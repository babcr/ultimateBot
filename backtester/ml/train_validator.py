"""Train, select and evaluate a frozen XGBoost validator from a SQLite snapshot."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from ultimateBot.backtester.embeddings.trade_dataset import code_provenance, dumps
from ultimateBot.backtester.embeddings.ts2vec_service import utc
from ultimateBot.backtester.ml.trade_features import FEATURE_CONTRACT, feature_names, feature_vector


DEFAULT_MODEL_DIRECTORY = Path("C:/TradingBotData/models/trade_validators")


class InsufficientTrainingData(ValueError):
    """Collection can resolve this condition without changing the training contract."""


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float = .60
    validation_fraction: float = .20
    min_train_samples: int = 100
    min_validation_samples: int = 30
    min_test_samples: int = 30
    min_class_samples: int = 5
    min_accepted_validation: int = 10
    target_r: float = 0.0
    threshold: float | None = None
    n_estimators: int = 600
    max_depth: int = 3
    learning_rate: float = .03
    early_stopping_rounds: int = 40
    seed: int = 1
    threads: int = 2

    def __post_init__(self):
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("Train/validation fractions must be positive and leave a test partition")
        if min(self.min_train_samples, self.min_validation_samples, self.min_test_samples,
               self.min_class_samples, self.min_accepted_validation, self.n_estimators,
               self.max_depth, self.early_stopping_rounds, self.threads) < 1:
            raise ValueError("Training counts, depth, rounds and threads must be positive")
        if not math.isfinite(self.target_r) or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("Require a finite target R and learning rate in (0,1]")
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("Validator threshold must be in [0,1]")
        if not 0 <= self.seed < 2**31:
            raise ValueError("XGBoost seed must be in [0,2**31)")


def timestamp(value):
    return utc(datetime.fromisoformat(value))


def read_snapshot(database, *, model_id=None, campaign_id=None, completed_only=False, cutoff=None):
    """A single read transaction also admits committed closes from running runs."""
    path = Path(database).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        columns = {r[1] for r in db.execute("PRAGMA table_info(closed_trades)")}
        if "feature_point" not in columns:
            raise ValueError("No order-creation captures yet; legacy fill-time data cannot train this validator")
        where = "r.status='completed'" if completed_only else "r.status IN ('completed','running')"
        where += " AND t.feature_point='order_creation'"
        params = []
        for column, value in (("t.model_id", model_id), ("r.campaign_id", campaign_id)):
            if value is not None:
                where += f" AND {column}=?"
                params.append(value)
        records = [dict(row) for row in db.execute(
            f"SELECT t.*,r.adaptive_seed,r.campaign_id FROM closed_trades t JOIN runs r USING(run_id) WHERE {where}", params)]
        if cutoff is not None:
            limit = timestamp(cutoff)
            records = [r for r in records if timestamp(r["closed_at"]) <= limit]
        models = {r["model_id"] for r in records}
        if not models:
            raise InsufficientTrainingData("No closed order-creation captures yet; legacy fill rows are excluded")
        if len(models) != 1:
            raise ValueError("Select exactly one embedding model with closed order-creation captures; legacy fill rows are excluded")
        selected = models.pop()
        model_row = db.execute("SELECT * FROM embedding_models WHERE model_id=?", (selected,)).fetchone()
        if model_row is None:
            raise ValueError("Embedding model provenance missing from database")
        model = dict(model_row)
        model["metadata"] = json.loads(model["metadata_json"])
        run_ids = sorted({r["run_id"] for r in records})
        # Preserve the fixed strategy/risk snapshots used to generate these labels.
        configurations = [dict(row) for row in db.execute("SELECT * FROM run_parameters ORDER BY run_id,section")
                          if row["run_id"] in set(run_ids)]
    records.sort(key=lambda r: (timestamp(r["feature_observed_at"]), r["run_id"], r["trade_id"]))
    if not records:
        raise ValueError("No closed trades available for training")
    dimensions = model["metadata"]["config"]["output_dims"]
    for row in records:
        if row["embedding_dims"] != dimensions or row["pnl_r"] is None or not math.isfinite(row["pnl_r"]):
            raise ValueError("Training requires consistent embeddings and a finite realized PnL in R for every trade")
        if not (timestamp(row["context_start"]) < timestamp(row["context_end"])
                <= timestamp(row["feature_observed_at"]) <= timestamp(row["opened_at"]) <= timestamp(row["closed_at"])):
            raise ValueError("Invalid trade/context chronology in training database")
        if timestamp(row["feature_observed_at"]) < timestamp(model["metadata"]["trained_through"]):
            raise ValueError("Embedding checkpoint postdates a training trade")
        row["features"] = feature_vector(json.loads(row["embedding_json"]), row["regression_lookback"],
                                        row["trailing"], row["side"], dimensions=dimensions)
    fingerprint = hashlib.sha256(dumps([
        {k: v for k, v in row.items() if k != "features"} for row in records
    ]).encode()).hexdigest()
    return records, model, {"database": str(path), "sha256": fingerprint, "rows": len(records),
                            "run_ids": run_ids, "run_parameters": configurations,
                            "completed_only": completed_only, "cutoff": cutoff}


def chronological_split(records, config):
    """Keep UTC market days together across seeds, then purge targets and contexts."""
    days = sorted({timestamp(r["feature_observed_at"]).date() for r in records})
    train_end = int(len(days) * config.train_fraction)
    validation_end = int(len(days) * (config.train_fraction + config.validation_fraction))
    if not 0 < train_end < validation_end < len(days):
        raise InsufficientTrainingData("Need more distinct trading days for three chronological partitions")
    validation_start, test_start = days[train_end], days[validation_end]
    train, validation, test = [], [], []
    for row in records:
        day = timestamp(row["feature_observed_at"]).date()
        (train if day < validation_start else validation if day < test_start else test).append(row)
    # Purge both overlapping return labels and observations inside the next
    # partition's input window. The real context_start includes weekend gaps.
    test_information_start = min(timestamp(r["context_start"]) for r in test)
    validation_information_start = min(timestamp(r["context_start"]) for r in validation)
    original = {"train": len(train), "validation": len(validation), "test": len(test)}
    train = [r for r in train if timestamp(r["closed_at"]) < validation_information_start]
    validation = [r for r in validation if timestamp(r["closed_at"]) < test_information_start]
    splits = {"train": train, "validation": validation, "test": test}
    for name, minimum in (("train", config.min_train_samples), ("validation", config.min_validation_samples),
                          ("test", config.min_test_samples)):
        effective = float(observation_weights(splits[name]).sum()) if splits[name] else 0.
        if effective + 1e-5 < minimum:
            raise InsufficientTrainingData(f"Insufficient {name} data after purging: {effective:g} effective observations < {minimum}")
    info = {"validation_start_day": validation_start.isoformat(), "test_start_day": test_start.isoformat(),
            "train_labels_before": validation_information_start.isoformat(),
            "validation_labels_before": test_information_start.isoformat(),
            "rows_before_purge": original,
            "rows_after_purge": {name: len(rows) for name, rows in splits.items()}}
    return splits, info


def observation_weights(records):
    # Repeated seeds/passes must not multiply the influence of the same observed
    # context and action. All such replicas stay in the same time partition.
    keys = [(r["symbol"], r["context_end"], r["regression_lookback"], r["trailing"], r["side"])
            for r in records]
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys], dtype=np.float32)


def partition_arrays(records, config):
    return (np.stack([r["features"] for r in records]),
            np.asarray([int(r["pnl_r"] > config.target_r) for r in records], dtype=np.int32),
            observation_weights(records))


def trading_metrics(records, accepted, weights):
    returns = np.asarray([r["pnl_r"] for r in records], dtype=np.float64)
    pnl = np.asarray([r["pnl"] for r in records], dtype=np.float64)
    accepted = np.asarray(accepted, dtype=bool)
    w = weights * accepted
    count = float(w.sum())
    gross_profit = float(np.sum(w * np.maximum(returns, 0)))
    gross_loss = float(np.sum(w * np.maximum(-returns, 0)))
    total = float(np.sum(w * returns))
    return {"trades": int(accepted.sum()), "weighted_trades": count,
            "coverage": float(w.sum() / weights.sum()),
            "win_rate": float(np.sum(w * (returns > 0)) / count) if count else None,
            "total_r": total, "mean_r": total / count if count else None,
            "profit_factor_r": gross_profit / gross_loss if gross_loss else None,
            "gross_profit_r": gross_profit, "gross_loss_r": gross_loss,
            "raw_pnl_sum": float(pnl[accepted].sum())}


def select_threshold(records, probabilities, weights, config):
    thresholds = [config.threshold] if config.threshold is not None else np.linspace(0, .95, 39)
    candidates = []
    for threshold in thresholds:
        metrics = trading_metrics(records, probabilities >= threshold, weights)
        eligible = metrics["weighted_trades"] >= config.min_accepted_validation
        candidates.append({"threshold": float(threshold), "eligible": eligible, **metrics})
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise InsufficientTrainingData("No threshold retains enough validation observations; collect more data or change --min-accepted-validation")
    # Weighted total R scores rejected opportunities as zero. Ties keep more trades.
    best = max(eligible, key=lambda row: (row["total_r"], row["coverage"]))
    return best["threshold"], candidates


def evaluate(records, probabilities, weights, config, threshold):
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    labels = np.asarray([int(r["pnl_r"] > config.target_r) for r in records])
    return {"rows": len(records), "effective_observations": float(weights.sum()),
            "positive_labels": int(labels.sum()),
            "roc_auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)) if len(set(labels)) > 1 else None,
            "log_loss": float(log_loss(labels, probabilities, sample_weight=weights, labels=[0, 1])),
            "brier_score": float(brier_score_loss(labels, probabilities, sample_weight=weights)),
            "baseline": trading_metrics(records, np.ones(len(records), dtype=bool), weights),
            "selected": trading_metrics(records, probabilities >= threshold, weights)}


def register_training(database, validator_id, directory, metadata, report):
    with closing(sqlite3.connect(database, timeout=60)) as db:
        with db:
            db.execute("""CREATE TABLE IF NOT EXISTS trade_validator_models (
                validator_id TEXT PRIMARY KEY, embedding_model_id TEXT NOT NULL,
                created_at TEXT NOT NULL, available_after TEXT NOT NULL,
                artifact_directory TEXT NOT NULL, metadata_json TEXT NOT NULL, report_json TEXT NOT NULL)""")
            db.execute("INSERT INTO trade_validator_models VALUES(?,?,?,?,?,?,?)",
                       (validator_id, metadata["embedding_model_id"], metadata["created_at"], metadata["available_after"],
                        str(directory), dumps(metadata), dumps(report)))


def train_validator(database, *, config=TrainingConfig(), output_dir=None, model_id=None,
                    campaign_id=None, completed_only=False, cutoff=None):
    try:
        import xgboost as xgb
        import sklearn
    except ImportError as error:
        raise RuntimeError("Install validator dependencies: py -m pip install -r requirements-validator.txt") from error
    records, embedding_model, snapshot = read_snapshot(database, model_id=model_id, campaign_id=campaign_id,
                                                      completed_only=completed_only, cutoff=cutoff)
    splits, split_info = chronological_split(records, config)
    arrays = {name: partition_arrays(rows, config) for name, rows in splits.items()}
    for name in ("train", "validation"):
        _, labels, weights = arrays[name]
        if any(float(weights[labels == label].sum()) < config.min_class_samples for label in (0, 1)):
            raise InsufficientTrainingData(f"Need at least {config.min_class_samples} effective observations of both classes in {name}")
    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", eval_metric="logloss",
        n_estimators=config.n_estimators, max_depth=config.max_depth, learning_rate=config.learning_rate,
        min_child_weight=5, reg_lambda=5, reg_alpha=.1, subsample=.8, colsample_bytree=.8,
        early_stopping_rounds=config.early_stopping_rounds, random_state=config.seed, n_jobs=config.threads)
    x_train, y_train, w_train = arrays["train"]
    x_validation, y_validation, w_validation = arrays["validation"]
    model.fit(x_train, y_train, sample_weight=w_train,
              eval_set=[(x_validation, y_validation)], sample_weight_eval_set=[w_validation], verbose=False)
    probabilities = {name: model.predict_proba(data[0])[:, 1] for name, data in arrays.items()}
    threshold, selection = select_threshold(splits["validation"], probabilities["validation"], w_validation, config)
    report = {"split": split_info, "threshold_selection": selection,
              "partitions": {name: evaluate(rows, probabilities[name], arrays[name][2], config, threshold)
                             for name, rows in splits.items()},
              "evaluation_kind": "historical_trade_selection_not_portfolio_replay"}
    now = datetime.now(timezone.utc)
    directory = (Path(output_dir) if output_dir is not None else
                 DEFAULT_MODEL_DIRECTORY / f"{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    names = feature_names(embedding_model["metadata"]["config"]["output_dims"])
    model.get_booster().set_attr(trade_feature_contract=FEATURE_CONTRACT)
    model.save_model(directory / "model.ubj")
    model_sha256 = hashlib.sha256((directory / "model.ubj").read_bytes()).hexdigest()
    validator_id = hashlib.sha256(dumps({"model_sha256": model_sha256, "threshold": threshold,
                                       "created_at": now.isoformat(), "snapshot": snapshot["sha256"]}).encode()).hexdigest()
    metadata = {
        "format_version": 1, "feature_contract": FEATURE_CONTRACT, "feature_names": names,
        "decision_point": "order_creation", "validator_id": validator_id, "model_sha256": model_sha256,
        "created_at": now.isoformat(), "embedding_model_id": embedding_model["model_id"],
        "embedding_metadata": embedding_model["metadata"], "embedding_checkpoint": embedding_model["checkpoint_path"],
        "threshold": threshold, "target": {"name": "pnl_r_greater_than", "r": config.target_r},
        # Threshold selection and evaluation have also consumed their outcomes.
        "available_after": max(timestamp(r["closed_at"]) for r in records).isoformat(),
        "trained_labels_through": max(timestamp(r["closed_at"]) for r in splits["train"]).isoformat(),
        "symbols": sorted({r["symbol"] for r in splits["train"]}),
        "training_config": asdict(config), "database_snapshot": snapshot,
        "best_iteration": int(model.best_iteration), "versions": {"xgboost": xgb.__version__, "sklearn": sklearn.__version__},
        "code": code_provenance(),
    }
    (directory / "metadata.json").write_text(dumps(metadata), encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["partition", "run_id", "trade_id", "symbol", "feature_observed_at", "opened_at", "closed_at", "pnl_r", "label", "score", "accepted", "weight"])
        for name, rows in splits.items():
            for row, probability, weight in zip(rows, probabilities[name], arrays[name][2]):
                writer.writerow([name, row["run_id"], row["trade_id"], row["symbol"], row["feature_observed_at"], row["opened_at"], row["closed_at"],
                                 row["pnl_r"], int(row["pnl_r"] > config.target_r), float(probability),
                                 int(probability >= threshold), float(weight)])
    register_training(database, validator_id, directory, metadata, report)
    return directory, metadata, report
