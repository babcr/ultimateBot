"""Learn on 2020-2023, validate on 2024-2025, then freeze and test on 2026."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from uuid import uuid4

import numpy as np
from ultimateBot.backtester.research.swing_selection import (
    StopSelection, acceptance_mask, selection_metrics, exact_stop_threshold,
)

from ultimateBot.backtester.research.swing_encoder import SwingEncoder, SwingEncoderConfig, encoder_training_plan
from ultimateBot.backtester.research.swing_opportunities import (
    OPPORTUNITY_SCHEMA, OpportunityConfig, connect, json_dump, uses_stop_distance, stop_distance_feature,
)
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, FEATURE_SCHEMAS, VOLUME_FEATURE_SCHEMA, SWING_SCHEMA, SwingWindow, feature_names, model_feature_schema, swing_features


def at(value):
    return datetime.fromisoformat(value)


BINARY_TARGETS = ('win', 'trailing-step-1')
TRAILING_TARGETS = ('trailing-step-1', 'trailing-step-raw', 'trailing-step-square', 'trailing-step-exp')
RESOLVER_TARGETS = ('pnl-r', *BINARY_TARGETS, *TRAILING_TARGETS[1:])


def is_binary_target(target):
    if target not in RESOLVER_TARGETS:
        raise ValueError(f'Unknown XGBoost target: {target}')
    return target in BINARY_TARGETS


def validate_regression_labels(labels, target):
    # XGBoost uses float32 gradients. Keep squared errors representable too.
    values = np.asarray(labels, dtype=np.float64)
    if not np.isfinite(values).all() or (np.abs(values) > math.sqrt(np.finfo(np.float32).max)).any():
        raise ValueError(f'Target {target} has labels too large for finite XGBoost squared errors; '
                         'use trailing-step-raw or trailing-step-square for large trailing levels')


def target_label(row, target):
    if target == 'pnl-r':
        return row['pnl_r']
    if target == 'win':
        return int(row['pnl_r'] > 0)
    if target in TRAILING_TARGETS:
        steps = row.get('trailing_steps')
        if (not isinstance(steps, Real) or not math.isfinite(steps)
                or steps < 0 or not float(steps).is_integer()):
            return None
        if target == 'trailing-step-1':
            return int(steps >= 1)
        if target == 'trailing-step-raw':
            label = float(steps)
        elif target == 'trailing-step-square':
            label = float(steps) ** 2
        else:
            # Check before exp() to give an actionable error instead of overflow.
            if steps > math.log(math.sqrt(np.finfo(np.float32).max)):
                raise ValueError('Target trailing-step-exp has labels too large for finite XGBoost squared errors; '
                                 'use trailing-step-raw or trailing-step-square for large trailing levels')
            label = math.exp(steps)
        validate_regression_labels([label], target)
        return label
    raise ValueError(f'Unknown XGBoost target: {target}')


def target_definition(target):
    return {'name': target,
            'label': {'pnl-r': 'realized pnl_r', 'win': 'int(pnl_r > 0)',
                      'trailing-step-1': 'int(trailing_steps >= 1)',
                      'trailing-step-raw': 'trailing_steps',
                      'trailing-step-square': 'trailing_steps ** 2',
                      'trailing-step-exp': 'exp(trailing_steps)'}[target],
            'prediction': ('probability' if is_binary_target(target) else
                           'pnl_r' if target == 'pnl-r' else 'transformed_trailing_steps'),
            'threshold_selection_metric': ('validation_total_pnl_r' if target == 'pnl-r'
                                           else 'validation_mean_target_label')}


def selection_objective(target):
    objective = dict(version=1, target=target, metric='total_r' if target == 'pnl-r' else 'mean_label',
                     direction='maximize', abstention_score=1. if target == 'trailing-step-exp' else 0.)
    if target != 'pnl-r':
        objective['acceptance_count_filter'] = False
    return objective


def validation_score(evaluation, target):
    """Selection utility in target units; realized PnL remains descriptive."""
    selected = evaluation['selected']
    if target == 'pnl-r':
        return selected['total_r']
    return selected['mean_label'] if selected['mean_label'] is not None else selection_objective(target)['abstention_score']


def validation_has_gain(evaluation, target):
    return validation_score(evaluation, target) > selection_objective(target)['abstention_score']


@dataclass(frozen=True)
class TrainingSelection:
    """Select existing observations, never resample a collected swing series."""
    symbols: tuple[str, ...] | None = None
    interval_seconds: int | None = None
    window_mode: str = "both"
    durations: tuple[int, ...] | None = None
    pairs: tuple[int, ...] | None = None
    include_volume: bool | None = None

    def __post_init__(self):
        if self.include_volume is not None and not isinstance(self.include_volume, bool):
            raise ValueError('include_volume must be a boolean or None')
        if self.window_mode not in ("both", "duration", "pairs"):
            raise ValueError("Invalid window mode")
        if self.interval_seconds is not None:
            OpportunityConfig(interval_seconds=self.interval_seconds)
        for name in ("durations", "pairs"):
            values = getattr(self, name)
            if values is not None:
                if not values:
                    raise ValueError(f"{name} must not be empty")
                mode = "duration" if name == "durations" else "pairs"
                for value in values:
                    SwingWindow(mode, value)
                object.__setattr__(self, name, tuple(dict.fromkeys(values)))
        if self.window_mode == "duration" and self.pairs is not None or self.window_mode == "pairs" and self.durations is not None:
            raise ValueError("Requested window values conflict with --window-mode")
        if self.symbols is not None:
            symbols = tuple(sorted({s.strip().upper() for s in self.symbols}))
            if not symbols or any(not s for s in symbols):
                raise ValueError("Select at least one nonempty symbol")
            object.__setattr__(self, "symbols", symbols)

    def windows(self, parameters):
        available = list(dict.fromkeys(SwingWindow(**w) for w in parameters["windows"]))
        explicit = self.durations is not None or self.pairs is not None
        requested = []
        for mode, values in (("duration", self.durations), ("pairs", self.pairs)):
            if self.window_mode not in ("both", mode):
                continue
            if values is not None:
                requested.extend(SwingWindow(mode, v) for v in values)
            elif not explicit:
                requested.extend(w for w in available if w.mode == mode)
        missing = [w.key for w in requested if w not in available]
        if (missing and parameters.get('store_contexts', True)) or not requested:
            raise ValueError(f"Requested windows unavailable in this collection: {missing or self.window_mode}; "
                             f"available: {[w.key for w in available]}")
        return requested


def collection_symbols(db, run_id, parameters):
    return sorted(parameters.get("symbols") or [r[0] for r in db.execute(
        "SELECT DISTINCT symbol FROM swing_events WHERE run_id=?", (run_id,))])


def select_collection(db, run_id, selection):
    query = "SELECT * FROM swing_runs WHERE status='completed'"
    args = ()
    if run_id is not None:
        query += " AND run_id=?"
        args = (run_id,)
    for run in db.execute(query + " ORDER BY started_at DESC", args).fetchall():
        parameters = json.loads(run["parameters_json"])
        try:
            model_feature_schema(parameters['feature_schema'], selection.include_volume)
            if selection.interval_seconds is not None and parameters["config"]["interval_seconds"] != selection.interval_seconds:
                raise ValueError("Candle interval differs from the collection; collect that interval first")
            symbols = collection_symbols(db, run["run_id"], parameters)
            if selection.symbols is not None and not set(selection.symbols) <= set(symbols):
                raise ValueError(f"Requested symbols unavailable; collected symbols: {symbols}")
            windows = selection.windows(parameters)
        except ValueError:
            if run_id is not None:
                raise
            continue
        return run, parameters, windows, list(selection.symbols or symbols)
    raise ValueError("No completed collection matches the requested run, symbols, candle interval and windows")


def scope_predicate(symbols, alias="e"):
    clauses, args = [], []
    if symbols:
        clauses.append(f"{alias}.symbol IN ({','.join('?' for _ in symbols)})")
        args.extend(symbols)
    return " AND ".join(clauses) or "1=1", args


@dataclass(frozen=True)
class ResolverConfig:
    target: str = "pnl-r"
    min_encoder_samples: int = 50
    min_train_samples: int = 100
    min_validation_samples: int = 30
    min_accepted_validation: int = 10
    n_estimators: int = 400
    max_depth: int = 3
    learning_rate: float = .03
    early_stopping_rounds: int = 30
    seed: int = 1
    threads: int = 2

    def __post_init__(self):
        if self.target not in RESOLVER_TARGETS:
            raise ValueError(f"XGBoost target must be one of {RESOLVER_TARGETS}")
        for key in ("min_encoder_samples", "min_train_samples", "min_validation_samples",
                    "min_accepted_validation", "n_estimators", "max_depth", "early_stopping_rounds", "threads"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1 or not 0 <= self.seed < 2**31:
            raise ValueError("Invalid learning rate or seed")


def day_start(value):
    return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=timezone.utc)


@dataclass(frozen=True)
class ResearchProtocol:
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2025-12-31"

    def __post_init__(self):
        a,b,c,d = (date.fromisoformat(v) for v in (self.train_start,self.train_end,self.validation_start,self.validation_end))
        if not a <= b < c <= d:
            raise ValueError("Require train_start <= train_end < validation_start <= validation_end")

    @property
    def bounds(self):
        return (day_start(self.train_start), day_start(self.train_end)+timedelta(days=1),
                day_start(self.validation_start), day_start(self.validation_end)+timedelta(days=1))


def purged_split(records, protocol, config):
    train_start,train_end,validation_start,validation_end = protocol.bounds
    train = [r for r in records if train_start <= at(r["confirmed_at"]) < train_end]
    validation = [r for r in records if validation_start <= at(r["confirmed_at"]) < validation_end]
    before = {"train":len(train),"validation":len(validation)}
    # A 2023 entry closed in 2024 cannot provide a 2023 training label. Likewise
    # validation never sees a trade's 2026 outcome. Past context overlap is legitimate.
    train = [r for r in train if at(r["closed_at"]) < train_end]
    validation = [r for r in validation if at(r["closed_at"]) < validation_end]
    partitions = {"train":train,"validation":validation}
    for name,minimum in (("train",config.min_train_samples),("validation",config.min_validation_samples)):
        if len(partitions[name]) < minimum:
            raise ValueError(f"Need {minimum} common {name} samples with outcomes known within that period; got {len(partitions[name])}")
    return partitions,{"before_purge":before,"after_purge":{k:len(v) for k,v in partitions.items()},
                        "train_labels_before":train_end.isoformat(),"validation_labels_before":validation_end.isoformat(),
                        "observed_signal_ranges":{name:{"first":samples[0]["confirmed_at"],"last":samples[-1]["confirmed_at"]}
                                                  for name,samples in partitions.items()}}


def metrics(rows, accepted):
    accepted = np.asarray(accepted, dtype=bool)
    returns = np.asarray([r["pnl_r"] for r in rows], dtype=np.float64)[accepted]
    positive, negative = float(np.maximum(returns, 0).sum()), float(-np.minimum(returns, 0).sum())
    return {"opportunities": int(accepted.sum()), "coverage": float(accepted.mean()) if len(accepted) else 0.,
            "total_r": float(returns.sum()), "mean_r": float(returns.mean()) if len(returns) else None,
            "win_rate": float((returns > 0).mean()) if len(returns) else None,
            "profit_factor_r": positive / negative if negative else None,
            "gross_profit_r": positive, "gross_loss_r": negative}


def label_metrics(labels, accepted):
    """Describe accepted outcomes in the model's label units, without using PnL."""
    known = np.asarray([value for value, keep in zip(labels, accepted) if keep and value is not None], dtype=np.float64)
    return dict(known_target_labels=len(known), total_label=float(known.sum()),
                mean_label=float(known.mean()) if len(known) else None)


def choose_threshold(rows, predictions, config, opportunity_config=None):
    # All selection is on validation. Test PnLs never enter this function during training.
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(rows),) or not np.isfinite(predictions).all():
        raise ValueError("Invalid XGBoost predictions")
    if is_binary_target(config.target) and ((predictions < 0) | (predictions > 1)).any():
        raise ValueError('Binary XGBoost predictions must be probabilities between 0 and 1')
    selection = StopSelection.from_rows(rows, opportunity_config)
    if config.target != 'pnl-r':
        labels = [target_label(row, config.target) for row in rows]
        if not len(labels) or any(label is None for label in labels):
            raise ValueError('Threshold selection requires known target labels')
        labels = np.asarray(labels, dtype=np.float64)
        if not np.isfinite(labels).all():
            raise ValueError('Threshold selection requires finite target labels')
        policy = selection or StopSelection(np.arange(len(rows)).astype(str), np.zeros(len(rows), dtype=int), 1)
        floor = getattr(config, 'min_threshold', min(0., float(predictions.min())))
        objective = selection_objective(config.target)
        best, points = exact_stop_threshold(labels, predictions, floor, policy, score_metric='mean_r',
            empty_score=objective['abstention_score'], max_threshold=1. if is_binary_target(config.target) else None)
        pnl = np.asarray([row['pnl_r'] for row in rows], dtype=np.float64)
        candidates = []
        for point in points:
            mask = acceptance_mask(predictions, point['threshold'], selection)
            count = int(mask.sum())
            outcome = selection_metrics(pnl, mask, policy)
            outcome.update(known_target_labels=count, total_label=float(labels[mask].sum()),
                mean_label=float(labels[mask].mean()) if count else None)
            candidates.append(dict(threshold=point['threshold'], **outcome,
                selection_score=point['mean_r'] if count else objective['abstention_score'],
                selection_metric=objective['metric']))
        return best['threshold'], candidates
    if selection is not None:
        floor = getattr(config, 'min_threshold', min(0., float(predictions.min())))
        selected, candidates = exact_stop_threshold([r['pnl_r'] for r in rows], predictions, floor, selection)
        return selected['threshold'], candidates
    thresholds = np.unique(np.r_[0., np.nextafter(predictions.min(), -np.inf),
                                np.quantile(predictions, np.linspace(0, .95, 20))])
    if is_binary_target(config.target):
        thresholds = np.unique(np.clip(thresholds, 0., 1.))
    candidates = [{"threshold": float(t), **metrics(rows, predictions >= t)} for t in thresholds]
    eligible = [c for c in candidates if c["opportunities"] >= config.min_accepted_validation]
    if not eligible:
        raise ValueError("Too few accepted validation opportunities for threshold selection")
    best = max(eligible, key=lambda c: (c["total_r"], c["coverage"]))
    return best["threshold"], candidates


TABULAR_FEATURES = ["side", "log_swing_count", "log_span_periods"]


def tabular(row, interval_seconds, config=None):
    values = [row["side"], math.log1p(row["n_swings"]), math.log1p(row["span_seconds"] / interval_seconds)]
    if uses_stop_distance(config or {}):
        values.append(stop_distance_feature(config, row['symbol'], row['pivot_price'], row['stop_buffer_pips']))
    return values


def fit_xgb(arrays, config):
    import xgboost as xgb
    binary = is_binary_target(config.target)
    if not binary:
        for _, labels in arrays.values():
            validate_regression_labels(labels, config.target)
    constructor = xgb.XGBClassifier if binary else xgb.XGBRegressor
    model = constructor(n_estimators=config.n_estimators, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=1., colsample_bytree=1.,
                        n_jobs=config.threads, random_state=config.seed, tree_method="hist",
                        objective="binary:logistic" if binary else "reg:squarederror",
                        eval_metric=["auc", "logloss"] if binary else "rmse",
                        early_stopping_rounds=config.early_stopping_rounds)
    if binary and len(np.unique(arrays['train'][1])) != 2:
        raise ValueError(f"Binary target {config.target} requires both label 0 and label 1 in training trades")
    # The final dataset/metric still controls early stopping. Training curves
    # and AUC provide diagnostics without replacing validation logloss/RMSE.
    model.fit(*arrays["train"], eval_set=[arrays['train'], arrays["validation"]], verbose=False)
    return model


def predict(model, features, target):
    return model.predict_proba(features)[:, 1] if is_binary_target(target) else model.predict(features)


def learning_diagnostics(model, arrays, target):
    """Compare predictions with a constant learned only on training labels."""
    from sklearn.metrics import log_loss, roc_auc_score
    prior = float(np.mean(arrays['train'][1]))
    partitions = {}
    for phase, (features, labels) in arrays.items():
        prediction = predict(model, features, target)
        if not is_binary_target(target):
            error = float(np.mean((labels-prediction)**2))
            constant_error = float(np.mean((labels-prior)**2))
            auc = None
        else:
            error = float(log_loss(labels, prediction, labels=[0, 1]))
            constant_error = float(log_loss(labels, np.full(len(labels), prior), labels=[0, 1]))
            auc = float(roc_auc_score(labels, prediction)) if len(np.unique(labels)) == 2 else None
        partitions[phase] = {'error': error, 'constant_error': constant_error,
            'improvement_over_constant': constant_error-error, 'auc': auc,
            'prediction_min': float(prediction.min()), 'prediction_max': float(prediction.max()),
            'unique_predictions': int(len(np.unique(prediction)))}
    validation = partitions['validation']
    gain = validation['improvement_over_constant'] > 0 and (
        not is_binary_target(target) or validation['auc'] is not None and validation['auc'] > .5)
    return {'error_metric': 'logloss' if is_binary_target(target) else 'mse',
        'constant_from_training': prior, 'partitions': partitions,
        'validation_predictive_gain': bool(gain),
        'best_iteration': int(model.best_iteration), 'trees_used': int(model.best_iteration+1),
        'rounds_trained': int(model.get_booster().num_boosted_rounds()),
        'learning_curves': model.evals_result(),
        'early_stopping_metric': 'logloss' if is_binary_target(target) else 'rmse',
        'status': 'validation_gain_observed' if gain else 'no_validation_predictive_gain'}


def evaluate(rows, predictions, threshold, target, opportunity_config=None):
    labels = [target_label(r, target) for r in rows]
    known = np.asarray([label is not None for label in labels], dtype=bool)
    actual = np.asarray([label for label in labels if label is not None])
    error = float(np.mean((actual - np.asarray(predictions)[known]) ** 2)) if known.any() else None
    result = {"rows": len(rows), "prediction_mse": error, "prediction_target": target,
            "known_target_labels": int(known.sum()),
            "target_positive_labels": int(actual.sum()) if is_binary_target(target) else None,
            "all_opportunities": metrics(rows, np.ones(len(rows), dtype=bool)),
            "selected": metrics(rows, predictions >= threshold)}
    selection = StopSelection.from_rows(rows, opportunity_config)
    if selection is not None:
        y = [r['pnl_r'] for r in rows]
        result.update(selection.diagnostic(),
            all_opportunities=selection_metrics(y, selection.mask(np.ones(len(rows)), 0.), selection),
            selected=selection_metrics(y, selection.mask(predictions, threshold), selection))
    all_mask = acceptance_mask(np.ones(len(rows)), 0., selection)
    selected_mask = acceptance_mask(predictions, threshold, selection)
    result['all_opportunities'].update(label_metrics(labels, all_mask))
    result['selected'].update(label_metrics(labels, selected_mask))
    result['selection_objective'] = selection_objective(target)
    result['selection_score'] = validation_score(result, target)
    return result


def train(database, *, run_id=None, output_dir, config=ResolverConfig(), encoder_config=SwingEncoderConfig(),
          protocol=ResearchProtocol(), selection=TrainingSelection(), representation='ts2vec', cohort_policy='common',
          grid_normalization='swing-mean'):
    if grid_normalization != 'swing-mean' and representation != 'swing-grid':
        raise ValueError('Grid normalization applies only to swing-grid')
    if cohort_policy not in ('common', 'per-window') or (cohort_policy != 'common' and representation != 'swing-grid'):
        raise ValueError('Per-window cohorts are supported only for swing-grid')
    if representation == 'swing-grid':
        from ultimateBot.backtester.research.swing_grid import train_grid, train_grid_per_window
        trainer = train_grid_per_window if cohort_policy == 'per-window' else train_grid
        return trainer(database, run_id=run_id, output_dir=output_dir, config=config,
                          encoder_config=encoder_config, protocol=protocol, selection=selection,
                          grid_normalization=grid_normalization)
    if representation != 'ts2vec':
        raise ValueError('Unknown swing representation')
    if not Path(database).is_file():
        raise FileNotFoundError("Collect the swing database before training")
    db = connect(database)
    try:
        run, parameters, windows, symbols = select_collection(db, run_id, selection)
        run_id = run["run_id"]
        if parameters["feature_schema"] not in FEATURE_SCHEMAS or parameters.get("swing_schema") != SWING_SCHEMA:
            raise ValueError("Incompatible collected swing feature schema")
        feature_schema = model_feature_schema(parameters['feature_schema'], selection.include_volume)
        input_features = feature_names(feature_schema)
        if parameters.get('opportunity_schema')!=OPPORTUNITY_SCHEMA or parameters.get('risk_management_filters') is not False:
            raise ValueError("Recollect with the unrestricted swing opportunity collector; this run uses an older execution policy")
        keys = list(dict.fromkeys(w.key for w in windows))
        scope = {"symbols": symbols,
                 "interval_seconds": parameters["config"]["interval_seconds"],
                 "window_keys": keys, "opportunity_config": parameters["config"]}
        predicate, scope_args = scope_predicate(symbols)
        window_placeholders = ','.join('?' for _ in keys)
        train_start,train_end,validation_start,validation_end = protocol.bounds
        from ultimateBot.backtester.research.swing_contexts import ensure_windows
        ensure_windows(db, run_id, windows, start=train_start, before=validation_end, symbols=symbols)
        event_source = 'swing_event_labels' if config.target in TRAILING_TARGETS else 'swing_events'
        target_filter = 'AND e.trailing_step_1_label IS NOT NULL' if config.target in TRAILING_TARGETS else ''
        common = [dict(r) for r in db.execute(f"""SELECT e.*,min(c.information_start) AS information_start
            FROM {event_source} e JOIN swing_contexts c USING(event_id)
            WHERE e.run_id=? AND e.status='closed' AND e.pnl_r IS NOT NULL
            {target_filter}
            AND {predicate} AND c.window_key IN ({window_placeholders})
            AND ((e.confirmed_at>=? AND e.confirmed_at<? AND e.closed_at<?)
              OR (e.confirmed_at>=? AND e.confirmed_at<? AND e.closed_at<?))
            GROUP BY e.event_id HAVING count(DISTINCT c.window_key)=? ORDER BY e.confirmed_at,e.symbol""",
            (run_id,*scope_args,*keys,train_start.isoformat(),train_end.isoformat(),train_end.isoformat(),
             validation_start.isoformat(),validation_end.isoformat(),validation_end.isoformat(),len(keys)))]
        if not common:
            raise ValueError("No labeled opportunities have every requested duration/couple context"
                             + (f" and known trailing_steps for target {config.target}" if target_filter else ''))
        if parameters['feature_schema'] == FEATURE_SCHEMA:
            # Adding a nullable volume column must not change the fingerprint of
            # old cohorts, otherwise resuming saved six-feature trials would fail.
            for row in common:
                row.pop('mean_quote_volume', None)
        if not uses_stop_distance(parameters['config']):
            for row in common:
                for column in ('swing_id', 'variant_index', 'stop_buffer_pips'):
                    row.pop(column, None)
        partitions, split_info = purged_split(common, protocol, config)
        for row in common:
            if row["pnl_r"] is None or not math.isfinite(row["pnl_r"]):
                raise ValueError("A closed opportunity has no finite realized PnL/R")
            # Fail on unsafe transformations before starting expensive TS2Vec fitting.
            target_label(row, config.target)
        fingerprint = hashlib.sha256(json_dump({"parameters": parameters, "scope": scope, "labeled_common_events": common}).encode()).hexdigest()
        experiment = Path(output_dir).resolve() / f"{run_id[:12]}_{uuid4().hex[:12]}"
        experiment.mkdir(parents=True, exist_ok=False)
        experiment_id = experiment.name
        with db:
            db.execute("INSERT INTO swing_experiments(experiment_id,run_id,status,directory,parameters_json) VALUES(?,?,'running',?,?)",
                       (experiment_id,run_id,str(experiment),json_dump({"protocol":asdict(protocol),"config":asdict(config),"encoder_config":asdict(encoder_config),"selection":asdict(selection),"scope":scope})))
        print(f"[swing-resolver] experiment={experiment}", flush=True)
        comparison = {"run_id": run_id, "database": str(Path(database).resolve()), "dataset_sha256": fingerprint,
                      'feature_schema': feature_schema,
                      "config": asdict(config), "encoder_config": asdict(encoder_config), "scope": scope,
                      "protocol":asdict(protocol),"experiment_id":experiment_id,"test_evaluated":False,
                      "target_definition": target_definition(config.target),
                      "split": split_info, "common_labeled_opportunities": len(common), "windows": {},
                      "interpretation": "Independent overlapping opportunities; sums of R are not a portfolio return."}
        for key in keys:
            rows = [dict(r) for r in db.execute(f"""SELECT c.context_id,c.event_id,c.window_key,c.context_end,
                c.information_start,c.n_swings,c.span_seconds,c.features,c.payload_context_id,e.side,e.symbol,e.status,e.pivot_price,e.stop_buffer_pips
                FROM swing_contexts_resolved c JOIN swing_events e USING(event_id)
                WHERE e.run_id=? AND c.window_key=? AND {predicate} AND
                ((c.context_end>=? AND c.context_end<?) OR (c.context_end>=? AND c.context_end<?))
                ORDER BY c.context_end,e.symbol""", (run_id,key,*scope_args,train_start.isoformat(),train_end.isoformat(),
                                                     validation_start.isoformat(),validation_end.isoformat()))]
            pretrain = [r for r in rows if train_start <= at(r["context_end"]) < train_end]
            unique_pretrain = list({r['payload_context_id']: r for r in pretrain}.values())
            validation_rows = [r for r in rows if validation_start <= at(r["context_end"]) < validation_end]
            observed = pretrain + validation_rows
            if len(unique_pretrain) < config.min_encoder_samples:
                raise ValueError(f"{key}: need {config.min_encoder_samples} training contexts for TS2Vec; got {len(unique_pretrain)}")
            # Resolve workload before decoding feature blobs or allocating tensors.
            encoder_plan = encoder_training_plan(encoder_config, [r['n_swings'] for r in unique_pretrain], feature_schema=feature_schema)
            if encoder_config.max_train_length and any(r['n_swings'] > encoder_config.max_train_length for r in observed):
                raise ValueError(f'{key}: an observed full window exceeds the explicit TS2Vec length limit')
            print(f"[swing-resolver-plan] window={key} " + json_dump(encoder_plan), flush=True)
            def sequence(row):
                source = np.frombuffer(row["features"], dtype="<f4").reshape(row["n_swings"], len(feature_names(parameters['feature_schema'])))
                return source[:, :len(input_features)].copy()
            directory = experiment / key
            directory.mkdir()
            checkpoint = directory / "ts2vec.pt"
            (directory/'training_plan.json').write_text(json_dump(encoder_plan), encoding='utf-8')
            pretrain_hash = hashlib.sha256(b"".join(r["features"] for r in unique_pretrain)).hexdigest()
            encoder = SwingEncoder(encoder_config, feature_schema=feature_schema).fit(
                [sequence(r) for r in unique_pretrain], observed_at=[at(r["context_end"]) for r in unique_pretrain],
                cutoff=train_end, checkpoint=checkpoint,
                provenance={"run_id": run_id, "window_key": key, "scope": {**scope, "window_keys": [key]}, "dataset_sha256": fingerprint,
                            "training_period":{"start":protocol.train_start,"end":protocol.train_end},
                            "validation_inputs_used_for_fit":False,
                            "pretrain_features_sha256": pretrain_hash, "source_fingerprint": parameters["source_fingerprint"]})
            with db:
                db.execute("INSERT INTO swing_embedding_models VALUES(?,?,?,?,?)",
                           (encoder.model_id, run_id, key, str(checkpoint), json_dump(encoder.metadata)))
            def encode_rows(records, training_reconstruction=False):
                unique = {r['payload_context_id']: r for r in records}
                encoded = encoder.encode([sequence(r) for r in unique.values()],
                    observed_at=[at(r['context_end']) for r in unique.values()],
                    training_reconstruction=training_reconstruction)
                lookup = dict(zip(unique, encoded))
                return np.stack([lookup[r['payload_context_id']] for r in records])
            training_vectors = encode_rows(pretrain, training_reconstruction=True)
            validation_vectors = encode_rows(validation_rows)
            vectors = np.concatenate((training_vectors,validation_vectors))
            with db:
                db.executemany("INSERT INTO swing_embeddings VALUES(?,?,?)",
                               [(r["context_id"], encoder.model_id, v.astype("<f4").tobytes()) for r, v in zip(observed, vectors)])
                eligible_ids = {r["event_id"] for samples in partitions.values() for r in samples}
                db.executemany("INSERT INTO swing_assignments VALUES(?,?,?,?,?)",
                    [(experiment_id,r["context_id"],"train" if at(r["context_end"])<train_end else "validation",
                      int(at(r["context_end"])<train_end),int(r["event_id"] in eligible_ids)) for r in observed])
            indexed = {r["event_id"]: (r, v) for r, v in zip(observed, vectors)}
            interval = parameters["config"]["interval_seconds"]
            full, base = {}, {}
            for name, samples in partitions.items():
                metadata = np.asarray([tabular(indexed[r["event_id"]][0], interval, parameters['config']) for r in samples], dtype=np.float32)
                embeddings = np.stack([indexed[r["event_id"]][1] for r in samples])
                targets = np.asarray([target_label(r, config.target) for r in samples])
                full[name] = (np.column_stack((embeddings, metadata)), targets)
                base[name] = (metadata, targets)
            model, baseline = fit_xgb(full, config), fit_xgb(base, config)
            diagnostics = learning_diagnostics(model, full, config.target)
            baseline_diagnostics = learning_diagnostics(baseline, base, config.target)
            predictions = {name: predict(model, x, config.target) for name, (x, y) in full.items()}
            baseline_predictions = {name: predict(baseline, x, config.target) for name, (x, y) in base.items()}
            threshold, candidates = choose_threshold(partitions["validation"], predictions["validation"], config, parameters['config'])
            baseline_threshold, _ = choose_threshold(partitions["validation"], baseline_predictions["validation"], config, parameters['config'])
            report = {"embedding_model_id": encoder.model_id, "threshold": threshold,
                      "encoder_training": encoder.metadata['training_plan'],
                      "learning_diagnostics": diagnostics, "tabular_learning_diagnostics": baseline_diagnostics,
                      "target_definition": target_definition(config.target),
                      "baseline_threshold": baseline_threshold, "threshold_candidates_validation_only": candidates,
                      "partitions": {name: evaluate(samples, predictions[name], threshold, config.target, parameters['config']) for name, samples in partitions.items()},
                      "tabular_only": {name: evaluate(samples, baseline_predictions[name], baseline_threshold, config.target, parameters['config']) for name, samples in partitions.items()}}
            report['validation_qualified'] = bool(diagnostics['validation_predictive_gain'] and
                validation_has_gain(report['partitions']['validation'], config.target))
            model.save_model(directory / "xgboost.json")
            baseline.save_model(directory / "xgboost_tabular_only.json")
            resolver_id = uuid4().hex
            metadata = {"version": 1, "resolver_id": resolver_id, "run_id": run_id, "window_key": key,
                        "feature_schema": feature_schema, "embedding_model_id": encoder.model_id,
                        "scope": {**scope, "window_keys": [key]},
                        "feature_names": [f"embedding_{i:03d}" for i in range(encoder_config.output_dims)] + TABULAR_FEATURES
                            + (['stop_buffer_ratio'] if uses_stop_distance(parameters['config']) else []),
                        "target": config.target, "threshold": threshold, "available_after": validation_end.isoformat(),
                        "target_definition": target_definition(config.target),
                        "selection_objective": selection_objective(config.target),
                        "validation_qualified": report['validation_qualified'],
                        "validation_predictive_gain": diagnostics['validation_predictive_gain'],
                        "protocol":asdict(protocol),"opportunity_config":parameters["config"],
                        "swing_schema":SWING_SCHEMA,"experiment_id":experiment_id,
                        "opportunity_schema":OPPORTUNITY_SCHEMA,
                        "interval_seconds": interval, "source_fingerprint": parameters["source_fingerprint"],
                        "dataset_sha256": fingerprint,
                        "xgboost_sha256": hashlib.sha256((directory / "xgboost.json").read_bytes()).hexdigest()}
            metadata["baseline_xgboost_sha256"] = hashlib.sha256((directory/"xgboost_tabular_only.json").read_bytes()).hexdigest()
            (directory / "metadata.json").write_text(json_dump(metadata), encoding="utf-8")
            (directory / "report.json").write_text(json_dump(report), encoding="utf-8")
            with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["event_id", "symbol", "partition", "confirmed_at", "closed_at",
                                                           "pnl_r", "trailing_steps", "target_label", "prediction", "accepted", "tabular_prediction", "tabular_accepted"])
                writer.writeheader()
                for name, samples in partitions.items():
                    policy = StopSelection.from_rows(samples, parameters['config'])
                    accepted = acceptance_mask(predictions[name], threshold, policy)
                    base_accepted = acceptance_mask(baseline_predictions[name], baseline_threshold, policy)
                    for i, (r, value, bv) in enumerate(zip(samples, predictions[name], baseline_predictions[name])):
                        writer.writerow({**{k: r[k] for k in ("event_id", "symbol", "confirmed_at", "closed_at", "pnl_r")},
                                         "trailing_steps": r['trailing_steps'], "target_label": target_label(r, config.target),
                                         "partition": name, "prediction": float(value), "accepted": int(accepted[i]),
                                         "tabular_prediction": float(bv), "tabular_accepted": int(base_accepted[i])})
            with db:
                db.execute("INSERT INTO swing_resolvers VALUES(?,?,?,?,?,?,?)",
                           (resolver_id, run_id, key, encoder.model_id, str(directory), json_dump(metadata), json_dump(report)))
                all_simple=np.asarray([tabular(r,interval,parameters['config']) for r in observed],dtype=np.float32)
                all_predictions=predict(model,np.column_stack((vectors,all_simple)),config.target)
                all_baseline=predict(baseline,all_simple,config.target)
                policy = StopSelection.from_rows(observed, parameters['config'])
                accepted = acceptance_mask(all_predictions, threshold, policy)
                base_accepted = acceptance_mask(all_baseline, baseline_threshold, policy)
                db.executemany("INSERT INTO swing_predictions VALUES(?,?,?,?,?,?,?)",
                    [(resolver_id,r['context_id'],'train' if at(r['context_end'])<train_end else 'validation',float(value),
                      int(accepted[i]),float(bv),int(base_accepted[i]))
                     for i,(r,value,bv) in enumerate(zip(observed,all_predictions,all_baseline))])
            comparison["windows"][key] = {"directory": str(directory), "resolver_id": resolver_id,
                       "encoder_train_contexts": len(pretrain), "embedded_contexts": len(observed), "report": report}
            (experiment / "comparison.partial.json").write_text(json_dump(comparison), encoding="utf-8")
            print(f"[swing-resolver] {key}: validation R={report['partitions']['validation']['selected']['total_r']:.3f} "
                  f"selection_metric={selection_objective(config.target)['metric']} "
                  f"selection_score={validation_score(report['partitions']['validation'], config.target):.9g} "
                  f"predictive_status={diagnostics['status']} trees_used={diagnostics['trees_used']} "
                  f"validation_qualified={report['validation_qualified']} (test not evaluated)", flush=True)
        selected = max(keys, key=lambda key: (validation_score(comparison["windows"][key]["report"]["partitions"]["validation"], config.target),
                                             comparison["windows"][key]["report"]["partitions"]["validation"]["selected"]["coverage"]))
        selected_data = comparison["windows"][selected]
        positive_validation = selected_data["report"]["partitions"]["validation"]["selected"]["total_r"] > 0
        comparison.update(selected_window_validation_only=selected, selected_directory=selected_data["directory"],
                          selection_objective=selection_objective(config.target),
                          selected_validation_score=validation_score(selected_data['report']['partitions']['validation'], config.target),
                          selected_threshold=selected_data['report']['threshold'], selected_target=config.target,
                          validation_qualified=selected_data['report']['validation_qualified'],
                          positive_validation=positive_validation, directory=str(experiment))
        (experiment / "comparison.json").write_text(json_dump(comparison), encoding="utf-8")
        (experiment / "selected.json").write_text(json_dump({"window_key": selected, "directory": selected_data["directory"],
                                                             "selection_objective": selection_objective(config.target),
                                                             "selected_validation_score": comparison['selected_validation_score'],
                                                             "threshold": comparison['selected_threshold'],
                                                             "target": config.target,
                                                             "positive_validation": positive_validation,
                                                             "validation_qualified": selected_data['report']['validation_qualified'],
                                                             "scope": {**scope, "window_keys": [selected]}}), encoding="utf-8")
        write_comparison(experiment, comparison)
        with db:
            db.execute("UPDATE swing_experiments SET status='completed',selected_resolver_id=?,report_json=? WHERE experiment_id=?",
                       (selected_data["resolver_id"],json_dump(comparison),experiment_id))
        return comparison
    except BaseException:
        if 'experiment_id' in locals():
            db.rollback()
            with db:
                db.execute("UPDATE swing_experiments SET status='failed' WHERE experiment_id=?",(experiment_id,))
        raise
    finally:
        db.close()


def write_comparison(directory, comparison):
    grid = comparison.get('representation') == 'swing-grid'
    strategy = 'swing-grid' if grid else 'swing-ts2vec'
    columns = ["window", "encoder_train_contexts", "embedded_contexts", "validation_all_count", "validation_all_r",
               "validation_selected_count", "validation_selected_r", "validation_selected_mean_r",
               "validation_selected_pf", "validation_tabular_selected_r", "selected_on_validation",
               "validation_target_score", "validation_target_mean", "validation_target_sum"]
    table = []
    for key, entry in comparison["windows"].items():
        report = entry["report"]
        val = report["partitions"]["validation"]
        row = dict(zip(columns, [key, entry["encoder_train_contexts"], entry["embedded_contexts"],
                         val["all_opportunities"]["opportunities"], val["all_opportunities"]["total_r"],
                         val["selected"]["opportunities"], val["selected"]["total_r"], val["selected"]["mean_r"],
                         val["selected"]["profit_factor_r"], report["tabular_only"]["validation"]["selected"]["total_r"],
                         key == comparison["selected_window_validation_only"]]))
        row.update(validation_target_score=validation_score(val, comparison['config']['target']),
                   validation_target_mean=val['selected'].get('mean_label'), validation_target_sum=val['selected'].get('total_label'))
        table.append(row)
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Recherche XGBoost sur grilles de swings" if grid else "# Recherche TS2Vec sur les swings", "", f"Run : `{comparison['run_id']}`.", "",
             f"Périmètre : `{json_dump(comparison['scope'])}`.", "",
             f"Cible XGBoost : `{comparison['config']['target']}`. Objectif du seuil et des fenêtres : "
             f"`{selection_objective(comparison['config']['target'])['metric']}` dans les unités de cette cible.", "",
             "Opportunités indépendantes et potentiellement simultanées. Les sommes de R ne sont pas des rendements de portefeuille.", "",
             f"Fenêtre sélectionnée **sur la validation uniquement** : `{comparison['selected_window_validation_only']}`.", "",
             f"**Seuil initial du modèle retenu : `{comparison['selected_threshold']!r}`** "
             f"(cible `{comparison['selected_target']}`).", "",
             f"Option de backtest : `--{strategy}-threshold {comparison['selected_threshold']!r}`. "
             "Sans cette option, le robot utilise déjà ce seuil enregistré dans le modèle.", "",
             f"Périodes : `{json_dump(comparison['protocol'])}` ; effectifs : `{json_dump(comparison['split']['after_purge'])}`.", "",
             "**Aucune donnée de test n'est utilisée par cette commande.** Le test 2026 est une étape séparée avec modèles figés.", "",
             "| Fenêtre | N validation retenu | Score cible retenu | R descriptif | R sans embedding |",
             "|---|---:|---:|---:|---:|"]
    if grid:
        lines[4:4] = [f"Normalisation : `{comparison.get('grid_normalization', 'swing-mean')}`.", '']
    for row in table:
        lines.append(f"| {row['window']} | {row['validation_selected_count']} | {row['validation_target_score']:.9g} | {row['validation_selected_r']:.3f} | "
                     f"{row['validation_tabular_selected_r']:.3f} |")
    lines += ['', '## Diagnostic des apprentissages', '',
              '| Fenêtre | Arbres utilisés | AUC validation | Gain d’erreur vs constante | Qualification validation |',
              '|---|---:|---:|---:|---|']
    for key, entry in comparison['windows'].items():
        report = entry['report']
        diagnostic = report.get('learning_diagnostics')
        if diagnostic is not None:
            val = diagnostic['partitions']['validation']
            auc = 'n/a' if val['auc'] is None else f"{val['auc']:.6f}"
            lines.append(f"| {key} | {diagnostic['trees_used']} | {auc} | "
                         f"{val['improvement_over_constant']:.8f} | {report['validation_qualified']} |")
    lines += ['', 'La constante est calculée sur les labels d’apprentissage. La qualification exige un gain d’erreur '
              'sur cette constante, une AUC supérieure à 0,5 pour la classification et un score de validation supérieur '
              'au label neutre (0, ou exp(0)=1 pour trailing-step-exp). '
              'Ce diagnostic ne constitue pas une preuve de rentabilité future.']
    if comparison.get('cohort_policy') == 'per-window':
        lines += ['', '## Cohortes stables par fenetre', '',
                  'Chaque fenetre apprend, choisit ses arbres et son seuil sur ses propres evenements eligibles. '
                  'Ajouter une autre duree ne change pas ces choix. Le classement utilise ensuite les memes '
                  'evenements de validation pour toutes les fenetres, avec les seuils deja figes.', '',
                  '| Fenetre | Train propre | Validation propre | Validation commune | Score cible commun |',
                  '|---|---:|---:|---:|---:|']
        for key, entry in comparison['windows'].items():
            report = entry['report']
            counts = report['own_split']
            common = report['comparison_validation']
            lines.append(f"| {key} | {counts['train']} | {counts['validation']} | {common['rows']} | "
                         f"{validation_score(common, comparison['config']['target']):.9g} |")
    lines.extend(["", "Les modèles témoins utilisent uniquement le sens, le nombre de swings et la durée du contexte, avec leur propre seuil sélectionné sur validation.",
                  "", ("Grille dense : prix / moyenne des prix des swings presents dans la fenetre - 1. Trous a zero. Une case par bougie de confirmation, sans TS2Vec, volume ni variables tabulaires supplementaires en entree du modele principal."
                         if grid else "TS2Vec et XGBoost apprennent sur la même période d'apprentissage. Les embeddings de cette période sont des reconstructions d'entraînement ; validation et test utilisent l'encodeur figé."),
                  "", "Les positions encore ouvertes et les ordres non exécutés n'ont aucun label de PnL. Les résultats qui débordent de leur période sont exclus de l'apprentissage ou de la sélection."])
    (directory / "rapport.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class SwingResolver:
    """Frozen inference for a causal swing context; never receives realized returns."""
    def __init__(self, directory, *, device="cpu", threads=2):
        import xgboost as xgb
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["version"] != 1 or self.metadata["feature_schema"] not in FEATURE_SCHEMAS:
            raise ValueError("Unsupported swing resolver")
        if hashlib.sha256((self.directory / "xgboost.json").read_bytes()).hexdigest() != self.metadata["xgboost_sha256"]:
            raise ValueError("XGBoost checkpoint hash mismatch")
        self.encoder = SwingEncoder.load(self.directory / "ts2vec.pt", device=device, threads=threads)
        if self.encoder.feature_schema != self.metadata['feature_schema']:
            raise ValueError('Resolver and encoder feature schemas differ')
        if self.encoder.model_id != self.metadata["embedding_model_id"]:
            raise ValueError("TS2Vec checkpoint hash mismatch")
        self.model = xgb.XGBClassifier() if is_binary_target(self.metadata["target"]) else xgb.XGBRegressor()
        self.model.load_model(self.directory / "xgboost.json")

    def predict(self, points, *, as_of, symbol=None, stop_buffer_pips=None):
        if as_of < at(self.metadata["available_after"]):
            raise ValueError("Resolver validation/training postdates this decision")
        interval = timedelta(seconds=self.metadata["interval_seconds"])
        points = tuple(points)
        scope = self.metadata.get("scope")
        if scope:
            if symbol is None and len(scope["symbols"]) == 1:
                symbol = scope["symbols"][0]
            if symbol is None or symbol.upper() not in scope["symbols"]:
                raise ValueError("Specify a symbol within this resolver's training scope")
        mode, value = self.metadata["window_key"].rsplit("_", 1)
        if mode == "pairs" and len(points) != 2 * int(value):
            raise ValueError("Wrong number of swing couples for this resolver")
        if mode == "duration" and any(s.confirmed_at <= as_of - timedelta(seconds=int(value)) for s in points):
            raise ValueError("Swings fall outside this resolver's duration window")
        sequence = swing_features(points, interval, as_of, feature_schema=self.metadata['feature_schema'])
        if points[-1].confirmed_at != as_of:
            raise ValueError("A decision must coincide with the final confirmed swing")
        vector = self.encoder.encode([sequence], observed_at=[as_of])[0]
        values = [points[-1].side, math.log1p(len(points)),
                  math.log1p((points[-1].confirmed_at - points[0].confirmed_at) / interval)]
        if uses_stop_distance(self.metadata['opportunity_config']):
            values.append(stop_distance_feature(self.metadata['opportunity_config'], symbol, points[-1].price, stop_buffer_pips))
        features = np.asarray([np.r_[vector, values]], dtype=np.float32)
        prediction = float(predict(self.model, features, self.metadata["target"])[0])
        return {"prediction": prediction, "target": self.metadata["target"], "threshold": self.metadata["threshold"],
                "accepted": prediction >= self.metadata["threshold"], "side": points[-1].side,
                "resolver_id": self.metadata["resolver_id"]}


def resolver_directory(directory):
    directory=Path(directory).resolve()
    if not (directory/"metadata.json").is_file():
        directory=Path(json.loads((directory/"selected.json").read_text(encoding="utf-8"))["directory"])
    return directory


def frozen_test(database, *, directory, run_id=None, start="2026-01-01", end="2026-12-31",
                output_dir=None, min_samples=30, threads=2):
    """Only transform/predict. There is deliberately no fit or threshold selection here."""
    resolved = resolver_directory(directory)
    if json.loads((resolved/'metadata.json').read_text(encoding='utf-8')).get('model_type') == 'swing-nn':
        from ultimateBot.backtester.research.swing_neural import frozen_test as frozen_neural_test
        destination = output_dir or resolved.parent/'tests'/('neural_'+uuid4().hex[:12])
        return frozen_neural_test(database, directory=resolved, run_id=run_id, start=start, end=end,
                                  output_dir=destination, min_samples=min_samples, threads=threads)
    if json.loads((resolved/'metadata.json').read_text(encoding='utf-8')).get('model_type') == 'swing-grid':
        from ultimateBot.backtester.research.swing_grid import frozen_grid_test
        return frozen_grid_test(database, directory=resolved, run_id=run_id, start=start, end=end,
                                output_dir=output_dir, min_samples=min_samples, threads=threads)
    import xgboost as xgb
    if not Path(database).is_file():
        raise FileNotFoundError("Collect the test observations into SQLite first")
    if date.fromisoformat(start)>date.fromisoformat(end) or min_samples<1:
        raise ValueError("Invalid test date bounds or minimum sample count")
    directory=resolver_directory(directory)
    resolver=SwingResolver(directory,threads=threads)
    metadata=resolver.metadata
    test_start,test_end=day_start(start),day_start(end)+timedelta(days=1)
    if test_start<at(metadata["available_after"]):
        raise ValueError("The test period must follow the frozen resolver's validation period")
    baseline_path=directory/"xgboost_tabular_only.json"
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=metadata["baseline_xgboost_sha256"]:
        raise ValueError("Baseline checkpoint hash mismatch")
    baseline=xgb.XGBClassifier() if is_binary_target(metadata["target"]) else xgb.XGBRegressor()
    baseline.load_model(baseline_path)
    training_report=json.loads((directory/"report.json").read_text(encoding="utf-8"))
    baseline_threshold=training_report["baseline_threshold"]
    db=connect(database)
    try:
        scope = metadata.get("scope", {})
        mode, value = metadata["window_key"].rsplit("_", 1)
        selection = TrainingSelection(symbols=tuple(scope["symbols"]) if scope else None,
                                      include_volume=metadata['feature_schema'] == VOLUME_FEATURE_SCHEMA,
                                      interval_seconds=metadata["interval_seconds"], window_mode=mode,
                                      durations=(int(value),) if mode == "duration" else None,
                                      pairs=(int(value),) if mode == "pairs" else None)
        # A search may have selected a size not present in the test collection.
        # Rebuild that frozen context from the stored test swings, never refit.
        from ultimateBot.backtester.research.swing_contexts import ensure_windows
        from dataclasses import replace
        base_selection = replace(selection, window_mode='both', durations=None, pairs=None)
        run, parameters, _, _ = select_collection(db, run_id, base_selection)
        run_id = run["run_id"]
        if (model_feature_schema(parameters['feature_schema'], metadata['feature_schema'] == VOLUME_FEATURE_SCHEMA)
                != metadata['feature_schema'] or parameters["swing_schema"]!=SWING_SCHEMA):
            raise ValueError("Test data uses a different swing representation")
        if parameters.get('opportunity_schema')!=OPPORTUNITY_SCHEMA or metadata.get('opportunity_schema')!=OPPORTUNITY_SCHEMA:
            raise ValueError("Test/model opportunity execution policies do not match the unrestricted collector")
        original=asdict(OpportunityConfig(**metadata["opportunity_config"]))
        current=asdict(OpportunityConfig(**parameters["config"]))
        # R-normalized labels may use another constant virtual cash unit, never a different stop/trailing policy.
        if {k:v for k,v in original.items() if k!='risk_cash'}!={k:v for k,v in current.items() if k!='risk_cash'}:
            raise ValueError("Test collection must use the frozen model's candle, stop buffer, expiry and trailing parameters")
        ensure_windows(db, run_id, (SwingWindow(mode, int(value)),), start=test_start, before=test_end,
                       symbols=scope.get('symbols'))
        predicate, scope_args = scope_predicate(scope.get("symbols"))
        rows=[dict(r) for r in db.execute(f"""SELECT c.*,e.symbol,e.side,e.status,e.confirmed_at,e.closed_at,e.pnl,e.pnl_r
            ,e.trailing_steps,e.pivot_price,e.stop_buffer_pips
            FROM swing_contexts_resolved c JOIN swing_events e USING(event_id) WHERE e.run_id=? AND c.window_key=?
            AND {predicate}
            AND c.context_end>=? AND c.context_end<? ORDER BY c.context_end,e.symbol""",
            (run_id,metadata["window_key"],*scope_args,test_start.isoformat(),test_end.isoformat()))]
        labeled=[i for i,r in enumerate(rows) if r["status"]=='closed' and r["closed_at"]<test_end.isoformat()
                 and r["pnl_r"] is not None and math.isfinite(r["pnl_r"])]
        if len(labeled)<min_samples:
            raise ValueError(f"Need {min_samples} closed test observations within the test dates; got {len(labeled)}")
        sequences=[np.frombuffer(r["features"],dtype='<f4').reshape(r["n_swings"],len(feature_names(parameters['feature_schema'])))
                   [:, :len(feature_names(metadata['feature_schema']))].copy() for r in rows]
        vectors=resolver.encoder.encode(sequences,observed_at=[at(r["context_end"]) for r in rows])
        simple=np.asarray([tabular(r,metadata["interval_seconds"],metadata['opportunity_config']) for r in rows],dtype=np.float32)
        predictions=predict(resolver.model,np.column_stack((vectors,simple)),metadata["target"])
        baseline_predictions=predict(baseline,simple,metadata["target"])
        labels=[rows[i] for i in labeled]
        selected_predictions=predictions[labeled]
        test_id=uuid4().hex
        destination=Path(output_dir).resolve() if output_dir else directory.parent/'tests'
        destination=destination/f"{run_id[:12]}_{test_id[:12]}"
        destination.mkdir(parents=True,exist_ok=False)
        report={"test_id":test_id,"resolver_id":metadata["resolver_id"],"window_key":metadata["window_key"],
                "scope": scope,
                "run_id":run_id,"test_start":start,"test_end":end,"last_observed_signal":rows[-1]["context_end"],
                "predicted_contexts":len(rows),"labeled_contexts":len(labeled),"unlabeled_or_censored":len(rows)-len(labeled),
                "selected":evaluate(labels,selected_predictions,metadata["threshold"],metadata["target"],metadata['opportunity_config']),
                "tabular_only":evaluate(labels,baseline_predictions[labeled],baseline_threshold,metadata["target"],metadata['opportunity_config']),
                "threshold":metadata["threshold"],"fit_called":False,"directory":str(destination),
                "target_definition": target_definition(metadata['target']),
                "interpretation":"Independent opportunities; no capital, margin, spread, exposure or RR risk veto."}
        report["snapshot_sha256"]=hashlib.sha256(json_dump([
            {k:r[k] for k in ('context_id','event_id','context_end','closed_at','pnl_r','trailing_steps')} for r in rows]).encode()).hexdigest()
        policy = StopSelection.from_rows(rows, metadata['opportunity_config'])
        accepted = acceptance_mask(predictions, metadata['threshold'], policy)
        base_accepted = acceptance_mask(baseline_predictions, baseline_threshold, policy)
        with db:
            db.execute("INSERT OR IGNORE INTO swing_embedding_models VALUES(?,?,?,?,?)",
                (resolver.encoder.model_id,run_id,metadata["window_key"],str(directory/'ts2vec.pt'),json_dump(resolver.encoder.metadata)))
            db.execute("INSERT OR IGNORE INTO swing_experiments(experiment_id,run_id,status,directory,parameters_json,selected_resolver_id) VALUES(?,?,'frozen_import',?,?,?)",
                (metadata["experiment_id"],run_id,str(directory.parent),json_dump(metadata["protocol"]),metadata["resolver_id"]))
            eligible=set(labeled)
            for i,(r,v,value,bv) in enumerate(zip(rows,vectors,predictions,baseline_predictions)):
                db.execute("INSERT OR IGNORE INTO swing_embeddings VALUES(?,?,?)",(r["context_id"],resolver.encoder.model_id,v.astype('<f4').tobytes()))
                db.execute("INSERT OR IGNORE INTO swing_assignments VALUES(?,?,'test',0,?)",
                           (metadata["experiment_id"],r["context_id"],int(i in eligible)))
                db.execute("INSERT OR IGNORE INTO swing_predictions VALUES(?,?,'test',?,?,?,?)",
                           (metadata["resolver_id"],r["context_id"],float(value),int(accepted[i]),float(bv),int(base_accepted[i])))
            db.execute("INSERT INTO swing_tests VALUES(?,?,?,?,?,?,?)",(test_id,metadata["resolver_id"],run_id,
                datetime.now(timezone.utc).isoformat(),str(destination),json_dump({"start":start,"end":end,"model_metadata":metadata}),json_dump(report)))
        with (destination/'predictions.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['context_id','event_id','symbol','confirmed_at','closed_at','pnl_r','trailing_steps','target_label','prediction','accepted','label_eligible'])
            writer.writeheader()
            for i,(r,value) in enumerate(zip(rows,predictions)):
                writer.writerow({**{k:r[k] for k in ('context_id','event_id','symbol','confirmed_at','closed_at','pnl_r')},
                                 'trailing_steps':r['trailing_steps'],
                                 'target_label':target_label(r,metadata['target']) if i in eligible else None,
                                 'prediction':float(value),'accepted':int(accepted[i]),'label_eligible':int(i in eligible)})
        (destination/'report.json').write_text(json_dump(report),encoding='utf-8')
        return report
    finally:
        db.close()
