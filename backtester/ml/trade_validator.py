"""XGBoost validation at order creation, before submission to the platform."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ultimateBot.backtester.embeddings.trade_dataset import dumps
from ultimateBot.backtester.embeddings.ts2vec_service import EmbeddingConfig, TS2VecService, utc
from ultimateBot.backtester.ml.trade_features import FEATURE_CONTRACT, feature_names, feature_vector, stamp_order_parameters


@dataclass(frozen=True)
class ValidationDecision:
    accepted: bool
    probability: float | None
    threshold: float
    reason: str


class TradeValidator:
    def __init__(self, directory, *, threshold=None, threads=2):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("Install validator dependencies: py -m pip install -r requirements-validator.txt") from error
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.dimensions = self.metadata["embedding_metadata"]["config"]["output_dims"]
        if (self.metadata["format_version"] != 1 or self.metadata["feature_contract"] != FEATURE_CONTRACT
                or self.metadata["decision_point"] != "order_creation"
                or self.metadata["feature_names"] != feature_names(self.dimensions)):
            raise ValueError("Unsupported trade validator feature contract")
        self.threshold = float(self.metadata["threshold"] if threshold is None else threshold)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Trade validator threshold must be finite and in [0,1]")
        path = self.directory / "model.ubj"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["model_sha256"]:
            raise ValueError("Trade validator model checksum differs from its metadata")
        self.model = XGBClassifier(n_jobs=threads)
        self.model.load_model(path)
        self.model.set_params(n_jobs=threads)
        if (self.model.n_features_in_ != len(self.metadata["feature_names"])
                or self.model.get_booster().attr("trade_feature_contract") != FEATURE_CONTRACT):
            raise ValueError("XGBoost model does not match the validator feature contract")
        self.encoder = None
        self.engine = None
        self.log_path = None
        self.stats = {"accepted": 0, "rejected": 0}

    def evaluate(self, snapshot, *, lookback, trailing, side, symbol, as_of):
        if snapshot.model_id != self.metadata["embedding_model_id"]:
            raise ValueError("Trade validator requires the same TS2Vec checkpoint as its training data")
        if utc(as_of) <= utc(datetime.fromisoformat(self.metadata["available_after"])):
            raise ValueError("Trade validator uses outcomes from this time or later (look-ahead); use a later backtest period")
        if utc(datetime.fromisoformat(snapshot.context_end)) > utc(as_of):
            raise ValueError("Trade validator embedding includes future data")
        if symbol not in self.metadata["symbols"]:
            return ValidationDecision(False, None, self.threshold, "unseen_symbol")
        vector = feature_vector(snapshot.values, lookback, trailing, side, dimensions=self.dimensions)
        probability = float(self.model.predict_proba(vector[None, :])[0, 1])
        if not math.isfinite(probability):
            raise ValueError("Trade validator produced a non-finite score")
        accepted = probability >= self.threshold
        return ValidationDecision(accepted, probability, self.threshold,
                                  "score_accepted" if accepted else "score_below_threshold")

    def bind(self, engine, encoder, *, log_path=None):
        if encoder.model_id != self.metadata["embedding_model_id"]:
            raise ValueError("Trade validator and TS2Vec model identities differ")
        if encoder.metadata["candle_interval_seconds"] != self.metadata["embedding_metadata"]["candle_interval_seconds"]:
            raise ValueError("Trade validator and TS2Vec candle intervals differ")
        self.engine, self.encoder = engine, encoder
        self.log_path = (Path(log_path) if log_path else
                         Path("C:/TradingBotData/validator_decisions") / f"{uuid4().hex}.jsonl").resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        engine.market_state.ensure_history_capacity(encoder.config.context_bars)
        engine.trade_validator = self

    def prepare_order(self, order):
        stamp_order_parameters(order, self.engine.strategy)

    def before_order(self, order, as_of, snapshot=None):
        # Called once, before submission. The platform controls subsequent fills.
        if len(self.encoder.completed_candles(self.engine.market_state, order.symbol, as_of)) < 2:
            decision = ValidationDecision(False, None, self.threshold, "insufficient_context")
        else:
            snapshot = snapshot or self.encoder.embed_market(self.engine.market_state, order.symbol, as_of=as_of)
            decision = self.evaluate(snapshot, lookback=order.strategy_lookback_bars,
                                     trailing=int(bool(order.trailing_stop_enabled)),
                                     side=1 if order.side.name == "BUY" else -1,
                                     symbol=order.symbol, as_of=as_of)
        self.stats["accepted" if decision.accepted else "rejected"] += 1
        record = {"validator_id": self.metadata["validator_id"], "order_id": order.order_id,
                  "timestamp": utc(as_of).isoformat(), "symbol": order.symbol, "decision_point": "order_creation",
                  "regression_lookback": order.strategy_lookback_bars,
                  "trailing": bool(order.trailing_stop_enabled), "side": order.side.name,
                  "score": decision.probability, "threshold": decision.threshold,
                  "accepted": decision.accepted, "reason": decision.reason}
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(dumps(record) + "\n")
        print(f"[trade-validator] order_id={order.order_id} symbol={order.symbol} "
              f"timestamp={as_of.isoformat()} accepted={decision.accepted} "
              f"score={decision.probability} threshold={self.threshold} reason={decision.reason}", flush=True)
        return decision.accepted

    def finish(self):
        summary = {"validator_id": self.metadata["validator_id"], "threshold": self.threshold,
                   "stats": self.stats, "decisions": str(self.log_path),
                   "finished_at": datetime.now(timezone.utc).isoformat()}
        self.log_path.with_suffix(".summary.json").write_text(dumps(summary), encoding="utf-8")
        print(f"[trade-validator-summary] {dumps(summary)}", flush=True)


def attach_trade_validator(engine, args):
    validator = TradeValidator(args.trade_validator, threshold=args.trade_validator_threshold,
                               threads=args.ts2vec_threads)
    config = EmbeddingConfig(context_bars=args.ts2vec_context_bars, output_dims=args.ts2vec_output_dims,
                             hidden_dims=args.ts2vec_hidden_dims, depth=args.ts2vec_depth,
                             train_iters=args.ts2vec_train_iters, seed=args.ts2vec_seed,
                             device=args.ts2vec_device, threads=args.ts2vec_threads)
    encoder = TS2VecService(args.ts2vec_model, config, candle_interval=engine.market_state.candle_interval)
    validator.bind(engine, encoder, log_path=args.trade_validator_log)
    print(f"[trade-validator-ready] model={validator.directory} threshold={validator.threshold} "
          f"available_after={validator.metadata['available_after']} log={validator.log_path}", flush=True)
    return validator
