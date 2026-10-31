"""Durable order-creation snapshots, fill linkage and close-only labels in SQLite."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from ultimateBot.backtester.embeddings.ts2vec_service import DEFAULT_DATA_DIRECTORY, utc


DEFAULT_DATABASE = DEFAULT_DATA_DIRECTORY / "trade_embeddings.sqlite3"


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def dumps(value):
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    sources = [*sorted((root / "backtester").rglob("*.py")), root / "run_backtest.py"]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "bot_source_sha256": digest.hexdigest(), "ts2vec_commit": git("-C", "ts2vec", "rev-parse", "HEAD")}


SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_schema (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO dataset_schema VALUES (1);
CREATE TABLE IF NOT EXISTS embedding_models (
    model_id TEXT PRIMARY KEY,
    checkpoint_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json))
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    campaign_id TEXT,
    adaptive_seed INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted')),
    model_id TEXT REFERENCES embedding_models(model_id),
    summary_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS run_parameters (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    section TEXT NOT NULL,
    parameters_json TEXT NOT NULL CHECK(json_valid(parameters_json)),
    PRIMARY KEY(run_id, section)
);
CREATE TABLE IF NOT EXISTS run_symbols (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    symbol TEXT NOT NULL,
    PRIMARY KEY(run_id, symbol)
);
CREATE TABLE IF NOT EXISTS pending_trade_features (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    trade_id INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL CHECK(json_valid(snapshot_json)),
    PRIMARY KEY(run_id, trade_id)
);
CREATE TABLE IF NOT EXISTS order_feature_snapshots (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    order_id INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL CHECK(json_valid(snapshot_json)),
    filled_trade_id INTEGER,
    PRIMARY KEY(run_id, order_id)
);
CREATE TABLE IF NOT EXISTS closed_trades (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    trade_id INTEGER NOT NULL,
    model_id TEXT NOT NULL REFERENCES embedding_models(model_id),
    symbol TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    context_start TEXT NOT NULL,
    context_end TEXT NOT NULL,
    context_bars INTEGER NOT NULL CHECK(context_bars >= 2),
    embedding_json TEXT NOT NULL CHECK(json_valid(embedding_json)),
    embedding_dims INTEGER NOT NULL,
    regression_lookback INTEGER NOT NULL CHECK(regression_lookback >= 2),
    trailing INTEGER NOT NULL CHECK(trailing IN (0,1)),
    side INTEGER NOT NULL CHECK(side IN (-1,1)),
    strategy_configuration_id INTEGER,
    timeframe_minutes REAL NOT NULL,
    global_validation INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    initial_stop_loss REAL,
    initial_take_profit REAL,
    trailing_activation_price REAL,
    quantity REAL NOT NULL,
    profit_multiplier REAL NOT NULL,
    account_currency TEXT NOT NULL,
    initial_risk REAL,
    exit_price REAL NOT NULL,
    pnl REAL NOT NULL,
    pnl_r REAL,
    won INTEGER NOT NULL CHECK(won IN (0,1)),
    close_reason TEXT NOT NULL,
    feature_point TEXT NOT NULL DEFAULT 'fill',
    order_id INTEGER,
    order_created_at TEXT,
    signal_at TEXT,
    feature_observed_at TEXT,
    PRIMARY KEY(run_id, trade_id)
);
CREATE INDEX IF NOT EXISTS closed_trades_model_time ON closed_trades(model_id, opened_at);
CREATE TABLE IF NOT EXISTS order_signal_contexts (
    run_id TEXT NOT NULL,
    order_id INTEGER NOT NULL,
    context_json TEXT NOT NULL CHECK(json_valid(context_json)),
    PRIMARY KEY(run_id, order_id),
    FOREIGN KEY(run_id, order_id) REFERENCES order_feature_snapshots(run_id, order_id)
);
CREATE INDEX IF NOT EXISTS runs_campaign_seed ON runs(campaign_id, adaptive_seed);
CREATE VIEW IF NOT EXISTS training_samples AS
    SELECT t.*, r.adaptive_seed, r.campaign_id, r.status AS run_status
    FROM closed_trades t JOIN runs r ON r.run_id=t.run_id;
"""


def ensure_capture_schema(connection):
    """Add order-time provenance without relabelling legacy fill-time captures."""
    connection.executescript(SCHEMA)
    columns = {r[1] for r in connection.execute("PRAGMA table_info(closed_trades)")}
    additions = {"feature_point": "TEXT NOT NULL DEFAULT 'fill'", "order_id": "INTEGER",
                 "order_created_at": "TEXT", "signal_at": "TEXT", "feature_observed_at": "TEXT"}
    with connection:
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE closed_trades ADD COLUMN {name} {declaration}")


class TradeDataset:
    def __init__(self, database, encoder, *, parameters, symbols=(), seed=None,
                 campaign_id=None, run_id=None):
        self.path = Path(database).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.run_id = run_id or uuid4().hex
        self.engine = None
        self.closed = False
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        ensure_capture_schema(self.connection)
        if [r[0] for r in self.connection.execute("SELECT version FROM dataset_schema")] != [1]:
            self.connection.close()
            raise ValueError("Unsupported trade dataset schema")
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO runs(run_id,campaign_id,adaptive_seed,started_at,status) VALUES(?,?,?,?, 'running')",
                    (self.run_id, campaign_id, seed, datetime.now(timezone.utc).isoformat()))
                self.connection.executemany("INSERT INTO run_parameters VALUES(?,?,?)",
                    [(self.run_id, name, dumps(value)) for name, value in parameters.items()])
                self.connection.executemany("INSERT INTO run_symbols VALUES(?,?)",
                    [(self.run_id, symbol) for symbol in dict.fromkeys(symbols)])
            if encoder.model_id is not None:
                self._register_model()
        except BaseException:
            self.connection.close()
            raise

    def _register_model(self):
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO embedding_models VALUES(?,?,?)",
                (self.encoder.model_id, str(self.encoder.checkpoint), dumps(self.encoder.metadata)))
            self.connection.execute("UPDATE runs SET model_id=? WHERE run_id=?",
                                    (self.encoder.model_id, self.run_id))

    def bind(self, engine):
        self.engine = engine
        engine.market_state.ensure_history_capacity(self.encoder.config.context_bars)
        engine.trade_dataset = self
        engine.reports.add_trade_listener(self.on_trade_closed)

    def prepare_order(self, order):
        # Capture the configuration that generated this order, even if adaptation
        # changes while a pending limit order waits for its fill.
        from ultimateBot.backtester.ml.trade_features import stamp_order_parameters
        stamp_order_parameters(order, self.engine.strategy)

    def snapshot_for_order(self, order, as_of):
        if self.encoder.model is None:
            market = self.engine.market_state
            symbols = sorted(set(market.last_ticks) | {order.symbol})
            histories = {symbol: self.encoder.completed_candles(market, symbol, as_of)
                         for symbol in symbols}
            self.encoder.fit_histories(histories, as_of=as_of, source="first_order_creation_bootstrap")
            self._register_model()
        return self.encoder.embed_market(self.engine.market_state, order.symbol, as_of=as_of)

    def on_order_created(self, order, as_of, snapshot=None):
        snapshot = snapshot or self.snapshot_for_order(order, as_of)
        if order.strategy_lookback_bars is None:
            raise ValueError("Missing regression lookback on created order")
        record = {
            "run_id": self.run_id, "order_id": order.order_id, "model_id": snapshot.model_id,
            "symbol": order.symbol, "feature_point": "order_creation",
            "order_created_at": utc(as_of).isoformat(), "feature_observed_at": utc(as_of).isoformat(),
            "signal_at": utc(order.created_at).isoformat(),
            "context_start": snapshot.context_start, "context_end": snapshot.context_end,
            "context_bars": snapshot.context_bars, "embedding_json": dumps(snapshot.values),
            "embedding_dims": len(snapshot.values), "regression_lookback": order.strategy_lookback_bars,
            "trailing": int(bool(order.trailing_stop_enabled)), "side": 1 if order.side.name == "BUY" else -1,
            "strategy_configuration_id": order.strategy_configuration_id,
            "timeframe_minutes": order.strategy_timeframe_minutes,
            "global_validation": int(bool(order.strategy_use_global_timeframe_validation)),
        }
        with self.connection:
            self.connection.execute("INSERT INTO order_feature_snapshots(run_id,order_id,snapshot_json) VALUES(?,?,?)",
                                    (self.run_id, order.order_id, dumps(record)))
            if order.strategy_entry_context:
                self.connection.execute("INSERT INTO order_signal_contexts VALUES(?,?,?)",
                                        (self.run_id, order.order_id, dumps(order.strategy_entry_context)))

    def on_position_open(self, position, tick, order):
        row = self.connection.execute("SELECT snapshot_json FROM order_feature_snapshots WHERE run_id=? AND order_id=?",
                                      (self.run_id, order.order_id)).fetchone()
        if row is None:
            raise ValueError(f"Filled order {order.order_id} has no persisted creation-time snapshot")
        record = json.loads(row[0])
        initial_risk = (abs(position.entry_price - position.stop_loss) * position.quantity * position.profit_multiplier
                        if position.stop_loss is not None else None)
        record.update({
            "trade_id": position.trade_id, "opened_at": utc(position.opened_at).isoformat(),
            "entry_price": position.entry_price, "initial_stop_loss": position.stop_loss,
            "initial_take_profit": position.take_profit,
            "trailing_activation_price": position.trailing_stop_activation_price,
            "quantity": position.quantity, "profit_multiplier": position.profit_multiplier,
            "account_currency": position.account_currency, "initial_risk": initial_risk,
        })
        with self.connection:
            self.connection.execute("INSERT INTO pending_trade_features VALUES(?,?,?)",
                                    (self.run_id, position.trade_id, dumps(record)))
            self.connection.execute("UPDATE order_feature_snapshots SET filled_trade_id=? WHERE run_id=? AND order_id=?",
                                    (position.trade_id, self.run_id, order.order_id))

    def on_trade_closed(self, report):
        if not report.is_complete:
            return
        row = self.connection.execute("SELECT snapshot_json FROM pending_trade_features WHERE run_id=? AND trade_id=?",
                                      (self.run_id, report.trade_id)).fetchone()
        if row is None:
            if self.connection.execute("SELECT 1 FROM closed_trades WHERE run_id=? AND trade_id=?",
                                       (self.run_id, report.trade_id)).fetchone():
                return  # Idempotent final-close notification.
            raise ValueError(f"Trade {report.trade_id} closed without a persisted fill snapshot")
        record = json.loads(row[0])
        if record["symbol"] != report.symbol or record["opened_at"] != utc(report.opened_at).isoformat():
            raise ValueError("Closed trade does not match its fill snapshot")
        if not math.isfinite(report.pnl):
            raise ValueError("A trade label must have a finite realized PnL")
        record.update(closed_at=utc(report.closed_at).isoformat(), exit_price=report.exit_price,
                      pnl=report.pnl, won=int(report.pnl > 0), close_reason=report.close_reason.name,
                      pnl_r=report.pnl / record["initial_risk"] if record["initial_risk"] else None)
        columns = ",".join(record)
        placeholders = ",".join("?" for _ in record)
        with self.connection:
            self.connection.execute(f"INSERT INTO closed_trades({columns}) VALUES({placeholders})", tuple(record.values()))
            self.connection.execute("DELETE FROM pending_trade_features WHERE run_id=? AND trade_id=?",
                                    (self.run_id, report.trade_id))

    def finish(self, status="completed", error=None):
        if self.closed:
            return
        summary = dict(self.connection.execute(
            "SELECT COUNT(*) AS closed_trades, COALESCE(SUM(pnl),0) AS pnl FROM closed_trades WHERE run_id=?",
            (self.run_id,)).fetchone())
        summary["unclosed_fills"] = self.connection.execute(
            "SELECT COUNT(*) FROM pending_trade_features WHERE run_id=?", (self.run_id,)).fetchone()[0]
        summary["captured_orders"] = self.connection.execute(
            "SELECT COUNT(*) FROM order_feature_snapshots WHERE run_id=?", (self.run_id,)).fetchone()[0]
        if self.engine is not None:
            summary.update(final_balance=self.engine.portfolio.balance,
                           open_positions=len(self.engine.portfolio.positions),
                           last_market_time=self.engine._market_clock)
        try:
            with self.connection:
                self.connection.execute("UPDATE runs SET finished_at=?,status=?,summary_json=?,error=? WHERE run_id=?",
                    (datetime.now(timezone.utc).isoformat(), status, dumps(summary),
                     str(error) if error is not None else None, self.run_id))
        finally:
            self.connection.close()
            self.closed = True
        print(f"[trade-dataset] run_id={self.run_id} status={status} "
              f"closed={summary['closed_trades']} unclosed={summary['unclosed_fills']} database={self.path}", flush=True)


def attach_trade_dataset(engine, args, encoder=None):
    from ultimateBot.backtester.embeddings.ts2vec_service import EmbeddingConfig, TS2VecService
    config = EmbeddingConfig(context_bars=args.ts2vec_context_bars, output_dims=args.ts2vec_output_dims,
                             hidden_dims=args.ts2vec_hidden_dims, depth=args.ts2vec_depth,
                             train_iters=args.ts2vec_train_iters, seed=args.ts2vec_seed,
                             device=args.ts2vec_device, threads=args.ts2vec_threads)
    if encoder is None:
        encoder = TS2VecService(args.ts2vec_model, config, candle_interval=engine.market_state.candle_interval)
    parameters = {
        "arguments": vars(args), "regression_rules": engine.strategy.regression_rules,
        "strategy_config": engine.strategy.config,
        "risk": engine.risk_manager.risk_config, "volatility": engine.risk_manager.volatility_config,
        "schedule": engine.trading_schedule.config, "embedding": config,
        "adaptive": engine.adaptive_optimizer.config if engine.adaptive_optimizer else None,
        "orders": {"expiration_bars": engine.order_builder.expiration_bars,
                   "execution_delay": engine.order_builder.execution_delay,
                   "candle_interval": engine.order_builder.candle_interval},
        "code": code_provenance(),
        "capture": {"feature_point": "order_creation", "clock": "engine_decision_clock", "version": 1},
        "validator": ({"metadata": engine.trade_validator.metadata, "threshold": engine.trade_validator.threshold,
                       "log_path": engine.trade_validator.log_path} if engine.trade_validator is not None else None),
    }
    dataset = TradeDataset(args.trade_database, encoder, parameters=parameters, symbols=args.symbols,
                           seed=args.adaptive_seed, campaign_id=args.dataset_campaign_id,
                           run_id=args.dataset_run_id)
    dataset.bind(engine)
    print(f"[trade-dataset] run_id={dataset.run_id} database={dataset.path}", flush=True)
    return dataset
