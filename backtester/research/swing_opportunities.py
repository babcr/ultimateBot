"""Independent swing opportunities, durable causal snapshots, exact quote exits."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np
import pandas as pd

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.embeddings.trade_dataset import code_provenance
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.monitoring.trailing_stop import TrailingStopManager
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.research.swing_series import (
    FEATURE_SCHEMA, FEATURE_SCHEMAS, VOLUME_FEATURE_SCHEMA, SWING_SCHEMA, SwingSeries, SwingWindow, feature_names, swing_features,
)

OPPORTUNITY_SCHEMA="unrestricted_independent_swing_orders_with_stop_buffer_v1"


def instant(ms):
    return datetime.fromtimestamp(float(ms) / 1000, timezone.utc)


def milliseconds(at):
    return round(at.timestamp() * 1000)


def json_dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


SCHEMA = """
CREATE TABLE IF NOT EXISTS swing_runs(
 run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
 parameters_json TEXT NOT NULL, summary_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS swing_events(
 event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES swing_runs(run_id), symbol TEXT NOT NULL,
 kind TEXT NOT NULL, side INTEGER NOT NULL, pivot_at TEXT NOT NULL, pair_start TEXT NOT NULL,
 confirmed_at TEXT NOT NULL, stop_price REAL NOT NULL, pivot_price REAL, expires_at TEXT NOT NULL,
 status TEXT NOT NULL, opened_at TEXT, entry_price REAL, initial_risk REAL, quantity REAL,
 closed_at TEXT, exit_price REAL, close_reason TEXT, pnl REAL, pnl_r REAL,
 mfe_r REAL, mae_r REAL, trailing_steps INTEGER, mean_quote_volume REAL,
 swing_id TEXT, variant_index INTEGER NOT NULL DEFAULT 0, stop_buffer_pips REAL);
CREATE INDEX IF NOT EXISTS swing_events_run_time ON swing_events(run_id,confirmed_at);
CREATE VIEW IF NOT EXISTS swing_event_labels AS
 SELECT e.*,
 CASE WHEN status='closed' AND typeof(trailing_steps) IN ('integer','real')
           AND trailing_steps>=0 AND trailing_steps=CAST(trailing_steps AS INTEGER)
      THEN CASE WHEN trailing_steps>=1 THEN 1 ELSE 0 END
      ELSE NULL END AS trailing_step_1_label
 FROM swing_events e;
CREATE TABLE IF NOT EXISTS swing_contexts(
 context_id INTEGER PRIMARY KEY, event_id TEXT NOT NULL REFERENCES swing_events(event_id),
 window_key TEXT NOT NULL, information_start TEXT NOT NULL, context_end TEXT NOT NULL,
 n_swings INTEGER NOT NULL, span_seconds REAL NOT NULL, features BLOB NOT NULL,
 swings_json TEXT NOT NULL, shared_context_id INTEGER REFERENCES swing_contexts(context_id), UNIQUE(event_id,window_key));
CREATE VIEW IF NOT EXISTS swing_contexts_resolved AS
 SELECT c.context_id,c.event_id,c.window_key,c.information_start,c.context_end,c.n_swings,c.span_seconds,
 CASE WHEN c.shared_context_id IS NULL THEN c.features ELSE p.features END AS features,
 CASE WHEN c.shared_context_id IS NULL THEN c.swings_json ELSE p.swings_json END AS swings_json,
 coalesce(c.shared_context_id,c.context_id) AS payload_context_id
 FROM swing_contexts c LEFT JOIN swing_contexts p ON p.context_id=c.shared_context_id;
CREATE TABLE IF NOT EXISTS swing_embedding_models(
 model_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES swing_runs(run_id),
 window_key TEXT NOT NULL, checkpoint TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS swing_embeddings(
 context_id INTEGER NOT NULL REFERENCES swing_contexts(context_id),
 model_id TEXT NOT NULL REFERENCES swing_embedding_models(model_id), embedding BLOB NOT NULL,
 PRIMARY KEY(context_id,model_id));
CREATE TABLE IF NOT EXISTS swing_resolvers(
 resolver_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES swing_runs(run_id),
 window_key TEXT NOT NULL, model_id TEXT NOT NULL REFERENCES swing_embedding_models(model_id),
 directory TEXT NOT NULL, metadata_json TEXT NOT NULL, report_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS swing_experiments(
 experiment_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES swing_runs(run_id),
 status TEXT NOT NULL, directory TEXT NOT NULL, parameters_json TEXT NOT NULL,
 selected_resolver_id TEXT, report_json TEXT);
CREATE TABLE IF NOT EXISTS swing_assignments(
 experiment_id TEXT NOT NULL REFERENCES swing_experiments(experiment_id),
 context_id INTEGER NOT NULL REFERENCES swing_contexts(context_id),
 phase TEXT NOT NULL, encoder_training INTEGER NOT NULL, label_eligible INTEGER NOT NULL,
 PRIMARY KEY(experiment_id,context_id));
CREATE TABLE IF NOT EXISTS swing_predictions(
 resolver_id TEXT NOT NULL, context_id INTEGER NOT NULL REFERENCES swing_contexts(context_id),
 phase TEXT NOT NULL, prediction REAL NOT NULL, accepted INTEGER NOT NULL,
 baseline_prediction REAL, baseline_accepted INTEGER,
 PRIMARY KEY(resolver_id,context_id,phase));
CREATE TABLE IF NOT EXISTS swing_tests(
 test_id TEXT PRIMARY KEY, resolver_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES swing_runs(run_id),
 started_at TEXT NOT NULL, directory TEXT NOT NULL, parameters_json TEXT NOT NULL, report_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS swing_context_builds(
 run_id TEXT NOT NULL REFERENCES swing_runs(run_id), window_key TEXT NOT NULL,
 completed_before TEXT NOT NULL, metadata_json TEXT NOT NULL,
 PRIMARY KEY(run_id,window_key));
CREATE TABLE IF NOT EXISTS swing_context_build_checkpoints(
 run_id TEXT NOT NULL REFERENCES swing_runs(run_id), window_key TEXT NOT NULL,
 symbol TEXT NOT NULL, completed_before TEXT NOT NULL,
 PRIMARY KEY(run_id,window_key,symbol));
CREATE TABLE IF NOT EXISTS swing_context_coverage(
 run_id TEXT NOT NULL REFERENCES swing_runs(run_id), window_key TEXT NOT NULL,
 symbol TEXT NOT NULL, start_at TEXT NOT NULL, end_at TEXT NOT NULL,
 PRIMARY KEY(run_id,window_key,symbol,start_at,end_at));
CREATE TABLE IF NOT EXISTS swing_window_searches(
 search_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES swing_runs(run_id),
 status TEXT NOT NULL, directory TEXT NOT NULL, parameters_json TEXT NOT NULL,
 report_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS swing_run_extensions(
 extension_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES swing_runs(run_id),
 status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
 parameters_json TEXT NOT NULL, error TEXT);
"""


def connect(database):
    path = Path(database).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(SCHEMA)
    if "pivot_price" not in {row[1] for row in db.execute("PRAGMA table_info(swing_events)")}:
        # Earlier research datasets used the exact swing as stop (buffer was always zero).
        with db:
            db.execute("ALTER TABLE swing_events ADD COLUMN pivot_price REAL")
            db.execute("UPDATE swing_events SET pivot_price=stop_price")
    if 'mean_quote_volume' not in {row[1] for row in db.execute('PRAGMA table_info(swing_events)')}:
        with db:
            db.execute('ALTER TABLE swing_events ADD COLUMN mean_quote_volume REAL')
    for table, columns in {'swing_events': {'swing_id': 'TEXT', 'variant_index': 'INTEGER NOT NULL DEFAULT 0',
                                          'stop_buffer_pips': 'REAL'},
                           'swing_contexts': {'shared_context_id': 'INTEGER REFERENCES swing_contexts(context_id)'}}.items():
        present = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
        with db:
            for name, declaration in columns.items():
                if name not in present:
                    db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {declaration}')
    return db


@dataclass(frozen=True)
class OpportunityConfig:
    interval_seconds: int = 3600
    trailing_start_r: float = 1.
    order_expiration_bars: int = 2
    execution_delay_seconds: float = 0.
    risk_cash: float = 100.
    stop_buffer_pips: float = 0.
    pip_size: float = 0.
    stop_variants: int = 1
    stop_step_pips: float = 5.
    # Zero reads legacy configurations without changing their execution policy.
    max_holding_seconds: int | None = None

    def __post_init__(self):
        if (isinstance(self.interval_seconds, bool) or not isinstance(self.interval_seconds, int)
                or self.interval_seconds < 1 or 86400 % self.interval_seconds):
            raise ValueError("Candle period must divide a UTC day into whole seconds")
        if isinstance(self.order_expiration_bars, bool) or not isinstance(self.order_expiration_bars, int) or self.order_expiration_bars < 1:
            raise ValueError("Order expiration must be a positive number of candles")
        if not math.isfinite(self.trailing_start_r) or self.trailing_start_r <= 0:
            raise ValueError("Trailing activation must be finite and > 0 R")
        if not math.isfinite(self.risk_cash) or self.risk_cash <= 0:
            raise ValueError("Independent risk cash must be finite and positive")
        if not math.isfinite(self.execution_delay_seconds) or self.execution_delay_seconds < 0:
            raise ValueError("Execution delay must be finite and nonnegative")
        for name in ("stop_buffer_pips", "pip_size"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.stop_variants) is not int or self.stop_variants < 1:
            raise ValueError('stop_variants must be a positive integer (including the base stop)')
        if not math.isfinite(self.stop_step_pips) or self.stop_step_pips <= 0:
            raise ValueError('stop_step_pips must be finite and positive')
        if self.max_holding_seconds is None:
            object.__setattr__(self, 'max_holding_seconds', 14*86400 if self.stop_variants > 1 else 0)
        if type(self.max_holding_seconds) is not int or self.max_holding_seconds < 0:
            raise ValueError('max_holding_seconds must be a nonnegative integer')
        if self.stop_variants > 1 and not self.max_holding_seconds:
            raise ValueError('Variable stops require a positive maximum holding time')

    def pip_for(self, symbol):
        size = self.pip_size
        if not size:
            normalized = symbol.upper().replace("/", "")
            currencies = {"EUR", "USD", "GBP", "CAD", "CHF", "JPY", "AUD", "NZD"}
            if len(normalized) != 6 or normalized[:3] not in currencies or normalized[3:] not in currencies:
                raise ValueError("--pip-size is required for a nonstandard FX symbol when adding a stop buffer")
            size = .01 if normalized[3:] == "JPY" else .0001
        return size

    def buffer_for(self, variant_index=0):
        if type(variant_index) is not int or not 0 <= variant_index < self.stop_variants:
            raise ValueError('Stop variant index is outside the configured grid')
        return self.stop_buffer_pips + variant_index*self.stop_step_pips

    def stop_for(self, swing, symbol, variant_index=0):
        buffer = self.buffer_for(variant_index)
        return swing.price - swing.side * buffer * self.pip_for(symbol) if buffer else swing.price


def uses_stop_distance(config):
    return config.get('stop_variants', 1) > 1


def stop_distance_feature(config, symbol, pivot_price, buffer_pips=None):
    """Known at confirmation: extra structural-stop margin / current swing price."""
    settings = OpportunityConfig(**config)
    buffer = settings.stop_buffer_pips if buffer_pips is None else float(buffer_pips)
    if not math.isfinite(buffer) or buffer < 0 or not math.isfinite(pivot_price) or pivot_price <= 0:
        raise ValueError('Stop input requires a finite nonnegative buffer and a positive swing price')
    return buffer*settings.pip_for(symbol)/pivot_price


def variant_groups(rows):
    """One reversal can have multiple trades, never multiple temporal swing points."""
    grouped = {}
    for original in rows:
        row = dict(original)
        grouped.setdefault(row.get('swing_id') or row['event_id'], []).append(row)
    result = []
    for values in grouped.values():
        values.sort(key=lambda r: r.get('variant_index', 0))
        geometry = ('kind', 'pivot_at', 'pair_start', 'confirmed_at')
        if (len({r.get('variant_index', 0) for r in values}) != len(values)
                or any(any(r[k] != values[0][k] for k in geometry) for r in values)):
            raise ValueError('Inconsistent trade variants for a collected swing')
        result.append(values)
    return result


def store_context(db, event_ids, window, points, interval, as_of, feature_schema, features=None):
    """Share the large payload; keep distinct context IDs for per-trade labels."""
    if features is None:
        features = swing_features(points, interval, as_of, feature_schema=feature_schema)
    features = features.astype('<f4').tobytes()
    payload = json_dump([s.to_dict() for s in points])
    owner, inserted = None, 0
    # Legacy databases retain NOT NULL swings_json. Empty payloads resolve via owner.
    for event_id in event_ids:
        result = db.execute('''INSERT OR IGNORE INTO swing_contexts(event_id,window_key,information_start,
            context_end,n_swings,span_seconds,features,swings_json,shared_context_id) VALUES(?,?,?,?,?,?,?,?,?)''',
            (event_id, window.key, min(s.pair_start for s in points).isoformat(), as_of.isoformat(), len(points),
             (points[-1].confirmed_at-points[0].confirmed_at).total_seconds(),
             features if owner is None else b'', payload if owner is None else '', owner))
        inserted += result.rowcount
        if owner is None:
            owner = db.execute('SELECT context_id FROM swing_contexts WHERE event_id=? AND window_key=?',
                               (event_id, window.key)).fetchone()[0]
    return inserted


def read_day(path, symbol, *, include_volume=False):
    columns = ['timestamp', 'bid', 'ask'] + (['bv', 'av'] if include_volume else [])
    try:
        frame = pd.read_csv(path, sep="\t", header=None, encoding="utf-8-sig", float_precision="round_trip",
                            names=["frame", "symbol", "timestamp", "session", "sequence", "bid", "ask", "bv", "av"],
                            usecols=[0, 1, 2, 5, 6] + ([7, 8] if include_volume else []))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    if not frame.symbol.eq(symbol).all() or not frame.frame.isin(("START", "STOP", "QUOTE")).all():
        raise ValueError(f"Unexpected tick file contract: {path}")
    frame = frame.loc[frame.frame == "QUOTE", columns].copy()
    values = frame.to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or (values[:, 1:3] <= 0).any()
            or include_volume and (values[:, 3:] < 0).any()
            or (values[:, 2] < values[:, 1]).any() or (np.diff(values[:, 0]) < 0).any()):
        raise ValueError(f"Invalid or crossed/out-of-order quotes: {path}")
    if len(values) and any(instant(ms).date() != date.fromisoformat(path.stem) for ms in (values[0, 0], values[-1, 0])):
        raise ValueError(f"Quote dates do not match the daily file: {path}")
    return frame


def day_candles(frame, interval_seconds):
    if frame.empty:
        return []
    groups = frame.groupby(frame.timestamp // (interval_seconds * 1000), sort=True)
    ohlc = groups[["bid", "ask"]].agg(["first", "max", "min", "last"])
    counts = groups.size()
    sums = groups[['bv', 'av']].sum() if {'bv', 'av'} <= set(frame.columns) else None
    def volume_fields(bucket):
        if sums is None:
            return {}
        return {'tick_count': int(counts.loc[bucket]),
                'bid_volume_sum': float(sums.loc[bucket, 'bv']), 'ask_volume_sum': float(sums.loc[bucket, 'av']),
                'bid_volume_count': int(counts.loc[bucket]), 'ask_volume_count': int(counts.loc[bucket])}
    return [(Candle(instant(int(bucket) * interval_seconds * 1000),
                    *(float(row["bid", k]) for k in ("first", "max", "min", "last")), **volume_fields(bucket)),
             Candle(instant(int(bucket) * interval_seconds * 1000),
                    *(float(row["ask", k]) for k in ("first", "max", "min", "last")), **volume_fields(bucket)))
            for bucket, row in ohlc.iterrows()]


class OpportunitySimulator:
    """Trades are independent. Jump only over quotes unable to change a stop/exit.

    Setting accelerated=False is the all-tick reference, used for equivalence tests.
    Both paths call the existing monitor and trailing manager at every state change.
    """
    def __init__(self, db, config: OpportunityConfig, accelerated=True):
        self.db, self.config = db, config
        self.accelerated = accelerated
        self.pending = []
        self.positions: dict[str, Position] = {}
        self.manager, self.monitor = TrailingStopManager(), TradeMonitor()
        self.execution = ExecutionModel()
        self.next_trade_id = 1

    def add(self, event_id, symbol, swing, variant_index=0):
        self.db.execute("UPDATE swing_events SET pivot_price=?,stop_price=? WHERE event_id=?",
                        (swing.price, self.config.stop_for(swing, symbol, variant_index), event_id))
        self.pending.append((event_id, symbol, swing, variant_index))

    def process(self, quotes: np.ndarray):
        if not len(quotes):
            return
        # Existing positions precede the new independent fills, as in the real engine.
        for event_id, position in list(self.positions.items()):
            self._advance(event_id, position, quotes, 0)
        still_pending = []
        for event_id, symbol, swing, variant_index in self.pending:
            created = swing.confirmed_at
            eligible_ms = milliseconds(created + timedelta(seconds=self.config.execution_delay_seconds))
            expiry = created + timedelta(seconds=self.config.interval_seconds * self.config.order_expiration_bars)
            index = int(np.searchsorted(quotes[:, 0], eligible_ms, side="right"))
            if index == len(quotes):
                if quotes[-1, 0] >= milliseconds(expiry):
                    self.db.execute("UPDATE swing_events SET status='expired' WHERE event_id=?", (event_id,))
                else:
                    still_pending.append((event_id, symbol, swing, variant_index))
                continue
            if quotes[index, 0] >= milliseconds(expiry):
                self.db.execute("UPDATE swing_events SET status='expired' WHERE event_id=?", (event_id,))
                continue
            ms, bid, ask = quotes[index]
            tick = Tick(instant(ms), symbol, bid, ask, 0, 0)
            side = OrderSide.BUY if swing.side > 0 else OrderSide.SELL
            order = OrderRequest(self.next_trade_id, symbol, created, created,
                                 created + timedelta(seconds=self.config.execution_delay_seconds), side, 1.,
                                 expires_at=expiry)
            result = self.execution.try_execute(order, tick)
            assert result is not None
            entry = result.fill.price
            stop = self.config.stop_for(swing, symbol, variant_index)
            if stop <= 0:
                raise ValueError("The requested stop buffer produces a nonpositive stop price")
            risk = abs(entry - stop)
            quantity = self.config.risk_cash / risk if risk else 1.
            position = Position(self.next_trade_id, symbol, side, quantity, tick.timestamp, entry, stop,
                                trailing_stop_activation_price=entry + swing.side * self.config.trailing_start_r * risk,
                                trailing_stop_enabled=True)
            if self.config.max_holding_seconds:
                position.close_after = tick.timestamp+timedelta(seconds=self.config.max_holding_seconds)
            self.next_trade_id += 1
            self.positions[event_id] = position
            self.db.execute("""UPDATE swing_events SET status='open',opened_at=?,entry_price=?,initial_risk=?,quantity=?
                               WHERE event_id=?""", (tick.timestamp.isoformat(), entry, risk, quantity, event_id))
            # No risk veto: even a stop already crossed at the fill yields a real
            # observed immediate loss. A zero-distance R stays NULL, never invented.
            request = self.monitor.check(position, tick)
            if request is not None:
                self._close(event_id, position, request)
                continue
            self.manager.prepare_position(position)
            self._advance(event_id, position, quotes, index + 1)
        self.pending = still_pending

    def _advance(self, event_id, position, quotes, index):
        direction = 1 if position.side == OrderSide.BUY else -1
        prices = quotes[:, 1 if direction > 0 else 2]
        while index < len(quotes):
            end = index
            if self.accelerated:
                state = position.trailing_stop_state
                threshold = state.activation_price + direction * ((state.steps + 1) if state.active else 0) * state.initial_stop_distance
                # Conservative ULP envelope admits harmless extra callbacks near a step.
                tolerance = 8 * np.maximum(np.spacing(np.abs(prices[index:])), math.ulp(abs(threshold))) if state.active else 0.
                mask = (direction * (prices[index:] - position.stop_loss) <= 0) | (direction * (prices[index:] - threshold) >= -tolerance)
                if position.close_after is not None:
                    mask |= quotes[index:, 0] >= milliseconds(position.close_after)
                hits = np.flatnonzero(mask)
                end = index + int(hits[0]) if len(hits) else len(quotes) - 1
            segment = direction * (prices[index:end + 1] - position.entry_price) * position.quantity
            position.max_favorable_pnl = max(position.max_favorable_pnl, float(segment.max()))
            position.max_adverse_pnl = min(position.max_adverse_pnl, float(segment.min()))
            ms, bid, ask = quotes[end]
            tick = Tick(instant(ms), position.symbol, bid, ask, 0, 0)
            request = self.monitor.check(position, tick)
            if request is None:
                self.manager.update(position, tick)
                request = self.monitor.check(position, tick)
            if request is not None:
                self._close(event_id, position, request)
                return
            index = end + 1

    def _close(self, event_id, position, request):
        direction = 1 if position.side == OrderSide.BUY else -1
        state = position.trailing_stop_state
        initial_distance = state.initial_stop_distance if state else abs(position.entry_price-position.stop_loss)
        cash_risk = initial_distance * position.quantity
        pnl = direction * (request.price-position.entry_price) * position.quantity
        self.db.execute("""UPDATE swing_events SET status='closed',closed_at=?,exit_price=?,close_reason=?,
            pnl=?,pnl_r=?,mfe_r=?,mae_r=?,trailing_steps=? WHERE event_id=?""",
            (request.timestamp.isoformat(),request.price,request.reason.name,pnl,pnl/cash_risk if cash_risk else None,
             position.max_favorable_pnl/cash_risk if cash_risk else None,
             position.max_adverse_pnl/cash_risk if cash_risk else None,state.steps if state else 0,event_id))
        del self.positions[event_id]

    def finish(self, end_exclusive: datetime):
        for event_id, symbol, swing, variant_index in self.pending:
            expiry = swing.confirmed_at + timedelta(seconds=self.config.interval_seconds * self.config.order_expiration_bars)
            status = "expired" if expiry <= end_exclusive else "pending_at_end"
            self.db.execute("UPDATE swing_events SET status=? WHERE event_id=?", (status, event_id))
        # Open positions deliberately have no realized PnL, even when their orders would have expired.


def existing_collection(database, run_id):
    """Read-only metadata lookup, also used by the append CLI preview."""
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError('The database containing the existing run does not exist')
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        row = db.execute('SELECT * FROM swing_runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None:
            raise ValueError(f'Unknown collection run_id: {run_id}')
        if row['status'] != 'completed':
            raise ValueError('Adding symbols requires a completed collection; wait for the running collector to finish')
        parameters = json.loads(row['parameters_json'])
        if (parameters.get('opportunity_schema') != OPPORTUNITY_SCHEMA
                or parameters.get('swing_schema') != SWING_SCHEMA or parameters.get('feature_schema') not in FEATURE_SCHEMAS
                or parameters.get('risk_management_filters') is not False):
            raise ValueError('Cannot extend an incompatible swing collection')
        return dict(row), parameters
    finally:
        db.close()


def collect(database, *, history, symbols, start: date | None = None, end: date | None = None, config=None,
            windows=None, accelerated=None, run_id=None, include_volume=None, store_contexts=None):
    original, original_parameters = (existing_collection(database, run_id) if run_id is not None else (None, None))
    if original is not None:
        stored_contexts = original_parameters.get('store_contexts', True)
        if store_contexts is not None and store_contexts != stored_contexts:
            raise ValueError('Adding symbols must preserve the context storage setting')
        store_contexts = stored_contexts
        stored_volume = original_parameters['feature_schema'] == VOLUME_FEATURE_SCHEMA
        if include_volume is not None and bool(include_volume) != stored_volume:
            raise ValueError('Adding symbols must preserve the swing volume feature setting')
        include_volume = stored_volume
        stored_config = OpportunityConfig(**original_parameters['config'])
        stored_windows = tuple(SwingWindow(**w) for w in original_parameters['windows'])
        stored_start, stored_end = (date.fromisoformat(original_parameters[k]) for k in ('start', 'end'))
        if ((config is not None and config != stored_config)
                or (windows is not None and set(windows) != set(stored_windows))
                or (start is not None and start != stored_start) or (end is not None and end != stored_end)):
            raise ValueError('Adding symbols must preserve the existing dates, candle interval, windows, stop and trailing parameters')
        config, windows, start, end = stored_config, stored_windows, stored_start, stored_end
    config = config if config is not None else OpportunityConfig(max_holding_seconds=14*86400)
    if store_contexts is None:
        store_contexts = False
    if type(store_contexts) is not bool:
        raise ValueError('store_contexts must be a boolean or None')
    include_volume = bool(include_volume)
    feature_schema = VOLUME_FEATURE_SCHEMA if include_volume else FEATURE_SCHEMA
    windows = tuple(windows) if windows is not None else (SwingWindow('duration', 86400), SwingWindow('pairs', 8))
    accelerated = (original_parameters.get('accelerated', True) if original_parameters else True) if accelerated is None else accelerated
    if start is None or end is None:
        raise ValueError('A new collection requires start and end dates')
    if start > end or not symbols:
        raise ValueError("Require symbols and chronological date bounds")
    symbols = tuple(dict.fromkeys(s.strip().upper() for s in symbols))
    if not windows or any(not s for s in symbols):
        raise ValueError('Require nonempty symbols and swing windows')
    previous_symbols = list(original_parameters['symbols']) if original_parameters else []
    already_collected = [s for s in symbols if s in previous_symbols]
    symbols = tuple(s for s in symbols if s not in previous_symbols)
    if not symbols:
        print(f'[swing-collect] run={run_id} already_collected={already_collected} nothing_to_add', flush=True)
        return {**json.loads(original['summary_json'] or '{}'), 'run_id': run_id,
                'added_symbols': [], 'skipped_existing_symbols': already_collected}
    files = {}
    for symbol in symbols:
        files[symbol] = sorted(p for p in (Path(history) / symbol).glob("*/*/*.tsv")
                               if start <= date.fromisoformat(p.stem) <= end)
        if not files[symbol]:
            raise ValueError(f"No tick files for {symbol} in the requested dates")
    manifest = [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for paths in files.values() for p in paths]
    run_id = run_id if original is not None else uuid4().hex
    parameters = {"config": asdict(config), "windows": [asdict(w) for w in windows], "symbols": list(symbols),
                  "start": str(start), "end": str(end), "swing_schema": SWING_SCHEMA,
                  "opportunity_schema":OPPORTUNITY_SCHEMA,"risk_management_filters":False,
                  "feature_schema": feature_schema, "feature_names": feature_names(feature_schema),
                  "store_contexts": store_contexts,
                  "independent_opportunities": True, "accelerated": accelerated,
                  "source_files": manifest, "source_fingerprint": hashlib.sha256(json_dump(manifest).encode()).hexdigest(),
                  "code_provenance": code_provenance()}
    db = connect(database)
    counts = Counter(json.loads(original['summary_json'] or '{}').get('counts', {})) if original else Counter()
    claimed = False
    extension_id = uuid4().hex if original else None
    try:
        with db:
            if original is None:
                db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES(?,?,'running',?)",
                           (run_id, datetime.now(timezone.utc).isoformat(), json_dump(parameters)))
            else:
                present = db.execute(f"SELECT 1 FROM swing_events WHERE run_id=? AND symbol IN ({','.join('?' for _ in symbols)}) LIMIT 1",
                                     (run_id, *symbols)).fetchone()
                if present:
                    raise ValueError('A requested symbol already has observations outside the run symbol list; refusing to overwrite them')
                result = db.execute("""UPDATE swing_runs SET status='running',finished_at=NULL,error=NULL
                    WHERE run_id=? AND status='completed' AND parameters_json=?""", (run_id, original['parameters_json']))
                if result.rowcount != 1:
                    raise ValueError('The run changed or another collector is already extending it; retry after it completes')
                db.execute("""INSERT INTO swing_run_extensions(extension_id,run_id,status,started_at,parameters_json)
                    VALUES(?,?,'running',?,?)""", (extension_id, run_id, datetime.now(timezone.utc).isoformat(), json_dump(parameters)))
        claimed = True
        print(f'[swing-collect] run={run_id} context_storage={"enabled" if store_contexts else "deferred"}', flush=True)
        interval = timedelta(seconds=config.interval_seconds)
        for symbol in symbols:
            # Detection only needs adjacent candles and the latest swing type.
            # Keep two points when window construction is deferred.
            tracking = tuple(windows) if store_contexts else (SwingWindow('pairs', 1),)
            series = SwingSeries(interval, tracking, include_volume=include_volume)
            sim = OpportunitySimulator(db, config, accelerated)
            total_ticks = 0
            for day_index, path in enumerate(files[symbol]):
                frame = read_day(path, symbol, include_volume=True) if include_volume else read_day(path, symbol)
                total_ticks += len(frame)
                # No feature depends on subsequent candles/quotes: append one closed pair at a time.
                with db:
                    for bid, ask in day_candles(frame, config.interval_seconds):
                        series.append(bid,ask)
                        for swing in series.last_candidates:
                            base_id=f"{run_id}:{symbol}:{swing.event_key}"
                            expiry=swing.confirmed_at+interval*config.order_expiration_bars
                            event_ids = []
                            for variant in range(config.stop_variants):
                                event_id = base_id if variant == 0 else f'{base_id}:stop{variant}'
                                event_ids.append(event_id)
                                db.execute("""INSERT INTO swing_events(event_id,run_id,symbol,kind,side,pivot_at,pair_start,
                                    confirmed_at,stop_price,expires_at,status,mean_quote_volume,swing_id,variant_index,stop_buffer_pips)
                                    VALUES(?,?,?,?,?,?,?,?,?,?,'pending',?,?,?,?)""",
                                    (event_id,run_id,symbol,swing.kind,swing.side,swing.pivot_at.isoformat(),
                                     swing.pair_start.isoformat(),swing.confirmed_at.isoformat(),swing.price,expiry.isoformat(),
                                     swing.mean_quote_volume, f'{symbol}:{swing.event_key}' if config.stop_variants > 1 else None,
                                     variant,config.buffer_for(variant) if config.stop_variants > 1 else None))
                                sim.add(event_id,symbol,swing,variant)
                            for window in windows if store_contexts else ():
                                points=series.context(window,swing.confirmed_at)
                                if points is None:
                                    counts[window.key+":warmup_or_ambiguous"]+=1
                                    continue
                                store_context(db,event_ids,window,points,interval,swing.confirmed_at,feature_schema)
                    sim.process(frame[['timestamp', 'bid', 'ask']].to_numpy(dtype=np.float64))
                if day_index % 20 == 0 or day_index == len(files[symbol]) - 1:
                    print(f"[swing-collect] run={run_id} symbol={symbol} day={path.stem} "
                          f"files={day_index+1}/{len(files[symbol])} ticks={total_ticks} "
                          f"swings={series.counts['high']+series.counts['low']} open={len(sim.positions)}", flush=True)
            with db:
                sim.finish(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
            counts.update({f"{symbol}:{key}": value for key, value in series.counts.items()})
        statuses = dict(db.execute("SELECT status,count(*) FROM swing_events WHERE run_id=? GROUP BY status", (run_id,)))
        context_counts = dict(db.execute("""SELECT c.window_key,count(*) FROM swing_contexts c
            JOIN swing_events e USING(event_id) WHERE e.run_id=? GROUP BY c.window_key""", (run_id,)))
        summary = {"run_id": run_id, "statuses": statuses, "contexts": context_counts, "counts": dict(counts),
                   "store_contexts": store_contexts}
        if original is not None:
            summary.update(added_symbols=list(symbols), skipped_existing_symbols=already_collected)
            combined_files = original_parameters['source_files'] + manifest
            parameters = {**original_parameters, 'symbols': previous_symbols+list(symbols), 'source_files': combined_files,
                'source_fingerprint': hashlib.sha256(json_dump(combined_files).encode()).hexdigest(),
                'extension_ids': original_parameters.get('extension_ids', [])+[extension_id]}
        with db:
            completed_at = datetime.now(timezone.utc).isoformat()
            coverage_start = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).isoformat()
            coverage_end = datetime.combine(end+timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).isoformat()
            db.executemany('''INSERT OR IGNORE INTO swing_context_coverage
                (run_id,window_key,symbol,start_at,end_at) VALUES(?,?,?,?,?)''',
                [(run_id, window.key, symbol, coverage_start, coverage_end)
                 for window in (windows if store_contexts else ()) for symbol in symbols])
            db.execute("UPDATE swing_runs SET status='completed',finished_at=?,summary_json=?,parameters_json=?,error=NULL WHERE run_id=?",
                       (completed_at, json_dump(summary), json_dump(parameters), run_id))
            if extension_id is not None:
                db.execute("UPDATE swing_run_extensions SET status='completed',finished_at=? WHERE extension_id=?", (completed_at, extension_id))
        return summary
    except BaseException as error:
        db.rollback()
        if claimed:
            with db:
                if original is None:
                    db.execute("UPDATE swing_runs SET status='failed',finished_at=?,error=? WHERE run_id=?",
                               (datetime.now(timezone.utc).isoformat(), str(error), run_id))
                else:
                    # Only this extension's new symbols are removed on failure.
                    # Previously collected events, contexts and models stay intact.
                    placeholders = ','.join('?' for _ in symbols)
                    db.execute(f"""DELETE FROM swing_contexts WHERE event_id IN
                        (SELECT event_id FROM swing_events WHERE run_id=? AND symbol IN ({placeholders}))""", (run_id, *symbols))
                    db.execute(f'DELETE FROM swing_events WHERE run_id=? AND symbol IN ({placeholders})', (run_id, *symbols))
                    db.execute("""UPDATE swing_runs SET status='completed',finished_at=?,error=?,summary_json=?,parameters_json=?
                        WHERE run_id=?""", (original['finished_at'], original['error'], original['summary_json'], original['parameters_json'], run_id))
                    db.execute("UPDATE swing_run_extensions SET status='failed',finished_at=?,error=? WHERE extension_id=?",
                               (datetime.now(timezone.utc).isoformat(), str(error), extension_id))
        raise
    finally:
        db.close()
