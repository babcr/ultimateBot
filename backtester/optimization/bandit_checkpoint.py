"""Inspectable JSON checkpoints; no pickle or dynamic deserialization."""
from __future__ import annotations

from collections import Counter, deque, defaultdict
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.monitoring.trailing_stop import TrailingStopState
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide, OrderType, OrderStatus
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.reports.trade_report import TradeReport
from ultimateBot.backtester.optimization.bandit_core import (
    BanditAction, MarketContext, DecisionSnapshot, PositionLearningRecord,
)

CLASSES = {cls.__name__: cls for cls in (Tick, Candle, TrailingStopState, OrderRequest, Position,
                                         TradeReport, BanditAction, MarketContext, DecisionSnapshot, PositionLearningRecord)}
ENUMS = {cls.__name__: cls for cls in (OrderSide, OrderType, OrderStatus, CloseReason)}


def encode(value):
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "name": value.name}
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    if is_dataclass(value):
        return {"class": type(value).__name__, "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, dict):
        return {"mapping": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple, deque)):
        return {"sequence": [encode(item) for item in value], "tuple": isinstance(value, tuple)}
    if isinstance(value, float) and not isfinite(value):
        return {"nonfinite": "inf" if value > 0 else "-inf" if value < 0 else "nan"}
    return value


def decode(value):
    if not isinstance(value, dict):
        return value
    if "enum" in value:
        return ENUMS[value["enum"]][value["name"]]
    if "datetime" in value:
        return datetime.fromisoformat(value["datetime"])
    if "seconds" in value:
        return timedelta(seconds=value["seconds"])
    if "class" in value:
        return CLASSES[value["class"]](**{key: decode(item) for key, item in value["fields"].items()})
    if "mapping" in value:
        return {decode(key): decode(item) for key, item in value["mapping"]}
    if "sequence" in value:
        items = [decode(item) for item in value["sequence"]]
        return tuple(items) if value["tuple"] else items
    if "nonfinite" in value:
        return float(value["nonfinite"])
    raise ValueError("Unsupported checkpoint value")


def candle_builder_state(builder):
    return {"active": encode(builder._active), "closed": encode(builder._closed),
            "counts": encode(builder._candle_counts)}


def restore_candle_builder(builder, state):
    builder._active = decode(state["active"])
    builder._closed.clear()
    for symbol, candles in decode(state["closed"]).items():
        builder._closed[symbol] = deque(candles, maxlen=builder.lookback_candles)
    builder._swing_histories.clear()
    builder._candle_counts = decode(state["counts"])


def market_state_snapshot(market):
    return {"last_ticks": encode(market.last_ticks), "point_sizes": market._point_sizes,
            "builders": {str(interval.total_seconds()): {source: candle_builder_state(builder)
                         for source, builder in builders.items()}
                         for interval, builders in market._candle_builders_by_interval.items()}}


def restore_market_state(market, state):
    market.last_ticks = decode(state["last_ticks"])
    market._point_sizes = dict(state["point_sizes"])
    for interval, builders in state["builders"].items():
        duration = timedelta(seconds=float(interval))
        market.register_candle_interval(duration)
        for source, snapshot in builders.items():
            restore_candle_builder(market._candle_builders_by_interval[duration][source], snapshot)


STRATEGY_STATE_FIELDS = (
    "_last_closed_candle", "_lookback_by_symbol", "_stop_buffer_by_symbol", "last_analysis", "last_global_analysis",
    "_last_global_closed_candle", "_analysis_windows", "_global_analysis_windows", "_analysis_windows_by_symbol",
    "_global_analysis_windows_by_symbol", "_reason_counts", "_global_reason_counts", "_reason_counts_by_symbol",
    "_global_reason_counts_by_symbol",
)


def engine_snapshot(engine):
    return {"balance": engine.portfolio.balance, "initial_balance": engine.portfolio.initial_balance,
            "next_trade_id": engine.portfolio._next_trade_id, "positions": encode(engine.portfolio.positions),
            "orders": encode(engine.pending_orders_processor.queue._orders_by_symbol),
            "next_order_id": engine.order_builder._next_order_id,
            "strategy": {key: encode(getattr(engine.strategy, key)) for key in STRATEGY_STATE_FIELDS if hasattr(engine.strategy, key)},
            "risk_candles": candle_builder_state(engine.risk_manager.candle_builder),
            "batch_buffers": encode(engine.batcher.buffers), "batch_windows": encode(engine.batcher._analysis_windows),
            "trades": encode(engine.reports.trades), "partials": encode(engine.reports._partials),
            "seen_deals": list(engine.reports._seen_deals),
            "market_clock": encode(engine._market_clock), "trading_schedule": engine.trading_schedule.checkpoint()}


def restore_engine(engine, state):
    engine.portfolio.balance = state["balance"]
    engine.portfolio.initial_balance = state["initial_balance"]
    engine.portfolio._next_trade_id = state["next_trade_id"]
    engine.portfolio.positions = decode(state["positions"])
    engine.pending_orders_processor.queue._orders_by_symbol.clear()
    engine.pending_orders_processor.queue._orders_by_symbol.update(decode(state["orders"]))
    engine.order_builder._next_order_id = state["next_order_id"]
    for key, value in state["strategy"].items():
        decoded = decode(value)
        if isinstance(getattr(engine.strategy, key, None), Counter):
            decoded = Counter(decoded)
        elif key in ("_reason_counts_by_symbol", "_global_reason_counts_by_symbol"):
            decoded = {symbol: Counter(counts) for symbol, counts in decoded.items()}
        current = getattr(engine.strategy, key, None)
        if isinstance(current, defaultdict):
            decoded = defaultdict(current.default_factory, decoded)
        setattr(engine.strategy, key, decoded)
    restore_candle_builder(engine.risk_manager.candle_builder, state["risk_candles"])
    engine.batcher.buffers = decode(state["batch_buffers"])
    engine.batcher._analysis_windows = decode(state["batch_windows"])
    engine.reports.trades = decode(state["trades"])
    engine.reports._partials = decode(state["partials"])
    engine.reports._seen_deals = set(state["seen_deals"])
    engine._market_clock = decode(state.get("market_clock"))
    engine.trading_schedule.restore(state.get("trading_schedule", {}))
