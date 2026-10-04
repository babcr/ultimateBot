from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import (
    LIMIT_ENTRY_STYLES,
    STOP_ENTRY_STYLES,
    decide_market_structure,
    resolve_structure_rules,
)


class StructureStrategy(Strategy):
    """Strategy that emits signals from the shared market-structure engine only."""

    def __init__(self, structure_rules: dict[str, Any] | None = None, minimum_candles: int | None = None):
        self.analysis_interval = timedelta(hours=1)
        self.structure_rules = resolve_structure_rules(structure_rules)
        self.minimum_candles = max(
            30,
            int(minimum_candles or self.structure_rules["level_lookback_bars"]),
        )
        self._last_closed_candle: dict[str, datetime] = {}

    def on_market_data(self, batch: TickBatch, context: StrategyContext) -> list[Signal]:
        closed_candles = context.market_state.get_candles(batch.symbol, include_active=False)
        if len(closed_candles) < self.minimum_candles:
            return []

        latest_closed = closed_candles[-1]
        if self._last_closed_candle.get(batch.symbol) == latest_closed.start_time:
            return []
        self._last_closed_candle[batch.symbol] = latest_closed.start_time

        candle_frame = pd.DataFrame(
            {
                "time": candle.start_time,
                "open": candle.open_price,
                "high": candle.high_price,
                "low": candle.low_price,
                "close": candle.close_price,
            }
            for candle in closed_candles
        )
        point_size = max(context.market_state.get_point_size(batch.symbol), 1e-9)
        decision, _analysis = decide_market_structure(
            candles=candle_frame,
            point_size=point_size,
            config=self.structure_rules,
        )
        if decision is None:
            return []

        side = SignalSide.BUY if decision.side == "buy" else SignalSide.SELL
        return [
            Signal(
                symbol=batch.symbol,
                timestamp=batch.ticks[-1].timestamp,
                side=side,
                confidence=decision.confidence,
                entry_mode=self._entry_mode_for_style(decision.entry_style),
                protective_level_price=decision.protective_level_price,
                protective_level_quality=decision.protective_level_quality,
            )
        ]

    def _entry_mode_for_style(self, entry_style: str | None) -> SignalEntryMode:
        if entry_style in LIMIT_ENTRY_STYLES:
            return SignalEntryMode.LIMIT
        if entry_style in STOP_ENTRY_STYLES:
            return SignalEntryMode.STOP
        return SignalEntryMode.MARKET