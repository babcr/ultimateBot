from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from typing import Any

import pandas as pd

from ultimateBot.backtester.config.backtest_config import DEFAULT_CANDLE_INTERVAL, normalize_candle_interval
from ultimateBot.backtester.market.candle_builder import Candle, PriceSource
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import (
    SWING_PRICE_SOURCES, _collect_swings, _normalize_swings, _rolling_atr,
    merge_structure_reversal_swings, resolve_structure_rules,
)


STRUCTURE_PRICE_SOURCES: tuple[PriceSource, PriceSource] = ("bid", "ask")


class SwingStrategy(Strategy):
    """Shared closed-candle input and swing extraction for both decision engines."""

    minimum_lookback_bars = 2

    def __init__(
        self,
        structure_rules: dict[str, Any],
        minimum_candles: int | None = None,
        analysis_interval: timedelta = DEFAULT_CANDLE_INTERVAL,
    ):
        self.analysis_interval = normalize_candle_interval(analysis_interval, default=DEFAULT_CANDLE_INTERVAL)
        self.structure_rules = dict(structure_rules)
        self._minimum_candles_override = minimum_candles
        self.minimum_candles = max(2, int(minimum_candles or self.structure_rules["level_lookback_bars"]))
        self._last_closed_candle: dict[str, datetime] = {}
        self._lookback_by_symbol: dict[str, int] = {}
        self._analysis_interval_by_symbol: dict[str, timedelta] = {}
        self._stop_buffer_by_symbol: dict[str, float] = {}

    @property
    def history_lookback_bars(self) -> int:
        return int(self.structure_rules["level_lookback_bars"])

    def history_capacity_for_lookback(self, bars: int) -> int:
        if bars < self.minimum_lookback_bars:
            raise ValueError(f"{type(self).__name__} requires a lookback of at least {self.minimum_lookback_bars}")
        return max(bars, int(self._minimum_candles_override or bars))

    def get_history_lookback_bars(self, symbol: str) -> int:
        return self._lookback_by_symbol.get(symbol, self.history_lookback_bars)

    def rules_for_symbol(self, symbol: str) -> dict[str, Any]:
        return {**self.structure_rules, "level_lookback_bars": self.get_history_lookback_bars(symbol),
                "protective_stop_buffer_atr": self.get_protective_stop_buffer_atr(symbol)}

    def get_protective_stop_buffer_atr(self, symbol: str) -> float:
        return self._stop_buffer_by_symbol.get(symbol, float(self.structure_rules["protective_stop_buffer_atr"]))

    def set_protective_stop_buffer_atr(self, value: float, symbol: str) -> None:
        if not isfinite(value) or value < 0:
            raise ValueError("Additional stop coefficient must be finite and nonnegative")
        self._stop_buffer_by_symbol[symbol] = float(value)

    def set_history_lookback_bars(self, bars: int, symbol: str | None = None) -> None:
        required = self.history_capacity_for_lookback(bars)
        if symbol is None:
            self.structure_rules["level_lookback_bars"] = bars
            self.minimum_candles = required
        else:
            self._lookback_by_symbol[symbol] = bars

    def get_analysis_interval(self, symbol: str) -> timedelta:
        return self._analysis_interval_by_symbol.get(symbol, self.analysis_interval)

    def set_analysis_interval(self, interval: timedelta, symbol: str | None = None) -> None:
        normalized = normalize_candle_interval(interval, default=self.analysis_interval)
        if symbol is None:
            self.analysis_interval = normalized
            self._analysis_interval_by_symbol.clear()
            self._last_closed_candle.clear()
        else:
            self._analysis_interval_by_symbol[symbol] = normalized
            self._last_closed_candle.pop(symbol, None)

    def _get_structure_candles(self, symbol: str, context: StrategyContext) -> dict[str, list[Candle]] | None:
        circuit_candles: dict[str, list[Candle]] = {}
        latest_start: datetime | None = None
        capacity = self.history_capacity_for_lookback(self.get_history_lookback_bars(symbol))
        interval = self.get_analysis_interval(symbol)
        for price_source in STRUCTURE_PRICE_SOURCES:
            closed_candles = context.market_state.get_candles(
                symbol, include_active=False, price_source=price_source, limit=capacity,
                candle_interval=interval,
            )
            if len(closed_candles) < capacity:
                return None
            current_start = closed_candles[-1].start_time
            if latest_start is None:
                latest_start = current_start
            elif current_start != latest_start:
                return None
            circuit_candles[price_source] = closed_candles
        return circuit_candles

    def _build_shared_swing_candidates(
        self, frames: dict[str, pd.DataFrame], *, symbol: str | None = None,
        market_state: MarketState | None = None,
    ) -> list[dict[str, Any]]:
        """One raw candidate vector for every indexed analysis window."""
        candidates = []
        for source in STRUCTURE_PRICE_SOURCES:
            frame = frames[source]
            if symbol is not None and market_state is not None:
                swings = market_state.get_reversal_swings(
                    symbol, 1.0, price_source=source, limit=len(frame),
                    candle_interval=self.get_analysis_interval(symbol),
                )
            else:
                swings = _collect_swings(frame, 2, 1.0, normalize=False)
            candidates.extend({**swing, "source": source, "pair_range": float(swing["strength"])}
                              for swing in swings)
        candidates.sort(key=lambda swing: (int(swing["index"]), str(swing["source"])))
        return candidates

    def _build_shared_structure_swings(
        self, bid_frame: pd.DataFrame, ask_frame: pd.DataFrame, point_size: float,
        rules: dict[str, Any] | None = None,
        symbol: str | None = None, market_state: MarketState | None = None,
    ) -> list[dict[str, Any]]:
        rules = self.structure_rules if rules is None else rules
        if symbol is not None and market_state is not None:
            resolved_rules = resolve_structure_rules(rules)
            shared_swings = []
            frames = {"bid": bid_frame, "ask": ask_frame}
            for kind, source in SWING_PRICE_SOURCES.items():
                frame = frames[source].tail(max(int(resolved_rules["level_lookback_bars"]), 500))
                if frame.empty:
                    continue
                atr = _rolling_atr(frame, point_size, int(resolved_rules["level_lookback_bars"]))
                swings = market_state.get_reversal_swings(
                    symbol,
                    atr,
                    price_source=source,
                    limit=len(frame),
                    candle_interval=self.get_analysis_interval(symbol),
                )
                shared_swings.extend({**swing, "source": source}
                                     for swing in _normalize_swings(swings) if swing["kind"] == kind)
            return _normalize_swings(shared_swings)
        return merge_structure_reversal_swings(
            bid_candles=bid_frame, ask_candles=ask_frame,
            point_size=point_size, config=rules,
        )

    def _build_candle_frame(self, closed_candles: list[Candle]) -> pd.DataFrame:
        return pd.DataFrame(
            {"time": candle.start_time, "open": candle.open_price, "high": candle.high_price,
             "low": candle.low_price, "close": candle.close_price}
            for candle in closed_candles
        )
