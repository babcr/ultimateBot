from __future__ import annotations

from collections import deque
from datetime import timedelta
from typing import Any, Optional

from ultimateBot.backtester.config.backtest_config import DEFAULT_CANDLE_INTERVAL, normalize_candle_interval, VolatilityConfig
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle, CandleBuilder, PriceSource, normalize_price_source


_FIAT_CODES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"}

DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()

class MarketState:
    def __init__(self, candle_interval: timedelta = DEFAULT_CANDLE_INTERVAL, candle_lookback: int = DEFAULT_VOLATILITY_CONFIG.candle_lookback):
        self.last_ticks: dict[str, Tick] = {}
        normalized_interval = normalize_candle_interval(candle_interval, default=DEFAULT_CANDLE_INTERVAL)
        self.candle_interval = normalized_interval
        self.candle_lookback = max(1, int(candle_lookback))
        self._candle_builders: dict[str, CandleBuilder] = {
            "mid": CandleBuilder(
                candle_interval=normalized_interval,
                lookback_candles=self.candle_lookback,
                price_source="mid",
            ),
            "bid": CandleBuilder(
                candle_interval=normalized_interval,
                lookback_candles=self.candle_lookback,
                price_source="bid",
            ),
            "ask": CandleBuilder(
                candle_interval=normalized_interval,
                lookback_candles=self.candle_lookback,
                price_source="ask",
            ),
        }
        self._candle_builders_by_interval: dict[timedelta, dict[str, CandleBuilder]] = {
            normalized_interval: self._candle_builders,
        }
        self._point_sizes: dict[str, float] = {}

    def register_candle_interval(self, candle_interval: timedelta) -> None:
        normalized = normalize_candle_interval(candle_interval, default=self.candle_interval)
        if normalized in self._candle_builders_by_interval:
            return
        self._candle_builders_by_interval[normalized] = {
            source: CandleBuilder(
                candle_interval=normalized,
                lookback_candles=self.candle_lookback,
                price_source=source,
            )
            for source in ("mid", "bid", "ask")
        }

    def ensure_history_capacity(self, capacity: int) -> None:
        """Grow the shared bounded history, retaining only already observed data."""
        capacity = max(self.candle_lookback, int(capacity))
        if capacity == self.candle_lookback:
            return
        self.candle_lookback = capacity
        for builders in self._candle_builders_by_interval.values():
            for builder in builders.values():
                builder.lookback_candles = capacity
                for symbol, candles in list(builder._closed.items()):
                    builder._closed[symbol] = deque(candles, maxlen=capacity)
                builder._swing_histories.clear()

    def update(self, tick: Tick):
        self.last_ticks[tick.symbol] = tick
        for builders in self._candle_builders_by_interval.values():
            for candle_builder in builders.values():
                candle_builder.update(tick)
        mid_price = 0.5 * (tick.bid + tick.ask)
        self._point_sizes[tick.symbol] = self._infer_point_size(tick.symbol, mid_price)

    def get_tick(self, symbol: str) -> Optional[Tick]:
        return self.last_ticks.get(symbol)

    def get_candles(
        self, symbol: str, include_active: bool = False,
        price_source: PriceSource = "mid", limit: int | None = None,
        candle_interval: timedelta | None = None,
    ) -> list[Candle]:
        return self._get_candle_builder(price_source, candle_interval).get_candles(
            symbol, include_active=include_active, limit=limit,
        )

    def get_point_size(self, symbol: str) -> float:
        cached = self._point_sizes.get(symbol)
        if cached is not None and cached > 0.0:
            return cached

        tick = self.get_tick(symbol)
        if tick is None:
            return 0.0001

        mid_price = 0.5 * (tick.bid + tick.ask)
        point_size = self._infer_point_size(symbol, mid_price)
        self._point_sizes[symbol] = point_size
        return point_size

    def get_reversal_swings(
        self, symbol: str, atr: float, price_source: PriceSource = "mid", limit: int | None = None,
        candle_interval: timedelta | None = None,
    ) -> list[dict[str, Any]]:
        return self._get_candle_builder(price_source, candle_interval).get_reversal_swings(symbol, atr, limit=limit)

    def finalize(self) -> None:
        for builders in self._candle_builders_by_interval.values():
            for candle_builder in builders.values():
                candle_builder.finalize_active_candles()

    def _get_candle_builder(
        self,
        price_source: PriceSource,
        candle_interval: timedelta | None = None,
    ) -> CandleBuilder:
        interval = self.candle_interval if candle_interval is None else normalize_candle_interval(
            candle_interval,
            default=self.candle_interval,
        )
        builders = self._candle_builders_by_interval.get(interval)
        if builders is None:
            raise ValueError(f"Candle interval {interval} is not registered")
        return builders[normalize_price_source(price_source)]

    def _infer_point_size(self, symbol: str, reference_price: float) -> float:
        base = symbol[:3]
        quote = symbol[3:]
        if len(symbol) == 6 and base in _FIAT_CODES and quote in _FIAT_CODES:
            return 0.001 if symbol.endswith("JPY") else 0.00001

        if reference_price >= 1000.0:
            return 0.1
        if reference_price >= 100.0:
            return 0.01
        if reference_price >= 10.0:
            return 0.001
        return 0.0001
