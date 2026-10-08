from __future__ import annotations

from datetime import timedelta
from typing import Optional

from ultimateBot.backtester.config.backtest_config import DEFAULT_CANDLE_INTERVAL, normalize_candle_interval, VolatilityConfig
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle, CandleBuilder, PriceSource, normalize_price_source


_FIAT_CODES = {"AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD"}

DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()

class MarketState:
    def __init__(self, candle_interval: timedelta = DEFAULT_CANDLE_INTERVAL, candle_lookback: int = DEFAULT_VOLATILITY_CONFIG.candle_lookback):
        self.last_ticks: dict[str, Tick] = {}
        normalized_interval = normalize_candle_interval(candle_interval, default=DEFAULT_CANDLE_INTERVAL)
        self._candle_builders: dict[str, CandleBuilder] = {
            "mid": CandleBuilder(
                candle_interval=normalized_interval,
                lookback_candles=candle_lookback,
                price_source="mid",
            ),
            "bid": CandleBuilder(
                candle_interval=normalized_interval,
                lookback_candles=candle_lookback,
                price_source="bid",
            ),
            "ask": CandleBuilder(
                candle_interval=normalized_interval,
                lookback_candles=candle_lookback,
                price_source="ask",
            ),
        }
        self._point_sizes: dict[str, float] = {}

    def update(self, tick: Tick):
        self.last_ticks[tick.symbol] = tick
        for candle_builder in self._candle_builders.values():
            candle_builder.update(tick)
        mid_price = 0.5 * (tick.bid + tick.ask)
        self._point_sizes[tick.symbol] = self._infer_point_size(tick.symbol, mid_price)

    def get_tick(self, symbol: str) -> Optional[Tick]:
        return self.last_ticks.get(symbol)

    def get_candles(self, symbol: str, include_active: bool = False, price_source: PriceSource = "mid") -> list[Candle]:
        return self._get_candle_builder(price_source).get_candles(symbol, include_active=include_active)

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

    def finalize(self) -> None:
        for candle_builder in self._candle_builders.values():
            candle_builder.finalize_active_candles()

    def _get_candle_builder(self, price_source: PriceSource) -> CandleBuilder:
        return self._candle_builders[normalize_price_source(price_source)]

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
