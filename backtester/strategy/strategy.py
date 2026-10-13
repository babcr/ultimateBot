from abc import ABC, abstractmethod
from datetime import timedelta

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.strategy.signal import Signal
from ultimateBot.backtester.strategy.strategy_context import StrategyContext


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    analysis_interval: timedelta | None = None

    @property
    def history_lookback_bars(self) -> int:
        raise ValueError(f"{type(self).__name__} does not expose a configurable candle history")

    def history_capacity_for_lookback(self, bars: int) -> int:
        raise ValueError(f"{type(self).__name__} does not support adaptive candle lookbacks")

    def get_history_lookback_bars(self, symbol: str) -> int:
        return self.history_lookback_bars

    def set_history_lookback_bars(self, bars: int, symbol: str | None = None) -> None:
        raise ValueError(f"{type(self).__name__} does not support adaptive candle lookbacks")

    def get_analysis_interval(self, symbol: str) -> timedelta:
        if self.analysis_interval is None:
            raise ValueError(f"{type(self).__name__} does not expose an analysis timeframe")
        return self.analysis_interval

    def set_analysis_interval(self, interval: timedelta, symbol: str | None = None) -> None:
        raise ValueError(f"{type(self).__name__} does not support adaptive analysis timeframes")

    def supports_global_timeframe_validation(self) -> bool:
        return False

    def get_global_timeframe_validation_enabled(self, symbol: str | None = None) -> bool:
        return False

    def set_global_timeframe_validation_enabled(self, enabled: bool, symbol: str | None = None) -> None:
        if enabled:
            raise ValueError(f"{type(self).__name__} does not support global timeframe validation")

    @abstractmethod
    def on_market_data(self, batch: TickBatch, context: StrategyContext) -> list[Signal]:
        pass
