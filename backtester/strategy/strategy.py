from abc import ABC, abstractmethod
from datetime import timedelta

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.strategy.signal import Signal
from ultimateBot.backtester.strategy.strategy_context import StrategyContext


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    analysis_interval: timedelta | None = None

    @abstractmethod
    def on_market_data(self, batch: TickBatch, context: StrategyContext) -> list[Signal]:
        pass