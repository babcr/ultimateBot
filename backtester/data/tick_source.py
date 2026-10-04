from typing import Iterator
from abc import ABC, abstractmethod
from ultimateBot.backtester.data.tick import Tick

class TickSource(ABC):
    """Abstract base class for tick data sources."""
    @abstractmethod
    def __iter__(self) -> Iterator[Tick]:
        pass