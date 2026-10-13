from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime


class SignalSide(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()


class SignalEntryMode(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()


@dataclass(frozen=True)
class Signal:
    symbol: str
    timestamp: datetime
    side: SignalSide
    confidence: float = 0.0
    entry_mode: SignalEntryMode = SignalEntryMode.MARKET
    protective_level_price: float | None = None
    protective_level_quality: float = 0.0
    protective_stop_buffer: float = 0.0
