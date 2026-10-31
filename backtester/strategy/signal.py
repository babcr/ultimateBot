from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    minimum_reward_risk_ratio: float = 0.0
    entry_context: dict[str, Any] | None = None
    minimum_entry_price: float | None = None
    maximum_entry_price: float | None = None
