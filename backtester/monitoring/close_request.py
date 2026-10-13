from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime

class CloseReason(Enum):
   STOP_LOSS = auto()
   TAKE_PROFIT = auto()
   STRATEGY = auto()
   END_OF_DAY = auto()
   END_OF_WEEK = auto()
   END_OF_BACKTEST = auto()
   MANUAL = auto()
   TIMEOUT = auto()
   TRAILING_STOP = auto()

@dataclass
class CloseRequest:
   trade_id: int
   timestamp: datetime
   price: float
   reason: CloseReason
   quantity: float | None = None
   event_id: str | None = None
