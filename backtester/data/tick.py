from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Tick:
   timestamp: datetime
   symbol: str
   bid: float
   ask: float
   bid_volume: float
   ask_volume: float

   @property
   def spread(self) -> float:
      return self.ask - self.bid