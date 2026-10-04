from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from ultimateBot.backtester.orders.order import OrderSide

@dataclass
class Position:
   trade_id: int

   symbol: str
   side: OrderSide

   quantity: float

   opened_at: datetime
   entry_price: float

   stop_loss: Optional[float] = None
   take_profit: Optional[float] = None

   max_favorable_pnl: float = 0.0
   max_adverse_pnl: float = 0.0