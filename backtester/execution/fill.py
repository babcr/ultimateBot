from dataclasses import dataclass
from datetime import datetime
from ultimateBot.backtester.orders.order import OrderSide

@dataclass(frozen=True)
class Fill:
   order_id: int
   symbol: str
   timestamp: datetime
   side: OrderSide
   quantity: float
   price: float
