from dataclasses import dataclass
from datetime import datetime
from ultimateBot.backtester.orders.order import OrderStatus

@dataclass(frozen=True)
class OrderReport:
   order_id: int

   symbol: str

   created_at: datetime
   completed_at: datetime | None

   status: OrderStatus
   
   requested_price: float | None
   execution_price: float | None

   reason: str | None