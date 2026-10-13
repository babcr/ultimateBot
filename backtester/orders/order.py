from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from ultimateBot.backtester.data.tick import Tick

if TYPE_CHECKING:
   from ultimateBot.backtester.execution.execution_model import ExecutionModel, ExecutionResult
   from ultimateBot.backtester.optimization.bandit_core import PositionLearningRecord

class OrderSide(Enum):
   BUY = auto()
   SELL = auto()

class OrderType(Enum):
   MARKET = auto()
   LIMIT = auto()
   STOP = auto()

class OrderStatus(Enum):
   CREATED = auto()
   PENDING = auto()
   FILLED = auto()
   CANCELLED = auto()
   REJECTED = auto()
   EXPIRED = auto()

@dataclass
class OrderRequest:
   order_id: int

   symbol: str
   timestamp: datetime
   created_at: datetime
   eligible_at: datetime

   side: OrderSide
   quantity: float

   order_type: OrderType = OrderType.MARKET

   limit_price: Optional[float] = None
   stop_price: Optional[float] = None

   stop_loss: Optional[float] = None
   take_profit: Optional[float] = None

   expires_at: Optional[datetime] = None
   trailing_stop_activation_price: Optional[float] = None
   trailing_stop_enabled: Optional[bool] = None
   strategy_configuration_id: Optional[int] = None
   strategy_lookback_bars: Optional[int] = None
   strategy_timeframe_minutes: Optional[int] = None
   strategy_use_global_timeframe_validation: Optional[bool] = None
   learning_record: PositionLearningRecord | None = None
   profit_multiplier: float = 1.0
   account_currency: str = "engine_price_quantity"


class PendingOrderQueue:
   def __init__(self) -> None:
      self._orders_by_symbol: dict[str, list[OrderRequest]] = defaultdict(list)

   def add(self, order: OrderRequest) -> None:
      self._orders_by_symbol[order.symbol].append(order)

   def add_many(self, orders: list[OrderRequest]) -> None:
      for order in orders:
         self.add(order)
   def get_for_symbol(self, symbol: str) -> list[OrderRequest]:
      return self._orders_by_symbol.get(symbol, [])

   def take_for_symbol(self, symbol: str) -> list[OrderRequest]:
      """Take and remove all pending orders for a given symbol from the queue.
         The processor will put back only the PENDING orders.

      Args:
         symbol (str): the symbol for which to take and remove all pending orders from the queue.

      Returns:
         list[OrderRequest]: the list of pending orders that were removed from the queue for the given symbol.
      """
      return self._orders_by_symbol.pop(symbol, [])

   def restore_for_symbol(self, symbol: str, orders: list[OrderRequest]) -> None:
      """Restore a list of orders for a given symbol back into the queue.

      Args:
         symbol (str): the symbol for which to restore orders.
         orders (list[OrderRequest]): the list of orders to restore.
      """
      if not orders:
         return
      self._orders_by_symbol[symbol].extend(orders)

   def count_for_symbol(self, symbol: str) -> int:
      return len(self._orders_by_symbol.get(symbol, []))

   def __len__(self) -> int:
      return sum(len(orders) for orders in self._orders_by_symbol.values())

   """
   def remove(self, order: OrderRequest) -> None:
      if order is None:
         return

      orders = self._orders_by_symbol.get(order.symbol, [])
      if order in orders:
         orders.remove(order)

      if not orders:
         del self._orders_by_symbol[order.symbol]

   def cancel(self, order_id: int) -> bool:
      for symbol, orders in list(self._orders_by_symbol.items()):
         for order in orders:
            if order.order_id == order_id:
               orders.remove(order)
               if not orders:
                  del self._orders_by_symbol[symbol]
               return True
      return False
   """


class PendingOrderProcessor:
   def __init__(self, queue: PendingOrderQueue, execution_model: ExecutionModel):
      self.queue = queue
      self.execution_model = execution_model

   def on_tick(self, tick: Tick, max_fills: int | None = None) -> list[ExecutionResult]:

      fills = []
      orders = self.queue.take_for_symbol(tick.symbol)
      if not orders:
         return []
      fills: list[ExecutionResult] = []
      still_pending: list[OrderRequest] = []

      for order in orders:
         # ----------------------------------------
         # 1. Ordres expirés
         # ----------------------------------------
         if order.expires_at is not None and order.expires_at <= tick.timestamp:
            print(
               f"[order-expired] order_id={order.order_id} symbol={order.symbol} "
               f"timestamp={tick.timestamp.isoformat()} side={order.side.name.lower()} "
               f"order_type={order.order_type.name.lower()} "
               f"created_at={order.created_at.isoformat()} expires_at={order.expires_at.isoformat()}",
               flush=True,
            )
            continue
         # ----------------------------------------
         # 2. Pas encore executable
         # ----------------------------------------
         if order.eligible_at is not None and order.eligible_at > tick.timestamp:
            still_pending.append(order)
            continue
         if max_fills is not None and len(fills) >= max_fills:
            still_pending.append(order)
            continue
         # ----------------------------------------
         # 3. Tentative d'exécution
         # ----------------------------------------
         result = self.execution_model.try_execute(order=order, tick=tick)
         if result is None:
            # Order could not be executed (Exemple; LIMIT non-atteint)
            still_pending.append(order)
            continue
         # ----------------------------------------
         # 4. Order exécutée avec succès
         # ----------------------------------------
         fills.append(result)
      # Restore the order to the queue if it was not executed
      if still_pending:
         self.queue.restore_for_symbol(tick.symbol, still_pending)

      return fills
