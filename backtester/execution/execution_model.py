from ultimateBot.backtester.orders.order import OrderRequest, OrderSide, OrderType
from ultimateBot.backtester.data.tick import Tick
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from ultimateBot.backtester.execution.fill import Fill

@dataclass(frozen=True)
class ExecutionResult:
   order: OrderRequest
   fill: Fill

class ExecutionModel:
   def try_execute(self, order: OrderRequest, tick: Tick) -> Optional[ExecutionResult]:

      # Reject ticks that are older than the order creation time
      if tick.timestamp <= order.created_at:
         return None

      if order.eligible_at is not None and tick.timestamp <= order.eligible_at:
         return None

      if order.symbol != tick.symbol:
         return None

      if order.order_type == OrderType.LIMIT:
         return self._execute_limit(order, tick)

      if order.order_type == OrderType.STOP:
         return self._execute_stop(order, tick)

      if order.order_type == OrderType.MARKET:
         return self._execute_market(order, tick)

      return None


   def _execute_limit(self, order: OrderRequest, tick: Tick) -> Optional[ExecutionResult]:
      # Reject if limit price is not set
      assert order.limit_price is not None, "Limit price must be set for limit orders"
      # Reject if limit price is not a positive number
      assert order.limit_price > 0, "Limit price must be a positive number for limit orders"


      if order.side == OrderSide.BUY and tick.ask > order.limit_price:
         return None
      if order.side == OrderSide.SELL and tick.bid < order.limit_price:
         return None
      if order.side == OrderSide.BUY:
         price = min(order.limit_price, tick.ask)
      elif order.side == OrderSide.SELL:
         price = max(order.limit_price, tick.bid)
      else:
         price = None

      if price is None:
         return None

      return ExecutionResult(
         order=order,
         fill=Fill(
         order_id=order.order_id,
         symbol=order.symbol,
         timestamp=tick.timestamp,
         side=order.side,
         quantity=order.quantity,
         price=price
         )
      )


   def _execute_stop(self, order: OrderRequest, tick: Tick) -> Optional[ExecutionResult]:
      if order.side == OrderSide.BUY and tick.ask < order.stop_price:
         return None
      if order.side == OrderSide.SELL and tick.bid > order.stop_price:
         return None
      if order.side == OrderSide.BUY:
         price = tick.ask
      elif order.side == OrderSide.SELL:
         price = tick.bid
      else:
         price = None

      if price is None:
         return None

      return ExecutionResult(
         order=order,
         fill=Fill(
         order_id=order.order_id,
         symbol=order.symbol,
         timestamp=tick.timestamp,
         side=order.side,
         quantity=order.quantity,
         price=price
         )
      )

   def _execute_market(self, order: OrderRequest, tick: Tick) -> Optional[ExecutionResult]:
      if order.side == OrderSide.BUY:
         price = tick.ask
      elif order.side == OrderSide.SELL:
         price = tick.bid
      else:
         price = None

      if price is None:
         return None

      return ExecutionResult(
         order=order,
         fill=Fill(
         order_id=order.order_id,
         symbol=order.symbol,
         timestamp=tick.timestamp,
         side=order.side,
         quantity=order.quantity,
         price=price
         )
      )

   def is_expired(self, order: OrderRequest, now: datetime) -> bool:
      if order.expires_at is None:
         return False
      return now >= order.expires_at