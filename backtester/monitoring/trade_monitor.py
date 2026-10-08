from typing import Optional
from ultimateBot.backtester.monitoring.close_request import CloseRequest, CloseReason
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.portfolio.position import Position


class TradeMonitor:

   def _mark_position_excursion(
      self,
      position: Position,
      exit_price: float,
   ) -> None:
      if position.side == OrderSide.BUY:
         unrealized_pnl = (exit_price - position.entry_price) * position.quantity
      else:
         unrealized_pnl = (position.entry_price - exit_price) * position.quantity

      position.max_favorable_pnl = max(position.max_favorable_pnl, unrealized_pnl)
      position.max_adverse_pnl = min(position.max_adverse_pnl, unrealized_pnl)

   def check(
      self,
      position: Position,
      tick: Tick,
   ) -> Optional[CloseRequest]:
      if position.symbol != tick.symbol:
         return None

      if position.side == OrderSide.BUY:
         exit_price = tick.bid
         self._mark_position_excursion(position, exit_price)
         if position.stop_loss is not None and exit_price <= position.stop_loss:
            return CloseRequest(
               trade_id=position.trade_id,
               timestamp=tick.timestamp,
               price=exit_price,
               reason=CloseReason.STOP_LOSS,
            )
         if position.take_profit is not None and exit_price >= position.take_profit:
            return CloseRequest(
               trade_id=position.trade_id,
               timestamp=tick.timestamp,
               price=exit_price,
               reason=CloseReason.TAKE_PROFIT,
            )
      elif position.side == OrderSide.SELL:
         exit_price = tick.ask
         self._mark_position_excursion(position, exit_price)
         if position.stop_loss is not None and exit_price >= position.stop_loss:
            return CloseRequest(
               trade_id=position.trade_id,
               timestamp=tick.timestamp,
               price=exit_price,
               reason=CloseReason.STOP_LOSS,
            )
         if position.take_profit is not None and exit_price <= position.take_profit:
            return CloseRequest(
               trade_id=position.trade_id,
               timestamp=tick.timestamp,
               price=exit_price,
               reason=CloseReason.TAKE_PROFIT,
            )
      else:
         pass

      return None