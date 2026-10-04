from typing import Optional
from ultimateBot.backtester.monitoring.close_request import CloseRequest, CloseReason
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.portfolio.position import Position


class TradeMonitor:

   def check(
      self,
      position: Position,
      tick: Tick,
   ) -> Optional[CloseRequest]:
      if position.symbol != tick.symbol:
         return None

      if position.side == OrderSide.BUY:
         exit_price = tick.bid
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