from __future__ import annotations

from ultimateBot.backtester.execution.fill import Fill
from ultimateBot.backtester.orders.order import OrderRequest
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position import Position


class PositionManager:
   def __init__(self):
      self.positions: dict[int, Position] = {}
      self._next_trade_id = 1

   def open_position(self, portfolio: Portfolio, fill: Fill, order: OrderRequest) -> Position:
      trade_id = portfolio._next_trade_id
      portfolio._next_trade_id += 1

      position = Position(
         trade_id=trade_id,
         symbol=fill.symbol,
         side=fill.side,
         quantity=fill.quantity,
         opened_at=fill.timestamp,
         entry_price=fill.price,
         stop_loss=order.stop_loss,
         take_profit=order.take_profit,
         trailing_stop_activation_price=order.trailing_stop_activation_price,
         trailing_stop_enabled=order.trailing_stop_enabled,
         strategy_configuration_id=order.strategy_configuration_id,
         strategy_lookback_bars=order.strategy_lookback_bars,
         strategy_timeframe_minutes=order.strategy_timeframe_minutes,
         strategy_use_global_timeframe_validation=order.strategy_use_global_timeframe_validation,
         learning_record=order.learning_record,
         profit_multiplier=order.profit_multiplier,
         account_currency=order.account_currency,
      )

      portfolio.positions[trade_id] = position
      return position
