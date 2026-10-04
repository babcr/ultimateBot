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
      )

      portfolio.positions[trade_id] = position
      return position