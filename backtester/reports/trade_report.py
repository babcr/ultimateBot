from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ultimateBot.backtester.monitoring.close_request import CloseReason, CloseRequest
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.portfolio.portfolio import Portfolio

@dataclass(frozen=True)
class TradeReport:
   trade_id: int
   symbol: str

   side: OrderSide

   opened_at: datetime
   closed_at: datetime

   entry_price: float
   exit_price: float

   quantity: float

   pnl: float

   close_reason: CloseReason


class TradeCloser:
   def close(
      self,
      portfolio: Portfolio,
      request: CloseRequest
   ) -> TradeReport:
      position = portfolio.positions.pop(request.trade_id)

      if position.side == OrderSide.BUY:
         pnl = (request.price - position.entry_price) * position.quantity
      else:
         pnl = (position.entry_price - request.price) * position.quantity

      portfolio.balance += pnl

      return TradeReport(
         trade_id=request.trade_id,
         symbol=position.symbol,
         side=position.side,
         opened_at=position.opened_at,
         closed_at=request.timestamp,
         entry_price=position.entry_price,
         exit_price=request.price,
         quantity=position.quantity,
         pnl=pnl,
         close_reason=request.reason
      )