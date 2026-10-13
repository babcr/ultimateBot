from collections.abc import Callable
from dataclasses import replace

from ultimateBot.backtester.reports.trade_report import TradeReport

class ReportCollector:
   def __init__(self):
      self.trades: list[TradeReport] = []
      self._trade_listeners: list[Callable[[TradeReport], None]] = []
      self._event_listeners: list[Callable[[TradeReport], None]] = []
      self._partials: dict[tuple, list[TradeReport]] = {}
      self._seen_deals: set[str] = set()

   def add_trade_event_listener(self, listener: Callable[[TradeReport], None]) -> None:
      if listener not in self._event_listeners:
         self._event_listeners.append(listener)

   def add_trade_listener(self, listener: Callable[[TradeReport], None]) -> None:
      if listener not in self._trade_listeners:
         self._trade_listeners.append(listener)

   def add_trade(self, report: TradeReport):
      if report.event_id is not None:
         if report.event_id in self._seen_deals:
            return
         self._seen_deals.add(report.event_id)
      for listener in tuple(self._event_listeners):
         listener(report)
      key = (report.symbol, report.trade_id, report.opened_at)
      if not report.is_complete:
         self._partials.setdefault(key, []).append(report)
         return
      parts = self._partials.pop(key, [])
      if parts:
         report = replace(report, pnl=report.pnl + sum(part.pnl for part in parts),
                          quantity=report.quantity + sum(part.quantity for part in parts),
                          pnl_gross=(sum(part.pnl_gross for part in [*parts, report])
                                     if all(part.pnl_gross is not None for part in [*parts, report]) else None),
                          deal_ids=tuple(deal for part in [*parts, report] for deal in part.deal_ids))
      self.trades.append(report)
      for listener in tuple(self._trade_listeners):
         listener(report)
