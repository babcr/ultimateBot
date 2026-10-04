from ultimateBot.backtester.reports.trade_report import TradeReport

class ReportCollector:
   def __init__(self):
      self.trades: list[TradeReport] = []

   def add_trade(self, report: TradeReport):
      self.trades.append(report)