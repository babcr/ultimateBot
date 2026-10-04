from typing import Iterable
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.orders.order import PendingOrderProcessor, PendingOrderQueue
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.signal import Signal
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.backtester.strategy.strategy_context import StrategyContext

class BacktestEngine:
   def __init__(
      self,
      feed: Iterable[Tick],
      batcher: TickBatcher,
      strategy: Strategy,
      risk_manager: RiskManager,
      order_builder: OrderBuilder,
      execution_model: ExecutionModel,
      portfolio: Portfolio,
      position_manager: PositionManager,
      trade_monitor: TradeMonitor,
      trade_closer: TradeCloser,
      reports: ReportCollector,
      market_state: MarketState | None = None,
   ):
      self.feed = feed
      self.batcher = batcher
      self.strategy = strategy
      self.risk_manager = risk_manager
      self.order_builder = order_builder
      self.execution_model = execution_model
      self.portfolio = portfolio
      self.position_manager = position_manager
      self.trade_monitor = trade_monitor
      self.trade_closer = trade_closer
      self.reports = reports

      self.market_state = market_state if market_state is not None else MarketState()
      self.pending_orders_processor = PendingOrderProcessor(PendingOrderQueue(), self.execution_model)

   def _process_pending_orders(self, tick: Tick) -> None:
      execution_results = self.pending_orders_processor.on_tick(tick)
      for result in execution_results:
         self.position_manager.open_position(portfolio=self.portfolio, fill=result.fill, order=result.order)

   def run(self) -> None:
      for tick in self.feed:
         self.market_state.update(tick)
         self.risk_manager.on_tick(tick)
         self._monitor_positions(tick)
         self._process_pending_orders(tick)

         batch = self.batcher.push(tick)
         if batch is None:
            continue

         self._run_strategy(batch)

      self.market_state.finalize()
      for batch in self.batcher.flush():
         self._run_strategy(batch)

   def _run_strategy(self, batch) -> None:
      context = StrategyContext(portfolio=self.portfolio, market_state=self.market_state)
      signals = self.strategy.on_market_data(batch, context)

      for signal in signals:
         self._process_signal(signal, batch.ticks[-1])

   def _process_signal(self, signal: Signal, tick: Tick) -> None:
      evaluation = self.risk_manager.evaluate(
         signal=signal,
         portfolio=self.portfolio,
         tick=tick,
         pending_orders_for_symbol=self.pending_orders_processor.queue.count_for_symbol(signal.symbol),
      )
      if evaluation is None:
         return

      order = self.order_builder.build(signal, evaluation)
      self.pending_orders_processor.queue.add(order)

   def _monitor_positions(self, tick: Tick) -> None:
      positions = list(self.portfolio.positions.values())
      for position in positions:
         request = self.trade_monitor.check(position, tick)
         if request is None:
            continue

         report = self.trade_closer.close(self.portfolio, request)
         self.reports.add_trade(report)