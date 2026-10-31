from typing import Iterable
from collections import Counter
from math import isfinite
from ultimateBot.backtester.config.trading_schedule import TradingSchedule, TradingScheduleConfig
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.monitoring.close_request import CloseReason, CloseRequest
from ultimateBot.backtester.orders.order import PendingOrderProcessor, PendingOrderQueue
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.optimization.adaptive_strategy import AdaptiveStrategyOptimizer
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
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
      adaptive_optimizer: AdaptiveStrategyOptimizer | None = None,
      contextual_optimizer=None,
      learning_hooks=None,
      trading_schedule: TradingScheduleConfig | None = None,
      trade_dataset=None,
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
      self.adaptive_optimizer = adaptive_optimizer
      self.contextual_optimizer = contextual_optimizer
      self.learning_hooks = learning_hooks if learning_hooks is not None else contextual_optimizer
      self.trade_dataset = trade_dataset
      self.trade_validator = None
      self.trading_schedule = TradingSchedule(trading_schedule)
      self._market_clock = None
      self.trade_entry_modes: dict[int, str] = {}
      self.entry_rejection_counts = Counter()
      if adaptive_optimizer is not None:
         reports.add_trade_listener(adaptive_optimizer.on_trade_closed)

      self.market_state = market_state if market_state is not None else MarketState()
      self.pending_orders_processor = PendingOrderProcessor(PendingOrderQueue(), self.execution_model)

   def _process_pending_orders(self, tick: Tick) -> None:
      self._cancel_disallowed_orders(tick)
      available_slots = max(0, self.risk_manager.risk_config.max_positions - len(self.portfolio.positions))
      execution_results = self.pending_orders_processor.on_tick(tick, max_fills=available_slots)
      for result in execution_results:
         position = self.position_manager.open_position(portfolio=self.portfolio, fill=result.fill, order=result.order)
         prepare_filled = getattr(self.strategy, "prepare_filled_position", None)
         if prepare_filled is not None:
            prepare_filled(position)
         self.trade_closer.prepare_position(position)
         entry_context = result.order.strategy_entry_context or {}
         if entry_context.get("entry_mode"):
            self.trade_entry_modes[position.trade_id] = entry_context["entry_mode"]
         if self.learning_hooks is not None:
            self.learning_hooks.on_position_open(position, tick)
         if self.trade_dataset is not None:
            self.trade_dataset.on_position_open(position, tick, result.order)
         trailing_details = (
            f" trailing_activation_price={position.trailing_stop_activation_price}"
            if position.trailing_stop_enabled else ""
         )

         adaptive_details = (
            f" configuration_id={position.strategy_configuration_id} lookback={position.strategy_lookback_bars} "
            f"timeframe_minutes={position.strategy_timeframe_minutes} "
            f"global_validation={position.strategy_use_global_timeframe_validation} "
            f"trailing={position.trailing_stop_enabled}"
            if self.adaptive_optimizer is not None else ""
         )
         print(
            f"[trade-filled] order_id={result.order.order_id} trade_id={position.trade_id} "
            f"symbol={position.symbol} timestamp={position.opened_at.isoformat()} "
            f"side={position.side.name.lower()} quantity={position.quantity} "
            f"entry_mode={self.trade_entry_modes.get(position.trade_id,'unknown')} "
            f"entry_price={position.entry_price} stop_loss={position.stop_loss} "
            f"take_profit={position.take_profit}{trailing_details}{adaptive_details}",
            flush=True,
         )

   def _cancel_disallowed_orders(self, tick: Tick) -> None:
      if not self.trading_schedule.config.enabled:
         return
      queue = self.pending_orders_processor.queue
      for symbol in queue.symbols():
         keep = []
         for order in queue.take_for_symbol(symbol):
            reason = self.trading_schedule.pending_rejection(order.created_at, tick.timestamp)
            if reason is None:
               keep.append(order)
               continue
            self.trading_schedule.stats["cancelled_pending_orders"] += 1
            if self.learning_hooks is not None and hasattr(self.learning_hooks, "on_order_cancelled"):
               self.learning_hooks.on_order_cancelled(order)
            print(f"[order-cancelled] order_id={order.order_id} symbol={order.symbol} "
                  f"timestamp={tick.timestamp.isoformat()} reason={reason}", flush=True)
         queue.restore_for_symbol(symbol, keep)

   def run(self) -> None:
      try:
         self._run()
      except BaseException as error:
         if self.trade_validator is not None:
            self.trade_validator.finish()
         if self.trade_dataset is not None:
            status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            self.trade_dataset.finish(status, error)
         raise
      else:
         if self.trade_validator is not None:
            self.trade_validator.finish()
         if self.trade_dataset is not None:
            self.trade_dataset.finish()

   def _run(self) -> None:
      for tick in self.feed:
         self.on_tick(tick)

      self.market_state.finalize()
      for batch in self.batcher.flush():
         self._run_strategy(batch)
      if self.contextual_optimizer is not None:
         self.contextual_optimizer.finish()

   def on_tick(self, tick: Tick, *, update_market=True, allow_entries=True) -> None:
      optimizer = self.contextual_optimizer
      if optimizer is not None and optimizer.should_skip_replayed_tick(tick):
         return
      self._market_clock = tick.timestamp
      if update_market:
         self.market_state.update(tick)
      if optimizer is not None:
         optimizer.on_market_tick(tick)
      self.risk_manager.on_tick(tick)
      self._monitor_positions(tick)
      if optimizer is not None:
         optimizer.shadow.on_tick(tick)
         optimizer.flush_closures(tick.timestamp)
      self._process_pending_orders(tick)
      batch = self.batcher.push(tick)
      if batch is not None and allow_entries:
         self._run_strategy(batch)
         if getattr(self.strategy, "fill_on_signal_release", False):
            # Frozen swing decisions use the first quote releasing a closed candle.
            self._process_pending_orders(tick)
      if getattr(self.strategy, "fill_on_signal_release", False):
         # Also check newly filled pending orders when no new candle was released.
         self._monitor_positions(tick)
      if optimizer is not None:
         optimizer.after_tick(tick)

   def _run_strategy(self, batch) -> None:
      if self.adaptive_optimizer is not None:
         interval = self.strategy.get_analysis_interval(batch.symbol)
         candles = self.market_state.get_candles(batch.symbol, include_active=False, limit=1,
                                                  candle_interval=interval)
         if candles:
            closed_at = candles[-1].start_time + interval
            if closed_at <= (self._market_clock or batch.ticks[-1].timestamp):
               busy = (any(p.symbol == batch.symbol for p in self.portfolio.positions.values())
                       or self.pending_orders_processor.queue.count_for_symbol(batch.symbol) > 0)
               self.adaptive_optimizer.on_analysis_window(batch.symbol, closed_at, has_active_exposure=busy)
      context = StrategyContext(portfolio=self.portfolio, market_state=self.market_state,
                                pending_orders_for_symbol=self.pending_orders_processor.queue.count_for_symbol(batch.symbol))
      signals = self.strategy.on_market_data(batch, context)

      for signal in signals:
         quote = (self.market_state.last_ticks[batch.symbol]
                  if getattr(self.strategy, "fill_on_signal_release", False) else batch.ticks[-1])
         self._process_signal(signal, quote)

   def _process_signal(self, signal: Signal, tick: Tick) -> None:
      if signal.side != SignalSide.HOLD and self._hybrid_symbol_busy(signal.symbol):
         print(f"[signal-rejected] symbol={signal.symbol} timestamp={signal.timestamp.isoformat()} "
               "reason=hybrid_symbol_busy", flush=True)
         return
      if signal.side != SignalSide.HOLD:
         reason = self.trading_schedule.pending_rejection(signal.timestamp, self._market_clock or tick.timestamp)
         if reason is not None:
            self.trading_schedule.stats["rejected_signals"] += 1
            print(f"[signal-rejected] symbol={signal.symbol} timestamp={signal.timestamp.isoformat()} "
                  f"reason={reason}", flush=True)
            return
      evaluation = self.risk_manager.evaluate(
         signal=signal,
         portfolio=self.portfolio,
         tick=tick,
      )
      if evaluation is None:
         if getattr(self.strategy, "fill_on_signal_release", False):
            self.strategy.counts["rejected_by_risk"] += 1
            print(f"[signal-rejected] symbol={signal.symbol} timestamp={signal.timestamp.isoformat()} reason=risk_manager", flush=True)
         return

      threshold = getattr(self.strategy,"regression_rules",{}).get("regression_max_spread_stop_ratio",0.)
      submission_details = {}
      if threshold and signal.side != SignalSide.HOLD:
         # The batch tick closes the signal candle. The live quote is already
         # known at submission and may have a different spread after a gap.
         current = self.market_state.last_ticks.get(signal.symbol)
         if current is None or current.timestamp != (self._market_clock or tick.timestamp):
            current = tick
         risk_distance = abs(evaluation.entry_price-evaluation.stop_loss)
         spread = current.ask-current.bid
         ratio = spread/risk_distance if risk_distance > 0 else float("inf")
         if not isfinite(ratio) or spread < 0 or ratio > threshold:
            branch = (signal.entry_context or {}).get("entry_mode","unknown")
            self.entry_rejection_counts[(branch,"spread_stop_ratio")] += 1
            print(f"[signal-rejected] symbol={signal.symbol} timestamp={signal.timestamp.isoformat()} "
                  f"reason=spread_stop_ratio branch={branch} ratio={ratio} maximum={threshold}",flush=True)
            return
         submission_details = {"submission_spread":spread,"submission_spread_stop_ratio":ratio,
                               "planned_entry":evaluation.entry_price,"max_spread_stop_ratio":threshold}

      order = self.order_builder.build(signal, evaluation)
      if submission_details:
         order.strategy_entry_context = {**(order.strategy_entry_context or {}),**submission_details}
      if self.adaptive_optimizer is not None:
         self.adaptive_optimizer.prepare_order(order)
         self.order_builder.set_expiration_interval(
            order,
            self.strategy.get_analysis_interval(order.symbol),
         )
      if self.learning_hooks is not None:
         self.learning_hooks.prepare_order(order, evaluation, signal, tick)
      self.trade_closer.prepare_order(order)
      if self.trade_validator is not None:
         self.trade_validator.prepare_order(order)
      if self.trade_dataset is not None:
         self.trade_dataset.prepare_order(order)
      trailing_details = (
         f" trailing_activation_price={order.trailing_stop_activation_price}"
         if order.trailing_stop_enabled else ""
      )
      adaptive_details = (
         f" configuration_id={order.strategy_configuration_id} lookback={order.strategy_lookback_bars} "
         f"timeframe_minutes={order.strategy_timeframe_minutes} "
         f"global_validation={order.strategy_use_global_timeframe_validation} "
         f"trailing={order.trailing_stop_enabled}"
         if self.adaptive_optimizer is not None else ""
      )
      expiration_details = f" expires_at={order.expires_at.isoformat()}" if order.expires_at is not None else ""
      if not self._queue_order(order, tick):
         return
      print(
         f"[trade-created] order_id={order.order_id} symbol={order.symbol} "
         f"timestamp={(self._market_clock or tick.timestamp).isoformat()} signal_timestamp={order.created_at.isoformat()} side={order.side.name.lower()} "
         f"entry_mode={(order.strategy_entry_context or {}).get('entry_mode','unknown')} "
         f"order_type={order.order_type.name.lower()} quantity={order.quantity} "
         f"risk_percent={evaluation.risk_percent} "
         f"entry_price={evaluation.entry_price} stop_loss={order.stop_loss} "
         f"take_profit={order.take_profit}{expiration_details}{trailing_details}{adaptive_details}",
         flush=True,
      )

   def _hybrid_symbol_busy(self, symbol: str) -> bool:
      return (getattr(self.strategy, "regression_rules", {}).get("regression_entry_mode") == "hybrid"
              and (self.portfolio.has_open_position(symbol)
                   or self.pending_orders_processor.queue.count_for_symbol(symbol) > 0))

   def _queue_order(self, order, tick) -> bool:
      if self._hybrid_symbol_busy(order.symbol):
         print(f"[order-rejected] order_id={order.order_id} symbol={order.symbol} "
               "reason=hybrid_symbol_busy", flush=True)
         return False
      # The strategy signal may refer to the last tick of the previous candle.
      # The replay clock is the actual order-creation/submission decision time.
      decision_at = self._market_clock or tick.timestamp
      snapshot = (self.trade_dataset.snapshot_for_order(order, decision_at)
                  if self.trade_dataset is not None else None)
      if self.trade_validator is not None and not self.trade_validator.before_order(order, decision_at, snapshot):
         return False
      if self.trade_dataset is not None:
         self.trade_dataset.on_order_created(order, decision_at, snapshot)
      self.pending_orders_processor.queue.add(order)
      if self.adaptive_optimizer is not None:
         self.adaptive_optimizer.on_order_queued(order)
      if self.learning_hooks is not None:
         self.learning_hooks.on_order_queued(order)
      return True

   def _monitor_positions(self, tick: Tick) -> None:
      positions = list(self.portfolio.positions.values())
      for position in positions:
         if position.symbol == tick.symbol and self.trading_schedule.position_due(position.opened_at, tick.timestamp):
            request = CloseRequest(position.trade_id, tick.timestamp,
                                   tick.bid if position.side.name == "BUY" else tick.ask, CloseReason.END_OF_WEEK)
            self.trade_monitor._mark_position_excursion(position, request.price)
            self.trading_schedule.stats["weekly_closed_positions"] += 1
         else:
            request = self.trade_closer.check(position, tick, self.trade_monitor)
         if request is None:
            continue

         report = self.trade_closer.close(self.portfolio, request)
         adaptive_details = (
            f" configuration_id={report.strategy_configuration_id} lookback={report.strategy_lookback_bars} "
            f"timeframe_minutes={report.strategy_timeframe_minutes} "
            f"global_validation={report.strategy_use_global_timeframe_validation} "
            f"trailing={report.trailing_stop_enabled}"
            if self.adaptive_optimizer is not None else ""
         )
         print(
            f"[trade-closed] trade_id={report.trade_id} symbol={report.symbol} "
            f"timestamp={report.closed_at.isoformat()} side={report.side.name.lower()} "
            f"quantity={report.quantity} entry_price={report.entry_price} "
            f"exit_price={report.exit_price} pnl={report.pnl:.2f} "
            f"reason={report.close_reason.name.lower()}{adaptive_details}",
            flush=True,
         )
         self.reports.add_trade(report)
