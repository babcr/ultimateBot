import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.regression_breakout import RegressionBreakoutTracker
from ultimateBot.backtester.strategy.regression_hybrid import RegressionHybridTracker
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel, RegressionReentryTracker
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.swing_regression_config import SwingRegressionConfig
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, parse_args, print_regression_parameters, print_swing_regression_diagnostics


START = datetime(2024, 1, 2, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=1)


def candle(index=0, open=105, high=112, low=104, close=109):
    from ultimateBot.backtester.market.candle_builder import Candle
    return Candle(START + index * INTERVAL, open, high, low, close)


def ask_for(bid):
    return replace(bid, open_price=bid.open_price + .1, high_price=bid.high_price + .1,
                   low_price=bid.low_price + .1, close_price=bid.close_price + .1)


def channel():
    return FrozenChannel({
        "bar_count": 15, "atr": 2., "reasons": [],
        "anchored_line": {"kind": "low", "intercept": 100., "slope": 0.},
        "floating_line": {"kind": "high", "intercept": 110., "slope": 0.},
    }, START - INTERVAL, 105, 105.1)


def breakout(index=0):
    return candle(index, open=109.4, high=111.1, low=109.2, close=111)


def retest(index=1):
    return candle(index, open=110.4, high=110.6, low=109.8, close=110.3)


def reclaim(index=2):
    return candle(index, open=110.3, high=111.2, low=110.25, close=111.05)


def hybrid_config(trailing=True):
    config = SwingRegressionConfig()
    return replace(config,
                   regression=replace(config.regression, entry_mode="hybrid", lookback_bars=15,
                                      min_market_volatility_ratio=0),
                   breakout=replace(config.breakout, confirmation_mode="confirmed"),
                   volatility=replace(config.volatility, candle_interval=INTERVAL, minimum_candles=1),
                   risk=replace(config.risk, risk_per_trade_percent=1, reward_risk_ratio=3, trailing_start_rr_ratio=1.5),
                   execution=replace(config.execution, balance=1000, trailing_stop=trailing))


def engine_for(config=None, *options):
    with patch.object(sys, "argv", ["run", *options]):
        return build_engine([], parse_args(config or hybrid_config()))


class HybridTrackerTests(unittest.TestCase):
    def setUp(self):
        self.failed = RegressionReentryTracker()
        self.breakout = RegressionBreakoutTracker()
        self.hybrid = RegressionHybridTracker(self.failed, self.breakout)
        self.rules = resolve_swing_regression_rules(hybrid_config())

    def evaluate(self, bid=None, frozen=None, symbol="EURUSD", fit=None):
        bid = bid or candle()
        return self.hybrid.evaluate(symbol, bid, ask_for(bid), bid.start_time + INTERVAL,
                                    INTERVAL, self.rules, frozen, fit)

    def test_failed_confirmation_keeps_its_original_exits_and_context(self):
        expected, _ = RegressionReentryTracker().evaluate("EURUSD", candle(), ask_for(candle()),
                                                         START + INTERVAL, INTERVAL, self.rules, channel())
        signal, analysis = self.evaluate(frozen=channel())
        self.assertEqual(signal.side, expected.side)
        self.assertEqual(signal.stop_loss_price, expected.stop_loss_price)
        self.assertEqual(signal.take_profit_price, 105)
        self.assertEqual(signal.entry_context["entry_mode"], "failed-breakout")
        self.assertEqual(signal.entry_context["strategy_entry_mode"], "hybrid")
        self.assertEqual(analysis["selected_entry_mode"], "failed-breakout")
        self.assertFalse(self.failed.events or self.breakout.events)

    def test_retest_confirmation_cancels_a_pending_failed_branch(self):
        self.evaluate(breakout(), channel())
        self.assertIn("EURUSD", self.failed.events)
        self.assertIn("EURUSD", self.breakout.events)
        self.assertIsNone(self.evaluate(retest())[0])
        signal, analysis = self.evaluate(reclaim())
        self.assertEqual(signal.side, SignalSide.BUY)
        self.assertIsNone(signal.take_profit_price)
        self.assertAlmostEqual(signal.stop_loss_price, 109.4)
        self.assertAlmostEqual(signal.minimum_entry_price, 110.7)
        self.assertEqual(signal.entry_context["entry_mode"], "breakout-retest")
        self.assertEqual(analysis["cancelled_pending_branches"], ["failed-breakout"])
        self.assertFalse(self.failed.events or self.breakout.events)

    def test_failed_confirmation_cancels_a_pending_retest_without_reusing_it(self):
        self.evaluate(breakout(), channel())
        signal, analysis = self.evaluate(candle(1, open=110.3, high=110.5, low=109.2, close=109.8))
        self.assertEqual(signal.side, SignalSide.SELL)
        self.assertEqual(analysis["cancelled_pending_branches"], ["breakout-retest"])
        self.assertFalse(self.failed.events or self.breakout.events)
        self.assertIsNone(self.evaluate(reclaim())[0])

    def test_failed_expiry_and_invalid_new_fit_do_not_interrupt_the_pending_retest(self):
        self.rules["regression_reentry_max_bars"] = 0
        self.evaluate(breakout(), channel())
        self.evaluate(retest())
        self.assertNotIn("EURUSD", self.failed.events)
        self.assertIn("EURUSD", self.breakout.events)
        signal, analysis = self.evaluate(reclaim(), fit={"reasons": ["insufficient_parallelism"]})
        self.assertEqual(signal.entry_context["entry_mode"], "breakout-retest")
        self.assertEqual(analysis["branches"]["failed-breakout"]["reasons"], ["insufficient_parallelism"])
        self.assertEqual(signal.entry_context["channel_upper"], 110)

    def test_new_fit_is_ignored_by_each_pending_branch(self):
        self.evaluate(breakout(), channel())
        changed = channel()
        changed.analysis["anchored_line"]["intercept"] = 200
        changed.analysis["floating_line"]["intercept"] = 210
        self.evaluate(retest(), changed)
        signal, _ = self.evaluate(reclaim(), changed)
        self.assertEqual(signal.entry_context["channel_upper"], 110)

    def test_symbol_states_are_isolated(self):
        self.evaluate(breakout(), channel(), symbol="EURUSD")
        self.evaluate(breakout(), channel(), symbol="GBPUSD")
        self.evaluate(candle(1, open=110.3, high=110.5, low=109.2, close=109.8))
        self.assertEqual(set(self.failed.events), {"GBPUSD"})
        self.assertEqual(set(self.breakout.events), {"GBPUSD"})

    def test_opposite_side_simultaneous_signals_are_both_consumed_and_rejected(self):
        failed = Signal("EURUSD", START, SignalSide.SELL, entry_context={"entry_mode": "failed-breakout"})
        continuation = Signal("EURUSD", START, SignalSide.BUY, entry_context={"entry_mode": "breakout-retest"})
        self.failed.events["EURUSD"] = object()
        self.breakout.events["EURUSD"] = object()
        with patch.object(self.failed, "evaluate", return_value=(failed, {"direction": "sell", "reasons": []})), \
                patch.object(self.breakout, "evaluate", return_value=(continuation, {"direction": "buy", "reasons": []})):
            signal, analysis = self.evaluate()
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["hybrid_conflicting_signals"])
        self.assertFalse(self.failed.events or self.breakout.events)
        self.assertEqual(self.hybrid.diagnostics_summary()["conflicting_confirmations"], 1)
        self.assertFalse(self.hybrid.selected)

    def test_same_side_duplicates_choose_failed_once_without_mutating_original_signal(self):
        failed = Signal("EURUSD", START, SignalSide.BUY, entry_context={"entry_mode": "failed-breakout"})
        continuation = replace(failed, entry_context={"entry_mode": "breakout-retest"})
        with patch.object(self.failed, "evaluate", return_value=(failed, {"direction": "buy", "reasons": []})), \
                patch.object(self.breakout, "evaluate", return_value=(continuation, {"direction": "buy", "reasons": []})):
            signal, _ = self.evaluate(frozen=channel())
        self.assertEqual(signal.entry_context["entry_mode"], "failed-breakout")
        self.assertEqual(signal.entry_context["hybrid_confirmation_count"], 2)
        self.assertNotIn("strategy_entry_mode", failed.entry_context)
        self.assertEqual(self.hybrid.selected["failed-breakout"], 1)
        self.assertEqual(self.hybrid.confirmed["breakout-retest"], 1)
        self.assertEqual(self.hybrid.diagnostics_summary()["same_side_duplicates"], 1)

    def test_fit_failure_reasons_are_retained_for_both_idle_branches(self):
        signal, analysis = self.evaluate(fit={"reasons": ["missing_swing_family"]})
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["hybrid_no_confirmation"])
        for result in analysis["branches"].values():
            self.assertEqual(result["reasons"], ["missing_swing_family"])


class HybridStrategyTests(unittest.TestCase):
    def setUp(self):
        self.strategy = SwingRegressionStrategy(config=hybrid_config())
        self.bid = [candle(i, high=108, low=102, close=105) for i in range(16)]
        self.bid[-1] = breakout(15)
        self.market = SimpleNamespace(get_candles=lambda *args, **kwargs: self.bid, get_point_size=lambda _: .01)
        self.context = StrategyContext(Portfolio(1000), self.market)

    def batch(self):
        return TickBatch("EURUSD", [Tick(self.bid[-1].start_time + INTERVAL, "EURUSD", 111, 111.1, 1, 1)])

    def circuits(self, *args):
        return {"bid": self.bid, "ask": [ask_for(bar) for bar in self.bid]}

    def test_one_pre_event_fit_is_shared_and_each_closed_candle_is_evaluated_once(self):
        with patch.object(self.strategy, "_get_structure_candles", side_effect=self.circuits), \
                patch.object(self.strategy, "_evaluate_regression_pipeline", return_value=(None, deepcopy(channel().analysis))) as fit:
            self.assertEqual(self.strategy.on_market_data(self.batch(), self.context), [])
            self.assertEqual(self.strategy.on_market_data(self.batch(), self.context), [])
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.args[1][-1].start_time, START + 14 * INTERVAL)
            self.assertTrue(fit.call_args.kwargs["channel_only"])
            self.assertNotIn("context", fit.call_args.kwargs)
            self.bid = self.bid[1:] + [retest(16)]
            self.assertEqual(self.strategy.on_market_data(self.batch(), self.context), [])
            self.bid = self.bid[1:] + [reclaim(17)]
            signals = self.strategy.on_market_data(self.batch(), self.context)
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].entry_context["entry_mode"], "breakout-retest")
            self.assertEqual(fit.call_count, 1)
        diagnostics = self.strategy.diagnostics_summary()
        self.assertEqual(diagnostics["analysis_windows"], 3)
        self.assertEqual(diagnostics["hybrid"]["branches"]["breakout-retest"]["selected_signals"], 1)

    def test_open_position_or_pending_order_blocks_both_branches_and_clears_states(self):
        for exposure in ("position", "order"):
            with self.subTest(exposure=exposure):
                strategy = SwingRegressionStrategy(config=hybrid_config())
                strategy._reentry.events["EURUSD"] = object()
                strategy._breakout.events["EURUSD"] = object()
                portfolio = Portfolio(1000)
                if exposure == "position":
                    portfolio.positions[1] = Position(1, "EURUSD", OrderSide.BUY, 1, START, 105)
                context = StrategyContext(portfolio, self.market, int(exposure == "order"))
                with patch.object(strategy, "_get_structure_candles", side_effect=self.circuits), \
                        patch.object(strategy, "_evaluate_regression_pipeline") as fit:
                    self.assertEqual(strategy.on_market_data(self.batch(), context), [])
                    fit.assert_not_called()
                self.assertFalse(strategy._reentry.events or strategy._breakout.events)
                self.assertEqual(strategy.last_analysis["EURUSD"]["reasons"], ["hybrid_symbol_busy"])

    def test_another_symbols_position_does_not_block_this_symbol(self):
        self.context.portfolio.positions[1] = Position(1, "GBPUSD", OrderSide.BUY, 1, START, 105)
        with patch.object(self.strategy, "_get_structure_candles", side_effect=self.circuits), \
                patch.object(self.strategy, "_evaluate_regression_pipeline", return_value=(None, deepcopy(channel().analysis))) as fit:
            self.strategy.on_market_data(self.batch(), self.context)
        self.assertEqual(fit.call_count, 1)
        self.assertIn("EURUSD", self.strategy._breakout.events)

    def test_configuration_changes_clear_both_branches_only_for_the_selected_symbol(self):
        for tracker in (self.strategy._reentry, self.strategy._breakout):
            tracker.events.update(EURUSD=object(), GBPUSD=object())
        self.strategy.set_history_lookback_bars(20, "EURUSD")
        self.assertEqual(set(self.strategy._reentry.events), {"GBPUSD"})
        self.assertEqual(set(self.strategy._breakout.events), {"GBPUSD"})
        self.strategy.set_analysis_interval(2 * INTERVAL)
        self.assertFalse(self.strategy._reentry.events or self.strategy._breakout.events)


class HybridEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = engine_for()
        self.tick = Tick(START, "EURUSD", 109, 109.1, 1, 1)
        self.signal = Signal("EURUSD", START, SignalSide.SELL, stop_loss_price=112.5,
                             take_profit_price=105, minimum_reward_risk_ratio=1,
                             entry_context={"entry_mode": "failed-breakout", "strategy_entry_mode": "hybrid"})
        self.engine.risk_manager = RiskManager(minimum_candles=1, risk_per_trade=1, max_spread_ratio=100,
                                               limit_spread_ratio=100, reward_risk_ratio=3, trailing_start_rr_ratio=1.5)
        for price in (108, 110, 109):
            self.engine.risk_manager.on_tick(Tick(START, "EURUSD", price, price + .1, 1, 1))

    def test_config_cli_roundtrip_and_exit_parameters(self):
        rules = self.engine.strategy.regression_rules
        self.assertEqual(rules["regression_entry_mode"], "hybrid")
        self.assertEqual(self.engine.strategy.history_capacity_for_lookback(15), 16)
        self.assertFalse(rules["regression_require_level_reaction"])
        self.assertFalse(rules["regression_use_global_timeframe_validation"])
        self.assertEqual(self.engine.strategy.config.with_rules(rules).regression.entry_mode, "hybrid")
        with redirect_stdout(io.StringIO()) as out:
            print_regression_parameters(rules)
            print_swing_regression_diagnostics(self.engine)
        self.assertIn("config.reentry", out.getvalue())
        self.assertIn("config.breakout", out.getvalue())
        self.assertIn("Hybrid diagnostics", out.getvalue())

    def test_adaptation_is_rejected_but_trailing_is_allowed(self):
        for option in ("--adaptive-strategy", "--contextual-bandit"):
            with patch.object(sys, "argv", ["run", option]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(hybrid_config())
        self.assertTrue(engine_for(hybrid_config(), "--trailing-stop").trade_closer.trailing_stop_enabled)

    def test_only_one_order_is_queued_for_the_symbol_even_if_signals_are_opposite(self):
        with redirect_stdout(io.StringIO()):
            self.engine._process_signal(self.signal, self.tick)
        order = self.engine.pending_orders_processor.queue.get_for_symbol("EURUSD")[0]
        self.assertFalse(order.trailing_stop_enabled)
        self.assertEqual(order.take_profit, 105)
        with patch.object(self.engine.risk_manager, "evaluate") as risk, redirect_stdout(io.StringIO()):
            self.engine._process_signal(replace(self.signal, side=SignalSide.BUY), self.tick)
            risk.assert_not_called()
        self.assertEqual(self.engine.pending_orders_processor.queue.count_for_symbol("EURUSD"), 1)

    def test_open_position_rejects_another_entry_before_risk_evaluation(self):
        self.engine.portfolio.positions[1] = Position(1, "EURUSD", OrderSide.BUY, 1, START, 105)
        with patch.object(self.engine.risk_manager, "evaluate") as risk, redirect_stdout(io.StringIO()):
            self.engine._process_signal(self.signal, self.tick)
        risk.assert_not_called()
        self.assertEqual(len(self.engine.pending_orders_processor.queue), 0)

    def test_queue_guard_applies_before_dataset_capture(self):
        order = OrderRequest(1, "EURUSD", START, START, START, OrderSide.SELL, 1)
        self.engine.pending_orders_processor.queue.add(order)
        self.engine.trade_dataset = SimpleNamespace(snapshot_for_order=lambda *args: self.fail("Must not capture a rejected duplicate"))
        with redirect_stdout(io.StringIO()):
            self.assertFalse(self.engine._queue_order(replace(order, order_id=2), self.tick))
        self.assertEqual(len(self.engine.pending_orders_processor.queue), 1)

    def test_engine_passes_per_symbol_pending_count_into_the_strategy_context(self):
        self.engine.pending_orders_processor.queue.add(OrderRequest(1, "EURUSD", START, START, START, OrderSide.SELL, 1))
        self.engine.pending_orders_processor.queue.add(OrderRequest(2, "GBPUSD", START, START, START, OrderSide.SELL, 1))
        with patch.object(self.engine.strategy, "on_market_data", return_value=[]) as evaluate:
            self.engine._run_strategy(TickBatch("EURUSD", [self.tick]))
        self.assertEqual(evaluate.call_args.args[1].pending_orders_for_symbol, 1)

    def test_selected_event_is_consumed_even_if_risk_rejects_it(self):
        tracker = self.engine.strategy._hybrid
        rules = self.engine.strategy.regression_rules
        bar = breakout()
        tracker.evaluate("EURUSD", bar, ask_for(bar), START + INTERVAL, INTERVAL, rules, channel())
        bar = candle(1, open=110.3, high=110.5, low=109.2, close=109.8)
        signal, _ = tracker.evaluate("EURUSD", bar, ask_for(bar), START + 2 * INTERVAL, INTERVAL, rules, None)
        with patch.object(self.engine.risk_manager, "evaluate", return_value=None):
            self.engine._process_signal(signal, self.tick)
        self.assertEqual(len(self.engine.pending_orders_processor.queue), 0)
        self.assertFalse(self.engine.strategy._reentry.events or self.engine.strategy._breakout.events)

    def test_other_modes_keep_their_existing_multiple_order_policy(self):
        config = hybrid_config()
        config = replace(config, regression=replace(config.regression, entry_mode="breakout-retest"))
        engine = engine_for(config)
        order = OrderRequest(1, "EURUSD", START, START, START, OrderSide.SELL, 1)
        self.assertTrue(engine._queue_order(order, self.tick))
        self.assertTrue(engine._queue_order(replace(order, order_id=2), self.tick))
        self.assertEqual(len(engine.pending_orders_processor.queue), 2)


class HybridLifecycleTests(unittest.TestCase):
    def test_real_engine_handles_both_entries_with_independent_exit_policies(self):
        ticks = []
        for index in range(15):
            for seconds, price in ((0, 105), (10, 108), (20, 102), (50, 105)):
                ticks.append(Tick(START + index * INTERVAL + timedelta(seconds=seconds), "EURUSD", price, price + .1, 1, 1))
        patterns = {
            15: ((0, 105), (10, 112), (20, 104), (50, 109)),
            16: ((0, 109), (5, 109), (15, 104.9), (50, 105)),
            17: ((0, 109.4), (10, 111.1), (20, 109.2), (50, 111)),
            18: ((0, 110.4), (10, 110.6), (20, 109.8), (50, 110.3)),
            19: ((0, 110.3), (10, 111.2), (20, 110.25), (50, 111.05)),
            20: ((0, 111.05), (5, 111.05), (10, 114), (20, 119), (50, 114.6)),
        }
        for index, pattern in patterns.items():
            for seconds, price in pattern:
                ticks.append(Tick(START + index * INTERVAL + timedelta(seconds=seconds), "EURUSD", price, price + .1, 1, 1))
        for trailing in (True, False):
            with self.subTest(trailing=trailing):
                engine = engine_for(hybrid_config(trailing))
                engine.feed = ticks
                with patch.object(engine.strategy, "_evaluate_regression_pipeline", side_effect=lambda *args, **kwargs: (None, deepcopy(channel().analysis))), \
                        redirect_stdout(io.StringIO()):
                    engine.run()
                self.assertEqual(len(engine.reports.trades), 2)
                self.assertEqual(engine.trade_entry_modes, {1: "failed-breakout", 2: "breakout-retest"})
                failed, continuation = engine.reports.trades
                self.assertEqual(failed.close_reason, CloseReason.TAKE_PROFIT)
                self.assertFalse(failed.trailing_stop_enabled)
                self.assertEqual(continuation.trailing_stop_enabled, trailing)
                self.assertEqual(continuation.close_reason, CloseReason.TRAILING_STOP if trailing else CloseReason.TAKE_PROFIT)
                self.assertEqual(len(engine.pending_orders_processor.queue), 0)
                self.assertFalse(engine.portfolio.positions)
                with redirect_stdout(io.StringIO()) as out:
                    print_swing_regression_diagnostics(engine)
                self.assertIn("failed-breakout: raw_confirmations=1 selected_signals=1 closed=1", out.getvalue())
                self.assertIn("breakout-retest: raw_confirmations=1 selected_signals=1 closed=1", out.getvalue())


if __name__ == "__main__":
    unittest.main()
