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
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.regression_breakout import RegressionBreakoutTracker
from ultimateBot.backtester.strategy.regression_breakout_config import RegressionBreakoutParameters
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, parse_args, print_swing_regression_diagnostics
from ultimateBot.run_regression_breakout import breakout_config
from ultimateBot.compare_regression_breakout import summarize


START = datetime(2024, 1, 2, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=1)


def candle(index=0, open=109, high=112, low=108, close=111):
    return Candle(START + index * INTERVAL, open, high, low, close)


def ask_for(bid):
    return replace(bid, open_price=bid.open_price + .1, high_price=bid.high_price + .1,
                   low_price=bid.low_price + .1, close_price=bid.close_price + .1)


def channel(slope=0):
    return FrozenChannel({
        "bar_count": 15, "atr": 2., "reasons": [],
        "anchored_line": {"kind": "low", "intercept": 100., "slope": slope},
        "floating_line": {"kind": "high", "intercept": 110., "slope": slope},
    }, START - INTERVAL, 105., 105.1)


class BreakoutEventTests(unittest.TestCase):
    def setUp(self):
        self.tracker = RegressionBreakoutTracker()
        self.rules = resolve_swing_regression_rules({"regression_entry_mode": "breakout-retest",
                                                    "regression_breakout_confirmation_mode": "legacy"})

    def evaluate(self, bid=None, frozen=None, symbol="EURUSD", ask=None):
        bid = bid or candle()
        return self.tracker.evaluate(symbol, bid, ask or ask_for(bid), bid.start_time + INTERVAL,
                                     INTERVAL, self.rules, frozen)

    def buy_signal(self):
        self.evaluate(frozen=channel())
        return self.evaluate(candle(1, open=110, low=109.8, close=110.6))

    def test_buy_requires_later_retest_and_uses_bid_extreme(self):
        signal, analysis = self.evaluate(frozen=channel())
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_retest"])
        signal, analysis = self.evaluate(candle(1, open=110, low=109.8, close=110.6))
        self.assertEqual(signal.side, SignalSide.BUY)
        self.assertAlmostEqual(signal.stop_loss_price, 109.4)
        self.assertIsNone(signal.take_profit_price)
        self.assertEqual((signal.minimum_entry_price, signal.maximum_entry_price), (110, 111.5))
        self.assertEqual(signal.entry_context["breakout_bars"], 1)
        self.assertEqual(analysis["reasons"], [])
        self.assertFalse(self.tracker.events)
        self.assertIsNone(self.evaluate(candle(2))[0])

    def test_sell_is_symmetric_with_ask_extreme(self):
        self.evaluate(candle(open=101, high=102, low=98, close=99), frozen=channel())
        signal, _ = self.evaluate(candle(1, open=100, high=100.2, low=99, close=99.3))
        self.assertEqual(signal.side, SignalSide.SELL)
        self.assertAlmostEqual(signal.stop_loss_price, 100.7)
        self.assertEqual((signal.minimum_entry_price, signal.maximum_entry_price), (98.5, 100))

    def test_wick_and_insufficient_close_are_not_breakouts(self):
        for close in (109, 110, 110.3):
            with self.subTest(close=close):
                signal, analysis = self.evaluate(candle(close=close), frozen=channel())
                self.assertIsNone(signal)
                self.assertEqual(analysis["reasons"], ["breakout_no_close_outside"])

    def test_prior_close_must_be_inside_and_ambiguous_candles_are_rejected(self):
        _, analysis = self.evaluate(frozen=replace(channel(), previous_bid_close=111))
        self.assertEqual(analysis["reasons"], ["breakout_previous_close_outside"])
        _, analysis = self.evaluate(candle(low=99), frozen=channel())
        self.assertEqual(analysis["reasons"], ["breakout_ambiguous_candle"])

    def test_frozen_lines_are_projected_and_replacement_fit_is_ignored(self):
        self.evaluate(candle(open=111, high=114, low=110, close=112.5), frozen=channel(.1))
        signal, _ = self.evaluate(candle(1, open=111.6, high=113, low=111.4, close=112), frozen=channel(5))
        self.assertEqual(signal.entry_context["channel_upper"], 111.6)
        self.assertEqual(signal.entry_context["anchored_line"]["slope"], .1)

    def test_retest_can_confirm_later_and_retains_its_extreme(self):
        self.evaluate(frozen=channel())
        _, analysis = self.evaluate(candle(1, open=110.5, high=111, low=109.5, close=109.9))
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_confirmation"])
        signal, _ = self.evaluate(candle(2, open=110, low=110, close=110.6))
        self.assertAlmostEqual(signal.stop_loss_price, 109.1)
        self.assertEqual(signal.entry_context["breakout_bars"], 2)

    def test_no_touch_and_wrong_candle_direction_do_not_confirm(self):
        self.evaluate(frozen=channel())
        _, analysis = self.evaluate(candle(1, open=111, low=110.5, close=111.2))
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_retest"])
        _, analysis = self.evaluate(candle(2, open=111.2, low=110, close=110.8))
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_confirmation"])

    def test_deep_close_inside_or_opposite_boundary_invalidates(self):
        for bid, reason in ((candle(1, low=109, close=109.5), "close_back_inside"),
                            (candle(1, low=99, close=110.5), "opposite_boundary_touched")):
            with self.subTest(reason=reason):
                self.evaluate(frozen=channel())
                _, analysis = self.evaluate(bid)
                self.assertEqual(analysis["reasons"], ["breakout_" + reason])
                self.assertFalse(self.tracker.events)

    def test_expiry_is_inclusive_and_gap_clears_pending_event(self):
        self.evaluate(frozen=channel())
        for index in (1, 2):
            self.evaluate(candle(index, low=110.6))
        signal, _ = self.evaluate(candle(3, open=110, low=109.8, close=110.6))
        self.assertIsNotNone(signal)
        self.evaluate(frozen=channel())
        for index in (1, 2, 3):
            self.evaluate(candle(index, low=110.6))
        _, analysis = self.evaluate(candle(4))
        self.assertEqual(analysis["reasons"], ["breakout_expired"])
        self.evaluate(frozen=channel())
        _, analysis = self.evaluate(candle(2))
        self.assertEqual(analysis["reasons"], ["breakout_history_gap"])
        self.assertFalse(self.tracker.events)

    def test_confirmation_too_far_including_spread_is_consumed(self):
        self.evaluate(frozen=channel())
        signal, analysis = self.evaluate(candle(1, open=110, low=110, close=111.45))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_entry_too_far"])
        self.assertFalse(self.tracker.events)

    def test_two_finite_non_crossing_lines_are_required(self):
        for kind in ("missing", "crossing", "future_crossing", "invalid_atr"):
            frozen = channel()
            if kind == "missing":
                frozen.analysis["floating_line"] = None
                reason = "requires_two_lines"
            elif kind == "invalid_atr":
                frozen.analysis["atr"] = float("nan")
                reason = "invalid_atr"
            else:
                frozen.analysis["floating_line"]["intercept"] = 99 if kind == "crossing" else 110
                frozen.analysis["floating_line"]["slope"] = 0 if kind == "crossing" else -.6
                reason = "invalid_channel"
            with self.subTest(kind=kind):
                _, analysis = self.evaluate(frozen=frozen)
                self.assertEqual(analysis["reasons"], ["breakout_" + reason])

    def test_pending_events_are_isolated_by_symbol(self):
        self.evaluate(frozen=channel())
        self.evaluate(frozen=channel(), symbol="GBPUSD")
        self.evaluate(candle(1, open=110, low=109.8, close=110.6))
        self.assertEqual(set(self.tracker.events), {"GBPUSD"})


class BreakoutIntegrationTests(unittest.TestCase):
    def risk_evaluation(self, signal, bid=110.6, ask=110.7, **options):
        parameters = dict(minimum_candles=1, risk_per_trade=1, max_spread_ratio=100,
                          limit_spread_ratio=100, reward_risk_ratio=3, trailing_start_rr_ratio=1.5)
        parameters.update(options)
        manager = RiskManager(**parameters)
        for price in (108, 112, 110):
            manager.on_tick(Tick(START, "EURUSD", price, price + .1, 1, 1))
        return manager.evaluate(signal, Portfolio(initial_balance=1000), Tick(START, "EURUSD", bid, ask, 1, 1))

    def test_stop_only_sizes_risk_and_reference_target_for_both_sides(self):
        for side, stop, entry in ((SignalSide.BUY, 109.4, 110.7), (SignalSide.SELL, 112, 110.6)):
            with self.subTest(side=side):
                signal = Signal("EURUSD", START, side, stop_loss_price=stop)
                result = self.risk_evaluation(signal)
                distance = abs(entry - stop)
                direction = 1 if side == SignalSide.BUY else -1
                self.assertAlmostEqual(result.quantity * distance, 10)
                self.assertAlmostEqual(result.take_profit, entry + direction * (3 * (distance + .1) + .1))
                self.assertAlmostEqual(result.trailing_stop_activation_price, entry + direction * 1.5 * distance)
                self.assertAlmostEqual(result.take_profit_distance, abs(result.take_profit - entry))

    def test_executable_entry_outside_zone_or_invalid_stop_is_rejected(self):
        signal = Signal("EURUSD", START, SignalSide.BUY, stop_loss_price=109.4,
                        minimum_entry_price=110, maximum_entry_price=111.5)
        self.assertIsNotNone(self.risk_evaluation(signal))
        for bid, ask in ((111.45, 111.55), (109.8, 109.9)):
            self.assertIsNone(self.risk_evaluation(signal, bid=bid, ask=ask))
        for stop in (float("nan"), float("inf"), 111):
            self.assertIsNone(self.risk_evaluation(replace(signal, stop_loss_price=stop)))
        self.assertIsNone(self.risk_evaluation(replace(signal, maximum_entry_price=float("nan"))))

    def test_entry_zone_also_rejects_a_limit_fallback_inside_the_channel(self):
        signal = Signal("EURUSD", START, SignalSide.BUY, stop_loss_price=109.4)
        # Live ask is inside the allowed entry zone, but the fallback limit is below it.
        self.assertIsNotNone(self.risk_evaluation(signal, bid=110.001, ask=110.101, limit_spread_ratio=0))
        bounded = replace(signal, minimum_entry_price=110, maximum_entry_price=111.5)
        self.assertIsNone(self.risk_evaluation(bounded, bid=110.001, ask=110.101, limit_spread_ratio=0))

    def test_sell_entry_zone_is_checked_on_the_executable_bid(self):
        signal = Signal("EURUSD", START, SignalSide.SELL, stop_loss_price=100.7,
                        minimum_entry_price=98.5, maximum_entry_price=100)
        self.assertIsNotNone(self.risk_evaluation(signal, bid=99.3, ask=99.4))
        self.assertIsNone(self.risk_evaluation(signal, bid=98.45, ask=98.55))
        self.assertIsNone(self.risk_evaluation(signal, bid=100.05, ask=100.15))

    def test_instrument_minimum_stop_is_used_for_quantity_target_and_trailing(self):
        terms = SimpleNamespace(
            account_profit_multiplier=1, account_currency="USD",
            stop_distance=lambda entry, distance, spread, buying: max(distance, 2),
            target_price=lambda price, buying: price,
            quantity=lambda quantity, fixed: quantity,
        )
        signal = Signal("EURUSD", START, SignalSide.BUY, stop_loss_price=109.4)
        result = self.risk_evaluation(signal, instrument_rules_provider=lambda _: terms)
        self.assertAlmostEqual(result.stop_loss, 108.7)
        self.assertAlmostEqual(result.quantity, 5)
        self.assertAlmostEqual(result.take_profit, 110.7 + 3 * 2.1 + .1)
        self.assertAlmostEqual(result.trailing_stop_activation_price, 113.7)

    def test_actual_order_fill_and_trailing_continue_beyond_reference_tp(self):
        signal = Signal("EURUSD", START, SignalSide.BUY, stop_loss_price=109.4,
                        entry_context={"entry_mode": "breakout-retest"})
        result = self.risk_evaluation(signal)
        order = OrderBuilder().build(signal, result)
        closer = TradeCloser(True)
        closer.prepare_order(order)
        self.assertTrue(order.trailing_stop_enabled)
        self.assertIsNone(order.take_profit)
        self.assertEqual(order.strategy_entry_context, signal.entry_context)
        execution = ExecutionModel().try_execute(order, Tick(START + timedelta(seconds=1), "EURUSD", 110.6, 110.7, 1, 1))
        portfolio = Portfolio(1000)
        position = PositionManager().open_position(portfolio, execution.fill, order)
        closer.prepare_position(position)
        with redirect_stdout(io.StringIO()):
            price = result.take_profit + 3
            self.assertIsNone(closer.check(position, Tick(START + INTERVAL, "EURUSD", price, price + .1, 1, 1), TradeMonitor()))
            self.assertGreater(position.stop_loss, position.entry_price)
            request = closer.check(position, Tick(START + 2 * INTERVAL, "EURUSD", position.stop_loss, position.stop_loss + .1, 1, 1), TradeMonitor())
        self.assertEqual(request.reason, CloseReason.TRAILING_STOP)

    def test_config_cli_roundtrip_and_fixed_baseline(self):
        config = breakout_config()
        config = replace(config, breakout=replace(config.breakout, max_bars=5))
        with patch.object(sys, "argv", ["run", "--regression-breakout-stop-buffer-atr", "0.4"]):
            args = parse_args(config)
        engine = build_engine([], args)
        self.assertEqual(engine.strategy.config.breakout.max_bars, 5)
        self.assertEqual(engine.strategy.config.breakout.stop_buffer_atr, .4)
        self.assertEqual(engine.strategy.history_capacity_for_lookback(15), 16)
        self.assertTrue(engine.trade_closer.trailing_stop_enabled)
        self.assertEqual(engine.risk_manager.risk_config.trailing_start_rr_ratio, 1.5)
        self.assertFalse(engine.strategy.get_global_timeframe_validation_enabled())
        self.assertEqual(config.with_rules(config.to_rules()), config)
        with redirect_stdout(io.StringIO()) as out:
            print_swing_regression_diagnostics(engine)
        self.assertIn("Confirmed breakout retests", out.getvalue())
        # The ordinary launcher also supports the same entry with fixed TP for comparison.
        with patch.object(sys, "argv", ["run", "--strategy", "swing-regression", "--regression-entry-mode", "breakout-retest"]):
            fixed = build_engine([], parse_args())
        self.assertFalse(fixed.trade_closer.trailing_stop_enabled)
        for option in ("--adaptive-strategy", "--contextual-bandit"):
            with patch.object(sys, "argv", ["run", option]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(config)

    def test_invalid_parameters_are_rejected(self):
        for name, value in (("max_bars", 0), ("max_bars", 1.5), ("min_close_atr", -.1),
                            ("stop_buffer_atr", float("nan")), ("retest_tolerance_atr", float("inf"))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_swing_regression_rules({"regression_breakout_" + name: value})

    def test_shared_quality_filters_remain_active_but_directional_gates_are_disabled(self):
        strategy = SwingRegressionStrategy(config=breakout_config())
        rules = strategy.regression_rules
        for name in ("min_parallelism", "line_max_volatility_atr", "max_swing_density", "min_market_volatility_ratio"):
            self.assertEqual(rules["regression_" + name], getattr(strategy.config.regression, name))
        self.assertFalse(rules["regression_require_level_reaction"])
        self.assertFalse(rules["regression_require_candle_confirmation"])

    def test_strategy_fits_before_breakout_once_and_uses_pending_frozen_channel(self):
        config = breakout_config()
        config = replace(config, breakout=replace(config.breakout, confirmation_mode="legacy"))
        strategy = SwingRegressionStrategy(config=config, analysis_interval=INTERVAL)
        bid = [candle(i, high=108, low=102, close=105) for i in range(16)]
        bid[-1] = candle(15)
        quotes = {"bid": bid, "ask": [ask_for(c) for c in bid]}
        market = SimpleNamespace(get_candles=lambda *args, **kwargs: quotes["bid"], get_point_size=lambda _: .01)
        batch = TickBatch("EURUSD", [Tick(START + 16 * INTERVAL, "EURUSD", 111, 111.1, 1, 1)])
        with patch.object(strategy, "_get_structure_candles", side_effect=lambda *args: quotes), \
                patch.object(strategy, "_evaluate_regression_pipeline", return_value=(None, deepcopy(channel().analysis))) as fit:
            self.assertEqual(strategy.on_market_data(batch, SimpleNamespace(market_state=market)), [])
            self.assertEqual(fit.call_args.args[1][-1].start_time, START + 14 * INTERVAL)
            self.assertTrue(fit.call_args.kwargs["channel_only"])
            self.assertNotIn("context", fit.call_args.kwargs)
            self.assertEqual(strategy.on_market_data(batch, SimpleNamespace(market_state=market)), [])
            bid = bid[1:] + [candle(16, open=110, low=109.8, close=110.6)]
            quotes.update(bid=bid, ask=[ask_for(c) for c in bid])
            signals = strategy.on_market_data(batch, SimpleNamespace(market_state=market))
            self.assertEqual(len(signals), 1)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(strategy.on_market_data(batch, SimpleNamespace(market_state=market)), [])

    def test_history_requirements_and_configuration_changes_clear_events(self):
        config = breakout_config()
        config = replace(config, regression=replace(config.regression, market_volatility_window_bars=40))
        strategy = SwingRegressionStrategy(config=config)
        self.assertEqual(strategy.minimum_candles, 41)
        self.assertEqual(strategy.history_capacity_for_lookback(15), 41)
        strategy._breakout.events["EURUSD"] = object()
        strategy.set_history_lookback_bars(20, "EURUSD")
        self.assertFalse(strategy._breakout.events)
        strategy._breakout.events["EURUSD"] = object()
        strategy.set_analysis_interval(INTERVAL, "EURUSD")
        self.assertFalse(strategy._breakout.events)


class ConfirmedBreakoutTests(unittest.TestCase):
    def setUp(self):
        self.tracker = RegressionBreakoutTracker()
        self.rules = resolve_swing_regression_rules({"regression_entry_mode": "breakout-retest",
                                                    "regression_breakout_confirmation_mode": "confirmed"})

    def evaluate(self, bid, ask=None, frozen=None):
        return self.tracker.evaluate("EURUSD", bid, ask or ask_for(bid), bid.start_time + INTERVAL,
                                     INTERVAL, self.rules, frozen)

    def breakout(self):
        return candle(open=109.4, high=111.1, low=109.2, close=111)

    def retest(self, index=1):
        return candle(index, open=110.4, high=110.6, low=109.8, close=110.3)

    def reclaim(self, index=2):
        return candle(index, open=110.3, high=111.2, low=110.25, close=111.05)

    def arm_retest(self):
        self.evaluate(self.breakout(), frozen=channel())
        return self.evaluate(self.retest())

    def test_color_and_touch_alone_do_not_confirm_then_later_reclaim_does(self):
        self.evaluate(self.breakout(), frozen=channel())
        signal, analysis = self.evaluate(candle(1, open=110, high=110.6, low=109.8, close=110.5))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_reclaim"])
        signal, analysis = self.evaluate(self.reclaim())
        self.assertEqual(signal.side, SignalSide.BUY)
        self.assertAlmostEqual(signal.stop_loss_price, 109.4)
        self.assertAlmostEqual(signal.minimum_entry_price, 110.7)
        self.assertEqual(signal.maximum_entry_price, 111.5)
        self.assertAlmostEqual(signal.entry_context["retest_trigger"], 110.6)
        self.assertEqual(signal.entry_context["confirmation_mode"], "confirmed")
        self.assertEqual(analysis["reasons"], [])
        self.assertFalse(self.tracker.events)

    def test_retest_extreme_is_from_an_earlier_candle_not_the_confirmation_candle(self):
        self.arm_retest()
        # Its own high is 111.2, above its close. The previously closed trigger is 110.6.
        signal, _ = self.evaluate(self.reclaim())
        self.assertIsNotNone(signal)
        self.assertEqual(signal.entry_context["retest_trigger"], 110.6)
        self.assertGreater(self.reclaim().high_price, self.reclaim().close_price)

    def test_sell_rules_mirror_buy_quotes_with_spread(self):
        def mirrored_quotes(bid):
            ask = ask_for(bid)
            def mirror(quote):
                return replace(quote, open_price=210 - quote.open_price,
                               high_price=210 - quote.low_price, low_price=210 - quote.high_price,
                               close_price=210 - quote.close_price)
            return mirror(ask), mirror(bid)

        for index, bar in enumerate((self.breakout(), self.retest(), self.reclaim())):
            bid, ask = mirrored_quotes(bar)
            signal, _ = self.evaluate(bid, ask, frozen=channel() if index == 0 else None)
        self.assertEqual(signal.side, SignalSide.SELL)
        self.assertAlmostEqual(signal.stop_loss_price, 100.6)
        self.assertAlmostEqual(signal.maximum_entry_price, 99.3)
        self.assertAlmostEqual(signal.minimum_entry_price, 98.5)

    def test_breakout_must_have_directional_body_and_favorable_close(self):
        for bar, reason in (
            (candle(open=110.9, high=112, low=108, close=111), "weak_breakout_body"),
            (candle(open=111.2, high=111.3, low=110, close=111), "weak_breakout_body"),
            (candle(open=108, high=112, low=108, close=110.5), "weak_breakout_close"),
        ):
            with self.subTest(reason=reason):
                signal, analysis = self.evaluate(bar, frozen=channel())
                self.assertIsNone(signal)
                self.assertEqual(analysis["reasons"], ["breakout_" + reason])
                self.assertFalse(self.tracker.events)

    def test_a_moving_line_without_price_progress_does_not_arm(self):
        frozen = replace(channel(-.2), previous_bid_close=107.1, previous_ask_close=107.2)
        self.rules["regression_breakout_min_close_atr"] = .04
        bar = candle(open=107.05, high=107.12, low=107, close=107.1)
        _, analysis = self.evaluate(bar, frozen=frozen)
        self.assertEqual(analysis["reasons"], ["breakout_no_price_impulse"])
        self.assertFalse(self.tracker.events)

    def test_deep_wick_invalidates_even_if_the_close_recovers(self):
        self.evaluate(self.breakout(), frozen=channel())
        bar = candle(1, open=110.2, high=110.8, low=108.8, close=110.5)
        signal, analysis = self.evaluate(bar)
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_retest_too_deep"])
        self.assertFalse(self.tracker.events)

    def test_prior_extreme_needs_a_close_not_just_a_wick(self):
        self.arm_retest()
        signal, analysis = self.evaluate(candle(2, open=110.2, high=111.3, low=110.2, close=110.65))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_reclaim"])
        self.assertAlmostEqual(analysis["confirmation_level"], 110.7)

    def test_weak_reclaim_body_is_rejected(self):
        self.arm_retest()
        signal, analysis = self.evaluate(candle(2, open=111, high=111.15, low=110.5, close=111.05))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_weak_confirmation_body"])

    def test_reclaim_must_close_near_the_favorable_end_of_its_candle(self):
        self.arm_retest()
        signal, analysis = self.evaluate(candle(2, open=110.2, high=111.6, low=110.2, close=111))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_weak_confirmation_close"])

    def test_new_retest_refreshes_trigger_only_for_the_following_candle(self):
        self.arm_retest()
        signal, _ = self.evaluate(candle(2, open=110.4, high=110.5, low=109.9, close=110.2))
        self.assertIsNone(signal)
        signal, _ = self.evaluate(candle(3, open=110.2, high=110.9, low=110.2, close=110.8))
        self.assertIsNotNone(signal)
        self.assertAlmostEqual(signal.entry_context["retest_trigger"], 110.5)
        self.assertAlmostEqual(signal.minimum_entry_price, 110.6)
        self.assertAlmostEqual(signal.stop_loss_price, 109.4)

    def test_final_allowed_bar_can_reclaim_but_an_expired_bar_cannot(self):
        self.evaluate(self.breakout(), frozen=channel())
        self.evaluate(candle(1, open=111, high=111.3, low=110.8, close=111.1))
        self.evaluate(self.retest(2))
        self.assertIsNotNone(self.evaluate(self.reclaim(3))[0])
        self.arm_retest()
        for index in (2, 3):
            self.evaluate(candle(index, open=110.5, high=110.6, low=110.45, close=110.5))
        signal, analysis = self.evaluate(self.reclaim(4))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_expired"])

    def test_large_reclaim_extension_is_consumed_without_entry(self):
        self.arm_retest()
        signal, analysis = self.evaluate(candle(2, open=110.3, high=111.7, low=110.25, close=111.6))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_entry_too_far"])
        self.assertFalse(self.tracker.events)

    def test_invalid_ohlc_cannot_arm_or_keep_an_event(self):
        self.evaluate(self.breakout(), frozen=channel())
        signal, analysis = self.evaluate(replace(self.retest(), high_price=float("nan")))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_invalid_candle"])
        self.assertFalse(self.tracker.events)
        _, analysis = self.evaluate(candle(open=115, high=112, close=111), frozen=channel())
        self.assertEqual(analysis["reasons"], ["breakout_invalid_candle"])

    def test_confirmation_zone_rejects_a_later_price_back_below_the_reclaim(self):
        self.arm_retest()
        signal, _ = self.evaluate(self.reclaim())
        risk_tests = BreakoutIntegrationTests()
        self.assertIsNotNone(risk_tests.risk_evaluation(signal, bid=111.05, ask=111.15))
        self.assertIsNone(risk_tests.risk_evaluation(signal, bid=110.5, ask=110.6))

    def test_cli_and_config_accept_legacy_and_validate_quality_ranges(self):
        with patch.object(sys, "argv", ["run", "--regression-breakout-confirmation-mode", "legacy",
                                      "--regression-breakout-max-bars", "1"]):
            args = parse_args(breakout_config())
        self.assertEqual(build_engine([], args).strategy.config.breakout.confirmation_mode, "legacy")
        for options in ({"confirmation_mode": "unknown"}, {"min_close_location": .4},
                        {"min_breakout_body_ratio": 1.1}, {"min_confirmation_body_ratio": -.1},
                        {"confirmation_buffer_atr": float("nan")}, {"max_retest_depth_atr": -1},
                        {"confirmation_mode": "confirmed", "max_bars": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                RegressionBreakoutParameters(**options)

    def test_strategy_keeps_one_frozen_fit_until_the_later_confirmation(self):
        config = breakout_config()
        config = replace(config, breakout=replace(config.breakout, confirmation_mode="confirmed"))
        strategy = SwingRegressionStrategy(config=config, analysis_interval=INTERVAL)
        history = [candle(i, open=105, high=108, low=102, close=105) for i in range(15)]
        quotes = {}
        market = SimpleNamespace(get_candles=lambda *args, **kwargs: quotes["bid"], get_point_size=lambda _: .01)
        context = SimpleNamespace(market_state=market)
        with patch.object(strategy, "_get_structure_candles", side_effect=lambda *args: quotes), \
                patch.object(strategy, "_evaluate_regression_pipeline", return_value=(None, deepcopy(channel().analysis))) as fit:
            for index, bar in enumerate((self.breakout(), self.retest(), self.reclaim()), start=15):
                history.append(replace(bar, start_time=START + index * INTERVAL))
                quotes.update(bid=history[-16:], ask=[ask_for(c) for c in history[-16:]])
                batch = TickBatch("EURUSD", [Tick(START + (index + 1) * INTERVAL, "EURUSD", bar.close_price,
                                                 bar.close_price + .1, 1, 1)])
                signals = strategy.on_market_data(batch, context)
                self.assertEqual(len(signals), int(index == 17))
                self.assertEqual(strategy.on_market_data(batch, context), [])
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.args[1][-1].start_time, START + 14 * INTERVAL)
            self.assertAlmostEqual(signals[0].minimum_entry_price, 110.7)

    def test_separate_confirmation_window_allows_a_late_retest_to_reclaim(self):
        self.rules["regression_breakout_confirmation_max_bars"] = 3
        self.evaluate(self.breakout(), frozen=channel())
        for index in (1, 2):
            self.evaluate(candle(index, open=111, high=111.3, low=110.8, close=111.1))
        self.evaluate(self.retest(3))
        signal, analysis = self.evaluate(self.reclaim(4))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.entry_context["first_retest_bar"], 3)
        self.assertEqual(signal.entry_context["confirmation_bars"], 1)
        self.assertEqual(analysis["confirmation_deadline_bar"], 6)

    def test_retouches_do_not_extend_the_first_contact_deadline(self):
        self.rules["regression_breakout_confirmation_max_bars"] = 3
        self.arm_retest()
        for index in (2, 3, 4):
            self.evaluate(candle(index, open=110.4, high=110.5, low=109.9, close=110.2))
        signal, analysis = self.evaluate(self.reclaim(5))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_confirmation_expired"])
        self.assertEqual(analysis["first_retest_bar"], 1)
        self.assertEqual(analysis["confirmation_deadline_bar"], 4)
        self.assertFalse(self.tracker.events)

    def test_separate_confirmation_window_does_not_allow_a_late_first_contact(self):
        self.rules["regression_breakout_confirmation_max_bars"] = 3
        self.evaluate(self.breakout(), frozen=channel())
        for index in (1, 2, 3):
            self.evaluate(candle(index, open=111, high=111.3, low=110.8, close=111.1))
        signal, analysis = self.evaluate(self.retest(4))
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["breakout_retest_expired"])
        self.assertFalse(self.tracker.events)

    def test_separate_confirmation_deadline_is_inclusive(self):
        self.rules["regression_breakout_confirmation_max_bars"] = 3
        self.evaluate(self.breakout(), frozen=channel())
        for index in (1, 2):
            self.evaluate(candle(index, open=111, high=111.3, low=110.8, close=111.1))
        for index in (3, 4, 5):
            self.evaluate(self.retest(index))
        signal, _ = self.evaluate(self.reclaim(6))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.entry_context["first_retest_bar"], 3)
        self.assertEqual(signal.entry_context["confirmation_bars"], 3)

    def test_channel_must_remain_valid_through_the_extended_horizon(self):
        frozen = replace(channel(), previous_bid_close=102.8, previous_ask_close=102.9)
        frozen.analysis["floating_line"]["slope"] = -.5
        bar = candle(open=102.7, high=103.5, low=102.4, close=103.4)
        _, analysis = self.evaluate(bar, frozen=frozen)
        self.assertEqual(analysis["reasons"], ["breakout_awaiting_retest"])
        self.tracker.clear()
        self.rules["regression_breakout_confirmation_max_bars"] = 3
        _, analysis = self.evaluate(bar, frozen=frozen)
        self.assertEqual(analysis["reasons"], ["breakout_invalid_channel"])

    def test_separate_window_config_and_cli_are_validated_and_round_trip(self):
        with patch.object(sys, "argv", ["run", "--strategy", "swing-regression",
                                      "--regression-entry-mode", "breakout-retest",
                                      "--regression-breakout-confirmation-mode", "confirmed",
                                      "--regression-breakout-max-bars", "1",
                                      "--regression-breakout-confirmation-max-bars", "3"]):
            engine = build_engine([], parse_args())
        self.assertEqual(engine.strategy.config.breakout.confirmation_max_bars, 3)
        self.assertEqual(engine.strategy.config.breakout.max_bars, 1)
        self.assertEqual(engine.strategy.config.with_rules(engine.strategy.config.to_rules()), engine.strategy.config)
        for value in (-1, 1.5, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RegressionBreakoutParameters(confirmation_max_bars=value)

    def test_separate_window_is_ignored_for_legacy_entries(self):
        self.rules["regression_breakout_confirmation_mode"] = "legacy"
        self.rules["regression_breakout_confirmation_max_bars"] = 100
        self.evaluate(self.breakout(), frozen=channel())
        signal, _ = self.evaluate(candle(1, open=110, high=110.6, low=109.8, close=110.5))
        self.assertIsNotNone(signal)
        self.assertEqual(signal.entry_context["confirmation_max_bars"], 0)


class BreakoutComparisonTests(unittest.TestCase):
    def test_report_uses_each_trades_fill_risk_and_counts_trailing_exits(self):
        def trade(identifier, pnl, reason):
            return SimpleNamespace(trade_id=identifier, symbol="EURUSD", side=SignalSide.BUY,
                                   opened_at=START, closed_at=START + identifier * INTERVAL,
                                   pnl=pnl, max_favorable_pnl=max(0, pnl), max_adverse_pnl=min(0, pnl),
                                   close_reason=reason)
        engine = SimpleNamespace(
            reports=SimpleNamespace(trades=[trade(1, 200, CloseReason.TRAILING_STOP),
                                           trade(2, -50, CloseReason.STOP_LOSS)]),
            portfolio=SimpleNamespace(positions={}, balance=1150, initial_balance=1000),
            pending_orders_processor=SimpleNamespace(queue=SimpleNamespace(symbols=lambda: [])),
            strategy=SimpleNamespace(diagnostics_summary=lambda: {}),
        )
        report = summarize(engine, {1: 100, 2: 50})
        self.assertEqual(report["total_r"], 1)
        self.assertEqual(report["expectancy_r"], .5)
        self.assertEqual(report["profit_factor_r"], 2)
        self.assertEqual(report["realized_drawdown_r"], 1)
        self.assertEqual(report["trailing_exits"], 1)
        self.assertAlmostEqual(report["return_percent"], 15)


if __name__ == "__main__":
    unittest.main()
