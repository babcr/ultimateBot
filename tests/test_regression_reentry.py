import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel, RegressionReentryTracker
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.swing_regression_decision_engine import decide_swing_regression, resolve_swing_regression_rules
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.backtester.embeddings.trade_dataset import TradeDataset, ensure_capture_schema
from ultimateBot.run_backtest import parse_args, build_engine
from ultimateBot.run_regression_reentry import reentry_config


START = datetime(2024, 1, 2, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=1)


def candle(index=0, high=112, low=104, close=109):
    return Candle(START + index * INTERVAL, 105, high, low, close)


def channel(slope=0, low=100, high=110):
    return FrozenChannel({
        "bar_count": 15, "atr": 2.0, "reasons": [],
        "anchored_line": {"kind": "low", "intercept": low, "slope": slope},
        "floating_line": {"kind": "high", "intercept": high, "slope": slope},
    }, START - INTERVAL, 105, 105.1)


class ReentryEventTests(unittest.TestCase):
    def setUp(self):
        self.tracker = RegressionReentryTracker()
        self.rules = resolve_swing_regression_rules({"regression_entry_mode": "failed-breakout",
                                                    "regression_reentry_max_bars": 2})

    def evaluate(self, bid=None, ask=None, frozen=None, symbol="EURUSD"):
        bid = bid or candle()
        ask = ask or replace(bid, high_price=bid.high_price + .1, low_price=bid.low_price + .1,
                             close_price=bid.close_price + .1)
        return self.tracker.evaluate(symbol, bid, ask, bid.start_time + INTERVAL,
                                     INTERVAL, self.rules, frozen)

    def test_sell_after_upper_excursion_targets_center_with_ask_stop(self):
        signal, analysis = self.evaluate(frozen=channel())
        self.assertEqual(signal.side, SignalSide.SELL)
        self.assertAlmostEqual(signal.stop_loss_price, 112.5)
        self.assertEqual(signal.take_profit_price, 105)
        self.assertEqual(analysis["reasons"], [])
        self.assertNotIn("EURUSD", self.tracker.events)

    def test_buy_after_lower_excursion_uses_bid_stop(self):
        signal, _ = self.evaluate(candle(high=106, low=98, close=101), frozen=channel())
        self.assertEqual(signal.side, SignalSide.BUY)
        self.assertAlmostEqual(signal.stop_loss_price, 97.6)
        self.assertEqual(signal.take_profit_price, 105)

    def test_pending_event_keeps_original_channel_and_extreme(self):
        signal, analysis = self.evaluate(candle(close=111), frozen=channel())
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["reentry_awaiting_close_inside"])
        signal, _ = self.evaluate(candle(1, high=113, close=109), frozen=channel(low=200, high=210))
        self.assertEqual(signal.take_profit_price, 105)
        self.assertAlmostEqual(signal.stop_loss_price, 113.5)
        self.assertEqual(signal.entry_context["reentry_bars"], 1)

    def test_expiry_allows_two_extra_bars_then_discards(self):
        for index in range(3):
            signal, analysis = self.evaluate(candle(index, close=111), frozen=channel())
            self.assertIsNone(signal)
            self.assertEqual(analysis["reasons"], ["reentry_awaiting_close_inside"])
        signal, analysis = self.evaluate(candle(3), frozen=channel())
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["reentry_expired"])
        self.assertFalse(self.tracker.events)

    def test_gap_invalidates_pending_event(self):
        self.evaluate(candle(close=111), frozen=channel())
        signal, analysis = self.evaluate(candle(2), frozen=channel())
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["reentry_history_gap"])

    def test_rejects_ambiguous_candle_and_opposite_boundary(self):
        _, analysis = self.evaluate(candle(high=112, low=98), frozen=channel())
        self.assertEqual(analysis["reasons"], ["reentry_ambiguous_excursion"])
        self.evaluate(candle(close=111), frozen=channel())
        _, analysis = self.evaluate(candle(1, high=109, low=98, close=105), frozen=channel())
        self.assertEqual(analysis["reasons"], ["reentry_opposite_boundary_touched"])
        self.assertFalse(self.tracker.events)

    def test_proximity_without_excursion_does_not_trigger(self):
        signal, analysis = self.evaluate(candle(high=109.9), frozen=channel())
        self.assertIsNone(signal)
        self.assertEqual(analysis["reasons"], ["reentry_no_excursion"])

    def test_requires_preceding_close_inside_and_two_lines(self):
        _, analysis = self.evaluate(frozen=replace(channel(), previous_bid_close=111))
        self.assertEqual(analysis["reasons"], ["reentry_previous_close_outside"])
        frozen = channel()
        frozen.analysis["floating_line"] = None
        _, analysis = self.evaluate(frozen=frozen)
        self.assertEqual(analysis["reasons"], ["reentry_requires_two_lines"])

    def test_rejects_strong_drift_and_crossing_lines(self):
        _, analysis = self.evaluate(frozen=channel(slope=1))
        self.assertEqual(analysis["reasons"], ["reentry_channel_too_directional"])
        _, analysis = self.evaluate(frozen=channel(low=111, high=110))
        self.assertEqual(analysis["reasons"], ["reentry_invalid_channel"])

    def test_symbols_have_separate_pending_events(self):
        self.evaluate(candle(close=111), frozen=channel(), symbol="EURUSD")
        self.evaluate(candle(close=111), frozen=channel(), symbol="GBPUSD")
        self.evaluate(candle(1), symbol="EURUSD")
        self.assertEqual(set(self.tracker.events), {"GBPUSD"})

    def test_channel_projects_frozen_slopes_to_confirmation_bar(self):
        frozen = channel(slope=.1)
        signal, _ = self.evaluate(candle(high=114, close=113), frozen=frozen)
        self.assertIsNone(signal)
        signal, _ = self.evaluate(candle(1, high=114, close=110))
        self.assertAlmostEqual(signal.take_profit_price, 106.6)
        self.assertEqual(signal.entry_context["channel_fitted_through"], frozen.fitted_through.isoformat())

    def test_larger_volatility_window_keeps_regression_coordinates_and_lookback(self):
        frozen = channel(slope=.1)
        frozen.analysis.update(bar_count=40, regression_window_start=25, regression_lookback_bars=15)
        signal, analysis = self.evaluate(candle(high=116, close=113), frozen=frozen)
        self.assertEqual(signal.entry_context["regression_lookback"], 15)
        self.assertAlmostEqual(signal.take_profit_price, 109)
        self.assertAlmostEqual(analysis["channel_drift_ratio"], .15)


class ReentryExecutionTests(unittest.TestCase):
    def evaluate(self, signal, bid=109, ask=109.1):
        manager = RiskManager(minimum_candles=1, risk_per_trade=1, max_spread_ratio=100,
                              limit_spread_ratio=100, reward_risk_ratio=3)
        tick = Tick(START, "EURUSD", bid, ask, 1, 1)
        for price in (108, 110, 109):
            manager.on_tick(Tick(START, "EURUSD", price, price + .1, 1, 1))
        return manager.evaluate(signal, Portfolio(initial_balance=1000), tick)

    def test_fixed_exits_size_risk_and_survive_trailing_default(self):
        signal = Signal("EURUSD", START, SignalSide.SELL, stop_loss_price=112.5,
                        take_profit_price=105, minimum_reward_risk_ratio=1, entry_context={"entry_mode": "failed-breakout"})
        evaluation = self.evaluate(signal)
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation.stop_loss, 112.5)
        self.assertEqual(evaluation.take_profit, 105)
        self.assertAlmostEqual(evaluation.quantity * 3.5, 10)
        order = OrderBuilder().build(signal, evaluation)
        TradeCloser(trailing_stop_enabled=True).prepare_order(order)
        self.assertEqual(order.take_profit, 105)
        self.assertFalse(order.trailing_stop_enabled)
        self.assertEqual(order.strategy_entry_context, signal.entry_context)

    def test_rechecks_reward_risk_at_current_executable_price(self):
        signal = Signal("EURUSD", START, SignalSide.SELL, stop_loss_price=112.5,
                        take_profit_price=105, minimum_reward_risk_ratio=1)
        self.assertIsNone(self.evaluate(signal, bid=108, ask=108.1))
        self.assertIsNone(self.evaluate(signal, bid=104, ask=104.1))
        self.assertIsNone(self.evaluate(signal, bid=113, ask=113.1))

    def test_buy_exits_and_invalid_values(self):
        signal = Signal("EURUSD", START, SignalSide.BUY, stop_loss_price=105,
                        take_profit_price=115, minimum_reward_risk_ratio=1)
        self.assertIsNotNone(self.evaluate(signal))
        for stop, target in ((float("nan"), 115), (105, float("inf")), (None, 115), (110, 115)):
            self.assertIsNone(self.evaluate(replace(signal, stop_loss_price=stop, take_profit_price=target)))

    def test_config_cli_and_incompatible_exit_policies(self):
        with patch.object(sys, "argv", ["run", "--symbols", "EURUSD"]):
            args = parse_args(reentry_config())
        engine = build_engine([], args)
        self.assertEqual(engine.strategy.history_capacity_for_lookback(15), 16)
        self.assertFalse(engine.strategy.regression_rules["regression_require_level_reaction"])
        self.assertFalse(engine.strategy.get_global_timeframe_validation_enabled())
        for option in ("--trailing-stop", "--adaptive-strategy", "--contextual-bandit"):
            with self.subTest(option=option), patch.object(sys, "argv", ["run", option]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(reentry_config())

    def test_schema_accepts_event_context_joined_to_order(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("PRAGMA foreign_keys=ON")
            ensure_capture_schema(db)
            db.execute("INSERT INTO runs(run_id,started_at,status) VALUES('r','now','running')")
            db.execute("INSERT INTO order_feature_snapshots VALUES('r',1,'{}',NULL)")
            db.execute("INSERT INTO order_signal_contexts VALUES('r',1,?)", (json.dumps({"entry_mode": "failed-breakout"}),))
            ensure_capture_schema(db)
            self.assertEqual(db.execute("SELECT count(*) FROM order_signal_contexts").fetchone()[0], 1)

    def test_event_context_is_captured_at_creation_and_joined_to_close_label(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = SimpleNamespace(model_id="test-encoder", checkpoint="test.pt", metadata={})
            dataset = TradeDataset(Path(directory)/"test.sqlite3", encoder, parameters={})
            try:
                order = OrderRequest(1, "EURUSD", START, START, START, OrderSide.SELL, 1,
                                     strategy_lookback_bars=15, strategy_timeframe_minutes=1,
                                     strategy_entry_context={"entry_mode": "failed-breakout", "excursion_atr": 1})
                snapshot = SimpleNamespace(model_id="test-encoder", values=(.1, .2), context_bars=15,
                                           context_start=(START-15*INTERVAL).isoformat(), context_end=START.isoformat())
                dataset.on_order_created(order, START, snapshot=snapshot)
                self.assertEqual(dataset.connection.execute("SELECT count(*) FROM closed_trades").fetchone()[0], 0)
                # Later live state cannot rewrite the already persisted event.
                order.strategy_entry_context["excursion_atr"] = 99
                opened = START + timedelta(seconds=1)
                position = SimpleNamespace(trade_id=1, opened_at=opened, entry_price=109, stop_loss=112.5,
                                           take_profit=105, trailing_stop_activation_price=None, quantity=1,
                                           profit_multiplier=1, account_currency="engine_price_quantity")
                dataset.on_position_open(position, None, order)
                report = SimpleNamespace(is_complete=True, trade_id=1, symbol="EURUSD", opened_at=opened,
                                         closed_at=opened+INTERVAL, pnl=4, exit_price=105, close_reason=CloseReason.TAKE_PROFIT)
                dataset.on_trade_closed(report)
                row = dataset.connection.execute("SELECT c.context_json,t.pnl,t.pnl_r FROM order_signal_contexts c "
                                                 "JOIN closed_trades t USING(run_id,order_id)").fetchone()
                self.assertEqual(json.loads(row["context_json"])["excursion_atr"], 1)
                self.assertEqual(row["pnl"], 4)
                self.assertAlmostEqual(row["pnl_r"], 4/3.5)
            finally:
                with redirect_stdout(io.StringIO()):
                    dataset.finish()

    def test_channel_only_does_not_require_nonzero_trend(self):
        swings = [{"kind": kind, "index": index, "price": price, "strength": 3}
                  for kind, price, indices in (("low", 100, (1, 5, 9)), ("high", 110, (3, 7, 11)))
                  for index in indices]
        _, analysis = decide_swing_regression(swings, 105, 2, 15,
                                              {"regression_min_consecutive_swing_distance_atr": 0}, channel_only=True)
        self.assertEqual(analysis["reasons"], [])
        self.assertEqual(analysis["shared_slope"], 0)

    def test_strategy_excludes_event_from_fit_and_evaluates_closed_candle_once(self):
        strategy = SwingRegressionStrategy({"regression_entry_mode": "failed-breakout"}, analysis_interval=INTERVAL)
        bid = [candle(i, high=108, low=102, close=105) for i in range(16)]
        bid[-1] = candle(15)
        ask = [replace(c, high_price=c.high_price+.1, low_price=c.low_price+.1,
                       close_price=c.close_price+.1) for c in bid]
        market = SimpleNamespace(get_candles=lambda *args, **kwargs: bid, get_point_size=lambda _: .01)
        context = SimpleNamespace(market_state=market)
        tick = Tick(START+16*INTERVAL, "EURUSD", 109, 109.1, 1, 1)
        batch = TickBatch("EURUSD", [tick])
        analysis = deepcopy(channel().analysis)
        with patch.object(strategy, "_get_structure_candles", return_value={"bid": bid, "ask": ask}), \
                patch.object(strategy, "_evaluate_regression_pipeline", return_value=(None, analysis)) as fit:
            signals = strategy.on_market_data(batch, context)
            self.assertEqual(len(signals), 1)
            self.assertEqual(fit.call_args.args[1][-1].start_time, START + 14*INTERVAL)
            self.assertEqual(len(fit.call_args.args[1]), 15)
            self.assertNotIn("context", fit.call_args.kwargs)
            self.assertTrue(fit.call_args.kwargs["channel_only"])
            self.assertEqual(strategy.on_market_data(batch, context), [])
            self.assertEqual(fit.call_count, 1)

    def test_changing_lookback_invalidates_frozen_event(self):
        strategy = SwingRegressionStrategy({"regression_entry_mode": "failed-breakout"})
        strategy._reentry.events["EURUSD"] = object()
        strategy.set_history_lookback_bars(20, "EURUSD")
        self.assertFalse(strategy._reentry.events)

    def test_history_capacity_respects_minimum_and_pre_event_volatility(self):
        strategy = SwingRegressionStrategy({"regression_entry_mode": "failed-breakout",
                                            "regression_lookback_bars": 15}, minimum_candles=40)
        self.assertEqual(strategy.history_capacity_for_lookback(15), 40)
        self.assertEqual(strategy.minimum_candles, 40)
        strategy = SwingRegressionStrategy({"regression_entry_mode": "failed-breakout",
                                            "regression_lookback_bars": 15,
                                            "regression_market_volatility_window_bars": 40})
        self.assertEqual(strategy.history_capacity_for_lookback(15), 41)
        self.assertEqual(strategy.minimum_candles, 41)

    def test_invalid_event_parameters_are_rejected(self):
        for name, value in (("reentry_max_bars", -1), ("reentry_max_bars", 1.5),
                            ("reentry_min_rr", float("nan")), ("reentry_stop_buffer_atr", -.1)):
            with self.assertRaises(ValueError):
                resolve_swing_regression_rules({"regression_"+name: value})


if __name__ == "__main__":
    unittest.main()
