import io
import sys
import unittest
from contextlib import redirect_stderr
from datetime import timedelta
from unittest.mock import patch

import pandas as pd

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.strategy.signal import SignalSide
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import _candle_context
from ultimateBot.backtester.strategy.swing_regression_config import DEFAULT_SWING_REGRESSION_CONFIG
from ultimateBot.backtester.strategy.swing_regression_decision_engine import (
    decide_swing_regression, resolve_swing_regression_rules,
)
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.tests.test_swing_regression import START, candle_ticks, reflected, rising_channel, swing
from ultimateBot.run_backtest import build_engine, parse_args, print_swing_regression_diagnostics


class SwingCandleConfirmationCliTests(unittest.TestCase):
    def test_confirmation_is_optional_and_uses_existing_default_ratio(self):
        rules = resolve_swing_regression_rules()
        self.assertFalse(rules["regression_require_candle_confirmation"])
        self.assertEqual(rules["confirmation_body_ratio"],
                         DEFAULT_SWING_REGRESSION_CONFIG.candles.confirmation_body_ratio)

    def test_cli_propagates_confirmation_and_ratio_to_adaptive_strategy(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                argv = ["run_backtest", "--strategy", "swing-regression", "--adaptive-strategy",
                        "--regression-candle-confirmation" if enabled else "--no-regression-candle-confirmation",
                        "--confirmation-body-ratio", "0.75"]
                with patch.object(sys, "argv", argv):
                    engine = build_engine([], parse_args())
                engine.strategy.set_history_lookback_bars(15, symbol="EURUSD")
                rules = engine.strategy.rules_for_symbol("EURUSD")
                self.assertEqual(rules["regression_require_candle_confirmation"], enabled)
                self.assertEqual(rules["confirmation_body_ratio"], 0.75)

    def test_ratio_option_also_configures_structure_strategy(self):
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "structure",
                                        "--confirmation-body-ratio", "0.75"]):
            engine = build_engine([], parse_args())
        self.assertEqual(engine.strategy.structure_rules["confirmation_body_ratio"], 0.75)

    def test_cli_rejects_invalid_ratio(self):
        for value in ("0", "0.14", "1.01", "nan", "inf"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with patch.object(sys, "argv", ["run_backtest", "--confirmation-body-ratio", value]):
                    with self.assertRaises(SystemExit) as raised:
                        parse_args()
                self.assertEqual(raised.exception.code, 2)


class SwingCandleConfirmationDecisionTests(unittest.TestCase):
    def decide(self, side="buy", candle=None, enabled=True, ratio=0.68, fresh=True, contexts=None):
        points = rising_channel() + [{**swing("low", 29, 102.9), "source": "ask"}]
        if side == "sell":
            points = [{**item, "source": "ask" if item["kind"] == "low" else "bid"}
                      for item in reflected(points)]
        rules = resolve_swing_regression_rules({
            "level_lookback_bars": 30, "regression_min_swings_per_side": 3,
            "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.1,
            "regression_min_parallelism": 0, "regression_min_consecutive_swing_distance_atr": 0,
            "regression_require_level_reaction": False,
            "regression_require_candle_confirmation": enabled, "confirmation_body_ratio": ratio,
        })
        if candle is not None:
            source = "ask" if side == "buy" else "bid"
            contexts = {source: _candle_context(pd.DataFrame([candle]), 2.0, rules)}
        return decide_swing_regression(points, 104.0 if side == "buy" else 106.0, 2.0, 30,
                                       rules, [points[-1]] if fresh else None, candle_contexts=contexts)

    def test_strong_reversal_candle_confirms_buy_and_sell(self):
        for side, candle in (
            ("buy", {"open": 102, "close": 104, "high": 104.5, "low": 102}),
            ("sell", {"open": 108, "close": 106, "high": 108, "low": 105.5}),
        ):
            with self.subTest(side=side):
                decision, analysis = self.decide(side, candle)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.side, side)
                self.assertTrue(analysis["candle_confirmation_passed"])
                self.assertAlmostEqual(analysis["confirmation_candle"]["body_ratio"], 0.8)
                self.assertEqual(analysis["candle_confirmation_price_source"], "ask" if side == "buy" else "bid")

    def test_weak_body_wrong_direction_doji_and_zero_range_reject(self):
        cases = (
            {"open": 103, "close": 104, "high": 104.5, "low": 102},
            {"open": 106, "close": 104, "high": 106, "low": 103.5},
            {"open": 104, "close": 104, "high": 105, "low": 103},
            {"open": 104, "close": 104, "high": 104, "low": 104},
        )
        for side in ("buy", "sell"):
            for base_candle in cases:
                candle = base_candle if side == "buy" else {
                    "open": 210 - base_candle["open"], "close": 210 - base_candle["close"],
                    "high": 210 - base_candle["low"], "low": 210 - base_candle["high"],
                }
                with self.subTest(side=side, candle=candle):
                    decision, analysis = self.decide(side, candle)
                    self.assertIsNone(decision)
                    self.assertEqual(analysis["reasons"], ["trigger_candle_not_confirmed"])

    def test_body_threshold_changes_acceptance(self):
        candle = {"open": 103, "close": 104, "high": 104.5, "low": 102}
        decision, _ = self.decide(candle=candle, ratio=0.4)
        self.assertIsNotNone(decision)  # Egalite au seuil acceptee.
        decision, _ = self.decide(candle=candle, ratio=0.41)
        self.assertIsNone(decision)

    def test_small_threshold_still_requires_directional_close_position(self):
        candle = {"open": 103, "close": 104, "high": 107, "low": 102}
        decision, analysis = self.decide(candle=candle, ratio=0.15)
        self.assertIsNone(decision)  # Cloture a 40 % du range, trop basse pour confirmer un achat.
        self.assertEqual(analysis["reasons"], ["trigger_candle_not_confirmed"])

    def test_enabled_filter_refuses_missing_candle(self):
        decision, analysis = self.decide()
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["missing_confirmation_candle"])

    def test_disabled_filter_preserves_entry_without_candle(self):
        decision, analysis = self.decide(enabled=False)
        self.assertIsNotNone(decision)
        self.assertFalse(analysis["candle_confirmation_applied"])

    def test_global_trend_analysis_does_not_require_reversal_candle(self):
        decision, analysis = self.decide(fresh=False)
        self.assertIsNotNone(decision)
        self.assertFalse(analysis["candle_confirmation_applied"])

    def test_spread_sources_use_the_actual_trigger_candle(self):
        rules = resolve_swing_regression_rules()
        strong_bullish = _candle_context(pd.DataFrame([
            {"open": 102, "close": 104, "high": 104.5, "low": 102}]), 2.0, rules)
        doji = _candle_context(pd.DataFrame([
            {"open": 104, "close": 104, "high": 105, "low": 103}]), 2.0, rules)
        decision, _ = self.decide(contexts={"ask": strong_bullish, "bid": doji})
        self.assertIsNotNone(decision)
        decision, _ = self.decide(contexts={"ask": doji, "bid": strong_bullish})
        self.assertIsNone(decision)

    def test_high_cannot_trigger_buy_even_when_mislabeled_as_ask(self):
        trigger = {**swing("high", 29, 104.9), "source": "ask"}
        decision, analysis = decide_swing_regression(
            rising_channel() + [trigger], 111, 2, 30,
            {"level_lookback_bars": 30, "regression_require_level_reaction": False,
             "regression_score_threshold": 0, "regression_min_parallelism": 0,
             "regression_min_consecutive_swing_distance_atr": 0}, [trigger],
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["expected_trigger_kind"], "low")
        self.assertEqual(analysis["reasons"], ["trigger_swings_filtered"])

    def test_low_cannot_trigger_sell_even_when_mislabeled_as_bid(self):
        trigger = {**swing("low", 29, 105.1), "source": "bid"}
        decision, analysis = decide_swing_regression(
            reflected(rising_channel()) + [trigger], 99, 2, 30,
            {"level_lookback_bars": 30, "regression_require_level_reaction": False,
             "regression_score_threshold": 0, "regression_min_parallelism": 0,
             "regression_min_consecutive_swing_distance_atr": 0}, [trigger],
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["expected_trigger_kind"], "high")
        self.assertEqual(analysis["reasons"], ["trigger_swings_filtered"])


class SwingCandleConfirmationStrategyTests(unittest.TestCase):
    def test_nonzero_spread_uses_ask_lows_bid_highs_and_matching_boundary_prices(self):
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=30)
        for index in range(30):
            for tick in candle_ticks(index):
                state.update(Tick(tick.timestamp, tick.symbol, tick.bid, tick.bid + 0.7, 1, 1))
        state.finalize()
        strategy = SwingRegressionStrategy({
            "level_lookback_bars": 30, "regression_min_swings_per_side": 3,
            "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.1,
            "regression_min_parallelism": 0, "regression_min_consecutive_swing_distance_atr": 0,
            "regression_global_timeframe_multiplier": 1, "regression_require_level_reaction": False,
        }, analysis_interval=timedelta(minutes=1))
        candles = {source: state.get_candles("TEST", price_source=source) for source in ("bid", "ask", "mid")}
        with patch("ultimateBot.backtester.strategy.swing_regression_strategy.decide_swing_regression",
                   wraps=decide_swing_regression) as decider:
            decision, analysis = strategy._evaluate_regression_pipeline(
                "TEST", candles["bid"], candles["ask"], candles["mid"], 0.01,
                timeframe_label="local", timeframe_multiplier=1, context=StrategyContext(Portfolio(1000), state),
            )
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["expected_trigger_source"], "ask")
        self.assertEqual(analysis["fresh_swings"][0]["kind"], "low")
        self.assertAlmostEqual(analysis["fresh_swings"][0]["price"], 104.8)
        self.assertEqual(analysis["fresh_swings"][0]["source"], "ask")
        boundaries = {item["kind"]: item for item in decider.call_args.args[6]}
        self.assertAlmostEqual(boundaries["low"]["price"], 99.2)
        self.assertEqual(boundaries["low"]["price_source"], "ask")
        self.assertAlmostEqual(boundaries["high"]["price"], 101.5)
        self.assertEqual(boundaries["high"]["price_source"], "bid")
        swings = strategy._build_shared_structure_swings(
            strategy._build_candle_frame(candles["bid"]), strategy._build_candle_frame(candles["ask"]), 0.01,
        )
        self.assertEqual({item["source"] for item in swings if item["kind"] == "low"}, {"ask"})
        self.assertEqual({item["source"] for item in swings if item["kind"] == "high"}, {"bid"})
        self.assertAlmostEqual(swings[-2]["price"], 107.1)  # High bid, et non le high ask a 107.8.

    def test_pipeline_uses_closed_candle_threshold_and_records_rejections(self):
        for direction in (1, -1):
            for ratio, expected_count in ((0.68, 0), (0.3, 1)):
                with self.subTest(direction=direction, ratio=ratio):
                    state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=30)
                    ticks = [tick for index in range(30) for tick in candle_ticks(index)]
                    if direction == -1:
                        ticks = [Tick(t.timestamp, t.symbol, 210 - t.bid, 210 - t.ask, 1, 1) for t in ticks]
                    for tick in ticks:
                        state.update(tick)
                    # Une bougie active extreme ne doit pas confirmer le swing ferme.
                    state.update(Tick(START + timedelta(minutes=30), "TEST", 10000, 10000, 1, 1))
                    strategy = SwingRegressionStrategy({
                        "level_lookback_bars": 30, "regression_score_threshold": 0.15,
                        "regression_min_swings_per_side": 3, "regression_slope_scale_atr": 0.1,
                        "regression_global_timeframe_multiplier": 1, "regression_min_parallelism": 0,
                        "regression_min_consecutive_swing_distance_atr": 0,
                        "regression_require_level_reaction": False,
                        "regression_require_candle_confirmation": True, "confirmation_body_ratio": ratio,
                    }, analysis_interval=timedelta(minutes=1))
                    context = StrategyContext(Portfolio(1000), state)
                    signals = strategy.on_market_data(TickBatch("TEST", [ticks[-1]]), context)
                    self.assertEqual(len(signals), expected_count)
                    analysis = strategy.last_analysis["TEST"]
                    self.assertAlmostEqual(analysis["confirmation_candle"]["body_ratio"], 1 / 3)
                    self.assertEqual(strategy.on_market_data(TickBatch("TEST", [ticks[-1]]), context), [])
                    if expected_count:
                        self.assertEqual(signals[0].side, SignalSide.BUY if direction == 1 else SignalSide.SELL)
                    else:
                        self.assertEqual(strategy.diagnostics_summary()["reason_counts"]["trigger_candle_not_confirmed"], 1)
                        with patch("sys.stdout", new_callable=io.StringIO) as output:
                            print_swing_regression_diagnostics(type("Engine", (), {"strategy": strategy})())
                        self.assertIn("Candle confirmation rejections: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
