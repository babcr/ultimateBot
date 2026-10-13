import io
import sys
import unittest
from datetime import timedelta
from random import Random
from unittest.mock import patch

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import _rolling_atr, merge_structure_reversal_swings
from ultimateBot.backtester.strategy.swing_regression_decision_engine import (
    decide_swing_regression, resolve_swing_regression_rules, select_swing_window,
)
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, parse_args
from ultimateBot.tests.test_swing_level_reaction import START, bounce_swings, mirror, reaction_ticks, rules


class StrategyLookbackTests(unittest.TestCase):
    def engine(self, regression=15, levels=30, *options):
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression",
                                        "--regression-lookback-bars", str(regression),
                                        "--level-lookback-bars", str(levels), "--candle-interval", "1m",
                                        "--regression-global-timeframe-multiplier", "1",
                                        "--regression-min-market-volatility-ratio", "0", *options]):
            return build_engine([], parse_args())

    def test_cli_keeps_independent_windows_and_allocates_the_maximum(self):
        for regression, levels in ((15, 30), (50, 10)):
            with self.subTest(regression=regression, levels=levels):
                engine = self.engine(regression, levels)
                strategy = engine.strategy
                self.assertEqual(strategy.history_lookback_bars, regression)
                self.assertEqual(strategy.level_history_lookback_bars, levels)
                self.assertEqual(strategy.minimum_candles, max(regression, levels))
                self.assertEqual(engine.market_state.candle_lookback, max(regression, levels))
                self.assertEqual(set(engine.market_state._candle_builders_by_interval), {timedelta(minutes=1)})

    def test_legacy_configuration_uses_equal_windows_until_regression_is_explicit(self):
        resolved = resolve_swing_regression_rules({"level_lookback_bars": 17})
        self.assertEqual(resolved["regression_lookback_bars"], 17)
        self.assertEqual(resolved["level_lookback_bars"], 17)
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression", "--level-lookback-bars", "17"]):
            engine = build_engine([], parse_args())
        self.assertEqual(engine.strategy.history_lookback_bars, 17)

    def test_invalid_regression_windows_are_rejected(self):
        for value in (0, 1, -5):
            with self.subTest(value=value), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.engine(value)

    def test_levels_can_use_history_older_than_regression_without_changing_the_fit(self):
        for swings, price in ((bounce_swings(), 107), (mirror(bounce_swings()), 103)):
            with self.subTest(price=price):
                common = rules(regression_lookback_bars=15)
                long_decision, long_analysis = decide_swing_regression(
                    swings, price, 2, 30, common, [swings[-1]], point_size=0.001,
                )
                short_decision, short_analysis = decide_swing_regression(
                    swings, price, 2, 30, {**common, "level_lookback_bars": 15},
                    [swings[-1]], point_size=0.001,
                )
                self.assertIsNotNone(long_decision)
                self.assertIsNone(short_decision)
                self.assertEqual(short_analysis["reasons"], ["trigger_swing_without_level_reaction"])
                for field in ("anchored_line", "floating_line", "trend_score", "swing_density"):
                    self.assertEqual(long_analysis[field], short_analysis[field])
                self.assertEqual(long_analysis["regression_window_start"], 15)
                self.assertEqual(long_analysis["level_window_start"], 0)
                self.assertEqual(short_analysis["level_window_start"], 15)

    def test_pipeline_uses_one_raw_vector_and_preserves_shared_coordinates(self):
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=30)
        for tick in reaction_ticks():
            state.update(tick)
        state.finalize()
        strategy = SwingRegressionStrategy(rules(regression_lookback_bars=10), analysis_interval=timedelta(minutes=1))
        candles = {source: state.get_candles("TEST", price_source=source) for source in ("bid", "ask", "mid")}
        with patch.object(strategy, "_build_shared_swing_candidates", wraps=strategy._build_shared_swing_candidates) as collect:
            decision, analysis = strategy._evaluate_regression_pipeline(
                "TEST", candles["bid"], candles["ask"], candles["mid"], 0.001,
                timeframe_label="local", timeframe_multiplier=1,
                context=StrategyContext(None, state),
            )
        self.assertIsNotNone(decision)
        self.assertEqual(collect.call_count, 1)
        self.assertEqual({source: len(frame) for source, frame in collect.call_args.args[0].items()},
                         {"bid": 30, "ask": 30, "mid": 30})
        self.assertEqual(analysis["bar_count"], 30)
        self.assertEqual(analysis["regression_window_start"], 20)
        self.assertEqual(analysis["level_window_start"], 0)
        self.assertGreaterEqual(analysis["anchor"]["index"], 20)
        self.assertEqual(analysis["anchor"]["time"], candles["mid"][analysis["anchor"]["index"]].start_time)
        self.assertTrue(all(swing["index"] == 29 for swing in analysis["fresh_swings"]))
        self.assertEqual(analysis["entry_level_reaction"]["origin"]["index"], 28)
        self.assertLess(analysis["entry_level_reaction"]["level"]["latest_index"], 28)

    def test_index_projection_matches_detection_on_each_window_with_variable_spreads(self):
        rng = Random(41)
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=650)
        for index in range(650):
            opening = 100 + rng.uniform(-2, 2)
            close = opening if index % 7 == 0 else opening + rng.choice((-1, 1))
            for seconds, bid in zip((0, 10, 20, 50), (opening, opening + 3, opening - 3, close)):
                state.update(Tick(START + timedelta(minutes=index, seconds=seconds), "TEST", bid, bid + rng.uniform(0.01, 0.8), 1, 1))
        state.finalize()
        strategy = SwingRegressionStrategy(rules(), analysis_interval=timedelta(minutes=1))
        frames = {source: strategy._build_candle_frame(state.get_candles("TEST", price_source=source))
                  for source in ("bid", "ask")}
        raw = strategy._build_shared_swing_candidates(frames, symbol="TEST", market_state=state)
        original = [dict(item) for item in raw]
        for window in (2, 5, 15, 65, 650):
            start = len(frames["bid"]) - window
            source_atrs = {source: _rolling_atr(frame.iloc[start:], 0.001, window) for source, frame in frames.items()}
            projected = select_swing_window(raw, start, 650, 1, source_atrs)
            expected = merge_structure_reversal_swings(
                frames["bid"].iloc[start:], frames["ask"].iloc[start:], 0.001,
                {"level_lookback_bars": window},
            )
            self.assertEqual(projected, [{**item, "index": item["index"] + start} for item in expected])
        self.assertEqual(raw, original)

    def test_adaptive_regression_changes_keep_level_window_and_buffer_identity(self):
        engine = self.engine(15, 40, "--adaptive-strategy", "--adaptive-lookback-min", "5",
                             "--adaptive-lookback-max", "61", "--adaptive-lookback-step", "5")
        for tick in reaction_ticks(70):
            engine.market_state.update(tick)
        buffers = {source: builder._closed["TEST"] for source, builder in engine.market_state._candle_builders.items()}
        self.assertEqual(engine.market_state.candle_lookback, 60)
        for window in (5, 50, 10):
            engine.strategy.set_history_lookback_bars(window, symbol="TEST")
            self.assertEqual(engine.strategy.get_history_lookback_bars("TEST"), window)
            self.assertEqual(engine.strategy.get_history_lookback_bars("OTHER"), 15)
            self.assertEqual(engine.strategy.rules_for_symbol("TEST")["level_lookback_bars"], 40)
            self.assertEqual(engine.strategy.history_capacity_for_lookback(window), max(window, 40))
            for source, builder in engine.market_state._candle_builders.items():
                self.assertIs(builder._closed["TEST"], buffers[source])
                self.assertEqual(builder._closed["TEST"].maxlen, 60)

    def test_global_confirmation_capacity_uses_regression_window(self):
        strategy = SwingRegressionStrategy(rules(regression_lookback_bars=10, level_lookback_bars=80,
                                                regression_global_timeframe_multiplier=3,
                                                regression_min_market_volatility_ratio=0))
        self.assertEqual(strategy.global_history_lookback_bars, 10)
        self.assertEqual(strategy.global_history_capacity, 30)
        self.assertEqual(strategy.minimum_candles, 80)
        strategy.set_history_lookback_bars(20)
        self.assertEqual(strategy.history_lookback_bars, 20)
        self.assertEqual(strategy.level_history_lookback_bars, 80)
        self.assertEqual(strategy.global_history_lookback_bars, 10)


if __name__ == "__main__":
    unittest.main()
