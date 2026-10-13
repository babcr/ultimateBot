import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.strategy.signal import SignalSide
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.swing_regression_decision_engine import (
    decide_swing_regression, evaluate_trigger_level_reactions, resolve_swing_regression_rules,
)
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, parse_args, print_swing_regression_diagnostics


START = datetime(2020, 1, 2, tzinfo=timezone.utc)


def point(kind, index, price):
    return {"kind": kind, "index": index, "price": price, "strength": 3.0,
            "source": "bid" if kind == "low" else "ask"}


def bounce_swings():
    swings = []
    for index in range(5):
        swings += [point("low", 1 + index * 4, 100.0),
                   point("high", 3 + index * 4, 102.0 + index)]
    return swings + [point("low", 21, 100.1), point("high", 29, 108.0)]


def mirror(swings):
    return [point("high" if item["kind"] == "low" else "low",
                  item["index"], 210.0 - item["price"]) for item in swings]


def rules(**overrides):
    return resolve_swing_regression_rules({
        "level_lookback_bars": 30, "regression_score_threshold": 0.0,
        "regression_slope_scale_atr": 0.1, "regression_min_parallelism": 0.0,
        "regression_min_consecutive_swing_distance_atr": 0.0,
        "regression_trigger_max_distance_atr": float("inf"),
        "regression_global_timeframe_multiplier": 1,
        "min_level_touches": 5,
        "level_merge_atr_ratio": 0.01,
        **overrides,
    })


def reaction_ticks(count=30, direction=1, drift=0.001):
    ticks = []
    for index in range(count):
        opening = 101.0 + 0.08 * index
        prices = (opening, 104.0 + 0.2 * index, 100.0 + drift * index,
                  opening + (0.4 if index % 2 == 0 else -0.4))
        for seconds, price in zip((0, 10, 20, 50), prices):
            bid, ask = (price, price + 0.01) if direction == 1 else (210.0 - price - 0.01, 210.0 - price)
            ticks.append(Tick(START + timedelta(minutes=index, seconds=seconds), "TEST", bid, ask, 1, 1))
    return ticks


class SwingLevelReactionTests(unittest.TestCase):
    def decide(self, swings=None, price=107.0, triggers=None, closes=None, **overrides):
        swings = bounce_swings() if swings is None else swings
        return decide_swing_regression(
            swings, price, 2.0, 30, rules(**overrides),
            [swings[-1]] if triggers is None else triggers,
            point_size=0.001, candle_closes=closes,
        )

    def test_filter_is_enabled_by_default(self):
        self.assertTrue(resolve_swing_regression_rules()["regression_require_level_reaction"])

    def test_support_bounce_and_resistance_rejection_validate_entries(self):
        for swings, price, side, style in (
            (bounce_swings(), 107.0, "buy", "support_bounce"),
            (mirror(bounce_swings()), 103.0, "sell", "resistance_rejection"),
        ):
            with self.subTest(side=side):
                decision, analysis = self.decide(swings, price)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.side, side)
                self.assertTrue(analysis["level_reaction_passed"])
                reaction = analysis["entry_level_reaction"]
                self.assertEqual(reaction["style"], style)
                self.assertEqual(reaction["level"]["touches"], 5)
                self.assertLess(reaction["level"]["latest_index"], reaction["origin"]["index"])

    def test_regression_entry_without_level_is_rejected_and_can_be_compared(self):
        swings = [point("low", i, 100 + 0.1 * i) for i in (1, 10, 20)]
        swings += [point("high", i, 102 + 0.1 * i) for i in (5, 15, 29)]
        legacy, _ = self.decide(swings, regression_require_level_reaction=False)
        decision, analysis = self.decide(swings)
        self.assertIsNotNone(legacy)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["trigger_swing_without_level_reaction"])
        self.assertFalse(analysis["level_reaction_passed"])

    def test_touching_origin_cannot_supply_the_fifth_touch_itself(self):
        swings = bounce_swings()[2:]
        decision, analysis = self.decide(swings)
        self.assertIsNone(decision)
        self.assertEqual(analysis["trigger_level_reactions"][0]["reason"], "no_preexisting_level")

    def test_trigger_cannot_create_its_own_level(self):
        # The only point near 108 is the new trigger, even with one touch allowed.
        swings = bounce_swings()
        swings[-2] = point("low", 21, 107.9)
        reactions = evaluate_trigger_level_reactions(
            swings, [swings[-1]], "buy", 109, 2, 0.001, rules(min_level_touches=1),
        )
        self.assertFalse(reactions[0]["passed"])
        self.assertTrue(all(level["latest_index"] < 21 for level in reactions[0]["levels"]))

    def test_origin_must_touch_the_level_even_if_trigger_is_near_another_level(self):
        swings = bounce_swings()
        swings[-2] = point("low", 21, 101.0)
        decision, analysis = self.decide(swings)
        self.assertIsNone(decision)
        self.assertEqual(analysis["trigger_level_reactions"][0]["reason"], "origin_too_far_from_level")

    def test_close_must_leave_the_zone_in_the_trade_direction(self):
        for swings, price, side in ((bounce_swings(), 100.4, "buy"),
                                    (mirror(bounce_swings()), 109.6, "sell")):
            with self.subTest(side=side):
                reactions = evaluate_trigger_level_reactions(
                    swings, [swings[-1]], side, price, 2, 0.001, rules(),
                )
                self.assertFalse(reactions[0]["passed"])
                self.assertEqual(reactions[0]["reason"], "level_zone_not_exited")

    def test_breakouts_from_the_other_side_are_rejected(self):
        for swings, price, approach, side in (
            (bounce_swings(), 107, 99, "buy"),
            (mirror(bounce_swings()), 103, 111, "sell"),
        ):
            with self.subTest(side=side):
                closes = [approach] * 30
                decision, analysis = self.decide(swings, price, closes=closes)
                self.assertIsNone(decision)
                self.assertEqual(analysis["trigger_level_reactions"][0]["reason"],
                                 "level_approached_from_wrong_side")

    def test_quality_threshold_is_shared_with_structure_strategy(self):
        decision, analysis = self.decide(min_level_quality=1.0)
        self.assertIsNone(decision)
        self.assertEqual(analysis["trigger_level_reactions"][0]["reason"], "no_preexisting_level")

    def test_empty_trigger_list_cannot_bypass_entry_filter(self):
        decision, analysis = self.decide(triggers=[])
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["trigger_swing_without_level_reaction"])

    def test_global_trend_analysis_does_not_require_a_local_reaction(self):
        decision, analysis = decide_swing_regression(bounce_swings(), 107, 2, 30, rules())
        self.assertIsNotNone(decision)
        self.assertFalse(analysis["level_reaction_applied"])
        self.assertIsNone(analysis["level_reaction_passed"])

    def test_distance_and_reaction_must_be_valid_for_the_same_trigger(self):
        # A close-to-line swing without a reaction must not borrow a reaction
        # from a second swing that fails its own regression distance check.
        swings = bounce_swings()
        swings[-1] = point("high", 25, 108.0)
        swings += [point("low", 27, 101.0), point("high", 29, 109.0)]
        trigger_with_reaction = swings[11]
        trigger_without_reaction = swings[-1]
        reactions = evaluate_trigger_level_reactions(
            swings, [trigger_with_reaction, trigger_without_reaction], "buy", 107, 2, 0.001, rules(),
        )
        self.assertEqual([reaction["passed"] for reaction in reactions], [True, False])
        with patch("ultimateBot.backtester.strategy.swing_regression_decision_engine.evaluate_trigger_swings",
                   return_value=[{**trigger_with_reaction, "distance_atr": 2.0},
                                 {**trigger_without_reaction, "distance_atr": 0.0}]):
            decision, analysis = self.decide(
                swings, triggers=[trigger_with_reaction, trigger_without_reaction],
                regression_trigger_max_distance_atr=1.0,
            )
        self.assertIsNone(decision)
        self.assertEqual(len(analysis["trigger_level_reactions"]), 1)
        self.assertEqual(analysis["trigger_level_reactions"][0]["trigger"], trigger_without_reaction)

    def test_closed_candle_pipeline_emits_both_sides_after_valid_reactions(self):
        for direction, side in ((1, SignalSide.BUY), (-1, SignalSide.SELL)):
            with self.subTest(direction=direction):
                ticks = reaction_ticks(direction=direction)
                state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=30)
                for tick in ticks:
                    state.update(tick)
                last = ticks[-1]
                state.update(Tick(START + timedelta(minutes=30), "TEST", last.bid, last.ask, 1, 1))
                strategy = SwingRegressionStrategy(rules(), analysis_interval=timedelta(minutes=1))
                signals = strategy.on_market_data(TickBatch("TEST", [last]), StrategyContext(Portfolio(1000), state))
                self.assertEqual(len(signals), 1)
                self.assertEqual(signals[0].side, side)
                self.assertTrue(strategy.last_analysis["TEST"]["level_reaction_passed"])

    def test_pipeline_blocks_missing_levels_and_reports_the_rejection(self):
        ticks = reaction_ticks(drift=0.05)
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=30)
        for tick in ticks:
            state.update(tick)
        last = ticks[-1]
        state.update(Tick(START + timedelta(minutes=30), "TEST", last.bid, last.ask, 1, 1))
        strategy = SwingRegressionStrategy(rules(), analysis_interval=timedelta(minutes=1))
        signals = strategy.on_market_data(TickBatch("TEST", [last]), StrategyContext(Portfolio(1000), state))
        self.assertEqual(signals, [])
        self.assertEqual(strategy.last_analysis["TEST"]["reasons"], ["trigger_swing_without_level_reaction"])
        output = io.StringIO()
        with redirect_stdout(output):
            print_swing_regression_diagnostics(type("Engine", (), {"strategy": strategy})())
        self.assertIn("Support/resistance reaction rejections: 1", output.getvalue())

    def test_cli_enables_filter_and_allows_explicit_comparison_without_it(self):
        for options, enabled in (([], True), (["--no-regression-level-reaction"], False),
                                 (["--regression-level-reaction"], True)):
            with self.subTest(options=options), patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression", *options]):
                engine = build_engine([], parse_args())
                self.assertEqual(engine.strategy.regression_rules["regression_require_level_reaction"], enabled)

    def test_combined_strategy_creates_and_fills_trades_for_both_directions(self):
        for direction, side in ((1, OrderSide.BUY), (-1, OrderSide.SELL)):
            with self.subTest(direction=direction), patch.object(sys, "argv", [
                "run_backtest", "--strategy", "swing-regression", "--level-lookback-bars", "30",
                "--candle-interval", "1m", "--candle-lookback", "30",
                "--regression-global-timeframe-multiplier", "1",
            ]):
                strategy = SwingRegressionStrategy(rules(), analysis_interval=timedelta(minutes=1))
                engine = build_engine(reaction_ticks(100, direction), parse_args(), strategy_override=strategy)
                with redirect_stdout(io.StringIO()):
                    engine.run()
                filled = [*engine.reports.trades, *engine.portfolio.positions.values()]
                self.assertGreater(len(filled), 0)
                self.assertTrue(all(trade.side == side for trade in filled))
                self.assertGreater(strategy.diagnostics_summary()["reason_counts"].get("regression_trend", 0), 0)


if __name__ == "__main__":
    unittest.main()
