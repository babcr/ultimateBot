import io
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_strategy import StructureStrategy
from ultimateBot.backtester.strategy.swing_regression_decision_engine import (
    decide_swing_regression,
    fit_swing_line,
    regression_trend_metrics,
    resolve_swing_regression_rules,
    select_reference_anchor,
)
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, parse_args, print_backtest_summary


START = datetime(2020, 1, 2, tzinfo=timezone.utc)


def swing(kind, index, price):
    return {"kind": kind, "index": index, "price": price, "strength": 3.0}


def rising_channel():
    return ([swing("low", index, 100.0 + 0.1 * index) for index in (0, 10, 20)]
            + [swing("high", index, 102.0 + 0.1 * index) for index in (5, 15, 25)])


def noisy_rising_channel():
    return ([swing("low", index, price) for index, price in ((0, 100.0), (10, 101.0), (20, 104.0))]
            + [swing("high", index, price) for index, price in ((5, 102.0), (15, 103.0), (25, 106.0))])


def dense_rising_channel():
    return ([swing("low", index, 100.0 + 0.05 * index) for index in range(0, 30, 2)]
            + [swing("high", index, 102.0 + 0.05 * index) for index in range(1, 30, 2)])


def reflected(swings):
    return [{**item, "kind": "high" if item["kind"] == "low" else "low",
             "price": 210.0 - item["price"]} for item in swings]


def candle_ticks(index, symbol="TEST", direction=1, bullish=None):
    base = 100.0 + direction * 0.2 * index
    color = (1 if index % 2 == 0 else -1) if bullish is None else (1 if bullish else -1)
    start = START + timedelta(minutes=index)
    prices = ((0, base), (1, base), (10, base + 1.5),
              (20, base - 1.5), (50, base + color))
    return [Tick(start + timedelta(seconds=second), symbol, price, price, 1.0, 1.0)
            for second, price in prices]


class SwingRegressionDecisionTests(unittest.TestCase):
    def decide(self, swings=None, price=104.0, historical_anchor_candidates=None, **rules):
        return decide_swing_regression(
            rising_channel() if swings is None else swings, price, 2.0, 30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_max_swing_density": float("inf"), "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0, **rules},
            historical_anchor_candidates=historical_anchor_candidates,
        )

    def test_reference_anchor_is_the_most_distant_swing_from_price(self):
        points = [swing("high", 5, 110.0), swing("high", 20, 109.0),
                  swing("low", 10, 90.0), swing("low", 25, 92.0)]
        self.assertEqual(select_reference_anchor(points, 104.0), points[2])
        self.assertEqual(select_reference_anchor(points[1:], 104.0), points[2])
        earlier = swing("low", 2, 88.0)
        self.assertEqual(select_reference_anchor([*points, earlier], 104.0), earlier)
        self.assertEqual(select_reference_anchor([swing("low", 1, 100.0), swing("low", 5, 101.0)], 104.0), swing("low", 1, 100.0))
        tied = [swing("low", 1, 100.0), swing("low", 6, 108.0), swing("high", 2, 100.0), swing("high", 7, 108.0)]
        self.assertEqual(select_reference_anchor(tied, 104.0), swing("high", 7, 108.0))
        self.assertIsNone(select_reference_anchor([swing("low", 1, 100.0)], 104.0))

    def test_oldest_lookback_candle_extremes_join_anchor_candidates(self):
        points = [swing("high", 5, 110.0), swing("high", 20, 109.0),
                  swing("low", 10, 96.0), swing("low", 25, 97.0)]
        boundary_candidates = [
            {"kind": "high", "price": 116.0, "index": 0, "source": "lookback_boundary"},
            {"kind": "low", "price": 93.0, "index": 0, "source": "lookback_boundary"},
        ]
        anchor = select_reference_anchor(points, 104.0, historical_anchor_candidates=boundary_candidates)
        self.assertEqual(anchor["price"], 116.0)
        self.assertEqual(anchor["kind"], "high")
        self.assertEqual(anchor["source"], "lookback_boundary")

    def test_constrained_fit_passes_through_anchor_and_floating_fit_optimizes_intercept(self):
        points = [swing("low", index, price) for index, price in enumerate((1.0, 3.0, 2.0))]
        anchored = fit_swing_line(points, points[0])
        floating = fit_swing_line(points)
        self.assertIsNotNone(anchored)
        self.assertIsNotNone(floating)
        self.assertAlmostEqual(anchored.slope, 0.8)
        self.assertAlmostEqual(anchored.price_at(0), 1.0)
        self.assertAlmostEqual(floating.slope, 0.5)
        self.assertAlmostEqual(floating.intercept, 1.5)
        self.assertGreater(floating.r_squared, anchored.r_squared)

    def test_buy_and_sell_keep_the_most_distant_swing_family_as_anchor(self):
        for points, price, side, anchor_kind in (
            (rising_channel(), 104.0, "buy", "low"),
            (rising_channel(), 100.5, "buy", "high"),
            (reflected(rising_channel()), 106.0, "sell", "high"),
            (reflected(rising_channel()), 109.5, "sell", "low"),
        ):
            with self.subTest(side=side, anchor=anchor_kind):
                decision, analysis = self.decide(points, price)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.side, side)
                self.assertEqual(analysis["anchor"]["kind"], anchor_kind)
                self.assertEqual(analysis["anchored_line"]["kind"], anchor_kind)
                self.assertNotEqual(analysis["floating_line"]["kind"], anchor_kind)
                self.assertIsNone(analysis["floating_line"]["anchor_index"])
                self.assertAlmostEqual(analysis["parallelism"], 1.0)
                self.assertAlmostEqual(analysis["shared_slope"], 0.1 if side == "buy" else -0.1)
                if side == "buy":
                    self.assertGreater(decision.trend_score, 0.0)
                else:
                    self.assertLess(decision.trend_score, 0.0)

    def test_horizontal_or_opposite_slopes_are_neutral(self):
        flat = [swing("low", index, 100.0) for index in (0, 10, 20)]
        flat += [swing("high", index, 102.0) for index in (5, 15, 25)]
        divergent = [swing("low", index, 100.0 - 0.2 * index) for index in (0, 10, 20)]
        divergent += [swing("high", index, 102.0 + 0.2 * index) for index in (5, 15, 25)]
        for points in (flat, divergent):
            with self.subTest(points=points):
                decision, analysis = self.decide(points, 101.0, regression_score_threshold=0.0)
                self.assertIsNone(decision)
                self.assertEqual(analysis["trend_score"], 0.0)
                self.assertEqual(analysis["reasons"], ["neutral_trend"])
        self.assertEqual(analysis["parallelism"], 0.0)

    def test_threshold_can_filter_a_weak_trend(self):
        decision, analysis = self.decide(regression_score_threshold=0.99)
        self.assertIsNone(decision)
        self.assertGreater(analysis["slope_strength"], 0.15)
        self.assertLess(analysis["slope_strength"], 0.99)
        self.assertEqual(analysis["reasons"], ["neutral_trend"])

    def test_parallelism_has_its_own_independent_filter(self):
        points = ([swing("low", index, 100.0 + 0.1 * index) for index in (0, 10, 20)]
                  + [swing("high", index, 102.0 + 0.3 * index) for index in (5, 15, 25)])
        decision, analysis = self.decide(points, 108.0, regression_score_threshold=0.15, regression_min_parallelism=0.0)
        self.assertIsNotNone(decision)
        self.assertAlmostEqual(analysis["parallelism"], 0.5)
        self.assertGreater(abs(analysis["slope_strength"]), 0.15)
        decision, analysis = self.decide(points, 108.0, regression_score_threshold=0.15, regression_min_parallelism=0.6)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["insufficient_parallelism"])

    def test_trigger_swing_must_stay_close_to_its_family_regression_line(self):
        trigger = {**swing("high", 29, 104.9), "source": "ask"}
        swings = rising_channel() + [trigger]
        decision, analysis = decide_swing_regression(
            swings, 111.0, 2.0, 30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_trigger_max_distance_atr": 2.0,
             "regression_require_level_reaction": False},
            [trigger],
        )
        self.assertIsNotNone(decision)
        self.assertEqual(len(analysis["trigger_swings"]), 1)
        self.assertAlmostEqual(analysis["trigger_swings"][0]["distance_atr"], 0.0)
        far_trigger = {**swing("high", 29, 108.9), "source": "ask"}
        decision, analysis = decide_swing_regression(
            rising_channel() + [far_trigger], 111.0, 2.0, 30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_trigger_max_distance_atr": 0.5,
             "regression_require_level_reaction": False},
            [far_trigger],
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["trigger_swing_too_far_from_regression"])

    def test_buy_trigger_must_come_from_ask_curve(self):
        wrong_trigger = {**swing("low", 29, 102.9), "source": "bid"}
        decision, analysis = decide_swing_regression(
            rising_channel() + [wrong_trigger],
            111.0,
            2.0,
            30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_trigger_max_distance_atr": 1.0,
             "regression_require_level_reaction": False},
            [wrong_trigger],
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["expected_trigger_source"], "ask")
        self.assertEqual(analysis["reasons"], ["trigger_swings_filtered"])

        good_trigger = {**swing("high", 29, 104.9), "source": "ask"}
        decision, analysis = decide_swing_regression(
            rising_channel() + [good_trigger],
            111.0,
            2.0,
            30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_trigger_max_distance_atr": 1.0,
             "regression_require_level_reaction": False},
            [good_trigger],
        )
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["expected_trigger_source"], "ask")

    def test_sell_trigger_must_come_from_bid_curve(self):
        sell_swings = reflected(rising_channel())
        wrong_trigger = {**swing("high", 29, 107.1), "source": "ask"}
        decision, analysis = decide_swing_regression(
            sell_swings + [wrong_trigger],
            99.0,
            2.0,
            30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_trigger_max_distance_atr": 1.0,
             "regression_require_level_reaction": False},
            [wrong_trigger],
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["expected_trigger_source"], "bid")
        self.assertEqual(analysis["reasons"], ["trigger_swings_filtered"])

        good_trigger = {**swing("low", 29, 105.1), "source": "bid"}
        decision, analysis = decide_swing_regression(
            sell_swings + [good_trigger],
            99.0,
            2.0,
            30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_trigger_max_distance_atr": 1.0,
             "regression_require_level_reaction": False},
            [good_trigger],
        )
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["expected_trigger_source"], "bid")

    def test_reference_line_can_decide_without_floating_regression(self):
        for points, price, side, anchor_kind in (
            ([swing("low", 0, 100.0), swing("low", 10, 101.0), swing("low", 20, 102.0), swing("high", 25, 104.0)], 104.5, "buy", "low"),
            ([swing("high", 0, 110.0), swing("high", 10, 109.0), swing("high", 20, 108.0), swing("low", 25, 106.0)], 105.5, "sell", "high"),
        ):
            with self.subTest(side=side):
                decision, analysis = self.decide(points, price)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.side, side)
                self.assertEqual(analysis["anchor"]["kind"], anchor_kind)
                self.assertIsNone(analysis["floating_line"])
                self.assertTrue(analysis["reference_line_only"])
                self.assertAlmostEqual(analysis["parallelism"], 1.0)
                self.assertAlmostEqual(analysis["shared_slope"], 0.1 if side == "buy" else -0.1)

    def test_close_consecutive_swing_pair_is_filtered_before_regressions(self):
        points = [
            swing("low", 0, 100.0),
            swing("high", 5, 100.4),
            swing("low", 10, 99.0),
            swing("high", 15, 103.0),
            swing("low", 20, 100.0),
            swing("high", 25, 104.0),
        ]
        decision, analysis = self.decide(
            points,
            104.0,
            regression_min_swings_per_side=2,
            regression_min_consecutive_swing_distance_atr=0.3,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["raw_swing_count"], 6)
        self.assertEqual(analysis["swing_count"], 4)
        self.assertEqual(analysis["filtered_consecutive_swing_pair_count"], 1)
        self.assertAlmostEqual(analysis["filtered_consecutive_swing_pairs"][0]["distance_atr"], 0.2)

    def test_close_pair_filter_runs_before_swing_density_gate(self):
        points = [
            swing("low", 0, 100.0),
            swing("high", 2, 100.3),
            swing("low", 4, 100.0),
            swing("high", 8, 102.0),
            swing("low", 12, 100.8),
            swing("high", 16, 102.8),
            swing("low", 20, 101.6),
            swing("high", 24, 103.6),
        ]
        decision, analysis = self.decide(
            points,
            104.0,
            regression_max_swing_density=0.25,
            regression_min_consecutive_swing_distance_atr=0.3,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["raw_swing_count"], 8)
        self.assertEqual(analysis["filtered_consecutive_swing_pair_count"], 1)
        self.assertEqual(analysis["swing_count"], 6)
        self.assertAlmostEqual(analysis["swing_density"], 0.2)
        self.assertNotEqual(analysis["reasons"], ["swing_agitation_too_high"])

    def test_filtered_pair_density_can_filter_false_swing_zones(self):
        points = [
            swing("low", 0, 100.0),
            swing("high", 5, 100.4),
            swing("low", 10, 99.0),
            swing("high", 15, 103.0),
            swing("low", 20, 100.0),
            swing("high", 25, 104.0),
        ]
        decision, analysis = self.decide(
            points,
            104.0,
            regression_min_swings_per_side=2,
            regression_min_consecutive_swing_distance_atr=0.3,
            regression_max_filtered_consecutive_swing_pair_density=0.02,
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["filtered_consecutive_swing_pair_count"], 1)
        self.assertAlmostEqual(analysis["filtered_consecutive_swing_pair_density"], 1.0 / 30.0)
        self.assertEqual(analysis["reasons"], ["false_swing_pair_density_too_high"])

    def test_false_pair_immediately_before_trigger_blocks_trade(self):
        points = [
            swing("low", 0, 100.0),
            swing("high", 5, 102.0),
            swing("low", 10, 100.4),
            swing("high", 20, 101.2),
            swing("low", 25, 101.0),
            swing("high", 29, 103.0),
        ]
        trigger = [points[-1]]
        decision, analysis = decide_swing_regression(
            points,
            103.5,
            2.0,
            30,
            {
                "level_lookback_bars": 30,
                "protective_stop_buffer_atr": 0.25,
                "regression_min_swings_per_side": 2,
                "regression_score_threshold": 0.15,
                "regression_slope_scale_atr": 0.10,
                "regression_min_parallelism": 0.0,
                "regression_min_consecutive_swing_distance_atr": 0.25,
                "regression_max_filtered_consecutive_swing_pair_density": float("inf"),
                "regression_trigger_max_distance_atr": 1.0,
            },
            trigger,
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["false_swing_pair_before_trigger"])
        self.assertEqual(len(analysis["trigger_blocking_filtered_pairs"]), 1)
        self.assertEqual(analysis["trigger_blocking_filtered_pairs"][0]["next_swing"]["index"], 29)

    def test_filtered_latest_trigger_swings_do_not_trigger_regression(self):
        points = [
            swing("low", 0, 100.0),
            swing("high", 10, 103.0),
            swing("low", 20, 101.0),
            swing("high", 28, 104.0),
            swing("low", 29, 103.8),
        ]
        latest = points[-2:]
        decision, analysis = decide_swing_regression(
            points,
            104.0,
            2.0,
            30,
            {
                "level_lookback_bars": 30,
                "protective_stop_buffer_atr": 0.25,
                "regression_min_swings_per_side": 2,
                "regression_score_threshold": 0.15,
                "regression_slope_scale_atr": 0.10,
                "regression_min_parallelism": 0.0,
                "regression_min_consecutive_swing_distance_atr": 0.2,
            },
            latest,
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["trigger_swings_filtered"])

    def test_oldest_lookback_candle_can_become_reference_anchor(self):
        points = ([swing("low", 10, 100.0), swing("low", 20, 101.0), swing("low", 29, 102.0)]
                  + [swing("high", 5, 102.5), swing("high", 15, 103.5), swing("high", 25, 104.5)])
        boundary_candidates = [
            {"kind": "high", "price": 104.5, "index": 0, "source": "lookback_boundary", "label": "oldest_candle_high"},
            {"kind": "low", "price": 99.0, "index": 0, "source": "lookback_boundary", "label": "oldest_candle_low"},
        ]
        decision, analysis = self.decide(points, 104.5, historical_anchor_candidates=boundary_candidates)
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["anchor"]["price"], 99.0)
        self.assertEqual(analysis["anchor"]["kind"], "low")
        self.assertEqual(analysis["anchor"]["source"], "lookback_boundary")
        self.assertEqual(analysis["anchored_line"]["point_count"], 4)

    def test_two_bar_window_can_use_lookback_boundary_as_second_regression_point(self):
        boundary_candidates = [
            {"kind": "low", "price": 99.0, "index": 0, "source": "lookback_boundary", "label": "oldest_candle_low"},
        ]
        decision, analysis = decide_swing_regression(
            [{**swing("low", 1, 100.0), "source": "bid"}],
            102.0,
            2.0,
            2,
            {
                "level_lookback_bars": 2,
                "protective_stop_buffer_atr": 0.25,
                "regression_min_swings_per_side": 2,
                "regression_score_threshold": 0.0,
                "regression_slope_scale_atr": 0.10,
                "regression_max_swing_density": float("inf"),
                "regression_min_parallelism": 0.0,
                "regression_min_consecutive_swing_distance_atr": 0.0,
            },
            historical_anchor_candidates=boundary_candidates,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.side, "buy")
        self.assertEqual(analysis["anchor"]["kind"], "low")
        self.assertEqual(analysis["anchored_line"]["point_count"], 2)
        self.assertEqual(analysis["reasons"], ["regression_trend"])

    def test_line_volatility_can_filter_noisy_regressions(self):
        swings = noisy_rising_channel()
        decision, analysis = self.decide(swings, 106.0, regression_line_max_volatility_atr=0.30)
        self.assertIsNotNone(decision)
        self.assertGreater(analysis["anchored_line_volatility_atr"], 0.2)
        self.assertGreater(analysis["floating_line_volatility_atr"], 0.2)
        decision, analysis = self.decide(swings, 106.0, regression_line_max_volatility_atr=0.20)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["regression_line_too_volatile"])
        self.assertEqual(analysis["volatile_lines"], ["low", "high"])

    def test_market_volatility_filter_blocks_quiet_windows(self):
        decision, analysis = decide_swing_regression(
            rising_channel(),
            104.0,
            2.0,
            30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_market_volatility_window_bars": 12,
             "regression_min_market_volatility_ratio": 0.02},
            market_volatility_atr=1.0,
            market_volatility_ratio=0.009,
        )
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["market_volatility_too_low"])
        self.assertEqual(analysis["market_volatility_window_bars"], 12)
        self.assertAlmostEqual(analysis["market_volatility_ratio"], 0.009)

        decision, analysis = decide_swing_regression(
            rising_channel(),
            104.0,
            2.0,
            30,
            {"level_lookback_bars": 30, "protective_stop_buffer_atr": 0.25,
             "regression_min_swings_per_side": 3,
             "regression_score_threshold": 0.15, "regression_slope_scale_atr": 0.10,
             "regression_min_parallelism": 0.0,
             "regression_min_consecutive_swing_distance_atr": 0.0,
             "regression_market_volatility_window_bars": 12,
             "regression_min_market_volatility_ratio": 0.005},
            market_volatility_atr=1.0,
            market_volatility_ratio=0.009,
        )
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["reasons"], ["regression_trend"])

    def test_swing_density_can_filter_agitated_windows(self):
        swings = dense_rising_channel()
        decision, analysis = self.decide(swings, 104.0, regression_max_swing_density=1.1)
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["swing_count"], 30)
        self.assertAlmostEqual(analysis["swing_density"], 1.0)
        decision, analysis = self.decide(swings, 104.0, regression_max_swing_density=0.8)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["swing_agitation_too_high"])

    def test_score_is_invariant_to_price_units(self):
        _, baseline = self.decide()
        for scale in (0.01, 1000.0):
            with self.subTest(scale=scale):
                points = [{**point, "price": point["price"] * scale} for point in rising_channel()]
                decision, analysis = decide_swing_regression(points, 104.0 * scale, 2.0 * scale, 30,
                                                            {"level_lookback_bars": 30, "regression_score_threshold": 0.15,
                                                             "regression_min_swings_per_side": 3,
                                                             "regression_slope_scale_atr": 0.10,
                                                             "regression_min_consecutive_swing_distance_atr": 0.0})
                self.assertIsNotNone(decision)
                self.assertAlmostEqual(analysis["trend_score"], baseline["trend_score"])
                self.assertAlmostEqual(analysis["parallelism"], baseline["parallelism"])

    def test_only_current_window_swings_influence_origin_and_regressions(self):
        points = [{**point, "index": point["index"] + 70} for point in rising_channel()]
        points += [swing("high", 5, 10000.0), swing("low", 6, 1.0),
                   swing("high", 100, 20000.0), swing("low", 99, float("nan"))]
        decision, analysis = decide_swing_regression(points, 104.0, 2.0, 100,
                                                    {"level_lookback_bars": 30, "regression_score_threshold": 0.15,
                                                     "regression_min_swings_per_side": 3,
                                                     "regression_slope_scale_atr": 0.10,
                                                     "regression_min_consecutive_swing_distance_atr": 0.0})
        _, baseline = self.decide()
        self.assertIsNotNone(decision)
        self.assertEqual(analysis["anchor"]["index"], 70)
        self.assertEqual(analysis["anchor"]["price"], 100.0)
        self.assertEqual(analysis["anchored_line"]["point_count"], 3)
        self.assertEqual(analysis["floating_line"]["point_count"], 3)
        self.assertAlmostEqual(analysis["trend_score"], baseline["trend_score"])

    def test_single_opposite_swing_no_longer_blocks_reference_regression(self):
        points = [swing("low", 0, 100.0), swing("low", 10, 101.0), swing("low", 20, 102.0), swing("high", 25, 103.0)]
        decision, analysis = decide_swing_regression(
            points, 104.0, 2.0, 30,
            {"level_lookback_bars": 30, "regression_score_threshold": 0.15,
               "regression_min_swings_per_side": 3, "regression_slope_scale_atr": 0.10,
               "regression_min_consecutive_swing_distance_atr": 0.0},
        )
        self.assertIsNotNone(decision)
        self.assertNotEqual(analysis["reasons"], ["insufficient_swings"])

    def test_window_length_alone_does_not_amplify_the_score(self):
        scores = []
        for window in (30, 50, 700):
            _, analysis = decide_swing_regression(
                rising_channel(), 104.0, 2.0, window,
                {"level_lookback_bars": window, "regression_score_threshold": 0.95,
                 "regression_min_swings_per_side": 3,
                  "regression_slope_scale_atr": 0.10,
                  "regression_min_consecutive_swing_distance_atr": 0.0},
            )
            scores.append(analysis["trend_score"])
        self.assertAlmostEqual(scores[0], 0.4621171573)
        for score in scores[1:]:
            self.assertAlmostEqual(score, scores[0])

    def test_high_threshold_requires_strong_per_candle_drift_in_both_directions(self):
        # A perfectly parallel channel drifting 0.01 ATR/bar must remain weak even over 700 bars.
        # At 0.2 ATR/bar, the new score passes 0.95 with the 0.1 ATR reference.
        for direction in (1, -1):
            for drift, expected_pass in ((0.01, False), (0.10, False), (0.20, True)):
                with self.subTest(direction=direction, drift=drift):
                    points = ([swing("low", index, 100.0 + direction * drift * index) for index in (0, 10, 20)]
                              + [swing("high", index, 102.0 + direction * drift * index) for index in (5, 15, 25)])
                    price = 102.0 if direction == 1 else 97.0
                    decision, analysis = decide_swing_regression(
                        points, price, 1.0, 700,
                        {"level_lookback_bars": 700, "regression_score_threshold": 0.95,
                         "regression_min_swings_per_side": 3,
                        "regression_slope_scale_atr": 0.10,
                        "regression_min_consecutive_swing_distance_atr": 0.0},
                    )
                    self.assertEqual(decision is not None, expected_pass)
                    self.assertEqual(analysis["trend_score"] > 0.0, direction == 1)
                    if expected_pass:
                        self.assertGreaterEqual(abs(analysis["trend_score"]), 0.95)
                    else:
                        self.assertLess(abs(analysis["trend_score"]), 0.95)

    def test_reference_scale_and_volatility_reduce_trend_strength(self):
        points = rising_channel()
        lows = [point for point in points if point["kind"] == "low"]
        highs = [point for point in points if point["kind"] == "high"]
        anchored = fit_swing_line(lows, lows[0])
        floating = fit_swing_line(highs)
        normal = regression_trend_metrics(anchored, floating, 2.0, 0.1)
        larger_scale = regression_trend_metrics(anchored, floating, 2.0, 0.2)
        larger_atr = regression_trend_metrics(anchored, floating, 4.0, 0.1)
        self.assertLess(larger_scale["trend_score"], normal["trend_score"])
        self.assertAlmostEqual(larger_scale["trend_score"], larger_atr["trend_score"])
        self.assertAlmostEqual(normal["slope_strength"], normal["trend_score"])

    def test_invalid_reference_uses_a_positive_default(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertGreater(resolve_swing_regression_rules({"regression_slope_scale_atr": value})["regression_slope_scale_atr"], 0.0)

    def test_trigger_distance_rule_accepts_zero_or_infinite_thresholds(self):
        self.assertEqual(resolve_swing_regression_rules({"regression_trigger_max_distance_atr": 0.0})["regression_trigger_max_distance_atr"], 0.0)
        self.assertEqual(resolve_swing_regression_rules({"regression_trigger_max_distance_atr": float("inf")})["regression_trigger_max_distance_atr"], float("inf"))

    def test_line_volatility_rule_accepts_zero_or_infinite_thresholds(self):
        self.assertEqual(resolve_swing_regression_rules({"regression_line_max_volatility_atr": 0.0})["regression_line_max_volatility_atr"], 0.0)
        self.assertEqual(resolve_swing_regression_rules({"regression_line_max_volatility_atr": float("inf")})["regression_line_max_volatility_atr"], float("inf"))

    def test_market_volatility_rules_accept_zero_and_clamp_window(self):
        rules = resolve_swing_regression_rules({
            "regression_market_volatility_window_bars": 1,
            "regression_min_market_volatility_ratio": 0.0,
            "regression_global_timeframe_multiplier": 0,
            "regression_global_signal_lookback_bars": 0,
            "regression_global_reference_slope_threshold_atr": -1.0,
            "regression_global_min_parallelism": 2.0,
        })
        self.assertEqual(rules["regression_market_volatility_window_bars"], 2)
        self.assertEqual(rules["regression_min_market_volatility_ratio"], 0.0)
        self.assertEqual(rules["regression_global_timeframe_multiplier"], 1)
        self.assertEqual(rules["regression_global_signal_lookback_bars"], 1)
        self.assertEqual(rules["regression_global_reference_slope_threshold_atr"], 0.0)
        self.assertEqual(rules["regression_global_min_parallelism"], 1.0)
        self.assertEqual(
            resolve_swing_regression_rules({"regression_min_market_volatility_ratio": float("inf")})["regression_min_market_volatility_ratio"],
            float("inf"),
        )

    def test_swing_density_rule_accepts_zero_or_infinite_thresholds(self):
        self.assertEqual(resolve_swing_regression_rules({"regression_max_swing_density": 0.0})["regression_max_swing_density"], 0.0)
        self.assertEqual(resolve_swing_regression_rules({"regression_max_swing_density": float("inf")})["regression_max_swing_density"], float("inf"))

    def test_consecutive_swing_distance_rule_accepts_zero_or_infinite_thresholds(self):
        self.assertEqual(resolve_swing_regression_rules({"regression_min_consecutive_swing_distance_atr": 0.0})["regression_min_consecutive_swing_distance_atr"], 0.0)
        self.assertEqual(resolve_swing_regression_rules({"regression_min_consecutive_swing_distance_atr": float("inf")})["regression_min_consecutive_swing_distance_atr"], float("inf"))

    def test_filtered_pair_density_rule_accepts_zero_or_infinite_thresholds(self):
        self.assertEqual(resolve_swing_regression_rules({"regression_max_filtered_consecutive_swing_pair_density": 0.0})["regression_max_filtered_consecutive_swing_pair_density"], 0.0)
        self.assertEqual(resolve_swing_regression_rules({"regression_max_filtered_consecutive_swing_pair_density": float("inf")})["regression_max_filtered_consecutive_swing_pair_density"], float("inf"))

    def test_parallelism_rule_is_clamped_between_zero_and_one(self):
        self.assertEqual(resolve_swing_regression_rules({"regression_min_parallelism": -1.0})["regression_min_parallelism"], 0.0)
        self.assertEqual(resolve_swing_regression_rules({"regression_min_parallelism": 2.0})["regression_min_parallelism"], 1.0)

    def test_insufficient_and_degenerate_swings_do_not_produce_orders(self):
        cases = (
            ([swing("low", 1, 100.0)], "missing_swing_family"),
            ([swing("low", 1, 100.0), swing("high", 2, 102.0)], "insufficient_swings"),
            ([swing("low", 1, 100.0), swing("low", 2, 101.0), swing("low", 3, 101.5),
              swing("high", 5, 102.0), swing("high", 5, 103.0), swing("high", 5, 104.0)], "degenerate_regression"),
        )
        for points, reason in cases:
            with self.subTest(reason=reason):
                decision, analysis = self.decide(points)
                self.assertIsNone(decision)
                self.assertEqual(analysis["reasons"], [reason])
        decision, analysis = self.decide(regression_min_swings_per_side=4)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["insufficient_swings"])

    def test_invalid_market_or_disabled_engine_does_not_decide(self):
        for price, atr in ((float("nan"), 2.0), (104.0, 0.0), (104.0, float("inf"))):
            with self.subTest(price=price, atr=atr):
                decision, analysis = decide_swing_regression(rising_channel(), price, atr, 30)
                self.assertIsNone(decision)
                self.assertEqual(analysis["reasons"], ["invalid_market_data"])
        decision, analysis = self.decide(enabled=False)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["regression_disabled"])

    def test_protective_line_recent_swing_and_fallback_for_both_directions(self):
        for points, price, line_price, recent_price in (
            (rising_channel(), 104.0, 102.9, 102.0),
            (reflected(rising_channel()), 106.0, 107.1, 108.0),
        ):
            for source, expected in (("structure_level", line_price), ("recent_swing", recent_price)):
                with self.subTest(price=price, source=source):
                    decision, analysis = self.decide(points, price, protective_level_source=source)
                    self.assertIsNotNone(decision)
                    self.assertAlmostEqual(decision.protective_level_price, expected)
                    self.assertAlmostEqual(decision.protective_stop_buffer, 0.5)
                    self.assertEqual(analysis["protective_level_source"],
                                     "regression_line" if source == "structure_level" else "recent_swing")
        decision, analysis = self.decide(price=100.5)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.protective_level_price, 100.0)
        self.assertEqual(analysis["protective_level_source"], "recent_swing")
        decision, analysis = self.decide(price=99.0)
        self.assertIsNone(decision)
        self.assertEqual(analysis["reasons"], ["no_protective_swing"])

    def test_recent_swing_protection_stays_on_adverse_graph_after_trigger_filtering(self):
        cases = (
            (
                [{**item, "source": "ask" if item["kind"] == "high" else "bid"} for item in rising_channel()]
                + [{**swing("high", 29, 104.9), "source": "ask"}],
                111.0,
                "buy",
                "ask",
                "bid",
                "low",
                102.0,
            ),
            (
                [{**item, "source": "ask" if item["kind"] == "high" else "bid"} for item in reflected(rising_channel())]
                + [{**swing("low", 29, 105.1), "source": "bid"}],
                100.0,
                "sell",
                "bid",
                "ask",
                "high",
                108.0,
            ),
        )
        for points, price, side, trigger_source, protective_source, protective_kind, expected_price in cases:
            trigger = points[-1]
            with self.subTest(side=side, trigger_source=trigger_source, protective_source=protective_source):
                decision, analysis = decide_swing_regression(
                    points,
                    price,
                    2.0,
                    30,
                    {
                        "level_lookback_bars": 30,
                        "protective_level_source": "recent_swing",
                        "protective_stop_buffer_atr": 0.25,
                        "regression_min_swings_per_side": 3,
                        "regression_score_threshold": 0.15,
                        "regression_slope_scale_atr": 0.10,
                        "regression_min_parallelism": 0.0,
                        "regression_min_consecutive_swing_distance_atr": 0.0,
                        "regression_trigger_max_distance_atr": 1.0,
                        "regression_require_level_reaction": False,
                    },
                    [trigger],
                )
                self.assertIsNotNone(decision)
                self.assertEqual(decision.side, side)
                self.assertEqual(analysis["expected_trigger_source"], trigger_source)
                self.assertEqual(analysis["protective_level_source"], "recent_swing")
                self.assertIsNotNone(analysis["protective_swing"])
                self.assertEqual(analysis["protective_swing"]["source"], protective_source)
                self.assertEqual(analysis["protective_swing"]["kind"], protective_kind)
                self.assertAlmostEqual(decision.protective_level_price, expected_price)
                self.assertAlmostEqual(decision.protective_stop_buffer, 0.5)

    def test_atr_padding_moves_stop_and_preserves_risk_budget(self):
        for points, price in ((rising_channel(), 104.0), (reflected(rising_channel()), 106.0)):
            with self.subTest(price=price):
                decision, _ = self.decide(points, price)
                manager = RiskManager(risk_per_trade=0.1, minimum_candles=1, reward_risk_ratio=2.0)
                manager.on_tick(Tick(START, "TEST", price - 1.0, price - 1.0, 1.0, 1.0))
                tick = Tick(START + timedelta(seconds=30), "TEST", price, price, 1.0, 1.0)
                manager.on_tick(tick)
                signal = Signal("TEST", tick.timestamp, SignalSide.BUY if decision.side == "buy" else SignalSide.SELL,
                                protective_level_price=decision.protective_level_price,
                                protective_level_quality=decision.protective_level_quality,
                                protective_stop_buffer=decision.protective_stop_buffer)
                padded = manager.evaluate(signal, Portfolio(1000.0), tick)
                original = manager.evaluate(replace(signal, protective_stop_buffer=0.0), Portfolio(1000.0), tick)
                self.assertIsNotNone(padded)
                self.assertIsNotNone(original)
                self.assertAlmostEqual(padded.stop_distance - original.stop_distance, 0.5)
                self.assertLess(padded.quantity, original.quantity)
                self.assertAlmostEqual(padded.quantity * padded.stop_distance, 1.0)
                self.assertAlmostEqual(padded.take_profit_distance, padded.stop_distance * 2.0)

    def test_risk_manager_allows_multiple_positions_on_same_symbol(self):
        manager = RiskManager(risk_per_trade=0.1, minimum_candles=1, reward_risk_ratio=2.0, max_positions=2)
        portfolio = Portfolio(1000.0)
        portfolio.positions[1] = Position(
            trade_id=1,
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=1.0,
            opened_at=START,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=102.0,
        )
        manager.on_tick(Tick(START, "TEST", 103.0, 103.0, 1.0, 1.0))
        tick = Tick(START + timedelta(seconds=30), "TEST", 104.0, 104.0, 1.0, 1.0)
        manager.on_tick(tick)
        evaluation = manager.evaluate(Signal("TEST", tick.timestamp, SignalSide.BUY), portfolio, tick)
        self.assertIsNotNone(evaluation)


class SwingRegressionStrategyTests(unittest.TestCase):
    def context(self, count=30, symbol="TEST", direction=1):
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=30)
        for index in range(count):
            for tick in candle_ticks(index, symbol, direction):
                state.update(tick)
        return StrategyContext(Portfolio(1000.0), state)

    def strategy(self, **rules):
        return SwingRegressionStrategy({"level_lookback_bars": 30, "regression_score_threshold": 0.15,
                                        "regression_min_swings_per_side": 3,
                                        "regression_slope_scale_atr": 0.10,
                                        "regression_global_timeframe_multiplier": 1,
                                        "regression_min_parallelism": 0.0,
                                        "regression_max_swing_density": float("inf"),
                                        "regression_min_consecutive_swing_distance_atr": 0.0,
                                        "regression_require_level_reaction": False,
                                        **rules}, analysis_interval=timedelta(minutes=1))

    def test_active_candles_do_not_confirm_swings_or_set_current_price(self):
        context = self.context()
        strategy = self.strategy()
        previous_tick = candle_ticks(29)[-1]
        batch = TickBatch("TEST", [previous_tick])
        self.assertEqual(strategy.on_market_data(batch, context), [])
        # Closing bar 29 confirms its swing. An extreme tick in active bar 30 stays outside the fit.
        context.market_state.update(Tick(START + timedelta(minutes=30), "TEST", 10000.0, 10000.0, 1.0, 1.0))
        signals = strategy.on_market_data(batch, context)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].side, SignalSide.BUY)
        self.assertEqual(signals[0].timestamp, previous_tick.timestamp)
        analysis = strategy.last_analysis["TEST"]
        self.assertAlmostEqual(analysis["current_price"], 104.8)
        self.assertLess(analysis["anchor"]["price"], 110.0)
        self.assertEqual(analysis["fresh_swings"][0]["index"], 29)
        self.assertEqual(strategy.on_market_data(batch, context), [])

    def test_rolling_window_only_triggers_on_a_new_swing(self):
        context = self.context()
        strategy = self.strategy()
        context.market_state.update(candle_ticks(30)[0])
        self.assertEqual(len(strategy.on_market_data(TickBatch("TEST", [candle_ticks(29)[-1]]), context)), 1)
        for tick in candle_ticks(30):
            context.market_state.update(tick)
        context.market_state.update(candle_ticks(31, bullish=True)[0])
        second_signals = strategy.on_market_data(TickBatch("TEST", [candle_ticks(30)[-1]]), context)
        self.assertEqual(second_signals, [])
        self.assertEqual(strategy.last_analysis["TEST"]["expected_trigger_source"], "ask")
        self.assertEqual(strategy.last_analysis["TEST"]["reasons"], ["trigger_swings_filtered"])
        for tick in candle_ticks(31, bullish=True):
            context.market_state.update(tick)
        context.market_state.update(candle_ticks(32)[0])
        # Two consecutive bullish candles confirm no new pivot, although all window indices shifted.
        self.assertEqual(strategy.on_market_data(TickBatch("TEST", [candle_ticks(31, bullish=True)[-1]]), context), [])
        self.assertEqual(strategy._analysis_windows, 2)

    def test_short_lookback_still_produces_analysis(self):
        context = self.context()
        strategy = self.strategy(
            level_lookback_bars=5,
            regression_market_volatility_window_bars=5,
            regression_min_market_volatility_ratio=0.0,
        )
        previous_tick = candle_ticks(29)[-1]
        batch = TickBatch("TEST", [previous_tick])
        self.assertEqual(strategy.on_market_data(batch, context), [])
        self.assertEqual(strategy._analysis_windows, 1)
        self.assertIn("TEST", strategy.last_analysis)
        context.market_state.update(candle_ticks(30)[0])
        strategy.on_market_data(batch, context)
        self.assertEqual(strategy._analysis_windows, 2)
        self.assertIn("TEST", strategy.last_analysis)
        self.assertTrue(strategy.last_analysis["TEST"]["fresh_swings"])
        self.assertEqual(strategy.last_analysis["TEST"]["fresh_swings"][0]["index"], 4)
        self.assertEqual(strategy.diagnostics_summary()["analysis_windows_by_symbol"], {"TEST": 2})

    def test_backtest_summary_reports_low_volatility_rejections(self):
        context = self.context()
        strategy = self.strategy(
            regression_market_volatility_window_bars=5,
            regression_min_market_volatility_ratio=0.05,
        )
        batch = TickBatch("TEST", [candle_ticks(29)[-1]])
        self.assertEqual(strategy.on_market_data(batch, context), [])
        context.market_state.update(candle_ticks(30)[0])
        self.assertEqual(strategy.on_market_data(batch, context), [])
        diagnostics = strategy.diagnostics_summary()
        self.assertEqual(diagnostics["reason_counts"].get("market_volatility_too_low"), 1)
        self.assertEqual(diagnostics["reason_counts_by_symbol"].get("TEST", {}).get("market_volatility_too_low"), 1)

        engine = SimpleNamespace(
            portfolio=Portfolio(1000.0),
            reports=SimpleNamespace(trades=[]),
            strategy=strategy,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_backtest_summary(engine)
        text = output.getvalue()
        self.assertIn("Swing-regression diagnostics:", text)
        self.assertIn("Low-volatility rejections: 1 (100.00%)", text)
        self.assertIn("TEST: 1 (100.00%) / 1 analyses", text)

    def test_global_trend_filter_blocks_local_signal_on_reference_slope_mismatch(self):
        context = self.context(count=12)
        context.market_state.update(candle_ticks(12)[0])
        strategy = self.strategy(
            level_lookback_bars=5,
            regression_global_timeframe_multiplier=2,
            regression_global_reference_slope_threshold_atr=0.05,
            regression_market_volatility_window_bars=5,
            regression_min_market_volatility_ratio=0.0,
        )
        local_decision = SimpleNamespace(
            side="buy",
            confidence=0.8,
            protective_level_price=99.0,
            protective_level_quality=0.7,
            protective_stop_buffer=0.2,
        )
        global_analysis = {
            "anchor": None,
            "anchored_line": {"kind": "high", "slope": -0.08},
            "floating_line": None,
            "parallelism": 1.0,
            "atr": 1.0,
            "reasons": ["neutral_trend"],
            "closed_at": START + timedelta(minutes=10),
        }
        local_analysis = {
            "anchor": None,
            "anchored_line": None,
            "floating_line": None,
            "direction": "buy",
            "reasons": ["regression_trend"],
            "closed_at": START + timedelta(minutes=11),
        }
        with patch.object(strategy, "_evaluate_regression_pipeline", side_effect=[
            (None, global_analysis),
            (local_decision, local_analysis),
        ]):
            signals = strategy.on_market_data(TickBatch("TEST", [candle_ticks(11)[-1]]), context)
        self.assertEqual(signals, [])
        self.assertEqual(strategy.last_analysis["TEST"]["reasons"], ["global_reference_slope_direction_mismatch"])
        self.assertFalse(strategy.last_analysis["TEST"]["global_confirmation_passed"])
        self.assertLess(strategy.last_analysis["TEST"]["global_reference_slope_atr"], 0.0)

    def test_global_trend_filter_accepts_aligned_reference_slope_without_global_signal(self):
        context = self.context(count=12)
        context.market_state.update(candle_ticks(12)[0])
        strategy = self.strategy(
            level_lookback_bars=5,
            regression_global_timeframe_multiplier=2,
            regression_global_reference_slope_threshold_atr=0.05,
            regression_market_volatility_window_bars=5,
            regression_min_market_volatility_ratio=0.0,
        )
        global_analysis = {
            "anchor": None,
            "anchored_line": {"kind": "low", "slope": 0.08},
            "floating_line": None,
            "parallelism": 0.75,
            "atr": 1.0,
            "reasons": ["neutral_trend"],
            "closed_at": START + timedelta(minutes=10),
        }
        local_decision = SimpleNamespace(
            side="buy",
            confidence=0.8,
            protective_level_price=99.0,
            protective_level_quality=0.7,
            protective_stop_buffer=0.2,
        )
        local_analysis = {
            "anchor": None,
            "anchored_line": None,
            "floating_line": None,
            "direction": "buy",
            "reasons": ["regression_trend"],
            "closed_at": START + timedelta(minutes=11),
        }
        with patch.object(strategy, "_evaluate_regression_pipeline", side_effect=[
            (None, global_analysis),
            (local_decision, local_analysis),
        ]):
            signals = strategy.on_market_data(TickBatch("TEST", [candle_ticks(11)[-1]]), context)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].side, SignalSide.BUY)
        self.assertTrue(strategy.last_analysis["TEST"]["global_confirmation_passed"])
        self.assertEqual(strategy.last_analysis["TEST"]["global_confirmation_reason"], "global_trend_confirmed")
        self.assertGreater(strategy.last_analysis["TEST"]["global_reference_slope_atr"], 0.0)

    def test_global_trend_filter_can_require_global_parallelism(self):
        context = self.context(count=12)
        context.market_state.update(candle_ticks(12)[0])
        strategy = self.strategy(
            level_lookback_bars=5,
            regression_global_timeframe_multiplier=2,
            regression_global_reference_slope_threshold_atr=0.05,
            regression_global_min_parallelism=0.6,
            regression_market_volatility_window_bars=5,
            regression_min_market_volatility_ratio=0.0,
        )
        local_decision = SimpleNamespace(
            side="buy",
            confidence=0.8,
            protective_level_price=99.0,
            protective_level_quality=0.7,
            protective_stop_buffer=0.2,
        )
        global_analysis = {
            "anchor": None,
            "anchored_line": {"kind": "low", "slope": 0.08},
            "floating_line": {"kind": "high", "slope": 0.12},
            "parallelism": 0.4,
            "atr": 1.0,
            "reasons": ["regression_trend"],
            "closed_at": START + timedelta(minutes=10),
        }
        local_analysis = {
            "anchor": None,
            "anchored_line": None,
            "floating_line": None,
            "direction": "buy",
            "reasons": ["regression_trend"],
            "closed_at": START + timedelta(minutes=11),
        }
        with patch.object(strategy, "_evaluate_regression_pipeline", side_effect=[
            (None, global_analysis),
            (local_decision, local_analysis),
        ]):
            signals = strategy.on_market_data(TickBatch("TEST", [candle_ticks(11)[-1]]), context)
        self.assertEqual(signals, [])
        self.assertEqual(strategy.last_analysis["TEST"]["reasons"], ["global_parallelism_too_low"])
        self.assertAlmostEqual(strategy.last_analysis["TEST"]["global_parallelism"], 0.4)

    def test_adaptive_local_windows_preserve_global_analysis_for_each_symbol(self):
        context = self.context(count=24)
        for index in range(24):
            for tick in candle_ticks(index, "OTHER", direction=-1):
                context.market_state.update(tick)
        for symbol in ("TEST", "OTHER"):
            context.market_state.update(candle_ticks(24, symbol)[0])
        settings = dict(level_lookback_bars=5, regression_global_timeframe_multiplier=2,
                        regression_market_volatility_window_bars=5,
                        regression_min_market_volatility_ratio=0.0)
        control, adaptive = self.strategy(**settings), self.strategy(**settings)
        for symbol in ("TEST", "OTHER"):
            batch = TickBatch(symbol, [candle_ticks(24, symbol)[0]])
            control.on_market_data(batch, context)
            adaptive.on_market_data(batch, context)
            cached = adaptive.last_global_analysis[symbol]
            closed = adaptive._last_global_closed_candle[symbol]
            for lookback in (3, 20, 8):
                adaptive.set_history_lookback_bars(lookback, symbol=symbol)
                adaptive.on_market_data(batch, context)
                self.assertIs(adaptive.last_global_analysis[symbol], cached)
                self.assertEqual(adaptive._last_global_closed_candle[symbol], closed)
        adaptive.set_history_lookback_bars(20, symbol="TEST")
        adaptive.set_history_lookback_bars(3, symbol="OTHER")
        for index in (24, 25):
            for symbol in ("TEST", "OTHER"):
                for tick in candle_ticks(index, symbol):
                    context.market_state.update(tick)
        for symbol in ("TEST", "OTHER"):
            next_tick = candle_ticks(26, symbol)[0]
            context.market_state.update(next_tick)
            batch = TickBatch(symbol, [next_tick])
            control.on_market_data(batch, context)
            adaptive.on_market_data(batch, context)
            self.assertEqual(adaptive.last_global_analysis[symbol], control.last_global_analysis[symbol])
            self.assertEqual(adaptive.last_global_analysis[symbol]["bar_count"], 5)
            self.assertEqual(adaptive.global_history_lookback_bars, 5)
            self.assertEqual(adaptive.global_history_capacity, 10)

    def test_global_confirmation_continues_while_larger_local_window_warms_up(self):
        context = self.context(count=10)
        context.market_state.update(candle_ticks(10)[0])
        strategy = self.strategy(level_lookback_bars=5, regression_global_timeframe_multiplier=2,
                                 regression_market_volatility_window_bars=5,
                                 regression_min_market_volatility_ratio=0.0)
        strategy.on_market_data(TickBatch("TEST", [candle_ticks(10)[0]]), context)
        cached = strategy.last_global_analysis["TEST"]
        strategy.set_history_lookback_bars(20, symbol="TEST")
        self.assertIs(strategy.last_global_analysis["TEST"], cached)
        self.assertIsNone(strategy._get_structure_candles("TEST", context))
        for index in (10, 11):
            for tick in candle_ticks(index):
                context.market_state.update(tick)
        next_tick = candle_ticks(12)[0]
        context.market_state.update(next_tick)
        self.assertEqual(strategy.on_market_data(TickBatch("TEST", [next_tick]), context), [])
        self.assertIsNone(strategy._get_structure_candles("TEST", context))
        self.assertEqual(strategy.last_global_analysis["TEST"]["bar_count"], 5)
        self.assertGreater(strategy.last_global_analysis["TEST"]["closed_at"], cached["closed_at"])
        self.assertEqual(strategy._global_analysis_windows, 2)

    def test_closed_candle_tracking_is_independent_for_each_symbol(self):
        context = self.context()
        for index in range(30):
            for tick in candle_ticks(index, "OTHER", direction=-1):
                context.market_state.update(tick)
        for symbol, direction in (("TEST", 1), ("OTHER", -1)):
            context.market_state.update(candle_ticks(30, symbol, direction)[0])
        strategy = self.strategy()
        for symbol, direction, expected in (("TEST", 1, SignalSide.BUY), ("OTHER", -1, None)):
            batch = TickBatch(symbol, [candle_ticks(29, symbol, direction)[-1]])
            signals = strategy.on_market_data(batch, context)
            if expected is None:
                self.assertEqual(signals, [])
                self.assertEqual(strategy.last_analysis[symbol]["expected_trigger_source"], "bid")
                self.assertEqual(strategy.last_analysis[symbol]["reasons"], ["trigger_swings_filtered"])
            else:
                self.assertEqual(len(signals), 1)
                self.assertEqual(signals[0].side, expected)
            self.assertEqual(strategy.on_market_data(batch, context), [])

    def test_structure_and_regression_share_the_same_swing_vectors(self):
        context = self.context()
        context.market_state.update(candle_ticks(30)[0])
        regression = self.strategy()
        structure = StructureStrategy({"level_lookback_bars": 60})
        frames = {source: regression._build_candle_frame(context.market_state.get_candles("TEST", price_source=source))
                  for source in ("bid", "ask")}
        regression_swings = regression._build_shared_structure_swings(frames["bid"], frames["ask"], 0.01)
        structure_swings = structure._build_shared_structure_swings(frames["bid"], frames["ask"], 0.01)
        self.assertTrue(regression_swings)
        self.assertEqual(regression_swings, structure_swings)
        self.assertEqual({item["source"] for item in regression_swings if item["kind"] == "low"}, {"bid"})
        self.assertEqual({item["source"] for item in regression_swings if item["kind"] == "high"}, {"ask"})

    def test_cli_selects_new_engine_and_preserves_structure_default(self):
        with patch.object(sys, "argv", ["run_backtest"]):
            engine = build_engine([], parse_args())
        self.assertIsInstance(engine.strategy, StructureStrategy)
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression",
                                        "--level-lookback-bars", "45", "--regression-min-swings", "3",
                                        "--regression-score-threshold", "0.4", "--regression-min-parallelism", "0.65", "--regression-slope-scale-atr", "0.2",
                                        "--regression-global-timeframe-multiplier", "3",
                                        "--regression-global-reference-slope-threshold-atr", "0.07",
                                        "--regression-global-min-parallelism", "0.8",
                                        "--regression-global-signal-lookback-bars", "2",
                                        "--regression-market-volatility-window-bars", "60",
                                        "--regression-min-market-volatility-ratio", "0.003",
                                        "--regression-line-max-volatility-atr", "0.24",
                                        "--regression-min-consecutive-swing-distance-atr", "0.35",
                                        "--regression-max-filtered-consecutive-swing-pair-density", "0.08",
                                        "--regression-max-swing-density", "0.75",
                                        "--regression-trigger-max-distance-atr", "0.6",
                                        "--candle-interval", "30m",
                                        "--structure-stop-buffer-atr", "0.75", "--structure-stop-source", "recent_swing"]):
            engine = build_engine([], parse_args())
        self.assertIsInstance(engine.strategy, SwingRegressionStrategy)
        self.assertEqual(engine.strategy.minimum_candles, 180)
        self.assertEqual(engine.strategy.regression_rules["level_lookback_bars"], 45)
        self.assertEqual(engine.strategy.regression_rules["regression_min_swings_per_side"], 3)
        self.assertEqual(engine.strategy.regression_rules["regression_score_threshold"], 0.4)
        self.assertEqual(engine.strategy.regression_rules["regression_min_parallelism"], 0.65)
        self.assertEqual(engine.strategy.regression_rules["regression_slope_scale_atr"], 0.2)
        self.assertEqual(engine.strategy.regression_rules["regression_global_timeframe_multiplier"], 3)
        self.assertEqual(engine.strategy.regression_rules["regression_global_reference_slope_threshold_atr"], 0.07)
        self.assertEqual(engine.strategy.regression_rules["regression_global_min_parallelism"], 0.8)
        self.assertEqual(engine.strategy.regression_rules["regression_global_signal_lookback_bars"], 2)
        self.assertEqual(engine.strategy.regression_rules["regression_market_volatility_window_bars"], 60)
        self.assertEqual(engine.strategy.regression_rules["regression_min_market_volatility_ratio"], 0.003)
        self.assertEqual(engine.strategy.regression_rules["regression_line_max_volatility_atr"], 0.24)
        self.assertEqual(engine.strategy.regression_rules["regression_min_consecutive_swing_distance_atr"], 0.35)
        self.assertEqual(engine.strategy.regression_rules["regression_max_filtered_consecutive_swing_pair_density"], 0.08)
        self.assertEqual(engine.strategy.regression_rules["regression_max_swing_density"], 0.75)
        self.assertEqual(engine.strategy.regression_rules["regression_trigger_max_distance_atr"], 0.6)
        self.assertEqual(engine.strategy.regression_rules["protective_stop_buffer_atr"], 0.75)
        self.assertEqual(engine.strategy.regression_rules["protective_level_source"], "recent_swing")
        self.assertEqual(engine.batcher.analysis_interval, timedelta(minutes=30))
        self.assertEqual(engine.market_state._candle_builders["mid"].lookback_candles, 180)

    def test_cli_accepts_two_bar_regression_lookback(self):
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression",
                                        "--level-lookback-bars", "2",
                                        "--regression-global-timeframe-multiplier", "1",
                                        "--regression-min-market-volatility-ratio", "0"]):
            engine = build_engine([], parse_args())
        self.assertIsInstance(engine.strategy, SwingRegressionStrategy)
        self.assertEqual(engine.strategy.regression_rules["level_lookback_bars"], 2)
        self.assertEqual(engine.strategy.minimum_candles, 2)

    def test_full_backtest_creates_fills_and_closes_regression_trades(self):
        feed = [tick for index in range(100) for tick in candle_ticks(index)]
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression",
                                        "--level-lookback-bars", "30", "--candle-interval", "1m",
                                        "--regression-global-timeframe-multiplier", "1",
                                        "--regression-min-swings", "3",
                                        "--regression-score-threshold", "0.15",
                                        "--regression-slope-scale-atr", "0.1",
                                        "--regression-min-parallelism", "0.0",
                                        "--regression-min-consecutive-swing-distance-atr", "0",
                                        "--reward-risk-ratio", "1",
                                        "--structure-stop-buffer-atr", "0.25",
                                        "--regression-max-swing-density", "inf",
                                        "--no-regression-level-reaction",
                                        "--regression-live-analysis"]):
            engine = build_engine(feed, parse_args())
        output = io.StringIO()
        with redirect_stdout(output):
            engine.run()
        self.assertGreater(len(engine.reports.trades), 0)
        self.assertTrue(all(trade.side == OrderSide.BUY for trade in engine.reports.trades))
        for marker in ("[swing-regression]", "[trade-created]", "[trade-filled]", "[trade-closed]"):
            self.assertIn(marker, output.getvalue())


if __name__ == "__main__":
    unittest.main()
