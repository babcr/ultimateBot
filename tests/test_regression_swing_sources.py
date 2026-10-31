import unittest
from datetime import timedelta
from unittest.mock import patch

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.swing_regression_decision_engine import (
    fit_swing_line, historical_family_candidates, select_reference_anchor, select_swing_window,
)
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.tests.test_swing_regression import START, candle_ticks


class RegressionSwingSourceTests(unittest.TestCase):
    def test_normalized_swings_reject_wrong_or_unknown_quote_sources(self):
        swings = [
            {"kind": "low", "index": 1, "price": 101, "source": "ask"},
            {"kind": "high", "index": 2, "price": 103, "source": "bid"},
            {"kind": "low", "index": 3, "price": 90, "source": "bid"},
            {"kind": "high", "index": 4, "price": 110, "source": "ask"},
            {"kind": "high", "index": 5, "price": 112, "source": "mid"},
            {"kind": "low", "index": 6, "price": 89, "source": "ask", "price_source": "bid"},
        ]
        self.assertEqual(select_swing_window(swings, 0, 10, 1), swings[:2])

    def test_historical_origins_reject_wrong_quote_sources(self):
        valid_low = {"kind": "low", "index": 0, "price": 99,
                     "source": "lookback_boundary", "price_source": "ask"}
        valid_high = {"kind": "high", "index": 0, "price": 105,
                      "source": "lookback_boundary", "price_source": "bid"}
        invalid = [
            {**valid_low, "price": 1, "price_source": "bid"},
            {**valid_high, "price": 1000, "price_source": "ask"},
            {**valid_low, "price": 0.5, "price_source": "mid"},
        ]
        candidates = [*invalid, valid_low, valid_high]
        self.assertEqual(historical_family_candidates("low", candidates), [valid_low])
        self.assertEqual(historical_family_candidates("high", candidates), [valid_high])
        family = [{"kind": "low", "index": 3, "price": 101, "source": "ask"},
                  {"kind": "high", "index": 4, "price": 104, "source": "bid"}]
        self.assertEqual(select_reference_anchor([*family, *invalid], 104, 2, candidates), valid_low)

    def test_raw_swings_preserve_only_matching_families_without_mutating_cache(self):
        swings = [
            {"kind": "low", "index": 1, "price": 101, "source": "ask", "pair_range": 3},
            {"kind": "high", "index": 2, "price": 103, "source": "bid", "pair_range": 3},
            {"kind": "low", "index": 3, "price": 80, "source": "ask", "price_source": "bid", "pair_range": 30},
            {"kind": "high", "index": 4, "price": 120, "source": "ask", "pair_range": 30},
            {"kind": "low", "index": 5, "price": 70, "source": "bid", "pair_range": 30},
        ]
        before = [dict(point) for point in swings]
        selected = select_swing_window(swings, 0, 10, 1)
        self.assertEqual([(p["kind"], p["source"], p["price"]) for p in selected],
                         [("low", "ask", 101), ("high", "bid", 103)])
        self.assertEqual(swings, before)

    def test_legacy_synthetic_swings_without_source_remain_supported(self):
        swings = [{"kind": "low", "index": 1, "price": 101},
                  {"kind": "high", "index": 2, "price": 103}]
        self.assertEqual(select_swing_window(swings, 0, 10, 1), swings)

    def test_every_point_fitted_locally_and_globally_matches_its_quote_candles(self):
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=60)
        for bar in range(60):
            for tick in candle_ticks(bar):
                spread = 0.3 + 0.002 * bar + 0.01 * (tick.timestamp.second % 7)
                state.update(Tick(tick.timestamp, "TEST", tick.bid, tick.bid + spread, 1, 1))
        state.finalize()
        strategy = SwingRegressionStrategy({
            "regression_lookback_bars": 10, "level_lookback_bars": 20,
            "regression_min_swings_per_side": 3, "regression_score_threshold": 0,
            "regression_min_parallelism": 0, "regression_min_consecutive_swing_distance_atr": 0,
            "regression_min_market_volatility_ratio": 0, "regression_require_level_reaction": False,
            "regression_global_timeframe_multiplier": 3,
        }, analysis_interval=timedelta(minutes=1))
        candles = {source: state.get_candles("TEST", price_source=source) for source in ("bid", "ask", "mid")}

        def verify_fits(fitter, bars, multiplier):
            self.assertEqual(fitter.call_count, 2)  # Droite ancree et droite libre.
            self.assertEqual({call.args[0][0]["kind"] for call in fitter.call_args_list}, {"low", "high"})
            for call in fitter.call_args_list:
                for point in call.args[0]:
                    kind, index = point["kind"], point["index"]
                    source = "bid" if kind == "high" else "ask"
                    self.assertEqual(point.get("price_source", point["source"]), source)
                    # Une origine de bord utilise une bougie ; un swing est l'extreme des deux.
                    start = index if point["source"] == "lookback_boundary" else index - 1
                    self.assertGreaterEqual(start, 0)
                    group = bars[source][start * multiplier:(index + 1) * multiplier]
                    expected_price = (max(c.high_price for c in group) if kind == "high"
                                      else min(c.low_price for c in group))
                    self.assertAlmostEqual(point["price"], expected_price)

        with patch("ultimateBot.backtester.strategy.swing_regression_decision_engine.fit_swing_line",
                   wraps=fit_swing_line) as fitter:
            evaluation = strategy._evaluate_regression_pipeline(
                "TEST", candles["bid"], candles["ask"], candles["mid"], 0.01,
                timeframe_label="local", timeframe_multiplier=1, context=StrategyContext(None, state),
            )
            self.assertIsNotNone(evaluation)
            self.assertIsNotNone(evaluation[0])
            verify_fits(fitter, {source: values[-10:] for source, values in candles.items()}, 1)
            fitter.reset_mock()
            strategy._refresh_global_pipeline("TEST", candles["bid"], candles["ask"], candles["mid"], 0.01)
            self.assertIn("TEST", strategy.last_global_analysis)
            verify_fits(fitter, {source: values[-30:] for source, values in candles.items()}, 3)


if __name__ == "__main__":
    unittest.main()
