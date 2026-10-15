import unittest
from datetime import datetime, timedelta, timezone
from random import Random
from unittest.mock import patch

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import CandleBuilder
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.strategy.structure_decision_engine import merge_structure_reversal_swings
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy


START = datetime(2020, 1, 2, tzinfo=timezone.utc)


class SwingHistoryTests(unittest.TestCase):
    def strategy(self):
        return SwingRegressionStrategy({"level_lookback_bars": 5,
                                        "regression_global_timeframe_multiplier": 1,
                                        "regression_min_market_volatility_ratio": 0.0},
                                       analysis_interval=timedelta(minutes=1))

    def project(self, strategy, state, symbol, lookback):
        strategy.set_history_lookback_bars(lookback, symbol=symbol)
        frames = {source: strategy._build_candle_frame(
            state.get_candles(symbol, price_source=source, limit=lookback))
            for source in ("bid", "ask")}
        return strategy._build_shared_structure_swings(
            frames["bid"], frames["ask"], 0.001,
            rules=strategy.rules_for_symbol(symbol), symbol=symbol, market_state=state,
        ), frames

    def test_incremental_vectors_match_full_detection_across_windows_symbols_and_gaps(self):
        rng = Random(91)
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=31)
        strategy = self.strategy()
        for bar in range(48):
            for symbol in ("EURUSD", "USDCHF"):
                minute = bar + bar // 9
                opening = 100.0 + rng.uniform(-2, 2)
                closing = opening if bar % 6 == 0 else opening + rng.choice((-1, 1)) * rng.uniform(0.1, 2)
                opening_spread = rng.uniform(0.1, 1)
                prices = (opening, opening + 3, opening - 3, closing)
                for offset, bid in zip((0, 12, 32, 50), prices):
                    spread = opening_spread if offset == 0 or (offset == 50 and bar % 6 == 0) else rng.uniform(0.1, 1)
                    state.update(Tick(START + timedelta(minutes=minute, seconds=offset),
                                      symbol, bid, bid + spread, 1, 1))
                if bar < 2:
                    continue
                for lookback in (2, 5, 13, 31):
                    with self.subTest(bar=bar, symbol=symbol, lookback=lookback):
                        incremental, frames = self.project(strategy, state, symbol, lookback)
                        expected = merge_structure_reversal_swings(
                            frames["bid"], frames["ask"], 0.001, strategy.rules_for_symbol(symbol))
                        self.assertEqual(incremental, expected)
                        self.assertTrue(all(0 < swing["index"] < len(frames["bid"]) for swing in incremental))

    def test_boundary_pair_is_excluded_and_normalization_is_redone_after_window_change(self):
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=5)
        strategy = self.strategy()
        # Two lows separated by a doji: the older extreme suppresses the newer
        # low only when the two candles that confirmed it are inside the window.
        candles = ((100, 101, 90, 99), (100, 102, 95, 101), (100, 101, 96, 100),
                   (100, 101, 98, 99), (100, 102, 98, 101))
        for minute, prices in enumerate(candles):
            for offset, bid in zip((0, 12, 32, 50), prices):
                state.update(Tick(START + timedelta(minutes=minute, seconds=offset),
                                  "EURUSD", bid, bid + 0.2, 1, 1))
        state.update(Tick(START + timedelta(minutes=5), "EURUSD", 100, 100.2, 1, 1))
        # Les lows sont des prix ask : le spread de 0.2 est inclus.
        for lookback, expected_price, expected_index in ((5, 90.2, 1), (4, 98.2, 3), (2, 98.2, 1), (5, 90.2, 1)):
            with self.subTest(lookback=lookback):
                swings, _ = self.project(strategy, state, "EURUSD", lookback)
                self.assertEqual([(s["kind"], s["price"], s["index"]) for s in swings],
                                 [("low", expected_price, expected_index)])

    def test_detection_runs_once_per_new_closed_candle_and_queries_reuse_history(self):
        builder = CandleBuilder(timedelta(minutes=1), lookback_candles=6)
        for minute in range(6):
            builder.update(Tick(START + timedelta(minutes=minute), "EURUSD", 100, 100, 1, 1))
            price = 101 if minute % 2 == 0 else 99
            builder.update(Tick(START + timedelta(minutes=minute, seconds=20), "EURUSD", price, price, 1, 1))
        before = builder.get_reversal_swings("EURUSD", 1, limit=3)
        history = builder._swing_histories["EURUSD"]
        with patch.object(history, "append", wraps=history.append) as append:
            builder.update(Tick(START + timedelta(minutes=5, seconds=30), "EURUSD", 80, 80, 1, 1))
            self.assertEqual(builder.get_reversal_swings("EURUSD", 1, limit=3), before)
            for limit in (2, 6, 3, 5):
                builder.get_reversal_swings("EURUSD", 2, limit=limit)
            self.assertEqual(append.call_count, 0)
            builder.update(Tick(START + timedelta(minutes=6), "EURUSD", 100, 100, 1, 1))
            self.assertEqual(append.call_count, 1)
            builder.get_reversal_swings("EURUSD", 1, limit=6)
            self.assertIs(builder._swing_histories["EURUSD"], history)
            builder.finalize_active_candles()
            self.assertEqual(append.call_count, 2)

    def test_strength_uses_current_atr_and_projection_cannot_modify_candidates(self):
        builder = CandleBuilder(timedelta(minutes=1), lookback_candles=3)
        for minute, closing in enumerate((101, 99, 101)):
            builder.update(Tick(START + timedelta(minutes=minute), "EURUSD", 100, 100, 1, 1))
            builder.update(Tick(START + timedelta(minutes=minute, seconds=20), "EURUSD", closing, closing, 1, 1))
        builder.finalize_active_candles()
        first = builder.get_reversal_swings("EURUSD", 1)
        second = builder.get_reversal_swings("EURUSD", 2)
        self.assertEqual([s["strength"] for s in first], [2 * s["strength"] for s in second])
        first[0]["price"] = -1000
        self.assertEqual(builder.get_reversal_swings("EURUSD", 2), second)

    def test_candidate_storage_is_bounded_and_no_missing_candles_are_invented(self):
        builder = CandleBuilder(timedelta(minutes=1), lookback_candles=4)
        self.assertEqual(builder.get_reversal_swings("EURUSD", 1), [])
        for minute in range(80):
            builder.update(Tick(START + timedelta(minutes=minute), "EURUSD", 100, 100, 1, 1))
            price = 101 if minute % 2 == 0 else 99
            builder.update(Tick(START + timedelta(minutes=minute, seconds=20), "EURUSD", price, price, 1, 1))
            swings = builder.get_reversal_swings("EURUSD", 1)
            self.assertLessEqual(len(swings), 3)
            if minute:
                self.assertLessEqual(len(builder._swing_histories["EURUSD"]._candidates), 3)
        self.assertEqual(builder.get_reversal_swings("EURUSD", 1, limit=0), [])
        self.assertEqual(builder.get_reversal_swings("EURUSD", 1, limit=1), [])
        with self.assertRaises(ValueError):
            builder.get_reversal_swings("EURUSD", 1, limit=-1)


if __name__ == "__main__":
    unittest.main()
