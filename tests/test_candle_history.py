import unittest
from datetime import datetime, timedelta, timezone

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import CandleBuilder
from ultimateBot.backtester.market.market_state import MarketState


START = datetime(2020, 1, 2, tzinfo=timezone.utc)


def tick(minute, symbol="EURUSD"):
    return Tick(START + timedelta(minutes=minute), symbol, 100.0 + minute, 100.2 + minute, 1, 1)


class CandleHistoryTests(unittest.TestCase):
    def test_projection_returns_latest_closed_candles_in_time_order(self):
        builder = CandleBuilder(timedelta(minutes=1), lookback_candles=5)
        self.assertEqual(builder.get_candles("UNKNOWN", limit=3), [])
        for minute in range(8):
            builder.update(tick(minute))
        history = builder.get_candles("EURUSD", include_active=False)
        projection = builder.get_candles("EURUSD", include_active=False, limit=3)
        self.assertEqual([c.start_time for c in history],
                         [START + timedelta(minutes=m) for m in range(2, 7)])
        self.assertEqual([c.start_time for c in projection],
                         [START + timedelta(minutes=m) for m in range(4, 7)])
        for original, projected in zip(history[-3:], projection):
            self.assertIs(original, projected)
        projection.clear()
        self.assertEqual(builder.get_candles("EURUSD", include_active=False), history)
        self.assertEqual(builder.get_candles("EURUSD", include_active=False, limit=20), history)

    def test_limit_includes_active_candle_and_preserves_unlimited_reads(self):
        builder = CandleBuilder(timedelta(minutes=1), lookback_candles=3)
        for minute in range(5):
            builder.update(tick(minute))
        self.assertEqual(len(builder.get_candles("EURUSD")), 4)
        for limit in (0, 1, 2, 3, 8):
            with self.subTest(limit=limit):
                candles = builder.get_candles("EURUSD", limit=limit)
                expected = list(range(1, 5))[-limit:] if limit else []
                self.assertEqual([c.start_time for c in candles],
                                 [START + timedelta(minutes=m) for m in expected])
        builder.finalize_active_candles()
        self.assertEqual([c.start_time for c in builder.get_candles("EURUSD", limit=2)],
                         [START + timedelta(minutes=m) for m in (3, 4)])

    def test_warmup_returns_available_candles_without_padding(self):
        builder = CandleBuilder(timedelta(minutes=1), lookback_candles=10)
        builder.update(tick(0))
        self.assertEqual(builder.get_candles("EURUSD", include_active=False, limit=5), [])
        builder.update(tick(1))
        candles = builder.get_candles("EURUSD", include_active=False, limit=5)
        self.assertEqual([c.start_time for c in candles], [START])
        with self.assertRaises(ValueError):
            builder.get_candles("EURUSD", limit=-1)

    def test_market_state_projects_each_price_source_and_symbol(self):
        state = MarketState(candle_interval=timedelta(minutes=1), candle_lookback=5)
        for minute in range(8):
            for symbol in ("EURUSD", "USDCHF"):
                state.update(tick(minute, symbol))
        for source, offset in (("bid", 0.0), ("ask", 0.2), ("mid", 0.1)):
            for symbol, limit in (("EURUSD", 2), ("USDCHF", 4)):
                with self.subTest(source=source, symbol=symbol):
                    candles = state.get_candles(symbol, price_source=source, limit=limit)
                    self.assertEqual([c.start_time for c in candles],
                                     [START + timedelta(minutes=m) for m in range(7 - limit, 7)])
                    for candle, minute in zip(candles, range(7 - limit, 7)):
                        self.assertAlmostEqual(candle.close_price, 100.0 + minute + offset)

    def test_market_state_keeps_registered_timeframes_in_parallel(self):
        state = MarketState(candle_interval=timedelta(minutes=10), candle_lookback=10)
        state.register_candle_interval(timedelta(minutes=5))
        state.register_candle_interval(timedelta(minutes=15))
        for minute in range(31):
            state.update(tick(minute))
        expected = {
            5: [0, 5, 10, 15, 20, 25],
            10: [0, 10, 20],
            15: [0, 15],
        }
        for minutes, starts in expected.items():
            with self.subTest(timeframe=minutes):
                candles = state.get_candles(
                    "EURUSD",
                    include_active=False,
                    candle_interval=timedelta(minutes=minutes),
                )
                self.assertEqual(
                    [candle.start_time for candle in candles],
                    [START + timedelta(minutes=value) for value in starts],
                )


if __name__ == "__main__":
    unittest.main()
