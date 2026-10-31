import io
import json
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from ultimateBot.audit_breakout_frequency import aggregate_quotes, build_audit_engine, timestamp
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import CandleBuilder
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.replay_breakout_signals import replay


START = datetime(2024, 1, 2, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=30)


class AggregationTests(unittest.TestCase):
    def test_quotes_match_all_three_candle_builders_including_mid_extrema_and_volumes(self):
        quotes = [(0, 100, 112, 1, 2), (10, 109, 109.2, -1, 3),
                  (20, 106, 114, 4, -1), (1800, 100, 100.1, 2, 2)]
        frame = pd.DataFrame([dict(timestamp=int(START.timestamp() * 1000) + sec * 1000,
                                   bid=bid, ask=ask, bid_volume=bv, ask_volume=av)
                              for sec, bid, ask, bv, av in quotes])
        rows = aggregate_quotes(frame, 1_800_000)
        for source in ("bid", "ask", "mid"):
            builder = CandleBuilder(INTERVAL, 150, source)
            for row in frame.itertuples():
                builder.update(Tick(timestamp(row.timestamp), "EURUSD", row.bid, row.ask, row.bid_volume, row.ask_volume))
            builder.finalize_active_candles()
            for cached, actual in zip(rows, builder.get_candles("EURUSD")):
                self.assertEqual(cached[source], [actual.open_price, actual.high_price, actual.low_price, actual.close_price])
                self.assertEqual(cached["count"], actual.tick_count)
                self.assertEqual(cached["volumes"], [actual.bid_volume_sum, actual.ask_volume_sum,
                                                     actual.bid_volume_count, actual.ask_volume_count])
        self.assertEqual(rows[0]["mid"][1], 110)
        self.assertNotEqual(rows[0]["mid"][1], (rows[0]["bid"][1] + rows[0]["ask"][1]) / 2)

    def test_nonchronological_or_nonfinite_quotes_are_rejected(self):
        frame = pd.DataFrame(dict(timestamp=[2, 1], bid=[1., 1.], ask=[1.1, 1.1],
                                  bid_volume=[1., 1.], ask_volume=[1., 1.]))
        with self.assertRaises(ValueError):
            aggregate_quotes(frame, 1000)
        frame.timestamp = [1, 2]
        frame.loc[0, "bid"] = float("inf")
        with self.assertRaises(ValueError):
            aggregate_quotes(frame, 1000)


class SignalReplayTests(unittest.TestCase):
    def compare_with_full_tick_engine(self, spread, expire=False, signal_index=15):
        ticks = []
        for index in range(24):
            prices = (1.1, 1.1004 if index < 18 else 1.102, 1.1002)
            if expire and index in (16, 17):
                prices = (1.1004, 1.1008, 1.1006)
            for minutes, bid in zip((0, 5, 20), prices):
                ticks.append(Tick(START + index * INTERVAL + timedelta(minutes=minutes),
                                  "EURUSD", bid, bid + spread, 1, 2))
        frame = pd.DataFrame([dict(timestamp=round(tick.timestamp.timestamp() * 1000),
                                   bid=tick.bid, ask=tick.ask, bid_volume=tick.bid_volume, ask_volume=tick.ask_volume)
                              for tick in ticks])
        candles = aggregate_quotes(frame, 1_800_000)
        context = {"entry_mode": "breakout-retest", "confirmation_mode": "confirmed", "stop_loss": 1.0997,
                   "minimum_entry_price": 1., "maximum_entry_price": 1.2}
        signal = Signal("EURUSD", ticks[signal_index * 3 + 2].timestamp, SignalSide.BUY, stop_loss_price=1.0997,
                        minimum_entry_price=1, maximum_entry_price=1.2, entry_context=context)
        pending = {signal.timestamp: signal}
        full = build_audit_engine("EURUSD", 30, 3, 0)
        full.strategy = SimpleNamespace(on_market_data=lambda batch, context: (
            [pending.pop(batch.ticks[-1].timestamp)] if batch.ticks[-1].timestamp in pending else []))
        full.feed = ticks
        with redirect_stdout(io.StringIO()):
            full.run()
        audited = {"max_bars": 3, "confirmation_max_bars": 0, "diagnostics": {},
                   "signals": [{"timestamp": signal.timestamp.isoformat(), "side": "BUY", "context": context}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "EURUSD/2024/01/2024-01-02.tsv"
            target.parent.mkdir(parents=True)
            target.write_text("".join(f"QUOTE\tEURUSD\t{round(t.timestamp.timestamp()*1000)}\ts\t{i}\t{t.bid!r}\t{t.ask!r}\t1\t2\n"
                                      for i, t in enumerate(ticks)), encoding="utf-8")
            sparse = replay(root, "EURUSD", date(2024, 1, 2), date(2024, 1, 2), 30,
                            candles, audited, root / "replay.log")
        self.assertEqual(sparse["closed_trades"], len(full.reports.trades))
        self.assertAlmostEqual(sparse["return_percent"], (full.portfolio.balance / full.portfolio.initial_balance - 1) * 100)
        self.assertEqual(sparse["open_positions"], len(full.portfolio.positions))
        json.dumps(sparse)
        for actual, recorded in zip(full.reports.trades, sparse["trades"]):
            self.assertEqual(recorded["opened_at"], actual.opened_at.isoformat())
            self.assertEqual(recorded["closed_at"], actual.closed_at.isoformat())
            self.assertAlmostEqual(recorded["pnl_r"], actual.pnl / (abs(actual.entry_price - 1.0997) * actual.quantity))
        return sparse

    def test_market_fill_price_gap_and_tp_match_full_replay(self):
        report = self.compare_with_full_tick_engine(.000002)
        self.assertEqual(report["closed_trades"], 1)
        # Signal is decided on the next candle's first quote; it fills on a later quote.
        self.assertEqual(report["trades"][0]["opened_at"], (START + 16 * INTERVAL + timedelta(minutes=5)).isoformat())

    def test_limit_fallback_and_waiting_order_match_full_replay(self):
        report = self.compare_with_full_tick_engine(.00002)
        self.assertEqual(report["closed_trades"], 1)
        self.assertEqual(report["trades"][0]["opened_at"], (START + 17 * INTERVAL).isoformat())

    def test_expired_limit_does_not_fill_on_a_later_price(self):
        report = self.compare_with_full_tick_engine(.00002, expire=True)
        self.assertEqual(report["closed_trades"], 0)
        self.assertEqual(report["pending_orders"], 0)

    def test_a_signal_from_final_flush_cannot_fill_without_a_future_tick(self):
        report = self.compare_with_full_tick_engine(.000002, signal_index=23)
        self.assertEqual(report["closed_trades"], 0)
        self.assertEqual(report["pending_orders"], 1)


if __name__ == "__main__":
    unittest.main()
