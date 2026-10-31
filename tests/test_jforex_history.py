import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ultimateBot.backtester.other_tick_busters import import_jforex_history as history
from ultimateBot.backtester.other_tick_busters.download_dukascopy_ticks import (
    MonthlyQuoteWriter, _timestamp_ms,
)


DAY = datetime(2020, 1, 1, tzinfo=timezone.utc)
START = 1577836800000
STOP = START + 86_400_000 - 1


def frame(kind, timestamp, sequence, values="0\t0\t0\t0", session="session", symbol="EURUSD"):
    return f"{kind}\t{symbol}\t{timestamp}\t{session}\t{sequence}\t{values}\n"


def valid_day():
    return (frame("START", START, 0)
            + frame("QUOTE", START + 1, 1, "1.1\t1.2\t2\t3")
            + frame("STOP", STOP, 1))


class JForexHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = history._day_file(self.root, "EURUSD", DAY)
        self.path.parent.mkdir(parents=True)

    def quotes(self, content):
        self.path.write_text(content, encoding="utf-8")
        return list(history._iter_day_quotes(self.path, "EURUSD", DAY))

    def test_complete_day_and_bid_ask_volume_mapping(self):
        quotes = self.quotes(valid_day())
        self.assertEqual(len(quotes), 1)
        quote = quotes[0]
        self.assertEqual((quote.bid, quote.ask, quote.bid_size, quote.ask_size), (1.1, 1.2, 2, 3))
        self.assertEqual(quote.timestamp, DAY + timedelta(milliseconds=1))
        self.assertEqual(quote.sequence, 1)

    def test_empty_closed_market_day_is_complete(self):
        self.assertEqual(self.quotes(frame("START", START, 0) + frame("STOP", STOP, 0)), [])

    def test_multiple_ticks_at_same_millisecond_are_preserved(self):
        content = (frame("START", START, 0)
                   + frame("QUOTE", START + 1, 1, "1.1\t1.2\t2\t3")
                   + frame("QUOTE", START + 1, 2, "1.11\t1.21\t4\t5")
                   + frame("STOP", STOP, 2))
        quotes = self.quotes(content)
        self.assertEqual([q.sequence for q in quotes], [1, 2])
        self.assertEqual(quotes[0].timestamp, quotes[1].timestamp)

    def test_incomplete_or_inconsistent_files_are_rejected_with_location(self):
        start, quote, stop = valid_day().splitlines(keepends=True)
        cases = {
            "no start": quote + stop,
            "no stop": start + quote,
            "truncated final line": start + quote + stop.rstrip("\n"),
            "missing column": start + quote.rsplit("\t", 1)[0] + "\n" + stop,
            "missing quote": start + frame("QUOTE", START + 1, 2, "1.1\t1.2\t2\t3") + stop,
            "duplicate quote": start + quote + quote + stop,
            "changed session": start + quote.replace("session", "different") + stop,
            "absent session": start + quote.replace("session", "") + stop,
            "wrong symbol": start + quote.replace("EURUSD", "USDCHF") + stop,
            "second start": start + start + quote + stop,
            "after stop": start + quote + stop + quote,
            "wrong stop count": start + quote + frame("STOP", STOP, 2),
            "wrong stop time": start + quote + frame("STOP", STOP - 1, 1),
            "wrong start time": frame("START", START + 1, 0) + quote + stop,
            "quote outside day": start + frame("QUOTE", STOP + 1, 1, "1.1\t1.2\t2\t3") + stop,
            "control has prices": frame("START", START, 0, "1\t0\t0\t0") + quote + stop,
            "heartbeat": start + frame("HEARTBEAT", START + 1, 0) + quote + stop,
            "blank line": start + "\n" + quote + stop,
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, r"2020-01-01.tsv:\d+:"):
                    self.quotes(content)

    def test_empty_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "START/STOP"):
            self.quotes("")

    def test_invalid_prices_and_volumes_are_rejected(self):
        for values in ("0\t1.2\t2\t3", "1.2\t1.1\t2\t3", "NaN\t1.2\t2\t3",
                       "1.1\tInfinity\t2\t3", "1.1\t1.2\tNaN\t3", "1.1\t1.2\t2\t-1"):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.quotes(frame("START", START, 0) + frame("QUOTE", START, 1, values)
                                + frame("STOP", STOP, 1))

    def test_decreasing_timestamps_are_rejected(self):
        content = (frame("START", START, 0)
                   + frame("QUOTE", START + 2, 1, "1.1\t1.2\t2\t3")
                   + frame("QUOTE", START + 1, 2, "1.1\t1.2\t2\t3")
                   + frame("STOP", STOP, 2))
        with self.assertRaisesRegex(ValueError, "timestamps decroissants"):
            self.quotes(content)

    def test_source_month_detects_residual_partial_and_missing_days(self):
        end = DAY + timedelta(days=1)
        self.assertIn("jour source absent", history._validate_source_month(self.root, "EURUSD", DAY, end))
        self.path.write_text(valid_day(), encoding="utf-8")
        self.assertIsNone(history._validate_source_month(self.root, "EURUSD", DAY, end))
        self.path.with_suffix(".tsv.part").write_text("partial", encoding="utf-8")
        self.assertIn("temporaire residuel", history._validate_source_month(self.root, "EURUSD", DAY, end))

    def test_defaults_match_four_requested_symbols_and_java_source_directory(self):
        args = history.build_argument_parser().parse_args([])
        self.assertEqual(args.symbols, "USDCHF,GBPUSD,USDCAD,EURUSD")
        self.assertEqual(args.source_dir, Path(r"C:\jforex_history"))

    def test_validate_only_does_not_load_pyarrow_or_create_output(self):
        self.path.write_text(valid_day(), encoding="utf-8")
        output = self.root / "unused-output"
        argv = ["import_jforex_history.py", "--source-dir", str(self.root), "--symbols", "EURUSD",
                "--start-date", "2020-01-01", "--end-date", "2020-01-01",
                "--output-dir", str(output), "--validate-only"]
        with patch("sys.argv", argv), patch.object(history, "_load_parquet_modules") as load:
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertEqual(history.main(), 0)
            load.assert_not_called()
        self.assertIn("validated_days=1 validated_quotes=1", log.getvalue())
        self.assertFalse(output.exists())

    def test_invalid_day_aborts_month_instead_of_finalizing(self):
        self.path.write_text(valid_day().replace(frame("STOP", STOP, 1), ""), encoding="utf-8")
        with patch.object(history, "_load_parquet_modules", return_value=(None, None)):
            with patch.object(history, "MonthlyQuoteWriter") as writer_class:
                writer = writer_class.return_value
                writer.pending_rows.return_value = 0
                with self.assertRaisesRegex(ValueError, "START/STOP"):
                    history._write_month_from_exports(self.root, "EURUSD", DAY, DAY + timedelta(days=1),
                                                      self.root / "result.parquet", 100)
                writer.abort.assert_called_once()
                writer.finalize.assert_not_called()

    def test_all_millisecond_timestamps_survive_parquet_buffering(self):
        for base in (datetime(1970, 1, 1, tzinfo=timezone.utc), DAY,
                     datetime(2026, 10, 13, tzinfo=timezone.utc)):
            base_ms = _timestamp_ms(base)
            writer = MonthlyQuoteWriter(self.root / "result.parquet", "EURUSD", 100_000)
            for offset in range(1000):
                line = frame("QUOTE", base_ms + offset, offset + 1, "1.1\t1.2\t2\t3")
                _, quote = history._parse_export_line(line, "EURUSD")
                writer.append(quote)
            self.assertEqual(writer._timestamp_ms, list(range(base_ms, base_ms + 1000)))


if __name__ == "__main__":
    unittest.main()
