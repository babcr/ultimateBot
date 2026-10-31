import csv
import io
import json
import random
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.embeddings.trade_dataset import TradeDataset
from ultimateBot.backtester.embeddings.ts2vec_service import EmbeddingConfig, TS2VecService, candle_features
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeReport
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.export_trade_dataset import export_dataset
from ultimateBot.run_backtest import build_engine, parse_args
from ultimateBot.run_embedding_sweep import command_for, parse_args as sweep_args


START = datetime(2020, 1, 2, 8, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=1)
CONFIG = EmbeddingConfig(context_bars=8, output_dims=8, hidden_dims=4, depth=1, train_iters=2)


def candles(count=8):
    return [Candle(START + i * INTERVAL, 1 + i * .0001, 1.0003 + i * .0001,
                   .9998 + i * .0001, 1.0001 + i * .0001, tick_count=1)
            for i in range(count)]


class QuietRegression(SwingRegressionStrategy):
    def on_market_data(self, batch, context):
        return []


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Install requirements-embeddings.txt for TS2Vec integration tests")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.temporary.name) / "model.pt"
        cls.service = TS2VecService(cls.checkpoint, CONFIG, INTERVAL)
        np.random.seed(20)
        random.seed(20)
        torch.manual_seed(20)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        with redirect_stdout(io.StringIO()):
            cls.service.fit_histories({"EURUSD": candles(), "GBPUSD": candles()}, as_of=START + 8 * INTERVAL)
        cls.rng_preserved = (np.array_equal(numpy_state[1], np.random.get_state()[1])
                             and python_state == random.getstate()
                             and torch.equal(torch_state, torch.random.get_rng_state()))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_training_reload_is_finite_deterministic_and_does_not_change_rng(self):
        self.assertTrue(self.rng_preserved)
        before = self.service.embed_candles(candles(), as_of=START + 8 * INTERVAL)
        loaded = TS2VecService(self.checkpoint, CONFIG, INTERVAL)
        after = loaded.embed_candles(candles(), as_of=START + 8 * INTERVAL)
        self.assertEqual(before, after)
        self.assertEqual(len(before.values), 8)
        self.assertTrue(np.isfinite(before.values).all())
        self.assertFalse(loaded.model.net.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.model.net.parameters()))
        self.assertNotIn("models", sys.modules)

    def test_rejects_future_model_future_candles_and_incompatible_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            self.service.embed_candles(candles(6), as_of=START + 7 * INTERVAL)
        with self.assertRaisesRegex(ValueError, "future candle"):
            self.service.embed_candles(candles(10), as_of=START + 8 * INTERVAL)
        with self.assertRaisesRegex(ValueError, "output_dims"):
            TS2VecService(self.checkpoint, replace(CONFIG, output_dims=16), INTERVAL)
        with self.assertRaisesRegex(ValueError, "interval"):
            TS2VecService(self.checkpoint, CONFIG, 2 * INTERVAL)

    def test_preprocessing_is_price_scale_invariant_and_scaler_stays_frozen(self):
        data = candles()
        scaled = [replace(c, open_price=c.open_price * 100, high_price=c.high_price * 100,
                          low_price=c.low_price * 100, close_price=c.close_price * 100) for c in data]
        np.testing.assert_allclose(candle_features(data), candle_features(scaled), atol=1e-5)
        mean, scale = self.service.mean.copy(), self.service.scale.copy()
        self.service.embed_candles(candles(12), as_of=START + 12 * INTERVAL)
        np.testing.assert_array_equal(self.service.mean, mean)
        np.testing.assert_array_equal(self.service.scale, scale)

    def make_engine(self, directory, feed=()):
        options = ["run_backtest", "--strategy", "swing-regression", "--symbols", "EURUSD",
                   "--trade-database", str(Path(directory) / "trades.sqlite3"),
                   "--ts2vec-model", str(self.checkpoint), "--ts2vec-context-bars", "8",
                   "--ts2vec-output-dims", "8", "--ts2vec-hidden-dims", "4", "--ts2vec-depth", "1",
                   "--candle-interval", "1m", "--regression-lookback-bars", "3",
                   "--level-lookback-bars", "8", "--no-regression-global-timeframe-validation"]
        with patch.object(sys, "argv", options), redirect_stdout(io.StringIO()):
            args = parse_args()
            strategy = QuietRegression(regression_rules={"regression_lookback_bars": 3})
            if isinstance(feed, list):
                feed = [tick for tick in feed if tick.timestamp > START + 8 * INTERVAL]
            engine = build_engine(feed, args, strategy_override=strategy)
            for i in range(9):
                engine.on_tick(Tick(START + i * INTERVAL, "EURUSD", 1., 1.0001, 1., 1.))
        return engine

    def queue(self, engine, order_id=1, side=OrderSide.BUY, eligible_minute=7.5):
        order = OrderRequest(order_id, "EURUSD", START, START, START + eligible_minute * INTERVAL,
                             side, 1000, stop_loss=.999 if side == OrderSide.BUY else 1.5,
                             take_profit=1.001 if side == OrderSide.BUY else .5,
                             trailing_stop_activation_price=.999 if side == OrderSide.SELL else None,
                             trailing_stop_enabled=(side == OrderSide.SELL))
        engine.trade_closer.prepare_order(order)
        engine.trade_dataset.prepare_order(order)
        engine._queue_order(order, engine.market_state.get_tick(order.symbol))
        return order

    def test_fill_close_pipeline_persists_original_action_and_only_closed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            feed = [Tick(START + i * INTERVAL, "EURUSD", 1.0 if i < 10 else 1.002, 1.0001 if i < 10 else 1.0021, 1., 1.)
                    for i in range(12)]
            engine = self.make_engine(directory, feed)
            self.queue(engine)
            self.queue(engine, 2, OrderSide.SELL, 9.5)
            engine.strategy.set_history_lookback_bars(7, "EURUSD")
            with redirect_stdout(io.StringIO()):
                engine.run()
            with closing(sqlite3.connect(Path(directory) / "trades.sqlite3")) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute("SELECT * FROM closed_trades").fetchall()
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual((row["regression_lookback"], row["trailing"], row["side"]), (3, 0, 1))
                self.assertAlmostEqual(row["pnl"], 1.9)
                self.assertEqual(len(json.loads(row["embedding_json"])), 8)
                self.assertLessEqual(row["context_end"], row["opened_at"])
                self.assertEqual(row["context_bars"], 8)
                self.assertEqual(row["feature_point"], "order_creation")
                self.assertLess(row["feature_observed_at"], row["opened_at"])
                self.assertEqual(row["order_created_at"], row["feature_observed_at"])
                self.assertLess(row["signal_at"], row["order_created_at"])
                pending = json.loads(db.execute("SELECT snapshot_json FROM pending_trade_features").fetchone()[0])
                self.assertEqual((pending["side"], pending["trailing"]), (-1, 1))
                self.assertEqual(db.execute("SELECT status FROM runs").fetchone()[0], "completed")
                self.assertGreater(db.execute("SELECT COUNT(*) FROM run_parameters").fetchone()[0], 5)
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            output = Path(directory) / "trades.csv"
            self.assertEqual(export_dataset(Path(directory) / "trades.sqlite3", output), 1)
            with output.open(newline="", encoding="utf-8") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(exported[0]["regression_lookback"], "3")
            self.assertIn("embedding_007", exported[0])

    def test_partial_close_aggregation_and_duplicate_close_are_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory)
            self.queue(engine, side=OrderSide.SELL)
            with redirect_stdout(io.StringIO()):
                for i in (9,):
                    engine.on_tick(Tick(START + i * INTERVAL, "EURUSD", 1, 1.0001, 1., 1.))
            dataset = engine.trade_dataset
            stored = dataset.connection.execute("SELECT snapshot_json FROM pending_trade_features").fetchone()[0]
            position = next(iter(engine.portfolio.positions.values()))
            position.strategy_lookback_bars = 49
            report = TradeReport(position.trade_id, "EURUSD", OrderSide.SELL, position.opened_at,
                                 START + 10 * INTERVAL, 1, .99, 600, 6, 0, 0, CloseReason.TAKE_PROFIT,
                                 is_complete=False, event_id="partial")
            collector = ReportCollector()
            collector.add_trade_listener(dataset.on_trade_closed)
            collector.add_trade(report)
            self.assertEqual(dataset.connection.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0], 0)
            collector.add_trade(replace(report, quantity=400, pnl=-2, is_complete=True, event_id="final"))
            collector.add_trade(replace(report, quantity=400, pnl=-2, is_complete=True, event_id="final"))
            dataset.on_trade_closed(collector.trades[0])
            row = dataset.connection.execute("SELECT * FROM closed_trades").fetchone()
            self.assertEqual((row["pnl"], row["regression_lookback"], row["side"], row["trailing"]), (4, 3, -1, 1))
            self.assertEqual(row["embedding_json"], json.loads(stored)["embedding_json"])
            self.assertEqual(dataset.connection.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0], 1)
            self.assertEqual(dataset.connection.execute("SELECT COUNT(*) FROM pending_trade_features").fetchone()[0], 0)
            with redirect_stdout(io.StringIO()):
                dataset.finish()

    def test_failed_run_keeps_fill_durable_and_has_no_fabricated_label(self):
        with tempfile.TemporaryDirectory() as directory:
            def broken_feed():
                for i in (9,):
                    yield Tick(START + i * INTERVAL, "EURUSD", 1, 1.0001, 1., 1.)
                raise RuntimeError("broken tick source")
            engine = self.make_engine(directory, broken_feed())
            self.queue(engine)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "broken tick"):
                engine.run()
            with closing(sqlite3.connect(Path(directory) / "trades.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT status FROM runs").fetchone()[0], "failed")
                self.assertEqual(db.execute("SELECT COUNT(*) FROM pending_trade_features").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0], 0)

    def test_repeated_seeds_get_distinct_runs_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.sqlite3"
            with redirect_stdout(io.StringIO()):
                first = TradeDataset(path, self.service, parameters={}, seed=1)
                first.finish()
                second = TradeDataset(path, self.service, parameters={}, seed=1)
                second.finish()
            self.assertNotEqual(first.run_id, second.run_id)
            with self.assertRaises(sqlite3.IntegrityError):
                TradeDataset(path, self.service, parameters={}, seed=1, run_id=first.run_id)


class SweepTests(unittest.TestCase):
    def test_requested_configuration_is_preserved_and_seed_is_variable(self):
        args = sweep_args(["--seed-start", "1", "--seed-end", "10", "--passes", "2"])
        command = command_for(args, 7, "campaign", "run")
        with patch.object(sys, "argv", ["run_backtest", *command[4:]]):
            parsed = parse_args()
        self.assertEqual(parsed.symbols, ["USDCHF", "GBPUSD", "USDCAD", "EURUSD"])
        self.assertEqual((parsed.adaptive_seed, parsed.regression_lookback_bars, parsed.adaptive_lookback_max), (7, 15, 50))
        self.assertEqual((parsed.risk_percent, parsed.reward_risk_ratio, parsed.trailing_start_rr_ratio), (2.2, 3., 2.99))
        self.assertEqual((parsed.start_date.isoformat(), parsed.end_date.isoformat()), ("2020-01-01", "2023-12-31"))
        self.assertTrue(parsed.session_filter and parsed.trailing_stop and parsed.adaptive_strategy)
        self.assertFalse(parsed.adaptive_optimize_risk or parsed.adaptive_tighten_search_interval)
        self.assertFalse(parsed.regression_global_timeframe_validation or parsed.regression_level_reaction)
        self.assertEqual(parsed.adaptive_optimization_metric, "profit_factor")


if __name__ == "__main__":
    unittest.main()
