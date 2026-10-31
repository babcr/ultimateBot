import csv
import io
import json
import shutil
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
from ultimateBot.backtester.embeddings.trade_dataset import SCHEMA, dumps, ensure_capture_schema
from ultimateBot.backtester.embeddings.ts2vec_service import EmbeddingConfig, EmbeddingSnapshot, TS2VecService
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.ml.train_validator import (
    TrainingConfig, chronological_split, observation_weights, read_snapshot, select_threshold, train_validator,
)
from ultimateBot.backtester.ml.trade_validator import TradeValidator
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide, PendingOrderProcessor, PendingOrderQueue
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, parse_args


BASE = datetime(2020, 1, 2, 8, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=1)
ENCODER_CONFIG = EmbeddingConfig(context_bars=8, output_dims=8, hidden_dims=4, depth=1, train_iters=2)
TRAINING_CONFIG = TrainingConfig(n_estimators=80, early_stopping_rounds=10)


class QuietRegression(SwingRegressionStrategy):
    def on_market_data(self, batch, context):
        return []


class TradeValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import xgboost
            import sklearn
        except ImportError:
            raise unittest.SkipTest("Install requirements-validator.txt to run XGBoost tests")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.checkpoint = cls.directory / "encoder.pt"
        encoder = TS2VecService(cls.checkpoint, ENCODER_CONFIG, INTERVAL)
        candles = [Candle(BASE + i * INTERVAL, 1, 1.0003 + i * .0001, .9999, 1.0001 + i * .0001)
                   for i in range(8)]
        with redirect_stdout(io.StringIO()):
            encoder.fit_histories({"EURUSD": candles}, as_of=BASE + 8 * INTERVAL)
        cls.encoder_id = encoder.model_id
        cls.database = cls.directory / "trades.sqlite3"
        rng = np.random.RandomState(1)
        with closing(sqlite3.connect(cls.database)) as db:
            db.executescript(SCHEMA)
            with db:
                db.execute("INSERT INTO runs(run_id,adaptive_seed,started_at,status,model_id) VALUES(?,?,?,?,?)",
                           ("run-1", 1, BASE.isoformat(), "running", cls.encoder_id))
                db.execute("INSERT INTO embedding_models VALUES(?,?,?)", (cls.encoder_id, str(cls.checkpoint), dumps(encoder.metadata)))
                for index in range(240):
                    embedding = rng.normal(size=8)
                    pnl_r = 2.0 if embedding[0] > 0 else -1.0
                    opened = BASE + timedelta(days=1 + index // 6, hours=index % 6)
                    record = {
                        "run_id": "run-1", "trade_id": index + 1, "model_id": cls.encoder_id,
                        "symbol": "EURUSD", "opened_at": opened.isoformat(),
                        "closed_at": (opened + timedelta(minutes=45)).isoformat(),
                        "context_start": (opened - 8 * INTERVAL).isoformat(), "context_end": opened.isoformat(),
                        "context_bars": 8, "embedding_json": dumps(embedding.tolist()), "embedding_dims": 8,
                        "regression_lookback": 5 + index % 10, "trailing": index % 2,
                        "side": 1 if index % 2 else -1, "strategy_configuration_id": 0,
                        "timeframe_minutes": 1, "global_validation": 0,
                        "entry_price": 1., "initial_stop_loss": .99, "initial_take_profit": 1.02,
                        "trailing_activation_price": None, "quantity": 100, "profit_multiplier": 1.,
                        "account_currency": "engine_price_quantity", "initial_risk": 1.,
                        "exit_price": 1 + pnl_r / 100, "pnl": pnl_r, "pnl_r": pnl_r,
                        "won": int(pnl_r > 0), "close_reason": "TAKE_PROFIT" if pnl_r > 0 else "STOP_LOSS",
                        "feature_point": "order_creation", "order_id": index + 1,
                        "order_created_at": opened.isoformat(), "feature_observed_at": opened.isoformat(),
                        "signal_at": opened.isoformat(),
                    }
                    db.execute(f"INSERT INTO closed_trades({','.join(record)}) VALUES({','.join('?' for _ in record)})",
                               tuple(record.values()))
        cls.model_dir, cls.metadata, cls.report = train_validator(
            cls.database, config=TRAINING_CONFIG, output_dir=cls.directory / "validator")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_xgboost_training_reload_and_feature_contract(self):
        self.assertGreater(self.report["partitions"]["test"]["roc_auc"], .9)
        self.assertEqual(len(self.metadata["feature_names"]), 11)
        validator = TradeValidator(self.model_dir)
        records, _, _ = read_snapshot(self.database)
        row = records[-1]
        snapshot = EmbeddingSnapshot(self.encoder_id, tuple(json.loads(row["embedding_json"])),
                                     row["context_start"], row["context_end"], 8)
        decision = validator.evaluate(snapshot, lookback=row["regression_lookback"], trailing=row["trailing"],
                                      side=row["side"], symbol="EURUSD", as_of=datetime(2020, 4, 1, tzinfo=timezone.utc))
        with (self.model_dir / "predictions.csv").open(newline="", encoding="utf-8") as source:
            predictions = list(csv.DictReader(source))
        self.assertAlmostEqual(decision.probability, float(predictions[-1]["score"]), places=6)
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM trade_validator_models").fetchone()[0], 1)

    def test_runtime_rejects_future_labels_wrong_encoder_and_modified_weights(self):
        validator = TradeValidator(self.model_dir)
        snapshot = EmbeddingSnapshot(self.encoder_id, (0.,) * 8, BASE.isoformat(), (BASE + INTERVAL).isoformat(), 8)
        options = dict(lookback=10, trailing=0, side=1, symbol="EURUSD",
                       as_of=datetime(2020, 4, 1, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            validator.evaluate(snapshot, **{**options, "as_of": datetime.fromisoformat(self.metadata["available_after"])})
        with self.assertRaisesRegex(ValueError, "same TS2Vec"):
            validator.evaluate(replace(snapshot, model_id="other"), **options)
        unseen = validator.evaluate(snapshot, **{**options, "symbol": "GBPUSD"})
        self.assertFalse(unseen.accepted)
        self.assertEqual(unseen.reason, "unseen_symbol")
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "broken"
            shutil.copytree(self.model_dir, bad)
            (bad / "model.ubj").write_bytes(b"bad model")
            with self.assertRaisesRegex(ValueError, "checksum"):
                TradeValidator(bad)

    def test_split_keeps_seeds_on_same_days_and_purges_overlapping_labels(self):
        records, _, _ = read_snapshot(self.database)
        splits, info = chronological_split(records, TRAINING_CONFIG)
        crossing = dict(splits["train"][-1])
        crossing["closed_at"] = splits["validation"][0]["closed_at"]
        replicated = [{**r, "run_id": "another-seed"} for r in records]
        with_crossing = [crossing if r["trade_id"] == crossing["trade_id"] else r for r in records]
        result, _ = chronological_split(with_crossing + replicated, TRAINING_CONFIG)
        self.assertNotIn(crossing, result["train"])
        for left, right in (("train", "validation"), ("validation", "test")):
            self.assertLess(max(r["closed_at"] for r in result[left]), min(r["context_start"] for r in result[right]))
        weights = observation_weights([records[0], {**records[0], "run_id": "seed2"}, records[1]])
        np.testing.assert_allclose(weights, [.5, .5, 1.])

    def test_test_labels_do_not_change_trained_model_or_selected_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "trades.sqlite3"
            shutil.copyfile(self.database, copied)
            _, info = chronological_split(read_snapshot(copied)[0], TRAINING_CONFIG)
            with closing(sqlite3.connect(copied)) as db:
                with db:
                    db.execute("UPDATE closed_trades SET pnl_r=-pnl_r,pnl=-pnl WHERE opened_at>=?", (info["test_start_day"],))
            _, metadata, report = train_validator(copied, config=TRAINING_CONFIG, output_dir=Path(directory) / "other")
            self.assertEqual(metadata["model_sha256"], self.metadata["model_sha256"])
            self.assertEqual(metadata["threshold"], self.metadata["threshold"])
            self.assertNotEqual(report["partitions"]["test"]["roc_auc"], self.report["partitions"]["test"]["roc_auc"])

    def test_completed_only_and_insufficient_data_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "No closed order-creation"):
            read_snapshot(self.database, completed_only=True)
        with self.assertRaisesRegex(ValueError, "Insufficient train"):
            chronological_split(read_snapshot(self.database)[0], replace(TRAINING_CONFIG, min_train_samples=1000))

    def test_legacy_fill_migration_keeps_rows_and_excludes_them_from_training(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "legacy.sqlite3"
            shutil.copyfile(self.database, copied)
            with closing(sqlite3.connect(copied)) as db:
                with db:
                    for column in ("feature_point", "order_id", "order_created_at", "signal_at", "feature_observed_at"):
                        db.execute(f"ALTER TABLE closed_trades DROP COLUMN {column}")
                ensure_capture_schema(db)
                self.assertEqual(db.execute("SELECT count(*) FROM closed_trades WHERE feature_point='fill'").fetchone()[0], 240)
            with self.assertRaisesRegex(ValueError, "legacy fill rows are excluded"):
                read_snapshot(copied)

    def make_engine(self, directory, threshold, capture=True):
        options = ["run_backtest", "--strategy", "swing-regression", "--symbols", "EURUSD",
                   "--trade-validator", str(self.model_dir), "--trade-validator-threshold", str(threshold),
                   "--trade-validator-log", str(Path(directory) / "decisions.jsonl"),
                   "--trade-database", str(Path(directory) / "filtered.sqlite3"),
                   "--ts2vec-model", str(self.checkpoint), "--ts2vec-context-bars", "8",
                   "--ts2vec-output-dims", "8", "--ts2vec-hidden-dims", "4", "--ts2vec-depth", "1",
                   "--candle-interval", "1m", "--regression-lookback-bars", "3", "--level-lookback-bars", "8"]
        if not capture:
            index = options.index("--trade-database")
            del options[index:index + 2]
        future = datetime(2020, 4, 1, 8, tzinfo=timezone.utc)
        feed = [Tick(future + i * INTERVAL, "EURUSD", 1.0 if i < 9 else 1.002,
                     1.0001 if i < 9 else 1.0021, 1., 1.) for i in range(11)]
        with patch.object(sys, "argv", options), redirect_stdout(io.StringIO()):
            engine = build_engine(feed[9:], parse_args(), strategy_override=QuietRegression(regression_rules={"regression_lookback_bars": 3}))
            for tick in feed[:9]:
                engine.on_tick(tick)
        order = OrderRequest(1, "EURUSD", future, future, future + 7.5 * INTERVAL,
                             OrderSide.BUY, 1000, stop_loss=.999, take_profit=1.001)
        engine.trade_closer.prepare_order(order)
        engine.trade_validator.prepare_order(order)
        with redirect_stdout(io.StringIO()):
            engine._queue_order(order, engine.market_state.get_tick(order.symbol))
        engine.strategy.set_history_lookback_bars(49, "EURUSD")
        return engine

    def test_rejection_happens_before_order_submission_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, 1.0)
            self.assertEqual(len(engine.pending_orders_processor.queue), 0)
            with redirect_stdout(io.StringIO()):
                engine.run()
            self.assertEqual(len(engine.portfolio.positions), 0)
            self.assertEqual(len(engine.reports.trades), 0)
            self.assertEqual(engine.portfolio._next_trade_id, 1)
            self.assertEqual(engine.trade_validator.stats, {"accepted": 0, "rejected": 1})
            with closing(sqlite3.connect(Path(directory) / "filtered.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM pending_trade_features").fetchone()[0], 0)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0], 0)
            audit = [json.loads(line) for line in (Path(directory) / "decisions.jsonl").read_text().splitlines()]
            self.assertEqual(audit[0]["regression_lookback"], 3)

    def test_acceptance_uses_creation_snapshot_and_persists_it_unchanged_at_close(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, 0.)
            self.assertIs(engine.trade_validator.encoder, engine.trade_dataset.encoder)
            stored = engine.trade_dataset.connection.execute("SELECT snapshot_json FROM order_feature_snapshots").fetchone()[0]
            with redirect_stdout(io.StringIO()), patch.object(engine.trade_validator.encoder, "embed_market", side_effect=AssertionError("Fill must not recompute embedding")):
                engine.run()
            self.assertEqual(len(engine.reports.trades), 1)
            self.assertEqual(engine.trade_validator.stats, {"accepted": 1, "rejected": 0})
            with closing(sqlite3.connect(Path(directory) / "filtered.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT regression_lookback FROM closed_trades").fetchone()[0], 3)
                self.assertEqual(db.execute("SELECT embedding_json FROM closed_trades").fetchone()[0], json.loads(stored)["embedding_json"])

    def test_order_validation_does_not_require_a_capture_database(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.make_engine(directory, 0., capture=False)
            self.assertIsNone(engine.trade_dataset)
            self.assertEqual(engine.trade_validator.stats, {"accepted": 1, "rejected": 0})
            with redirect_stdout(io.StringIO()):
                engine.run()
            self.assertEqual(len(engine.reports.trades), 1)


class PlatformFillTests(unittest.TestCase):
    def test_platform_fill_processor_has_no_model_callback(self):
        queue = PendingOrderQueue()
        for identifier in (1, 2):
            queue.add(OrderRequest(identifier, "EURUSD", BASE, BASE, BASE, OrderSide.BUY, 1))
        processor = PendingOrderProcessor(queue, ExecutionModel())
        tick = Tick(BASE + INTERVAL, "EURUSD", 1, 1.0001, 1., 1.)
        results = processor.on_tick(tick, max_fills=1)
        self.assertEqual([result.order.order_id for result in results], [1])
        self.assertEqual(len(queue), 1)


if __name__ == "__main__":
    unittest.main()
