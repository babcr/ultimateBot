import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from random import Random
from unittest.mock import Mock, patch

import numpy as np

from ultimateBot.backtester.market.candle_builder import CandleBuilder
from ultimateBot.backtester.optimization.bandit_core import (
    ActionSpace, BanditAction, BanditConfig, ClosedTradeObservation, ContextBuilder,
    DecisionSnapshot, FrozenScaler, LinUCBPolicy, MarketContext, ModelIdentity,
    PositionLearningRecord, canonical_json,
)
from ultimateBot.backtester.optimization.bandit_store import LearningStore, LearningStorageError
from ultimateBot.backtester.optimization.bandit_broker import BrokerBanditBridge, BrokerDeal
from ultimateBot.backtester.optimization.contextual_bandit import ContextualBanditOptimizer
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.monitoring.close_request import CloseReason, CloseRequest
from ultimateBot.backtester.risk.instrument_rules import InstrumentRules
from ultimateBot.tests import test_stop_buffer
from ultimateBot.tests.test_swing_level_reaction import START, reaction_ticks
from ultimateBot.run_backtest import build_engine, parse_args


def context(index=0, slope=1.0):
    return MarketContext((START + timedelta(minutes=index)).isoformat(),
                         (slope, .01, 1., 2., 3., 4.), (True,) * 6, True, "ready", 10)


def args_for(memory=None, extra=()):
    arguments = ["run_backtest", "--symbols", "TEST", "--strategy", "swing-regression",
                 "--candle-interval", "1m", "--regression-lookback-bars", "10", "--level-lookback-bars", "10",
                 "--regression-global-timeframe-multiplier", "1", "--regression-min-market-volatility-ratio", "0",
                 "--regression-score-threshold", "0", "--regression-min-parallelism", "0",
                 "--regression-min-consecutive-swing-distance-atr", "0", "--no-regression-level-reaction",
                 "--reward-risk-ratio", ".5", "--trailing-start-rr-ratio", ".1"]
    if memory is not None:
        arguments += ["--contextual-bandit", "--bandit-memory", str(memory), "--bandit-context-bars", "10",
                      "--bandit-warmup", "2", "--bandit-shadow-limit", "0", "--adaptive-lookback-min", "5",
                      "--adaptive-lookback-max", "16"]
    with patch.object(sys, "argv", arguments + list(extra)):
        return parse_args()


class BanditCoreTests(unittest.TestCase):
    def test_effective_identity_and_isolation(self):
        config = BanditConfig(lookback_min=5, lookback_max=16, lookback_grid_size=9)
        baseline = BanditAction(10, False, .4)
        space = ActionSpace(config, .4, baseline)
        explicit = replace(config, lookback_grid=tuple(space.domain["lookbacks"]),
                           stop_grid=tuple(space.domain["stop_coefficients"]))
        other_space = ActionSpace(explicit, .4, baseline)
        a = ModelIdentity.create(" eurusd ", 60, {"risk": 1.0, "reaction": True}, space.domain, config)
        b = ModelIdentity.create("EURUSD", 60, {"reaction": True, "risk": 1}, other_space.domain, explicit)
        self.assertEqual(a.model_id, b.model_id)
        for symbol, seconds, fixed in (("BTCUSD", 60, a.payload["fixed"]), ("EURUSD", 300, a.payload["fixed"]),
                                        ("EURUSD", 60, {"risk": 2, "reaction": True})):
            self.assertNotEqual(a.model_id, ModelIdentity.create(symbol, seconds, fixed, space.domain, config).model_id)
        self.assertEqual(90, len(space.actions))
        self.assertEqual(5, min(action.lookback for action in space.actions))
        self.assertEqual(15, max(action.lookback for action in space.actions))
        self.assertIn(10, space.domain["lookbacks"])
        self.assertEqual(18, len(ActionSpace(config, 0., BanditAction(10, False, 0.)).actions))

    def test_grid_bounds_and_numeric_validation(self):
        for changes in ({"alpha": float("nan")}, {"regularization": 0}, {"context_bars": 0},
                        {"warmup": 0}, {"feature_clip": 0}, {"lookback_step": 0}):
            with self.assertRaises(ValueError):
                BanditConfig(**changes)
        with self.assertRaises(ValueError):
            ActionSpace(BanditConfig(lookback_grid=(1,)), .2, BanditAction(10, False, .2))
        with self.assertRaises(ValueError):
            ActionSpace(BanditConfig(stop_grid=(0., .3)), .2, BanditAction(10, False, .2))
        with self.assertRaises(ValueError):
            MarketContext("bad", values=(float("nan"),) * 6)

    def test_scaler_freezes_masks_clips_and_reload(self):
        scaler = FrozenScaler(2)
        first, second = context(1), context(2, 2)
        self.assertFalse(scaler.collect(first))
        self.assertFalse(scaler.collect(first))
        self.assertTrue(scaler.collect(second))
        old = scaler.to_dict()
        self.assertFalse(scaler.collect(context(3, 999)))
        self.assertEqual(old, scaler.to_dict())
        partial = replace(context(4, 1e9), available=(True, True, False, False, False, False))
        x = scaler.transform(partial)
        self.assertEqual(13, len(x))
        self.assertEqual(10., x[1])
        np.testing.assert_array_equal(x[3:7], np.zeros(4))
        np.testing.assert_array_equal(x, FrozenScaler(2, json.loads(canonical_json(old))).transform(partial))

    def test_linucb_learns_contextual_action_and_determinism(self):
        actions = [BanditAction(5, False, 0), BanditAction(6, False, 0)]
        def train():
            policy = LinUCBPolicy(BanditConfig(alpha=.3), actions)
            rng = Random(13)
            rewards, baseline_rewards, choices = [], [], []
            for index in range(500):
                direction = 1 if index % 2 == 0 else -1
                x = np.zeros(13); x[0] = 1.; x[1] = direction
                chosen = policy.select(x, [a.action_id for a in actions], rng)["action_id"]
                reward = direction if chosen == actions[0].action_id else -direction
                policy.update(chosen, x, reward, 1.)
                choices.append(chosen); rewards.append(reward); baseline_rewards.append(direction)
            return policy, choices, np.mean(rewards[-200:]), np.mean(baseline_rewards[-200:])
        first, choices, reward, baseline = train()
        second, choices2, _, _ = train()
        self.assertEqual(choices, choices2)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertGreater(reward, baseline + .8)


class BanditStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "learning.sqlite3"
        self.config = BanditConfig(lookback_min=5, lookback_max=8, context_bars=10, warmup=1)
        self.baseline = BanditAction(5, False, .2)
        self.space = ActionSpace(self.config, .2, self.baseline)
        self.identity = ModelIdentity.create("TEST", 60, {"risk": 1}, self.space.domain, self.config)
        self.store = LearningStore(self.path)
        self.state, _ = self.store.load_or_create(self.identity, self.space, self.config, "first", {}, context().timestamp)

    def tearDown(self):
        self.directory.cleanup()

    def observation(self, key="one", action=None, source="backtest_executed", risk=10., pnl=20., **changes):
        snapshot = DecisionSnapshot(self.identity.model_id, action or self.baseline, 0, context())
        record = PositionLearningRecord(snapshot, context(1, 2), "first", source, "trajectory-" + key,
                                        "scenario-" + key, 2., 99., initial_risk=risk, initial_quantity=1., **changes)
        return ClosedTradeObservation(key, record, context(2).timestamp, context(2, -10), pnl, "stop_loss", pnl_gross=pnl + 2)

    def learn(self, observations, store=None, run_id="first", close_context=None):
        return (store or self.store).learn_batch(self.identity, self.space, self.config, run_id,
                                                observations, close_context or context(2), 20)

    def test_reward_attribution_dedup_reload_and_rng(self):
        obs = self.observation()
        state, decision, statuses = self.learn([obs])
        self.assertEqual(.4, statuses[0]["reward"])
        self.assertEqual(2., statuses[0]["raw_r"])
        scaler = FrozenScaler(1, state["scaler"])
        x = scaler.transform(obs.record.decision.context)
        np.testing.assert_allclose(state["policy"][self.baseline.action_id]["A"], np.eye(13) + np.outer(x, x))
        np.testing.assert_allclose(state["policy"][self.baseline.action_id]["b"], .4 * x)
        old = self.store.export()
        _, _, duplicate = self.learn([obs])
        self.assertEqual("duplicate", duplicate[0]["status"])
        self.assertEqual(old, self.store.export())
        reloaded = LearningStore(self.path)
        state2, loaded = reloaded.load_or_create(self.identity, self.space, self.config, "second", {}, context().timestamp)
        self.assertTrue(loaded)
        self.assertEqual(canonical_json(state), canonical_json(state2))
        self.assertEqual(decision, reloaded.run_decision(self.identity.model_id, "second"))
        rng_a = Random(); rng_a.setstate((state["rng"][0], tuple(state["rng"][1]), state["rng"][2]))
        rng_b = Random(); rng_b.setstate((state2["rng"][0], tuple(state2["rng"][1]), state2["rng"][2]))
        self.assertEqual([rng_a.random() for _ in range(5)], [rng_b.random() for _ in range(5)])

    def test_invalid_rewards_forced_close_and_missing_context_audited(self):
        observations = [self.observation("zero", risk=0), self.observation("nan", pnl=float("nan")),
                        replace(self.observation("forced"), forced=True)]
        _, previous, statuses = self.learn(observations)
        self.assertEqual(0, sum(item["weight"] for item in statuses))
        self.assertEqual(3, self.store.summary(self.identity.model_id)["observations"])
        empty = MarketContext(context().timestamp)
        _, decision, _ = self.learn([self.observation("valid")], close_context=empty)
        self.assertEqual("fallback_history_insufficient", decision.reason)
        self.assertEqual(previous.action, decision.action)

    def test_shadow_lifetime_budget_and_executed_copy(self):
        arm = self.space.actions[0]
        group = {"group_id": "g", "learners": ["a", "b"], "counts": {}, "spent": 0., "budget": .25}
        self.store.register_group(self.identity.model_id, group)
        observations = [self.observation(f"shadow-{variant}-{i}", action=arm, source="shadow", group_id="g", variant_id=variant)
                        for variant in ("a", "b") for i in range(12)]
        _, _, statuses = self.learn(observations)
        self.assertLessEqual(sum(item["weight"] for item in statuses), .25)
        self.assertLessEqual(self.store.group("g")["spent"], .25)
        copied = self.observation("copy", action=arm, source="shadow", group_id="g", variant_id="active")
        _, _, copies = self.learn([copied])
        self.assertEqual("executed_action_copy", copies[0]["status"])
        executed = self.observation("executed", action=arm)
        shadow = replace(self.observation("same", action=arm, source="shadow", group_id="g", variant_id="a"),
                         record=replace(executed.record, source="shadow", group_id="g", variant_id="a"))
        self.learn([shadow]); self.learn([executed])
        rows = self.store.export()["tables"]["observations"]
        self.assertEqual(0, next(row for row in rows if row["event_key"] == "same")["weight"])

    def test_atomic_rollback_and_outbox_recovery(self):
        observation = self.observation()
        with patch.object(self.store, "_select", side_effect=RuntimeError("power failure")):
            with self.assertRaises(RuntimeError):
                self.learn([observation])
        self.assertEqual(0, self.store.summary(self.identity.model_id)["observations"])
        inserted, _, journaled = self.store.record_deal(self.identity.model_id, "trajectory", "deal", {"net": 20},
                                                       final_factory=lambda _: observation)
        self.assertTrue(inserted)
        self.assertEqual(observation, journaled)
        reopened = LearningStore(self.path)
        self.assertEqual([observation], reopened.closure_journal("first"))
        self.learn(reopened.closure_journal("first"), store=reopened)
        self.assertEqual([], reopened.closure_journal("first"))
        _, _, already_committed = reopened.record_deal(self.identity.model_id, "trajectory", "deal", {"net": 20},
                                                       final_factory=lambda _: observation)
        self.assertIsNone(already_committed)
        self.assertEqual(1, reopened.summary(self.identity.model_id)["observations"])

    def test_concurrent_writers_refresh_latest_state(self):
        stores = [LearningStore(self.path), LearningStore(self.path)]
        for index, store in enumerate(stores):
            store.load_or_create(self.identity, self.space, self.config, str(index), {}, context().timestamp)
        def worker(index):
            for number in range(5):
                self.learn([self.observation(f"process-{index}-{number}")], store=stores[index], run_id=str(index))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, range(2)))
        state, _ = self.store.load_or_create(self.identity, self.space, self.config, "verify", {}, "")
        self.assertEqual(10, state["policy"][self.baseline.action_id]["count"])
        self.assertEqual(10., state["policy"][self.baseline.action_id]["weight"])

    def test_separate_processes_share_atomic_learning(self):
        script = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from ultimateBot.tests.test_contextual_bandit import context
from ultimateBot.backtester.optimization.bandit_core import *
from ultimateBot.backtester.optimization.bandit_store import LearningStore
config = BanditConfig(lookback_min=5, lookback_max=8, context_bars=10, warmup=1)
baseline = BanditAction(5, False, .2)
space = ActionSpace(config, .2, baseline)
identity = ModelIdentity.create('TEST', 60, {'risk': 1}, space.domain, config)
store = LearningStore(Path(sys.argv[1]))
run = 'worker-' + sys.argv[2]
store.load_or_create(identity, space, config, run, {'worker': run}, '')
for index in range(4):
    key = run + '-' + str(index)
    decision = DecisionSnapshot(identity.model_id, baseline, 0, context())
    record = PositionLearningRecord(decision, context(), run, 'backtest_executed', key, key,
                                    1., 99., initial_risk=10., initial_quantity=1.)
    obs = ClosedTradeObservation(key, record, context(index + 2).timestamp, context(index + 2), 20., 'stop_loss')
    store.learn_batch(identity, space, config, run, [obs], context(index + 2), 20)
'''
        processes = [subprocess.Popen([sys.executable, "-c", script, str(self.path), str(index),
                                        str(Path(__file__).resolve().parents[2])], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True) for index in range(2)]
        for process in processes:
            _, errors = process.communicate(timeout=30)
            self.assertEqual(0, process.returncode, errors)
        state, _ = self.store.load_or_create(self.identity, self.space, self.config, "verify", {}, "")
        self.assertEqual(8, state["policy"][self.baseline.action_id]["count"])
        self.assertEqual(8, state["policy_version"])

    def test_simultaneous_batch_order_is_deterministic(self):
        second = LearningStore(Path(self.directory.name) / "second.sqlite3")
        second.load_or_create(self.identity, self.space, self.config, "first", {}, "")
        batch = [self.observation("b", pnl=-10.), self.observation("a", pnl=30.)]
        state, decision, _ = self.learn(batch)
        other_state, other_decision, _ = self.learn(list(reversed(batch)), store=second)
        self.assertEqual(canonical_json(state), canonical_json(other_state))
        self.assertEqual(decision, other_decision)
        self.assertEqual(1, state["policy_version"])

    def test_warmup_observations_replayed_when_scaler_freezes(self):
        self.config = replace(self.config, warmup=2)
        self.identity = ModelIdentity.create("TEST", 60, {"risk": 1}, self.space.domain, self.config)
        self.store.load_or_create(self.identity, self.space, self.config, "first", {}, "")
        state, decision, _ = self.learn([self.observation("a")], close_context=context(2, 2))
        self.assertFalse(state["scaler"]["frozen"])
        self.assertEqual(0, state["policy"][self.baseline.action_id]["count"])
        self.assertEqual(self.baseline, decision.action)
        state, _, _ = self.learn([self.observation("b")], close_context=context(3, 3))
        self.assertTrue(state["scaler"]["frozen"])
        self.assertEqual(2, state["policy"][self.baseline.action_id]["count"])
        self.assertEqual(2.5, state["scaler"]["mean"][0])

    def test_schema_and_lock_failure_do_not_reset_memory(self):
        wrong_path = Path(self.directory.name) / "wrong.sqlite3"
        with closing(sqlite3.connect(wrong_path)) as db:
            db.execute("PRAGMA user_version=99")
        with self.assertRaises(LearningStorageError):
            LearningStore(wrong_path)
        with closing(sqlite3.connect(wrong_path)) as db:
            self.assertEqual(99, db.execute("PRAGMA user_version").fetchone()[0])
        with self.store.connection(write=True):
            with self.assertRaises(LearningStorageError):
                LearningStore(self.path, timeout=.01)
        self.assertEqual(1, len(self.store.export()["tables"]["models"]))

    def test_read_only_export_import_reset_and_corruption(self):
        self.learn([self.observation()])
        exported = self.store.export()
        readonly = LearningStore(self.path, read_only=True)
        state, _ = readonly.load_or_create(self.identity, self.space, self.config, "readonly", {}, "")
        readonly.learn_batch(self.identity, self.space, self.config, "readonly", [self.observation("new")],
                            context(3), 20, local_state=state)
        self.assertEqual(exported, self.store.export())
        imported = LearningStore(Path(self.directory.name) / "copy.sqlite3")
        imported.import_memory(exported)
        self.assertEqual(exported, imported.export())
        imported.reset_model(self.identity.model_id)
        self.assertEqual([], imported.export()["tables"]["models"])
        with self.store.connection(write=True) as db:
            db.execute("UPDATE models SET state_json='{' WHERE model_id=?", (self.identity.model_id,))
        with self.assertRaises(LearningStorageError):
            self.store.load_or_create(self.identity, self.space, self.config, "bad", {}, "")
        self.assertTrue(self.path.with_suffix(".sqlite3.errors.jsonl").exists())


class BanditIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "learning.sqlite3"
        self.quiet = redirect_stdout(io.StringIO()); self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.directory.cleanup()

    def engine(self, count=45, extra=(), memory=True):
        return build_engine(reaction_ticks(count), args_for(self.path if memory else None, extra))

    def test_no_selection_without_closure_and_positions_keep_snapshot(self):
        engine = self.engine()
        runtime = engine.contextual_optimizer
        for tick in reaction_ticks(10):
            engine.on_tick(tick)
        self.assertEqual(0, runtime.models["TEST"]["decision"].policy_version)
        engine.run()
        self.assertGreater(len(engine.reports.trades), 0)
        for report in engine.reports.trades:
            self.assertEqual(report.trailing_stop_enabled, report.learning_record.decision.action.trailing_enabled)
            self.assertEqual(0, report.learning_record.decision.context.bars % 10)
        for position in engine.portfolio.positions.values():
            self.assertEqual(position.trailing_stop_enabled, position.learning_record.decision.action.trailing_enabled)

    def test_partials_unique_final_observation_and_final_risk(self):
        engine = self.engine()
        runtime = engine.contextual_optimizer
        for tick in engine.feed:
            engine.on_tick(tick)
            if engine.portfolio.positions:
                break
        position = next(iter(engine.portfolio.positions.values()))
        record = position.learning_record
        half = position.quantity / 2
        time = runtime.market_clock + timedelta(seconds=1)
        first = engine.trade_closer.close(engine.portfolio, CloseRequest(position.trade_id, time, position.entry_price + .1,
                                                                       CloseReason.TAKE_PROFIT, half, "partial-a"))
        runtime.queue_trade(first); runtime.queue_trade(first)
        self.assertEqual([], runtime.pending_observations)
        final = engine.trade_closer.close(engine.portfolio, CloseRequest(position.trade_id, time, position.entry_price + .2,
                                                                       CloseReason.TAKE_PROFIT, half, "partial-b"))
        runtime.queue_trade(final); runtime.queue_trade(final)
        self.assertEqual(1, len(runtime.pending_observations))
        observation = runtime.pending_observations[0][1]
        self.assertAlmostEqual(first.pnl + final.pnl, observation.pnl_net)
        self.assertEqual(record.initial_risk, observation.record.initial_risk)
        runtime.flush_closures(time)
        self.assertEqual(1, runtime.store.summary(record.decision.model_id)["observations"])

    def test_closed_features_no_future_missing_volumes(self):
        engine = self.engine(count=14)
        for tick in engine.feed:
            engine.market_state.update(replace(tick, bid_volume=None, ask_volume=float("nan")))
        runtime = engine.contextual_optimizer
        snapshot = runtime.context("TEST", engine.feed[-1].timestamp)
        self.assertTrue(snapshot.valid)
        self.assertEqual((False,) * 4, snapshot.available[2:])
        last = engine.feed[-1]
        engine.market_state.update(replace(last, timestamp=last.timestamp + timedelta(seconds=1), bid=1e9, ask=1e9 + 1,
                                           bid_volume=1e20, ask_volume=1e20))
        after = runtime.context("TEST", engine.feed[-1].timestamp)
        self.assertEqual(snapshot.values, after.values)
        self.assertEqual(snapshot.available, after.available)

    def test_resume_matches_continuous_and_reload_flags_training(self):
        engine = self.engine(count=45, extra=("--bandit-run-id", "original"))
        ticks = engine.feed
        for tick in ticks[:105]:
            engine.on_tick(tick)
        runtime = engine.contextual_optimizer
        runtime.save_checkpoint(ticks[104].timestamp)
        exported = runtime.store.export()
        copy_path = Path(self.directory.name) / "continuous.sqlite3"
        LearningStore(copy_path).import_memory(exported)
        resumed = self.engine(count=45, extra=("--bandit-run-id", "original", "--bandit-resume"))
        engine.contextual_optimizer.store = LearningStore(copy_path)
        engine.feed = ticks[105:]
        engine.run(); resumed.run()
        self.assertEqual(engine.portfolio.balance, resumed.portfolio.balance)
        self.assertEqual(len(engine.reports.trades), len(resumed.reports.trades))
        self.assertEqual(engine.contextual_optimizer.models["TEST"]["state"], resumed.contextual_optimizer.models["TEST"]["state"])
        again = self.engine(count=45)
        self.assertTrue(again.contextual_optimizer.models["TEST"]["loaded"])
        self.assertTrue(again.contextual_optimizer.summary()["models"]["TEST"]["knowledge_at_or_after_run_start"])
        self.assertEqual(resumed.contextual_optimizer.models["TEST"]["decision"].action,
                         again.contextual_optimizer.models["TEST"]["decision"].action)

    def test_shadow_virtual_executor_own_strategies_and_six_limit(self):
        engine = self.engine(count=35, extra=("--bandit-shadow-limit", "6", "--bandit-group-max-bars", "5"))
        spy = Mock(); spy.try_fill.side_effect = AssertionError("real execution path")
        engine.execution_model = spy
        runtime = engine.contextual_optimizer
        for tick in engine.feed:
            engine.market_state.update(tick)
            runtime.on_market_tick(tick)
            engine.risk_manager.on_tick(tick)
            runtime.shadow.on_tick(tick)
            runtime.flush_closures(tick.timestamp)
            self.assertLessEqual(len(runtime.shadow.variants), 6)
        spy.assert_not_called()
        self.assertEqual(0, spy.try_fill.call_count)
        shadow_reports = runtime.summary()["shadow"]
        self.assertGreater(sum(item["closed_trades"] for item in shadow_reports), 0)
        self.assertGreater(len({item["action"]["lookback"] for item in shadow_reports}), 1)
        for variant in runtime.shadow.variants:
            self.assertIs(engine.market_state, variant.engine.market_state)
            self.assertIsNot(engine.strategy, variant.engine.strategy)
            self.assertIsNot(engine.portfolio, variant.engine.portfolio)
            self.assertIsNot(spy, variant.engine.execution_model)
            self.assertEqual(variant.action.lookback, variant.engine.strategy.rules_for_symbol("TEST")["regression_lookback_bars"])
        for group in runtime.summary()["shadow_groups"]:
            self.assertLessEqual(group["spent"], .25)

    def test_bandit_disabled_preserves_reference_and_no_memory_access(self):
        with patch("ultimateBot.backtester.optimization.contextual_bandit.LearningStore", side_effect=AssertionError("memory access")):
            first = self.engine(memory=False); second = self.engine(memory=False)
            first.run(); second.run()
        self.assertIsNone(first.contextual_optimizer)
        self.assertEqual(first.portfolio.balance, second.portfolio.balance)
        self.assertEqual(first.reports.trades, second.reports.trades)
        self.assertFalse(self.path.exists())

    def test_idle_market_no_fake_reward_or_per_tick_learning(self):
        engine = self.engine(count=25)
        runtime = engine.contextual_optimizer
        with patch.object(runtime.store, "learn_batch", side_effect=AssertionError("per-tick learning")):
            for tick in engine.feed:
                engine.on_tick(replace(tick, bid=100., ask=100.01, bid_volume=None, ask_volume=None))
        self.assertEqual(0, runtime.store.summary(runtime.models["TEST"]["identity"].model_id)["observations"])
        self.assertEqual(0, runtime.models["TEST"]["decision"].policy_version)

    def test_replay_prefix_mismatch_is_explicit(self):
        engine = self.engine(count=25, extra=("--bandit-run-id", "same"))
        for tick in engine.feed[:30]:
            engine.on_tick(tick)
        engine.contextual_optimizer.save_checkpoint(engine.feed[29].timestamp)
        resumed = self.engine(count=25, extra=("--bandit-run-id", "same", "--bandit-resume"))
        changed = [replace(engine.feed[0], bid=engine.feed[0].bid + .01), *engine.feed[1:30]]
        with self.assertRaises(LearningStorageError):
            for tick in changed:
                resumed.on_tick(tick)

    def test_modes_share_identity_and_effective_stop_arms_are_coalesced(self):
        engines, runtimes = [], []
        for source in ("backtest_executed", "demo_executed", "live_real", "development_executed"):
            engine = self.engine(count=20, memory=False)
            engine.risk_manager.instrument_rules_provider = lambda _: InstrumentRules(tick_size=1000.)
            runtime = ContextualBanditOptimizer(BanditConfig(lookback_min=5, lookback_max=16, context_bars=10,
                                                             warmup=2, shadow_limit=0), memory=self.path, source=source,
                                               strategy_factory=lambda: engine.strategy)
            runtime.bind_engine(engine, ["TEST"])
            engines.append(engine); runtimes.append(runtime)
        self.assertEqual(1, len({runtime.models["TEST"]["identity"].model_id for runtime in runtimes}))
        self.assertTrue(all(runtime.models["TEST"]["loaded"] for runtime in runtimes[1:]))
        engine, runtime = engines[0], runtimes[0]
        for tick in engine.feed:
            engine.market_state.update(tick)
        valid = runtime.valid_action_ids("TEST")
        self.assertEqual(18, len(valid))
        self.assertTrue(all(runtime.space.by_id[action].stop_extra_atr_multiplier == 0 for action in valid))

    def test_crash_after_journaling_preserves_one_closure_and_pf(self):
        engine = self.engine(count=35, extra=("--bandit-run-id", "crash"))
        runtime = engine.contextual_optimizer
        for index, tick in enumerate(engine.feed):
            engine.on_tick(tick)
            if engine.portfolio.positions:
                break
        runtime.save_checkpoint(tick.timestamp)
        position = next(iter(engine.portfolio.positions.values()))
        closing_tick = replace(engine.feed[index + 1], bid=position.stop_loss - 1., ask=position.stop_loss - .99)
        with patch.object(runtime.store, "learn_batch", side_effect=RuntimeError("power failure before learning")):
            with self.assertRaises(RuntimeError):
                engine.on_tick(closing_tick)
        resumed = self.engine(count=35, extra=("--bandit-run-id", "crash", "--bandit-resume"))
        for prior in engine.feed[:index + 1]:
            resumed.on_tick(prior)
        resumed.on_tick(closing_tick)
        info = resumed.contextual_optimizer.store.summary(runtime.models["TEST"]["identity"].model_id)
        self.assertEqual(1, info["observations"])
        self.assertEqual(1, len(resumed.contextual_optimizer.main_pf.pnl_samples))
        self.assertEqual([], runtime.store.closure_journal("crash"))

    def test_broker_net_deals_currency_reconciliation_and_replay(self):
        engine = self.engine(count=15, memory=False)
        config = BanditConfig(lookback_min=5, lookback_max=16, warmup=1, context_bars=10, shadow_limit=0)
        runtime = ContextualBanditOptimizer(config, memory=self.path, source="demo_executed", run_id="demo",
                                             strategy_factory=lambda: engine.strategy, monetary_contract="USD_per_price_quantity_v1")
        engine.contextual_optimizer = runtime; runtime.bind_engine(engine, ["TEST"])
        bridge = BrokerBanditBridge(runtime, account_scope="broker:account")
        for tick in engine.feed:
            bridge.on_market_tick(tick)
        decision = runtime.models["TEST"]["decision"]
        record = PositionLearningRecord(decision, context(), "demo", "demo_executed", "ignored", "broker-scenario",
                                         1., 99., account_currency="USD")
        frozen = bridge.register_position("p1", record, symbol="TEST", side=OrderSide.BUY, entry_price=100.,
                                           opened_at=START, initial_risk_account_currency=10., quantity=2., account_currency="USD")
        self.assertEqual("broker_open", bridge.reconcile_positions(["p1"])["p1"])
        self.assertEqual("awaiting_authoritative_close_deals", bridge.reconcile_positions([])["p1"])
        stamp = engine.feed[-1].timestamp
        first = BrokerDeal("p1", "a", stamp, 1., 101., 8., "USD", False, pnl_gross_account_currency=10.)
        final = BrokerDeal("p1", "b", stamp, 1., 102., 18., "USD", True, pnl_gross_account_currency=20.)
        bridge.process_deals([final, first], timestamp=stamp)
        rows = runtime.store.export()["tables"]["observations"]
        self.assertEqual(1, len(rows)); self.assertEqual(2.6, rows[0]["raw_r"])
        self.assertEqual(26., json.loads(rows[0]["payload_json"])["pnl_net"])
        self.assertEqual(30., json.loads(rows[0]["payload_json"])["pnl_gross"])
        bridge.process_deals([first, final], timestamp=stamp)
        restarted = BrokerBanditBridge(runtime, account_scope="broker:account")
        restarted.process_deals([first, final], timestamp=stamp)
        self.assertEqual(1, runtime.store.summary(frozen.decision.model_id)["observations"])
        with self.assertRaises(ValueError):
            bridge.process_deals([replace(final, account_currency="EUR")], timestamp=stamp)


class InstrumentIntegrationTests(unittest.TestCase):
    def test_ticks_extra_bound_both_directions_and_quantity(self):
        terms = InstrumentRules(tick_size=.1, minimum_stop_distance=.3, quantity_step=.25, minimum_quantity=.5)
        for buy in (False, True):
            base = terms.stop_distance(100.03, .2, 0., buy)
            padded = terms.stop_distance(100.03, .2, .26, buy)
            self.assertAlmostEqual(.2, padded - base)
            self.assertLessEqual(padded - base, .26)
            stop = 100.03 - padded if buy else 100.03 + padded
            self.assertAlmostEqual(round(stop / .1), stop / .1)
        self.assertEqual(.75, terms.quantity(.99))
        self.assertEqual(0., terms.quantity(.99, fixed=True))
        self.assertEqual(0., terms.quantity(.3))

    def test_currency_conversion_resizes_risk_without_exceeding_budget(self):
        helper = test_stop_buffer.StopBufferTests()
        manager, tick = helper.manager_and_tick()
        from ultimateBot.backtester.strategy.signal import Signal, SignalSide
        from ultimateBot.backtester.portfolio.portfolio import Portfolio
        signal = Signal("TEST", SignalSide.BUY, tick.timestamp, protective_level_price=98., protective_level_quality=1.)
        native = manager.evaluate(signal, Portfolio(1000.), tick)
        manager.instrument_rules_provider = lambda _: InstrumentRules(account_profit_multiplier=2., account_currency="USD")
        converted = manager.evaluate(signal, Portfolio(1000.), tick)
        self.assertAlmostEqual(native.quantity / 2., converted.quantity)
        self.assertAlmostEqual(1., converted.capital_at_risk)
        self.assertEqual("USD", converted.account_currency)


if __name__ == "__main__":
    unittest.main()
