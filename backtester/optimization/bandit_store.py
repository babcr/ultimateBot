"""Transactional shared learning, deduplication and explicit memory administration."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from ultimateBot.backtester.optimization.bandit_core import (
    BanditAction, ClosedTradeObservation, DecisionSnapshot, FrozenScaler, LinUCBPolicy,
    MarketContext, canonical_json, stable_id,
)

SCHEMA_VERSION = 1


class LearningStorageError(RuntimeError):
    pass


def restore_rng(value):
    rng = Random()
    rng.setstate((int(value[0]), tuple(value[1]), value[2]))
    return rng


class LearningStore:
    def __init__(self, path, *, read_only=False, timeout=10.0):
        self.path = Path(path).resolve()
        self.read_only = read_only
        self.timeout = timeout
        self._readonly_groups = {}
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.connection(write=True) as db:
                    version = db.execute("PRAGMA user_version").fetchone()[0]
                    if version not in (0, SCHEMA_VERSION):
                        raise LearningStorageError(f"Unsupported learning schema {version}; migration required")
                    if version == 0 and db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone():
                        raise LearningStorageError("Existing unversioned memory requires an explicit migration")
                    db.executescript("""
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY, identity_json TEXT NOT NULL, state_json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS runs (
                        model_id TEXT NOT NULL, run_id TEXT NOT NULL, provenance_json TEXT NOT NULL,
                        state_json TEXT NOT NULL, PRIMARY KEY(model_id, run_id));
                    CREATE TABLE IF NOT EXISTS observations (
                        event_key TEXT PRIMARY KEY, model_id TEXT NOT NULL, scenario_key TEXT NOT NULL,
                        action_id TEXT NOT NULL, source TEXT NOT NULL, group_id TEXT, variant_id TEXT,
                        closed_at TEXT NOT NULL, payload_json TEXT NOT NULL, raw_r REAL, reward REAL,
                        assigned_weight REAL NOT NULL, weight REAL NOT NULL, learned INTEGER NOT NULL,
                        status TEXT NOT NULL);
                    CREATE INDEX IF NOT EXISTS observations_model ON observations(model_id);
                    CREATE INDEX IF NOT EXISTS observations_scenario ON observations(model_id, scenario_key, action_id);
                    CREATE TABLE IF NOT EXISTS decisions (
                        decision_key TEXT PRIMARY KEY, model_id TEXT NOT NULL, run_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL, payload_json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS groups (
                        group_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, payload_json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS deals (
                        event_key TEXT PRIMARY KEY, model_id TEXT NOT NULL, trajectory_key TEXT NOT NULL,
                        payload_json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS pending (
                        model_id TEXT NOT NULL, run_id TEXT NOT NULL, record_key TEXT NOT NULL,
                        kind TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY(model_id,run_id,record_key));
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        model_id TEXT NOT NULL, run_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                        PRIMARY KEY(model_id,run_id));
                    """)
                    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            except (sqlite3.Error, LearningStorageError) as error:
                self._fail(error)
        elif self.path.exists():
            with self.connection() as db:
                if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
                    self._fail(LearningStorageError("Read-only memory has an incompatible schema"))

    def _fail(self, error):
        message = {"error": str(error), "type": type(error).__name__, "database": str(self.path),
                   "recorded_at": datetime.now(timezone.utc).isoformat()}
        if not self.read_only:
            try:
                with self.path.with_suffix(self.path.suffix + ".errors.jsonl").open("a", encoding="utf-8") as file:
                    file.write(canonical_json(message) + "\n")
            except OSError:
                pass
        raise LearningStorageError(f"Learning storage unavailable; memory was not reset: {error}") from error

    @contextmanager
    def connection(self, write=False):
        if write and self.read_only:
            raise LearningStorageError("Cannot write a read-only learning memory")
        db = None
        try:
            if self.read_only:
                db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=self.timeout)
            else:
                db = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
            if not self.read_only:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write and db.in_transaction:
                db.commit()
        except sqlite3.Error as error:
            if db is not None and db.in_transaction:
                db.rollback()
            self._fail(error)
        except Exception:
            if db is not None and db.in_transaction:
                db.rollback()
            raise
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def initial_state(identity, space, config, timestamp):
        baseline = space.baseline
        actions = {action.action_id: action for action in [*space.actions, baseline]}
        rng = Random(config.seed ^ int(identity.model_id[:16], 16))
        decision = DecisionSnapshot(identity.model_id, baseline, 0, MarketContext(timestamp))
        return {"baseline": baseline.to_dict(), "actions": [action.to_dict() for action in actions.values()],
                "scaler": FrozenScaler(config.warmup, clip=config.feature_clip).to_dict(),
                "policy": LinUCBPolicy(config, actions.values()).to_dict(), "rng": rng.getstate(),
                "policy_version": 0, "decision": decision.to_dict(), "changes": 0}

    def load_or_create(self, identity, space, config, run_id, provenance, timestamp):
        fresh = self.initial_state(identity, space, config, timestamp)
        if self.read_only and not self.path.exists():
            return fresh, False
        with self.connection(write=not self.read_only) as db:
            row = db.execute("SELECT * FROM models WHERE model_id=?", (identity.model_id,)).fetchone()
            loaded = row is not None
            if row:
                if row["identity_json"] != canonical_json(identity.payload):
                    self._fail(LearningStorageError("Identity contract mismatch; explicit migration required"))
                # Validate the state even in read-only mode; no silent neutral prior reset.
                try:
                    state = json.loads(row["state_json"])
                    LinUCBPolicy(config, [BanditAction.from_dict(action) for action in state["actions"]], state["policy"])
                    FrozenScaler(config.warmup, state["scaler"])
                    if state["scaler"]["clip"] != config.feature_clip:
                        raise ValueError("Stored feature clipping contract changed")
                    expected = {*space.by_id, BanditAction.from_dict(state["baseline"]).action_id}
                    if expected != {action["action_id"] for action in state["actions"]}:
                        raise ValueError("Persisted model action domain changed")
                    DecisionSnapshot.from_dict(state["decision"])
                    restore_rng(state["rng"])
                except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                    self._fail(error)
            else:
                state = fresh
                if not self.read_only:
                    db.execute("INSERT INTO models VALUES (?,?,?)",
                               (identity.model_id, canonical_json(identity.payload), canonical_json(state)))
            if not self.read_only:
                db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?)",
                           (identity.model_id, run_id, canonical_json(provenance), canonical_json(state["decision"])))
            return state, loaded

    def run_decision(self, model_id, run_id):
        if not self.path.exists():
            return None
        with self.connection() as db:
            row = db.execute("SELECT state_json FROM runs WHERE model_id=? AND run_id=?", (model_id, run_id)).fetchone()
            return DecisionSnapshot.from_dict(json.loads(row[0])) if row else None

    def set_initial_context(self, model_id, run_id, decision):
        if self.read_only:
            return
        with self.connection(write=True) as db:
            row = db.execute("SELECT state_json FROM models WHERE model_id=?", (model_id,)).fetchone()
            state = json.loads(row[0])
            if state["policy_version"] == 0:
                state["decision"] = decision.to_dict()
                db.execute("UPDATE models SET state_json=? WHERE model_id=?", (canonical_json(state), model_id))
            db.execute("UPDATE runs SET state_json=? WHERE model_id=? AND run_id=?",
                       (canonical_json(decision.to_dict()), model_id, run_id))

    def register_group(self, model_id, group):
        if self.read_only:
            self._readonly_groups.setdefault(group["group_id"], dict(group))
            return
        with self.connection(write=True) as db:
            db.execute("INSERT OR IGNORE INTO groups VALUES (?,?,?)",
                       (group["group_id"], model_id, canonical_json(group)))

    def _shadow_weight(self, db, observation):
        record = observation.record
        row = db.execute("SELECT payload_json FROM groups WHERE group_id=?", (record.group_id,)).fetchone()
        if row is None:
            return 0.0, "shadow_group_missing"
        group = json.loads(row[0])
        if record.variant_id not in group["learners"]:
            return 0.0, "executed_action_copy"
        index = int(group["counts"].get(record.variant_id, 0)) + 1
        group["counts"][record.variant_id] = index
        budget = float(group["budget"])
        allocated = budget / len(group["learners"]) * 2.0 ** (-index)
        remaining = max(0.0, budget - float(group.get("spent", 0)))
        weight = min(allocated, remaining)
        group["spent"] = float(group.get("spent", 0)) + weight
        db.execute("UPDATE groups SET payload_json=? WHERE group_id=?", (canonical_json(group), record.group_id))
        return weight, None

    def _rebuild_policy(self, db, model_id, config, state, scaler):
        actions = [BanditAction.from_dict(value) for value in state["actions"]]
        policy = LinUCBPolicy(config, actions)
        rows = db.execute("SELECT * FROM observations WHERE model_id=? AND reward IS NOT NULL AND weight>0 ORDER BY event_key",
                          (model_id,)).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            context = MarketContext.from_dict(payload["record"]["decision"]["context"])
            if context.valid:
                policy.update(row["action_id"], scaler.transform(context), row["reward"], row["weight"])
                db.execute("UPDATE observations SET learned=1 WHERE event_key=?", (row["event_key"],))
        return policy

    @staticmethod
    def _select(model_id, space, config, state, context, available_bars, policy, scaler, valid_action_ids=None):
        previous = BanditAction.from_dict(state["decision"]["action"])
        rng = restore_rng(state["rng"])
        valid_actions = [action for action in space.actions if action.lookback <= available_bars
                         and (valid_action_ids is None or action.action_id in valid_action_ids)]
        reason = "linucb"
        mean = bonus = 0.0
        if not context.valid or not valid_actions:
            action = previous
            reason = "fallback_" + context.reason if not context.valid else "fallback_history_insufficient"
        elif not scaler.frozen:
            action = BanditAction.from_dict(state["baseline"])
            reason = "baseline_warmup"
        else:
            score = policy.select(scaler.transform(context), [action.action_id for action in valid_actions], rng)
            action = space.by_id[score["action_id"]]
            mean, bonus = score["mean"], score["bonus"]
        version = int(state["policy_version"]) + 1
        decision = DecisionSnapshot(model_id, action, version, context, mean, bonus, reason)
        state.update({"decision": decision.to_dict(), "policy_version": version,
                      "rng": rng.getstate(), "policy": policy.to_dict(), "scaler": scaler.to_dict(),
                      "changes": int(state["changes"]) + int(action.action_id != previous.action_id)})
        return decision

    def learn_batch(self, identity, space, config, run_id, observations, context, available_bars,
                    *, local_state=None, valid_action_ids=None):
        """Read latest shared knowledge and commit observations/model/decision together."""
        if self.read_only:
            state = json.loads(canonical_json(local_state))
            scaler = FrozenScaler(config.warmup, state["scaler"])
            policy = LinUCBPolicy(config, [BanditAction.from_dict(a) for a in state["actions"]], state["policy"])
            decision = self._select(identity.model_id, space, config, state, context, available_bars, policy, scaler, valid_action_ids)
            return state, decision, [{"event_key": obs.event_key, "status": "read_only", "weight": 0} for obs in observations]
        statuses = []
        with self.connection(write=True) as db:
            row = db.execute("SELECT state_json FROM models WHERE model_id=?", (identity.model_id,)).fetchone()
            if row is None:
                raise LearningStorageError("Model removed during run; explicit restart required")
            state = json.loads(row[0])
            scaler = FrozenScaler(config.warmup, state["scaler"])
            policy = LinUCBPolicy(config, [BanditAction.from_dict(a) for a in state["actions"]], state["policy"])
            inserted = []
            rebuild = False
            for obs in sorted(observations, key=lambda obs: (obs.closed_at, obs.event_key)):
                # The journal entry and the open-position entry are cleared
                # in the same transaction as learning, including replays.
                db.execute("DELETE FROM pending WHERE model_id=? AND run_id=? AND record_key=?",
                           (identity.model_id, obs.record.run_id, obs.event_key))
                if obs.record.provenance.get("broker_scope"):
                    db.execute("UPDATE pending SET kind='broker_completed' WHERE model_id=? AND run_id=? AND record_key=?",
                               (identity.model_id, obs.record.run_id, obs.record.trajectory_key))
                else:
                    db.execute("DELETE FROM pending WHERE model_id=? AND run_id=? AND record_key=?",
                               (identity.model_id, obs.record.run_id, obs.record.trajectory_key))
                if db.execute("SELECT 1 FROM observations WHERE event_key=?", (obs.event_key,)).fetchone():
                    statuses.append({"event_key": obs.event_key, "status": "duplicate", "weight": 0})
                    continue
                record = obs.record
                if record.decision.model_id != identity.model_id:
                    raise ValueError("Observation belongs to another model")
                raw, reward, invalid = obs.reward(config.reward_clip)
                assigned, diagnostic = (self._shadow_weight(db, obs) if record.source == "shadow" else (1.0, None))
                if record.source == "shadow" and db.execute(
                    "SELECT 1 FROM observations WHERE model_id=? AND scenario_key=? AND action_id=? AND source!='shadow'",
                    (identity.model_id, record.scenario_key, record.decision.action.action_id),
                ).fetchone():
                    diagnostic = "executed_action_copy"
                elif record.source != "shadow":
                    copies = db.execute(
                        "SELECT weight FROM observations WHERE model_id=? AND scenario_key=? AND action_id=? AND source='shadow'",
                        (identity.model_id, record.scenario_key, record.decision.action.action_id),
                    ).fetchall()
                    if copies:
                        rebuild |= any(row[0] > 0 for row in copies)
                        db.execute("UPDATE observations SET weight=0, learned=0, status='executed_action_copy' "
                                   "WHERE model_id=? AND scenario_key=? AND action_id=? AND source='shadow'",
                                   (identity.model_id, record.scenario_key, record.decision.action.action_id))
                if not record.decision.context.valid:
                    invalid = invalid or "selection_context_invalid"
                status = invalid or diagnostic or "eligible"
                repeated = config.repeat_weight if record.source in ("backtest_executed", "development_executed", "shadow") else None
                weight = assigned * (repeated or 1.0) if status == "eligible" else 0.0
                payload = obs.to_dict()
                db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    obs.event_key, identity.model_id, record.scenario_key, record.decision.action.action_id,
                    record.source, record.group_id, record.variant_id, obs.closed_at, canonical_json(payload),
                    raw, reward, assigned, weight, 0, status,
                ))
                inserted.append((obs, reward, weight))
                statuses.append({"event_key": obs.event_key, "status": status, "weight": weight,
                                 "assigned_weight": assigned, "raw_r": raw, "reward": reward,
                                 "source": record.source, "action_id": record.decision.action.action_id})
            if observations and not inserted:
                decision = DecisionSnapshot.from_dict((local_state or state)["decision"])
                return state, decision, statuses
            newly_frozen = scaler.collect(context)
            if scaler.frozen:
                if newly_frozen or rebuild:
                    policy = self._rebuild_policy(db, identity.model_id, config, state, scaler)
                else:
                    for obs, reward, weight in inserted:
                        if reward is not None and weight > 0:
                            policy.update(obs.record.decision.action.action_id, scaler.transform(obs.record.decision.context), reward, weight)
                            db.execute("UPDATE observations SET learned=1 WHERE event_key=?", (obs.event_key,))
            decision = self._select(identity.model_id, space, config, state, context, available_bars, policy, scaler, valid_action_ids)
            db.execute("UPDATE models SET state_json=? WHERE model_id=?", (canonical_json(state), identity.model_id))
            db.execute("UPDATE runs SET state_json=? WHERE model_id=? AND run_id=?",
                       (canonical_json(decision.to_dict()), identity.model_id, run_id))
            key = stable_id([identity.model_id, run_id, decision.policy_version])
            db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                       (key, identity.model_id, run_id, context.timestamp, canonical_json(decision.to_dict())))
            return state, decision, statuses

    def record_deal(self, model_id, trajectory_key, event_key, payload, *, final_factory=None):
        """Persist partials once; the final deal alone unlocks a full observation."""
        if self.read_only:
            return True, [payload], None
        with self.connection(write=True) as db:
            inserted = db.execute("INSERT OR IGNORE INTO deals VALUES (?,?,?,?)",
                                  (event_key, model_id, trajectory_key, canonical_json(payload))).rowcount > 0
            parts = [json.loads(row[0]) for row in db.execute(
                "SELECT payload_json FROM deals WHERE model_id=? AND trajectory_key=? ORDER BY event_key",
                (model_id, trajectory_key))]
            observation = final_factory(parts) if final_factory is not None else None
            if observation is not None:
                if db.execute("SELECT 1 FROM observations WHERE event_key=?", (observation.event_key,)).fetchone():
                    observation = None
                else:
                    db.execute("INSERT OR IGNORE INTO pending VALUES (?,?,?,?,?)", (
                        model_id, observation.record.run_id, observation.event_key, "closed_observation",
                        canonical_json(observation.to_dict()),
                    ))
            return inserted, parts, observation

    def closure_journal(self, run_id):
        if not self.path.exists():
            return []
        with self.connection() as db:
            return [ClosedTradeObservation.from_dict(json.loads(row[0])) for row in db.execute(
                "SELECT payload_json FROM pending WHERE run_id=? AND kind='closed_observation' ORDER BY record_key",
                (run_id,))]

    def pending_records(self, kind):
        if not self.path.exists():
            return []
        with self.connection() as db:
            return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in db.execute(
                "SELECT * FROM pending WHERE kind=? ORDER BY record_key", (kind,))]

    def observation(self, event_key):
        if not self.path.exists():
            return None
        with self.connection() as db:
            row = db.execute("SELECT payload_json FROM observations WHERE event_key=?", (event_key,)).fetchone()
            return ClosedTradeObservation.from_dict(json.loads(row[0])) if row else None

    def save_pending(self, model_id, run_id, key, kind, payload):
        if self.read_only:
            return
        with self.connection(write=True) as db:
            db.execute("INSERT OR REPLACE INTO pending VALUES (?,?,?,?,?)",
                       (model_id, run_id, key, kind, canonical_json(payload)))

    def remove_pending(self, model_id, run_id, key):
        if not self.read_only:
            with self.connection(write=True) as db:
                db.execute("DELETE FROM pending WHERE model_id=? AND run_id=? AND record_key=?", (model_id, run_id, key))

    def save_checkpoint(self, model_id, run_id, payload):
        if not self.read_only:
            with self.connection(write=True) as db:
                db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)", (model_id, run_id, canonical_json(payload)))

    def checkpoint(self, model_id, run_id):
        if not self.path.exists():
            return None
        with self.connection() as db:
            row = db.execute("SELECT payload_json FROM checkpoints WHERE model_id=? AND run_id=?", (model_id, run_id)).fetchone()
            return json.loads(row[0]) if row else None

    def latest_checkpoint(self, run_id):
        if not self.path.exists():
            return None
        with self.connection() as db:
            records = [json.loads(row[0]) for row in db.execute("SELECT payload_json FROM checkpoints WHERE run_id=?", (run_id,))]
            return max(records, key=lambda record: record["timestamp"]) if records else None

    def group(self, group_id):
        if self.read_only and group_id in self._readonly_groups:
            return self._readonly_groups[group_id]
        if not self.path.exists():
            return None
        with self.connection() as db:
            row = db.execute("SELECT payload_json FROM groups WHERE group_id=?", (group_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def summary(self, model_id):
        if not self.path.exists():
            return {"sources": {}, "latest_observation": None, "observations": 0}
        with self.connection() as db:
            sources = {row["source"]: {"observations": row["n"], "learned": row["learned"], "weight": row["weight"]}
                       for row in db.execute("SELECT source,COUNT(*) n,SUM(learned) learned,SUM(weight) weight "
                                             "FROM observations WHERE model_id=? GROUP BY source", (model_id,))}
            latest = db.execute("SELECT MAX(closed_at) FROM observations WHERE model_id=?", (model_id,)).fetchone()[0]
            return {"sources": sources, "latest_observation": latest,
                    "observations": sum(value["observations"] for value in sources.values())}

    def export(self, model_id=None):
        result = {"schema_version": SCHEMA_VERSION, "tables": {}}
        with self.connection() as db:
            for table in ("models", "runs", "observations", "decisions", "groups", "deals", "pending", "checkpoints"):
                query = f"SELECT * FROM {table}" + (" WHERE model_id=?" if model_id is not None else "")
                result["tables"][table] = [dict(row) for row in db.execute(query, (model_id,) if model_id else ())]
        return result

    def import_memory(self, payload):
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise LearningStorageError("Import requires matching schema or an explicit migration")
        with self.connection(write=True) as db:
            # Importing into an existing identity is deliberately explicit:
            # raw observations can be merged, frozen sufficient statistics cannot.
            for model in payload["tables"]["models"]:
                if stable_id(json.loads(model["identity_json"])) != model["model_id"]:
                    raise LearningStorageError("Imported identity hash mismatch")
                existing = db.execute("SELECT 1 FROM models WHERE model_id=?", (model["model_id"],)).fetchone()
                if existing:
                    raise LearningStorageError("Target model already exists; import into another memory or explicitly reset this model")
            for table in ("models", "runs", "observations", "decisions", "groups", "deals", "pending", "checkpoints"):
                for row in payload["tables"].get(table, []):
                    expected = [column["name"] for column in db.execute(f"PRAGMA table_info({table})")]
                    if set(row) != set(expected):
                        raise LearningStorageError(f"Invalid imported columns for {table}")
                    if row["model_id"] not in {model["model_id"] for model in payload["tables"]["models"]}:
                        raise LearningStorageError("Imported record belongs to a missing model")
                    columns = expected
                    db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                               [row[column] for column in columns])

    def reset_model(self, model_id):
        if not model_id or len(model_id) != 64:
            raise ValueError("Reset requires one complete model SHA256 identifier")
        with self.connection(write=True) as db:
            for table in ("runs", "observations", "decisions", "groups", "deals", "pending", "checkpoints", "models"):
                db.execute(f"DELETE FROM {table} WHERE model_id=?", (model_id,))
