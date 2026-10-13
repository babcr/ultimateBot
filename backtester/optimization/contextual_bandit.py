"""Trade-close coordinator shared by replay and externally supplied live events."""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from uuid import uuid4

from ultimateBot.backtester.optimization.adaptive_strategy import RollingTradeProfitFactor
from ultimateBot.backtester.optimization.bandit_core import (
    ActionSpace, BanditAction, BanditConfig, ClosedTradeObservation, ContextBuilder, DecisionSnapshot,
    MarketContext, ModelIdentity, PositionLearningRecord, canonical_json, stable_id,
)
from ultimateBot.backtester.optimization.bandit_store import LearningStore, LearningStorageError
from ultimateBot.backtester.optimization.bandit_shadow import ShadowRunner
from ultimateBot.backtester.optimization.bandit_checkpoint import (
    encode, decode, engine_snapshot, restore_engine, market_state_snapshot, restore_market_state,
)

DEFAULT_MEMORY = Path(__file__).resolve().parents[2] / "artifacts" / "bandit_memory" / "learning.sqlite3"


class ContextualBanditOptimizer:
    def __init__(self, config: BanditConfig, *, memory=DEFAULT_MEMORY, read_only=False, store=None,
                 run_id=None, resume=False, provenance=None, strategy_factory=None,
                 source="backtest_executed", monetary_contract="price_delta_times_quantity_v1",
                 volume_contract="jforex_ITick_native_available_quote_units_v1"):
        if source not in ("backtest_executed", "development_executed", "demo_executed", "live_real"):
            raise ValueError("Explicit execution provenance is required")
        self.config = config
        self.store = store or LearningStore(memory, read_only=read_only)
        self.run_id = run_id or uuid4().hex
        self.resume = resume
        self.provenance = dict(provenance or {})
        self.source = source
        self.monetary_contract = monetary_contract
        self.volume_contract = volume_contract
        self.strategy_factory = strategy_factory
        self.context_builder = ContextBuilder(config.context_bars)
        self.engine = None
        self.models = {}
        self.pending_observations = []
        self.shadow = ShadowRunner(self)
        self.trace_digest = "00" * 32
        self.processed_ticks = 0
        self._resume_remaining = 0
        self._resume_trace = "00" * 32
        self.market_clock = None
        self.resume_timestamp = None
        self.last_context_bar = {}
        self._context_cache = {}
        self.dirty = False
        self.change_count = 0
        self.observation_statuses = []
        self.decisions = []
        self.duplicate_deals = 0
        self.main_pf = RollingTradeProfitFactor(config.profit_factor_window)
        self.peak_equity = 0.0
        self.drawdown = 0.0
        self.initial_actions = {}
        self._partial_readonly = {}
        self._started_symbols = set()
        self._diagnostic_keys = set()

    @property
    def required_history_capacity(self):
        largest = max(range(self.config.lookback_min, self.config.lookback_max, self.config.lookback_step))
        return max(self.config.context_bars, largest)

    def bind_engine(self, engine, symbols=()):
        self.engine = engine
        self.peak_equity = engine.portfolio.initial_balance
        self.static_rules = dict(engine.strategy.regression_rules)
        self.static_global_rules = dict(engine.strategy._global_regression_rules)
        self.baseline = BanditAction(engine.strategy.history_lookback_bars, engine.trade_closer.trailing_stop_enabled,
                                     float(self.static_rules["protective_stop_buffer_atr"]))
        self.space = ActionSpace(self.config, self.baseline.stop_extra_atr_multiplier, self.baseline)
        capacity = max(self.required_history_capacity, *(engine.strategy.history_capacity_for_lookback(action.lookback)
                                                        for action in self.space.actions))
        engine.market_state.ensure_history_capacity(capacity)
        fixed_rules = {key: value for key, value in self.static_rules.items()
                       if key not in ("regression_lookback_bars", "protective_stop_buffer_atr", "regression_live_analysis")}
        global_rules = {key: value for key, value in self.static_global_rules.items()
                        if key not in ("protective_stop_buffer_atr", "regression_live_analysis")}
        if not engine.strategy._global_confirmation_enabled():
            global_rules = {"enabled": False}
        self.fixed_parameters = {
            "strategy": fixed_rules, "global_confirmation": global_rules,
            "risk": asdict(engine.risk_manager.risk_config), "volatility": asdict(engine.risk_manager.volatility_config),
            "order_delay_seconds": engine.order_builder.execution_delay.total_seconds(),
            "order_expiration_bars": engine.order_builder.expiration_bars,
            "monetary_contract": self.monetary_contract,
            "volume_convention": "available quote volume, valid nonnegative samples, no executed volume",
            "volume_unit_contract": self.volume_contract,
        }
        engine.reports.add_trade_event_listener(self.queue_trade)
        for symbol in symbols:
            self.ensure_model(symbol, self.provenance.get("start_date") or "")
        if self.resume:
            saved = self.store.latest_checkpoint(self.run_id)
            if saved is None:
                raise LearningStorageError(f"No resumable checkpoint for run {self.run_id}")
            self.restore(saved)
        for observation in self.store.closure_journal(self.run_id):
            symbol = next((name for name, bundle in self.models.items()
                           if bundle["identity"].model_id == observation.record.decision.model_id), None)
            if symbol is not None:
                self.pending_observations.append((symbol, observation))

    def ensure_model(self, symbol, timestamp):
        if symbol in self.models:
            return self.models[symbol]
        fixed = dict(self.fixed_parameters)
        provider = self.engine.risk_manager.instrument_rules_provider
        if provider is not None:
            fixed["instrument_rules"] = asdict(provider(symbol))
        identity = ModelIdentity.create(symbol, self.engine.strategy.get_analysis_interval(symbol).total_seconds(),
                                        fixed, self.space.domain, self.config)
        state, loaded = self.store.load_or_create(identity, self.space, self.config, self.run_id,
                                                  {**self.provenance, "source": self.source}, timestamp)
        decision = self.store.run_decision(identity.model_id, self.run_id) or DecisionSnapshot.from_dict(state["decision"])
        bundle = {"identity": identity, "space": self.space, "state": state, "decision": decision, "loaded": loaded}
        self.models[symbol] = bundle
        self.initial_actions[symbol] = decision.action.to_dict()
        self.apply(symbol, decision)
        summary = self.store.summary(identity.model_id)
        bundle["prior_latest_observation"] = summary["latest_observation"]
        print(f"[bandit-memory] model_id={identity.model_id} symbol={symbol} loaded={loaded} "
              f"memory={self.store.path} read_only={self.store.read_only} observations={summary['observations']} "
              f"sources={canonical_json(summary['sources'])} action={canonical_json(decision.action.to_dict())}", flush=True)
        return bundle

    def apply(self, symbol, decision):
        strategy = self.engine.strategy
        strategy.set_history_lookback_bars(decision.action.lookback, symbol=symbol)
        strategy.set_protective_stop_buffer_atr(decision.action.stop_extra_atr_multiplier, symbol=symbol)

    def available_bars(self, symbol):
        return len(self.engine.market_state.get_candles(symbol, include_active=False, price_source="mid",
                                                       candle_interval=self.engine.strategy.get_analysis_interval(symbol)))

    def closed_bars(self, symbol):
        builder = self.engine.market_state._get_candle_builder("mid", self.engine.strategy.get_analysis_interval(symbol))
        return builder._candle_counts.get(symbol, 0)

    def valid_action_ids(self, symbol):
        provider = self.engine.risk_manager.instrument_rules_provider
        capacity = self.available_bars(symbol)
        actions = [action for action in self.space.actions if action.lookback <= capacity]
        if provider is None or provider(symbol).tick_size is None:
            return {action.action_id for action in actions}
        from math import floor
        from ultimateBot.backtester.strategy.structure_decision_engine import _rolling_atr
        terms = provider(symbol)
        candles = self.engine.market_state.get_candles(symbol, include_active=False, price_source="mid",
                                                       candle_interval=self.engine.strategy.get_analysis_interval(symbol))
        frame = self.engine.strategy._build_candle_frame(candles)
        atrs = {lookback: _rolling_atr(frame.iloc[-lookback:], self.engine.market_state.get_point_size(symbol), lookback)
                for lookback in {action.lookback for action in actions}}
        representatives = {}
        for action in sorted(actions, key=lambda action: action.stop_extra_atr_multiplier):
            steps = floor(action.stop_extra_atr_multiplier * atrs[action.lookback] / terms.tick_size + 1e-10)
            key = (action.lookback, action.trailing_enabled, steps)
            representatives.setdefault(key, action.action_id)
        return set(representatives.values())

    def context(self, symbol, timestamp=None):
        timestamp = timestamp or self.market_clock
        key = (symbol, self.closed_bars(symbol))
        cached = self._context_cache.get(symbol)
        if cached is None or cached[0] != key:
            cached = (key, self.context_builder.build(symbol, timestamp, self.engine.market_state, self.engine.strategy))
            self._context_cache[symbol] = cached
        return replace(cached[1], timestamp=timestamp.isoformat())

    @staticmethod
    def next_trace(previous, tick):
        volumes = [float(value) if value is not None else float("nan") for value in (tick.bid_volume, tick.ask_volume)]
        frame = struct.pack(">qdddd", int(tick.timestamp.timestamp() * 1_000_000), tick.bid, tick.ask, *volumes)
        return sha256(bytes.fromhex(previous) + tick.symbol.encode() + frame).hexdigest()

    def should_skip_replayed_tick(self, tick):
        if self._resume_remaining:
            self._resume_trace = self.next_trace(self._resume_trace, tick)
            self._resume_remaining -= 1
            if not self._resume_remaining and self._resume_trace != self.trace_digest:
                raise LearningStorageError("Resume market prefix differs from the checkpoint; use the original feed/date bounds")
            return True
        return False

    def on_market_tick(self, tick):
        self.market_clock = tick.timestamp
        self.ensure_model(tick.symbol, tick.timestamp.isoformat())
        if not self.provenance.get("start_date"):
            self.provenance["start_date"] = tick.timestamp.isoformat()
        self.trace_digest = self.next_trace(self.trace_digest, tick)
        self.processed_ticks += 1
        bar = self.closed_bars(tick.symbol)
        if self.last_context_bar.get(tick.symbol) != bar:
            self.last_context_bar[tick.symbol] = bar
            bundle = self.models[tick.symbol]
            context = self.context(tick.symbol)
            if context.valid and bundle["decision"].policy_version == 0 and not bundle["decision"].context.valid:
                # Establish the initial baseline snapshot once past data is
                # available. This changes no action and is not tick selection.
                decision = replace(bundle["decision"], context=context)
                bundle["decision"] = decision
                self.store.set_initial_context(bundle["identity"].model_id, self.run_id, decision)
                self.dirty = True
            if tick.symbol not in self._started_symbols and context.valid:
                self.shadow.start(tick.symbol, tick.timestamp, context)
                self._started_symbols.add(tick.symbol)

    def prepare_order(self, order, evaluation, signal, tick):
        bundle = self.ensure_model(order.symbol, tick.timestamp.isoformat())
        self.attach_order(order, evaluation, signal, tick, bundle["decision"], source=self.source)

    def attach_order(self, order, evaluation, signal, tick, snapshot, *, source, group_id=None, variant_id=None):
        context = self.context(order.symbol)
        analysis = self.engine.strategy.last_analysis.get(order.symbol, {})
        if source == "shadow":
            variants = [variant for variant in self.shadow.variants if variant.group["group_id"] == group_id and variant.variant_id == variant_id]
            if variants:
                analysis = variants[0].engine.strategy.last_analysis.get(order.symbol, {})
        atr = float(analysis.get("atr", 0.0))
        scenario = stable_id([snapshot.model_id, order.symbol, signal.timestamp.isoformat(),
                              order.side.name, snapshot.action.action_id, self.trace_digest])
        trajectory = stable_id([scenario, evaluation.entry_price, evaluation.stop_loss,
                                evaluation.quantity, source])
        order.trailing_stop_enabled = snapshot.action.trailing_enabled
        order.learning_record = PositionLearningRecord(
            snapshot, context, self.run_id, source, trajectory, scenario, atr, order.stop_loss,
            group_id=group_id, variant_id=variant_id,
            provenance={**self.provenance, "trace_at_entry": self.trace_digest,
                        "planned_entry": evaluation.entry_price, "planned_risk": evaluation.capital_at_risk,
                        "quote_spread": tick.spread, "costs": "existing engine: bid/ask spread, no commissions or funding",
                        "account_profit_multiplier": evaluation.profit_multiplier, "account_currency": evaluation.account_currency,
                        "decision_clock": self.market_clock.isoformat()},
        )

    def on_order_queued(self, order):
        record = order.learning_record
        if record is not None:
            self.store.save_pending(record.decision.model_id, self.run_id, record.trajectory_key, "pending_order", encode(order))
            self.dirty = True

    def on_position_open(self, position, tick):
        self.position_opened(position, tick)

    def position_opened(self, position, tick, *, persist=True):
        record = position.learning_record
        if record is None:
            return
        direction = 1.0 if position.side.name == "BUY" else -1.0
        risk = direction * (position.entry_price - record.initial_stop) * position.quantity * position.profit_multiplier
        position.learning_record = replace(record, initial_risk=risk if isfinite(risk) and risk > 0 else None,
                                           initial_quantity=position.quantity, entry_context=self.context(position.symbol, tick.timestamp),
                                           account_currency=position.account_currency)
        if persist:
            self.store.save_pending(record.decision.model_id, self.run_id, record.trajectory_key, "open_position", encode(position))
        self.dirty = True

    def queue_trade(self, report):
        record = report.learning_record
        if record is None:
            return
        deal_key = report.event_id or stable_id([record.decision.model_id, record.trajectory_key,
                                               report.closed_at.isoformat(), report.exit_price, report.quantity,
                                               report.pnl, report.is_complete, report.close_reason.name])
        if self.config.repeat_weight is not None and self.source in ("backtest_executed", "development_executed"):
            deal_key = stable_id([deal_key, "explicit_repeat", self.run_id])
        payload = {"pnl": report.pnl, "pnl_gross": report.pnl_gross if report.pnl_gross is not None else report.pnl,
                   "quantity": report.quantity, "event_key": deal_key, "is_complete": report.is_complete,
                   "closed_at": report.closed_at.isoformat(), "report": encode(report)}
        def finalize(parts):
            def number(value):
                return float(value["float"]) if isinstance(value, dict) and "float" in value else value
            return ClosedTradeObservation(
                stable_id([record.decision.model_id, record.trajectory_key, sorted(part["event_key"] for part in parts)]),
                record, report.closed_at.isoformat(), self.context(report.symbol, report.closed_at),
                sum(number(part["pnl"]) for part in parts), report.close_reason.name.lower(),
                pnl_gross=sum(number(part["pnl_gross"]) for part in parts),
                forced=report.close_reason.name == "END_OF_BACKTEST",
                broker_deal_ids=tuple(part["event_key"] for part in parts),
            )
        if self.store.read_only:
            bucket = self._partial_readonly.setdefault(record.trajectory_key, {})
            inserted = deal_key not in bucket
            bucket.setdefault(deal_key, payload)
            parts = list(bucket.values())
            observation = finalize(parts) if inserted and report.is_complete else None
        else:
            inserted, parts, observation = self.store.record_deal(
                record.decision.model_id, record.trajectory_key, deal_key, payload,
                final_factory=finalize if report.is_complete else None,
            )
        if not inserted:
            self.duplicate_deals += 1
            if report.is_complete:
                # A replay after a crash between commit and the next checkpoint
                # restores the already committed run policy, without relearning.
                decision = self.store.run_decision(record.decision.model_id, self.run_id)
                if decision is not None and record.source != "shadow":
                    self.models[report.symbol]["decision"] = decision
                    self.models[report.symbol]["state"]["decision"] = decision.to_dict()
                    self.apply(report.symbol, decision)
            if observation is None:
                if report.is_complete:
                    candidate = finalize(parts)
                    if candidate.event_key not in self._diagnostic_keys:
                        observation = self.store.observation(candidate.event_key)
                if observation is None:
                    return
        self.dirty = True
        if not report.is_complete:
            return
        if observation is None:
            return
        self.record_diagnostics(report.symbol, observation)
        if any(obs.event_key == observation.event_key for _, obs in self.pending_observations):
            return
        self.pending_observations.append((report.symbol, observation))

    def record_diagnostics(self, symbol, observation):
        if observation.event_key in self._diagnostic_keys:
            return
        self._diagnostic_keys.add(observation.event_key)
        if observation.record.source != "shadow" and isfinite(observation.pnl_net):
            self.main_pf.add(observation.pnl_net)
            self.shadow.main_closed(symbol)

    def flush_closures(self, timestamp):
        if not self.pending_observations:
            return
        batches = {}
        for symbol, observation in self.pending_observations:
            self.record_diagnostics(symbol, observation)
            batches.setdefault(symbol, []).append(observation)
        self.pending_observations = []
        for symbol in sorted(batches):
            bundle = self.models[symbol]
            context = self.context(symbol, timestamp)
            previous = bundle["decision"]
            state, decision, statuses = self.store.learn_batch(
                bundle["identity"], bundle["space"], self.config, self.run_id, batches[symbol], context,
                self.available_bars(symbol), local_state=bundle["state"],
                valid_action_ids=self.valid_action_ids(symbol),
            )
            by_event = {obs.event_key: obs for obs in batches[symbol]}
            for status in statuses:
                obs = by_event[status["event_key"]]
                status.update({"closed_at": obs.closed_at, "run_id": self.run_id,
                               "trajectory_key": obs.record.trajectory_key, "scenario_key": obs.record.scenario_key,
                               "group_id": obs.record.group_id, "variant_id": obs.record.variant_id,
                               "action": obs.record.decision.action.to_dict(), "pnl_net": obs.pnl_net,
                               "pnl_gross": obs.pnl_gross, "close_reason": obs.close_reason,
                               "initial_risk": obs.record.initial_risk, "account_currency": obs.record.account_currency,
                               "selection_policy_version": obs.record.decision.policy_version})
            bundle.update(state=state, decision=decision)
            self.change_count += int(previous.action.action_id != decision.action.action_id)
            self.apply(symbol, decision)
            self.observation_statuses.extend(statuses)
            self.decisions.append(decision.to_dict())
            for status in statuses:
                print(f"[bandit-close] run_id={self.run_id} model_id={decision.model_id} {canonical_json(status)}", flush=True)
            print(f"[bandit-decision] run_id={self.run_id} symbol={symbol} "
                  f"{canonical_json(decision.to_dict())}", flush=True)
            self.shadow.rotate(symbol, timestamp, context)
        self.dirty = True

    def after_tick(self, tick):
        equity = self.engine.portfolio.balance
        for position in self.engine.portfolio.positions.values():
            quote = self.engine.market_state.get_tick(position.symbol)
            if quote:
                equity += ((quote.bid - position.entry_price) if position.side.name == "BUY"
                           else (position.entry_price - quote.ask)) * position.quantity * position.profit_multiplier
        self.peak_equity = max(self.peak_equity, equity)
        self.drawdown = max(self.drawdown, self.peak_equity - equity)
        if self.dirty:
            self.save_checkpoint(tick.timestamp)
            self.dirty = False

    def save_checkpoint(self, timestamp):
        if not self.models:
            return
        payload = {"timestamp": timestamp.isoformat(), "trace_digest": self.trace_digest,
                   "processed_ticks": self.processed_ticks,
                   "main": engine_snapshot(self.engine), "market": market_state_snapshot(self.engine.market_state),
                   "shadow": self.shadow.checkpoint(), "last_context_bar": self.last_context_bar,
                   "decisions": {symbol: bundle["decision"].to_dict() for symbol, bundle in self.models.items()},
                   "model_ids": {symbol: bundle["identity"].model_id for symbol, bundle in self.models.items()},
                   "main_pf": list(self.main_pf.pnl_samples), "peak_equity": self.peak_equity,
                   "drawdown": self.drawdown, "change_count": self.change_count,
                   "started_symbols": sorted(self._started_symbols)}
        payload["diagnostic_keys"] = sorted(self._diagnostic_keys)
        for bundle in self.models.values():
            self.store.save_checkpoint(bundle["identity"].model_id, self.run_id, payload)

    def restore(self, payload):
        self.resume_timestamp = datetime.fromisoformat(payload["timestamp"])
        self.market_clock = self.resume_timestamp
        for symbol, model_id in payload["model_ids"].items():
            bundle = self.ensure_model(symbol, payload["timestamp"])
            if bundle["identity"].model_id != model_id:
                raise LearningStorageError("Resume fixed parameters/model identity changed")
        restore_market_state(self.engine.market_state, payload["market"])
        self._context_cache.clear()
        restore_engine(self.engine, payload["main"])
        self.trace_digest = payload["trace_digest"]
        self.processed_ticks = payload["processed_ticks"]
        self._resume_remaining = self.processed_ticks
        self.last_context_bar = payload["last_context_bar"]
        for symbol, value in payload["decisions"].items():
            self.models[symbol]["decision"] = DecisionSnapshot.from_dict(value)
            self.apply(symbol, self.models[symbol]["decision"])
        self.shadow.restore(payload["shadow"])
        self.main_pf.clear()
        for pnl in payload["main_pf"]:
            self.main_pf.add(pnl)
        self.peak_equity, self.drawdown, self.change_count = payload["peak_equity"], payload["drawdown"], payload["change_count"]
        self._started_symbols = set(payload["started_symbols"])
        self._diagnostic_keys = set(payload.get("diagnostic_keys", []))

    def finish(self):
        if self._resume_remaining:
            raise LearningStorageError("Resume feed ended before its saved checkpoint prefix")
        if self.market_clock is not None:
            self.flush_closures(self.market_clock)
            self.save_checkpoint(self.market_clock)

    def summary(self):
        models = {}
        start = self.provenance.get("start_date")
        for symbol, bundle in self.models.items():
            info = self.store.summary(bundle["identity"].model_id)
            latest = bundle["prior_latest_observation"]
            contamination = bool(start and latest and latest >= start)
            models[symbol] = {"model_id": bundle["identity"].model_id, "loaded": bundle["loaded"],
                              "initial_action": self.initial_actions[symbol], "final_action": bundle["decision"].action.to_dict(),
                              "memory": info, "knowledge_at_or_after_run_start": contamination,
                              "independent_validation": not contamination and self.store.read_only,
                              "scaler_frozen": bundle["state"]["scaler"]["frozen"],
                              "bounds": bundle["space"].domain}
        return {"run_id": self.run_id, "memory": str(self.store.path), "read_only": self.store.read_only,
                "models": models, "policy_changes": self.change_count, "duplicate_deals": self.duplicate_deals,
                "rolling_main_profit_factor": self.main_pf.value, "main_drawdown": self.drawdown,
                "main_pnl_net": self.engine.portfolio.balance - self.engine.portfolio.initial_balance,
                "main_closed_trades": len(self.engine.reports.trades),
                "shadow": [*self.shadow.archived, *(variant.summary() for variant in self.shadow.variants)],
                "shadow_limit": self.config.shadow_limit, "shadow_budget_per_group": self.config.shadow_budget,
                "shadow_groups": [self.store.group(group_id) or group for group_id, group in self.shadow.groups.items()],
                "observations_this_run": self.observation_statuses, "decisions": self.decisions,
                "provenance": {**self.provenance, "trace_digest": self.trace_digest, "source": self.source}}
