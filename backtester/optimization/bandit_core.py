"""Finite contextual actions, immutable attribution and a disjoint LinUCB model."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from hashlib import sha256
from math import isfinite, log1p, sqrt
from random import Random
from typing import Any

import numpy as np

from ultimateBot.backtester.optimization.adaptive_strategy import AdaptiveStrategyConfig

FEATURE_VERSION = "quote-volumes-closed-window-atr-slope-v1"
ACTION_VERSION = "lookback-trailing-extra-atr-v1"
REWARD_VERSION = "net-initial-risk-clipped-v1"
BEHAVIOR_VERSION = "swing-regression-low-ask-buy-high-bid-sell-v3"
FEATURE_DIMENSION = 13


def canonical_json(value: Any) -> str:
    def normalize(item):
        if isinstance(item, timedelta):
            return {"seconds": normalize(item.total_seconds())}
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if isinstance(item, Enum):
            return {"enum": type(item).__name__, "name": item.name}
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, dict):
            return {str(key): normalize(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        if isinstance(item, float):
            if not isfinite(item):
                return {"float": "inf" if item > 0 else "-inf" if item < 0 else "nan"}
            return int(item) if item.is_integer() else item
        return item
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_id(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BanditConfig:
    lookback_min: int = AdaptiveStrategyConfig().lookback_min
    lookback_max: int = AdaptiveStrategyConfig().lookback_max
    lookback_step: int = AdaptiveStrategyConfig().lookback_step
    lookback_grid: tuple[int, ...] = ()
    lookback_grid_size: int = 9
    stop_grid: tuple[float, ...] = ()
    stop_grid_size: int = 5
    alpha: float = 1.0
    regularization: float = 1.0
    seed: int = 42
    warmup: int = 20
    context_bars: int = 30
    reward_clip: float = 5.0
    feature_clip: float = 10.0
    shadow_limit: int = 6
    shadow_budget: float = 0.25
    profit_factor_window: int = AdaptiveStrategyConfig().profit_factor_window
    group_main_trades: int = 0
    group_max_bars: int = 0
    repeat_weight: float | None = None

    def __post_init__(self):
        if not 2 <= self.lookback_min < self.lookback_max or self.lookback_step < 1:
            raise ValueError("Bandit lookback uses the adaptive domain: 2 <= min < max, positive step, max excluded")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (
            self.lookback_min, self.lookback_max, self.lookback_step, self.seed, self.warmup,
            self.context_bars, self.shadow_limit, self.profit_factor_window, self.group_main_trades,
            self.group_max_bars, self.lookback_grid_size, self.stop_grid_size,
        )):
            raise ValueError("Bandit counts, bounds and seed must be integers")
        if self.warmup < 1 or self.context_bars < 2 or self.profit_factor_window < 1:
            raise ValueError("Bandit warmup/PF window must be positive; context needs at least two candles")
        if self.lookback_grid_size < 2 or self.stop_grid_size < 2:
            raise ValueError("Non-degenerate bandit grids must include their two endpoints")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.lookback_grid):
            raise ValueError("Explicit lookback grid needs integer values")
        if self.shadow_limit < 0 or self.group_main_trades < 0 or self.group_max_bars < 0:
            raise ValueError("Shadow limits must be nonnegative")
        for name in ("alpha", "regularization", "reward_clip", "shadow_budget", "feature_clip"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0 or name in ("regularization", "reward_clip", "feature_clip") and value == 0:
                raise ValueError(f"Invalid bandit {name}")
        if self.repeat_weight is not None and (not isfinite(self.repeat_weight) or not 0 < self.repeat_weight <= 1):
            raise ValueError("Explicit repeated-training weight must be in (0, 1]")

    @property
    def effective_group_trades(self):
        return self.group_main_trades or self.profit_factor_window

    @property
    def effective_group_bars(self):
        # PF is measured in trades. This separate finite time guard also
        # bounds a scenario when the principal path produces no trades.
        return self.group_max_bars or self.context_bars * self.profit_factor_window


@dataclass(frozen=True)
class BanditAction:
    lookback: int
    trailing_enabled: bool
    stop_extra_atr_multiplier: float

    @property
    def action_id(self):
        return stable_id({"contract": ACTION_VERSION, **asdict(self), "stop_unit": "ATR"})

    def to_dict(self):
        return {**asdict(self), "action_id": self.action_id}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data["trailing_enabled"], bool):
            raise ValueError("Stored trailing action must be a boolean")
        action = cls(int(data["lookback"]), data["trailing_enabled"], float(data["stop_extra_atr_multiplier"]))
        if action.lookback < 2 or not isfinite(action.stop_extra_atr_multiplier) or action.stop_extra_atr_multiplier < 0:
            raise ValueError("Invalid stored action")
        if "action_id" in data and action.action_id != data["action_id"]:
            raise ValueError("Stored action identifier mismatch")
        return action


class ActionSpace:
    def __init__(self, config: BanditConfig, max_stop_atr: float, baseline: BanditAction):
        if not isfinite(max_stop_atr) or max_stop_atr < 0:
            raise ValueError("Maximum extra stop must be a nonnegative ATR coefficient")
        allowed = list(range(config.lookback_min, config.lookback_max, config.lookback_step))
        if config.lookback_grid:
            lookbacks = sorted(set(config.lookback_grid))
            if not lookbacks or any(value not in allowed for value in lookbacks):
                raise ValueError("Lookback grid must belong to the existing adaptive integer lattice")
        elif len(allowed) <= config.lookback_grid_size:
            lookbacks = allowed
        else:
            indices = np.linspace(0, len(allowed) - 1, config.lookback_grid_size).round().astype(int)
            lookbacks = sorted({allowed[int(index)] for index in indices})
            if baseline.lookback in allowed and baseline.lookback not in lookbacks:
                interior = lookbacks[1:-1]
                if interior:
                    lookbacks.remove(min(interior, key=lambda value: abs(value - baseline.lookback)))
                    lookbacks.append(baseline.lookback)
                    lookbacks.sort()
        if max_stop_atr == 0:
            stops = [0.0]
        elif config.stop_grid:
            stops = sorted(set(float(value) for value in config.stop_grid))
            if (not stops or stops[0] != 0 or stops[-1] != max_stop_atr
                    or any(not isfinite(value) or not 0 <= value <= max_stop_atr for value in stops)):
                raise ValueError("Stop grid must include zero and the configured maximum, within the ATR bound")
        else:
            stops = [float(value) for value in np.linspace(0, max_stop_atr, config.stop_grid_size)]
        self.actions = tuple(BanditAction(lookback, trailing, stop) for lookback in lookbacks
                             for trailing in (False, True) for stop in stops)
        self.by_id = {action.action_id: action for action in self.actions}
        self.baseline = baseline
        self.maximum_stop_atr = max_stop_atr
        self.domain = {"lookbacks": lookbacks, "lookback_bounds": [config.lookback_min, config.lookback_max],
                       "lookback_step": config.lookback_step, "upper_bound": "exclusive",
                       "trailing": [False, True], "stop_coefficients": stops, "stop_unit": "ATR",
                       "maximum_stop_atr": max_stop_atr}


@dataclass(frozen=True)
class ModelIdentity:
    payload: dict[str, Any]

    @property
    def model_id(self):
        return stable_id(self.payload)

    @classmethod
    def create(cls, symbol, timeframe_seconds, fixed_parameters, domain, config):
        parameters = asdict(config)
        # Fingerprint the resolved domain, not how an equivalent CLI grid
        # was spelled. Execution-only repetition is not a policy parameter.
        for key in ("repeat_weight", "lookback_grid", "stop_grid", "lookback_grid_size", "stop_grid_size"):
            parameters.pop(key)
        parameters["group_main_trades"] = config.effective_group_trades
        parameters["group_max_bars"] = config.effective_group_bars
        return cls({"symbol": str(symbol).strip().upper(), "timeframe_seconds": int(timeframe_seconds),
                    "strategy": BEHAVIOR_VERSION, "fixed": fixed_parameters, "domain": domain,
                    "bandit": parameters, "features": FEATURE_VERSION, "actions": ACTION_VERSION,
                    "reward": REWARD_VERSION})


@dataclass(frozen=True)
class MarketContext:
    timestamp: str
    values: tuple[float, ...] = (0.0,) * 6
    available: tuple[bool, ...] = (False,) * 6
    valid: bool = False
    reason: str = "history_insufficient"
    bars: int = 0
    convention: str = FEATURE_VERSION

    def __post_init__(self):
        if len(self.values) != 6 or len(self.available) != 6:
            raise ValueError("Context has six features and six availability masks")
        if not all(isfinite(value) for value in self.values):
            raise ValueError("Context cannot contain NaN or infinity")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "values": tuple(value["values"]), "available": tuple(value["available"])})


class FrozenScaler:
    def __init__(self, warmup: int, state=None, clip=10.0):
        if state is not None and not {"warmup", "clip", "samples", "mean", "scale", "frozen"}.issubset(state):
            raise ValueError("Persisted scaler is incomplete; explicit migration required")
        state = state or {}
        self.warmup = warmup
        self.clip = float(state.get("clip", clip))
        self.samples = list(state.get("samples", []))
        self.mean = np.array(state.get("mean", [0.0] * 6), dtype=float)
        self.scale = np.array(state.get("scale", [1.0] * 6), dtype=float)
        self.frozen = bool(state.get("frozen", False))
        if state and int(state.get("warmup", -1)) != warmup:
            raise ValueError("Stored scaler warmup contract changed")
        if (self.mean.shape != (6,) or self.scale.shape != (6,) or not isfinite(self.clip) or self.clip <= 0
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)):
            raise ValueError("Stored scaler is incompatible or nonfinite")
        for sample in self.samples:
            if (len(sample["values"]) != 6 or len(sample["available"]) != 6
                    or not all(isfinite(value) for value in sample["values"])):
                raise ValueError("Invalid persisted warmup sample")

    def collect(self, context: MarketContext) -> bool:
        if self.frozen or not context.valid:
            return False
        key = stable_id(context.to_dict())
        if any(sample["key"] == key for sample in self.samples):
            return False
        self.samples.append({"key": key, "values": context.values, "available": context.available})
        if len(self.samples) < self.warmup:
            return False
        for index in range(6):
            values = [sample["values"][index] for sample in self.samples if sample["available"][index]]
            if values:
                self.mean[index] = float(np.mean(values))
                self.scale[index] = max(float(np.std(values)), 1e-8)
        self.frozen = True
        return True

    def transform(self, context: MarketContext):
        if not self.frozen or not context.valid:
            raise ValueError("Learning needs a valid context and a frozen scaler")
        mask = np.array(context.available, dtype=float)
        continuous = np.clip((np.array(context.values) - self.mean) / self.scale, -self.clip, self.clip)
        x = np.concatenate(([1.0], continuous * mask, mask))
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite normalized context")
        return x

    def to_dict(self):
        return {"warmup": self.warmup, "clip": self.clip, "samples": self.samples, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "frozen": self.frozen}


class LinUCBPolicy:
    def __init__(self, config: BanditConfig, actions, state=None):
        self.config = config
        self.arms = {}
        actions = tuple(actions)
        if state is not None and set(state) != {action.action_id for action in actions}:
            raise ValueError("Persisted action statistics are missing or incompatible")
        for action in actions:
            data = (state or {}).get(action.action_id)
            self.arms[action.action_id] = {
                "A": np.array(data["A"], dtype=float) if data else np.eye(FEATURE_DIMENSION) * config.regularization,
                "b": np.array(data["b"], dtype=float) if data else np.zeros(FEATURE_DIMENSION),
                "count": int(data.get("count", 0)) if data else 0,
                "weight": float(data.get("weight", 0)) if data else 0.0,
            }
        for arm in self.arms.values():
            if arm["A"].shape != (FEATURE_DIMENSION, FEATURE_DIMENSION) or arm["b"].shape != (FEATURE_DIMENSION,):
                raise ValueError("Incompatible LinUCB matrix dimensions")
            if not np.isfinite(arm["A"]).all() or not np.isfinite(arm["b"]).all():
                raise ValueError("Nonfinite persisted LinUCB state")
            if (not np.allclose(arm["A"], arm["A"].T) or arm["count"] < 0
                    or not isfinite(arm["weight"]) or arm["weight"] < 0):
                raise ValueError("Invalid persisted LinUCB statistics")
            np.linalg.cholesky(arm["A"])

    def update(self, action_id, x, reward, weight):
        x = np.asarray(x, dtype=float)
        if x.shape != (FEATURE_DIMENSION,) or not np.isfinite(x).all() or not isfinite(reward) or not isfinite(weight) or weight < 0:
            raise ValueError("Invalid LinUCB observation")
        if weight == 0:
            return
        arm = self.arms[action_id]
        arm["A"] += weight * np.outer(x, x)
        arm["b"] += weight * reward * x
        arm["count"] += 1
        arm["weight"] += weight

    def scores(self, x, action_ids=None):
        result = []
        for action_id in self.arms if action_ids is None else action_ids:
            arm = self.arms[action_id]
            cholesky = np.linalg.cholesky(arm["A"])
            solved_x = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, x))
            theta = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, arm["b"]))
            mean = float(np.dot(x, theta))
            bonus = self.config.alpha * sqrt(max(0.0, float(np.dot(x, solved_x))))
            result.append({"action_id": action_id, "mean": mean, "bonus": bonus, "score": mean + bonus})
        return result

    def select(self, x, action_ids, rng: Random):
        scores = self.scores(x, action_ids)
        if not scores:
            raise ValueError("No valid bandit action")
        best = max(item["score"] for item in scores)
        return rng.choice([item for item in scores if abs(item["score"] - best) <= 1e-12])

    def to_dict(self):
        return {key: {**arm, "A": arm["A"].tolist(), "b": arm["b"].tolist()} for key, arm in self.arms.items()}


@dataclass(frozen=True)
class DecisionSnapshot:
    model_id: str
    action: BanditAction
    policy_version: int
    context: MarketContext
    mean: float = 0.0
    bonus: float = 0.0
    reason: str = "baseline_warmup"

    def to_dict(self):
        return {**asdict(self), "action": self.action.to_dict()}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "action": BanditAction.from_dict(value["action"]),
                      "context": MarketContext.from_dict(value["context"])})


@dataclass(frozen=True)
class PositionLearningRecord:
    decision: DecisionSnapshot
    entry_context: MarketContext
    run_id: str
    source: str
    trajectory_key: str
    scenario_key: str
    atr_entry: float
    initial_stop: float
    initial_risk: float | None = None
    initial_quantity: float | None = None
    group_id: str | None = None
    variant_id: str | None = None
    account_currency: str = "engine_price_quantity"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {**asdict(self), "decision": self.decision.to_dict()}

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        for key in ("initial_risk", "initial_quantity", "atr_entry", "initial_stop"):
            if isinstance(data.get(key), dict) and "float" in data[key]:
                data[key] = float(data[key]["float"])
        return cls(**{**data, "decision": DecisionSnapshot.from_dict(value["decision"]),
                      "entry_context": MarketContext.from_dict(value["entry_context"])})


@dataclass(frozen=True)
class ClosedTradeObservation:
    event_key: str
    record: PositionLearningRecord
    closed_at: str
    close_context: MarketContext
    pnl_net: float
    close_reason: str
    pnl_gross: float | None = None
    forced: bool = False
    broker_deal_ids: tuple[str, ...] = ()

    def reward(self, clip):
        risk = self.record.initial_risk
        if risk is None or not isfinite(risk) or risk <= 0 or not isfinite(self.pnl_net):
            return None, None, "initial_risk_or_pnl_invalid"
        raw = self.pnl_net / risk
        if not isfinite(raw):
            return None, None, "reward_nonfinite"
        if self.forced:
            return raw, None, "forced_close_excluded"
        return raw, max(-clip, min(clip, raw)) / clip, None

    def to_dict(self):
        return {**asdict(self), "record": self.record.to_dict()}

    @classmethod
    def from_dict(cls, value):
        def number(item):
            if isinstance(item, dict) and "float" in item:
                return float(item["float"])
            return item
        return cls(**{**value, "record": PositionLearningRecord.from_dict(value["record"]),
                      "close_context": MarketContext.from_dict(value["close_context"]),
                      "pnl_net": number(value["pnl_net"]), "pnl_gross": number(value["pnl_gross"]),
                      "broker_deal_ids": tuple(value["broker_deal_ids"])})


class ContextBuilder:
    """Read a fixed closed-candle window, never the mutable active candle."""
    def __init__(self, bars):
        self.bars = bars

    def build(self, symbol, timestamp, market_state, strategy):
        interval = strategy.get_analysis_interval(symbol)
        circuits = {source: market_state.get_candles(symbol, include_active=False, price_source=source,
                                                    limit=self.bars, candle_interval=interval)
                    for source in ("bid", "ask", "mid")}
        if any(len(candles) < self.bars for candles in circuits.values()):
            return MarketContext(timestamp.isoformat(), bars=min(map(len, circuits.values())))
        point_size = market_state.get_point_size(symbol)
        fixed_rules = {**strategy.rules_for_symbol(symbol), "regression_lookback_bars": self.bars,
                       "regression_min_market_volatility_ratio": 0.0}
        result = strategy._evaluate_regression_pipeline(
            symbol, circuits["bid"], circuits["ask"], circuits["mid"], point_size,
            timeframe_label="context", timeframe_multiplier=1, require_fresh_swings=False,
            rules=fixed_rules, context=type("Context", (), {"market_state": market_state})(),
        )
        analysis = result[1] if result is not None else {}
        line = analysis.get("anchored_line")
        atr = float(analysis.get("atr", 0.0))
        price = float(circuits["mid"][-1].close_price)
        slope = float(line["slope"]) / atr if line and atr > 0 else None
        volatility = atr / abs(price) if atr > 0 and price != 0 else None
        values = [0.0] * 6
        masks = [False] * 6
        invalid = False
        for index, value in enumerate((slope, volatility)):
            if value is not None:
                invalid |= not isfinite(value)
                if isfinite(value):
                    values[index], masks[index] = value, True
        for side, tick_index, candle_index in (("bid", 2, 4), ("ask", 3, 5)):
            candles = [candle for candle in circuits["mid"] if getattr(candle, f"{side}_volume_count", 0) > 0]
            if candles:
                total = sum(getattr(candle, f"{side}_volume_sum") for candle in candles)
                count = sum(getattr(candle, f"{side}_volume_count") for candle in candles)
                if not isfinite(total) or total < 0:
                    invalid = True
                else:
                    values[tick_index], masks[tick_index] = log1p(total / count), True
                    values[candle_index], masks[candle_index] = log1p(total / len(candles)), True
        valid = masks[0] and masks[1] and not invalid
        return MarketContext(timestamp.isoformat(), tuple(values), tuple(masks), valid,
                             "ready" if valid else "nonfinite_features" if invalid else "trend_unavailable", self.bars)
