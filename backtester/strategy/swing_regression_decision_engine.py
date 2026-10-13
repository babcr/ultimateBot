from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, tanh
from typing import Any

import numpy as np

from ultimateBot.backtester.strategy.structure_decision_engine import (
    DEFAULT_STRUCTURE_RULES, _normalize_swings, build_structure_levels, resolve_structure_rules,
)


DEFAULT_SWING_REGRESSION_RULES = {
    # None keeps legacy commands/configurations on the level window until
    # an independent regression lookback is explicitly supplied.
    "regression_lookback_bars": None,
    "regression_min_swings_per_side": 2,
    "regression_score_threshold": 0.5,
    # Shared drift per candle, in ATR units: reaching this reference gives tanh(1).
    "regression_slope_scale_atr": 0.3, # 0.43 optimal
    "regression_use_global_timeframe_validation": True,
    "regression_global_timeframe_multiplier": 2,
    "regression_global_signal_lookback_bars": 5,
    "regression_global_reference_slope_threshold_atr": 0.3,
    "regression_global_min_parallelism": 0.5,
    "regression_market_volatility_window_bars": 30,
    "regression_min_market_volatility_ratio": 0.0001, # 0.002 optimal
    "regression_line_max_volatility_atr": 1.6,
    "regression_trigger_max_distance_atr": 5.3,
    "regression_require_level_reaction": True,
    "regression_min_consecutive_swing_distance_atr": 1.,
    "regression_max_filtered_consecutive_swing_pair_density": .3,
    "regression_max_swing_density": 1.,
    "regression_min_parallelism": .5,
    "regression_live_analysis": False,
}


def resolve_swing_regression_rules(config: dict[str, Any] | None = None) -> dict[str, Any]:
    overrides = {key: value for key, value in (config or {}).items() if value is not None}
    rules = resolve_structure_rules({**DEFAULT_SWING_REGRESSION_RULES, **overrides})
    # This branch fits the exact requested window for the regression analysis.
    rules["level_lookback_bars"] = max(2, int(overrides.get("level_lookback_bars", DEFAULT_STRUCTURE_RULES["level_lookback_bars"])))
    rules["regression_lookback_bars"] = max(2, int(
        overrides.get("regression_lookback_bars", rules["level_lookback_bars"])
    ))
    rules["regression_min_swings_per_side"] = max(2, int(rules["regression_min_swings_per_side"]))
    threshold = float(rules["regression_score_threshold"])
    rules["regression_score_threshold"] = (
        min(1.0, max(0.0, threshold)) if isfinite(threshold)
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_score_threshold"])
    )
    slope_scale = float(rules["regression_slope_scale_atr"])
    rules["regression_slope_scale_atr"] = (
        slope_scale if isfinite(slope_scale) and slope_scale > 0.0
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_slope_scale_atr"])
    )
    rules["regression_use_global_timeframe_validation"] = bool(
        rules["regression_use_global_timeframe_validation"]
    )
    rules["regression_global_timeframe_multiplier"] = max(1, int(rules["regression_global_timeframe_multiplier"]))
    rules["regression_global_signal_lookback_bars"] = max(1, int(rules["regression_global_signal_lookback_bars"]))
    global_reference_slope_threshold_atr = float(rules["regression_global_reference_slope_threshold_atr"])
    rules["regression_global_reference_slope_threshold_atr"] = (
        max(0.0, global_reference_slope_threshold_atr)
        if isfinite(global_reference_slope_threshold_atr)
        or global_reference_slope_threshold_atr == float("inf")
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_global_reference_slope_threshold_atr"])
    )
    global_min_parallelism = float(rules["regression_global_min_parallelism"])
    rules["regression_global_min_parallelism"] = (
        min(1.0, max(0.0, global_min_parallelism))
        if isfinite(global_min_parallelism)
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_global_min_parallelism"])
    )
    rules["regression_market_volatility_window_bars"] = max(2, int(rules["regression_market_volatility_window_bars"]))
    min_market_volatility_ratio = float(rules["regression_min_market_volatility_ratio"])
    rules["regression_min_market_volatility_ratio"] = (
        min_market_volatility_ratio
        if min_market_volatility_ratio >= 0.0 and (isfinite(min_market_volatility_ratio) or min_market_volatility_ratio == float("inf"))
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_min_market_volatility_ratio"])
    )
    line_max_volatility = float(rules["regression_line_max_volatility_atr"])
    rules["regression_line_max_volatility_atr"] = (
        line_max_volatility
        if line_max_volatility >= 0.0 and (isfinite(line_max_volatility) or line_max_volatility == float("inf"))
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_line_max_volatility_atr"])
    )
    trigger_max_distance = float(rules["regression_trigger_max_distance_atr"])
    rules["regression_trigger_max_distance_atr"] = (
        trigger_max_distance
        if trigger_max_distance >= 0.0 and (isfinite(trigger_max_distance) or trigger_max_distance == float("inf"))
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_trigger_max_distance_atr"])
    )
    min_consecutive_swing_distance = float(rules["regression_min_consecutive_swing_distance_atr"])
    rules["regression_min_consecutive_swing_distance_atr"] = (
        min_consecutive_swing_distance
        if min_consecutive_swing_distance >= 0.0 and (isfinite(min_consecutive_swing_distance) or min_consecutive_swing_distance == float("inf"))
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_min_consecutive_swing_distance_atr"])
    )
    max_filtered_pair_density = float(rules["regression_max_filtered_consecutive_swing_pair_density"])
    rules["regression_max_filtered_consecutive_swing_pair_density"] = (
        max_filtered_pair_density
        if max_filtered_pair_density >= 0.0 and (isfinite(max_filtered_pair_density) or max_filtered_pair_density == float("inf"))
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_max_filtered_consecutive_swing_pair_density"])
    )
    max_swing_density = float(rules["regression_max_swing_density"])
    rules["regression_max_swing_density"] = (
        max_swing_density
        if max_swing_density >= 0.0 and (isfinite(max_swing_density) or max_swing_density == float("inf"))
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_max_swing_density"])
    )
    min_parallelism = float(rules["regression_min_parallelism"])
    rules["regression_min_parallelism"] = (
        min(1.0, max(0.0, min_parallelism)) if isfinite(min_parallelism)
        else float(DEFAULT_SWING_REGRESSION_RULES["regression_min_parallelism"])
    )
    rules["regression_live_analysis"] = bool(rules["regression_live_analysis"])
    rules["regression_require_level_reaction"] = bool(rules["regression_require_level_reaction"])
    return rules


@dataclass(frozen=True)
class RegressionLine:
    kind: str
    slope: float
    intercept: float
    point_count: int
    r_squared: float
    rmse: float
    anchor_index: int | None = None
    anchor_price: float | None = None

    def price_at(self, index: float) -> float:
        return self.intercept + self.slope * index


@dataclass(frozen=True)
class SwingRegressionDecision:
    side: str
    confidence: float
    trend_score: float
    protective_level_price: float
    protective_level_quality: float
    protective_stop_buffer: float


def historical_family_candidates(
    kind: str,
    historical_anchor_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        dict(candidate)
        for candidate in (historical_anchor_candidates or [])
        if candidate.get("kind") == kind
        and "index" in candidate
        and "price" in candidate
        and isfinite(float(candidate["price"]))
    ]


def select_reference_anchor(
    swings: list[dict[str, Any]], current_price: float, minimum_points: int = 2,
    historical_anchor_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for kind in ("high", "low"):
        family = [swing for swing in swings if swing.get("kind") == kind]
        historical_family = historical_family_candidates(kind, historical_anchor_candidates)
        if family and len(family) + len(historical_family) >= minimum_points:
            candidates.extend(family)
            candidates.extend(historical_family)
    if not candidates:
        return None
    return dict(max(candidates, key=lambda swing: (abs(float(swing["price"]) - current_price), int(swing["index"]))))


def fit_swing_line(
    swings: list[dict[str, Any]], anchor: dict[str, Any] | None = None,
) -> RegressionLine | None:
    if len(swings) < 2:
        return None
    x = np.array([float(swing["index"]) for swing in swings], dtype=np.float64)
    y = np.array([float(swing["price"]) for swing in swings], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(np.unique(x)) < 2:
        return None
    if anchor is None:
        dx = x - x.mean()
        dy = y - y.mean()
        anchor_index = None
        anchor_price = None
    else:
        anchor_index = int(anchor["index"])
        anchor_price = float(anchor["price"])
        dx = x - anchor_index
        dy = y - anchor_price
    denominator = float(np.dot(dx, dx))
    if denominator <= 0.0:
        return None
    slope = float(np.dot(dx, dy) / denominator)
    intercept = float(y.mean() - slope * x.mean()) if anchor is None else anchor_price - slope * anchor_index
    residuals = y - (intercept + slope * x)
    residual_sum = float(np.dot(residuals, residuals))
    total_sum = float(np.dot(y - y.mean(), y - y.mean()))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else float(residual_sum <= 1e-20)
    return RegressionLine(str(swings[0]["kind"]), slope, intercept, len(swings), r_squared,
                          float(np.sqrt(residual_sum / len(swings))), anchor_index, anchor_price)


def extend_family_with_anchor(swings: list[dict[str, Any]], anchor: dict[str, Any] | None) -> list[dict[str, Any]]:
    if anchor is None:
        return list(swings)
    anchor_kind = anchor.get("kind")
    anchor_index = int(anchor["index"])
    anchor_price = float(anchor["price"])
    family = [dict(swing) for swing in swings]
    if any(
        swing.get("kind") == anchor_kind
        and int(swing["index"]) == anchor_index
        and float(swing["price"]) == anchor_price
        for swing in family
    ):
        return family
    family.append(dict(anchor))
    family.sort(key=lambda swing: int(swing["index"]))
    return family


def extend_family_with_candidates(
    swings: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    family = [dict(swing) for swing in swings]
    for candidate in candidates or []:
        candidate_kind = candidate.get("kind")
        candidate_index = int(candidate["index"])
        candidate_price = float(candidate["price"])
        if any(
            swing.get("kind") == candidate_kind
            and int(swing["index"]) == candidate_index
            and float(swing["price"]) == candidate_price
            for swing in family
        ):
            continue
        family.append(dict(candidate))
    family.sort(key=lambda swing: int(swing["index"]))
    return family


def swing_key(swing: dict[str, Any]) -> tuple[str, int, float]:
    return str(swing.get("kind", "")), int(swing["index"]), float(swing["price"])


def inferred_swing_source(swing: dict[str, Any]) -> str | None:
    source = swing.get("source")
    if source is not None:
        normalized = str(source).strip().lower()
        if normalized in {"ask", "bid"}:
            return normalized
    kind = str(swing.get("kind", "")).strip().lower()
    if kind == "high":
        return "ask"
    if kind == "low":
        return "bid"
    return None


def filter_close_consecutive_swing_pairs(
    swings: list[dict[str, Any]], atr: float, min_distance_atr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = [dict(swing) for swing in swings]
    if len(ordered) < 2 or atr <= 0.0 or min_distance_atr <= 0.0:
        return ordered, []
    kept: list[dict[str, Any]] = []
    filtered_pairs: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        if index + 1 >= len(ordered):
            kept.append(current)
            break
        following = ordered[index + 1]
        if current.get("kind") == following.get("kind"):
            kept.append(current)
            index += 1
            continue
        distance = abs(float(current["price"]) - float(following["price"]))
        distance_atr = distance / atr if atr > 0.0 else float("inf")
        if distance_atr < min_distance_atr:
            next_swing = dict(ordered[index + 2]) if index + 2 < len(ordered) else None
            filtered_pairs.append({
                "first": dict(current),
                "second": dict(following),
                "next_swing": next_swing,
                "distance": distance,
                "distance_atr": distance_atr,
            })
            index += 2
            continue
        kept.append(current)
        index += 1
    return kept, filtered_pairs


def filtered_pairs_before_triggers(
    filtered_pairs: list[dict[str, Any]], trigger_swings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not filtered_pairs or not trigger_swings:
        return []
    trigger_keys = {swing_key(swing) for swing in trigger_swings}
    return [
        dict(pair)
        for pair in filtered_pairs
        if pair.get("next_swing") is not None and swing_key(pair["next_swing"]) in trigger_keys
    ]


def regression_trend_metrics(
    anchored: RegressionLine, floating: RegressionLine | None, atr: float, slope_scale_atr: float,
) -> dict[str, float | bool]:
    if floating is None:
        parallelism = 1.0
        shared_slope = anchored.slope
    else:
        denominator = abs(anchored.slope) + abs(floating.slope)
        parallelism = max(0.0, 1.0 - abs(anchored.slope - floating.slope) / denominator) if denominator > 0.0 else 1.0
        shared_slope = (anchored.slope + floating.slope) / 2.0
    normalized_slope = shared_slope / atr
    # A per-candle drift, not a whole-window displacement: length alone cannot saturate the score.
    slope_strength = tanh(normalized_slope / slope_scale_atr)
    trend_score = parallelism * slope_strength
    if abs(trend_score) < 1e-12:
        trend_score = 0.0
    return {"parallelism": parallelism, "shared_slope": shared_slope,
            "normalized_slope": normalized_slope, "slope_strength": slope_strength,
            "slope_scale_atr": slope_scale_atr, "trend_score": trend_score,
            "reference_line_only": floating is None}


def regression_line_volatility_metrics(
    anchored: RegressionLine, floating: RegressionLine | None, atr: float,
) -> dict[str, float]:
    anchored_volatility = anchored.rmse / atr if atr > 0.0 else float("inf")
    floating_volatility = 0.0 if floating is None else floating.rmse / atr if atr > 0.0 else float("inf")
    return {
        "anchored_line_volatility_atr": anchored_volatility,
        "floating_line_volatility_atr": floating_volatility,
        "max_line_volatility_atr": max(anchored_volatility, floating_volatility),
    }


def evaluate_trigger_swings(
    trigger_swings: list[dict[str, Any]], anchored: RegressionLine, floating: RegressionLine | None, atr: float,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for swing in trigger_swings:
        kind = str(swing.get("kind", ""))
        line = anchored if kind == anchored.kind else floating if floating is not None and kind == floating.kind else None
        if line is None:
            continue
        index = int(swing["index"])
        price = float(swing["price"])
        line_price = float(line.price_at(index))
        distance = abs(price - line_price)
        evaluations.append({
            "kind": kind,
            "index": index,
            "price": price,
            "line_kind": line.kind,
            "line_price": line_price,
            "distance": distance,
            "distance_atr": distance / atr if atr > 0.0 else float("inf"),
        })
    return evaluations


def select_swing_window(
    swings: list[dict[str, Any]], start: int, end: int, atr: float,
    source_atrs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Select by shared indices; normalize raw candidates after windowing.

    Raw two-candle pivots need their preceding candle inside the window.
    Legacy callers can still supply already normalized swing vectors.
    """
    selected = [dict(swing) for swing in swings
                if swing.get("kind") in ("high", "low")
                and start <= int(swing["index"]) < end
                and ("pair_range" not in swing or int(swing["index"]) > start)
                and isfinite(float(swing["price"]))]
    if not any("pair_range" in swing for swing in selected):
        return selected
    shared: list[dict[str, Any]] = []
    for source, kind in (("bid", "low"), ("ask", "high")):
        denominator = max(float((source_atrs or {}).get(source, atr)), 1e-9)
        candidates = []
        for swing in selected:
            if swing.get("source") != source:
                continue
            pair_range = float(swing.pop("pair_range"))
            candidates.append({**swing, "strength": pair_range / denominator})
        shared.extend(swing for swing in _normalize_swings(candidates) if swing["kind"] == kind)
    return _normalize_swings(shared)


def evaluate_trigger_level_reactions(
    swings: list[dict[str, Any]], trigger_swings: list[dict[str, Any]],
    side: str, current_price: float, atr: float, point_size: float,
    rules: dict[str, Any],
    *, level_swings: list[dict[str, Any]] | None = None,
    candle_closes: list[float] | None = None,
    level_window_start: int = 0,
    level_source_atrs: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Match the entry leg to a level established before its opposite pivot.

    Buys still trigger on ask highs and sells on bid lows. The preceding
    opposite pivot must touch an existing level, then the closed mid price
    must leave that level's reaction zone in the regression direction.
    """
    origin_kind = "low" if side == "buy" else "high"
    trigger_kind = "high" if side == "buy" else "low"
    sign = 1.0 if side == "buy" else -1.0
    zone = float(rules["entry_distance_atr"]) * atr
    evaluations: list[dict[str, Any]] = []
    for trigger in trigger_swings:
        evaluation: dict[str, Any] = {
            "trigger": dict(trigger), "origin": None, "approach_price": None, "levels": [],
            "level": None, "distance_atr": None, "departure_atr": None,
            "style": "support_bounce" if side == "buy" else "resistance_rejection",
            "passed": False, "reason": "missing_reaction_origin",
        }
        evaluations.append(evaluation)
        if trigger.get("kind") != trigger_kind:
            continue
        origins = [swing for swing in swings if swing.get("kind") == origin_kind
                   and int(swing["index"]) < int(trigger["index"])]
        if not origins:
            continue
        origin = max(origins, key=lambda swing: int(swing["index"]))
        evaluation["origin"] = dict(origin)
        origin_price = float(origin["price"])
        # The price must approach from the support/resistance side. A move
        # crossing a level from the other side is a breakout, not a reaction.
        approach_index = int(origin["index"]) - 2
        if candle_closes is not None:
            approach_price = (
                float(candle_closes[approach_index])
                if 0 <= approach_index < len(candle_closes) else None
            )
        else:
            approaches = [swing for swing in swings if swing.get("kind") == trigger_kind
                          and int(swing["index"]) < int(origin["index"])]
            approach_price = (
                float(max(approaches, key=lambda swing: int(swing["index"]))["price"])
                if approaches else None
            )
        evaluation["approach_price"] = approach_price
        if approach_price is None or not isfinite(approach_price):
            evaluation["reason"] = "missing_reaction_approach"
            continue
        # Exclude both the touching pivot and the trigger: neither may create
        # or strengthen the level that validates its own entry leg.
        levels = build_structure_levels(
            select_swing_window(
                swings if level_swings is None else level_swings,
                level_window_start, int(origin["index"]), atr, level_source_atrs,
            ),
            origin_price, atr, point_size, config=rules,
            max_age=int(origin["index"]) - 1,
        )
        evaluation["levels"] = levels
        evaluation["reason"] = "no_preexisting_level" if not levels else "origin_too_far_from_level"
        touching_levels = [level for level in levels
                           if abs(origin_price - float(level["price"])) <= zone]
        if not touching_levels:
            continue
        touching_levels = [level for level in touching_levels
                           if sign * (approach_price - float(level["price"])) > zone]
        if not touching_levels:
            evaluation["reason"] = "level_approached_from_wrong_side"
            continue
        evaluation["reason"] = "level_zone_not_exited"
        if sign * (float(trigger["price"]) - origin_price) <= 0.0:
            continue
        exited_levels = [level for level in touching_levels
                         if sign * (current_price - float(level["price"])) > zone]
        if not exited_levels:
            continue
        level = min(exited_levels, key=lambda item: (
            abs(origin_price - float(item["price"])), -float(item["quality"]),
        ))
        evaluation.update({
            "level": {**level, "side": "support" if side == "buy" else "resistance"},
            "distance_atr": abs(origin_price - float(level["price"])) / atr,
            "departure_atr": sign * (current_price - float(level["price"])) / atr,
            "passed": True, "reason": "level_reaction_confirmed",
        })
    return evaluations


def decide_swing_regression(
    swings: list[dict[str, Any]], current_price: float, atr: float, bar_count: int,
    config: dict[str, Any] | None = None,
    trigger_swings: list[dict[str, Any]] | None = None,
    historical_anchor_candidates: list[dict[str, Any]] | None = None,
    market_volatility_atr: float | None = None,
    market_volatility_ratio: float | None = None,
    point_size: float = 1e-9,
    candle_closes: list[float] | None = None,
    level_atr: float | None = None,
    regression_source_atrs: dict[str, float] | None = None,
    level_source_atrs: dict[str, float] | None = None,
) -> tuple[SwingRegressionDecision | None, dict[str, Any]]:
    rules = resolve_swing_regression_rules(config)
    analysis: dict[str, Any] = {
        "current_price": current_price, "atr": atr, "bar_count": bar_count,
        "regression_lookback_bars": rules["regression_lookback_bars"],
        "level_lookback_bars": rules["level_lookback_bars"],
        "regression_window_start": max(0, bar_count - rules["regression_lookback_bars"]),
        "level_window_start": max(0, bar_count - rules["level_lookback_bars"]),
        "window_end": bar_count, "level_atr": atr if level_atr is None else level_atr,
        "anchor": None, "anchored_line": None, "floating_line": None,
        "protective_swing": None,
        "parallelism": 0.0, "shared_slope": 0.0, "normalized_slope": 0.0,
        "anchored_slope_atr": 0.0, "floating_slope_atr": 0.0,
        "slope_strength": 0.0, "slope_scale_atr": rules["regression_slope_scale_atr"],
        "market_volatility_window_bars": rules["regression_market_volatility_window_bars"],
        "market_volatility_atr": 0.0,
        "market_volatility_ratio": 0.0,
        "regression_min_market_volatility_ratio": rules["regression_min_market_volatility_ratio"],
        "trend_score": 0.0, "direction": "flat", "reference_line_only": False,
        "raw_swing_count": 0, "swing_count": 0, "swing_density": 0.0,
        "regression_min_consecutive_swing_distance_atr": rules["regression_min_consecutive_swing_distance_atr"],
        "filtered_consecutive_swing_pairs": [],
        "filtered_consecutive_swing_pair_count": 0,
        "filtered_consecutive_swing_pair_density": 0.0,
        "regression_max_filtered_consecutive_swing_pair_density": rules["regression_max_filtered_consecutive_swing_pair_density"],
        "regression_max_swing_density": rules["regression_max_swing_density"],
        "regression_min_parallelism": rules["regression_min_parallelism"],
        "regression_line_max_volatility_atr": rules["regression_line_max_volatility_atr"],
        "anchored_line_volatility_atr": 0.0,
        "floating_line_volatility_atr": 0.0,
        "max_line_volatility_atr": 0.0,
        "volatile_lines": [],
        "trigger_max_distance_atr": rules["regression_trigger_max_distance_atr"],
        "trigger_blocking_filtered_pairs": [],
        "level_reaction_enabled": rules["regression_require_level_reaction"],
        "level_reaction_applied": False, "level_reaction_passed": None,
        "level_reaction_distance_atr": rules["entry_distance_atr"],
        "trigger_level_reactions": [], "entry_level_reaction": None,
        "trigger_swings": [], "reasons": [],
    }
    if not rules["enabled"]:
        analysis["reasons"] = ["regression_disabled"]
        return None, analysis
    if not isfinite(current_price) or current_price <= 0.0 or not isfinite(atr) or atr <= 0.0 or bar_count < 2:
        analysis["reasons"] = ["invalid_market_data"]
        return None, analysis
    effective_market_volatility_atr = float(atr if market_volatility_atr is None else market_volatility_atr)
    effective_market_volatility_ratio = (
        (effective_market_volatility_atr / max(abs(current_price), 1e-9))
        if market_volatility_ratio is None
        else float(market_volatility_ratio)
    )
    analysis["market_volatility_atr"] = effective_market_volatility_atr
    analysis["market_volatility_ratio"] = effective_market_volatility_ratio
    min_market_volatility_ratio = float(rules["regression_min_market_volatility_ratio"])
    if min_market_volatility_ratio > 0.0:
        if not isfinite(effective_market_volatility_ratio) or effective_market_volatility_ratio < 0.0:
            analysis["reasons"] = ["invalid_market_volatility"]
            return None, analysis
        if effective_market_volatility_ratio < min_market_volatility_ratio:
            analysis["reasons"] = ["market_volatility_too_low"]
            return None, analysis
    window_start = analysis["regression_window_start"]
    clean_swings = select_swing_window(swings, window_start, bar_count, atr, regression_source_atrs)
    analysis["raw_swing_count"] = len(clean_swings)
    clean_swings, filtered_pairs = filter_close_consecutive_swing_pairs(
        clean_swings,
        atr,
        float(rules["regression_min_consecutive_swing_distance_atr"]),
    )
    analysis["filtered_consecutive_swing_pairs"] = filtered_pairs
    analysis["filtered_consecutive_swing_pair_count"] = len(filtered_pairs)
    analysis["filtered_consecutive_swing_pair_density"] = (
        len(filtered_pairs) / rules["regression_lookback_bars"]
    )
    analysis["swing_count"] = len(clean_swings)
    analysis["swing_density"] = len(clean_swings) / rules["regression_lookback_bars"]
    if analysis["filtered_consecutive_swing_pair_density"] > float(rules["regression_max_filtered_consecutive_swing_pair_density"]):
        analysis["reasons"] = ["false_swing_pair_density_too_high"]
        return None, analysis
    if analysis["swing_density"] > float(rules["regression_max_swing_density"]):
        analysis["reasons"] = ["swing_agitation_too_high"]
        return None, analysis
    swing_counts = {
        "high": sum(1 for swing in clean_swings if swing["kind"] == "high"),
        "low": sum(1 for swing in clean_swings if swing["kind"] == "low"),
    }
    anchor = select_reference_anchor(
        clean_swings,
        current_price,
        rules["regression_min_swings_per_side"],
        historical_anchor_candidates,
    )
    analysis["anchor"] = anchor
    if anchor is None:
        populated_families = sum(1 for count in swing_counts.values() if count > 0)
        analysis["reasons"] = ["missing_swing_family" if populated_families < 2 else "insufficient_swings"]
        return None, analysis
    same_side = extend_family_with_candidates(
        extend_family_with_anchor(
            [swing for swing in clean_swings if swing["kind"] == anchor["kind"]],
            anchor,
        ),
        historical_family_candidates(str(anchor["kind"]), historical_anchor_candidates),
    )
    opposite_side = [swing for swing in clean_swings if swing["kind"] != anchor["kind"]]
    if len(same_side) < rules["regression_min_swings_per_side"]:
        analysis["reasons"] = ["insufficient_swings"]
        return None, analysis
    anchored = fit_swing_line(same_side, anchor)
    floating = fit_swing_line(opposite_side) if len(opposite_side) >= 2 else None
    if anchored is None or (len(opposite_side) >= 2 and floating is None):
        analysis["reasons"] = ["degenerate_regression"]
        return None, analysis
    analysis["anchored_line"] = asdict(anchored)
    analysis["floating_line"] = asdict(floating) if floating is not None else None
    analysis["anchored_slope_atr"] = anchored.slope / atr if atr > 0.0 else float("inf")
    analysis["floating_slope_atr"] = 0.0 if floating is None else floating.slope / atr if atr > 0.0 else float("inf")
    analysis.update(regression_trend_metrics(anchored, floating, atr, rules["regression_slope_scale_atr"]))
    analysis.update(regression_line_volatility_metrics(anchored, floating, atr))
    max_line_volatility = float(rules["regression_line_max_volatility_atr"])
    if analysis["max_line_volatility_atr"] > max_line_volatility:
        volatile_lines = []
        if analysis["anchored_line_volatility_atr"] > max_line_volatility:
            volatile_lines.append(anchored.kind)
        if floating is not None and analysis["floating_line_volatility_atr"] > max_line_volatility:
            volatile_lines.append(floating.kind)
        analysis["volatile_lines"] = volatile_lines
        analysis["reasons"] = ["regression_line_too_volatile"]
        return None, analysis
    if analysis["parallelism"] < rules["regression_min_parallelism"]:
        analysis["reasons"] = ["insufficient_parallelism"]
        return None, analysis
    slope_strength = analysis["slope_strength"]
    if slope_strength == 0.0 or abs(slope_strength) < rules["regression_score_threshold"]:
        analysis["reasons"] = ["neutral_trend"]
        return None, analysis
    score = analysis["trend_score"]
    side = "buy" if score > 0.0 else "sell"
    analysis["direction"] = side
    if trigger_swings is not None:
        surviving_swing_keys = {
            swing_key(swing)
            for swing in clean_swings
        }
        expected_trigger_source = "ask" if side == "buy" else "bid"
        analysis["expected_trigger_source"] = expected_trigger_source
        available_line_kinds = {anchored.kind}
        if floating is not None:
            available_line_kinds.add(floating.kind)
        trigger_candidates = [dict(swing) for swing in trigger_swings
                              if swing_key(swing) in surviving_swing_keys
                              if inferred_swing_source(swing) == expected_trigger_source
                              if swing.get("kind") in available_line_kinds
                              and isfinite(float(swing["price"]))]
        trigger_blocking_pairs = filtered_pairs_before_triggers(filtered_pairs, trigger_candidates)
        analysis["trigger_blocking_filtered_pairs"] = trigger_blocking_pairs
        if trigger_blocking_pairs:
            analysis["reasons"] = ["false_swing_pair_before_trigger"]
            return None, analysis
        if trigger_swings and not trigger_candidates:
            analysis["reasons"] = ["trigger_swings_filtered"]
            return None, analysis
        trigger_analysis = evaluate_trigger_swings(trigger_candidates, anchored, floating, atr)
        analysis["trigger_swings"] = trigger_analysis
        max_distance = float(rules["regression_trigger_max_distance_atr"])
        if trigger_analysis and not any(item["distance_atr"] <= max_distance for item in trigger_analysis):
            analysis["reasons"] = ["trigger_swing_too_far_from_regression"]
            return None, analysis
        if rules["regression_require_level_reaction"]:
            analysis["level_reaction_applied"] = True
            analysis["level_reaction_passed"] = False
            # The same trigger must pass both gates; two different swings
            # cannot each satisfy one half of the combined entry condition.
            close_trigger_keys = {swing_key(item) for item in trigger_analysis
                                  if item["distance_atr"] <= max_distance}
            reactions = evaluate_trigger_level_reactions(
                clean_swings,
                [swing for swing in trigger_candidates if swing_key(swing) in close_trigger_keys],
                side, current_price, float(analysis["level_atr"]), max(point_size, 1e-9), rules,
                level_swings=swings, candle_closes=candle_closes,
                level_window_start=analysis["level_window_start"],
                level_source_atrs=level_source_atrs,
            )
            analysis["trigger_level_reactions"] = reactions
            reaction = next((item for item in reactions if item["passed"]), None)
            if reaction is None:
                analysis["reasons"] = ["trigger_swing_without_level_reaction"]
                return None, analysis
            analysis["level_reaction_passed"] = True
            analysis["entry_level_reaction"] = reaction
    protective_kind = "low" if side == "buy" else "high"
    protective_line = anchored if anchored.kind == protective_kind else floating
    protective_price = float("nan") if protective_line is None else protective_line.price_at(bar_count - 1)

    def protective(price: float) -> bool:
        return isfinite(price) and price > 0.0 and (price < current_price if side == "buy" else price > current_price)

    source = "regression_line"
    quality = 0.0 if protective_line is None else min(1.0, max(0.0, protective_line.r_squared))
    if protective_line is None or rules["protective_level_source"] == "recent_swing" or not protective(protective_price):
        candidates = [swing for swing in clean_swings
                      if swing["kind"] == protective_kind and protective(float(swing["price"]))]
        if not candidates:
            analysis["reasons"] = ["no_protective_swing"]
            return None, analysis
        latest = max(candidates, key=lambda swing: int(swing["index"]))
        protective_price = float(latest["price"])
        quality = min(1.0, max(0.0, float(latest.get("strength", 0.0)) / 3.0))
        source = "recent_swing"
        analysis["protective_swing"] = dict(latest)
    buffer = atr * rules["protective_stop_buffer_atr"]
    analysis.update({"protective_level_price": protective_price, "protective_level_source": source,
                     "protective_stop_buffer": buffer, "reasons": ["regression_trend"]})
    return SwingRegressionDecision(side, abs(score), score, protective_price, quality, buffer), analysis
