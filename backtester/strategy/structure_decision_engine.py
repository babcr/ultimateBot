from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


LIMIT_ENTRY_STYLES = frozenset(
    {
        "support_bounce",
        "resistance_rejection",
        "resistance_breakout_retest",
        "support_breakdown_retest",
    }
)
STOP_ENTRY_STYLES = frozenset({"resistance_breakout", "support_breakdown"})


@dataclass(frozen=True)
class StructureDecision:
    side: str
    entry_style: str | None
    confidence: float
    rule_margin: float
    rr_ratio: float
    protective_level_price: float | None
    protective_level_quality: float
    reasons: tuple[str, ...]


DEFAULT_STRUCTURE_RULES = {
    # Coupe completement le decideur de structure s'il faut revenir au signal brut.
    "enabled": True,
    # Garde pour compatibilite de config. Le detecteur actuel de swings repose sur deux bougies opposees,
    # donc ce parametre n'influe plus vraiment sur les swings detectes.
    "pivot_window": 3,
    # Nombre de bougies fermees analysees pour extraire swings, tendance, supports et resistances.
    "level_lookback_bars": 700,
    # Nombre maximal de niveaux de support et de resistance conserves apres classement.
    "max_levels_per_side": 100,
    # Reglage principal du clustering: deux swings sont fusionnes dans le meme cluster si leur ecart
    # est inferieur ou egal a ATR * ce ratio. Le diminuer rend les niveaux plus nets et plus separes.
    "level_merge_atr_ratio": 0.001,
    # Plancher absolu du clustering en nombre de points instrument. Meme avec un ATR tres faible,
    # un cluster ne sera pas plus etroit que point_size * cette valeur. Le diminuer aussi si besoin.
    "level_merge_min_points": 2.0,
    # Distance max au prix courant pour qu'un niveau pese encore dans le biais directionnel global.
    "near_level_atr": 0.1,
    # Distance max au niveau pour considerer une entree bounce/rejection exploitable.
    "entry_distance_atr": 0.05,
    # Depassement minimal d'un niveau casse pour reconnaitre un breakout / breakdown valide.
    "breakout_distance_atr": 0.5,
    # Tolere qu'une cloture precedente soit legerement deja au-dessus/en-dessous du niveau casse.
    "breakout_cross_allowance_atr": 0.05,
    # Tolere qu'un retest deborde legerement le niveau casse avant reprise.
    "breakout_retest_tolerance_atr": 0.10,
    # Extension maximale acceptee apres cassure; au-dela on considere qu'on chasse trop le mouvement.
    "breakout_max_extension_atr": 1.5,
    # Qualite minimale d'un cluster pour qu'il survive comme niveau tradable.
    "min_level_quality": 0.65,
    # Nombre minimal de touches/swing-points dans un cluster pour le garder.
    "min_level_touches": 7,
    # Confiance minimale pour laisser un pattern chartiste influencer le biais final.
    "pattern_confidence_threshold": 0.65,
    # Seuil minimal du score directionnel pour sortir du mode sideways.
    "directional_bias_threshold": 0.2,
    # Qualite minimale pour accepter une entree de type support_bounce / resistance_rejection.
    "min_entry_quality": 0.55,
    # Qualite minimale pour accepter une entree breakout / breakdown, souvent plus exigeante.
    "min_breakout_quality": 0.65,
    # Reward/risk minimal exige une fois le stop structurel et la cible deduits.
    "min_rr_ratio": 1.4,
    # Source prioritaire du niveau protecteur: `structure_level` garde le stop sur le niveau qui porte
    # le setup (support/retest/resistance), `recent_swing` privilegie le dernier retournement local.
    "protective_level_source": "structure_level",
    # Distance minimale restante jusqu'a l'objectif structurel pour accepter l'entree.
    "min_target_distance_atr": 0.25,
    # Marge minimale buy-vs-sell pour sortir de l'indecision structurelle pure.
    "decision_margin_threshold": 0.12,
    # Exige ou non que la bougie hebdomadaire soit dans le meme sens que l'entree.
    "require_weekly_alignment": False,
    # Bonus directionnel apporte par l'alignement de la bougie hebdomadaire.
    "weekly_alignment_weight": 0.60,
    # Marge structurelle minimale pour que l'overlay confirme, veto ou inverse un signal amont.
    "rule_margin_threshold": 0.36,
    # Poids du decideur de structure dans la fusion finale avec le signal de base.
    "overlay_weight": 1.0,
    # Taille minimale du corps d'une bougie (vs range) pour compter comme vraie confirmation haussiere/baissiere.
    "confirmation_body_ratio": 0.68,
    # Taille maximale du corps (vs range) pour classer une bougie en doji.
    "doji_body_ratio": 0.18,
    # Autorise le decideur de structure a creer une entree meme si le signal amont est neutre.
    "allow_rule_entry_from_flat": True,
    # Autorise le decideur de structure a retourner un signal oppose au lieu de seulement le bloquer.
    "allow_rule_reversal": False,
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_protective_level_source(value: Any) -> str:
    normalized = str(value or "structure_level").strip().lower()
    if normalized in {"recent_swing", "swing", "recent_reversal", "latest_reversal"}:
        return "recent_swing"
    return "structure_level"


def _select_protective_level(
    structural_level: dict[str, Any] | None,
    recent_swing: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    preferred_source = _normalize_protective_level_source(config.get("protective_level_source"))
    if preferred_source == "recent_swing":
        if recent_swing is not None:
            return recent_swing, "recent_swing"
        if structural_level is not None:
            return structural_level, "structure_level"
    else:
        if structural_level is not None:
            return structural_level, "structure_level"
        if recent_swing is not None:
            return recent_swing, "recent_swing"
    return None, None


def resolve_structure_rules(config: dict[str, Any] | None) -> dict[str, Any]:
    # Les parametres *_atr sont des multiples d'ATR. `level_merge_min_points` est le plancher
    # absolu du clustering en points instrument, utile quand on veut resserrer les niveaux.
    merged = dict(DEFAULT_STRUCTURE_RULES)
    if isinstance(config, dict):
        for key, value in config.items():
            if value is not None:
                merged[key] = value
    merged["pivot_window"] = max(2, int(merged.get("pivot_window", 3)))
    merged["level_lookback_bars"] = max(60, int(merged.get("level_lookback_bars", 2000)))
    merged["max_levels_per_side"] = max(1, int(merged.get("max_levels_per_side", 3)))
    merged["level_merge_atr_ratio"] = max(0.0001, float(merged.get("level_merge_atr_ratio", 0.0001)))
    merged["level_merge_min_points"] = max(1.0, float(merged.get("level_merge_min_points", 8.0)))
    merged["near_level_atr"] = max(0.15, float(merged.get("near_level_atr", 0.65)))
    merged["entry_distance_atr"] = max(0.10, float(merged.get("entry_distance_atr", 0.45)))
    merged["breakout_distance_atr"] = max(0.05, float(merged.get("breakout_distance_atr", 0.18)))
    merged["breakout_cross_allowance_atr"] = max(0.0, float(merged.get("breakout_cross_allowance_atr", 0.05)))
    merged["breakout_retest_tolerance_atr"] = max(0.0, float(merged.get("breakout_retest_tolerance_atr", 0.20)))
    merged["breakout_max_extension_atr"] = max(0.2, float(merged.get("breakout_max_extension_atr", 0.85)))
    merged["min_level_quality"] = _clamp(float(merged.get("min_level_quality", 0.62)), 0.0, 1.0)
    merged["min_level_touches"] = max(1, int(merged.get("min_level_touches", 1)))
    merged["pattern_confidence_threshold"] = _clamp(float(merged.get("pattern_confidence_threshold", 0.65)), 0.0, 1.0)
    merged["directional_bias_threshold"] = _clamp(float(merged.get("directional_bias_threshold", 0.18)), 0.0, 1.0)
    merged["min_entry_quality"] = _clamp(float(merged.get("min_entry_quality", 0.65)), 0.0, 1.0)
    merged["min_breakout_quality"] = _clamp(float(merged.get("min_breakout_quality", 0.75)), 0.0, 1.0)
    merged["min_rr_ratio"] = max(0.5, float(merged.get("min_rr_ratio", 1.25)))
    merged["protective_level_source"] = _normalize_protective_level_source(merged.get("protective_level_source", "structure_level"))
    merged["min_target_distance_atr"] = max(0.0, float(merged.get("min_target_distance_atr", 0.70)))
    merged["decision_margin_threshold"] = _clamp(float(merged.get("decision_margin_threshold", 0.12)), 0.0, 1.0)
    merged["require_weekly_alignment"] = bool(merged.get("require_weekly_alignment", True))
    merged["weekly_alignment_weight"] = _clamp(float(merged.get("weekly_alignment_weight", 0.60)), 0.0, 2.0)
    merged["rule_margin_threshold"] = _clamp(float(merged.get("rule_margin_threshold", 0.36)), 0.0, 1.0)
    merged["overlay_weight"] = _clamp(float(merged.get("overlay_weight", 0.30)), 0.0, 1.0)
    merged["confirmation_body_ratio"] = _clamp(float(merged.get("confirmation_body_ratio", 0.38)), 0.15, 1.0)
    merged["doji_body_ratio"] = _clamp(float(merged.get("doji_body_ratio", 0.18)), 0.01, 0.40)
    merged["allow_rule_entry_from_flat"] = bool(merged.get("allow_rule_entry_from_flat", True))
    merged["allow_rule_reversal"] = bool(merged.get("allow_rule_reversal", False))
    merged["enabled"] = bool(merged.get("enabled", True))
    return merged


def _empty_analysis(current_price: float, atr: float) -> dict[str, Any]:
    return {
        "current_price": float(current_price),
        "atr": float(atr),
        "trend": "sideways",
        "trend_strength": 0.0,
        "trend_score": 0.0,
        "pattern": None,
        "pattern_direction": None,
        "pattern_confidence": 0.0,
        "pattern_confirmed": False,
        "support": None,
        "resistance": None,
        "nearest_support": None,
        "nearest_resistance": None,
        "structure_levels": [],
        "support_levels": [],
        "resistance_levels": [],
        "buy_setup": {
            "direction_ok": True,
            "entry_ok": False,
            "rr_ok": False,
            "setup_ok": False,
            "direction_score": 0.0,
            "quality": 0.0,
            "style": None,
            "rr_ratio": 0.0,
            "stop_distance_atr": 0.0,
            "protective_distance_atr": 0.0,
            "protective_level_price": None,
            "protective_level_quality": 0.0,
            "target_distance_atr": 0.0,
            "reasons": [],
            "blockers": [],
        },
        "sell_setup": {
            "direction_ok": True,
            "entry_ok": False,
            "rr_ok": False,
            "setup_ok": False,
            "direction_score": 0.0,
            "quality": 0.0,
            "style": None,
            "rr_ratio": 0.0,
            "stop_distance_atr": 0.0,
            "protective_distance_atr": 0.0,
            "protective_level_price": None,
            "protective_level_quality": 0.0,
            "target_distance_atr": 0.0,
            "reasons": [],
            "blockers": [],
        },
        "buy_score": 0.0,
        "sell_score": 0.0,
        "rule_signal": 1,
        "rule_bias": 0.0,
        "rule_margin": 0.0,
        "allow_buy": True,
        "allow_sell": True,
        "buy_blockers": [],
        "sell_blockers": [],
        "reasons": [],
    }


def _rolling_atr(frame: pd.DataFrame, point_size: float, atr_window: int) -> float:
    ranges = (frame["high"].astype(float) - frame["low"].astype(float)).clip(lower=0.0)
    if ranges.empty:
        return max(point_size * 8.0, 1e-9)
    # L'ATR structurel doit refleter la volatilite de toute la fenetre d'analyse, pas seulement 14 bougies.
    effective_window = max(1, min(int(atr_window), len(ranges)))
    atr_series = ranges.rolling(effective_window).mean()
    atr = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else float(ranges.tail(effective_window).mean())
    median_range = float(ranges.median()) if len(ranges) else 0.0
    return max(atr, median_range, point_size * 8.0, 1e-9)


def _normalize_swings(swings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    swings.sort(key=lambda item: (int(item["index"]), 0 if item["kind"] == "low" else 1))
    normalized: list[dict[str, Any]] = []
    for swing in swings:
        if not normalized:
            normalized.append(swing)
            continue
        previous = normalized[-1]
        if swing["kind"] != previous["kind"]:
            normalized.append(swing)
            continue
        if swing["kind"] == "high":
            if float(swing["price"]) >= float(previous["price"]):
                normalized[-1] = swing
        else:
            if float(swing["price"]) <= float(previous["price"]):
                normalized[-1] = swing
    return normalized


def _collect_swings(frame: pd.DataFrame, pivot_window: int, atr: float) -> list[dict[str, Any]]:
    # Le swing engine actuel detecte un pivot quand une bougie haussiere est suivie d'une baissiere
    # (swing high) ou l'inverse (swing low). `pivot_window` reste uniquement pour compatibilite.
    _ = pivot_window

    opens = frame["open"].to_numpy(dtype=np.float64)
    highs = frame["high"].to_numpy(dtype=np.float64)
    lows = frame["low"].to_numpy(dtype=np.float64)
    closes = frame["close"].to_numpy(dtype=np.float64)
    swings: list[dict[str, Any]] = []
    for index in range(1, len(frame)):
        previous_index = index - 1

        previous_open = float(opens[previous_index])
        previous_close = float(closes[previous_index])
        current_open = float(opens[index])
        current_close = float(closes[index])

        previous_bullish = previous_close > previous_open
        previous_bearish = previous_close < previous_open
        current_bullish = current_close > current_open
        current_bearish = current_close < current_open

        pair_high = max(float(highs[previous_index]), float(highs[index]))
        pair_low = min(float(lows[previous_index]), float(lows[index]))
        pair_strength = max((pair_high - pair_low) / max(atr, 1e-9), 0.0)

        if previous_bullish and current_bearish:
            swings.append({
                "kind": "high",
                "price": pair_high,
                "index": index,
                "strength": float(pair_strength),
            })
        if previous_bearish and current_bullish:
            swings.append({
                "kind": "low",
                "price": pair_low,
                "index": index,
                "strength": float(pair_strength),
            })
    return _normalize_swings(swings)


def collect_structure_swings(
    candles: pd.DataFrame,
    point_size: float,
    config: dict[str, Any] | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    rules = resolve_structure_rules(config)
    if candles is None or candles.empty:
        return []

    lookback = max(int(rules["level_lookback_bars"]), 500)
    frame = candles.copy().sort_values("time").tail(lookback).reset_index(drop=True)
    if len(frame) < 30:
        return []

    frame["time"] = pd.to_datetime(frame["time"])
    atr = _rolling_atr(frame, point_size, int(rules["level_lookback_bars"]))
    swings = _collect_swings(frame, int(rules["pivot_window"]), atr)
    if kind is None:
        return [dict(item) for item in swings]

    normalized_kind = str(kind).strip().lower()
    return [dict(item) for item in swings if str(item.get("kind")) == normalized_kind]


def merge_structure_reversal_swings(
    bid_candles: pd.DataFrame,
    ask_candles: pd.DataFrame,
    point_size: float,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    shared_swings = [
        {**swing, "source": "bid"}
        for swing in collect_structure_swings(
            candles=bid_candles,
            point_size=point_size,
            config=config,
            kind="low",
        )
    ]
    shared_swings.extend(
        {**swing, "source": "ask"}
        for swing in collect_structure_swings(
            candles=ask_candles,
            point_size=point_size,
            config=config,
            kind="high",
        )
    )
    return _normalize_swings(shared_swings)


def _latest_swing_level(
    swings: list[dict[str, Any]],
    kind: str,
    current_price: float,
    atr: float,
    prefer_below_price: bool,
) -> dict[str, Any] | None:
    candidates = [item for item in swings if str(item.get("kind")) == kind]
    if not candidates:
        return None

    if prefer_below_price:
        price_aligned = [item for item in candidates if float(item["price"]) <= current_price]
    else:
        price_aligned = [item for item in candidates if float(item["price"]) >= current_price]
    selected_pool = price_aligned or candidates

    latest = max(selected_pool, key=lambda item: int(item["index"]))
    strength = float(latest.get("strength", 0.0))
    quality = _clamp(strength / 3.0, 0.0, 1.0)
    return {
        "side": "swing",
        "kind": kind,
        "price": float(latest["price"]),
        "quality": float(quality),
        "avg_strength": float(strength),
        "latest_index": int(latest["index"]),
        "distance_atr": abs(current_price - float(latest["price"])) / max(atr, 1e-9),
    }


def _build_level_summary(cluster: dict[str, Any], current_price: float, atr: float, tolerance: float, max_age: int) -> dict[str, Any]:
    prices = [float(item["price"]) for item in cluster["points"]]
    strengths = [float(item["strength"]) for item in cluster["points"]]
    touches = len(prices)
    low_reactions = sum(1 for item in cluster["points"] if str(item.get("kind")) == "low")
    high_reactions = sum(1 for item in cluster["points"] if str(item.get("kind")) == "high")
    latest_index = max(int(item["index"]) for item in cluster["points"])
    spread = max(prices) - min(prices) if touches > 1 else 0.0
    touch_score = _clamp(touches / 4.0, 0.0, 1.0)
    strength_score = _clamp((sum(strengths) / max(touches, 1)) / 2.5, 0.0, 1.0)
    recency_score = _clamp(1.0 - ((max_age - latest_index) / max(max_age, 1)), 0.0, 1.0)
    tightness_score = 1.0 - _clamp(spread / max(tolerance * 2.0, 1e-9), 0.0, 1.0)
    quality = 0.35 * touch_score + 0.30 * strength_score + 0.20 * recency_score + 0.15 * tightness_score
    distance_atr = abs(current_price - float(cluster["price"])) / max(atr, 1e-9)
    return {
        "side": "level",
        "price": float(cluster["price"]),
        "touches": int(touches),
        "low_reactions": int(low_reactions),
        "high_reactions": int(high_reactions),
        "quality": float(_clamp(quality, 0.0, 1.0)),
        "avg_strength": float(sum(strengths) / max(touches, 1)),
        "latest_index": int(latest_index),
        "distance_atr": float(distance_atr),
    }


def _filter_trade_levels(levels: list[dict[str, Any]], min_touches: int, min_quality: float) -> list[dict[str, Any]]:
    return [
        level
        for level in levels
        if int(level["touches"]) >= min_touches and float(level["quality"]) >= min_quality
    ]


def _level_rank(level: dict[str, Any]) -> tuple[float, float]:
    return (float(level["quality"]) - 0.08 * float(level["distance_atr"]), -float(level["distance_atr"]))


def _cluster_levels(swings: list[dict[str, Any]], current_price: float, atr: float, tolerance: float, max_age: int) -> list[dict[str, Any]]:
    # Tous les swings high/low sont fusionnes dans les memes clusters: un niveau casse peut donc
    # naturellement changer de role plus tard sans etre duplique dans deux familles separees.
    candidates = [dict(item) for item in swings]
    if not candidates:
        return []

    candidates.sort(key=lambda item: float(item["price"]))
    clusters: list[dict[str, Any]] = []
    for candidate in candidates:
        weight = float(candidate["strength"]) + 1.0
        if not clusters or abs(float(candidate["price"]) - float(clusters[-1]["price"])) > tolerance:
            clusters.append({"price": float(candidate["price"]), "weight": weight, "points": [candidate]})
            continue
        cluster = clusters[-1]
        total_weight = float(cluster["weight"]) + weight
        cluster["price"] = (float(cluster["price"]) * float(cluster["weight"]) + float(candidate["price"]) * weight) / max(total_weight, 1e-9)
        cluster["weight"] = total_weight
        cluster["points"].append(candidate)

    levels = [_build_level_summary(cluster, current_price, atr, tolerance, max_age) for cluster in clusters]
    levels.sort(key=_level_rank, reverse=True)
    return levels


def build_structure_levels(
    level_swings: list[dict[str, Any]] | None,
    current_price: float,
    atr: float,
    point_size: float,
    config: dict[str, Any] | None = None,
    max_age: int | None = None,
) -> list[dict[str, Any]]:
    rules = resolve_structure_rules(config)
    normalized_swings = _normalize_swings(
        [dict(item) for item in (level_swings or []) if isinstance(item, dict)]
    )
    if len(normalized_swings) < 4:
        return []

    effective_max_age = max(
        int(max_age if max_age is not None else 0),
        max(int(item["index"]) for item in normalized_swings),
    )
    effective_atr = max(float(atr), point_size, 1e-9)
    tolerance = max(
        float(rules["level_merge_atr_ratio"]) * effective_atr,
        point_size * float(rules["level_merge_min_points"]),
    )
    return _filter_trade_levels(
        _cluster_levels(normalized_swings, current_price, effective_atr, tolerance, effective_max_age),
        int(rules["min_level_touches"]),
        float(rules["min_level_quality"]),
    )


def build_shared_structure_levels(
    bid_candles: pd.DataFrame,
    ask_candles: pd.DataFrame,
    point_size: float,
    config: dict[str, Any] | None = None,
    level_swings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rules = resolve_structure_rules(config)
    if bid_candles is None or bid_candles.empty or ask_candles is None or ask_candles.empty:
        return []

    lookback = max(int(rules["level_lookback_bars"]), 500)
    bid_frame = bid_candles.copy().sort_values("time").tail(lookback).reset_index(drop=True)
    ask_frame = ask_candles.copy().sort_values("time").tail(lookback).reset_index(drop=True)
    if len(bid_frame) < 30 or len(ask_frame) < 30:
        return []

    bid_frame["time"] = pd.to_datetime(bid_frame["time"])
    ask_frame["time"] = pd.to_datetime(ask_frame["time"])
    shared_current_price = (float(bid_frame["close"].iloc[-1]) + float(ask_frame["close"].iloc[-1])) / 2.0
    bid_atr = _rolling_atr(bid_frame, point_size, int(rules["level_lookback_bars"]))
    ask_atr = _rolling_atr(ask_frame, point_size, int(rules["level_lookback_bars"]))
    shared_atr = max((bid_atr + ask_atr) / 2.0, point_size, 1e-9)
    shared_swings = level_swings or merge_structure_reversal_swings(
        bid_candles=bid_frame,
        ask_candles=ask_frame,
        point_size=point_size,
        config=rules,
    )
    return build_structure_levels(
        level_swings=shared_swings,
        current_price=shared_current_price,
        atr=shared_atr,
        point_size=point_size,
        config=rules,
        max_age=min(len(bid_frame), len(ask_frame)) - 1,
    )


def _project_trade_levels(
    structure_levels: list[dict[str, Any]],
    current_price: float,
    tolerance: float,
    atr: float,
    max_levels: int,
    side: str,
) -> list[dict[str, Any]]:
    if side == "support":
        selected = [level for level in structure_levels if float(level["price"]) <= current_price + tolerance]
    else:
        selected = [level for level in structure_levels if float(level["price"]) >= current_price - tolerance]
    selected = [
        {
            **level,
            "distance_atr": abs(current_price - float(level["price"])) / max(atr, 1e-9),
        }
        for level in selected
    ]
    selected.sort(key=_level_rank, reverse=True)
    return [{**level, "side": side} for level in selected[:max_levels]]


def _pick_primary_level(levels: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not levels:
        return None
    return max(levels, key=lambda level: float(level["quality"]) - 0.10 * float(level["distance_atr"]))


def _detect_trend(frame: pd.DataFrame, swings: list[dict[str, Any]], atr: float) -> dict[str, float | str]:
    close = frame["close"].astype(float)
    if len(close) < 12:
        return {"direction": "sideways", "strength": 0.0, "score": 0.0}

    fast = close.ewm(span=8, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    baseline = close.ewm(span=55, adjust=False).mean()
    current = float(close.iloc[-1])
    fast_now = float(fast.iloc[-1])
    slow_now = float(slow.iloc[-1])
    baseline_now = float(baseline.iloc[-1])
    momentum_span = min(12, len(close) - 1)
    momentum_reference = float(close.iloc[-1 - momentum_span]) if momentum_span > 0 else current

    alignment = math.tanh((fast_now - slow_now) / max(atr * 1.5, 1e-9))
    location = math.tanh((current - baseline_now) / max(atr * 2.0, 1e-9))
    momentum = math.tanh((current - momentum_reference) / max(atr * 3.0, 1e-9))

    swing_score = 0.0
    recent_highs = [item for item in swings if item["kind"] == "high"][-2:]
    recent_lows = [item for item in swings if item["kind"] == "low"][-2:]
    if len(recent_highs) == 2 and len(recent_lows) == 2:
        high_delta = float(recent_highs[-1]["price"]) - float(recent_highs[-2]["price"])
        low_delta = float(recent_lows[-1]["price"]) - float(recent_lows[-2]["price"])
        swing_score = math.tanh((high_delta + low_delta) / max(atr * 2.0, 1e-9))

    score = 0.35 * alignment + 0.30 * location + 0.20 * momentum + 0.15 * swing_score
    if score > 0.18:
        direction = "bullish"
    elif score < -0.18:
        direction = "bearish"
    else:
        direction = "sideways"
    return {
        "direction": direction,
        "strength": float(_clamp(abs(score) * 1.6, 0.0, 1.0)),
        "score": float(score),
    }


def _line_value(x1: int, y1: float, x2: int, y2: float, x: int) -> float:
    if x2 == x1:
        return float(y2)
    slope = (y2 - y1) / float(x2 - x1)
    return float(y1 + slope * (x - x1))


def _detect_head_shoulders(swings: list[dict[str, Any]], current_index: int, current_close: float, atr: float, inverse: bool) -> dict[str, Any] | None:
    expected = ["low", "high", "low", "high", "low"] if inverse else ["high", "low", "high", "low", "high"]
    best: dict[str, Any] | None = None
    best_confidence = -1.0
    for start in range(len(swings) - 4):
        sequence = swings[start:start + 5]
        if [item["kind"] for item in sequence] != expected:
            continue
        if current_index - int(sequence[-1]["index"]) > 18:
            continue

        left = sequence[0]
        neck_left = sequence[1]
        head = sequence[2]
        neck_right = sequence[3]
        right = sequence[4]

        shoulder_gap = abs(float(left["price"]) - float(right["price"]))
        neckline_gap = abs(float(neck_left["price"]) - float(neck_right["price"]))
        left_width = max(int(head["index"]) - int(left["index"]), 1)
        right_width = max(int(right["index"]) - int(head["index"]), 1)
        timing_balance = 1.0 - _clamp(abs(left_width - right_width) / max(left_width + right_width, 1), 0.0, 1.0)
        shoulder_balance = 1.0 - _clamp(shoulder_gap / max(atr * 2.0, 1e-9), 0.0, 1.0)
        neckline_balance = 1.0 - _clamp(neckline_gap / max(atr * 2.0, 1e-9), 0.0, 1.0)
        neckline_now = _line_value(int(neck_left["index"]), float(neck_left["price"]), int(neck_right["index"]), float(neck_right["price"]), current_index)

        if inverse:
            head_margin = min(float(left["price"]), float(right["price"])) - float(head["price"])
            if head_margin < atr * 0.6 or shoulder_gap > atr * 1.2:
                continue
            confirmation_margin = current_close - neckline_now
            direction = "buy"
            name = "inverse_head_and_shoulders"
        else:
            head_margin = float(head["price"]) - max(float(left["price"]), float(right["price"]))
            if head_margin < atr * 0.6 or shoulder_gap > atr * 1.2:
                continue
            confirmation_margin = neckline_now - current_close
            direction = "sell"
            name = "head_and_shoulders"

        recency = 1.0 - _clamp((current_index - int(right["index"])) / 18.0, 0.0, 1.0)
        prominence = _clamp(head_margin / max(atr * 2.5, 1e-9), 0.0, 1.0)
        confirmation = _clamp(confirmation_margin / max(atr * 1.5, 1e-9), 0.0, 1.0)
        confidence = 0.30 * prominence + 0.20 * shoulder_balance + 0.15 * neckline_balance + 0.20 * timing_balance + 0.15 * recency
        confidence += 0.10 * confirmation
        confidence = _clamp(confidence, 0.0, 1.0)
        confirmed = confirmation_margin > atr * 0.15

        candidate = {
            "name": name,
            "direction": direction,
            "confidence": float(confidence),
            "confirmed": bool(confirmed),
            "status": "confirmed" if confirmed else "forming",
            "neckline": float(neckline_now),
            "completion_index": int(right["index"]),
        }
        if float(candidate["confidence"]) > best_confidence:
            best = candidate
            best_confidence = float(candidate["confidence"])
    return best


def _detect_zeus_lightning(swings: list[dict[str, Any]], current_index: int, current_close: float, atr: float, bullish: bool) -> dict[str, Any] | None:
    expected = ["low", "high", "low", "high"] if bullish else ["high", "low", "high", "low"]
    best: dict[str, Any] | None = None
    best_confidence = -1.0
    for start in range(len(swings) - 3):
        sequence = swings[start:start + 4]
        if [item["kind"] for item in sequence] != expected:
            continue
        if current_index - int(sequence[-1]["index"]) > 14:
            continue

        a, b, c, d = sequence
        if bullish:
            first_leg = float(b["price"]) - float(a["price"])
            retrace = float(b["price"]) - float(c["price"])
            second_leg = float(d["price"]) - float(c["price"])
            breakout_margin = current_close - float(b["price"])
            confirmed = current_close >= float(b["price"]) - atr * 0.10 or float(d["price"]) > float(b["price"])
            direction = "buy"
        else:
            first_leg = float(a["price"]) - float(b["price"])
            retrace = float(c["price"]) - float(b["price"])
            second_leg = float(c["price"]) - float(d["price"])
            breakout_margin = float(b["price"]) - current_close
            confirmed = current_close <= float(b["price"]) + atr * 0.10 or float(d["price"]) < float(b["price"])
            direction = "sell"

        if first_leg < atr * 1.8 or second_leg < atr * 1.2:
            continue
        retrace_ratio = retrace / max(first_leg, 1e-9)
        if retrace_ratio < 0.20 or retrace_ratio > 0.75:
            continue

        impulse_1 = _clamp(first_leg / max(atr * 3.0, 1e-9), 0.0, 1.0)
        impulse_2 = _clamp(second_leg / max(atr * 2.5, 1e-9), 0.0, 1.0)
        retrace_fit = 1.0 - _clamp(abs(retrace_ratio - 0.5) / 0.3, 0.0, 1.0)
        timing_balance = 1.0 - _clamp(abs((int(b["index"]) - int(a["index"])) - (int(d["index"]) - int(c["index"]))) / max(current_index, 1), 0.0, 1.0)
        breakout = _clamp(breakout_margin / max(atr * 1.2, 1e-9), 0.0, 1.0)
        confidence = 0.25 * impulse_1 + 0.25 * impulse_2 + 0.20 * retrace_fit + 0.10 * timing_balance + 0.20 * breakout
        confidence = _clamp(confidence, 0.0, 1.0)

        candidate = {
            "name": "zeus_lightning",
            "direction": direction,
            "confidence": float(confidence),
            "confirmed": bool(confirmed),
            "status": "confirmed" if confirmed else "forming",
            "completion_index": int(d["index"]),
        }
        if float(candidate["confidence"]) > best_confidence:
            best = candidate
            best_confidence = float(candidate["confidence"])
    return best


def _detect_patterns(swings: list[dict[str, Any]], current_index: int, current_close: float, atr: float) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = [
        _detect_head_shoulders(swings, current_index, current_close, atr, inverse=False),
        _detect_head_shoulders(swings, current_index, current_close, atr, inverse=True),
        _detect_zeus_lightning(swings, current_index, current_close, atr, bullish=True),
        _detect_zeus_lightning(swings, current_index, current_close, atr, bullish=False),
    ]
    patterns = [candidate for candidate in candidates if candidate is not None]
    patterns.sort(key=lambda item: (bool(item.get("confirmed", False)), float(item.get("confidence", 0.0))), reverse=True)
    active = patterns[0] if patterns else None
    return active, patterns


def _nearest_level(levels: list[dict[str, Any]], current_price: float) -> dict[str, Any] | None:
    if not levels:
        return None
    return min(levels, key=lambda level: abs(float(level["price"]) - current_price))


def _closest_level_below(levels: list[dict[str, Any]], current_price: float) -> dict[str, Any] | None:
    candidates = [level for level in levels if float(level["price"]) <= current_price]
    if not candidates:
        return None
    return max(candidates, key=lambda level: float(level["price"]))


def _closest_level_above(levels: list[dict[str, Any]], current_price: float) -> dict[str, Any] | None:
    candidates = [level for level in levels if float(level["price"]) >= current_price]
    if not candidates:
        return None
    return min(candidates, key=lambda level: float(level["price"]))


def _candle_context(frame: pd.DataFrame, atr: float, config: dict[str, Any]) -> dict[str, float]:
    last = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else last
    open_price = float(last["open"])
    close_price = float(last["close"])
    high_price = float(last["high"])
    low_price = float(last["low"])
    previous_open = float(previous["open"])
    previous_close = float(previous["close"])
    previous_high = float(previous["high"])
    previous_low = float(previous["low"])
    doji_body_ratio = float(config["doji_body_ratio"])
    confirmation_body_ratio = float(config["confirmation_body_ratio"])
    total_range = max(high_price - low_price, 1e-9)
    previous_total_range = max(previous_high - previous_low, 1e-9)
    body = close_price - open_price
    previous_body = previous_close - previous_open
    body_ratio = abs(body) / total_range
    previous_body_ratio = abs(previous_body) / previous_total_range
    upper_wick = max(high_price - max(open_price, close_price), 0.0) / total_range
    lower_wick = max(min(open_price, close_price) - low_price, 0.0) / total_range
    close_position = _clamp((close_price - low_price) / total_range, 0.0, 1.0)
    move_atr = (close_price - previous_close) / max(atr, 1e-9)
    bullish_rejection = _clamp(0.45 * lower_wick + 0.30 * max(close_position - 0.5, 0.0) * 2.0 + 0.25 * body_ratio * float(body > 0.0), 0.0, 1.0)
    bearish_rejection = _clamp(0.45 * upper_wick + 0.30 * max(0.5 - close_position, 0.0) * 2.0 + 0.25 * body_ratio * float(body < 0.0), 0.0, 1.0)
    previous_is_doji = float(previous_body_ratio <= doji_body_ratio)
    positive_candle = float(body > 0.0 and body_ratio >= confirmation_body_ratio and close_position >= 0.55)
    negative_candle = float(body < 0.0 and body_ratio >= confirmation_body_ratio and close_position <= 0.45)
    bullish_engulfing = 0.0
    bearish_engulfing = 0.0
    doji_bullish_confirmation = 0.0
    doji_bearish_confirmation = 0.0
    bullish_confirmation = float(positive_candle)
    bearish_confirmation = float(negative_candle)
    return {
        "close": close_price,
        "previous_close": previous_close,
        "high": high_price,
        "low": low_price,
        "body_ratio": float(body_ratio),
        "close_position": float(close_position),
        "bullish_rejection": float(bullish_rejection),
        "bearish_rejection": float(bearish_rejection),
        "up_impulse": float(_clamp(move_atr / 1.25, 0.0, 1.0)),
        "down_impulse": float(_clamp(-move_atr / 1.25, 0.0, 1.0)),
        "is_bullish": float(body > 0.0),
        "is_bearish": float(body < 0.0),
        "previous_is_doji": float(previous_is_doji),
        "positive_candle": float(positive_candle),
        "negative_candle": float(negative_candle),
        "bullish_engulfing": float(bullish_engulfing),
        "bearish_engulfing": float(bearish_engulfing),
        "doji_bullish_confirmation": float(doji_bullish_confirmation),
        "doji_bearish_confirmation": float(doji_bearish_confirmation),
        "bullish_confirmation": float(bullish_confirmation),
        "bearish_confirmation": float(bearish_confirmation),
    }


def _aggregate_candles_by_timeframe(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    working = frame.copy().sort_values("time").reset_index(drop=True)
    timestamps = pd.to_datetime(working["time"], utc=True).dt.tz_localize(None)
    if timeframe == "D":
        buckets = timestamps.dt.floor("D")
    elif timeframe == "W":
        buckets = timestamps.dt.to_period("W-SUN").dt.start_time
    else:
        raise ValueError(f"Unsupported timeframe {timeframe}")

    aggregated = (
        working.assign(_bucket=buckets)
        .groupby("_bucket", as_index=False, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .rename(columns={"_bucket": "time"})
    )
    return aggregated[["time", "open", "high", "low", "close"]]


def _higher_timeframe_candle_bias(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float | str]:
    if frame.empty:
        return {
            "direction": "sideways",
            "strength": 0.0,
            "body_ratio": 0.0,
            "close_position": 0.5,
        }

    last = frame.iloc[-1]
    open_price = float(last["open"])
    close_price = float(last["close"])
    high_price = float(last["high"])
    low_price = float(last["low"])
    total_range = max(high_price - low_price, 1e-9)
    body = close_price - open_price
    body_ratio = abs(body) / total_range
    close_position = _clamp((close_price - low_price) / total_range, 0.0, 1.0)
    if body_ratio <= float(config["doji_body_ratio"]):
        direction = "sideways"
        strength = 0.0
    elif body > 0.0:
        direction = "bullish"
        strength = _clamp(0.70 * body_ratio + 0.30 * close_position, 0.0, 1.0)
    else:
        direction = "bearish"
        strength = _clamp(0.70 * body_ratio + 0.30 * (1.0 - close_position), 0.0, 1.0)
    return {
        "direction": direction,
        "strength": float(strength),
        "body_ratio": float(body_ratio),
        "close_position": float(close_position),
    }


def _estimate_rr(
    side: str,
    current_price: float,
    atr: float,
    protective_level: dict[str, Any] | None,
    target_level: dict[str, Any] | None,
) -> tuple[float, float, float]:
    if side == "buy":
        stop_distance = max(current_price - float(protective_level["price"]), atr * 0.35) if protective_level is not None else atr
        target_distance = float(target_level["price"]) - current_price if target_level is not None else atr * 2.0
    else:
        stop_distance = max(float(protective_level["price"]) - current_price, atr * 0.35) if protective_level is not None else atr
        target_distance = current_price - float(target_level["price"]) if target_level is not None else atr * 2.0
    if target_distance <= 0.0:
        target_distance = atr * 2.0
    rr_ratio = target_distance / max(stop_distance, 1e-9)
    return float(rr_ratio), float(stop_distance / max(atr, 1e-9)), float(target_distance / max(atr, 1e-9))


def _evaluate_direction_setup(
    side: str,
    trend: dict[str, float | str],
    pattern: dict[str, Any] | None,
    daily_bias: dict[str, float | str],
    weekly_bias: dict[str, float | str],
    config: dict[str, Any],
) -> dict[str, Any]:
    side_sign = 1.0 if side == "buy" else -1.0
    expected_direction = "bullish" if side == "buy" else "bearish"
    score = side_sign * float(trend["score"])
    blockers: list[str] = []
    reasons: list[str] = []
    if score > 0.0:
        reasons.append("trend_alignment")

    _ = daily_bias
    weekly_direction = str(weekly_bias.get("direction", "sideways"))
    if weekly_direction != expected_direction:
        if bool(config.get("require_weekly_alignment", True)):
            blockers.append("weekly_candle_mismatch")
    else:
        weekly_alignment_weight = float(config.get("weekly_alignment_weight", 0.60))
        score += weekly_alignment_weight * float(weekly_bias.get("strength", 0.0))
        if weekly_alignment_weight > 0.0:
            reasons.append("weekly_candle_alignment")

    if pattern is not None and float(pattern.get("confidence", 0.0)) >= float(config["pattern_confidence_threshold"]):
        pattern_side = 1.0 if str(pattern.get("direction")) == "buy" else -1.0
        pattern_score = side_sign * pattern_side * float(pattern["confidence"])
        if bool(pattern.get("confirmed", False)):
            pattern_score *= 1.15
        score += 0.65 * pattern_score
        reasons.append(f"pattern_{pattern['name']}")
        if pattern_score < -0.7:
            blockers.append("opposite_confirmed_pattern")

    if score < -float(config["directional_bias_threshold"]):
        blockers.append("opposite_trend_bias")

    return {
        "direction_score": float(score),
        "direction_ok": not blockers,
        "reasons": reasons,
        "blockers": blockers,
    }


def _evaluate_buy_entry(
    current_price: float,
    atr: float,
    candle: dict[str, float],
    floor_support: dict[str, Any] | None,
    broken_resistance: dict[str, Any] | None,
    upside_resistance: dict[str, Any] | None,
    last_swing_low: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    entry_distance_atr = float(config["entry_distance_atr"])
    breakout_distance_atr = float(config["breakout_distance_atr"])
    breakout_cross_allowance_atr = float(config["breakout_cross_allowance_atr"])
    breakout_retest_tolerance_atr = float(config["breakout_retest_tolerance_atr"])
    breakout_max_extension_atr = float(config["breakout_max_extension_atr"])
    min_entry_quality = float(config["min_entry_quality"])
    min_breakout_quality = float(config["min_breakout_quality"])
    min_rr_ratio = float(config["min_rr_ratio"])
    min_target_distance_atr = float(config["min_target_distance_atr"])

    best_quality = 0.0
    best_style: str | None = None
    structural_level: dict[str, Any] | None = None
    reasons: list[str] = []
    blockers: list[str] = []
    target_level = upside_resistance
    candle_confirm = float(candle["bullish_confirmation"])
    if candle_confirm <= 0.0:
        blockers.append("no_bullish_confirmation")
    else:
        reasons.append("bullish_candle_confirmation")

    if floor_support is not None:
        support_distance = max(current_price - float(floor_support["price"]), 0.0) / max(atr, 1e-9)
        if support_distance <= entry_distance_atr:
            proximity = 1.0 - _clamp(support_distance / max(entry_distance_atr, 1e-9), 0.0, 1.0)
            bounce_quality = 0.35 * proximity + 0.35 * float(floor_support["quality"]) + 0.30 * candle_confirm
            if bounce_quality >= min_entry_quality:
                best_quality = float(bounce_quality)
                best_style = "support_bounce"
                structural_level = floor_support
                reasons.append("support_entry")

    if broken_resistance is not None:
        clearance = (current_price - float(broken_resistance["price"])) / max(atr, 1e-9)
        previous_clearance = (float(candle["previous_close"]) - float(broken_resistance["price"])) / max(atr, 1e-9)
        crossed = previous_clearance <= breakout_cross_allowance_atr and clearance >= breakout_distance_atr
        retest = (
            float(candle["low"]) <= float(broken_resistance["price"]) + breakout_retest_tolerance_atr * atr
            and current_price >= float(broken_resistance["price"])
        )
        if clearance > breakout_max_extension_atr:
            blockers.append("chasing_breakout")
        if (crossed or retest) and clearance <= breakout_max_extension_atr:
            breakout_strength = _clamp(clearance / max(breakout_distance_atr * 2.0, 1e-9), 0.0, 1.0)
            retest_bonus = 1.0 if retest else 0.55
            breakout_quality = 0.25 * float(broken_resistance["quality"]) + 0.25 * breakout_strength + 0.20 * candle_confirm + 0.30 * retest_bonus
            if breakout_quality > best_quality and breakout_quality >= min_breakout_quality:
                best_quality = float(breakout_quality)
                best_style = "resistance_breakout_retest" if retest else "resistance_breakout"
                structural_level = broken_resistance
                reasons.append("breakout_entry")

    protective_level, protective_level_source = _select_protective_level(structural_level, last_swing_low, config)
    rr_ratio, stop_distance_atr, target_distance_atr = _estimate_rr("buy", current_price, atr, protective_level, target_level)
    protective_distance_atr = (
        max(current_price - float(protective_level["price"]), 0.0) / max(atr, 1e-9)
        if protective_level is not None
        else 0.0
    )
    if protective_level is None:
        blockers.append("no_low_swing_stop")
    if target_level is not None and target_distance_atr < min_target_distance_atr:
        blockers.append("resistance_too_close")
    rr_ok = rr_ratio >= min_rr_ratio
    if not rr_ok:
        blockers.append("poor_rr")
    entry_ok = best_style is not None and not blockers and rr_ok
    return {
        "entry_ok": bool(entry_ok),
        "quality": float(best_quality),
        "style": best_style,
        "rr_ratio": float(rr_ratio),
        "stop_distance_atr": float(stop_distance_atr),
        "protective_distance_atr": float(protective_distance_atr),
        "protective_level_price": float(protective_level["price"]) if protective_level is not None else None,
        "protective_level_quality": float(protective_level.get("quality", 0.0)) if protective_level is not None else 0.0,
        "protective_level_source": protective_level_source,
        "target_distance_atr": float(target_distance_atr),
        "reasons": reasons,
        "blockers": blockers,
        "rr_ok": bool(rr_ok),
    }


def _evaluate_sell_entry(
    current_price: float,
    atr: float,
    candle: dict[str, float],
    ceiling_resistance: dict[str, Any] | None,
    broken_support: dict[str, Any] | None,
    downside_support: dict[str, Any] | None,
    last_swing_high: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    entry_distance_atr = float(config["entry_distance_atr"])
    breakout_distance_atr = float(config["breakout_distance_atr"])
    breakout_cross_allowance_atr = float(config["breakout_cross_allowance_atr"])
    breakout_retest_tolerance_atr = float(config["breakout_retest_tolerance_atr"])
    breakout_max_extension_atr = float(config["breakout_max_extension_atr"])
    min_entry_quality = float(config["min_entry_quality"])
    min_breakout_quality = float(config["min_breakout_quality"])
    min_rr_ratio = float(config["min_rr_ratio"])
    min_target_distance_atr = float(config["min_target_distance_atr"])

    best_quality = 0.0
    best_style: str | None = None
    structural_level: dict[str, Any] | None = None
    reasons: list[str] = []
    blockers: list[str] = []
    target_level = downside_support
    candle_confirm = float(candle["bearish_confirmation"])
    if candle_confirm <= 0.0:
        blockers.append("no_bearish_confirmation")
    else:
        reasons.append("bearish_candle_confirmation")

    if ceiling_resistance is not None:
        resistance_distance = max(float(ceiling_resistance["price"]) - current_price, 0.0) / max(atr, 1e-9)
        if resistance_distance <= entry_distance_atr:
            proximity = 1.0 - _clamp(resistance_distance / max(entry_distance_atr, 1e-9), 0.0, 1.0)
            rejection_quality = 0.35 * proximity + 0.35 * float(ceiling_resistance["quality"]) + 0.30 * candle_confirm
            if rejection_quality >= min_entry_quality:
                best_quality = float(rejection_quality)
                best_style = "resistance_rejection"
                structural_level = ceiling_resistance
                reasons.append("resistance_entry")

    if broken_support is not None:
        clearance = (float(broken_support["price"]) - current_price) / max(atr, 1e-9)
        previous_clearance = (float(broken_support["price"]) - float(candle["previous_close"])) / max(atr, 1e-9)
        crossed = previous_clearance <= breakout_cross_allowance_atr and clearance >= breakout_distance_atr
        retest = (
            float(candle["high"]) >= float(broken_support["price"]) - breakout_retest_tolerance_atr * atr
            and current_price <= float(broken_support["price"])
        )
        if clearance > breakout_max_extension_atr:
            blockers.append("chasing_breakdown")
        if (crossed or retest) and clearance <= breakout_max_extension_atr:
            breakdown_strength = _clamp(clearance / max(breakout_distance_atr * 2.0, 1e-9), 0.0, 1.0)
            retest_bonus = 1.0 if retest else 0.55
            breakdown_quality = 0.25 * float(broken_support["quality"]) + 0.25 * breakdown_strength + 0.20 * candle_confirm + 0.30 * retest_bonus
            if breakdown_quality > best_quality and breakdown_quality >= min_breakout_quality:
                best_quality = float(breakdown_quality)
                best_style = "support_breakdown_retest" if retest else "support_breakdown"
                structural_level = broken_support
                reasons.append("breakdown_entry")

    protective_level, protective_level_source = _select_protective_level(structural_level, last_swing_high, config)
    rr_ratio, stop_distance_atr, target_distance_atr = _estimate_rr("sell", current_price, atr, protective_level, target_level)
    protective_distance_atr = (
        max(float(protective_level["price"]) - current_price, 0.0) / max(atr, 1e-9)
        if protective_level is not None
        else 0.0
    )
    if protective_level is None:
        blockers.append("no_high_swing_stop")
    if target_level is not None and target_distance_atr < min_target_distance_atr:
        blockers.append("support_too_close")
    rr_ok = rr_ratio >= min_rr_ratio
    if not rr_ok:
        blockers.append("poor_rr")
    entry_ok = best_style is not None and not blockers and rr_ok
    return {
        "entry_ok": bool(entry_ok),
        "quality": float(best_quality),
        "style": best_style,
        "rr_ratio": float(rr_ratio),
        "stop_distance_atr": float(stop_distance_atr),
        "protective_distance_atr": float(protective_distance_atr),
        "protective_level_price": float(protective_level["price"]) if protective_level is not None else None,
        "protective_level_quality": float(protective_level.get("quality", 0.0)) if protective_level is not None else 0.0,
        "protective_level_source": protective_level_source,
        "target_distance_atr": float(target_distance_atr),
        "reasons": reasons,
        "blockers": blockers,
        "rr_ok": bool(rr_ok),
    }


def _combine_rule_and_entry(rule_bias: dict[str, Any], buy_setup: dict[str, Any], sell_setup: dict[str, Any]) -> dict[str, Any]:
    decision_margin_threshold = float(rule_bias.get("decision_margin_threshold", 0.12))
    buy_blockers = list(rule_bias["buy_blockers"]) + list(buy_setup["blockers"])
    sell_blockers = list(rule_bias["sell_blockers"]) + list(sell_setup["blockers"])

    allow_buy = bool(rule_bias["allow_buy"]) and bool(buy_setup["direction_ok"]) and bool(buy_setup["entry_ok"]) and bool(buy_setup["rr_ok"])
    allow_sell = bool(rule_bias["allow_sell"]) and bool(sell_setup["direction_ok"]) and bool(sell_setup["entry_ok"]) and bool(sell_setup["rr_ok"])

    buy_score = float(rule_bias["buy_score"]) + 0.65 * float(buy_setup["quality"]) + 0.20 * max(float(buy_setup["direction_score"]), 0.0)
    sell_score = float(rule_bias["sell_score"]) + 0.65 * float(sell_setup["quality"]) + 0.20 * max(float(sell_setup["direction_score"]), 0.0)
    if not allow_buy:
        buy_score *= 0.45
    if not allow_sell:
        sell_score *= 0.45

    total_score = max(buy_score + sell_score, 1.0)
    rule_bias_value = (buy_score - sell_score) / total_score
    rule_margin = abs(rule_bias_value)
    if buy_score > sell_score and allow_buy and rule_margin >= decision_margin_threshold:
        rule_signal = 2
    elif sell_score > buy_score and allow_sell and rule_margin >= decision_margin_threshold:
        rule_signal = 0
    else:
        rule_signal = 1

    combined_reasons = list(dict.fromkeys(list(rule_bias["reasons"]) + list(buy_setup["reasons"]) + list(sell_setup["reasons"])))
    return {
        "buy_score": float(buy_score),
        "sell_score": float(sell_score),
        "rule_signal": int(rule_signal),
        "rule_bias": float(rule_bias_value),
        "rule_margin": float(rule_margin),
        "allow_buy": bool(allow_buy),
        "allow_sell": bool(allow_sell),
        "buy_blockers": buy_blockers,
        "sell_blockers": sell_blockers,
        "reasons": combined_reasons,
    }


def _build_side_setup(direction: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    blockers = list(dict.fromkeys(list(direction["blockers"]) + list(entry["blockers"])))
    reasons = list(dict.fromkeys(list(direction["reasons"]) + list(entry["reasons"])))
    direction_ok = bool(direction["direction_ok"])
    if (
        not direction_ok
        and "opposite_confirmed_pattern" not in direction["blockers"]
        and bool(entry["entry_ok"])
        and bool(entry["rr_ok"])
        and float(entry["quality"]) >= 0.68
        and float(direction["direction_score"]) >= -0.20
    ):
        direction_ok = True
        blockers = [blocker for blocker in blockers if blocker != "opposite_trend_bias"]
        reasons = list(dict.fromkeys(reasons + ["entry_override"]))

    setup_ok = direction_ok and bool(entry["entry_ok"]) and bool(entry["rr_ok"]) and not blockers
    return {
        **direction,
        **entry,
        "direction_ok": bool(direction_ok),
        "setup_ok": bool(setup_ok),
        "reasons": reasons,
        "blockers": blockers,
    }


def _evaluate_rule_bias(
    trend: dict[str, float | str],
    support: dict[str, Any] | None,
    resistance: dict[str, Any] | None,
    pattern: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    near_level_atr = float(config["near_level_atr"])
    min_level_quality = float(config["min_level_quality"])
    pattern_threshold = float(config["pattern_confidence_threshold"])
    decision_margin_threshold = float(config["decision_margin_threshold"])

    buy_score = 0.0
    sell_score = 0.0
    reasons: list[str] = []
    buy_blockers: list[str] = []
    sell_blockers: list[str] = []

    direction = str(trend["direction"])
    strength = float(trend["strength"])
    if direction == "bullish":
        buy_score += 0.9 + 0.7 * strength
        reasons.append("bullish_trend")
    elif direction == "bearish":
        sell_score += 0.9 + 0.7 * strength
        reasons.append("bearish_trend")
    else:
        buy_score += 0.15
        sell_score += 0.15

    if support is not None:
        support_quality = float(support["quality"])
        support_distance = float(support["distance_atr"])
        if support_quality >= min_level_quality and support_distance <= near_level_atr:
            proximity = 1.0 - _clamp(support_distance / max(near_level_atr, 1e-9), 0.0, 1.0)
            buy_score += 0.9 * support_quality * proximity
            reasons.append("support_rebound")
        if support_quality >= 0.72 and support_distance <= near_level_atr * 0.70 and direction != "bullish":
            sell_blockers.append("strong_support")

    if resistance is not None:
        resistance_quality = float(resistance["quality"])
        resistance_distance = float(resistance["distance_atr"])
        if resistance_quality >= min_level_quality and resistance_distance <= near_level_atr:
            proximity = 1.0 - _clamp(resistance_distance / max(near_level_atr, 1e-9), 0.0, 1.0)
            sell_score += 0.9 * resistance_quality * proximity
            reasons.append("resistance_rejection")
        if resistance_quality >= 0.72 and resistance_distance <= near_level_atr * 0.70 and direction != "bearish":
            buy_blockers.append("strong_resistance")

    if pattern is not None and float(pattern.get("confidence", 0.0)) >= pattern_threshold:
        confidence = float(pattern["confidence"])
        direction = str(pattern["direction"])
        if direction == "buy":
            buy_score += (1.15 if pattern["name"] == "inverse_head_and_shoulders" else 0.95) * confidence
            reasons.append(str(pattern["name"]))
            if bool(pattern.get("confirmed", False)):
                sell_blockers.append(str(pattern["name"]))
        elif direction == "sell":
            sell_score += (1.15 if pattern["name"] == "head_and_shoulders" else 0.95) * confidence
            reasons.append(str(pattern["name"]))
            if bool(pattern.get("confirmed", False)):
                buy_blockers.append(str(pattern["name"]))

    total_score = max(buy_score + sell_score, 1.0)
    rule_bias = (buy_score - sell_score) / total_score
    rule_margin = abs(rule_bias)

    if buy_score > sell_score and not buy_blockers and rule_margin >= decision_margin_threshold:
        rule_signal = 2
    elif sell_score > buy_score and not sell_blockers and rule_margin >= decision_margin_threshold:
        rule_signal = 0
    else:
        rule_signal = 1

    return {
        "buy_score": float(buy_score),
        "sell_score": float(sell_score),
        "rule_signal": int(rule_signal),
        "rule_bias": float(rule_bias),
        "rule_margin": float(rule_margin),
        "decision_margin_threshold": float(decision_margin_threshold),
        "allow_buy": not buy_blockers,
        "allow_sell": not sell_blockers,
        "buy_blockers": buy_blockers,
        "sell_blockers": sell_blockers,
        "reasons": reasons,
    }


def analyze_market_structure(
    candles: pd.DataFrame,
    point_size: float,
    config: dict[str, Any] | None = None,
    level_swings: list[dict[str, Any]] | None = None,
    structure_levels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rules = resolve_structure_rules(config)
    if candles is None or candles.empty:
        return _empty_analysis(0.0, max(point_size, 1e-9))

    lookback = max(int(rules["level_lookback_bars"]), 500)
    frame = candles.copy().sort_values("time").tail(lookback).reset_index(drop=True)
    if len(frame) < 30:
        current_price = float(frame["close"].iloc[-1]) if not frame.empty else 0.0
        return _empty_analysis(current_price, max(point_size, 1e-9))

    frame["time"] = pd.to_datetime(frame["time"])
    current_price = float(frame["close"].iloc[-1])
    atr = _rolling_atr(frame, point_size, int(rules["level_lookback_bars"]))
    swings = _collect_swings(frame, int(rules["pivot_window"]), atr)
    if level_swings is None:
        level_swings = swings
    else:
        level_swings = _normalize_swings(
            [
                dict(item)
                for item in level_swings
                if isinstance(item, dict) and 0 <= int(item.get("index", -1)) < len(frame)
            ]
        )
    if len(swings) < 4:
        return _empty_analysis(current_price, atr)
    if structure_levels is None:
        if len(level_swings) < 4:
            level_swings = swings
        structure_levels = build_structure_levels(
            level_swings=level_swings,
            current_price=current_price,
            atr=atr,
            point_size=point_size,
            config=rules,
            max_age=len(frame) - 1,
        )
    else:
        structure_levels = [dict(level) for level in structure_levels if isinstance(level, dict)]

    # Largeur effective d'un cluster de niveau. Pour resserrer les supports/resistances, diminuer
    # d'abord `level_merge_atr_ratio`, puis `level_merge_min_points` si le plancher en points bloque.
    tolerance = max(
        float(rules["level_merge_atr_ratio"]) * atr,
        point_size * float(rules["level_merge_min_points"]),
    )
    support_levels = _project_trade_levels(
        structure_levels,
        current_price,
        tolerance,
        atr,
        int(rules["max_levels_per_side"]),
        "support",
    )
    resistance_levels = _project_trade_levels(
        structure_levels,
        current_price,
        tolerance,
        atr,
        int(rules["max_levels_per_side"]),
        "resistance",
    )
    support = _pick_primary_level(support_levels)
    resistance = _pick_primary_level(resistance_levels)
    nearest_support = _nearest_level(support_levels, current_price)
    nearest_resistance = _nearest_level(resistance_levels, current_price)
    floor_support = _closest_level_below(support_levels, current_price) or nearest_support
    ceiling_resistance = _closest_level_above(resistance_levels, current_price) or nearest_resistance
    broken_resistance = _closest_level_below(resistance_levels, current_price)
    broken_support = _closest_level_above(support_levels, current_price)
    last_swing_low = _latest_swing_level(swings, "low", current_price, atr, prefer_below_price=True)
    last_swing_high = _latest_swing_level(swings, "high", current_price, atr, prefer_below_price=False)
    trend = _detect_trend(frame, swings, atr)
    daily_frame = _aggregate_candles_by_timeframe(frame, "D")
    weekly_frame = _aggregate_candles_by_timeframe(frame, "W")
    daily_bias = _higher_timeframe_candle_bias(daily_frame, rules)
    weekly_bias = _higher_timeframe_candle_bias(weekly_frame, rules)
    active_pattern, patterns = _detect_patterns(swings, len(frame) - 1, current_price, atr)
    rule_bias = _evaluate_rule_bias(trend, support, resistance, active_pattern, rules)
    candle = _candle_context(frame, atr, rules)
    buy_direction = _evaluate_direction_setup("buy", trend, active_pattern, daily_bias, weekly_bias, rules)
    sell_direction = _evaluate_direction_setup("sell", trend, active_pattern, daily_bias, weekly_bias, rules)
    buy_entry = _evaluate_buy_entry(current_price, atr, candle, floor_support, broken_resistance, ceiling_resistance, last_swing_low, rules)
    sell_entry = _evaluate_sell_entry(current_price, atr, candle, ceiling_resistance, broken_support, floor_support, last_swing_high, rules)
    buy_setup = _build_side_setup(buy_direction, buy_entry)
    sell_setup = _build_side_setup(sell_direction, sell_entry)
    combined = _combine_rule_and_entry(rule_bias, buy_setup, sell_setup)

    analysis = {
        "current_price": float(current_price),
        "atr": float(atr),
        "trend": str(trend["direction"]),
        "trend_strength": float(trend["strength"]),
        "trend_score": float(trend["score"]),
        "pattern": active_pattern["name"] if active_pattern else None,
        "pattern_direction": active_pattern["direction"] if active_pattern else None,
        "pattern_confidence": float(active_pattern["confidence"]) if active_pattern else 0.0,
        "pattern_confirmed": bool(active_pattern.get("confirmed", False)) if active_pattern else False,
        "daily_candle_trend": str(daily_bias["direction"]),
        "daily_candle_strength": float(daily_bias["strength"]),
        "weekly_candle_trend": str(weekly_bias["direction"]),
        "weekly_candle_strength": float(weekly_bias["strength"]),
        "structure_levels": structure_levels,
        "support": support,
        "resistance": resistance,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "last_swing_low": last_swing_low,
        "last_swing_high": last_swing_high,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "patterns": patterns,
        "buy_setup": buy_setup,
        "sell_setup": sell_setup,
    }
    analysis.update(combined)
    return analysis


def _level_value(level: dict[str, Any] | None, field: str) -> float | None:
    if not level:
        return None
    value = level.get(field)
    return None if value is None else float(value)


def apply_structure_overlay(
    signal: int,
    consensus: dict[str, Any],
    analysis: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    rules = resolve_structure_rules(config)
    snapshot = dict(consensus)
    if not rules.get("enabled", True) or analysis is None:
        snapshot["overlay_reason"] = "structure_disabled"
        snapshot["final_signal"] = int(signal)
        return signal, snapshot

    snapshot.update({
        "trend": analysis.get("trend"),
        "trend_strength": float(analysis.get("trend_strength", 0.0)),
        "trend_score": float(analysis.get("trend_score", 0.0)),
        "pattern": analysis.get("pattern"),
        "pattern_direction": analysis.get("pattern_direction"),
        "pattern_confidence": float(analysis.get("pattern_confidence", 0.0)),
        "pattern_confirmed": bool(analysis.get("pattern_confirmed", False)),
        "support": _level_value(analysis.get("support"), "price"),
        "support_quality": _level_value(analysis.get("support"), "quality"),
        "support_distance_atr": _level_value(analysis.get("support"), "distance_atr"),
        "nearest_support": _level_value(analysis.get("nearest_support"), "price"),
        "nearest_support_quality": _level_value(analysis.get("nearest_support"), "quality"),
        "nearest_support_distance_atr": _level_value(analysis.get("nearest_support"), "distance_atr"),
        "resistance": _level_value(analysis.get("resistance"), "price"),
        "resistance_quality": _level_value(analysis.get("resistance"), "quality"),
        "resistance_distance_atr": _level_value(analysis.get("resistance"), "distance_atr"),
        "nearest_resistance": _level_value(analysis.get("nearest_resistance"), "price"),
        "nearest_resistance_quality": _level_value(analysis.get("nearest_resistance"), "quality"),
        "nearest_resistance_distance_atr": _level_value(analysis.get("nearest_resistance"), "distance_atr"),
        "buy_score": float(analysis.get("buy_score", 0.0)),
        "sell_score": float(analysis.get("sell_score", 0.0)),
        "rule_signal": int(analysis.get("rule_signal", 1)),
        "rule_bias": float(analysis.get("rule_bias", 0.0)),
        "rule_margin": float(analysis.get("rule_margin", 0.0)),
        "allow_buy": bool(analysis.get("allow_buy", True)),
        "allow_sell": bool(analysis.get("allow_sell", True)),
        "buy_setup_ok": bool(analysis.get("buy_setup", {}).get("setup_ok", False)),
        "buy_direction_score": float(analysis.get("buy_setup", {}).get("direction_score", 0.0)),
        "buy_entry_quality": float(analysis.get("buy_setup", {}).get("quality", 0.0)),
        "buy_entry_style": analysis.get("buy_setup", {}).get("style"),
        "buy_rr_ratio": float(analysis.get("buy_setup", {}).get("rr_ratio", 0.0)),
        "sell_setup_ok": bool(analysis.get("sell_setup", {}).get("setup_ok", False)),
        "sell_direction_score": float(analysis.get("sell_setup", {}).get("direction_score", 0.0)),
        "sell_entry_quality": float(analysis.get("sell_setup", {}).get("quality", 0.0)),
        "sell_entry_style": analysis.get("sell_setup", {}).get("style"),
        "sell_rr_ratio": float(analysis.get("sell_setup", {}).get("rr_ratio", 0.0)),
        "buy_blockers": list(analysis.get("buy_blockers", [])),
        "sell_blockers": list(analysis.get("sell_blockers", [])),
        "structure_reasons": list(analysis.get("reasons", [])),
        "structure_source_timeframe": analysis.get("source_timeframe"),
    })

    rule_signal = int(snapshot["rule_signal"])
    effective_margin = float(snapshot["rule_margin"]) * max(float(rules["overlay_weight"]), 0.0)
    ready = int(float(snapshot.get("ready", 0.0)))
    final_signal = int(signal)
    reason = "unchanged"

    if signal == 2:
        if not bool(snapshot["allow_buy"]):
            final_signal = 1
            reason = "buy_blocked"
        elif rule_signal == 0 and effective_margin >= float(rules["rule_margin_threshold"]):
            if bool(rules["allow_rule_reversal"]) and bool(snapshot["allow_sell"]):
                final_signal = 0
                reason = "sell_reversal"
            else:
                final_signal = 1
                reason = "buy_veto"
        elif rule_signal == 2:
            reason = "buy_confirmed"
    elif signal == 0:
        if not bool(snapshot["allow_sell"]):
            final_signal = 1
            reason = "sell_blocked"
        elif rule_signal == 2 and effective_margin >= float(rules["rule_margin_threshold"]):
            if bool(rules["allow_rule_reversal"]) and bool(snapshot["allow_buy"]):
                final_signal = 2
                reason = "buy_reversal"
            else:
                final_signal = 1
                reason = "sell_veto"
        elif rule_signal == 0:
            reason = "sell_confirmed"
    else:
        if ready > 0 and bool(rules["allow_rule_entry_from_flat"]):
            if rule_signal == 2 and bool(snapshot["allow_buy"]) and effective_margin >= float(rules["rule_margin_threshold"]):
                final_signal = 2
                reason = "buy_rule_entry"
            elif rule_signal == 0 and bool(snapshot["allow_sell"]) and effective_margin >= float(rules["rule_margin_threshold"]):
                final_signal = 0
                reason = "sell_rule_entry"
            else:
                reason = "flat"
        else:
            reason = "flat"

    snapshot["overlay_reason"] = reason
    snapshot["effective_rule_margin"] = float(effective_margin)
    snapshot["final_signal"] = int(final_signal)
    return final_signal, snapshot


def build_structure_decision(analysis: dict[str, Any] | None) -> StructureDecision | None:
    if not isinstance(analysis, dict):
        return None

    buy_setup = analysis.get("buy_setup") or {}
    sell_setup = analysis.get("sell_setup") or {}
    rule_signal = int(analysis.get("rule_signal", 1))

    if rule_signal == 2 and bool(buy_setup.get("setup_ok", False)):
        return _decision_from_setup("buy", buy_setup, analysis)
    if rule_signal == 0 and bool(sell_setup.get("setup_ok", False)):
        return _decision_from_setup("sell", sell_setup, analysis)
    return None


def decide_market_structure(
    candles: pd.DataFrame,
    point_size: float,
    config: dict[str, Any] | None = None,
    level_swings: list[dict[str, Any]] | None = None,
    structure_levels: list[dict[str, Any]] | None = None,
) -> tuple[StructureDecision | None, dict[str, Any]]:
    analysis = analyze_market_structure(
        candles=candles,
        point_size=point_size,
        config=config,
        level_swings=level_swings,
        structure_levels=structure_levels,
    )
    return build_structure_decision(analysis), analysis


def _decision_from_setup(side: str, setup: dict[str, Any], analysis: dict[str, Any]) -> StructureDecision:
    quality = float(setup.get("quality", 0.0))
    rr_ratio = float(setup.get("rr_ratio", 0.0))
    rr_score = _clamp(rr_ratio / 3.0, 0.0, 1.0)
    rule_margin = float(analysis.get("rule_margin", 0.0))
    confidence = _clamp(0.55 * quality + 0.25 * rule_margin + 0.20 * rr_score, 0.0, 1.0)

    reasons = tuple(
        dict.fromkeys(
            list(analysis.get("reasons", []))
            + list(setup.get("reasons", []))
        )
    )

    return StructureDecision(
        side=side,
        entry_style=setup.get("style"),
        confidence=confidence,
        rule_margin=rule_margin,
        rr_ratio=rr_ratio,
        protective_level_price=_optional_float(setup.get("protective_level_price")),
        protective_level_quality=float(setup.get("protective_level_quality", 0.0)),
        reasons=reasons,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
