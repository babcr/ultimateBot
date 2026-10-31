"""Causal price/time groups, independent of the regression entry engine."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.signal import Signal, SignalSide


@dataclass(frozen=True)
class TemporalPatternConfig:
    family: str = "all"
    head_shoulders_entry: str = "breakout"
    max_span_bars: int = 48
    min_span_bars: int = 6
    max_swing_gap_bars: int = 12
    max_market_gap_bars: int = 3
    max_completion_age_bars: int = 6
    max_timing_ratio: float = 2.5
    min_swing_leg_atr: float = .5
    shoulder_tolerance_atr: float = .6
    head_prominence_atr: float = .6
    neckline_tolerance_atr: float = .6
    level_tolerance_atr: float = .35
    min_height_atr: float = 1.
    prior_trend_atr: float = .5
    breakout_buffer_atr: float = .15
    retest_tolerance_atr: float = .2
    max_retest_depth_atr: float = .5
    retest_max_bars: int = 6
    confirmation_max_bars: int = 3
    confirmation_buffer_atr: float = .05
    min_body_ratio: float = .35
    min_close_location: float = .65
    max_entry_extension_atr: float = 1.
    stop_buffer_atr: float = .2
    min_target_rr: float = 1.

    def __post_init__(self):
        if self.family not in ("all", "head-shoulders", "continuation"):
            raise ValueError("Unknown temporal pattern family")
        if self.head_shoulders_entry not in ("breakout", "retest"):
            raise ValueError("Unknown head-and-shoulders confirmation")
        integer_fields = ("max_span_bars", "min_span_bars", "max_swing_gap_bars",
                          "max_market_gap_bars", "max_completion_age_bars", "retest_max_bars",
                          "confirmation_max_bars")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in self.__dataclass_fields__:
            if name in ("family", "head_shoulders_entry") or name in integer_fields:
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_span_bars > self.max_span_bars or self.max_timing_ratio < 1:
            raise ValueError("Invalid temporal group duration or symmetry")
        for name in ("min_body_ratio", "min_close_location"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if self.max_retest_depth_atr < self.retest_tolerance_atr:
            raise ValueError("Retest depth must cover the touch tolerance")
        if self.min_height_atr <= 0 or self.max_entry_extension_atr <= self.breakout_buffer_atr:
            raise ValueError("Invalid height or entry extension")


@dataclass(frozen=True)
class TimedSwing:
    kind: str
    price: float
    pivot_at: datetime
    confirmed_at: datetime


@dataclass(frozen=True)
class SwingGroup:
    name: str
    direction: int
    swings: tuple[TimedSwing, ...]
    line_start: datetime
    line_price: float
    slope_per_second: float
    height: float
    invalidation_price: float
    atr: float

    def level(self, at: datetime) -> float:
        return self.line_price + self.slope_per_second * (at - self.line_start).total_seconds()

    @property
    def key(self) -> tuple:
        return self.name, self.direction, tuple((s.kind, s.pivot_at, s.confirmed_at) for s in self.swings)


def confirmed_swings(bid: list[Candle], ask: list[Candle], interval: timedelta,
                     max_market_gap_bars: int) -> list[TimedSwing]:
    """Extreme timestamp and availability timestamp are deliberately different.

    Opposite quote reversals on the same candle are ambiguous: split the history.
    A missing market segment also splits it, so unrelated sessions cannot be glued.
    """
    if len(bid) != len(ask) or any(b.start_time != a.start_time for b, a in zip(bid, ask)):
        raise ValueError("Bid and ask candle histories must align")
    swings: list[TimedSwing] = []
    for i in range(1, len(bid)):
        gap = bid[i].start_time - bid[i - 1].start_time
        if gap <= timedelta(0):
            raise ValueError("Candle times must increase")
        if gap > max_market_gap_bars * interval:
            swings.clear()
            continue
        candidates = []
        for kind, bars in (("high", bid), ("low", ask)):
            prev, current = bars[i - 1], bars[i]
            signs = (prev.close_price - prev.open_price, current.close_price - current.open_price)
            if not ((kind == "high" and signs[0] > 0 > signs[1]) or
                    (kind == "low" and signs[0] < 0 < signs[1])):
                continue
            extreme = (max if kind == "high" else min)(
                (prev, current), key=lambda c: c.high_price if kind == "high" else c.low_price)
            candidates.append(TimedSwing(kind, extreme.high_price if kind == "high" else extreme.low_price,
                                         extreme.start_time, current.start_time + interval))
        if len(candidates) > 1:
            swings.clear()
            continue
        for swing in candidates:
            if swings and swings[-1].kind == swing.kind:
                if (swing.price - swings[-1].price) * (1 if swing.kind == "high" else -1) >= 0:
                    swings[-1] = swing
            else:
                swings.append(swing)
    return swings


def _localized(sequence: list[TimedSwing], at: datetime, interval: timedelta,
               cfg: TemporalPatternConfig) -> bool:
    times = [s.pivot_at for s in sequence]
    gaps = [(b - a) / interval for a, b in zip(times, times[1:])]
    span = (times[-1] - times[0]) / interval
    return (all(s.kind in ("high", "low") and isfinite(s.price) and s.price > 0
                and s.pivot_at + interval <= s.confirmed_at <= at for s in sequence)
            and all(a.kind != b.kind for a, b in zip(sequence, sequence[1:]))
            and all(0 < gap <= cfg.max_swing_gap_bars for gap in gaps)
            and cfg.min_span_bars <= span <= cfg.max_span_bars
            and 0 <= (at - sequence[-1].confirmed_at) / interval <= cfg.max_completion_age_bars)


def structural_swings(swings: list[TimedSwing], atr: float, minimum_leg_atr: float) -> list[TimedSwing]:
    """Absorb small counter-moves before grouping, using only the observed prefix.

    This defines one structural scale. It does not search arbitrary subsets to fit a figure.
    Replacing an extreme keeps the new confirmation time, never the earlier availability.
    """
    retained: list[TimedSwing] = []
    for swing in swings:
        if not retained:
            retained.append(swing)
        elif retained[-1].kind == swing.kind:
            if (swing.price - retained[-1].price) * (1 if swing.kind == "high" else -1) >= 0:
                retained[-1] = swing
        elif (swing.price - retained[-1].price) * (1 if swing.kind == "high" else -1) >= minimum_leg_atr * atr:
            retained.append(swing)
    return retained


def _prior_trend(swings: list[TimedSwing], start: int, seq: list[TimedSwing],
                 direction: int, atr: float, cfg: TemporalPatternConfig,
                 interval: timedelta) -> bool:
    if start < 2:
        return False
    prior = swings[start - 2:start]
    if (seq[0].pivot_at - prior[0].pivot_at) / interval > cfg.max_span_bars:
        return False
    deltas = [direction * (next(s.price for s in seq if s.kind == p.kind) - p.price) for p in prior]
    return min(deltas) > 0 and max(deltas) >= cfg.prior_trend_atr * atr


def detect_groups(swings: list[TimedSwing], at: datetime, interval: timedelta,
                  atr: float, cfg: TemporalPatternConfig) -> list[SwingGroup]:
    """Use adjacent swings only, never arbitrary combinations of distant pivots."""
    if not isfinite(atr) or atr <= 0:
        return []
    groups = []
    # The latest available swing must belong to the figure. No stale figure behind newer pivots.
    if cfg.family in ("all", "head-shoulders") and len(swings) >= 7:
        seq = swings[-5:]
        if _localized(seq, at, interval, cfg):
            direction = -1 if seq[0].kind == "high" else 1
            expected = ["high", "low", "high", "low", "high"] if direction < 0 else ["low", "high", "low", "high", "low"]
            left, n1, head, n2, right = seq
            width1, width2 = head.pivot_at - left.pivot_at, right.pivot_at - head.pivot_at
            ratio = max(width1, width2) / min(width1, width2)
            prominence = min(direction * (left.price - head.price), direction * (right.price - head.price))
            slope = (n2.price - n1.price) / (n2.pivot_at - n1.pivot_at).total_seconds()
            shoulder_depth = min(-direction * (s.price - (n1.price + slope * (s.pivot_at - n1.pivot_at).total_seconds())) for s in (left, right))
            if ([s.kind for s in seq] == expected and ratio <= cfg.max_timing_ratio
                    and abs(left.price - right.price) <= cfg.shoulder_tolerance_atr * atr
                    and abs(n1.price - n2.price) <= cfg.neckline_tolerance_atr * atr
                    and prominence >= cfg.head_prominence_atr * atr
                    and shoulder_depth >= cfg.min_height_atr * atr
                    and _prior_trend(swings, len(swings) - 5, seq, -direction, atr, cfg, interval)):
                height = -direction * (head.price - (n1.price + slope * (head.pivot_at - n1.pivot_at).total_seconds()))
                groups.append(SwingGroup("head-shoulders" if direction < 0 else "inverse-head-shoulders",
                                         direction, tuple(seq), n1.pivot_at, n1.price, slope, height, right.price, atr))
    if cfg.family in ("all", "continuation"):
        # Prefer the longest valid consolidation, with at least two touches on each side.
        for size in (8, 6, 4):
            if len(swings) < size + 2:
                continue
            seq = swings[-size:]
            if not _localized(seq, at, interval, cfg):
                continue
            highs = [s.price for s in seq if s.kind == "high"]
            lows = [s.price for s in seq if s.kind == "low"]
            if len(highs) != size // 2 or len(lows) != size // 2:
                continue
            upper, lower = max(highs), min(lows)
            if (max(highs) - min(highs) > cfg.level_tolerance_atr * atr
                    or max(lows) - min(lows) > cfg.level_tolerance_atr * atr
                    or upper - lower < cfg.min_height_atr * atr):
                continue
            for direction in (1, -1):
                if _prior_trend(swings, len(swings) - size, seq, direction, atr, cfg, interval):
                    groups.append(SwingGroup("consolidation", direction, tuple(seq), seq[0].pivot_at,
                                             upper if direction > 0 else lower, 0., upper - lower,
                                             lower if direction > 0 else upper, atr))
            if groups and groups[-1].name == "consolidation":
                break
    return groups


@dataclass
class PatternEvent:
    group: SwingGroup
    breakout_at: datetime
    last_at: datetime
    touch_at: datetime | None = None
    trigger: float | None = None
    extreme: float | None = None


def _strong_bar(candle: Candle, direction: int, cfg: TemporalPatternConfig) -> bool:
    span = candle.high_price - candle.low_price
    return (span > 0 and direction * (candle.close_price - candle.open_price) / span >= cfg.min_body_ratio
            and (candle.close_price - candle.low_price if direction > 0 else candle.high_price - candle.close_price) / span >= cfg.min_close_location)


class TemporalPatternTracker:
    def __init__(self, config: TemporalPatternConfig):
        self.config = config
        self.events: dict[str, PatternEvent] = {}
        self.forming: dict[str, dict[tuple, SwingGroup]] = {}
        self.consumed: dict[str, dict[tuple, datetime]] = {}
        self.counts = Counter()
        self.last_analysis: dict[str, dict] = {}

    def clear(self, symbol: str):
        self.events.pop(symbol, None)
        self.forming.pop(symbol, None)

    def evaluate(self, symbol: str, bid: Candle, ask: Candle, mid: Candle,
                 previous_bid: Candle, previous_ask: Candle, interval: timedelta,
                 timestamp: datetime, groups: list[SwingGroup]) -> Signal | None:
        cfg = self.config
        at = bid.start_time
        def reason(value):
            self.counts[value] += 1
            self.last_analysis[symbol] = {"closed_at": at.isoformat(), "reason": value}

        seen = self.consumed.setdefault(symbol, {})
        for key, end in list(seen.items()):
            if at - end > (cfg.max_span_bars + cfg.max_completion_age_bars + cfg.retest_max_bars + cfg.confirmation_max_bars + 2) * interval:
                del seen[key]
        event = self.events.get(symbol)
        if event is None:
            forming = self.forming.setdefault(symbol, {})
            for group in groups:
                if group.key not in seen and group.swings[-1].confirmed_at <= at:
                    # Freeze at recognition, not only on the eventual crossing. Additional
                    # post-shoulder swings cannot erase a still valid completed figure.
                    forming.setdefault(group.key, group)
            for key, group in list(forming.items()):
                if (at - group.swings[-1].confirmed_at) / interval > cfg.max_completion_age_bars:
                    del forming[key]
                    seen[key] = at
                    self.counts["group_expired_before_breakout"] += 1
            candidates = []
            for group in list(forming.values()):
                if group.key in seen or group.swings[-1].confirmed_at > at:
                    continue
                direction = group.direction
                before, bar = (previous_bid, bid) if direction > 0 else (previous_ask, ask)
                adverse = bar.low_price if direction > 0 else bar.high_price
                if direction * (adverse - group.invalidation_price) < -cfg.stop_buffer_atr * group.atr:
                    seen[group.key] = at
                    forming.pop(group.key)
                    self.counts["group_invalidated_before_breakout"] += 1
                    continue
                distance = direction * (bar.close_price - group.level(at)) / group.atr
                if (direction * (before.close_price - group.level(before.start_time)) <= cfg.breakout_buffer_atr * group.atr
                        and cfg.breakout_buffer_atr <= distance <= cfg.max_entry_extension_atr
                        and _strong_bar(mid, direction, cfg)):
                    candidates.append(group)
                elif distance >= cfg.breakout_buffer_atr:
                    # A weak first break cannot become a fresh strong crossing later.
                    seen[group.key] = at
                    forming.pop(group.key)
                    self.counts["weak_or_extended_first_breakout"] += 1
            if not candidates:
                reason("no_group_breakout")
                return None
            for group in candidates:
                seen[group.key] = at
            self.forming.pop(symbol, None)
            if len({g.direction for g in candidates}) != 1:
                reason("conflicting_groups")
                return None
            candidates.sort(key=lambda g: (g.name == "consolidation", -len(g.swings)))
            group = candidates[0]  # Reversal then longest consolidation: deterministic priority.
            self.counts[group.name + ":breakouts"] += 1
            if group.name != "consolidation" and cfg.head_shoulders_entry == "breakout":
                event = PatternEvent(group, at, at)
                stop = group.invalidation_price - group.direction * cfg.stop_buffer_atr * group.atr
                return self._signal(symbol, event, bid, ask, interval, timestamp, stop, "breakout")
            self.events[symbol] = PatternEvent(group, at, at)
            reason("breakout_waiting_retest")
            return None

        group = event.group
        direction, atr = group.direction, group.atr
        if at <= event.last_at:
            reason("duplicate_candle")
            return None
        gap = at - event.last_at
        event.last_at = at
        if gap != interval:
            self.clear(symbol)
            reason("event_history_gap")
            return None
        level = group.level(at)
        bar = bid if direction > 0 else ask
        adverse = bar.low_price if direction > 0 else bar.high_price
        structural_invalid = direction * (adverse - group.invalidation_price) <= 0
        deadline = event.breakout_at + cfg.retest_max_bars * interval if event.touch_at is None else event.touch_at + cfg.confirmation_max_bars * interval
        if at > deadline or structural_invalid or direction * (adverse - level) < -cfg.max_retest_depth_atr * atr:
            self.clear(symbol)
            reason("event_expired" if at > deadline else "event_invalidated")
            return None
        if event.touch_at is None:
            if direction * (adverse - level) <= cfg.retest_tolerance_atr * atr:
                event.touch_at = at
                event.trigger = bar.high_price if direction > 0 else bar.low_price
                event.extreme = adverse
                reason("retest_waiting_confirmation")
            else:
                reason("waiting_retest")
            return None
        event.extreme = min(event.extreme, adverse) if direction > 0 else max(event.extreme, adverse)
        reclaimed = (direction * (bar.close_price - event.trigger) >= cfg.confirmation_buffer_atr * atr
                     and direction * (bar.close_price - level) >= cfg.breakout_buffer_atr * atr)
        if not reclaimed or not _strong_bar(mid, direction, cfg):
            reason("waiting_confirmation")
            return None
        # Every raw confirmation consumes the setup, including a rejected entry extension/RR.
        self.clear(symbol)
        stop = event.extreme - direction * cfg.stop_buffer_atr * atr
        return self._signal(symbol, event, bid, ask, interval, timestamp, stop, "retest")

    def _signal(self, symbol, event, bid, ask, interval, timestamp, stop, stage):
        cfg, group = self.config, event.group
        at, direction, atr = bid.start_time, group.direction, group.atr
        level = group.level(at)
        def reason(value):
            self.counts[value] += 1
            self.last_analysis[symbol] = {"closed_at": at.isoformat(), "reason": value, "pattern": group.name}
        entry = ask.close_price if direction > 0 else bid.close_price
        target = level + direction * group.height
        risk, reward = direction * (entry - stop), direction * (target - entry)
        if (risk <= 0 or reward <= 0 or reward / risk < cfg.min_target_rr
                or direction * (entry - level) > cfg.max_entry_extension_atr * atr):
            reason("entry_extension_or_target_space")
            return None
        # The entry envelope preserves measured-target space even if the next quote gaps.
        rr_bound = (target + cfg.min_target_rr * stop) / (1 + cfg.min_target_rr)
        lower, upper = ((level + cfg.breakout_buffer_atr * atr,
                         min(level + cfg.max_entry_extension_atr * atr, rr_bound)) if direction > 0 else
                        (max(level - cfg.max_entry_extension_atr * atr, rr_bound),
                         level - cfg.breakout_buffer_atr * atr))
        if not 0 < lower <= entry <= upper or stop <= 0:
            reason("entry_outside_zone")
            return None
        reason("confirmed")
        self.counts[group.name + ":confirmed"] += 1
        return Signal(symbol, timestamp, SignalSide.BUY if direction > 0 else SignalSide.SELL,
                      confidence=.7, stop_loss_price=stop,
                      minimum_entry_price=lower, maximum_entry_price=upper,
                      entry_context={"entry_mode": group.name + "-" + stage, "strategy_entry_mode": "swing-patterns",
                                     "pattern": group.name, "breakout_at": event.breakout_at.isoformat(),
                                     "retest_at": event.touch_at.isoformat() if event.touch_at else None,
                                     "confirmation_at": at.isoformat(),
                                     "group_start": group.swings[0].pivot_at.isoformat(),
                                     "group_end": group.swings[-1].pivot_at.isoformat(),
                                     "group_span_bars": (group.swings[-1].pivot_at - group.swings[0].pivot_at) / interval,
                                     "frozen_atr": atr, "neckline_or_level": level, "measured_target": target,
                                     "measured_target_rr": reward / risk,
                                     "swings": [{"kind": s.kind, "price": s.price, "pivot_at": s.pivot_at.isoformat(),
                                                 "confirmed_at": s.confirmed_at.isoformat()} for s in group.swings]})
