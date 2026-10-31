"""Point-in-time alternating bid highs / ask lows. Latest wins, not most extreme."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math

import numpy as np

from ultimateBot.backtester.market.candle_builder import Candle


SWING_SCHEMA = "alternating_bid_high_ask_low_latest_v1"
FEATURE_SCHEMA = "swing_relative_price_return_kind_pivot_time_confirmation_time_delay_v1"
FEATURE_NAMES = ("relative_price_bps", "leg_return_bps", "kind", "log_pivot_gap_periods",
                 "log_confirmation_gap_periods", "log_confirmation_delay_periods")
VOLUME_FEATURE_SCHEMA = FEATURE_SCHEMA + "_mean_quote_volume_v2"
FEATURE_SCHEMAS = {FEATURE_SCHEMA: FEATURE_NAMES,
                   VOLUME_FEATURE_SCHEMA: (*FEATURE_NAMES, 'mean_quote_volume')}


def feature_names(schema):
    if schema not in FEATURE_SCHEMAS:
        raise ValueError(f'Unsupported swing feature schema: {schema}')
    return FEATURE_SCHEMAS[schema]


def model_feature_schema(source_schema, include_volume=None):
    feature_names(source_schema)
    if include_volume is True and source_schema != VOLUME_FEATURE_SCHEMA:
        raise ValueError('This collection has no swing volumes; collect with --swing-volume first')
    return FEATURE_SCHEMA if include_volume is False else source_schema


@dataclass(frozen=True)
class SwingPoint:
    kind: str
    price: float
    pivot_at: datetime
    pair_start: datetime
    confirmed_at: datetime
    mean_quote_volume: float | None = None

    @property
    def side(self):
        return 1 if self.kind == "low" else -1

    def to_dict(self):
        # Preserve legacy serialization and event identities when volume is absent.
        return {key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in asdict(self).items() if key != 'mean_quote_volume' or value is not None}

    @property
    def event_key(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class SwingWindow:
    mode: str
    value: int

    def __post_init__(self):
        if self.mode not in ("duration", "pairs") or isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 1:
            raise ValueError("Swing window requires positive integer seconds (duration) or couples (pairs)")

    @property
    def key(self):
        return f"{self.mode}_{self.value}"


class SwingSeries:
    def __init__(self, interval: timedelta, windows: tuple[SwingWindow, ...], *, include_volume=False):
        if interval <= timedelta(0) or not windows:
            raise ValueError("Require a positive candle interval and at least one swing window")
        self.interval = interval
        self.include_volume = include_volume
        self.windows = tuple(dict.fromkeys(windows))
        self.swings: deque[SwingPoint] = deque()
        self.previous: tuple[Candle, Candle] | None = None
        self.observed_since: datetime | None = None
        self.counts = Counter()
        self.last_candidates: tuple[SwingPoint, ...] = ()
        self.max_seconds = max((w.value for w in windows if w.mode == "duration"), default=0)
        self.max_points = max((2 * w.value for w in windows if w.mode == "pairs"), default=2)

    def append(self, bid: Candle, ask: Candle) -> SwingPoint | None:
        self.last_candidates = ()
        if bid.start_time != ask.start_time:
            raise ValueError("Bid and ask candles must align")
        for c in (bid, ask):
            if (not all(math.isfinite(p) and p > 0 for p in (c.open_price, c.high_price, c.low_price, c.close_price))
                    or not c.low_price <= min(c.open_price, c.close_price) <= max(c.open_price, c.close_price) <= c.high_price):
                raise ValueError("Invalid candle OHLC")
        if self.observed_since is None:
            self.observed_since = bid.start_time
        previous, self.previous = self.previous, (bid, ask)
        if previous is None:
            return None
        gap = bid.start_time - previous[0].start_time
        if gap <= timedelta(0):
            raise ValueError("Candles must be strictly chronological")
        if gap != self.interval:
            # A reversal is two adjacent candles, never Friday + Sunday.
            self.counts["non_adjacent_candle_pair"] += 1
            return None
        found = []
        for kind, before, current in (("high", previous[0], bid), ("low", previous[1], ask)):
            a, b = before.close_price - before.open_price, current.close_price - current.open_price
            if (kind == "high" and a > 0 > b) or (kind == "low" and a < 0 < b):
                extreme = (max if kind == "high" else min)(
                    (before, current), key=lambda c: c.high_price if kind == "high" else c.low_price)
                volume = None
                if self.include_volume:
                    side = 'bid' if kind == 'high' else 'ask'
                    count = before.tick_count + current.tick_count
                    valid = getattr(before, f'{side}_volume_count') + getattr(current, f'{side}_volume_count')
                    total = getattr(before, f'{side}_volume_sum') + getattr(current, f'{side}_volume_sum')
                    if count <= 0 or valid != count or not math.isfinite(total) or total < 0:
                        raise ValueError(f'Missing or invalid {side} volumes in the two swing candles')
                    volume = total/count
                found.append(SwingPoint(kind, extreme.high_price if kind == "high" else extreme.low_price,
                                        extreme.start_time, before.start_time, current.start_time + self.interval, volume))
        return self.append_confirmed(found)

    def append_confirmed(self, found: tuple[SwingPoint, ...] | list[SwingPoint]) -> SwingPoint | None:
        """Replay stored reversal events without candles or realized trade outcomes."""
        self.last_candidates = tuple(found)
        if not found:
            return None
        if (len(found) > 2 or len({s.confirmed_at for s in found}) != 1
                or len({s.kind for s in found}) != len(found)):
            raise ValueError("Reversal candidates must share one confirmation time")
        if self.observed_since is None:
            # Older collections do not record the very first candle. The first
            # reversal's pair start is a conservative, provably observed boundary.
            self.observed_since = min(s.pair_start for s in found)
        if len(found) > 1:
            # Opposite bid/ask bodies can confirm both at once. No invented ordering.
            self.counts["ambiguous_simultaneous_reversals"] += 1
            self.last_candidates = tuple(found)
            self.counts.update(s.kind for s in found)
            self.swings.clear()
            self.observed_since = found[0].confirmed_at - self.interval
            return None
        swing = found[0]
        self.last_candidates = (swing,)
        if self.swings and self.swings[-1].kind == swing.kind:
            self.swings[-1] = swing  # The user explicitly requires latest, regardless of price.
            self.counts["same_kind_latest_replacement"] += 1
        else:
            self.swings.append(swing)
        self.counts[swing.kind] += 1
        cutoff = swing.confirmed_at - timedelta(seconds=self.max_seconds)
        while len(self.swings) > self.max_points and self.swings[0].confirmed_at < cutoff:
            self.swings.popleft()
        return swing

    def context(self, window: SwingWindow, as_of: datetime) -> tuple[SwingPoint, ...] | None:
        points = tuple(s for s in self.swings if s.confirmed_at <= as_of)
        if not points or points[-1].confirmed_at != as_of:
            return None
        if window.mode == "pairs":
            count = 2 * window.value
            return points[-count:] if len(points) >= count else None
        start = as_of - timedelta(seconds=window.value)
        if self.observed_since is None or self.observed_since > start:
            return None
        selected = tuple(s for s in points if start < s.confirmed_at <= as_of)
        return selected if len(selected) >= 2 else None


def swing_features(points: tuple[SwingPoint, ...], interval: timedelta, as_of: datetime, *, feature_schema=FEATURE_SCHEMA) -> np.ndarray:
    feature_names(feature_schema)
    if len(points) < 2 or interval <= timedelta(0):
        raise ValueError("An embedding requires at least two swings and a positive candle interval")
    if any(s.confirmed_at > as_of or not s.pair_start <= s.pivot_at < s.confirmed_at
           or s.kind not in ("high", "low") or not math.isfinite(s.price) or s.price <= 0 for s in points):
        raise ValueError("Invalid or future swing")
    if any(a.kind == b.kind or a.confirmed_at >= b.confirmed_at or a.pivot_at > b.pivot_at
           for a, b in zip(points, points[1:])):
        raise ValueError("Swings must alternate and follow confirmation/pivot chronology")
    prices = np.asarray([s.price for s in points], dtype=np.float64)
    delta_price = np.log(prices / np.r_[prices[0], prices[:-1]]) * 10000
    pivot_gaps = [0., *[(b.pivot_at - a.pivot_at) / interval for a, b in zip(points, points[1:])]]
    confirmation_gaps = [0., *[(b.confirmed_at - a.confirmed_at) / interval for a, b in zip(points, points[1:])]]
    delays = [(s.confirmed_at - s.pivot_at) / interval for s in points]
    features = np.column_stack((np.log(prices / prices[0]) * 10000, delta_price,
                                [1 if s.kind == "high" else -1 for s in points],
                                np.log1p(pivot_gaps), np.log1p(confirmation_gaps), np.log1p(delays)))
    if feature_schema == VOLUME_FEATURE_SCHEMA:
        volumes = [s.mean_quote_volume for s in points]
        if any(v is None or not math.isfinite(v) or v < 0 for v in volumes):
            raise ValueError('Volume features require a finite nonnegative mean quote volume for every swing')
        features = np.column_stack((features, volumes))
    if not np.isfinite(features).all() or (np.abs(features) > np.finfo(np.float32).max).any():
        raise ValueError("Nonfinite swing features")
    return features.astype(np.float32)
