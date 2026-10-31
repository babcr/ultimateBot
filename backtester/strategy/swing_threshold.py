"""Causal feedback targeting one accepted signal per reciprocal-rate period."""
from dataclasses import dataclass
from datetime import timedelta, timezone
from math import isfinite


@dataclass(frozen=True)
class DailyTradeTargetConfig:
    trades_per_day: float
    threshold_step: float = .001

    def __post_init__(self):
        if isinstance(self.trades_per_day, bool) or not isfinite(self.trades_per_day) or self.trades_per_day <= 0:
            raise ValueError("Daily trade target must be positive and finite, including fractional values")
        if not isfinite(self.threshold_step) or self.threshold_step <= 0:
            raise ValueError("Dynamic threshold step must be positive and finite")
        try:
            period = timedelta(days=self.period_days)
        except OverflowError as error:
            raise ValueError("Trade target produces an unrepresentable adjustment period") from error
        if period <= timedelta(0):
            raise ValueError("Trade target produces a zero-length adjustment period")

    @property
    def period_days(self):
        return 1. / self.trades_per_day


def daily_target_config(target=None, step=None):
    if target is None and step is None:
        return None
    if target is None:
        raise ValueError("--swing-ts2vec-threshold-step requires --swing-ts2vec-target-trades-per-day")
    return DailyTradeTargetConfig(target, .001 if step is None else step)


def validate_min_threshold(value, *, probability=False):
    if value is None:
        return None
    value = float(value)
    if not isfinite(value):
        raise ValueError('Minimum threshold must be finite')
    if probability and not 0 <= value <= 1:
        raise ValueError('A minimum probability threshold must be between 0 and 1')
    return value


class DailyTradeTarget:
    def __init__(self, threshold, config, *, probability=False, min_threshold=None):
        if not isfinite(threshold):
            raise ValueError("Initial threshold must be finite")
        self.threshold = float(threshold)
        self.min_threshold = validate_min_threshold(min_threshold, probability=probability)
        if self.min_threshold is not None:
            self.threshold = max(self.threshold, self.min_threshold)
        self.config = config
        self.probability = probability
        self.day = None
        self.period = timedelta(days=config.period_days)
        self.period_start = self.period_end = None
        self.scored = self.accepted = self.raised_steps = 0
        self.threshold_start = self.threshold
        self.history = []

    def _move(self, direction):
        value = self.threshold + direction*self.config.threshold_step
        if not isfinite(value):
            raise ValueError("Dynamic threshold overflow; use a smaller step")
        if self.probability:
            value = min(1., max(0., value))
        if self.min_threshold is not None:
            value = max(self.min_threshold, value)
        previous, self.threshold = self.threshold, value
        return value != previous

    def advance(self, timestamp):
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Daily threshold needs a timezone-aware decision timestamp")
        current = timestamp.astimezone(timezone.utc)
        if self.period_start is not None and current < self.period_start:
            return None
        self.day = current.date()
        if self.period_end is not None and current < self.period_end:
            return None
        closed = None
        if self.period_start is not None:
            before_close = self.threshold
            # No relaxation during context warmup or days with no usable input.
            lowered = self._move(-1) if self.scored and self.accepted < 1 else False
            elapsed = (current-self.period_start)//self.period
            closed = {"period_start": self.period_start.isoformat(), "period_end": self.period_end.isoformat(),
                      "period_days": self.config.period_days, "scored": self.scored, "accepted": self.accepted,
                      "target": 1, "target_per_day": self.config.trades_per_day,
                      "threshold_step": self.config.threshold_step,
                      "min_threshold": self.min_threshold,
                      "threshold_start": self.threshold_start, "threshold_before_close": before_close,
                      "next_threshold": self.threshold, "raised_steps": self.raised_steps,
                      "lowered": lowered, "no_context": self.scored == 0,
                      "unobserved_periods_skipped": elapsed-1}
            self.history.append(closed)
            # Preserve the original grid; do not round fractional-day periods.
            self.period_start += elapsed*self.period
        else:
            anchor = current.replace(hour=0, minute=0, second=0, microsecond=0)
            self.period_start = anchor+((current-anchor)//self.period)*self.period
        self.period_end = self.period_start+self.period
        self.scored = self.accepted = self.raised_steps = 0
        self.threshold_start = self.threshold
        return closed

    def record(self, accepted):
        if self.day is None:
            raise ValueError("Advance to a decision period before recording a signal")
        self.scored += 1
        if accepted:
            self.accepted += 1
            # The signal crossing the objective uses the previous threshold;
            # the increase applies to subsequent decisions, never retroactively.
            if self.accepted > 1 and self._move(1):
                self.raised_steps += 1

    def summary(self):
        return {"target_per_day": self.config.trades_per_day, "threshold_step": self.config.threshold_step,
                "min_threshold": self.min_threshold,
                "target_per_period": 1, "period_days": self.config.period_days,
                "timezone": "UTC", "completed_observed_periods": len(self.history),
                "raised_steps": sum(r["raised_steps"] for r in self.history)+self.raised_steps,
                "lowered_periods": sum(r["lowered"] for r in self.history),
                "period_start": self.period_start.isoformat() if self.period_start else None,
                "period_end": self.period_end.isoformat() if self.period_end else None,
                "current_period_scored": self.scored, "current_period_accepted": self.accepted,
                "current_threshold": self.threshold}
