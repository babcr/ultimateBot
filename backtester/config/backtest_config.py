from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite
import re
from typing import Any


SUPPORTED_CAPITAL_SOURCES = ("equity", "free_margin", "balance")

DEFAULT_CANDLE_INTERVAL = timedelta(minutes=10)

_CANDLE_INTERVAL_PATTERN = re.compile(r"^\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[a-z]+)?\s*$", re.IGNORECASE)
_CANDLE_INTERVAL_FACTORS = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}


def normalize_backtest_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid backtest date {value!r}. Expected YYYY-MM-DD.") from exc


def resolve_backtest_bounds(
    start_date: Any | None = None,
    end_date: Any | None = None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    default_start_year: int | None = None,
    default_end_year: int | None = None,
) -> tuple[date | None, date | None]:
    has_explicit_bounds = any(value is not None for value in (start_date, end_date, start_year, end_year))

    resolved_start = normalize_backtest_date(start_date)
    resolved_end = normalize_backtest_date(end_date)

    if resolved_start is None and start_year is not None:
        resolved_start = date(start_year, 1, 1)
    if resolved_end is None and end_year is not None:
        resolved_end = date(end_year, 12, 31)

    if not has_explicit_bounds:
        if default_start_year is not None:
            resolved_start = date(default_start_year, 1, 1)
        if default_end_year is not None:
            resolved_end = date(default_end_year, 12, 31)

    if resolved_start is not None and resolved_end is not None and resolved_start > resolved_end:
        raise ValueError(
            "Invalid backtest bounds: start date "
            f"{resolved_start.isoformat()} is after end date {resolved_end.isoformat()}."
        )

    return resolved_start, resolved_end


def format_backtest_bound(value: Any | None) -> str:
    normalized = normalize_backtest_date(value)
    return normalized.isoformat() if normalized is not None else "open"


def normalize_candle_interval(value: Any, default: timedelta = DEFAULT_CANDLE_INTERVAL) -> timedelta:
    if isinstance(value, timedelta):
        interval = value
    else:
        match = _CANDLE_INTERVAL_PATTERN.match(str(value or ""))
        if match is None:
            return default

        try:
            amount = float(match.group("amount"))
        except (TypeError, ValueError):
            return default
        if amount <= 0.0:
            return default

        unit = str(match.group("unit") or "m").lower()
        factor = _CANDLE_INTERVAL_FACTORS.get(unit)
        if factor is None:
            return default
        interval = timedelta(seconds=amount * factor)

    if interval.total_seconds() <= 0.0:
        return default
    return interval


def format_candle_interval(value: Any) -> str:
    interval = normalize_candle_interval(value)
    total_seconds = int(interval.total_seconds())
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"


@dataclass(frozen=True)
class OrderConfig:
    execution_delay: timedelta = timedelta(milliseconds=50)
    expiration_bars: int = 2
    candle_interval: timedelta = DEFAULT_CANDLE_INTERVAL

    def __post_init__(self) -> None:
        if not isinstance(self.expiration_bars, int) or isinstance(self.expiration_bars, bool) or self.expiration_bars < 1:
            raise ValueError("Order expiration bars must be a positive integer")
        if self.candle_interval <= timedelta(0):
            raise ValueError("Order candle interval must be positive")
        if self.execution_delay < timedelta(0):
            raise ValueError("Order execution delay must be non-negative")
        try:
            expiration_delay = self.expiration_delay
        except OverflowError as error:
            raise ValueError("Order expiration bars are too large for the candle interval") from error
        if expiration_delay <= self.execution_delay:
            raise ValueError("Order expiration delay must be greater than the execution delay")

    @property
    def expiration_delay(self) -> timedelta:
        return self.candle_interval * self.expiration_bars


@dataclass(frozen=True)
class VolatilityConfig:
    candle_interval: timedelta = DEFAULT_CANDLE_INTERVAL
    candle_lookback: int = 150
    minimum_candles: int = 2
    granularity_factor: float = 10.0
    loss_shrink_ratio: float = 0.24
    offset_shrink_ratio: float = 0.003
    medium_point_spread_normalization_factor: float = 2.0
    large_point_spread_normalization_factor: float = 4.0
    mode: str = "swing"


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_percent: float = 5.
    available_money_ratio: float = 1.0
    available_money_source: str = "equity"
    accepted_risk_overrun: float = 0.15
    reward_risk_ratio: float = 2.5
    limit_spread_ratio: float = 0.10
    max_spread_ratio: float = 0.50
    max_positions: int = 10
    fixed_quantity: float | None = None
    # None starts trailing at half the configured TP reward/risk ratio.
    trailing_start_rr_ratio: float | None = None


def resolve_trailing_start_rr_ratio(tp_rr_ratio: float, start_rr_ratio: float | None = None) -> float:
    """Resolve an earlier trailing threshold using the risk manager's TP policy."""
    effective_tp_rr = float(tp_rr_ratio)
    if not isfinite(effective_tp_rr) or effective_tp_rr < 0.0:
        effective_tp_rr = RiskConfig().reward_risk_ratio
    effective_tp_rr = max(0.1, effective_tp_rr)
    start = effective_tp_rr / 2.0 if start_rr_ratio is None else float(start_rr_ratio)
    if not isfinite(start) or not 0.0 < start < effective_tp_rr:
        raise ValueError("Trailing start RR must be positive, finite and strictly below the take-profit RR")
    return start
