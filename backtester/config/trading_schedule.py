"""Optional entry windows and a weekly exit, using market time and IANA zones."""
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from math import isfinite
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def utc_timestamp(value: datetime) -> datetime:
    # Match the candle builders' convention for timestamps without a zone.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def parse_clock(value: str) -> time:
    if not isinstance(value, str) or re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value.strip()) is None:
        raise ValueError("Clock time must use HH:MM or HH:MM:SS, without a timezone")
    pieces = [int(piece) for piece in value.strip().split(":")]
    if pieces[0] >= 24 or any(piece >= 60 for piece in pieces[1:]):
        raise ValueError("Clock time must be between 00:00:00 and 23:59:59")
    try:
        clock = time.fromisoformat(value.strip())
    except (ValueError, TypeError) as error:
        raise ValueError("Clock time must use HH:MM or HH:MM:SS, without a timezone") from error
    if clock.tzinfo is not None or clock.microsecond:
        raise ValueError("Clock time must use whole seconds, without a timezone")
    return clock


def resolve_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as error:
        raise ValueError(f"Unknown timezone {name!r}; an IANA timezone database (tzdata on Windows) is required") from error


@dataclass(frozen=True)
class MarketSession:
    name: str
    timezone: str
    opens: str
    closes: str

    def __post_init__(self):
        opening, closing = parse_clock(self.opens), parse_clock(self.closes)
        if opening >= closing:
            raise ValueError("A market session must open before it closes on the same local day")

    def identity(self):
        return {"name": self.name, "timezone": self.timezone,
                "opens": parse_clock(self.opens).isoformat(), "closes": parse_clock(self.closes).isoformat()}


LONDON_SESSION = MarketSession("London", "Europe/London", "08:00", "16:30")
NEW_YORK_SESSION = MarketSession("New York", "America/New_York", "09:30", "16:00")


@dataclass(frozen=True)
class TradingScheduleConfig:
    session_filter: bool = False
    session_entry_fraction: float = .75
    close_friday: bool = False
    friday_close_time: str = "23:59"
    friday_close_timezone: str = "UTC"
    sessions: tuple[MarketSession, ...] = (LONDON_SESSION, NEW_YORK_SESSION)

    def __post_init__(self):
        if not isinstance(self.session_filter, bool) or not isinstance(self.close_friday, bool):
            raise ValueError("Trading schedule activation flags must be booleans")
        if not isfinite(self.session_entry_fraction) or not 0 < self.session_entry_fraction <= 1:
            raise ValueError("Session entry fraction must be finite and in (0, 1]")
        parse_clock(self.friday_close_time)
        if self.session_filter and not self.sessions:
            raise ValueError("An enabled session filter needs at least one session")
        if self.session_filter:
            for session in self.sessions:
                resolve_zone(session.timezone)
        if self.close_friday:
            resolve_zone(self.friday_close_timezone)

    @property
    def enabled(self):
        return self.session_filter or self.close_friday

    def identity(self):
        """Only effective rules belong in the model identity; disabled rules do not."""
        if not self.enabled:
            return None
        value = {"contract": "weekday-session-union-weekly-quote-exit-v1"}
        if self.session_filter:
            value["entry_sessions"] = {"fraction": self.session_entry_fraction,
                                       "sessions": [session.identity() for session in self.sessions],
                                       "weekdays": [0, 1, 2, 3, 4], "end_exclusive": True,
                                       "pending_orders": "cancel_at_end_of_continuous_entry_window"}
        if self.close_friday:
            zone = self.friday_close_timezone
            if zone in ("Etc/UTC", "Etc/GMT", "GMT", "Zulu", "UCT"):
                zone = "UTC"
            value["weekly_exit"] = {"weekday": 4, "time": parse_clock(self.friday_close_time).isoformat(),
                                    "timezone": zone, "execution": "first_symbol_quote_at_or_after_cutoff",
                                    "block_entries_until": "Monday_00:00_local",
                                    "cancel_pending_before_cutoff": True}
        return value


class TradingSchedule:
    def __init__(self, config: TradingScheduleConfig | None = None):
        self.config = config or TradingScheduleConfig()
        self.sessions = [(session, resolve_zone(session.timezone)) for session in self.config.sessions] if self.config.session_filter else []
        self.friday_zone = resolve_zone(self.config.friday_close_timezone) if self.config.close_friday else None
        self.friday_clock = parse_clock(self.config.friday_close_time)
        self.stats = {"rejected_signals": 0, "cancelled_pending_orders": 0, "weekly_closed_positions": 0}

    def entry_rejection(self, timestamp: datetime) -> str | None:
        if not self.config.enabled:
            return None
        now = utc_timestamp(timestamp)
        if self.friday_zone is not None:
            local = now.astimezone(self.friday_zone)
            if local.weekday() >= 5 or local.weekday() == 4 and local.time() >= self.friday_clock:
                return "weekly_entry_lock"
        if self.config.session_filter:
            if any(opening <= now < cutoff for opening, cutoff in self.session_windows(now)):
                return None
            return "outside_market_sessions"
        return None

    def session_windows(self, timestamp: datetime):
        now = utc_timestamp(timestamp)
        windows = []
        for session, zone in self.sessions:
            local = now.astimezone(zone)
            if local.weekday() >= 5:
                continue
            opening = datetime.combine(local.date(), parse_clock(session.opens), zone)
            closing = datetime.combine(local.date(), parse_clock(session.closes), zone)
            cutoff = opening + (closing - opening) * self.config.session_entry_fraction
            windows.append((opening.astimezone(timezone.utc), cutoff.astimezone(timezone.utc)))
        return sorted(windows)

    def entry_window_end(self, timestamp: datetime) -> datetime | None:
        # The session union can overlap during weeks with different DST dates.
        # A pending order may cross an overlap, but cannot survive a closed gap.
        merged = []
        for opening, cutoff in self.session_windows(timestamp):
            if merged and opening <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], cutoff))
            else:
                merged.append((opening, cutoff))
        now = utc_timestamp(timestamp)
        return next((cutoff for opening, cutoff in merged if opening <= now < cutoff), None)

    def entries_allowed(self, timestamp: datetime) -> bool:
        return self.entry_rejection(timestamp) is None

    def latest_weekly_cutoff(self, timestamp: datetime) -> datetime | None:
        if self.friday_zone is None:
            return None
        local = utc_timestamp(timestamp).astimezone(self.friday_zone)
        friday = local.date() - timedelta(days=(local.weekday() - 4) % 7)
        cutoff = datetime.combine(friday, self.friday_clock, self.friday_zone)
        if cutoff > local:
            cutoff = datetime.combine(friday - timedelta(days=7), self.friday_clock, self.friday_zone)
        return cutoff.astimezone(timezone.utc)

    def position_due(self, opened_at: datetime, timestamp: datetime) -> bool:
        cutoff = self.latest_weekly_cutoff(timestamp)
        return cutoff is not None and utc_timestamp(opened_at) <= cutoff

    def pending_rejection(self, created_at: datetime, timestamp: datetime) -> str | None:
        cutoff = self.latest_weekly_cutoff(timestamp)
        if cutoff is not None and utc_timestamp(created_at) <= cutoff:
            return "before_weekly_cutoff"
        reason = self.entry_rejection(created_at) or self.entry_rejection(timestamp)
        if reason is None and self.config.session_filter:
            cutoff = self.entry_window_end(created_at)
            if cutoff is not None and utc_timestamp(timestamp) >= cutoff:
                return "entry_session_ended"
        return reason

    def checkpoint(self):
        return dict(self.stats)

    def restore(self, snapshot):
        self.stats.update({name: int(snapshot.get(name, 0)) for name in self.stats})
