"""Parameters specific to the frozen-channel breakout/retest entry."""
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from typing import Any


BREAKOUT_RULE_PREFIX = "regression_breakout_"


@dataclass(frozen=True)
class RegressionBreakoutParameters:
    confirmation_mode: str = field(default="legacy", metadata={"help": "legacy preserves the original entry; experimental confirmed requires impulse quality and a later retest-extreme reclaim"})
    max_bars: int = field(default=3, metadata={"help": "Shared retest/confirmation deadline, or first-retest deadline when confirmation-max-bars is positive (closed candles after breakout, inclusive)"})
    confirmation_max_bars: int = field(default=0, metadata={"help": "confirmed mode: extra confirmation window after the FIRST retest contact; 0 preserves the shared max-bars deadline (inclusive)"})
    min_close_atr: float = field(default=0.2, metadata={"help": "Minimum breakout close beyond the frozen boundary, in pre-breakout ATR"})
    retest_tolerance_atr: float = field(default=0.2, metadata={"help": "Maximum distance from the boundary for a retest touch, in frozen ATR"})
    invalidation_atr: float = field(default=0.2, metadata={"help": "Maximum closing penetration back inside the channel before invalidation, in frozen ATR"})
    max_entry_distance_atr: float = field(default=0.75, metadata={"help": "Maximum executable entry extension beyond the retested boundary, in frozen ATR"})
    stop_buffer_atr: float = field(default=0.2, metadata={"help": "Stop buffer beyond the retest bid low / ask high, in frozen ATR"})
    min_breakout_body_ratio: float = field(default=0.5, metadata={"help": "Minimum directional body/range of the breakout candle (confirmed mode, 0..1)"})
    min_confirmation_body_ratio: float = field(default=0.35, metadata={"help": "Minimum directional body/range of the reclaim candle (confirmed mode, 0..1)"})
    min_close_location: float = field(default=0.7, metadata={"help": "Minimum close location toward the favorable end of breakout and reclaim candles (confirmed mode, 0.5..1)"})
    max_retest_depth_atr: float = field(default=0.5, metadata={"help": "Maximum retest wick penetration inside the frozen channel (confirmed mode, frozen ATR)"})
    confirmation_buffer_atr: float = field(default=0.05, metadata={"help": "Required close beyond both the previous retest extreme and boundary (confirmed mode, frozen ATR)"})

    def __post_init__(self):
        for item in fields(self):
            if item.name == "confirmation_mode":
                if self.confirmation_mode not in ("legacy", "confirmed"):
                    raise ValueError("regression_breakout_confirmation_mode must be legacy or confirmed")
                continue
            try:
                value = float(getattr(self, item.name))
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{BREAKOUT_RULE_PREFIX}{item.name} must be finite and nonnegative") from error
            if not isfinite(value) or value < 0:
                raise ValueError(f"{BREAKOUT_RULE_PREFIX}{item.name} must be finite and nonnegative")
            if item.name in ("max_bars", "confirmation_max_bars"):
                minimum = 1 if item.name == "max_bars" else 0
                if value < minimum or not value.is_integer():
                    raise ValueError(f"{BREAKOUT_RULE_PREFIX}{item.name} must be an integer at least {minimum}")
                value = int(value)
            if item.name in ("min_breakout_body_ratio", "min_confirmation_body_ratio", "min_close_location"):
                minimum = 0.5 if item.name == "min_close_location" else 0
                if not minimum <= value <= 1:
                    raise ValueError(f"{BREAKOUT_RULE_PREFIX}{item.name} must be between {minimum} and 1")
            object.__setattr__(self, item.name, value)
        if self.confirmation_mode == "confirmed" and self.max_bars < 2 and not self.confirmation_max_bars:
            raise ValueError("regression_breakout_max_bars must be at least 2 for a later retest reclaim")

    def to_rules(self) -> dict[str, Any]:
        return {BREAKOUT_RULE_PREFIX + name: value for name, value in asdict(self).items()}

    @classmethod
    def from_rules(cls, rules: dict[str, Any]):
        return cls(**{item.name: rules[BREAKOUT_RULE_PREFIX + item.name]
                      for item in fields(cls) if BREAKOUT_RULE_PREFIX + item.name in rules})


DEFAULT_BREAKOUT_PARAMETERS = RegressionBreakoutParameters()
