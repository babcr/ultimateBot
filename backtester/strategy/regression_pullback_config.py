"""Optional directional-channel pullback; experimental defaults, not optimized."""
from dataclasses import asdict, dataclass, field, fields
from math import isfinite


PULLBACK_RULE_PREFIX = "regression_pullback_"


@dataclass(frozen=True)
class RegressionPullbackParameters:
    enabled: bool = field(default=False, metadata={"help":"Enable the experimental directional pullback branch inside hybrid mode"})
    regime: str = field(default="trend",metadata={"help":"trend: directional edge pullback; range: internal edge bounce without a failed-breakout excursion"})
    max_bars: int = field(default=3, metadata={"help":"Closed candles after the initial touch to confirm, inclusive"})
    min_parallelism: float = field(default=.7, metadata={"help":"Minimum channel parallelism for a directional pullback (0..1)"})
    min_drift_ratio: float = field(default=.6, metadata={"help":"Minimum weaker-line slope times lookback / channel width; both lines must trend together"})
    range_max_drift_ratio: float = field(default=.5,metadata={"help":"range regime: maximum stronger-line drift / minimum frozen channel width"})
    range_max_excursion_atr: float = field(default=.1,metadata={"help":"range regime: invalidate a wick reaching this outside excursion (frozen ATR), reserving failed breakouts for the primary branch"})
    min_width_atr: float = field(default=1.5, metadata={"help":"Minimum frozen channel width in pre-touch ATR"})
    touch_tolerance_atr: float = field(default=.2, metadata={"help":"Touch zone around the trend-side boundary, in frozen ATR"})
    max_depth_atr: float = field(default=.5, metadata={"help":"Maximum wick penetration beyond the trend-side boundary, in frozen ATR"})
    min_body_ratio: float = field(default=.35, metadata={"help":"Minimum directional body / range on a later reclaim candle (0..1)"})
    min_close_location: float = field(default=.65, metadata={"help":"Minimum directional close location on the reclaim candle (0.5..1)"})
    confirmation_buffer_atr: float = field(default=.05, metadata={"help":"Reclaim margin beyond the initial touch candle high/low, in frozen ATR"})
    max_entry_fraction: float = field(default=.6, metadata={"help":"Maximum entry progress from the trend-side boundary toward the opposite boundary (0..1)"})
    min_opposite_rr: float = field(default=1., metadata={"help":"Minimum space to the frozen opposite boundary / initial stop risk at confirmation"})
    stop_buffer_atr: float = field(default=.2, metadata={"help":"Stop buffer beyond the pullback extreme, in frozen ATR"})

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("regression_pullback_enabled must be a boolean")
        for item in fields(self):
            if item.name == "enabled":
                continue
            if item.name == "regime":
                if self.regime not in ("trend","range"):
                    raise ValueError("regression_pullback_regime must be trend or range")
                continue
            try:
                value = float(getattr(self,item.name))
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(PULLBACK_RULE_PREFIX + item.name + " must be finite and nonnegative") from error
            if not isfinite(value) or value < 0:
                raise ValueError(PULLBACK_RULE_PREFIX + item.name + " must be finite and nonnegative")
            if item.name == "max_bars":
                if value < 1 or not value.is_integer():
                    raise ValueError("regression_pullback_max_bars must be a positive integer")
                value = int(value)
            if item.name in ("min_parallelism","min_body_ratio","min_close_location","max_entry_fraction"):
                minimum = .5 if item.name == "min_close_location" else 0
                if not minimum <= value <= 1:
                    raise ValueError(PULLBACK_RULE_PREFIX + item.name + " must be in its probability range")
            object.__setattr__(self,item.name,value)

    def to_rules(self):
        return {PULLBACK_RULE_PREFIX + key:value for key,value in asdict(self).items()}

    @classmethod
    def from_rules(cls,rules):
        return cls(**{item.name:rules[PULLBACK_RULE_PREFIX+item.name] for item in fields(cls) if PULLBACK_RULE_PREFIX+item.name in rules})
