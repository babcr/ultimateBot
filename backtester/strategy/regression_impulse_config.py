"""Fixed initial hypothesis: compressed channel breakout then outside hold."""
from dataclasses import dataclass,field,fields,asdict
from math import isfinite


IMPULSE_RULE_PREFIX = "regression_impulse_"


@dataclass(frozen=True)
class RegressionImpulseParameters:
    enabled: bool = field(default=False,metadata={"help":"Experimental hybrid breakout-hold branch, without waiting for a retest"})
    max_compression_ratio: float = field(default=.85,metadata={"help":"Maximum pre-breakout short ATR / regression ATR"})
    max_width_atr: float = field(default=4.,metadata={"help":"Maximum frozen channel width in pre-breakout ATR"})
    min_body_ratio: float = field(default=.7,metadata={"help":"Minimum initial breakout directional body / range (0..1)"})
    min_close_location: float = field(default=.8,metadata={"help":"Minimum favorable close location on initial breakout (0.5..1)"})
    min_close_atr: float = field(default=.2,metadata={"help":"Minimum initial close outside the channel in frozen ATR"})
    max_breakout_distance_atr: float = field(default=.75,metadata={"help":"Maximum initial breakout close distance in frozen ATR"})
    min_hold_body_ratio: float = field(default=.35,metadata={"help":"Minimum directional body / range on the next outside hold candle (0..1)"})
    min_hold_close_location: float = field(default=.65,metadata={"help":"Minimum favorable close location on hold candle (0.5..1)"})
    min_hold_low_atr: float = field(default=.1,metadata={"help":"The entire hold candle must remain at least this distance outside the channel, in frozen ATR"})
    confirmation_buffer_atr: float = field(default=.05,metadata={"help":"Minimum progress beyond the breakout close on the next candle, in frozen ATR"})
    max_entry_distance_atr: float = field(default=1.,metadata={"help":"Maximum executable entry distance beyond frozen boundary, in ATR"})
    stop_buffer_atr: float = field(default=.2,metadata={"help":"Stop just inside the frozen boundary, in ATR"})

    def __post_init__(self):
        if not isinstance(self.enabled,bool):
            raise ValueError("regression_impulse_enabled must be a boolean")
        for item in fields(self):
            if item.name == "enabled":
                continue
            try:
                value = float(getattr(self,item.name))
            except (ValueError,TypeError,OverflowError) as error:
                raise ValueError(IMPULSE_RULE_PREFIX+item.name+" must be finite and nonnegative") from error
            if not isfinite(value) or value < 0:
                raise ValueError(IMPULSE_RULE_PREFIX+item.name+" must be finite and nonnegative")
            if item.name in ("min_body_ratio","min_hold_body_ratio","min_close_location","min_hold_close_location"):
                minimum = .5 if 'location' in item.name else 0
                if not minimum <= value <= 1:
                    raise ValueError(IMPULSE_RULE_PREFIX+item.name+" is outside its ratio range")
            object.__setattr__(self,item.name,value)
        if self.max_breakout_distance_atr < self.min_close_atr or self.max_entry_distance_atr < self.min_close_atr:
            raise ValueError("Impulse maximum distances must cover the minimum breakout distance")

    def to_rules(self):
        return {IMPULSE_RULE_PREFIX+key:value for key,value in asdict(self).items()}

    @classmethod
    def from_rules(cls,rules):
        return cls(**{item.name:rules[IMPULSE_RULE_PREFIX+item.name] for item in fields(cls) if IMPULSE_RULE_PREFIX+item.name in rules})
