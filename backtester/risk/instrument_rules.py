"""Optional broker/instrument terms, supplied by an integration rather than guessed."""
from dataclasses import dataclass
from math import ceil, floor, isfinite


@dataclass(frozen=True)
class InstrumentRules:
    tick_size: float | None = None
    minimum_stop_distance: float = 0.0
    quantity_step: float | None = None
    minimum_quantity: float = 0.0
    maximum_quantity: float | None = None
    account_profit_multiplier: float = 1.0
    account_currency: str = "engine_price_quantity"
    monetary_contract: str = "price_delta_times_quantity_v1"

    def __post_init__(self):
        for name in ("tick_size", "quantity_step", "maximum_quantity"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"Instrument {name} must be finite and positive")
        for name in ("minimum_stop_distance", "minimum_quantity"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"Instrument {name} must be finite and nonnegative")
        if not isfinite(self.account_profit_multiplier) or self.account_profit_multiplier <= 0:
            raise ValueError("Account-currency conversion must be finite and positive")
        if self.maximum_quantity is not None and self.maximum_quantity < self.minimum_quantity:
            raise ValueError("Instrument quantity range is empty")

    def stop_distance(self, entry, base, extra, buy):
        base = max(base, self.minimum_stop_distance)
        if self.tick_size is None:
            return base + extra
        step = self.tick_size
        stop = entry - base if buy else entry + base
        stop = floor(stop / step + 1e-10) * step if buy else ceil(stop / step - 1e-10) * step
        # Structural/broker protection is rounded first. The additional
        # action is rounded down separately, never beyond its ATR bound.
        extra_steps = max(0, floor(extra / step + 1e-10))
        return abs(entry - stop) + extra_steps * step

    def target_price(self, target, buy):
        if self.tick_size is None:
            return target
        return (ceil(target / self.tick_size - 1e-10) if buy else floor(target / self.tick_size + 1e-10)) * self.tick_size

    def quantity(self, requested, *, fixed=False):
        result = requested
        if self.maximum_quantity is not None:
            result = min(result, self.maximum_quantity)
        if self.quantity_step is not None:
            result = floor(result / self.quantity_step + 1e-10) * self.quantity_step
        if fixed and abs(result - requested) > 1e-10:
            return 0.0
        return result if result >= self.minimum_quantity else 0.0
