from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


SUPPORTED_CAPITAL_SOURCES = ("equity", "free_margin", "balance")


@dataclass(frozen=True)
class VolatilityConfig:
    candle_interval: timedelta = timedelta(hours=1)
    candle_lookback: int = 700
    minimum_candles: int = 5
    granularity_factor: float = 10.0
    loss_shrink_ratio: float = 0.24
    offset_shrink_ratio: float = 0.003
    mode: str = "swing"


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_percent: float = 10.0
    available_money_ratio: float = 1.0
    available_money_source: str = "equity"
    accepted_risk_overrun: float = 0.25
    reward_risk_ratio: float = 2.5
    limit_spread_ratio: float = 0.10
    max_spread_ratio: float = 0.50
    max_positions: int = 10
    fixed_quantity: float | None = None
