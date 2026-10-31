"""Paramètres exclusivement dédiés à l'entrée ``failed-breakout``.

Configuration Python : ``SwingRegressionConfig.reentry``.
Options CLI et clés du moteur : ``regression_reentry_<nom>``.
Les paramètres du canal restent dans ``SwingRegressionConfig.regression``.
Voir ``docs/regression_reentry.md`` pour les règles, unités et limites exactes.
"""
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from typing import Any


REENTRY_RULE_PREFIX = "regression_reentry_"


@dataclass(frozen=True)
class RegressionReentryParameters:
    # 0 = même bougie uniquement ; 3 = bougie de dépassement et jusqu'à t+3 inclus.
    max_bars: int = field(default=3, metadata={
        "help": "Extra closed candles allowed after the excursion; 0 means same candle only (inclusive deadline)",
    })
    # Dérive d'une droite sur N bougies / largeur minimale du canal ; sans unité.
    max_drift_ratio: float = field(default=0.5, metadata={
        "help": "Maximum absolute line slope times regression lookback / minimum frozen channel width",
    })
    # High bid au-dessus / low ask au-dessous ; ATR de la fenêtre précédant l'événement.
    min_excursion_atr: float = field(default=0.1, metadata={
        "help": "Minimum excursion beyond the frozen boundary in pre-event ATR; 0 still requires a strict crossing",
    })
    # SL au-delà du plus haut ask (vente) / plus bas bid (achat), sur tout l'événement.
    stop_buffer_atr: float = field(default=0.2, metadata={
        "help": "Stop buffer beyond the event's executable quote extreme, in frozen pre-event ATR",
    })
    # Gain vers le centre / risque au stop, vérifié au prix d'entrée planifié.
    min_rr: float = field(default=1.0, metadata={
        "help": "Minimum center-target reward / stop risk at planned executable entry, with spread included",
    })
    quality_mode: str = field(default="legacy", metadata={
        "help": "legacy keeps all original confirmations; selective accepts next-bar returns or short-stopside-wick same-bar returns",
    })
    max_same_bar_stopside_wick: float = field(default=0.25, metadata={
        "help": "selective mode: maximum stop-side wick / range of the closed mid candle for same-bar reentries (0..1)",
    })

    def __post_init__(self):
        try:
            expiry = float(self.max_bars)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("regression_reentry_max_bars must be a nonnegative integer") from error
        if not isfinite(expiry) or expiry < 0 or not expiry.is_integer():
            raise ValueError("regression_reentry_max_bars must be a nonnegative integer")
        object.__setattr__(self, "max_bars", int(expiry))
        for item in fields(self):
            if item.name == "quality_mode":
                if self.quality_mode not in ("legacy", "selective"):
                    raise ValueError("regression_reentry_quality_mode must be legacy or selective")
                continue
            if item.name == "max_bars":
                continue
            try:
                value = float(getattr(self, item.name))
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"{REENTRY_RULE_PREFIX}{item.name} must be finite and nonnegative") from error
            if not isfinite(value) or value < 0:
                raise ValueError(f"{REENTRY_RULE_PREFIX}{item.name} must be finite and nonnegative")
            if item.name == "max_same_bar_stopside_wick" and value > 1:
                raise ValueError("regression_reentry_max_same_bar_stopside_wick must be between 0 and 1")
            object.__setattr__(self, item.name, value)

    def to_rules(self) -> dict[str, Any]:
        return {REENTRY_RULE_PREFIX + name: value for name, value in asdict(self).items()}

    @classmethod
    def from_rules(cls, rules: dict[str, Any]):
        return cls(**{item.name: rules[REENTRY_RULE_PREFIX + item.name]
                      for item in fields(cls) if REENTRY_RULE_PREFIX + item.name in rules})


DEFAULT_REENTRY_PARAMETERS = RegressionReentryParameters()
