"""Point d'entrée des paramètres de swing-regression, indépendants de structure.

Les objets sont immuables : utiliser dataclasses.replace pour préparer un run.
Les dictionnaires des moteurs restent des adaptateurs de compatibilité.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from ultimateBot.backtester.config.backtest_config import RiskConfig, VolatilityConfig
from ultimateBot.backtester.config.trading_schedule import TradingScheduleConfig
from ultimateBot.backtester.optimization.adaptive_strategy import AdaptiveStrategyConfig
from ultimateBot.backtester.optimization.bandit_core import BanditConfig
from ultimateBot.backtester.strategy.regression_reentry_config import RegressionReentryParameters, REENTRY_RULE_PREFIX
from ultimateBot.backtester.strategy.regression_breakout_config import RegressionBreakoutParameters, BREAKOUT_RULE_PREFIX
from ultimateBot.backtester.strategy.regression_pullback_config import RegressionPullbackParameters, PULLBACK_RULE_PREFIX
from ultimateBot.backtester.strategy.regression_impulse_config import RegressionImpulseParameters, IMPULSE_RULE_PREFIX


@dataclass(frozen=True)
class RegressionParameters:
    entry_mode: str = "trend"
    # Canal partagé par les modes d'entrée.
    # None reprend la fenêtre des niveaux pour les anciennes configurations.
    lookback_bars: int | None = 20
    min_swings_per_side: int = 2
    line_max_volatility_atr: float = 1.6
    min_consecutive_swing_distance_atr: float = 1.
    max_filtered_consecutive_swing_pair_density: float = 0.5
    max_swing_density: float = .7
    min_parallelism: float = 0.2
    market_volatility_window_bars: int = 6
    min_market_volatility_ratio: float = 0.00005
    # Entrée historique trend uniquement ; le score reste un diagnostic dans l'autre mode.
    score_threshold: float = 0.5
    slope_scale_atr: float = 0.00001
    trigger_max_distance_atr: float = 1.0
    require_level_reaction: bool = True
    require_candle_confirmation: bool = False
    live_analysis: bool = False
    max_spread_stop_ratio: float = 0.0


@dataclass(frozen=True)
class GlobalRegressionParameters:
    use_global_timeframe_validation: bool = True
    global_timeframe_multiplier: int = 3
    # Conservé pour compatibilité ; ignoré par le filtre de tendance actuel.
    global_signal_lookback_bars: int = 20
    global_reference_slope_threshold_atr: float = 0.00001
    global_min_parallelism: float = 0.2


@dataclass(frozen=True)
class LevelParameters:
    level_lookback_bars: int = 33
    max_levels_per_side: int = 100
    level_merge_atr_ratio: float = 0.08
    level_merge_min_points: float = 2.0
    min_level_quality: float = 0.5
    min_level_touches: int = 3
    entry_distance_atr: float = 0.30


@dataclass(frozen=True)
class CandleConfirmationParameters:
    # Valeur effective historique (le résolveur impose un plancher à 0.15).
    confirmation_body_ratio: float = 0.15
    doji_body_ratio: float = 0.18


@dataclass(frozen=True)
class ProtectionParameters:
    protective_level_source: str = "structure_level"
    protective_stop_buffer_atr: float = 3.


@dataclass(frozen=True)
class StructureCompatibilityParameters:
    """Règles encore transportées par les helpers partagés, sans filtre ajouté.

    Certaines sont inactives en régression (overlay, patterns, breakouts, etc.).
    Les exposer ici permet de supprimer progressivement l'héritage structurel.
    """
    enabled: bool = True
    pivot_window: int = 3
    near_level_atr: float = 0.50
    breakout_distance_atr: float = 0.65
    breakout_cross_allowance_atr: float = 0.05
    breakout_retest_tolerance_atr: float = 0.10
    breakout_max_extension_atr: float = 0.5
    pattern_confidence_threshold: float = 0.2
    directional_bias_threshold: float = 0.2
    min_entry_quality: float = 0.55
    min_breakout_quality: float = 0.65
    min_rr_ratio: float = 1.4
    min_target_distance_atr: float = 0.25
    decision_margin_threshold: float = 0.62
    require_weekly_alignment: bool = False
    weekly_alignment_weight: float = 0.60
    rule_margin_threshold: float = 0.36
    overlay_weight: float = 1.0
    allow_rule_entry_from_flat: bool = True
    allow_rule_reversal: bool = False
    live_level_count: bool = False
    live_level_count_every: int = 1


@dataclass(frozen=True)
class RegressionOrderParameters:
    # L'intervalle des ordres est celui de volatility.candle_interval.
    execution_delay: timedelta = timedelta(milliseconds=50)
    expiration_bars: int = 2


@dataclass(frozen=True)
class RegressionExecutionParameters:
    balance: float = 10_000_000.0
    batch_size: int = 500
    trailing_stop: bool = False
    # Minimum propre à la stratégie, distinct du minimum du risque.
    minimum_candles: int | None = None


@dataclass(frozen=True)
class RegressionOptimizationParameters:
    adaptive_enabled: bool = False
    adaptive: AdaptiveStrategyConfig = field(default_factory=AdaptiveStrategyConfig)
    # None désactive le bandit ; ses bornes et sa fenêtre PF sont alors celles
    # d'adaptive lorsqu'il est activé uniquement par la ligne de commande.
    bandit: BanditConfig | None = None
    bandit_memory: Path = field(default_factory=lambda: (
        Path(__file__).resolve().parents[2] / "artifacts" / "bandit_memory" / "learning.sqlite3"
    ))
    bandit_read_only: bool = False
    bandit_report: Path | None = None
    bandit_run_id: str | None = None
    bandit_resume: bool = False


@dataclass(frozen=True)
class SwingRegressionConfig:
    regression: RegressionParameters = field(default_factory=RegressionParameters)
    # Réintégration uniquement : délai, dérive, excursion, marge SL, RR vers le centre.
    reentry: RegressionReentryParameters = field(default_factory=RegressionReentryParameters)
    breakout: RegressionBreakoutParameters = field(default_factory=RegressionBreakoutParameters)
    global_timeframe: GlobalRegressionParameters = field(default_factory=GlobalRegressionParameters)
    levels: LevelParameters = field(default_factory=LevelParameters)
    candles: CandleConfirmationParameters = field(default_factory=CandleConfirmationParameters)
    protection: ProtectionParameters = field(default_factory=ProtectionParameters)
    structure_compatibility: StructureCompatibilityParameters = field(default_factory=StructureCompatibilityParameters)
    risk: RiskConfig = field(default_factory=RiskConfig)
    # Intervalle unique pour stratégie, marché, volatilité et expiration.
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    orders: RegressionOrderParameters = field(default_factory=RegressionOrderParameters)
    execution: RegressionExecutionParameters = field(default_factory=RegressionExecutionParameters)
    schedule: TradingScheduleConfig = field(default_factory=TradingScheduleConfig)
    optimization: RegressionOptimizationParameters = field(default_factory=RegressionOptimizationParameters)
    pullback: RegressionPullbackParameters = field(default_factory=RegressionPullbackParameters)
    impulse: RegressionImpulseParameters = field(default_factory=RegressionImpulseParameters)

    def to_rules(self) -> dict[str, Any]:
        """Valeurs brutes ; les moteurs conservent leurs normalisations existantes."""
        rules = {"regression_" + key: value for section in (self.regression, self.global_timeframe)
                 for key, value in asdict(section).items()}
        rules.update(self.reentry.to_rules())
        rules.update(self.breakout.to_rules())
        rules.update(self.pullback.to_rules())
        rules.update(self.impulse.to_rules())
        for section in (self.levels, self.candles, self.protection, self.structure_compatibility):
            rules.update(asdict(section))
        return rules

    def with_rules(self, rules: dict[str, Any]) -> SwingRegressionConfig:
        """Reporte les règles effectives dans leurs rubriques, sans modifier l'objet."""
        sections = {}
        for name in ("regression", "reentry", "breakout", "pullback", "impulse", "global_timeframe", "levels", "candles", "protection", "structure_compatibility"):
            section = getattr(self, name)
            prefix = (REENTRY_RULE_PREFIX if name == "reentry" else BREAKOUT_RULE_PREFIX if name == "breakout"
                      else PULLBACK_RULE_PREFIX if name == "pullback"
                      else IMPULSE_RULE_PREFIX if name == "impulse"
                      else "regression_" if name in ("regression", "global_timeframe") else "")
            sections[name] = replace(section, **{
                item.name: rules[prefix + item.name]
                for item in fields(section) if prefix + item.name in rules
            })
        return replace(self, **sections)

    def to_arg_defaults(self) -> dict[str, Any]:
        """Adaptateur vers le lanceur ; les arguments explicites restent prioritaires."""
        aliases = {
            "regression_min_swings_per_side": "regression_min_swings",
            "regression_use_global_timeframe_validation": "regression_global_timeframe_validation",
            "regression_require_level_reaction": "regression_level_reaction",
            "regression_require_candle_confirmation": "regression_candle_confirmation",
            "protective_level_source": "structure_stop_source",
            "protective_stop_buffer_atr": "structure_stop_buffer_atr",
            "min_rr_ratio": "structure_min_rr_ratio",
            "live_level_count": "structure_live_level_count",
            "live_level_count_every": "structure_live_level_count_every",
        }
        values = {aliases.get(key, key): value for key, value in self.to_rules().items()}
        values.update(asdict(self.risk))
        values["risk_percent"] = values.pop("risk_per_trade_percent")
        values["quantity"] = values.pop("fixed_quantity")
        values.update(asdict(self.volatility))
        values["volatility_mode"] = values.pop("mode")
        values.update(asdict(self.execution))
        values["strategy_minimum_candles"] = self.execution.minimum_candles
        # --minimum-candles est historiquement réservé au gestionnaire de risque.
        values["minimum_candles"] = self.volatility.minimum_candles
        values["order_expiration_bars"] = self.orders.expiration_bars
        values["execution_delay"] = self.orders.execution_delay
        values.update({key: getattr(self.schedule, key) for key in (
            "session_filter", "session_entry_fraction", "close_friday", "friday_close_time", "friday_close_timezone",
        )})
        values["trading_sessions"] = self.schedule.sessions
        optimization = self.optimization
        values.update({"adaptive_" + key: value for key, value in asdict(optimization.adaptive).items()})
        values["adaptive_strategy"] = optimization.adaptive_enabled
        values["contextual_bandit"] = optimization.bandit is not None
        for key in ("bandit_memory", "bandit_read_only", "bandit_report", "bandit_run_id", "bandit_resume"):
            values[key] = getattr(optimization, key)
        if optimization.bandit is not None:
            bandit_aliases = {"shadow_budget": "bandit_shadow_weight", "group_main_trades": "bandit_group_trades"}
            for key, value in asdict(optimization.bandit).items():
                name = ("adaptive_" + key if key in (
                    "lookback_min", "lookback_max", "lookback_step", "profit_factor_window",
                ) else bandit_aliases.get(key, "bandit_" + key))
                values[name] = value
        return values


DEFAULT_SWING_REGRESSION_CONFIG = SwingRegressionConfig()
