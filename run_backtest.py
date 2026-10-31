#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import fields, replace
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
   # Les imports ultimateBot.* necessitent le dossier parent du depot.
   sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimateBot.backtester.config.backtest_config import (
   OrderConfig,
   RiskConfig,
   VolatilityConfig,
   format_backtest_bound,
   format_candle_interval,
   normalize_candle_interval,
   resolve_backtest_bounds,
   resolve_trailing_start_rr_ratio,
)
from ultimateBot.backtester.config.trading_schedule import TradingScheduleConfig
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.file_tick_source import JForexTickSource
from ultimateBot.backtester.data.merged_feed import MergedTickFeed
from ultimateBot.backtester.engine.backtest_engine import BacktestEngine
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.optimization.adaptive_strategy import AdaptiveStrategyConfig, AdaptiveStrategyOptimizer
from ultimateBot.backtester.optimization.bandit_core import BanditConfig, BanditAction, ActionSpace
from ultimateBot.backtester.optimization.contextual_bandit import ContextualBanditOptimizer, DEFAULT_MEMORY
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.structure_decision_engine import resolve_structure_rules
from ultimateBot.backtester.strategy.dummy_strategy import DummyStrategy
from ultimateBot.backtester.strategy.structure_strategy import StructureStrategy
from ultimateBot.backtester.strategy.temporal_pattern_strategy import TemporalPatternStrategy
from ultimateBot.backtester.strategy.temporal_patterns import TemporalPatternConfig
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.backtester.strategy.swing_regression_config import (
   DEFAULT_SWING_REGRESSION_CONFIG, SwingRegressionConfig,
)
from ultimateBot.backtester.strategy.regression_reentry_config import (
   DEFAULT_REENTRY_PARAMETERS, RegressionReentryParameters,
)
from ultimateBot.backtester.strategy.regression_breakout_config import (
   DEFAULT_BREAKOUT_PARAMETERS, RegressionBreakoutParameters,
)
from ultimateBot.backtester.strategy.regression_pullback_config import RegressionPullbackParameters
from ultimateBot.backtester.strategy.regression_impulse_config import RegressionImpulseParameters

BASE_HISTORY_PATH = Path(r"C:\jforex_history")

all_symbols = [
   #"BTCUSD",
   #"ETHUSD",
   #"EURGBP",
   "EURCHF",
   "EURUSD",
   "EURCAD",
   #"EURAUD",
   #"EURNZD",
   "USDCHF",
   "USDCAD",
   "GBPUSD",
   #"GBPCAD",
   "GBPCHF",
   #"GBPAUD",
   #"GBPNZD",
]

working_symbols = ["USDCHF","GBPUSD","USDCAD","EURUSD"]

btc_eur_symbols = ["BTCUSD", "EURUSD"]
eur_symbol = ["EURUSD"]
DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_ORDER_CONFIG = OrderConfig()
DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()
DEFAULT_ADAPTIVE_CONFIG = AdaptiveStrategyConfig()
DEFAULT_STRUCTURE_RULES = resolve_structure_rules(None)
DEFAULT_REGRESSION_RULES = resolve_swing_regression_rules(None)
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2020


def _integer_grid(value):
   try:
      return tuple(int(item.strip()) for item in value.split(",") if item.strip())
   except ValueError as error:
      raise argparse.ArgumentTypeError("Expected comma-separated integer lookbacks") from error


def _stop_grid(value):
   try:
      return tuple(float(item.strip()) for item in value.split(",") if item.strip())
   except ValueError as error:
      raise argparse.ArgumentTypeError("Expected comma-separated ATR coefficients") from error


def parse_args(regression_config: SwingRegressionConfig | None = None) -> argparse.Namespace:
   parser = argparse.ArgumentParser(
      description="Launch tick-based trading backtest"
   )

   parser.add_argument(
      "--history",
      type=Path,
      default=BASE_HISTORY_PATH,
      help="Path to the historical tick data directory",
   )
   parser.add_argument(
      "--symbols",
      type=str,
      nargs="+",
      default=all_symbols,
      help="List of symbols to include in the backtest",
   )
   parser.add_argument(
      "--start-date",
      type=str,
      default=None,
      help="Inclusive start date for the backtest in YYYY-MM-DD format",
   )
   parser.add_argument(
      "--end-date",
      type=str,
      default=None,
      help="Inclusive end date for the backtest in YYYY-MM-DD format",
   )
   parser.add_argument(
      "--start-year",
      type=int,
      default=None,
      help="Fallback inclusive start year when --start-date is not provided",
   )
   parser.add_argument(
      "--end-year",
      type=int,
      default=None,
      help="Fallback inclusive end year when --end-date is not provided",
   )
   parser.add_argument(
      "--balance",
      type=float,
      default=10_000_000.0,
      help="Initial balance for the backtest",
   )
   parser.add_argument(
      "--batch-size",
      type=int,
      default=500,
      help="Batch size for the backtest",
   )
   parser.add_argument(
      "--order-expiration-bars",
      type=int,
      default=DEFAULT_ORDER_CONFIG.expiration_bars,
      help=(
         "Positive integer lifetime multiplier of the engine's base candle interval "
         f"for every pending order (default: {DEFAULT_ORDER_CONFIG.expiration_bars}). "
         "Filled positions keep their SL/TP or trailing exit"
      ),
   )
   parser.add_argument(
      "--strategy",
      type=str,
      choices=["structure", "swing-regression", "swing-patterns", "swing-ts2vec", "swing-informer", "swing-grid", "swing-nn", "dummy"],
      default="structure",
      help="Decision engine: structure, swing regressions, swing patterns, TS2Vec/XGBoost, Informer, or dummy",
   )
   parser.add_argument("--swing-ts2vec-model", "--swing-informer-model", "--swing-grid-model", "--swing-nn-model", type=Path, default=None,
                       help="Frozen swing TS2Vec/XGBoost experiment or window directory, or Informer model directory")
   parser.add_argument("--swing-ts2vec-threshold", "--swing-informer-threshold", "--swing-grid-threshold", "--swing-nn-threshold", type=float, default=None,
                       help="Override the model's saved acceptance threshold (PnL/R, transformed trailing score, or binary-target probability)")
   parser.add_argument("--swing-ts2vec-min-threshold", "--swing-informer-min-threshold", "--swing-grid-min-threshold", "--swing-nn-min-threshold", type=float, default=None,
                       help="Minimum acceptance threshold, including adaptive decreases; also raises a lower initial threshold")
   parser.add_argument("--swing-ts2vec-take-profit-r", "--swing-informer-take-profit-r", "--swing-grid-take-profit-r", "--swing-nn-take-profit-r", type=float, default=None,
                       help="Fixed TP in multiples of actual fill risk; disables trailing even for models trained with trailing")
   parser.add_argument("--swing-ts2vec-stop-buffer-pips", "--swing-informer-stop-buffer-pips", "--swing-grid-stop-buffer-pips", "--swing-nn-stop-buffer-pips", type=float, default=None,
                       help="Override the model's stop margin beyond the swing pivot, in pips; 0 uses the pivot without a margin")
   parser.add_argument("--swing-ts2vec-decision-log", "--swing-informer-decision-log", "--swing-grid-decision-log", "--swing-nn-decision-log", choices=("off", "accepted", "all"),
                       nargs="?", const="accepted", default="off",
                       help="Model decision traces: off (default), model-accepted signals only, or all for debugging; flag alone selects accepted")
   parser.add_argument("--no-swing-ts2vec-decision-log", "--no-swing-informer-decision-log", "--no-swing-grid-decision-log", "--no-swing-nn-decision-log", dest="swing_ts2vec_decision_log",
                       action="store_const", const="off", help="Disable model decision traces")
   parser.add_argument("--swing-ts2vec-target-trades-per-day", "--swing-informer-target-trades-per-day", "--swing-grid-target-trades-per-day", "--swing-nn-target-trades-per-day", type=float, default=None,
                       help="Positive model-accepted trade cadence per day across symbols (UTC), including fractional values")
   parser.add_argument("--swing-ts2vec-threshold-step", "--swing-informer-threshold-step", "--swing-grid-threshold-step", "--swing-nn-threshold-step", type=float, default=None,
                       help="Dynamic threshold adjustment step (default: 0.001 when a target cadence is set)")
   parser.add_argument(
      "--candle-interval",
      type=str,
      default=format_candle_interval(DEFAULT_VOLATILITY_CONFIG.candle_interval),
      help="Base candle period shared by market-state candles, strategy analysis and risk volatility (examples: 15m, 1h, 4h, 1d)",
   )
   parser.add_argument("--session-filter", action=argparse.BooleanOptionalAction, default=False,
                       help="Allow entries only during the first part of London (08:00-16:30 local) or New York (09:30-16:00 local), weekdays, with DST")
   parser.add_argument("--session-entry-fraction", type=float, default=.75,
                       help="Fraction of each session after its opening that accepts entries; (0,1], default 0.75")
   parser.add_argument("--close-friday", action=argparse.BooleanOptionalAction, default=False,
                       help="Close all positions at/after the Friday cutoff on each symbol's first available tick, cancel pending entries and block new entries until Monday")
   parser.add_argument("--friday-close-time", default="23:59", help="Friday cutoff HH:MM or HH:MM:SS (default 23:59)")
   parser.add_argument("--friday-close-timezone", default="UTC", help="IANA timezone for the Friday cutoff (default UTC)")
   parser.add_argument(
      "--level-lookback-bars",
      type=int,
      default=None,
      help="Support/resistance window; also supplies the regression window when --regression-lookback-bars is omitted (minimum: 2 for swing-regression, 60 for structure). Independent of --candle-lookback",
   )
   parser.add_argument(
      "--regression-lookback-bars",
      type=int,
      default=None,
      help="Independent regression window, minimum 2 candles. Shares one maximum history buffer with --level-lookback-bars",
   )
   parser.add_argument(
      "--regression-score-threshold",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_score_threshold"],
      help="Minimum absolute slope-strength required by swing-regression, between 0 and 1; parallelism now has its own independent filter",
   )
   parser.add_argument("--regression-entry-mode", choices=("trend", "failed-breakout", "breakout-retest", "hybrid"), default="trend",
                       help="Directional swing entry, failed breakout, breakout/retest, or hybrid coordination of both channel entries")
   patterns = parser.add_argument_group("Temporal swing groups (experimental --strategy swing-patterns)")
   pattern_defaults = TemporalPatternConfig()
   for field in fields(pattern_defaults):
      option = "--pattern-" + field.name.replace("_", "-")
      default = getattr(pattern_defaults, field.name)
      if field.name == "family":
         patterns.add_argument(option, choices=("all", "head-shoulders", "continuation"), default=default)
      elif field.name == "head_shoulders_entry":
         patterns.add_argument(option, choices=("breakout", "retest"), default=default)
      else:
         patterns.add_argument(option, type=type(default), default=default,
                               help=f"TemporalPatternConfig.{field.name} (default: {default})")
   parser.add_argument("--regression-max-spread-stop-ratio",type=float,default=0.,
                       help="Reject submission spread / final planned stop distance above this value; 0 disables the check")
   reentry_group = parser.add_argument_group(
      "Failed-breakout entry parameters",
      "Active with --regression-entry-mode failed-breakout or hybrid. Channel quality uses shared --regression-* filters. "
      "Python settings: SwingRegressionConfig.reentry; see docs/regression_reentry.md.",
   )
   for parameter in fields(RegressionReentryParameters):
      reentry_group.add_argument(
         "--regression-reentry-" + parameter.name.replace("_", "-"),
         type=str if parameter.name == "quality_mode" else int if parameter.name == "max_bars" else float,
         default=getattr(DEFAULT_REENTRY_PARAMETERS, parameter.name),
         help=parameter.metadata["help"] + " (default: %(default)s)",
      )
   breakout_group = parser.add_argument_group(
      "Breakout-retest entry parameters",
      "Active with --regression-entry-mode breakout-retest or hybrid. Python settings: SwingRegressionConfig.breakout; "
      "see docs/regression_breakout.md. Supports --trailing-stop or a fixed reference TP for comparison.",
   )
   for parameter in fields(RegressionBreakoutParameters):
      breakout_group.add_argument(
         "--regression-breakout-" + parameter.name.replace("_", "-"),
         type=str if parameter.name == "confirmation_mode" else int if parameter.name in ("max_bars", "confirmation_max_bars") else float,
         default=getattr(DEFAULT_BREAKOUT_PARAMETERS, parameter.name),
         help=parameter.metadata["help"] + " (default: %(default)s)",
      )
   pullback_group = parser.add_argument_group("Experimental hybrid directional pullback")
   for parameter in fields(RegressionPullbackParameters):
      options = dict(default=getattr(RegressionPullbackParameters(),parameter.name),help=parameter.metadata["help"])
      if parameter.name == "enabled":
         options["action"] = argparse.BooleanOptionalAction
      else:
         options["type"] = str if parameter.name == "regime" else int if parameter.name == "max_bars" else float
      pullback_group.add_argument("--regression-pullback-"+parameter.name.replace("_","-"),**options)
   impulse_group = parser.add_argument_group("Experimental hybrid breakout-hold")
   for parameter in fields(RegressionImpulseParameters):
      options = dict(default=getattr(RegressionImpulseParameters(),parameter.name),help=parameter.metadata["help"])
      if parameter.name == "enabled":
         options["action"] = argparse.BooleanOptionalAction
      else:
         options["type"] = float
      impulse_group.add_argument("--regression-impulse-"+parameter.name.replace("_","-"),**options)
   parser.add_argument(
      "--adaptive-strategy", action="store_true", default=False,
      help=(
         "Randomly change the strategy lookback and, unless removed from the adaptive grid, "
         "the trailing mode and swing-regression global validation choice when the rolling adaptive metric declines"
      ),
   )
   parser.add_argument(
      "--adaptive-optimize-trailing-stop",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_ADAPTIVE_CONFIG.optimize_trailing_stop,
      help=(
         "Whether trailing-stop activation belongs to the adaptive candidate grid. "
         "Use --no-adaptive-optimize-trailing-stop to keep it fixed to --trailing-stop"
      ),
   )
   bandit_defaults = BanditConfig()
   parser.add_argument("--contextual-bandit", action="store_true", help="Learn lookback, trailing mode and extra structural-stop ATR using shared LinUCB memory and virtual portfolios")
   parser.add_argument("--bandit-memory", type=Path, default=DEFAULT_MEMORY, help="Common SQLite learning memory across compatible environments")
   parser.add_argument("--bandit-read-only", action="store_true", help="Load knowledge and simulate without modifying the learning memory")
   parser.add_argument("--bandit-alpha", type=float, default=bandit_defaults.alpha)
   parser.add_argument("--bandit-lambda", dest="bandit_regularization", type=float, default=bandit_defaults.regularization)
   parser.add_argument("--bandit-seed", type=int, default=bandit_defaults.seed)
   parser.add_argument("--bandit-warmup", type=int, default=bandit_defaults.warmup, help="Valid past closure-context snapshots before freezing standardization")
   parser.add_argument("--bandit-context-bars", type=int, default=None, help="Fixed context window; defaults to the existing regression volatility window")
   parser.add_argument("--bandit-lookback-grid", type=_integer_grid, default=(), help="Explicit grid within --adaptive-lookback-min/max/step; max is excluded")
   parser.add_argument("--bandit-stop-grid", type=_stop_grid, default=(), help="ATR coefficients including zero and --structure-stop-buffer-atr as the maximum")
   parser.add_argument("--bandit-lookback-grid-size", type=int, default=bandit_defaults.lookback_grid_size)
   parser.add_argument("--bandit-stop-grid-size", type=int, default=bandit_defaults.stop_grid_size)
   parser.add_argument("--bandit-shadow-limit", type=int, default=bandit_defaults.shadow_limit, help="Maximum simultaneously allocated virtual portfolios, including draining variants")
   parser.add_argument("--bandit-shadow-weight", type=float, default=bandit_defaults.shadow_budget, help="Total lifetime learning-weight budget per correlated comparison group")
   parser.add_argument("--bandit-reward-clip", type=float, default=bandit_defaults.reward_clip)
   parser.add_argument("--bandit-feature-clip", type=float, default=bandit_defaults.feature_clip, help="Bound standardized continuous features after the scaler is frozen")
   parser.add_argument("--bandit-group-trades", type=int, default=0, help="Main closes per comparison group; defaults to the existing PF trade window")
   parser.add_argument("--bandit-group-max-bars", type=int, default=0, help="Finite entry-window guard even without main trades; default context-bars times PF-window")
   parser.add_argument("--bandit-report", type=Path, default=None, help="Structured run report; default artifacts/bandit_reports/RUN_ID.json")
   parser.add_argument("--bandit-run-id", default=None, help="Explicit provenance identifier, required for checkpoint resume")
   parser.add_argument("--bandit-resume", action="store_true", help="Restore this run's portfolios, pending orders and shadow state, then skip consumed replay ticks")
   parser.add_argument("--bandit-repeat-weight", type=float, default=None, help="Explicitly permit weighted repeated training; omitted means idempotent observations")
   parser.add_argument(
      "--adaptive-optimize-global-timeframe-validation",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_ADAPTIVE_CONFIG.optimize_global_timeframe_validation,
      help=(
         "Whether swing-regression global timeframe validation belongs to the adaptive candidate grid. "
         "Use --no-adaptive-optimize-global-timeframe-validation to keep it fixed to "
         "--regression-global-timeframe-validation"
      ),
   )
   parser.add_argument(
      "--adaptive-lookback-min", type=int, default=DEFAULT_ADAPTIVE_CONFIG.lookback_min,
      help=f"Inclusive lower strategy lookback bound (default: {DEFAULT_ADAPTIVE_CONFIG.lookback_min})",
   )
   parser.add_argument(
      "--adaptive-lookback-max", type=int, default=DEFAULT_ADAPTIVE_CONFIG.lookback_max,
      help=f"Exclusive upper strategy lookback bound (default: {DEFAULT_ADAPTIVE_CONFIG.lookback_max})",
   )
   parser.add_argument(
      "--adaptive-lookback-step", type=int, default=DEFAULT_ADAPTIVE_CONFIG.lookback_step,
      help=f"Discrete step between candidate lookbacks (default: {DEFAULT_ADAPTIVE_CONFIG.lookback_step})",
   )
   parser.add_argument(
      "--adaptive-max-idle-bars", type=int, default=DEFAULT_ADAPTIVE_CONFIG.max_idle_bars,
      help="Explore another configuration after this many closed analysis candles without a queued order, once adaptation has started and no positions/orders remain; 0 disables recovery",
   )
   parser.add_argument(
      "--adaptive-accuracy-window", type=int, default=DEFAULT_ADAPTIVE_CONFIG.accuracy_window,
      help=f"Maximum closes under the current configuration used for rolling win rate; evaluated from the first close (default: {DEFAULT_ADAPTIVE_CONFIG.accuracy_window})",
   )
   parser.add_argument(
      "--adaptive-accuracy-drop-percent", type=float, default=DEFAULT_ADAPTIVE_CONFIG.accuracy_drop_percent,
      help=f"Win-rate drop that triggers another random pair (default: {DEFAULT_ADAPTIVE_CONFIG.accuracy_drop_percent:g} percent)",
   )
   parser.add_argument(
      "--adaptive-min-accuracy-percent", type=float, default=DEFAULT_ADAPTIVE_CONFIG.min_accuracy_percent,
      help=f"Per-symbol win-rate target: at or below this threshold, search again at each close under the current configuration, starting with the first (60 means it must exceed 60 percent; 0 disables this trigger; default: {DEFAULT_ADAPTIVE_CONFIG.min_accuracy_percent:g})",
   )
   parser.add_argument(
      "--adaptive-accuracy-drop-mode", choices=("relative", "points"), default=DEFAULT_ADAPTIVE_CONFIG.accuracy_drop_mode,
      help=f"Relative drop (80 to 76 is 5 percent) or percentage-point drop (80 to 75 is 5 points) (default: {DEFAULT_ADAPTIVE_CONFIG.accuracy_drop_mode})",
   )
   parser.add_argument(
      "--adaptive-profit-factor-window", type=int,
      default=DEFAULT_ADAPTIVE_CONFIG.profit_factor_window,
      help=f"Maximum closes used by the per-symbol rolling profit factor (default: {DEFAULT_ADAPTIVE_CONFIG.profit_factor_window})",
   )
   parser.add_argument(
      "--adaptive-profit-factor-drop-percent", type=float,
      default=DEFAULT_ADAPTIVE_CONFIG.profit_factor_drop_percent,
      help=f"Profit-factor decline that triggers another configuration; a percentage in relative mode or an absolute ratio-point drop in points mode (default: {DEFAULT_ADAPTIVE_CONFIG.profit_factor_drop_percent:g})",
   )
   parser.add_argument(
      "--adaptive-profit-factor-drop-mode", choices=("relative", "points"),
      default=DEFAULT_ADAPTIVE_CONFIG.profit_factor_drop_mode,
      help=f"Compare profit factor by relative percentage decline or absolute ratio-point decline (default: {DEFAULT_ADAPTIVE_CONFIG.profit_factor_drop_mode})",
   )
   parser.add_argument(
      "--adaptive-min-profit-factor", type=float,
      default=DEFAULT_ADAPTIVE_CONFIG.min_profit_factor,
      help=f"Per-symbol rolling profit-factor target; at or below it, search again (0 disables this trigger; default: {DEFAULT_ADAPTIVE_CONFIG.min_profit_factor:g})",
   )
   parser.add_argument(
      "--adaptive-optimization-metric", choices=("accuracy", "profit_factor"),
      default=DEFAULT_ADAPTIVE_CONFIG.optimization_metric,
      help=(
         "Rolling metric used to trigger adaptive switches: accuracy or profit factor "
         f"(default: {DEFAULT_ADAPTIVE_CONFIG.optimization_metric})"
      ),
   )
   parser.add_argument(
      "--adaptive-seed", type=int, default=DEFAULT_ADAPTIVE_CONFIG.seed,
      help=f"Seed for reproducible adaptive parameter choices (default: {DEFAULT_ADAPTIVE_CONFIG.seed})",
   )
   parser.add_argument(
      "--adaptive-tighten-search-interval",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_ADAPTIVE_CONFIG.tighten_search_interval,
      help=(
         "Whether wins narrow the current lookback candidate interval around the winning configuration. "
         "Use --no-adaptive-tighten-search-interval to keep the full interval available during the whole run"
      ),
   )
   parser.add_argument(
      "--adaptive-search-long-arm-delta", type=int, default=DEFAULT_ADAPTIVE_CONFIG.search_long_arm_delta,
      help=f"Candidate indices removed from the longer side after a win, or restored after a loss (default: {DEFAULT_ADAPTIVE_CONFIG.search_long_arm_delta})",
   )
   parser.add_argument(
      "--adaptive-search-short-arm-delta", type=int, default=DEFAULT_ADAPTIVE_CONFIG.search_short_arm_delta,
      help=f"Candidate indices removed from the shorter side after a win, or restored after a loss (default: {DEFAULT_ADAPTIVE_CONFIG.search_short_arm_delta})",
   )
   parser.add_argument(
      "--adaptive-min-risk-percent", type=float, default=DEFAULT_ADAPTIVE_CONFIG.min_risk_percent,
      help=f"Initial and minimum risk per trade for each symbol; --risk-percent remains the maximum (default: {DEFAULT_ADAPTIVE_CONFIG.min_risk_percent:g} percent)",
   )
   parser.add_argument(
      "--adaptive-optimize-risk",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_ADAPTIVE_CONFIG.optimize_risk,
      help=(
         "Whether risk per trade varies with each symbol's rolling accuracy. "
         "Use --no-adaptive-optimize-risk to keep --risk-percent fixed"
      ),
   )
   parser.add_argument(
      "--adaptive-timeframe-min-minutes", type=int, default=DEFAULT_ADAPTIVE_CONFIG.timeframe_min_minutes,
      help=(
         "Legacy lower adaptive-timeframe bound retained for CLI compatibility. "
         "Adaptive timeframe switching is disabled, so each symbol stays on its original timeframe."
      ),
   )
   parser.add_argument(
      "--adaptive-timeframe-max-minutes", type=int, default=DEFAULT_ADAPTIVE_CONFIG.timeframe_max_minutes,
      help=(
         "Legacy upper adaptive-timeframe bound retained for CLI compatibility. "
         "Adaptive timeframe switching is disabled, so each symbol stays on its original timeframe."
      ),
   )
   parser.add_argument(
      "--adaptive-timeframe-step-minutes", type=int, default=DEFAULT_ADAPTIVE_CONFIG.timeframe_step_minutes,
      help=(
         "Legacy adaptive-timeframe step retained for CLI compatibility. "
         "Adaptive timeframe switching is disabled, so each symbol stays on its original timeframe."
      ),
   )
   parser.add_argument(
      "--regression-min-parallelism",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_min_parallelism"],
      help="Minimum parallelism required between the anchored and floating regression lines, between 0 and 1; ignored when the floating line is absent",
   )
   parser.add_argument(
      "--regression-slope-scale-atr",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_slope_scale_atr"],
      help=f"Positive reference for shared drift per candle in ATR units; larger values make the trend score more selective (configured default: {DEFAULT_REGRESSION_RULES['regression_slope_scale_atr']})",
   )
   parser.add_argument(
      "--regression-global-timeframe-multiplier",
      type=int,
      default=DEFAULT_REGRESSION_RULES["regression_global_timeframe_multiplier"],
      help="Higher-timeframe trend multiplier G applied to --candle-interval. Use 1 to disable the global trend filter.",
   )
   parser.add_argument(
      "--regression-global-timeframe-validation",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_REGRESSION_RULES["regression_use_global_timeframe_validation"],
      help=(
         "Whether the higher-timeframe trend gate is allowed to block local swing-regression trades "
         "when the global multiplier is above 1"
      ),
   )
   parser.add_argument(
      "--regression-global-reference-slope-threshold-atr",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_global_reference_slope_threshold_atr"],
      help="Minimum ATR-normalized slope required on the global anchored regression line to validate a local trade. Buys require a positive slope above this threshold, sells require a negative slope below -threshold. Use 0 to only enforce sign alignment.",
   )
   parser.add_argument(
      "--regression-global-min-parallelism",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_global_min_parallelism"],
      help="Minimum parallelism required on the global regression pair before a local trade may follow the global trend. Use 0 to disable.",
   )
   parser.add_argument(
      "--regression-global-signal-lookback-bars",
      type=int,
      default=DEFAULT_REGRESSION_RULES["regression_global_signal_lookback_bars"],
      help="Legacy recent-signal window from the previous global confirmation mode. It is retained for backward compatibility but ignored by the current global trend filter.",
   )
   parser.add_argument(
      "--regression-market-volatility-window-bars",
      type=int,
      default=DEFAULT_REGRESSION_RULES["regression_market_volatility_window_bars"],
      help="Closed-candle ATR window used to estimate background market volatility for the low-volatility filter",
   )
   parser.add_argument(
      "--regression-min-market-volatility-ratio",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_min_market_volatility_ratio"],
      help="Minimum normalized market volatility required to trade: ATR over the configured volatility window divided by the latest mid close. Use 0 to disable",
   )
   parser.add_argument(
      "--regression-line-max-volatility-atr",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_line_max_volatility_atr"],
      help="Maximum ATR-normalized volatility allowed around each regression line; internally this uses the line RMSE over its swing family. Use inf to disable this legitimacy filter",
   )
   parser.add_argument(
      "--regression-level-reaction",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_REGRESSION_RULES["regression_require_level_reaction"],
      help="Require a support bounce for buys or a resistance rejection for sells before validating a fresh entry swing; enabled by default",
   )
   parser.add_argument(
      "--regression-candle-confirmation",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_REGRESSION_RULES["regression_require_candle_confirmation"],
      help="Require a directional candle confirming the entry swing (bullish ask candle for a low/buy, bearish bid candle for a high/sell); disabled by default",
   )
   parser.add_argument(
      "--confirmation-body-ratio",
      type=float,
      default=DEFAULT_STRUCTURE_RULES["confirmation_body_ratio"],
      help="Minimum abs(close-open)/(high-low) for candle confirmation, between 0.15 and 1 (shared by structure and optional regression confirmation)",
   )
   parser.add_argument(
      "--regression-trigger-max-distance-atr",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_trigger_max_distance_atr"],
      help="Maximum ATR-normalized distance allowed between a trigger swing and the regression line of its own family before the signal is rejected",
   )
   parser.add_argument(
      "--regression-min-consecutive-swing-distance-atr",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_min_consecutive_swing_distance_atr"],
      help="Minimum ATR-normalized distance required between two consecutive opposite swings; if they are closer, the pair is removed before the regressions",
   )
   parser.add_argument(
      "--regression-max-filtered-consecutive-swing-pair-density",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_max_filtered_consecutive_swing_pair_density"],
      help="Maximum normalized density of filtered false consecutive swing pairs inside the regression window; this is filtered-pair count divided by regression-lookback-bars. Use inf to disable this zone filter",
   )
   parser.add_argument(
      "--regression-max-swing-density",
      type=float,
      default=DEFAULT_REGRESSION_RULES["regression_max_swing_density"],
      help="Maximum normalized swing agitation allowed inside the regression window; this is total confirmed swings divided by regression-lookback-bars. Use inf to disable this filter",
   )
   parser.add_argument(
      "--regression-min-swings",
      type=int,
      default=DEFAULT_REGRESSION_RULES["regression_min_swings_per_side"],
      help="Minimum number of confirmed swings required in the reference family to fit the anchored line; the floating line becomes optional until it has enough swings",
   )
   parser.add_argument(
      "--regression-live-analysis",
      action="store_true",
      help="Print the anchor, trend metrics, both R-squared values and decision reasons at each new confirmed swing",
   )
   parser.add_argument(
      "--structure-min-rr-ratio",
      type=float,
      default=float(DEFAULT_STRUCTURE_RULES["min_rr_ratio"]),
      help="Minimum reward/risk ratio required by the structure engine before it accepts an entry",
   )
   parser.add_argument(
      "--structure-stop-source",
      type=str,
      choices=["structure_level", "recent_swing"],
      default=str(DEFAULT_STRUCTURE_RULES["protective_level_source"]),
      help="Protective stop source: structure level (regression line in swing-regression mode) or most recent swing",
   )
   parser.add_argument(
      "--structure-stop-buffer-atr",
      type=float,
      default=float(DEFAULT_STRUCTURE_RULES["protective_stop_buffer_atr"]),
      help="Additional stop distance in ATR multiples for structure-level and recent-swing protection; 0 disables the extra buffer",
   )
   parser.add_argument(
      "--structure-live-level-count",
      action="store_true",
      help="Print the number of elements in the structure level vector during the backtest",
   )
   parser.add_argument(
      "--structure-live-level-count-every",
      type=int,
      default=1,
      help="When live level-count logging is enabled, print one line every N analysis windows",
   )
   parser.add_argument(
      "--quantity",
      type=float,
      default=None,
      help="Optional fixed quantity override. Leave empty for automatic risk-based sizing.",
   )
   parser.add_argument(
      "--risk-percent",
      type=float,
      default=DEFAULT_RISK_CONFIG.risk_per_trade_percent,
      help="Risk percentage applied to the selected available capital budget for each trade (0.1 means 0.1 percent; 25.0 means 25 percent)",
   )
   parser.add_argument(
      "--available-money-ratio",
      type=float,
      default=DEFAULT_RISK_CONFIG.available_money_ratio,
      help="Share of the selected capital source available to the risk budget",
   )
   parser.add_argument(
      "--available-money-source",
      type=str,
      default=DEFAULT_RISK_CONFIG.available_money_source,
      choices=["equity", "free_margin", "balance"],
      help="Capital source used to build the per-trade risk budget",
   )
   parser.add_argument(
      "--accepted-risk-overrun",
      type=float,
      default=DEFAULT_RISK_CONFIG.accepted_risk_overrun,
      help="Maximum allowed overshoot above the configured risk percentage when a fixed quantity is supplied",
   )
   parser.add_argument(
      "--max-positions",
      type=int,
      default=DEFAULT_RISK_CONFIG.max_positions,
      help="Maximum number of open positions in the backtest",
   )
   parser.add_argument(
      "--reward-risk-ratio",
      type=float,
      default=DEFAULT_RISK_CONFIG.reward_risk_ratio,
      help="Take-profit distance expressed as a multiple of the stop distance; in trailing mode this is a reference RR and does not close the trade",
   )
   parser.add_argument(
      "--trailing-stop",
      action="store_true",
      help="Move the SL to breakeven at a separate activation level, then advance it by one original stop distance per favorable step; TP exits are disabled in this mode",
   )
   parser.add_argument(
      "--trailing-start-rr-ratio",
      type=float,
      default=DEFAULT_RISK_CONFIG.trailing_start_rr_ratio,
      help="Start trailing at this positive reward/risk ratio, strictly below --reward-risk-ratio. Defaults to half the reference TP RR; the trade stays open beyond the planned TP",
   )
   parser.add_argument(
      "--limit-spread-ratio",
      type=float,
      default=DEFAULT_RISK_CONFIG.limit_spread_ratio,
      help="Spread ratio above which market intent is degraded to a limit order",
   )
   parser.add_argument(
      "--max-spread-ratio",
      type=float,
      default=DEFAULT_RISK_CONFIG.max_spread_ratio,
      help="Spread ratio above which the trade is rejected",
   )
   parser.add_argument(
      "--candle-lookback",
      type=int,
      default=DEFAULT_VOLATILITY_CONFIG.candle_lookback,
      help="Risk manager candle history used for volatility, stop/target sizing and spread checks. Independent of --regression-lookback-bars and --level-lookback-bars",
   )
   parser.add_argument(
      "--minimum-candles",
      type=int,
      default=DEFAULT_VOLATILITY_CONFIG.minimum_candles,
      help="Minimum number of candles required before risk evaluation is allowed",
   )
   parser.add_argument(
      "--granularity-factor",
      type=float,
      default=DEFAULT_VOLATILITY_CONFIG.granularity_factor,
      help="Scaling factor used to discretize volatility extremes",
   )
   parser.add_argument(
      "--loss-shrink-ratio",
      type=float,
      default=DEFAULT_VOLATILITY_CONFIG.loss_shrink_ratio,
      help="Multiplier applied to candle volatility to derive stop distance",
   )
   parser.add_argument(
      "--offset-shrink-ratio",
      type=float,
      default=DEFAULT_VOLATILITY_CONFIG.offset_shrink_ratio,
      help="Multiplier applied to candle volatility to derive pending-order entry offset",
   )
   parser.add_argument(
      "--medium-point-spread-normalization-factor",
      type=float,
      default=DEFAULT_VOLATILITY_CONFIG.medium_point_spread_normalization_factor,
      help="Spread normalization multiplier for non-FX instruments with medium inferred point size (for example many 100+ priced CFDs)",
   )
   parser.add_argument(
      "--large-point-spread-normalization-factor",
      type=float,
      default=DEFAULT_VOLATILITY_CONFIG.large_point_spread_normalization_factor,
      help="Spread normalization multiplier for non-FX instruments with large inferred point size (for example BTCUSD-like symbols)",
   )
   parser.add_argument(
      "--volatility-mode",
      type=str,
      choices=["swing", "intraday"],
      default=DEFAULT_VOLATILITY_CONFIG.mode,
      help="Controls whether volatility scaling rounds up or down",
   )

   dataset = parser.add_argument_group("TS2Vec trade dataset (optional)")
   dataset.add_argument("--trade-database", type=Path, default=None,
                        help="Capture order-creation embeddings and close-only labels in this SQLite database")
   dataset.add_argument("--dataset-run-id", default=None, help="Unique run identity; existing IDs are rejected")
   dataset.add_argument("--dataset-campaign-id", default=None, help="Group runs from one seed sweep")
   dataset.add_argument("--ts2vec-model", type=Path, default=Path("C:/TradingBotData/models/ts2vec_order_creation.pt"),
                        help="Frozen checkpoint; trained once from observed candles at first order creation if missing")
   dataset.add_argument("--ts2vec-context-bars", type=int, default=150, help="Fixed embedding window, independent of adaptive lookback")
   dataset.add_argument("--ts2vec-output-dims", type=int, default=320)
   dataset.add_argument("--ts2vec-hidden-dims", type=int, default=64)
   dataset.add_argument("--ts2vec-depth", type=int, default=10)
   dataset.add_argument("--ts2vec-train-iters", type=int, default=200)
   dataset.add_argument("--ts2vec-seed", type=int, default=1, help="Encoder seed, kept fixed across adaptive seeds")
   dataset.add_argument("--ts2vec-device", default="cpu")
   dataset.add_argument("--ts2vec-threads", "--informer-threads", "--swing-grid-threads", "--swing-nn-threads", type=int, default=2)
   dataset.add_argument("--trade-validator", type=Path, default=None,
                        help="XGBoost artifact directory: approve each order before queuing/submitting it")
   dataset.add_argument("--trade-validator-threshold", type=float, default=None,
                        help="Override the artifact's validation-selected acceptance threshold, in [0,1]")
   dataset.add_argument("--trade-validator-log", type=Path, default=None, help="JSONL decision audit; default C:/TradingBotData/validator_decisions/UUID.jsonl")

   # Select strategy defaults before parsing explicit options. The other engines
   # keep their existing defaults, including the shared structure CLI options.
   selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
   selector.add_argument("--strategy")
   selector.add_argument("--swing-ts2vec-model", "--swing-informer-model", "--swing-grid-model", "--swing-nn-model", type=Path)
   selector.add_argument("--trailing-start-rr-ratio", type=float)
   selection, _ = selector.parse_known_args()
   if selection.strategy == "swing-regression" or (selection.strategy is None and regression_config is not None):
      config = regression_config if regression_config is not None else DEFAULT_SWING_REGRESSION_CONFIG
      parser.set_defaults(strategy="swing-regression", swing_regression_config=config, **config.to_arg_defaults())
   if selection.strategy in ("swing-ts2vec", "swing-informer", "swing-grid", "swing-nn"):
      if selection.swing_ts2vec_model is None:
         parser.error(f"--strategy {selection.strategy} requires --{selection.strategy}-model")
      try:
         from ultimateBot.backtester.strategy.swing_ts2vec_strategy import model_configuration
         _, metadata, opportunity, _ = model_configuration(selection.swing_ts2vec_model)
      except (OSError, ValueError, KeyError) as error:
         parser.error(f"Cannot load frozen swing model: {error}")
      effective_trailing_start = (opportunity.trailing_start_r if selection.trailing_start_rr_ratio is None
                                  else selection.trailing_start_rr_ratio)
      model_defaults = dict(candle_interval=format_candle_interval(timedelta(seconds=opportunity.interval_seconds)),
                            order_expiration_bars=opportunity.order_expiration_bars,
                            execution_delay=timedelta(seconds=opportunity.execution_delay_seconds),
                            trailing_stop=True, trailing_start_rr_ratio=opportunity.trailing_start_r,
                            reward_risk_ratio=max(DEFAULT_RISK_CONFIG.reward_risk_ratio, effective_trailing_start+1))
      if metadata.get("scope"):
         model_defaults["symbols"] = metadata["scope"]["symbols"]
      parser.set_defaults(**model_defaults)
   args = parser.parse_args()
   try:
      if args.strategy in ('swing-ts2vec', 'swing-informer', 'swing-grid', 'swing-nn') and args.swing_ts2vec_take_profit_r is not None:
         args.trailing_stop = False
         args.reward_risk_ratio = args.swing_ts2vec_take_profit_r
         # This unused trailing reference must remain valid for the shared risk manager.
         args.trailing_start_rr_ratio = None
      validate_regression_entry_execution(args)
      if args.trade_database is not None or args.trade_validator is not None:
         from ultimateBot.backtester.embeddings.ts2vec_service import EmbeddingConfig
         EmbeddingConfig(context_bars=args.ts2vec_context_bars, output_dims=args.ts2vec_output_dims,
                         hidden_dims=args.ts2vec_hidden_dims, depth=args.ts2vec_depth,
                         train_iters=args.ts2vec_train_iters, seed=args.ts2vec_seed,
                         device=args.ts2vec_device, threads=args.ts2vec_threads)
         if args.strategy != "swing-regression":
            raise ValueError("Trade embeddings/validation require --strategy swing-regression")
         if args.bandit_resume:
            raise ValueError("Trade dataset capture cannot resume a bandit checkpoint without its fill snapshots")
      if args.trade_validator_threshold is not None:
         if args.trade_validator is None or not isfinite(args.trade_validator_threshold) or not 0 <= args.trade_validator_threshold <= 1:
            raise ValueError("--trade-validator-threshold requires --trade-validator and a value in [0,1]")
      if args.trade_validator is not None and args.contextual_bandit:
         raise ValueError("Trade validation currently supports legacy adaptive or fixed strategies, not contextual-bandit shadow learning")
      resolve_trailing_start_rr_ratio(args.reward_risk_ratio, args.trailing_start_rr_ratio)
      resolve_backtest_trading_schedule(args)
      if args.regression_lookback_bars is not None and args.regression_lookback_bars < 2:
         raise ValueError("--regression-lookback-bars must be at least 2")
      if not isfinite(args.confirmation_body_ratio) or not 0.15 <= args.confirmation_body_ratio <= 1.0:
         raise ValueError("--confirmation-body-ratio must be finite and between 0.15 and 1")
      resolve_backtest_order_config(args)
      adaptive_config = resolve_backtest_adaptive_config(args)
      bandit_config = resolve_backtest_bandit_config(args)
      if bandit_config is not None:
         if adaptive_config is not None:
            raise ValueError("--contextual-bandit and --adaptive-strategy cannot control the same entries")
         if args.strategy != "swing-regression":
            raise ValueError("The contextual bandit optimizes --strategy swing-regression")
         if args.bandit_resume and not args.bandit_run_id:
            raise ValueError("--bandit-resume requires --bandit-run-id")
         rules = resolve_backtest_regression_rules(args)
         ActionSpace(bandit_config, rules["protective_stop_buffer_atr"],
                     BanditAction(rules["regression_lookback_bars"], args.trailing_stop, rules["protective_stop_buffer_atr"]))
      if adaptive_config is not None:
         if (
            adaptive_config.optimize_risk
            and (not isfinite(args.risk_percent) or args.risk_percent < adaptive_config.min_risk_percent)
         ):
            raise ValueError("--risk-percent must be finite and greater than or equal to --adaptive-min-risk-percent")
         if args.strategy == "dummy":
            raise ValueError("Dummy strategy has no configurable candle history; use structure or swing-regression")
         if args.strategy == "structure" and adaptive_config.lookback_min < 60:
            raise ValueError("Structure strategy requires --adaptive-lookback-min >= 60")
      args.start_date, args.end_date = resolve_backtest_bounds(
         start_date=args.start_date,
         end_date=args.end_date,
         start_year=args.start_year,
         end_year=args.end_year,
         default_start_year=DEFAULT_START_YEAR,
         default_end_year=DEFAULT_END_YEAR,
      )
   except ValueError as error:
      parser.error(str(error))
   return args


def resolve_backtest_trading_schedule(args: argparse.Namespace) -> TradingScheduleConfig:
   return TradingScheduleConfig(
      session_filter=bool(getattr(args, "session_filter", False)),
      session_entry_fraction=getattr(args, "session_entry_fraction", .75),
      close_friday=bool(getattr(args, "close_friday", False)),
      friday_close_time=getattr(args, "friday_close_time", "23:59"),
      friday_close_timezone=getattr(args, "friday_close_timezone", "UTC"),
      sessions=getattr(args, "trading_sessions", TradingScheduleConfig().sessions),
   )


def resolve_backtest_order_config(
   args: argparse.Namespace, candle_interval: timedelta | None = None,
) -> OrderConfig:
   bars = getattr(args, "order_expiration_bars", None)
   return OrderConfig(
      execution_delay=getattr(args, "execution_delay", DEFAULT_ORDER_CONFIG.execution_delay),
      expiration_bars=DEFAULT_ORDER_CONFIG.expiration_bars if bars is None else bars,
      candle_interval=candle_interval if candle_interval is not None else resolve_backtest_candle_interval(args),
   )


def resolve_backtest_adaptive_config(args: argparse.Namespace) -> AdaptiveStrategyConfig | None:
   if not bool(getattr(args, "adaptive_strategy", False)):
      return None
   config_values = {}
   for field in fields(DEFAULT_ADAPTIVE_CONFIG):
      value = getattr(args, f"adaptive_{field.name}", None)
      config_values[field.name] = getattr(DEFAULT_ADAPTIVE_CONFIG, field.name) if value is None else value
   return AdaptiveStrategyConfig(**config_values)


def resolve_backtest_bandit_config(args: argparse.Namespace) -> BanditConfig | None:
   if not bool(getattr(args, "contextual_bandit", False)):
      return None
   defaults = BanditConfig()
   context_bars = getattr(args, "bandit_context_bars", None)
   return BanditConfig(
      lookback_min=getattr(args, "adaptive_lookback_min", defaults.lookback_min),
      lookback_max=getattr(args, "adaptive_lookback_max", defaults.lookback_max),
      lookback_step=getattr(args, "adaptive_lookback_step", defaults.lookback_step),
      lookback_grid=getattr(args, "bandit_lookback_grid", ()), stop_grid=getattr(args, "bandit_stop_grid", ()),
      lookback_grid_size=getattr(args, "bandit_lookback_grid_size", defaults.lookback_grid_size),
      stop_grid_size=getattr(args, "bandit_stop_grid_size", defaults.stop_grid_size),
      alpha=getattr(args, "bandit_alpha", defaults.alpha), regularization=getattr(args, "bandit_regularization", defaults.regularization),
      seed=getattr(args, "bandit_seed", defaults.seed), warmup=getattr(args, "bandit_warmup", defaults.warmup),
      context_bars=getattr(args, "regression_market_volatility_window_bars", defaults.context_bars) if context_bars is None else context_bars,
      reward_clip=getattr(args, "bandit_reward_clip", defaults.reward_clip),
      feature_clip=getattr(args, "bandit_feature_clip", defaults.feature_clip),
      shadow_limit=getattr(args, "bandit_shadow_limit", defaults.shadow_limit), shadow_budget=getattr(args, "bandit_shadow_weight", defaults.shadow_budget),
      profit_factor_window=getattr(args, "adaptive_profit_factor_window", defaults.profit_factor_window),
      group_main_trades=getattr(args, "bandit_group_trades", 0), group_max_bars=getattr(args, "bandit_group_max_bars", 0),
      repeat_weight=getattr(args, "bandit_repeat_weight", None),
   )


def resolve_backtest_adaptive_choice_values(
   args: argparse.Namespace,
   adaptive_config: AdaptiveStrategyConfig,
   regression_rules: dict[str, Any] | None = None,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
   trailing_stop_values = (
      (False, True)
      if adaptive_config.optimize_trailing_stop
      else (bool(getattr(args, "trailing_stop", False)),)
   )
   if getattr(args, "strategy", None) != "swing-regression":
      return trailing_stop_values, (False,)
   effective_regression_rules = (
      resolve_backtest_regression_rules(args)
      if regression_rules is None else regression_rules
   )
   global_validation_values = (
      (True, False)
      if adaptive_config.optimize_global_timeframe_validation
      else (bool(effective_regression_rules["regression_use_global_timeframe_validation"]),)
   )
   return trailing_stop_values, global_validation_values


def resolve_backtest_structure_rules(args: argparse.Namespace) -> dict[str, Any]:
   return resolve_structure_rules(
      {
         "level_lookback_bars": getattr(args, "level_lookback_bars", None),
         "confirmation_body_ratio": getattr(args, "confirmation_body_ratio", DEFAULT_STRUCTURE_RULES["confirmation_body_ratio"]),
         "min_rr_ratio": getattr(args, "structure_min_rr_ratio", DEFAULT_STRUCTURE_RULES["min_rr_ratio"]),
         "protective_level_source": getattr(args, "structure_stop_source", DEFAULT_STRUCTURE_RULES["protective_level_source"]),
         "protective_stop_buffer_atr": getattr(args, "structure_stop_buffer_atr", DEFAULT_STRUCTURE_RULES["protective_stop_buffer_atr"]),
         "live_level_count": bool(getattr(args, "structure_live_level_count", False)),
         "live_level_count_every": max(1, int(getattr(args, "structure_live_level_count_every", 1))),
      }
   )


def resolve_backtest_regression_rules(args: argparse.Namespace) -> dict[str, Any]:
   config = getattr(args, "swing_regression_config", DEFAULT_SWING_REGRESSION_CONFIG)
   rules = config.to_rules()
   defaults = resolve_swing_regression_rules(config)
   for key in ("regression_entry_mode", *DEFAULT_REENTRY_PARAMETERS.to_rules(), *DEFAULT_BREAKOUT_PARAMETERS.to_rules(), *RegressionPullbackParameters().to_rules(), *RegressionImpulseParameters().to_rules()):
      rules[key] = getattr(args, key, defaults[key])
   for key, argument in {
      "confirmation_body_ratio": "confirmation_body_ratio",
      "min_rr_ratio": "structure_min_rr_ratio",
      "protective_level_source": "structure_stop_source",
      "protective_stop_buffer_atr": "structure_stop_buffer_atr",
      "live_level_count": "structure_live_level_count",
      "live_level_count_every": "structure_live_level_count_every",
   }.items():
      value = getattr(args, argument, None)
      if value is not None:
         rules[key] = value
   lookback = getattr(args, "level_lookback_bars", None)
   rules.update({
      "level_lookback_bars": defaults["level_lookback_bars"] if lookback is None else lookback,
      "regression_max_spread_stop_ratio":getattr(args,"regression_max_spread_stop_ratio",defaults["regression_max_spread_stop_ratio"]),
      "regression_lookback_bars": getattr(args, "regression_lookback_bars", rules["regression_lookback_bars"]),
      "regression_score_threshold": getattr(args, "regression_score_threshold", defaults["regression_score_threshold"]),
      "regression_min_parallelism": getattr(args, "regression_min_parallelism", defaults["regression_min_parallelism"]),
      "regression_slope_scale_atr": getattr(args, "regression_slope_scale_atr", defaults["regression_slope_scale_atr"]),
      "regression_use_global_timeframe_validation": getattr(
         args,
         "regression_global_timeframe_validation",
         defaults["regression_use_global_timeframe_validation"],
      ),
      "regression_global_timeframe_multiplier": getattr(args, "regression_global_timeframe_multiplier", defaults["regression_global_timeframe_multiplier"]),
      "regression_global_reference_slope_threshold_atr": getattr(args, "regression_global_reference_slope_threshold_atr", defaults["regression_global_reference_slope_threshold_atr"]),
      "regression_global_min_parallelism": getattr(args, "regression_global_min_parallelism", defaults["regression_global_min_parallelism"]),
      "regression_global_signal_lookback_bars": getattr(args, "regression_global_signal_lookback_bars", defaults["regression_global_signal_lookback_bars"]),
      "regression_market_volatility_window_bars": getattr(args, "regression_market_volatility_window_bars", defaults["regression_market_volatility_window_bars"]),
      "regression_min_market_volatility_ratio": getattr(args, "regression_min_market_volatility_ratio", defaults["regression_min_market_volatility_ratio"]),
      "regression_line_max_volatility_atr": getattr(args, "regression_line_max_volatility_atr", defaults["regression_line_max_volatility_atr"]),
      "regression_trigger_max_distance_atr": getattr(args, "regression_trigger_max_distance_atr", defaults["regression_trigger_max_distance_atr"]),
      "regression_min_consecutive_swing_distance_atr": getattr(args, "regression_min_consecutive_swing_distance_atr", defaults["regression_min_consecutive_swing_distance_atr"]),
      "regression_max_filtered_consecutive_swing_pair_density": getattr(args, "regression_max_filtered_consecutive_swing_pair_density", defaults["regression_max_filtered_consecutive_swing_pair_density"]),
      "regression_max_swing_density": getattr(args, "regression_max_swing_density", defaults["regression_max_swing_density"]),
      "regression_min_swings_per_side": getattr(args, "regression_min_swings", defaults["regression_min_swings_per_side"]),
      "regression_live_analysis": bool(getattr(args, "regression_live_analysis", defaults["regression_live_analysis"])),
      "regression_require_level_reaction": getattr(args, "regression_level_reaction", defaults["regression_require_level_reaction"]),
      "regression_require_candle_confirmation": getattr(args, "regression_candle_confirmation", defaults["regression_require_candle_confirmation"]),
   })
   return resolve_swing_regression_rules(rules)


def validate_regression_entry_execution(args: argparse.Namespace) -> None:
   if args.strategy in ("swing-ts2vec", "swing-informer", "swing-grid", "swing-nn"):
      validate_swing_ts2vec_execution(args)
   elif (getattr(args, "swing_ts2vec_model", None) is not None
         or getattr(args, "swing_ts2vec_threshold", None) is not None
         or getattr(args, "swing_ts2vec_min_threshold", None) is not None
         or getattr(args, "swing_ts2vec_take_profit_r", None) is not None
         or getattr(args, "swing_ts2vec_stop_buffer_pips", None) is not None
         or getattr(args, "swing_ts2vec_decision_log", "off") != "off"
         or getattr(args, "swing_ts2vec_target_trades_per_day", None) is not None
         or getattr(args, "swing_ts2vec_threshold_step", None) is not None):
      raise ValueError("Swing model options require --strategy swing-ts2vec, swing-informer, swing-grid or swing-nn")
   if args.strategy == "swing-patterns":
      resolve_temporal_pattern_config(args)
      if any(bool(getattr(args, name, False)) for name in ("adaptive_strategy", "contextual_bandit")):
         raise ValueError("swing-patterns requires fixed temporal-group parameters: disable adaptive strategy and contextual bandit")
      if args.level_lookback_bars is not None and args.level_lookback_bars < 14:
         raise ValueError("swing-patterns requires --level-lookback-bars >= 14")
   if args.strategy != "swing-regression":
      return
   rules = resolve_backtest_regression_rules(args)
   if rules["regression_entry_mode"] == "failed-breakout":
      if any(bool(getattr(args, name, False)) for name in ("adaptive_strategy", "contextual_bandit", "trailing_stop")):
         raise ValueError("failed-breakout requires fixed parameters and its center target: disable adaptive strategy, contextual bandit and trailing stop")
   if rules["regression_entry_mode"] in ("breakout-retest", "hybrid"):
      if any(bool(getattr(args, name, False)) for name in ("adaptive_strategy", "contextual_bandit")):
         raise ValueError(f"{rules['regression_entry_mode']} requires fixed parameters: disable adaptive strategy and contextual bandit")


def resolve_temporal_pattern_config(args: argparse.Namespace) -> TemporalPatternConfig:
   defaults = TemporalPatternConfig()
   return TemporalPatternConfig(**{field.name: getattr(args, "pattern_" + field.name, getattr(defaults, field.name))
                                  for field in fields(defaults)})


def validate_swing_ts2vec_execution(args):
   from ultimateBot.backtester.strategy.swing_ts2vec_strategy import model_configuration, resolve_swing_threshold
   from ultimateBot.backtester.strategy.swing_threshold import daily_target_config
   if getattr(args, "swing_ts2vec_model", None) is None:
      raise ValueError(f"--strategy {args.strategy} requires --{args.strategy}-model")
   try:
      _, metadata, opportunity, _ = model_configuration(args.swing_ts2vec_model)
   except (OSError, KeyError) as error:
      raise ValueError(f"Cannot load frozen swing model: {error}") from error
   if metadata.get('model_type', 'swing-ts2vec') != args.strategy:
      raise ValueError(f"Model type does not match --strategy {args.strategy}")
   resolve_swing_threshold(metadata, getattr(args, "swing_ts2vec_threshold", None),
                           getattr(args, "swing_ts2vec_min_threshold", None))
   daily_target_config(getattr(args, "swing_ts2vec_target_trades_per_day", None),
                       getattr(args, "swing_ts2vec_threshold_step", None))
   stop_buffer = getattr(args, 'swing_ts2vec_stop_buffer_pips', None)
   if stop_buffer is not None and (not isfinite(stop_buffer) or stop_buffer < 0):
      raise ValueError('Swing stop buffer must be finite and nonnegative')
   if args.adaptive_strategy or args.contextual_bandit:
      raise ValueError("Frozen swing models require fixed parameters: disable adaptive strategy and contextual bandit")
   if resolve_backtest_candle_interval(args).total_seconds() != opportunity.interval_seconds:
      raise ValueError("Candle interval must match the frozen swing model")
   fixed_tp = getattr(args, 'swing_ts2vec_take_profit_r', None)
   if fixed_tp is not None:
      if not isfinite(fixed_tp) or fixed_tp <= 0:
         raise ValueError('Swing fixed take profit must be positive and finite')
      if args.trailing_stop:
         raise ValueError('A fixed swing TP requires trailing to be disabled')
   elif not args.trailing_stop:
      raise ValueError(f'Disable trailing by providing --{args.strategy}-take-profit-r')
   if (args.order_expiration_bars != opportunity.order_expiration_bars
         or getattr(args, "execution_delay", timedelta(0)).total_seconds() != opportunity.execution_delay_seconds):
      raise ValueError("Use the frozen swing model's order expiry and execution delay")
   if metadata.get("scope") and not set(args.symbols) <= set(metadata["scope"]["symbols"]):
      raise ValueError("Selected symbols must belong to the frozen model's training scope")


def resolve_backtest_candle_interval(args: argparse.Namespace, fallback: Any | None = None) -> timedelta:
   default_interval = normalize_candle_interval(
      fallback if fallback is not None else DEFAULT_VOLATILITY_CONFIG.candle_interval,
      default=DEFAULT_VOLATILITY_CONFIG.candle_interval,
   )
   return normalize_candle_interval(
      getattr(args, "candle_interval", default_interval),
      default=default_interval,
   )


def resolve_backtest_regression_config(args: argparse.Namespace) -> SwingRegressionConfig:
   """Snapshot organisé des entrées effectives, options CLI comprises."""
   config = getattr(args, "swing_regression_config", DEFAULT_SWING_REGRESSION_CONFIG)
   candle_interval = resolve_backtest_candle_interval(args)
   risk_aliases = {"risk_per_trade_percent": "risk_percent", "fixed_quantity": "quantity"}
   risk = replace(config.risk, **{
      item.name: getattr(args, risk_aliases.get(item.name, item.name), getattr(config.risk, item.name))
      for item in fields(config.risk)
   })
   volatility = replace(config.volatility, **{
      item.name: getattr(args, "volatility_mode" if item.name == "mode" else item.name,
                         getattr(config.volatility, item.name))
      for item in fields(config.volatility) if item.name != "candle_interval"
   }, candle_interval=candle_interval)
   order_config = resolve_backtest_order_config(args, candle_interval)
   optimization = replace(
      config.optimization,
      adaptive_enabled=bool(getattr(args, "adaptive_strategy", False)),
      adaptive=resolve_backtest_adaptive_config(args) or config.optimization.adaptive,
      bandit=resolve_backtest_bandit_config(args),
      **{key: getattr(args, key, getattr(config.optimization, key)) for key in (
         "bandit_memory", "bandit_read_only", "bandit_report", "bandit_run_id", "bandit_resume",
      )},
   )
   return replace(
      config.with_rules(resolve_backtest_regression_rules(args)), risk=risk, volatility=volatility,
      orders=replace(config.orders, execution_delay=order_config.execution_delay,
                     expiration_bars=order_config.expiration_bars),
      execution=replace(config.execution, balance=args.balance, batch_size=args.batch_size,
                        trailing_stop=bool(getattr(args, "trailing_stop", False)),
                        minimum_candles=getattr(args, "strategy_minimum_candles", config.execution.minimum_candles)),
      schedule=resolve_backtest_trading_schedule(args), optimization=optimization,
   )


def build_feed(history_path: Path, symbols: list[str], start_date: date | None, end_date: date | None):
   sources = [
      JForexTickSource(
         base_path=history_path,
         symbol=symbol,
         start_date=start_date,
         end_date=end_date,
      )
      for symbol in symbols
   ]
   return MergedTickFeed(sources=sources)


def build_engine(
   feed,
   args: argparse.Namespace,
   strategy_override=None,
) -> BacktestEngine:
   validate_regression_entry_execution(args)
   portfolio = Portfolio(initial_balance=args.balance)
   reports = ReportCollector()
   execution_model = ExecutionModel()
   adaptive_config = resolve_backtest_adaptive_config(args)
   bandit_config = resolve_backtest_bandit_config(args)
   if bandit_config is not None and adaptive_config is not None:
      raise ValueError("Contextual and legacy adaptive optimization are mutually exclusive")
   trade_closer = TradeCloser(trailing_stop_enabled=bool(getattr(args, "trailing_stop", False)))
   adaptive_optimizer = None
   strategy = strategy_override
   if strategy is None and args.strategy == "dummy":
      strategy = DummyStrategy(seed=42)

   candle_interval = resolve_backtest_candle_interval(
      args,
      fallback=getattr(strategy, "analysis_interval", None),
   )
   order_config = resolve_backtest_order_config(args, candle_interval=candle_interval)

   if isinstance(strategy, DummyStrategy):
      if adaptive_config is not None:
         raise ValueError("Dummy strategy has no configurable candle history")
      market_state = MarketState(candle_interval=candle_interval)
   else:
      if strategy is None:
         if args.strategy == "swing-regression":
            strategy = SwingRegressionStrategy(
               regression_rules=resolve_backtest_regression_rules(args),
               analysis_interval=candle_interval,
               minimum_candles=getattr(args, "strategy_minimum_candles", None),
               config=resolve_backtest_regression_config(args),
            )
         elif args.strategy == "swing-patterns":
            strategy = TemporalPatternStrategy(
               config=resolve_temporal_pattern_config(args), lookback=args.level_lookback_bars,
               analysis_interval=candle_interval,
            )
         elif args.strategy in ("swing-ts2vec", "swing-informer", "swing-grid", "swing-nn"):
            from ultimateBot.backtester.strategy.swing_ts2vec_strategy import SwingTS2VecStrategy
            from ultimateBot.backtester.strategy.swing_threshold import daily_target_config
            strategy_type = SwingTS2VecStrategy
            if args.strategy == 'swing-informer':
               from ultimateBot.backtester.strategy.swing_informer_strategy import SwingInformerStrategy
               strategy_type = SwingInformerStrategy
            elif args.strategy == 'swing-nn':
               from ultimateBot.backtester.strategy.swing_neural_strategy import SwingNeuralStrategy
               strategy_type = SwingNeuralStrategy
            elif args.strategy == 'swing-grid':
               from ultimateBot.backtester.strategy.swing_grid_strategy import SwingGridStrategy
               strategy_type = SwingGridStrategy
            strategy = strategy_type(args.swing_ts2vec_model, threads=args.ts2vec_threads,
                                          threshold=getattr(args, "swing_ts2vec_threshold", None),
                                          min_threshold=getattr(args, "swing_ts2vec_min_threshold", None),
                                          trailing_start_r=args.trailing_start_rr_ratio,
                                          take_profit_r=getattr(args, 'swing_ts2vec_take_profit_r', None),
                                          stop_buffer_pips=getattr(args, 'swing_ts2vec_stop_buffer_pips', None),
                                          decision_log=getattr(args, "swing_ts2vec_decision_log", "off"),
                                          daily_target=daily_target_config(
                                             getattr(args, "swing_ts2vec_target_trades_per_day", None),
                                             getattr(args, "swing_ts2vec_threshold_step", None)))
         else:
            strategy = StructureStrategy(
               structure_rules=resolve_backtest_structure_rules(args),
               analysis_interval=candle_interval,
            )
      elif hasattr(strategy, "analysis_interval"):
         strategy.analysis_interval = candle_interval
      if adaptive_config is not None:
         adaptive_optimizer = AdaptiveStrategyOptimizer(
            adaptive_config,
            strategy,
            trade_closer,
            max_risk_percent=args.risk_percent,
         )
      # Retain the full analysis window required by the selected swing-based engine.
      market_state = MarketState(
         candle_interval=candle_interval,
         candle_lookback=max(
            int(getattr(strategy, "minimum_candles", 30)),
            int(getattr(strategy, "structure_rules", {}).get("level_lookback_bars", getattr(strategy, "minimum_candles", 30))),
            adaptive_optimizer.required_history_capacity if adaptive_optimizer is not None else 0,
            max(bandit_config.context_bars, max(range(bandit_config.lookback_min, bandit_config.lookback_max,
                                                    bandit_config.lookback_step))) if bandit_config is not None else 0,
         ),
      )
   analysis_interval = getattr(strategy, "analysis_interval", None)

   engine = BacktestEngine(
      feed=feed,
      batcher=TickBatcher(
         batch_size=args.batch_size,
         analysis_interval=analysis_interval,
         analysis_interval_provider=(
            strategy.get_analysis_interval if adaptive_optimizer is not None else None
         ),
      ),
      strategy=strategy,
      risk_manager=RiskManager(
         risk_per_trade=args.risk_percent,
         max_positions=args.max_positions,
         available_money_ratio=args.available_money_ratio,
         available_money_source=args.available_money_source,
         accepted_risk_overrun=args.accepted_risk_overrun,
         reward_risk_ratio=getattr(args, "reward_risk_ratio", DEFAULT_RISK_CONFIG.reward_risk_ratio),
         trailing_start_rr_ratio=getattr(args, "trailing_start_rr_ratio", DEFAULT_RISK_CONFIG.trailing_start_rr_ratio),
         limit_spread_ratio=(args.max_spread_ratio if args.strategy in ("swing-ts2vec", "swing-informer", "swing-grid", "swing-nn")
                             else getattr(args, "limit_spread_ratio", DEFAULT_RISK_CONFIG.limit_spread_ratio)),
         max_spread_ratio=getattr(args, "max_spread_ratio", DEFAULT_RISK_CONFIG.max_spread_ratio),
         fixed_quantity=args.quantity,
         candle_interval=candle_interval,
         candle_lookback=getattr(args, "candle_lookback", DEFAULT_VOLATILITY_CONFIG.candle_lookback),
         minimum_candles=getattr(args, "minimum_candles", DEFAULT_VOLATILITY_CONFIG.minimum_candles),
         granularity_factor=getattr(args, "granularity_factor", DEFAULT_VOLATILITY_CONFIG.granularity_factor),
         loss_shrink_ratio=getattr(args, "loss_shrink_ratio", DEFAULT_VOLATILITY_CONFIG.loss_shrink_ratio),
         offset_shrink_ratio=getattr(args, "offset_shrink_ratio", DEFAULT_VOLATILITY_CONFIG.offset_shrink_ratio),
         medium_point_spread_normalization_factor=getattr(
            args,
            "medium_point_spread_normalization_factor",
            DEFAULT_VOLATILITY_CONFIG.medium_point_spread_normalization_factor,
         ),
         large_point_spread_normalization_factor=getattr(
            args,
            "large_point_spread_normalization_factor",
            DEFAULT_VOLATILITY_CONFIG.large_point_spread_normalization_factor,
         ),
         volatility_mode=getattr(args, "volatility_mode", DEFAULT_VOLATILITY_CONFIG.mode),
         risk_percent_provider=(
            adaptive_optimizer.risk_percent_for
            if adaptive_optimizer is not None and adaptive_optimizer.config.optimize_risk
            else None
         ),
      ),
      order_builder=OrderBuilder(
         execution_delay=order_config.execution_delay,
         expiration_bars=order_config.expiration_bars,
         candle_interval=order_config.candle_interval,
      ),
      execution_model=execution_model,
      portfolio=portfolio,
      position_manager=PositionManager(),
      trade_monitor=TradeMonitor(),
      trade_closer=trade_closer,
      reports=reports,
      market_state=market_state,
      adaptive_optimizer=adaptive_optimizer,
      trading_schedule=resolve_backtest_trading_schedule(args),
   )
   if bandit_config is not None:
      if not isinstance(strategy, SwingRegressionStrategy):
         raise ValueError("Contextual optimization requires a swing-regression strategy")
      original_rules = dict(strategy.regression_rules)
      factory = getattr(strategy, "shadow_clone_factory", None) or (
         lambda: SwingRegressionStrategy(original_rules, minimum_candles=strategy._minimum_candles_override,
                                          analysis_interval=candle_interval)
      )
      optimizer = ContextualBanditOptimizer(
         bandit_config, memory=getattr(args, "bandit_memory", DEFAULT_MEMORY),
         read_only=bool(getattr(args, "bandit_read_only", False)),
         run_id=getattr(args, "bandit_run_id", None), resume=bool(getattr(args, "bandit_resume", False)),
         provenance={"environment": "backtest", "history": str(getattr(args, "history", "")),
                     "start_date": args.start_date.isoformat() if getattr(args, "start_date", None) else None,
                     "end_date": args.end_date.isoformat() if getattr(args, "end_date", None) else None},
         strategy_factory=factory, source="backtest_executed",
      )
      engine.contextual_optimizer = optimizer
      engine.learning_hooks = optimizer
      optimizer.bind_engine(engine, getattr(args, "symbols", ()))
   if getattr(args, "trade_validator", None) is not None:
      from ultimateBot.backtester.ml.trade_validator import attach_trade_validator
      attach_trade_validator(engine, args)
   if getattr(args, "trade_database", None) is not None:
      from ultimateBot.backtester.embeddings.trade_dataset import attach_trade_dataset
      attach_trade_dataset(engine, args, encoder=engine.trade_validator.encoder if engine.trade_validator else None)
   return engine


def print_symbol_summaries(engine: BacktestEngine) -> None:
   trades = engine.reports.trades
   open_positions_by_symbol: dict[str, int] = {}

   for position in engine.portfolio.positions.values():
      open_positions_by_symbol[position.symbol] = open_positions_by_symbol.get(position.symbol, 0) + 1

   symbols = sorted({trade.symbol for trade in trades} | set(open_positions_by_symbol))
   if not symbols:
      return

   print("Per-symbol results:")
   for symbol in symbols:
      symbol_trades = [trade for trade in trades if trade.symbol == symbol]
      wins = [trade for trade in symbol_trades if trade.pnl > 0]
      losses = [trade for trade in symbol_trades if trade.pnl <= 0]
      pnl = sum(trade.pnl for trade in symbol_trades)
      gross_wins = sum(trade.pnl for trade in wins)
      gross_losses = abs(sum(trade.pnl for trade in losses))
      open_positions = open_positions_by_symbol.get(symbol, 0)
      avg_max_favorable_pnl = sum(trade.max_favorable_pnl for trade in symbol_trades) / max(1, len(symbol_trades))
      avg_max_adverse_pnl = sum(trade.max_adverse_pnl for trade in symbol_trades) / max(1, len(symbol_trades))

      if symbol_trades:
         print(
            f"  {symbol}: closed={len(symbol_trades)} wins={len(wins)} losses={len(losses)} "
            f"win_rate={len(wins) / len(symbol_trades):.2%} pnl={pnl:.2f} "
            f"open={open_positions} profit_factor={gross_wins / max(1.0, gross_losses):.2f} "
            f"avg_mfe={avg_max_favorable_pnl:.2f} avg_mae={avg_max_adverse_pnl:.2f}"
         )
      else:
         print(f"  {symbol}: closed=0 wins=0 losses=0 win_rate=n/a pnl=0.00 open={open_positions} profit_factor=n/a")


def print_swing_regression_diagnostics(engine: BacktestEngine) -> None:
   strategy = engine.strategy
   if getattr(strategy, "fill_on_signal_release", False):
      print(f"Frozen swing model: {strategy.directory}")
      print(f"Swing context: {strategy.window.key}; model available after {strategy.available_after.isoformat()}")
      print(f"Acceptance threshold: {strategy.threshold} (saved={strategy.trained_threshold}; source={strategy.threshold_source})")
      print(f"Minimum acceptance threshold: {strategy.min_threshold}")
      print(f"Structural stop buffer: {strategy.opportunity_config.stop_buffer_pips} pips "
            f"(training={strategy.trained_stop_buffer_pips} pips)")
      if strategy.take_profit_r is None:
         print(f"Trailing activation: {strategy.opportunity_config.trailing_start_r} R "
               f"(training={strategy.trained_trailing_start_r} R)")
      else:
         print(f"Fixed take profit: {strategy.take_profit_r} R; trailing disabled "
               f"(training trailing activation={strategy.trained_trailing_start_r} R)")
      print(f"Model decision traces: {strategy.decision_log}")
      if strategy.daily_target is not None:
         print(f"Dynamic threshold: {json.dumps(strategy.daily_target.summary(), sort_keys=True)}")
         print(f"Initial threshold: {strategy.initial_threshold} (source={strategy.initial_threshold_source})")
      print(f"Swing model decisions: {json.dumps(dict(strategy.counts), sort_keys=True)}")
      print("Accepted signals still pass through the robot's risk and portfolio limits.")
      return
   if isinstance(strategy, TemporalPatternStrategy):
      print("Temporal swing-pattern diagnostics:")
      for name, count in sorted(strategy.tracker.counts.items()):
         print(f"  {name}: {count}")
      for name in ("head-shoulders-breakout", "inverse-head-shoulders-breakout",
                   "head-shoulders-retest", "inverse-head-shoulders-retest", "consolidation-retest"):
         trades = [t for t in engine.reports.trades if engine.trade_entry_modes.get(t.trade_id) == name]
         print(f"  {name}: closed={len(trades)} pnl={sum(t.pnl for t in trades):.2f}")
      return
   if not isinstance(strategy, SwingRegressionStrategy):
      return

   diagnostics = strategy.diagnostics_summary()
   analysis_windows = int(diagnostics.get("analysis_windows", 0))
   reason_counts = diagnostics.get("reason_counts", {})
   entry_mode = strategy.regression_rules["regression_entry_mode"]
   if entry_mode == "hybrid":
      print("Hybrid diagnostics:")
      print(f"  Closed-candle analysis windows: {analysis_windows}")
      hybrid = diagnostics["hybrid"]
      print(f"  Opposite-side confirmation conflicts: {hybrid['conflicting_confirmations']}")
      print(f"  Same-side duplicate confirmations: {hybrid['same_side_duplicates']}")
      print(f"  Complementary confirmations suppressed by primary entries: {hybrid['suppressed_complementary_confirmations']}")
      print(f"  Windows blocked by symbol exposure: {reason_counts.get('hybrid_symbol_busy', 0)}")
      for reason,count in reason_counts.items():
         if reason.startswith("reentry_quality_"):
            print(f"  {reason}: {count}")
      for branch, counts in hybrid["branches"].items():
         trades = [trade for trade in engine.reports.trades if engine.trade_entry_modes.get(trade.trade_id) == branch]
         opened = sum(engine.trade_entry_modes.get(position.trade_id) == branch for position in engine.portfolio.positions.values())
         print(f"  {branch}: raw_confirmations={counts['confirmations']} selected_signals={counts['selected_signals']} "
               f"closed={len(trades)} wins={sum(trade.pnl > 0 for trade in trades)} "
               f"pnl={sum(trade.pnl for trade in trades):.2f} open={opened}")
         for reason, count in sorted(counts["reason_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"    {reason}: {count}")
         print(f"    submission_spread_stop_rejections: {engine.entry_rejection_counts[(branch,'spread_stop_ratio')]}")
      return
   if entry_mode in ("failed-breakout", "breakout-retest"):
      print(f"{entry_mode.capitalize()} diagnostics:")
      print(f"  Closed-candle analysis windows: {analysis_windows}")
      confirmations = analysis_windows - sum(int(count) for count in reason_counts.values())
      label = "reentries" if entry_mode == "failed-breakout" else "breakout retests"
      print(f"  Confirmed {label} before session/risk/validator checks: {confirmations}")
      print("  Rejected or waiting analysis windows by reason:")
      for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
         print(f"    {reason}: {count}")
      return
   low_volatility_rejections = int(reason_counts.get("market_volatility_too_low", 0))
   global_confirmation_rejections = sum(
      int(reason_counts.get(reason, 0))
      for reason in (
         "global_analysis_missing",
         "global_reference_slope_missing",
         "global_reference_slope_direction_mismatch",
         "global_reference_slope_too_weak",
         "global_parallelism_too_low",
      )
   )
   analysis_windows_by_symbol = diagnostics.get("analysis_windows_by_symbol", {})
   reason_counts_by_symbol = diagnostics.get("reason_counts_by_symbol", {})

   print("Swing-regression diagnostics:")
   print(f"  Analysis windows: {analysis_windows}")
   print(
      f"  Low-volatility rejections: {low_volatility_rejections} "
      f"({low_volatility_rejections / max(1, analysis_windows):.2%})"
   )
   level_reaction_rejections = int(reason_counts.get("trigger_swing_without_level_reaction", 0))
   print(
      f"  Support/resistance reaction rejections: {level_reaction_rejections} "
      f"({level_reaction_rejections / max(1, analysis_windows):.2%})"
   )
   candle_confirmation_rejections = sum(int(reason_counts.get(reason, 0)) for reason in (
      "trigger_candle_not_confirmed", "missing_confirmation_candle",
   ))
   if strategy.regression_rules["regression_require_candle_confirmation"]:
      print(
         f"  Candle confirmation rejections: {candle_confirmation_rejections} "
         f"({candle_confirmation_rejections / max(1, analysis_windows):.2%})"
      )
   if global_confirmation_rejections > 0:
      print(
         f"  Global-trend rejections: {global_confirmation_rejections} "
         f"({global_confirmation_rejections / max(1, analysis_windows):.2%})"
      )

   symbols = sorted(
      symbol for symbol in set(analysis_windows_by_symbol) | set(reason_counts_by_symbol)
      if int(analysis_windows_by_symbol.get(symbol, 0)) > 0 or int(reason_counts_by_symbol.get(symbol, {}).get("market_volatility_too_low", 0)) > 0
   )
   if not symbols:
      return

   print("  Low-volatility rejections by symbol:")
   for symbol in symbols:
      symbol_analyses = int(analysis_windows_by_symbol.get(symbol, 0))
      symbol_low_volatility_rejections = int(reason_counts_by_symbol.get(symbol, {}).get("market_volatility_too_low", 0))
      print(
         f"    {symbol}: {symbol_low_volatility_rejections} "
         f"({symbol_low_volatility_rejections / max(1, symbol_analyses):.2%}) / {symbol_analyses} analyses"
      )


def _format_metric_value(value: float | None, *, percent: bool = False) -> str:
   if value is None:
      return "not reached"
   if value == float("inf"):
      return "inf"
   return f"{value:.2%}" if percent else f"{value:.2f}"


def print_backtest_summary(engine: BacktestEngine) -> None:
   portfolio = engine.portfolio
   trades = engine.reports.trades
   print("=======================================================")
   print("Backtest summary")
   print("=======================================================")

   print(f"Initial balance: {portfolio.initial_balance:.2f}")
   print(f"Final balance: {portfolio.balance:.2f}")

   pnl = portfolio.balance - portfolio.initial_balance
   print(f"Total PnL: {pnl:.2f}")

   if portfolio.initial_balance != 0:
      return_pct = pnl / portfolio.initial_balance * 100
      print(f"Return: {return_pct:.2f}%")

   print(f"Total closed trades: {len(trades)}")
   print(f"Open positions: {len(portfolio.positions)}")
   pending_orders = getattr(getattr(engine, "pending_orders_processor", None), "queue", ())
   print(f"Pending orders: {len(pending_orders)}")
   schedule = getattr(engine, "trading_schedule", None)
   if schedule is not None and schedule.config.enabled:
      print(f"Trading time rules: {schedule.config.identity()}")
      print(f"Trading time diagnostics: {schedule.stats}")

   if trades:
      wins = [trade for trade in trades if trade.pnl > 0]
      losses = [trade for trade in trades if trade.pnl <= 0]
      gross_wins = sum(trade.pnl for trade in wins)
      gross_losses = abs(sum(trade.pnl for trade in losses))
      avg_max_favorable_pnl = sum(trade.max_favorable_pnl for trade in trades) / max(1, len(trades))
      avg_max_adverse_pnl = sum(trade.max_adverse_pnl for trade in trades) / max(1, len(trades))
      best_max_favorable_pnl = max(trade.max_favorable_pnl for trade in trades)
      worst_max_adverse_pnl = min(trade.max_adverse_pnl for trade in trades)

      print(f"Winning trades: {len(wins)}")
      print(f"Losing trades: {len(losses)}")
      print(f"Win rate: {len(wins) / max(1, len(trades)):.2%}")
      print(f"Profit factor (all closed trades): {gross_wins / max(1.0, gross_losses):.2f}")
      print(f"Average max favorable PnL: {avg_max_favorable_pnl:.2f}")
      print(f"Average max adverse PnL: {avg_max_adverse_pnl:.2f}")
      print(f"Best max favorable PnL: {best_max_favorable_pnl:.2f}")
      print(f"Worst max adverse PnL: {worst_max_adverse_pnl:.2f}")
   else:
      print("No trades were executed.")

   print_symbol_summaries(engine)
   print_swing_regression_diagnostics(engine)
   bandit = getattr(engine, "contextual_optimizer", None)
   if bandit is not None:
      summary = bandit.summary()
      print(f"Contextual bandit: run_id={bandit.run_id} memory={bandit.store.path} "
            f"changes={bandit.change_count} shadow_variants={len(bandit.shadow.variants)} "
            f"rolling_PF={bandit.main_pf.value} drawdown={bandit.drawdown}")
      for symbol, model in summary["models"].items():
         print(f"  {symbol}: model_id={model['model_id']} loaded={model['loaded']} "
               f"initial={model['initial_action']} final={model['final_action']} "
               f"observations={model['memory']['observations']} memory_on_or_after_start={model['knowledge_at_or_after_run_start']} "
               f"independent_validation={model['independent_validation']}")
      print("  Shadow balances are separate hypothetical portfolios; they are not added to main PnL.")
   optimizer = getattr(engine, "adaptive_optimizer", None)
   if optimizer is not None:
      print(
         f"Adaptive strategy per symbol: switches={optimizer.switch_count} "
         f"previous-configuration closes ignored={optimizer.ignored_closes}"
      )
      for symbol, state in sorted(optimizer.states.items()):
         current = state.current_parameters
         accuracy_text = f"{state.last_accuracy:.2%}" if state.last_accuracy is not None else "no closes yet"
         maximum_accuracy_text = (
            f"{state.maximum_accuracy:.2%}" if state.maximum_accuracy is not None else "not reached"
         )
         metric_text = _format_metric_value(
            state.last_metric_value,
            percent=optimizer.config.optimization_metric == "accuracy",
         ) if state.last_metric_value is not None else "no closes yet"
         reference_metric_text = _format_metric_value(
            state.reference_metric_value,
            percent=optimizer.config.optimization_metric == "accuracy",
         )
         metric_sample_count = (
            len(state.profit_factor.pnl_samples)
            if optimizer.config.optimization_metric == "profit_factor"
            else len(state.accuracy.outcomes)
         )
         metric_window_size = (
            optimizer.config.profit_factor_window
            if optimizer.config.optimization_metric == "profit_factor"
            else optimizer.config.accuracy_window
         )
         print(
            f"  {symbol}: switches={state.switch_count} lookback={current.lookback_bars} "
            f"timeframe_minutes={current.timeframe_minutes} "
            f"global_validation={current.use_global_timeframe_validation} "
            f"trailing={current.trailing_stop_enabled} accuracy={accuracy_text} "
            f"{optimizer.config.optimization_metric}={metric_text} "
            f"reference_{optimizer.config.optimization_metric}={reference_metric_text} "
            f"risk_mode={'adaptive' if optimizer.config.optimize_risk else 'fixed'} "
            f"risk_percent={state.current_risk_percent:g} "
            f"risk_accuracy_ceiling={state.risk_accuracy_ceiling:.2%} "
            f"maximum_accuracy={maximum_accuracy_text} "
            f"metric_sample={metric_sample_count}/{metric_window_size} "
            f"accuracy_sample={len(state.accuracy.outcomes)}/{optimizer.config.accuracy_window} "
            f"search_indices=[{state.search_lower_index},{state.search_upper_index}] "
            f"winner_index={state.winner_index} "
            f"previous-configuration closes ignored={state.ignored_closes}"
         )

   print("=======================================================")


def print_regression_parameters(rules: dict[str, Any]) -> None:
   hybrid = rules["regression_entry_mode"] == "hybrid"
   failed_breakout = rules["regression_entry_mode"] == "failed-breakout" or hybrid
   breakout_retest = rules["regression_entry_mode"] == "breakout-retest" or hybrid
   print(f"Regression entry mode: {rules['regression_entry_mode']}")
   print(f"Maximum submission spread / final planned stop risk: {rules['regression_max_spread_stop_ratio']} (0 = disabled)")
   if rules["regression_pullback_enabled"]:
      print("Experimental trend-pullback parameters (config.pullback):")
      for name,value in RegressionPullbackParameters.from_rules(rules).to_rules().items():
         print(f"  --{name.replace('_','-')} = {value}")
   if rules["regression_impulse_enabled"]:
      print("Experimental breakout-hold parameters (config.impulse):")
      for name,value in RegressionImpulseParameters.from_rules(rules).to_rules().items():
         print(f"  --{name.replace('_','-')} = {value}")
   if failed_breakout:
      print("Failed-breakout dedicated entry parameters (config.reentry):")
      for name, value in RegressionReentryParameters.from_rules(rules).to_rules().items():
         print(f"  --{name.replace('_', '-')} = {value}")
      print("Failed-breakout exits: excursion extreme + frozen ATR buffer / fixed channel-center target")
   if breakout_retest:
      print("Breakout-retest dedicated entry parameters (config.breakout):")
      for name, value in RegressionBreakoutParameters.from_rules(rules).to_rules().items():
         print(f"  --{name.replace('_', '-')} = {value}")
      print("Breakout-retest stop: retest extreme + frozen ATR buffer; reference target uses final stop risk")
      print("Shared regression channel parameters (config.regression):")
   if failed_breakout and not breakout_retest:
      print("Shared regression channel parameters (config.regression):")
   if hybrid:
      print("Hybrid arbitration: first confirmation consumes both branches; opposite-side conflicts rejected")
      print("Hybrid exposure: at most one pending order or open position per symbol")
   for key, label in (
      ("lookback_bars", "Regression lookback bars"),
      ("min_parallelism", "Regression minimum parallelism"),
      ("market_volatility_window_bars", "Regression market volatility window"),
      ("min_market_volatility_ratio", "Regression minimum market volatility"),
      ("line_max_volatility_atr", "Regression line max volatility (ATR)"),
      ("min_consecutive_swing_distance_atr", "Regression minimum consecutive swing distance (ATR)"),
      ("max_filtered_consecutive_swing_pair_density", "Regression max filtered consecutive swing pair density"),
      ("max_swing_density", "Regression max swing density"),
      ("min_swings_per_side", "Regression minimum reference swings"),
      ("live_analysis", "Regression live analysis"),
   ):
      print(f"  {label}: {rules['regression_' + key]}")
   if failed_breakout or breakout_retest:
      print("Directional trend, level-reaction and candle-color gates: disabled in this mode")
      print("Global timeframe validation: disabled; entry confirmation uses the frozen channel event")
      reference = "hybrid" if hybrid else "reentry" if failed_breakout else "breakout"
      print(f"Parameter reference: docs/regression_{reference}.md")
      return
   print(f"Regression score threshold: {rules['regression_score_threshold']}")
   print(f"Regression slope reference: {rules['regression_slope_scale_atr']} ATR per candle")
   print(f"Regression global timeframe validation: {rules['regression_use_global_timeframe_validation']}")
   print(f"Regression global timeframe multiplier: {rules['regression_global_timeframe_multiplier']}x")
   print(f"Regression global reference slope threshold: {rules['regression_global_reference_slope_threshold_atr']} ATR per candle")
   print(f"Regression global minimum parallelism: {rules['regression_global_min_parallelism']}")
   print(f"Regression trigger max distance: {rules['regression_trigger_max_distance_atr']} ATR")
   print(f"Regression support/resistance reaction required: {rules['regression_require_level_reaction']}")
   print(f"Regression candle confirmation required: {rules['regression_require_candle_confirmation']}")
   print(f"Confirmation minimum body/range ratio: {rules['confirmation_body_ratio']}")


def main(regression_config: SwingRegressionConfig | None = None) -> None:
   args = parse_args(regression_config=regression_config)
   structure_rules = resolve_backtest_structure_rules(args)
   regression_rules = resolve_backtest_regression_rules(args)
   failed_breakout = args.strategy == "swing-regression" and regression_rules["regression_entry_mode"] == "failed-breakout"
   channel_entry = args.strategy == "swing-regression" and regression_rules["regression_entry_mode"] in ("failed-breakout", "breakout-retest", "hybrid")
   candle_interval = resolve_backtest_candle_interval(args)

   strategy_analysis_mode = (
      f"{format_candle_interval(candle_interval)} analysis-window batching"
      if args.strategy in ("structure", "swing-regression", "swing-patterns", "swing-ts2vec", "swing-informer", "swing-grid", "swing-nn")
      else f"fixed-size batching ({args.batch_size} ticks)"
   )

   print("=======================================================")
   print("Starting backtest with the following parameters:")
   print(f"History path: {args.history}")
   print(f"Symbols: {args.symbols}")
   print(f"Start date: {format_backtest_bound(args.start_date)}")
   print(f"End date: {format_backtest_bound(args.end_date)}")
   print(f"Initial balance: {args.balance}")
   print(f"Batch size: {args.batch_size}")
   print(f"Batching mode: {strategy_analysis_mode}")
   print(f"Candle interval: {format_candle_interval(candle_interval)}")
   print(f"Strategy: {args.strategy}")
   if args.strategy == "structure":
      print(f"Structure min RR ratio: {structure_rules['min_rr_ratio']}")
   if args.strategy == "swing-regression":
      print_regression_parameters(regression_rules)
   if args.strategy == "swing-patterns":
      print(f"Temporal swing groups (experimental): {resolve_temporal_pattern_config(args)}")
   if args.strategy in ("swing-ts2vec", "swing-informer", "swing-grid", "swing-nn"):
      from ultimateBot.backtester.strategy.swing_ts2vec_strategy import model_configuration, resolve_swing_threshold
      from ultimateBot.backtester.strategy.swing_threshold import daily_target_config
      directory, metadata, opportunity, window = model_configuration(args.swing_ts2vec_model)
      print(f"Frozen swing model: {directory}")
      threshold = resolve_swing_threshold(metadata, getattr(args, "swing_ts2vec_threshold", None),
                                          getattr(args, "swing_ts2vec_min_threshold", None))
      print(f"Swing context: {window.key}; threshold={threshold}; saved_threshold={metadata['threshold']}; target={metadata['target']}")
      print(f"Minimum acceptance threshold: {getattr(args, 'swing_ts2vec_min_threshold', None)}")
      print(f"Model decision traces: {args.swing_ts2vec_decision_log}")
      if args.swing_ts2vec_target_trades_per_day is not None:
         cadence = daily_target_config(args.swing_ts2vec_target_trades_per_day, args.swing_ts2vec_threshold_step)
         print(f"Accepted-signal cadence (all symbols, UTC): {cadence.trades_per_day}/day; "
               f"objective=1 per {cadence.period_days:g} days; threshold step={cadence.threshold_step}; "
               "raise immediately after extra acceptances, lower after a period with none")
      stop_buffer = (opportunity.stop_buffer_pips if args.swing_ts2vec_stop_buffer_pips is None
                     else args.swing_ts2vec_stop_buffer_pips)
      print(f"Structural stop buffer: {stop_buffer} pips (training={opportunity.stop_buffer_pips} pips)")
      if args.swing_ts2vec_take_profit_r is None:
         print(f"Trailing activation: {args.trailing_start_rr_ratio} R (training={opportunity.trailing_start_r} R)")
      else:
         print(f"Fixed take profit: {args.swing_ts2vec_take_profit_r} R; trailing disabled "
               f"(training trailing activation={opportunity.trailing_start_r} R)")
      print(f"Entries start after {metadata['available_after']}; earlier candles only warm up swing history")
   if args.strategy in ("structure", "swing-regression") and not channel_entry:
      active_rules = regression_rules if args.strategy == "swing-regression" else structure_rules
      print(f"Support/resistance lookback bars: {active_rules['level_lookback_bars']}")
      print(f"Protective stop source: {active_rules['protective_level_source']}")
      print(f"Protective stop buffer: {active_rules['protective_stop_buffer_atr']} ATR")
   print(f"Quantity override: {args.quantity}")
   print(f"Risk percent: {args.risk_percent}")
   print(f"Available money ratio: {args.available_money_ratio}")
   print(f"Available money source: {args.available_money_source}")
   print(f"Accepted risk overrun: {args.accepted_risk_overrun}")
   print(f"Max positions: {args.max_positions}")
   order_config = resolve_backtest_order_config(args, candle_interval=candle_interval)
   print(
      f"Order lifetime: {order_config.expiration_bars} x {format_candle_interval(order_config.candle_interval)} "
      f"({order_config.expiration_delay.total_seconds():g} seconds)"
   )
   if failed_breakout:
      print("Position exit mode: fixed channel-center target; minimum entry RR is config.reentry.min_rr")
   elif args.strategy == "swing-regression" and regression_rules["regression_entry_mode"] == "hybrid":
      print("Position exit mode: failed-breakout = fixed channel center; breakout-retest = "
            + ("trailing stop" if args.trailing_stop else "fixed reference TP"))
      print(f"Breakout-retest reference reward/risk ratio: {args.reward_risk_ratio}")
   else:
      print(f"Reward/risk ratio: {args.reward_risk_ratio}")
      print(f"Position exit mode: {'trailing stop only (separate activation, breakeven, then one-risk steps)' if args.trailing_stop else 'fixed take profit'}")
   adaptive_config = resolve_backtest_adaptive_config(args)
   if args.trailing_stop or adaptive_config is not None and adaptive_config.optimize_trailing_stop:
      if args.strategy not in ('swing-ts2vec', 'swing-informer', 'swing-grid', 'swing-nn') or args.trailing_stop:
         print(f"Trailing start RR ratio: {resolve_trailing_start_rr_ratio(args.reward_risk_ratio, args.trailing_start_rr_ratio)}")
   if adaptive_config is not None:
      adaptive_trailing_values, adaptive_global_validation_values = resolve_backtest_adaptive_choice_values(
         args,
         adaptive_config,
         regression_rules,
      )
      adaptive_risk_text = (
         f"range=[{adaptive_config.min_risk_percent:g}%, {args.risk_percent:g}%]"
         if adaptive_config.optimize_risk
         else f"fixed={args.risk_percent:g}%"
      )
      print(
         f"Adaptive strategy per symbol: lookbacks=[{adaptive_config.lookback_min}, {adaptive_config.lookback_max}) "
         f"timeframes_minutes=[{adaptive_config.timeframe_min_minutes}, {adaptive_config.timeframe_max_minutes}) "
         f"timeframe_step={adaptive_config.timeframe_step_minutes} "
         f"step={adaptive_config.lookback_step} "
         f"trailing_values={list(adaptive_trailing_values)} "
         f"global_validation_values={list(adaptive_global_validation_values)} "
         f"optimization_metric={adaptive_config.optimization_metric} "
         f"candidates={len(adaptive_config.candidates(trailing_stop_values=adaptive_trailing_values, global_timeframe_validation_values=adaptive_global_validation_values))} "
         f"accuracy_window={adaptive_config.accuracy_window} "
         f"accuracy_drop={adaptive_config.accuracy_drop_percent}% ({adaptive_config.accuracy_drop_mode}) "
         f"minimum_accuracy={adaptive_config.min_accuracy_percent}% "
         f"profit_factor_window={adaptive_config.profit_factor_window} "
         f"profit_factor_drop={adaptive_config.profit_factor_drop_percent} ({adaptive_config.profit_factor_drop_mode}) "
         f"minimum_profit_factor={adaptive_config.min_profit_factor} "
         f"tighten_search_interval={adaptive_config.tighten_search_interval} "
         f"search_arm_deltas={adaptive_config.search_long_arm_delta}/{adaptive_config.search_short_arm_delta} "
         f"optimize_risk={adaptive_config.optimize_risk} "
         f"risk={adaptive_risk_text} "
         f"seed={adaptive_config.seed}"
      )
   print(f"Limit spread ratio: {args.limit_spread_ratio}")
   print(f"Max spread ratio: {args.max_spread_ratio}")
   print(f"Risk candle lookback (volatility/SL/TP/spread): {args.candle_lookback}")
   print(f"Minimum candles: {args.minimum_candles}")
   print(f"Granularity factor: {args.granularity_factor}")
   print(f"Loss shrink ratio: {args.loss_shrink_ratio}")
   print(f"Offset shrink ratio: {args.offset_shrink_ratio}")
   print(f"Medium-point spread normalization factor: {args.medium_point_spread_normalization_factor}")
   print(f"Large-point spread normalization factor: {args.large_point_spread_normalization_factor}")
   print(f"Volatility mode: {args.volatility_mode}")
   print("=======================================================")

   feed = build_feed(
      history_path=args.history,
      symbols=args.symbols,
      start_date=args.start_date,
      end_date=args.end_date,
   )
   engine = build_engine(feed=feed, args=args)

   print("Running backtest...")
   engine.run()
   print("Backtest completed.")
   print("=======================================================")
   print_backtest_summary(engine)
   if engine.contextual_optimizer is not None:
      from ultimateBot.backtester.optimization.bandit_core import canonical_json
      report_path = getattr(args, "bandit_report", None) or Path(__file__).parent / "artifacts" / "bandit_reports" / f"{engine.contextual_optimizer.run_id}.json"
      report_path.parent.mkdir(parents=True, exist_ok=True)
      report_path.write_text(json.dumps(json.loads(canonical_json(engine.contextual_optimizer.summary())), indent=2,
                                        ensure_ascii=False), encoding="utf-8")
      print(f"Bandit report: {report_path}")


if __name__ == "__main__":
   main()
