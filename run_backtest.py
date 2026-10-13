#!/usr/bin/env python3
import argparse
import json
from dataclasses import fields
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from typing import Any

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
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy

BASE_HISTORY_PATH = Path(r"D:\jforex_history")

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


def parse_args() -> argparse.Namespace:
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
      default=50,
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
      choices=["structure", "swing-regression", "dummy"],
      default="structure",
      help="Decision engine to run: market structure, swing regressions, or the dummy smoke strategy",
   )
   parser.add_argument(
      "--candle-interval",
      type=str,
      default=format_candle_interval(DEFAULT_VOLATILITY_CONFIG.candle_interval),
      help="Base candle period shared by market-state candles, strategy analysis and risk volatility (examples: 15m, 1h, 4h, 1d)",
   )
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

   args = parser.parse_args()
   try:
      resolve_trailing_start_rr_ratio(args.reward_risk_ratio, args.trailing_start_rr_ratio)
      if args.regression_lookback_bars is not None and args.regression_lookback_bars < 2:
         raise ValueError("--regression-lookback-bars must be at least 2")
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


def resolve_backtest_order_config(
   args: argparse.Namespace, candle_interval: timedelta | None = None,
) -> OrderConfig:
   bars = getattr(args, "order_expiration_bars", None)
   return OrderConfig(
      execution_delay=DEFAULT_ORDER_CONFIG.execution_delay,
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
         "min_rr_ratio": getattr(args, "structure_min_rr_ratio", DEFAULT_STRUCTURE_RULES["min_rr_ratio"]),
         "protective_level_source": getattr(args, "structure_stop_source", DEFAULT_STRUCTURE_RULES["protective_level_source"]),
         "protective_stop_buffer_atr": getattr(args, "structure_stop_buffer_atr", DEFAULT_STRUCTURE_RULES["protective_stop_buffer_atr"]),
         "live_level_count": bool(getattr(args, "structure_live_level_count", False)),
         "live_level_count_every": max(1, int(getattr(args, "structure_live_level_count_every", 1))),
      }
   )


def resolve_backtest_regression_rules(args: argparse.Namespace) -> dict[str, Any]:
   rules = resolve_backtest_structure_rules(args)
   lookback = getattr(args, "level_lookback_bars", None)
   rules.update({
      "level_lookback_bars": DEFAULT_REGRESSION_RULES["level_lookback_bars"] if lookback is None else lookback,
      "regression_lookback_bars": getattr(args, "regression_lookback_bars", None),
      "regression_score_threshold": getattr(args, "regression_score_threshold", DEFAULT_REGRESSION_RULES["regression_score_threshold"]),
      "regression_min_parallelism": getattr(args, "regression_min_parallelism", DEFAULT_REGRESSION_RULES["regression_min_parallelism"]),
      "regression_slope_scale_atr": getattr(args, "regression_slope_scale_atr", DEFAULT_REGRESSION_RULES["regression_slope_scale_atr"]),
      "regression_use_global_timeframe_validation": getattr(
         args,
         "regression_global_timeframe_validation",
         DEFAULT_REGRESSION_RULES["regression_use_global_timeframe_validation"],
      ),
      "regression_global_timeframe_multiplier": getattr(args, "regression_global_timeframe_multiplier", DEFAULT_REGRESSION_RULES["regression_global_timeframe_multiplier"]),
      "regression_global_reference_slope_threshold_atr": getattr(args, "regression_global_reference_slope_threshold_atr", DEFAULT_REGRESSION_RULES["regression_global_reference_slope_threshold_atr"]),
      "regression_global_min_parallelism": getattr(args, "regression_global_min_parallelism", DEFAULT_REGRESSION_RULES["regression_global_min_parallelism"]),
      "regression_global_signal_lookback_bars": getattr(args, "regression_global_signal_lookback_bars", DEFAULT_REGRESSION_RULES["regression_global_signal_lookback_bars"]),
      "regression_market_volatility_window_bars": getattr(args, "regression_market_volatility_window_bars", DEFAULT_REGRESSION_RULES["regression_market_volatility_window_bars"]),
      "regression_min_market_volatility_ratio": getattr(args, "regression_min_market_volatility_ratio", DEFAULT_REGRESSION_RULES["regression_min_market_volatility_ratio"]),
      "regression_line_max_volatility_atr": getattr(args, "regression_line_max_volatility_atr", DEFAULT_REGRESSION_RULES["regression_line_max_volatility_atr"]),
      "regression_trigger_max_distance_atr": getattr(args, "regression_trigger_max_distance_atr", DEFAULT_REGRESSION_RULES["regression_trigger_max_distance_atr"]),
      "regression_min_consecutive_swing_distance_atr": getattr(args, "regression_min_consecutive_swing_distance_atr", DEFAULT_REGRESSION_RULES["regression_min_consecutive_swing_distance_atr"]),
      "regression_max_filtered_consecutive_swing_pair_density": getattr(args, "regression_max_filtered_consecutive_swing_pair_density", DEFAULT_REGRESSION_RULES["regression_max_filtered_consecutive_swing_pair_density"]),
      "regression_max_swing_density": getattr(args, "regression_max_swing_density", DEFAULT_REGRESSION_RULES["regression_max_swing_density"]),
      "regression_min_swings_per_side": getattr(args, "regression_min_swings", DEFAULT_REGRESSION_RULES["regression_min_swings_per_side"]),
      "regression_live_analysis": bool(getattr(args, "regression_live_analysis", False)),
      "regression_require_level_reaction": getattr(args, "regression_level_reaction", DEFAULT_REGRESSION_RULES["regression_require_level_reaction"]),
   })
   return resolve_swing_regression_rules(rules)


def resolve_backtest_candle_interval(args: argparse.Namespace, fallback: Any | None = None) -> timedelta:
   default_interval = normalize_candle_interval(
      fallback if fallback is not None else DEFAULT_VOLATILITY_CONFIG.candle_interval,
      default=DEFAULT_VOLATILITY_CONFIG.candle_interval,
   )
   return normalize_candle_interval(
      getattr(args, "candle_interval", default_interval),
      default=default_interval,
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
            )
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
         limit_spread_ratio=getattr(args, "limit_spread_ratio", DEFAULT_RISK_CONFIG.limit_spread_ratio),
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
   if not isinstance(strategy, SwingRegressionStrategy):
      return

   diagnostics = strategy.diagnostics_summary()
   analysis_windows = int(diagnostics.get("analysis_windows", 0))
   reason_counts = diagnostics.get("reason_counts", {})
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


def main() -> None:
   args = parse_args()
   structure_rules = resolve_backtest_structure_rules(args)
   regression_rules = resolve_backtest_regression_rules(args)
   candle_interval = resolve_backtest_candle_interval(args)

   strategy_analysis_mode = (
      f"{format_candle_interval(candle_interval)} analysis-window batching"
      if args.strategy in ("structure", "swing-regression")
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
      print(f"Regression score threshold: {regression_rules['regression_score_threshold']}")
      print(f"Regression minimum parallelism: {regression_rules['regression_min_parallelism']}")
      print(f"Regression slope reference: {regression_rules['regression_slope_scale_atr']} ATR per candle")
      print(f"Regression global timeframe validation: {regression_rules['regression_use_global_timeframe_validation']}")
      print(f"Regression global timeframe multiplier: {regression_rules['regression_global_timeframe_multiplier']}x")
      print(f"Regression global reference slope threshold: {regression_rules['regression_global_reference_slope_threshold_atr']} ATR per candle")
      print(f"Regression global minimum parallelism: {regression_rules['regression_global_min_parallelism']}")
      print(f"Regression market volatility window: {regression_rules['regression_market_volatility_window_bars']} candles")
      print(f"Regression minimum market volatility: {regression_rules['regression_min_market_volatility_ratio']}")
      print(f"Regression line max volatility: {regression_rules['regression_line_max_volatility_atr']} ATR")
      print(f"Regression trigger max distance: {regression_rules['regression_trigger_max_distance_atr']} ATR")
      print(f"Regression support/resistance reaction required: {regression_rules['regression_require_level_reaction']}")
      print(f"Regression minimum consecutive swing distance: {regression_rules['regression_min_consecutive_swing_distance_atr']} ATR")
      print(f"Regression max filtered consecutive swing pair density: {regression_rules['regression_max_filtered_consecutive_swing_pair_density']}")
      print(f"Regression max swing density: {regression_rules['regression_max_swing_density']}")
      print(f"Regression minimum reference swings: {regression_rules['regression_min_swings_per_side']}")
   if args.strategy in ("structure", "swing-regression"):
      active_rules = regression_rules if args.strategy == "swing-regression" else structure_rules
      print(f"Support/resistance lookback bars: {active_rules['level_lookback_bars']}")
      if args.strategy == "swing-regression":
         print(f"Regression lookback bars: {active_rules['regression_lookback_bars']}")
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
   print(f"Reward/risk ratio: {args.reward_risk_ratio}")
   print(f"Position exit mode: {'trailing stop only (separate activation, breakeven, then one-risk steps)' if args.trailing_stop else 'fixed take profit'}")
   adaptive_config = resolve_backtest_adaptive_config(args)
   if args.trailing_stop or adaptive_config is not None and adaptive_config.optimize_trailing_stop:
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
