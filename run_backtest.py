#!/usr/bin/env python3
import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ultimateBot.backtester.config.backtest_config import (
   RiskConfig,
   VolatilityConfig,
   format_backtest_bound,
   format_candle_interval,
   normalize_candle_interval,
   resolve_backtest_bounds,
)
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.file_tick_source import JForexTickSource
from ultimateBot.backtester.data.merged_feed import MergedTickFeed
from ultimateBot.backtester.engine.backtest_engine import BacktestEngine
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.structure_decision_engine import resolve_structure_rules
from ultimateBot.backtester.strategy.dummy_strategy import DummyStrategy
from ultimateBot.backtester.strategy.structure_strategy import StructureStrategy

BASE_HISTORY_PATH = Path(r"D:\jforex_history")

all_symbols = [
   #"BTCUSD",
   #"ETHUSD",
   "EURGBP",
   "EURCHF",
   "EURUSD",
   "EURCAD",
   "EURAUD",
   "EURNZD",
   "USDCHF",
   "USDCAD",
   "GBPUSD",
   "GBPCAD",
   "GBPCHF",
   "GBPAUD",
   "GBPNZD",
]

btc_eur_symbols = ["BTCUSD", "EURUSD"]
eur_symbol = ["EURUSD"]
DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()
DEFAULT_STRUCTURE_RULES = resolve_structure_rules(None)
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2020


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
      default=50.0,
      help="Initial balance for the backtest",
   )
   parser.add_argument(
      "--batch-size",
      type=int,
      default=500,
      help="Batch size for the backtest",
   )
   parser.add_argument(
      "--strategy",
      type=str,
      choices=["structure", "dummy"],
      default="structure",
      help="Strategy to run: the legacy market-structure adapter or the dummy smoke strategy",
   )
   parser.add_argument(
      "--candle-interval",
      type=str,
      default=format_candle_interval(DEFAULT_VOLATILITY_CONFIG.candle_interval),
      help="Base candle period shared by market-state candles, structure analysis and risk volatility (examples: 15m, 1h, 4h, 1d)",
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
      help="Protective stop source for structure signals: origin structure level or most recent swing",
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
      help="Risk percentage applied to the selected available capital budget for each trade (for example 25.0 means 25 percent)",
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
      help="Take-profit distance expressed as a multiple of the stop distance",
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
      help="Number of candles used to estimate recent volatility",
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
      "--volatility-mode",
      type=str,
      choices=["swing", "intraday"],
      default=DEFAULT_VOLATILITY_CONFIG.mode,
      help="Controls whether volatility scaling rounds up or down",
   )

   args = parser.parse_args()
   try:
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


def resolve_backtest_structure_rules(args: argparse.Namespace) -> dict[str, Any]:
   return resolve_structure_rules(
      {
         "min_rr_ratio": getattr(args, "structure_min_rr_ratio", DEFAULT_STRUCTURE_RULES["min_rr_ratio"]),
         "protective_level_source": getattr(args, "structure_stop_source", DEFAULT_STRUCTURE_RULES["protective_level_source"]),
         "live_level_count": bool(getattr(args, "structure_live_level_count", False)),
         "live_level_count_every": max(1, int(getattr(args, "structure_live_level_count_every", 1))),
      }
   )


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
   strategy = strategy_override
   if strategy is None and args.strategy == "dummy":
      strategy = DummyStrategy(seed=42)

   candle_interval = resolve_backtest_candle_interval(
      args,
      fallback=getattr(strategy, "analysis_interval", None),
   )

   if isinstance(strategy, DummyStrategy):
      market_state = MarketState(candle_interval=candle_interval)
   else:
      if strategy is None:
         strategy = StructureStrategy(
            structure_rules=resolve_backtest_structure_rules(args),
            analysis_interval=candle_interval,
         )
      elif hasattr(strategy, "analysis_interval"):
         strategy.analysis_interval = candle_interval
      # The market state must retain at least as many closed candles as the structure engine requires.
      market_state = MarketState(
         candle_interval=candle_interval,
         candle_lookback=max(
            int(getattr(strategy, "minimum_candles", 30)),
            int(getattr(strategy, "structure_rules", {}).get("level_lookback_bars", getattr(strategy, "minimum_candles", 30))),
         ),
      )

   analysis_interval = getattr(strategy, "analysis_interval", None)

   return BacktestEngine(
      feed=feed,
      batcher=TickBatcher(batch_size=args.batch_size, analysis_interval=analysis_interval),
      strategy=strategy,
      risk_manager=RiskManager(
         risk_per_trade=args.risk_percent,
         max_positions=args.max_positions,
         available_money_ratio=args.available_money_ratio,
         available_money_source=args.available_money_source,
         accepted_risk_overrun=args.accepted_risk_overrun,
         reward_risk_ratio=getattr(args, "reward_risk_ratio", DEFAULT_RISK_CONFIG.reward_risk_ratio),
         limit_spread_ratio=getattr(args, "limit_spread_ratio", DEFAULT_RISK_CONFIG.limit_spread_ratio),
         max_spread_ratio=getattr(args, "max_spread_ratio", DEFAULT_RISK_CONFIG.max_spread_ratio),
         fixed_quantity=args.quantity,
         candle_interval=candle_interval,
         candle_lookback=getattr(args, "candle_lookback", DEFAULT_VOLATILITY_CONFIG.candle_lookback),
         minimum_candles=getattr(args, "minimum_candles", DEFAULT_VOLATILITY_CONFIG.minimum_candles),
         granularity_factor=getattr(args, "granularity_factor", DEFAULT_VOLATILITY_CONFIG.granularity_factor),
         loss_shrink_ratio=getattr(args, "loss_shrink_ratio", DEFAULT_VOLATILITY_CONFIG.loss_shrink_ratio),
         offset_shrink_ratio=getattr(args, "offset_shrink_ratio", DEFAULT_VOLATILITY_CONFIG.offset_shrink_ratio),
         volatility_mode=getattr(args, "volatility_mode", DEFAULT_VOLATILITY_CONFIG.mode),
      ),
      order_builder=OrderBuilder(),
      execution_model=execution_model,
      portfolio=portfolio,
      position_manager=PositionManager(),
      trade_monitor=TradeMonitor(),
      trade_closer=TradeCloser(),
      reports=reports,
      market_state=market_state,
   )


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
      print(f"Profit factor: {gross_wins / max(1.0, gross_losses):.2f}")
      print(f"Average max favorable PnL: {avg_max_favorable_pnl:.2f}")
      print(f"Average max adverse PnL: {avg_max_adverse_pnl:.2f}")
      print(f"Best max favorable PnL: {best_max_favorable_pnl:.2f}")
      print(f"Worst max adverse PnL: {worst_max_adverse_pnl:.2f}")
   else:
      print("No trades were executed.")

   print_symbol_summaries(engine)

   print("=======================================================")


def main() -> None:
   args = parse_args()
   structure_rules = resolve_backtest_structure_rules(args)
   candle_interval = resolve_backtest_candle_interval(args)

   strategy_analysis_mode = "hourly analysis-window batching" if args.strategy == "structure" else f"fixed-size batching ({args.batch_size} ticks)"

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
      print(f"Structure stop source: {structure_rules['protective_level_source']}")
   print(f"Quantity override: {args.quantity}")
   print(f"Risk percent: {args.risk_percent}")
   print(f"Available money ratio: {args.available_money_ratio}")
   print(f"Available money source: {args.available_money_source}")
   print(f"Accepted risk overrun: {args.accepted_risk_overrun}")
   print(f"Max positions: {args.max_positions}")
   print(f"Reward/risk ratio: {args.reward_risk_ratio}")
   print(f"Limit spread ratio: {args.limit_spread_ratio}")
   print(f"Max spread ratio: {args.max_spread_ratio}")
   print(f"Candle lookback: {args.candle_lookback}")
   print(f"Minimum candles: {args.minimum_candles}")
   print(f"Granularity factor: {args.granularity_factor}")
   print(f"Loss shrink ratio: {args.loss_shrink_ratio}")
   print(f"Offset shrink ratio: {args.offset_shrink_ratio}")
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


if __name__ == "__main__":
   main()