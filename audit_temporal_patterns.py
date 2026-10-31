"""Count causal pattern confirmations on true closed candles, without simulating PnL.

Examples (from the parent directory):
python -m ultimateBot.audit_temporal_patterns --start-date 2024-01-01 --end-date 2025-12-31
Extra --pattern-* arguments are passed to the regular backtest CLI.
"""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from ultimateBot.audit_breakout_frequency import timestamp
from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.research_regression_complement import load_history
from ultimateBot.run_backtest import build_engine, parse_args


def scan(strategy, symbol, rows):
    history = {source: [Candle(timestamp(r["start_ms"]), *r[source]) for r in rows]
               for source in ("bid", "ask", "mid")}
    capacity = strategy.history_capacity_for_lookback(strategy.history_lookback_bars)
    current = 0
    def get_candles(_symbol, *, price_source, include_active, limit, candle_interval):
        assert not include_active
        return history[price_source][max(0, current - limit + 1):current + 1]
    market = SimpleNamespace(get_candles=get_candles, last_ticks={})
    context = StrategyContext(Portfolio(10000), market)
    signals = []
    for current in range(capacity - 1, len(rows) - 1):
        row, next_row = rows[current], rows[current + 1]
        # The real first quote after the close releases the decision. No future bar enters history.
        tick = Tick(timestamp(next_row["first_ms"]), symbol, row["bid"][3], row["ask"][3], 0, 0)
        market.last_ticks[symbol] = tick
        signals.extend(strategy.on_market_data(TickBatch(symbol, [tick]), context))
    return signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("C:/jforex_history"))
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDCAD", "USDCHF"])
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/temporal_patterns")
    args, extra = parser.parse_known_args()
    previous = sys.argv
    try:
        sys.argv = ["audit", "--strategy", "swing-patterns", "--candle-interval", "60m", *extra]
        strategy = build_engine([], parse_args()).strategy
    finally:
        sys.argv = previous
    signals = []
    summary = {"start": str(args.start_date), "end": str(args.end_date), "config": vars(strategy.config),
               "method": "Causal closed-candle detection; always flat; no execution or profitability simulation.",
               "by_symbol": {}}
    cache = Path(__file__).parent / "artifacts/hybrid_trade_audit/candle_cache"
    for symbol in args.symbols:
        rows = load_history(args.history, symbol, args.start_date, args.end_date, 60, cache)
        before = Counter(strategy.tracker.counts)
        found = scan(strategy, symbol, rows)
        summary["by_symbol"][symbol] = {"candles": len(rows), "confirmations": len(found),
                "by_pattern": dict(Counter(s.entry_context["pattern"] for s in found)),
                "diagnostics": dict(strategy.tracker.counts - before)}
        signals.extend({"symbol": s.symbol, "timestamp": s.timestamp.isoformat(), "side": s.side.name,
                        "stop": s.stop_loss_price, **s.entry_context} for s in found)
        print(f"{symbol}: {len(found)} raw confirmations", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary["total_confirmations"] = len(signals)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "signals.json").write_text(json.dumps(signals, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
