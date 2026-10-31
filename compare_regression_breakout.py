"""Paired, fixed-parameter tick replays of legacy and confirmed breakout entries."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from ultimateBot.run_backtest import build_engine, build_feed, parse_args
from ultimateBot.run_regression_breakout import breakout_config


def summarize(engine, initial_risks):
    records = []
    cumulative_r = peak_r = drawdown_r = 0.0
    for trade in engine.reports.trades:
        risk = initial_risks[trade.trade_id]
        pnl_r = trade.pnl / risk
        cumulative_r += pnl_r
        peak_r = max(peak_r, cumulative_r)
        drawdown_r = max(drawdown_r, peak_r - cumulative_r)
        records.append({
            "symbol": trade.symbol, "side": trade.side.name,
            "opened_at": trade.opened_at.isoformat(), "closed_at": trade.closed_at.isoformat(),
            "pnl_r": pnl_r, "mfe_r": trade.max_favorable_pnl / risk,
            "mae_r": trade.max_adverse_pnl / risk, "close_reason": trade.close_reason.name.lower(),
        })
    gains = sum(max(row["pnl_r"], 0) for row in records)
    losses = sum(max(-row["pnl_r"], 0) for row in records)
    return {
        "closed_trades": len(records), "wins": sum(row["pnl_r"] > 0 for row in records),
        "total_r": cumulative_r, "expectancy_r": cumulative_r / len(records) if records else None,
        "profit_factor_r": gains / losses if losses else None,
        "realized_drawdown_r": drawdown_r,
        "trailing_exits": sum(row["close_reason"] == "trailing_stop" for row in records),
        "open_positions": len(engine.portfolio.positions),
        "pending_orders": sum(engine.pending_orders_processor.queue.count_for_symbol(symbol)
                              for symbol in engine.pending_orders_processor.queue.symbols()),
        "return_percent": (engine.portfolio.balance / engine.portfolio.initial_balance - 1) * 100,
        "diagnostics": engine.strategy.diagnostics_summary(), "trades": records,
    }


def run_comparison(history, symbols, periods, output):
    output.mkdir(parents=True, exist_ok=True)
    tracker_path = Path(__file__).parent / "backtester/strategy/regression_breakout.py"
    tracker_source = tracker_path.read_bytes()
    (output / "tracker_evaluated.py").write_bytes(tracker_source)
    settings = breakout_config()
    settings = replace(settings, risk=replace(settings.risk, risk_per_trade_percent=0.1))
    results = []
    report = {
        "contract": "fixed-entry-rules-identical-risk-trailing-tick-replays-v1",
        "note": "No parameter search; small samples do not establish an advantage. Drawdown is realized R only.",
        "history": str(history), "parameters": asdict(settings), "results": results,
        "tracker_sha256": hashlib.sha256(tracker_source).hexdigest(),
    }
    for label, start, end in periods:
        if start > end:
            raise ValueError("Period start must not be after its end")
        for symbol in symbols:
            for mode in ("legacy", "confirmed"):
                config = replace(settings, breakout=replace(settings.breakout, confirmation_mode=mode))
                previous_argv = sys.argv
                try:
                    sys.argv = ["comparison", "--symbols", symbol, "--start-date", start.isoformat(),
                                "--end-date", end.isoformat(), "--history", str(history), "--session-filter"]
                    arguments = parse_args(config)
                finally:
                    sys.argv = previous_argv
                engine = build_engine(build_feed(history, [symbol], start, end), arguments)
                initial_risks = {}
                open_position = engine.position_manager.open_position

                def record_open(*args, **kwargs):
                    position = open_position(*args, **kwargs)
                    initial_risks[position.trade_id] = (abs(position.entry_price - position.stop_loss)
                                                        * position.quantity * position.profit_multiplier)
                    return position

                engine.position_manager.open_position = record_open
                stem = f"{label}_{symbol.lower()}_{mode}"
                started = perf_counter()
                with (output / f"{stem}.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
                    engine.run()
                result = {"period": label, "symbol": symbol, "mode": mode,
                          "start": start.isoformat(), "end": end.isoformat(),
                          "elapsed_seconds": perf_counter() - started,
                          **summarize(engine, initial_risks)}
                if not result["diagnostics"]["analysis_windows"]:
                    raise ValueError(f"No analysis windows for {stem}; check historical data coverage")
                results.append(result)
                (output / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                print(json.dumps({key: value for key, value in result.items() if key not in ("trades", "diagnostics")}), flush=True)
    report["source_changed_during_run"] = hashlib.sha256(tracker_path.read_bytes()).hexdigest() != report["tracker_sha256"]
    (output / "comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("C:/jforex_history"))
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD"])
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 2))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 1, 12))
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2024, 1, 15))
    parser.add_argument("--validation-end", type=date.fromisoformat, default=date(2024, 1, 26))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/regression_breakout_detection_v2/comparison")
    args = parser.parse_args()
    if args.end_date >= args.validation_start:
        parser.error("Validation must start strictly after the comparison period ends")
    run_comparison(args.history, args.symbols,
                   [("comparison", args.start_date, args.end_date),
                    ("validation", args.validation_start, args.validation_end)], args.output)


if __name__ == "__main__":
    main()
