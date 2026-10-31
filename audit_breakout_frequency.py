"""Audit closed-candle signals and isolated risk checks, without simulating fills or PnL.

Defaults to the user's fixed 15-bar, 30-minute, 2.3R breakout configuration. Daily TSVs
are aggregated from synchronous quotes; an on-disk OHLC cache avoids repeated
full tick parsing. The actual strategy and risk manager evaluate every close.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.run_backtest import build_engine, parse_args


def aggregate_quotes(frame, interval_ms):
    if frame.empty:
        return []
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Quote timestamps must be chronological")
    if not np.isfinite(frame[["bid", "ask"]].to_numpy()).all():
        raise ValueError("Non-finite quotes cannot be audited")
    frame = frame.loc[(frame.bid > 0) & (frame.ask > 0)].copy()
    frame["mid"] = (frame.bid + frame.ask) / 2
    frame["bucket"] = frame.timestamp // interval_ms * interval_ms
    for name in ("bid_volume", "ask_volume"):
        valid = np.isfinite(frame[name]) & (frame[name] >= 0)
        frame[name + "_valid"] = valid.astype(int)
        frame[name + "_clean"] = frame[name].where(valid, 0)
    aggregation = {"timestamp": ["first", "last"], "bid": ["first", "max", "min", "last", "size"],
                   "ask": ["first", "max", "min", "last"], "mid": ["first", "max", "min", "last"],
                   "bid_volume_clean": ["sum"], "bid_volume_valid": ["sum"],
                   "ask_volume_clean": ["sum"], "ask_volume_valid": ["sum"]}
    grouped = frame.groupby("bucket", sort=True).agg(aggregation)
    rows = []
    for bucket, values in grouped.iterrows():
        row = {"start_ms": int(bucket), "first_ms": int(values["timestamp", "first"]),
               "last_ms": int(values["timestamp", "last"]), "count": int(values["bid", "size"]),
               "volumes": [float(values["bid_volume_clean", "sum"]), float(values["ask_volume_clean", "sum"]),
                           int(values["bid_volume_valid", "sum"]), int(values["ask_volume_valid", "sum"])]}
        for source in ("bid", "ask", "mid"):
            row[source] = [float(values[source, metric]) for metric in ("first", "max", "min", "last")]
        rows.append(row)
    return rows


def load_candles(history, symbol, start, end, minutes, cache):
    if minutes < 1 or 1440 % minutes:
        raise ValueError("Audit interval must divide a UTC day into whole-minute candles")
    files = sorted(path for path in (history / symbol).glob("*/*/*.tsv")
                   if start <= date.fromisoformat(path.stem) <= end)
    if not files:
        raise ValueError("No tick files in the selected period")
    manifest = [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns] for path in files]
    contract = {"version": 1, "minutes": minutes, "symbol": symbol, "files": manifest}
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored["contract"] == contract:
            return stored["candles"]
    candles = []
    for index, path in enumerate(files):
        frame = pd.read_csv(path, sep="\t", header=None,
                            names=["frame", "symbol", "timestamp", "session", "sequence", "bid", "ask", "bid_volume", "ask_volume"],
                            usecols=[0, 1, 2, 5, 6, 7, 8], encoding="utf-8-sig",
                            dtype={"timestamp": "int64", "bid": "float64", "ask": "float64",
                                   "bid_volume": "float64", "ask_volume": "float64"})
        if not frame.frame.isin(("START", "STOP", "QUOTE")).all() or not frame.symbol.eq(symbol).all():
            raise ValueError(f"Unexpected frame or symbol in {path}")
        candles.extend(aggregate_quotes(frame.loc[frame.frame == "QUOTE"], minutes * 60_000))
        if index % 30 == 0 or index == len(files) - 1:
            print(f"Aggregated {index + 1}/{len(files)} files through {path.stem}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"contract": contract, "candles": candles}), encoding="utf-8")
    return candles


def timestamp(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def build_audit_engine(symbol, minutes, max_bars, confirmation_bars, lookback=15):
    previous_argv = sys.argv
    try:
        sys.argv = ["audit", "--strategy", "swing-regression", "--symbols", symbol,
                    "--candle-interval", f"{minutes}m", "--candle-lookback", "150", "--reward-risk-ratio", "2.3",
                    "--level-lookback-bars", "150", "--regression-lookback-bars", str(lookback), "--risk-percent", "1",
                    "--regression-entry-mode", "breakout-retest", "--no-regression-global-timeframe-validation",
                    "--no-regression-level-reaction", "--regression-breakout-confirmation-mode", "confirmed",
                    "--regression-breakout-max-bars", str(max_bars),
                    "--regression-breakout-confirmation-max-bars", str(confirmation_bars)]
        args = parse_args()
    finally:
        sys.argv = previous_argv
    return build_engine([], args)


def audit(candles, symbol, minutes, max_bars, confirmation_bars, lookback=15):
    engine = build_audit_engine(symbol, minutes, max_bars, confirmation_bars, lookback)
    market = engine.market_state
    context = StrategyContext(engine.portfolio, market)
    signals, risk_accepted = [], []
    for row in candles:
        start_time = timestamp(row["start_ms"])
        vbid, vask, cbid, cask = row["volumes"]
        for source in ("bid", "ask", "mid"):
            bar = Candle(start_time, *row[source], row["count"], vbid, vask, cbid, cask)
            market._candle_builders[source]._append_closed_candle(symbol, bar)
            if source == "mid":
                engine.risk_manager.candle_builder._append_closed_candle(symbol, bar)
        # All inputs are closed candles; the next active candle is deliberately absent.
        # Its single first tick has zero range and does not change risk volatility stats.
        last_tick = Tick(timestamp(row["last_ms"]), symbol, row["bid"][3], row["ask"][3], 0, 0)
        market.last_ticks[symbol] = last_tick
        for signal in engine.strategy.on_market_data(TickBatch(symbol, [last_tick]), context):
            record = {"timestamp": signal.timestamp.isoformat(), "side": signal.side.name,
                      "context": signal.entry_context}
            signals.append(record)
            evaluation = engine.risk_manager.evaluate(signal, engine.portfolio, last_tick)
            if evaluation is not None:
                risk_accepted.append({**record, "planned_entry": evaluation.entry_price,
                                      "stop_loss": evaluation.stop_loss, "take_profit": evaluation.take_profit})
    return {"max_bars": max_bars, "confirmation_max_bars": confirmation_bars,
            "regression_lookback_bars": lookback,
            "signal_count": len(signals), "isolated_risk_accepted": len(risk_accepted),
            "signals_by_month": dict(Counter(item["timestamp"][:7] for item in signals)),
            "risk_accepted_by_month": dict(Counter(item["timestamp"][:7] for item in risk_accepted)),
            "diagnostics": engine.strategy.diagnostics_summary(),
            "signals": signals, "risk_accepted_signals": risk_accepted,
            "effective_rules": engine.strategy.regression_rules}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("C:/jforex_history"))
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--regression-lookback-bars", type=int, default=15)
    parser.add_argument("--variants", nargs="+", default=["3:0", "3:3", "6:3"],
                        help="Timing variants as max-bars:confirmation-max-bars; not a PnL parameter search")
    parser.add_argument("--candle-cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts/regression_breakout_timing")
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("Start must precede end")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.candle_cache or args.output / f"{args.symbol.lower()}_{args.start_date}_{args.end_date}_{args.minutes}m.json"
    candles = load_candles(args.history, args.symbol, args.start_date, args.end_date, args.minutes, cache)
    results = []
    report = {"contract": "closed-candle-frequency-and-isolated-risk-v1", "symbol": args.symbol,
              "start": args.start_date.isoformat(), "end": args.end_date.isoformat(), "minutes": args.minutes,
              "candle_cache": str(cache.resolve()), "regression_lookback_bars": args.regression_lookback_bars,
              "note": "No fill or PnL simulation. Isolated risk checks assume a flat portfolio. No quality thresholds are tuned.",
              "results": results}
    for variant in args.variants:
        max_bars, confirmation_bars = map(int, variant.split(":"))
        result = audit(candles, args.symbol, args.minutes, max_bars, confirmation_bars, args.regression_lookback_bars)
        results.append(result)
        (args.output / "frequency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("max_bars", "confirmation_max_bars", "signal_count", "isolated_risk_accepted")}), flush=True)


if __name__ == "__main__":
    main()
