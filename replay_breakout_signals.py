"""Replay an audited fixed strategy's signals on original ticks with the existing engine.

Skips ticks only while there are no positions, pending orders or signal decisions.
At each signal, restores the risk candle history observed at that decision.
This is restricted to the fixed, single-symbol configuration of the audit.
"""
import argparse
from contextlib import redirect_stdout
from datetime import date, datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ultimateBot.audit_breakout_frequency import build_audit_engine, timestamp
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.compare_regression_breakout import summarize


def seed_risk(engine, candles, bar_index, symbol, current_tick):
    builder = engine.risk_manager.candle_builder
    builder._active.pop(symbol, None)
    builder._closed[symbol].clear()
    builder._swing_histories.pop(symbol, None)
    builder._candle_counts[symbol] = 0
    terminal_flush = current_tick.timestamp == timestamp(candles[bar_index]["last_ms"])
    closed_end = bar_index if terminal_flush else bar_index + 1
    for row in candles[max(0, closed_end - builder.lookback_candles):closed_end]:
        builder._append_closed_candle(symbol, Candle(timestamp(row["start_ms"]), *row["mid"], row["count"], *row["volumes"]))
    # The engine has already observed the first tick that closed the signal candle.
    # Recreate this active candle after restoring the preceding closed history.
    if terminal_flush:
        # MarketState is finalized at end of feed, while the risk builder still
        # holds this final full candle as active (in addition to its closed history).
        row = candles[bar_index]
        builder._active[symbol] = Candle(timestamp(row["start_ms"]), *row["mid"], row["count"], *row["volumes"])
    else:
        builder.update(current_tick)


def replay(history, symbol, start, end, minutes, candles, result, log_path):
    engine = build_audit_engine(symbol, minutes, result["max_bars"], result["confirmation_max_bars"],
                                result.get("regression_lookback_bars", 15))
    engine.batcher.keep_last_tick_only = True
    by_last_ms = {row["last_ms"]: index for index, row in enumerate(candles)}
    scheduled = []
    for record in result["signals"]:
        index = by_last_ms[round(datetime.fromisoformat(record["timestamp"]).timestamp() * 1000)]
        decision_ms = candles[index + 1]["first_ms"] if index + 1 < len(candles) else candles[index]["last_ms"]
        scheduled.append((decision_ms, index, record))
    scheduled.sort(key=lambda item: item[0])
    next_signal = 0
    initial_risks = {}
    open_position = engine.position_manager.open_position

    def record_open(*args, **kwargs):
        position = open_position(*args, **kwargs)
        initial_risks[position.trade_id] = abs(position.entry_price - position.stop_loss) * position.quantity * position.profit_multiplier
        return position

    engine.position_manager.open_position = record_open

    def busy():
        return bool(engine.portfolio.positions) or engine.pending_orders_processor.queue.count_for_symbol(symbol) > 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
        files = sorted(path for path in (history / symbol).glob("*/*/*.tsv")
                       if start <= date.fromisoformat(path.stem) <= end)
        for path in files:
            day = date.fromisoformat(path.stem)
            if not busy():
                if next_signal == len(scheduled):
                    break
                if day < timestamp(scheduled[next_signal][0]).date():
                    continue
            frame = pd.read_csv(path, sep="\t", header=None,
                                names=["frame", "symbol", "timestamp", "session", "sequence", "bid", "ask", "bid_volume", "ask_volume"],
                                usecols=[0, 1, 2, 5, 6, 7, 8], encoding="utf-8-sig")
            frame = frame.loc[(frame.frame == "QUOTE") & (frame.symbol == symbol) & (frame.bid > 0) & (frame.ask > 0)]
            times = frame.timestamp.to_numpy(dtype="int64")
            quotes = frame[["bid", "ask", "bid_volume", "ask_volume"]].to_numpy(dtype="float64")
            index = 0
            while index < len(times):
                if not busy():
                    if next_signal == len(scheduled) or timestamp(scheduled[next_signal][0]).date() > day:
                        break
                    index = max(index, int(np.searchsorted(times, scheduled[next_signal][0])))
                    if index >= len(times):
                        raise ValueError("An audited signal decision tick is missing from the replay")
                ms = int(times[index])
                tick = Tick(timestamp(ms), symbol, *(float(value) for value in quotes[index]))
                engine.on_tick(tick, update_market=False, allow_entries=False)
                if next_signal < len(scheduled) and ms >= scheduled[next_signal][0]:
                    decision_ms, bar_index, record = scheduled[next_signal]
                    if ms != decision_ms:
                        raise ValueError("Signal decision timing differs from the original quote history")
                    seed_risk(engine, candles, bar_index, symbol, tick)
                    context = record["context"]
                    signal = Signal(symbol, datetime.fromisoformat(record["timestamp"]), SignalSide[record["side"]],
                                    stop_loss_price=context["stop_loss"], entry_context=context,
                                    minimum_entry_price=context["minimum_entry_price"],
                                    maximum_entry_price=context["maximum_entry_price"])
                    row = candles[bar_index]
                    closing_tick = Tick(timestamp(row["last_ms"]), symbol, row["bid"][3], row["ask"][3], 0, 0)
                    # Monitor/fill phases have already run for this tick, as in BacktestEngine.on_tick.
                    # The newly queued order can first execute on a subsequent tick.
                    engine._process_signal(signal, closing_tick)
                    next_signal += 1
                index += 1
        if next_signal != len(scheduled):
            raise ValueError("Not all audited signals were replayed")
    summary = summarize(engine, initial_risks)
    summary["diagnostics"] = result["diagnostics"]
    summary["max_bars"] = result["max_bars"]
    summary["confirmation_max_bars"] = result["confirmation_max_bars"]
    summary["replayed_signals"] = next_signal
    summary["regression_lookback_bars"] = result.get("regression_lookback_bars", 15)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("C:/jforex_history"))
    parser.add_argument("--audit", type=Path, default=Path(__file__).parent / "artifacts/regression_breakout_timing/frequency.json")
    args = parser.parse_args()
    audit_report = json.loads(args.audit.read_text(encoding="utf-8"))
    symbol, minutes = audit_report["symbol"], audit_report["minutes"]
    start, end = date.fromisoformat(audit_report["start"]), date.fromisoformat(audit_report["end"])
    cache_path = Path(audit_report.get("candle_cache", args.audit.parent / f"{symbol.lower()}_{start}_{end}_{minutes}m.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    for path, size, modified in cached["contract"]["files"]:
        if not Path(path).resolve().is_relative_to((args.history / symbol).resolve()):
            raise ValueError("Replay history root differs from the audited source")
        stat = Path(path).stat()
        if stat.st_size != size or stat.st_mtime_ns != modified:
            raise ValueError("History changed since the audit; rebuild the candle cache")
    report = {"contract": "fixed-closed-candle-signal-replay-original-ticks-v1",
              "symbol": symbol, "start": start.isoformat(), "end": end.isoformat(),
              "note": "Fixed strategy only; original risk, queue, fill, portfolio and closer. Idle quote ticks are skipped; risk history is restored at decisions.",
              "results": []}
    for result in audit_report["results"]:
        stem = f"replay_{result['max_bars']}_{result['confirmation_max_bars']}"
        summary = replay(args.history, symbol, start, end, minutes, cached["candles"], result, args.audit.parent / f"{stem}.log")
        report["results"].append(summary)
        (args.audit.parent / "replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key not in ("diagnostics", "trades")}), flush=True)


if __name__ == "__main__":
    main()
