"""Reproducible local replay, reload and overhead measurement in fresh processes."""
import argparse
import ctypes
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter, process_time
from uuid import uuid4

from ultimateBot.backtester.data.file_tick_source import JForexTickSource
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.optimization.bandit_core import canonical_json
from ultimateBot.run_backtest import build_engine, parse_args


def peak_process_bytes():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                       "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not measure(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize


def data(args):
    if args.synthetic:
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        return [Tick(start + timedelta(minutes=index, seconds=seconds), "TEST", price, price + .01, 1., 2.)
                for index in range(args.sample_minutes)
                for seconds, price in zip((0, 10, 20, 50),
                    (101. + .08 * index, 104. + .2 * index, 100. + .001 * index,
                     101. + .08 * index + (.4 if index % 2 == 0 else -.4)))]
    ticks, first = [], None
    for tick in JForexTickSource(args.history, args.symbol, args.date, args.date):
        first = first or tick.timestamp
        if tick.timestamp >= first + timedelta(minutes=args.sample_minutes):
            break
        ticks.append(tick)
    if not ticks:
        raise RuntimeError("No local ticks in the requested historical sample")
    return ticks


def worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    ticks = data(args)
    symbol = ticks[0].symbol
    arguments = ["run_backtest", "--symbols", symbol, "--strategy", "swing-regression", "--candle-interval", "1m",
                 "--regression-lookback-bars", "10", "--level-lookback-bars", "30",
                 "--regression-global-timeframe-multiplier", "1", "--regression-min-market-volatility-ratio", "0",
                 "--regression-score-threshold", "0", "--regression-min-parallelism", "0",
                 "--regression-min-consecutive-swing-distance-atr", "0", "--no-regression-level-reaction",
                 "--reward-risk-ratio", ".5", "--trailing-start-rr-ratio", ".1"]
    if args.case != "disabled":
        arguments += ["--contextual-bandit", "--bandit-memory", str(args.memory), "--bandit-context-bars", "10",
                      "--bandit-warmup", "2", "--bandit-shadow-limit", "6", "--bandit-group-max-bars", "30",
                      "--adaptive-lookback-min", "5", "--adaptive-lookback-max", "20"]
    if args.case == "max_lookback":
        arguments += ["--regression-lookback-bars", "19", "--bandit-lookback-grid", "5,10,19"]
    previous = sys.argv
    try:
        sys.argv = arguments
        launch = parse_args()
    finally:
        sys.argv = previous
    with (args.output / (args.case + ".log")).open("w", encoding="utf-8") as log, redirect_stdout(log):
        engine = build_engine(ticks, launch)
        if engine.contextual_optimizer:
            engine.contextual_optimizer.provenance.update(
                start_date=ticks[0].timestamp.isoformat(), end_date=ticks[-1].timestamp.isoformat(),
                data_source="synthetic fixture" if args.synthetic else str(args.history), validation_case=args.case)
        cpu, wall = process_time(), perf_counter()
        engine.run()
        wall, cpu = perf_counter() - wall, process_time() - cpu
        report = engine.contextual_optimizer.summary() if engine.contextual_optimizer else {
            "main_closed_trades": len(engine.reports.trades),
            "main_pnl_net": engine.portfolio.balance - engine.portfolio.initial_balance}
        report.update(validation={"case": args.case, "source": "synthetic" if args.synthetic else "local JForex history",
                                  "symbol": symbol, "ticks": len(ticks), "start": ticks[0].timestamp.isoformat(),
                                  "end": ticks[-1].timestamp.isoformat(), "wall_seconds": wall, "cpu_seconds": cpu,
                                  "peak_process_bytes": peak_process_bytes(), "arguments": arguments[1:],
                                  "data_fingerprint": sha256(canonical_json([[tick.timestamp, tick.bid, tick.ask,
                                      tick.bid_volume, tick.ask_volume] for tick in ticks]).encode()).hexdigest()})
    path = args.output / (args.case + ".json")
    path.write_text(json.dumps(json.loads(canonical_json(report)), indent=2, ensure_ascii=False), encoding="utf-8")
    print(canonical_json({"report": str(path), "trades": report["main_closed_trades"], "wall_seconds": wall}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("D:/jforex_history"))
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--date", type=date.fromisoformat, default=date(2020, 1, 2))
    parser.add_argument("--sample-minutes", type=int, default=120)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "contextual_bandit_validation")
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--case", choices=("disabled", "initial", "reload", "max_lookback"))
    args = parser.parse_args()
    if args.sample_minutes < 25:
        parser.error("Use at least 25 minutes to observe the maximum default lookback")
    if args.case:
        worker(args)
        return
    args.memory = args.memory or args.output / ("learning-" + uuid4().hex + ".sqlite3")
    commands = [sys.executable, "-m", "ultimateBot.backtester.optimization.validate_contextual_bandit",
                "--history", str(args.history), "--symbol", args.symbol, "--date", args.date.isoformat(),
                "--sample-minutes", str(args.sample_minutes), "--output", str(args.output)]
    if args.synthetic:
        commands.append("--synthetic")
    for case in ("disabled", "initial", "reload", "max_lookback"):
        memory = args.memory if case != "max_lookback" else args.memory.with_name("max-" + args.memory.name)
        subprocess.run([*commands, "--memory", str(memory), "--case", case], check=True)
    reports = {case: json.loads((args.output / (case + ".json")).read_text(encoding="utf-8"))
               for case in ("disabled", "initial", "reload", "max_lookback")}
    baseline = reports["disabled"]["validation"]
    metrics = {case: {"cpu_ratio_to_disabled": report["validation"]["cpu_seconds"] / max(baseline["cpu_seconds"], 1e-9),
                      "wall_ratio_to_disabled": report["validation"]["wall_seconds"] / max(baseline["wall_seconds"], 1e-9),
                      "extra_peak_process_bytes": report["validation"]["peak_process_bytes"] - baseline["peak_process_bytes"]}
               for case, report in reports.items()}
    (args.output / "overhead.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
