#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shlex
import sys
import time
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimateBot.run_backtest import build_engine, build_feed, parse_args as parse_backtest_args


DEFAULT_BASE_BACKTEST_ARGS = [
    "--strategy", "swing-regression",
    "--symbols", "USDCHF", "GBPUSD", "EURUSD",
    "--candle-lookback", "150",
    "--reward-risk-ratio", "3.0",
    "--level-lookback-bars", "20",
    "--risk-percent", "5.0",
    "--structure-stop-source", "structure_level",
]
DEFAULT_G_GRID = [5, 6, 8, 10, 12]
DEFAULT_GLOBAL_SLOPE_THRESHOLD_GRID = [0.0, 0.02, 0.05, 0.08, 0.12]
DEFAULT_GLOBAL_PARALLELISM_GRID = [0.0]
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date(2025, 12, 31)


@dataclass(frozen=True)
class Candidate:
    global_timeframe_multiplier: int
    global_reference_slope_threshold_atr: float
    global_min_parallelism: float

    @property
    def label(self) -> str:
        return (
            f"g={self.global_timeframe_multiplier}_"
            f"slope={self.global_reference_slope_threshold_atr:g}_"
            f"parallelism={self.global_min_parallelism:g}"
        )


@dataclass(frozen=True)
class SweepResult:
    candidate_label: str
    global_timeframe_multiplier: int
    global_reference_slope_threshold_atr: float
    global_min_parallelism: float
    start_date: date
    end_date: date
    runtime_seconds: float
    final_balance: float
    total_pnl: float
    trade_count: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    stop_loss_closes: int
    take_profit_closes: int
    other_closes: int
    analysis_windows: int
    global_analysis_windows: int
    low_volatility_rejections: int
    global_analysis_missing: int
    global_reference_slope_missing: int
    global_reference_slope_direction_mismatch: int
    global_reference_slope_too_weak: int
    global_parallelism_too_low: int
    global_trend_rejections: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep swing-regression global trend-filter settings over a long historical window, "
            "persist per-candidate results, and print a final ranking."
        ),
    )
    parser.add_argument(
        "--base-command",
        type=str,
        default=None,
        help=(
            "Exact command line used to launch ultimateBot.run_backtest. Everything up to and including "
            "`-m ultimateBot.run_backtest` is ignored, then this script forces its own date range and grid."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=parse_iso_date,
        default=DEFAULT_START_DATE,
        help="Inclusive sweep start date in YYYY-MM-DD format (default: 2020-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_iso_date,
        default=DEFAULT_END_DATE,
        help="Inclusive sweep end date in YYYY-MM-DD format (default: 2025-12-31).",
    )
    parser.add_argument(
        "--g-grid",
        type=int,
        nargs="+",
        default=DEFAULT_G_GRID,
        help="Candidate values for --regression-global-timeframe-multiplier. Defaults start at G=5.",
    )
    parser.add_argument(
        "--global-slope-threshold-grid",
        type=float,
        nargs="+",
        default=DEFAULT_GLOBAL_SLOPE_THRESHOLD_GRID,
        help=(
            "Candidate values for --regression-global-reference-slope-threshold-atr. "
            "Buys require a positive global anchored slope above the threshold; sells require the symmetric negative slope."
        ),
    )
    parser.add_argument(
        "--global-parallelism-grid",
        type=float,
        nargs="+",
        default=DEFAULT_GLOBAL_PARALLELISM_GRID,
        help=(
            "Candidate values for --regression-global-min-parallelism. "
            "Use 0 to ignore global parallelism and only follow the global reference slope."
        ),
    )
    parser.add_argument(
        "--global-lookback-grid",
        type=int,
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("ultimateBot") / "artifacts" / "global_confirmation_sweep",
        help="Directory where CSV, JSON, and resumable per-candidate results are written.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional stable subdirectory name under --artifacts-dir.",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=10,
        help="How many ranked candidates to print at the end.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum number of closed trades for a candidate to count as active in the ranking.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already computed candidates from the artifact directory.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Keep backtest warnings visible instead of silencing them during the sweep.",
    )
    parser.add_argument(
        "backtest_args",
        nargs=argparse.REMAINDER,
        help=(
            "Alternative to --base-command: pass run_backtest arguments after `--`. "
            "If omitted, the script uses the current default swing-regression command template."
        ),
    )
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.top_results <= 0:
        parser.error("--top-results must be positive")
    if args.min_trades < 0:
        parser.error("--min-trades must be non-negative")
    return args


def parse_iso_date(raw_value: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid date '{raw_value}': expected YYYY-MM-DD") from error


def parse_backtest_tokens(base_command: str | None, remainder: list[str]) -> list[str]:
    if base_command:
        tokens = shlex.split(base_command, posix=False)
    else:
        tokens = list(remainder)
        if tokens and tokens[0] == "--":
            tokens = tokens[1:]
    if not tokens:
        return list(DEFAULT_BASE_BACKTEST_ARGS)

    if "-m" in tokens:
        module_index = tokens.index("-m")
        if module_index + 1 < len(tokens) and tokens[module_index + 1] == "ultimateBot.run_backtest":
            return tokens[module_index + 2:]

    if "ultimateBot.run_backtest" in tokens:
        module_index = tokens.index("ultimateBot.run_backtest")
        return tokens[module_index + 1:]

    if tokens and tokens[0].lower().endswith("run_backtest.py"):
        return tokens[1:]

    return tokens


def build_runtime_args(tokens: list[str]) -> argparse.Namespace:
    with patch.object(sys, "argv", ["run_backtest", *tokens]):
        args = parse_backtest_args()
    if getattr(args, "strategy", None) != "swing-regression":
        raise ValueError("This sweep only supports `--strategy swing-regression`.")
    return args


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[int, float, float]] = set()
    for multiplier in args.g_grid:
        for slope_threshold in args.global_slope_threshold_grid:
            for parallelism in args.global_parallelism_grid:
                candidate = Candidate(
                    global_timeframe_multiplier=max(1, int(multiplier)),
                    global_reference_slope_threshold_atr=max(0.0, float(slope_threshold)),
                    global_min_parallelism=min(1.0, max(0.0, float(parallelism))),
                )
                key = (
                    candidate.global_timeframe_multiplier,
                    candidate.global_reference_slope_threshold_atr,
                    candidate.global_min_parallelism,
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item.global_timeframe_multiplier,
            item.global_reference_slope_threshold_atr,
            item.global_min_parallelism,
        )
    )
    return candidates


def clone_runtime_args(
    base_args: argparse.Namespace,
    *,
    start_date: date,
    end_date: date,
    candidate: Candidate,
) -> argparse.Namespace:
    values = dict(vars(base_args))
    values["start_date"] = start_date
    values["end_date"] = end_date
    values["regression_global_timeframe_multiplier"] = candidate.global_timeframe_multiplier
    values["regression_global_reference_slope_threshold_atr"] = candidate.global_reference_slope_threshold_atr
    values["regression_global_min_parallelism"] = candidate.global_min_parallelism
    return argparse.Namespace(**values)


def run_backtest_candidate(
    runtime_args: argparse.Namespace,
    candidate: Candidate,
    *,
    show_warnings: bool,
) -> SweepResult:
    candidate_args = clone_runtime_args(
        runtime_args,
        start_date=runtime_args.start_date,
        end_date=runtime_args.end_date,
        candidate=candidate,
    )
    feed = build_feed(
        candidate_args.history,
        candidate_args.symbols,
        candidate_args.start_date,
        candidate_args.end_date,
    )
    engine = build_engine(feed=feed, args=candidate_args)

    started_at = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        if show_warnings:
            engine.run()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engine.run()
    runtime_seconds = time.perf_counter() - started_at

    trades = list(engine.reports.trades)
    pnls = [float(trade.pnl) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl <= 0.0]
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0.0 else float("inf") if gross_wins > 0.0 else 0.0
    close_counts: dict[str, int] = {}
    for trade in trades:
        reason_name = str(trade.close_reason.name)
        close_counts[reason_name] = close_counts.get(reason_name, 0) + 1

    diagnostics = engine.strategy.diagnostics_summary()
    reason_counts = diagnostics.get("reason_counts", {})
    final_balance = float(engine.portfolio.balance)
    total_pnl = final_balance - float(engine.portfolio.initial_balance)
    trade_count = len(trades)

    return SweepResult(
        candidate_label=candidate.label,
        global_timeframe_multiplier=candidate.global_timeframe_multiplier,
        global_reference_slope_threshold_atr=candidate.global_reference_slope_threshold_atr,
        global_min_parallelism=candidate.global_min_parallelism,
        start_date=candidate_args.start_date,
        end_date=candidate_args.end_date,
        runtime_seconds=runtime_seconds,
        final_balance=final_balance,
        total_pnl=total_pnl,
        trade_count=trade_count,
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / trade_count) if trade_count else 0.0,
        profit_factor=profit_factor,
        avg_pnl=(total_pnl / trade_count) if trade_count else 0.0,
        avg_win=(gross_wins / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        stop_loss_closes=int(close_counts.get("STOP_LOSS", 0)),
        take_profit_closes=int(close_counts.get("TAKE_PROFIT", 0)),
        other_closes=trade_count - int(close_counts.get("STOP_LOSS", 0)) - int(close_counts.get("TAKE_PROFIT", 0)),
        analysis_windows=int(diagnostics.get("analysis_windows", 0)),
        global_analysis_windows=int(diagnostics.get("global_analysis_windows", 0)),
        low_volatility_rejections=int(reason_counts.get("market_volatility_too_low", 0)),
        global_analysis_missing=int(reason_counts.get("global_analysis_missing", 0)),
        global_reference_slope_missing=int(reason_counts.get("global_reference_slope_missing", 0)),
        global_reference_slope_direction_mismatch=int(reason_counts.get("global_reference_slope_direction_mismatch", 0)),
        global_reference_slope_too_weak=int(reason_counts.get("global_reference_slope_too_weak", 0)),
        global_parallelism_too_low=int(reason_counts.get("global_parallelism_too_low", 0)),
        global_trend_rejections=(
            int(reason_counts.get("global_analysis_missing", 0))
            + int(reason_counts.get("global_reference_slope_missing", 0))
            + int(reason_counts.get("global_reference_slope_direction_mismatch", 0))
            + int(reason_counts.get("global_reference_slope_too_weak", 0))
            + int(reason_counts.get("global_parallelism_too_low", 0))
        ),
    )


def build_default_label(runtime_args: argparse.Namespace, start_date: date, end_date: date) -> str:
    symbols = [str(symbol).lower() for symbol in getattr(runtime_args, "symbols", [])]
    symbol_part = "-".join(symbols) if symbols else "no-symbols"
    return f"{start_date.isoformat()}_{end_date.isoformat()}_{symbol_part}_global-trend-sweep"


def result_to_payload(result: SweepResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["start_date"] = result.start_date.isoformat()
    payload["end_date"] = result.end_date.isoformat()
    return payload


def result_from_payload(payload: dict[str, Any]) -> SweepResult:
    values = dict(payload)
    values["start_date"] = date.fromisoformat(values["start_date"])
    values["end_date"] = date.fromisoformat(values["end_date"])
    return SweepResult(**values)


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    return value


def load_existing_results(results_path: Path) -> dict[str, SweepResult]:
    if not results_path.exists():
        return {}
    loaded: dict[str, SweepResult] = {}
    for raw_line in results_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        result = result_from_payload(payload)
        loaded[result.candidate_label] = result
    return loaded


def append_result(results_path: Path, result: SweepResult) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_payload(result_to_payload(result)), allow_nan=False) + "\n")


def rank_results(results: list[SweepResult], min_trades: int) -> list[SweepResult]:
    infinity_score = 1e12
    return sorted(
        results,
        key=lambda item: (
            0 if item.trade_count >= min_trades else 1,
            -item.total_pnl,
            -(item.profit_factor if math.isfinite(item.profit_factor) else infinity_score),
            item.stop_loss_closes,
            -item.trade_count,
            item.global_timeframe_multiplier,
            item.global_reference_slope_threshold_atr,
            item.global_min_parallelism,
        ),
    )


def write_csv(csv_path: Path, ranked_results: list[SweepResult]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result_to_payload(ranked_results[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in ranked_results:
            writer.writerow(result_to_payload(result))


def write_summary(
    summary_path: Path,
    *,
    script_args: argparse.Namespace,
    template_backtest_args: list[str],
    runtime_args: argparse.Namespace,
    candidates: list[Candidate],
    ranked_results: list[SweepResult],
) -> None:
    active_results = [item for item in ranked_results if item.trade_count >= script_args.min_trades]
    best_by_pnl = active_results[0] if active_results else None
    best_by_profit_factor = None
    if active_results:
        best_by_profit_factor = min(
            active_results,
            key=lambda item: (
                -(item.profit_factor if math.isfinite(item.profit_factor) else 1e12),
                -item.total_pnl,
                item.stop_loss_closes,
                -item.trade_count,
            ),
        )
    fewest_stop_losses = None
    if active_results:
        fewest_stop_losses = min(
            active_results,
            key=lambda item: (
                item.stop_loss_closes,
                -item.total_pnl,
                -(item.profit_factor if math.isfinite(item.profit_factor) else 1e12),
                -item.trade_count,
            ),
        )

    payload = {
        "window": {
            "start_date": script_args.start_date.isoformat(),
            "end_date": script_args.end_date.isoformat(),
        },
        "template_backtest_args": template_backtest_args,
        "symbols": list(getattr(runtime_args, "symbols", [])),
        "grid": {
            "g_grid": list(script_args.g_grid),
            "global_slope_threshold_grid": list(script_args.global_slope_threshold_grid),
            "global_parallelism_grid": list(script_args.global_parallelism_grid),
            "candidate_count": len(candidates),
        },
        "ranking": {
            "min_trades": script_args.min_trades,
            "best_by_pnl": None if best_by_pnl is None else result_to_payload(best_by_pnl),
            "best_by_profit_factor": None if best_by_profit_factor is None else result_to_payload(best_by_profit_factor),
            "fewest_stop_losses": None if fewest_stop_losses is None else result_to_payload(fewest_stop_losses),
            "top_results": [result_to_payload(item) for item in ranked_results[: script_args.top_results]],
        },
        "all_results": [result_to_payload(item) for item in ranked_results],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(sanitize_payload(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def print_progress(index: int, total: int, result: SweepResult) -> None:
    print(
        f"[{index}/{total}] G={result.global_timeframe_multiplier} "
        f"slope_thr={result.global_reference_slope_threshold_atr:g} "
        f"gpar={result.global_min_parallelism:g} pnl={result.total_pnl:.2f} "
        f"trades={result.trade_count} pf={result.profit_factor:.2f} "
        f"sl={result.stop_loss_closes} rejects={result.global_trend_rejections} "
        f"runtime={result.runtime_seconds:.1f}s"
    )


def print_ranked_summary(ranked_results: list[SweepResult], top_results: int, min_trades: int) -> None:
    print("\nTop candidates")
    for index, result in enumerate(ranked_results[:top_results], start=1):
        active_marker = "active" if result.trade_count >= min_trades else "filtered"
        print(
            f"{index:>2}. G={result.global_timeframe_multiplier:<2} "
            f"slope_thr={result.global_reference_slope_threshold_atr:<4g} "
            f"gpar={result.global_min_parallelism:<4g} "
            f"pnl={result.total_pnl:>8.2f} trades={result.trade_count:>5} "
            f"win_rate={result.win_rate:>7.2%} pf={result.profit_factor:>6.2f} "
            f"sl={result.stop_loss_closes:>4} rejects={result.global_trend_rejections:>6} "
            f"[{active_marker}]"
        )


def main() -> None:
    args = parse_args()
    template_backtest_args = parse_backtest_tokens(args.base_command, list(args.backtest_args))
    runtime_args = build_runtime_args(template_backtest_args)
    runtime_args.start_date = args.start_date
    runtime_args.end_date = args.end_date
    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("No candidate generated from the requested grids.")

    label = args.label or build_default_label(runtime_args, args.start_date, args.end_date)
    artifact_dir = args.artifacts_dir / label
    results_path = artifact_dir / "candidate_results.jsonl"
    csv_path = artifact_dir / "results.csv"
    summary_path = artifact_dir / "summary.json"

    existing = load_existing_results(results_path) if args.resume else {}
    collected: dict[str, SweepResult] = dict(existing)

    print("Starting global-trend sweep")
    print(f"Window: {args.start_date.isoformat()} -> {args.end_date.isoformat()}")
    print(f"Symbols: {', '.join(getattr(runtime_args, 'symbols', []))}")
    print(
        f"Grid: G={list(args.g_grid)} "
        f"slope={list(args.global_slope_threshold_grid)} "
        f"parallelism={list(args.global_parallelism_grid)}"
    )
    print(f"Candidates: {len(candidates)}")
    print(f"Artifacts: {artifact_dir}")
    if args.resume and existing:
        print(f"Resume: reusing {len(existing)} completed candidate(s)")

    for index, candidate in enumerate(candidates, start=1):
        cached = collected.get(candidate.label)
        if cached is not None:
            print_progress(index, len(candidates), cached)
            continue
        result = run_backtest_candidate(runtime_args, candidate, show_warnings=args.show_warnings)
        collected[candidate.label] = result
        append_result(results_path, result)
        ranked_results = rank_results(list(collected.values()), args.min_trades)
        write_csv(csv_path, ranked_results)
        write_summary(
            summary_path,
            script_args=args,
            template_backtest_args=template_backtest_args,
            runtime_args=runtime_args,
            candidates=candidates,
            ranked_results=ranked_results,
        )
        print_progress(index, len(candidates), result)

    ranked_results = rank_results(list(collected.values()), args.min_trades)
    write_csv(csv_path, ranked_results)
    write_summary(
        summary_path,
        script_args=args,
        template_backtest_args=template_backtest_args,
        runtime_args=runtime_args,
        candidates=candidates,
        ranked_results=ranked_results,
    )
    print_ranked_summary(ranked_results, args.top_results, args.min_trades)
    if ranked_results:
        best = next((item for item in ranked_results if item.trade_count >= args.min_trades), ranked_results[0])
        print("\nSuggested overrides")
        print(
            "--regression-global-timeframe-multiplier "
            f"{best.global_timeframe_multiplier} "
            "--regression-global-reference-slope-threshold-atr "
            f"{best.global_reference_slope_threshold_atr:g} "
            "--regression-global-min-parallelism "
            f"{best.global_min_parallelism:g}"
        )
    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()