#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pandas as pd
import shlex
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from ultimateBot.backtester.engine.backtest_engine import BacktestEngine
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.reports.trade_report import TradeReport
from ultimateBot.backtester.risk.risk_manager import RiskEvaluation
from ultimateBot.backtester.strategy.signal import Signal
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import _rolling_atr
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import build_engine, build_feed, parse_args as parse_backtest_args


DEFAULT_THRESHOLD_GRID = [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]
DEFAULT_WINDOW_GRID = [20, 30, 40, 60]


@dataclass(frozen=True)
class Candidate:
    market_volatility_window_bars: int
    min_market_volatility_ratio: float

    @property
    def label(self) -> str:
        return (
            f"window={self.market_volatility_window_bars}_"
            f"threshold={self.min_market_volatility_ratio:.6f}"
        )


@dataclass(frozen=True)
class WindowResult:
    candidate_label: str
    candidate: dict[str, Any]
    start_date: date
    end_date: date
    runtime_seconds: float
    total_pnl: float
    trade_count: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    stop_loss_closes: int
    take_profit_closes: int
    other_closes: int
    analysis_windows: int
    low_volatility_rejections: int
    avg_stop_loss_entry_volatility_ratio: float | None
    avg_stop_loss_entry_volatility_atr: float | None
    avg_stop_loss_close_volatility_ratio: float | None
    avg_stop_loss_close_volatility_atr: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze stop-loss-heavy periods for swing-regression and optimize the market-volatility filter "
            "with resumable multi-day sweeps."
        ),
    )
    parser.add_argument(
        "--base-command",
        type=str,
        default=None,
        help=(
            "Exact command line used to launch ultimateBot.run_backtest. Everything up to and including "
            "`-m ultimateBot.run_backtest` is ignored, the remaining args are replayed."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("ultimateBot") / "artifacts" / "stoploss_volatility_optimizer",
        help="Directory where the baseline audit, period summaries, and optimization results are written.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional stable subdirectory name under --artifacts-dir. Defaults to a name derived from the date range and symbols.",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=7,
        help="Bucket stop-loss closes into N-day periods to find the worst clusters.",
    )
    parser.add_argument(
        "--top-periods",
        type=int,
        default=10,
        help="Number of stop-loss-heavy periods to print and export.",
    )
    parser.add_argument(
        "--evaluation-window-days",
        type=int,
        default=30,
        help="Evaluate each volatility candidate on rolling contiguous windows of this many days. Use 0 to score only the full date range.",
    )
    parser.add_argument(
        "--max-evaluation-windows",
        type=int,
        default=0,
        help="Optional cap on the number of evaluation windows. 0 means no cap.",
    )
    parser.add_argument(
        "--trade-retention-floor",
        type=float,
        default=0.60,
        help="Reject candidate windows that keep less than this fraction of the baseline trade count.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLD_GRID,
        help="Candidate values for regression_min_market_volatility_ratio.",
    )
    parser.add_argument(
        "--window-grid",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOW_GRID,
        help="Candidate values for regression_market_volatility_window_bars.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=10,
        help="How many ranked static candidates to print at the end.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse previously written baseline and per-window results from the artifact directory.",
    )
    parser.add_argument(
        "backtest_args",
        nargs=argparse.REMAINDER,
        help="Alternative to --base-command: pass the run_backtest arguments after `--`.",
    )
    args = parser.parse_args()
    if args.period_days <= 0:
        parser.error("--period-days must be positive")
    if args.evaluation_window_days < 0:
        parser.error("--evaluation-window-days must be non-negative")
    if not 0.0 <= args.trade_retention_floor <= 1.0:
        parser.error("--trade-retention-floor must be between 0 and 1")
    return args


def parse_backtest_tokens(base_command: str | None, remainder: list[str]) -> list[str]:
    if base_command:
        tokens = shlex.split(base_command, posix=False)
    else:
        tokens = list(remainder)
        if tokens and tokens[0] == "--":
            tokens = tokens[1:]
    if not tokens:
        raise ValueError("Provide either --base-command or run_backtest args after `--`.")

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
        raise ValueError("This optimizer only supports `--strategy swing-regression`.")
    return args


def clone_runtime_args(
    base_args: argparse.Namespace,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    candidate: Candidate | None = None,
) -> argparse.Namespace:
    values = dict(vars(base_args))
    if start_date is not None:
        values["start_date"] = start_date
    if end_date is not None:
        values["end_date"] = end_date
    if candidate is not None:
        values["regression_market_volatility_window_bars"] = candidate.market_volatility_window_bars
        values["regression_min_market_volatility_ratio"] = candidate.min_market_volatility_ratio
    return argparse.Namespace(**values)


def build_candidates(base_args: argparse.Namespace, args: argparse.Namespace) -> list[Candidate]:
    base_candidate = Candidate(
        market_volatility_window_bars=int(getattr(base_args, "regression_market_volatility_window_bars", 30)),
        min_market_volatility_ratio=float(getattr(base_args, "regression_min_market_volatility_ratio", 0.0)),
    )
    candidates: list[Candidate] = [base_candidate]
    seen: set[tuple[int, float]] = {
        (base_candidate.market_volatility_window_bars, round(base_candidate.min_market_volatility_ratio, 12))
    }
    for threshold in args.threshold_grid:
        if threshold <= 0.0:
            continue
        for window_bars in args.window_grid:
            candidate = Candidate(
                market_volatility_window_bars=max(2, int(window_bars)),
                min_market_volatility_ratio=max(0.0, float(threshold)),
            )
            key = (candidate.market_volatility_window_bars, round(candidate.min_market_volatility_ratio, 12))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def split_date_range(
    start_date: date,
    end_date: date,
    window_days: int,
    max_windows: int,
) -> list[tuple[date, date]]:
    if window_days <= 0:
        return [(start_date, end_date)]

    windows: list[tuple[date, date]] = []
    current_start = start_date
    while current_start <= end_date:
        current_end = min(end_date, current_start + timedelta(days=window_days - 1))
        windows.append((current_start, current_end))
        if max_windows > 0 and len(windows) >= max_windows:
            break
        current_start = current_end + timedelta(days=1)
    return windows


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def default_json(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name"):
        return str(getattr(value, "name")).lower()
    raise TypeError(type(value).__name__)


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value]
    return value


def build_candle_frame(closed_candles: list[Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": candle.start_time,
            "open": candle.open_price,
            "high": candle.high_price,
            "low": candle.low_price,
            "close": candle.close_price,
        }
        for candle in closed_candles
    )


class VolatilityAuditEngine(BacktestEngine):
    def __init__(self, template: BacktestEngine):
        super().__init__(**{
            key: getattr(template, key)
            for key in (
                "feed",
                "batcher",
                "strategy",
                "risk_manager",
                "order_builder",
                "execution_model",
                "portfolio",
                "position_manager",
                "trade_monitor",
                "trade_closer",
                "reports",
                "market_state",
                "adaptive_optimizer",
            )
        })
        self._analysis_snapshot_by_symbol: dict[str, dict[str, Any]] = {}
        self.order_snapshots: dict[int, dict[str, Any]] = {}
        self.trade_to_order: dict[int, int] = {}
        self.trade_records: list[dict[str, Any]] = []

    def _capture_analysis_snapshot(self, symbol: str) -> dict[str, Any]:
        strategy = self.strategy
        if not isinstance(strategy, SwingRegressionStrategy):
            return {}
        analysis = strategy.last_analysis.get(symbol)
        if not analysis:
            return {}
        return {
            "symbol": symbol,
            "analysis_closed_at": analysis.get("closed_at"),
            "analysis_reason": next(iter(analysis.get("reasons", [])), None),
            "analysis_direction": analysis.get("direction"),
            "analysis_current_price": analysis.get("current_price"),
            "analysis_atr": analysis.get("atr"),
            "market_volatility_atr_entry": analysis.get("market_volatility_atr"),
            "market_volatility_ratio_entry": analysis.get("market_volatility_ratio"),
            "market_volatility_window_bars": analysis.get("market_volatility_window_bars"),
            "analysis_swing_density": analysis.get("swing_density"),
            "analysis_trend_score": analysis.get("trend_score"),
            "analysis_parallelism": analysis.get("parallelism"),
        }

    def _current_close_volatility_snapshot(self, symbol: str) -> dict[str, Any]:
        strategy = self.strategy
        if not isinstance(strategy, SwingRegressionStrategy):
            return {}
        window_bars = int(strategy.regression_rules["regression_market_volatility_window_bars"])
        candles = self.market_state.get_candles(symbol, include_active=False, price_source="mid")
        effective_window = min(len(candles), max(2, window_bars))
        if effective_window < 2:
            return {}
        sample = candles[-effective_window:]
        frame = build_candle_frame(sample)
        point_size = max(self.market_state.get_point_size(symbol), 1e-9)
        market_volatility_atr = _rolling_atr(frame, point_size, effective_window)
        current_price = float(frame["close"].iloc[-1])
        return {
            "market_volatility_atr_close": market_volatility_atr,
            "market_volatility_ratio_close": market_volatility_atr / max(abs(current_price), point_size, 1e-9),
            "market_volatility_close_reference_time": sample[-1].start_time,
        }

    def _run_strategy(self, batch) -> None:
        context = StrategyContext(portfolio=self.portfolio, market_state=self.market_state)
        signals = self.strategy.on_market_data(batch, context)
        self._analysis_snapshot_by_symbol[batch.symbol] = self._capture_analysis_snapshot(batch.symbol)
        for signal in signals:
            self._process_signal(signal, batch.ticks[-1])

    def _process_signal(self, signal: Signal, tick) -> None:
        evaluation = self.risk_manager.evaluate(
            signal=signal,
            portfolio=self.portfolio,
            tick=tick,
            pending_orders_for_symbol=self.pending_orders_processor.queue.count_for_symbol(signal.symbol),
        )
        if evaluation is None:
            return

        order = self.order_builder.build(signal, evaluation)
        if self.adaptive_optimizer is not None:
            self.adaptive_optimizer.prepare_order(order)
        self.trade_closer.prepare_order(order)
        snapshot = dict(self._analysis_snapshot_by_symbol.get(signal.symbol, {}))
        snapshot.update(self._serialize_evaluation(signal, evaluation))
        self.order_snapshots[order.order_id] = snapshot
        self.pending_orders_processor.queue.add(order)

    def _process_pending_orders(self, tick) -> None:
        execution_results = self.pending_orders_processor.on_tick(tick)
        for result in execution_results:
            position = self.position_manager.open_position(portfolio=self.portfolio, fill=result.fill, order=result.order)
            self.trade_closer.prepare_position(position)
            self.trade_to_order[position.trade_id] = result.order.order_id

    def _monitor_positions(self, tick) -> None:
        positions = list(self.portfolio.positions.values())
        for position in positions:
            request = self.trade_closer.check(position, tick, self.trade_monitor)
            if request is None:
                continue

            close_snapshot = self._current_close_volatility_snapshot(position.symbol)
            report = self.trade_closer.close(self.portfolio, request)
            self.reports.add_trade(report)
            self.trade_records.append(self._build_trade_record(report, close_snapshot))

    def _serialize_evaluation(self, signal: Signal, evaluation: RiskEvaluation) -> dict[str, Any]:
        return {
            "signal_timestamp": signal.timestamp,
            "signal_side": signal.side.name.lower(),
            "planned_entry_price": evaluation.entry_price,
            "planned_stop_loss": evaluation.stop_loss,
            "planned_take_profit": evaluation.take_profit,
            "planned_stop_distance": evaluation.stop_distance,
            "planned_take_profit_distance": evaluation.take_profit_distance,
            "planned_spread_ratio": evaluation.spread_ratio,
            "planned_quantity": evaluation.quantity,
        }

    def _build_trade_record(self, report: TradeReport, close_snapshot: dict[str, Any]) -> dict[str, Any]:
        order_snapshot = dict(self.order_snapshots.get(self.trade_to_order.get(report.trade_id, -1), {}))
        record = {
            **order_snapshot,
            **close_snapshot,
            "trade_id": report.trade_id,
            "symbol": report.symbol,
            "side": report.side.name.lower(),
            "opened_at": report.opened_at,
            "closed_at": report.closed_at,
            "entry_price": report.entry_price,
            "exit_price": report.exit_price,
            "quantity": report.quantity,
            "pnl": report.pnl,
            "max_favorable_pnl": report.max_favorable_pnl,
            "max_adverse_pnl": report.max_adverse_pnl,
            "close_reason": report.close_reason.name.lower(),
            "duration_hours": (report.closed_at - report.opened_at).total_seconds() / 3600.0,
        }
        return record


def run_audited_backtest(runtime_args: argparse.Namespace) -> tuple[WindowResult, list[dict[str, Any]]]:
    feed = build_feed(runtime_args.history, runtime_args.symbols, runtime_args.start_date, runtime_args.end_date)
    base_engine = build_engine(feed=feed, args=runtime_args)
    engine = VolatilityAuditEngine(base_engine)

    started_at = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        engine.run()
    runtime_seconds = time.perf_counter() - started_at

    trades = list(engine.reports.trades)
    wins = [trade for trade in trades if trade.pnl > 0.0]
    losses = [trade for trade in trades if trade.pnl <= 0.0]
    gross_wins = sum(trade.pnl for trade in wins)
    gross_losses = abs(sum(trade.pnl for trade in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0.0 else float("inf") if gross_wins > 0.0 else 0.0
    close_counts = Counter(trade.close_reason.name for trade in trades)
    diagnostics = engine.strategy.diagnostics_summary() if isinstance(engine.strategy, SwingRegressionStrategy) else {}

    stop_loss_records = [record for record in engine.trade_records if record.get("close_reason") == CloseReason.STOP_LOSS.name.lower()]
    result = WindowResult(
        candidate_label=(
            f"window={getattr(runtime_args, 'regression_market_volatility_window_bars', 30)}_"
            f"threshold={float(getattr(runtime_args, 'regression_min_market_volatility_ratio', 0.0)):.6f}"
        ),
        candidate={
            "regression_market_volatility_window_bars": int(getattr(runtime_args, "regression_market_volatility_window_bars", 30)),
            "regression_min_market_volatility_ratio": float(getattr(runtime_args, "regression_min_market_volatility_ratio", 0.0)),
        },
        start_date=runtime_args.start_date,
        end_date=runtime_args.end_date,
        runtime_seconds=runtime_seconds,
        total_pnl=float(engine.portfolio.balance - engine.portfolio.initial_balance),
        trade_count=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        profit_factor=profit_factor,
        stop_loss_closes=close_counts.get(CloseReason.STOP_LOSS.name, 0),
        take_profit_closes=close_counts.get(CloseReason.TAKE_PROFIT.name, 0),
        other_closes=len(trades) - close_counts.get(CloseReason.STOP_LOSS.name, 0) - close_counts.get(CloseReason.TAKE_PROFIT.name, 0),
        analysis_windows=int(diagnostics.get("analysis_windows", 0)),
        low_volatility_rejections=int(diagnostics.get("reason_counts", {}).get("market_volatility_too_low", 0)),
        avg_stop_loss_entry_volatility_ratio=mean_or_none(record.get("market_volatility_ratio_entry") for record in stop_loss_records),
        avg_stop_loss_entry_volatility_atr=mean_or_none(record.get("market_volatility_atr_entry") for record in stop_loss_records),
        avg_stop_loss_close_volatility_ratio=mean_or_none(record.get("market_volatility_ratio_close") for record in stop_loss_records),
        avg_stop_loss_close_volatility_atr=mean_or_none(record.get("market_volatility_atr_close") for record in stop_loss_records),
    )
    return result, engine.trade_records


def to_serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (datetime, date)):
            serializable[key] = value.isoformat()
        elif isinstance(value, float):
            serializable[key] = None if not math.isfinite(value) else value
        else:
            serializable[key] = value
    return serializable


def bucket_start(day: date, anchor: date, period_days: int) -> date:
    offset = max(0, (day - anchor).days)
    return anchor + timedelta(days=(offset // period_days) * period_days)


def summarize_stop_loss_periods(
    trade_records: list[dict[str, Any]],
    anchor: date,
    period_days: int,
) -> list[dict[str, Any]]:
    buckets: dict[date, list[dict[str, Any]]] = {}
    for record in trade_records:
        if record.get("close_reason") != CloseReason.STOP_LOSS.name.lower():
            continue
        closed_at = record.get("closed_at")
        if not isinstance(closed_at, datetime):
            continue
        start = bucket_start(closed_at.date(), anchor, period_days)
        buckets.setdefault(start, []).append(record)

    period_summaries: list[dict[str, Any]] = []
    for start, records in buckets.items():
        end = start + timedelta(days=period_days - 1)
        period_summaries.append({
            "period_start": start,
            "period_end": end,
            "stop_loss_count": len(records),
            "symbols": dict(Counter(str(record.get("symbol")) for record in records)),
            "avg_entry_volatility_ratio": mean_or_none(record.get("market_volatility_ratio_entry") for record in records),
            "avg_entry_volatility_atr": mean_or_none(record.get("market_volatility_atr_entry") for record in records),
            "avg_close_volatility_ratio": mean_or_none(record.get("market_volatility_ratio_close") for record in records),
            "avg_close_volatility_atr": mean_or_none(record.get("market_volatility_atr_close") for record in records),
            "avg_loss_pnl": mean_or_none(record.get("pnl") for record in records),
            "gross_loss_pnl": sum(float(record.get("pnl") or 0.0) for record in records),
        })

    period_summaries.sort(
        key=lambda item: (
            -int(item["stop_loss_count"]),
            float(item["gross_loss_pnl"]),
            item["period_start"],
        )
    )
    return period_summaries


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, default=default_json, ensure_ascii=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_payload(payload), indent=2, default=default_json, allow_nan=False), encoding="utf-8")


def artifact_slug(base_args: argparse.Namespace, label: str | None) -> str:
    if label:
        return label
    symbols = "-".join(base_args.symbols)
    return f"{base_args.start_date.isoformat()}_{base_args.end_date.isoformat()}_{symbols}"


def serialize_window_result(result: WindowResult) -> dict[str, Any]:
    return {
        **asdict(result),
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
    }


def load_cached_window_results(path: Path) -> dict[tuple[str, str, str], WindowResult]:
    if not path.exists():
        return {}
    cached: dict[tuple[str, str, str], WindowResult] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            result = WindowResult(
                candidate_label=str(payload["candidate_label"]),
                candidate=dict(payload["candidate"]),
                start_date=date.fromisoformat(payload["start_date"]),
                end_date=date.fromisoformat(payload["end_date"]),
                runtime_seconds=float(payload["runtime_seconds"]),
                total_pnl=float(payload["total_pnl"]),
                trade_count=int(payload["trade_count"]),
                wins=int(payload["wins"]),
                losses=int(payload["losses"]),
                win_rate=float(payload["win_rate"]),
                profit_factor=float(payload["profit_factor"]),
                stop_loss_closes=int(payload["stop_loss_closes"]),
                take_profit_closes=int(payload["take_profit_closes"]),
                other_closes=int(payload["other_closes"]),
                analysis_windows=int(payload["analysis_windows"]),
                low_volatility_rejections=int(payload["low_volatility_rejections"]),
                avg_stop_loss_entry_volatility_ratio=payload.get("avg_stop_loss_entry_volatility_ratio"),
                avg_stop_loss_entry_volatility_atr=payload.get("avg_stop_loss_entry_volatility_atr"),
                avg_stop_loss_close_volatility_ratio=payload.get("avg_stop_loss_close_volatility_ratio"),
                avg_stop_loss_close_volatility_atr=payload.get("avg_stop_loss_close_volatility_atr"),
            )
            key = (result.candidate_label, payload["start_date"], payload["end_date"])
            cached[key] = result
    return cached


def append_window_result(path: Path, result: WindowResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_payload(serialize_window_result(result)), allow_nan=False) + "\n")


def evaluate_candidates(
    base_args: argparse.Namespace,
    candidates: list[Candidate],
    windows: list[tuple[date, date]],
    cache_path: Path,
    resume: bool,
) -> tuple[dict[tuple[str, str, str], WindowResult], list[WindowResult]]:
    cached = load_cached_window_results(cache_path) if resume else {}
    scheduled_runs = [(candidate, start, end) for candidate in candidates for start, end in windows]
    total_runs = len(scheduled_runs)
    completed = 0
    executed_results: list[WindowResult] = []
    overall_started_at = time.perf_counter()

    for candidate, start, end in scheduled_runs:
        key = (candidate.label, start.isoformat(), end.isoformat())
        if key in cached:
            executed_results.append(cached[key])
            completed += 1
            continue

        runtime_args = clone_runtime_args(base_args, start_date=start, end_date=end, candidate=candidate)
        result, _trade_records = run_audited_backtest(runtime_args)
        cached[key] = result
        executed_results.append(result)
        append_window_result(cache_path, result)
        completed += 1
        elapsed = time.perf_counter() - overall_started_at
        average_runtime = elapsed / max(1, completed)
        remaining = max(0, total_runs - completed)
        eta_seconds = average_runtime * remaining
        print(
            f"[optimizer] {completed}/{total_runs} {candidate.label} {start.isoformat()}..{end.isoformat()} "
            f"sl={result.stop_loss_closes} trades={result.trade_count} pnl={result.total_pnl:.2f} "
            f"low-vol-rejects={result.low_volatility_rejections} runtime={result.runtime_seconds:.1f}s eta={eta_seconds / 60.0:.1f}m",
            flush=True,
        )
    return cached, executed_results


def group_results_by_candidate(results: Iterable[WindowResult]) -> dict[str, list[WindowResult]]:
    grouped: dict[str, list[WindowResult]] = {}
    for result in results:
        grouped.setdefault(result.candidate_label, []).append(result)
    for items in grouped.values():
        items.sort(key=lambda item: item.start_date)
    return grouped


def aggregate_candidate(
    candidate_label: str,
    results: list[WindowResult],
    baseline_results: list[WindowResult],
    trade_retention_floor: float,
) -> dict[str, Any]:
    baseline_by_window = {
        (result.start_date, result.end_date): result
        for result in baseline_results
    }
    violations = 0
    windows_beating_baseline = 0
    windows_matching_baseline = 0
    for result in results:
        baseline = baseline_by_window[(result.start_date, result.end_date)]
        minimum_trades = math.ceil(baseline.trade_count * trade_retention_floor)
        if baseline.trade_count > 0 and result.trade_count < minimum_trades:
            violations += 1
        if result.stop_loss_closes < baseline.stop_loss_closes:
            windows_beating_baseline += 1
        elif result.stop_loss_closes == baseline.stop_loss_closes:
            windows_matching_baseline += 1

    total_stop_losses = sum(result.stop_loss_closes for result in results)
    total_trades = sum(result.trade_count for result in results)
    total_pnl = sum(result.total_pnl for result in results)
    total_low_vol_rejections = sum(result.low_volatility_rejections for result in results)
    baseline_stop_losses = sum(result.stop_loss_closes for result in baseline_results)
    baseline_trades = sum(result.trade_count for result in baseline_results)

    return {
        "candidate_label": candidate_label,
        "candidate": results[0].candidate,
        "windows": len(results),
        "trade_retention_violations": violations,
        "windows_beating_baseline_stop_losses": windows_beating_baseline,
        "windows_matching_baseline_stop_losses": windows_matching_baseline,
        "total_stop_loss_closes": total_stop_losses,
        "baseline_total_stop_loss_closes": baseline_stop_losses,
        "stop_loss_delta": total_stop_losses - baseline_stop_losses,
        "total_trades": total_trades,
        "baseline_total_trades": baseline_trades,
        "trade_retention_ratio": total_trades / baseline_trades if baseline_trades > 0 else 1.0,
        "total_pnl": total_pnl,
        "total_low_volatility_rejections": total_low_vol_rejections,
        "avg_stop_loss_entry_volatility_ratio": mean_or_none(result.avg_stop_loss_entry_volatility_ratio for result in results),
        "avg_stop_loss_close_volatility_ratio": mean_or_none(result.avg_stop_loss_close_volatility_ratio for result in results),
    }


def select_best_candidate_for_window(
    window_results: list[WindowResult],
    baseline: WindowResult,
    trade_retention_floor: float,
) -> WindowResult:
    minimum_trades = math.ceil(baseline.trade_count * trade_retention_floor)
    eligible = [
        result for result in window_results
        if baseline.trade_count == 0 or result.trade_count >= minimum_trades
    ]
    ranked_pool = eligible or window_results
    return sorted(
        ranked_pool,
        key=lambda result: (
            result.stop_loss_closes,
            -result.total_pnl,
            -result.trade_count,
            result.low_volatility_rejections,
            result.candidate_label,
        ),
    )[0]


def main() -> None:
    args = parse_args()
    backtest_tokens = parse_backtest_tokens(args.base_command, args.backtest_args)
    base_args = build_runtime_args(backtest_tokens)

    artifact_dir = args.artifacts_dir / artifact_slug(base_args, args.label)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"[optimizer] artifacts={artifact_dir}")

    baseline_path = artifact_dir / "baseline_full.json"
    baseline_records_path = artifact_dir / "baseline_trade_records.json"
    period_csv_path = artifact_dir / "worst_stop_loss_periods.csv"
    period_json_path = artifact_dir / "worst_stop_loss_periods.json"
    window_cache_path = artifact_dir / "candidate_window_results.jsonl"
    adaptive_csv_path = artifact_dir / "adaptive_schedule.csv"
    summary_path = artifact_dir / "summary.json"

    if args.resume and baseline_path.exists() and baseline_records_path.exists():
        baseline_full_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_records = json.loads(baseline_records_path.read_text(encoding="utf-8"))
        baseline_full_result = WindowResult(
            candidate_label=str(baseline_full_payload["candidate_label"]),
            candidate=dict(baseline_full_payload["candidate"]),
            start_date=date.fromisoformat(baseline_full_payload["start_date"]),
            end_date=date.fromisoformat(baseline_full_payload["end_date"]),
            runtime_seconds=float(baseline_full_payload["runtime_seconds"]),
            total_pnl=float(baseline_full_payload["total_pnl"]),
            trade_count=int(baseline_full_payload["trade_count"]),
            wins=int(baseline_full_payload["wins"]),
            losses=int(baseline_full_payload["losses"]),
            win_rate=float(baseline_full_payload["win_rate"]),
            profit_factor=float(baseline_full_payload["profit_factor"]),
            stop_loss_closes=int(baseline_full_payload["stop_loss_closes"]),
            take_profit_closes=int(baseline_full_payload["take_profit_closes"]),
            other_closes=int(baseline_full_payload["other_closes"]),
            analysis_windows=int(baseline_full_payload["analysis_windows"]),
            low_volatility_rejections=int(baseline_full_payload["low_volatility_rejections"]),
            avg_stop_loss_entry_volatility_ratio=baseline_full_payload.get("avg_stop_loss_entry_volatility_ratio"),
            avg_stop_loss_entry_volatility_atr=baseline_full_payload.get("avg_stop_loss_entry_volatility_atr"),
            avg_stop_loss_close_volatility_ratio=baseline_full_payload.get("avg_stop_loss_close_volatility_ratio"),
            avg_stop_loss_close_volatility_atr=baseline_full_payload.get("avg_stop_loss_close_volatility_atr"),
        )
        print("[optimizer] baseline audit loaded from cache")
    else:
        baseline_full_result, baseline_trade_records_raw = run_audited_backtest(base_args)
        baseline_records = [to_serializable_record(record) for record in baseline_trade_records_raw]
        write_json(baseline_path, serialize_window_result(baseline_full_result))
        write_json(baseline_records_path, baseline_records)
        print(
            f"[optimizer] baseline full-range run done: sl={baseline_full_result.stop_loss_closes} "
            f"trades={baseline_full_result.trade_count} pnl={baseline_full_result.total_pnl:.2f}"
        )

    baseline_trade_records: list[dict[str, Any]] = []
    for record in baseline_records:
        rebuilt = dict(record)
        for field_name in (
            "analysis_closed_at",
            "signal_timestamp",
            "opened_at",
            "closed_at",
            "market_volatility_close_reference_time",
        ):
            if rebuilt.get(field_name):
                rebuilt[field_name] = datetime.fromisoformat(str(rebuilt[field_name]))
        baseline_trade_records.append(rebuilt)

    worst_periods = summarize_stop_loss_periods(
        baseline_trade_records,
        anchor=base_args.start_date,
        period_days=args.period_days,
    )
    write_json(period_json_path, [
        {
            **item,
            "period_start": item["period_start"].isoformat(),
            "period_end": item["period_end"].isoformat(),
        }
        for item in worst_periods
    ])
    save_csv(period_csv_path, [
        {
            **item,
            "period_start": item["period_start"].isoformat(),
            "period_end": item["period_end"].isoformat(),
        }
        for item in worst_periods
    ])

    print("[optimizer] worst stop-loss periods:")
    for period in worst_periods[:args.top_periods]:
        print(
            f"  {period['period_start'].isoformat()}..{period['period_end'].isoformat()} "
            f"sl={period['stop_loss_count']} avg_entry_vol_ratio={period['avg_entry_volatility_ratio']} "
            f"avg_close_vol_ratio={period['avg_close_volatility_ratio']} symbols={period['symbols']}"
        )

    windows = split_date_range(
        base_args.start_date,
        base_args.end_date,
        window_days=args.evaluation_window_days,
        max_windows=args.max_evaluation_windows,
    )
    candidates = build_candidates(base_args, args)
    print(f"[optimizer] evaluating {len(candidates)} candidates over {len(windows)} windows")

    executed_results = evaluate_candidates(
        base_args=base_args,
        candidates=candidates,
        windows=windows,
        cache_path=window_cache_path,
        resume=args.resume,
    )[1]
    grouped = group_results_by_candidate(executed_results)
    baseline_label = candidates[0].label
    baseline_window_results = grouped[baseline_label]

    aggregated_candidates = [
        aggregate_candidate(
            candidate_label=label,
            results=results,
            baseline_results=baseline_window_results,
            trade_retention_floor=args.trade_retention_floor,
        )
        for label, results in grouped.items()
    ]
    aggregated_candidates.sort(
        key=lambda item: (
            int(item["trade_retention_violations"]),
            int(item["total_stop_loss_closes"]),
            -int(item["windows_beating_baseline_stop_losses"]),
            -float(item["total_pnl"]),
            -float(item["trade_retention_ratio"]),
            item["candidate_label"],
        )
    )

    adaptive_schedule: list[dict[str, Any]] = []
    for start, end in windows:
        per_window_results = [
            grouped[candidate.label][index]
            for index, _window in enumerate(windows)
            if _window == (start, end)
            for candidate in candidates
        ]
        baseline = next(result for result in per_window_results if result.candidate_label == baseline_label)
        best = select_best_candidate_for_window(per_window_results, baseline, args.trade_retention_floor)
        adaptive_schedule.append({
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "candidate_label": best.candidate_label,
            "regression_market_volatility_window_bars": best.candidate["regression_market_volatility_window_bars"],
            "regression_min_market_volatility_ratio": best.candidate["regression_min_market_volatility_ratio"],
            "stop_loss_closes": best.stop_loss_closes,
            "trade_count": best.trade_count,
            "total_pnl": best.total_pnl,
            "baseline_stop_loss_closes": baseline.stop_loss_closes,
            "baseline_trade_count": baseline.trade_count,
            "baseline_total_pnl": baseline.total_pnl,
        })
    save_csv(adaptive_csv_path, adaptive_schedule)

    recommendation = aggregated_candidates[0] if aggregated_candidates else None
    summary = {
        "base_command_tokens": backtest_tokens,
        "baseline_full_result": serialize_window_result(baseline_full_result),
        "worst_stop_loss_periods": [
            {
                **item,
                "period_start": item["period_start"].isoformat(),
                "period_end": item["period_end"].isoformat(),
            }
            for item in worst_periods[:args.top_periods]
        ],
        "evaluation_windows": [
            {"start_date": start.isoformat(), "end_date": end.isoformat()}
            for start, end in windows
        ],
        "top_static_candidates": aggregated_candidates[:args.top_candidates],
        "adaptive_schedule": adaptive_schedule,
        "recommended_static_candidate": recommendation,
    }
    write_json(summary_path, summary)

    print("[optimizer] top static candidates:")
    for item in aggregated_candidates[:args.top_candidates]:
        print(
            f"  {item['candidate_label']} sl={item['total_stop_loss_closes']} "
            f"delta={item['stop_loss_delta']:+d} pnl={item['total_pnl']:.2f} "
            f"trade_retention={item['trade_retention_ratio']:.2%} violations={item['trade_retention_violations']} "
            f"low-vol-rejects={item['total_low_volatility_rejections']}"
        )

    if recommendation is not None:
        print("[optimizer] recommended static candidate:")
        print(
            f"  window={recommendation['candidate']['regression_market_volatility_window_bars']} "
            f"threshold={recommendation['candidate']['regression_min_market_volatility_ratio']} "
            f"sl={recommendation['total_stop_loss_closes']} pnl={recommendation['total_pnl']:.2f}"
        )
    print(f"[optimizer] summary={summary_path}")


if __name__ == "__main__":
    main()
