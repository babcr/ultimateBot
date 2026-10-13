#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from ultimateBot.backtester.config.backtest_config import (
    RiskConfig,
    VolatilityConfig,
    format_backtest_bound,
    resolve_backtest_bounds,
)
from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.strategy.structure_decision_engine import resolve_structure_rules
from ultimateBot.backtester.strategy.structure_strategy import StructureStrategy
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.run_backtest import BASE_HISTORY_PATH, build_engine, build_feed


DEFAULT_RISK_CONFIG = RiskConfig()
DEFAULT_VOLATILITY_CONFIG = VolatilityConfig()
DEFAULT_START_YEAR = 2024
DEFAULT_END_YEAR = 2024

QUICK_SWEEP_PROFILE: dict[str, list[Any]] = {
    "level_lookback_bars": [500, 1000, 2000],
    "level_merge_atr_ratio": [0.03, 0.05, 0.08],
    "entry_distance_atr": [0.35, 0.55, 0.75],
    "breakout_distance_atr": [0.10, 0.20, 0.35],
    "breakout_cross_allowance_atr": [0.00, 0.05, 0.12],
    "breakout_retest_tolerance_atr": [0.10, 0.20, 0.35],
    "breakout_max_extension_atr": [0.35, 0.60, 0.90],
    "min_level_quality": [0.50, 0.65, 0.80],
    "pattern_confidence_threshold": [0.45, 0.65, 0.80],
    "directional_bias_threshold": [0.05, 0.20, 0.35],
    "min_entry_quality": [0.40, 0.55, 0.70],
    "min_breakout_quality": [0.50, 0.65, 0.80],
    "min_rr_ratio": [0.75, 1.00, 1.25],
    "min_target_distance_atr": [0.35, 0.50, 0.70],
    "decision_margin_threshold": [0.08, 0.12, 0.20],
    "confirmation_body_ratio": [0.20, 0.38, 0.55],
    "require_weekly_alignment": [False, True],
    "weekly_alignment_weight": [0.0, 0.30, 0.60],
}

WIDE_SWEEP_PROFILE: dict[str, list[Any]] = {
    **QUICK_SWEEP_PROFILE,
    "max_levels_per_side": [4, 8, 12],
    "level_merge_min_points": [1.0, 3.0, 5.0],
    "near_level_atr": [0.15, 0.30, 0.50],
    "min_level_touches": [2, 3, 4],
}

SWEEP_PROFILES = {
    "quick": QUICK_SWEEP_PROFILE,
    "wide": WIDE_SWEEP_PROFILE,
}


@dataclass(frozen=True)
class SweepResult:
    label: str
    phase: str
    runtime_seconds: float
    overrides: dict[str, Any]
    resolved_rules: dict[str, Any]
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
    expectancy: float
    max_drawdown_value: float
    max_drawdown_pct: float
    stop_loss_closes: int
    take_profit_closes: int
    other_closes: int
    diagnostics: dict[str, Any]


class DiagnosticStructureStrategy(StructureStrategy):
    def __init__(self, structure_rules: dict[str, Any] | None = None, minimum_candles: int | None = None):
        super().__init__(structure_rules=structure_rules, minimum_candles=minimum_candles)
        self.analysis_windows = 0
        self.decision_windows = 0
        self.rule_signal_counts: Counter[str] = Counter()
        self.signal_side_counts: Counter[str] = Counter()
        self.buy_blocker_counts: Counter[str] = Counter()
        self.sell_blocker_counts: Counter[str] = Counter()
        self.buy_style_counts: Counter[str] = Counter()
        self.sell_style_counts: Counter[str] = Counter()
        self.buy_setup_ok_count = 0
        self.sell_setup_ok_count = 0

    def on_market_data(self, batch: TickBatch, context: StrategyContext):
        circuit_candles = self._get_structure_candles(batch.symbol, context)
        if circuit_candles is None:
            return []

        latest_closed = circuit_candles["bid"][-1]
        if self._last_closed_candle.get(batch.symbol) == latest_closed.start_time:
            return []
        self._last_closed_candle[batch.symbol] = latest_closed.start_time

        point_size = max(context.market_state.get_point_size(batch.symbol), 1e-9)
        circuit_results = self._analyze_structure_circuits(
            circuit_candles, point_size, rules=self.rules_for_symbol(batch.symbol),
            symbol=batch.symbol, market_state=context.market_state,
        )

        selected_source, selected_decision = self._select_converged_decision(
            bid_analysis=circuit_results["bid"][1],
            ask_analysis=circuit_results["ask"][1],
            bid_decision=circuit_results["bid"][0],
            ask_decision=circuit_results["ask"][0],
        )
        self._record_window(
            analyses=[analysis for _decision, analysis in circuit_results.values()],
            selected_decision=selected_decision,
            selected_analysis=circuit_results[selected_source][1] if selected_source is not None else None,
        )
        if selected_decision is None:
            return []

        protective_level_price, protective_level_quality = self._resolve_risk_manager_protection(
            side=selected_decision.side,
            bid_analysis=circuit_results["bid"][1],
            ask_analysis=circuit_results["ask"][1],
            fallback_decision=selected_decision,
        )
        self.signal_side_counts[selected_decision.side] += 1
        return [
            self._build_signal(
                batch=batch,
                decision=selected_decision,
                protective_level_price=protective_level_price,
                protective_level_quality=protective_level_quality,
            )
        ]

    def _record_window(
        self,
        analyses: list[dict[str, Any]],
        selected_decision: Any,
        selected_analysis: dict[str, Any] | None,
    ) -> None:
        self.analysis_windows += 1
        if selected_decision is None:
            self.rule_signal_counts["flat"] += 1
        else:
            self.rule_signal_counts[str(selected_decision.side)] += 1
            self.decision_windows += 1

        if selected_analysis is not None:
            self._record_analysis_details(selected_analysis)
            return

        for analysis in analyses:
            self._record_analysis_details(analysis)

    def _record_analysis_details(self, analysis: dict[str, Any]) -> None:
        if not analysis:
            return

        buy_setup = analysis.get("buy_setup") or {}
        sell_setup = analysis.get("sell_setup") or {}
        for blocker in buy_setup.get("blockers", []):
            self.buy_blocker_counts[str(blocker)] += 1
        for blocker in sell_setup.get("blockers", []):
            self.sell_blocker_counts[str(blocker)] += 1

        buy_style = buy_setup.get("style")
        if buy_style:
            self.buy_style_counts[str(buy_style)] += 1
        sell_style = sell_setup.get("style")
        if sell_style:
            self.sell_style_counts[str(sell_style)] += 1

        if bool(buy_setup.get("setup_ok", False)):
            self.buy_setup_ok_count += 1
        if bool(sell_setup.get("setup_ok", False)):
            self.sell_setup_ok_count += 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "analysis_windows": self.analysis_windows,
            "decision_windows": self.decision_windows,
            "decision_rate": self.decision_windows / self.analysis_windows if self.analysis_windows else 0.0,
            "buy_setup_ok_count": self.buy_setup_ok_count,
            "sell_setup_ok_count": self.sell_setup_ok_count,
            "rule_signal_counts": dict(self.rule_signal_counts),
            "signal_side_counts": dict(self.signal_side_counts),
            "top_buy_blockers": self.buy_blocker_counts.most_common(8),
            "top_sell_blockers": self.sell_blocker_counts.most_common(8),
            "buy_styles": dict(self.buy_style_counts),
            "sell_styles": dict(self.sell_style_counts),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep high-impact structure-strategy parameters and rank the outcomes.",
    )
    parser.add_argument("--history", type=Path, default=BASE_HISTORY_PATH, help="Path to the historical tick data directory")
    parser.add_argument("--symbols", type=str, nargs="+", default=["EURUSD"], help="Symbols to include in the backtest")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive start date for the sweep backtests in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive end date for the sweep backtests in YYYY-MM-DD format")
    parser.add_argument("--start-year", type=int, default=None, help="Fallback inclusive start year when --start-date is not provided")
    parser.add_argument("--end-year", type=int, default=None, help="Fallback inclusive end year when --end-date is not provided")
    parser.add_argument("--balance", type=float, default=50.0, help="Initial balance for each backtest run")
    parser.add_argument("--batch-size", type=int, default=500, help="Fallback tick batch size used by non-time-based strategies")
    parser.add_argument("--quantity", type=float, default=None, help="Optional fixed quantity override")
    parser.add_argument("--risk-percent", type=float, default=1.0, help="Risk percentage per trade; 0.1 means 0.1 percent, 1.0 means 1 percent")
    parser.add_argument("--available-money-ratio", type=float, default=1.0, help="Share of the selected capital source available to the risk budget")
    parser.add_argument("--available-money-source", type=str, default="equity", choices=["equity", "free_margin", "balance"], help="Capital source used for risk sizing")
    parser.add_argument("--accepted-risk-overrun", type=float, default=0.25, help="Maximum allowed overshoot above the configured risk percentage when a fixed quantity is supplied")
    parser.add_argument("--max-positions", type=int, default=5, help="Maximum number of open positions")
    parser.add_argument("--reward-risk-ratio", type=float, default=DEFAULT_RISK_CONFIG.reward_risk_ratio, help="Take-profit distance as a multiple of the stop distance")
    parser.add_argument("--limit-spread-ratio", type=float, default=DEFAULT_RISK_CONFIG.limit_spread_ratio, help="Spread ratio above which market intent is degraded to a limit order")
    parser.add_argument("--max-spread-ratio", type=float, default=DEFAULT_RISK_CONFIG.max_spread_ratio, help="Spread ratio above which the trade is rejected")
    parser.add_argument("--candle-lookback", type=int, default=DEFAULT_VOLATILITY_CONFIG.candle_lookback, help="Number of candles used to estimate recent volatility")
    parser.add_argument("--minimum-candles", type=int, default=DEFAULT_VOLATILITY_CONFIG.minimum_candles, help="Minimum number of candles required before risk evaluation is allowed")
    parser.add_argument("--granularity-factor", type=float, default=DEFAULT_VOLATILITY_CONFIG.granularity_factor, help="Scaling factor used to discretize volatility extremes")
    parser.add_argument("--loss-shrink-ratio", type=float, default=DEFAULT_VOLATILITY_CONFIG.loss_shrink_ratio, help="Multiplier applied to candle volatility to derive stop distance")
    parser.add_argument("--offset-shrink-ratio", type=float, default=DEFAULT_VOLATILITY_CONFIG.offset_shrink_ratio, help="Multiplier applied to candle volatility to derive pending-order entry offset")
    parser.add_argument("--volatility-mode", type=str, choices=["swing", "intraday"], default=DEFAULT_VOLATILITY_CONFIG.mode, help="Controls whether volatility scaling rounds up or down")
    parser.add_argument("--mode", type=str, choices=["manual", "one-factor", "staged-grid"], default="staged-grid", help="Run one explicit override set, a single-parameter sensitivity scan, or a sensitivity scan followed by a focused combinatorial grid")
    parser.add_argument("--profile", type=str, choices=sorted(SWEEP_PROFILES), default="quick", help="Parameter profile to evaluate")
    parser.add_argument("--parameter", action="append", default=[], metavar="NAME", help="Limit the sweep to specific parameter names from the selected profile; can be repeated")
    parser.add_argument("--top-parameters", type=int, default=4, help="How many parameters to carry from the one-factor scan into the staged grid")
    parser.add_argument("--values-per-parameter", type=int, default=3, help="How many top values per selected parameter to keep for the staged grid")
    parser.add_argument("--max-grid-runs", type=int, default=81, help="Safety cap on the number of staged-grid combinations")
    parser.add_argument("--top-results", type=int, default=10, help="How many ranked runs to print")
    parser.add_argument("--rule", action="append", default=[], metavar="KEY=VALUE", help="Base structure-rule override applied before the sweep; can be repeated")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the full sweep results as JSON")
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


def parse_rule_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --rule '{item}'. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --rule '{item}'. Missing key.")
        overrides[key] = parse_scalar(raw_value.strip())
    return overrides


def parse_scalar(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in raw_value for char in (".", "e", "E")):
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def ranking_key(result: SweepResult) -> tuple[float, float, float, float]:
    return (
        result.total_pnl,
        result.profit_factor,
        result.win_rate,
        -result.max_drawdown_value,
    )


def select_sweep_profile(profile: dict[str, list[Any]], selected_parameters: list[str]) -> dict[str, list[Any]]:
    if not selected_parameters:
        return dict(profile)

    selected = {name.strip() for name in selected_parameters if name.strip()}
    unknown = sorted(selected - set(profile))
    if unknown:
        raise ValueError(
            "Unknown --parameter value(s): " + ", ".join(unknown) + ". Available: " + ", ".join(sorted(profile))
        )
    return {key: list(values) for key, values in profile.items() if key in selected}


def build_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        history=args.history,
        symbols=list(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        balance=args.balance,
        batch_size=args.batch_size,
        strategy="structure",
        quantity=args.quantity,
        risk_percent=args.risk_percent,
        available_money_ratio=args.available_money_ratio,
        available_money_source=args.available_money_source,
        accepted_risk_overrun=args.accepted_risk_overrun,
        max_positions=args.max_positions,
        reward_risk_ratio=args.reward_risk_ratio,
        limit_spread_ratio=args.limit_spread_ratio,
        max_spread_ratio=args.max_spread_ratio,
        candle_lookback=args.candle_lookback,
        minimum_candles=args.minimum_candles,
        granularity_factor=args.granularity_factor,
        loss_shrink_ratio=args.loss_shrink_ratio,
        offset_shrink_ratio=args.offset_shrink_ratio,
        volatility_mode=args.volatility_mode,
    )


def run_structure_backtest(runtime_args: argparse.Namespace, label: str, phase: str, overrides: dict[str, Any]) -> SweepResult:
    strategy = DiagnosticStructureStrategy(structure_rules=overrides)
    feed = build_feed(runtime_args.history, runtime_args.symbols, runtime_args.start_date, runtime_args.end_date)
    engine = build_engine(feed=feed, args=runtime_args, strategy_override=strategy)

    started_at = time.perf_counter()
    engine.run()
    runtime_seconds = time.perf_counter() - started_at

    trades = list(engine.reports.trades)
    pnls = [float(trade.pnl) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl <= 0.0]
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0.0 else float("inf") if gross_wins > 0.0 else 0.0

    balance_curve = [float(engine.portfolio.initial_balance)]
    running_balance = float(engine.portfolio.initial_balance)
    peak_balance = running_balance
    max_drawdown_value = 0.0
    for trade in trades:
        running_balance += float(trade.pnl)
        balance_curve.append(running_balance)
        peak_balance = max(peak_balance, running_balance)
        max_drawdown_value = min(max_drawdown_value, running_balance - peak_balance)

    close_counts = Counter(trade.close_reason.name for trade in trades)
    final_balance = float(engine.portfolio.balance)
    total_pnl = final_balance - float(engine.portfolio.initial_balance)
    trade_count = len(trades)
    avg_pnl = total_pnl / trade_count if trade_count else 0.0
    avg_win = gross_wins / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = len(wins) / trade_count if trade_count else 0.0
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss if trade_count else 0.0
    max_drawdown_pct = abs(max_drawdown_value) / max(balance_curve) * 100.0 if balance_curve and max(balance_curve) > 0.0 else 0.0

    return SweepResult(
        label=label,
        phase=phase,
        runtime_seconds=runtime_seconds,
        overrides=dict(overrides),
        resolved_rules=dict(strategy.structure_rules),
        final_balance=final_balance,
        total_pnl=total_pnl,
        trade_count=trade_count,
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        max_drawdown_value=abs(max_drawdown_value),
        max_drawdown_pct=max_drawdown_pct,
        stop_loss_closes=close_counts.get("STOP_LOSS", 0),
        take_profit_closes=close_counts.get("TAKE_PROFIT", 0),
        other_closes=trade_count - close_counts.get("STOP_LOSS", 0) - close_counts.get("TAKE_PROFIT", 0),
        diagnostics=strategy.diagnostics(),
    )


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_result(result: SweepResult, baseline: SweepResult | None = None) -> None:
    delta = result.total_pnl - baseline.total_pnl if baseline is not None else 0.0
    changed = ", ".join(f"{key}={format_value(value)}" for key, value in sorted(result.overrides.items())) or "baseline"
    diagnostics = result.diagnostics
    print(
        f"[{result.phase}] {result.label}: pnl={result.total_pnl:.2f} "
        f"delta={delta:+.2f} balance={result.final_balance:.2f} trades={result.trade_count} "
        f"win_rate={result.win_rate:.2%} pf={result.profit_factor:.2f} "
        f"dd={result.max_drawdown_value:.2f} sl/tp={result.stop_loss_closes}/{result.take_profit_closes} "
        f"analysis={diagnostics.get('decision_windows', 0)}/{diagnostics.get('analysis_windows', 0)} "
        f"runtime={result.runtime_seconds:.1f}s | {changed}"
    )


def print_diagnostics_summary(result: SweepResult) -> None:
    diagnostics = result.diagnostics
    print("Diagnostics:")
    print(
        f"  analysis_windows={diagnostics.get('analysis_windows', 0)} "
        f"decision_windows={diagnostics.get('decision_windows', 0)} "
        f"decision_rate={diagnostics.get('decision_rate', 0.0):.2%}"
    )
    print(f"  rule_signals={diagnostics.get('rule_signal_counts', {})}")
    print(f"  emitted_sides={diagnostics.get('signal_side_counts', {})}")
    print(f"  buy_setup_ok_count={diagnostics.get('buy_setup_ok_count', 0)} sell_setup_ok_count={diagnostics.get('sell_setup_ok_count', 0)}")
    print(f"  top_buy_blockers={diagnostics.get('top_buy_blockers', [])}")
    print(f"  top_sell_blockers={diagnostics.get('top_sell_blockers', [])}")
    print(f"  buy_styles={diagnostics.get('buy_styles', {})}")
    print(f"  sell_styles={diagnostics.get('sell_styles', {})}")


def print_ranked_results(results: list[SweepResult], top_results: int, title: str) -> None:
    print(f"\n{title}")
    ranked = sorted(results, key=ranking_key, reverse=True)
    for index, result in enumerate(ranked[:top_results], start=1):
        changed = ", ".join(f"{key}={format_value(value)}" for key, value in sorted(result.overrides.items())) or "baseline"
        print(
            f"{index:>2}. {result.label:<36} pnl={result.total_pnl:>8.2f} "
            f"balance={result.final_balance:>8.2f} trades={result.trade_count:>4} "
            f"win_rate={result.win_rate:>7.2%} pf={result.profit_factor:>5.2f} "
            f"dd={result.max_drawdown_value:>7.2f} | {changed}"
        )


def build_one_factor_tasks(
    baseline_rules: dict[str, Any],
    sweep_profile: dict[str, list[Any]],
) -> list[tuple[str, dict[str, Any]]]:
    tasks: list[tuple[str, dict[str, Any]]] = []
    for key, candidate_values in sweep_profile.items():
        seen_values: set[str] = set()
        for raw_value in candidate_values:
            resolved_value = resolve_structure_rules({key: raw_value})[key]
            signature = repr(resolved_value)
            if signature in seen_values or resolved_value == baseline_rules[key]:
                continue
            seen_values.add(signature)
            label = f"{key}={format_value(resolved_value)}"
            tasks.append((label, {key: raw_value}))
    return tasks


def shortlist_parameters(
    baseline_result: SweepResult,
    one_factor_results: list[SweepResult],
    top_parameters: int,
    values_per_parameter: int,
) -> dict[str, list[Any]]:
    results_by_param: dict[str, list[SweepResult]] = defaultdict(list)
    baseline_rules = baseline_result.resolved_rules
    for result in one_factor_results:
        if len(result.overrides) != 1:
            continue
        key = next(iter(result.overrides))
        results_by_param[key].append(result)

    ranking_rows: list[tuple[float, float, str, list[SweepResult]]] = []
    for key, results in results_by_param.items():
        best = max(results, key=ranking_key)
        delta = best.total_pnl - baseline_result.total_pnl
        ranking_rows.append((delta, abs(delta), key, sorted(results, key=ranking_key, reverse=True)))

    positive_rows = [row for row in ranking_rows if row[0] > 0.0]
    selected_rows = sorted(positive_rows or ranking_rows, key=lambda row: (row[0], row[1]), reverse=True)[:top_parameters]

    shortlisted: dict[str, list[Any]] = {}
    for _delta, _impact, key, ranked_results in selected_rows:
        values: list[Any] = [baseline_rules[key]]
        for result in ranked_results:
            candidate = result.resolved_rules[key]
            if candidate in values:
                continue
            values.append(candidate)
            if len(values) >= max(1, values_per_parameter):
                break
        shortlisted[key] = values
    return shortlisted


def build_staged_grid_tasks(
    shortlisted_parameters: dict[str, list[Any]],
    max_grid_runs: int,
) -> list[tuple[str, dict[str, Any]]]:
    if not shortlisted_parameters:
        return []

    keys = list(shortlisted_parameters)
    value_lists = [shortlisted_parameters[key] for key in keys]
    tasks: list[tuple[str, dict[str, Any]]] = []

    for combination in product(*value_lists):
        overrides = {key: value for key, value in zip(keys, combination)}
        label = "grid:" + ",".join(f"{key}={format_value(value)}" for key, value in overrides.items())
        tasks.append((label, overrides))
        if len(tasks) >= max_grid_runs:
            break
    return tasks


def write_results(output_path: Path, baseline_result: SweepResult, all_results: list[SweepResult], shortlisted_parameters: dict[str, list[Any]]) -> None:
    payload = {
        "baseline": asdict(baseline_result),
        "shortlisted_parameters": shortlisted_parameters,
        "results": [asdict(result) for result in all_results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_rule_overrides = parse_rule_overrides(list(args.rule))
    baseline_rules = resolve_structure_rules(base_rule_overrides)
    sweep_profile = select_sweep_profile(SWEEP_PROFILES[args.profile], list(args.parameter))
    runtime_args = build_runtime_args(args)

    print("Starting structure-rule sweep")
    print(f"Symbols: {runtime_args.symbols}")
    print(f"Window: {format_backtest_bound(runtime_args.start_date)} -> {format_backtest_bound(runtime_args.end_date)}")
    print(f"Mode: {args.mode}")
    print(f"Profile: {args.profile}")
    if args.parameter:
        print(f"Parameters: {', '.join(sorted(sweep_profile))}")
    if base_rule_overrides:
        print("Base overrides:")
        for key, value in sorted(base_rule_overrides.items()):
            print(f"  {key}={format_value(value)}")

    if args.mode == "manual":
        result = run_structure_backtest(runtime_args, label="manual", phase="manual", overrides=baseline_rules)
        print_result(result)
        print_diagnostics_summary(result)
        if args.output is not None:
            write_results(args.output, result, [result], {})
            print(f"\nFull results written to {args.output}")
        return

    baseline_result = run_structure_backtest(runtime_args, label="baseline", phase="baseline", overrides=baseline_rules)
    print_result(baseline_result)

    one_factor_tasks = build_one_factor_tasks(baseline_rules, sweep_profile)
    one_factor_results: list[SweepResult] = []
    for label, overrides in one_factor_tasks:
        merged_overrides = dict(base_rule_overrides)
        merged_overrides.update(overrides)
        result = run_structure_backtest(runtime_args, label=label, phase="one-factor", overrides=merged_overrides)
        one_factor_results.append(result)
        print_result(result, baseline=baseline_result)

    shortlisted_parameters = shortlist_parameters(
        baseline_result=baseline_result,
        one_factor_results=one_factor_results,
        top_parameters=args.top_parameters,
        values_per_parameter=args.values_per_parameter,
    )

    staged_grid_results: list[SweepResult] = []
    if args.mode == "staged-grid":
        staged_grid_tasks = build_staged_grid_tasks(shortlisted_parameters, args.max_grid_runs)
        print("\nSelected parameters for staged grid:")
        if not shortlisted_parameters:
            print("  none")
        else:
            for key, values in shortlisted_parameters.items():
                formatted_values = ", ".join(format_value(value) for value in values)
                print(f"  {key}: {formatted_values}")

        for label, overrides in staged_grid_tasks:
            merged_overrides = dict(base_rule_overrides)
            merged_overrides.update(overrides)
            result = run_structure_backtest(runtime_args, label=label, phase="staged-grid", overrides=merged_overrides)
            staged_grid_results.append(result)
            print_result(result, baseline=baseline_result)

    all_results = [baseline_result] + one_factor_results + staged_grid_results
    print_ranked_results(all_results, args.top_results, title="Top results")

    if args.output is not None:
        write_results(args.output, baseline_result, all_results, shortlisted_parameters)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
