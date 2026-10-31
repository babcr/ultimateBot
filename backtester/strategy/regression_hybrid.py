"""Coordinate independent frozen-channel entries with one decision per symbol."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.regression_breakout import RegressionBreakoutTracker
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel, RegressionReentryTracker
from ultimateBot.backtester.strategy.signal import Signal


class RegressionHybridTracker:
    def __init__(self, reentry: RegressionReentryTracker, breakout: RegressionBreakoutTracker, pullback=None, impulse=None):
        # Ordering also gives failed-breakout priority for same-side duplicates.
        self.branches = {"failed-breakout": reentry, "breakout-retest": breakout}
        if pullback is not None:
            self.branches[pullback.entry_name] = pullback
        if impulse is not None:
            self.branches[impulse.entry_name] = impulse
        self.windows = Counter()
        self.confirmed = Counter()
        self.selected = Counter()
        self.reasons = {name: Counter() for name in self.branches}
        self.conflicts = 0
        self.duplicates = 0
        self.suppressed_complementary = 0

    def clear(self, symbol: str | None = None) -> None:
        for tracker in self.branches.values():
            tracker.clear(symbol)

    def needs_channel(self, symbol: str) -> bool:
        return any(symbol not in tracker.events for tracker in self.branches.values())

    def diagnostics_summary(self) -> dict[str, Any]:
        return {
            "conflicting_confirmations": self.conflicts,
            "same_side_duplicates": self.duplicates,
            "suppressed_complementary_confirmations": self.suppressed_complementary,
            "branches": {name: {"analysis_windows": self.windows[name],
                                "confirmations": self.confirmed[name],
                                "selected_signals": self.selected[name],
                                "reason_counts": dict(self.reasons[name])}
                         for name in self.branches},
        }

    def evaluate(self, symbol: str, bid: Candle, ask: Candle, timestamp: datetime,
                 interval: timedelta, rules: dict[str, Any], channel: FrozenChannel | None,
                 fit_analysis: dict[str, Any] | None = None) -> tuple[Signal | None, dict[str, Any]]:
        analyses = {}
        candidates = []
        for name, tracker in self.branches.items():
            if symbol not in tracker.events and channel is None:
                analysis = deepcopy(fit_analysis or {})
                analysis.update(entry_mode=name, closed_at=bid.start_time, symbol=symbol, direction="flat")
                analysis["reasons"] = analysis.get("reasons") or ["hybrid_channel_unavailable"]
                signal = None
            else:
                # A pending tracker ignores the new fit and keeps its own frozen channel.
                signal, analysis = tracker.evaluate(symbol, bid, ask, timestamp, interval, rules, channel)
            analyses[name] = analysis
            self.windows[name] += 1
            self.reasons[name].update(analysis["reasons"])
            if signal is not None:
                self.confirmed[name] += 1
                candidates.append((name, signal))

        summary = {"entry_mode": "hybrid", "symbol": symbol, "closed_at": bid.start_time,
                   "direction": "flat", "selected_entry_mode": None,
                   "reasons": ["hybrid_no_confirmation"], "branches": analyses}
        if not candidates:
            return None, summary

        summary["cancelled_pending_branches"] = [name for name, tracker in self.branches.items()
                                                  if symbol in tracker.events]
        # Consume both branches before downstream risk/session/validator checks.
        self.clear(symbol)
        primary = [(name,signal) for name,signal in candidates if name in ("failed-breakout","breakout-retest")]
        if primary:
            # The experimental branch never overrides/rejects a primary entry.
            self.suppressed_complementary += len(candidates)-len(primary)
            candidates = primary
        if len({signal.side for _, signal in candidates}) > 1:
            self.conflicts += 1
            summary["reasons"] = ["hybrid_conflicting_signals"]
            return None, summary
        if len(candidates) > 1:
            self.duplicates += 1
        name, signal = candidates[0]
        self.selected[name] += 1
        summary.update(direction=analyses[name]["direction"], selected_entry_mode=name, reasons=[])
        context = {**(signal.entry_context or {}), "strategy_entry_mode": "hybrid",
                   "hybrid_selected_branch": name, "hybrid_confirmation_count": len(candidates)}
        return replace(signal, entry_context=context), summary
