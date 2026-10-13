from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ultimateBot.backtester.config.backtest_config import DEFAULT_CANDLE_INTERVAL
from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.market.candle_builder import Candle, PriceSource
from ultimateBot.backtester.market.market_state import MarketState
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide
from ultimateBot.backtester.strategy.swing_strategy import SwingStrategy, STRUCTURE_PRICE_SOURCES
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import (
    LIMIT_ENTRY_STYLES,
    STOP_ENTRY_STYLES,
    StructureDecision,
    build_shared_structure_levels,
    decide_market_structure,
    resolve_structure_rules,
)


class StructureStrategy(SwingStrategy):
    """Strategy that emits signals from the shared market-structure engine only."""

    minimum_lookback_bars = 60

    def __init__(
        self,
        structure_rules: dict[str, Any] | None = None,
        minimum_candles: int | None = None,
        analysis_interval: timedelta = DEFAULT_CANDLE_INTERVAL,
    ):
        super().__init__(
            structure_rules=resolve_structure_rules(structure_rules),
            minimum_candles=minimum_candles,
            analysis_interval=analysis_interval,
        )
        self._level_count_log_window = 0

    def on_market_data(self, batch: TickBatch, context: StrategyContext) -> list[Signal]:
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
        bid_decision, bid_analysis = circuit_results["bid"]
        ask_decision, ask_analysis = circuit_results["ask"]
        self._maybe_log_level_vector_size(
            symbol=batch.symbol,
            closed_start=latest_closed.start_time,
            bid_analysis=bid_analysis,
            ask_analysis=ask_analysis,
        )

        _selected_source, selected_decision = self._select_converged_decision(
            bid_analysis=bid_analysis,
            ask_analysis=ask_analysis,
            bid_decision=bid_decision,
            ask_decision=ask_decision,
        )
        if selected_decision is None:
            return []

        protective_level_price, protective_level_quality = self._resolve_risk_manager_protection(
            side=selected_decision.side,
            bid_analysis=bid_analysis,
            ask_analysis=ask_analysis,
            fallback_decision=selected_decision,
        )

        return [
            self._build_signal(
                batch=batch,
                decision=selected_decision,
                protective_level_price=protective_level_price,
                protective_level_quality=protective_level_quality,
            )
        ]

    def _analyze_structure_circuits(
        self,
        circuit_candles: dict[PriceSource, list[Candle]],
        point_size: float,
        rules: dict[str, Any] | None = None,
        symbol: str | None = None,
        market_state: MarketState | None = None,
    ) -> dict[PriceSource, tuple[StructureDecision | None, dict[str, Any]]]:
        rules = self.structure_rules if rules is None else rules
        circuit_frames = {
            price_source: self._build_candle_frame(closed_candles)
            for price_source, closed_candles in circuit_candles.items()
        }
        shared_swings = self._build_shared_structure_swings(
            bid_frame=circuit_frames["bid"],
            ask_frame=circuit_frames["ask"],
            point_size=point_size, rules=rules, symbol=symbol, market_state=market_state,
        )
        shared_levels = build_shared_structure_levels(
            bid_candles=circuit_frames["bid"],
            ask_candles=circuit_frames["ask"],
            point_size=point_size,
            config=rules,
            level_swings=shared_swings,
        )
        return {
            price_source: self._analyze_structure_frame(
                candle_frame,
                point_size,
                level_swings=shared_swings,
                structure_levels=shared_levels,
                rules=rules,
            )
            for price_source, candle_frame in circuit_frames.items()
        }

    def _analyze_structure_circuit(
        self,
        closed_candles: list[Candle],
        point_size: float,
        level_swings: list[dict[str, Any]] | None = None,
        structure_levels: list[dict[str, Any]] | None = None,
        rules: dict[str, Any] | None = None,
    ) -> tuple[StructureDecision | None, dict[str, Any]]:
        return self._analyze_structure_frame(
            candle_frame=self._build_candle_frame(closed_candles),
            point_size=point_size,
            level_swings=level_swings,
            structure_levels=structure_levels,
            rules=rules,
        )

    def _analyze_structure_frame(
        self,
        candle_frame: pd.DataFrame,
        point_size: float,
        level_swings: list[dict[str, Any]] | None = None,
        structure_levels: list[dict[str, Any]] | None = None,
        rules: dict[str, Any] | None = None,
    ) -> tuple[StructureDecision | None, dict[str, Any]]:
        return decide_market_structure(
            candles=candle_frame,
            point_size=point_size,
            config=self.structure_rules if rules is None else rules,
            level_swings=level_swings,
            structure_levels=structure_levels,
        )

    def _maybe_log_level_vector_size(
        self,
        symbol: str,
        closed_start: datetime,
        bid_analysis: dict[str, Any],
        ask_analysis: dict[str, Any],
    ) -> None:
        if not bool(self.structure_rules.get("live_level_count", False)):
            return

        self._level_count_log_window += 1
        every = max(1, int(self.structure_rules.get("live_level_count_every", 1)))
        if self._level_count_log_window % every != 0:
            return

        def merged_projected_level_count(analysis: dict[str, Any]) -> int:
            merged_prices = {
                float(level["price"])
                for level in (analysis.get("support_levels") or []) + (analysis.get("resistance_levels") or [])
                if isinstance(level, dict) and level.get("price") is not None
            }
            return len(merged_prices)

        bid_merged_count = merged_projected_level_count(bid_analysis)
        ask_merged_count = merged_projected_level_count(ask_analysis)
        count_summary = f"merged_levels={bid_merged_count}"
        if bid_merged_count != ask_merged_count:
            count_summary = f"bid_merged_levels={bid_merged_count} ask_merged_levels={ask_merged_count}"

        print(
            f"[structure-level-count] window={self._level_count_log_window} "
            f"symbol={symbol} closed={closed_start.isoformat()} {count_summary}",
            flush=True,
        )

    def _select_converged_decision(
        self,
        bid_analysis: dict[str, Any],
        ask_analysis: dict[str, Any],
        bid_decision: StructureDecision | None,
        ask_decision: StructureDecision | None,
    ) -> tuple[PriceSource | None, StructureDecision | None]:
        bid_side = self._suggested_side(bid_analysis)
        ask_side = self._suggested_side(ask_analysis)
        if bid_side is None or ask_side is None:
            return None, None
        if bid_side != ask_side:
            return None, None
        if bid_side == "buy":
            if ask_decision is None or ask_decision.side != "buy":
                return "ask", None
            return "ask", ask_decision
        if bid_decision is None or bid_decision.side != "sell":
            return "bid", None
        return "bid", bid_decision

    def _rule_signal_side(self, analysis: dict[str, Any] | None) -> str | None:
        if not isinstance(analysis, dict):
            return None
        rule_signal = int(analysis.get("rule_signal", 1))
        if rule_signal == 2:
            return "buy"
        if rule_signal == 0:
            return "sell"
        return None

    def _suggested_side(self, analysis: dict[str, Any] | None) -> str | None:
        if not isinstance(analysis, dict):
            return None
        buy_score = float(analysis.get("buy_score", 0.0))
        sell_score = float(analysis.get("sell_score", 0.0))
        margin = abs(float(analysis.get("rule_bias", 0.0)))
        threshold = float(self.structure_rules.get("decision_margin_threshold", 0.12))
        if margin < threshold:
            return None
        if buy_score > sell_score:
            return "buy"
        if sell_score > buy_score:
            return "sell"
        return None

    def _resolve_risk_manager_protection(
        self,
        side: str,
        bid_analysis: dict[str, Any],
        ask_analysis: dict[str, Any],
        fallback_decision: StructureDecision,
    ) -> tuple[float | None, float]:
        if side == "buy":
            setup = bid_analysis.get("buy_setup") or {}
        else:
            setup = ask_analysis.get("sell_setup") or {}

        protective_level_price = setup.get("protective_level_price")
        if protective_level_price is None:
            return fallback_decision.protective_level_price, fallback_decision.protective_level_quality
        return float(protective_level_price), float(setup.get("protective_level_quality", 0.0))

    def _build_signal(
        self,
        batch: TickBatch,
        decision: StructureDecision,
        protective_level_price: float | None,
        protective_level_quality: float,
    ) -> Signal:
        return Signal(
            symbol=batch.symbol,
            timestamp=batch.ticks[-1].timestamp,
            side=self._signal_side_from_decision(decision.side),
            confidence=decision.confidence,
            entry_mode=self._entry_mode_for_style(decision.entry_style),
            protective_level_price=protective_level_price,
            protective_level_quality=protective_level_quality,
            protective_stop_buffer=decision.protective_stop_buffer,
        )

    def _signal_side_from_decision(self, side: str) -> SignalSide:
        return SignalSide.BUY if side == "buy" else SignalSide.SELL

    def _entry_mode_for_style(self, entry_style: str | None) -> SignalEntryMode:
        if entry_style in LIMIT_ENTRY_STYLES:
            return SignalEntryMode.LIMIT
        if entry_style in STOP_ENTRY_STYLES:
            return SignalEntryMode.STOP
        return SignalEntryMode.MARKET
