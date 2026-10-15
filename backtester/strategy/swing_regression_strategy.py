from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_decision_engine import SWING_PRICE_SOURCES, _candle_context, _rolling_atr
from ultimateBot.backtester.strategy.swing_regression_config import (
    DEFAULT_SWING_REGRESSION_CONFIG, SwingRegressionConfig,
)
from ultimateBot.backtester.strategy.swing_regression_decision_engine import (
    decide_swing_regression, resolve_swing_regression_rules, select_swing_window,
)
from ultimateBot.backtester.strategy.swing_strategy import SwingStrategy


def _bucket_start(timestamp: datetime, interval: timedelta) -> datetime:
    interval_seconds = max(1, int(interval.total_seconds()))
    normalized = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
    epoch = int(normalized.timestamp())
    bucket_epoch = epoch - (epoch % interval_seconds)
    return datetime.fromtimestamp(bucket_epoch, tz=normalized.tzinfo)


def _aggregate_candle_group(bucket_start: datetime, candles: list[Candle]) -> Candle:
    return Candle(
        start_time=bucket_start,
        open_price=float(candles[0].open_price),
        high_price=max(float(candle.high_price) for candle in candles),
        low_price=min(float(candle.low_price) for candle in candles),
        close_price=float(candles[-1].close_price),
        tick_count=sum(int(candle.tick_count) for candle in candles),
        bid_volume_sum=sum(candle.bid_volume_sum for candle in candles),
        ask_volume_sum=sum(candle.ask_volume_sum for candle in candles),
        bid_volume_count=sum(candle.bid_volume_count for candle in candles),
        ask_volume_count=sum(candle.ask_volume_count for candle in candles),
    )


def _aggregate_candles_by_multiplier(
    candles: list[Candle],
    base_interval: timedelta,
    multiplier: int,
) -> list[Candle]:
    if multiplier <= 1:
        return list(candles)
    aggregated: list[Candle] = []
    grouped: list[Candle] = []
    higher_interval = base_interval * multiplier
    current_bucket: datetime | None = None
    for candle in sorted(candles, key=lambda item: item.start_time):
        bucket = _bucket_start(candle.start_time, higher_interval)
        if current_bucket is None or bucket != current_bucket:
            if len(grouped) == multiplier:
                aggregated.append(_aggregate_candle_group(current_bucket, grouped))
            current_bucket = bucket
            grouped = [candle]
            continue
        grouped.append(candle)
    if current_bucket is not None and len(grouped) == multiplier:
        aggregated.append(_aggregate_candle_group(current_bucket, grouped))
    return aggregated


class SwingRegressionStrategy(SwingStrategy):
    """Fork at the shared swing vectors, evaluated once per fresh confirmed swing."""

    def __init__(
        self, regression_rules: dict[str, Any] | None = None,
        minimum_candles: int | None = None, analysis_interval: timedelta | None = None,
        *, config: SwingRegressionConfig | None = None,
    ):
        self.config = config if config is not None else DEFAULT_SWING_REGRESSION_CONFIG
        configured_rules = self.config.to_rules()
        configured_rules.update({key: value for key, value in (regression_rules or {}).items() if value is not None})
        resolved_rules = resolve_swing_regression_rules(configured_rules)
        if minimum_candles is None:
            minimum_candles = self.config.execution.minimum_candles
        if analysis_interval is None:
            analysis_interval = self.config.volatility.candle_interval
        regression_lookback = int(resolved_rules["regression_lookback_bars"])
        required_candles = max(int(minimum_candles or 2), regression_lookback)
        if resolved_rules["regression_require_level_reaction"]:
            required_candles = max(required_candles, int(resolved_rules["level_lookback_bars"]))
        if float(resolved_rules["regression_min_market_volatility_ratio"]) > 0.0:
            required_candles = max(required_candles, int(resolved_rules["regression_market_volatility_window_bars"]))
        global_multiplier = int(resolved_rules["regression_global_timeframe_multiplier"])
        global_capacity = 0
        if global_multiplier > 1:
            global_capacity = regression_lookback * global_multiplier
            if float(resolved_rules["regression_min_market_volatility_ratio"]) > 0.0:
                global_capacity = max(
                    global_capacity,
                    int(resolved_rules["regression_market_volatility_window_bars"]) * global_multiplier,
                )
            required_candles = max(required_candles, global_capacity)
        super().__init__(resolved_rules, required_candles, analysis_interval)
        # Snapshot of effective startup inputs. Adaptive per-symbol settings
        # continue to use their existing dedicated setters.
        self.config = replace(
            self.config.with_rules(resolved_rules),
            volatility=replace(self.config.volatility, candle_interval=self.analysis_interval),
            execution=replace(self.config.execution, minimum_candles=minimum_candles),
        )
        self.regression_rules = self.structure_rules
        self._minimum_candles_override = minimum_candles
        # Higher-timeframe confirmation keeps its startup configuration.
        self._global_regression_rules = dict(resolved_rules)
        self._global_timeframe_validation_enabled = bool(
            self._global_regression_rules["regression_use_global_timeframe_validation"]
        )
        self._global_timeframe_validation_by_symbol: dict[str, bool] = {}
        self._global_history_capacity = global_capacity
        self.last_analysis: dict[str, dict[str, Any]] = {}
        self.last_global_analysis: dict[str, dict[str, Any]] = {}
        self._analysis_windows = 0
        self._analysis_windows_by_symbol: Counter[str] = Counter()
        self._reason_counts: Counter[str] = Counter()
        self._reason_counts_by_symbol: dict[str, Counter[str]] = {}
        self._global_analysis_windows = 0
        self._global_analysis_windows_by_symbol: Counter[str] = Counter()
        self._global_reason_counts: Counter[str] = Counter()
        self._global_reason_counts_by_symbol: dict[str, Counter[str]] = {}
        self._last_global_closed_candle: dict[str, datetime] = {}

    @property
    def history_lookback_bars(self) -> int:
        return int(self.regression_rules["regression_lookback_bars"])

    @property
    def level_history_lookback_bars(self) -> int:
        return int(self.regression_rules["level_lookback_bars"])

    def rules_for_symbol(self, symbol: str) -> dict[str, Any]:
        return {**self.regression_rules, "regression_lookback_bars": self.get_history_lookback_bars(symbol),
                "protective_stop_buffer_atr": self.get_protective_stop_buffer_atr(symbol)}

    @property
    def global_history_lookback_bars(self) -> int:
        return int(self._global_regression_rules["regression_lookback_bars"])

    @property
    def global_history_capacity(self) -> int:
        return self._global_history_capacity

    def history_capacity_for_lookback(self, bars: int) -> int:
        required = max(super().history_capacity_for_lookback(bars), self.global_history_capacity)
        if self.regression_rules["regression_require_level_reaction"]:
            required = max(required, self.level_history_lookback_bars)
        if float(self.regression_rules["regression_min_market_volatility_ratio"]) > 0.0:
            required = max(required, int(self.regression_rules["regression_market_volatility_window_bars"]))
        return required

    def set_history_lookback_bars(self, bars: int, symbol: str | None = None) -> None:
        required = self.history_capacity_for_lookback(bars)
        if symbol is None:
            self.regression_rules["regression_lookback_bars"] = bars
            self.minimum_candles = required
        else:
            self._lookback_by_symbol[symbol] = bars
        # Only the local analysis depends on the adaptive window.
        if symbol is None:
            self.last_analysis.clear()
        else:
            self.last_analysis.pop(symbol, None)

    def set_analysis_interval(self, interval: timedelta, symbol: str | None = None) -> None:
        super().set_analysis_interval(interval, symbol=symbol)
        # The higher-timeframe confirmation remains based on the fixed startup interval.
        if symbol is None:
            self.last_analysis.clear()
        else:
            self.last_analysis.pop(symbol, None)

    def supports_global_timeframe_validation(self) -> bool:
        return True

    def get_global_timeframe_validation_enabled(self, symbol: str | None = None) -> bool:
        if symbol is None:
            return self._global_timeframe_validation_enabled
        return self._global_timeframe_validation_by_symbol.get(symbol, self._global_timeframe_validation_enabled)

    def set_global_timeframe_validation_enabled(self, enabled: bool, symbol: str | None = None) -> None:
        if symbol is None:
            self._global_timeframe_validation_enabled = bool(enabled)
            self._global_timeframe_validation_by_symbol.clear()
            self.last_analysis.clear()
            return
        self._global_timeframe_validation_by_symbol[symbol] = bool(enabled)
        self.last_analysis.pop(symbol, None)

    def diagnostics_summary(self) -> dict[str, Any]:
        return {
            "analysis_windows": self._analysis_windows,
            "analysis_windows_by_symbol": dict(self._analysis_windows_by_symbol),
            "reason_counts": dict(self._reason_counts),
            "reason_counts_by_symbol": {
                symbol: dict(counts)
                for symbol, counts in self._reason_counts_by_symbol.items()
            },
            "global_analysis_windows": self._global_analysis_windows,
            "global_analysis_windows_by_symbol": dict(self._global_analysis_windows_by_symbol),
            "global_reason_counts": dict(self._global_reason_counts),
            "global_reason_counts_by_symbol": {
                symbol: dict(counts)
                for symbol, counts in self._global_reason_counts_by_symbol.items()
            },
        }

    def _record_analysis(self, symbol: str, analysis: dict[str, Any], *, global_pipeline: bool = False) -> None:
        if global_pipeline:
            self._global_analysis_windows += 1
            self._global_analysis_windows_by_symbol[symbol] += 1
            counter = self._global_reason_counts
            by_symbol = self._global_reason_counts_by_symbol
        else:
            self._analysis_windows += 1
            self._analysis_windows_by_symbol[symbol] += 1
            counter = self._reason_counts
            by_symbol = self._reason_counts_by_symbol
        reasons = [str(reason) for reason in analysis.get("reasons", []) if reason]
        if reasons:
            counter.update(reasons)
            by_symbol.setdefault(symbol, Counter()).update(reasons)

    def _evaluate_regression_pipeline(
        self,
        symbol: str,
        bid_candles: list[Candle],
        ask_candles: list[Candle],
        mid_candles: list[Candle],
        point_size: float,
        *,
        timeframe_label: str,
        timeframe_multiplier: int,
        require_fresh_swings: bool = True,
        rules: dict[str, Any] | None = None,
        context: StrategyContext | None = None,
    ) -> tuple[Any, dict[str, Any]] | None:
        rules = self.rules_for_symbol(symbol) if rules is None else rules
        regression_lookback = int(rules["regression_lookback_bars"])
        level_lookback = int(rules["level_lookback_bars"])
        use_levels = require_fresh_swings and bool(rules["regression_require_level_reaction"])
        min_market_volatility_ratio = float(rules["regression_min_market_volatility_ratio"])
        market_volatility_window = int(rules["regression_market_volatility_window_bars"])
        shared_bars = max(
            regression_lookback, level_lookback if use_levels else 0,
            market_volatility_window if min_market_volatility_ratio > 0.0 else 0,
        )
        if min(len(bid_candles), len(ask_candles), len(mid_candles)) < shared_bars:
            return None
        latest_closed = bid_candles[-1].start_time
        if ask_candles[-1].start_time != latest_closed or mid_candles[-1].start_time != latest_closed:
            return None

        # Each quote source has one maximum candle vector. All analysis views
        # use offsets into it and keep those coordinates for their swings.
        frames = {
            source: self._build_candle_frame(candles[-shared_bars:])
            for source, candles in (("bid", bid_candles), ("ask", ask_candles), ("mid", mid_candles))
        }
        bar_count = shared_bars
        regression_start = bar_count - regression_lookback
        level_start = max(0, bar_count - level_lookback)
        mid_frame = frames["mid"]
        atr = _rolling_atr(mid_frame.iloc[regression_start:], point_size, regression_lookback)
        level_atr = (
            _rolling_atr(mid_frame.iloc[level_start:], point_size, level_lookback)
            if use_levels else atr
        )
        regression_source_atrs = {
            source: _rolling_atr(frames[source].iloc[regression_start:], point_size, regression_lookback)
            for source in ("bid", "ask")
        }
        level_source_atrs = {
            source: _rolling_atr(frames[source].iloc[level_start:], point_size, level_lookback)
            for source in ("bid", "ask")
        } if use_levels else regression_source_atrs
        swings = self._build_shared_swing_candidates(
            frames,
            symbol=symbol if context is not None else None,
            market_state=context.market_state if context is not None else None,
        )
        regression_swings = select_swing_window(
            swings, regression_start, bar_count, atr, regression_source_atrs,
        )
        fresh_swings = [swing for swing in regression_swings if int(swing["index"]) == bar_count - 1]
        if require_fresh_swings and not fresh_swings:
            return None

        effective_market_volatility_window = (
            market_volatility_window
            if min_market_volatility_ratio > 0.0
            else max(2, min(market_volatility_window, bar_count))
        )
        volatility_frame = mid_frame.iloc[bar_count - effective_market_volatility_window:]
        market_volatility_atr = _rolling_atr(volatility_frame, point_size, effective_market_volatility_window)
        current_price = float(mid_frame["close"].iloc[-1])
        market_volatility_ratio = market_volatility_atr / max(abs(current_price), point_size, 1e-9)
        oldest_candle_anchor_candidates = [
            {"kind": kind, "price": float(frames[SWING_PRICE_SOURCES[kind]][kind].iloc[regression_start]),
             "index": regression_start, "source": "lookback_boundary",
             "price_source": SWING_PRICE_SOURCES[kind], "label": "oldest_candle_" + kind}
            for kind in ("high", "low")
        ]
        decision, analysis = decide_swing_regression(
            swings,
            current_price,
            atr,
            bar_count,
            rules,
            fresh_swings if require_fresh_swings else None,
            oldest_candle_anchor_candidates,
            market_volatility_atr=market_volatility_atr,
            market_volatility_ratio=market_volatility_ratio,
            point_size=point_size,
            candle_closes=mid_frame["close"].astype(float).tolist(),
            level_atr=level_atr,
            regression_source_atrs=regression_source_atrs,
            level_source_atrs=level_source_atrs,
            candle_contexts=(
                {source: _candle_context(frames[source], regression_source_atrs[source], rules)
                 for source in ("bid", "ask")}
                if require_fresh_swings and rules["regression_require_candle_confirmation"] else None
            ),
        )
        analysis["symbol"] = symbol
        analysis["closed_at"] = latest_closed
        analysis["pipeline"] = timeframe_label
        analysis["timeframe_multiplier"] = timeframe_multiplier
        analysis["fresh_swings"] = [dict(swing) for swing in fresh_swings]
        if analysis["anchor"] is not None:
            analysis["anchor"]["time"] = mid_candles[-bar_count + int(analysis["anchor"]["index"])].start_time
        return decision, analysis

    def _global_confirmation_enabled(self, symbol: str | None = None) -> bool:
        return (
            self.get_global_timeframe_validation_enabled(symbol)
            and int(self._global_regression_rules["regression_global_timeframe_multiplier"]) > 1
        )

    def _refresh_global_pipeline(
        self,
        symbol: str,
        bid_candles: list[Candle],
        ask_candles: list[Candle],
        mid_candles: list[Candle],
        point_size: float,
    ) -> None:
        if not self._global_confirmation_enabled(symbol):
            return
        multiplier = int(self._global_regression_rules["regression_global_timeframe_multiplier"])
        capacity = self.global_history_capacity
        aggregated_bid = _aggregate_candles_by_multiplier(bid_candles[-capacity:], self.analysis_interval, multiplier)
        aggregated_ask = _aggregate_candles_by_multiplier(ask_candles[-capacity:], self.analysis_interval, multiplier)
        aggregated_mid = _aggregate_candles_by_multiplier(mid_candles[-capacity:], self.analysis_interval, multiplier)
        if not aggregated_bid or not aggregated_ask or not aggregated_mid:
            return
        latest_closed = aggregated_bid[-1].start_time
        if aggregated_ask[-1].start_time != latest_closed or aggregated_mid[-1].start_time != latest_closed:
            return
        if self._last_global_closed_candle.get(symbol) == latest_closed:
            return
        self._last_global_closed_candle[symbol] = latest_closed
        evaluation = self._evaluate_regression_pipeline(
            symbol,
            aggregated_bid,
            aggregated_ask,
            aggregated_mid,
            point_size,
            timeframe_label="global",
            timeframe_multiplier=multiplier,
            require_fresh_swings=False,
            rules=self._global_regression_rules,
        )
        if evaluation is None:
            return
        _decision, analysis = evaluation
        self.last_global_analysis[symbol] = analysis
        self._record_analysis(symbol, analysis, global_pipeline=True)

    def _refresh_global_from_fixed_history(self, symbol: str, context: StrategyContext, point_size: float) -> None:
        if not self._global_confirmation_enabled(symbol):
            return
        global_candles = {
            source: context.market_state.get_candles(
                symbol,
                include_active=False,
                price_source=source,
                limit=self.global_history_capacity,
                candle_interval=self.analysis_interval,
            )
            for source in ("bid", "ask", "mid")
        }
        if not all(len(candles) == self.global_history_capacity for candles in global_candles.values()):
            return
        self._refresh_global_pipeline(
            symbol,
            global_candles["bid"],
            global_candles["ask"],
            global_candles["mid"],
            point_size,
        )

    def _apply_global_trend_filter(
        self,
        symbol: str,
        local_side: str,
        analysis: dict[str, Any],
    ) -> bool:
        enabled = self._global_confirmation_enabled(symbol)
        analysis["global_confirmation_enabled"] = enabled
        analysis["global_timeframe_multiplier"] = int(self._global_regression_rules["regression_global_timeframe_multiplier"])
        analysis["global_reference_slope_threshold_atr"] = float(
            self._global_regression_rules["regression_global_reference_slope_threshold_atr"]
        )
        analysis["global_min_parallelism"] = float(self._global_regression_rules["regression_global_min_parallelism"])
        analysis["global_confirmation_passed"] = not enabled
        analysis["global_confirmation_reason"] = None
        analysis["global_reference_slope_atr"] = None
        analysis["global_reference_line_kind"] = None
        analysis["global_parallelism"] = None
        analysis["global_trend_direction"] = None
        analysis["global_pipeline_closed_at"] = self._last_global_closed_candle.get(symbol)
        if not enabled:
            return True

        def reject(reason: str) -> bool:
            analysis["global_confirmation_passed"] = False
            analysis["global_confirmation_reason"] = reason
            analysis["reasons"] = [reason]
            return False

        global_analysis = self.last_global_analysis.get(symbol)
        if global_analysis is None:
            return reject("global_analysis_missing")

        global_closed_at = global_analysis.get("closed_at")
        if isinstance(global_closed_at, datetime):
            analysis["global_pipeline_closed_at"] = global_closed_at

        anchored_line = global_analysis.get("anchored_line")
        global_atr = float(global_analysis.get("atr", 0.0))
        if not isinstance(anchored_line, dict) or not isfinite(global_atr) or global_atr <= 0.0:
            return reject("global_reference_slope_missing")

        anchored_slope = float(anchored_line.get("slope", float("nan")))
        if not isfinite(anchored_slope):
            return reject("global_reference_slope_missing")

        global_reference_slope_atr = anchored_slope / global_atr
        global_parallelism = float(global_analysis.get("parallelism", 0.0))
        if not isfinite(global_parallelism):
            global_parallelism = 0.0

        analysis["global_reference_slope_atr"] = global_reference_slope_atr
        analysis["global_reference_line_kind"] = anchored_line.get("kind")
        analysis["global_parallelism"] = global_parallelism
        analysis["global_trend_direction"] = (
            "buy" if global_reference_slope_atr > 0.0
            else "sell" if global_reference_slope_atr < 0.0
            else "flat"
        )

        slope_threshold = float(self._global_regression_rules["regression_global_reference_slope_threshold_atr"])
        if local_side == "buy":
            if global_reference_slope_atr <= 0.0:
                return reject("global_reference_slope_direction_mismatch")
            if global_reference_slope_atr < slope_threshold:
                return reject("global_reference_slope_too_weak")
        else:
            if global_reference_slope_atr >= 0.0:
                return reject("global_reference_slope_direction_mismatch")
            if abs(global_reference_slope_atr) < slope_threshold:
                return reject("global_reference_slope_too_weak")

        if global_parallelism < float(self._global_regression_rules["regression_global_min_parallelism"]):
            return reject("global_parallelism_too_low")

        analysis["global_confirmation_passed"] = True
        analysis["global_confirmation_reason"] = "global_trend_confirmed"
        return True

    def on_market_data(self, batch: TickBatch, context: StrategyContext) -> list[Signal]:
        point_size = max(context.market_state.get_point_size(batch.symbol), 1e-9)
        self._refresh_global_from_fixed_history(batch.symbol, context, point_size)
        circuits = self._get_structure_candles(batch.symbol, context)
        if circuits is None:
            return []
        latest_closed = circuits["bid"][-1].start_time
        if self._last_closed_candle.get(batch.symbol) == latest_closed:
            return []
        self._last_closed_candle[batch.symbol] = latest_closed
        local_interval = self.get_analysis_interval(batch.symbol)
        all_mid_candles = context.market_state.get_candles(
            batch.symbol, include_active=False, price_source="mid",
            limit=self.history_capacity_for_lookback(self.get_history_lookback_bars(batch.symbol)),
            candle_interval=local_interval,
        )
        evaluation = self._evaluate_regression_pipeline(
            batch.symbol,
            circuits["bid"],
            circuits["ask"],
            all_mid_candles,
            point_size,
            timeframe_label="local",
            timeframe_multiplier=1,
            context=context,
        )
        if evaluation is None:
            return []
        decision, analysis = evaluation
        if decision is not None and not self._apply_global_trend_filter(batch.symbol, str(decision.side), analysis):
            decision = None
        self.last_analysis[batch.symbol] = analysis
        self._record_analysis(batch.symbol, analysis)
        if self.regression_rules["regression_live_analysis"]:
            anchor = analysis["anchor"] or {}
            anchored_line = analysis["anchored_line"] or {}
            floating_line = analysis["floating_line"] or {}
            global_suffix = ""
            if analysis.get("global_confirmation_enabled"):
                global_suffix = (
                    f" global_mult={analysis['global_timeframe_multiplier']}"
                    f" global_pass={analysis.get('global_confirmation_passed')}"
                    f" global_reason={analysis.get('global_confirmation_reason')}"
                    f" global_ref_slope_atr={analysis.get('global_reference_slope_atr')}"
                    f" global_parallelism={analysis.get('global_parallelism')}"
                )
            print(
                f"[swing-regression] window={self._analysis_windows} symbol={batch.symbol} "
                f"closed={latest_closed.isoformat()} origin={anchor.get('kind')} origin_price={anchor.get('price')} "
                f"shared_bars={analysis['bar_count']} "
                f"regression_start={analysis.get('regression_window_start')} "
                f"level_start={analysis.get('level_window_start')} "
                f"swing_count={analysis['swing_count']} swing_density={analysis['swing_density']:.4f} "
                f"filtered_pairs={analysis['filtered_consecutive_swing_pair_count']} filtered_pair_density={analysis['filtered_consecutive_swing_pair_density']:.4f} "
                f"parallelism={analysis['parallelism']:.4f} shared_slope={analysis['shared_slope']:.8g} "
                f"normalized_slope={analysis['normalized_slope']:.4f} "
                f"slope_strength={analysis['slope_strength']:.4f} slope_scale_atr={analysis['slope_scale_atr']:.4g} "
                f"market_vol_atr={analysis['market_volatility_atr']:.6g} "
                f"market_vol_ratio={analysis['market_volatility_ratio']:.6g} "
                f"vol_anchored_atr={analysis['anchored_line_volatility_atr']:.4f} "
                f"vol_floating_atr={analysis['floating_line_volatility_atr']:.4f} "
                f"score={analysis['trend_score']:.4f} direction={analysis['direction']} "
                f"r2_anchored={anchored_line.get('r_squared')} r2_floating={floating_line.get('r_squared')} "
                f"level_reaction={analysis.get('level_reaction_passed')} "
                f"candle_confirmation={analysis.get('candle_confirmation_passed')} "
                f"body_ratio={(analysis.get('confirmation_candle') or {}).get('body_ratio')} "
                f"entry_style={(analysis.get('entry_level_reaction') or {}).get('style')} "
                f"entry_level={((analysis.get('entry_level_reaction') or {}).get('level') or {}).get('price')} "
                f"reason={','.join(analysis['reasons'])}{global_suffix}",
                flush=True,
            )
        if decision is None:
            return []
        return [Signal(
            symbol=batch.symbol, timestamp=batch.ticks[-1].timestamp,
            side=SignalSide.BUY if decision.side == "buy" else SignalSide.SELL,
            confidence=decision.confidence, entry_mode=SignalEntryMode.MARKET,
            protective_level_price=decision.protective_level_price,
            protective_level_quality=decision.protective_level_quality,
            protective_stop_buffer=decision.protective_stop_buffer,
        )]
