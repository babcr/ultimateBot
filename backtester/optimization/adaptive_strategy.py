from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from random import Random

from ultimateBot.backtester.config.backtest_config import RiskConfig
from ultimateBot.backtester.orders.order import OrderRequest
from ultimateBot.backtester.reports.trade_report import TradeCloser, TradeReport
from ultimateBot.backtester.strategy.strategy import Strategy


@dataclass(frozen=True)
class StrategyParameters:
    lookback_bars: int
    trailing_stop_enabled: bool
    timeframe_minutes: int = 10
    use_global_timeframe_validation: bool = False


@dataclass(frozen=True)
class AdaptiveStrategyConfig:
    lookback_min: int = 5
    lookback_max: int = 20
    lookback_step: int = 1
    accuracy_window: int = 15
    accuracy_drop_percent: float = 2.
    accuracy_drop_mode: str = "relative"
    profit_factor_window: int = 15
    profit_factor_drop_percent: float = 2.
    profit_factor_drop_mode: str = "relative"
    min_profit_factor: float = 1.01
    optimization_metric: str = "accuracy"
    seed: int = 10
    min_accuracy_percent: float = 23.
    tighten_search_interval: bool = False
    search_long_arm_delta: int = 2
    search_short_arm_delta: int = 1
    min_risk_percent: float = .01
    optimize_risk: bool = True
    optimize_trailing_stop: bool = True
    optimize_global_timeframe_validation: bool = False
    timeframe_min_minutes: int = 9
    timeframe_max_minutes: int = 10
    timeframe_step_minutes: int = 1

    def __post_init__(self) -> None:
        if self.lookback_min < 2 or self.lookback_max <= self.lookback_min:
            raise ValueError("Adaptive lookback bounds require 2 <= min < max (max is exclusive)")
        if self.lookback_step < 1:
            raise ValueError("Adaptive lookback step must be positive")
        if self.accuracy_window < 1:
            raise ValueError("Adaptive accuracy window must contain at least one closed trade")
        if not isfinite(self.accuracy_drop_percent) or not 0.0 < self.accuracy_drop_percent <= 100.0:
            raise ValueError("Adaptive accuracy drop percent must be in (0, 100]")
        if self.accuracy_drop_mode not in ("relative", "points"):
            raise ValueError("Adaptive accuracy drop mode must be relative or points")
        if self.profit_factor_window < 1:
            raise ValueError("Adaptive profit-factor window must contain at least one closed trade")
        if not isfinite(self.profit_factor_drop_percent) or self.profit_factor_drop_percent <= 0.0:
            raise ValueError("Adaptive profit-factor drop must be positive")
        if self.profit_factor_drop_mode not in ("relative", "points"):
            raise ValueError("Adaptive profit-factor drop mode must be relative or points")
        if self.profit_factor_drop_mode == "relative" and self.profit_factor_drop_percent > 100.0:
            raise ValueError("Adaptive relative profit-factor drop percent must be in (0, 100]")
        if not isfinite(self.min_profit_factor) or self.min_profit_factor < 0.0:
            raise ValueError("Adaptive minimum profit factor must be finite and non-negative")
        if self.optimization_metric not in ("accuracy", "profit_factor"):
            raise ValueError("Adaptive optimization metric must be accuracy or profit_factor")
        if not isfinite(self.min_accuracy_percent) or not 0.0 <= self.min_accuracy_percent <= 100.0:
            raise ValueError("Adaptive minimum accuracy percent must be in [0, 100]")
        if self.search_long_arm_delta < 1 or self.search_short_arm_delta < 1:
            raise ValueError("Adaptive search arm deltas must be positive")
        if self.search_long_arm_delta < self.search_short_arm_delta:
            raise ValueError("Adaptive long-arm delta must be greater than or equal to the short-arm delta")
        if not isfinite(self.min_risk_percent) or self.min_risk_percent <= 0.0:
            raise ValueError("Adaptive minimum risk percent must be positive")
        if self.timeframe_min_minutes < 1 or self.timeframe_max_minutes <= self.timeframe_min_minutes:
            raise ValueError("Adaptive timeframe bounds require 1 <= min < max (max is exclusive)")
        if self.timeframe_step_minutes < 1:
            raise ValueError("Adaptive timeframe step must be positive")

    def candidates(
        self,
        timeframe_minutes: int | None = None,
        trailing_stop_values: tuple[bool, ...] = (False, True),
        global_timeframe_validation_values: tuple[bool, ...] = (False,),
    ) -> tuple[StrategyParameters, ...]:
        resolved_timeframe = self.timeframe_min_minutes if timeframe_minutes is None else int(timeframe_minutes)
        trailing_priority = {
            value: index
            for index, value in enumerate(trailing_stop_values)
        }
        validation_priority = {
            value: index
            for index, value in enumerate(global_timeframe_validation_values)
        }
        candidates = (
            StrategyParameters(lookback, trailing, resolved_timeframe, use_global_timeframe_validation)
            for use_global_timeframe_validation in global_timeframe_validation_values
            for lookback in range(self.lookback_min, self.lookback_max, self.lookback_step)
            for trailing in trailing_stop_values
        )
        # Adaptive search explores lookback and any enabled boolean dimensions;
        # timeframe stays at the symbol's origin.
        return tuple(sorted(
            candidates,
            key=lambda item: (
                validation_priority[item.use_global_timeframe_validation],
                item.lookback_bars,
                trailing_priority[item.trailing_stop_enabled],
            ),
        ))


class RollingTradeAccuracy:
    """Measure available positive-PnL closes, up to the rolling window limit."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("Accuracy window must contain at least one closed trade")
        self.outcomes: deque[bool] = deque(maxlen=window_size)
        self._wins = 0

    def add(self, pnl: float) -> float:
        if not isfinite(pnl):
            raise ValueError("Adaptive accuracy requires finite realized PnL")
        if len(self.outcomes) == self.outcomes.maxlen:
            self._wins -= self.outcomes[0]
        won = pnl > 0.0
        self.outcomes.append(won)
        self._wins += won
        return self._wins / len(self.outcomes)

    def clear(self) -> None:
        self.outcomes.clear()
        self._wins = 0


class RollingTradeProfitFactor:
    """Measure rolling profit factor from realized PnL inside the window."""

    def __init__(self, window_size: int):
        if window_size < 1:
            raise ValueError("Profit-factor window must contain at least one closed trade")
        self.pnl_samples: deque[float] = deque(maxlen=window_size)
        self._gross_wins = 0.0
        self._gross_losses = 0.0

    def add(self, pnl: float) -> float:
        if not isfinite(pnl):
            raise ValueError("Adaptive profit factor requires finite realized PnL")
        if len(self.pnl_samples) == self.pnl_samples.maxlen:
            outgoing = self.pnl_samples[0]
            if outgoing > 0.0:
                self._gross_wins -= outgoing
            elif outgoing < 0.0:
                self._gross_losses -= -outgoing
        self.pnl_samples.append(pnl)
        if pnl > 0.0:
            self._gross_wins += pnl
        elif pnl < 0.0:
            self._gross_losses += -pnl
        return self.value

    @property
    def value(self) -> float:
        if self._gross_wins <= 0.0:
            return 0.0
        if self._gross_losses <= 0.0:
            return float("inf")
        return self._gross_wins / self._gross_losses

    def clear(self) -> None:
        self.pnl_samples.clear()
        self._gross_wins = 0.0
        self._gross_losses = 0.0


@dataclass(frozen=True)
class AccuracyRiskScaler:
    minimum_percent: float
    maximum_percent: float

    def __post_init__(self) -> None:
        if not isfinite(self.minimum_percent) or self.minimum_percent <= 0.0:
            raise ValueError("Dynamic minimum risk percent must be positive")
        if not isfinite(self.maximum_percent) or self.maximum_percent < self.minimum_percent:
            raise ValueError("Dynamic maximum risk percent must be greater than or equal to the minimum")

    def for_accuracy(self, accuracy: float | None, accuracy_ceiling: float) -> float:
        if accuracy is None:
            return self.minimum_percent
        if not isfinite(accuracy_ceiling) or accuracy_ceiling < 0.0:
            raise ValueError("Dynamic risk accuracy ceiling must be finite and non-negative")
        if accuracy_ceiling == 0.0:
            progress = 1.0 if accuracy > 0.0 else 0.0
        else:
            progress = min(1.0, max(0.0, accuracy / accuracy_ceiling))
        return self.minimum_percent + progress * (self.maximum_percent - self.minimum_percent)


@dataclass
class SymbolAdaptiveState:
    current_parameters: StrategyParameters
    accuracy: RollingTradeAccuracy
    profit_factor: RollingTradeProfitFactor
    rng: Random
    configuration_id: int = 0
    switch_count: int = 0
    ignored_closes: int = 0
    reference_accuracy: float | None = None
    last_accuracy: float | None = None
    reference_metric_value: float | None = None
    last_metric_value: float | None = None
    search_lower_index: int = 0
    search_upper_index: int = 0
    winner_index: int | None = None
    current_risk_percent: float = 0.01
    maximum_accuracy: float | None = None
    risk_accuracy_ceiling: float = 0.0


class AdaptiveStrategyOptimizer:
    """Explore strategy variants independently per symbol on declining metrics."""

    def __init__(
        self,
        config: AdaptiveStrategyConfig,
        strategy: Strategy,
        trade_closer: TradeCloser,
        max_risk_percent: float = RiskConfig().risk_per_trade_percent,
    ):
        self.config = config
        self.strategy = strategy
        self.trade_closer = trade_closer
        self.max_risk_percent = max_risk_percent
        self.risk_scaler = AccuracyRiskScaler(
            config.min_risk_percent if config.optimize_risk else max_risk_percent,
            max_risk_percent,
        )
        # The common strategy interface validates support and every proposed window.
        origin = strategy.history_lookback_bars
        origin_interval = strategy.analysis_interval
        origin_trailing_stop_enabled = trade_closer.trailing_stop_enabled
        self._supports_global_timeframe_validation = strategy.supports_global_timeframe_validation()
        origin_global_timeframe_validation = strategy.get_global_timeframe_validation_enabled()
        origin_minutes = origin_interval.total_seconds() / 60.0 if origin_interval is not None else 0.0
        if not origin_minutes.is_integer() or origin_minutes < 1.0:
            raise ValueError("Adaptive strategy requires an initial timeframe expressed in whole minutes")
        self.candidates = config.candidates(
            timeframe_minutes=int(origin_minutes),
            trailing_stop_values=(False, True)
            if config.optimize_trailing_stop else (origin_trailing_stop_enabled,),
            global_timeframe_validation_values=(True, False)
            if self._supports_global_timeframe_validation and config.optimize_global_timeframe_validation
            else (origin_global_timeframe_validation,),
        )
        self._candidate_indices = {candidate: index for index, candidate in enumerate(self.candidates)}
        for candidate in self.candidates:
            strategy.history_capacity_for_lookback(candidate.lookback_bars)
        self.origin_parameters = StrategyParameters(
            origin,
            origin_trailing_stop_enabled,
            int(origin_minutes),
            origin_global_timeframe_validation,
        )
        self.states: dict[str, SymbolAdaptiveState] = {}
        self.required_history_capacity = max(
            strategy.history_capacity_for_lookback(origin),
            strategy.history_capacity_for_lookback(self.candidates[-1].lookback_bars),
        )

    @property
    def switch_count(self) -> int:
        return sum(state.switch_count for state in self.states.values())

    @property
    def ignored_closes(self) -> int:
        return sum(state.ignored_closes for state in self.states.values())

    def state_for(self, symbol: str) -> SymbolAdaptiveState:
        if symbol not in self.states:
            # A symbol's random sequence is unaffected by interleaving other symbols.
            digest = sha256(f"{self.config.seed}:{symbol}".encode("utf-8")).digest()
            accuracy = RollingTradeAccuracy(self.config.accuracy_window)
            self.states[symbol] = SymbolAdaptiveState(
                current_parameters=self.origin_parameters,
                accuracy=accuracy,
                profit_factor=RollingTradeProfitFactor(self.config.profit_factor_window),
                rng=Random(int.from_bytes(digest, "big")),
                search_upper_index=len(self.candidates) - 1,
                current_risk_percent=(
                    self.risk_scaler.minimum_percent
                    if self.config.optimize_risk
                    else self.max_risk_percent
                ),
                risk_accuracy_ceiling=(
                    self.config.min_accuracy_percent / 100.0
                    if self.config.optimize_risk
                    else 0.0
                ),
            )
        return self.states[symbol]

    def risk_percent_for(self, symbol: str) -> float:
        return self.state_for(symbol).current_risk_percent

    def _arm_deltas(self, winner_index: int) -> tuple[int, int]:
        left_span = winner_index
        right_span = len(self.candidates) - 1 - winner_index
        if left_span > right_span:
            return self.config.search_long_arm_delta, self.config.search_short_arm_delta
        if right_span > left_span:
            return self.config.search_short_arm_delta, self.config.search_long_arm_delta
        return self.config.search_short_arm_delta, self.config.search_short_arm_delta

    def _tighten_search_interval(self, state: SymbolAdaptiveState, winner_index: int) -> None:
        state.winner_index = winner_index
        if not self.config.tighten_search_interval:
            return
        left_delta, right_delta = self._arm_deltas(winner_index)
        state.search_lower_index = min(winner_index, state.search_lower_index + left_delta)
        state.search_upper_index = max(winner_index, state.search_upper_index - right_delta)

    def _loosen_search_interval(self, state: SymbolAdaptiveState) -> None:
        if state.winner_index is None:
            return
        left_delta, right_delta = self._arm_deltas(state.winner_index)
        state.search_lower_index = max(0, state.search_lower_index - left_delta)
        state.search_upper_index = min(len(self.candidates) - 1, state.search_upper_index + right_delta)

    def _selection_indices(self, state: SymbolAdaptiveState) -> list[int]:
        current_index = self._candidate_indices.get(state.current_parameters)
        return [
            index
            for index in range(state.search_lower_index, state.search_upper_index + 1)
            if index != current_index
        ]

    def prepare_order(self, order: OrderRequest) -> None:
        state = self.state_for(order.symbol)
        order.strategy_configuration_id = state.configuration_id
        order.strategy_lookback_bars = state.current_parameters.lookback_bars
        order.strategy_timeframe_minutes = state.current_parameters.timeframe_minutes
        order.strategy_use_global_timeframe_validation = (
            state.current_parameters.use_global_timeframe_validation
        )
        order.trailing_stop_enabled = state.current_parameters.trailing_stop_enabled

    @staticmethod
    def _metric_drop(reference: float, current: float, mode: str) -> float:
        if reference == float("inf"):
            return 0.0 if current == float("inf") else float("inf")
        raw_drop = reference - current
        if mode == "relative":
            return raw_drop / reference if reference > 0.0 else 0.0
        return raw_drop

    def _optimization_settings(self) -> tuple[float, str, float, str]:
        if self.config.optimization_metric == "profit_factor":
            threshold = (
                self.config.profit_factor_drop_percent / 100.0
                if self.config.profit_factor_drop_mode == "relative"
                else self.config.profit_factor_drop_percent
            )
            return (
                threshold,
                self.config.profit_factor_drop_mode,
                self.config.min_profit_factor,
                "profit_factor_below_minimum",
            )
        return (
            self.config.accuracy_drop_percent / 100.0,
            self.config.accuracy_drop_mode,
            self.config.min_accuracy_percent / 100.0,
            "accuracy_below_minimum",
        )

    def on_trade_closed(self, report: TradeReport) -> None:
        # Trades still open from an earlier period cannot rate the new parameters.
        state = self.state_for(report.symbol)
        if report.strategy_configuration_id != state.configuration_id:
            state.ignored_closes += 1
            return
        current_index = self._candidate_indices.get(state.current_parameters)
        if report.pnl > 0.0:
            if current_index is not None:
                self._tighten_search_interval(state, current_index)
        else:
            self._loosen_search_interval(state)
        accuracy = state.accuracy.add(report.pnl)
        state.last_accuracy = accuracy
        profit_factor = state.profit_factor.add(report.pnl)
        metric_value = profit_factor if self.config.optimization_metric == "profit_factor" else accuracy
        state.last_metric_value = metric_value
        minimum_accuracy = self.config.min_accuracy_percent / 100.0
        if self.config.optimize_risk:
            if accuracy > minimum_accuracy + 1e-12:
                if state.maximum_accuracy is None or accuracy > state.maximum_accuracy:
                    state.maximum_accuracy = accuracy
            state.risk_accuracy_ceiling = (
                state.maximum_accuracy if state.maximum_accuracy is not None else minimum_accuracy
            )
            state.current_risk_percent = self.risk_scaler.for_accuracy(
                accuracy,
                state.risk_accuracy_ceiling,
            )
        if state.reference_accuracy is None or accuracy > state.reference_accuracy:
            state.reference_accuracy = accuracy
        reference_accuracy = state.reference_accuracy
        if state.reference_metric_value is None or metric_value > state.reference_metric_value:
            state.reference_metric_value = metric_value
        reference_metric_value = state.reference_metric_value
        drop_threshold, drop_mode, minimum_metric, below_minimum_reason = self._optimization_settings()
        drop = (
            0.0
            if reference_metric_value is None
            else self._metric_drop(reference_metric_value, metric_value, drop_mode)
        )
        declining = drop + 1e-12 >= drop_threshold
        below_minimum = minimum_metric > 0.0 and metric_value <= minimum_metric + 1e-12
        if not declining and not below_minimum:
            return

        metric_samples = (
            state.profit_factor.pnl_samples
            if self.config.optimization_metric == "profit_factor"
            else state.accuracy.outcomes
        )
        metric_window_size = (
            self.config.profit_factor_window
            if self.config.optimization_metric == "profit_factor"
            else self.config.accuracy_window
        )
        sample_size = len(metric_samples)
        old = state.current_parameters
        selectable_indices = self._selection_indices(state)
        if not selectable_indices:
            return
        # Uniform random choice inside this symbol's current interval.
        index = selectable_indices[state.rng.randrange(len(selectable_indices))]
        selected = self.candidates[index]
        self.strategy.set_history_lookback_bars(selected.lookback_bars, symbol=report.symbol)
        if self._supports_global_timeframe_validation:
            self.strategy.set_global_timeframe_validation_enabled(
                selected.use_global_timeframe_validation,
                symbol=report.symbol,
            )
        state.current_parameters = selected
        state.configuration_id += 1
        state.switch_count += 1
        state.accuracy.clear()
        state.profit_factor.clear()
        state.reference_accuracy = None
        state.last_accuracy = None
        state.reference_metric_value = None
        state.last_metric_value = None
        state.current_risk_percent = (
            self.risk_scaler.minimum_percent
            if self.config.optimize_risk
            else self.max_risk_percent
        )
        reasons = ",".join(reason for reason, enabled in (
            (f"{self.config.optimization_metric}_drop", declining),
            (below_minimum_reason, below_minimum),
        ) if enabled)
        print(
            f"[adaptive-strategy] symbol={report.symbol} timestamp={report.closed_at.isoformat()} "
            f"trigger_trade_id={report.trade_id} configuration_id={state.configuration_id} "
            f"reason={reasons} minimum_accuracy_percent={self.config.min_accuracy_percent} "
            f"minimum_profit_factor={self.config.min_profit_factor} "
            f"accuracy={accuracy:.6f} reference_accuracy={reference_accuracy:.6f} "
            f"optimization_metric={self.config.optimization_metric} "
            f"metric_value={metric_value:.6f} reference_metric_value={reference_metric_value:.6f} "
            f"sample={sample_size}/{metric_window_size} "
            f"metric_drop={drop:.6f} drop_percent={drop * 100.0:.6f} drop_mode={drop_mode} "
            f"candidate_index={index} search_lower_index={state.search_lower_index} "
            f"search_upper_index={state.search_upper_index} winner_index={state.winner_index} "
            f"risk_percent={state.current_risk_percent} "
            f"risk_accuracy_ceiling={state.risk_accuracy_ceiling:.6f} "
            f"maximum_accuracy={state.maximum_accuracy} "
            f"previous_lookback={old.lookback_bars} "
            f"lookback={selected.lookback_bars} previous_timeframe_minutes={old.timeframe_minutes} "
            f"timeframe_minutes={selected.timeframe_minutes} "
            f"previous_global_validation={old.use_global_timeframe_validation} "
            f"global_validation={selected.use_global_timeframe_validation} "
            f"previous_trailing={old.trailing_stop_enabled} trailing={selected.trailing_stop_enabled}",
            flush=True,
        )
