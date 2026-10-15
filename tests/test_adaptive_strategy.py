import io
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from ultimateBot.backtester.config.backtest_config import DEFAULT_CANDLE_INTERVAL
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.execution.fill import Fill
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.optimization.adaptive_strategy import (
    AccuracyRiskScaler, AdaptiveStrategyConfig, AdaptiveStrategyOptimizer, RollingTradeAccuracy,
    RollingTradeProfitFactor,
)
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.report_collector import ReportCollector
from ultimateBot.backtester.reports.trade_report import TradeCloser, TradeReport
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.structure_strategy import StructureStrategy
from ultimateBot.backtester.strategy.swing_regression_config import SwingRegressionConfig
from ultimateBot.backtester.strategy.swing_regression_strategy import SwingRegressionStrategy
from ultimateBot.run_backtest import (
    build_engine, parse_args, resolve_backtest_adaptive_choice_values,
    resolve_backtest_adaptive_config, resolve_backtest_regression_rules,
)


START = datetime(2020, 1, 2, tzinfo=timezone.utc)


class HistoryStrategy(Strategy):
    def __init__(
        self,
        lookback=3,
        supports_global_timeframe_validation=False,
        use_global_timeframe_validation=True,
    ):
        self.lookback = lookback
        self.lookbacks_by_symbol = {}
        self.analysis_interval = timedelta(minutes=10)
        self.analysis_intervals_by_symbol = {}
        self._supports_global_timeframe_validation = supports_global_timeframe_validation
        self._use_global_timeframe_validation = bool(use_global_timeframe_validation)
        self.use_global_timeframe_validation_by_symbol = {}

    @property
    def history_lookback_bars(self):
        return self.lookback

    def history_capacity_for_lookback(self, bars):
        if bars < 2:
            raise ValueError("At least two candles required")
        return bars

    def get_history_lookback_bars(self, symbol):
        return self.lookbacks_by_symbol.get(symbol, self.lookback)

    def set_history_lookback_bars(self, bars, symbol=None):
        self.history_capacity_for_lookback(bars)
        if symbol is None:
            self.lookback = bars
        else:
            self.lookbacks_by_symbol[symbol] = bars

    def get_analysis_interval(self, symbol):
        return self.analysis_intervals_by_symbol.get(symbol, self.analysis_interval)

    def set_analysis_interval(self, interval, symbol=None):
        if symbol is None:
            self.analysis_interval = interval
        else:
            self.analysis_intervals_by_symbol[symbol] = interval

    def supports_global_timeframe_validation(self):
        return self._supports_global_timeframe_validation

    def get_global_timeframe_validation_enabled(self, symbol=None):
        if symbol is None:
            return self._use_global_timeframe_validation
        return self.use_global_timeframe_validation_by_symbol.get(symbol, self._use_global_timeframe_validation)

    def set_global_timeframe_validation_enabled(self, enabled, symbol=None):
        if not self._supports_global_timeframe_validation and enabled:
            raise ValueError("HistoryStrategy does not support global timeframe validation")
        if symbol is None:
            self._use_global_timeframe_validation = bool(enabled)
            self.use_global_timeframe_validation_by_symbol.clear()
        else:
            self.use_global_timeframe_validation_by_symbol[symbol] = bool(enabled)

    def on_market_data(self, batch, context):
        return []


def report(pnl, configuration_id=0, trade_id=1, symbol="TEST"):
    return TradeReport(
        trade_id, symbol, OrderSide.BUY, START, START + timedelta(seconds=trade_id),
        100.0, 100.0 + pnl, 1.0, pnl, max(0.0, pnl), min(0.0, pnl),
        CloseReason.TAKE_PROFIT if pnl > 0.0 else CloseReason.STOP_LOSS,
        strategy_configuration_id=configuration_id,
    )


def order(order_id=1, target=106.0):
    return OrderRequest(order_id, "TEST", START, START, START, OrderSide.BUY, 1.0,
                        stop_loss=98.0, take_profit=target,
                        trailing_stop_activation_price=100.0 + (target - 100.0) / 2.0)


class AdaptiveStrategyTests(unittest.TestCase):
    def optimizer(self, strategy=None, **overrides):
        parameters = dict(lookback_min=3, lookback_max=6, accuracy_window=4,
                          accuracy_drop_percent=25.0, min_accuracy_percent=0.0,
                          tighten_search_interval=True,
                          optimize_global_timeframe_validation=True)
        parameters.update(overrides)
        return AdaptiveStrategyOptimizer(
            AdaptiveStrategyConfig(**parameters),
            HistoryStrategy() if strategy is None else strategy,
            TradeCloser(),
        )

    def close(self, optimizer, pnl, configuration_id=None, symbol="TEST"):
        state = optimizer.state_for(symbol)
        current_id = state.configuration_id if configuration_id is None else configuration_id
        with redirect_stdout(io.StringIO()):
            optimizer.on_trade_closed(report(pnl, current_id, symbol=symbol))

    def test_candidate_vector_has_both_modes_and_excludes_upper_bound(self):
        pairs = AdaptiveStrategyConfig(10, 13).candidates()
        self.assertEqual([(p.lookback_bars, p.trailing_stop_enabled) for p in pairs],
                         [(10, False), (10, True), (11, False), (11, True), (12, False), (12, True)])
        self.assertEqual([p.lookback_bars for p in AdaptiveStrategyConfig(10, 20, lookback_step=4).candidates()],
                         [10, 10, 14, 14, 18, 18])
        ordered = AdaptiveStrategyConfig(
            10, 12, timeframe_min_minutes=5, timeframe_max_minutes=16,
            timeframe_step_minutes=5,
        ).candidates(timeframe_minutes=10)
        self.assertEqual(
            [(p.timeframe_minutes, p.lookback_bars, p.trailing_stop_enabled) for p in ordered],
            [(10, lookback, trailing)
             for lookback in (10, 11)
             for trailing in (False, True)],
        )
        global_ordered = AdaptiveStrategyConfig(10, 11).candidates(
            timeframe_minutes=10,
            global_timeframe_validation_values=(True, False),
        )
        self.assertEqual(
            [
                (p.use_global_timeframe_validation, p.lookback_bars, p.trailing_stop_enabled)
                for p in global_ordered
            ],
            [
                (True, 10, False),
                (True, 10, True),
                (False, 10, False),
                (False, 10, True),
            ],
        )

    def test_optimizer_can_freeze_trailing_stop_to_its_origin_value(self):
        optimizer = AdaptiveStrategyOptimizer(
            AdaptiveStrategyConfig(
                3, 5, accuracy_window=2, accuracy_drop_percent=100,
                min_accuracy_percent=60, optimize_trailing_stop=False,
            ),
            HistoryStrategy(),
            TradeCloser(trailing_stop_enabled=True),
        )
        self.assertEqual(
            [(p.lookback_bars, p.trailing_stop_enabled) for p in optimizer.candidates],
            [(3, True), (4, True)],
        )
        with patch.object(optimizer, "_selection_indices", return_value=[1]):
            self.close(optimizer, -1)
        state = optimizer.state_for("TEST")
        self.assertTrue(state.current_parameters.trailing_stop_enabled)
        prepared = order()
        optimizer.prepare_order(prepared)
        optimizer.trade_closer.prepare_order(prepared)
        self.assertTrue(prepared.trailing_stop_enabled)
        self.assertIsNone(prepared.take_profit)
        self.assertEqual(prepared.trailing_stop_activation_price, 103)

    def test_optimizer_can_freeze_global_validation_to_its_origin_value(self):
        strategy = HistoryStrategy(
            supports_global_timeframe_validation=True,
            use_global_timeframe_validation=False,
        )
        optimizer = AdaptiveStrategyOptimizer(
            AdaptiveStrategyConfig(
                3, 5, accuracy_window=2, accuracy_drop_percent=100,
                min_accuracy_percent=60, optimize_global_timeframe_validation=False,
            ),
            strategy,
            TradeCloser(),
        )
        self.assertEqual(
            {p.use_global_timeframe_validation for p in optimizer.candidates},
            {False},
        )
        with patch.object(optimizer, "_selection_indices", return_value=[1]):
            self.close(optimizer, -1)
        state = optimizer.state_for("TEST")
        self.assertFalse(state.current_parameters.use_global_timeframe_validation)
        self.assertFalse(strategy.get_global_timeframe_validation_enabled("TEST"))

    def test_optimizer_can_freeze_both_boolean_dimensions_to_their_origin_values(self):
        optimizer = AdaptiveStrategyOptimizer(
            AdaptiveStrategyConfig(
                3, 5,
                optimize_trailing_stop=False,
                optimize_global_timeframe_validation=False,
            ),
            HistoryStrategy(
                supports_global_timeframe_validation=True,
                use_global_timeframe_validation=True,
            ),
            TradeCloser(trailing_stop_enabled=True),
        )
        self.assertEqual(
            [
                (
                    p.lookback_bars,
                    p.trailing_stop_enabled,
                    p.use_global_timeframe_validation,
                )
                for p in optimizer.candidates
            ],
            [(3, True, True), (4, True, True)],
        )

    def test_adaptive_switch_keeps_the_origin_timeframe_for_the_selected_symbol(self):
        config = AdaptiveStrategyConfig(
            3, 4, accuracy_window=50, accuracy_drop_percent=100,
            min_accuracy_percent=60, timeframe_min_minutes=5,
            timeframe_max_minutes=16, timeframe_step_minutes=5,
        )
        strategy = HistoryStrategy()
        optimizer = AdaptiveStrategyOptimizer(config, strategy, TradeCloser())
        state = optimizer.state_for("EURUSD")
        target_index = next(
            index for index, candidate in enumerate(optimizer.candidates)
            if not candidate.trailing_stop_enabled
        )
        with patch.object(optimizer, "_selection_indices", return_value=[target_index]):
            self.close(optimizer, -1, symbol="EURUSD")
        self.assertEqual(state.current_parameters.timeframe_minutes, 10)
        self.assertEqual(strategy.get_analysis_interval("EURUSD"), timedelta(minutes=10))
        self.assertEqual(strategy.get_analysis_interval("USDCHF"), timedelta(minutes=10))
        prepared = order()
        prepared.symbol = "EURUSD"
        optimizer.prepare_order(prepared)
        self.assertEqual(prepared.strategy_timeframe_minutes, 10)

    def test_each_symbol_starts_with_the_full_candidate_interval(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8)
        eur = optimizer.state_for("EURUSD")
        chf = optimizer.state_for("USDCHF")
        for state in (eur, chf):
            self.assertEqual(state.search_lower_index, 0)
            self.assertEqual(state.search_upper_index, len(optimizer.candidates) - 1)
            self.assertIsNone(state.winner_index)

    def test_risk_scaler_maps_accuracy_linearly_between_its_bounds(self):
        scaler = AccuracyRiskScaler(0.01, 2.01)
        self.assertEqual(scaler.for_accuracy(None, 0.6), 0.01)
        self.assertEqual(scaler.for_accuracy(0.0, 0.6), 0.01)
        self.assertAlmostEqual(scaler.for_accuracy(0.3, 0.6), 1.01)
        self.assertEqual(scaler.for_accuracy(0.6, 0.6), 2.01)
        self.assertEqual(scaler.for_accuracy(0.8, 0.6), 2.01)
        with self.assertRaises(ValueError):
            AccuracyRiskScaler(0.02, 0.01)

    def test_risk_optimizer_can_be_disabled_without_affecting_strategy_switches(self):
        config = AdaptiveStrategyConfig(
            3, 6,
            accuracy_window=4,
            accuracy_drop_percent=5,
            min_accuracy_percent=0,
            min_risk_percent=5.0,
            optimize_risk=False,
        )
        optimizer = AdaptiveStrategyOptimizer(
            config, HistoryStrategy(), TradeCloser(), max_risk_percent=2.0,
        )
        state = optimizer.state_for("TEST")
        self.assertEqual(optimizer.risk_percent_for("TEST"), 2.0)
        self.close(optimizer, 1)
        self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 1)
        self.assertEqual(optimizer.risk_percent_for("TEST"), 2.0)
        self.assertIsNone(state.maximum_accuracy)

    def test_risk_ceiling_starts_at_target_then_tracks_symbols_best_accuracy(self):
        config = AdaptiveStrategyConfig(3, 6, accuracy_window=4,
                                        accuracy_drop_percent=100,
                                        min_accuracy_percent=60,
                                        min_risk_percent=0.01)
        optimizer = AdaptiveStrategyOptimizer(
            config, HistoryStrategy(), TradeCloser(), max_risk_percent=1.01,
        )
        state = optimizer.state_for("EURUSD")
        state.current_parameters = optimizer.candidates[0]
        state.search_lower_index = 0
        state.search_upper_index = 0
        self.assertEqual(state.risk_accuracy_ceiling, 0.6)
        self.assertEqual(optimizer.risk_percent_for("EURUSD"), 0.01)
        self.assertEqual(optimizer.risk_percent_for("USDCHF"), 0.01)
        self.close(optimizer, -1, symbol="EURUSD")
        self.close(optimizer, 1, symbol="EURUSD")
        self.assertIsNone(state.maximum_accuracy)
        self.assertAlmostEqual(optimizer.risk_percent_for("EURUSD"), 0.01 + 0.5 / 0.6)
        self.close(optimizer, 1, symbol="EURUSD")
        self.assertAlmostEqual(state.maximum_accuracy, 2 / 3)
        self.assertAlmostEqual(state.risk_accuracy_ceiling, 2 / 3)
        self.assertEqual(optimizer.risk_percent_for("EURUSD"), 1.01)
        self.assertEqual(optimizer.risk_percent_for("USDCHF"), 0.01)

    def test_configuration_switch_resets_symbol_risk_to_minimum(self):
        config = AdaptiveStrategyConfig(3, 6, accuracy_window=50,
                                        accuracy_drop_percent=5,
                                        min_accuracy_percent=0,
                                        min_risk_percent=0.02)
        optimizer = AdaptiveStrategyOptimizer(
            config, HistoryStrategy(), TradeCloser(), max_risk_percent=2.0,
        )
        self.close(optimizer, 1)
        self.assertEqual(optimizer.risk_percent_for("TEST"), 2.0)
        self.assertEqual(optimizer.state_for("TEST").maximum_accuracy, 1.0)
        self.close(optimizer, -1)
        self.assertEqual(optimizer.state_for("TEST").configuration_id, 1)
        self.assertEqual(optimizer.risk_percent_for("TEST"), 0.02)
        self.assertEqual(optimizer.state_for("TEST").maximum_accuracy, 1.0)
        self.assertEqual(optimizer.state_for("TEST").risk_accuracy_ceiling, 1.0)

    def test_risk_declines_against_preserved_symbol_record(self):
        config = AdaptiveStrategyConfig(3, 6, accuracy_window=4,
                                        accuracy_drop_percent=100,
                                        min_accuracy_percent=40,
                                        min_risk_percent=0.01)
        optimizer = AdaptiveStrategyOptimizer(
            config, HistoryStrategy(), TradeCloser(), max_risk_percent=1.01,
        )
        state = optimizer.state_for("TEST")
        state.current_parameters = optimizer.candidates[0]
        state.search_lower_index = 0
        state.search_upper_index = 0
        for pnl in (-1, 1, 1):
            self.close(optimizer, pnl)
        self.assertAlmostEqual(state.maximum_accuracy, 2 / 3)
        self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 0)
        self.assertAlmostEqual(state.last_accuracy, 0.5)
        self.assertAlmostEqual(state.risk_accuracy_ceiling, 2 / 3)
        self.assertAlmostEqual(state.current_risk_percent, 0.76)

    def test_wins_tighten_long_arm_by_two_and_short_arm_by_one(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8,
                                   accuracy_window=50, accuracy_drop_percent=100,
                                   search_long_arm_delta=2, search_short_arm_delta=1)
        state = optimizer.state_for("TEST")
        # Index 2 has two candidates on its left and seven on its right.
        state.current_parameters = optimizer.candidates[2]
        expected_intervals = ((1, 7), (2, 5), (2, 3), (2, 2), (2, 2))
        for expected in expected_intervals:
            self.close(optimizer, 1)
            self.assertEqual((state.search_lower_index, state.search_upper_index), expected)
            self.assertEqual(state.winner_index, 2)

    def test_configured_arm_deltas_control_tightening_and_loosening(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=9,
                                   accuracy_window=50, accuracy_drop_percent=100,
                                   search_long_arm_delta=3, search_short_arm_delta=2)
        state = optimizer.state_for("TEST")
        state.current_parameters = optimizer.candidates[2]
        self.close(optimizer, 1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (2, 8))
        self.close(optimizer, -1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (0, 11))

    def test_wins_can_keep_the_full_interval_when_tightening_is_disabled(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8,
                                   accuracy_window=50, accuracy_drop_percent=100,
                                   tighten_search_interval=False,
                                   search_long_arm_delta=2, search_short_arm_delta=1)
        state = optimizer.state_for("TEST")
        state.current_parameters = optimizer.candidates[2]
        full_interval = (0, len(optimizer.candidates) - 1)
        for _ in range(4):
            self.close(optimizer, 1)
            self.assertEqual((state.search_lower_index, state.search_upper_index), full_interval)
            self.assertEqual(state.winner_index, 2)

    def test_losses_gradually_restore_original_bounds_around_winner(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8,
                                   accuracy_window=50, accuracy_drop_percent=100,
                                   search_long_arm_delta=2, search_short_arm_delta=1)
        state = optimizer.state_for("TEST")
        state.current_parameters = optimizer.candidates[2]
        for _ in range(4):
            self.close(optimizer, 1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (2, 2))
        for expected in ((1, 4), (0, 6), (0, 8), (0, 9), (0, 9)):
            self.close(optimizer, -1)
            self.assertEqual((state.search_lower_index, state.search_upper_index), expected)

    def test_loss_trigger_selects_only_inside_newly_loosened_interval(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=6,
                                   accuracy_window=50, accuracy_drop_percent=5, seed=13,
                                   search_long_arm_delta=2, search_short_arm_delta=1)
        state = optimizer.state_for("TEST")
        state.current_parameters = optimizer.candidates[4]
        for _ in range(4):
            self.close(optimizer, 1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (4, 4))
        self.close(optimizer, -1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (2, 5))
        self.assertIn(state.current_parameters, optimizer.candidates[2:6])
        self.assertNotEqual(state.current_parameters, optimizer.candidates[4])

    def test_a_win_under_a_new_pair_recenters_the_interval_on_that_pair(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8,
                                   accuracy_window=50, accuracy_drop_percent=100,
                                   search_long_arm_delta=2, search_short_arm_delta=1)
        state = optimizer.state_for("TEST")
        state.current_parameters = optimizer.candidates[2]
        self.close(optimizer, 1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (1, 7))
        state.current_parameters = optimizer.candidates[6]
        self.close(optimizer, 1)
        self.assertEqual(state.winner_index, 6)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (3, 6))

    def test_old_configuration_close_does_not_change_search_interval(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8)
        state = optimizer.state_for("TEST")
        state.configuration_id = 1
        before = (state.search_lower_index, state.search_upper_index, state.winner_index)
        self.close(optimizer, 1, configuration_id=0)
        self.assertEqual((state.search_lower_index, state.search_upper_index, state.winner_index), before)

    def test_symbols_tighten_and_loosen_their_intervals_independently(self):
        optimizer = self.optimizer(lookback_min=3, lookback_max=8,
                                   accuracy_window=50, accuracy_drop_percent=100,
                                   search_long_arm_delta=2, search_short_arm_delta=1)
        eur = optimizer.state_for("EURUSD")
        chf = optimizer.state_for("USDCHF")
        eur.current_parameters = optimizer.candidates[2]
        chf.current_parameters = optimizer.candidates[7]
        self.close(optimizer, 1, symbol="EURUSD")
        self.close(optimizer, 1, symbol="EURUSD")
        self.close(optimizer, 1, symbol="USDCHF")
        self.assertEqual((eur.search_lower_index, eur.search_upper_index, eur.winner_index), (2, 5, 2))
        self.assertEqual((chf.search_lower_index, chf.search_upper_index, chf.winner_index), (2, 8, 7))
        self.close(optimizer, -1, symbol="EURUSD")
        self.assertEqual((eur.search_lower_index, eur.search_upper_index), (1, 7))
        self.assertEqual((chf.search_lower_index, chf.search_upper_index), (2, 8))

    def test_origin_outside_grid_keeps_full_interval_until_a_candidate_wins(self):
        optimizer = self.optimizer(lookback_min=10, lookback_max=13,
                                   accuracy_window=50, accuracy_drop_percent=5)
        state = optimizer.state_for("TEST")
        self.close(optimizer, 1)
        self.assertEqual((state.search_lower_index, state.search_upper_index), (0, 5))
        self.assertIsNone(state.winner_index)
        self.close(optimizer, -1)
        self.assertIn(state.current_parameters, optimizer.candidates)
        self.close(optimizer, 1)
        self.assertIsNotNone(state.winner_index)
        self.assertNotEqual((state.search_lower_index, state.search_upper_index), (0, 5))

    def test_invalid_configuration_is_rejected(self):
        for fields in (dict(lookback_min=1), dict(lookback_max=3), dict(lookback_step=0),
                       dict(accuracy_window=0), dict(accuracy_window=-1), dict(accuracy_drop_percent=0),
                       dict(accuracy_drop_percent=float("nan")), dict(accuracy_drop_percent=101),
                       dict(profit_factor_window=0), dict(profit_factor_window=-1),
                       dict(profit_factor_drop_percent=0),
                       dict(profit_factor_drop_percent=float("nan")),
                       dict(profit_factor_drop_percent=101, profit_factor_drop_mode="relative"),
                       dict(profit_factor_drop_mode="unknown"), dict(min_profit_factor=-1),
                       dict(min_profit_factor=float("nan")),
                       dict(optimization_metric="sharpe"),
                       dict(accuracy_drop_mode="unknown"), dict(min_accuracy_percent=-1),
                       dict(min_accuracy_percent=101), dict(min_accuracy_percent=float("nan")),
                       dict(search_long_arm_delta=0), dict(search_short_arm_delta=0),
                       dict(search_long_arm_delta=1, search_short_arm_delta=2),
                       dict(min_risk_percent=0), dict(min_risk_percent=float("nan")),
                       dict(timeframe_min_minutes=0), dict(timeframe_max_minutes=9),
                       dict(timeframe_min_minutes=10, timeframe_max_minutes=10),
                       dict(timeframe_step_minutes=0)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.optimizer(**fields)

    def test_origin_is_preserved_even_outside_candidate_bounds(self):
        optimizer = self.optimizer(lookback_min=10, lookback_max=12)
        state = optimizer.state_for("TEST")
        self.assertEqual(state.current_parameters.lookback_bars, 3)
        self.assertFalse(state.current_parameters.trailing_stop_enabled)
        self.assertEqual(optimizer.required_history_capacity, 11)
        for pnl in (1, 1, 1, 1, -1):
            self.close(optimizer, pnl)
        self.assertIn(state.current_parameters, optimizer.candidates)

    def test_accuracy_drop_switches_before_measurement_window_is_full(self):
        optimizer = self.optimizer(accuracy_window=50, accuracy_drop_percent=5)
        state = optimizer.state_for("TEST")
        self.assertIsNone(state.last_accuracy)
        self.close(optimizer, 1)
        self.assertEqual(state.configuration_id, 0)
        self.assertEqual(state.last_accuracy, 1.0)
        self.assertEqual(state.reference_accuracy, 1.0)
        self.assertEqual(len(state.accuracy.outcomes), 1)
        self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 1)
        self.assertIsNone(state.last_accuracy)
        self.assertFalse(state.accuracy.outcomes)
        # The next pair is evaluated from its own first close as well.
        self.close(optimizer, 1)
        self.assertEqual(state.last_accuracy, 1.0)
        self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 2)

    def test_relative_five_percent_triggers_at_80_to_76(self):
        optimizer = self.optimizer(accuracy_window=25, accuracy_drop_percent=5)
        state = optimizer.state_for("TEST")
        for pnl in [-1] * 5 + [1] * 20:
            self.close(optimizer, pnl)
        self.assertAlmostEqual(state.reference_accuracy, 0.8)
        old = state.current_parameters
        # Replace the five oldest losses first; the sixth loss removes a win.
        for _ in range(5):
            self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 0)
        self.assertAlmostEqual(state.last_accuracy, 0.8)
        self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 1)
        self.assertNotEqual(state.current_parameters, old)
        self.assertEqual(optimizer.strategy.get_history_lookback_bars("TEST"), state.current_parameters.lookback_bars)
        new = order()
        optimizer.prepare_order(new)
        self.assertEqual(new.trailing_stop_enabled, state.current_parameters.trailing_stop_enabled)
        self.assertIsNone(state.reference_accuracy)
        self.assertFalse(state.accuracy.outcomes)

    def test_percentage_points_do_not_trigger_on_80_to_76(self):
        optimizer = self.optimizer(accuracy_window=25, accuracy_drop_percent=5, accuracy_drop_mode="points")
        state = optimizer.state_for("TEST")
        for pnl in [-1] * 5 + [1] * 20 + [-1] * 6:
            self.close(optimizer, pnl)
        self.assertEqual(state.configuration_id, 0)
        self.assertAlmostEqual(state.last_accuracy, 0.76)
        self.close(optimizer, -1)
        self.assertEqual(state.configuration_id, 1)

    def test_gradual_decline_is_compared_with_peak_not_previous_close(self):
        optimizer = self.optimizer(accuracy_window=50, accuracy_drop_percent=5)
        for _ in range(50):
            self.close(optimizer, 1)
        self.close(optimizer, -1)
        self.close(optimizer, -1)
        self.assertEqual(optimizer.switch_count, 0)
        self.close(optimizer, -1)
        self.assertEqual(optimizer.switch_count, 1)

    def test_improving_accuracy_raises_reference(self):
        optimizer = self.optimizer()
        state = optimizer.state_for("TEST")
        for pnl in (-1, 1, 1, 1):
            self.close(optimizer, pnl)
        self.assertEqual(state.reference_accuracy, 0.75)
        self.close(optimizer, 1)
        self.assertEqual(state.reference_accuracy, 1.0)
        self.close(optimizer, -1)
        self.assertEqual(optimizer.switch_count, 1)

    def test_old_configuration_closes_cannot_rate_new_configuration(self):
        optimizer = self.optimizer()
        state = optimizer.state_for("TEST")
        for pnl in (1, 1, 1, 1, -1):
            self.close(optimizer, pnl)
        for _ in range(20):
            self.close(optimizer, -1, configuration_id=0)
        self.assertEqual(state.configuration_id, 1)
        self.assertEqual(optimizer.ignored_closes, 20)
        self.assertFalse(state.accuracy.outcomes)
        self.close(optimizer, 1)
        self.assertEqual(len(state.accuracy.outcomes), 1)

    def test_random_sequence_is_reproducible_and_always_changes_pair(self):
        sequences = []
        for _ in range(2):
            optimizer = self.optimizer(seed=51)
            state = optimizer.state_for("TEST")
            sequence = [state.current_parameters]
            for _ in range(12):
                for pnl in (1, 1, 1, 1, -1):
                    self.close(optimizer, pnl)
                self.assertNotEqual(state.current_parameters, sequence[-1])
                sequence.append(state.current_parameters)
            sequences.append(sequence)
        self.assertEqual(sequences[0], sequences[1])

    def test_single_lookback_still_explores_both_trailing_modes(self):
        optimizer = self.optimizer(lookback_max=4)
        for pnl in (1, 1, 1, 1, -1):
            self.close(optimizer, pnl)
        self.assertEqual(optimizer.state_for("TEST").current_parameters.lookback_bars, 3)
        self.assertTrue(optimizer.state_for("TEST").current_parameters.trailing_stop_enabled)

    def test_zero_accuracy_does_not_divide_by_zero_or_invent_decline(self):
        optimizer = self.optimizer()
        for _ in range(10):
            self.close(optimizer, -1)
        self.assertEqual(optimizer.state_for("TEST").reference_accuracy, 0.0)
        self.assertEqual(optimizer.switch_count, 0)

    def test_minimum_accuracy_retries_from_each_new_pairs_first_defined_rate(self):
        for outcomes in ((1, -1), (-1,)):
            with self.subTest(outcomes=outcomes):
                optimizer = self.optimizer(min_accuracy_percent=60, accuracy_window=50,
                                           accuracy_drop_percent=100)
                state = optimizer.state_for("TEST")
                for period in range(3):
                    for pnl in outcomes[:-1]:
                        self.close(optimizer, pnl)
                    self.assertEqual(state.configuration_id, period)
                    self.close(optimizer, outcomes[-1])
                    self.assertEqual(state.configuration_id, period + 1)
                    self.assertFalse(state.accuracy.outcomes)
                self.assertEqual(state.switch_count, 3)

    def test_minimum_accuracy_requires_strict_excess_and_zero_disables_it(self):
        optimizer = self.optimizer(min_accuracy_percent=75, accuracy_drop_percent=100)
        for pnl in (1, 1, 1, -1):
            self.close(optimizer, pnl)
        self.assertEqual(optimizer.switch_count, 1)
        self.assertIsNone(optimizer.state_for("TEST").last_accuracy)
        disabled = self.optimizer(accuracy_drop_percent=100)
        for pnl in (1, -1, 1, -1):
            self.close(disabled, pnl)
        self.assertEqual(disabled.switch_count, 0)

    def test_minimum_accuracy_triggers_on_first_close_and_logs_partial_sample(self):
        optimizer = self.optimizer(min_accuracy_percent=80, accuracy_window=50)
        output = io.StringIO()
        with redirect_stdout(output):
            optimizer.on_trade_closed(report(-1))
        self.assertEqual(optimizer.state_for("TEST").switch_count, 1)
        self.assertIn("reason=accuracy_below_minimum", output.getvalue())
        self.assertIn("drop_percent=0.000000", output.getvalue())
        self.assertIn("sample=1/50", output.getvalue())

    def test_symbol_precision_configuration_and_order_modes_are_isolated(self):
        optimizer = self.optimizer(min_accuracy_percent=60, lookback_max=4)
        for _ in range(4):
            self.close(optimizer, 1, symbol="EURUSD")
        self.close(optimizer, -1, symbol="USDCHF")
        eur = optimizer.state_for("EURUSD")
        chf = optimizer.state_for("USDCHF")
        self.assertEqual(eur.configuration_id, 0)
        self.assertEqual(eur.last_accuracy, 1)
        self.assertEqual(chf.configuration_id, 1)
        self.assertTrue(chf.current_parameters.trailing_stop_enabled)
        self.assertFalse(eur.current_parameters.trailing_stop_enabled)
        eur_order = replace(order(), symbol="EURUSD")
        chf_order = replace(order(2), symbol="USDCHF")
        for request in (eur_order, chf_order):
            optimizer.prepare_order(request)
            optimizer.trade_closer.prepare_order(request)
        self.assertEqual(eur_order.strategy_configuration_id, 0)
        self.assertEqual(eur_order.take_profit, 106)
        self.assertEqual(chf_order.strategy_configuration_id, 1)
        self.assertIsNone(chf_order.take_profit)
        self.assertEqual(chf_order.trailing_stop_activation_price, 103)
        # An epoch of zero is still current for EURUSD, but outdated for USDCHF.
        self.close(optimizer, 1, configuration_id=0, symbol="EURUSD")
        self.close(optimizer, -1, configuration_id=0, symbol="USDCHF")
        self.assertEqual(eur.ignored_closes, 0)
        self.assertEqual(chf.ignored_closes, 1)
        self.assertFalse(chf.accuracy.outcomes)

    def test_each_symbol_is_evaluated_from_its_own_first_close(self):
        optimizer = self.optimizer(min_accuracy_percent=90, accuracy_window=50)
        self.close(optimizer, 1, symbol="EURUSD")
        eur = optimizer.state_for("EURUSD")
        chf = optimizer.state_for("USDCHF")
        self.assertEqual(eur.last_accuracy, 1.0)
        self.assertIsNone(chf.last_accuracy)
        self.assertFalse(chf.accuracy.outcomes)
        self.close(optimizer, -1, symbol="USDCHF")
        self.assertEqual(chf.switch_count, 1)
        self.assertEqual(eur.switch_count, 0)
        self.assertEqual(len(eur.accuracy.outcomes), 1)
        self.close(optimizer, -1, symbol="EURUSD")
        self.assertEqual(eur.switch_count, 1)
        self.assertEqual(chf.switch_count, 1)
        self.assertFalse(chf.accuracy.outcomes)

    def test_symbol_random_choices_do_not_depend_on_other_symbol_activity(self):
        isolated = self.optimizer(seed=19)
        mixed = self.optimizer(seed=19)
        for _ in range(8):
            for pnl in (1, 1, 1, 1, -1):
                self.close(isolated, pnl, symbol="EURUSD")
                self.close(mixed, pnl, symbol="EURUSD")
                self.close(mixed, pnl, symbol="USDCHF")
            self.assertEqual(isolated.state_for("EURUSD").current_parameters,
                             mixed.state_for("EURUSD").current_parameters)
        self.assertEqual(mixed.state_for("USDCHF").switch_count, 8)

    def test_breakeven_is_not_a_win_and_accuracy_uses_realized_pnl(self):
        accuracy = RollingTradeAccuracy(4)
        for pnl, expected in ((1, 1.0), (0, 0.5), (-1, 1 / 3)):
            self.assertAlmostEqual(accuracy.add(pnl), expected)
        self.assertEqual(accuracy.add(1), 0.5)
        self.assertEqual(accuracy.add(0), 0.25)
        with self.assertRaises(ValueError):
            accuracy.add(float("nan"))
        accuracy.clear()
        self.assertEqual(accuracy.add(0), 0.0)

    def test_profit_factor_uses_realized_pnl_window(self):
        profit_factor = RollingTradeProfitFactor(3)
        self.assertEqual(profit_factor.add(-1), 0.0)
        self.assertEqual(profit_factor.add(4), 4.0)
        self.assertEqual(profit_factor.add(2), 6.0)
        self.assertEqual(profit_factor.add(-2), 3.0)
        self.assertEqual(profit_factor.add(1), 1.5)
        with self.assertRaises(ValueError):
            profit_factor.add(float("nan"))
        profit_factor.clear()
        self.assertEqual(profit_factor.add(0), 0.0)

    def test_profit_factor_can_drive_switches_even_when_accuracy_is_flat(self):
        optimizer = self.optimizer(
            accuracy_window=7,
            accuracy_drop_percent=100,
            profit_factor_window=2,
            profit_factor_drop_percent=50,
            optimization_metric="profit_factor",
            min_accuracy_percent=0,
            min_profit_factor=0,
        )
        state = optimizer.state_for("TEST")
        for pnl in (-1, 4, -1):
            self.close(optimizer, pnl)
        self.assertEqual(state.configuration_id, 0)
        self.assertAlmostEqual(state.last_accuracy, 1 / 3)
        self.assertEqual(state.last_metric_value, 4.0)
        output = io.StringIO()
        with redirect_stdout(output):
            optimizer.on_trade_closed(report(1, state.configuration_id))
        self.assertEqual(state.configuration_id, 1)
        self.assertIn("reason=profit_factor_drop", output.getvalue())
        self.assertIn("optimization_metric=profit_factor", output.getvalue())
        self.assertIn("sample=2/2", output.getvalue())

    def test_profit_factor_has_its_own_rolling_window(self):
        optimizer = self.optimizer(
            accuracy_window=4,
            profit_factor_window=2,
            profit_factor_drop_percent=100,
            optimization_metric="profit_factor",
            min_profit_factor=0,
        )
        state = optimizer.state_for("TEST")
        with patch.object(optimizer, "_selection_indices", return_value=[]):
            for pnl in (-1, 4, 2, -2):
                self.close(optimizer, pnl)
        self.assertEqual(len(state.accuracy.outcomes), 4)
        self.assertEqual(list(state.profit_factor.pnl_samples), [2, -2])
        self.assertEqual(state.profit_factor.value, 1.0)

    def test_profit_factor_is_isolated_per_symbol_instead_of_using_global_results(self):
        optimizer = self.optimizer(
            profit_factor_window=4,
            profit_factor_drop_percent=100,
            optimization_metric="profit_factor",
            min_profit_factor=0,
        )
        with patch.object(optimizer, "_selection_indices", return_value=[]):
            for pnl in (-1, 4):
                self.close(optimizer, pnl, symbol="EURUSD")
            for pnl in (-10, 1):
                self.close(optimizer, pnl, symbol="USDCHF")
        self.assertEqual(optimizer.state_for("EURUSD").profit_factor.value, 4.0)
        self.assertEqual(optimizer.state_for("USDCHF").profit_factor.value, 0.1)

    def test_minimum_profit_factor_is_an_independent_search_trigger(self):
        optimizer = self.optimizer(
            accuracy_drop_percent=100,
            profit_factor_window=10,
            profit_factor_drop_percent=100,
            optimization_metric="profit_factor",
            min_accuracy_percent=0,
            min_profit_factor=1.0,
        )
        state = optimizer.state_for("TEST")
        self.close(optimizer, 2)
        self.assertEqual(state.configuration_id, 0)
        output = io.StringIO()
        with redirect_stdout(output):
            optimizer.on_trade_closed(report(-2, state.configuration_id))
        self.assertEqual(state.configuration_id, 1)
        self.assertIn("profit_factor_below_minimum", output.getvalue())

    def test_switch_updates_symbol_global_validation_when_strategy_supports_it(self):
        strategy = HistoryStrategy(
            supports_global_timeframe_validation=True,
            use_global_timeframe_validation=True,
        )
        optimizer = self.optimizer(
            strategy=strategy,
            lookback_min=3,
            lookback_max=4,
            accuracy_drop_percent=100,
            min_accuracy_percent=60,
        )
        target_index = next(
            index for index, candidate in enumerate(optimizer.candidates)
            if not candidate.use_global_timeframe_validation and not candidate.trailing_stop_enabled
        )
        with patch.object(optimizer, "_selection_indices", return_value=[target_index]):
            self.close(optimizer, -1, symbol="EURUSD")
        state = optimizer.state_for("EURUSD")
        self.assertFalse(state.current_parameters.use_global_timeframe_validation)
        self.assertFalse(strategy.get_global_timeframe_validation_enabled("EURUSD"))

    def test_one_trade_accuracy_window_is_supported_and_discards_previous_outcome(self):
        accuracy = RollingTradeAccuracy(1)
        self.assertEqual(accuracy.add(1), 1.0)
        self.assertEqual(accuracy.add(-1), 0.0)
        self.assertEqual(list(accuracy.outcomes), [False])
        optimizer = self.optimizer(accuracy_window=1)
        self.close(optimizer, 1)
        self.close(optimizer, -1)
        self.assertEqual(optimizer.switch_count, 1)
        with self.assertRaises(ValueError):
            RollingTradeAccuracy(0)

    def test_trade_collector_notifies_after_recording_without_duplicate_listener(self):
        collector = ReportCollector()
        seen = []

        def listener(trade):
            seen.append((trade, len(collector.trades)))

        collector.add_trade_listener(listener)
        collector.add_trade_listener(listener)
        trade = report(1)
        collector.add_trade(trade)
        self.assertEqual(seen, [(trade, 1)])


class AdaptiveIntegrationTests(unittest.TestCase):
    def args(self, *extra):
        options = ["run_backtest", "--strategy", "swing-regression", "--balance", "1000",
                   "--candle-interval", "10m", "--candle-lookback", "17",
                   "--minimum-candles", "1", "--limit-spread-ratio", "1", "--max-spread-ratio", "1",
                   "--adaptive-strategy", "--adaptive-lookback-min", "3", "--adaptive-lookback-max", "4",
                   "--adaptive-accuracy-window", "2"]
        with patch.object(sys, "argv", options + list(extra)):
            return parse_args()

    def test_batcher_uses_each_symbols_current_timeframe(self):
        intervals = {"EURUSD": timedelta(minutes=5), "USDCHF": timedelta(minutes=15)}
        batcher = TickBatcher(
            batch_size=100,
            analysis_interval=timedelta(minutes=10),
            analysis_interval_provider=lambda symbol: intervals[symbol],
        )
        self.assertIsNone(batcher.push(Tick(START, "EURUSD", 1, 1.1, 1, 1)))
        self.assertIsNone(batcher.push(Tick(START, "USDCHF", 1, 1.1, 1, 1)))
        eur_batch = batcher.push(Tick(START + timedelta(minutes=5), "EURUSD", 1, 1.1, 1, 1))
        self.assertIsNotNone(eur_batch)
        self.assertIsNone(batcher.push(Tick(START + timedelta(minutes=5), "USDCHF", 1, 1.1, 1, 1)))
        chf_batch = batcher.push(Tick(START + timedelta(minutes=15), "USDCHF", 1, 1.1, 1, 1))
        self.assertIsNotNone(chf_batch)

    def test_engine_keeps_only_the_origin_timeframe_registered(self):
        engine = build_engine([], self.args(
            "--adaptive-timeframe-min-minutes", "5",
            "--adaptive-timeframe-max-minutes", "16",
            "--adaptive-timeframe-step-minutes", "5",
        ))
        self.assertEqual(
            {int(interval.total_seconds() // 60) for interval in engine.market_state._candle_builders_by_interval},
            {10},
        )
        self.assertIsNotNone(engine.batcher.analysis_interval_provider)

    def test_strategy_keeps_each_symbol_on_the_origin_timeframe(self):
        engine = build_engine([], self.args(
            "--level-lookback-bars", "3",
            "--adaptive-timeframe-min-minutes", "5",
            "--adaptive-timeframe-max-minutes", "16",
            "--adaptive-timeframe-step-minutes", "5",
            "--regression-global-timeframe-multiplier", "1",
            "--regression-min-market-volatility-ratio", "0",
        ))
        strategy = engine.strategy
        for minute in range(61):
            for symbol in ("EURUSD", "USDCHF"):
                engine.market_state.update(Tick(
                    START + timedelta(minutes=minute), symbol, 100, 100.2, 1, 1,
                ))
        context = StrategyContext(engine.portfolio, engine.market_state)
        eur = strategy._get_structure_candles("EURUSD", context)
        chf = strategy._get_structure_candles("USDCHF", context)
        self.assertEqual(
            [candle.start_time for candle in eur["bid"]],
            [START + timedelta(minutes=minute) for minute in (30, 40, 50)],
        )
        self.assertEqual(
            [candle.start_time for candle in chf["bid"]],
            [START + timedelta(minutes=minute) for minute in (30, 40, 50)],
        )
        self.assertEqual(strategy.analysis_interval, timedelta(minutes=10))
        self.assertEqual(strategy.get_analysis_interval("EURUSD"), timedelta(minutes=10))

    def test_cli_rejects_invalid_ranges_and_unsupported_strategy(self):
        options = (["--adaptive-strategy", "--adaptive-lookback-min", "-1"],
                   ["--adaptive-strategy", "--adaptive-lookback-max", "60"],
                   ["--adaptive-strategy", "--adaptive-lookback-min", "5", "--adaptive-lookback-max", "4"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-min-risk-percent", "0"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--risk-percent", "0.005"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-profit-factor-window", "0"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-profit-factor-drop-percent", "0"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-min-profit-factor", "-1"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-timeframe-min-minutes", "0"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-timeframe-min-minutes", "10", "--adaptive-timeframe-max-minutes", "10"],
                   ["--strategy", "swing-regression", "--adaptive-strategy", "--adaptive-timeframe-step-minutes", "0"],
                   ["--strategy", "structure", "--adaptive-strategy", "--adaptive-lookback-min", "2", "--adaptive-lookback-max", "4"],
                   ["--strategy", "dummy", "--adaptive-strategy", "--adaptive-lookback-min", "2", "--adaptive-lookback-max", "4"])
        for flags in options:
            with self.subTest(flags=flags), patch.object(sys, "argv", ["run_backtest"] + flags):
                with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()

    def test_cli_enables_optimizer_with_only_activation_flag_for_both_strategies(self):
        for strategy, extra_options, expected_config in (
            ("structure", ["--adaptive-lookback-min", "60", "--adaptive-lookback-max", "61"],
             replace(AdaptiveStrategyConfig(), lookback_min=60, lookback_max=61)),
            ("swing-regression", [], AdaptiveStrategyConfig()),
        ):
            with self.subTest(strategy=strategy), patch.object(
                sys, "argv", ["run_backtest", "--strategy", strategy, "--adaptive-strategy", *extra_options],
            ):
                args = parse_args()
                engine = build_engine([], args)
                self.assertEqual(engine.adaptive_optimizer.config, expected_config)
                trailing_values, global_values = resolve_backtest_adaptive_choice_values(
                    args,
                    expected_config,
                    resolve_backtest_regression_rules(args) if strategy == "swing-regression" else None,
                )
                self.assertEqual(
                    engine.adaptive_optimizer.candidates,
                    expected_config.candidates(
                        timeframe_minutes=int(DEFAULT_CANDLE_INTERVAL.total_seconds() // 60),
                        trailing_stop_values=trailing_values,
                        global_timeframe_validation_values=global_values,
                    ),
                )

    def test_cli_reads_all_default_values_from_one_configuration(self):
        configured = AdaptiveStrategyConfig(64, 145, lookback_step=4, accuracy_window=9,
                                            accuracy_drop_percent=8, accuracy_drop_mode="points",
                                            profit_factor_window=12,
                                            profit_factor_drop_percent=0.4,
                                            profit_factor_drop_mode="points",
                                            min_profit_factor=1.2,
                                            optimization_metric="profit_factor",
                                            seed=11, min_accuracy_percent=55,
                                            search_long_arm_delta=4, search_short_arm_delta=2,
                                            min_risk_percent=0.02,
                                            optimize_risk=False,
                                            optimize_trailing_stop=False,
                                            optimize_global_timeframe_validation=False,
                                            timeframe_min_minutes=5,
                                            timeframe_max_minutes=31,
                                            timeframe_step_minutes=5)
        config = SwingRegressionConfig()
        regression_config = replace(config, optimization=replace(config.optimization, adaptive=configured))
        for strategy, constant, defaults in (
            ("structure", "DEFAULT_ADAPTIVE_CONFIG", configured),
            ("swing-regression", "DEFAULT_SWING_REGRESSION_CONFIG", regression_config),
        ):
            with self.subTest(strategy=strategy), patch("ultimateBot.run_backtest." + constant, defaults), patch.object(
                sys, "argv", ["run_backtest", "--strategy", strategy, "--adaptive-strategy"],
            ):
                engine = build_engine([], parse_args())
            self.assertEqual(engine.adaptive_optimizer.config, configured)

    def test_cli_accepts_individual_bound_overrides_and_keeps_other_defaults(self):
        for option, field, value in (("--adaptive-lookback-min", "lookback_min", 10),
                                     ("--adaptive-lookback-max", "lookback_max", 90),
                                     ("--adaptive-min-accuracy-percent", "min_accuracy_percent", 0),
                                     ("--adaptive-profit-factor-window", "profit_factor_window", 17),
                                     ("--adaptive-profit-factor-drop-percent", "profit_factor_drop_percent", 9),
                                     ("--adaptive-profit-factor-drop-mode", "profit_factor_drop_mode", "points"),
                                     ("--adaptive-min-profit-factor", "min_profit_factor", 1.25),
                                     ("--adaptive-optimization-metric", "optimization_metric", "profit_factor"),
                                     ("--adaptive-search-long-arm-delta", "search_long_arm_delta", 3),
                                     ("--adaptive-search-short-arm-delta", "search_short_arm_delta", 1),
                                     ("--adaptive-min-risk-percent", "min_risk_percent", 0.02),
                                     ("--adaptive-timeframe-min-minutes", "timeframe_min_minutes", 5),
                                     ("--adaptive-timeframe-max-minutes", "timeframe_max_minutes", 31),
                                     ("--adaptive-timeframe-step-minutes", "timeframe_step_minutes", 5)):
            with self.subTest(option=option), patch.object(
                sys, "argv", ["run_backtest", "--strategy", "swing-regression", "--adaptive-strategy", option, str(value)],
            ):
                engine = build_engine([], parse_args())
                self.assertEqual(engine.adaptive_optimizer.config, replace(AdaptiveStrategyConfig(), **{field: value}))

    def test_cli_can_disable_interval_tightening(self):
        with patch.object(
            sys, "argv", [
                "run_backtest", "--strategy", "swing-regression", "--adaptive-strategy",
                "--no-adaptive-tighten-search-interval",
            ],
        ):
            engine = build_engine([], parse_args())
        self.assertEqual(
            engine.adaptive_optimizer.config,
            replace(AdaptiveStrategyConfig(), tighten_search_interval=False),
        )

    def test_cli_can_disable_dynamic_risk_and_use_fixed_risk_percent(self):
        engine = build_engine([], self.args(
            "--risk-percent", "0.005",
            "--no-adaptive-optimize-risk",
        ))
        self.assertFalse(engine.adaptive_optimizer.config.optimize_risk)
        self.assertIsNone(engine.risk_manager.risk_percent_provider)
        self.assertEqual(engine.risk_manager.risk_percent_for_symbol("EURUSD"), 0.005)
        state = engine.adaptive_optimizer.state_for("EURUSD")
        self.assertEqual(state.current_risk_percent, 0.005)
        with redirect_stdout(io.StringIO()):
            engine.adaptive_optimizer.on_trade_closed(report(-1, symbol="EURUSD"))
        self.assertEqual(engine.risk_manager.risk_percent_for_symbol("EURUSD"), 0.005)

    def test_cli_can_remove_trailing_and_global_validation_from_adaptive_grid(self):
        engine = build_engine([], self.args(
            "--trailing-stop",
            "--regression-global-timeframe-validation",
            "--no-adaptive-optimize-trailing-stop",
            "--no-adaptive-optimize-global-timeframe-validation",
        ))
        self.assertEqual(
            engine.adaptive_optimizer.config,
            replace(
                AdaptiveStrategyConfig(),
                lookback_min=3,
                lookback_max=4,
                accuracy_window=2,
                optimize_trailing_stop=False,
                optimize_global_timeframe_validation=False,
            ),
        )
        self.assertEqual(
            {
                (candidate.trailing_stop_enabled, candidate.use_global_timeframe_validation)
                for candidate in engine.adaptive_optimizer.candidates
            },
            {(True, True)},
        )

    def test_cli_can_disable_fixed_global_validation(self):
        engine = build_engine([], self.args("--no-regression-global-timeframe-validation"))
        self.assertFalse(engine.strategy.get_global_timeframe_validation_enabled())

    def test_cli_can_choose_profit_factor_optimization_metric(self):
        engine = build_engine([], self.args("--adaptive-optimization-metric", "profit_factor"))
        self.assertEqual(engine.adaptive_optimizer.config.optimization_metric, "profit_factor")

    def test_optimizer_remains_disabled_by_default_even_with_bound_values(self):
        for options in ([], ["--adaptive-lookback-min", "70"]):
            with self.subTest(options=options), patch.object(sys, "argv", ["run_backtest"] + options):
                engine = build_engine([], parse_args())
                self.assertIsNone(engine.adaptive_optimizer)

    def test_programmatic_configuration_uses_defaults_for_missing_or_none_values(self):
        args = SimpleNamespace(adaptive_strategy=True, adaptive_lookback_min=None,
                               adaptive_lookback_max=None)
        self.assertEqual(resolve_backtest_adaptive_config(args), AdaptiveStrategyConfig())
        args.adaptive_min_accuracy_percent = 0
        self.assertEqual(resolve_backtest_adaptive_config(args), replace(AdaptiveStrategyConfig(), min_accuracy_percent=0))
        args.adaptive_optimize_risk = False
        args.adaptive_optimize_trailing_stop = False
        args.adaptive_optimize_global_timeframe_validation = False
        self.assertEqual(
            resolve_backtest_adaptive_config(args),
            replace(
                AdaptiveStrategyConfig(),
                min_accuracy_percent=0,
                optimize_risk=False,
                optimize_trailing_stop=False,
                optimize_global_timeframe_validation=False,
            ),
        )

    def test_cli_resolves_effective_adaptive_boolean_choice_values(self):
        args = self.args(
            "--trailing-stop",
            "--no-adaptive-optimize-trailing-stop",
            "--no-adaptive-optimize-global-timeframe-validation",
        )
        self.assertEqual(
            resolve_backtest_adaptive_choice_values(
                args,
                resolve_backtest_adaptive_config(args),
                resolve_backtest_regression_rules(args),
            ),
            ((True,), (True,)),
        )
        false_args = self.args(
            "--no-regression-global-timeframe-validation",
            "--no-adaptive-optimize-global-timeframe-validation",
        )
        self.assertEqual(
            resolve_backtest_adaptive_choice_values(
                false_args,
                resolve_backtest_adaptive_config(false_args),
                resolve_backtest_regression_rules(false_args),
            ),
            ((False, True), (False,)),
        )

    def test_history_reserves_maximum_but_initial_active_window_is_unchanged(self):
        engine = build_engine([], self.args("--level-lookback-bars", "5", "--adaptive-lookback-max", "11",
                                            "--regression-global-timeframe-multiplier", "1",
                                            "--regression-min-market-volatility-ratio", "0"))
        self.assertEqual(engine.strategy.history_lookback_bars, 5)
        self.assertEqual(engine.adaptive_optimizer.required_history_capacity, 10)
        self.assertEqual(engine.risk_manager.volatility_config.candle_lookback, 17)
        for builder in engine.market_state._candle_builders.values():
            self.assertEqual(builder.lookback_candles, 10)
        for seconds in range(11):
            engine.market_state.update(Tick(START + timedelta(minutes=10 * seconds), "TEST", 100, 100.2, 1, 1))
        circuits = engine.strategy._get_structure_candles("TEST", StrategyContext(engine.portfolio, engine.market_state))
        self.assertEqual(len(circuits["bid"]), 5)
        engine.strategy.set_history_lookback_bars(10)
        circuits = engine.strategy._get_structure_candles("TEST", StrategyContext(engine.portfolio, engine.market_state))
        self.assertEqual(len(circuits["bid"]), 10)

    def test_engine_wires_each_symbols_adaptive_risk_into_position_sizing(self):
        engine = build_engine([], self.args("--risk-percent", "1", "--adaptive-min-risk-percent", "0.01"))
        self.assertEqual(engine.risk_manager.risk_percent_for_symbol("EURUSD"), 0.01)
        self.assertEqual(engine.risk_manager.risk_percent_for_symbol("USDCHF"), 0.01)
        with redirect_stdout(io.StringIO()):
            engine.adaptive_optimizer.on_trade_closed(report(1, symbol="EURUSD"))
        self.assertEqual(engine.risk_manager.risk_percent_for_symbol("EURUSD"), 1.0)
        self.assertEqual(engine.risk_manager.risk_percent_for_symbol("USDCHF"), 0.01)

    def test_structure_and_regression_implement_common_history_interface(self):
        structure = StructureStrategy({"level_lookback_bars": 60})
        structure.set_history_lookback_bars(80)
        self.assertEqual(structure.structure_rules["level_lookback_bars"], 80)
        self.assertEqual(structure.minimum_candles, 80)
        with self.assertRaises(ValueError):
            structure.set_history_lookback_bars(30)
        regression = SwingRegressionStrategy({"level_lookback_bars": 5,
                                             "regression_global_timeframe_multiplier": 4,
                                             "regression_min_market_volatility_ratio": 1.0,
                                             "regression_market_volatility_window_bars": 15})
        self.assertEqual(regression.history_capacity_for_lookback(20), 60)
        regression.last_analysis["TEST"] = {"old": True}
        regression.last_global_analysis["TEST"] = {"old": True}
        regression._last_global_closed_candle["TEST"] = START
        regression.set_history_lookback_bars(20)
        self.assertIs(regression.regression_rules, regression.structure_rules)
        self.assertEqual(regression.minimum_candles, 60)
        self.assertFalse(regression.last_analysis)
        self.assertEqual(regression.global_history_lookback_bars, 5)
        self.assertEqual(regression.last_global_analysis, {"TEST": {"old": True}})
        self.assertEqual(regression._last_global_closed_candle, {"TEST": START})

    def test_fixed_history_capacity_covers_global_and_volatility_requirements(self):
        for volatility_ratio, expected_capacity in ((0, 20), (1, 60)):
            with self.subTest(volatility_ratio=volatility_ratio):
                engine = build_engine([], self.args(
                    "--level-lookback-bars", "5", "--adaptive-lookback-max", "11",
                    "--regression-global-timeframe-multiplier", "4",
                    "--regression-market-volatility-window-bars", "15",
                    "--regression-min-market-volatility-ratio", str(volatility_ratio),
                ))
                self.assertEqual(engine.adaptive_optimizer.required_history_capacity, expected_capacity)
                for builder in engine.market_state._candle_builders.values():
                    self.assertEqual(builder.lookback_candles, expected_capacity)
                engine.strategy.set_history_lookback_bars(100, symbol="EURUSD")
                self.assertEqual(engine.strategy.history_capacity_for_lookback(100), 100)
                self.assertEqual(engine.strategy.global_history_capacity, expected_capacity)

    def test_cli_parses_minimum_accuracy_percent_and_rejects_invalid_values(self):
        engine = build_engine([], self.args("--adaptive-min-accuracy-percent", "60"),
                              strategy_override=HistoryStrategy())
        self.assertEqual(engine.adaptive_optimizer.config.min_accuracy_percent, 60)
        for invalid in ("-1", "101", "nan", "inf"):
            with self.subTest(value=invalid), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                self.args("--adaptive-min-accuracy-percent", invalid)

    def test_symbol_lookbacks_keep_other_symbol_history_and_regression_caches(self):
        engine = build_engine([], self.args("--level-lookback-bars", "5", "--adaptive-lookback-max", "11",
                                            "--regression-global-timeframe-multiplier", "1",
                                            "--regression-min-market-volatility-ratio", "0"))
        strategy = engine.strategy
        for minutes in range(11):
            for symbol in ("EURUSD", "USDCHF"):
                engine.market_state.update(Tick(START + timedelta(minutes=10 * minutes), symbol, 100, 100.2, 1, 1))
                strategy.last_analysis[symbol] = {"local": symbol}
                strategy.last_global_analysis[symbol] = {"global": symbol}
                strategy._last_global_closed_candle[symbol] = START
        strategy.set_history_lookback_bars(10, symbol="EURUSD")
        self.assertEqual(strategy.history_lookback_bars, 5)
        self.assertEqual(strategy.get_history_lookback_bars("EURUSD"), 10)
        self.assertEqual(strategy.get_history_lookback_bars("USDCHF"), 5)
        context = StrategyContext(engine.portfolio, engine.market_state)
        self.assertEqual(len(strategy._get_structure_candles("EURUSD", context)["bid"]), 10)
        self.assertEqual(len(strategy._get_structure_candles("USDCHF", context)["bid"]), 5)
        self.assertNotIn("EURUSD", strategy.last_analysis)
        self.assertEqual(strategy.last_global_analysis["EURUSD"], {"global": "EURUSD"})
        self.assertEqual(strategy._last_global_closed_candle["EURUSD"], START)
        self.assertEqual(strategy.last_analysis["USDCHF"], {"local": "USDCHF"})
        self.assertEqual(strategy.last_global_analysis["USDCHF"], {"global": "USDCHF"})
        self.assertEqual(strategy._last_global_closed_candle["USDCHF"], START)
        original_buffers = {(source, symbol): builder._closed[symbol]
                            for source, builder in engine.market_state._candle_builders.items()
                            for symbol in ("EURUSD", "USDCHF")}
        strategy.set_history_lookback_bars(3, symbol="EURUSD")
        for minutes in range(11, 16):
            for symbol in ("EURUSD", "USDCHF"):
                engine.market_state.update(Tick(START + timedelta(minutes=10 * minutes), symbol, 100, 100.2, 1, 1))
        # The level window stays at 5 while the regression window shrinks to 3.
        self.assertEqual(len(strategy._get_structure_candles("EURUSD", context)["bid"]), 5)
        self.assertEqual(strategy.get_history_lookback_bars("EURUSD"), 3)
        self.assertEqual(strategy.level_history_lookback_bars, 5)
        strategy.set_history_lookback_bars(10, symbol="EURUSD")
        for symbol, bars in (("EURUSD", 10), ("USDCHF", 5)):
            projected = strategy._get_structure_candles(symbol, context)["bid"]
            self.assertEqual([c.start_time for c in projected],
                             [START + timedelta(minutes=10 * m) for m in range(15 - bars, 15)])
        for source, builder in engine.market_state._candle_builders.items():
            for symbol in ("EURUSD", "USDCHF"):
                self.assertIs(builder._closed[symbol], original_buffers[(source, symbol)])
                self.assertEqual(builder._closed[symbol].maxlen, 10)
                self.assertEqual(len(builder._closed[symbol]), 10)

    def test_decision_engines_receive_symbol_specific_windows_and_rules(self):
        engine = build_engine([], self.args("--level-lookback-bars", "5", "--adaptive-lookback-max", "11",
                                            "--regression-global-timeframe-multiplier", "1",
                                            "--regression-min-market-volatility-ratio", "0"))
        strategy = engine.strategy
        for minutes in range(11):
            for symbol in ("EURUSD", "USDCHF"):
                engine.market_state.update(Tick(START + timedelta(minutes=10 * minutes), symbol, 100, 100.2, 1, 1))
        strategy.set_history_lookback_bars(10, symbol="EURUSD")
        with patch("ultimateBot.backtester.strategy.swing_regression_strategy.decide_swing_regression",
                   return_value=(None, {"anchor": None})) as decider:
            for symbol, bars in (("EURUSD", 10), ("USDCHF", 5)):
                circuits = strategy._get_structure_candles(symbol, StrategyContext(engine.portfolio, engine.market_state))
                mid = engine.market_state.get_candles(symbol, include_active=False)
                strategy._evaluate_regression_pipeline(symbol, circuits["bid"], circuits["ask"], mid,
                                                       0.01, timeframe_label="local", timeframe_multiplier=1,
                                                       require_fresh_swings=False)
                self.assertEqual(decider.call_args.args[3], bars)
                self.assertEqual(decider.call_args.args[4]["regression_lookback_bars"], bars)
                self.assertEqual(decider.call_args.args[4]["level_lookback_bars"], 5)
        structure = StructureStrategy({"level_lookback_bars": 60})
        structure.set_history_lookback_bars(80, symbol="EURUSD")
        circuits = {source: engine.market_state.get_candles("EURUSD", price_source=source)
                    for source in ("bid", "ask")}
        with patch("ultimateBot.backtester.strategy.structure_strategy.decide_market_structure",
                   return_value=(None, {})) as decider:
            for symbol, bars in (("EURUSD", 80), ("USDCHF", 60)):
                structure._analyze_structure_circuits(circuits, 0.01, rules=structure.rules_for_symbol(symbol))
                self.assertEqual(decider.call_args.kwargs["config"]["level_lookback_bars"], bars)

    def test_pending_orders_keep_their_exit_mode_across_toggle_in_both_directions(self):
        for enabled in (False, True):
            with self.subTest(initial=enabled):
                closer = TradeCloser(enabled)
                previous = order()
                closer.prepare_order(previous)
                closer.set_trailing_stop_enabled(not enabled)
                current = PositionManager().open_position(
                    Portfolio(1000), Fill(1, "TEST", START, OrderSide.BUY, 1, 100), previous)
                closer.prepare_position(current)
                self.assertEqual(current.trailing_stop_enabled, enabled)
                self.assertEqual(current.take_profit, None if enabled else 106)
                with redirect_stdout(io.StringIO()):
                    request = closer.check(current, Tick(START, "TEST", 104, 104.2, 1, 1), TradeMonitor())
                self.assertIsNone(request)
                if enabled:
                    self.assertTrue(current.trailing_stop_state.active)
                else:
                    self.assertIsNone(current.trailing_stop_state)
                with redirect_stdout(io.StringIO()):
                    at_target = closer.check(current, Tick(START, "TEST", 106, 106.2, 1, 1), TradeMonitor())
                if enabled:
                    self.assertIsNone(at_target)
                else:
                    self.assertEqual(at_target.reason, CloseReason.TAKE_PROFIT)
                new = order(2)
                closer.prepare_order(new)
                self.assertEqual(new.trailing_stop_enabled, not enabled)
                self.assertEqual(new.take_profit, 106 if enabled else None)
                self.assertEqual(new.trailing_stop_activation_price, None if enabled else 103)

    def test_live_engine_switches_after_closes_and_preserves_open_and_pending_trades(self):
        engine = build_engine([], self.args("--adaptive-accuracy-window", "50"), strategy_override=HistoryStrategy())
        engine.batcher = TickBatcher(1)
        optimizer = engine.adaptive_optimizer
        for target, stop in ((101, 98), (102, 98), (110, 99), (110, 95)):
            request = replace(order(target=target), stop_loss=stop)
            optimizer.prepare_order(request)
            engine.trade_closer.prepare_order(request)
            position = engine.position_manager.open_position(
                engine.portfolio, Fill(1, "TEST", START, OrderSide.BUY, 1, 100), request)
            engine.trade_closer.prepare_position(position)
        pending = replace(order(99, target=110), symbol="DELAYED", eligible_at=START + timedelta(seconds=100))
        optimizer.prepare_order(pending)
        engine.trade_closer.prepare_order(pending)
        engine.pending_orders_processor.queue.add(pending)

        def quote(price, seconds):
            return Tick(START + timedelta(seconds=seconds), "TEST", price, price + 0.2, 1, 1)

        def feed():
            yield quote(101, 1)
            yield quote(102, 2)
            yield quote(98, 3)
            self.assertEqual(optimizer.state_for("TEST").configuration_id, 1)
            self.assertTrue(optimizer.state_for("TEST").current_parameters.trailing_stop_enabled)
            self.assertFalse(engine.trade_closer.trailing_stop_enabled)
            old = engine.portfolio.positions[4]
            self.assertFalse(old.trailing_stop_enabled)
            self.assertEqual(old.take_profit, 110)
            signal = Signal("TEST", START + timedelta(seconds=3), SignalSide.BUY,
                            protective_level_price=95, protective_level_quality=1)
            engine._process_signal(signal, quote(98, 3))
            new = engine.pending_orders_processor.queue.get_for_symbol("TEST")[-1]
            self.assertEqual(new.strategy_configuration_id, 1)
            self.assertEqual(new.strategy_lookback_bars, 3)
            self.assertEqual(new.strategy_timeframe_minutes, 10)
            self.assertTrue(new.strategy_use_global_timeframe_validation)
            self.assertTrue(new.trailing_stop_enabled)
            self.assertIsNone(new.take_profit)
            self.assertIsNotNone(new.trailing_stop_activation_price)
            yield quote(100, 4)
            fresh = engine.portfolio.positions[5]
            self.assertTrue(fresh.trailing_stop_enabled)
            self.assertEqual(fresh.strategy_configuration_id, 1)
            self.assertTrue(fresh.strategy_use_global_timeframe_validation)
            yield replace(quote(100, 101), symbol="DELAYED")
            delayed = engine.portfolio.positions[6]
            self.assertFalse(delayed.trailing_stop_enabled)
            self.assertEqual(delayed.strategy_configuration_id, 0)
            self.assertEqual(delayed.take_profit, 110)

        engine.feed = feed()
        output = io.StringIO()
        with redirect_stdout(output):
            engine.run()
        self.assertEqual(len(engine.reports.trades), 3)
        self.assertEqual(optimizer.switch_count, 1)
        self.assertIn("[adaptive-strategy]", output.getvalue())
        self.assertIn("sample=3/50", output.getvalue())
        self.assertIn(
            "configuration_id=1 lookback=3 timeframe_minutes=10 global_validation=True trailing=True",
            output.getvalue(),
        )
        self.assertTrue(all(trade.strategy_configuration_id == 0 for trade in engine.reports.trades))


if __name__ == "__main__":
    unittest.main()
