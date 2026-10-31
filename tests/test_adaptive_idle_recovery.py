import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ultimateBot.backtester.optimization.adaptive_strategy import AdaptiveStrategyConfig, AdaptiveStrategyOptimizer
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.tests.test_adaptive_strategy import HistoryStrategy


START = datetime(2024, 1, 4, tzinfo=timezone.utc)


class AdaptiveIdleRecoveryTests(unittest.TestCase):
    def optimizer(self, bars=3):
        return AdaptiveStrategyOptimizer(
            AdaptiveStrategyConfig(lookback_min=5, lookback_max=20, max_idle_bars=bars,
                                   optimize_risk=False, optimize_global_timeframe_validation=False),
            HistoryStrategy(lookback=15), TradeCloser(),
        )

    def window(self, optimizer, index, symbol="EURUSD", busy=False):
        with redirect_stdout(io.StringIO()):
            optimizer.on_analysis_window(symbol, START + timedelta(minutes=10 * index), has_active_exposure=busy)

    def test_exploration_recovers_without_any_new_trade_closure(self):
        optimizer = self.optimizer()
        state = optimizer.state_for("EURUSD")
        state.configuration_id = 1
        previous = state.current_parameters
        self.window(optimizer, 1)
        self.window(optimizer, 2)
        self.assertEqual(state.configuration_id, 1)
        output = io.StringIO()
        with redirect_stdout(output):
            optimizer.on_analysis_window("EURUSD", START + timedelta(minutes=30))
        self.assertEqual(state.configuration_id, 2)
        self.assertNotEqual(state.current_parameters, previous)
        self.assertEqual(state.idle_bars, 0)
        self.assertEqual(optimizer.strategy.get_history_lookback_bars("EURUSD"), state.current_parameters.lookback_bars)
        self.assertIn("reason=no_trade_activity", output.getvalue())

    def test_startup_and_existing_exposure_are_preserved(self):
        optimizer = self.optimizer()
        state = optimizer.state_for("EURUSD")
        for i in range(6):
            self.window(optimizer, i)
        self.assertEqual((state.configuration_id, state.idle_bars), (0, 0))
        state.configuration_id = 1
        self.window(optimizer, 6)
        self.window(optimizer, 7)
        self.window(optimizer, 8, busy=True)
        self.assertEqual((state.configuration_id, state.idle_bars), (1, 0))
        self.window(optimizer, 9)
        self.window(optimizer, 10)
        self.assertEqual(state.configuration_id, 1)
        self.window(optimizer, 11)
        self.assertEqual(state.configuration_id, 2)

    def test_duplicate_candles_and_old_clocks_do_not_count(self):
        optimizer = self.optimizer()
        state = optimizer.state_for("EURUSD")
        state.configuration_id = 1
        for i in (2, 2, 1, 2):
            self.window(optimizer, i)
        self.assertEqual(state.idle_bars, 1)
        self.assertEqual(state.configuration_id, 1)

    def test_queued_order_resets_inactivity_and_symbols_are_independent(self):
        optimizer = self.optimizer()
        state = optimizer.state_for("EURUSD")
        state.configuration_id = 1
        self.window(optimizer, 1)
        self.window(optimizer, 2)
        optimizer.on_order_queued(SimpleNamespace(symbol="EURUSD", strategy_configuration_id=1))
        self.assertEqual(state.idle_bars, 0)
        self.window(optimizer, 3, symbol="GBPUSD")
        self.window(optimizer, 3)
        self.assertEqual(state.idle_bars, 1)
        self.assertEqual(optimizer.state_for("GBPUSD").configuration_id, 0)

    def test_recovery_can_be_disabled_and_negative_limits_are_rejected(self):
        optimizer = self.optimizer(0)
        state = optimizer.state_for("EURUSD")
        state.configuration_id = 1
        for i in range(10):
            self.window(optimizer, i)
        self.assertEqual(state.configuration_id, 1)
        with self.assertRaisesRegex(ValueError, "idle bars"):
            AdaptiveStrategyConfig(max_idle_bars=-1)


if __name__ == "__main__":
    unittest.main()
