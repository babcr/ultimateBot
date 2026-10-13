import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide
from ultimateBot.run_backtest import build_engine, parse_args


class RiskPercentTests(unittest.TestCase):
    def evaluate(self, manager, side=SignalSide.BUY):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        portfolio = Portfolio(initial_balance=1000.0)
        ticks = [
            Tick(start, "EURUSD", 1.1000, 1.1000, 1.0, 1.0),
            Tick(start + timedelta(seconds=30), "EURUSD", 1.1010, 1.1010, 1.0, 1.0),
        ]
        for tick in ticks:
            manager.on_tick(tick)
        signal = Signal("EURUSD", ticks[-1].timestamp, side)
        evaluation = manager.evaluate(signal, portfolio, ticks[-1])
        return evaluation, portfolio, signal, ticks[-1]

    def evaluate_symbol(self, manager, symbol, ticks, side=SignalSide.BUY):
        portfolio = Portfolio(initial_balance=1000.0)
        for tick in ticks:
            manager.on_tick(tick)
        signal = Signal(symbol, ticks[-1].timestamp, side)
        return manager.evaluate(signal, portfolio, ticks[-1])

    def test_stop_loss_matches_percentage_budget_below_and_above_one(self):
        for percent in (0.01, 0.1, 0.5, 0.99, 1.0, 1.01, 25.0):
            for side in (SignalSide.BUY, SignalSide.SELL):
                with self.subTest(percent=percent, side=side):
                    manager = RiskManager(risk_per_trade=percent, minimum_candles=1)
                    evaluation, portfolio, signal, tick = self.evaluate(manager, side)
                    self.assertIsNotNone(evaluation)
                    expected_loss = 1000.0 * percent / 100.0
                    self.assertAlmostEqual(evaluation.risk_percent, percent)
                    self.assertAlmostEqual(evaluation.capital_at_risk, expected_loss)

                    order = OrderBuilder().build(signal, evaluation)
                    fill_tick = Tick(tick.timestamp + timedelta(seconds=1), tick.symbol,
                                     tick.bid, tick.ask, 1.0, 1.0)
                    execution = ExecutionModel().try_execute(order, fill_tick)
                    self.assertIsNotNone(execution)
                    position = PositionManager().open_position(portfolio, execution.fill, order)
                    stop_tick = Tick(fill_tick.timestamp + timedelta(seconds=1), tick.symbol,
                                     position.stop_loss, position.stop_loss, 1.0, 1.0)
                    request = TradeMonitor().check(position, stop_tick)
                    self.assertIsNotNone(request)
                    self.assertEqual(request.reason, CloseReason.STOP_LOSS)
                    report = TradeCloser().close(portfolio, request)
                    self.assertAlmostEqual(report.pnl, -expected_loss, places=7)

    def test_fixed_quantity_respects_fractional_percentage_limit(self):
        for quantity, accepted in ((1000.0, True), (10000.0, False)):
            with self.subTest(quantity=quantity):
                manager = RiskManager(risk_per_trade=0.1, minimum_candles=1,
                                      fixed_quantity=quantity, accepted_risk_overrun=0.0)
                evaluation, _, _, _ = self.evaluate(manager)
                if accepted:
                    self.assertIsNotNone(evaluation)
                    self.assertLessEqual(evaluation.capital_at_risk, 1.0)
                else:
                    self.assertIsNone(evaluation)

    def test_symbol_risk_provider_controls_position_sizing_and_is_capped(self):
        risks = {"EURUSD": 0.25, "USDCHF": 1.5, "GBPUSD": 8.0}
        manager = RiskManager(
            risk_per_trade=5.0,
            minimum_candles=1,
            risk_percent_provider=lambda symbol: risks[symbol],
        )
        evaluation, _, _, _ = self.evaluate(manager)
        self.assertAlmostEqual(evaluation.risk_percent, 0.25)
        self.assertAlmostEqual(evaluation.capital_at_risk, 2.5)
        self.assertEqual(manager.risk_percent_for_symbol("USDCHF"), 1.5)
        self.assertEqual(manager.risk_percent_for_symbol("GBPUSD"), 5.0)

    def test_cli_fractional_percentage_reaches_engine_unchanged(self):
        with patch.object(sys, "argv", ["run_backtest", "--risk-percent", "0.1"]):
            args = parse_args()
        engine = build_engine(feed=[], args=args)
        self.assertEqual(engine.risk_manager.risk_config.risk_per_trade_percent, 0.1)

    def test_large_point_non_fx_symbols_scale_spread_ratio_before_rejection(self):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        ticks = [
            Tick(start, "BTCUSD", 10000.0, 10020.0, 1.0, 1.0),
            Tick(start + timedelta(seconds=30), "BTCUSD", 10100.0, 10120.0, 1.0, 1.0),
        ]
        strict = RiskManager(
            minimum_candles=1,
            medium_point_spread_normalization_factor=1.0,
            large_point_spread_normalization_factor=1.0,
        )
        self.assertIsNone(self.evaluate_symbol(strict, "BTCUSD", ticks))

        relaxed = RiskManager(minimum_candles=1)
        evaluation = self.evaluate_symbol(relaxed, "BTCUSD", ticks)
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation.entry_mode, SignalEntryMode.LIMIT)
        self.assertGreater(evaluation.spread_ratio, relaxed.risk_config.limit_spread_ratio)
        self.assertLessEqual(evaluation.spread_ratio, relaxed.risk_config.max_spread_ratio)

    def test_fiat_pairs_do_not_use_large_point_spread_scaling(self):
        start = datetime(2020, 1, 2, tzinfo=timezone.utc)
        ticks = [
            Tick(start, "EURUSD", 1.0000, 1.0020, 1.0, 1.0),
            Tick(start + timedelta(seconds=30), "EURUSD", 1.0100, 1.0120, 1.0, 1.0),
        ]
        manager = RiskManager(
            minimum_candles=1,
            medium_point_spread_normalization_factor=20.0,
            large_point_spread_normalization_factor=20.0,
        )
        self.assertIsNone(self.evaluate_symbol(manager, "EURUSD", ticks))

    def test_cli_spread_normalization_factors_reach_engine(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_backtest",
                "--medium-point-spread-normalization-factor",
                "3",
                "--large-point-spread-normalization-factor",
                "5",
            ],
        ):
            args = parse_args()
        engine = build_engine(feed=[], args=args)
        self.assertEqual(engine.risk_manager.volatility_config.medium_point_spread_normalization_factor, 3.0)
        self.assertEqual(engine.risk_manager.volatility_config.large_point_spread_normalization_factor, 5.0)


if __name__ == "__main__":
    unittest.main()
