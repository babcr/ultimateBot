import io
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.execution.fill import Fill
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.monitoring.trailing_stop import TrailingStopManager
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide, OrderType
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.portfolio.position_manager import PositionManager
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.risk.risk_manager import RiskManager
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.run_backtest import build_engine, parse_args


START = datetime(2020, 1, 2, tzinfo=timezone.utc)


def tick(bid, ask=None, symbol="TEST", seconds=1):
    return Tick(START + timedelta(seconds=seconds), symbol, bid, bid if ask is None else ask, 1.0, 1.0)


def position(side=OrderSide.BUY, trade_id=1, symbol="TEST"):
    return Position(trade_id, symbol, side, 2.0, START, 100.0,
                    stop_loss=98.0 if side == OrderSide.BUY else 102.0,
                    take_profit=120.0 if side == OrderSide.BUY else 80.0,
                    trailing_stop_activation_price=106.0 if side == OrderSide.BUY else 94.0)


class TrailingStopTests(unittest.TestCase):
    def setUp(self):
        self.closer = TradeCloser(trailing_stop_enabled=True)
        self.monitor = TradeMonitor()

    def check(self, current, quote):
        with redirect_stdout(io.StringIO()):
            return self.closer.check(current, quote, self.monitor)

    def test_fixed_tp_is_the_default_and_initial_sl_still_closes(self):
        for side, target, stop in ((OrderSide.BUY, 106.0, 98.0), (OrderSide.SELL, 94.0, 102.0)):
            with self.subTest(side=side):
                closer = TradeCloser(False)
                current = replace(position(side), take_profit=target, trailing_stop_activation_price=None)
                self.assertEqual(closer.check(current, tick(target), self.monitor).reason, CloseReason.TAKE_PROFIT)
                self.assertIsNone(current.trailing_stop_state)
                self.assertEqual(self.check(position(side), tick(stop)).reason, CloseReason.STOP_LOSS)

    def test_activation_uses_bid_for_buy_and_ask_for_sell(self):
        for side, wrong_quote, activation_quote in (
            (OrderSide.BUY, tick(105.9, 106.1), tick(106.0, 106.2)),
            (OrderSide.SELL, tick(93.9, 94.1), tick(93.8, 94.0)),
        ):
            with self.subTest(side=side):
                current = position(side)
                self.assertIsNone(self.check(current, wrong_quote))
                self.assertFalse(current.trailing_stop_state.active)
                self.assertIsNone(current.take_profit)
                self.assertIsNone(self.check(current, activation_quote))
                self.assertTrue(current.trailing_stop_state.active)
                self.assertEqual(current.stop_loss, 100.0)
                self.assertEqual(current.trailing_stop_state.steps, 0)

    def test_one_original_risk_step_moves_stop_one_step_for_both_directions(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            with self.subTest(side=side):
                current = position(side)
                target = current.trailing_stop_activation_price
                original_stop = current.stop_loss
                self.check(current, tick(target))
                self.check(current, tick(target + direction * 1.9))
                self.assertEqual(current.stop_loss, 100.0)
                self.check(current, tick(target + direction * 2.0))
                self.assertEqual(current.stop_loss, 100.0 + direction * 2.0)
                self.check(current, tick(target + direction * 6.0))
                self.assertEqual(current.stop_loss, 100.0 + direction * 6.0)
                self.assertEqual(current.trailing_stop_state.steps, 3)
                self.assertEqual(current.trailing_stop_state.initial_stop_loss, original_stop)
                self.assertEqual(current.trailing_stop_state.initial_stop_distance, 2.0)

    def test_stop_never_retreats_and_reversal_between_tp_and_stop_keeps_position(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            with self.subTest(side=side):
                current = position(side)
                target = current.trailing_stop_activation_price
                self.check(current, tick(target + direction * 4.0))
                self.assertEqual(current.stop_loss, 100.0 + direction * 4.0)
                self.assertIsNone(self.check(current, tick(target + direction * 1.0)))
                self.assertEqual(current.stop_loss, 100.0 + direction * 4.0)
                self.assertEqual(current.trailing_stop_state.steps, 2)

    def test_gap_on_activation_catches_up_every_crossed_step(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            with self.subTest(side=side):
                current = position(side)
                self.assertIsNone(self.check(current, tick(current.trailing_stop_activation_price + direction * 6.5)))
                self.assertEqual(current.stop_loss, 100.0 + direction * 6.0)
                self.assertEqual(current.trailing_stop_state.steps, 3)

    def test_forex_float_boundary_counts_exact_step_without_early_advance(self):
        current = replace(position(), entry_price=1.10, stop_loss=1.09, take_profit=1.20,
                          trailing_stop_activation_price=1.13)
        self.check(current, tick(1.13))
        self.check(current, tick(1.14999999))
        self.assertEqual(current.trailing_stop_state.steps, 1)
        self.check(current, tick(1.15))
        self.assertEqual(current.trailing_stop_state.steps, 2)
        self.assertAlmostEqual(current.stop_loss, 1.12)

    def test_breakeven_closes_at_zero_realized_pnl_for_both_sides(self):
        for side in (OrderSide.BUY, OrderSide.SELL):
            with self.subTest(side=side):
                current = position(side)
                portfolio = Portfolio(1000.0)
                portfolio.positions[current.trade_id] = current
                self.check(current, tick(current.trailing_stop_activation_price))
                quote = tick(100.0, 100.2) if side == OrderSide.BUY else tick(99.8, 100.0)
                request = self.check(current, quote)
                self.assertEqual(request.reason, CloseReason.TRAILING_STOP)
                report = self.closer.close(portfolio, request)
                self.assertEqual(report.pnl, 0.0)
                self.assertEqual(portfolio.balance, 1000.0)
                self.assertFalse(portfolio.positions)

    def test_stop_gap_exits_at_executable_quote_rather_than_posted_stop(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            with self.subTest(side=side):
                current = position(side)
                self.check(current, tick(current.trailing_stop_activation_price + direction * 4.0))
                price = current.stop_loss - direction * 0.5
                quote = tick(price, price + 0.2) if side == OrderSide.BUY else tick(price - 0.2, price)
                request = self.check(current, quote)
                self.assertEqual(request.reason, CloseReason.TRAILING_STOP)
                self.assertEqual(request.price, price)

    def test_positions_and_symbols_keep_separate_trailing_state(self):
        first = position()
        second = position(trade_id=2)
        other = position(trade_id=3, symbol="OTHER")
        self.check(first, tick(108.0))
        self.assertIsNone(self.check(other, tick(110.0)))
        self.assertIsNone(other.trailing_stop_state)
        self.check(second, tick(105.0))
        self.assertFalse(second.trailing_stop_state.active)
        self.assertTrue(first.trailing_stop_state.active)
        self.assertEqual(first.stop_loss, 102.0)
        self.assertEqual(second.stop_loss, 98.0)

    def test_fill_price_sets_breakeven_and_original_distance_for_all_order_types(self):
        manager = TrailingStopManager()
        for order_type in OrderType:
            for side in (OrderSide.BUY, OrderSide.SELL):
                with self.subTest(order_type=order_type, side=side):
                    current = position(side)
                    order = OrderRequest(1, "TEST", START, START, START, side, 2.0,
                                         order_type=order_type, stop_loss=current.stop_loss,
                                         take_profit=current.take_profit)
                    order.trailing_stop_activation_price = current.trailing_stop_activation_price
                    manager.prepare_order(order)
                    self.assertIsNone(order.take_profit)
                    fill_price = 100.5 if side == OrderSide.BUY else 99.5
                    fill = Fill(1, "TEST", START, side, 2.0, fill_price)
                    filled = PositionManager().open_position(Portfolio(1000.0), fill, order)
                    state = manager.prepare_position(filled)
                    self.assertIsNone(filled.take_profit)
                    self.assertEqual(state.activation_price, current.trailing_stop_activation_price)
                    self.assertEqual(state.breakeven_price, fill_price)
                    self.assertEqual(state.initial_stop_distance, 2.5)
                    manager.prepare_position(filled)
                    manager.prepare_order(order)
                    self.assertIs(filled.trailing_stop_state, state)

    def test_protected_profit_does_not_increase_capital_at_risk(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            with self.subTest(side=side):
                current = position(side)
                portfolio = Portfolio(1000.0)
                portfolio.positions[1] = current
                self.assertEqual(portfolio.capital_at_risk, 4.0)
                self.check(current, tick(current.trailing_stop_activation_price + direction * 4.0))
                self.assertEqual(portfolio.capital_at_risk, 0.0)
                self.assertEqual(portfolio.free_margin, 1000.0)

    def test_initial_stop_takes_priority_before_trailing_activation(self):
        current = position()
        request = self.check(current, tick(97.0, 106.0))
        self.assertEqual(request.reason, CloseReason.STOP_LOSS)
        self.assertFalse(current.trailing_stop_state.active)
        self.assertEqual(current.stop_loss, 98.0)

    def test_fill_above_activation_can_close_immediately_after_breakeven_move(self):
        current = replace(position(), entry_price=107.0)
        request = self.check(current, tick(106.5, 107.0))
        self.assertEqual(current.stop_loss, 107.0)
        self.assertEqual(request.reason, CloseReason.TRAILING_STOP)
        self.assertEqual(request.price, 106.5)

    def test_planned_take_profit_does_not_close_after_trailing_activation(self):
        for side in (OrderSide.BUY, OrderSide.SELL):
            with self.subTest(side=side):
                current = position(side)
                target = current.take_profit
                self.assertIsNone(self.check(current, tick(current.trailing_stop_activation_price)))
                self.assertIsNone(current.take_profit)
                self.assertTrue(current.trailing_stop_state.active)
                self.assertIsNone(self.check(current, tick(target)))
                self.assertEqual(self.check(current, tick(current.stop_loss)).reason, CloseReason.TRAILING_STOP)

    def test_gap_past_planned_tp_catches_up_trailing_without_closing(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            with self.subTest(side=side):
                current = position(side)
                price = current.take_profit + direction * 0.5
                request = self.check(current, tick(price))
                self.assertIsNone(request)
                self.assertIsNone(current.take_profit)
                self.assertTrue(current.trailing_stop_state.active)
                self.assertEqual(current.trailing_stop_state.steps, 7)

    def test_activation_equal_to_or_beyond_tp_is_rejected(self):
        for side, direction in ((OrderSide.BUY, 1), (OrderSide.SELL, -1)):
            for excess in (0, 1):
                with self.subTest(side=side, excess=excess), self.assertRaises(ValueError):
                    current = position(side)
                    current.trailing_stop_activation_price = current.take_profit + direction * excess
                    self.closer.prepare_position(current)


class TrailingActivationRiskTests(unittest.TestCase):
    def evaluation(self, side, entry_mode, start_rr):
        manager = RiskManager(reward_risk_ratio=3, trailing_start_rr_ratio=start_rr,
                              minimum_candles=1, limit_spread_ratio=1, max_spread_ratio=1)
        manager.on_tick(tick(98, 98.2, seconds=0))
        current = tick(100, 100.2, seconds=1)
        manager.on_tick(current)
        signal = Signal("TEST", current.timestamp, side, entry_mode=entry_mode,
                        protective_level_price=98 if side == SignalSide.BUY else 102,
                        protective_level_quality=1)
        return manager.evaluate(signal, Portfolio(1000), current)

    def test_start_rr_sets_separate_level_for_all_order_types_and_sides(self):
        for side, direction in ((SignalSide.BUY, 1), (SignalSide.SELL, -1)):
            for mode in SignalEntryMode:
                with self.subTest(side=side, mode=mode):
                    result = self.evaluation(side, mode, 1)
                    self.assertIsNotNone(result)
                    self.assertAlmostEqual(result.trailing_stop_activation_price,
                                           result.entry_price + direction * result.stop_distance)
                    self.assertGreater(direction * (result.take_profit - result.trailing_stop_activation_price), 0)

    def test_changing_start_rr_preserves_tp_stop_and_risk_budget(self):
        first = self.evaluation(SignalSide.BUY, SignalEntryMode.MARKET, 0.5)
        second = self.evaluation(SignalSide.BUY, SignalEntryMode.MARKET, 2)
        for field in ("entry_price", "stop_loss", "take_profit", "quantity", "risk_percent"):
            self.assertEqual(getattr(first, field), getattr(second, field))
        self.assertLess(first.trailing_stop_activation_price, second.trailing_stop_activation_price)

    def test_default_start_rr_is_half_the_tp_rr(self):
        for target_rr in (0.1, 0.5, 2.5, 3):
            self.assertEqual(RiskManager(reward_risk_ratio=target_rr).risk_config.trailing_start_rr_ratio,
                             target_rr / 2)

    def test_invalid_start_rr_is_rejected_by_cli_and_risk_manager(self):
        for value in (0, -1, 3, 4, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RiskManager(reward_risk_ratio=3, trailing_start_rr_ratio=value)
                with patch.object(sys, "argv", ["run_backtest", "--trailing-stop", "--reward-risk-ratio", "3",
                                               "--trailing-start-rr-ratio", str(value)]), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()


class SingleSignalStrategy(Strategy):
    def __init__(self, side):
        self.side = side
        self.sent = False

    def on_market_data(self, batch, context):
        if self.sent or batch.ticks[-1].timestamp < START + timedelta(seconds=1):
            return []
        self.sent = True
        return [Signal("TEST", batch.ticks[-1].timestamp, self.side,
                       protective_level_price=98.0 if self.side == SignalSide.BUY else 102.0,
                       protective_level_quality=1.0)]


class TrailingStopIntegrationTests(unittest.TestCase):
    def engine(self, side, enabled, target_rr=10, start_rr=3):
        options = ["run_backtest", "--batch-size", "1", "--balance", "1000",
                   "--minimum-candles", "1", "--reward-risk-ratio", str(target_rr),
                   "--trailing-start-rr-ratio", str(start_rr),
                   "--limit-spread-ratio", "1", "--max-spread-ratio", "1"]
        if enabled:
            options.append("--trailing-stop")
        with patch.object(sys, "argv", options):
            engine = build_engine([], parse_args(), strategy_override=SingleSignalStrategy(side))
        # This synthetic strategy evaluates every tick, independently of candle batching.
        engine.batcher = TickBatcher(batch_size=1)
        return engine

    def test_cli_enables_trailing_closer_and_defaults_to_fixed_tp(self):
        self.assertFalse(self.engine(SignalSide.BUY, False).trade_closer.trailing_stop_enabled)
        self.assertTrue(self.engine(SignalSide.BUY, True).trade_closer.trailing_stop_enabled)

    def test_full_engine_starts_at_one_rr_runs_beyond_three_rr_and_exits_on_stop(self):
        for side, direction in ((SignalSide.BUY, 1), (SignalSide.SELL, -1)):
            with self.subTest(side=side):
                engine = self.engine(side, True, target_rr=3, start_rr=1)
                planned = []
                original_builder = engine.order_builder.build

                def record_order(signal, evaluation):
                    planned.append(evaluation)
                    return original_builder(signal, evaluation)

                def quote(price, seconds):
                    return tick(price, price + 0.2, seconds=seconds) if direction == 1 else tick(price - 0.2, price, seconds=seconds)

                def feed():
                    yield tick(98, 98.2, seconds=0)
                    yield tick(100, 100.2, seconds=1)
                    order = engine.pending_orders_processor.queue.get_for_symbol("TEST")[0]
                    activation = order.trailing_stop_activation_price
                    self.assertIsNone(order.take_profit)
                    target = planned[0].take_profit
                    self.assertGreater(direction * (target - activation), 0)
                    yield tick(100, 100.2, seconds=2)
                    current = engine.portfolio.positions[1]
                    yield quote(activation - direction * 0.01, 3)
                    self.assertFalse(current.trailing_stop_state.active)
                    yield quote(activation, 4)
                    self.assertTrue(current.trailing_stop_state.active)
                    self.assertEqual(current.stop_loss, current.entry_price)
                    self.assertIsNone(current.take_profit)
                    yield quote(target - direction * 0.01, 5)
                    self.assertIn(1, engine.portfolio.positions)
                    yield quote(target, 6)
                    self.assertIn(1, engine.portfolio.positions)
                    risk = current.trailing_stop_state.initial_stop_distance
                    yield quote(target + direction * 2 * risk, 7)
                    self.assertIn(1, engine.portfolio.positions)
                    yield quote(current.stop_loss - direction * 0.1, 8)

                engine.feed = feed()
                with patch.object(engine.order_builder, "build", side_effect=record_order), redirect_stdout(io.StringIO()):
                    engine.run()
                self.assertEqual(len(engine.reports.trades), 1)
                self.assertEqual(engine.reports.trades[0].close_reason, CloseReason.TRAILING_STOP)
                self.assertGreater(engine.reports.trades[0].pnl, 0)
                self.assertTrue(engine.reports.trades[0].trailing_stop_enabled)
                self.assertFalse(engine.portfolio.positions)

    def test_full_engine_starts_before_tp_then_closes_on_trailing_stop(self):
        for side, direction in ((SignalSide.BUY, 1), (SignalSide.SELL, -1)):
            with self.subTest(side=side):
                engine = self.engine(side, True)

                def quote(exit_price, seconds):
                    return (tick(exit_price, exit_price + 0.2, seconds=seconds) if direction == 1
                            else tick(exit_price - 0.2, exit_price, seconds=seconds))

                def feed():
                    yield tick(98.0, 98.2, seconds=0)
                    yield tick(100.0, 100.2, seconds=1)
                    order = engine.pending_orders_processor.queue.get_for_symbol("TEST")[0]
                    self.assertIsNone(order.take_profit)
                    target = order.trailing_stop_activation_price
                    initial_stop = order.stop_loss
                    yield (tick(100.3, 100.5, seconds=2) if direction == 1
                           else tick(99.5, 99.7, seconds=2))
                    current = engine.portfolio.positions[1]
                    state = current.trailing_stop_state
                    self.assertEqual(state.initial_stop_loss, initial_stop)
                    self.assertEqual(state.breakeven_price, current.entry_price)
                    risk = abs(current.entry_price - initial_stop)
                    self.assertEqual(state.initial_stop_distance, risk)
                    yield quote(target - direction * 0.1, 3)
                    self.assertFalse(state.active)
                    yield quote(target, 4)
                    self.assertIn(1, engine.portfolio.positions)
                    self.assertTrue(state.active)
                    self.assertEqual(current.stop_loss, current.entry_price)
                    yield quote(target + direction * risk, 5)
                    self.assertAlmostEqual(current.stop_loss, current.entry_price + direction * risk)
                    yield quote(target + direction * 3.0 * risk, 6)
                    self.assertEqual(state.steps, 3)
                    self.assertAlmostEqual(current.stop_loss, current.entry_price + direction * 3.0 * risk)
                    yield quote(current.stop_loss - direction * 0.5, 7)

                engine.feed = feed()
                output = io.StringIO()
                with redirect_stdout(output):
                    engine.run()
                self.assertEqual(len(engine.reports.trades), 1)
                report = engine.reports.trades[0]
                self.assertEqual(report.close_reason, CloseReason.TRAILING_STOP)
                self.assertGreater(report.pnl, 0.0)
                self.assertAlmostEqual(engine.portfolio.balance, 1000.0 + report.pnl)
                self.assertFalse(engine.portfolio.positions)
                for text in ("take_profit=None", "trailing_activation_price=", "event=activated", "event=advanced",
                             "reason=trailing_stop", "[trade-created]", "[trade-filled]", "[trade-closed]"):
                    self.assertIn(text, output.getvalue())

    def test_full_engine_without_flag_closes_at_fixed_tp(self):
        for side, direction in ((SignalSide.BUY, 1), (SignalSide.SELL, -1)):
            with self.subTest(side=side):
                engine = self.engine(side, False)

                def feed():
                    yield tick(98.0, 98.2, seconds=0)
                    yield tick(100.0, 100.2, seconds=1)
                    order = engine.pending_orders_processor.queue.get_for_symbol("TEST")[0]
                    self.assertIsNone(order.trailing_stop_activation_price)
                    yield tick(100.0, 100.2, seconds=2)
                    current = engine.portfolio.positions[1]
                    self.assertIsNone(current.trailing_stop_state)
                    target = current.take_profit
                    yield (tick(target, target + 0.2, seconds=3) if direction == 1
                           else tick(target - 0.2, target, seconds=3))

                engine.feed = feed()
                output = io.StringIO()
                with redirect_stdout(output):
                    engine.run()
                self.assertEqual(len(engine.reports.trades), 1)
                self.assertEqual(engine.reports.trades[0].close_reason, CloseReason.TAKE_PROFIT)
                self.assertNotIn("[trailing-stop]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
