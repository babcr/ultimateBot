import io
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from ultimateBot.backtester.config.backtest_config import OrderConfig
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.execution.execution_model import ExecutionModel
from ultimateBot.backtester.orders.order import OrderSide, OrderType, PendingOrderProcessor, PendingOrderQueue
from ultimateBot.backtester.orders.order_builder import OrderBuilder
from ultimateBot.backtester.risk.risk_manager import RiskEvaluation
from ultimateBot.backtester.strategy.signal import Signal, SignalEntryMode, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.run_backtest import build_engine, parse_args, resolve_backtest_order_config


START = datetime(2020, 5, 26, tzinfo=timezone.utc)


def quote(seconds, bid=100.0, ask=100.2, symbol="TEST"):
    return Tick(START + timedelta(seconds=seconds), symbol, bid, ask, 1.0, 1.0)


def make_order(order_type=OrderType.LIMIT, side=OrderSide.BUY, *, builder=None, timestamp=START, symbol="TEST"):
    entry_mode = SignalEntryMode[order_type.name]
    signal = Signal(symbol, timestamp, SignalSide.BUY if side == OrderSide.BUY else SignalSide.SELL,
                    entry_mode=entry_mode)
    evaluation = RiskEvaluation(
        side=side, order_type=order_type, entry_mode=entry_mode,
        entry_price=100.0, stop_loss=98.0 if side == OrderSide.BUY else 102.0,
        take_profit=106.0 if side == OrderSide.BUY else 94.0,
        quantity=1.0, available_capital=1000.0, capital_at_risk=2.0,
        risk_percent=0.2, stop_distance=2.0, take_profit_distance=6.0,
        raw_spread=0.2, spread_ratio=0.1, entry_offset=0.0,
    )
    return (builder or OrderBuilder()).build(signal, evaluation)


class OrderExpirationTests(unittest.TestCase):
    def test_every_order_type_and_side_expires_relative_to_creation(self):
        duration = timedelta(seconds=1.25)
        builder = OrderBuilder(execution_delay=timedelta(milliseconds=75), expiration_bars=5,
                               candle_interval=timedelta(milliseconds=250))
        for order_type in OrderType:
            for side in OrderSide:
                with self.subTest(order_type=order_type, side=side):
                    created_at = START + timedelta(seconds=30)
                    order = make_order(order_type, side, builder=builder, timestamp=created_at)
                    self.assertEqual(order.expires_at, created_at + duration)
                    self.assertEqual(order.eligible_at, created_at + timedelta(milliseconds=75))

    def test_constructor_rejects_impossible_lifetimes(self):
        for bars in (-1, 0, 1.5, True, "2"):
            with self.subTest(bars=bars), self.assertRaises(ValueError):
                OrderBuilder(expiration_bars=bars)
        for interval in (timedelta(seconds=-1), timedelta(0), timedelta(milliseconds=50)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                OrderBuilder(expiration_bars=1, candle_interval=interval)
        with self.assertRaises(ValueError):
            OrderBuilder(execution_delay=timedelta(seconds=-1))

    def test_default_lifetime_is_shared_by_builder_and_cli(self):
        with patch.object(sys, "argv", ["run_backtest"]):
            args = parse_args()
        config = OrderConfig()
        engine = build_engine([], args)
        self.assertEqual(make_order().expires_at, START + config.expiration_delay)
        self.assertEqual(args.order_expiration_bars, config.expiration_bars)
        self.assertEqual(engine.order_builder.expiration_delay, config.expiration_delay)
        self.assertEqual(resolve_backtest_order_config(SimpleNamespace()), config)

    def test_cli_accepts_custom_lifetime_and_centralized_defaults(self):
        configured = OrderConfig(execution_delay=timedelta(milliseconds=75),
                                 expiration_bars=3)
        for extra, expected_bars in (([], 3), (["--order-expiration-bars", "2"], 2)):
            with self.subTest(extra=extra), patch("ultimateBot.run_backtest.DEFAULT_ORDER_CONFIG", configured), \
                    patch.object(sys, "argv", ["run_backtest", "--candle-interval", "15m"] + extra):
                engine = build_engine([], parse_args())
                self.assertEqual(engine.order_builder.expiration_bars, expected_bars)
                self.assertEqual(engine.order_builder.candle_interval, timedelta(minutes=15))
                self.assertEqual(engine.order_builder.expiration_delay, timedelta(minutes=15 * expected_bars))
                self.assertEqual(engine.order_builder.execution_delay, configured.execution_delay)

    def test_cli_expiration_uses_base_timeframe_independently_of_global_confirmation(self):
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression",
                                       "--candle-interval", "15m", "--order-expiration-bars", "2",
                                       "--regression-global-timeframe-multiplier", "3"]):
            engine = build_engine([], parse_args())
        self.assertEqual(engine.order_builder.candle_interval, engine.strategy.analysis_interval)
        self.assertEqual(engine.order_builder.expiration_delay, timedelta(minutes=30))

    def test_cli_rejects_nonpositive_noninteger_and_unusable_lifetimes(self):
        for value in ("0", "-1", "nan", "inf", "-inf", "2.5", "1.0", str(10 ** 20)):
            with self.subTest(value=value), patch.object(
                sys, "argv", ["run_backtest", f"--order-expiration-bars={value}"],
            ), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as raised:
                parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_expiration_wins_over_a_possible_fill_at_the_deadline(self):
        for order_type in OrderType:
            for side in OrderSide:
                with self.subTest(order_type=order_type, side=side):
                    queue = PendingOrderQueue()
                    execution = ExecutionModel()
                    processor = PendingOrderProcessor(queue, execution)
                    order = make_order(order_type, side, builder=OrderBuilder(expiration_bars=2,
                                                                            candle_interval=timedelta(seconds=1)))
                    queue.add(order)
                    output = io.StringIO()
                    with redirect_stdout(output), patch.object(execution, "try_execute", wraps=execution.try_execute) as execute:
                        fills = processor.on_tick(quote(2, bid=100, ask=100))
                        self.assertEqual(fills, [])
                        execute.assert_not_called()
                        self.assertEqual(processor.on_tick(quote(3, bid=100, ask=100)), [])
                    self.assertEqual(queue.count_for_symbol("TEST"), 0)
                    self.assertIn("[order-expired]", output.getvalue())
                    self.assertIn(f"expires_at={order.expires_at.isoformat()}", output.getvalue())

    def test_order_can_fill_just_before_expiration(self):
        for order_type in OrderType:
            with self.subTest(order_type=order_type):
                queue = PendingOrderQueue()
                order = make_order(order_type, builder=OrderBuilder(expiration_bars=2,
                                                                   candle_interval=timedelta(seconds=1)))
                queue.add(order)
                processor = PendingOrderProcessor(queue, ExecutionModel())
                current = Tick(order.expires_at - timedelta(microseconds=1), "TEST", 100, 100, 1, 1)
                fills = processor.on_tick(current)
                self.assertEqual(len(fills), 1)
                self.assertIs(fills[0].order, order)
                self.assertEqual(len(queue), 0)

    def test_expiration_keeps_other_orders_and_symbols_intact(self):
        queue = PendingOrderQueue()
        builder = OrderBuilder(expiration_bars=2, candle_interval=timedelta(seconds=1))
        first = make_order(builder=builder)
        later = make_order(builder=builder, timestamp=START + timedelta(seconds=1))
        other = make_order(builder=builder, symbol="OTHER")
        queue.add_many([first, later, other])
        processor = PendingOrderProcessor(queue, ExecutionModel())
        self.assertEqual(processor.on_tick(quote(1.5, bid=101, ask=101.2)), [])
        self.assertEqual(queue.get_for_symbol("TEST"), [first, later])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(processor.on_tick(quote(2, bid=101, ask=101.2)), [])
        self.assertEqual(queue.get_for_symbol("TEST"), [later])
        self.assertEqual(queue.get_for_symbol("OTHER"), [other])
        with redirect_stdout(io.StringIO()):
            processor.on_tick(quote(3, bid=101, ask=101.2))
            processor.on_tick(quote(3, symbol="OTHER"))
        self.assertEqual(len(queue), 0)

    def test_fill_capacity_retains_orders_and_expiration_still_runs_at_zero_capacity(self):
        queue = PendingOrderQueue()
        builder = OrderBuilder(expiration_bars=2, candle_interval=timedelta(seconds=1))
        orders = [make_order(OrderType.MARKET, builder=builder) for _ in range(3)]
        queue.add_many(orders)
        processor = PendingOrderProcessor(queue, ExecutionModel())
        fills = processor.on_tick(quote(1), max_fills=1)
        self.assertEqual([result.order for result in fills], orders[:1])
        self.assertEqual(queue.get_for_symbol("TEST"), orders[1:])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(processor.on_tick(quote(2), max_fills=0), [])
        self.assertEqual(len(queue), 0)


class ReenteringStrategy(Strategy):
    def on_market_data(self, batch, context):
        current = batch.ticks[-1]
        if current.timestamp == START:
            return []
        mode = SignalEntryMode.LIMIT if current.timestamp == START + timedelta(seconds=1) else SignalEntryMode.MARKET
        return [Signal(current.symbol, current.timestamp, SignalSide.BUY, entry_mode=mode)]


class OrderExpirationIntegrationTests(unittest.TestCase):
    def engine(self, *extra):
        options = ["run_backtest", "--balance", "1000", "--minimum-candles", "1",
                   "--max-positions", "1", "--reward-risk-ratio", "1",
                   "--limit-spread-ratio", "1", "--max-spread-ratio", "1",
                   "--candle-interval", "10s", "--order-expiration-bars", "1"]
        with patch.object(sys, "argv", options + list(extra)):
            engine = build_engine([], parse_args(), strategy_override=ReenteringStrategy())
        engine.batcher = TickBatcher(batch_size=1)
        return engine

    def test_pending_limit_allows_a_new_opportunity_before_expiring_and_positions_outlive_expiration(self):
        for trailing in (False, True):
            with self.subTest(trailing=trailing):
                engine = self.engine(*(["--trailing-stop"] if trailing else []))

                def feed():
                    yield quote(0, bid=98, ask=98.2)
                    yield quote(1)
                    first = engine.pending_orders_processor.queue.get_for_symbol("TEST")[0]
                    self.assertEqual(first.expires_at, START + timedelta(seconds=11))
                    yield quote(2)
                    pending = engine.pending_orders_processor.queue.get_for_symbol("TEST")
                    self.assertEqual(len(pending), 2)
                    self.assertIs(pending[0], first)
                    second = pending[1]
                    self.assertEqual(second.order_id, 2)
                    self.assertEqual(second.expires_at, START + timedelta(seconds=12))
                    yield quote(3)
                    self.assertEqual(len(engine.portfolio.positions), 1)
                    self.assertEqual(engine.pending_orders_processor.queue.get_for_symbol("TEST"), [first])
                    yield quote(11)
                    self.assertEqual(len(engine.pending_orders_processor.queue), 0)
                    yield quote(12)

                engine.feed = feed()
                output = io.StringIO()
                with redirect_stdout(output):
                    engine.run()
                self.assertEqual(len(engine.portfolio.positions), 1)
                self.assertEqual(len(engine.pending_orders_processor.queue), 0)
                self.assertEqual(engine.reports.trades, [])
                self.assertEqual(engine.portfolio.positions[1].trailing_stop_enabled, trailing)
                self.assertIn("[order-expired] order_id=1", output.getvalue())
                self.assertIn("[trade-filled] order_id=2", output.getvalue())

    def test_fill_capacity_is_global_and_expired_orders_are_removed_while_positions_are_full(self):
        engine = self.engine("--max-positions", "2")
        builder = OrderBuilder(expiration_bars=2, candle_interval=timedelta(seconds=1))
        orders = [make_order(OrderType.MARKET, builder=builder) for _ in range(3)]
        other = make_order(OrderType.MARKET, builder=builder, symbol="OTHER")
        engine.pending_orders_processor.queue.add_many(orders + [other])
        with redirect_stdout(io.StringIO()):
            engine._process_pending_orders(quote(1))
            engine._process_pending_orders(quote(1, symbol="OTHER"))
        self.assertEqual(len(engine.portfolio.positions), 2)
        self.assertEqual(engine.pending_orders_processor.queue.get_for_symbol("TEST"), orders[2:])
        self.assertEqual(engine.pending_orders_processor.queue.get_for_symbol("OTHER"), [other])
        with redirect_stdout(io.StringIO()):
            engine._process_pending_orders(quote(2))
            engine._process_pending_orders(quote(2, symbol="OTHER"))
        self.assertEqual(len(engine.portfolio.positions), 2)
        self.assertEqual(len(engine.pending_orders_processor.queue), 0)


if __name__ == "__main__":
    unittest.main()
