import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from ultimateBot.backtester.config.trading_schedule import (
    TradingSchedule, TradingScheduleConfig, LONDON_SESSION, NEW_YORK_SESSION,
)
from ultimateBot.backtester.data.batcher import TickBatcher
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.monitoring.close_request import CloseReason
from ultimateBot.backtester.optimization.bandit_checkpoint import engine_snapshot, restore_engine
from ultimateBot.backtester.optimization.bandit_broker import BrokerBanditBridge
from ultimateBot.backtester.optimization.bandit_core import PositionLearningRecord
from ultimateBot.backtester.orders.order import OrderSide, OrderType
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.tests.test_order_expiration import make_order
from ultimateBot.tests.test_contextual_bandit import args_for, context
from ultimateBot.run_backtest import build_engine, parse_args, resolve_backtest_trading_schedule


def stamp(value):
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def quote(value, symbol="TEST", bid=100., ask=100.2):
    return Tick(stamp(value), symbol, bid, ask, 1., 1.)


class QuietStrategy(Strategy):
    minimum_candles = 2
    analysis_interval = timedelta(minutes=1)
    history_lookback_bars = 2

    def on_market_data(self, batch, context):
        return []


class SessionWindowTests(unittest.TestCase):
    def policy(self, session=None, fraction=.75):
        kwargs = {"sessions": (session,)} if session is not None else {}
        return TradingSchedule(TradingScheduleConfig(session_filter=True, session_entry_fraction=fraction, **kwargs))

    def test_local_open_inclusive_three_quarters_exclusive_both_sessions(self):
        for session, opening in ((LONDON_SESSION, (8, 0)), (NEW_YORK_SESSION, (9, 30))):
            zone = ZoneInfo(session.timezone)
            policy = self.policy(session)
            start = datetime(2026, 1, 15, *opening, tzinfo=zone)
            cutoff = datetime(2026, 1, 15, 14, 22, 30, tzinfo=zone)
            with self.subTest(session=session.name):
                self.assertFalse(policy.entries_allowed(start - timedelta(microseconds=1)))
                self.assertTrue(policy.entries_allowed(start))
                self.assertTrue(policy.entries_allowed(cutoff - timedelta(microseconds=1)))
                self.assertFalse(policy.entries_allowed(cutoff))
                self.assertEqual(cutoff.astimezone(timezone.utc), policy.entry_window_end(start))

    def test_union_accepts_either_session_and_preserves_gap(self):
        policy = self.policy()
        for value in ("2026-01-15T08:00:00", "2026-01-15T14:30:00", "2026-01-15T19:22:29"):
            self.assertTrue(policy.entries_allowed(stamp(value)))
        for value in ("2026-01-15T07:59:59", "2026-01-15T14:22:30", "2026-01-15T14:29:59", "2026-01-15T19:22:30"):
            self.assertFalse(policy.entries_allowed(stamp(value)))

    def test_dst_dates_are_independent_in_europe_and_us(self):
        london, new_york = self.policy(LONDON_SESSION), self.policy(NEW_YORK_SESSION)
        self.assertFalse(london.entries_allowed(stamp("2026-03-09T07:30:00")))
        self.assertTrue(london.entries_allowed(stamp("2026-03-30T07:30:00")))
        self.assertFalse(new_york.entries_allowed(stamp("2026-03-02T13:30:00")))
        self.assertTrue(new_york.entries_allowed(stamp("2026-03-09T13:30:00")))
        self.assertTrue(new_york.entries_allowed(stamp("2026-03-30T13:30:00")))
        self.assertTrue(london.entries_allowed(stamp("2026-10-23T07:30:00")))
        self.assertFalse(london.entries_allowed(stamp("2026-10-26T07:30:00")))
        self.assertTrue(new_york.entries_allowed(stamp("2026-10-26T13:30:00")))
        self.assertFalse(new_york.entries_allowed(stamp("2026-11-02T13:30:00")))

    def test_weekend_rejected_fraction_configurable_naive_means_utc(self):
        policy = self.policy()
        self.assertFalse(policy.entries_allowed(stamp("2026-01-17T12:00:00")))
        self.assertFalse(policy.entries_allowed(stamp("2026-01-18T16:00:00")))
        self.assertTrue(policy.entries_allowed(datetime(2026, 1, 15, 8)))
        half = self.policy(LONDON_SESSION, .5)
        self.assertTrue(half.entries_allowed(stamp("2026-01-15T12:14:59")))
        self.assertFalse(half.entries_allowed(stamp("2026-01-15T12:15:00")))
        full = self.policy(LONDON_SESSION, 1.)
        self.assertTrue(full.entries_allowed(stamp("2026-01-15T16:29:59")))
        self.assertFalse(full.entries_allowed(stamp("2026-01-15T16:30:00")))

    def test_pending_cannot_cross_gap_or_next_day_but_can_cross_overlap(self):
        policy = self.policy()
        self.assertEqual("entry_session_ended", policy.pending_rejection(
            stamp("2026-01-15T14:22:29"), stamp("2026-01-15T14:30:00")))
        self.assertEqual("entry_session_ended", policy.pending_rejection(
            stamp("2026-01-15T19:22:29"), stamp("2026-01-16T08:00:00")))
        self.assertIsNone(policy.pending_rejection(stamp("2026-03-09T14:22:29"), stamp("2026-03-09T14:30:00")))

    def test_invalid_config_and_identity_only_effective_rules(self):
        for fraction in (0, -.1, 1.01, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                TradingScheduleConfig(session_entry_fraction=fraction)
        for clock in ("24:00", "2359", "23:59Z", "23:59:00.1", "bad"):
            with self.subTest(clock=clock), self.assertRaises(ValueError):
                TradingScheduleConfig(friday_close_time=clock)
        with self.assertRaises(ValueError):
            TradingScheduleConfig(close_friday=True, friday_close_timezone="invalid/zone")
        self.assertIsNone(TradingScheduleConfig(session_entry_fraction=.5).identity())
        self.assertEqual(TradingScheduleConfig(close_friday=True).identity(),
                         TradingScheduleConfig(close_friday=True, friday_close_time="23:59:00", friday_close_timezone="Etc/UTC").identity())


class WeeklyDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.policy = TradingSchedule(TradingScheduleConfig(close_friday=True))

    def test_exact_friday_cutoff_and_weekend_lock(self):
        opened = stamp("2026-10-15T09:00:00")
        self.assertFalse(self.policy.position_due(opened, stamp("2026-10-16T23:58:59.999999")))
        self.assertTrue(self.policy.position_due(opened, stamp("2026-10-16T23:59:00")))
        self.assertTrue(self.policy.entries_allowed(stamp("2026-10-16T23:58:59")))
        for value in ("2026-10-16T23:59:00", "2026-10-17T00:00:00", "2026-10-18T23:59:59"):
            self.assertFalse(self.policy.entries_allowed(stamp(value)))
        self.assertTrue(self.policy.entries_allowed(stamp("2026-10-19T00:00:00")))

    def test_missed_quote_still_closes_old_position_next_week(self):
        friday_position = stamp("2026-10-16T09:00:00")
        monday = stamp("2026-10-19T09:00:00")
        self.assertTrue(self.policy.position_due(friday_position, monday))
        self.assertFalse(self.policy.position_due(monday, monday + timedelta(seconds=1)))
        self.assertEqual("before_weekly_cutoff", self.policy.pending_rejection(friday_position, monday))
        self.assertIsNone(self.policy.pending_rejection(monday, monday + timedelta(seconds=1)))

    def test_timezone_uses_local_friday_with_dst_and_year_boundary(self):
        policy = TradingSchedule(TradingScheduleConfig(close_friday=True, friday_close_timezone="Europe/Paris"))
        opened = stamp("2026-07-02T09:00:00")
        self.assertFalse(policy.position_due(opened, stamp("2026-07-03T21:58:59")))
        self.assertTrue(policy.position_due(opened, stamp("2026-07-03T21:59:00")))
        opened = stamp("2025-12-31T09:00:00")
        self.assertFalse(self.policy.position_due(opened, stamp("2026-01-02T23:58:59")))
        self.assertTrue(self.policy.position_due(opened, stamp("2026-01-02T23:59:00")))


class TradingScheduleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.quiet = redirect_stdout(io.StringIO()); self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)

    def engine(self, *extra, feed=()):
        with patch.object(sys, "argv", ["run_backtest", "--strategy", "swing-regression", "--symbols", "TEST",
                                       "--candle-interval", "1m", "--minimum-candles", "1", *extra]):
            engine = build_engine(feed, parse_args(), strategy_override=QuietStrategy())
        engine.batcher = TickBatcher(batch_size=1)
        return engine

    def add_position(self, engine, side=OrderSide.BUY, trade_id=1, symbol="TEST", opened="2026-10-15T09:00:00", trailing=False):
        current = Position(trade_id, symbol, side, 2., stamp(opened), 100.,
                           stop_loss=90. if side == OrderSide.BUY else 110.,
                           take_profit=120. if side == OrderSide.BUY else 80.,
                           trailing_stop_enabled=trailing,
                           trailing_stop_activation_price=106. if trailing else None)
        engine.portfolio.positions[trade_id] = current
        engine.portfolio._next_trade_id = max(engine.portfolio._next_trade_id, trade_id + 1)
        return current

    def test_flags_default_off_enable_and_disable_independently(self):
        engine = self.engine()
        self.assertFalse(engine.trading_schedule.config.enabled)
        with patch.object(sys, "argv", ["run_backtest", "--session-filter", "--close-friday"]):
            cfg = resolve_backtest_trading_schedule(parse_args())
        self.assertTrue(cfg.session_filter); self.assertTrue(cfg.close_friday)
        self.assertEqual("UTC", cfg.friday_close_timezone)
        with patch.object(sys, "argv", ["run_backtest", "--session-filter", "--close-friday", "--no-session-filter", "--no-close-friday"]):
            self.assertFalse(resolve_backtest_trading_schedule(parse_args()).enabled)
        for option, value in (("--session-entry-fraction", "1.5"), ("--friday-close-time", "24:00")):
            with patch.object(sys, "argv", ["run_backtest", option, value]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_signal_gate_before_sizing_and_creation_and_accepted_signal_fills(self):
        engine = self.engine("--session-filter", "--limit-spread-ratio", "1", "--max-spread-ratio", "1")
        rejected = quote("2026-01-15T07:59:59")
        signal = Signal("TEST", rejected.timestamp, SignalSide.BUY)
        with patch.object(engine.risk_manager, "evaluate", side_effect=AssertionError("sizing outside session")):
            engine._process_signal(signal, rejected)
        self.assertEqual(0, len(engine.pending_orders_processor.queue))
        accepted = quote("2026-01-15T08:00:00", bid=98., ask=98.2)
        engine.risk_manager.on_tick(accepted)
        accepted = quote("2026-01-15T08:00:01")
        engine.risk_manager.on_tick(accepted)
        engine._process_signal(Signal("TEST", accepted.timestamp, SignalSide.BUY), accepted)
        self.assertEqual(1, len(engine.pending_orders_processor.queue))
        engine._process_pending_orders(quote("2026-01-15T08:00:02"))
        self.assertEqual(1, len(engine.portfolio.positions))
        self.assertEqual(1, engine.trading_schedule.stats["rejected_signals"])

    def test_late_batch_and_final_flush_cannot_emit_outside_window(self):
        engine = self.engine("--session-filter")
        current = quote("2026-01-15T19:23:00")
        engine.on_tick(current)
        previous = quote("2026-01-15T19:22:29")
        with patch.object(engine.risk_manager, "evaluate", side_effect=AssertionError("late sizing")):
            engine._process_signal(Signal("TEST", previous.timestamp, SignalSide.BUY), previous)
            engine.feed = [current]
            with patch.object(engine.strategy, "on_market_data", return_value=[Signal("TEST", previous.timestamp, SignalSide.BUY)]):
                engine.run()
        self.assertEqual(0, len(engine.pending_orders_processor.queue))

    def test_pending_market_limit_and_stop_cancelled_at_window_end(self):
        for order_type in OrderType:
            engine = self.engine("--session-filter")
            order = make_order(order_type, timestamp=stamp("2026-01-15T19:22:29"))
            engine.pending_orders_processor.queue.add(order)
            with patch.object(engine.execution_model, "try_execute", side_effect=AssertionError("late fill")):
                engine._process_pending_orders(quote("2026-01-15T19:22:30"))
            self.assertEqual(0, len(engine.portfolio.positions))
            self.assertEqual(0, len(engine.pending_orders_processor.queue))
            self.assertEqual([], engine.reports.trades)
            self.assertEqual(1, engine.trading_schedule.stats["cancelled_pending_orders"])

    def test_existing_positions_keep_stops_outside_entry_windows(self):
        engine = self.engine("--session-filter")
        self.add_position(engine, opened="2026-01-15T10:00:00")
        engine.on_tick(quote("2026-01-15T22:00:00", bid=89., ask=89.2))
        self.assertEqual(1, len(engine.reports.trades))
        self.assertEqual(CloseReason.STOP_LOSS, engine.reports.trades[0].close_reason)

    def test_friday_closes_all_positions_at_correct_bid_ask_per_symbol(self):
        engine = self.engine("--close-friday")
        self.add_position(engine)
        self.add_position(engine, OrderSide.SELL, 2, "OTHER")
        engine.on_tick(quote("2026-10-16T23:58:59"))
        self.assertEqual(2, len(engine.portfolio.positions))
        engine.on_tick(quote("2026-10-16T23:59:00", bid=101., ask=101.2))
        self.assertEqual([2], list(engine.portfolio.positions))
        engine.on_tick(quote("2026-10-16T23:59:01", "OTHER", 98.8, 99.))
        self.assertEqual({}, engine.portfolio.positions)
        self.assertEqual([CloseReason.END_OF_WEEK] * 2, [report.close_reason for report in engine.reports.trades])
        self.assertEqual([101., 99.], [report.exit_price for report in engine.reports.trades])
        self.assertEqual([2., 2.], [report.pnl for report in engine.reports.trades])
        self.assertEqual(54., engine.portfolio.balance)
        self.assertEqual(2, engine.reports.trades[0].max_favorable_pnl)
        engine.on_tick(quote("2026-10-17T00:00:00"))
        self.assertEqual(2, len(engine.reports.trades))

    def test_pending_all_symbols_cancelled_no_reentry_until_monday(self):
        engine = self.engine("--close-friday")
        for symbol in ("TEST", "OTHER"):
            engine.pending_orders_processor.queue.add(make_order(OrderType.MARKET, symbol=symbol,
                                                                  timestamp=stamp("2026-10-16T23:58:59")))
        engine.on_tick(quote("2026-10-16T23:59:00"))
        self.assertEqual(0, len(engine.pending_orders_processor.queue))
        self.assertEqual(2, engine.trading_schedule.stats["cancelled_pending_orders"])
        sunday = quote("2026-10-18T23:00:00")
        engine._process_signal(Signal("TEST", sunday.timestamp, SignalSide.BUY), sunday)
        self.assertEqual(0, len(engine.pending_orders_processor.queue))
        engine.on_tick(quote("2026-10-19T00:00:00"))
        self.assertTrue(engine.trading_schedule.entries_allowed(engine._market_clock))

    def test_quote_gap_and_checkpoint_close_on_next_quote_without_future_price(self):
        engine = self.engine("--close-friday")
        self.add_position(engine)
        engine.on_tick(quote("2026-10-16T23:58:00"))
        saved = engine_snapshot(engine)
        restored = self.engine("--close-friday")
        restore_engine(restored, saved)
        restored.on_tick(quote("2026-10-19T08:00:00", bid=95., ask=95.2))
        report = restored.reports.trades[0]
        self.assertEqual(stamp("2026-10-19T08:00:00"), report.closed_at)
        self.assertEqual(95., report.exit_price)
        self.assertEqual(CloseReason.END_OF_WEEK, report.close_reason)
        self.assertEqual(1, restored.trading_schedule.stats["weekly_closed_positions"])
        self.add_position(restored, trade_id=2, opened="2026-10-19T08:00:01")
        restored.on_tick(quote("2026-10-19T08:00:02"))
        self.assertIn(2, restored.portfolio.positions)

    def test_weekly_exit_applies_to_active_trailing_and_preserves_mode(self):
        engine = self.engine("--close-friday", "--trailing-stop")
        current = self.add_position(engine, trailing=True)
        engine.on_tick(quote("2026-10-16T23:58:59", bid=106., ask=106.2))
        self.assertTrue(current.trailing_stop_state.active)
        engine.on_tick(quote("2026-10-16T23:59:00", bid=108., ask=108.2))
        self.assertEqual(CloseReason.END_OF_WEEK, engine.reports.trades[0].close_reason)
        self.assertTrue(engine.reports.trades[0].trailing_stop_enabled)

    def test_disabled_options_preserve_existing_orders_and_weekend_positions(self):
        engine = self.engine()
        self.add_position(engine)
        order = make_order(OrderType.MARKET, timestamp=stamp("2026-10-16T23:58:59"))
        engine.pending_orders_processor.queue.add(order)
        engine.on_tick(quote("2026-10-16T23:59:00"))
        self.assertEqual(2, len(engine.portfolio.positions))
        self.assertEqual([], engine.reports.trades)
        self.assertEqual(0, engine.trading_schedule.stats["weekly_closed_positions"])


class BanditTimeRulesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "learning.sqlite3"
        self.quiet = redirect_stdout(io.StringIO()); self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.directory.cleanup()

    def engine(self, *extra):
        return build_engine([], args_for(self.path, extra))

    def test_identity_separates_enabled_rules_and_keeps_disabled_memory(self):
        plain = self.engine()
        ignored = self.engine("--session-entry-fraction", ".5", "--friday-close-time", "22:00")
        session = self.engine("--session-filter")
        weekly = self.engine("--close-friday")
        both = self.engine("--session-filter", "--close-friday")
        def model(engine):
            return engine.contextual_optimizer.models["TEST"]["identity"].model_id
        self.assertEqual(model(plain), model(ignored))
        self.assertEqual(4, len({model(plain), model(session), model(weekly), model(both)}))
        changed = self.engine("--session-filter", "--session-entry-fraction", ".5")
        self.assertNotEqual(model(session), model(changed))
        self.assertNotIn("trading_time_rules", plain.contextual_optimizer.fixed_parameters)

    def test_shadow_inherits_filter_and_weekly_close_with_isolated_stats(self):
        from ultimateBot.backtester.optimization.bandit_shadow import ShadowVariant
        engine = self.engine("--session-filter", "--close-friday")
        runtime = engine.contextual_optimizer
        bundle = runtime.models["TEST"]
        group = {"group_id": "group", "budget": .25}
        variant = ShadowVariant(runtime.shadow, group, bundle["decision"].action, bundle["decision"], "TEST")
        self.assertEqual(engine.trading_schedule.config, variant.engine.trading_schedule.config)
        self.assertIsNot(engine.trading_schedule.stats, variant.engine.trading_schedule.stats)
        position = Position(1, "TEST", OrderSide.BUY, 1., stamp("2026-10-15T09:00:00"), 100.,
                            stop_loss=90., take_profit=120., trailing_stop_enabled=False)
        variant.engine.portfolio.positions[1] = position
        tick = quote("2026-10-16T23:59:00", bid=101., ask=101.2)
        engine.market_state.update(tick)
        variant.on_tick(tick)
        self.assertEqual(CloseReason.END_OF_WEEK, variant.engine.reports.trades[0].close_reason)
        self.assertEqual(1, variant.engine.trading_schedule.stats["weekly_closed_positions"])
        self.assertEqual(0, engine.trading_schedule.stats["weekly_closed_positions"])

    def test_weekly_closure_is_a_normal_learnable_exit_and_cancel_is_not_loss(self):
        engine = self.engine("--close-friday", "--bandit-warmup", "1")
        runtime = engine.contextual_optimizer
        bundle = runtime.models["TEST"]
        decision = replace(bundle["decision"], context=context())
        record = PositionLearningRecord(decision, context(), runtime.run_id, runtime.source, "filled", "scenario",
                                         1., 98., initial_risk=2., initial_quantity=1.)
        position = Position(1, "TEST", OrderSide.BUY, 1., stamp("2026-10-15T09:00:00"), 100.,
                            stop_loss=98., take_profit=120., learning_record=record)
        engine.portfolio.positions[1] = position
        pending = make_order(OrderType.MARKET, timestamp=stamp("2026-10-16T23:58:59"))
        pending.learning_record = replace(record, trajectory_key="pending")
        engine.pending_orders_processor.queue.add(pending)
        runtime.on_order_queued(pending)
        tick = quote("2026-10-16T23:59:00", bid=101., ask=101.2)
        with patch.object(runtime, "context", return_value=context()):
            engine.on_tick(tick)
        self.assertEqual(1, len(runtime.observation_statuses))
        status = runtime.observation_statuses[0]
        self.assertEqual("end_of_week", status["close_reason"])
        self.assertEqual("eligible", status["status"])
        self.assertEqual(1., status["weight"])
        self.assertEqual(.5, status["raw_r"])
        self.assertEqual([], runtime.store.pending_records("pending_order"))


class BrokerTimeRulesTests(unittest.TestCase):
    def bridge(self):
        store = Mock(); store.pending_records.return_value = []
        coordinator = SimpleNamespace(source="demo_executed", engine=SimpleNamespace(
            trading_schedule=TradingSchedule(TradingScheduleConfig(session_filter=True, close_friday=True))),
            market_clock=None, store=store)
        return BrokerBanditBridge(coordinator, account_scope="account")

    def test_broker_entry_query_matches_session_and_weekly_rules(self):
        bridge = self.bridge()
        self.assertTrue(bridge.entry_allowed(stamp("2026-01-15T08:00:00")))
        self.assertFalse(bridge.entry_allowed(stamp("2026-01-15T07:59:59")))
        self.assertFalse(bridge.entry_allowed(stamp("2026-01-15T14:22:29"), timestamp=stamp("2026-01-15T14:30:00")))
        self.assertFalse(bridge.entry_allowed(stamp("2026-01-16T23:59:00")))

    def test_timer_requests_all_open_positions_and_never_learns_fake_fills(self):
        bridge = self.bridge()
        for position_id in ("one", "two"):
            bridge.positions[position_id] = {"opened_at": "2026-10-15T08:00:00+00:00"}
        self.assertEqual([], bridge.weekly_close_requests(stamp("2026-10-16T23:58:59")))
        requests = bridge.weekly_close_requests(stamp("2026-10-16T23:59:00"))
        self.assertEqual(["one", "two"], [request.position_id for request in requests])
        self.assertTrue(all(request.reason == CloseReason.END_OF_WEEK for request in requests))
        bridge.coordinator.store.record_deal.assert_not_called()
        bridge.coordinator.store.learn_batch.assert_not_called()

    def test_closed_or_unreconciled_broker_positions_are_not_requested_again(self):
        bridge = self.bridge()
        for position_id in ("closed", "missing", "new"):
            bridge.positions[position_id] = {"opened_at": "2026-10-15T08:00:00+00:00"}
        bridge.reconciliation.update(closed="closed_deals_committed", missing="awaiting_authoritative_close_deals")
        self.assertEqual(["new"], [request.position_id for request in bridge.weekly_close_requests(stamp("2026-10-16T23:59:00"))])


if __name__ == "__main__":
    unittest.main()
