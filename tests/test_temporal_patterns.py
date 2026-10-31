from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.strategy.signal import SignalSide
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.temporal_pattern_strategy import TemporalPatternStrategy
from ultimateBot.backtester.strategy.temporal_patterns import (
    TimedSwing, TemporalPatternConfig, TemporalPatternTracker, confirmed_swings, structural_swings, detect_groups,
)
from ultimateBot.run_backtest import parse_args, build_engine


START = datetime(2024, 1, 2, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def bar(hour, o, h, l, c):
    return Candle(START + hour * HOUR, o, h, l, c)


def shift(c, distance):
    return replace(c, open_price=c.open_price + distance, high_price=c.high_price + distance,
                   low_price=c.low_price + distance, close_price=c.close_price + distance)


def points(prices):
    return [TimedSwing("high" if i % 2 == 0 else "low", price, START + i * 2 * HOUR,
                       START + (i * 2 + 1) * HOUR) for i, price in enumerate(prices)]


def hs_group():
    swings = points([100, 98, 104, 101, 106, 101, 104.1])
    return detect_groups(swings, START + 14 * HOUR, HOUR, 1., TemporalPatternConfig())[0]


def mirror(c):
    return replace(c, open_price=210 - c.open_price, high_price=210 - c.low_price,
                   low_price=210 - c.high_price, close_price=210 - c.close_price)


def cli(*options):
    with patch.object(sys, "argv", ["run", "--strategy", "swing-patterns", *options]):
        return parse_args()


class TemporalGroupTests(unittest.TestCase):
    def test_head_shoulders_and_inverse_require_prior_trend_and_localized_geometry(self):
        cfg = TemporalPatternConfig()
        swings = points([100, 98, 104, 101, 106, 101, 104.1])
        at = START + 14 * HOUR
        group = detect_groups(swings, at, HOUR, 1., cfg)[0]
        self.assertEqual(group.name, "head-shoulders")
        self.assertEqual(group.direction, -1)
        self.assertEqual(group.height, 5)
        inverse = [replace(s, kind="low" if s.kind == "high" else "high", price=210 - s.price) for s in swings]
        self.assertEqual(detect_groups(inverse, at, HOUR, 1., cfg)[0].name, "inverse-head-shoulders")
        self.assertEqual(detect_groups(swings[-5:], at, HOUR, 1., cfg), [])
        no_trend = [replace(s, price=s.price + 8) if i < 2 else s for i, s in enumerate(swings)]
        self.assertEqual(detect_groups(no_trend, at, HOUR, 1., cfg), [])

    def test_time_span_gap_symmetry_staleness_and_confirmation_are_hard_limits(self):
        cfg = TemporalPatternConfig(family="head-shoulders")
        swings = points([100, 98, 104, 101, 106, 101, 104.1])
        at = START + 14 * HOUR
        for changed in (replace(cfg, max_span_bars=7), replace(cfg, max_swing_gap_bars=1),
                        replace(cfg, max_completion_age_bars=1)):
            self.assertEqual(detect_groups(swings, at + 2 * HOUR, HOUR, 1., changed), [])
        late = [*swings[:-1], replace(swings[-1], confirmed_at=at + HOUR)]
        self.assertEqual(detect_groups(late, at, HOUR, 1., cfg), [])
        # Even within the total duration, a head very near the left shoulder is rejected.
        asymmetric = list(swings)
        asymmetric[3] = replace(asymmetric[3], pivot_at=START + 4.5 * HOUR)
        asymmetric[4] = replace(asymmetric[4], pivot_at=START + 5 * HOUR)
        self.assertEqual(detect_groups(asymmetric, at, HOUR, 1., cfg), [])

    def test_no_cherry_picking_around_intervening_swings(self):
        swings = points([100, 98, 104, 101, 106, 101, 104.1])
        extra = TimedSwing("low", 102, START + 13 * HOUR, START + 14 * HOUR)
        self.assertEqual(detect_groups([*swings, extra], START + 15 * HOUR, HOUR, 1.,
                                       TemporalPatternConfig(family="head-shoulders")), [])

    def test_consolidation_has_two_touches_each_side_and_same_direction_prior_trend(self):
        cfg = TemporalPatternConfig(family="continuation")
        swings = points([100, 98, 104, 101, 104.1, 101.1])
        group = detect_groups(swings, START + 12 * HOUR, HOUR, 1., cfg)[0]
        self.assertEqual((group.name, group.direction), ("consolidation", 1))
        self.assertEqual(group.level(START + 80 * HOUR), 104.1)
        bad = [*swings[:-1], replace(swings[-1], price=102)]
        self.assertEqual(detect_groups(bad, START + 12 * HOUR, HOUR, 1., cfg), [])
        self.assertEqual(detect_groups(swings[-4:], START + 12 * HOUR, HOUR, 1., cfg), [])

    def test_swings_keep_extreme_and_confirmation_times_and_quote_sources(self):
        bid = [bar(0, 100, 104, 99, 103), bar(1, 103, 103.5, 100, 101), bar(2, 101, 103, 98, 102)]
        ask = [shift(c, .1) for c in bid]
        before = confirmed_swings(bid[:1], ask[:1], HOUR, 3)
        self.assertEqual(before, [])
        swings = confirmed_swings(bid, ask, HOUR, 3)
        self.assertEqual((swings[0].price, swings[0].pivot_at, swings[0].confirmed_at),
                         (104, START, START + 2 * HOUR))
        self.assertAlmostEqual(swings[1].price, 98.1)
        self.assertEqual(swings[1].confirmed_at, START + 3 * HOUR)
        self.assertEqual(confirmed_swings(bid[:2], ask[:2], HOUR, 3), swings[:1])

    def test_market_gap_splits_swing_history(self):
        bid = [bar(0, 100, 104, 99, 103), bar(1, 103, 104, 100, 101), bar(60, 101, 104, 100, 103)]
        self.assertEqual(confirmed_swings(bid, [shift(c, .1) for c in bid], HOUR, 3), [])

    def test_structural_scale_absorbs_small_legs_and_retains_new_availability(self):
        swings = points([104, 103.8, 105, 102, 102.2, 101])
        structural = structural_swings(swings, 1., .5)
        self.assertEqual([s.price for s in structural], [105, 101])
        self.assertEqual(structural[0].confirmed_at, swings[2].confirmed_at)
        self.assertEqual(structural[1].confirmed_at, swings[-1].confirmed_at)
        self.assertEqual(structural_swings(swings[:2], 1., .5), swings[:1])


class TemporalEntryTests(unittest.TestCase):
    def setUp(self):
        self.group = hs_group()
        # Express prices in ask for this sell example.
        self.bars = [bar(14, 101.2, 101.3, 100.6, 100.7),
                     bar(15, 100.9, 101.1, 100.65, 100.9),
                     bar(16, 100.9, 101.0, 100.4, 100.5)]
        self.tracker = TemporalPatternTracker(TemporalPatternConfig(head_shoulders_entry="retest"))

    def evaluate(self, ask, groups=None):
        return self.tracker.evaluate("EURUSD", shift(ask, -.1), ask, shift(ask, -.05),
                                     bar(13, 101.3, 101.4, 101.1, 101.2),
                                     bar(13, 101.4, 101.5, 101.2, 101.3), HOUR,
                                     ask.start_time + HOUR, [self.group] if groups is None else groups)

    def test_breakout_touch_and_later_confirmation_with_frozen_group_and_trailing(self):
        self.assertIsNone(self.evaluate(self.bars[0]))
        self.assertEqual(self.tracker.counts["breakout_waiting_retest"], 1)
        self.assertIsNone(self.evaluate(self.bars[1], []))
        signal = self.evaluate(self.bars[2], [])
        self.assertEqual(signal.side, SignalSide.SELL)
        self.assertAlmostEqual(signal.stop_loss_price, 101.3)
        self.assertIsNone(signal.take_profit_price)
        self.assertEqual(signal.entry_context["pattern"], "head-shoulders")
        self.assertEqual(signal.entry_context["group_span_bars"], 8)
        self.assertEqual(len(signal.entry_context["swings"]), 5)
        self.assertLess(signal.minimum_entry_price, 100.4)
        self.assertGreater(signal.maximum_entry_price, 100.4)
        self.assertFalse(self.tracker.events)

    def test_mirrored_buy_uses_bid_for_confirmation_and_stop(self):
        swings = [replace(s, kind="low" if s.kind == "high" else "high", price=210 - s.price)
                  for s in points([100, 98, 104, 101, 106, 101, 104.1])]
        group = detect_groups(swings, START + 14 * HOUR, HOUR, 1., TemporalPatternConfig())[0]
        results = []
        for ask in self.bars:
            bid = mirror(ask)
            results.append(self.tracker.evaluate("GBPUSD", bid, shift(bid, .1), shift(bid, .05),
                           mirror(bar(13, 101.4, 101.5, 101.2, 101.3)),
                           mirror(bar(13, 101.3, 101.4, 101.1, 101.2)), HOUR,
                           ask.start_time + HOUR, [group]))
        self.assertEqual(results[-1].side, SignalSide.BUY)
        self.assertAlmostEqual(results[-1].stop_loss_price, 108.7)

    def test_one_bar_cannot_break_retest_and_confirm(self):
        self.evaluate(self.bars[0])
        self.assertIsNone(self.evaluate(self.bars[0]))
        self.assertEqual(self.tracker.counts["duplicate_candle"], 1)
        self.assertIsNone(self.evaluate(self.bars[1]))
        self.assertIn("EURUSD", self.tracker.events)

    def test_expiry_and_gap_consume_setup_without_resurrection(self):
        for missing in (True, False):
            self.tracker = TemporalPatternTracker(TemporalPatternConfig(retest_max_bars=1, head_shoulders_entry="retest"))
            self.evaluate(self.bars[0])
            if not missing:
                self.evaluate(bar(15, 100.7, 100.75, 100.4, 100.5))
            self.assertIsNone(self.evaluate(bar(16, 101.2, 101.3, 100.6, 100.7)))
            self.assertFalse(self.tracker.events)
            self.assertIsNone(self.evaluate(bar(17, 101.2, 101.3, 100.6, 100.7)))
            self.assertFalse(self.tracker.events)

    def test_confirmation_deadline_is_anchored_at_first_touch(self):
        self.tracker = TemporalPatternTracker(TemporalPatternConfig(confirmation_max_bars=1, head_shoulders_entry="retest"))
        self.evaluate(self.bars[0])
        self.evaluate(self.bars[1])
        self.evaluate(replace(self.bars[1], start_time=START + 16 * HOUR))
        self.assertIsNone(self.evaluate(replace(self.bars[2], start_time=START + 17 * HOUR)))
        self.assertEqual(self.tracker.counts["event_expired"], 1)

    def test_stop_updates_with_retest_extremes_and_deep_retest_invalidates(self):
        self.evaluate(self.bars[0])
        self.assertIsNone(self.evaluate(replace(self.bars[1], high_price=101.6)))
        self.assertEqual(self.tracker.counts["event_invalidated"], 1)
        self.assertFalse(self.tracker.events)

    def test_extended_confirmation_is_consumed_and_other_symbol_is_independent(self):
        self.evaluate(self.bars[0])
        self.evaluate(self.bars[1])
        self.assertIsNone(self.evaluate(bar(16, 100.8, 100.9, 99, 99.1)))
        self.assertEqual(self.tracker.counts["entry_extension_or_target_space"], 1)
        self.assertFalse(self.tracker.events)
        self.assertEqual(self.tracker.consumed.get("GBPUSD"), None)

    def test_unconfirmed_group_cannot_trigger(self):
        late = replace(self.group, swings=(*self.group.swings[:-1],
                       replace(self.group.swings[-1], confirmed_at=START + 15 * HOUR)))
        self.assertIsNone(self.evaluate(self.bars[0], [late]))
        self.assertFalse(self.tracker.events)

    def test_recognized_figure_survives_intermediate_swings_until_fresh_breakout(self):
        forming_bar = bar(14, 102.5, 102.6, 102.1, 102.2)
        self.assertIsNone(self.evaluate(forming_bar))
        self.assertTrue(self.tracker.forming["EURUSD"])
        # No group is redetected now; the stored figure still permits its first crossing.
        self.assertIsNone(self.evaluate(replace(self.bars[0], start_time=START + 15 * HOUR), []))
        self.assertIn("EURUSD", self.tracker.events)

    def test_group_expires_before_breakout_and_weak_first_break_cannot_be_recycled(self):
        self.evaluate(bar(14, 102.5, 102.6, 102.1, 102.2))
        self.evaluate(replace(self.bars[0], start_time=START + 21 * HOUR), [])
        self.assertFalse(self.tracker.events)
        self.assertEqual(self.tracker.counts["group_expired_before_breakout"], 1)
        self.tracker = TemporalPatternTracker(TemporalPatternConfig(head_shoulders_entry="retest"))
        self.evaluate(bar(14, 100.75, 101.5, 100.6, 100.7))
        self.assertEqual(self.tracker.counts["weak_or_extended_first_breakout"], 1)
        self.evaluate(replace(self.bars[0], start_time=START + 15 * HOUR))
        self.assertFalse(self.tracker.events)

    def test_head_shoulders_neckline_confirmation_can_enter_directly_with_structural_stop(self):
        self.tracker = TemporalPatternTracker(TemporalPatternConfig())
        signal = self.evaluate(self.bars[0])
        self.assertEqual(signal.side, SignalSide.SELL)
        self.assertEqual(signal.entry_context["entry_mode"], "head-shoulders-breakout")
        self.assertIsNone(signal.entry_context["retest_at"])
        self.assertAlmostEqual(signal.stop_loss_price, 104.3)
        self.assertIsNone(signal.take_profit_price)

    def test_consolidation_always_requires_later_retest_and_reclaim_in_both_directions(self):
        swings = points([100, 98, 104, 101, 104.1, 101.1])
        original = detect_groups(swings, START + 12 * HOUR, HOUR, 1., TemporalPatternConfig())[0]
        bars = [bar(12, 103.9, 104.7, 103.8, 104.6), bar(13, 104.4, 104.6, 104., 104.3),
                bar(14, 104.35, 105., 104.2, 104.9)]
        for selling in (False, True):
            tracker = TemporalPatternTracker(TemporalPatternConfig())
            group = original
            if selling:
                group = replace(original, direction=-1, line_price=210 - original.line_price,
                                invalidation_price=210 - original.invalidation_price,
                                swings=tuple(replace(s, kind="low" if s.kind == "high" else "high",
                                                     price=210 - s.price) for s in original.swings))
            signals = []
            for b in bars:
                bid, ask = (mirror(shift(b, .1)), mirror(b)) if selling else (b, shift(b, .1))
                previous = bar(11, 103.8, 103.9, 103.5, 103.7)
                pb, pa = (mirror(shift(previous, .1)), mirror(previous)) if selling else (previous, shift(previous, .1))
                signals.append(tracker.evaluate("EURUSD", bid, ask, shift(bid, .05), pb, pa, HOUR,
                                                b.start_time + HOUR, [group]))
            self.assertEqual(signals[:2], [None, None])
            self.assertEqual(signals[-1].side, SignalSide.SELL if selling else SignalSide.BUY)
            self.assertEqual(signals[-1].entry_context["entry_mode"], "consolidation-retest")


class TemporalIntegrationTests(unittest.TestCase):
    def test_cli_builds_opt_in_strategy_and_preserves_existing_default(self):
        args = cli("--pattern-family", "head-shoulders", "--pattern-max-span-bars", "36", "--trailing-stop")
        engine = build_engine([], args)
        self.assertIsInstance(engine.strategy, TemporalPatternStrategy)
        self.assertEqual(engine.strategy.config.max_span_bars, 36)
        with patch.object(sys, "argv", ["run"]):
            self.assertEqual(parse_args().strategy, "structure")

    def test_invalid_config_and_unsupported_adaptive_fail_before_run(self):
        for args in (("--pattern-max-span-bars", "0"), ("--pattern-min-body-ratio", "nan"),
                     ("--adaptive-strategy",), ("--contextual-bandit",), ("--level-lookback-bars", "3")):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli(*args)

    def test_strategy_uses_closed_history_and_pre_breakout_fit_and_deduplicates(self):
        strategy = TemporalPatternStrategy(lookback=14, analysis_interval=HOUR)
        bids = [bar(i, 103, 104, 102, 103.2) for i in range(14)]
        asks, mids = [shift(c, .1) for c in bids], [shift(c, .05) for c in bids]
        calls = []
        def candles(symbol, **kwargs):
            calls.append(kwargs)
            return {"bid": bids, "ask": asks, "mid": mids}[kwargs["price_source"]]
        tick = Tick(START + 14 * HOUR, "EURUSD", 103, 103.1, 1, 1)
        market = SimpleNamespace(get_candles=candles, last_ticks={"EURUSD": tick})
        context = StrategyContext(Portfolio(10000), market)
        with patch("ultimateBot.backtester.strategy.temporal_pattern_strategy.confirmed_swings", return_value=[]) as extract:
            strategy.on_market_data(TickBatch("EURUSD", [tick]), context)
            strategy.on_market_data(TickBatch("EURUSD", [tick]), context)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(len(extract.call_args.args[0]), 13)
        self.assertTrue(all(c["include_active"] is False for c in calls))

    def test_busy_symbol_clears_pending_structure(self):
        strategy = TemporalPatternStrategy(lookback=14)
        strategy.tracker.events["EURUSD"] = object()
        market = SimpleNamespace(get_candles=lambda *a, **kw: [])
        strategy.on_market_data(TickBatch("EURUSD", []), StrategyContext(Portfolio(10000), market, 1))
        self.assertFalse(strategy.tracker.events)

    def test_final_partial_candle_and_late_weekend_release_cannot_create_an_entry(self):
        strategy = TemporalPatternStrategy(lookback=14, analysis_interval=HOUR)
        bids = [bar(i, 103, 104, 102, 103.2) for i in range(14)]
        market = SimpleNamespace(get_candles=lambda symbol, **kw: bids,
                                 last_ticks={"EURUSD": Tick(START + 13.5 * HOUR, "EURUSD", 103, 103.1, 1, 1)})
        context = StrategyContext(Portfolio(10000), market)
        with patch("ultimateBot.backtester.strategy.temporal_pattern_strategy.confirmed_swings") as extract:
            self.assertEqual(strategy.on_market_data(TickBatch("EURUSD", []), context), [])
            market.last_ticks["EURUSD"] = replace(market.last_ticks["EURUSD"], timestamp=START + 70 * HOUR)
            self.assertEqual(strategy.on_market_data(TickBatch("EURUSD", []), context), [])
            extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
