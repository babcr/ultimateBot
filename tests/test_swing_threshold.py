from datetime import datetime, timedelta, timezone
import unittest

from ultimateBot.backtester.strategy.swing_threshold import DailyTradeTarget, DailyTradeTargetConfig, daily_target_config


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class DailyTradeTargetTests(unittest.TestCase):
    def test_minimum_stops_decreases_across_periods_but_allows_later_increases(self):
        policy = DailyTradeTarget(.3, DailyTradeTargetConfig(1, .1), min_threshold=.25)
        policy.advance(START)
        policy.record(False)
        closed = policy.advance(START+timedelta(days=1))
        self.assertTrue(closed['lowered'])
        self.assertEqual(policy.threshold, .25)
        for day in range(2, 6):
            policy.record(False)
            closed = policy.advance(START+timedelta(days=day))
            self.assertFalse(closed['lowered'])
            self.assertEqual(closed['min_threshold'], .25)
            self.assertEqual(policy.threshold, .25)
        policy.record(True)
        policy.record(True)
        self.assertAlmostEqual(policy.threshold, .35)
        self.assertEqual(policy.summary()['min_threshold'], .25)

    def test_minimum_clamps_initial_value_and_validates_target_scale(self):
        config = DailyTradeTargetConfig(.25, .1)
        for minimum, probability in ((-.2, False), (8., False), (.4, True)):
            policy = DailyTradeTarget(minimum-1, config, min_threshold=minimum, probability=probability)
            self.assertEqual(policy.threshold, minimum)
        for minimum, probability in ((float('nan'), False), (float('inf'), False),
                                      (-.1, True), (1.1, True)):
            with self.assertRaises(ValueError):
                DailyTradeTarget(.25, config, min_threshold=minimum, probability=probability)

    def test_increase_after_each_extra_acceptance_only_affects_later_decisions(self):
        policy = DailyTradeTarget(.25,DailyTradeTargetConfig(2,.1))
        policy.advance(START)
        used = []
        for prediction in (.5,.5,.5,.3,.5):
            used.append(policy.threshold)
            policy.record(prediction >= policy.threshold)
        for actual,expected in zip(used,(.25,.25,.35,.45,.45)):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(policy.accepted,4)
        self.assertEqual(policy.scored,5)
        self.assertEqual(policy.raised_steps,3)
        self.assertAlmostEqual(policy.threshold,.55)

    def test_under_target_lowers_once_at_next_day_exact_or_over_target_does_not(self):
        policy = DailyTradeTarget(.25,DailyTradeTargetConfig(1,.1))
        policy.advance(START)
        for accepted in (True,True):
            policy.record(accepted)
        closed = policy.advance(START+timedelta(days=1))
        self.assertEqual(closed['accepted'],2)
        self.assertFalse(closed['lowered'])
        self.assertAlmostEqual(policy.threshold,.35)
        for _ in range(4):
            policy.record(False)
        closed = policy.advance(START+timedelta(days=2))
        self.assertTrue(closed['lowered'])
        self.assertEqual(closed['accepted'],0)
        self.assertAlmostEqual(policy.threshold,.25)
        policy.record(True)
        closed = policy.advance(START+timedelta(days=3))
        self.assertFalse(closed['lowered'])
        self.assertEqual(closed['raised_steps'],0)
        self.assertAlmostEqual(policy.threshold,.25)

    def test_missing_days_and_context_warmup_do_not_manufacture_adjustments(self):
        policy = DailyTradeTarget(.25,DailyTradeTargetConfig(1,.1))
        policy.advance(START)
        closed = policy.advance(START+timedelta(days=1))
        self.assertTrue(closed['no_context'])
        self.assertAlmostEqual(policy.threshold,.25)
        policy.record(False)
        policy.advance(START+timedelta(days=5))
        self.assertAlmostEqual(policy.threshold,.15)
        self.assertEqual(len(policy.history),2)
        self.assertIsNone(policy.advance(START+timedelta(days=4)))
        self.assertEqual(policy.day,(START+timedelta(days=5)).date())

    def test_day_uses_utc_and_probability_threshold_is_bounded(self):
        policy = DailyTradeTarget(.95,DailyTradeTargetConfig(1,.1),probability=True)
        policy.advance(START)
        policy.record(True)
        policy.record(True)
        policy.record(True)
        self.assertEqual(policy.threshold,1.)
        self.assertEqual(policy.raised_steps,1)
        same_day=datetime(2026,1,2,1,tzinfo=timezone(timedelta(hours=2)))
        self.assertIsNone(policy.advance(same_day))
        policy.advance(START+timedelta(days=1))
        policy.record(False)
        policy.advance(START+timedelta(days=2))
        self.assertAlmostEqual(policy.threshold,.9)
        low = DailyTradeTarget(.05,DailyTradeTargetConfig(1,.1),probability=True)
        low.advance(START)
        low.record(False)
        low.advance(START+timedelta(days=1))
        self.assertEqual(low.threshold,0.)

    def test_invalid_or_incomplete_configuration_is_rejected(self):
        self.assertIsNone(daily_target_config())
        for target,step in ((None,.1),(0,.1),(-1,.1),(True,.1),(float('nan'),.1),(float('inf'),.1),
                            (1e-300,.1),(1e300,.1),(2,0),(2,-.1),(2,float('nan')),(2,float('inf'))):
            with self.subTest(target=target,step=step),self.assertRaises(ValueError):
                daily_target_config(target,step)

    def test_fractional_cadence_accumulates_until_four_day_boundary_and_uses_default_step(self):
        config = daily_target_config(.25)
        self.assertEqual(config.period_days,4.)
        self.assertEqual(config.threshold_step,.001)
        policy = DailyTradeTarget(.25,config)
        policy.advance(START)
        policy.record(False)
        for day in (1,2,3):
            self.assertIsNone(policy.advance(START+timedelta(days=day)))
            self.assertEqual(policy.scored,1)
            self.assertEqual(policy.threshold,.25)
        closed = policy.advance(START+timedelta(days=4))
        self.assertEqual(closed['target'],1)
        self.assertTrue(closed['lowered'])
        self.assertAlmostEqual(policy.threshold,.249)
        policy.record(True)
        policy.advance(START+timedelta(days=7))
        self.assertEqual(policy.accepted,1)
        policy.record(True)
        self.assertAlmostEqual(policy.threshold,.25)
        closed = policy.advance(START+timedelta(days=8))
        self.assertEqual(closed['accepted'],2)
        self.assertFalse(closed['lowered'])

    def test_reciprocal_period_applies_above_one_and_does_not_round_days(self):
        for rate,hours in ((2.,12),(1.5,16),(.3,80)):
            with self.subTest(rate=rate):
                policy = DailyTradeTarget(.25,DailyTradeTargetConfig(rate))
                policy.advance(START+timedelta(minutes=10))
                policy.record(False)
                end = START+timedelta(hours=hours)
                self.assertEqual(policy.period_end,end)
                self.assertIsNone(policy.advance(end-timedelta(microseconds=1)))
                self.assertEqual(policy.threshold,.25)
                closed = policy.advance(end)
                self.assertTrue(closed['lowered'])
                self.assertAlmostEqual(policy.threshold,.249)
                self.assertEqual(policy.period_start,end)
                self.assertEqual(policy.period_end,end+timedelta(hours=hours))
        late = DailyTradeTarget(.25,DailyTradeTargetConfig(2))
        late.advance(START+timedelta(hours=18))
        self.assertEqual(late.period_start,START+timedelta(hours=12))
        self.assertEqual(late.period_end,START+timedelta(days=1))
        late.record(False)
        self.assertIsNone(late.advance(START+timedelta(hours=19)))
        self.assertEqual(late.threshold,.25)


if __name__ == '__main__':
    unittest.main()
