from copy import deepcopy
from dataclasses import replace
import unittest

from ultimateBot.backtester.strategy.regression_impulse import RegressionImpulseTracker
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel
from ultimateBot.backtester.strategy.signal import SignalSide
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.tests.test_regression_hybrid import START,INTERVAL,candle,ask_for,channel,engine_for


class ImpulseTests(unittest.TestCase):
    def setUp(self):
        self.rules = resolve_swing_regression_rules({'regression_entry_mode':'hybrid','regression_impulse_enabled':True})
        self.frozen = channel()
        self.frozen.analysis['floating_line']['intercept'] = 106.
        self.frozen.analysis.update(market_volatility_atr=1.2,shared_slope=0.)
        self.breakout = candle(0,open=105.7,high=106.8,low=105.6,close=106.7)
        self.hold = candle(1,open=106.7,high=107.5,low=106.4,close=107.4)

    def test_two_closed_candles_confirm_buy_and_sell_without_retest_or_fixed_target(self):
        for selling in (False,True):
            frozen = deepcopy(self.frozen)
            quotes = [(bar,ask_for(bar)) for bar in (self.breakout,self.hold)]
            if selling:
                for key in ('anchored_line','floating_line'):
                    line = frozen.analysis[key]
                    line.update(intercept=220-line['intercept'],kind='low' if line['kind']=='high' else 'high')
                frozen = replace(frozen,previous_bid_close=114.9,previous_ask_close=115.)
                quotes = [(replace(ask,open_price=220-ask.open_price,high_price=220-ask.low_price,
                                    low_price=220-ask.high_price,close_price=220-ask.close_price),
                           replace(bid,open_price=220-bid.open_price,high_price=220-bid.low_price,
                                    low_price=220-bid.high_price,close_price=220-bid.close_price)) for bid,ask in quotes]
            tracker = RegressionImpulseTracker()
            self.assertIsNone(tracker.evaluate('EURUSD',*quotes[0],START,INTERVAL,self.rules,frozen)[0])
            signal,_ = tracker.evaluate('EURUSD',*quotes[1],START,INTERVAL,self.rules,None)
            self.assertEqual(signal.side,SignalSide.SELL if selling else SignalSide.BUY)
            self.assertEqual(signal.entry_context['entry_mode'],'breakout-hold')
            self.assertIsNone(signal.take_profit_price)
            self.assertAlmostEqual(signal.stop_loss_price,114.4 if selling else 105.6)
            self.assertLess(signal.minimum_entry_price,signal.maximum_entry_price)
            self.assertFalse(tracker.events)

    def test_failed_hold_is_consumed_and_cannot_confirm_on_a_later_candle(self):
        tracker = RegressionImpulseTracker()
        tracker.evaluate('EURUSD',self.breakout,ask_for(self.breakout),START,INTERVAL,self.rules,self.frozen)
        touched = replace(self.hold,low_price=105.8)
        signal,analysis = tracker.evaluate('EURUSD',touched,ask_for(touched),START,INTERVAL,self.rules,None)
        self.assertIsNone(signal)
        self.assertEqual(analysis['reasons'],['impulse_hold_not_confirmed'])
        self.assertFalse(tracker.events)
        self.assertIsNone(tracker.evaluate('EURUSD',self.hold,ask_for(self.hold),START,INTERVAL,self.rules,None)[0])

    def test_compression_and_causal_history_are_required(self):
        tracker = RegressionImpulseTracker()
        expanded = deepcopy(self.frozen)
        expanded.analysis['market_volatility_atr'] = 2.
        self.assertEqual(tracker.evaluate('EURUSD',self.breakout,ask_for(self.breakout),START,INTERVAL,self.rules,expanded)[1]['reasons'],['impulse_no_compression'])
        tracker.evaluate('EURUSD',self.breakout,ask_for(self.breakout),START,INTERVAL,self.rules,self.frozen)
        gap = replace(self.hold,start_time=START+2*INTERVAL)
        self.assertEqual(tracker.evaluate('EURUSD',gap,ask_for(gap),START,INTERVAL,self.rules,None)[1]['reasons'],['impulse_history_gap'])
        self.assertFalse(tracker.events)

    def test_cli_roundtrip_and_disabled_default(self):
        self.assertNotIn('breakout-hold',engine_for().strategy._hybrid.branches)
        engine = engine_for(None,'--regression-impulse-enabled')
        self.assertIn('breakout-hold',engine.strategy._hybrid.branches)
        self.assertTrue(engine.strategy.config.impulse.enabled)
        with self.assertRaises(ValueError):
            resolve_swing_regression_rules({'regression_entry_mode':'trend','regression_impulse_enabled':True})


if __name__ == '__main__':
    unittest.main()
