import io
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ultimateBot.audit_breakout_frequency import aggregate_quotes
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.strategy.regression_entry_quality import select_reentry
from ultimateBot.backtester.strategy.regression_pullback import RegressionPullbackTracker
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.swing_regression_decision_engine import resolve_swing_regression_rules
from ultimateBot.research_regression_complement import replay,next_trigger
from ultimateBot.backtester.orders.order import OrderRequest,OrderSide,OrderType
from ultimateBot.tests.test_regression_hybrid import START, INTERVAL, candle, ask_for, channel, engine_for, hybrid_config


class ReentryQualityTests(unittest.TestCase):
    def setUp(self):
        self.rules = resolve_swing_regression_rules({'regression_entry_mode':'hybrid','regression_reentry_quality_mode':'selective'})

    def signal(self,side,bars):
        return Signal('EURUSD',START,side,entry_context={'entry_mode':'failed-breakout','reentry_bars':bars})

    def test_nextbar_needs_no_wick_condition_and_keeps_fixed_exits(self):
        signal = replace(self.signal(SignalSide.BUY,1),stop_loss_price=99,take_profit_price=105)
        accepted,reason = select_reentry(signal,candle(open=109,high=110,low=100,close=109.1),self.rules)
        self.assertIsNone(reason)
        self.assertEqual((accepted.stop_loss_price,accepted.take_profit_price),(99,105))
        self.assertEqual(accepted.entry_context['reentry_quality_case'],'next_bar')

    def test_samebar_uses_the_correct_wick_for_long_and_short(self):
        for side,mid in [(SignalSide.BUY,candle(open=101,high=110,low=100,close=108)),
                         (SignalSide.SELL,candle(open=109,high=110,low=100,close=102))]:
            accepted,reason = select_reentry(self.signal(side,0),mid,self.rules)
            self.assertIsNone(reason)
            self.assertAlmostEqual(accepted.entry_context['signal_stopside_wick'],.1)
        self.assertEqual(select_reentry(self.signal(SignalSide.BUY,0),candle(open=108,high=110,low=100,close=109),self.rules)[1],
                         'reentry_quality_same_bar_wick')

    def test_late_confirmation_and_invalid_candle_rejected(self):
        for bars in (2,3):
            self.assertEqual(select_reentry(self.signal(SignalSide.BUY,bars),candle(),self.rules)[1],'reentry_quality_late')
        self.assertEqual(select_reentry(self.signal(SignalSide.BUY,1),candle(high=float('nan')),self.rules)[1],'reentry_quality_invalid_candle')

    def test_legacy_and_other_branches_are_unchanged(self):
        signal = self.signal(SignalSide.BUY,3)
        rules = {**self.rules,'regression_reentry_quality_mode':'legacy'}
        self.assertIs(select_reentry(signal,candle(),rules)[0],signal)
        signal = replace(signal,entry_context={'entry_mode':'breakout-retest'})
        self.assertIs(select_reentry(signal,candle(),self.rules)[0],signal)

    def test_rejected_confirmation_consumes_both_setups_and_is_counted(self):
        from ultimateBot.tests.test_regression_hybrid import HybridStrategyTests
        fixture = HybridStrategyTests()
        fixture.setUp()
        strategy = fixture.strategy
        strategy.regression_rules['regression_reentry_quality_mode'] = 'selective'
        fixture.bid[-1] = candle(15,open=105,high=112,low=104,close=109)
        with patch.object(strategy,'_get_structure_candles',side_effect=fixture.circuits), \
                patch.object(strategy,'_evaluate_regression_pipeline',return_value=(None,deepcopy(channel().analysis))):
            self.assertEqual(strategy.on_market_data(fixture.batch(),fixture.context),[])
        self.assertFalse(strategy._reentry.events or strategy._breakout.events)
        self.assertEqual(strategy.diagnostics_summary()['reason_counts']['reentry_quality_same_bar_wick'],1)


class PullbackTests(unittest.TestCase):
    def setUp(self):
        self.rules = resolve_swing_regression_rules({'regression_entry_mode':'hybrid','regression_lookback_bars':15,
                                                    'regression_pullback_enabled':True})
        self.analysis = deepcopy(channel().analysis)
        self.analysis.update(parallelism=1.,regression_lookback_bars=15)
        for key in ('anchored_line','floating_line'):
            self.analysis[key]['slope'] = .5
        self.channel = FrozenChannel(self.analysis,START-INTERVAL,109,109.1)
        self.touch = candle(0,open=109,high=110,low=107.4,close=109.7)
        self.confirm = candle(1,open=109.7,high=111,low=108.8,close=110.9)

    def test_later_reclaim_confirms_both_directions_without_a_fixed_target(self):
        for selling in (False,True):
            tracker = RegressionPullbackTracker()
            frozen = deepcopy(self.channel)
            quotes = [(bar,ask_for(bar)) for bar in (self.touch,self.confirm)]
            if selling:
                for key in ('anchored_line','floating_line'):
                    line = frozen.analysis[key]
                    line.update(intercept=220-line['intercept'],slope=-.5,kind='high' if line['kind']=='low' else 'low')
                frozen = replace(frozen,previous_bid_close=110.9,previous_ask_close=111.)
                quotes = [(replace(ask,open_price=220-ask.open_price,high_price=220-ask.low_price,
                                    low_price=220-ask.high_price,close_price=220-ask.close_price),
                           replace(bid,open_price=220-bid.open_price,high_price=220-bid.low_price,
                                    low_price=220-bid.high_price,close_price=220-bid.close_price)) for bid,ask in quotes]
            first,analysis = tracker.evaluate('EURUSD',*quotes[0],START+INTERVAL,INTERVAL,self.rules,frozen)
            self.assertIsNone(first)
            signal,_ = tracker.evaluate('EURUSD',*quotes[1],START+2*INTERVAL,INTERVAL,self.rules,None)
            self.assertIsNotNone(signal)
            self.assertEqual(signal.side,SignalSide.SELL if selling else SignalSide.BUY)
            self.assertIsNone(signal.take_profit_price)
            self.assertLess(signal.minimum_entry_price,signal.maximum_entry_price)
            self.assertFalse(tracker.events)

    def test_nondirectional_channel_missing_history_and_deep_touch_are_rejected(self):
        tracker = RegressionPullbackTracker()
        flat = deepcopy(self.channel)
        flat.analysis['floating_line']['slope'] = -.5
        self.assertIsNone(tracker.evaluate('EURUSD',self.touch,ask_for(self.touch),START,INTERVAL,self.rules,flat)[0])
        signal,analysis = tracker.evaluate('EURUSD',replace(self.touch,low_price=105),ask_for(replace(self.touch,low_price=105)),START,INTERVAL,self.rules,self.channel)
        self.assertIsNone(signal)
        self.assertEqual(analysis['reasons'],['pullback_too_deep'])
        tracker.evaluate('EURUSD',self.touch,ask_for(self.touch),START,INTERVAL,self.rules,self.channel)
        gap = replace(self.confirm,start_time=START+3*INTERVAL)
        self.assertEqual(tracker.evaluate('EURUSD',gap,ask_for(gap),START,INTERVAL,self.rules,None)[1]['reasons'],['pullback_history_gap'])
        self.assertFalse(tracker.events)

    def test_primary_entry_keeps_priority_over_an_opposite_pullback_confirmation(self):
        from ultimateBot.backtester.strategy.regression_hybrid import RegressionHybridTracker
        from ultimateBot.backtester.strategy.regression_reentry import RegressionReentryTracker
        from ultimateBot.backtester.strategy.regression_breakout import RegressionBreakoutTracker
        primary,other,pullback = RegressionReentryTracker(),RegressionBreakoutTracker(),RegressionPullbackTracker()
        hybrid = RegressionHybridTracker(primary,other,pullback)
        sell = Signal('EURUSD',START,SignalSide.SELL,entry_context={'entry_mode':'failed-breakout'})
        buy = Signal('EURUSD',START,SignalSide.BUY,entry_context={'entry_mode':'trend-pullback'})
        with patch.object(primary,'evaluate',return_value=(sell,dict(direction='sell',reasons=[]))), \
             patch.object(other,'evaluate',return_value=(None,dict(direction='flat',reasons=['waiting']))), \
             patch.object(pullback,'evaluate',return_value=(buy,dict(direction='buy',reasons=[]))):
            signal,_ = hybrid.evaluate('EURUSD',self.touch,ask_for(self.touch),START,INTERVAL,self.rules,self.channel)
        self.assertEqual(signal.side,SignalSide.SELL)
        self.assertEqual(hybrid.suppressed_complementary,1)

    def test_pending_pullback_keeps_its_original_channel_and_expires_without_extension(self):
        tracker = RegressionPullbackTracker()
        tracker.evaluate('EURUSD',self.touch,ask_for(self.touch),START,INTERVAL,self.rules,self.channel)
        changed = deepcopy(self.channel)
        changed.analysis['atr'] = 200
        signal,_ = tracker.evaluate('EURUSD',self.confirm,ask_for(self.confirm),START,INTERVAL,self.rules,changed)
        self.assertEqual(signal.entry_context['atr'],2.)
        tracker = RegressionPullbackTracker()
        tracker.evaluate('EURUSD',self.touch,ask_for(self.touch),START,INTERVAL,self.rules,self.channel)
        for index in range(1,5):
            bar = candle(index,open=109.8,high=110,low=109.5,close=109.8)
            signal,analysis = tracker.evaluate('EURUSD',bar,ask_for(bar),START,INTERVAL,self.rules,None)
        self.assertEqual(analysis['reasons'],['pullback_expired'])
        self.assertFalse(tracker.events)

    def test_range_bounce_uses_a_flat_channel_and_no_failed_breakout_excursion(self):
        rules = {**self.rules,'regression_pullback_regime':'range'}
        frozen = channel()
        frozen.analysis['parallelism'] = 1.
        touch = candle(0,open=102,high=103,low=100.2,close=102.5)
        confirm = candle(1,open=102.5,high=104,low=101,close=103.8)
        tracker = RegressionPullbackTracker('channel-bounce')
        self.assertIsNone(tracker.evaluate('EURUSD',touch,ask_for(touch),START,INTERVAL,rules,frozen)[0])
        signal,_ = tracker.evaluate('EURUSD',confirm,ask_for(confirm),START,INTERVAL,rules,None)
        self.assertEqual(signal.entry_context['entry_mode'],'channel-bounce')
        self.assertEqual(signal.side,SignalSide.BUY)
        self.assertIsNone(signal.take_profit_price)
        tracker = RegressionPullbackTracker('channel-bounce')
        outside = replace(touch,low_price=99.7)
        _,analysis = tracker.evaluate('EURUSD',outside,ask_for(outside),START,INTERVAL,rules,frozen)
        self.assertEqual(analysis['reasons'],['pullback_reserved_for_failed_breakout'])
        self.assertFalse(tracker.events)


class SubmissionTests(unittest.TestCase):
    def test_launcher_prioritizes_frequency_and_experiment_recipes_stay_frozen(self):
        from ultimateBot.run_regression_hybrid import hybrid_config as selective_config
        from ultimateBot.research_regression_complement import build_variant
        config = selective_config()
        self.assertEqual(config.reentry.quality_mode,'selective')
        self.assertEqual(config.regression.max_spread_stop_ratio,0.)
        self.assertFalse(config.pullback.enabled or config.impulse.enabled)
        for name,cost,quality in [('original',0.,'legacy'),('quality',0.,'selective'),
                                  ('spread',.1,'legacy'),('filtered',.1,'selective')]:
            rules = build_variant(name).strategy.regression_rules
            self.assertEqual(rules['regression_max_spread_stop_ratio'],cost)
            self.assertEqual(rules['regression_reentry_quality_mode'],quality)

    def test_live_spread_is_checked_against_final_planned_risk_before_order_creation(self):
        from ultimateBot.tests.test_regression_hybrid import HybridEngineTests
        fixture = HybridEngineTests()
        fixture.setUp()
        engine = fixture.engine
        engine.strategy.regression_rules['regression_max_spread_stop_ratio'] = .1
        engine._market_clock = START+timedelta(seconds=1)
        engine.market_state.last_ticks['EURUSD'] = Tick(engine._market_clock,'EURUSD',109,110,1,1)
        with redirect_stdout(io.StringIO()):
            engine._process_signal(fixture.signal,fixture.tick)
        self.assertEqual(len(engine.pending_orders_processor.queue),0)
        self.assertEqual(engine.order_builder._next_order_id,1)
        self.assertEqual(engine.entry_rejection_counts[('failed-breakout','spread_stop_ratio')],1)

    def test_python_config_and_cli_roundtrip_for_both_new_options(self):
        config = hybrid_config()
        config = replace(config,reentry=replace(config.reentry,quality_mode='selective'),
                         regression=replace(config.regression,max_spread_stop_ratio=.1),
                         pullback=replace(config.pullback,enabled=True))
        engine = engine_for(config)
        self.assertTrue(engine.strategy.config.pullback.enabled)
        self.assertEqual(engine.strategy.regression_rules['regression_reentry_quality_mode'],'selective')
        self.assertEqual(engine.strategy.regression_rules['regression_max_spread_stop_ratio'],.1)
        engine = engine_for(config,'--no-regression-pullback-enabled','--regression-reentry-quality-mode','legacy')
        self.assertNotIn('trend-pullback',engine.strategy._hybrid.branches)
        self.assertEqual(engine.strategy.config.reentry.quality_mode,'legacy')


class TriggerReplayTests(unittest.TestCase):
    def test_limit_eligibility_and_expiration_precede_fill_at_the_deadline(self):
        start = round(START.timestamp()*1000)
        quotes = np.array([[start+offset,ask-.1,ask,1,1] for offset,ask in [(0,99),(10,99),(11,101),(49,99),(50,99)]])
        for expiry in (False,True):
            engine = engine_for()
            order = OrderRequest(1,'EURUSD',START,START,START+timedelta(milliseconds=10),OrderSide.BUY,1,
                order_type=OrderType.LIMIT,limit_price=100,stop_loss=98,take_profit=105,
                expires_at=START+timedelta(milliseconds=50),trailing_stop_enabled=False)
            engine.pending_orders_processor.queue.add(order)
            values = quotes.copy()
            if expiry:
                values[3,1:3] = [100.9,101]
            index = next_trigger(engine,'EURUSD',values,0)
            self.assertEqual(index,4 if expiry else 3)
            tick = Tick(START+timedelta(milliseconds=50 if expiry else 49),'EURUSD',*values[index,1:])
            with redirect_stdout(io.StringIO()):
                engine._process_pending_orders(tick)
            self.assertEqual(len(engine.portfolio.positions),0 if expiry else 1)
            self.assertEqual(len(engine.pending_orders_processor.queue),0)

    def test_original_tick_jumps_match_full_engine_including_trailing_and_excursions(self):
        ticks = []
        patterns = {15:((0,105),(10,112),(20,104),(50,109)),
                    16:((0,109),(5,109),(15,104.9),(50,105)),
                    17:((0,109.4),(10,111.1),(20,109.2),(50,111)),
                    18:((0,110.4),(10,110.6),(20,109.8),(50,110.3)),
                    19:((0,110.3),(10,111.2),(20,110.25),(50,111.05)),
                    20:((0,111.05),(5,111.05),(10,114),(20,119),(50,114.6))}
        for index in range(21):
            for sec,price in patterns.get(index,((0,105),(10,108),(20,102),(50,105))):
                ticks.append(Tick(START+index*INTERVAL+timedelta(seconds=sec),'EURUSD',price,price+.1,1,1))
        symbols = ['EURUSD','GBPUSD']
        ticks += [replace(tick,symbol='GBPUSD') for tick in list(ticks)]
        ticks.sort(key=lambda tick:(tick.timestamp,symbols.index(tick.symbol)))
        frame = pd.DataFrame([dict(timestamp=round(t.timestamp.timestamp()*1000),bid=t.bid,ask=t.ask,
                                   bid_volume=1.,ask_volume=1.,symbol=t.symbol) for t in ticks])
        bars = {symbol:aggregate_quotes(frame.loc[frame.symbol.eq(symbol)],60_000) for symbol in symbols}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for symbol in symbols:
                path = root/symbol/'2024'/'01'/'2024-01-02.tsv'
                path.parent.mkdir(parents=True)
                path.write_text(''.join(f'QUOTE\t{symbol}\t{round(t.timestamp.timestamp()*1000)}\t0\t{i}\t{t.bid}\t{t.ask}\t1\t1\n'
                                       for i,t in enumerate(ticks) if t.symbol == symbol))
            full,jumped,lazy = engine_for(),engine_for(),engine_for()
            full.feed = ticks
            fit = lambda *args,**kwargs:(None,deepcopy(channel().analysis))
            with patch.object(full.strategy,'_evaluate_regression_pipeline',side_effect=fit),redirect_stdout(io.StringIO()):
                full.run()
            with patch.object(jumped.strategy,'_evaluate_regression_pipeline',side_effect=fit),redirect_stdout(io.StringIO()):
                result = replay({'test':jumped},root,symbols,date(2024,1,2),date(2024,1,2),bars,root/'replay.log')
            with patch.object(lazy.strategy,'_evaluate_regression_pipeline',side_effect=fit),redirect_stdout(io.StringIO()):
                replay({'test':lazy},root,symbols,date(2024,1,2),date(2024,1,2),bars,root/'lazy.log',lazy_quotes=True)
        self.assertEqual(len(full.reports.trades),4)
        self.assertEqual(jumped.reports.trades,full.reports.trades)
        self.assertEqual(lazy.reports.trades,full.reports.trades)
        self.assertLess(result['test']['processed_quote_events'],len(ticks))


if __name__ == '__main__':
    unittest.main()
