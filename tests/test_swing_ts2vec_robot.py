from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, SWING_SCHEMA, SwingSeries, SwingWindow
from ultimateBot.backtester.strategy.strategy_context import StrategyContext
from ultimateBot.backtester.strategy.swing_ts2vec_strategy import SwingTS2VecStrategy
from ultimateBot.run_backtest import build_engine, parse_args


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def metadata(directory, *, window='pairs_1', available=START, delay=0.):
    values = dict(version=1, feature_schema=FEATURE_SCHEMA, swing_schema=SWING_SCHEMA,
                  opportunity_schema=OPPORTUNITY_SCHEMA, window_key=window, interval_seconds=3600,
                  opportunity_config=dict(interval_seconds=3600,trailing_start_r=1.,stop_buffer_pips=2.,
                                          order_expiration_bars=2,execution_delay_seconds=delay),
                  available_after=available.isoformat(), threshold=.25,target='pnl-r',
                  resolver_id='resolver',embedding_model_id='encoder',scope=dict(symbols=['EURUSD','GBPUSD']))
    (directory/'metadata.json').write_text(json.dumps(values),encoding='utf-8')
    return values


def bars(count=7):
    result = []
    for i in range(count):
        o, c = (1.1000,1.1020) if i%2 == 0 else (1.1020,1.1000)
        result.append(Candle(START+i*HOUR,o,1.1030,1.0990,c))
    return result


def fake_resolver(accepted=True):
    def predict(points, *, as_of, symbol):
        return dict(prediction=.5 if accepted else -.5,threshold=.25,target='pnl-r',
                    accepted=accepted,side=points[-1].side,resolver_id='resolver')
    return SimpleNamespace(predict=predict)


def cli(directory, *options):
    with patch.object(sys,'argv',['run_backtest','--strategy','swing-ts2vec','--swing-ts2vec-model',str(directory),*options]):
        return parse_args()


class SwingRobotTests(unittest.TestCase):
    def test_stop_buffer_override_changes_buy_and_sell_risk_and_both_exit_modes(self):
        for selling in (False, True):
            for fixed_tp in (False, True):
                with self.subTest(selling=selling, fixed_tp=fixed_tp), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
                    directory = Path(tmp)
                    metadata(directory)
                    original = (directory/'metadata.json').read_bytes()
                    extra = ['--swing-ts2vec-take-profit-r', '2'] if fixed_tp else []
                    with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=fake_resolver()):
                        engine = build_engine([], cli(directory, '--max-spread-ratio', '10000', '--risk-percent', '1',
                            '--balance', '10000', '--max-positions', '1', '--swing-ts2vec-stop-buffer-pips', '5',
                            '--swing-ts2vec-decision-log', 'all', *extra))
                    feed_candles(engine, 4, selling=selling)
                    position = next(iter(engine.portfolio.positions.values()))
                    direction = -1 if selling else 1
                    self.assertAlmostEqual(position.stop_loss, 1.1025 if selling else 1.0996)
                    distance = abs(position.entry_price-position.stop_loss)
                    self.assertAlmostEqual(position.quantity*distance, 100.)
                    if fixed_tp:
                        self.assertAlmostEqual(position.take_profit, position.entry_price+direction*2*distance)
                        self.assertFalse(position.trailing_stop_enabled)
                    else:
                        self.assertIsNone(position.take_profit)
                        self.assertAlmostEqual(position.trailing_stop_activation_price, position.entry_price+direction*distance)
                    self.assertEqual(engine.strategy.trained_stop_buffer_pips, 2.)
                    self.assertEqual(engine.strategy.opportunity_config.stop_buffer_pips, 5.)
                    self.assertIn('"stop_buffer_pips": 5.0', output.getvalue())
                    self.assertIn('"trained_stop_buffer_pips": 2.0', output.getvalue())
                    self.assertEqual((directory/'metadata.json').read_bytes(), original)

    def test_zero_stop_buffer_removes_model_margin_and_invalid_cli_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            metadata(directory)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver'):
                strategy = build_engine([], cli(directory, '--swing-ts2vec-stop-buffer-pips', '0')).strategy
            self.assertEqual(strategy.opportunity_config.stop_buffer_pips, 0.)
            for side in (-1, 1):
                self.assertEqual(strategy.opportunity_config.stop_for(SimpleNamespace(price=1.1, side=side), 'EURUSD'), 1.1)
            for value in ('-1', 'nan', 'inf', '-inf'):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli(directory, '--swing-ts2vec-stop-buffer-pips='+value)
            with patch.object(sys, 'argv', ['run_backtest', '--swing-ts2vec-stop-buffer-pips', '2']), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_minimum_threshold_changes_execution_and_preserves_saved_model(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
                directory = Path(tmp)
                metadata(directory)
                original = (directory/'metadata.json').read_bytes()
                cadence = ['--swing-ts2vec-target-trades-per-day', '1'] if adaptive else []
                with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=fake_resolver()):
                    engine = build_engine([], cli(directory, '--swing-ts2vec-threshold', '.1',
                        '--swing-ts2vec-min-threshold', '.6', '--swing-ts2vec-decision-log', 'all', *cadence))
                feed_candles(engine, 3)
                strategy = engine.strategy
                self.assertEqual(strategy.initial_threshold, .6)
                self.assertEqual(strategy.initial_threshold_source, 'minimum')
                self.assertEqual(strategy.counts['accepted'], 0)
                self.assertGreater(strategy.counts['rejected_by_model'], 0)
                if adaptive:
                    for day in range(1, 5):
                        strategy.daily_target.record(False)
                        strategy.daily_target.advance(START+timedelta(days=day))
                        self.assertEqual(strategy.daily_target.threshold, .6)
                self.assertIn('"min_threshold": 0.6', output.getvalue())
                self.assertEqual((directory/'metadata.json').read_bytes(), original)

    def test_minimum_threshold_cli_validation_for_probabilities_and_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            values = metadata(directory)
            for target in ('pnl-r', 'trailing-step-raw', 'trailing-step-square', 'trailing-step-exp', 'win', 'trailing-step-1'):
                values['target'] = target
                (directory/'metadata.json').write_text(json.dumps(values))
                valid = (0., .5, 1.) if target in ('win', 'trailing-step-1') else (-2., 0., 8.)
                for minimum in valid:
                    args = cli(directory, f'--swing-ts2vec-min-threshold={minimum}')
                    with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver'):
                        strategy = build_engine([], args).strategy
                    self.assertEqual(strategy.threshold, max(.25, minimum))
                invalid = ('nan', 'inf', '-inf')
                if target in ('win', 'trailing-step-1'):
                    invalid += ('-.1', '1.1')
                for minimum in invalid:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        cli(directory, f'--swing-ts2vec-min-threshold={minimum}')
            with patch.object(sys, 'argv', ['run_backtest', '--swing-ts2vec-min-threshold', '0']), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_dynamic_threshold_pools_symbols_counts_acceptances_before_risk_and_logs_used_threshold(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as output:
            directory = Path(tmp)
            metadata(directory)
            original = (directory/'metadata.json').read_bytes()
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1',
                    '--max-positions','1','--swing-ts2vec-target-trades-per-day','1',
                    '--swing-ts2vec-threshold-step','1','--swing-ts2vec-decision-log','all'))
            ticks = []
            for bid in bars(4):
                for seconds,price in ((1,bid.open_price),(3599,bid.close_price)):
                    for index,symbol in enumerate(('EURUSD','GBPUSD')):
                        ticks.append(Tick(bid.start_time+timedelta(seconds=seconds,microseconds=index),symbol,price,price+.0001,1,1))
            for index,symbol in enumerate(('EURUSD','GBPUSD')):
                ticks.append(Tick(START+4*HOUR+timedelta(milliseconds=1,microseconds=index),symbol,1.1,1.1001,1,1))
            for tick in sorted(ticks,key=lambda t:t.timestamp):
                engine.on_tick(tick)
            self.assertEqual(engine.strategy.counts['accepted'],2)
            self.assertEqual(engine.strategy.counts['rejected_by_model'],2)
            self.assertEqual(engine.strategy.counts['rejected_by_risk'],1)
            self.assertEqual(len(engine.portfolio.positions),1)
            self.assertEqual(engine.strategy.daily_target.accepted,2)
            self.assertAlmostEqual(engine.strategy.threshold,1.25)
            decisions = [json.loads(line.split('] ',1)[1]) for line in output.getvalue().splitlines()
                         if line.startswith('[swing-ts2vec-decision]')]
            self.assertEqual([r['threshold'] for r in decisions],[.25,.25,1.25,1.25])
            self.assertTrue(all(r['threshold_source']=='adaptive' for r in decisions))
            self.assertEqual((directory/'metadata.json').read_bytes(),original)

    def test_dynamic_threshold_stays_inactive_before_model_availability_and_validates_cli_pair(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory,available=START+timedelta(days=2))
            for options in (('--swing-ts2vec-target-trades-per-day','nan'),('--swing-ts2vec-threshold-step','.1'),
                            ('--swing-ts2vec-target-trades-per-day','0','--swing-ts2vec-threshold-step','.1'),
                            ('--swing-ts2vec-target-trades-per-day','2','--swing-ts2vec-threshold-step','nan')):
                with self.subTest(options=options),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    cli(directory,*options)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                engine = build_engine([],cli(directory,'--swing-ts2vec-target-trades-per-day','2','--swing-ts2vec-threshold-step','.1'))
            feed_candles(engine,4)
            self.assertIsNone(engine.strategy.daily_target.day)
            self.assertEqual(engine.strategy.daily_target.history,[])
            self.assertEqual(engine.strategy.threshold,.25)
            engine.on_tick(Tick(START+timedelta(days=2),'EURUSD',1.1,1.1001,1,1))
            self.assertEqual(engine.strategy.daily_target.day,(START+timedelta(days=2)).date())
            self.assertEqual(engine.strategy.threshold,.25)

    def test_fractional_target_cli_uses_default_step_and_keeps_period_across_days(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver(False)):
                engine = build_engine([],cli(directory,'--swing-ts2vec-target-trades-per-day','.25'))
            feed_candles(engine,4)
            self.assertEqual(engine.strategy.daily_target.config.threshold_step,.001)
            self.assertEqual(engine.strategy.daily_target.period_end,START+timedelta(days=4))
            self.assertEqual(engine.strategy.daily_target.scored,2)
            for day in (1,2,3):
                engine.on_tick(Tick(START+timedelta(days=day),'EURUSD',1.1,1.1001,1,1))
                self.assertEqual(engine.strategy.threshold,.25)
            engine.on_tick(Tick(START+timedelta(days=4),'EURUSD',1.1,1.1001,1,1))
            self.assertAlmostEqual(engine.strategy.threshold,.249)
            self.assertEqual(len(engine.strategy.daily_target.history),1)

    def test_decision_log_is_opt_in_and_can_filter_rejections_without_affecting_execution(self):
        settings = (([],"off"),(["--swing-ts2vec-decision-log"],"accepted"),
                    (["--swing-ts2vec-decision-log","accepted"],"accepted"),
                    (["--swing-ts2vec-decision-log","all"],"all"),
                    (["--swing-ts2vec-decision-log","off"],"off"),
                    (["--swing-ts2vec-decision-log","all","--no-swing-ts2vec-decision-log"],"off"))
        for accepted in (True,False):
            for options,mode in settings:
                with self.subTest(accepted=accepted,mode=mode,options=options),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as output:
                    directory = Path(tmp)
                    metadata(directory)
                    with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver(accepted)):
                        engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1',*options))
                    feed_candles(engine,3)
                    lines = output.getvalue().splitlines()
                    decisions = [json.loads(line.split('] ',1)[1]) for line in lines if line.startswith('[swing-ts2vec-decision]')]
                    expected = mode == 'all' or mode == 'accepted' and accepted
                    self.assertEqual(len(decisions),int(expected))
                    self.assertEqual(engine.strategy.decision_log,mode)
                    self.assertEqual(engine.strategy.counts['scored'],1)
                    self.assertEqual(engine.strategy.counts['accepted'],int(accepted))
                    self.assertEqual(bool(engine.portfolio.positions),accepted)
                    self.assertTrue(any(line.startswith('[swing-ts2vec-progress]') for line in lines))
                    if decisions:
                        self.assertEqual(decisions[0]['accepted'],accepted)

    def test_progress_exposes_pre_model_replay_and_trading_period_without_scoring_partial_candles(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as output:
            directory = Path(tmp)
            metadata(directory,available=START+HOUR)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                strategy = SwingTS2VecStrategy(directory)
            tick = Tick(START,'EURUSD',1.1,1.1001,1,1)
            market = SimpleNamespace(last_ticks={'EURUSD':tick},get_candles=lambda *args,**kwargs: [])
            context = StrategyContext(Portfolio(10000),market)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.monotonic',side_effect=[0.,1.,31.,32.]):
                for _ in range(3):
                    self.assertEqual(strategy.on_market_data(TickBatch('EURUSD',[tick]),context),[])
                market.last_ticks['EURUSD'] = Tick(START+HOUR,'EURUSD',1.1,1.1001,1,1)
                self.assertEqual(strategy.on_market_data(TickBatch('EURUSD',[tick]),context),[])
            progress = [line for line in output.getvalue().splitlines() if line.startswith('[swing-ts2vec-progress]')]
            self.assertEqual(len(progress),3)
            self.assertIn('phase=warmup_before_model',progress[0])
            self.assertIn('phase=trading_period',progress[-1])
            self.assertIn('scored=0',progress[-1])

    def test_threshold_override_changes_execution_without_changing_saved_model(self):
        for threshold, prediction_positive, accepted in ((None,True,True),(.75,True,False),
                                                        (.5,True,True),(0.,True,True),
                                                        (0.,False,False),(-.5,False,True)):
            with self.subTest(threshold=threshold,positive=prediction_positive),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as output:
                directory = Path(tmp)
                metadata(directory)
                original = (directory/'metadata.json').read_bytes()
                options = [] if threshold is None else [f'--swing-ts2vec-threshold={threshold}']
                with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver(prediction_positive)):
                    engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1',
                                                '--swing-ts2vec-decision-log','all',*options))
                feed_candles(engine,3)
                self.assertEqual(bool(engine.portfolio.positions),accepted)
                self.assertEqual(engine.strategy.threshold,.25 if threshold is None else threshold)
                self.assertEqual(engine.strategy.trained_threshold,.25)
                self.assertEqual(engine.strategy.threshold_source,'model' if threshold is None else 'override')
                self.assertEqual((directory/'metadata.json').read_bytes(),original)
                decision = next(json.loads(line.split('] ',1)[1]) for line in output.getvalue().splitlines()
                                if line.startswith('[swing-ts2vec-decision]'))
                self.assertEqual(decision['accepted'],accepted)
                self.assertEqual(decision['threshold'],engine.strategy.threshold)
                self.assertEqual(decision['trained_threshold'],.25)

    def test_threshold_validation_rejects_nonfinite_values_and_invalid_win_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            values = metadata(directory)
            for threshold in ('nan','inf','-inf'):
                with self.subTest(threshold=threshold),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    cli(directory,f'--swing-ts2vec-threshold={threshold}')
            values['target'] = 'win'
            (directory/'metadata.json').write_text(json.dumps(values),encoding='utf-8')
            for threshold in (-.1,1.1):
                with self.subTest(threshold=threshold),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    cli(directory,f'--swing-ts2vec-threshold={threshold}')
            self.assertEqual(cli(directory,'--swing-ts2vec-threshold','0').swing_ts2vec_threshold,0.)
            self.assertEqual(cli(directory,'--swing-ts2vec-threshold','1').swing_ts2vec_threshold,1.)
            with patch.object(sys,'argv',['run','--swing-ts2vec-threshold','0']),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args()

    def test_cli_restores_model_settings_and_rejects_incompatible_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            metadata(directory)
            args = cli(directory)
            self.assertEqual(args.symbols,['EURUSD','GBPUSD'])
            self.assertEqual(args.candle_interval,'1h')
            self.assertEqual(args.execution_delay,timedelta(0))
            self.assertTrue(args.trailing_stop)
            self.assertEqual(args.trailing_start_rr_ratio,1.)
            for options in (('--candle-interval','4h'),('--trailing-start-rr-ratio','0'),
                            ('--order-expiration-bars','3'),('--symbols','USDJPY'),
                            ('--adaptive-strategy',),('--contextual-bandit',)):
                with self.subTest(options=options),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    cli(directory,*options)
            with patch.object(sys,'argv',['run','--strategy','swing-ts2vec']),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args()

    def test_strategy_matches_collector_contexts_keeps_observing_when_busy_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory)
            resolver = fake_resolver()
            seen = []
            original = resolver.predict
            resolver.predict = lambda points,**kwargs: (seen.append(points),original(points,**kwargs))[1]
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=resolver):
                strategy = SwingTS2VecStrategy(directory)
            reference = SwingSeries(HOUR,(SwingWindow('pairs',1),))
            expected, history, emitted = [], [], []
            for bid in bars():
                ask = Candle(bid.start_time,bid.open_price+.0001,bid.high_price+.0001,
                             bid.low_price+.0001,bid.close_price+.0001)
                history.append((bid,ask))
                reference.append(bid,ask)
                if reference.last_candidates:
                    points = reference.context(SwingWindow('pairs',1),bid.start_time+HOUR)
                    if points:
                        expected.append(points)
                tick = Tick(bid.start_time+HOUR+timedelta(milliseconds=1),'EURUSD',bid.close_price,bid.close_price+.0001,1,1)
                market = SimpleNamespace(last_ticks={'EURUSD':tick},get_candles=lambda symbol,**kwargs:
                                         [pair[0 if kwargs['price_source']=='bid' else 1] for pair in history])
                context = StrategyContext(Portfolio(10000),market,pending_orders_for_symbol=5)
                emitted.extend(strategy.on_market_data(TickBatch('EURUSD',[tick]),context))
                self.assertEqual(strategy.on_market_data(TickBatch('EURUSD',[tick]),context),[])
            self.assertEqual(seen,expected)
            self.assertEqual(len(emitted),5)
            self.assertAlmostEqual(emitted[0].stop_loss_price,1.0991-.0002)
            self.assertEqual(emitted[0].timestamp,START+3*HOUR)
            self.assertEqual(strategy.counts['accepted'],5)

    def test_no_entry_before_model_availability_or_from_partial_and_expired_candles(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory,available=START+10*HOUR)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                strategy = SwingTS2VecStrategy(directory)
            history = bars(4)
            tick = Tick(START+3.5*HOUR,'EURUSD',1.1,1.1001,1,1)
            market = SimpleNamespace(last_ticks={'EURUSD':tick},get_candles=lambda *a,**kw: history)
            context = StrategyContext(Portfolio(10000),market)
            self.assertEqual(strategy.on_market_data(TickBatch('EURUSD',[tick]),context),[])
            self.assertEqual(strategy.last_start['EURUSD'],START+2*HOUR)
            self.assertEqual(strategy.counts['scored'],0)
            metadata(directory,available=START)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                strategy = SwingTS2VecStrategy(directory)
            market.last_ticks['EURUSD'] = Tick(START+20*HOUR,'EURUSD',1.1,1.1001,1,1)
            self.assertEqual(strategy.on_market_data(TickBatch('EURUSD',[tick]),context),[])
            self.assertEqual(strategy.counts['scored'],0)
            self.assertGreater(strategy.counts['already_expired_at_release'],0)

    def test_model_rejection_produces_no_order(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver(False)):
                engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1'))
            feed_candles(engine,4)
            self.assertEqual(len(engine.portfolio.positions),0)
            self.assertEqual(len(engine.pending_orders_processor.queue),0)
            self.assertGreater(engine.strategy.counts['rejected_by_model'],0)

    def test_robot_fills_first_eligible_quote_with_structural_stop_trailing_and_risk_limits(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1','--balance','10000','--max-positions','1'))
            feed_candles(engine,4)
            self.assertEqual(len(engine.portfolio.positions),1)
            position = next(iter(engine.portfolio.positions.values()))
            self.assertEqual(position.opened_at,START+3*HOUR+timedelta(seconds=1))
            self.assertAlmostEqual(position.entry_price,1.1021)
            self.assertAlmostEqual(position.stop_loss,1.0999)
            self.assertAlmostEqual(position.quantity*(position.entry_price-position.stop_loss),100.)
            self.assertIsNone(position.take_profit)
            self.assertTrue(position.trailing_stop_enabled)
            self.assertAlmostEqual(position.trailing_stop_activation_price,1.1043)
            self.assertGreater(engine.strategy.counts['rejected_by_risk'],0)
            # A filled position survives the two-candle order lifetime.
            for hour in (5,6):
                engine.on_tick(Tick(START+hour*HOUR,'EURUSD',1.1000,1.1001,1,1))
            self.assertIn(position.trade_id,engine.portfolio.positions)
            engine.on_tick(Tick(START+6*HOUR+timedelta(seconds=1),'EURUSD',1.1043,1.1044,1,1))
            self.assertAlmostEqual(position.stop_loss,position.entry_price)
            engine.on_tick(Tick(START+6*HOUR+timedelta(seconds=2),'EURUSD',1.1021,1.1022,1,1))
            self.assertNotIn(position.trade_id,engine.portfolio.positions)
            self.assertEqual(engine.reports.trades[-1].close_reason.name,'TRAILING_STOP')

    def test_trailing_override_applies_to_buy_and_sell_fills_and_actual_activation(self):
        for selling in (False, True):
            with self.subTest(selling=selling), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
                directory = Path(tmp)
                metadata(directory)
                original = (directory/'metadata.json').read_bytes()
                with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=fake_resolver()):
                    engine = build_engine([], cli(directory, '--max-spread-ratio', '10000', '--risk-percent', '1',
                        '--max-positions', '1', '--trailing-start-rr-ratio', '2', '--swing-ts2vec-decision-log', 'all'))
                feed_candles(engine, 4, selling=selling)
                position = next(iter(engine.portfolio.positions.values()))
                direction = -1 if selling else 1
                original_stop = position.stop_loss
                distance = abs(position.entry_price-original_stop)
                expected = position.entry_price+direction*2*distance
                self.assertAlmostEqual(position.trailing_stop_activation_price, expected)
                self.assertIsNone(position.take_profit)
                self.assertEqual(engine.strategy.trained_trailing_start_r, 1.)
                self.assertEqual(engine.strategy.opportunity_config.trailing_start_r, 2.)
                for seconds, multiple in ((1, 1.1), (2, 2.1)):
                    price = position.entry_price+direction*multiple*distance
                    bid, ask = (price-.0001, price) if selling else (price, price+.0001)
                    engine.on_tick(Tick(START+6*HOUR+timedelta(seconds=seconds), 'EURUSD', bid, ask, 1, 1))
                    self.assertAlmostEqual(position.stop_loss, original_stop if seconds == 1 else position.entry_price)
                self.assertIn('"trailing_start_r": 2.0', output.getvalue())
                self.assertIn('"trained_trailing_start_r": 1.0', output.getvalue())
                self.assertEqual((directory/'metadata.json').read_bytes(), original)

    def test_trailing_override_cli_can_exceed_old_reference_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            metadata(directory)
            for value in (.5, 1.5, 5.):
                args = cli(directory, '--trailing-start-rr-ratio', str(value))
                self.assertEqual(args.trailing_start_rr_ratio, value)
                self.assertGreater(args.reward_risk_ratio, value)
            for value in ('0', '-1', 'nan', 'inf', '-inf'):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli(directory, '--trailing-start-rr-ratio='+value)

    def test_delayed_fill_with_override_recomputes_activation_from_actual_entry(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory, delay=10.)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=fake_resolver()):
                engine = build_engine([], cli(directory, '--max-spread-ratio', '10000', '--risk-percent', '1',
                    '--trailing-start-rr-ratio', '.5'))
            feed_candles(engine, 3)
            engine.on_tick(Tick(START+3*HOUR+timedelta(seconds=11), 'EURUSD', 1.1004, 1.1005, 1, 1))
            position = next(iter(engine.portfolio.positions.values()))
            self.assertAlmostEqual(position.entry_price, 1.1005)
            self.assertAlmostEqual(position.trailing_stop_activation_price,
                                   position.entry_price+.5*abs(position.entry_price-position.stop_loss))

    def test_fixed_tp_disables_trailing_and_closes_buys_and_sells_at_tp_or_initial_stop(self):
        for selling in (False, True):
            for take_profit in (False, True):
                with self.subTest(selling=selling, take_profit=take_profit), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
                    directory = Path(tmp)
                    metadata(directory)
                    original = (directory/'metadata.json').read_bytes()
                    args = cli(directory, '--max-spread-ratio', '10000', '--risk-percent', '1', '--max-positions', '1',
                               '--swing-ts2vec-take-profit-r', '2', '--trailing-stop', '--swing-ts2vec-decision-log', 'all')
                    self.assertFalse(args.trailing_stop)
                    self.assertEqual(args.reward_risk_ratio, 2.)
                    with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=fake_resolver()):
                        engine = build_engine([], args)
                    feed_candles(engine, 4, selling=selling)
                    position = next(iter(engine.portfolio.positions.values()))
                    original_stop = position.stop_loss
                    distance = abs(position.entry_price-original_stop)
                    direction = -1 if selling else 1
                    target = position.entry_price+direction*2*distance
                    cash_risk = distance*position.quantity
                    self.assertAlmostEqual(position.take_profit, target)
                    self.assertFalse(position.trailing_stop_enabled)
                    self.assertIsNone(position.trailing_stop_state)
                    self.assertIsNone(position.trailing_stop_activation_price)
                    # Passing the old 1R activation must leave the initial SL untouched.
                    price = position.entry_price+direction*1.1*distance
                    bid, ask = (price-.0001, price) if selling else (price, price+.0001)
                    engine.on_tick(Tick(START+6*HOUR+timedelta(seconds=1), 'EURUSD', bid, ask, 1, 1))
                    self.assertIn(position.trade_id, engine.portfolio.positions)
                    self.assertEqual(position.stop_loss, original_stop)
                    price = target if take_profit else original_stop
                    bid, ask = (price-.0001, price) if selling else (price, price+.0001)
                    engine.on_tick(Tick(START+6*HOUR+timedelta(seconds=2), 'EURUSD', bid, ask, 1, 1))
                    self.assertNotIn(position.trade_id, engine.portfolio.positions)
                    closed = engine.reports.trades[-1]
                    self.assertEqual(closed.close_reason.name, 'TAKE_PROFIT' if take_profit else 'STOP_LOSS')
                    self.assertAlmostEqual(closed.pnl/cash_risk, 2. if take_profit else -1.)
                    self.assertIn('"exit_mode": "fixed-take-profit"', output.getvalue())
                    self.assertIn('"take_profit_r": 2.0', output.getvalue())
                    self.assertEqual((directory/'metadata.json').read_bytes(), original)

    def test_delayed_fixed_tp_uses_actual_fill_and_allows_tp_below_training_activation(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory, delay=10.)
            args = cli(directory, '--max-spread-ratio', '10000', '--risk-percent', '1', '--swing-ts2vec-take-profit-r', '.5')
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=fake_resolver()):
                engine = build_engine([], args)
            feed_candles(engine, 3)
            self.assertEqual(len(engine.portfolio.positions), 0)
            engine.on_tick(Tick(START+3*HOUR+timedelta(seconds=11), 'EURUSD', 1.1004, 1.1005, 1, 1))
            position = next(iter(engine.portfolio.positions.values()))
            self.assertAlmostEqual(position.entry_price, 1.1005)
            self.assertAlmostEqual(position.take_profit, position.entry_price+.5*abs(position.entry_price-position.stop_loss))
            self.assertFalse(position.trailing_stop_enabled)
            self.assertIsNone(position.trailing_stop_state)

    def test_fixed_tp_cli_rejects_nonpositive_nonfinite_values_and_other_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            metadata(directory)
            for value in ('0', '-1', 'nan', 'inf', '-inf'):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli(directory, '--swing-ts2vec-take-profit-r='+value)
            with patch.object(sys, 'argv', ['run_backtest', '--swing-ts2vec-take-profit-r', '2']), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_delayed_fill_recomputes_trailing_from_actual_entry(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory,delay=10.)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1'))
            feed_candles(engine,3)
            self.assertEqual(len(engine.portfolio.positions),0)
            engine.on_tick(Tick(START+3*HOUR+timedelta(seconds=11),'EURUSD',1.1004,1.1005,1,1))
            position = next(iter(engine.portfolio.positions.values()))
            self.assertAlmostEqual(position.entry_price,1.1005)
            self.assertAlmostEqual(position.trailing_stop_activation_price,2*position.entry_price-position.stop_loss)


    def test_sell_fills_bid_with_bid_high_stop_and_actual_risk_trailing(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            metadata(directory)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1','--balance','10000','--max-positions','1'))
            feed_candles(engine,4,selling=True)
            position = next(iter(engine.portfolio.positions.values()))
            self.assertEqual(position.side.name,'SELL')
            self.assertAlmostEqual(position.entry_price,1.1000)
            self.assertAlmostEqual(position.stop_loss,1.1022)
            self.assertAlmostEqual(position.trailing_stop_activation_price,1.0978)
            self.assertAlmostEqual(position.quantity*(position.stop_loss-position.entry_price),100.)

    def test_delayed_fill_already_at_or_beyond_stop_closes_immediately(self):
        for ask in (1.0999,1.0998):
            with self.subTest(ask=ask),tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                directory = Path(tmp)
                metadata(directory,delay=10.)
                with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver',return_value=fake_resolver()):
                    engine = build_engine([],cli(directory,'--max-spread-ratio','10000','--risk-percent','1'))
                feed_candles(engine,3)
                engine.on_tick(Tick(START+3*HOUR+timedelta(seconds=11),'EURUSD',ask-.0001,ask,1,1))
                self.assertEqual(len(engine.portfolio.positions),0)
                self.assertEqual(len(engine.reports.trades),1)
                self.assertEqual(engine.reports.trades[0].close_reason.name,'STOP_LOSS')
                self.assertLess(engine.reports.trades[0].pnl,0)


def feed_candles(engine,count,selling=False):
    for bid in bars(count):
        # Small intrabar excursion keeps earlier positions open for limit checks.
        for seconds,price in ((1,bid.open_price),(3599,bid.close_price)):
            if selling:
                price = 2.2020-price
            engine.on_tick(Tick(bid.start_time+timedelta(seconds=seconds),'EURUSD',price,price+.0001,1,1))
    engine.on_tick(Tick(START+count*HOUR+timedelta(milliseconds=1),'EURUSD',1.1000,1.1001,1,1))


if __name__ == '__main__':
    unittest.main()
