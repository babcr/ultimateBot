from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.research.swing_opportunities import (
    OpportunityConfig, OpportunitySimulator, collect, connect, milliseconds, stop_distance_feature,
)
from ultimateBot.backtester.research.swing_series import SwingPoint, SwingWindow
from ultimateBot.backtester.research.swing_contexts import ensure_windows
from ultimateBot.backtester.research.swing_grid import prepare, train_grid, SwingGridResolver, append_stop_input, grid_vector
from ultimateBot.backtester.research.swing_neural import prepare_data, fit, SwingNeuralResolver, frozen_test
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol, TrainingSelection
from ultimateBot.run_swing_ts2vec import parse_args, opportunity_config
from ultimateBot.tests.test_swing_collection_append import history, START
from ultimateBot.tests.test_swing_ts2vec import memory_db, add_event
from ultimateBot.tests.test_swing_neural import config as neural_config
from ultimateBot.tests.test_swing_window_search import CONFIG, ENCODER


def collected(root):
    history(root/'ticks', ['EURUSD', 'USDJPY'])
    for symbol in ('EURUSD', 'USDJPY'):
        source = root/'ticks'/symbol/'2024/01/2024-01-01.tsv'
        target = source.with_name('2024-01-03.tsv')
        rows = []
        for line in source.read_text().splitlines():
            cells = line.split('\t')
            cells[2] = str(int(cells[2])+2*86400000)
            rows.append('\t'.join(cells))
        target.write_text('\n'.join(rows)+'\n')
    config = OpportunityConfig(interval_seconds=600, stop_buffer_pips=2., stop_variants=3,
                               stop_step_pips=5., max_holding_seconds=7200)
    path = root/'data.sqlite3'
    result = collect(path, history=root/'ticks', symbols=['EURUSD', 'USDJPY'], start=START.date(),
        end=(START+timedelta(days=2)).date(), config=config, windows=(SwingWindow('duration', 1800),), store_contexts=True)
    return path, result['run_id'], config


class SwingStopVariantTests(unittest.TestCase):
    def test_configuration_cli_pips_and_legacy_defaults(self):
        config = OpportunityConfig(stop_variants=4, stop_step_pips=5., stop_buffer_pips=2.)
        self.assertEqual([config.buffer_for(i) for i in range(4)], [2., 7., 12., 17.])
        self.assertEqual(config.max_holding_seconds, 14*86400)
        self.assertEqual(OpportunityConfig().max_holding_seconds, 0)
        low = SwingPoint('low', 1.1, START, START, START+timedelta(hours=2))
        self.assertAlmostEqual(config.stop_for(low, 'EURUSD', 2), 1.0988)
        self.assertAlmostEqual(config.stop_for(replace(low, kind='high', price=150.), 'USDJPY', 2), 150.12)
        args = parse_args(['collect', '--stop-variants', '4', '--stop-step-pips', '7', '--max-holding', '3w', '--dry-run'])
        self.assertEqual(opportunity_config(args).max_holding_seconds, 21*86400)
        self.assertEqual(opportunity_config(parse_args(['collect', '--dry-run'])).max_holding_seconds, 14*86400)
        for kwargs in (dict(stop_variants=0), dict(stop_variants=True), dict(stop_step_pips=0),
                       dict(stop_step_pips=float('nan')), dict(max_holding_seconds=-1),
                       dict(stop_variants=2, max_holding_seconds=0)):
            with self.assertRaises(ValueError):
                OpportunityConfig(**kwargs)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['collect', '--stop-variants', '0', '--dry-run'])

    def test_each_stop_has_its_own_risk_and_outcome_fast_matches_every_tick(self):
        for selling in (False, True):
            outcomes = []
            for accelerated in (False, True):
                db = memory_db()
                self.addCleanup(db.close)
                config = OpportunityConfig(stop_variants=3, max_holding_seconds=7200)
                swing = SwingPoint('low', 1.1000, START, START, START+timedelta(hours=2))
                t = milliseconds(swing.confirmed_at)
                quotes = np.asarray([[t+1, 1.1010, 1.1011], [t+2, 1.0997, 1.0998],
                                     [t+3, 1.1057, 1.1058], [t+4, 1.1020, 1.1021]])
                if selling:
                    swing = replace(swing, kind='high', price=2.2-swing.price)
                    quotes = np.column_stack((quotes[:, 0], 2.2-quotes[:, 2], 2.2-quotes[:, 1]))
                sim = OpportunitySimulator(db, config, accelerated)
                for i in range(3):
                    add_event(db, swing, str(i))
                    sim.add(str(i), 'EURUSD', swing, i)
                sim.process(quotes)
                rows = [dict(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
                self.assertTrue(all(r['status']=='closed' for r in rows))
                self.assertLess(rows[0]['pnl_r'], 0)
                self.assertGreater(rows[1]['pnl_r'], 0)
                self.assertGreater(rows[1]['initial_risk'], rows[0]['initial_risk'])
                self.assertNotEqual(rows[0]['quantity'], rows[1]['quantity'])
                outcomes.append(rows)
            self.assertEqual(outcomes[0], outcomes[1])

    def test_timeout_uses_first_available_quote_after_fill_deadline_even_without_price_change(self):
        for selling in (False, True):
            results = []
            for accelerated in (False, True):
                db = memory_db()
                self.addCleanup(db.close)
                config = OpportunityConfig(max_holding_seconds=3600)
                swing = SwingPoint('high' if selling else 'low', 1.2 if selling else 1.,
                                   START, START, START+timedelta(hours=2))
                add_event(db, swing)
                t = milliseconds(swing.confirmed_at)
                sim = OpportunitySimulator(db, config, accelerated)
                sim.add('event', 'EURUSD', swing)
                sim.process(np.asarray([[t+1, 1.1, 1.1001], [t+1000, 1.11, 1.1101]]))
                self.assertTrue(sim.positions)
                sim.process(np.asarray([[t+3600000, 1.1, 1.1001], [t+7200000, 1.1, 1.1001],
                                        [t+10800000, 1.12, 1.1201]]))
                row = dict(db.execute('SELECT * FROM swing_events').fetchone())
                self.assertEqual(row['close_reason'], 'MAX_HOLDING_TIME')
                self.assertEqual(row['closed_at'], (swing.confirmed_at+timedelta(hours=2)).isoformat())
                self.assertEqual(row['exit_price'], 1.1001 if selling else 1.1)
                results.append(row)
            self.assertEqual(results[0], results[1])

    def test_a_wider_stop_can_lose_when_its_trailing_activation_is_too_far(self):
        db = memory_db()
        self.addCleanup(db.close)
        settings = OpportunityConfig(stop_variants=4)
        swing = SwingPoint('low', 1.1000, START, START, START+timedelta(hours=2))
        sim = OpportunitySimulator(db, settings)
        for i in (0, 3):
            add_event(db, swing, str(i))
            sim.add(str(i), 'EURUSD', swing, i)
        t = milliseconds(swing.confirmed_at)
        sim.process(np.asarray([[t+1,1.1010,1.1011], [t+2,1.1034,1.1035],
                                [t+3,1.1018,1.1019], [t+4,1.0984,1.0985]]))
        rows = [dict(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
        self.assertGreater(rows[0]['pnl_r'], 0)
        self.assertGreaterEqual(rows[0]['trailing_steps'], 1)
        self.assertLess(rows[1]['pnl_r'], 0)
        self.assertEqual(rows[1]['trailing_steps'], 0)

    def test_connect_migrates_legacy_not_null_contexts_without_rewriting_payloads(self):
        from ultimateBot.backtester.research.swing_opportunities import SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.sqlite3'
            old = SCHEMA.replace(',\n swing_id TEXT, variant_index INTEGER NOT NULL DEFAULT 0, stop_buffer_pips REAL', '')
            old = old.replace(', shared_context_id INTEGER REFERENCES swing_contexts(context_id)', '')
            start = old.index('CREATE VIEW IF NOT EXISTS swing_contexts_resolved AS')
            end = old.index(';', start)+1
            old = old[:start]+old[end:]
            db = sqlite3.connect(path)
            db.executescript(old)
            db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES('run',?,'completed','{}')", (START.isoformat(),))
            db.execute("""INSERT INTO swing_events(event_id,run_id,symbol,kind,side,pivot_at,pair_start,
                confirmed_at,stop_price,expires_at,status) VALUES('old','run','EURUSD','low',1,?,?,?,1.1,?,'closed')""",
                (START.isoformat(),START.isoformat(),(START+timedelta(hours=2)).isoformat(),(START+timedelta(hours=4)).isoformat()))
            db.execute("""INSERT INTO swing_contexts(event_id,window_key,information_start,context_end,
                n_swings,span_seconds,features,swings_json) VALUES('old','pairs_1',?,?,2,3600,?,'[]')""",
                (START.isoformat(),(START+timedelta(hours=2)).isoformat(),b'original bytes'))
            db.commit()
            db.close()
            db = connect(path)
            try:
                row = db.execute('SELECT * FROM swing_contexts_resolved').fetchone()
                self.assertEqual(row['features'], b'original bytes')
                self.assertEqual(row['swings_json'], '[]')
                self.assertEqual(db.execute('SELECT variant_index FROM swing_events').fetchone()[0], 0)
            finally:
                db.close()

    def test_collection_shared_payload_reconstruction_append_and_grid_variant_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, run, config = collected(root)
            db = connect(path)
            try:
                groups = db.execute('SELECT swing_id,count(*) AS n FROM swing_events GROUP BY swing_id').fetchall()
                self.assertTrue(groups)
                self.assertTrue(all(r['n']==3 for r in groups))
                base_payloads = db.execute('SELECT count(*) FROM swing_contexts WHERE length(features)>0').fetchone()[0]
                count = db.execute('SELECT count(*) FROM swing_contexts').fetchone()[0]
                self.assertEqual(count, 3*base_payloads)
                resolved = db.execute('SELECT count(*) FROM swing_contexts_resolved WHERE length(features)>0 AND length(swings_json)>0').fetchone()[0]
                self.assertEqual(resolved, count)
                ensure_windows(db, run, (SwingWindow('duration', 2400),), before=START+timedelta(days=3))
                self.assertGreater(db.execute("SELECT count(*) FROM swing_contexts WHERE window_key='duration_2400'").fetchone()[0], 0)
            finally:
                db.close()
            args = parse_args(['collect', '--database', str(path), '--run-id', run, '--symbols', 'EURUSD', '--dry-run'])
            self.assertEqual(opportunity_config(args), config)
            periods = dict(train=['2024-01-01', '2024-01-01'], validation=['2024-01-02', '2024-01-02'])
            data, _ = prepare(path, run_id=run, windows=(SwingWindow('duration', 1800),), symbols=['EURUSD'],
                periods=periods, target='pnl-r', cache_root=root/'cache')
            try:
                self.assertEqual(data.matrices['duration_1800'].shape[1], 4)
                self.assertEqual(data.manifest['counts']['ambiguous'], 0)
                same = [i for i, r in enumerate(data.rows) if r['swing_id']==data.rows[0]['swing_id']]
                self.assertEqual(len(same), 3)
                x = data.matrices['duration_1800'][same]
                np.testing.assert_array_equal(x[:, :-1], np.repeat(x[:1, :-1], 3, axis=0))
                self.assertTrue(np.all(np.diff(x[:, -1]) > 0))
                self.assertTrue(all(r['closed_at'] < '2024-01-02' for r in data.rows if r['phase']=='train'))
            finally:
                data.close()

    def test_neural_xgboost_and_informer_use_stop_input_in_training_and_frozen_inference(self):
        from ultimateBot.backtester.research.swing_selection import StopSelection, FIRST_STOP_POLICY
        from ultimateBot.backtester.research.swing_neural import recalibrate_model
        import csv
        import hashlib
        from ultimateBot.backtester.research.swing_neural import predict_array
        from ultimateBot.backtester.research.swing_informer_data import extract, SwingInformerData
        from ultimateBot.backtester.research.swing_informer import fit as fit_informer, SwingInformerResolver
        from ultimateBot.tests.test_swing_informer import small_config
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path, run, settings = collected(root)
            protocol = ResearchProtocol('2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02')
            data = prepare_data(path, run_id=run, symbols=['EURUSD'], duration=1800, protocol=protocol,
                                grid_normalization='swing-mean', cache_root=root/'cache')
            try:
                meta = fit(data, root/'nn', neural_config(epochs=1, min_train_samples=10, min_validation_samples=8))
                self.assertEqual(meta['input_dims'], 4)
                self.assertEqual(meta['validation']['selection_policy'], FIRST_STOP_POLICY)
            finally:
                data.close()
            db = connect(path)
            row = dict(db.execute("SELECT c.*,e.stop_buffer_pips FROM swing_contexts_resolved c JOIN swing_events e USING(event_id) WHERE e.symbol='EURUSD' AND e.confirmed_at>='2024-01-03' AND c.window_key='duration_1800' LIMIT 1").fetchone())
            db.close()
            points = tuple(SwingPoint(**{k:datetime.fromisoformat(v) if k.endswith('_at') or k=='pair_start' else v for k,v in p.items()}) for p in json.loads(row['swings_json']))
            at = points[-1].confirmed_at
            resolver = SwingNeuralResolver(root/'nn', threads=1)
            with np.load(root/'nn/validation_predictions.npz', allow_pickle=False) as saved:
                policy = StopSelection.from_event_ids(saved['event_id'], asdict(settings))
                np.testing.assert_array_equal(saved['accepted'], policy.mask(saved['prediction'], meta['threshold']))
                self.assertEqual(int(saved['accepted'].sum()), meta['validation']['selected']['opportunities'])
            source_bytes = (root/'nn/validation_predictions.npz').read_bytes()
            updated = recalibrate_model(root/'nn', root/'recalibrated')
            self.assertEqual(meta['checkpoint_sha256'], updated['checkpoint_sha256'])
            self.assertEqual((root/'nn/validation_predictions.npz').read_bytes(), source_bytes)
            self.assertEqual(updated['validation']['selection_policy'], FIRST_STOP_POLICY)
            self.assertEqual(updated['validation_predictions_sha256'], hashlib.sha256((root/'recalibrated/validation_predictions.npz').read_bytes()).hexdigest())
            SwingNeuralResolver(root/'recalibrated', threads=1)
            for buffer in (2., 7., 12.):
                vector = append_stop_input(grid_vector(points, as_of=at, window=resolver.window, interval_seconds=600),
                                           asdict(settings), 'EURUSD', points[-1].price, buffer)
                expected = predict_array(resolver.model, vector[None], meta['normalization'])[0]
                self.assertEqual(expected, resolver.predict(points, as_of=at, symbol='EURUSD', stop_buffer_pips=buffer)['prediction'])
            result = frozen_test(path, directory=root/'nn', start='2024-01-03', end='2024-01-03',
                                 output_dir=root/'test', min_samples=5, threads=1)
            self.assertFalse(result['fitting_performed'])
            self.assertEqual(result['metrics']['selection_policy'], FIRST_STOP_POLICY)
            trained = train_grid(path, run_id=run, output_dir=root/'xgb', config=replace(CONFIG, min_train_samples=10,
                min_validation_samples=8, min_accepted_validation=1), encoder_config=ENCODER, protocol=protocol,
                selection=TrainingSelection(symbols=('EURUSD',), window_mode='duration', durations=(1800,)))
            tree = SwingGridResolver(trained['directory'], threads=1)
            self.assertEqual(tree.metadata['feature_names'][-1], 'stop_buffer_ratio')
            self.assertTrue(np.isfinite(tree.predict(points, as_of=at, symbol='EURUSD', stop_buffer_pips=12.)['prediction']))
            tree_report = trained['windows']['duration_1800']['report']['partitions']['validation']
            self.assertEqual(tree_report['selection_policy'], FIRST_STOP_POLICY)
            with (tree.directory/'predictions.csv').open(newline='', encoding='utf-8') as stream:
                records = [r for r in csv.DictReader(stream) if r['partition']=='validation']
            policy = StopSelection.from_event_ids([r['event_id'] for r in records], asdict(settings))
            expected = policy.mask([float(r['prediction']) for r in records], tree.metadata['threshold'])
            np.testing.assert_array_equal([int(r['accepted']) for r in records], expected)
            self.assertEqual(int(expected.sum()), tree_report['selected']['opportunities'])
            informer = extract(path, run_id=run, periods=dict(train=['2024-01-01','2024-01-01'],
                validation=['2024-01-02','2024-01-02']), seq_len=8, symbols=['EURUSD'])
            self.assertEqual(informer.batch([0])[0].shape, (1, 8, 8))  # six swing + symbol + stop
            informer.save(root/'data.npz')
            np.testing.assert_array_equal(SwingInformerData.load(root/'data.npz').batch([0])[0], informer.batch([0])[0])
            info_meta = fit_informer(informer, root/'informer', small_config(epochs=1, min_train_samples=10, min_validation_samples=8))
            self.assertEqual(info_meta['input_features'][-1], 'stop_buffer_ratio')
            self.assertEqual(info_meta['validation']['selection_policy'], FIRST_STOP_POLICY)
            with np.load(root/'informer/validation_predictions.npz', allow_pickle=False) as saved:
                policy = informer.stop_selection(informer.partition('validation'))
                np.testing.assert_array_equal(saved['accepted'], policy.mask(saved['prediction'], info_meta['threshold']))
            info_resolver = SwingInformerResolver(root/'informer', threads=1)
            from ultimateBot.tests.test_swing_informer import swing_points
            info_test = extract(path, run_id=run, periods=dict(test=['2024-01-03','2024-01-03']), seq_len=8, symbols=['EURUSD'])
            info_points = swing_points(info_test.points[info_test.contexts[0]])
            expected = info_resolver.regressor.predict(*info_test.batch([0]))[0]
            actual = info_resolver.predict(info_points, as_of=info_points[-1].confirmed_at, symbol='EURUSD',
                                          stop_buffer_pips=2.)['prediction']
            self.assertAlmostEqual(expected, actual, places=6)
            from ultimateBot.backtester.research.swing_resolver import train, SwingResolver, frozen_test as test_frozen
            old = train(path, run_id=run, output_dir=root/'ts2vec', config=replace(CONFIG, min_accepted_validation=1),
                encoder_config=ENCODER, protocol=protocol,
                selection=TrainingSelection(symbols=('EURUSD',), window_mode='duration', durations=(1800,)))
            encoded = SwingResolver(old['selected_directory'], threads=1)
            self.assertEqual(encoded.metadata['feature_names'][-1], 'stop_buffer_ratio')
            self.assertTrue(np.isfinite(encoded.predict(points, as_of=at, symbol='EURUSD', stop_buffer_pips=12.)['prediction']))
            evaluation = test_frozen(path, directory=old['directory'], start='2024-01-03', end='2024-01-03',
                                     output_dir=root/'ts2vec_test', min_samples=5, threads=1)
            self.assertFalse(evaluation['fit_called'])
            self.assertEqual(evaluation['selected']['selection_policy'], FIRST_STOP_POLICY)

    def test_real_engine_selects_first_accepted_stop_once_per_swing_and_time_closes(self):
        from ultimateBot.tests.test_swing_ts2vec_robot import metadata, feed_candles, cli
        from ultimateBot.run_backtest import build_engine
        from ultimateBot.backtester.data.tick import Tick
        from types import SimpleNamespace
        for selling in (False, True):
            for first_accepted in (2., 7., 12., None):
                with self.subTest(selling=selling, first_accepted=first_accepted), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                    root = Path(tmp)
                    meta = metadata(root)
                    meta['opportunity_config'].update(stop_variants=3, stop_step_pips=5., max_holding_seconds=3600)
                    (root/'metadata.json').write_text(json.dumps(meta))
                    seen, swings = [], []
                    def predict(points, *, as_of, symbol, stop_buffer_pips):
                        seen.append(stop_buffer_pips)
                        swings.append(points[-1])
                        accepted = first_accepted is not None and stop_buffer_pips >= first_accepted
                        return dict(prediction=.5 if accepted else 0., side=points[-1].side)
                    with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=SimpleNamespace(predict=predict)):
                        engine = build_engine([], cli(root, '--max-positions','10','--max-spread-ratio','10000',
                            '--swing-ts2vec-target-trades-per-day','1'))
                    feed_candles(engine, 3, selling=selling)
                    expected = [2., 7., 12.] if first_accepted is None else [v for v in (2., 7., 12.) if v <= first_accepted]
                    self.assertEqual(seen, expected)
                    self.assertEqual(engine.strategy.daily_target.scored, 1)
                    self.assertEqual(engine.strategy.daily_target.accepted, int(first_accepted is not None))
                    positions = list(engine.portfolio.positions.values())
                    self.assertEqual(len(positions), int(first_accepted is not None))
                    if first_accepted is None:
                        continue
                    position = positions[0]
                    variant = int((first_accepted-2)/5)
                    self.assertAlmostEqual(position.stop_loss,
                        engine.strategy.opportunity_config.stop_for(swings[-1], 'EURUSD', variant))
                    self.assertEqual(position.close_after, position.opened_at+timedelta(hours=1))
                    # Repeated quotes must not reevaluate the same confirmed swing.
                    engine.on_tick(Tick(position.opened_at+timedelta(seconds=1), 'EURUSD', 1.1, 1.1001, 1, 1))
                    self.assertEqual(seen, expected)
                    engine.on_tick(Tick(position.close_after, 'EURUSD', 1.1, 1.1001, 1, 1))
                    self.assertEqual(len(engine.portfolio.positions), 0)
                    self.assertEqual(len(engine.reports.trades), 1)
                    self.assertEqual(engine.reports.trades[0].close_reason.name, 'MAX_HOLDING_TIME')

    def test_risk_veto_does_not_try_larger_stops_for_the_same_swing(self):
        from ultimateBot.tests.test_swing_ts2vec_robot import metadata, feed_candles, cli
        from ultimateBot.run_backtest import build_engine
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            meta = metadata(root)
            meta['opportunity_config'].update(stop_variants=3, stop_step_pips=5., max_holding_seconds=86400)
            (root/'metadata.json').write_text(json.dumps(meta))
            seen = []
            def predict(points, *, as_of, symbol, stop_buffer_pips):
                seen.append((as_of, stop_buffer_pips))
                return dict(prediction=.5, side=points[-1].side)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver', return_value=SimpleNamespace(predict=predict)):
                engine = build_engine([], cli(root, '--max-positions','1','--max-spread-ratio','10000'))
            feed_candles(engine, 4)
            self.assertEqual(len(seen), 2)
            self.assertNotEqual(seen[0][0], seen[1][0])
            self.assertEqual([v for _, v in seen], [2., 2.])
            self.assertEqual(engine.strategy.counts['rejected_by_risk'], 1)
            self.assertEqual(len(engine.portfolio.positions), 1)

    def test_neural_nine_stops_and_base_override_reach_the_opened_position(self):
        from ultimateBot.tests.test_swing_ts2vec_robot import metadata, feed_candles, cli
        from ultimateBot.run_backtest import build_engine
        from types import SimpleNamespace
        for selling in (False, True):
            for override in (None, 5.):
                buffers = [10.+i*10 for i in range(9)] if override is None else [5.+i*10 for i in range(9)]
                for accepted_index in (2, 8, None):
                    with self.subTest(selling=selling, override=override, accepted_index=accepted_index), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                        root = Path(tmp)
                        meta = metadata(root, window='duration_7200')
                        meta.update(model_type='swing-nn', grid_normalization='swing-mean', min_threshold=0.)
                        meta['opportunity_config'].update(stop_buffer_pips=10., stop_variants=9,
                            stop_step_pips=10., max_holding_seconds=14*86400)
                        (root/'metadata.json').write_text(json.dumps(meta))
                        seen, swings = [], []
                        def predict(points, *, as_of, symbol, stop_buffer_pips):
                            seen.append(stop_buffer_pips)
                            swings.append(points[-1])
                            accepted = accepted_index is not None and stop_buffer_pips >= buffers[accepted_index]
                            return dict(prediction=.03 if accepted else .01, side=points[-1].side)
                        extra = [] if override is None else ['--swing-nn-stop-buffer-pips', str(override)]
                        with patch('ultimateBot.backtester.research.swing_neural.SwingNeuralResolver',
                                   return_value=SimpleNamespace(predict=predict)):
                            engine = build_engine([], cli(root, '--strategy', 'swing-nn',
                                '--swing-nn-threshold', '.02', '--max-spread-ratio', '10000',
                                '--risk-percent', '1', '--balance', '10000', *extra))
                        feed_candles(engine, 3, selling=selling)
                        expected = buffers if accepted_index is None else buffers[:accepted_index+1]
                        self.assertEqual(seen, expected)
                        self.assertEqual(engine.strategy.trained_stop_buffer_pips, 10.)
                        positions = list(engine.portfolio.positions.values())
                        self.assertEqual(len(positions), int(accepted_index is not None))
                        if accepted_index is not None:
                            position = positions[0]
                            expected_stop = swings[-1].price-swings[-1].side*buffers[accepted_index]*.0001
                            self.assertAlmostEqual(position.stop_loss, expected_stop)
                            distance = abs(position.entry_price-position.stop_loss)
                            self.assertAlmostEqual(position.quantity*distance, 100.)
                            self.assertAlmostEqual(position.trailing_stop_activation_price,
                                position.entry_price+swings[-1].side*distance)


if __name__ == '__main__':
    unittest.main()
