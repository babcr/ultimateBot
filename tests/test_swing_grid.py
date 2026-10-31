from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.research.swing_grid import GRID_SCHEMA, GridData, SwingGridResolver, grid_vector, grid_width, prepare
from ultimateBot.backtester.research.swing_resolver import TrainingSelection, frozen_test, predict, train
from ultimateBot.backtester.research.swing_series import SwingPoint, SwingWindow, VOLUME_FEATURE_SCHEMA
from ultimateBot.backtester.research.swing_window_search import search, search_plan
from ultimateBot.run_swing_ts2vec import parse_args, training_selection
from ultimateBot.tests.test_swing_window_search import START, HOUR, CONFIG, ENCODER, PROTOCOL, database


def point(at, price, kind='high'):
    return SwingPoint(kind, price, at-HOUR, at-2*HOUR, at)


def setup_database(path):
    db = database(path)
    db.execute("UPDATE swing_events SET trailing_steps=CAST(abs(pnl_r) AS INTEGER)")
    db.commit()
    db.close()


def selection():
    return TrainingSelection(window_mode='duration', durations=(6*3600, 8*3600))


class SwingGridTests(unittest.TestCase):
    def test_trailing_search_exports_target_scores_and_refuses_legacy_pnl_resume(self):
        from ultimateBot.backtester.research.swing_resolver import validation_score
        for target,policy in (('trailing-step-1','common'),('trailing-step-square','per-window')):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root=Path(tmp)
                path=root/'data.sqlite3'
                setup_database(path)
                options=dict(run_id='raw',output_dir=root/'search',config=replace(CONFIG,target=target),
                    encoder_config=ENCODER,protocol=PROTOCOL,selection=selection(),repeats=2,
                    max_depths=[2,4],representation='swing-grid',cohort_policy=policy)
                result=search(path,**options)
                self.assertEqual(result['plan']['validation_objective']['metric'],'mean_label')
                self.assertEqual(result['plan']['ranking'],'median_validation_mean_label_then_worst_then_coverage')
                trials=[json.loads(Path(entry['comparison']).read_text()) for entry in result['trials']]
                for row in result['leaderboard']:
                    scores=[validation_score(trial['windows'][row['window_key']]['report'].get('comparison_validation',
                        trial['windows'][row['window_key']]['report']['partitions']['validation']),target)
                        for trial in trials if trial['config']['max_depth']==row['max_depth']]
                    self.assertEqual(row['median_validation_score'],float(np.median(scores)))
                directory=Path(result['directory'])
                resolver=SwingGridResolver(directory,threads=1)
                self.assertEqual(resolver.metadata['selection_objective']['metric'],'mean_label')
                self.assertEqual(result['selected_validation_score'],result['leaderboard'][0]['median_validation_score'])
                with patch('ultimateBot.backtester.research.swing_window_search.train',side_effect=AssertionError('reuse')):
                    again=search(path,resume=directory,**options)
                self.assertEqual(again['selected_directory'],result['selected_directory'])
                old=json.loads((directory/'search.json').read_text())
                del old['plan']['validation_objective']
                (directory/'search.json').write_text(json.dumps(old))
                with self.assertRaisesRegex(ValueError,'previous PnL'):
                    search(path,resume=directory,**options)

    def test_exact_mean_excludes_holes_and_grid_preserves_candle_locations(self):
        at = START+6*HOUR
        points = (point(at-5*HOUR, 100.), point(at-3*HOUR, 130., 'low'), point(at, 160.))
        vector = grid_vector(points, as_of=at, window=SwingWindow('duration', 6*3600), interval_seconds=3600)
        expected = np.zeros(6, np.float32)
        expected[[0, 2, 5]] = np.asarray([100., 130., 160.])/130.-1
        np.testing.assert_allclose(vector, expected, atol=1e-8)
        self.assertEqual(vector.shape, (6,))
        self.assertEqual(vector[2], 0.)  # A swing equal to its mean is also zero, as requested.
        scaled = tuple(replace(p, price=p.price*5) for p in points)
        np.testing.assert_array_equal(vector, grid_vector(scaled, as_of=at,
            window=SwingWindow('duration', 6*3600), interval_seconds=3600))
        self.assertEqual(grid_width(SwingWindow('duration', 72*3600), 1200), 216)

    def test_invalid_future_non_alternating_and_left_boundary_swings_are_rejected(self):
        at, window = START+6*HOUR, SwingWindow('duration', 6*3600)
        def encode(points):
            return grid_vector(points, as_of=at, window=window, interval_seconds=3600)
        for points in ((point(at-6*HOUR, 100.), point(at, 120., 'low')),
                       (point(at-HOUR, 100.), point(at, 120.)),
                       (point(at-HOUR/2, 100.), point(at, 120., 'low')),
                       (point(at, 0.),), (point(at+HOUR, 100.),)):
            with self.subTest(points=points), self.assertRaises(ValueError):
                encode(points)
        with self.assertRaises(ValueError):
            grid_width(SwingWindow('pairs', 2), 3600)
        with self.assertRaises(ValueError):
            grid_width(SwingWindow('duration', 5400), 3600)

    def test_cli_selects_price_only_duration_grids_and_rejects_encoder_options(self):
        args = parse_args(['search', '--representation', 'swing-grid', '--durations', '72h', '96h',
                           '--target', 'trailing-step-square', '--dry-run'])
        selected = training_selection(args)
        self.assertEqual(selected.window_mode, 'duration')
        self.assertEqual(selected.durations, (259200, 345600))
        for options in (['--pairs', '2'], ['--swing-volume'], ['--ts2vec-depth', '2']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['train', '--representation', 'swing-grid', *options])

    def test_read_only_dense_cache_reuses_inputs_and_censors_future_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root, periods = Path(tmp), dict(train=[PROTOCOL.train_start, PROTOCOL.train_end],
                                          validation=[PROTOCOL.validation_start, PROTOCOL.validation_end])
            path = root/'data.sqlite3'
            setup_database(path)
            # All existing context blobs can be invalid: grid extraction never reads them.
            import sqlite3
            db = sqlite3.connect(path)
            db.execute("UPDATE swing_contexts SET features=X'00',swings_json='invalid'")
            db.execute("DELETE FROM swing_events WHERE event_id='EURUSD-30'")
            db.execute("UPDATE swing_events SET closed_at='2024-01-20T00:00:00+00:00' WHERE event_id='EURUSD-10'")
            params = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
            params['feature_schema'] = VOLUME_FEATURE_SCHEMA  # Volumes are intentionally irrelevant.
            db.execute('UPDATE swing_runs SET parameters_json=?', (json.dumps(params),))
            db.commit()
            db.close()
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            windows = tuple(SwingWindow('duration', s) for s in (21600, 28800))
            data, _ = prepare(path, run_id='raw', windows=windows, symbols=['EURUSD', 'GBPUSD'],
                periods=periods, target='pnl-r', cache_root=root/'cache')
            try:
                self.assertEqual(data.manifest['counts']['future_outcome'], 1)
                self.assertNotIn('EURUSD-10', {row['event_id'] for row in data.rows})
                self.assertIsInstance(data.matrices['duration_21600'], np.ndarray)
                self.assertEqual(data.matrices['duration_21600'].shape, (len(data.rows), 6))
                self.assertTrue(np.isfinite(data.matrices['duration_21600']).all())
                self.assertTrue((data.matrices['duration_28800'] == 0).any())
                directory = data.directory
            finally:
                data.close()
            with patch('ultimateBot.backtester.research.swing_grid.SwingSeries', side_effect=AssertionError('must reuse cache')):
                cached, _ = prepare(path, run_id='raw', windows=windows, symbols=['EURUSD', 'GBPUSD'],
                    periods=periods, target='pnl-r', cache_root=root/'cache')
                cached.close()
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            with (directory/'duration_21600.bin').open('r+b') as stream:
                stream.write(b'bad!')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                GridData(directory)

    def test_validation_changes_do_not_change_training_normalization_or_vectors(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            setup_database(path)
            periods = dict(train=[PROTOCOL.train_start, PROTOCOL.train_end], validation=[PROTOCOL.validation_start, PROTOCOL.validation_end])
            windows = (SwingWindow('duration', 21600),)
            before, _ = prepare(path, run_id='raw', windows=windows, symbols=['EURUSD'], periods=periods,
                                target='pnl-r', cache_root=root/'before')
            mask = [i for i, row in enumerate(before.rows) if row['phase'] == 'train']
            original = before.matrices[windows[0].key][mask].copy()
            before.close()
            db = sqlite3.connect(path)
            db.execute("UPDATE swing_events SET pivot_price=pivot_price*2,pnl_r=9999 WHERE confirmed_at>='2024-01-06'")
            db.commit()
            db.close()
            after, _ = prepare(path, run_id='raw', windows=windows, symbols=['EURUSD'], periods=periods,
                               target='pnl-r', cache_root=root/'after')
            try:
                mask = [i for i, row in enumerate(after.rows) if row['phase'] == 'train']
                np.testing.assert_array_equal(original, after.matrices[windows[0].key][mask])
            finally:
                after.close()

    def test_grid_training_inference_and_frozen_test_use_no_encoder_or_context_writes(self):
        import sqlite3
        from ultimateBot.run_backtest import build_engine, parse_args as backtest_args
        from ultimateBot.backtester.strategy.swing_grid_strategy import SwingGridStrategy
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            setup_database(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('ultimateBot.backtester.research.swing_encoder.SwingEncoder.fit', side_effect=AssertionError('no encoder')):
                result = train(path, run_id='raw', output_dir=root/'models', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=selection(), representation='swing-grid')
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(result['test_evaluated'])
            resolver = SwingGridResolver(result['directory'], threads=1)
            meta = resolver.metadata
            self.assertEqual(meta['grid_schema'], GRID_SCHEMA)
            self.assertEqual(meta['input_dims'], int(meta['window_key'].split('_')[1])//3600)
            self.assertFalse(meta['extra_tabular_inputs'])
            self.assertFalse((resolver.directory/'ts2vec.pt').exists())
            at = START+11*timedelta(days=1)
            db = sqlite3.connect(path)
            db.row_factory = sqlite3.Row
            rows = db.execute('''SELECT * FROM swing_events WHERE symbol='EURUSD' AND confirmed_at>? AND confirmed_at<=?
                ORDER BY confirmed_at''', ((at-timedelta(seconds=resolver.window.value)).isoformat(), at.isoformat())).fetchall()
            points = tuple(SwingPoint(row['kind'], row['pivot_price'],
                datetime.fromisoformat(row['pivot_at']), datetime.fromisoformat(row['pair_start']),
                datetime.fromisoformat(row['confirmed_at'])) for row in rows)
            db.close()
            decision = resolver.predict(points, as_of=at, symbol='EURUSD')
            vector = grid_vector(points, as_of=at, window=resolver.window, interval_seconds=3600)
            self.assertAlmostEqual(decision['prediction'], float(predict(resolver.model, vector[None], CONFIG.target)[0]))
            with self.assertRaisesRegex(ValueError, 'postdates'):
                resolver.predict(points, as_of=START, symbol='EURUSD')
            with self.assertRaisesRegex(ValueError, 'scope'):
                resolver.predict(points, as_of=at, symbol='USDJPY')
            original = (resolver.directory/'metadata.json').read_bytes()
            with patch('ultimateBot.backtester.research.swing_resolver.fit_xgb', side_effect=AssertionError('frozen test cannot fit')):
                report = frozen_test(path, directory=result['directory'], run_id='raw', start='2024-01-10', end='2024-01-12',
                    output_dir=root/'test', min_samples=8, threads=1)
            self.assertFalse(report['fitting_performed'])
            self.assertFalse(report['threshold_changed'])
            self.assertEqual(original, (resolver.directory/'metadata.json').read_bytes())
            with patch.object(sys, 'argv', ['run_backtest', '--strategy', 'swing-grid', '--swing-grid-model', result['directory'],
                '--swing-grid-threshold', '.2', '--swing-grid-min-threshold', '.1', '--swing-grid-threads', '1']):
                engine = build_engine([], backtest_args())
            self.assertIsInstance(engine.strategy, SwingGridStrategy)
            self.assertEqual(engine.strategy.threshold, .2)

    def test_search_uses_one_cached_common_cohort_for_repeats_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            setup_database(path)
            with patch('ultimateBot.backtester.research.swing_window_search.ensure_windows', side_effect=AssertionError('no context blobs')):
                plan = search_plan(path, run_id='raw', selection=selection(), representation='swing-grid', repeats=2)
                self.assertFalse(plan['grid_preflight']['duration_21600']['encoder_training'])
                result = search(path, run_id='raw', output_dir=root/'search', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=selection(), repeats=2, representation='swing-grid')
            self.assertEqual(result['representation'], 'swing-grid')
            self.assertEqual(len(result['trials']), 2)
            self.assertEqual(len(list((Path(result['directory'])/'trials/_swing_grid_cache').glob('*/manifest.json'))), 1)
            with patch('ultimateBot.backtester.research.swing_window_search.train', side_effect=AssertionError('reuse completed repeats')):
                resumed = search(path, run_id='raw', output_dir=root/'search', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=selection(), repeats=2, representation='swing-grid', resume=result['directory'])
            self.assertEqual(result['selected_directory'], resumed['selected_directory'])

    def test_depth_search_interrupt_resume_reuses_cohort_and_exports_selected_depth(self):
        from ultimateBot.backtester.research.swing_resolver import resolver_directory
        for policy in ('common', 'per-window'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root = Path(tmp)
                path = root/'data.sqlite3'
                setup_database(path)
                options = dict(run_id='raw', output_dir=root/'search', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=selection(), repeats=2, max_depths=[3, 5],
                    representation='swing-grid', cohort_policy=policy)
                with patch('ultimateBot.backtester.research.swing_window_search.train') as mocked:
                    def interrupt_at_second_depth(*args, **kwargs):
                        if mocked.call_count > 2:
                            raise KeyboardInterrupt()
                        return train(*args, **kwargs)
                    mocked.side_effect = interrupt_at_second_depth
                    with self.assertRaises(KeyboardInterrupt):
                        search(path, **options)
                directory = next((root/'search').glob('search_*'))
                partial = json.loads((directory/'search.json').read_text())
                self.assertEqual(partial['status'], 'interrupted')
                self.assertEqual([(r['max_depth'], r['repeat']) for r in partial['trials']], [(3, 1), (3, 2)])
                with patch('ultimateBot.backtester.research.swing_window_search.train', wraps=train) as mocked:
                    result = search(path, resume=directory, **{**options, 'max_depths': [5, 3, 3]})
                    self.assertEqual(mocked.call_count, 2)
                    self.assertEqual([(c.kwargs['config'].max_depth, c.kwargs['config'].seed)
                                      for c in mocked.call_args_list], [(5, CONFIG.seed), (5, CONFIG.seed+1)])
                self.assertEqual(result['plan']['model_pairs_to_train'], 8)
                self.assertEqual(len(result['leaderboard']), 4)
                self.assertEqual({(r['window_key'], r['max_depth']) for r in result['leaderboard']},
                    {(key, depth) for key in ('duration_21600', 'duration_28800') for depth in (3, 5)})
                self.assertEqual({r['repeats'] for r in result['leaderboard']}, {2})
                trials = [json.loads(Path(r['comparison']).read_text()) for r in result['trials']]
                self.assertEqual([(t['config']['max_depth'], t['config']['seed']) for t in trials],
                    [(3, 1), (3, 2), (5, 1), (5, 2)])
                self.assertEqual(len({t['dataset_sha256'] for t in trials}), 1)
                self.assertEqual(len(list((directory/'trials/_swing_grid_cache').glob('*/manifest.json'))),
                    1 if policy == 'common' else 2)
                selected = result['leaderboard'][0]
                representative = trials[selected['representative_trial']]
                self.assertEqual(result['selected_max_depth'], representative['config']['max_depth'])
                self.assertEqual(result['selected_directory'],
                    representative['windows'][selected['window_key']]['directory'])
                self.assertEqual(resolver_directory(directory), Path(result['selected_directory']))
                saved = json.loads((directory/'selected.json').read_text())
                self.assertEqual(saved['max_depth'], result['selected_max_depth'])
                self.assertEqual(saved['threshold'], result['selected_threshold'])
                resolver = SwingGridResolver(directory, threads=1)
                self.assertEqual(resolver.metadata['window_key'], result['selected_window_validation_only'])
                self.assertIn('Profondeur XGBoost retenue', (directory/'rapport.md').read_text(encoding='utf-8'))
                with (directory/'leaderboard.csv').open(newline='') as handle:
                    import csv
                    self.assertEqual({int(row['max_depth']) for row in csv.DictReader(handle)}, {3, 5})
                with patch('ultimateBot.backtester.research.swing_window_search.train', side_effect=AssertionError('reuse all')):
                    resumed = search(path, resume=directory, **options)
                self.assertEqual(resumed['selected_directory'], result['selected_directory'])
                with self.assertRaisesRegex(ValueError, 'Resume requires'):
                    search(path, resume=directory, **{**options, 'max_depths': [3, 7]})
                self.assertEqual(json.loads((directory/'search.json').read_text())['status'], 'completed')

    def test_per_window_fitting_is_invariant_to_adding_another_duration(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            setup_database(path)
            db = sqlite3.connect(path)
            db.row_factory = sqlite3.Row
            duplicate = dict(db.execute("SELECT * FROM swing_events WHERE event_id='EURUSD-160'").fetchone())
            duplicate.update(event_id='ambiguous', kind='low', side=1)
            columns = list(duplicate)
            db.execute(f"INSERT INTO swing_events ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                       [duplicate[name] for name in columns])
            db.commit()
            db.close()
            options = dict(run_id='raw', config=CONFIG, encoder_config=ENCODER, protocol=PROTOCOL,
                           representation='swing-grid', cohort_policy='per-window')
            single = train(path, output_dir=root/'models', selection=TrainingSelection(window_mode='duration', durations=(21600,)), **options)
            multiple = train(path, output_dir=root/'models', selection=selection(), **options)
            first = single['windows']['duration_21600']
            second = multiple['windows']['duration_21600']
            for name in ('metadata.json', 'predictions.csv'):
                a, b = Path(first['directory'])/name, Path(second['directory'])/name
                if name == 'metadata.json':
                    ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
                    for field in ('dataset_sha256', 'xgboost_sha256', 'threshold'):
                        self.assertEqual(ma[field], mb[field])
                else:
                    self.assertEqual(a.read_bytes(), b.read_bytes())
            report = second['report']
            self.assertGreater(report['own_split']['validation'], report['comparison_validation']['rows'])
            self.assertFalse(report['comparison_validation']['threshold_reselected'])
            self.assertEqual(multiple['common_validation_opportunities'], report['comparison_validation']['rows'])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['search', '--cohort-policy', 'per-window'])

    def test_per_window_search_ranks_common_validation_and_resumes_with_same_policy(self):
        from ultimateBot.backtester.research.swing_window_search import rank_windows
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            setup_database(path)
            options = dict(run_id='raw', output_dir=root/'search', config=CONFIG, encoder_config=ENCODER,
                           protocol=PROTOCOL, selection=selection(), repeats=1, representation='swing-grid', cohort_policy='per-window')
            result = search(path, **options)
            self.assertEqual(result['plan']['cohort_policy'], 'per-window')
            trial = json.loads(Path(result['trials'][0]['comparison']).read_text())
            expected = rank_windows([trial], list(trial['windows']))
            self.assertEqual(result['leaderboard'], expected)
            for key, entry in trial['windows'].items():
                common = entry['report']['comparison_validation']
                item = next(r for r in expected if r['window_key'] == key)
                self.assertEqual(item['median_validation_r'], common['selected']['total_r'])
            with patch('ultimateBot.backtester.research.swing_window_search.train', side_effect=AssertionError('reuse')):
                again = search(path, resume=result['directory'], **options)
            self.assertEqual(again['selected_directory'], result['selected_directory'])
            common_options = {**options, 'cohort_policy': 'common'}
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                search(path, resume=result['directory'], **common_options)


if __name__ == '__main__':
    unittest.main()
