from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.research.swing_contexts import ensure_windows
from ultimateBot.backtester.research.swing_encoder import SwingEncoderConfig
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, OpportunityConfig, collect, connect, milliseconds
from ultimateBot.backtester.research.swing_resolver import (
    ResearchProtocol, ResolverConfig, SwingResolver, TrainingSelection, frozen_test, resolver_directory,
    train,
)
from ultimateBot.backtester.research.swing_series import (
    FEATURE_SCHEMA, SWING_SCHEMA, SwingPoint, SwingSeries, SwingWindow, swing_features,
)
from ultimateBot.backtester.research.swing_window_search import rank_windows, rank_search_trials, search, search_plan
from ultimateBot.run_swing_ts2vec import main, parse_args, training_selection
from ultimateBot.tests.test_swing_ts2vec import bar, plus


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)
PROTOCOL = ResearchProtocol('2024-01-01', '2024-01-05', '2024-01-06', '2024-01-09')
CONFIG = ResolverConfig(min_encoder_samples=8, min_train_samples=10, min_validation_samples=8,
                        min_accepted_validation=3, n_estimators=8, early_stopping_rounds=2, threads=1)
ENCODER = SwingEncoderConfig(output_dims=4, hidden_dims=4, depth=1, train_iters=2, threads=1)


def database(path):
    db = connect(path)
    windows = (SwingWindow('pairs', 2), SwingWindow('duration', 4*3600))
    parameters = {'windows': [asdict(w) for w in windows], 'config': asdict(OpportunityConfig(stop_buffer_pips=2)),
        'symbols': ['EURUSD', 'GBPUSD'], 'source_fingerprint': 'raw-test',
        'feature_schema': FEATURE_SCHEMA, 'swing_schema': SWING_SCHEMA,
        'opportunity_schema': OPPORTUNITY_SCHEMA, 'risk_management_filters': False}
    db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES('raw',?,'completed',?)",
               (START.isoformat(), json.dumps(parameters)))
    for symbol in parameters['symbols']:
        series = SwingSeries(HOUR, windows)
        for i in range(320):
            end = START + (i+2)*HOUR
            kind = 'high' if i % 2 == 0 else 'low'
            price = 1.1 + (i % 7)*.0001 + (.001 if kind == 'high' else 0)
            swing = SwingPoint(kind, price, end-HOUR, end-2*HOUR, end)
            event_id = f'{symbol}-{i}'
            pnl = (-1., 0., 2.)[i % 3]
            db.execute("""INSERT INTO swing_events(event_id,run_id,symbol,kind,side,pivot_at,pair_start,
                confirmed_at,stop_price,pivot_price,expires_at,status,closed_at,pnl,pnl_r)
                VALUES(?,'raw',?,?,?,?,?,?,?,?,?,'closed',?,?,?)""", (event_id, symbol, kind, swing.side,
                swing.pivot_at.isoformat(), swing.pair_start.isoformat(), end.isoformat(),
                price-swing.side*.0002, price, (end+2*HOUR).isoformat(),
                (end+timedelta(minutes=20)).isoformat(), pnl*100, pnl))
            series.append_confirmed([swing])
            for window in windows:
                points = series.context(window, end)
                if points is not None:
                    db.execute("""INSERT INTO swing_contexts(event_id,window_key,information_start,context_end,
                        n_swings,span_seconds,features,swings_json) VALUES(?,?,?,?,?,?,?,?)""", (event_id, window.key,
                        points[0].pair_start.isoformat(), end.isoformat(), len(points),
                        (points[-1].confirmed_at-points[0].confirmed_at).total_seconds(),
                        swing_features(points, HOUR, end).astype('<f4').tobytes(),
                        json.dumps([s.to_dict() for s in points])))
    db.commit()
    return db


class WindowReplayTests(unittest.TestCase):
    def test_hard_shutdown_keeps_completed_symbols_and_resume_matches_uninterrupted_build(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path, reference_path = Path(tmp)/'data.sqlite3', Path(tmp)/'reference.sqlite3'
            db, reference = database(path), database(reference_path)
            original_events = [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
            db.close()
            windows = (SwingWindow('pairs', 3), SwingWindow('duration', 6*3600))
            ensure_windows(reference, 'raw', windows, before=PROTOCOL.bounds[3])
            # os._exit bypasses rollback/finally handlers, as a killed process does.
            script = '''
import os
import sys
from datetime import datetime
from unittest.mock import patch
from ultimateBot.backtester.research import swing_contexts
from ultimateBot.backtester.research.swing_opportunities import connect
from ultimateBot.backtester.research.swing_series import SwingWindow
db = connect(sys.argv[1])
features = swing_contexts.swing_features
def crash_during_second_symbol(*args, **kwargs):
    if db.execute("""SELECT 1 FROM swing_contexts c JOIN swing_events e USING(event_id)
                     WHERE e.symbol='GBPUSD' AND c.window_key='pairs_3' LIMIT 1""").fetchone():
        os._exit(73)
    return features(*args, **kwargs)
with patch.object(swing_contexts, 'swing_features', side_effect=crash_during_second_symbol):
    swing_contexts.ensure_windows(db, 'raw', (SwingWindow('pairs', 3), SwingWindow('duration', 21600)),
                                 before=datetime.fromisoformat(sys.argv[2]))
'''
            crashed = subprocess.run([sys.executable, '-c', script, str(path), PROTOCOL.bounds[3].isoformat()],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(crashed.returncode, 73, crashed.stdout+crashed.stderr)
            db = connect(path)
            checkpoints = [tuple(r) for r in db.execute('''SELECT window_key,symbol,completed_before
                FROM swing_context_build_checkpoints ORDER BY window_key,symbol''')]
            self.assertEqual(checkpoints, [(w.key, 'EURUSD', PROTOCOL.bounds[3].isoformat())
                                           for w in sorted(windows, key=lambda w: w.key)])
            self.assertGreater(db.execute("""SELECT count(*) FROM swing_contexts c
                JOIN swing_events e USING(event_id) WHERE e.symbol='EURUSD' AND c.window_key='pairs_3'""").fetchone()[0], 0)
            self.assertEqual(db.execute("""SELECT count(*) FROM swing_contexts c JOIN swing_events e USING(event_id)
                WHERE e.symbol='GBPUSD' AND c.window_key IN ('pairs_3','duration_21600')""").fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM swing_context_builds').fetchone()[0], 0)
            parameters = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
            self.assertNotIn(asdict(windows[0]), parameters['windows'])
            with patch('ultimateBot.backtester.research.swing_contexts.SwingSeries', wraps=SwingSeries) as replay:
                result = ensure_windows(db, 'raw', windows, before=PROTOCOL.bounds[3])
            self.assertEqual(replay.call_count, 1)
            self.assertGreater(result['inserted_contexts'], 0)
            self.assertEqual(original_events, [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')])
            query = '''SELECT event_id,window_key,information_start,context_end,n_swings,span_seconds,features,swings_json
                       FROM swing_contexts ORDER BY event_id,window_key'''
            self.assertEqual([tuple(r) for r in db.execute(query)], [tuple(r) for r in reference.execute(query)])
            self.assertEqual(db.execute('SELECT count(*) FROM swing_context_build_checkpoints').fetchone()[0], 4)
            self.assertEqual(db.execute('SELECT count(*) FROM swing_context_builds').fetchone()[0], 2)
            with patch('ultimateBot.backtester.research.swing_contexts.SwingSeries', side_effect=AssertionError('already saved')):
                self.assertEqual(ensure_windows(db, 'raw', windows, before=PROTOCOL.bounds[3])['inserted_contexts'], 0)
            db.close()
            reference.close()

    def test_checkpoints_are_specific_to_window_and_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db = database(Path(tmp)/'data.sqlite3')
            reference = database(Path(tmp)/'reference.sqlite3')
            windows = (SwingWindow('pairs', 3), SwingWindow('duration', 6*3600))
            features = swing_features

            def interrupt_second_symbol(*args, **kwargs):
                if db.execute("SELECT 1 FROM swing_context_build_checkpoints WHERE symbol='EURUSD'").fetchone():
                    raise KeyboardInterrupt()
                return features(*args, **kwargs)

            with patch('ultimateBot.backtester.research.swing_contexts.swing_features', side_effect=interrupt_second_symbol):
                with self.assertRaises(KeyboardInterrupt):
                    ensure_windows(db, 'raw', windows[:1], before=PROTOCOL.bounds[3])
            with patch('ultimateBot.backtester.research.swing_contexts.SwingSeries', wraps=SwingSeries) as replay:
                ensure_windows(db, 'raw', windows, before=PROTOCOL.bounds[3])
            self.assertEqual([call.args[1] for call in replay.call_args_list], [(windows[1],), windows])
            later = START+12*timedelta(days=1)
            with patch('ultimateBot.backtester.research.swing_contexts.SwingSeries', wraps=SwingSeries) as replay:
                ensure_windows(db, 'raw', windows, before=later)
            self.assertEqual([call.args[1] for call in replay.call_args_list], [windows, windows])
            self.assertEqual({r[0] for r in db.execute('SELECT completed_before FROM swing_context_build_checkpoints')},
                             {later.isoformat()})
            ensure_windows(reference, 'raw', windows, before=later)
            query = '''SELECT event_id,window_key,features,swings_json FROM swing_contexts ORDER BY event_id,window_key'''
            self.assertEqual([tuple(r) for r in db.execute(query)], [tuple(r) for r in reference.execute(query)])
            db.close()
            reference.close()

    def test_new_contexts_match_a_fresh_tick_collection_including_ambiguous_resets(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            folder = root/'history/EURUSD/2024/01'
            folder.mkdir(parents=True)
            lines = []
            for h in range(24):
                o, c = (100., 102.) if h % 2 == 0 else (102., 100.)
                ao, ac = o+10, c+10
                if h == 12:
                    ao, ac = 112., 111.
                elif h == 13:
                    ao, ac = 111., 112.
                for offset, bid, ask in ((1, o, ao), (1000, 103., 113.), (2000, 99., 109.), (3599, c, ac)):
                    ms = milliseconds(START+h*HOUR+timedelta(seconds=offset))
                    lines.append(f'QUOTE\tEURUSD\t{ms}\t1\t1\t{bid}\t{ask}\t1\t1')
            (folder/'2024-01-01.tsv').write_text('\n'.join(lines)+'\n')
            windows = (SwingWindow('pairs', 1), SwingWindow('pairs', 3), SwingWindow('duration', 21600))
            args = dict(history=root/'history', symbols=['EURUSD'], start=START.date(), end=START.date(),
                        config=OpportunityConfig(stop_buffer_pips=2))
            small = collect(root/'small.sqlite3', windows=windows[:1], store_contexts=True, **args)
            full = collect(root/'full.sqlite3', windows=windows, store_contexts=True, **args)
            db, reference = connect(root/'small.sqlite3'), connect(root/'full.sqlite3')
            ensure_windows(db, small['run_id'], windows[1:], before=START+timedelta(days=1))
            query = """SELECT window_key,information_start,context_end,n_swings,span_seconds,features,swings_json
                FROM swing_contexts WHERE window_key!='pairs_1' AND context_end<? ORDER BY window_key,context_end"""
            cutoff = ((START+timedelta(days=1)).isoformat(),)
            actual = [tuple(r) for r in db.execute(query, cutoff)]
            expected = [tuple(r) for r in reference.execute(query, cutoff)]
            db.close()
            reference.close()
            self.assertTrue(actual)
            self.assertEqual(actual, expected)
            self.assertGreater(full['counts']['EURUSD:ambiguous_simultaneous_reversals'], 0)

    def test_confirmed_replay_matches_candles_with_replacements_and_ambiguous_reset(self):
        windows = (SwingWindow('pairs', 1), SwingWindow('pairs', 2), SwingWindow('duration', 4*3600))
        live, replay = SwingSeries(HOUR, windows), SwingSeries(HOUR, windows)
        candles = [bar(i, o=100, h=103, l=99, c=102 if i % 2 == 0 else 99.5) for i in range(30)]
        for index, bid in enumerate(candles):
            ask = plus(bid)
            # Opposite bid/ask reversal candidates at the same time clear history.
            if index in (12, 13):
                ask = bar(index, o=101, h=104, l=98, c=99 if index == 12 else 103)
            live.append(bid, ask)
            replay.append_confirmed(live.last_candidates)
            if not live.last_candidates:
                continue
            as_of = live.last_candidates[0].confirmed_at
            for window in windows:
                expected = live.context(window, as_of)
                actual = replay.context(window, as_of)
                # At the very start the replay's known coverage can be shorter.
                if actual is not None:
                    self.assertEqual(actual, expected)
                    np.testing.assert_array_equal(swing_features(actual, HOUR, as_of),
                                                  swing_features(expected, HOUR, as_of))
        high = SwingPoint('high', 102, as_of, as_of-HOUR, as_of+HOUR)
        newer = SwingPoint('high', 101, as_of+HOUR, as_of, as_of+2*HOUR)
        replay.append_confirmed([high])
        replay.append_confirmed([newer])
        self.assertEqual(replay.swings[-1], newer)
        self.assertNotEqual(replay.swings[-2].kind, newer.kind)

    def test_new_windows_use_raw_pivot_not_buffered_stop_and_preserve_labels_and_existing_contexts(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db = database(Path(tmp)/'data.sqlite3')
            before = [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
            original = [tuple(r) for r in db.execute('SELECT * FROM swing_contexts ORDER BY context_id')]
            windows = (SwingWindow('pairs', 3), SwingWindow('duration', 6*3600))
            result = ensure_windows(db, 'raw', windows, before=PROTOCOL.bounds[3])
            self.assertGreater(result['inserted_contexts'], 0)
            self.assertEqual(before, [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')])
            self.assertEqual(original, [tuple(r) for r in db.execute(
                'SELECT * FROM swing_contexts WHERE context_id<=? ORDER BY context_id', (len(original),))])
            row = db.execute("""SELECT c.*, e.pivot_price,e.stop_price FROM swing_contexts c JOIN swing_events e USING(event_id)
                WHERE window_key='pairs_3' ORDER BY context_end LIMIT 1""").fetchone()
            self.assertEqual(row['n_swings'], 6)
            self.assertEqual(json.loads(row['swings_json'])[-1]['price'], row['pivot_price'])
            self.assertNotEqual(row['pivot_price'], row['stop_price'])
            self.assertEqual(db.execute("SELECT count(*) FROM swing_contexts WHERE window_key='pairs_3' AND context_end>=?",
                                      (PROTOCOL.bounds[3].isoformat(),)).fetchone()[0], 0)
            self.assertEqual(ensure_windows(db, 'raw', windows, before=PROTOCOL.bounds[3])['inserted_contexts'], 0)
            extended = ensure_windows(db, 'raw', windows, before=START+12*timedelta(days=1))
            self.assertGreater(extended['inserted_contexts'], 0)
            db.close()

    def test_build_does_not_read_future_raw_prices(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db = database(Path(tmp)/'data.sqlite3')
            db.execute('UPDATE swing_events SET pivot_price=-999 WHERE confirmed_at>=?', (PROTOCOL.bounds[3].isoformat(),))
            db.commit()
            result = ensure_windows(db, 'raw', (SwingWindow('pairs', 3),), before=PROTOCOL.bounds[3])
            self.assertGreater(result['inserted_contexts'], 0)
            db.close()


class WindowSearchTests(unittest.TestCase):
    def test_label_ranking_and_representative_ignore_pnl_across_depths(self):
        trials = []
        for depth,labels in ((3,[100.,.1,.2]),(5,[3.,4.,5.])):
            for i,label in enumerate(labels):
                trials.append(dict(config=dict(target='trailing-step-raw',max_depth=depth),
                    windows=dict(pairs_2=dict(directory=f'{depth}-{i}',report=dict(partitions=dict(validation=dict(
                        selected=dict(total_r=1000. if depth==3 else -1000.,mean_label=label,opportunities=1,coverage=.1))))))))
        ranked = rank_search_trials(trials,['pairs_2'],depth_grid=True)
        self.assertEqual([row['max_depth'] for row in ranked],[5,3])
        self.assertEqual(ranked[0]['median_validation_score'],4.)
        self.assertEqual(ranked[0]['median_validation_r'],-1000.)
        self.assertEqual(ranked[0]['representative_trial'],4)
        self.assertEqual(ranked[0]['directory'],'5-1')

    def test_depth_grid_cli_preview_is_read_only_and_single_depth_keeps_old_identity(self):
        args = parse_args(['search', '--max-depths', '9', '3', '5', '3', '--dry-run'])
        self.assertEqual(args.max_depths, [3, 5, 9])
        for options in (['--max-depths', '0'], ['--max-depths', '-1'], ['--max-depths', '2.5'],
                        ['--max-depths'], ['--max-depth', '3', '--max-depths', '3', '5']):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['search', *options])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            database(path).close()
            original = path.read_bytes()
            options = dict(config=CONFIG, encoder_config=ENCODER, protocol=PROTOCOL,
                           selection=TrainingSelection(pairs=(2,)), repeats=2)
            legacy = search_plan(path, **options)
            explicit = search_plan(path, max_depths=[3, 3], **options)
            self.assertEqual(explicit, legacy)
            self.assertNotIn('max_depths', legacy)
            plan = search_plan(path, max_depths=[5, 3, 5], **options)
            self.assertEqual(plan['max_depths'], [3, 5])
            self.assertEqual(plan['model_pairs_to_train'], 4)
            self.assertNotEqual(plan['plan_sha256'], legacy['plan_sha256'])
            for depths in ([], [True], [3.5], [-1], [0], ['5']):
                with self.subTest(depths=depths), self.assertRaisesRegex(ValueError, 'positive integers'):
                    search_plan(path, max_depths=depths, **options)
            # Exercise the actual CLI -> main -> search planner path.
            grid = main(['search', '--database', str(path), '--output-dir', str(root/'models'),
                '--run-id', 'raw', '--representation', 'swing-grid', '--durations', '6h', '8h',
                '--max-depths', '3', '5', '--repeats', '2', '--dry-run'])
            self.assertEqual(grid['model_pairs_to_train'], 8)
            self.assertEqual(grid['max_depths'], [3, 5])
            self.assertFalse((root/'models').exists())
            self.assertEqual(path.read_bytes(), original)

    def test_depth_ranking_keeps_seeds_separate_and_global_representative_index(self):
        def trial(depth, total, name):
            return dict(config=dict(max_depth=depth), windows=dict(pairs_2=dict(directory=name,
                report=dict(partitions=dict(validation=dict(selected=dict(total_r=total,
                    opportunities=10, coverage=.5)))))))
        trials = [trial(3, 100, 'shallow-best'), trial(3, -10, 'shallow-middle'), trial(3, -9, 'shallow-worst'),
                  trial(5, 4, 'deep-first'), trial(5, 5, 'deep-middle'), trial(5, 6, 'deep-last')]
        rows = rank_search_trials(trials, ['pairs_2'], depth_grid=True)
        self.assertEqual([row['max_depth'] for row in rows], [5, 3])
        self.assertEqual([row['repeats'] for row in rows], [3, 3])
        self.assertEqual(rows[0]['median_validation_r'], 5)
        self.assertEqual(rows[0]['representative_trial'], 4)
        self.assertEqual(rows[0]['directory'], 'deep-middle')
        self.assertEqual(rows[1]['median_validation_r'], -9)
        self.assertEqual(rows[1]['representative_trial'], 2)

    def test_depth_grid_also_trains_and_resumes_the_ts2vec_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'data.sqlite3'
            database(path).close()
            options = dict(output_dir=root/'models', config=CONFIG, encoder_config=ENCODER,
                protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)), repeats=1, max_depths=[2, 4])
            result = search(path, **options)
            self.assertEqual(len(result['leaderboard']), 2)
            self.assertEqual({row['max_depth'] for row in result['leaderboard']}, {2, 4})
            trials = [json.loads(Path(r['comparison']).read_text()) for r in result['trials']]
            self.assertEqual([t['config']['max_depth'] for t in trials], [2, 4])
            self.assertEqual([t['encoder_config']['seed'] for t in trials], [1, 1])
            self.assertEqual(trials[0]['dataset_sha256'], trials[1]['dataset_sha256'])
            resolver = SwingResolver(resolver_directory(result['directory']), threads=1)
            self.assertEqual(resolver.metadata['window_key'], 'pairs_2')
            with patch('ultimateBot.backtester.research.swing_window_search.train', side_effect=AssertionError('reuse')):
                resumed = search(path, resume=result['directory'], **options)
            self.assertEqual(resumed['selected_max_depth'], result['selected_max_depth'])
            self.assertEqual(resumed['selected_directory'], result['selected_directory'])

    def test_range_cli_merges_lists_and_rejects_invalid_or_conflicting_grids(self):
        args = parse_args(['search', '--duration-range', '6h', '24h', '6h', '--durations', '12h',
                           '--pairs-range', '2', '8', '2', '--pairs', '4', '--repeats', '5'])
        selection = training_selection(args)
        self.assertEqual(selection.durations, (43200, 21600, 64800, 86400))
        self.assertEqual(selection.pairs, (4, 2, 6, 8))
        self.assertEqual(args.repeats, 5)
        for options in (['--repeats', '0'], ['--pairs-range', '8', '2', '1'],
                        ['--pairs-range', '2', '8', '0'], ['--window-mode', 'duration', '--pairs-range', '2', '4', '1']):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['search', *options])

    def test_preview_is_read_only_and_preserves_full_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.sqlite3'
            db = database(path)
            db.close()
            previous = path.read_bytes()
            plan = search(path, output_dir=Path(tmp)/'models', selection=TrainingSelection(pairs=(1000,)), dry_run=True)
            self.assertEqual(plan['encoder_config']['max_train_length'], 0)
            self.assertEqual(plan['encoder_preflight']['pairs_1000']['status'], 'pending_context_build')
            self.assertEqual(plan['model_pairs_to_train'], 3)
            self.assertEqual(previous, path.read_bytes())
            self.assertFalse((Path(tmp)/'models').exists())
            with self.assertRaises(ValueError):
                search_plan(path, selection=TrainingSelection(interval_seconds=14400))
            with self.assertRaisesRegex(ValueError, 'length limit'):
                search_plan(path, selection=TrainingSelection(pairs=(1000,)),
                            encoder_config=SwingEncoderConfig(max_train_length=256))

    def test_ranking_uses_median_not_best_seed_and_selects_representative(self):
        def trial(a, b):
            return {'windows': {key: {'directory': key+str(a), 'report': {'partitions': {'validation': {
                'selected': {'total_r': value, 'opportunities': 10, 'coverage': .5}}}}}
                for key, value in [('pairs_2', a), ('duration_3600', b)]}}
        ranking = rank_windows([trial(100, 4), trial(-10, 5), trial(-9, 6)], ['pairs_2', 'duration_3600'])
        self.assertEqual(ranking[0]['window_key'], 'duration_3600')
        self.assertEqual(ranking[0]['median_validation_r'], 5)
        self.assertEqual(ranking[0]['representative_trial'], 1)
        self.assertEqual(ranking[1]['representative_trial'], 2)

    def test_real_search_new_modes_seeds_reload_and_frozen_test_extension(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            db = database(path)
            db.close()
            result = search(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                protocol=PROTOCOL, selection=TrainingSelection(durations=(21600,), pairs=(3,)), repeats=2)
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(len(result['trials']), 2)
            self.assertEqual({r['window_key'] for r in result['leaderboard']}, {'duration_21600', 'pairs_3'})
            resolver = SwingResolver(resolver_directory(result['directory']), threads=1)
            self.assertEqual(resolver.metadata['window_key'], result['selected_window_validation_only'])
            trials = [json.loads(Path(r['comparison']).read_text()) for r in result['trials']]
            self.assertEqual(trials[0]['dataset_sha256'], trials[1]['dataset_sha256'])
            self.assertEqual(trials[0]['split'], trials[1]['split'])
            self.assertEqual([r['encoder_config']['seed'] for r in trials], [1, 2])
            self.assertNotEqual(trials[0]['windows']['pairs_3']['report']['embedding_model_id'],
                                trials[1]['windows']['pairs_3']['report']['embedding_model_id'])
            before_hash = resolver.metadata['xgboost_sha256']
            tested = frozen_test(path, directory=result['directory'], run_id='raw', start='2024-01-10',
                                 end='2024-01-13', min_samples=8, threads=1)
            self.assertGreater(tested['labeled_contexts'], 0)
            self.assertEqual(SwingResolver(result['selected_directory'], threads=1).metadata['xgboost_sha256'], before_hash)
            db = connect(path)
            self.assertEqual(db.execute('SELECT status FROM swing_window_searches').fetchone()[0], 'completed')
            self.assertEqual(db.execute("SELECT count(*) FROM swing_predictions WHERE phase='test'").fetchone()[0],
                             tested['predicted_contexts'])
            db.close()
            with patch('ultimateBot.backtester.research.swing_window_search.train', side_effect=AssertionError('must reuse')):
                resumed = search(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=TrainingSelection(durations=(21600,), pairs=(3,)), repeats=2,
                    resume=result['directory'])
            self.assertEqual(resumed['selected_directory'], result['selected_directory'])
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                search(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=TrainingSelection(durations=(21600,), pairs=(3,)), repeats=3,
                    resume=result['directory'])
            db = connect(path)
            self.assertEqual(db.execute('SELECT status FROM swing_window_searches').fetchone()[0], 'completed')
            db.close()

    def test_interrupted_search_resumes_only_unfinished_repeats(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            db = database(path)
            db.close()
            with patch('ultimateBot.backtester.research.swing_window_search.train') as mocked:
                def interrupt_after_first(*args, **kwargs):
                    if mocked.call_count > 1:
                        raise KeyboardInterrupt()
                    return train(*args, **kwargs)
                mocked.side_effect = interrupt_after_first
                with self.assertRaises(KeyboardInterrupt):
                    search(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                           protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)), repeats=2)
            directory = next((Path(tmp)/'models').glob('search_*'))
            partial = json.loads((directory/'search.json').read_text())
            self.assertEqual(partial['status'], 'interrupted')
            self.assertEqual(len(partial['trials']), 1)
            with patch('ultimateBot.backtester.research.swing_window_search.train', wraps=train) as mocked:
                result = search(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                    protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)), repeats=2, resume=directory)
                self.assertEqual(mocked.call_count, 1)
            self.assertEqual(result['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
