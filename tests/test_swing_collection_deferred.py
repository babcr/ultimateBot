from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ultimateBot.backtester.research.swing_opportunities import collect, connect, OpportunityConfig
from ultimateBot.backtester.research.swing_contexts import ensure_windows, context_build_plan
from ultimateBot.backtester.research.swing_series import SwingWindow
from ultimateBot.backtester.research.swing_neural import prepare_data, fit, NeuralConfig
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol, TrainingSelection, train
from ultimateBot.run_swing_ts2vec import parse_args, main
from ultimateBot.tests.test_swing_collection_append import history, START
from ultimateBot.tests.test_swing_window_search import CONFIG, ENCODER

WINDOWS = (SwingWindow('duration', 1800), SwingWindow('pairs', 2))
SETTINGS = OpportunityConfig(interval_seconds=600, stop_variants=3, stop_buffer_pips=2., max_holding_seconds=7200)
PROTOCOL = ResearchProtocol('2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02')


def collected(root, filename='raw.sqlite3', symbols=('EURUSD',), **kwargs):
    return collect(root/filename, history=root/'ticks', symbols=symbols, start=START.date(),
        end=(START+timedelta(days=1)).date(), config=SETTINGS, windows=WINDOWS, **kwargs)


def outcomes(db):
    columns = [r[1] for r in db.execute('PRAGMA table_info(swing_events)') if r[1] not in ('event_id', 'run_id')]
    return [tuple(r) for r in db.execute(f"SELECT {','.join(columns)} FROM swing_events ORDER BY symbol,confirmed_at,kind,variant_index")]


def payloads(db):
    return [tuple(r) for r in db.execute('''SELECT e.symbol,e.confirmed_at,e.variant_index,c.window_key,
        c.information_start,c.context_end,c.n_swings,c.span_seconds,c.features,c.swings_json
        FROM swing_contexts_resolved c JOIN swing_events e USING(event_id)
        ORDER BY e.symbol,e.confirmed_at,e.kind,e.variant_index,c.window_key''')]


class DeferredCollectionTests(unittest.TestCase):
    def test_default_never_builds_contexts_and_matches_all_event_outcomes_with_volume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            history(root/'ticks', ['EURUSD','USDJPY'])
            with (patch('ultimateBot.backtester.research.swing_opportunities.store_context', side_effect=AssertionError('no blobs')),
                 patch('ultimateBot.backtester.research.swing_opportunities.swing_features', side_effect=AssertionError('no encoding')),
                 patch('ultimateBot.backtester.research.swing_series.SwingSeries.context', side_effect=AssertionError('no windows'))):
                raw = collected(root, symbols=('EURUSD','USDJPY'), include_volume=True)
            reference = collected(root, 'legacy.sqlite3', symbols=('EURUSD','USDJPY'), include_volume=True, store_contexts=True)
            db, other = connect(root/'raw.sqlite3'), connect(root/'legacy.sqlite3')
            try:
                self.assertFalse(raw['store_contexts'])
                self.assertEqual(raw['contexts'], {})
                self.assertTrue(reference['contexts'])
                self.assertEqual(outcomes(db), outcomes(other))
                self.assertEqual(db.execute('SELECT count(*) FROM swing_contexts').fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT count(*) FROM swing_context_coverage').fetchone()[0], 0)
                self.assertTrue(all(r[0] is not None for r in db.execute('SELECT mean_quote_volume FROM swing_events')))
                self.assertFalse(json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])['store_contexts'])
            finally:
                db.close()
                other.close()

    def test_later_windowing_matches_eager_contexts_and_is_reused_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            history(root/'ticks', ['EURUSD'])
            raw = collected(root)
            collected(root, 'legacy.sqlite3', store_contexts=True)
            db, other = connect(root/'raw.sqlite3'), connect(root/'legacy.sqlite3')
            try:
                original = outcomes(db)
                plan = context_build_plan(db, raw['run_id'], WINDOWS, before=START+timedelta(days=2))
                self.assertTrue(all(v['missing_ranges']['EURUSD'] for v in plan['windows'].values()))
                build = ensure_windows(db, raw['run_id'], WINDOWS, before=START+timedelta(days=2))
                self.assertGreater(build['inserted_contexts'], 0)
                self.assertEqual(payloads(db), payloads(other))
                self.assertEqual(outcomes(db), original)
                with patch('ultimateBot.backtester.research.swing_contexts._replay', side_effect=AssertionError('must reuse')):
                    repeated = ensure_windows(db, raw['run_id'], WINDOWS, before=START+timedelta(days=2))
                self.assertEqual(repeated['inserted_contexts'], 0)
            finally:
                db.close()
                other.close()

    def test_neural_grid_never_materializes_contexts_and_ts2vec_builds_new_windows_at_train(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            history(root/'ticks', ['EURUSD'])
            raw = collected(root)
            path = root/'raw.sqlite3'
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('ultimateBot.backtester.research.swing_opportunities.store_context', side_effect=AssertionError('no context blobs')):
                data = prepare_data(path, run_id=raw['run_id'], symbols=['EURUSD'], duration=2400,
                    protocol=PROTOCOL, cache_root=root/'cache')
                try:
                    self.assertEqual(data.x.shape[1], 5)
                    fit(data, root/'nn', NeuralConfig(epochs=1, batch_size=32, threads=1,
                        min_train_samples=10, min_validation_samples=8))
                finally:
                    data.close()
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            result = train(path, run_id=raw['run_id'], output_dir=root/'models',
                config=replace(CONFIG, n_estimators=2), encoder_config=replace(ENCODER, train_iters=1), protocol=PROTOCOL,
                selection=TrainingSelection(symbols=('EURUSD',), window_mode='duration', durations=(2400,)))
            self.assertIn('duration_2400', result['windows'])
            db = connect(path)
            try:
                self.assertGreater(db.execute('SELECT count(*) FROM swing_contexts').fetchone()[0], 0)
            finally:
                db.close()

    def test_raw_append_keeps_contexts_deferred_and_does_not_inherit_another_symbols_coverage(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            history(root/'ticks', ['EURUSD','USDJPY'])
            raw = collected(root)
            path, run = root/'raw.sqlite3', raw['run_id']
            window = SwingWindow('duration', 2400)
            db = connect(path)
            ensure_windows(db, run, (window,), before=START+timedelta(days=2))
            count = db.execute('SELECT count(*) FROM swing_contexts').fetchone()[0]
            db.close()
            args = ['collect','--database',str(path),'--run-id',run,'--symbols','USDJPY']
            self.assertFalse(parse_args(args).store_contexts)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([*args,'--store-contexts'])
            collect(path, run_id=run, history=root/'ticks', symbols=['USDJPY'])
            db = connect(path)
            try:
                self.assertEqual(db.execute('SELECT count(*) FROM swing_contexts').fetchone()[0], count)
                plan = context_build_plan(db, run, (window,), before=START+timedelta(days=2))
                coverage = plan['windows'][window.key]['missing_ranges']
                self.assertEqual(coverage['EURUSD'], [])
                self.assertTrue(coverage['USDJPY'])
                ensure_windows(db, run, (window,), before=START+timedelta(days=2), symbols=['USDJPY'])
                self.assertGreater(db.execute('SELECT count(*) FROM swing_contexts').fetchone()[0], count)
            finally:
                db.close()

    def test_cli_default_opt_in_and_append_of_legacy_run_without_storage_flag(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.assertFalse(parse_args(['collect']).store_contexts)
            self.assertTrue(parse_args(['collect','--store-contexts']).store_contexts)
            self.assertFalse(parse_args(['collect','--no-store-contexts']).store_contexts)
            preview = main(['collect','--dry-run'])
            self.assertEqual(preview['windowing'], 'deferred_to_train_search')
            history(root/'ticks', ['EURUSD','USDJPY'])
            old = collected(root, store_contexts=True)
            path = root/'raw.sqlite3'
            db = connect(path)
            params = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
            params.pop('store_contexts')
            db.execute('UPDATE swing_runs SET parameters_json=?', (json.dumps(params),))
            db.commit()
            db.close()
            args = parse_args(['collect','--database',str(path),'--run-id',old['run_id'],'--symbols','USDJPY'])
            self.assertTrue(args.store_contexts)
            appended = collect(path, run_id=old['run_id'], history=root/'ticks', symbols=['USDJPY'])
            self.assertTrue(appended['store_contexts'])
            db = connect(path)
            try:
                self.assertGreater(db.execute("SELECT count(*) FROM swing_contexts c JOIN swing_events e USING(event_id) WHERE e.symbol='USDJPY'").fetchone()[0], 0)
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
