from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ultimateBot.backtester.research.swing_encoder import SwingEncoderConfig
from ultimateBot.backtester.research.swing_opportunities import (
    OpportunityConfig, collect, connect, existing_collection, milliseconds, read_day,
)
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol, ResolverConfig, TrainingSelection, train
from ultimateBot.backtester.research.swing_series import SwingWindow
from ultimateBot.run_swing_ts2vec import main, opportunity_config, parse_args, windows_for


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CONFIG = OpportunityConfig(interval_seconds=600, stop_buffer_pips=10)
WINDOWS = (SwingWindow('duration', 1800), SwingWindow('pairs', 2))


def history(root, symbols):
    for symbol in symbols:
        for day in range(2):
            date = START+timedelta(days=day)
            path = root/symbol/'2024'/'01'/f'{date.date()}.tsv'
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            scale, base = (100, 150.) if symbol == 'USDJPY' else (1, 1.1)
            for h in range(48):
                level = base+.0006*scale*(h % 6)
                o, c = (level, level+.0008*scale) if h % 2 == 0 else (level+.0008*scale, level)
                for offset, price in ((1, o), (100, level+.0015*scale), (200, level-.0005*scale), (599, c)):
                    ms = milliseconds(date+timedelta(minutes=10*h, seconds=offset))
                    rows.append(f'QUOTE\t{symbol}\t{ms}\t1\t1\t{price}\t{price+.0001*scale}\t1\t1')
            path.write_text('\n'.join(rows)+'\n', encoding='utf-8')


def initial(root, symbols=('EURUSD',)):
    history(root/'history', [*symbols, 'USDJPY'])
    return collect(root/'data.sqlite3', history=root/'history', symbols=symbols,
                   start=START.date(), end=(START+timedelta(days=1)).date(), config=CONFIG, windows=WINDOWS, store_contexts=True)


def snapshot(db, run_id):
    return {'run': tuple(db.execute('SELECT * FROM swing_runs WHERE run_id=?', (run_id,)).fetchone()),
            'events': [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')],
            'contexts': [tuple(r) for r in db.execute('SELECT * FROM swing_contexts ORDER BY context_id')]}


class SwingCollectionAppendTests(unittest.TestCase):
    def test_append_same_run_preserves_previous_data_and_matches_independent_jpy_collection(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            first = initial(root)
            path, run_id = root/'data.sqlite3', first['run_id']
            db = connect(path)
            old = snapshot(db, run_id)
            original_parameters = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
            db.close()
            result = collect(path, history=root/'history', run_id=run_id, symbols=['EURUSD', 'USDJPY'])
            standalone = collect(root/'standalone.sqlite3', history=root/'history', symbols=['USDJPY'],
                start=START.date(), end=(START+timedelta(days=1)).date(), config=CONFIG, windows=WINDOWS, store_contexts=True)
            db, reference = connect(path), connect(root/'standalone.sqlite3')
            params = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
            old_events = [tuple(r) for r in db.execute("SELECT * FROM swing_events WHERE symbol='EURUSD' ORDER BY event_id")]
            old_contexts = [tuple(r) for r in db.execute('SELECT * FROM swing_contexts WHERE context_id<=? ORDER BY context_id',
                                                       (len(old['contexts']),))]
            columns = [r[1] for r in db.execute('PRAGMA table_info(swing_events)') if r[1] not in ('event_id', 'run_id')]
            fields = ','.join(columns)
            actual = [tuple(r) for r in db.execute(f"SELECT {fields} FROM swing_events WHERE symbol='USDJPY' ORDER BY confirmed_at")]
            expected = [tuple(r) for r in reference.execute(f'SELECT {fields} FROM swing_events ORDER BY confirmed_at')]
            contexts_query = """SELECT c.window_key,c.context_end,c.features,c.swings_json FROM swing_contexts c
                JOIN swing_events e USING(event_id) WHERE e.symbol='USDJPY' ORDER BY c.context_end,c.window_key"""
            actual_contexts = [tuple(r) for r in db.execute(contexts_query)]
            expected_contexts = [tuple(r) for r in reference.execute(contexts_query)]
            run_count = db.execute('SELECT count(*) FROM swing_runs').fetchone()[0]
            extension_status = db.execute('SELECT status FROM swing_run_extensions').fetchone()[0]
            db.close()
            reference.close()
            self.assertEqual(result['run_id'], run_id)
            self.assertEqual(result['added_symbols'], ['USDJPY'])
            self.assertEqual(result['skipped_existing_symbols'], ['EURUSD'])
            self.assertEqual(old_events, old['events'])
            self.assertEqual(old_contexts, old['contexts'])
            self.assertEqual(actual, expected)
            self.assertEqual(actual_contexts, expected_contexts)
            self.assertEqual(run_count, 1)
            self.assertEqual(extension_status, 'completed')
            self.assertEqual(params['symbols'], ['EURUSD', 'USDJPY'])
            self.assertEqual(params['config'], original_parameters['config'])
            self.assertEqual(params['windows'], original_parameters['windows'])
            self.assertNotEqual(params['source_fingerprint'], original_parameters['source_fingerprint'])
            self.assertEqual(len(params['source_files']), 2*len(original_parameters['source_files']))
            self.assertEqual(sum(result['statuses'].values()), sum(first['statuses'].values())+sum(standalone['statuses'].values()))
            with patch('ultimateBot.backtester.research.swing_opportunities.read_day', side_effect=AssertionError('already collected')):
                again = collect(path, history=root/'history', run_id=run_id, symbols=['USDJPY'])
            self.assertEqual(again['added_symbols'], [])
            self.assertEqual(again['run_id'], run_id)

    def test_conflicting_parameters_and_unknown_run_do_not_change_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_id = initial(root)['run_id']
            path = root/'data.sqlite3'
            db = connect(path)
            before = snapshot(db, run_id)
            db.close()
            for options in ({'config': OpportunityConfig()}, {'windows': (SwingWindow('pairs', 9),)},
                            {'start': START.date()-timedelta(days=1)}, {'end': START.date()}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'preserve'):
                    collect(path, history=root/'history', run_id=run_id, symbols=['USDJPY'], **options)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                collect(path, history=root/'history', run_id='missing', symbols=['USDJPY'])
            db = connect(path)
            self.assertEqual(snapshot(db, run_id), before)
            self.assertEqual(db.execute('SELECT count(*) FROM swing_run_extensions').fetchone()[0], 0)
            db.close()

    def test_failed_or_interrupted_append_removes_only_new_data_and_can_retry(self):
        for error_class in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error_class), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root = Path(tmp)
                run_id = initial(root)['run_id']
                path = root/'data.sqlite3'
                db = connect(path)
                before = snapshot(db, run_id)
                db.close()
                def fail_second_day(file, symbol):
                    if file.stem == '2024-01-02':
                        db = connect(path)
                        partial = db.execute("SELECT count(*) FROM swing_events WHERE symbol='USDJPY'").fetchone()[0]
                        db.close()
                        self.assertGreater(partial, 0)
                        raise error_class('simulated interruption')
                    return read_day(file, symbol)
                with patch('ultimateBot.backtester.research.swing_opportunities.read_day', side_effect=fail_second_day):
                    with self.assertRaises(error_class):
                        collect(path, history=root/'history', run_id=run_id, symbols=['USDJPY'])
                db = connect(path)
                after = snapshot(db, run_id)
                failed_status = db.execute('SELECT status FROM swing_run_extensions').fetchone()[0]
                db.close()
                self.assertEqual(after, before)
                self.assertEqual(failed_status, 'failed')
                result = collect(path, history=root/'history', run_id=run_id, symbols=['USDJPY'])
                self.assertEqual(result['added_symbols'], ['USDJPY'])

    def test_running_collection_cannot_be_extended(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_id = initial(root)['run_id']
            path = root/'data.sqlite3'
            db = connect(path)
            db.execute("UPDATE swing_runs SET status='running'")
            db.commit()
            db.close()
            with self.assertRaisesRegex(ValueError, 'wait for the running collector'):
                existing_collection(path, run_id)
            with self.assertRaisesRegex(ValueError, 'completed collection'):
                collect(path, history=root/'history', run_id=run_id, symbols=['USDJPY'])

    def test_cli_inherits_nondefault_settings_and_preview_does_not_modify_database(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_id = initial(root)['run_id']
            path = root/'data.sqlite3'
            options = ['collect', '--database', str(path), '--history', str(root/'history'), '--run-id', run_id]
            args = parse_args([*options, '--symbols', 'USDJPY'])
            self.assertEqual(opportunity_config(args), CONFIG)
            self.assertEqual(set(windows_for(args)), set(WINDOWS))
            self.assertEqual(str(args.start_date), '2024-01-01')
            self.assertEqual(str(args.end_date), '2024-01-02')
            original = path.read_bytes()
            preview = main([*options, '--symbols', 'USDJPY', '--dry-run'])
            self.assertEqual(preview['run_id'], run_id)
            self.assertEqual(preview['operation'], 'append_symbols')
            self.assertEqual(path.read_bytes(), original)
            for extra in ([], ['--symbols', 'USDJPY', '--candle-interval', '60m'],
                          ['--symbols', 'USDJPY', '--stop-buffer-pips', '0']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args([*options, *extra])
            self.assertEqual(main([*options, '--symbols', 'USDJPY'])['run_id'], run_id)

    def test_training_after_append_pools_only_selected_symbols(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_id = initial(root, symbols=('EURUSD', 'GBPUSD'))['run_id']
            path = root/'data.sqlite3'
            collect(path, history=root/'history', run_id=run_id, symbols=['USDJPY'])
            cfg = ResolverConfig(min_encoder_samples=8, min_train_samples=8, min_validation_samples=8,
                min_accepted_validation=3, n_estimators=6, early_stopping_rounds=2, threads=1)
            encoder = SwingEncoderConfig(output_dims=4, hidden_dims=4, depth=1, train_iters=2, threads=1)
            result = train(path, run_id=run_id, output_dir=root/'models', config=cfg, encoder_config=encoder,
                protocol=ResearchProtocol('2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'),
                selection=TrainingSelection(symbols=('EURUSD', 'USDJPY'), pairs=(2,)))
            self.assertEqual(result['scope']['symbols'], ['EURUSD', 'USDJPY'])
            db = connect(path)
            symbols = {r[0] for r in db.execute("""SELECT DISTINCT e.symbol FROM swing_assignments a
                JOIN swing_contexts c USING(context_id) JOIN swing_events e USING(event_id) WHERE experiment_id=?""",
                (result['experiment_id'],))}
            db.close()
            self.assertEqual(symbols, {'EURUSD', 'USDJPY'})


if __name__ == '__main__':
    unittest.main()
