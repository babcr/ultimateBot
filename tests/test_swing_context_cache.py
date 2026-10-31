from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.research import swing_contexts
from ultimateBot.backtester.research.swing_contexts import context_build_plan, ensure_windows
from ultimateBot.backtester.research.swing_opportunities import connect
from ultimateBot.backtester.research.swing_series import SwingWindow
from ultimateBot.tests.test_swing_window_search import database, START
from ultimateBot.tests.test_swing_collection_append import initial
from ultimateBot.backtester.research.swing_opportunities import collect


WINDOWS = (SwingWindow('pairs', 3), SwingWindow('duration', 6*3600))
DAY = timedelta(days=1)


def contexts(db, symbol, start, before):
    return [tuple(r) for r in db.execute('''SELECT c.event_id,c.window_key,c.information_start,c.context_end,
        c.n_swings,c.span_seconds,c.features,c.swings_json FROM swing_contexts c
        JOIN swing_events e USING(event_id) WHERE e.symbol=? AND c.window_key IN ('pairs_3','duration_21600')
        AND c.context_end>=? AND c.context_end<? ORDER BY c.event_id,c.window_key''',
        (symbol, start.isoformat(), before.isoformat()))]


def reset_and_replacements(db):
    # Place a reset shortly before the scoped range, and a run of replacements
    # across its boundary. Neither may require replaying the entire history.
    db.execute('''INSERT INTO swing_events(event_id,run_id,symbol,kind,side,pivot_at,pair_start,
        confirmed_at,stop_price,pivot_price,expires_at,status,closed_at,pnl,pnl_r)
        SELECT event_id||'-ambiguous',run_id,symbol,'high',-1,pivot_at,pair_start,
        confirmed_at,stop_price,pivot_price,expires_at,status,closed_at,pnl,pnl_r
        FROM swing_events WHERE event_id='EURUSD-115' ''')
    db.execute("UPDATE swing_events SET kind='high',side=-1 WHERE event_id IN ('EURUSD-118','EURUSD-119','EURUSD-120')")
    db.commit()


class ScopedContextCacheTests(unittest.TestCase):
    def test_newly_collected_symbol_records_full_coverage_of_a_previously_scoped_window(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            result = initial(root)
            db = connect(root/'data.sqlite3')
            window = SwingWindow('pairs', 3)
            ensure_windows(db, result['run_id'], (window,), start=START+DAY, before=START+2*DAY, symbols=['EURUSD'])
            db.close()
            collect(root/'data.sqlite3', history=root/'history', run_id=result['run_id'], symbols=['USDJPY'])
            db = connect(root/'data.sqlite3')
            with patch.object(swing_contexts, '_replay', side_effect=AssertionError('collector already built these contexts')):
                ensure_windows(db, result['run_id'], (window,), start=START, before=START+2*DAY, symbols=['USDJPY'])
            plan = context_build_plan(db, result['run_id'], (window,), start=START, before=START+2*DAY, symbols=['EURUSD'])
            self.assertEqual(plan['windows'][window.key]['missing_ranges']['EURUSD'],
                             [[START.isoformat(), (START+DAY).isoformat()]])
            db.close()

    def test_random_replacements_and_resets_match_full_replay_across_cached_ranges(self):
        windows = WINDOWS+(SwingWindow('pairs', 16), SwingWindow('duration', 24*3600))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db, reference = database(Path(tmp)/'scoped.db'), database(Path(tmp)/'reference.db')
            rng = np.random.default_rng(27)
            kinds = ['high' if value else 'low' for value in rng.integers(0, 2, 320)]
            resets = rng.choice(320, 12, replace=False)
            for connection in (db, reference):
                for i, kind in enumerate(kinds):
                    connection.execute('UPDATE swing_events SET kind=?,side=? WHERE event_id=?',
                                       (kind, -1 if kind == 'high' else 1, f'EURUSD-{i}'))
                for i in resets:
                    connection.execute('''INSERT INTO swing_events(event_id,run_id,symbol,kind,side,pivot_at,pair_start,
                        confirmed_at,stop_price,pivot_price,expires_at,status,closed_at,pnl,pnl_r)
                        SELECT event_id||'-reset',run_id,symbol,CASE kind WHEN 'high' THEN 'low' ELSE 'high' END,
                        -side,pivot_at,pair_start,confirmed_at,stop_price,pivot_price,expires_at,status,closed_at,pnl,pnl_r
                        FROM swing_events WHERE event_id=?''', (f'EURUSD-{i}',))
                connection.commit()
            ensure_windows(reference, 'raw', windows, before=START+13*DAY)
            for left, right in [(5, 7), (9, 12), (3, 10), (1, 13)]:
                ensure_windows(db, 'raw', windows, start=START+left*DAY, before=START+right*DAY, symbols=['EURUSD'])
                query = '''SELECT c.event_id,c.window_key,c.features,c.swings_json FROM swing_contexts c
                    JOIN swing_events e USING(event_id) WHERE e.symbol='EURUSD'
                    AND c.window_key IN ('pairs_3','pairs_16','duration_21600','duration_86400')
                    AND c.context_end>=? AND c.context_end<? ORDER BY c.event_id,c.window_key'''
                bounds = ((START+left*DAY).isoformat(), (START+right*DAY).isoformat())
                self.assertEqual([tuple(r) for r in db.execute(query, bounds)],
                                 [tuple(r) for r in reference.execute(query, bounds)])
            db.close()
            reference.close()

    def test_scoped_replay_matches_full_history_and_never_builds_unrequested_symbols_or_dates(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db, reference = database(Path(tmp)/'scoped.db'), database(Path(tmp)/'reference.db')
            for connection in (db, reference):
                reset_and_replacements(connection)
            start, before = START+5*DAY, START+9*DAY
            original = [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
            with patch.object(swing_contexts, '_replay', wraps=swing_contexts._replay) as replay:
                result = ensure_windows(db, 'raw', WINDOWS, start=start, before=before, symbols=['EURUSD'])
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.args[2], 'EURUSD')
            self.assertEqual(replay.call_args.args[4:6], (start, before))
            self.assertGreaterEqual(swing_contexts._warmup_start(db, 'raw', 'EURUSD', WINDOWS, start,
                START, timedelta(hours=1)), start-timedelta(hours=7))
            ensure_windows(reference, 'raw', WINDOWS, before=before)
            self.assertGreater(result['inserted_contexts'], 0)
            self.assertEqual(contexts(db, 'EURUSD', start, before), contexts(reference, 'EURUSD', start, before))
            self.assertEqual(len(contexts(db, 'EURUSD', START, before)), len(contexts(db, 'EURUSD', start, before)))
            self.assertEqual(contexts(db, 'GBPUSD', START, before), [])
            self.assertEqual(original, [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')])
            db.close()
            reference.close()

    def test_cached_subranges_skip_event_replay_and_extensions_only_fill_gaps(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db = database(Path(tmp)/'scoped.db')
            ensure_windows(db, 'raw', WINDOWS, start=START+5*DAY, before=START+7*DAY, symbols=['EURUSD'])
            with patch.object(swing_contexts, '_replay', side_effect=AssertionError('must reuse cached contexts')):
                result = ensure_windows(db, 'raw', WINDOWS, start=START+5*DAY, before=START+6*DAY, symbols=['EURUSD'])
            self.assertEqual(result['inserted_contexts'], 0)
            self.assertEqual(set(result['reused_windows']), {w.key for w in WINDOWS})
            with patch.object(swing_contexts, '_replay', wraps=swing_contexts._replay) as replay:
                ensure_windows(db, 'raw', WINDOWS, start=START+4*DAY, before=START+8*DAY, symbols=['EURUSD'])
            self.assertEqual([call.args[4:6] for call in replay.call_args_list],
                             [(START+4*DAY, START+5*DAY), (START+7*DAY, START+8*DAY)])
            # Coverage for EURUSD must never imply coverage for GBPUSD.
            with patch.object(swing_contexts, '_replay', wraps=swing_contexts._replay) as replay:
                ensure_windows(db, 'raw', WINDOWS, start=START+5*DAY, before=START+7*DAY, symbols=['GBPUSD'])
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.args[2], 'GBPUSD')
            self.assertEqual(replay.call_args.args[4:6], (START+5*DAY, START+7*DAY))
            # The old full-history marker must not claim all dates are built.
            plan = context_build_plan(db, 'raw', WINDOWS, start=START+3*DAY, before=START+9*DAY, symbols=['EURUSD'])
            for window in WINDOWS:
                self.assertEqual(plan['windows'][window.key]['missing_ranges']['EURUSD'], [
                    [(START+3*DAY).isoformat(), (START+4*DAY).isoformat()],
                    [(START+8*DAY).isoformat(), (START+9*DAY).isoformat()]])
            db.close()

    def test_disjoint_ranges_can_later_fill_full_history_without_changing_contexts(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db, reference = database(Path(tmp)/'scoped.db'), database(Path(tmp)/'reference.db')
            for start_day, end_day in [(5, 7), (9, 11), (3, 10)]:
                ensure_windows(db, 'raw', WINDOWS, start=START+start_day*DAY, before=START+end_day*DAY, symbols=['EURUSD'])
            ensure_windows(db, 'raw', WINDOWS, before=START+12*DAY, symbols=['EURUSD'])
            ensure_windows(reference, 'raw', WINDOWS, before=START+12*DAY)
            self.assertEqual(contexts(db, 'EURUSD', START, START+12*DAY),
                             contexts(reference, 'EURUSD', START, START+12*DAY))
            with patch.object(swing_contexts, '_replay', side_effect=AssertionError('full prefix already covered')):
                ensure_windows(db, 'raw', WINDOWS, start=START+2*DAY, before=START+11*DAY, symbols=['EURUSD'])
            self.assertTrue(any(context_build_plan(db, 'raw', WINDOWS, start=START+2*DAY,
                before=START+11*DAY, symbols=['GBPUSD'])['windows'][w.key]['missing_ranges']['GBPUSD'] for w in WINDOWS))
            db.close()
            reference.close()

    def test_legacy_full_builds_are_reused_and_initial_collection_coverage_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db = database(Path(tmp)/'legacy.db')
            ensure_windows(db, 'raw', WINDOWS, before=START+9*DAY)
            db.execute('DELETE FROM swing_context_coverage')
            db.execute('DELETE FROM swing_context_build_checkpoints')
            db.execute("UPDATE swing_context_builds SET metadata_json=?", (json.dumps({'method': 'confirmed_event_replay_v1'}),))
            db.execute('DROP TABLE swing_context_coverage')
            db.execute('DROP TABLE swing_context_build_checkpoints')
            db.commit()
            with patch.object(swing_contexts, '_replay', side_effect=AssertionError('legacy coverage is valid')):
                ensure_windows(db, 'raw', WINDOWS, start=START+5*DAY, before=START+7*DAY, symbols=['EURUSD'])
                ensure_windows(db, 'raw', (SwingWindow('pairs', 2),), start=START+5*DAY, before=START+7*DAY, symbols=['EURUSD'])
            db.close()

    def test_empty_results_are_cached_and_invalid_selectors_do_not_write(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            db = database(Path(tmp)/'data.db')
            window = SwingWindow('pairs', 1000)
            result = ensure_windows(db, 'raw', (window,), start=START+5*DAY, before=START+7*DAY, symbols=['EURUSD'])
            self.assertEqual(result['inserted_contexts'], 0)
            with patch.object(swing_contexts, '_replay', side_effect=AssertionError('a completed empty result is cached')):
                ensure_windows(db, 'raw', (window,), start=START+5*DAY, before=START+7*DAY, symbols=['EURUSD'])
            previous = db.total_changes
            for kwargs in ({'symbols': ['AUDUSD']}, {'symbols': []}, {'start': START.replace(tzinfo=None)}, {'start': START+9*DAY}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    ensure_windows(db, 'raw', WINDOWS, before=START+8*DAY, **kwargs)
            self.assertEqual(previous, db.total_changes)
            db.close()

    def test_hard_shutdown_preserves_scoped_coverage_and_resume_skips_completed_symbol(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.db'
            db = database(path)
            db.close()
            script = '''
import os, sys
from datetime import datetime
from unittest.mock import patch
from ultimateBot.backtester.research import swing_contexts
from ultimateBot.backtester.research.swing_opportunities import connect
from ultimateBot.backtester.research.swing_series import SwingWindow
db=connect(sys.argv[1])
original=swing_contexts.swing_features
def crash(*args, **kwargs):
    if db.execute("""SELECT 1 FROM swing_contexts c JOIN swing_events e USING(event_id)
                     WHERE e.symbol='GBPUSD' AND c.window_key='pairs_3' LIMIT 1""").fetchone():
        os._exit(73)
    return original(*args, **kwargs)
with patch.object(swing_contexts,'swing_features',side_effect=crash):
    swing_contexts.ensure_windows(db,'raw',(SwingWindow('pairs',3),),
        start=datetime.fromisoformat(sys.argv[2]),before=datetime.fromisoformat(sys.argv[3]),
        symbols=['EURUSD','GBPUSD'])
'''
            start, before = START+5*DAY, START+7*DAY
            crashed = subprocess.run([sys.executable, '-c', script, str(path), start.isoformat(), before.isoformat()],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(crashed.returncode, 73, crashed.stdout+crashed.stderr)
            db = connect(path)
            self.assertEqual([r[0] for r in db.execute('SELECT symbol FROM swing_context_coverage')], ['EURUSD'])
            self.assertEqual(contexts(db, 'GBPUSD', start, before), [])
            with patch.object(swing_contexts, '_replay', wraps=swing_contexts._replay) as replay:
                ensure_windows(db, 'raw', (WINDOWS[0],), start=start, before=before, symbols=['EURUSD','GBPUSD'])
            self.assertEqual(replay.call_count, 1)
            self.assertEqual(replay.call_args.args[2], 'GBPUSD')
            db.close()


if __name__ == '__main__':
    unittest.main()
