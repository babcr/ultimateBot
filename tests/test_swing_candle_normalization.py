from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.research.swing_candle_prices import CandleCloseIndex, candle_close_mean
from ultimateBot.backtester.research.swing_grid import CANDLE_CLOSE_GRID_SCHEMA, SwingGridResolver, grid_vector, prepare
from ultimateBot.backtester.research.swing_resolver import frozen_test, predict, train
from ultimateBot.backtester.research.swing_series import SwingWindow
from ultimateBot.backtester.research.swing_window_search import search, search_plan
from ultimateBot.run_swing_ts2vec import parse_args
from ultimateBot.tests.test_swing_grid import point, selection, setup_database
from ultimateBot.tests.test_swing_window_search import START, HOUR, CONFIG, ENCODER, PROTOCOL


def pair(at, close):
    bid = Candle(at, close-.1, close, close-.1, close-.1)
    ask = Candle(at, close+.1, close+.1, close, close+.1)
    return bid, ask


def add_sources(database, root):
    sources = []
    for symbol in ('EURUSD', 'GBPUSD'):
        for day in range(14):
            at = START+timedelta(days=day)
            path = root/symbol/str(at.year)/f'{at.month:02d}'/(str(at.date())+'.tsv')
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for hour in range(24):
                instant = at+hour*HOUR
                # Two quotes per candle, with an extreme early quote: only its close counts.
                for minute, mid in ((1, 1.3), (59, 1.1+.0001*((day*24+hour) % 11))):
                    stamp = int((instant+timedelta(minutes=minute)).timestamp()*1000)
                    lines.append(f'QUOTE\t{symbol}\t{stamp}\t0\t0\t{mid-.00005:.8f}\t{mid+.00005:.8f}\t1\t1\n')
            path.write_text(''.join(lines), encoding='utf-8')
            stat = path.stat()
            sources.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    with sqlite3.connect(database) as db:
        parameters = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
        parameters['source_files'] = sources
        db.execute('UPDATE swing_runs SET parameters_json=?', (json.dumps(parameters),))
    db.close()
    return parameters


class CandleNormalizationTests(unittest.TestCase):
    def test_all_closed_candles_contribute_equally_without_futures_or_zero_holes(self):
        at = START+6*HOUR
        window = SwingWindow('duration', 6*3600)
        candles = [pair(START-HOUR, 999.)]+[pair(START+i*HOUR, price)
            for i, price in enumerate((100., 120., 140., 160., 180., 200., 999.))]
        mean = candle_close_mean(candles, as_of=at, window=window, interval_seconds=3600)
        self.assertEqual(mean, 150.)
        points = (point(at-5*HOUR, 100.), point(at-3*HOUR, 130., 'low'), point(at, 160.))
        vector = grid_vector(points, as_of=at, window=window, interval_seconds=3600, normalization_mean=mean)
        expected = np.zeros(6, np.float32)
        expected[[0, 2, 5]] = np.asarray([100., 130., 160.])/150.-1
        np.testing.assert_allclose(vector, expected, atol=1e-8)
        with self.assertRaises(ValueError):
            grid_vector(points, as_of=at, window=window, interval_seconds=3600, normalization_mean=0)
        with self.assertRaises(ValueError):
            candle_close_mean([candles[-1]], as_of=at, window=window, interval_seconds=3600)

    def test_source_cache_uses_closes_reuses_across_windows_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            setup_database(database)
            parameters = add_sources(database, root/'ticks')
            args = dict(symbols=['EURUSD'], start=START, before=START+timedelta(days=1),
                        interval_seconds=3600, cache_root=root/'prices')
            prices = CandleCloseIndex(parameters, **args)
            mean = prices.mean('EURUSD', as_of=START+6*HOUR, window=SwingWindow('duration', 21600))
            self.assertAlmostEqual(mean, np.mean([1.1+.0001*i for i in range(6)]), places=14)
            with patch('ultimateBot.backtester.research.swing_opportunities.read_day', side_effect=AssertionError('cache required')):
                reused = CandleCloseIndex(parameters, **args)
                self.assertEqual(reused.mean('EURUSD', as_of=START+6*HOUR,
                    window=SwingWindow('duration', 21600)), mean)
            path = Path(parameters['source_files'][0][0])
            with path.open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                CandleCloseIndex(parameters, **args)

    def test_training_frozen_test_and_live_prediction_use_the_same_new_normalization(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            setup_database(database)
            parameters = add_sources(database, root/'ticks')
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            result = train(database, run_id='raw', output_dir=root/'models', config=CONFIG,
                encoder_config=ENCODER, protocol=PROTOCOL, selection=selection(),
                representation='swing-grid', grid_normalization='candle-close-mean')
            self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), before)
            resolver = SwingGridResolver(result['directory'], threads=1)
            self.assertEqual(resolver.metadata['grid_schema'], CANDLE_CLOSE_GRID_SCHEMA)
            at = START+11*timedelta(days=1)
            width = resolver.window.value//3600
            candles = []
            points = []
            for hour in range(width):
                candle_at = at-(width-hour)*HOUR
                number = int((candle_at-START)/HOUR)
                candles.append(pair(candle_at, 1.1+.0001*(number % 11)))
                points.append(point(candle_at+HOUR, 1.11 if hour % 2 == 0 else 1.10,
                    'high' if hour % 2 == 0 else 'low'))
            mean = candle_close_mean(candles, as_of=at, window=resolver.window, interval_seconds=3600)
            values = grid_vector(points, as_of=at, window=resolver.window,
                interval_seconds=3600, normalization_mean=mean)
            expected = float(predict(resolver.model, values[None, :], resolver.metadata['target'])[0])
            decision = resolver.predict(points, as_of=at, symbol='EURUSD', candles=candles)
            self.assertEqual(decision['prediction'], expected)
            with self.assertRaisesRegex(ValueError, 'requires its closed candle'):
                resolver.predict(points, as_of=at, symbol='EURUSD')
            prices = CandleCloseIndex(parameters, symbols=['EURUSD'], start=at-timedelta(seconds=resolver.window.value),
                before=at, interval_seconds=3600, cache_root=root/'swing_candle_close_cache')
            self.assertAlmostEqual(prices.mean('EURUSD', as_of=at, window=resolver.window), mean, places=14)
            tested = frozen_test(database, directory=result['directory'], run_id='raw', start='2024-01-10',
                end='2024-01-13', output_dir=root/'test', min_samples=5, threads=1)
            self.assertFalse(tested['fitting_performed'])
            self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), before)
            # Changing only future candle prices cannot affect already prepared training inputs.
            periods = dict(train=[PROTOCOL.train_start, PROTOCOL.train_end])
            kwargs = dict(run_id='raw', windows=(resolver.window,), symbols=['EURUSD'], periods=periods,
                          target='pnl-r', grid_normalization='candle-close-mean')
            data, _ = prepare(database, cache_root=root/'cache-a', **kwargs)
            arrays = data.matrices[resolver.window.key].copy()
            data.close()
            future = next(Path(row[0]) for row in parameters['source_files'] if row[0].endswith('2024-01-12.tsv'))
            future.write_text(future.read_text().replace('1.300', '9.300'))
            data, _ = prepare(database, cache_root=root/'cache-b', **kwargs)
            np.testing.assert_array_equal(data.matrices[resolver.window.key], arrays)
            data.close()

    def test_cli_search_resume_identity_and_missing_price_sources(self):
        args = parse_args(['search', '--representation', 'swing-grid', '--grid-normalization', 'candle-close-mean'])
        self.assertEqual(args.grid_normalization, 'candle-close-mean')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['search', '--grid-normalization', 'candle-close-mean'])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            setup_database(database)
            options = dict(run_id='raw', output_dir=root/'models', config=CONFIG, encoder_config=ENCODER,
                protocol=PROTOCOL, selection=selection(), repeats=1, representation='swing-grid',
                cohort_policy='per-window', grid_normalization='candle-close-mean')
            with self.assertRaisesRegex(ValueError, 'requires the collection source'):
                train(database, **{k:v for k,v in options.items() if k not in ('repeats', 'cohort_policy')})
            add_sources(database, root/'ticks')
            first = search(database, **options)
            second = search(database, **options, resume=first['directory'])
            self.assertEqual(first['selected_directory'], second['selected_directory'])
            with self.assertRaisesRegex(ValueError, 'Resume requires'):
                search(database, **{**options, 'grid_normalization': 'swing-mean'}, resume=first['directory'])
            plain = search_plan(database, **{k:v for k,v in options.items() if k!='output_dir'})
            self.assertEqual(plain['grid_schema'], CANDLE_CLOSE_GRID_SCHEMA)

    def test_backtest_tracks_non_swing_candles_and_excludes_active_candle(self):
        from ultimateBot.backtester.data.tick import Tick
        from ultimateBot.backtester.strategy.swing_grid_strategy import SwingGridStrategy
        from ultimateBot.run_backtest import build_engine, parse_args as backtest_args
        from ultimateBot.tests.test_swing_ts2vec_robot import START as ROBOT_START, bars, metadata
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            meta = metadata(root, window='duration_14400')
            meta.update(model_type='swing-grid', grid_normalization='candle-close-mean')
            (root/'metadata.json').write_text(json.dumps(meta))
            captured = []
            def score(points, *, as_of, symbol, candles):
                captured.append((as_of, tuple(candles)))
                return dict(prediction=-.5, threshold=.25, target='pnl-r', accepted=False,
                            side=points[-1].side, resolver_id='resolver')
            with patch.object(SwingGridStrategy, '_load_resolver', return_value=SimpleNamespace(predict=score)):
                with patch.object(sys, 'argv', ['run_backtest', '--strategy', 'swing-grid', '--swing-grid-model', str(root)]):
                    engine = build_engine([], backtest_args())
            candle_list = bars(8)
            candle_list[1] = replace(candle_list[1], open_price=1.1000, close_price=1.1020)
            for bid in candle_list:
                for seconds, price in ((1, bid.open_price), (3599, bid.close_price)):
                    engine.on_tick(Tick(bid.start_time+timedelta(seconds=seconds), 'EURUSD', price, price+.0001, 1, 1))
            engine.on_tick(Tick(ROBOT_START+8*HOUR, 'EURUSD', 99., 99.01, 1, 1))
            self.assertTrue(captured)
            for at, candles in captured:
                self.assertEqual(len(candles), 4)
                self.assertEqual(candles[0][0].start_time, at-4*HOUR)
                self.assertEqual(candles[-1][0].start_time+HOUR, at)
                self.assertTrue(all(bid.close_price < 2 for bid, _ in candles))


if __name__ == '__main__':
    unittest.main()
