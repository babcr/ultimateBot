"""Per-tick quote volume on both reversal candles, from collection to execution."""
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import timedelta
import csv
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import CandleBuilder
from ultimateBot.backtester.research.swing_contexts import ensure_windows
from ultimateBot.backtester.research.swing_encoder import SwingEncoder, SwingEncoderConfig, encoder_training_plan
from ultimateBot.backtester.research.swing_opportunities import SCHEMA, OpportunityConfig, collect, connect, day_candles, instant, read_day
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol, ResolverConfig, SwingResolver, TrainingSelection, frozen_test, train
from ultimateBot.backtester.research.swing_series import (
    FEATURE_NAMES, FEATURE_SCHEMA, VOLUME_FEATURE_SCHEMA, SwingPoint, SwingSeries, SwingWindow, feature_names, model_feature_schema, swing_features,
)
from ultimateBot.backtester.research.swing_window_search import search
from ultimateBot.run_backtest import build_engine
from ultimateBot.run_swing_ts2vec import parse_args
from ultimateBot.tests.test_swing_ts2vec import START, HOUR, PROTOCOL, add_event, create_training_database
from ultimateBot.tests.test_swing_ts2vec_robot import cli


def write_ticks(root, day, *, hours=12):
    folder = root/'history/EURUSD'/day.strftime('%Y/%m')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(day.strftime('%Y-%m-%d')+'.tsv')
    lines = []
    for h in range(hours):
        prices = (101., 103., 99., 102.) if h % 2 == 0 else (102., 103., 99., 101.)
        for i, (offset, price) in enumerate(zip((1, 1000, 2000, 3599), prices)):
            ms = round((day+h*HOUR+timedelta(seconds=offset)).timestamp()*1000)
            bid_volume = (day.day % 3)*100+h*10+i
            ask_volume = (day.day % 5)*200+h*20+2*i
            lines.append(f'QUOTE\tEURUSD\t{ms}\t1\t{i}\t{price}\t{price+.1}\t{bid_volume}\t{ask_volume}')
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return path


class SwingVolumeTests(unittest.TestCase):
    def test_tick_weighted_mean_uses_bid_for_high_and_ask_for_low_without_future_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'2024-01-01.tsv'
            # Different candle tick counts make an unweighted mean of candle means incorrect.
            payload = (((100., 2., 20.), (102., 4., 40.)),
                       ((102., 10., 100.), (101., 20., 200.), (100., 30., 300.)),
                       ((100., 1000., 10000.), (100., 2000., 20000.), (101., 3000., 30000.), (102., 4000., 40000.)))
            lines, ticks = [], []
            for h, values in enumerate(payload):
                for i, (bid, bv, av) in enumerate(values):
                    at = START+h*HOUR+timedelta(seconds=1+i*100)
                    ticks.append(Tick(at, 'EURUSD', bid, bid+.1, bv, av))
                    lines.append(f'QUOTE\tEURUSD\t{round(at.timestamp()*1000)}\t1\t{i}\t{bid}\t{bid+.1}\t{bv}\t{av}')
            path.write_text('\n'.join(lines)+'\n')
            frame = read_day(path, 'EURUSD', include_volume=True)
            candles = day_candles(frame, 3600)
            live_bid, live_ask = (CandleBuilder(HOUR, 10, source) for source in ('bid', 'ask'))
            for tick in ticks:
                live_bid.update(tick)
                live_ask.update(tick)
            live_bid.finalize_active_candles()
            live_ask.finalize_active_candles()
            self.assertEqual([c.tick_count for c, _ in candles], [2, 3, 4])
            offline = SwingSeries(HOUR, (SwingWindow('pairs', 1),), include_volume=True)
            online = SwingSeries(HOUR, (SwingWindow('pairs', 1),), include_volume=True)
            for i, ((bid, ask), lb, la) in enumerate(zip(candles, live_bid.get_candles('EURUSD'), live_ask.get_candles('EURUSD'))):
                offline.append(bid, ask)
                online.append(lb, la)
                self.assertEqual(offline.last_candidates, online.last_candidates)
                if i == 1:
                    high = offline.last_candidates[0]
                    self.assertEqual(high.kind, 'high')
                    self.assertEqual(high.mean_quote_volume, (2+4+10+20+30)/5)
                    self.assertNotEqual(high.mean_quote_volume, ((2+4)/2+(10+20+30)/3)/2)
            low = offline.last_candidates[0]
            self.assertEqual(low.kind, 'low')
            self.assertEqual(low.mean_quote_volume, (100+200+300+10000+20000+30000+40000)/7)
            points = offline.context(SwingWindow('pairs', 1), START+3*HOUR)
            new = swing_features(points, HOUR, START+3*HOUR, feature_schema=VOLUME_FEATURE_SCHEMA)
            old = swing_features(points, HOUR, START+3*HOUR)
            self.assertEqual(new.shape, (2, 7))
            np.testing.assert_array_equal(old, new[:, :6])
            np.testing.assert_allclose(new[:, 6], [high.mean_quote_volume, low.mean_quote_volume])

    def test_zero_volumes_are_valid_missing_and_invalid_volumes_are_not_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'2024-01-01.tsv'
            for value in ('0', '-1', 'nan', ''):
                path.write_text(f'QUOTE\tEURUSD\t{round((START+timedelta(seconds=1)).timestamp()*1000)}\t1\t1\t1.1\t1.2\t{value}\t0\n')
                if value == '0':
                    self.assertEqual(read_day(path, 'EURUSD', include_volume=True).bv.iloc[0], 0.)
                else:
                    with self.assertRaises(ValueError):
                        read_day(path, 'EURUSD', include_volume=True)
                self.assertEqual(list(read_day(path, 'EURUSD').columns), ['timestamp', 'bid', 'ask'])
        points = tuple(SwingPoint('high' if i == 0 else 'low', 1.2 if i == 0 else 1.1,
                                 START+i*HOUR, START+i*HOUR, START+(i+2)*HOUR) for i in range(2))
        self.assertNotIn('mean_quote_volume', points[0].to_dict())
        with self.assertRaisesRegex(ValueError, 'mean quote volume'):
            swing_features(points, HOUR, START+3*HOUR, feature_schema=VOLUME_FEATURE_SCHEMA)
        points = tuple(replace(p, mean_quote_volume=0.) for p in points)
        self.assertTrue((swing_features(points, HOUR, START+3*HOUR, feature_schema=VOLUME_FEATURE_SCHEMA)[:, 6] == 0).all())

    def test_cli_and_encoder_dimensions_are_explicit_and_preserve_legacy_defaults(self):
        self.assertTrue(parse_args(['collect', '--swing-volume']).swing_volume)
        self.assertTrue(parse_args(['run', '--swing-volume', '--dry-run']).swing_volume)
        self.assertIsNone(parse_args(['collect']).swing_volume)
        self.assertFalse(parse_args(['collect', '--no-swing-volume']).swing_volume)
        self.assertFalse(parse_args(['search', '--no-swing-volume']).swing_volume)
        self.assertTrue(parse_args(['train', '--swing-volume']).swing_volume)
        self.assertEqual(len(FEATURE_NAMES), 6)
        self.assertEqual(len(feature_names(VOLUME_FEATURE_SCHEMA)), 7)
        config = SwingEncoderConfig()
        old = encoder_training_plan(config, [4, 4])
        new = encoder_training_plan(config, [4, 4], feature_schema=VOLUME_FEATURE_SCHEMA)
        self.assertEqual(new['training_buffer_bytes']*6, old['training_buffer_bytes']*7)
        with self.assertRaisesRegex(ValueError, 'swings,7'):
            SwingEncoder(feature_schema=VOLUME_FEATURE_SCHEMA)._check(np.ones((4, 6)))
        self.assertEqual(model_feature_schema(VOLUME_FEATURE_SCHEMA, False), FEATURE_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'collect with --swing-volume'):
            model_feature_schema(FEATURE_SCHEMA, True)

    def test_legacy_database_migration_preserves_events_without_inventing_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'legacy.sqlite3'
            db = sqlite3.connect(path)
            db.executescript(SCHEMA.replace(', mean_quote_volume REAL', ''))
            db.execute("INSERT INTO swing_runs(run_id,started_at,status,parameters_json) VALUES('run',?,'completed','{}')", (START.isoformat(),))
            add_event(db, SwingPoint('low', 100., START, START, START+2*HOUR))
            before = db.execute('SELECT * FROM swing_events').fetchone()
            db.commit()
            db.close()
            db = connect(path)
            row = tuple(db.execute('SELECT * FROM swing_events').fetchone())
            self.assertEqual(row[:-1], before)
            self.assertIsNone(row[-1])
            db.close()

    def test_legacy_training_fingerprint_survives_nullable_column_migration(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            path = root/'legacy.sqlite3'
            create_training_database(path)
            db = sqlite3.connect(path)
            db.execute('ALTER TABLE swing_events DROP COLUMN mean_quote_volume')
            db.commit()
            db.close()
            def legacy_connection(database):
                connection = sqlite3.connect(database)
                connection.row_factory = sqlite3.Row
                return connection
            config = ResolverConfig(min_encoder_samples=8, min_train_samples=15, min_validation_samples=8,
                                    min_accepted_validation=3, n_estimators=6, early_stopping_rounds=2, threads=1)
            encoder = SwingEncoderConfig(output_dims=4, hidden_dims=4, depth=1, train_iters=2, threads=1)
            args = dict(output_dir=root/'models', config=config, encoder_config=encoder,
                        protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)))
            with patch('ultimateBot.backtester.research.swing_resolver.connect', side_effect=legacy_connection):
                before = train(path, **args)
            after = train(path, **args)
            self.assertEqual(before['dataset_sha256'], after['dataset_sha256'])
            self.assertEqual(before['split'], after['split'])

    def test_collection_cache_training_reload_frozen_test_and_robot_agree(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)
            paths = [write_ticks(root, START+timedelta(days=d)) for d in range(12)]
            path = root/'research.sqlite3'
            args = dict(history=root/'history', symbols=['EURUSD'], start=START.date(),
                        end=(START+timedelta(days=11)).date(), config=OpportunityConfig(stop_buffer_pips=2), include_volume=True)
            collected = collect(path, windows=(SwingWindow('pairs', 1),), store_contexts=True, **args)
            reference = collect(root/'reference.sqlite3', windows=(SwingWindow('pairs', 1), SwingWindow('pairs', 2)), store_contexts=True, **args)
            db = connect(path)
            run_id = collected['run_id']
            parameters = json.loads(db.execute('SELECT parameters_json FROM swing_runs').fetchone()[0])
            self.assertEqual(parameters['feature_schema'], VOLUME_FEATURE_SCHEMA)
            self.assertEqual(parameters['feature_names'][-1], 'mean_quote_volume')
            self.assertEqual(db.execute('SELECT count(*) FROM swing_events WHERE mean_quote_volume IS NULL').fetchone()[0], 0)
            with patch('ultimateBot.backtester.research.swing_opportunities.read_day', side_effect=AssertionError('must replay stored swings')):
                ensure_windows(db, run_id, (SwingWindow('pairs', 2),), before=START+timedelta(days=12))
            ref = connect(root/'reference.sqlite3')
            query = 'SELECT window_key,context_end,features,swings_json FROM swing_contexts ORDER BY window_key,context_end'
            self.assertEqual([tuple(r) for r in db.execute(query)], [tuple(r) for r in ref.execute(query)])
            original_events = [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
            db.close()
            ref.close()
            protocol = ResearchProtocol('2024-01-01', '2024-01-04', '2024-01-05', '2024-01-07')
            config = ResolverConfig(min_encoder_samples=8, min_train_samples=15, min_validation_samples=8,
                                    min_accepted_validation=3, n_estimators=6, early_stopping_rounds=2, threads=1)
            encoder = SwingEncoderConfig(output_dims=4, hidden_dims=4, depth=1, train_iters=2, threads=1)
            result = search(path, run_id=run_id, output_dir=root/'models', config=config, encoder_config=encoder,
                            protocol=protocol, selection=TrainingSelection(pairs=(2,)), repeats=1)
            resolver = SwingResolver(result['selected_directory'], threads=1)
            self.assertEqual(resolver.encoder.feature_schema, VOLUME_FEATURE_SCHEMA)
            self.assertEqual(len(resolver.encoder.mean), 7)
            # Only TS2Vec inputs grow: XGBoost still receives embedding + three tabular fields.
            self.assertEqual(len(resolver.metadata['feature_names']), 7)
            tested = frozen_test(path, directory=result['directory'], run_id=run_id, start='2024-01-08',
                                 end='2024-01-10', min_samples=5, threads=1)
            self.assertFalse(tested['fit_called'])
            with (Path(tested['directory'])/'predictions.csv').open(newline='') as f:
                expected = {r['confirmed_at']: float(r['prediction']) for r in csv.DictReader(f)}
            engine = build_engine([], cli(Path(result['selected_directory']), '--max-spread-ratio', '10000', '--swing-ts2vec-decision-log', 'all'))
            for tick_path in paths[7:10]:
                for row in read_day(tick_path, 'EURUSD', include_volume=True).itertuples(index=False):
                    engine.on_tick(Tick(instant(row.timestamp), 'EURUSD', row.bid, row.ask, row.bv, row.av))
            decisions = [json.loads(line.split('] ', 1)[1]) for line in output.getvalue().splitlines()
                         if line.startswith('[swing-ts2vec-decision]')]
            self.assertGreater(len(decisions), 5)
            for decision in decisions:
                self.assertIn(decision['confirmed_at'], expected)
                self.assertAlmostEqual(decision['prediction'], expected[decision['confirmed_at']], places=6)
            baseline = search(path, run_id=run_id, output_dir=root/'models', config=config, encoder_config=encoder,
                              protocol=protocol, selection=TrainingSelection(pairs=(2,), include_volume=False), repeats=1)
            baseline_resolver = SwingResolver(baseline['selected_directory'], threads=1)
            self.assertEqual(len(baseline_resolver.encoder.mean), 6)
            volume_comparison = json.loads(Path(result['trials'][0]['comparison']).read_text())
            baseline_comparison = json.loads(Path(baseline['trials'][0]['comparison']).read_text())
            self.assertEqual(volume_comparison['dataset_sha256'], baseline_comparison['dataset_sha256'])
            self.assertEqual(volume_comparison['split'], baseline_comparison['split'])
            self.assertNotEqual(result['plan']['plan_sha256'], baseline['plan']['plan_sha256'])
            baseline_test = frozen_test(path, directory=baseline['directory'], run_id=run_id, start='2024-01-08',
                                        end='2024-01-10', min_samples=5, threads=1)
            self.assertEqual(tested['labeled_contexts'], baseline_test['labeled_contexts'])
            db = connect(path)
            self.assertEqual(original_events, [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')])
            db.close()
            with self.assertRaisesRegex(ValueError, 'volume feature setting'):
                collect(path, run_id=run_id, history=root/'history', symbols=['GBPUSD'], include_volume=False)


if __name__ == '__main__':
    unittest.main()
