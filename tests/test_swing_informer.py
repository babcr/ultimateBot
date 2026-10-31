from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ultimateBot.backtester.research.swing_informer import (
    InformerConfig, InformerRegressor, SwingInformerResolver, fit, informer_class,
    atomic_torch, normalizer, predict_data, validation_threshold,
)
from ultimateBot.backtester.research.swing_informer_data import (
    CONFIRMED, SwingInformerData, extract, features,
)
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, OpportunityConfig
from ultimateBot.backtester.research.swing_series import (
    FEATURE_SCHEMA, SWING_SCHEMA, VOLUME_FEATURE_SCHEMA, SwingPoint, swing_features,
)

START = datetime(2022, 1, 1, tzinfo=timezone.utc)


def fixture_data():
    points = np.asarray([(1.1+.01*(i % 2 == 0)+i*.00001, 1 if i%2 == 0 else -1,
        (START+timedelta(hours=i-1)).timestamp(), (START+timedelta(hours=i)).timestamp(),
        100.+i%13, 0.) for i in range(192)], dtype=np.float64)
    contexts = np.asarray([np.arange(i-7, i+1) for i in range(7, len(points))], dtype=np.int32)
    last = contexts[:, -1]
    labels = np.where(last%2 == 0, 2., -1.).astype(np.float32)
    phases = (last >= 96).astype(np.uint8)
    metadata = dict(version=1, run_id='fixture', symbols=['EURUSD'], feature_schema=VOLUME_FEATURE_SCHEMA,
        opportunity_config=asdict(OpportunityConfig(interval_seconds=3600)), snapshot_sha256='fixture',
        periods={'train': ['2022-01-01', '2022-01-04'], 'validation': ['2022-01-05', '2022-01-08']},
        phase_names=['train', 'validation'], seq_len=8)
    return SwingInformerData(points, contexts, labels, phases, metadata)


def small_config(**overrides):
    return replace(InformerConfig(seq_len=8, label_len=4, d_model=8, n_heads=2,
        e_layers=1, d_layers=1, d_ff=16, batch_size=32, epochs=2, threads=1,
        min_train_samples=20, min_validation_samples=10, min_accepted_validation=5,
        checkpoint_batches=2), **overrides)


def swing_points(raw):
    return tuple(SwingPoint('high' if row[1] == 1 else 'low', row[0],
        datetime.fromtimestamp(row[2], timezone.utc), datetime.fromtimestamp(row[2]-3600, timezone.utc),
        datetime.fromtimestamp(row[3], timezone.utc), row[4]) for row in raw)


class InformerTests(unittest.TestCase):
    def test_features_match_causal_swing_encoding_and_have_no_gain_channel(self):
        data = fixture_data()
        raw = data.points[data.contexts[:2]]
        x, marks = features(raw, 3600, VOLUME_FEATURE_SCHEMA, 1)
        points = swing_points(raw[0])
        expected = swing_features(points, timedelta(hours=1), points[-1].confirmed_at,
                                  feature_schema=VOLUME_FEATURE_SCHEMA)
        np.testing.assert_allclose(x[0, :, :6], expected[:, :6], rtol=1e-6)
        np.testing.assert_allclose(x[0, :, 6], np.log1p(expected[:, 6]), rtol=1e-6)
        np.testing.assert_array_equal(x[:, :, -1], 1.)
        self.assertEqual(marks.shape, (2, 8, 5))
        original = data.batch([0])[0]
        data.labels[:] = 999999.
        np.testing.assert_array_equal(original, data.batch([0])[0])
        with self.assertRaises(ValueError):
            features(raw[:, ::-1], 3600, VOLUME_FEATURE_SCHEMA, 1)

    def test_normalizer_uses_training_only(self):
        data = fixture_data()
        train = data.partition('train')
        before = normalizer(data, train, 32)
        data.labels[data.partition('validation')] = 1e8
        data.points[96:, 0] *= 10
        data.points[96:, 4] *= 10
        after = normalizer(data, train, 32)
        self.assertEqual(before, after)
        self.assertEqual(before['fit_partition'], 'train_only')

    def test_upstream_is_isolated_and_prob_sparse_inference_is_repeatable(self):
        before = {name: sys.modules.get(name) for name in ('utils', 'models', 'utils.masking', 'models.model')}
        vendor = informer_class()
        self.assertTrue(vendor.__module__.startswith('_ultimatebot_informer_vendor.'))
        self.assertEqual(before, {name: sys.modules.get(name) for name in before})
        config = small_config(seq_len=64, label_len=32, factor=1)
        regressor = InformerRegressor(config, dict(mean=[0.]*7, scale=[1.]*7,
            target_mean=0., target_scale=1.), 7)
        x = np.random.default_rng(2).normal(size=(2, 64, 7)).astype(np.float32)
        marks = np.zeros((2, 64, 5), np.float32)
        state = torch.get_rng_state().clone()
        first = regressor.predict(x, marks)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        np.testing.assert_array_equal(first, regressor.predict(x, marks))
        np.testing.assert_allclose(first[:1], regressor.predict(x[:1], marks[:1]), atol=1e-6)

    def test_resume_reproduces_uninterrupted_training_and_inference_guards_cutoff(self):
        data, config = fixture_data(), small_config()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            uninterrupted = fit(data, root/'full', config)
            partial = fit(data, root/'resumed', config, stop_after_batches=2)
            self.assertTrue(partial['interrupted'])
            resumed = fit(data, root/'resumed', config, resume=True)
            a = torch.load(root/'full/latest.pt', weights_only=True)
            b = torch.load(root/'resumed/latest.pt', weights_only=True)
            for key, weights in a['model'].items():
                torch.testing.assert_close(weights, b['model'][key], rtol=0, atol=0)
            self.assertEqual(uninterrupted['history'], resumed['history'])
            self.assertEqual(uninterrupted['threshold'], resumed['threshold'])
            self.assertFalse(resumed['test_used_for_selection'])
            resolver = SwingInformerResolver(root/'resumed', threads=1)
            points = swing_points(data.points[data.contexts[-1]])
            with self.assertRaisesRegex(ValueError, 'postdates'):
                resolver.predict(points, as_of=points[-1].confirmed_at, symbol='EURUSD')
            shift = timedelta(days=100)
            future = tuple(replace(p, pivot_at=p.pivot_at+shift, pair_start=p.pair_start+shift,
                                   confirmed_at=p.confirmed_at+shift) for p in points)
            decision = resolver.predict(future, as_of=future[-1].confirmed_at, symbol='EURUSD')
            self.assertTrue(np.isfinite(decision['prediction']))
            self.assertEqual(decision['target'], 'pnl-r')
            with self.assertRaises(ValueError):
                resolver.predict(future, as_of=future[-1].confirmed_at, symbol='USDJPY')
            with self.assertRaisesRegex(ValueError, 'same model'):
                fit(data, root/'resumed', replace(config, learning_rate=.001), resume=True)
            (root/'resumed/best.pt').write_bytes(b'bad checkpoint')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                SwingInformerResolver(root/'resumed', threads=1)

    def test_threshold_floor_keeps_unprofitable_predictions_rejected(self):
        selected, candidates = validation_threshold(np.array([-1., 1., -1.]), np.array([-.5, -.2, -.1]), small_config())
        self.assertEqual(selected['threshold'], 0.)
        self.assertEqual(selected['opportunities'], 0)
        self.assertTrue(all(c['threshold'] >= 0 for c in candidates))

    def test_interrupted_checkpoint_write_preserves_last_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'latest.pt'
            atomic_torch(path, {'updates': 10})
            with patch('torch.save', side_effect=OSError('interrupted write')):
                with self.assertRaises(OSError):
                    atomic_torch(path, {'updates': 20})
            self.assertEqual(torch.load(path, weights_only=True)['updates'], 10)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_scalar_extraction_warmup_ambiguity_and_censoring_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            con = sqlite3.connect(database)
            parameters = dict(symbols=['EURUSD'], config=asdict(OpportunityConfig(interval_seconds=1200)),
                feature_schema=VOLUME_FEATURE_SCHEMA, swing_schema=SWING_SCHEMA,
                opportunity_schema=OPPORTUNITY_SCHEMA, risk_management_filters=False)
            con.execute('CREATE TABLE swing_runs (run_id TEXT,status TEXT,started_at TEXT,parameters_json TEXT)')
            con.execute('INSERT INTO swing_runs VALUES (?,?,?,?)', ('run', 'completed', '2022', json.dumps(parameters)))
            con.execute('''CREATE TABLE swing_events (event_id TEXT,run_id TEXT,symbol TEXT,kind TEXT,
                pivot_price REAL,pivot_at TEXT,pair_start TEXT,confirmed_at TEXT,status TEXT,
                closed_at TEXT,pnl_r REAL,mean_quote_volume REAL)''')
            for i in range(320):
                at = START+timedelta(minutes=i*20)
                closed = at+timedelta(minutes=5) if i != 65 else START+timedelta(days=2)
                row = (str(i), 'run', 'EURUSD', 'high' if i%2 == 0 else 'low', 1.11 if i%2 == 0 else 1.1,
                       (at-timedelta(minutes=20)).isoformat(), (at-timedelta(minutes=40)).isoformat(),
                       at.isoformat(), 'closed', closed.isoformat(), 1., 100.)
                con.execute('INSERT INTO swing_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', row)
                if i == 40:
                    con.execute('INSERT INTO swing_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                ('40b', *row[1:3], 'low', *row[4:]))
            con.commit()
            con.close()
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            data = extract(database, run_id='run', periods={'train': ['2022-01-01', '2022-01-01'],
                'validation': ['2022-01-02', '2022-01-02']}, seq_len=8)
            self.assertEqual(data.metadata['counts']['ambiguous'], 2)
            self.assertEqual(data.metadata['counts']['censored_outcome'], 1)
            for phase in ('train', 'validation'):
                self.assertGreater(len(data.partition(phase)), 10)
            later = extract(database, run_id='run', periods={'test': ['2022-01-04', '2022-01-04']}, seq_len=8)
            self.assertGreaterEqual(later.points[:, CONFIRMED].min(), (START+timedelta(days=3, minutes=-8*20)).timestamp())
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())
            data.save(root/'data.npz')
            saved = SwingInformerData.load(root/'data.npz')
            np.testing.assert_array_equal(data.contexts, saved.contexts)

    def test_backtest_cli_builds_informer_and_rejects_a_wrong_model_type(self):
        from ultimateBot.run_backtest import build_engine, parse_args
        from ultimateBot.backtester.strategy.swing_informer_strategy import SwingInformerStrategy
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            fit(fixture_data(), tmp, small_config(epochs=1))
            command = ['run_backtest', '--strategy', 'swing-informer', '--swing-informer-model', tmp,
                       '--swing-informer-threshold', '.2', '--swing-informer-min-threshold', '.1',
                       '--swing-informer-stop-buffer-pips', '5', '--informer-threads', '1']
            with patch.object(sys, 'argv', command):
                args = parse_args()
            engine = build_engine([], args)
            self.assertIsInstance(engine.strategy, SwingInformerStrategy)
            self.assertEqual(engine.strategy.threshold, .2)
            self.assertEqual(engine.strategy.window.value, 4)
            self.assertEqual(engine.strategy.opportunity_config.stop_buffer_pips, 5.)
            self.assertEqual(engine.strategy.strategy_name, 'swing-informer')
            with patch.object(sys, 'argv', command+['--swing-informer-take-profit-r', '2']):
                fixed = build_engine([], parse_args())
            self.assertEqual(fixed.strategy.exit_mode, 'fixed-take-profit')
            self.assertEqual(fixed.strategy.take_profit_r, 2.)
            with patch.object(sys, 'argv', [*command[:2], 'swing-ts2vec', *command[3:]]):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_cli_reuses_snapshot_and_test_never_changes_frozen_model_or_threshold(self):
        from ultimateBot.run_swing_informer import main
        data = fixture_data()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data.save(root/'dataset.npz')
            options = ['train', '--dataset', str(root/'dataset.npz'), '--output-dir', str(root/'model'),
                '--epochs', '1', '--d-model', '8', '--n-heads', '2', '--e-layers', '1', '--d-ff', '16',
                '--min-train-samples', '20', '--min-validation-samples', '10',
                '--min-accepted-validation', '5', '--threads', '1']
            with patch('ultimateBot.run_swing_informer.extract', side_effect=AssertionError('Cache must avoid DB reads')):
                metadata = main(options)
            original = (root/'model/metadata.json').read_bytes()
            test = fixture_data()
            test.points[:, 2:4] += 730*86400
            test.phases[:] = 0
            test.metadata['phase_names'] = ['test']
            test.metadata['periods'] = {'test': ['2024-01-01', '2024-12-31']}
            with patch('ultimateBot.run_swing_informer.extract', return_value=test):
                result = main(['test', '--model', str(root/'model'), '--start-date', '2024-01-01',
                    '--end-date', '2024-12-31', '--output-dir', str(root/'test'), '--threads', '1'])
            self.assertFalse(result['threshold_changed'])
            self.assertEqual(result['threshold'], metadata['threshold'])
            self.assertEqual(original, (root/'model/metadata.json').read_bytes())
            self.assertTrue((root/'test/test.json').exists())
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                main(['train', '--resume', str(root/'model')])


if __name__ == '__main__':
    unittest.main()
