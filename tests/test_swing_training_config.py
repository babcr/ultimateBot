from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import timedelta
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from ultimateBot.backtester.research.swing_encoder import SwingEncoder, SwingEncoderConfig, encoder_training_plan
from ultimateBot.backtester.research.swing_resolver import ResolverConfig, fit_xgb, learning_diagnostics
from ultimateBot.backtester.research.swing_window_search import search, search_plan
from ultimateBot.run_swing_ts2vec import parse_args
from ultimateBot.tests.test_swing_ts2vec import START
from ultimateBot.tests.test_swing_window_search import database, PROTOCOL


class EncoderBudgetTests(unittest.TestCase):
    def test_default_budget_scales_with_data_and_reports_upstream_remainder(self):
        config = SwingEncoderConfig()
        small = encoder_training_plan(config, [785]*160)
        large = encoder_training_plan(config, [785]*206775)
        self.assertEqual(small['optimizer_updates'], 10)
        self.assertEqual(large['optimizer_updates'], 12923)
        self.assertEqual(large['sequence_presentations'], 206768)
        self.assertEqual(large['dropped_sequences_per_epoch'], 7)
        self.assertEqual(large['depth'], 7)
        self.assertGreaterEqual(large['receptive_field_swings'], 785)
        self.assertEqual(large['truncated_sequences'], 0)
        self.assertEqual(large['warnings'], [])
        doubled = encoder_training_plan(replace(config, train_epochs=2), [785]*160)
        self.assertEqual(doubled['optimizer_updates'], 2*small['optimizer_updates'])

    def test_explicit_short_iteration_budget_is_visible_and_cap_never_silently_crops(self):
        plan = encoder_training_plan(SwingEncoderConfig(train_iters=200), [785]*206775)
        self.assertAlmostEqual(plan['equivalent_data_passes'], 3200/206775)
        self.assertIn('iteration_budget_below_one_data_pass', plan['warnings'])
        with self.assertRaisesRegex(ValueError, 'never silently truncated'):
            encoder_training_plan(SwingEncoderConfig(max_train_length=256), [300, 785])
        plan = encoder_training_plan(SwingEncoderConfig(depth=2, max_train_samples=16), [785]*32)
        self.assertIn('explicit_depth_receptive_field_shorter_than_training_window', plan['warnings'])
        self.assertIn('explicit_training_sample_limit', plan['warnings'])

    def test_epoch_fit_uses_reported_updates_and_persists_actual_loss(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            arrays = [np.random.default_rng(i).normal(size=(4+i % 3, 6)).astype(np.float32) for i in range(9)]
            config = SwingEncoderConfig(hidden_dims=4, output_dims=4, batch_size=4, train_epochs=2, threads=1)
            service = SwingEncoder(config).fit(arrays, observed_at=[START]*9, cutoff=START+timedelta(hours=1),
                                               checkpoint=Path(tmp)/'model.pt', provenance={})
            self.assertEqual(service.metadata['optimizer_updates'], 4)
            self.assertEqual(service.metadata['completed_epochs'], 2)
            self.assertEqual(len(service.metadata['epoch_losses']), 2)
            self.assertTrue(np.isfinite(service.metadata['last_loss']))
            self.assertEqual(service.metadata['training_plan']['optimizer_updates'], service.model.n_iters)

    def test_long_full_windows_and_legacy_checkpoints_keep_inference_reproducible(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            arrays = [np.random.default_rng(i).normal(size=(300, 6)).astype(np.float32) for i in range(2)]
            arrays[0][:44, 0] += 100  # The old tail crop would discard these values.
            config = SwingEncoderConfig(hidden_dims=2, output_dims=2, train_iters=1, threads=1)
            checkpoint = Path(tmp)/'full.pt'
            service = SwingEncoder(config).fit(arrays, observed_at=[START]*2, cutoff=START+timedelta(hours=1),
                                               checkpoint=checkpoint, provenance={})
            np.testing.assert_allclose(service.mean, np.concatenate(arrays).mean(axis=0, dtype=np.float64), rtol=1e-6)
            self.assertEqual(service.metadata['training_plan']['padded_training_length'], 300)
            self.assertIsNotNone(service.metadata['last_loss'])
            expected = service.encode(arrays, observed_at=[START+timedelta(hours=2)]*2)
            loaded = SwingEncoder.load(checkpoint, threads=1)
            np.testing.assert_array_equal(expected, loaded.encode(arrays, observed_at=[START+timedelta(hours=2)]*2))
            old = torch.load(checkpoint, map_location='cpu', weights_only=True)
            old['metadata']['version'] = 1
            old['metadata']['config'].pop('train_epochs')
            old['metadata']['config'].pop('learning_rate')
            old['metadata']['config']['max_train_length'] = 256
            old['metadata']['config']['train_iters'] = 200
            torch.save(old, Path(tmp)/'legacy.pt')
            legacy = SwingEncoder.load(Path(tmp)/'legacy.pt', threads=1)
            np.testing.assert_array_equal(expected, legacy.encode(arrays, observed_at=[START+timedelta(hours=2)]*2))
            with self.assertRaisesRegex(ValueError, 'never silently truncated'):
                SwingEncoder(replace(config, max_train_length=256)).fit(arrays, observed_at=[START]*2,
                    cutoff=START+timedelta(hours=1), checkpoint=Path(tmp)/'forbidden.pt', provenance={})
            self.assertFalse((Path(tmp)/'forbidden.pt').exists())

    def test_cli_exposes_epochs_and_rejects_ambiguous_budget(self):
        args = parse_args(['train'])
        self.assertEqual((args.ts2vec_train_epochs, args.ts2vec_train_iters, args.ts2vec_depth, args.ts2vec_max_train_length),
                         (1, 0, 0, 0))
        for flags in (['--ts2vec-train-epochs', '0'], ['--ts2vec-train-iters', '-1'],
                      ['--ts2vec-max-train-length', '1'],
                      ['--ts2vec-batch-size', '1'], ['--ts2vec-max-train-samples', '1'],
                      ['--ts2vec-train-epochs', '2', '--ts2vec-train-iters', '200']):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['train', *flags])

    def test_preflight_is_read_only_training_only_and_resume_identity_ignores_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.sqlite3'
            db = database(path)
            # A future input must not drive the depth or the training workload.
            db.execute("UPDATE swing_contexts SET n_swings=999999 WHERE context_end>=?", (PROTOCOL.bounds[1].isoformat(),))
            db.commit()
            db.close()
            before = path.read_bytes()
            from ultimateBot.backtester.research.swing_resolver import TrainingSelection
            selection = TrainingSelection(pairs=(2,))
            plan = search(path, output_dir=Path(tmp)/'models', protocol=PROTOCOL, selection=selection, dry_run=True)
            self.assertEqual(plan['encoder_preflight']['pairs_2']['sequence_length_max'], 4)
            self.assertEqual(before, path.read_bytes())
            self.assertFalse((Path(tmp)/'models').exists())
            # Changing the workload observation in the same collection must not
            # invalidate the requested configuration's resume fingerprint.
            db = sqlite3.connect(path)
            db.execute('''DELETE FROM swing_contexts WHERE context_id IN
                (SELECT context_id FROM swing_contexts WHERE context_end<? LIMIT 10)''', (PROTOCOL.bounds[1].isoformat(),))
            db.commit()
            db.close()
            changed = search_plan(path, protocol=PROTOCOL, selection=selection)
            self.assertEqual(plan['plan_sha256'], changed['plan_sha256'])
            self.assertNotEqual(plan['encoder_preflight'], changed['encoder_preflight'])


class XGBoostDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_find_predictive_gain_and_record_both_learning_curves(self):
        x = np.linspace(-1, 1, 400, dtype=np.float32).reshape(-1, 1)
        labels = (x[:, 0] > 0).astype(int)
        arrays = {'train': (x[::2], labels[::2]), 'validation': (x[1::2], labels[1::2])}
        config = ResolverConfig(target='trailing-step-1', n_estimators=40, early_stopping_rounds=5, threads=1)
        model = fit_xgb(arrays, config)
        result = learning_diagnostics(model, arrays, config.target)
        self.assertTrue(result['validation_predictive_gain'])
        self.assertGreater(result['partitions']['validation']['auc'], .99)
        self.assertGreater(result['partitions']['validation']['improvement_over_constant'], 0)
        self.assertEqual(set(result['learning_curves']), {'validation_0', 'validation_1'})
        self.assertIn('logloss', result['learning_curves']['validation_1'])
        self.assertIn('auc', result['learning_curves']['validation_1'])
        self.assertEqual(result['trees_used'], model.best_iteration+1)

    def test_constant_model_is_not_qualified_even_if_training_finished(self):
        arrays = {phase: (np.zeros((40, 2), dtype=np.float32), np.tile([0, 1], 20))
                  for phase in ('train', 'validation')}
        config = ResolverConfig(target='trailing-step-1', n_estimators=5, early_stopping_rounds=2, threads=1)
        model = fit_xgb(arrays, config)
        result = learning_diagnostics(model, arrays, config.target)
        self.assertFalse(result['validation_predictive_gain'])
        self.assertEqual(result['status'], 'no_validation_predictive_gain')
        self.assertEqual(result['partitions']['validation']['unique_predictions'], 1)
        self.assertEqual(result['partitions']['validation']['auc'], .5)
        self.assertEqual(result['constant_from_training'], .5)


if __name__ == '__main__':
    unittest.main()
