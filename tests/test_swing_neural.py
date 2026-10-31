from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
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
import torch

from ultimateBot.backtester.research.swing_grid import grid_schema, grid_vector, prepare
from ultimateBot.backtester.research.swing_neural import (
    NeuralConfig, NeuralData, SwingNeuralResolver, architecture_spec, build_model, choose_threshold,
    fit, frozen_test, normalizer, predict_array, prepare_data,
)
from ultimateBot.backtester.research.swing_series import SwingWindow
from ultimateBot.tests.test_swing_candle_normalization import add_sources
from ultimateBot.tests.test_swing_grid import setup_database, point
from ultimateBot.tests.test_swing_window_search import START, HOUR, PROTOCOL
from ultimateBot.run_swing_nn import main, search_plan


def synthetic(width=18, train=256, validation=128):
    rng = np.random.default_rng(17)
    x = rng.normal(0, .001, (train+validation, width)).astype(np.float32)
    x[rng.random(x.shape) < .35] = 0
    y = .8*x[:, -1]/.001+.5*x[:, -2]/.001+rng.normal(0, .03, len(x))
    source = dict(run_id='synthetic', source_fingerprint='synthetic', dataset_sha256='frozen',
        window_key=f'duration_{width*3600}', input_dims=width, interval_seconds=3600,
        grid_normalization='swing-mean', grid_schema=grid_schema('swing-mean'), symbols=['EURUSD'],
        opportunity_config=dict(interval_seconds=3600, order_expiration_bars=2, stop_buffer_pips=2.,
                                trailing_start_r=1., execution_delay_seconds=0., pip_size=0., risk_cash=100.),
        protocol=asdict(PROTOCOL), target='pnl-r')
    return SimpleNamespace(x=x, y=y, source=source, partitions=dict(train=np.arange(train),
        validation=np.arange(train, train+validation)), grid=SimpleNamespace(rows=[dict(event_id=str(i)) for i in range(len(x))]))


def config(**kwargs):
    return NeuralConfig(**dict(dict(epochs=4, patience=4, batch_size=32, min_train_samples=16,
        min_validation_samples=8, min_accepted_validation=5, threads=1, checkpoint_batches=2), **kwargs))


class NeuralGridTests(unittest.TestCase):
    def test_search_plan_reveals_cpu_convolution_switch_and_full_campaign_count(self):
        request = dict(config=asdict(NeuralConfig()), interval_seconds=1200,
            durations=[21600, 43200, 86400, 129600, 172800, 216000, 259200, 302400, 345600],
            architectures=['auto'], profiles=['compact', 'balanced'], seeds=[1, 2, 3])
        plan = search_plan(request)
        self.assertEqual(plan['trial_count'], 54)
        self.assertEqual(plan['counts_by_architecture'], dict(mlp=24, conv1d=30))
        self.assertEqual(plan['trials'][24]['window_seconds'], 172800)
        self.assertEqual(plan['trials'][24]['input_dims'], 144)
        # An explicit family overlapping auto must not inflate the estimate.
        request.update(durations=[43200], architectures=['auto', 'mlp', 'conv1d'])
        plan = search_plan(request)
        self.assertEqual(plan['trial_count'], 12)
        self.assertEqual(len({t['name'] for t in plan['trials']}), 12)

    def test_automatic_architectures_shapes_counts_and_configuration_validation(self):
        for width in (3, 18, 36, 72, 128, 129, 216, 1024):
            for profile in ('compact', 'balanced', 'wide'):
                spec = architecture_spec(width, config(profile=profile))
                model = build_model(spec)
                self.assertEqual(model(torch.ones(3, width)).shape, (3, 1))
                self.assertEqual(spec['architecture'], 'mlp' if width<=128 else 'conv1d')
                self.assertEqual(sum(p.numel() for p in model.parameters()), spec['parameters'])
        self.assertEqual(architecture_spec(36, NeuralConfig())['hidden'], [64, 32])
        self.assertEqual(architecture_spec(36, NeuralConfig())['parameters'], 4481)
        for values in (dict(dropout=1), dict(architecture='transformer'), dict(epochs=0), dict(learning_rate=float('nan'))):
            with self.assertRaises(ValueError):
                NeuralConfig(**values)

    def test_train_only_scalar_scale_preserves_zeros_and_ignores_validation(self):
        data = synthetic()
        indices = data.partitions['train']
        first = normalizer(data, indices, 32)
        data.x[data.partitions['validation']] = 999
        data.y[data.partitions['validation']] = 999
        self.assertEqual(first, normalizer(data, indices, 32))
        self.assertTrue((data.x[indices]/first['input_scale'])[data.x[indices]==0].sum() == 0)
        self.assertFalse(first['input_centering'])

    def test_network_learns_a_known_signal_on_unseen_samples(self):
        data = synthetic(train=512, validation=256)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            result = fit(data, tmp, config(epochs=20, patience=8, learning_rate=.01, dropout=0., batch_size=64,
                                         min_accepted_validation=1000))
            self.assertGreater(result['best_epoch'], 0)
            val = result['validation']
            self.assertLess(val['prediction_mse'], .3*val['constant_mse'])
            self.assertFalse(result['test_used_for_selection'])
            self.assertEqual(result['target'], 'pnl-r')
            self.assertTrue(result['validation_qualified'])
            self.assertLess(val['selected']['opportunities'], 1000)
            from ultimateBot.run_swing_nn import rank_models
            ranked = rank_models([dict(directory=str(Path(tmp)), metadata=result)])
            self.assertTrue(ranked['validation_qualified'])

    def test_mid_batch_resume_restores_optimizer_rng_scheduler_and_exact_weights(self):
        data = synthetic(train=96, validation=48)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            full = fit(data, root/'full', config())
            partial = fit(data, root/'resume', config(), stop_after_batches=2)
            self.assertTrue(partial['interrupted'])
            # Recover a shutdown between the two checkpoint publications.
            (root/'resume'/'best.pt').write_bytes(b'interrupted best publication')
            latest_before = hashlib.sha256((root/'resume'/'latest.pt').read_bytes()).hexdigest()
            with patch.object(torch.optim.AdamW, 'step', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                fit(data, root/'resume', config(), resume=True)
            self.assertEqual(latest_before, hashlib.sha256((root/'resume'/'latest.pt').read_bytes()).hexdigest())
            resumed = fit(data, root/'resume', config(), resume=True)
            a = torch.load(root/'full/latest.pt', weights_only=True)
            b = torch.load(root/'resume/latest.pt', weights_only=True)
            for name, value in a['model'].items():
                self.assertTrue(torch.equal(value, b['model'][name]), name)
            self.assertEqual(full['history'], resumed['history'])
            self.assertEqual(full['threshold'], resumed['threshold'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                fit(data, root/'resume', config(), resume=True)
            # An unfinished checkpoint refuses a different statistical setup.
            fit(data, root/'unfinished', config(), stop_after_batches=1)
            with self.assertRaisesRegex(ValueError, 'same data'):
                fit(data, root/'unfinished', config(learning_rate=.01), resume=True)

    def test_constant_selection_reports_real_validation_and_trained_epoch(self):
        data = synthetic(train=256, validation=128)
        train, validation = data.partitions['train'], data.partitions['validation']
        # The training relationship is learnable, but the validation labels
        # equal the training mean: no network can improve its zero baseline MSE.
        data.y -= data.y[train].mean()+.25
        data.y[validation] = data.y[train].mean()
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(log):
            result = fit(data, tmp, config(epochs=2, patience=2, dropout=0.))
            diagnostic = result['epoch_selection_diagnostics']
            self.assertEqual(result['best_epoch'], 0)
            self.assertEqual(result['completed_epochs'], 2)
            self.assertEqual(diagnostic['selected_predictor'], 'training_mean_constant')
            self.assertEqual(result['validation']['rows'], 128)
            self.assertEqual(result['validation']['selected']['opportunities'], 0)
            self.assertEqual(diagnostic['baseline_mse_r'], 0.)
            self.assertGreater(diagnostic['best_trained_mse_r'], 0.)
            best_trained = min(result['history'], key=lambda row: row['validation_mse_r'])
            self.assertEqual(diagnostic['best_trained_epoch'], best_trained['epoch'])
            latest = torch.load(Path(tmp)/'latest.pt', weights_only=True)
            model = build_model(latest['architecture'])
            model.load_state_dict(latest['model'])
            self.assertGreater(predict_array(model, data.x[validation], latest['normalization']).std(), 0.)
            self.assertIn('predictor=training_mean_constant validation_rows=128', log.getvalue())
            self.assertIn('prediction_min_r=', log.getvalue())
            report = (Path(tmp)/'rapport.md').read_text(encoding='utf-8')
            self.assertIn('Exemples de validation évalués : 128', report)
            self.assertIn('Zéro trade accepté ne signifie pas zéro donnée', report)

    def test_threshold_floor_and_abstention_never_force_losing_trades(self):
        cfg = config(min_threshold=.2, min_accepted_validation=3)
        pred = np.array([.1, .3, .4, .5, .6])
        selected, _ = choose_threshold(-np.ones(5), pred, cfg)
        self.assertEqual(selected['opportunities'], 0)
        self.assertGreaterEqual(selected['threshold'], .2)
        selected, _ = choose_threshold(np.ones(5), pred, cfg)
        self.assertEqual(selected['opportunities'], 4)
        selected, _ = choose_threshold(np.ones(5), np.array([-.1, -.1, -.1, .3, .4]), cfg)
        self.assertEqual(selected['opportunities'], 2)

    def test_a_single_accepted_event_is_eligible_even_with_a_legacy_minimum(self):
        self.assertEqual(NeuralConfig().min_accepted_validation, 0)
        y, predictions = np.array([5., -100., -1.]), np.array([.9, .8, -.1])
        selected, _ = choose_threshold(y, predictions, config(min_accepted_validation=1000))
        self.assertEqual(selected['opportunities'], 1)
        self.assertEqual(selected['total_r'], 5.)
        self.assertFalse(selected['diagnostics']['acceptance_count_filter'])
        self.assertEqual(selected['diagnostics']['minimum_accepted'], 0)
        self.assertEqual(selected['diagnostics']['at_floor']['opportunities'], 2)
        self.assertEqual(selected['diagnostics']['at_floor']['total_r'], -95.)
        with self.assertRaises(ValueError):
            NeuralConfig(min_accepted_validation=-1)

    def test_exact_threshold_search_finds_a_profitable_narrow_tail_missed_by_quantiles(self):
        pred = np.r_[np.linspace(.125, .001, 125), -np.ones(44000-125)]
        y = np.zeros(len(pred))
        y[:100] = -.1
        y[100:119] = 1.
        y[119:125] = -2.
        selected, candidates = choose_threshold(y, pred, config(min_accepted_validation=100))
        self.assertEqual(selected['opportunities'], 119)
        self.assertAlmostEqual(selected['total_r'], 9.)
        self.assertEqual(selected['threshold'], pred[118])
        self.assertEqual(selected['diagnostics']['method'], 'exact_unique_scores_no_count_filter_v3')
        self.assertTrue(any(c['threshold']==selected['threshold'] for c in candidates))

    def test_exact_thresholds_match_bruteforce_and_never_split_equal_scores(self):
        from ultimateBot.backtester.research.swing_informer import outcome_metrics
        rng = np.random.default_rng(19)
        for floor in (-.1, 0., .1):
            for _ in range(5):
                pred = np.round(rng.normal(0, .1, 200), 2)
                y = rng.normal(-.05, 1, 200)
                cfg = config(min_threshold=floor, min_accepted_validation=10)
                levels = np.unique(np.r_[floor, np.nextafter(max(float(pred.max()), floor), np.inf), pred[pred>=floor]])
                all_candidates = [dict(threshold=t, **outcome_metrics(y, pred>=t)) for t in levels]
                expected = max(all_candidates, key=lambda c:(c['total_r'], -c['opportunities']))
                actual, _ = choose_threshold(y, pred, cfg)
                self.assertEqual(actual['opportunities'], expected['opportunities'])
                self.assertAlmostEqual(actual['total_r'], expected['total_r'], places=11)

    def test_shared_grids_read_only_frozen_inference_and_real_backtest(self):
        from ultimateBot.backtester.research.swing_candle_prices import CandleCloseIndex
        from ultimateBot.run_backtest import build_engine, parse_args as backtest_args
        from ultimateBot.tests.test_swing_ts2vec_robot import feed_candles
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            setup_database(database)
            parameters = add_sources(database, root/'ticks')
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            data = prepare_data(database, run_id='raw', symbols=['EURUSD', 'GBPUSD'], duration=21600,
                                protocol=PROTOCOL, cache_root=root/'cache')
            try:
                data.snapshot(root)
                reference, _ = prepare(database, run_id='raw', windows=(SwingWindow('duration', 21600),),
                    symbols=['EURUSD','GBPUSD'], periods=dict(train=[PROTOCOL.train_start, PROTOCOL.train_end],
                    validation=[PROTOCOL.validation_start, PROTOCOL.validation_end]), target='pnl-r',
                    cache_root=root/'cache', grid_normalization='candle-close-mean')
                np.testing.assert_array_equal(data.x, reference.matrices['duration_21600'])
                reference.close()
                with patch('ultimateBot.backtester.research.swing_resolver.fit_xgb', side_effect=AssertionError('no trees')):
                    meta = fit(data, root/'model', config(epochs=2))
                loaded = NeuralData.load(root)
                self.assertEqual(loaded.source, data.source)
                loaded.close()
            finally:
                data.close()
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())
            resolver = SwingNeuralResolver(root/'model', threads=1)
            at = START+11*timedelta(days=1)
            points = [point(at-(5-i)*HOUR, 1.1+i*.0001, 'high' if i%2==0 else 'low') for i in range(6)]
            prices = CandleCloseIndex(parameters, symbols=['EURUSD'], start=at-6*HOUR, before=at,
                interval_seconds=3600, cache_root=root/'prices')
            mean = prices.mean('EURUSD', as_of=at, window=resolver.window)
            vector = grid_vector(points, as_of=at, window=resolver.window, interval_seconds=3600, normalization_mean=mean)
            expected = predict_array(resolver.model, vector[None, :], meta['normalization'])[0]
            decision = resolver.predict(points, as_of=at, symbol='EURUSD', normalization_mean=mean)
            self.assertEqual(expected, decision['prediction'])
            with self.assertRaisesRegex(ValueError, 'postdates'):
                resolver.predict(points, as_of=START, symbol='EURUSD', normalization_mean=mean)
            with self.assertRaisesRegex(ValueError, 'scope'):
                resolver.predict(points, as_of=at, symbol='USDJPY', normalization_mean=mean)
            with patch.object(sys, 'argv', ['run_backtest', '--strategy','swing-nn','--swing-nn-model',str(root/'model'),
                                          '--swing-nn-threads','1']):
                engine = build_engine([], backtest_args())
            self.assertEqual(engine.strategy.strategy_name, 'swing-nn')
            self.assertEqual(engine.strategy.min_threshold, meta['min_threshold'])
            feed_candles(engine, 12)
            self.assertGreater(engine.strategy.counts['scored'], 0)
            self.assertEqual(len(engine.strategy.normalization_candles['EURUSD']), 6)
            with patch('ultimateBot.backtester.research.swing_neural.fit', side_effect=AssertionError('frozen')):
                result = frozen_test(database, directory=root/'model', start='2024-01-10', end='2024-01-13',
                                     output_dir=root/'test', threads=1, min_samples=5)
            self.assertFalse(result['fitting_performed'])
            self.assertFalse(result['threshold_changed'])
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())
            checkpoint = root/'model'/'best.pt'
            checkpoint.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                SwingNeuralResolver(root/'model')

    def test_cli_plan_search_selection_resume_and_model_root_are_usable(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            setup_database(database)
            add_sources(database, root/'ticks')
            args = ['--database',str(database),'--run-id','raw','--train-start',PROTOCOL.train_start,
                '--train-end',PROTOCOL.train_end,'--validation-start',PROTOCOL.validation_start,
                '--validation-end',PROTOCOL.validation_end,'--epochs','1','--threads','1',
                '--min-train-samples','10','--min-validation-samples','8','--min-accepted-validation','3']
            planned = main(['plan',*args,'--duration','6h'])
            self.assertEqual(planned['architecture']['input_dims'], 6)
            result = main(['search',*args,'--durations','6h','8h','--architectures','mlp','conv1d',
                '--profiles','compact','--seeds','1','2','--output-dir',str(root/'search')])
            self.assertEqual(len(result['trials']), 8)
            self.assertEqual(len(result['leaderboard']), 4)
            self.assertFalse(result['threshold_reselected'])
            self.assertFalse(result['test_used_for_selection'])
            self.assertTrue(all(row['seeds']==2 for row in result['leaderboard']))
            resolver = SwingNeuralResolver(root/'search', threads=1)
            before = hashlib.sha256((resolver.directory/'best.pt').read_bytes()).hexdigest()
            resumed = main(['search','--resume',str(root/'search')])
            self.assertEqual(result['selected_directory'], resumed['selected_directory'])
            self.assertEqual(before, hashlib.sha256((resolver.directory/'best.pt').read_bytes()).hexdigest())
            source_metadata = (resolver.directory/'metadata.json').read_bytes()
            with patch('ultimateBot.run_swing_nn.fit', side_effect=AssertionError('no retraining')):
                revised = main(['recalibrate','--model',str(root/'search'),'--output-dir',str(root/'recalibrated')])
            copied = SwingNeuralResolver(root/'recalibrated', threads=1)
            self.assertEqual(before, hashlib.sha256((copied.directory/'best.pt').read_bytes()).hexdigest())
            self.assertEqual(source_metadata, (resolver.directory/'metadata.json').read_bytes())
            self.assertTrue(copied.metadata['calibration']['validation_reused'])
            self.assertFalse(copied.metadata['calibration']['training_performed'])
            self.assertEqual(revised['common_validation_events'], result['common_validation_events'])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['search','--resume',str(root/'search'),'--durations','12h'])

    def test_cli_train_resume_retains_extended_budget_and_refuses_to_mutate_frozen_run(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            setup_database(database)
            add_sources(database, root/'ticks')
            args = ['train','--database',str(database),'--run-id','raw','--duration','6h',
                '--train-start',PROTOCOL.train_start,'--train-end',PROTOCOL.train_end,
                '--validation-start',PROTOCOL.validation_start,'--validation-end',PROTOCOL.validation_end,
                '--epochs','2','--batch-size','32','--threads','1','--min-train-samples','10',
                '--min-validation-samples','8','--min-accepted-validation','3','--output-dir',str(root/'model')]
            def interrupt(*a, **kw):
                return fit(*a, **kw, stop_after_batches=2)
            with patch('ultimateBot.run_swing_nn.fit', side_effect=interrupt):
                result = main(args)
            self.assertTrue(result['interrupted'])
            result = main(['train','--resume',str(root/'model'),'--epochs','3','--min-accepted-validation','0'])
            self.assertEqual(result['completed_epochs'], 3)
            self.assertEqual(result['config']['min_accepted_validation'], 0)
            request = root/'model'/'request.json'
            before = request.read_bytes()
            self.assertEqual(json.loads(before)['config']['epochs'], 3)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['train','--resume',str(root/'model'),'--epochs','4'])
            self.assertEqual(request.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
