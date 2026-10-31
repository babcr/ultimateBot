"""Continuous trailing labels survive training, model reload and frozen evaluation."""
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import timedelta
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xgboost as xgb

from ultimateBot.backtester.research.swing_resolver import (
    SwingResolver, TrainingSelection, choose_threshold, evaluate, fit_xgb, frozen_test,
    target_definition, target_label, train,
)
from ultimateBot.backtester.research.swing_window_search import search
from ultimateBot.backtester.strategy.swing_threshold import DailyTradeTargetConfig
from ultimateBot.backtester.strategy.swing_ts2vec_strategy import SwingTS2VecStrategy, resolve_swing_threshold
from ultimateBot.run_swing_ts2vec import parse_args
from ultimateBot.tests.test_swing_trailing_target import CONFIG, ENCODER, training_database
from ultimateBot.tests.test_swing_ts2vec import START, PROTOCOL, points
from ultimateBot.tests.test_swing_ts2vec_robot import metadata


TRANSFORMS = {'trailing-step-raw': float, 'trailing-step-square': lambda s: float(s*s),
              'trailing-step-exp': math.exp}


class TrailingRegressionTests(unittest.TestCase):
    def test_mean_of_transformed_labels_changes_optimal_threshold(self):
        rows = [dict(event_id=str(i), pnl_r=1000., trailing_steps=step) for i,step in enumerate([4, 4, 0, 7])]
        scores = np.asarray([4., 3., 2., 1.])
        for target, expected in (('trailing-step-raw', 3.), ('trailing-step-square', 0.), ('trailing-step-exp', 0.)):
            threshold, _ = choose_threshold(rows, scores, replace(CONFIG, target=target, min_accepted_validation=1000))
            self.assertEqual(threshold, expected)
        # A zero-step exp label is 1, not 0; it carries no quality above abstention.
        for target in ('trailing-step-1', *TRANSFORMS):
            empty_rows = [dict(event_id=str(i), pnl_r=1000., trailing_steps=0) for i in range(3)]
            threshold, _ = choose_threshold(empty_rows, np.asarray([.1,.5,.9]), replace(CONFIG, target=target))
            self.assertGreater(threshold, .9)
            if target == 'trailing-step-1':
                self.assertLessEqual(threshold, 1.)
        # Exact probability 1 leaves no representable abstention threshold <= 1.
        threshold, _ = choose_threshold(empty_rows, np.asarray([.1,.5,1.]), CONFIG)
        self.assertEqual(threshold, 1.)

    def test_target_thresholds_match_bruteforce_with_and_without_stop_variants(self):
        rng = np.random.default_rng(842)
        for target in ('trailing-step-1', *TRANSFORMS):
            neutral = 1. if target == 'trailing-step-exp' else 0.
            for variants in (1, 3):
                for _ in range(5):
                    rows = [dict(event_id=f'g{g}' if variant == 0 else f'g{g}:stop{variant}',
                        pnl_r=float(rng.normal(-2., 8.)), trailing_steps=int(rng.integers(0, 9)))
                        for g in range(9) for variant in range(variants)]
                    scores = np.round(rng.uniform(.1,.9,len(rows)), 1)
                    order = rng.permutation(len(rows))[:len(rows)-1] if variants > 1 else rng.permutation(len(rows))
                    rows, scores = [rows[i] for i in order], scores[order]
                    settings = dict(stop_variants=variants)
                    levels = np.unique(np.r_[0.,scores,np.nextafter(scores.max(),np.inf)])
                    results = [evaluate(rows,scores,float(t),target,settings) for t in levels]
                    def key(result):
                        selected = result['selected']
                        value = selected['mean_label'] if selected['mean_label'] is not None else neutral
                        return value, selected['opportunities'] if value > neutral else -selected['opportunities']
                    expected = max(results,key=key)
                    threshold,candidates = choose_threshold(rows,scores,replace(CONFIG,target=target,min_accepted_validation=1000),settings)
                    actual = evaluate(rows,scores,threshold,target,settings)
                    self.assertAlmostEqual(actual['selection_score'],expected['selection_score'])
                    self.assertEqual(actual['selected']['opportunities'],expected['selected']['opportunities'])
                    for point in candidates:
                        report = evaluate(rows,scores,point['threshold'],target,settings)
                        self.assertAlmostEqual(point['total_r'],report['selected']['total_r'])
                        self.assertAlmostEqual(point['total_label'],report['selected']['total_label'])

    def test_cli_and_exact_transforms_independent_of_pnl(self):
        for target, transform in TRANSFORMS.items():
            for command in ('train', 'run', 'search'):
                self.assertEqual(parse_args([command, '--target', target]).target, target)
            for steps in (0, 1, 2, 10):
                for pnl in (-2., 0., 3.):
                    self.assertEqual(target_label({'trailing_steps': steps, 'pnl_r': pnl}, target), transform(steps))
            for invalid in (None, -1, .5, float('inf'), float('nan'), '2'):
                self.assertIsNone(target_label({'trailing_steps': invalid}, target))
            self.assertEqual(target_definition(target)['prediction'], 'transformed_trailing_steps')
        self.assertEqual(target_label({'trailing_steps': 0}, 'trailing-step-exp'), 1.)
        for steps in (45, 1000):
            with self.assertRaisesRegex(ValueError, 'too large'):
                target_label({'trailing_steps': steps}, 'trailing-step-exp')

    def test_thresholds_maximize_mean_label_without_a_minimum_or_pnl_filter(self):
        rows = [{'pnl_r': -5., 'trailing_steps': 8}, {'pnl_r': 4., 'trailing_steps': 0},
                {'pnl_r': 4., 'trailing_steps': 1}]
        predictions = np.asarray([9., 8., 2.])
        for target, transform in TRANSFORMS.items():
            cfg = replace(CONFIG, target=target, min_accepted_validation=1000)
            threshold, candidates = choose_threshold(rows, predictions, cfg)
            self.assertEqual(threshold, 9.)
            self.assertTrue(any(c['threshold'] > 1 for c in candidates))
            report = evaluate(rows, predictions, threshold, target)
            self.assertEqual(report['selected']['total_r'], -5.)
            self.assertEqual(report['selected']['opportunities'], 1)
            self.assertEqual(report['selected']['mean_label'], transform(8))
            self.assertEqual(choose_threshold([{**row, 'pnl_r': 100.} for row in rows], predictions, cfg)[0], threshold)
            self.assertIsNone(report['target_positive_labels'])
            self.assertAlmostEqual(report['prediction_mse'], np.mean((np.asarray([transform(8), transform(0), transform(1)])-predictions)**2))
            unknown = evaluate([{'pnl_r': 3., 'trailing_steps': None}], np.asarray([8.]), 5., target)
            self.assertEqual(unknown['known_target_labels'], 0)
            self.assertIsNone(unknown['prediction_mse'])

    def test_real_training_reload_and_frozen_test_preserve_continuous_labels(self):
        for target, transform in TRANSFORMS.items():
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                path = Path(tmp)/'data.sqlite3'
                db = training_database(path)
                db.execute("UPDATE swing_events SET trailing_steps=NULL WHERE event_id='event0'")
                db.execute("UPDATE swing_events SET closed_at='2024-02-16T01:00:00+00:00' WHERE event_id='event1'")
                # Outcome occurs outside frozen test period, so must not be exported as a known label.
                db.execute("UPDATE swing_events SET closed_at='2024-03-21T01:00:00+00:00' WHERE event_id='event79'")
                db.commit()
                original = [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
                with patch('ultimateBot.backtester.research.swing_resolver.fit_xgb', wraps=fit_xgb) as fitted:
                    result = train(path, output_dir=Path(tmp)/'models', config=replace(CONFIG, target=target),
                                   encoder_config=ENCODER, protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)))
                directory = Path(result['selected_directory'])
                with (directory/'predictions.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertFalse({'event0', 'event1'} & {r['event_id'] for r in rows})
                for phase in ('train', 'validation'):
                    samples = [r for r in rows if r['partition'] == phase]
                    expected = [transform(int(r['trailing_steps'])) for r in samples]
                    self.assertGreater(len(set(expected)), 2)
                    np.testing.assert_allclose([float(r['target_label']) for r in samples], expected)
                    for call in fitted.call_args_list:
                        np.testing.assert_allclose(call.args[0][phase][1], expected)
                resolver = SwingResolver(directory, threads=1)
                self.assertIsInstance(resolver.model, xgb.XGBRegressor)
                self.assertEqual(resolver.model.get_xgb_params()['objective'], 'reg:squarederror')
                report = json.loads((directory/'report.json').read_text())
                self.assertEqual(report['learning_diagnostics']['error_metric'], 'mse')
                self.assertIsNone(report['learning_diagnostics']['partitions']['validation']['auc'])
                self.assertFalse(any('trailing' in field or 'pnl' in field for field in resolver.metadata['feature_names']))
                seq = points(4, START+timedelta(days=70, hours=12))
                decision = resolver.predict(seq, as_of=seq[-1].confirmed_at)
                self.assertTrue(math.isfinite(decision['prediction']))
                self.assertEqual(decision['target'], target)
                with patch('ultimateBot.backtester.research.swing_resolver.fit_xgb', side_effect=AssertionError('frozen test fitted')):
                    tested = frozen_test(path, directory=result['directory'], run_id='run', start='2024-03-05',
                                         end='2024-03-20', min_samples=5, threads=1)
                self.assertFalse(tested['fit_called'])
                with (Path(tested['directory'])/'predictions.csv').open(newline='') as handle:
                    predictions = list(csv.DictReader(handle))
                self.assertTrue(any(r['label_eligible'] == '0' for r in predictions))
                for row in predictions:
                    if row['label_eligible'] == '1':
                        self.assertAlmostEqual(float(row['target_label']), transform(int(row['trailing_steps'])))
                    else:
                        self.assertEqual(row['target_label'], '')
                realized = sum(float(r['pnl_r']) for r in predictions if r['label_eligible']=='1' and r['accepted']=='1')
                self.assertAlmostEqual(realized, tested['selected']['selected']['total_r'])
                self.assertEqual(original, [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')])
                db.close()

    def test_search_accepts_all_continuous_targets(self):
        for target in TRANSFORMS:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                path = Path(tmp)/'data.sqlite3'
                training_database(path).close()
                result = search(path, output_dir=Path(tmp)/'models', config=replace(CONFIG, target=target),
                                encoder_config=ENCODER, protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)), repeats=1)
                self.assertEqual(result['plan']['config']['target'], target)
                report = json.loads(Path(result['trials'][0]['comparison']).read_text())
                self.assertEqual(report['target_definition']['prediction'], 'transformed_trailing_steps')

    def test_robot_static_and_adaptive_thresholds_use_regression_units(self):
        for target in TRANSFORMS:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                values = {**metadata(directory), 'target': target, 'threshold': 12.}
                (directory/'metadata.json').write_text(json.dumps(values))
                for threshold in (-2., 0., 12., 100.):
                    self.assertEqual(resolve_swing_threshold(values, threshold), threshold)
                with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver'):
                    strategy = SwingTS2VecStrategy(directory, daily_target=DailyTradeTargetConfig(1., 2.))
                controller = strategy.daily_target
                controller.advance(START)
                controller.record(True)
                controller.record(True)
                self.assertEqual(controller.threshold, 14.)
                with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver'):
                    strategy = SwingTS2VecStrategy(directory, threshold=0., daily_target=DailyTradeTargetConfig(1., 2.))
                controller = strategy.daily_target
                controller.advance(START)
                controller.record(False)
                controller.advance(START+timedelta(days=1))
                self.assertEqual(controller.threshold, -2.)

    def test_unsafe_exp_labels_fail_before_encoder_training(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            db = training_database(path)
            db.execute("UPDATE swing_events SET trailing_steps=1000 WHERE event_id='event2'")
            db.commit()
            db.close()
            with patch('ultimateBot.backtester.research.swing_resolver.SwingEncoder.fit', side_effect=AssertionError('encoder fitted')):
                with self.assertRaisesRegex(ValueError, 'trailing-step-exp.*too large'):
                    train(path, output_dir=Path(tmp)/'models', config=replace(CONFIG, target='trailing-step-exp'),
                          encoder_config=ENCODER, protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)))


if __name__ == '__main__':
    unittest.main()
