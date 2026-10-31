from contextlib import redirect_stdout
from dataclasses import replace
from datetime import timedelta
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.research.swing_encoder import SwingEncoderConfig
from ultimateBot.backtester.research.swing_opportunities import connect
from ultimateBot.backtester.research.swing_resolver import (
    ResolverConfig, SwingResolver, TrainingSelection, choose_threshold, evaluate, fit_xgb, frozen_test,
    target_label, train,
)
from ultimateBot.backtester.research.swing_window_search import search
from ultimateBot.backtester.strategy.swing_threshold import DailyTradeTargetConfig
from ultimateBot.backtester.strategy.swing_ts2vec_strategy import SwingTS2VecStrategy, resolve_swing_threshold
from ultimateBot.run_swing_ts2vec import parse_args
from ultimateBot.tests.test_swing_ts2vec import START, PROTOCOL, create_training_database, points
from ultimateBot.tests.test_swing_ts2vec_robot import metadata


CONFIG = ResolverConfig(target='trailing-step-1', min_encoder_samples=8, min_train_samples=15,
                        min_validation_samples=8, min_accepted_validation=3,
                        n_estimators=6, early_stopping_rounds=2, threads=1)
ENCODER = SwingEncoderConfig(output_dims=4, hidden_dims=4, depth=1, train_iters=2, threads=1)


def training_database(path):
    create_training_database(path)
    db = connect(path)
    # Labels intentionally disagree with PnL sign for many observations.
    db.execute('UPDATE swing_events SET trailing_steps=CAST(substr(event_id,6) AS INTEGER)%4')
    db.commit()
    return db


class TrailingTargetTests(unittest.TestCase):
    def test_evaluation_reports_labels_separately_from_realized_pnl_with_first_stop_policy(self):
        targets = {'trailing-step-1': lambda s: int(s >= 1), 'trailing-step-raw': float,
                   'trailing-step-square': lambda s: float(s*s), 'trailing-step-exp': np.exp}
        rows = [dict(event_id=event, pnl_r=pnl, trailing_steps=steps) for event,pnl,steps in (
            ('a', -10., 2), ('a:stop1', 100., 8), ('b', 5., 0), ('b:stop1', -7., 4))]
        pred = np.asarray([.9, .95, .1, .8])
        for target, transform in targets.items():
            with self.subTest(target=target):
                report = evaluate(rows, pred, .5, target, dict(stop_variants=2))
                self.assertEqual(report['selected']['opportunities'], 2)
                self.assertEqual(report['selected']['total_r'], -17.)
                self.assertEqual(report['selected']['known_target_labels'], 2)
                self.assertAlmostEqual(report['selected']['total_label'], transform(2)+transform(4))
                self.assertAlmostEqual(report['selected']['mean_label'], (transform(2)+transform(4))/2)
                self.assertAlmostEqual(report['all_opportunities']['total_label'], transform(2)+transform(0))
                empty = evaluate(rows, pred, 1., target, dict(stop_variants=2))
                self.assertEqual(empty['selected']['total_label'], 0.)
                self.assertIsNone(empty['selected']['mean_label'])
        unknown = evaluate([dict(event_id='x', pnl_r=3., trailing_steps=None)], np.asarray([.9]), .5, 'trailing-step-raw')
        self.assertEqual(unknown['selected']['known_target_labels'], 0)
        self.assertIsNone(unknown['selected']['mean_label'])

    def test_cli_and_label_definition_independent_of_pnl(self):
        for command in ('train', 'run', 'search'):
            self.assertEqual(parse_args([command, '--target', 'trailing-step-1']).target, 'trailing-step-1')
        for steps, expected in ((0, 0), (1, 1), (2, 1), (19, 1), (None, None), (-1, None), (.5, None), (float('inf'), None)):
            for pnl in (-2., 0., 3.):
                with self.subTest(steps=steps, pnl=pnl):
                    self.assertEqual(target_label({'trailing_steps': steps, 'pnl_r': pnl}, 'trailing-step-1'), expected)
        self.assertEqual(parse_args(['train']).target, 'pnl-r')

    def test_view_can_be_added_during_collection_and_does_not_materialize_or_modify_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.sqlite3'
            writer = training_database(path)
            writer.execute('DROP VIEW swing_event_labels')
            writer.execute("UPDATE swing_events SET status='open',trailing_steps=NULL WHERE event_id='event0'")
            writer.commit()
            original = [tuple(r) for r in writer.execute('SELECT * FROM swing_events ORDER BY event_id')]
            reader = connect(path)
            label = lambda: reader.execute("SELECT trailing_step_1_label FROM swing_event_labels WHERE event_id='event0'").fetchone()[0]
            self.assertIsNone(label())
            self.assertEqual(original, [tuple(r) for r in writer.execute('SELECT * FROM swing_events ORDER BY event_id')])
            for status, steps, expected in (('open', 2, None), ('expired', 0, None), ('closed', None, None),
                                            ('closed', 0, 0), ('closed', 1, 1), ('closed', 3, 1)):
                writer.execute("UPDATE swing_events SET status=?,trailing_steps=? WHERE event_id='event0'", (status, steps))
                writer.commit()
                self.assertEqual(label(), expected)
            writer.close()
            reader.close()

    def test_validation_chooses_probability_threshold_by_mean_label_independent_of_pnl(self):
        rows = [{'pnl_r': -5., 'trailing_steps': 1}, {'pnl_r': 4., 'trailing_steps': 0},
                {'pnl_r': 4., 'trailing_steps': 0}]
        predictions = np.asarray([.9, .8, .1])
        config = replace(CONFIG, min_accepted_validation=1000)
        threshold, candidates = choose_threshold(rows, predictions, config)
        self.assertEqual(threshold, .9)
        self.assertTrue(all(0 <= row['threshold'] <= 1 for row in candidates))
        report = evaluate(rows, predictions, threshold, 'trailing-step-1')
        self.assertEqual(report['selected']['total_r'], -5.)
        self.assertEqual(report['selected']['opportunities'], 1)
        self.assertEqual(report['selected']['mean_label'], 1.)
        changed = [{**row, 'pnl_r': -1000*row['pnl_r']} for row in rows]
        self.assertEqual(choose_threshold(changed, predictions, config)[0], threshold)
        self.assertEqual(report['target_positive_labels'], 1)
        self.assertAlmostEqual(report['prediction_mse'], np.mean((np.asarray([1.,0.,0.])-predictions)**2))
        _, candidates = choose_threshold(rows, np.zeros(3), config)
        self.assertTrue(all(0 <= row['threshold'] <= 1 for row in candidates))
        with self.assertRaisesRegex(ValueError, 'probabilities between 0 and 1'):
            choose_threshold(rows, np.asarray([-.1, .8, .9]), config)
        unknown = evaluate([{'pnl_r': 3., 'trailing_steps': None}], np.asarray([.8]), .5, 'trailing-step-1')
        self.assertEqual(unknown['selected']['total_r'], 3.)
        self.assertEqual(unknown['known_target_labels'], 0)
        self.assertIsNone(unknown['prediction_mse'])

    def test_training_receives_step_labels_and_reload_test_keep_probabilities_and_financial_results(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            db = training_database(path)
            # Unknown labels must not be silently changed into class 0.
            db.execute("UPDATE swing_events SET trailing_steps=NULL WHERE event_id='event0'")
            # This trade's label becomes known only in the next period.
            db.execute("UPDATE swing_events SET closed_at='2024-02-16T01:00:00+00:00' WHERE event_id='event1'")
            original = [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')]
            db.commit()
            db.close()
            with patch('ultimateBot.backtester.research.swing_resolver.fit_xgb', wraps=fit_xgb) as fitted:
                result = train(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                               protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)))
            directory = Path(result['selected_directory'])
            with (directory/'predictions.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            ids = {r['event_id'] for r in rows}
            self.assertNotIn('event0', ids)
            self.assertNotIn('event1', ids)
            for phase in ('train', 'validation'):
                samples = [r for r in rows if r['partition'] == phase]
                expected = [int(int(r['trailing_steps']) >= 1) for r in samples]
                for call in fitted.call_args_list:
                    np.testing.assert_array_equal(call.args[0][phase][1], expected)
                self.assertTrue(any(int(r['target_label']) != int(float(r['pnl_r']) > 0) for r in samples))
            resolver = SwingResolver(directory, threads=1)
            self.assertEqual(resolver.metadata['target'], 'trailing-step-1')
            self.assertFalse(any('trailing' in field or 'pnl' in field for field in resolver.metadata['feature_names']))
            seq = points(4, START+timedelta(days=70, hours=12))
            decision = resolver.predict(seq, as_of=seq[-1].confirmed_at)
            self.assertTrue(0 <= decision['prediction'] <= 1)
            self.assertTrue(0 <= decision['threshold'] <= 1)
            tested = frozen_test(path, directory=result['directory'], run_id='run', start='2024-03-05',
                                 end='2024-03-20', min_samples=5, threads=1)
            self.assertFalse(tested['fit_called'])
            with (Path(tested['directory'])/'predictions.csv').open(newline='') as handle:
                predictions = list(csv.DictReader(handle))
            realized = sum(float(r['pnl_r']) for r in predictions if r['label_eligible']=='1' and r['accepted']=='1')
            self.assertAlmostEqual(realized, tested['selected']['selected']['total_r'])
            db = connect(path)
            self.assertEqual(original, [tuple(r) for r in db.execute('SELECT * FROM swing_events ORDER BY event_id')])
            db.close()

    def test_missing_step_labels_and_single_class_are_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            create_training_database(path)
            with self.assertRaisesRegex(ValueError, 'known trailing_steps'):
                train(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                      protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)))
        arrays = {'train': (np.ones((4, 2)), np.ones(4)), 'validation': (np.ones((4, 2)), np.zeros(4))}
        with self.assertRaisesRegex(ValueError, 'both label 0 and label 1'):
            fit_xgb(arrays, CONFIG)

    def test_repeated_window_search_supports_new_target(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/'data.sqlite3'
            training_database(path).close()
            result = search(path, output_dir=Path(tmp)/'models', config=CONFIG, encoder_config=ENCODER,
                protocol=PROTOCOL, selection=TrainingSelection(pairs=(2,)), repeats=2)
            self.assertEqual(result['plan']['config']['target'], 'trailing-step-1')
            self.assertEqual(len(result['trials']), 2)
            self.assertFalse(result['test_evaluated'])
            for trial in result['trials']:
                report = json.loads(Path(trial['comparison']).read_text())
                self.assertEqual(report['target_definition']['prediction'], 'probability')
                self.assertEqual(report['target_definition']['threshold_selection_metric'], 'validation_mean_target_label')

    def test_robot_static_and_dynamic_thresholds_stay_in_probability_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            values = metadata(directory)
            values['target'] = 'trailing-step-1'
            values['threshold'] = .999
            (directory/'metadata.json').write_text(json.dumps(values))
            for threshold in (-.01, 1.01):
                with self.assertRaisesRegex(ValueError, 'between 0 and 1'):
                    resolve_swing_threshold(values, threshold)
                with self.assertRaisesRegex(ValueError, 'between 0 and 1'):
                    resolve_swing_threshold({**values, 'threshold': threshold})
            for threshold in (0., .5, 1.):
                self.assertEqual(resolve_swing_threshold(values, threshold), threshold)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver'):
                strategy = SwingTS2VecStrategy(directory, daily_target=DailyTradeTargetConfig(1., .001))
            controller = strategy.daily_target
            controller.advance(START)
            for _ in range(5):
                controller.record(True)
            self.assertEqual(controller.threshold, 1.)
            with patch('ultimateBot.backtester.strategy.swing_ts2vec_strategy.SwingResolver'):
                strategy = SwingTS2VecStrategy(directory, threshold=0., daily_target=DailyTradeTargetConfig(1., .001))
            controller = strategy.daily_target
            controller.advance(START)
            controller.record(False)
            controller.advance(START+timedelta(days=1))
            self.assertEqual(controller.threshold, 0.)


if __name__ == '__main__':
    unittest.main()
