import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ultimateBot.backtester.research.swing_selection import StopSelection, exact_stop_threshold, FIRST_STOP_POLICY


class StopSelectionTests(unittest.TestCase):
    def test_first_stop_is_selected_even_if_farther_stop_has_better_score_or_outcome(self):
        policy = StopSelection(['a','b','a','b','a','b'], [2,1,0,0,1,2], 3)
        pred = np.asarray([.99,.8,.7,.1,.9,.9])
        np.testing.assert_array_equal(policy.mask(pred, .5), [False,True,True,False,False,False])
        np.testing.assert_array_equal(policy.mask(pred, .85), [False,False,False,False,True,True])

    def test_lowering_threshold_replaces_wide_stop_and_can_reduce_pnl(self):
        policy = StopSelection.from_event_ids(['a','a:stop1'], dict(stop_variants=2))
        best, candidates = exact_stop_threshold([-3,5], [.4,.8], 0., policy)
        self.assertEqual(best['threshold'], .8)
        self.assertEqual(best['total_r'], 5.)
        floor = best['diagnostics']['at_floor']
        self.assertEqual(floor['total_r'], -3.)
        self.assertEqual(floor['opportunities'], 1)
        self.assertEqual(best['diagnostics']['selection_policy'], FIRST_STOP_POLICY)

    def test_profitable_wider_stop_cannot_be_used_if_shorter_stop_already_passes(self):
        policy = StopSelection.from_event_ids(['a','a:stop1'], dict(stop_variants=2))
        best, _ = exact_stop_threshold([-3,100], [.9,.8], 0., policy)
        self.assertEqual(best['opportunities'], 0)
        self.assertEqual(best['total_r'], 0.)

    def test_tied_scores_and_incomplete_swing_families(self):
        policy = StopSelection(['a','a','b','c'], [1,0,1,0], 2)
        np.testing.assert_array_equal(policy.mask(np.ones(4), 1.), [False,True,False,False])
        self.assertEqual(policy.diagnostic()['complete_swings'], 1)
        self.assertEqual(policy.diagnostic()['incomplete_swings'], 2)
        subset = policy.subset(np.array([True,False,False,False]))
        self.assertEqual(subset.mask(np.ones(1), 0.).sum(), 0)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            StopSelection(['a','a'], [0,0], 2)

    def test_exact_sweep_matches_independent_bruteforce_for_shuffled_groups_and_floors(self):
        rng = np.random.default_rng(71)
        for floor in (-.1, 0., .6):
            for _ in range(12):
                groups = np.repeat(np.arange(25).astype(str), 4)
                ranks = np.tile(np.arange(4), 25)
                order = rng.permutation(len(groups))[:95]  # include incomplete groups
                groups, ranks = groups[order], ranks[order]
                labels = rng.integers(-4, 8, len(groups)).astype(float)
                predictions = np.round(rng.normal(.2, .3, len(groups)), 2)
                policy = StopSelection(groups, ranks, 4)
                thresholds = np.unique(np.r_[floor, predictions[predictions >= floor],
                    max(floor, np.nextafter(predictions.max(), np.inf))])
                expected = []
                for t in thresholds:
                    selected = []
                    for group in set(groups):
                        indices = np.flatnonzero(groups == group)
                        if len(indices) != 4:
                            continue
                        passing = [i for i in indices if predictions[i] >= t]
                        if passing:
                            selected.append(min(passing, key=lambda i:ranks[i]))
                    expected.append(dict(threshold=t, total_r=float(labels[selected].sum()), opportunities=len(selected)))
                best = max(expected, key=lambda r:(r['total_r'], -r['opportunities']))
                actual, _ = exact_stop_threshold(labels, predictions, floor, policy)
                for name in ('threshold','total_r','opportunities'):
                    self.assertEqual(actual[name], best[name])

    def test_neural_informer_and_xgboost_share_first_stop_threshold_and_metrics(self):
        from ultimateBot.backtester.research.swing_neural import choose_threshold, NeuralConfig
        from ultimateBot.backtester.research.swing_informer import validation_threshold, evaluate_predictions, InformerConfig
        from ultimateBot.backtester.research.swing_resolver import choose_threshold as choose_xgb, evaluate, ResolverConfig
        ids = ['a','a:stop1']
        policy = StopSelection.from_event_ids(ids, dict(stop_variants=2))
        y, pred = np.asarray([-3.,5.]), np.asarray([.4,.8])
        for choose, config in ((choose_threshold, NeuralConfig(min_accepted_validation=1000)),
                               (validation_threshold, InformerConfig(min_accepted_validation=1000))):
            best, _ = choose(y, pred, config, policy)
            self.assertEqual(best['threshold'], .8)
            self.assertEqual(best['opportunities'], 1)
        metrics = evaluate_predictions(y, pred, .1, 0., policy)
        self.assertEqual(metrics['selected']['total_r'], -3.)
        self.assertEqual(metrics['selected']['opportunities'], 1)
        rows = [dict(event_id=event, pnl_r=value) for event,value in zip(ids,y)]
        t, _ = choose_xgb(rows, pred, ResolverConfig(min_accepted_validation=1000), dict(stop_variants=2))
        self.assertEqual(t, .8)
        self.assertEqual(evaluate(rows, pred, .1, 'pnl-r', dict(stop_variants=2))['selected']['total_r'], -3.)

    def test_neural_common_validation_ranking_recomputes_single_trade_metrics(self):
        from ultimateBot.run_swing_nn import rank_models
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'validation_predictions.npz'
            np.savez_compressed(path, event_id=['a','a:stop1','b','b:stop1'],
                                actual=[-1.,100.,-1.,100.], prediction=[.8,.9,.8,.9])
            meta = dict(validation_predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config=dict(min_validation_samples=2, profile='balanced', seed=1),
                opportunity_config=dict(stop_variants=2), threshold=0., normalization=dict(target_mean=0.),
                window_key='duration_43200', architecture=dict(architecture='mlp'), validation_qualified=True)
            result = rank_models([dict(directory=str(root), metadata=meta)])
            self.assertEqual(result['selected_comparison_validation']['selected']['opportunities'], 2)
            self.assertEqual(result['selected_comparison_validation']['selected']['total_r'], -2.)
            self.assertFalse(result['validation_qualified'])


if __name__ == '__main__':
    unittest.main()
