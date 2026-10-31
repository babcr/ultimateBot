from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, SWING_SCHEMA, SwingPoint
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA
from ultimateBot.backtester.research.swing_statistics import (
    COL, FEATURE_COLUMNS, NUMERIC_FEATURES, StatisticalProtocol, adjust_fdr,
    antecedent_features, block_bootstrap, cluster_test, extract_events, make_terms, phase_data, research, stamp,
)


def synthetic_data():
    rows = []
    for year in range(2015, 2026):
        start = stamp(f'{year}-01-01')
        for i in range(720):
            at = start+(i//2)*86400+(12+i%2)*3600
            row = np.ones(len(FEATURE_COLUMNS))
            row[COL['rowid']] = len(rows)+1
            row[COL['symbol']] = (i//3)%3
            row[COL['confirmed_at']] = at
            row[COL['closed_at']] = at+600
            row[COL['closed']] = 1
            row[COL['side']] = 1 if i%2 else -1
            row[COL['weekday']] = datetime.fromtimestamp(at, timezone.utc).weekday()
            row[COL['hour_bucket']] = 2
            row[COL['structure']] = (i//7)%3-1
            row[COL['leg_ratio']] = (.5, 2., 4.)[i%3]
            row[COL['volume_ratio']] = .5+(i%13)/8
            row[COL['opposite_volume_ratio']] = .5+(i%11)/8
            row[COL['pnl_r']] = (.7 if i%10 else -1.) if i%3 == 0 else -1.
            rows.append(row)
    return np.asarray(rows)


class SwingStatisticsTests(unittest.TestCase):
    def test_extraction_groups_simultaneous_symbols_censors_future_and_leaves_database_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            database = root/'research.sqlite3'
            parameters = {'symbols': ['A', 'B'], 'config': {'interval_seconds': 1200},
                          'swing_schema': SWING_SCHEMA, 'opportunity_schema': OPPORTUNITY_SCHEMA,
                          'feature_schema': FEATURE_SCHEMA, 'source_fingerprint': 'fixture'}
            with sqlite3.connect(database) as con:
                con.execute('CREATE TABLE swing_runs (run_id TEXT, status TEXT, started_at TEXT, parameters_json TEXT)')
                con.execute('INSERT INTO swing_runs VALUES (?,?,?,?)', ('fixture', 'completed', '2020', json.dumps(parameters)))
                con.execute('''CREATE TABLE swing_events (event_id TEXT PRIMARY KEY, run_id TEXT, symbol TEXT,
                    kind TEXT, pivot_price REAL, stop_price REAL, pivot_at TEXT, pair_start TEXT,
                    confirmed_at TEXT, status TEXT, closed_at TEXT, pnl_r REAL, mean_quote_volume REAL)''')
                start = datetime(2020, 1, 1, tzinfo=timezone.utc)
                def insert(event_id, symbol, i, *, opposite=False):
                    at = start+timedelta(minutes=20*i)
                    kind = 'high' if (i+opposite)%2 == 0 else 'low'
                    values = (event_id, 'fixture', symbol, kind, 1.12 if kind == 'high' else 1.1, 1.,
                              (at-timedelta(minutes=20)).isoformat(), (at-timedelta(minutes=40)).isoformat(),
                              at.isoformat(), 'closed', (at+timedelta(minutes=5)).isoformat(), -.5, 100.)
                    con.execute('INSERT INTO swing_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
                for i in range(24):
                    insert(f'{i:03d}a', 'A', i)
                    insert(f'{i:03d}b', 'B', i)
                    if i == 18:
                        # Sorting only by time/event_id interleaves B between A's opposite reversals.
                        insert(f'{i:03d}c', 'A', i, opposite=True)
                insert('future', 'A', 200000)
            con.close()
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            data, metadata = extract_events(database, 'fixture', StatisticalProtocol(), root/'export')
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())
            self.assertEqual(metadata['counts']['raw_events'], 49)
            self.assertEqual(metadata['counts']['ambiguous_events'], 2)
            self.assertEqual(metadata['usable_contexts'], 12)
            self.assertEqual(len(data), 12)
            self.assertEqual(metadata['extractor_version'], 2)
            self.assertTrue(metadata['read_only'])

    def test_geometry_and_volume_are_from_past_points_and_exclude_current_from_volume_reference(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        points = tuple(SwingPoint('high' if i%2 == 0 else 'low', 1.1+.01*(i%2)+.001*i,
                       start+timedelta(hours=i), start+timedelta(hours=i), start+timedelta(hours=i+2),
                       100. if i < 15 else 200.) for i in range(16))
        features = antecedent_features(points, 3600)
        self.assertEqual(features['volume_ratio'], 2.)
        self.assertEqual(features['opposite_volume_ratio'], 2.)
        self.assertTrue(all(k not in features for k in ('pnl_r', 'trailing_steps', 'mfe_r', 'mae_r', 'entry_price')))
        self.assertEqual(features['side'], 1)
        self.assertEqual(features['pivot_delay_bars'], 2)
        self.assertEqual(features['span_bars'], 15)

    def test_phase_split_censors_unknown_and_future_outcomes(self):
        data = synthetic_data()[:720].copy()
        protocol = replace(StatisticalProtocol(), discovery_start='2015-01-01', discovery_end='2015-12-31',
                           validation_start='2016-01-01', validation_end='2016-12-31',
                           test_start='2017-01-01', test_end='2017-12-31')
        data[0, COL['closed_at']] = stamp('2016-01-01')
        data[1, COL['closed']] = 0
        data[2, COL['pnl_r']] = float('nan')
        selected = phase_data(data, protocol, 'discovery')
        self.assertEqual(len(selected), len(data)-3)
        self.assertTrue((selected[:, COL['closed_at']] < stamp('2016-01-01')).all())
        self.assertEqual(stamp('2020-01-01T01:00:00+01:00'), stamp('2020-01-01'))

    def test_rule_thresholds_use_discovery_only(self):
        data = synthetic_data()
        protocol = StatisticalProtocol()
        before = make_terms(phase_data(data, protocol, 'discovery'), ['A', 'B', 'C'])
        data[data[:, COL['confirmed_at']] >= stamp('2024-01-01'), COL['leg_ratio']] = 1e12
        after = make_terms(phase_data(data, protocol, 'discovery'), ['A', 'B', 'C'])
        self.assertEqual(before, after)

    def test_calendar_clustering_does_not_treat_duplicate_overlapping_trades_as_new_information(self):
        data = phase_data(synthetic_data(), StatisticalProtocol(), 'validation')
        original = cluster_test(data, StatisticalProtocol(), 'validation')
        duplicated = cluster_test(np.repeat(data, 10, axis=0), StatisticalProtocol(), 'validation')
        self.assertAlmostEqual(original[0], duplicated[0], places=10)
        self.assertAlmostEqual(original[1], duplicated[1], places=10)
        self.assertEqual(original[2], duplicated[2])
        self.assertEqual(cluster_test(data[:2], StatisticalProtocol(), 'validation')[0], 1.)

    def test_block_bootstrap_is_reproducible_and_by_adjustment_is_conservative(self):
        protocol = replace(StatisticalProtocol(), bootstrap_samples=100)
        samples = phase_data(synthetic_data(), protocol, 'validation')
        first = block_bootstrap(samples, protocol, 'validation')
        self.assertEqual(first, block_bootstrap(samples, protocol, 'validation'))
        self.assertEqual(first['block_days'], 28)
        self.assertLess(first['mean_r_ci95'][0], first['mean_r_ci95'][1])
        records = [{'cluster_p_one_sided': p} for p in (.0001, .01, .1, .5)]
        adjust_fdr(records)
        self.assertTrue(all(r['fdr_by'] >= r['cluster_p_one_sided'] for r in records))

    def test_final_outcomes_cannot_change_which_rule_is_frozen(self):
        data = synthetic_data()
        metadata = {'run_id': 'synthetic', 'symbols': ['A', 'B', 'C'],
                    'opportunity_config': {'interval_seconds': 3600}, 'snapshot_sha256': 'synthetic'}
        protocol = replace(StatisticalProtocol(), min_discovery=300, min_validation=100, min_test=100, bootstrap_samples=100)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            first = research(data, metadata, Path(tmp)/'good', protocol)
            self.assertTrue(first['test_evaluated'])
            self.assertIsNotNone(first['selected_rule'])
            bad = data.copy()
            bad[bad[:, COL['confirmed_at']] >= stamp('2024-01-01'), COL['pnl_r']] = -1.
            second = research(bad, metadata, Path(tmp)/'bad', protocol)
            self.assertEqual(first['selected_rule'], second['selected_rule'])
            self.assertEqual(first['selected_validation'], second['selected_validation'])
            self.assertFalse(second['edge_confirmed'])
            frozen = json.loads((Path(tmp)/'bad/frozen_rule.json').read_text())
            self.assertFalse(frozen['test_used_for_selection'])
            self.assertNotIn('test', frozen)

    def test_no_discovery_rule_leaves_reserved_test_unevaluated_and_result_cannot_be_overwritten(self):
        data = synthetic_data()
        data[:, COL['pnl_r']] = -1.
        metadata = {'run_id': 'synthetic', 'symbols': ['A', 'B', 'C'],
                    'opportunity_config': {'interval_seconds': 3600}}
        def phase_guard(data, protocol, phase):
            self.assertNotEqual(phase, 'test', 'Reserved test must not be evaluated without a selected rule')
            return phase_data(data, protocol, phase)
        protocol = replace(StatisticalProtocol(), bootstrap_samples=50)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            with patch('ultimateBot.backtester.research.swing_statistics.phase_data', side_effect=phase_guard):
                result = research(data, metadata, tmp, protocol)
            self.assertEqual(result['discovery_qualified'], 0)
            self.assertIsNone(result['selected_rule'])
            self.assertFalse(result['test_evaluated'])
            self.assertFalse(result['edge_confirmed'])
            with self.assertRaisesRegex(ValueError, 'Do not overwrite'):
                research(data, metadata, tmp, protocol)


if __name__ == '__main__':
    unittest.main()
