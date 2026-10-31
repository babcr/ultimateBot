"""Transparent, chronological rule research on independent swing opportunities.

No embeddings or supervised models. Read only scalar event columns, never the
large context blobs. A rule is frozen before its reserved-period evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations, groupby
import json
import math
from pathlib import Path
import sqlite3
from statistics import median

import numpy as np
from scipy.stats import false_discovery_control, t as student_t

from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA
from ultimateBot.backtester.research.swing_series import SWING_SCHEMA, SwingPoint, SwingSeries, SwingWindow


CONTEXT_PAIRS = 8
NUMERIC_FEATURES = ('leg_bps', 'leg_ratio', 'previous_leg_ratio', 'gap_ratio',
                    'same_kind_change', 'path_efficiency', 'span_bars',
                    'volume_ratio', 'opposite_volume_ratio')
FEATURE_COLUMNS = ('rowid', 'symbol', 'confirmed_at', 'closed_at', 'pnl_r', 'closed',
                   'side', 'hour_bucket', 'weekday', 'pivot_delay_bars', 'structure', *NUMERIC_FEATURES)
COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


def stamp(value):
    instant = datetime.fromisoformat(value)
    return (instant.replace(tzinfo=timezone.utc) if instant.tzinfo is None else instant).timestamp()


@dataclass(frozen=True)
class StatisticalProtocol:
    discovery_start: str = '2015-01-01'
    discovery_end: str = '2021-12-31'
    validation_start: str = '2022-01-01'
    validation_end: str = '2023-12-31'
    test_start: str = '2024-01-01'
    test_end: str = '2025-12-31'
    min_discovery: int = 1000
    min_validation: int = 200
    min_test: int = 200
    fdr: float = .1
    min_profit_factor: float = 1.05
    min_positive_year_fraction: float = .6
    cluster_days: int = 28
    bootstrap_samples: int = 3000
    seed: int = 1729

    def __post_init__(self):
        dates = [stamp(getattr(self, k)) for k in ('discovery_start', 'discovery_end',
                 'validation_start', 'validation_end', 'test_start', 'test_end')]
        if not dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]:
            raise ValueError('Require chronological disjoint discovery, validation and test periods')
        if min(self.min_discovery, self.min_validation, self.min_test, self.cluster_days, self.bootstrap_samples) < 1:
            raise ValueError('Counts and block lengths must be positive')
        if not 0 < self.fdr < 1 or not 0 < self.min_positive_year_fraction <= 1:
            raise ValueError('Invalid statistical qualification levels')

    def bounds(self, phase):
        return stamp(getattr(self, phase+'_start')), stamp(getattr(self, phase+'_end'))+86400


def antecedent_features(points, interval_seconds):
    """Only confirmed geometry, elapsed time and quote volumes. No outcomes."""
    if len(points) != 2*CONTEXT_PAIRS:
        raise ValueError(f'Require {2*CONTEXT_PAIRS} alternating swings')
    p = [s.price for s in points]
    legs = [abs(b-a) for a, b in zip(p, p[1:])]
    current = points[-1]
    direction = current.side
    gaps = [(b.confirmed_at-a.confirmed_at).total_seconds()/interval_seconds
            for a, b in zip(points, points[1:])]
    def divide(a, b):
        return a/b if b > 0 and math.isfinite(a) and math.isfinite(b) else float('nan')
    prior_same = [s.mean_quote_volume for s in points[:-1] if s.kind == current.kind][-5:]
    prior_opposite = [s.mean_quote_volume for s in points[:-1] if s.kind != current.kind][-5:]
    def volume_ratio(values):
        if (current.mean_quote_volume is None or not math.isfinite(current.mean_quote_volume)
                or current.mean_quote_volume < 0 or len(values) < 5
                or any(v is None or not math.isfinite(v) or v < 0 for v in values)):
            return float('nan')
        return divide(current.mean_quote_volume, median(values))
    changes = (direction*(p[-1]-p[-3]), direction*(p[-2]-p[-4]))
    structure = 1 if min(changes) > 0 else -1 if max(changes) < 0 else 0
    at = current.confirmed_at.astimezone(timezone.utc)
    return {'side': direction, 'hour_bucket': at.hour//6, 'weekday': at.weekday(),
            'pivot_delay_bars': (at-current.pivot_at).total_seconds()/interval_seconds,
            'structure': structure,
            'leg_bps': abs(math.log(p[-1]/p[-2]))*10000,
            'leg_ratio': divide(legs[-1], legs[-2]),
            'previous_leg_ratio': divide(legs[-2], legs[-3]),
            'gap_ratio': divide(gaps[-1], median(gaps[:-1])),
            'same_kind_change': divide(changes[0], legs[-2]),
            'path_efficiency': divide(abs(p[-1]-p[0]), sum(legs)),
            'span_bars': (at-points[0].confirmed_at).total_seconds()/interval_seconds,
            'volume_ratio': volume_ratio(prior_same),
            'opposite_volume_ratio': volume_ratio(prior_opposite)}


def extract_events(database, run_id, protocol, output):
    """Stream a consistent read-only snapshot; do not read the unused 2026 data."""
    database, output = Path(database).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(database.as_uri()+'?mode=ro', uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    con.execute('PRAGMA cache_size=-32768')
    con.execute('BEGIN')
    digest = hashlib.sha256()
    chunks, batch = [], []
    counts = {'raw_events': 0, 'ambiguous_events': 0, 'context_not_ready': 0}
    try:
        if run_id is None:
            run = con.execute("SELECT * FROM swing_runs WHERE status='completed' ORDER BY started_at DESC LIMIT 1").fetchone()
        else:
            run = con.execute("SELECT * FROM swing_runs WHERE run_id=? AND status='completed'", (run_id,)).fetchone()
        if run is None:
            raise ValueError('A completed collection is required')
        parameters = json.loads(run['parameters_json'])
        if parameters.get('opportunity_schema') != OPPORTUNITY_SCHEMA or parameters.get('swing_schema') != SWING_SCHEMA:
            raise ValueError('Incompatible opportunity collection')
        run_id = run['run_id']
        symbols = sorted(parameters['symbols'])
        interval = parameters['config']['interval_seconds']
        series = {s: SwingSeries(timedelta(seconds=interval), (SwingWindow('pairs', CONTEXT_PAIRS),)) for s in symbols}
        ids = {s: i for i, s in enumerate(symbols)}
        end = datetime.fromtimestamp(protocol.bounds('test')[1], timezone.utc).isoformat()
        cursor = con.execute('''SELECT rowid,event_id,symbol,kind,coalesce(pivot_price,stop_price) AS price,
            pivot_at,pair_start,confirmed_at,status,closed_at,pnl_r,mean_quote_volume FROM swing_events
            WHERE run_id=? AND confirmed_at<? ORDER BY confirmed_at,symbol,event_id''', (run_id, end))
        for key, group in groupby(cursor, key=lambda r: (r['confirmed_at'], r['symbol'])):
            rows = list(group)
            counts['raw_events'] += len(rows)
            for row in rows:
                digest.update(json.dumps(tuple(row), allow_nan=False, separators=(',', ':')).encode())
            points = tuple(SwingPoint(r['kind'], r['price'], datetime.fromisoformat(r['pivot_at']),
                datetime.fromisoformat(r['pair_start']), datetime.fromisoformat(r['confirmed_at']), r['mean_quote_volume']) for r in rows)
            current_series = series[key[1]]
            swing = current_series.append_confirmed(points)
            if swing is None:
                counts['ambiguous_events'] += len(rows)
                continue
            context = current_series.context(SwingWindow('pairs', CONTEXT_PAIRS), swing.confirmed_at)
            if context is None:
                counts['context_not_ready'] += 1
                continue
            row = rows[0]
            features = antecedent_features(context, interval)
            values = {'rowid': row['rowid'], 'symbol': ids[row['symbol']],
                      'confirmed_at': swing.confirmed_at.timestamp(),
                      'closed_at': stamp(row['closed_at']) if row['closed_at'] else float('nan'),
                      'pnl_r': row['pnl_r'] if row['pnl_r'] is not None else float('nan'),
                      'closed': int(row['status'] == 'closed'), **features}
            batch.append(tuple(values[k] for k in FEATURE_COLUMNS))
            if len(batch) >= 10000:
                chunks.append(np.asarray(batch, dtype=np.float64))
                batch = []
                print(f"[swing-statistics] raw={counts['raw_events']} usable={sum(len(a) for a in chunks)}", flush=True)
        if batch:
            chunks.append(np.asarray(batch, dtype=np.float64))
        if not chunks:
            raise ValueError('No usable swing contexts')
        data = np.concatenate(chunks)
        metadata = {'run_id': run_id, 'database': str(database), 'symbols': symbols,
                    'opportunity_config': parameters['config'], 'source_fingerprint': parameters['source_fingerprint'],
                    'snapshot_sha256': digest.hexdigest(), 'counts': counts, 'usable_contexts': len(data),
                    'columns': FEATURE_COLUMNS, 'context_pairs': CONTEXT_PAIRS,
                    'read_before': end, 'read_only': True, 'extractor_version': 2,
                    'feature_schema': parameters['feature_schema']}
        np.savez_compressed(output/'features.npz', data=data, metadata=json.dumps(metadata))
        return data, metadata
    finally:
        con.close()


def phase_data(data, protocol, phase):
    start, end = protocol.bounds(phase)
    signals = (data[:, COL['confirmed_at']] >= start) & (data[:, COL['confirmed_at']] < end)
    known = (data[:, COL['closed']] == 1) & np.isfinite(data[:, COL['pnl_r']]) & (data[:, COL['closed_at']] < end)
    return data[signals & known]


def make_terms(discovery, symbols):
    terms = []
    for feature in NUMERIC_FEATURES:
        values = discovery[:, COL[feature]]
        values = values[np.isfinite(values)]
        if len(values) < 100:
            continue
        cuts = np.unique(np.quantile(values, [1/3, 2/3]))
        edges = [None, *[float(v) for v in cuts], None]
        for lo, hi in zip(edges, edges[1:]):
            terms.append({'feature': feature, 'op': 'range', 'lo': lo, 'hi': hi})
    for feature, values in (('symbol', range(len(symbols))), ('side', (-1, 1)),
                             ('hour_bucket', range(4)), ('weekday', range(5)), ('structure', (-1, 0, 1)),
                             ('pivot_delay_bars', sorted(set(discovery[:, COL['pivot_delay_bars']])))):
        for value in values:
            terms.append({'feature': feature, 'op': 'equal', 'value': float(value)})
    return terms


def term_mask(data, term):
    values = data[:, COL[term['feature']]]
    result = np.isfinite(values)
    if term['op'] == 'equal':
        return result & (values == term['value'])
    if term['lo'] is not None:
        result &= values >= term['lo']
    if term['hi'] is not None:
        result &= values < term['hi']
    return result


def rule_mask(data, rule):
    result = np.ones(len(data), dtype=bool)
    for term in rule['terms']:
        result &= term_mask(data, term)
    return result


def describe_term(term, symbols):
    feature = term['feature']
    if term['op'] == 'equal':
        value = term['value']
        if feature == 'symbol':
            return 'symbol='+symbols[int(value)]
        if feature == 'side':
            return 'side='+('buy' if value == 1 else 'sell')
        if feature == 'hour_bucket':
            return f'UTC hour in [{int(value)*6:02d}:00,{(int(value)+1)*6:02d}:00)'
        if feature == 'weekday':
            return 'weekday='+('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')[int(value)]
        return f'{feature}={value:g}'
    lo, hi = term['lo'], term['hi']
    return ((f'{feature} < {hi:.8g}' if lo is None else f'{feature} >= {lo:.8g}') if hi is None or lo is None
            else f'{lo:.8g} <= {feature} < {hi:.8g}')


def cluster_test(data, protocol, phase):
    """One-sided zero-expectancy test with all symbols clustered in 28-day blocks.

    Approximate Student inference on block influences, not an iid trade test.
    Repeating same-block trades cannot manufacture independent observations.
    """
    if not len(data):
        return 1., None, 0
    start, end = protocol.bounds(phase)
    slots = math.ceil((end-start)/(86400*protocol.cluster_days))
    index = ((data[:, COL['confirmed_at']]-start)//(86400*protocol.cluster_days)).astype(int)
    sums = np.bincount(index, weights=data[:, COL['pnl_r']], minlength=slots)
    counts = np.bincount(index, minlength=slots)
    active = int((counts > 0).sum())
    if active < 12:
        return 1., None, active
    mean = sums.sum()/counts.sum()
    variance = active/(active-1)*np.sum((sums-mean*counts)**2)/(counts.sum()**2)
    se = math.sqrt(variance)
    p = float(student_t.sf(mean/se, df=active-1)) if se else (0. if mean > 0 else 1.)
    return p, se, active


def summarize(data, protocol, phase):
    r = data[:, COL['pnl_r']]
    profit, loss = float(np.maximum(r, 0).sum()), float(-np.minimum(r, 0).sum())
    years = {}
    for year in range(int(getattr(protocol, phase+'_start')[:4]), int(getattr(protocol, phase+'_end')[:4])+1):
        selected = (data[:, COL['confirmed_at']] >= stamp(f'{year}-01-01')) & (data[:, COL['confirmed_at']] < stamp(f'{year+1}-01-01'))
        yr = r[selected]
        years[str(year)] = {'count': len(yr), 'total_r': float(yr.sum())}
    p, se, blocks = cluster_test(data, protocol, phase)
    return {'count': len(r), 'total_r': float(r.sum()), 'mean_r': float(r.mean()) if len(r) else None,
            'profit_factor': profit/loss if loss else None, 'gross_profit_r': profit, 'gross_loss_r': loss,
            'win_rate': float((r > 0).mean()) if len(r) else None, 'years': years,
            'positive_year_fraction': sum(v['total_r'] > 0 for v in years.values())/len(years),
            'cluster_p_one_sided': p, 'cluster_standard_error': se, 'active_28day_blocks': blocks}


def block_bootstrap(data, protocol, phase, *, block_days=None):
    """Moving calendar blocks: resample counts and sums jointly across all pairs."""
    if not len(data):
        return None
    start, end = protocol.bounds(phase)
    days = math.ceil((end-start)/86400)
    block_days = block_days or protocol.cluster_days
    block_days = min(block_days, days)
    day = ((data[:, COL['confirmed_at']]-start)//86400).astype(int)
    sums = np.bincount(day, weights=data[:, COL['pnl_r']], minlength=days)
    counts = np.bincount(day, minlength=days)
    rng = np.random.default_rng(protocol.seed)
    starts = rng.integers(0, days-block_days+1, size=(protocol.bootstrap_samples, math.ceil(days/block_days)))
    slots = (starts[:, :, None]+np.arange(block_days)).reshape(protocol.bootstrap_samples, -1)[:, :days]
    totals = sums[slots].sum(axis=1)
    n = counts[slots].sum(axis=1)
    values = totals[n > 0]/n[n > 0]
    if not len(values):
        return None
    lower, upper = np.quantile(values, [.025, .975])
    return {'block_days': block_days, 'resamples': len(values), 'mean_r_ci95': [float(lower), float(upper)],
            'bootstrap_fraction_nonpositive': float((values <= 0).mean()),
            'interpretation': 'Dependent-event calendar-block percentile interval; not a guarantee or portfolio return.'}


def adjust_fdr(records, metric='cluster_p_one_sided', output='fdr_by'):
    if records:
        q = false_discovery_control([r[metric] for r in records], method='by')
        for record, value in zip(records, q):
            record[output] = float(value)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def research(data, metadata, output, protocol=StatisticalProtocol()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'frozen_rule.json').exists():
        raise ValueError('Do not overwrite a frozen research result; use a new output directory')
    _write_json(output/'protocol.json', {'protocol': asdict(protocol), 'source': metadata,
        'candidate_family': 'all single predicates and all pairs on distinct features; discovery tertiles only',
        'multiple_testing': 'Benjamini-Yekutieli FDR (dependent rules); approximate calendar-cluster p-values',
        'selection': 'discovery BY<=fdr, mean>0, PF>=min_profit_factor, positive-year fraction>=min_positive_year_fraction; validation BY<=fdr, all years positive, CI95 lower>0; rank validation CI lower then mean',
        'test_policy': 'At most one selected rule; never choose or alter a rule using test outcomes; no events beyond test_end are extracted'})
    discovery = phase_data(data, protocol, 'discovery')
    validation = phase_data(data, protocol, 'validation')
    if len(discovery) < protocol.min_discovery or len(validation) < protocol.min_validation:
        raise ValueError('Insufficient discovery or validation observations')
    terms = make_terms(discovery, metadata['symbols'])
    specs = [(i,) for i in range(len(terms))]
    specs += [(i, j) for i, j in combinations(range(len(terms)), 2) if terms[i]['feature'] != terms[j]['feature']]
    masks = [term_mask(discovery, term) for term in terms]
    records = []
    for index, spec in enumerate(specs):
        chosen = masks[spec[0]].copy()
        if len(spec) == 2:
            chosen &= masks[spec[1]]
        rule = {'id': index, 'terms': [terms[i] for i in spec],
                'description': ' AND '.join(describe_term(terms[i], metadata['symbols']) for i in spec)}
        stats = summarize(discovery[chosen], protocol, 'discovery')
        if stats['count'] < protocol.min_discovery:
            stats['cluster_p_one_sided'] = 1.
        records.append({**rule, **stats})
    adjust_fdr(records)
    eligible = [r for r in records if r['count'] >= protocol.min_discovery and r['mean_r'] > 0
                and r['profit_factor'] is not None and r['profit_factor'] >= protocol.min_profit_factor
                and r['positive_year_fraction'] >= protocol.min_positive_year_fraction and r['fdr_by'] <= protocol.fdr]
    _write_json(output/'discovery_rules.json', records)
    print(f'[swing-statistics] hypotheses={len(records)} discovery_qualified={len(eligible)}', flush=True)
    validated = []
    for rule in eligible:
        samples = validation[rule_mask(validation, rule)]
        stats = summarize(samples, protocol, 'validation')
        if stats['count'] < protocol.min_validation:
            stats['cluster_p_one_sided'] = 1.
        validated.append({'id': rule['id'], 'description': rule['description'], **stats})
    adjust_fdr(validated)
    finalists = []
    for record in validated:
        if (record['count'] < protocol.min_validation or record['mean_r'] <= 0 or record['fdr_by'] > protocol.fdr
                or record['profit_factor'] is None or record['profit_factor'] < protocol.min_profit_factor
                or record['positive_year_fraction'] < 1):
            continue
        rule = records[record['id']]
        record['bootstrap'] = block_bootstrap(validation[rule_mask(validation, rule)], protocol, 'validation')
        if record['bootstrap'] and record['bootstrap']['mean_r_ci95'][0] > 0:
            finalists.append(record)
    _write_json(output/'validation_rules.json', validated)
    finalists.sort(key=lambda r: (-r['bootstrap']['mean_r_ci95'][0], -r['mean_r'], r['id']))
    selected = records[finalists[0]['id']] if finalists else None
    frozen = {'selected_rule': selected, 'selected_validation': finalists[0] if finalists else None,
              'available_after': datetime.fromtimestamp(protocol.bounds('validation')[1], timezone.utc).isoformat(),
              'source': metadata, 'protocol': asdict(protocol), 'test_used_for_selection': False,
              'no_selection_reason': None if selected else 'No rule passed the predeclared statistical requirements'}
    # This file exists BEFORE any held-out outcome is evaluated.
    _write_json(output/'frozen_rule.json', frozen)
    report = {'protocol': asdict(protocol), 'source': metadata, 'hypotheses': len(records),
              'discovery_qualified': len(eligible), 'validation_qualified': len(finalists),
              'discovery_baseline': summarize(discovery, protocol, 'discovery'),
              'validation_baseline': summarize(validation, protocol, 'validation'),
              'selected_rule': selected, 'selected_validation': finalists[0] if finalists else None,
              'test_evaluated': False, 'edge_confirmed': False,
              'interpretation': 'Independent overlapping opportunities at collected exits, not a portfolio backtest.'}
    # A discovery-only diagnostic, never a substitute for a selected strategy.
    largest = max((r for r in records if r['count'] >= protocol.min_discovery),
                  key=lambda r: r['mean_r'], default=None)
    report['positive_discovery_rules'] = sum(r['mean_r'] is not None and r['mean_r'] > 0 for r in records)
    if largest:
        report['exploratory_best'] = {**largest, 'bootstrap': block_bootstrap(
            discovery[rule_mask(discovery, largest)], protocol, 'discovery'),
            'selection_warning': 'Largest observed discovery mean among many trials; this is not a validated rule.'}
    if selected:
        test = phase_data(data, protocol, 'test')
        samples = test[rule_mask(test, selected)]
        result = summarize(samples, protocol, 'test')
        result['bootstrap'] = block_bootstrap(samples, protocol, 'test')
        result['bootstrap_14day'] = block_bootstrap(samples, protocol, 'test', block_days=14)
        result['bootstrap_56day'] = block_bootstrap(samples, protocol, 'test', block_days=56)
        result['per_symbol'] = {symbol: summarize(samples[samples[:, COL['symbol']] == index], protocol, 'test')
                                for index, symbol in enumerate(metadata['symbols'])}
        positive_bounds = all(result[k] and result[k]['mean_r_ci95'][0] > 0 for k in ('bootstrap', 'bootstrap_14day', 'bootstrap_56day'))
        report.update(test_evaluated=True, test=result, test_baseline=summarize(test, protocol, 'test'),
                      edge_confirmed=bool(result['count'] >= protocol.min_test and positive_bounds
                         and result['positive_year_fraction'] == 1 and result['profit_factor'] is not None
                         and result['profit_factor'] >= protocol.min_profit_factor))
    _write_json(output/'report.json', report)
    write_report(output, report, records)
    return report


def write_report(output, report, records):
    import csv
    top = sorted(records, key=lambda r: (r['fdr_by'], -(r['mean_r'] if r['mean_r'] is not None else -1e10)))[:40]
    columns = ('id', 'description', 'count', 'mean_r', 'total_r', 'profit_factor', 'win_rate',
               'positive_year_fraction', 'cluster_p_one_sided', 'fdr_by')
    with (output/'discovery_leaderboard.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(top)
    lines = ['# Recherche statistique des antecedents de swings', '',
             f"Collecte : `{report['source']['run_id']}`. Bougies : {report['source']['opportunity_config']['interval_seconds']/60:g} minutes.",
             f"Hypotheses : {report['hypotheses']}. Retenues en decouverte : {report['discovery_qualified']}. Retenues en validation : {report['validation_qualified']}.", '',
             f"Regles simplement positives en decouverte, avant controle statistique : {report['positive_discovery_rules']}.", '',
             f"Test reserve consulte : **{report['test_evaluated']}**. Avantage statistique confirme : **{report['edge_confirmed']}**.", '',
             'Les sommes de R portent sur des opportunites independantes et chevauchantes, sans capital ni contraintes de portefeuille.',
             'Les variables antecedent le signal ; PnL, MFE, MAE, trailing realise et prix de fill ne sont jamais des entrees.',
             'Les erreurs statistiques sont approximatives : blocs calendaires et correction BY limitent les faux positifs sans les eliminer.', '',
             'R est le PnL realise divise par le risque initial. Le spread bid/ask de la collecte est inclus ; commissions et slippage supplementaire ne sont pas ajoutes.',
             f"Parametres de sortie de la collecte : `{json.dumps(report['source']['opportunity_config'], ensure_ascii=False)}`.",
             f"Decouverte : {report['protocol']['discovery_start']} au {report['protocol']['discovery_end']}. Validation : {report['protocol']['validation_start']} au {report['protocol']['validation_end']}. Test reserve : {report['protocol']['test_start']} au {report['protocol']['test_end']}.", '',
             '| Periode | N | Somme R | Moyenne R | Profit factor |', '|---|---:|---:|---:|---:|']
    for name, key in (('Decouverte sans filtre', 'discovery_baseline'), ('Validation sans filtre', 'validation_baseline'),
                      ('Validation regle figee', 'selected_validation'), ('Test regle figee', 'test')):
        stats = report.get(key)
        if stats:
            pf = f"{stats['profit_factor']:.4f}" if stats['profit_factor'] is not None else 'n/a'
            mean = f"{stats['mean_r']:.6f}" if stats['mean_r'] is not None else 'n/a'
            lines.append(f"| {name} | {stats['count']} | {stats['total_r']:.3f} | {mean} | {pf} |")
    lines += ['', '## Regle figee', '', report['selected_rule']['description'] if report['selected_rule'] else
              'Aucune regle ne passe les criteres annonces. Ne pas presenter les meilleurs essais exploratoires comme une strategie validee.', '',
              'Cela ne prouve pas l\'absence de tout edge : seules des regles a une ou deux conditions, avec seuils aux tertiles, ont ete examinees. Les sorties restent celles de la collecte.', '']
    best = report.get('exploratory_best')
    if best and best['bootstrap']:
        lower, upper = best['bootstrap']['mean_r_ci95']
        lines += ['## Meilleure moyenne exploratoire', '', best['description'], '',
                  f"Moyenne : {best['mean_r']:.6f} R. Intervalle descriptif par blocs a 95 % : [{lower:.6f}, {upper:.6f}] R.",
                  'Cet intervalle ne corrige pas le choix du meilleur essai ; le q BY et les criteres de selection gouvernent la qualification.', '']
    lines += [
              '## Meilleurs essais de decouverte (non valides par ce classement)', '',
              '| Regle | N | Moyenne R | PF | q BY |', '|---|---:|---:|---:|---:|']
    for rule in top[:15]:
        mean = f"{rule['mean_r']:.5f}" if rule['mean_r'] is not None else 'n/a'
        pf = f"{rule['profit_factor']:.4f}" if rule['profit_factor'] is not None else 'n/a'
        lines.append(f"| {rule['description']} | {rule['count']} | {mean} | {pf} | {rule['fdr_by']:.5g} |")
    (output/'rapport.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
