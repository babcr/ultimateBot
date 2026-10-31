"""Direct dense candle grids: price / mean(actual window swings) - 1, holes=0."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import csv
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np
from ultimateBot.backtester.research.swing_selection import StopSelection, acceptance_mask

from ultimateBot.backtester.research.swing_opportunities import (
    OPPORTUNITY_SCHEMA, OpportunityConfig, json_dump, variant_groups, uses_stop_distance, stop_distance_feature,
)
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, SWING_SCHEMA, SwingPoint, SwingSeries, SwingWindow

GRID_SCHEMA = 'confirmed_candle_grid_swing_mean_ratio_minus_one_zero_holes_v1'
CANDLE_CLOSE_GRID_SCHEMA = 'confirmed_candle_grid_candle_close_mid_mean_ratio_minus_one_zero_holes_v1'
GRID_NORMALIZATIONS = ('swing-mean', 'candle-close-mean')


def grid_schema(normalization, config=None):
    if normalization not in GRID_NORMALIZATIONS:
        raise ValueError('Unknown swing grid normalization')
    base = GRID_SCHEMA if normalization == 'swing-mean' else CANDLE_CLOSE_GRID_SCHEMA
    return base+'_stop_buffer_ratio_v2' if uses_stop_distance(config or {}) else base


def normalization_description(normalization):
    grid_schema(normalization)
    return ('price / arithmetic_mean(actual_window_swing_prices) - 1; empty_candles=0'
            if normalization == 'swing-mean' else
            'price / arithmetic_mean((bid_close + ask_close) / 2 of all closed window candles) - 1; empty_candles=0')


def grid_width(window, interval_seconds):
    if (window.mode != 'duration' or type(interval_seconds) is not int or interval_seconds <= 0
            or window.value % interval_seconds):
        raise ValueError('Swing grid requires a duration that is an exact multiple of the collection candle width')
    return window.value//interval_seconds


def grid_input_width(window, interval_seconds, config=None):
    return grid_width(window, interval_seconds)+int(uses_stop_distance(config or {}))


def append_stop_input(vector, config, symbol, pivot_price, buffer_pips=None):
    if not uses_stop_distance(config):
        return vector
    return np.r_[vector, np.float32(stop_distance_feature(config, symbol, pivot_price, buffer_pips))].astype(np.float32)


def grid_vector(points, *, as_of, window, interval_seconds, normalization_mean=None):
    """Oldest candle first; final slot is the current confirmed swing. No labels."""
    width = grid_width(window, interval_seconds)
    points = tuple(points)
    if not points or as_of.tzinfo is None or points[-1].confirmed_at != as_of:
        raise ValueError('A grid must end at its last confirmed swing with an explicit timezone')
    step = timedelta(seconds=interval_seconds)
    left = as_of-timedelta(seconds=window.value)
    previous = None
    slots = []
    for point in points:
        if (point.kind not in ('high', 'low') or not math.isfinite(point.price) or point.price <= 0
                or not point.pair_start <= point.pivot_at < point.confirmed_at
                or not left < point.confirmed_at <= as_of):
            raise ValueError('Invalid or out-of-window swing')
        if previous and (previous.kind == point.kind or previous.confirmed_at >= point.confirmed_at
                         or previous.pivot_at > point.pivot_at):
            raise ValueError('Grid swings must alternate and follow point-in-time chronology')
        delay = as_of-point.confirmed_at
        if delay % step:
            raise ValueError('Swing confirmations must align with the collection candle width')
        slots.append(width-1-delay//step)
        previous = point
    # Dividing before summation avoids overflowing a mean of finite prices.
    mean = (math.fsum(point.price/len(points) for point in points)
            if normalization_mean is None else float(normalization_mean))
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError('Swing window mean must be finite and positive')
    result = np.zeros(width, dtype=np.float32)
    result[slots] = np.asarray([point.price/mean-1 for point in points], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized swing grid')
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_only(database):
    db = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.execute('BEGIN')
    return db


def collection(db, run_id, selection):
    """Duration grids can be built even if that duration has no stored context blobs."""
    query, args = "SELECT * FROM swing_runs WHERE status='completed'", []
    if run_id is not None:
        query += ' AND run_id=?'
        args.append(run_id)
    if selection.window_mode == 'pairs' or selection.pairs is not None:
        raise ValueError('Swing grid requires duration windows; couple-count windows have no fixed duration')
    if selection.include_volume is True:
        raise ValueError('Swing grid uses prices only; omit --swing-volume')
    for row in db.execute(query+' ORDER BY started_at DESC', args):
        params = json.loads(row['parameters_json'])
        if (params.get('swing_schema') != SWING_SCHEMA or params.get('opportunity_schema') != OPPORTUNITY_SCHEMA
                or params.get('risk_management_filters') is not False):
            if run_id is not None:
                raise ValueError('Require a current unrestricted swing collection')
            continue
        if selection.interval_seconds is not None and selection.interval_seconds != params['config']['interval_seconds']:
            if run_id is not None:
                raise ValueError('Candle width must match the collection')
            continue
        symbols = list(selection.symbols or sorted(params['symbols']))
        if not set(symbols) <= set(params['symbols']):
            if run_id is not None:
                raise ValueError('Requested symbols are unavailable in this collection')
            continue
        durations = selection.durations or tuple(w['value'] for w in params['windows'] if w['mode'] == 'duration')
        windows = tuple(dict.fromkeys(SwingWindow('duration', n) for n in durations))
        if not windows:
            raise ValueError('Select at least one duration for swing grids')
        for window in windows:
            grid_width(window, params['config']['interval_seconds'])
        return dict(row), params, windows, symbols
    raise ValueError('No completed collection matches the grid selectors')


class GridData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        for name, digest in self.manifest['files'].items():
            if _sha(self.directory/name) != digest:
                raise ValueError('Swing grid cache checksum mismatch; use a new cache directory')
        with gzip.open(self.directory/'rows.json.gz', 'rt', encoding='utf-8') as stream:
            self.rows = json.load(stream)
        self.matrices, self.geometry = {}, {}
        for key, width in self.manifest['dimensions'].items():
            self.matrices[key] = np.memmap(self.directory/(key+'.bin'), mode='r', dtype='<f4', shape=(len(self.rows), width))
            self.geometry[key] = np.memmap(self.directory/(key+'.geometry.bin'), mode='r', dtype='<f8', shape=(len(self.rows), 2))

    def close(self):
        for array in [*self.matrices.values(), *self.geometry.values()]:
            array._mmap.close()


def prepare(database, *, run_id, windows, symbols, periods, target, cache_root, grid_normalization='swing-mean'):
    """Replay scalar events once; reuse immutable on-disk dense arrays across seeds."""
    from ultimateBot.backtester.research.swing_resolver import target_label, validate_regression_labels, is_binary_target, day_start
    from ultimateBot.backtester.research.swing_resolver import TrainingSelection
    db = _read_only(database)
    try:
        run, params, checked, symbols = collection(db, run_id, TrainingSelection(symbols=tuple(symbols) if symbols else None,
            window_mode='duration', durations=tuple(w.value for w in windows)))
        windows = checked
        interval = params['config']['interval_seconds']
        request = dict(grid_schema=grid_schema(grid_normalization, params['config']), run_id=run['run_id'], source_fingerprint=params['source_fingerprint'],
            opportunity_config=params['config'], symbols=symbols, windows=[asdict(w) for w in windows], periods=periods, target=target)
        identity = hashlib.sha256(json_dump(request).encode()).hexdigest()
        cache_root = Path(cache_root).resolve()
        directory = cache_root/identity
        if (directory/'manifest.json').exists():
            print(f'[swing-grid-cache] reuse={directory}', flush=True)
            return GridData(directory), params
        temporary = cache_root/('.building_'+uuid4().hex)
        temporary.mkdir(parents=True, exist_ok=False)
        rows, counts = [], dict(raw_events=0, ambiguous=0, not_ready=0, unknown_label=0, future_outcome=0)
        digest = hashlib.sha256()
        maximum = max(w.value for w in windows)
        origin = (day_start(params['start']) if params.get('start') else datetime.fromisoformat(db.execute(
            'SELECT min(pair_start) FROM swing_events WHERE run_id=?', (run['run_id'],)).fetchone()[0]))
        with ExitStack() as streams:
            vectors = {w.key: streams.enter_context((temporary/(w.key+'.bin')).open('wb')) for w in windows}
            geometry = {w.key: streams.enter_context((temporary/(w.key+'.geometry.bin')).open('wb')) for w in windows}
            previous_end = None
            for phase, (start, end) in periods.items():
                start, end = day_start(start), day_start(end)+timedelta(days=1)
                if start >= end or (previous_end is not None and start < previous_end):
                    raise ValueError('Grid periods must be chronological and disjoint')
                previous_end = end
                warm = max(origin, start-timedelta(seconds=maximum+interval))
                series = {s: SwingSeries(timedelta(seconds=interval), windows) for s in symbols}
                for value in series.values():
                    value.observed_since = warm
                prices = None
                if grid_normalization == 'candle-close-mean':
                    from ultimateBot.backtester.research.swing_candle_prices import CandleCloseIndex
                    prices = CandleCloseIndex(params, symbols=symbols, start=warm, before=end,
                        interval_seconds=interval, cache_root=Path(database).resolve().parent/'swing_candle_close_cache')
                cursor = db.execute(f'''SELECT *,coalesce(pivot_price,stop_price) AS price FROM swing_events
                    WHERE run_id=? AND symbol IN ({','.join('?' for _ in symbols)})
                    AND confirmed_at>=? AND confirmed_at<? ORDER BY confirmed_at,symbol,event_id''',
                    (run['run_id'], *symbols, warm.isoformat(), end.isoformat()))
                for (_, symbol), group in groupby(cursor, key=lambda r: (r['confirmed_at'], r['symbol'])):
                    found = list(group)
                    counts['raw_events'] += len(found)
                    groups = variant_groups(found)
                    points = tuple(SwingPoint(g[0]['kind'], g[0]['price'], datetime.fromisoformat(g[0]['pivot_at']),
                        datetime.fromisoformat(g[0]['pair_start']), datetime.fromisoformat(g[0]['confirmed_at'])) for g in groups)
                    swing = series[symbol].append_confirmed(points)
                    if swing is None:
                        counts['ambiguous'] += len(found)
                        continue
                    if swing.confirmed_at < start:
                        continue
                    contexts = {w.key: series[symbol].context(w, swing.confirmed_at) for w in windows}
                    shared_arrays = {}
                    for row in groups[0]:
                        if row['status'] != 'closed' or row['pnl_r'] is None or not math.isfinite(row['pnl_r']):
                            counts['unknown_label'] += 1
                            continue
                        if not row['closed_at'] or datetime.fromisoformat(row['closed_at']) >= end:
                            counts['future_outcome'] += 1
                            continue
                        label = target_label(row, target)
                        if label is None:
                            counts['unknown_label'] += 1
                            continue
                        if not is_binary_target(target):
                            validate_regression_labels([label], target)
                        if any(value is None for value in contexts.values()):
                            counts['not_ready'] += 1
                            continue
                        for window in windows:
                            context = contexts[window.key]
                            if window.key not in shared_arrays:
                                mean = prices.mean(symbol, as_of=swing.confirmed_at, window=window) if prices is not None else None
                                shared_arrays[window.key] = grid_vector(context, as_of=swing.confirmed_at, window=window,
                                    interval_seconds=interval, normalization_mean=mean)
                            array = append_stop_input(shared_arrays[window.key], params['config'], symbol,
                                                      row['price'], row.get('stop_buffer_pips'))
                            vectors[window.key].write(array.astype('<f4', copy=False).tobytes())
                            geometry[window.key].write(np.asarray([len(context),
                                (context[-1].confirmed_at-context[0].confirmed_at).total_seconds()], dtype='<f8').tobytes())
                            digest.update(array.tobytes())
                        simple = {k: row[k] for k in ('event_id', 'symbol', 'side', 'confirmed_at', 'closed_at', 'pnl_r', 'trailing_steps')}
                        if uses_stop_distance(params['config']):
                            simple.update(swing_id=row['swing_id'], variant_index=row['variant_index'], stop_buffer_pips=row['stop_buffer_pips'])
                        simple['phase'] = phase
                        rows.append(simple)
                        digest.update(json_dump(simple).encode())
                        if len(rows) % 20000 == 0:
                            print(f'[swing-grid-data] raw={counts["raw_events"]} common_labeled={len(rows)}', flush=True)
        if not rows:
            raise ValueError('No labeled opportunities have every requested grid context')
        with gzip.open(temporary/'rows.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(rows, stream, allow_nan=False)
        manifest = dict(request=request, dataset_sha256=digest.hexdigest(), counts=counts, rows=len(rows),
            dimensions={w.key: grid_input_width(w, interval, params['config']) for w in windows},
            files={path.name: _sha(path) for path in temporary.iterdir()})
        (temporary/'manifest.json').write_text(json_dump(manifest), encoding='utf-8')
        temporary.rename(directory)
        print(f'[swing-grid-cache] saved={directory} rows={len(rows)}', flush=True)
        return GridData(directory), params
    finally:
        db.close()


def train_grid(database, *, run_id, output_dir, config, encoder_config, protocol, selection, grid_normalization='swing-mean'):
    from ultimateBot.backtester.research.swing_resolver import (
        purged_split, target_label, target_definition, fit_xgb, predict, learning_diagnostics,
        choose_threshold, evaluate, write_comparison, validation_score, validation_has_gain, selection_objective,
    )
    db = _read_only(database)
    try:
        run, params, windows, symbols = collection(db, run_id, selection)
    finally:
        db.close()
    periods = dict(train=[protocol.train_start, protocol.train_end], validation=[protocol.validation_start, protocol.validation_end])
    data, params = prepare(database, run_id=run['run_id'], windows=windows, symbols=symbols, periods=periods,
                           target=config.target, cache_root=Path(output_dir)/'_swing_grid_cache', grid_normalization=grid_normalization)
    partitions, split = purged_split(data.rows, protocol, config)
    indices = {phase: np.asarray([i for i, row in enumerate(data.rows) if row['phase'] == phase]) for phase in partitions}
    scope = dict(symbols=symbols, interval_seconds=params['config']['interval_seconds'],
                 window_keys=[w.key for w in windows], opportunity_config=params['config'])
    experiment = Path(output_dir).resolve()/f"grid_{run['run_id'][:12]}_{uuid4().hex[:12]}"
    experiment.mkdir(parents=True, exist_ok=False)
    comparison = dict(run_id=run['run_id'], directory=str(experiment), database=str(Path(database).resolve()),
        representation='swing-grid', feature_schema=FEATURE_SCHEMA, dataset_sha256=data.manifest['dataset_sha256'],
        config=asdict(config), encoder_config=asdict(encoder_config), protocol=asdict(protocol), scope=scope,
        experiment_id=experiment.name, test_evaluated=False, split=split, windows={},
        common_labeled_opportunities=len(data.rows), target_definition=target_definition(config.target),
        cache_directory=str(data.directory), interpretation='Independent overlapping opportunities, not a portfolio return.')
    if grid_normalization != 'swing-mean':
        comparison['grid_normalization'] = grid_normalization
    partial = experiment/'comparison.partial.json'
    sides = np.asarray([r['side'] for r in data.rows])
    for window in windows:
        key = window.key
        directory = experiment/key
        directory.mkdir()
        dimensions = grid_input_width(window, params['config']['interval_seconds'], params['config'])
        print(f'[swing-grid-plan] window={key} features={dimensions} train={len(indices["train"])} '
              f'validation={len(indices["validation"])} encoder=none normalization={grid_normalization}', flush=True)
        simple = np.column_stack((sides, np.log1p(data.geometry[key][:, 0]),
            np.log1p(data.geometry[key][:, 1]/params['config']['interval_seconds']))).astype(np.float32)
        arrays, base = {}, {}
        for phase in partitions:
            y = np.asarray([target_label(row, config.target) for row in partitions[phase]])
            arrays[phase] = (np.asarray(data.matrices[key][indices[phase]]), y)
            base[phase] = (simple[indices[phase]], y)
        model, baseline = fit_xgb(arrays, config), fit_xgb(base, config)
        diagnostic = learning_diagnostics(model, arrays, config.target)
        baseline_diagnostic = learning_diagnostics(baseline, base, config.target)
        predictions = {phase: predict(model, value[0], config.target) for phase, value in arrays.items()}
        baseline_predictions = {phase: predict(baseline, value[0], config.target) for phase, value in base.items()}
        threshold, candidates = choose_threshold(partitions['validation'], predictions['validation'], config, params['config'])
        baseline_threshold, _ = choose_threshold(partitions['validation'], baseline_predictions['validation'], config, params['config'])
        report = dict(representation='swing-grid', grid_schema=grid_schema(grid_normalization, params['config']), input_dims=dimensions,
            grid_normalization=grid_normalization,
            threshold=threshold, baseline_threshold=baseline_threshold, threshold_candidates_validation_only=candidates,
            learning_diagnostics=diagnostic, tabular_learning_diagnostics=baseline_diagnostic,
            target_definition=target_definition(config.target),
            partitions={phase: evaluate(rows, predictions[phase], threshold, config.target, params['config']) for phase, rows in partitions.items()},
            tabular_only={phase: evaluate(rows, baseline_predictions[phase], baseline_threshold, config.target, params['config'])
                          for phase, rows in partitions.items()})
        report['validation_qualified'] = bool(diagnostic['validation_predictive_gain'] and
            validation_has_gain(report['partitions']['validation'], config.target))
        model.save_model(directory/'xgboost.json')
        baseline.save_model(directory/'xgboost_tabular_only.json')
        metadata = dict(version=1, model_type='swing-grid', representation='swing-grid', grid_schema=grid_schema(grid_normalization, params['config']),
            grid_normalization=grid_normalization,
            resolver_id=uuid4().hex, run_id=run['run_id'], window_key=key, feature_schema=FEATURE_SCHEMA,
            scope={**scope, 'window_keys': [key]}, input_dims=dimensions,
            feature_names=[f'candle_{i:05d}' for i in range(grid_width(window, params['config']['interval_seconds']))]
                + (['stop_buffer_ratio'] if uses_stop_distance(params['config']) else []),
            target=config.target, threshold=threshold, available_after=protocol.bounds[3].isoformat(),
            target_definition=target_definition(config.target), validation_qualified=report['validation_qualified'],
            selection_objective=selection_objective(config.target),
            validation_predictive_gain=diagnostic['validation_predictive_gain'], protocol=asdict(protocol),
            opportunity_config=params['config'], swing_schema=SWING_SCHEMA, opportunity_schema=OPPORTUNITY_SCHEMA,
            interval_seconds=params['config']['interval_seconds'], source_fingerprint=params['source_fingerprint'],
            dataset_sha256=data.manifest['dataset_sha256'], experiment_id=experiment.name,
            normalization=normalization_description(grid_normalization),
            time_axis='confirmation; oldest first; (as_of-duration, as_of]; current swing in last slot',
            volume_used=False, extra_tabular_inputs=uses_stop_distance(params['config']),
            xgboost_sha256=_sha(directory/'xgboost.json'), baseline_xgboost_sha256=_sha(directory/'xgboost_tabular_only.json'))
        (directory/'metadata.json').write_text(json_dump(metadata), encoding='utf-8')
        (directory/'report.json').write_text(json_dump(report), encoding='utf-8')
        with (directory/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['event_id', 'symbol', 'partition', 'confirmed_at', 'closed_at',
                'pnl_r', 'trailing_steps', 'target_label', 'prediction', 'accepted', 'tabular_prediction', 'tabular_accepted'])
            writer.writeheader()
            for phase, rows in partitions.items():
                policy = StopSelection.from_rows(rows, params['config'])
                accepted = acceptance_mask(predictions[phase], threshold, policy)
                base_accepted = acceptance_mask(baseline_predictions[phase], baseline_threshold, policy)
                for i, (row, value, bv) in enumerate(zip(rows, predictions[phase], baseline_predictions[phase])):
                    writer.writerow({**{k: row[k] for k in ('event_id', 'symbol', 'confirmed_at', 'closed_at', 'pnl_r', 'trailing_steps')},
                        'partition': phase, 'target_label': target_label(row, config.target), 'prediction': float(value),
                        'accepted': int(accepted[i]), 'tabular_prediction': float(bv), 'tabular_accepted': int(base_accepted[i])})
        comparison['windows'][key] = dict(directory=str(directory), resolver_id=metadata['resolver_id'],
            encoder_train_contexts=0, embedded_contexts=len(data.rows), report=report)
        temporary = partial.with_suffix('.tmp')
        temporary.write_text(json_dump(comparison), encoding='utf-8')
        temporary.replace(partial)
        print(f'[swing-grid] {key}: validation R={report["partitions"]["validation"]["selected"]["total_r"]:.3f} '
              f'selection_metric={selection_objective(config.target)["metric"]} '
              f'selection_score={validation_score(report["partitions"]["validation"], config.target):.9g} '
              f'trees_used={diagnostic["trees_used"]} qualified={report["validation_qualified"]}', flush=True)
    selected = max(comparison['windows'], key=lambda key: (
        validation_score(comparison['windows'][key]['report']['partitions']['validation'], config.target),
        comparison['windows'][key]['report']['partitions']['validation']['selected']['coverage']))
    chosen = comparison['windows'][selected]
    comparison.update(selected_window_validation_only=selected, selected_directory=chosen['directory'],
        selection_objective=selection_objective(config.target),
        selected_validation_score=validation_score(chosen['report']['partitions']['validation'], config.target),
        selected_threshold=chosen['report']['threshold'], selected_target=config.target,
        validation_qualified=chosen['report']['validation_qualified'],
        positive_validation=chosen['report']['partitions']['validation']['selected']['total_r'] > 0)
    (experiment/'comparison.json').write_text(json_dump(comparison), encoding='utf-8')
    (experiment/'selected.json').write_text(json_dump(dict(window_key=selected, directory=chosen['directory'],
        selection_objective=comparison['selection_objective'], selected_validation_score=comparison['selected_validation_score'],
        threshold=comparison['selected_threshold'], target=config.target, model_type='swing-grid',
        validation_qualified=comparison['validation_qualified'], scope={**scope, 'window_keys': [selected]}, test_evaluated=False)), encoding='utf-8')
    write_comparison(experiment, comparison)
    data.close()
    return comparison


def train_grid_per_window(database, *, run_id, output_dir, config, encoder_config, protocol, selection, grid_normalization='swing-mean'):
    """Freeze each window on its own cohort, then compare on common validation IDs.

    Adding another duration must not alter a window's training, early stopping,
    threshold, weights or predictions. Only the comparison/ranking set changes.
    """
    from ultimateBot.backtester.research.swing_resolver import (
        evaluate, write_comparison, validation_score, validation_has_gain, selection_objective,
    )
    db = _read_only(database)
    try:
        run, _, windows, _ = collection(db, run_id, selection)
    finally:
        db.close()
    children, validation = {}, {}
    for window in windows:
        key = window.key
        children[key] = train_grid(database, run_id=run['run_id'], output_dir=output_dir, config=config,
            encoder_config=encoder_config, protocol=protocol,
            grid_normalization=grid_normalization,
            selection=replace(selection, window_mode='duration', durations=(window.value,), pairs=None))
        directory = Path(children[key]['windows'][key]['directory'])
        records = {}
        with (directory/'predictions.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['partition'] != 'validation':
                    continue
                for field in ('pnl_r', 'prediction', 'tabular_prediction'):
                    row[field] = float(row[field])
                row['trailing_steps'] = float(row['trailing_steps']) if row['trailing_steps'] else None
                records[row['event_id']] = row
        validation[key] = records
    common_ids = set.intersection(*(set(rows) for rows in validation.values()))
    if len(common_ids) < config.min_validation_samples:
        raise ValueError(f'Need {config.min_validation_samples} common validation observations for ranking; got {len(common_ids)}')
    first_key = windows[0].key
    ordered = sorted(common_ids, key=lambda event_id: (
        validation[first_key][event_id]['confirmed_at'], validation[first_key][event_id]['symbol'], event_id))
    common_hash = hashlib.sha256(json_dump(ordered).encode()).hexdigest()
    combined = {key: child['windows'][key] for key, child in children.items()}
    for key, entry in combined.items():
        records = [validation[key][event_id] for event_id in ordered]
        # Validate shared outcomes before using them to compare frozen predictions.
        if any((row['pnl_r'], row['closed_at'], row['trailing_steps']) != (
                validation[first_key][row['event_id']]['pnl_r'],
                validation[first_key][row['event_id']]['closed_at'],
                validation[first_key][row['event_id']]['trailing_steps']) for row in records):
            raise ValueError('Shared validation outcomes differ between window cohorts')
        report = entry['report']
        scores = np.asarray([row['prediction'] for row in records])
        base_scores = np.asarray([row['tabular_prediction'] for row in records])
        settings = children[key]['scope']['opportunity_config']
        report['comparison_validation'] = evaluate(records, scores, report['threshold'], config.target, settings)
        report['comparison_tabular_validation'] = evaluate(records, base_scores, report['baseline_threshold'], config.target, settings)
        report['comparison_validation']['cohort_sha256'] = common_hash
        report['comparison_validation']['threshold_reselected'] = False
        report['own_split'] = children[key]['split']['after_purge']
    selected = max(combined, key=lambda key: (
        validation_score(combined[key]['report']['comparison_validation'], config.target),
        combined[key]['report']['comparison_validation']['selected']['coverage']))
    chosen = combined[selected]
    directory = Path(output_dir).resolve()/f"grid_per_window_{run['run_id'][:12]}_{uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    source_hashes = {key: child['dataset_sha256'] for key, child in children.items()}
    comparison = {**children[first_key], 'cohort_policy': 'per-window', 'directory': str(directory),
        'experiment_id': directory.name, 'windows': combined,
        'scope': {**children[first_key]['scope'], 'window_keys': list(combined)},
        'dataset_sha256': hashlib.sha256(json_dump(dict(windows=source_hashes, common_validation=common_hash)).encode()).hexdigest(),
        'window_dataset_sha256': source_hashes, 'common_labeled_opportunities': None,
        'common_validation_opportunities': len(ordered), 'common_validation_sha256': common_hash,
        'split': {'after_purge': {key: child['split']['after_purge'] for key, child in children.items()},
                  'comparison_validation': len(ordered)},
        'selected_window_validation_only': selected, 'selected_directory': chosen['directory'],
        'selected_threshold': chosen['report']['threshold'], 'selected_target': config.target,
        'selection_objective': selection_objective(config.target),
        'selected_validation_score': validation_score(chosen['report']['comparison_validation'], config.target),
        'positive_validation': chosen['report']['comparison_validation']['selected']['total_r'] > 0,
        'validation_qualified': bool(chosen['report']['validation_qualified'] and
            validation_has_gain(chosen['report']['comparison_validation'], config.target))}
    (directory/'comparison.json').write_text(json_dump(comparison), encoding='utf-8')
    (directory/'selected.json').write_text(json_dump(dict(window_key=selected, directory=chosen['directory'],
        selection_objective=comparison['selection_objective'], selected_validation_score=comparison['selected_validation_score'],
        threshold=comparison['selected_threshold'], target=config.target, cohort_policy='per-window',
        validation_qualified=comparison['validation_qualified'], scope={**comparison['scope'], 'window_keys': [selected]},
        test_evaluated=False)), encoding='utf-8')
    write_comparison(directory, comparison)
    return comparison


class SwingGridResolver:
    def __init__(self, directory, *, threads=2, device='cpu'):
        import xgboost as xgb
        from ultimateBot.backtester.research.swing_resolver import is_binary_target, resolver_directory
        self.directory = resolver_directory(directory)
        self.metadata = json.loads((self.directory/'metadata.json').read_text(encoding='utf-8'))
        meta = self.metadata
        self.grid_normalization = meta.get('grid_normalization', 'swing-mean')
        if (meta.get('model_type') != 'swing-grid' or meta.get('grid_schema') != grid_schema(self.grid_normalization, meta['opportunity_config'])
                or meta.get('version') != 1 or meta.get('feature_schema') != FEATURE_SCHEMA):
            raise ValueError('Unsupported swing grid model')
        self.window = SwingWindow('duration', int(meta['window_key'].rsplit('_', 1)[1]))
        if grid_input_width(self.window, meta['interval_seconds'], meta['opportunity_config']) != meta['input_dims']:
            raise ValueError('Inconsistent swing grid dimensions')
        if _sha(self.directory/'xgboost.json') != meta['xgboost_sha256']:
            raise ValueError('XGBoost grid checkpoint hash mismatch')
        self.model = xgb.XGBClassifier() if is_binary_target(meta['target']) else xgb.XGBRegressor()
        self.model.load_model(self.directory/'xgboost.json')
        self.model.set_params(n_jobs=threads)
        if self.model.n_features_in_ != meta['input_dims']:
            raise ValueError('Grid model feature count mismatch')

    def predict(self, points, *, as_of, symbol=None, candles=None, normalization_mean=None, stop_buffer_pips=None):
        from ultimateBot.backtester.research.swing_resolver import predict
        points = tuple(points)
        if as_of < datetime.fromisoformat(self.metadata['available_after']):
            raise ValueError('Grid model training/validation postdates this decision')
        symbols = self.metadata['scope']['symbols']
        symbol = symbol.upper() if symbol else symbols[0] if len(symbols) == 1 else None
        if symbol not in symbols:
            raise ValueError('Specify a symbol in the frozen grid model scope')
        if self.grid_normalization == 'candle-close-mean':
            if candles is not None:
                if normalization_mean is not None:
                    raise ValueError('Provide candles or their mean, not both')
                from ultimateBot.backtester.research.swing_candle_prices import candle_close_mean
                normalization_mean = candle_close_mean(candles, as_of=as_of, window=self.window,
                    interval_seconds=self.metadata['interval_seconds'])
            if normalization_mean is None:
                raise ValueError('This frozen grid model requires its closed candle prices for normalization')
        elif candles is not None or normalization_mean is not None:
            raise ValueError('A frozen swing-mean model cannot change its normalization')
        values = grid_vector(points, as_of=as_of, window=self.window, interval_seconds=self.metadata['interval_seconds'],
            normalization_mean=normalization_mean)
        values = append_stop_input(values, self.metadata['opportunity_config'], symbol, points[-1].price, stop_buffer_pips)
        score = float(predict(self.model, values[None, :], self.metadata['target'])[0])
        if not math.isfinite(score):
            raise ValueError('Nonfinite XGBoost grid prediction')
        return dict(prediction=score, target=self.metadata['target'], threshold=self.metadata['threshold'],
            accepted=score >= self.metadata['threshold'], side=points[-1].side, resolver_id=self.metadata['resolver_id'])


def frozen_grid_test(database, *, directory, run_id=None, start='2026-01-01', end='2026-12-31', output_dir=None,
                     min_samples=30, threads=2):
    import xgboost as xgb
    from ultimateBot.backtester.research.swing_resolver import evaluate, predict, is_binary_target, day_start, target_label
    resolver = SwingGridResolver(directory, threads=threads)
    meta = resolver.metadata
    if min_samples < 1 or threads < 1:
        raise ValueError('Test sample and thread counts must be positive')
    if day_start(start) < datetime.fromisoformat(meta['available_after']) or day_start(start) > day_start(end):
        raise ValueError('Test must follow the frozen grid validation period')
    destination = Path(output_dir).resolve() if output_dir else resolver.directory.parent/'tests'
    data, params = prepare(database, run_id=run_id or meta['run_id'], windows=(resolver.window,), symbols=meta['scope']['symbols'],
        periods=dict(test=[start, end]), target=meta['target'], cache_root=destination/'_swing_grid_cache',
        grid_normalization=resolver.grid_normalization)
    original, current = [asdict(OpportunityConfig(**config)) for config in (meta['opportunity_config'], params['config'])]
    if {k: v for k, v in original.items() if k != 'risk_cash'} != {k: v for k, v in current.items() if k != 'risk_cash'}:
        raise ValueError('Frozen grid test must preserve candles, stop, expiry and trailing configuration')
    if len(data.rows) < min_samples:
        raise ValueError(f'Need {min_samples} closed test observations; got {len(data.rows)}')
    baseline_path = resolver.directory/'xgboost_tabular_only.json'
    if _sha(baseline_path) != meta['baseline_xgboost_sha256']:
        raise ValueError('Grid baseline checkpoint hash mismatch')
    baseline = xgb.XGBClassifier() if is_binary_target(meta['target']) else xgb.XGBRegressor()
    baseline.load_model(baseline_path)
    baseline.set_params(n_jobs=threads)
    training_report = json.loads((resolver.directory/'report.json').read_text(encoding='utf-8'))
    key = meta['window_key']
    simple = np.column_stack(([row['side'] for row in data.rows], np.log1p(data.geometry[key][:, 0]),
        np.log1p(data.geometry[key][:, 1]/meta['interval_seconds']))).astype(np.float32)
    scores = predict(resolver.model, data.matrices[key], meta['target'])
    base_scores = predict(baseline, simple, meta['target'])
    report = dict(resolver_id=meta['resolver_id'], run_id=data.manifest['request']['run_id'], representation='swing-grid',
        test_start=start, test_end=end, threshold=meta['threshold'],
        metrics=evaluate(data.rows, scores, meta['threshold'], meta['target'], meta['opportunity_config']),
        tabular_only=evaluate(data.rows, base_scores, training_report['baseline_threshold'], meta['target'], meta['opportunity_config']),
        fitting_performed=False, threshold_changed=False, dataset_sha256=data.manifest['dataset_sha256'],
        interpretation='Frozen overlapping opportunities, not portfolio performance.')
    result_dir = destination/('grid_'+uuid4().hex[:12])
    result_dir.mkdir(parents=True, exist_ok=False)
    report['directory'] = str(result_dir)
    (result_dir/'report.json').write_text(json_dump(report), encoding='utf-8')
    policy = StopSelection.from_rows(data.rows, meta['opportunity_config'])
    np.savez_compressed(result_dir/'predictions.npz', prediction=scores, actual=[row['pnl_r'] for row in data.rows],
                        target_label=np.asarray([target_label(row, meta['target']) for row in data.rows], dtype=np.float64),
                        event_id=np.asarray([r['event_id'] for r in data.rows]),
                        accepted=acceptance_mask(scores, meta['threshold'], policy))
    data.close()
    return report
