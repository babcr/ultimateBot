"""Persist causal windows once per symbol and covered time range, without redoing fills."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from itertools import groupby
import json

from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, json_dump, variant_groups, store_context
from ultimateBot.backtester.research.swing_series import (
    FEATURE_SCHEMA, FEATURE_SCHEMAS, VOLUME_FEATURE_SCHEMA, SWING_SCHEMA, SwingPoint, SwingSeries, SwingWindow, swing_features,
)


def _utc(value):
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Context build boundaries must include a timezone')
    return value.astimezone(timezone.utc)


def _request(db, run_id, windows, start, before, symbols):
    before = _utc(before)
    run = db.execute("SELECT * FROM swing_runs WHERE run_id=? AND status='completed'", (run_id,)).fetchone()
    if run is None:
        raise ValueError('Require a completed collection before building windows')
    parameters = json.loads(run['parameters_json'])
    if (parameters.get('feature_schema') not in FEATURE_SCHEMAS or parameters.get('swing_schema') != SWING_SCHEMA
            or parameters.get('opportunity_schema') != OPPORTUNITY_SCHEMA
            or parameters.get('risk_management_filters') is not False):
        raise ValueError('Incompatible swing collection for window reconstruction')
    origin = datetime.fromisoformat(parameters.get('start', '0001-01-01')).replace(tzinfo=timezone.utc)
    start = origin if start is None else max(origin, _utc(start))
    if start >= before:
        raise ValueError('Context build start must precede its cutoff')
    source_symbols = sorted(parameters.get('symbols') or [r[0] for r in db.execute(
        'SELECT DISTINCT symbol FROM swing_events WHERE run_id=?', (run_id,))])
    selected = source_symbols if symbols is None else sorted({s.strip().upper() for s in symbols})
    if not selected or not set(selected) <= set(source_symbols):
        raise ValueError(f'Requested symbols unavailable; collected symbols: {source_symbols}')
    return parameters, tuple(dict.fromkeys(windows)), origin, start, before, source_symbols, selected


def _merged(ranges):
    merged = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _missing(ranges, start, before):
    gaps = []
    for left, right in _merged(ranges):
        if right <= start:
            continue
        if left >= before:
            break
        if left > start:
            gaps.append((start, left))
        start = max(start, right)
        if start >= before:
            return gaps
    if start < before:
        gaps.append((start, before))
    return gaps


def _coverage(db, run_id, parameters, windows, origin, source_symbols):
    available = ({SwingWindow(**w).key for w in parameters['windows']}
                 if parameters.get('store_contexts', True) else set())
    coverage = {(w.key, symbol): [] for w in windows for symbol in source_symbols}
    tables = {r[0] for r in db.execute("""SELECT name FROM sqlite_master WHERE type='table'
        AND name IN ('swing_context_builds','swing_context_build_checkpoints','swing_context_coverage')""")}
    generated = ({r['window_key']: dict(r) for r in db.execute(
        'SELECT * FROM swing_context_builds WHERE run_id=?', (run_id,))}
        if 'swing_context_builds' in tables else {})
    source_end = (datetime.fromisoformat(parameters['end']).replace(tzinfo=timezone.utc)+timedelta(days=1)
                  if parameters.get('end') else datetime.max.replace(tzinfo=timezone.utc))
    collected = set()
    for window in windows:
        record = generated.get(window.key)
        metadata = json.loads(record['metadata_json']) if record else {}
        if window.key in available and (record is None or metadata.get('collected_with_source')):
            collected.add(window.key)
            for symbol in source_symbols:
                coverage[window.key, symbol].append((origin, source_end))
        if record is not None and metadata.get('coverage_table') != 'swing_context_coverage':
            # Legacy records cover the source prefix. New records retain only
            # the common contiguous prefix, never a scoped range's end.
            end = _utc(datetime.fromisoformat(record['completed_before']))
            for symbol in source_symbols:
                coverage[window.key, symbol].append((origin, end))
    checkpoints = (db.execute('SELECT * FROM swing_context_build_checkpoints WHERE run_id=?', (run_id,))
                   if 'swing_context_build_checkpoints' in tables else ())
    for row in checkpoints:
        key = row['window_key'], row['symbol']
        if key in coverage:
            coverage[key].append((origin, _utc(datetime.fromisoformat(row['completed_before']))))
    # A preview may open a pre-migration database read-only.
    if 'swing_context_coverage' in tables:
        for row in db.execute('SELECT * FROM swing_context_coverage WHERE run_id=?', (run_id,)):
            key = row['window_key'], row['symbol']
            if key in coverage:
                coverage[key].append((_utc(datetime.fromisoformat(row['start_at'])),
                                      _utc(datetime.fromisoformat(row['end_at']))))
    return coverage, collected


def context_build_plan(db, run_id, windows, *, before, start=None, symbols=None):
    """Read coverage metadata only; do not scan/rebuild events or create tables."""
    parameters, windows, origin, start, before, source, selected = _request(
        db, run_id, windows, start, before, symbols)
    coverage, _ = _coverage(db, run_id, parameters, windows, origin, source)
    return {'start': start.isoformat(), 'before': before.isoformat(), 'symbols': selected,
            'windows': {w.key: {'missing_ranges': {symbol: [[a.isoformat(), b.isoformat()]
                for a, b in _missing(coverage[w.key, symbol], start, before)] for symbol in selected}}
                for w in windows}}


def _warmup_start(db, run_id, symbol, windows, start, origin, interval):
    duration = timedelta(seconds=max((w.value for w in windows if w.mode == 'duration'), default=0))
    # Include boundary resets, but never materialize warmup-only contexts.
    lookback = duration+interval
    replay_start = start-lookback if start-origin >= lookback else origin
    points_needed = max((2*w.value for w in windows if w.mode == 'pairs'), default=0)
    if points_needed:
        cursor = db.execute('''SELECT DISTINCT kind,confirmed_at FROM swing_events WHERE run_id=? AND symbol=?
        AND confirmed_at<? ORDER BY confirmed_at DESC,kind DESC''', (run_id, symbol, start.isoformat()))
        boundary, previous, points = origin, None, 0
        try:
            for confirmed, batch in groupby(cursor, key=lambda r: r['confirmed_at']):
                rows = list(batch)
                boundary = _utc(datetime.fromisoformat(confirmed))
                if len(rows) > 1:
                    break
                if rows[0]['kind'] != previous:
                    points += 1
                    previous = rows[0]['kind']
                if points >= points_needed:
                    break
            else:
                boundary = origin
        finally:
            cursor.close()
        replay_start = min(replay_start, boundary)
    return replay_start


def _replay(db, run_id, symbol, windows, start, before, origin, interval, counts, *, feature_schema=FEATURE_SCHEMA):
    replay_start = _warmup_start(db, run_id, symbol, windows, start, origin, interval)
    series = SwingSeries(interval, windows, include_volume=feature_schema == VOLUME_FEATURE_SCHEMA)
    first = db.execute('''SELECT pair_start FROM swing_events WHERE run_id=? AND symbol=?
        ORDER BY confirmed_at,event_id LIMIT 1''', (run_id, symbol)).fetchone()
    if first is not None:
        series.observed_since = max(replay_start, _utc(datetime.fromisoformat(first['pair_start'])))
    print(f'[swing-windows] symbol={symbol} windows={",".join(w.key for w in windows)} '
          f'build_start={start.isoformat()} before={before.isoformat()} warmup_start={replay_start.isoformat()}', flush=True)
    cursor = db.execute('''SELECT *,
        coalesce(pivot_price,stop_price) AS price,mean_quote_volume FROM swing_events
        WHERE run_id=? AND symbol=? AND confirmed_at>=? AND confirmed_at<? ORDER BY confirmed_at,event_id''',
        (run_id, symbol, replay_start.isoformat(), before.isoformat()))
    try:
        for index, (confirmed, batch) in enumerate(groupby(cursor, key=lambda r: r['confirmed_at'])):
            groups = variant_groups(batch)
            rows = [g[0] for g in groups]
            candidates = tuple(SwingPoint(r['kind'], r['price'], datetime.fromisoformat(r['pivot_at']),
                datetime.fromisoformat(r['pair_start']), datetime.fromisoformat(r['confirmed_at']), r['mean_quote_volume']) for r in rows)
            swing = series.append_confirmed(candidates)
            if swing is not None and swing.confirmed_at >= start:
                for window in windows:
                    points = series.context(window, swing.confirmed_at)
                    if points is None:
                        continue
                    counts[window.key] += store_context(db, [r['event_id'] for r in groups[0]], window,
                        points, interval, swing.confirmed_at, feature_schema,
                        features=swing_features(points, interval, swing.confirmed_at, feature_schema=feature_schema))
            if index % 1000 == 0:
                print(f'[swing-windows] symbol={symbol} events_replayed={index+1} '
                      f'simulated_at={confirmed} before={before.isoformat()}', flush=True)
    finally:
        cursor.close()


def _prefix_end(ranges, origin):
    end = origin
    for left, right in _merged(ranges):
        if left > end:
            break
        end = max(end, right)
    return end


def ensure_windows(db, run_id, windows, *, before, start=None, symbols=None):
    """Build missing symbol/date ranges, with bounded causal warmup.

    Contexts and coverage commit together per symbol. Existing full-history
    builds/checkpoints remain valid and are reused without replaying events.
    """
    parameters, windows, origin, start, before, source, selected = _request(
        db, run_id, windows, start, before, symbols)
    coverage, collected = _coverage(db, run_id, parameters, windows, origin, source)
    gaps = {(w.key, symbol): _missing(coverage[w.key, symbol], start, before) for w in windows for symbol in selected}
    available = {SwingWindow(**w) for w in parameters['windows']}
    needed = tuple(w for w in windows if w not in available or any(gaps[w.key, s] for s in selected))
    reused = [w.key for w in windows if w not in needed]
    result = {'windows_built': [w.key for w in needed], 'reused_windows': reused, 'inserted_contexts': 0,
              'start': start.isoformat(), 'before': before.isoformat(), 'symbols': selected}
    if reused:
        print(f'[swing-windows] cache=reused windows={",".join(reused)} '
              f'start={start.isoformat()} before={before.isoformat()}', flush=True)
    if not needed:
        return result
    interval = timedelta(seconds=parameters['config']['interval_seconds'])
    counts = dict.fromkeys((w.key for w in needed), 0)
    for symbol in selected:
        segments = {}
        for window in needed:
            for bounds in gaps[window.key, symbol]:
                segments.setdefault(bounds, []).append(window)
        if not segments:
            print(f'[swing-windows] symbol={symbol} checkpoint=reused before={before.isoformat()}', flush=True)
            continue
        symbol_counts = dict.fromkeys(counts, 0)
        with db:
            for (left, right), pending in sorted(segments.items()):
                _replay(db, run_id, symbol, tuple(pending), left, right, origin, interval, symbol_counts,
                        feature_schema=parameters['feature_schema'])
                for window in pending:
                    db.execute('''INSERT OR IGNORE INTO swing_context_coverage
                        (run_id,window_key,symbol,start_at,end_at) VALUES(?,?,?,?,?)''',
                        (run_id, window.key, symbol, left.isoformat(), right.isoformat()))
            for window in needed:
                ranges = coverage[window.key, symbol]+gaps[window.key, symbol]
                prefix = _prefix_end(ranges, origin)
                if prefix > origin:
                    db.execute('''INSERT OR REPLACE INTO swing_context_build_checkpoints
                        (run_id,window_key,symbol,completed_before) VALUES(?,?,?,?)''',
                        (run_id, window.key, symbol, prefix.isoformat()))
        for window in needed:
            coverage[window.key, symbol].extend(gaps[window.key, symbol])
        for key, count in symbol_counts.items():
            counts[key] += count
        print(f'[swing-windows] symbol={symbol} checkpoint=saved '
              f'inserted_contexts={sum(symbol_counts.values())} before={before.isoformat()}', flush=True)
    with db:
        for window in needed:
            prefix = min(_prefix_end(coverage[window.key, symbol], origin) for symbol in source)
            db.execute('INSERT OR REPLACE INTO swing_context_builds VALUES(?,?,?,?)', (run_id, window.key,
                prefix.isoformat(), json_dump({'method': 'confirmed_event_replay_v2_scoped',
                    'collected_with_source': window.key in collected, 'coverage_table': 'swing_context_coverage',
                    'inserted_contexts': counts[window.key], 'labels_modified': False})))
            if window not in available:
                parameters['windows'].append(asdict(window))
        db.execute('UPDATE swing_runs SET parameters_json=? WHERE run_id=?', (json_dump(parameters), run_id))
    result.update(inserted_contexts=sum(counts.values()), counts=counts)
    return result
