"""Read-only scalar swing snapshots and compact indexed sequences for Informer."""
from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sqlite3

import numpy as np

from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, variant_groups, uses_stop_distance, stop_distance_feature
from ultimateBot.backtester.research.swing_series import (
    SWING_SCHEMA, VOLUME_FEATURE_SCHEMA, feature_names, model_feature_schema,
)

# Raw points have no realized outcomes. Contexts are integer indices into points.
PRICE, KIND, PIVOT, CONFIRMED, VOLUME, SYMBOL = range(6)
EVENT_COLUMNS = '*'


def instant(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Collected timestamps must have an explicit timezone')
    return value.astimezone(timezone.utc)


def bounds(start, end):
    a, b = (datetime.fromisoformat(v).replace(tzinfo=timezone.utc) for v in (start, end))
    if a > b:
        raise ValueError('Period start must precede its end')
    return a, b+timedelta(days=1)


def preceding(db, run_id, symbol, start, seq_len):
    """Read backward only until a complete warmup or an ambiguity reset is found."""
    cursor = db.execute(f'''SELECT {EVENT_COLUMNS} FROM swing_events WHERE run_id=?
        AND symbol=? AND confirmed_at<? ORDER BY confirmed_at DESC,event_id DESC''',
        (run_id, symbol, start.isoformat()))
    groups, count, last_kind = [], 0, None
    for _, group in groupby(cursor, key=lambda r: r['confirmed_at']):
        rows = [g[0] for g in variant_groups(group)]
        if len(rows) != 1:
            break
        groups.append(rows[0])
        if rows[0]['kind'] != last_kind:
            count += 1
            last_kind = rows[0]['kind']
        if count >= seq_len:
            break
    return reversed(groups)


def extract(database, *, run_id, periods, seq_len, symbols=None, include_volume=None):
    """No context blobs, DB writes, future outcomes, or arbitrary warmup truncation."""
    db = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    db.execute('BEGIN')
    try:
        query = "SELECT * FROM swing_runs WHERE status='completed'"
        args = ()
        if run_id is not None:
            query += ' AND run_id=?'
            args = (run_id,)
        run = db.execute(query+' ORDER BY started_at DESC LIMIT 1', args).fetchone()
        if run is None:
            raise ValueError('A completed collection is required')
        run_id = run['run_id']
        parameters = json.loads(run['parameters_json'])
        if (parameters.get('swing_schema') != SWING_SCHEMA or parameters.get('opportunity_schema') != OPPORTUNITY_SCHEMA
                or parameters.get('risk_management_filters') is not False):
            raise ValueError('Require the current unrestricted swing collection')
        schema = model_feature_schema(parameters['feature_schema'], include_volume)
        selected = sorted(set(s.upper() for s in (symbols or parameters['symbols'])))
        if not selected or not set(selected) <= set(parameters['symbols']):
            raise ValueError('Symbols must belong to the selected collection')
        phase_names = list(periods)
        limits = [bounds(*periods[name]) for name in phase_names]
        if any(left[1] > right[0] for left, right in zip(limits, limits[1:])):
            raise ValueError('Periods must be chronological and disjoint')
        start, end = limits[0][0], limits[-1][1]
        recent = {s: deque(maxlen=seq_len) for s in selected}
        points, contexts, labels, phases, stop_inputs = [], [], [], [], []
        digest = hashlib.sha256()
        counts = dict(events=0, ambiguous=0, context_not_ready=0, unknown_outcome=0, censored_outcome=0)

        def add_point(row):
            price = row['pivot_price']
            pivot, confirmed = instant(row['pivot_at']), instant(row['confirmed_at'])
            volume = row['mean_quote_volume']
            if (row['kind'] not in ('high', 'low') or price is None or not np.isfinite(price) or price <= 0
                    or not instant(row['pair_start']) <= pivot < confirmed):
                raise ValueError('Invalid collected swing geometry')
            if schema == VOLUME_FEATURE_SCHEMA and (volume is None or not np.isfinite(volume) or volume < 0):
                raise ValueError('Volume model requires finite nonnegative collected volumes')
            point = (price, 1 if row['kind'] == 'high' else -1, pivot.timestamp(), confirmed.timestamp(),
                     volume if volume is not None else 0., selected.index(row['symbol']))
            history = recent[row['symbol']]
            if history and points[history[-1]][KIND] == point[KIND]:
                history.pop()
            if history and points[history[-1]][CONFIRMED] >= point[CONFIRMED]:
                raise ValueError('Swings must follow strict confirmation chronology')
            history.append(len(points))
            points.append(point)

        # Warmup before requested dates is fetched once per pair, and only as far as needed.
        for symbol in selected:
            for row in preceding(db, run_id, symbol, start, seq_len):
                add_point(row)
        cursor = db.execute(f'''SELECT {EVENT_COLUMNS} FROM swing_events WHERE run_id=?
            AND symbol IN ({','.join('?' for _ in selected)}) AND confirmed_at>=? AND confirmed_at<?
            ORDER BY confirmed_at,symbol,event_id''', (run_id, *selected, start.isoformat(), end.isoformat()))
        for (at, symbol), group in groupby(cursor, key=lambda r: (r['confirmed_at'], r['symbol'])):
            rows = list(group)
            counts['events'] += len(rows)
            for row in rows:
                digest.update(json.dumps(tuple(row), allow_nan=False, separators=(',', ':')).encode())
            groups = variant_groups(rows)
            canonical = [g[0] for g in groups]
            if len(canonical) != 1:
                if len(canonical) != 2 or {r['kind'] for r in canonical} != {'high', 'low'}:
                    raise ValueError('Invalid duplicate swing confirmations')
                recent[symbol].clear()
                counts['ambiguous'] += len(rows)
                continue
            row = canonical[0]
            add_point(row)
            if len(recent[symbol]) < seq_len:
                counts['context_not_ready'] += 1
                continue
            confirmed = instant(at)
            phase = next((i for i, (a, b) in enumerate(limits) if a <= confirmed < b), None)
            if phase is None:
                continue
            for row in groups[0]:
                if row['status'] != 'closed' or row['pnl_r'] is None or not np.isfinite(row['pnl_r']):
                    counts['unknown_outcome'] += 1
                    continue
                if not row['closed_at'] or instant(row['closed_at']) >= limits[phase][1]:
                    counts['censored_outcome'] += 1
                    continue
                contexts.append(tuple(recent[symbol]))
                labels.append(row['pnl_r'])
                phases.append(phase)
                stop_inputs.append(stop_distance_feature(parameters['config'], symbol, row['pivot_price'], row.get('stop_buffer_pips'))
                                   if uses_stop_distance(parameters['config']) else 0.)
                if len(contexts) % 20000 == 0:
                    print(f'[swing-informer-data] events={counts["events"]} labeled_sequences={len(contexts)}', flush=True)
        if not contexts:
            raise ValueError('No closed opportunities with a complete swing context in the requested periods')
        raw = np.asarray(points, dtype=np.float64)
        indices = np.asarray(contexts, dtype=np.int32)
        outcomes = np.asarray(labels, dtype=np.float32)
        if not np.isfinite(outcomes).all():
            raise ValueError('Nonfinite or overflowing PnL/R labels')
        # Include warmup geometry in the input fingerprint as well as the labeled event stream.
        digest.update(raw.tobytes())
        metadata = dict(version=1, run_id=run_id, symbols=selected, feature_schema=schema,
                        opportunity_config=parameters['config'], source_fingerprint=parameters.get('source_fingerprint'),
                        snapshot_sha256=digest.hexdigest(), periods=periods, phase_names=phase_names,
                        seq_len=seq_len, counts=counts, labeled_sequences=len(indices), read_only=True,
                        source_database=str(Path(database).resolve()))
        return SwingInformerData(raw, indices, outcomes, np.asarray(phases, dtype=np.uint8), metadata, np.asarray(stop_inputs, dtype=np.float32))
    finally:
        db.close()


class SwingInformerData:
    def __init__(self, points, contexts, labels, phases, metadata, stop_inputs=None):
        self.points, self.contexts, self.labels, self.phases = points, contexts, labels, phases
        self.metadata = metadata
        self.stop_inputs = np.zeros(len(labels), dtype=np.float32) if stop_inputs is None else stop_inputs

    def save(self, path):
        np.savez_compressed(path, points=self.points, contexts=self.contexts, labels=self.labels,
                            phases=self.phases, metadata=json.dumps(self.metadata), stop_inputs=self.stop_inputs)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            return cls(saved['points'], saved['contexts'], saved['labels'], saved['phases'],
                       json.loads(str(saved['metadata'])), saved['stop_inputs'] if 'stop_inputs' in saved else None)

    def partition(self, phase):
        return np.flatnonzero(self.phases == self.metadata['phase_names'].index(phase))

    def stop_selection(self, indices):
        if not uses_stop_distance(self.metadata['opportunity_config']):
            return None
        from ultimateBot.backtester.research.swing_selection import StopSelection
        from ultimateBot.backtester.research.swing_opportunities import OpportunityConfig
        config = OpportunityConfig(**self.metadata['opportunity_config'])
        last = self.contexts[indices, -1]
        prices = self.points[last, PRICE]
        symbols = self.points[last, SYMBOL].astype(int)
        pips = np.asarray([config.pip_for(self.metadata['symbols'][i]) for i in symbols])
        buffers = self.stop_inputs[indices].astype(np.float64)*prices/pips
        ranks = np.rint((buffers-config.stop_buffer_pips)/config.stop_step_pips).astype(np.int64)
        return StopSelection(last.astype(str), ranks, config.stop_variants)

    def batch(self, indices):
        x, marks = features(self.points[self.contexts[indices]], self.metadata['opportunity_config']['interval_seconds'],
                        self.metadata['feature_schema'], len(self.metadata['symbols']))
        if uses_stop_distance(self.metadata['opportunity_config']):
            x = append_stop_feature(x, self.stop_inputs[indices])
        return x, marks


def features(raw, interval_seconds, schema, symbol_count):
    """Batch equivalent of swing_features plus known symbol and UTC calendar marks."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[2] != 6 or raw.shape[1] < 4 or not np.isfinite(raw).all():
        raise ValueError('Require finite [batch,swings,6] raw points')
    price, kind, pivot, confirmed = (raw[:, :, c] for c in (PRICE, KIND, PIVOT, CONFIRMED))
    if ((price <= 0).any() or (np.diff(confirmed, axis=1) <= 0).any()
            or (np.diff(pivot, axis=1) < 0).any() or (kind[:, 1:] == kind[:, :-1]).any()
            or not np.isin(kind, [-1, 1]).all() or (pivot >= confirmed).any()):
        raise ValueError('Invalid, nonalternating or out-of-order swing context')
    previous = np.concatenate([price[:, :1], price[:, :-1]], axis=1)
    channels = [np.log(price/price[:, :1])*10000, np.log(price/previous)*10000, kind,
                np.log1p(np.diff(pivot, axis=1, prepend=pivot[:, :1])/interval_seconds),
                np.log1p(np.diff(confirmed, axis=1, prepend=confirmed[:, :1])/interval_seconds),
                np.log1p((confirmed-pivot)/interval_seconds)]
    if schema == VOLUME_FEATURE_SCHEMA:
        if (raw[:, :, VOLUME] < 0).any():
            raise ValueError('Negative quote volumes')
        channels.append(np.log1p(raw[:, :, VOLUME]))
    if len(channels) != len(feature_names(schema)):
        raise ValueError('Unknown feature schema')
    x = np.stack(channels, axis=-1).astype(np.float32)
    symbol = raw[:, :, SYMBOL].astype(np.int64)
    if (symbol < 0).any() or (symbol >= symbol_count).any() or not (symbol == symbol[:, :1]).all():
        raise ValueError('Invalid or mixed-symbol context')
    x = np.concatenate([x, np.eye(symbol_count, dtype=np.float32)[symbol]], axis=-1)
    # Irregular swing times stay explicit. All timestamps were known at signal time.
    seconds = confirmed.astype(np.int64)
    dates = seconds.astype('datetime64[s]')
    days = dates.astype('datetime64[D]')
    month_starts = dates.astype('datetime64[M]').astype('datetime64[D]')
    year_starts = dates.astype('datetime64[Y]').astype('datetime64[D]')
    marks = np.stack([(seconds//60 % 60)/59-.5, (seconds//3600 % 24)/23-.5,
                      ((days.astype(np.int64)+3) % 7)/6-.5,
                      (days-month_starts).astype(np.int64)/30-.5,
                      (days-year_starts).astype(np.int64)/365-.5], axis=-1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite Informer input features')
    return x, marks


def append_stop_feature(x, values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(x),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid per-trade stop input')
    return np.concatenate((x, np.broadcast_to(values[:, None, None], (*x.shape[:2], 1))), axis=-1)
