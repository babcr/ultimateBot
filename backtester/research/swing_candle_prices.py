"""Causal candle-close means, with a small reusable cache outside the research DB."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np


PRICE_SCHEMA = 'arithmetic_mean_closed_candle_bid_ask_midpoints_v1'


def mean_closes(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Require finite positive closed candle prices for normalization')
    result = float(np.mean(values))
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Invalid candle-close normalization mean')
    return result


def candle_close_mean(candles, *, as_of, window, interval_seconds):
    """Equal weight per closed candle; exclude active/future candles and the left end."""
    if as_of.tzinfo is None or window.mode != 'duration':
        raise ValueError('Candle normalization requires a timezone and a duration')
    left, step = as_of-timedelta(seconds=window.value), timedelta(seconds=interval_seconds)
    values, previous = [], None
    for bid, ask in candles:
        if bid.start_time != ask.start_time or bid.start_time.tzinfo is None:
            raise ValueError('Normalization bid/ask candles must align with a timezone')
        if previous is not None and bid.start_time <= previous:
            raise ValueError('Normalization candles must be strictly chronological')
        previous = bid.start_time
        closed_at = bid.start_time+step
        if left < closed_at <= as_of:
            values.append(.5*bid.close_price+.5*ask.close_price)
    return mean_closes(values)


def _daily_closes(path, symbol, size, modified_ns, interval, cache_root):
    from ultimateBot.backtester.research.swing_opportunities import read_day
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, modified_ns):
        raise ValueError(f'Price source differs from the collected snapshot: {path}')
    request = json.dumps([PRICE_SCHEMA, str(path.resolve()), size, modified_ns, symbol, interval])
    key = hashlib.sha256(request.encode()).hexdigest()
    target = cache_root/(key+'.npz')
    reused = target.is_file()
    if reused:
        with np.load(target, allow_pickle=False) as data:
            if str(data['request']) != request:
                raise ValueError(f'Candle price cache identity mismatch: {target}')
            times, closes = data['times'].copy(), data['closes'].copy()
    else:
        frame = read_day(path, symbol)
        # Last bid and ask belong to the same quote. Each candle has equal weight,
        # independently of its tick count or the number of reversals it contains.
        last = frame.groupby(frame.timestamp//(interval*1000), sort=True)[['bid', 'ask']].last()
        times = np.asarray(last.index, dtype=np.int64)*interval
        closes = .5*last.bid.to_numpy(dtype=np.float64)+.5*last.ask.to_numpy(dtype=np.float64)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (size, modified_ns):
            raise ValueError(f'Price source changed while reading: {path}')
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cache_root/(key+'.'+uuid4().hex+'.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, request=np.asarray(request), times=times, closes=closes)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if (times.ndim != 1 or closes.shape != times.shape or times.dtype.kind not in 'iu'
            or (np.diff(times) <= 0).any() or not np.isfinite(closes).all() or (closes <= 0).any()):
        raise ValueError(f'Invalid candle price cache: {target}')
    return times, closes, reused


class CandleCloseIndex:
    def __init__(self, parameters, *, symbols, start, before, interval_seconds, cache_root):
        sources = parameters.get('source_files')
        if not sources:
            raise ValueError('Candle-close normalization requires the collection source tick files')
        self.prices = {}
        self.interval = interval_seconds
        for symbol in symbols:
            selected = []
            for name, size, modified_ns in sources:
                path = Path(name)
                if len(path.parents) < 3 or path.parents[2].name != symbol:
                    continue
                day = datetime.fromisoformat(path.stem).replace(tzinfo=timezone.utc)
                if day < before and day+timedelta(days=1) > start:
                    selected.append((path, size, modified_ns))
            if not selected:
                raise ValueError(f'No source candle prices for {symbol} in the requested period')
            parts, reused = [], 0
            for i, (path, size, modified_ns) in enumerate(sorted(selected), 1):
                times, closes, cached = _daily_closes(path, symbol, size, modified_ns, interval_seconds, Path(cache_root))
                parts.append((times, closes))
                reused += int(cached)
                if i % 100 == 0 or i == len(selected):
                    print(f'[swing-grid-prices] symbol={symbol} days={i}/{len(selected)} reused={reused}', flush=True)
            times = np.concatenate([part[0] for part in parts])
            closes = np.concatenate([part[1] for part in parts])
            # A candle can cross a daily file boundary. Keep its final quote.
            last = np.r_[times[1:] != times[:-1], True] if len(times) else np.empty(0, dtype=bool)
            self.prices[symbol] = times[last], closes[last]

    def mean(self, symbol, *, as_of, window):
        if as_of.tzinfo is None or window.mode != 'duration':
            raise ValueError('Candle normalization requires a timezone and a duration')
        # Stored times are candle starts. [t-D, t) corresponds to closed times
        # (t-D, t] when t and D align to the collection interval.
        if as_of.timestamp() % self.interval or window.value % self.interval:
            raise ValueError('Candle mean must align with the collection interval')
        times, closes = self.prices[symbol]
        lo, hi = np.searchsorted(times, [as_of.timestamp()-window.value, as_of.timestamp()], side='left')
        return mean_closes(closes[lo:hi])
