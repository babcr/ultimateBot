"""Reconstruct a logged hybrid run from local quotes, without changing the strategy.

Entry filters use information known at submission only. Exit counterfactuals keep
the original fills; they are diagnostics, not a new portfolio backtest.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from ultimateBot.audit_breakout_frequency import aggregate_quotes, timestamp
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.orders.order import OrderSide
from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.reports.trade_report import TradeCloser
from ultimateBot.backtester.strategy.regression_breakout import RegressionBreakoutTracker
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel, RegressionReentryTracker
from ultimateBot.backtester.strategy.regression_pullback import RegressionPullbackTracker
from ultimateBot.backtester.strategy.regression_impulse import RegressionImpulseTracker
from ultimateBot.run_backtest import build_engine, parse_args


def read_log(path):
    raw = path.read_bytes()
    text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    orders, fills, closes, trailing = {}, {}, {}, Counter()
    for line in text.splitlines():
        event = re.match(r'\[([\w-]+)\]', line)
        if not event:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        kind = event[1]
        if kind == 'trade-created':
            orders[int(fields['order_id'])] = fields
        elif kind == 'trade-filled':
            fills[int(fields['trade_id'])] = fields
        elif kind == 'trade-closed':
            trade_id = int(fields['trade_id'])
            if trade_id in closes:
                raise ValueError('Partial or repeated closes require a deal-aware audit')
            closes[trade_id] = fields
        elif kind == 'trailing-stop':
            trailing[int(fields['trade_id'])] += 1
    rows = []
    for trade_id, closed in closes.items():
        filled = fills[trade_id]
        created = orders[int(filled['order_id'])]
        number = lambda record, key: float(record[key])
        entry, stop = number(filled, 'entry_price'), number(filled, 'stop_loss')
        direction = 1 if filled['side'] == 'buy' else -1
        distance = direction * (entry - stop)
        if distance <= 0:
            raise ValueError(f'Nonpositive initial risk for trade {trade_id}')
        target = None if filled['take_profit'] == 'None' else number(filled, 'take_profit')
        planned_entry = number(created, 'entry_price')
        planned_risk = direction * (planned_entry - stop)
        if planned_risk <= 0:
            raise ValueError(f'Nonpositive planned risk for trade {trade_id}')
        opened = pd.Timestamp(filled['timestamp'])
        submitted = pd.Timestamp(created['timestamp'])
        signal_at = pd.Timestamp(created['signal_timestamp'])
        close_at = pd.Timestamp(closed['timestamp'])
        pnl = number(closed, 'pnl')
        quantity = number(filled, 'quantity')
        rows.append(dict(trade_id=trade_id, order_id=int(filled['order_id']), symbol=filled['symbol'],
                         branch=filled.get('entry_mode','breakout-retest' if target is None else 'failed-breakout'), side=filled['side'],
                         direction=direction, signal_at=signal_at, submitted_at=submitted, opened_at=opened,
                         closed_at=close_at, year=opened.year, hour_utc=submitted.hour,
                         weekday=submitted.dayofweek, order_type=created['order_type'],
                         entry=entry, planned_entry=planned_entry, planned_risk=planned_risk, stop=stop, target=target,
                         activation=float(filled.get('trailing_activation_price', 'nan')),
                         quantity=quantity, initial_risk=distance, risk_money=distance * quantity,
                         pnl=pnl, r=direction * (number(closed, 'exit_price') - entry) / distance,
                         outcome='win' if pnl > 0 else 'loss' if pnl < 0 else 'flat',
                         exit_price=number(closed, 'exit_price'), close_reason=closed['reason'],
                         planned_rr=direction * (target - planned_entry) / planned_risk if target else np.nan,
                         filled_rr=direction * (target - entry) / distance if target else np.nan,
                         hold_hours=(close_at - opened).total_seconds() / 3600,
                         fill_delay_seconds=(opened - submitted).total_seconds(),
                         signal_age_seconds=(submitted - signal_at).total_seconds(),
                         trailing_updates=trailing[trade_id]))
    frame = pd.DataFrame(rows).sort_values('opened_at').reset_index(drop=True)
    if len(frame) != len(fills):
        raise ValueError('Unclosed fills: this audit currently requires a completed flat run')
    expected = re.search(r'Total closed trades: (\d+)', text)
    if expected and int(expected[1]) != len(frame):
        raise ValueError('Log trade total does not match event records')
    return text, orders, frame


def stats(frame):
    if 'closed_at' in frame:
        frame = frame.sort_values('closed_at')
    values = frame.r.to_numpy(dtype=float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    cash_gains = frame.loc[frame.pnl > 0, 'pnl'].sum()
    cash_losses = -frame.loc[frame.pnl < 0, 'pnl'].sum()
    equity = np.r_[0., np.cumsum(values)]
    return dict(n=len(frame), wins=int((values > 0).sum()), losses=int((values < 0).sum()),
                flats=int((values == 0).sum()), sum_r=float(values.sum()),
                mean_r=float(values.mean()) if len(values) else None,
                profit_factor_r=float(gains / losses) if losses else None,
                profit_factor_cash=float(cash_gains / cash_losses) if cash_losses else None,
                pnl=float(frame.pnl.sum()), max_drawdown_r=float((np.maximum.accumulate(equity) - equity).max()))


def grouped_stats(frame, columns):
    result = []
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        result.append({**dict(zip(columns, (str(key) for key in keys))), **stats(group)})
    return result


def engine_from_log(text, minutes):
    options = ['audit', '--strategy', 'swing-regression', '--regression-entry-mode', 'hybrid',
               '--regression-lookback-bars', '25', '--level-lookback-bars', '150', '--candle-lookback', '150',
               '--candle-interval', f'{minutes}m', '--risk-percent', '1', '--reward-risk-ratio', '2.3',
               '--trailing-stop', '--trailing-start-rr-ratio', '1']
    for name, value in re.findall(r'  --([\w-]+) = ([^\s]+)', text):
        if name in ('regression-pullback-enabled','regression-impulse-enabled'):
            options.append('--'+name if value == 'True' else '--no-'+name)
        else:
            options.extend(['--' + name, value])
    labels = {'Regression lookback bars': 'regression-lookback-bars',
              'Regression minimum parallelism': 'regression-min-parallelism',
              'Regression market volatility window': 'regression-market-volatility-window-bars',
              'Regression minimum market volatility': 'regression-min-market-volatility-ratio',
              'Regression line max volatility \\(ATR\\)': 'regression-line-max-volatility-atr',
              'Regression minimum consecutive swing distance \\(ATR\\)': 'regression-min-consecutive-swing-distance-atr',
              'Regression max filtered consecutive swing pair density': 'regression-max-filtered-consecutive-swing-pair-density',
              'Regression max swing density': 'regression-max-swing-density',
              'Regression minimum reference swings': 'regression-min-swings'}
    for label, option in labels.items():
        match = re.search(r'^\s*' + label + r': (\S+)', text, re.M)
        if match:
            options.extend(['--' + option, match[1]])
    spread = re.search(r'^Maximum submission spread / final planned stop risk: (\S+)',text,re.M)
    if spread:
        options.extend(['--regression-max-spread-stop-ratio',spread[1]])
    previous = sys.argv
    try:
        sys.argv = options
        engine = build_engine([], parse_args())
    finally:
        sys.argv = previous
    return engine


def dates_between(start, end):
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def source_path(history, symbol, day):
    return history / symbol / str(day.year) / f'{day.month:02d}' / f'{day}.tsv'


def read_quotes(path, symbol, float_precision=None):
    frame = pd.read_csv(path, sep='\t', header=None,
                        names=['frame', 'symbol', 'timestamp', 'session', 'sequence', 'bid', 'ask', 'bid_volume', 'ask_volume'],
                        usecols=[0, 1, 2, 5, 6, 7, 8], encoding='utf-8-sig',float_precision=float_precision)
    if not frame.symbol.eq(symbol).all() or not frame.frame.isin(['QUOTE', 'START', 'STOP']).all():
        raise ValueError(f'Invalid quote contract: {path}')
    frame = frame.loc[(frame.frame == 'QUOTE') & (frame.bid > 0) & (frame.ask > 0)]
    if not frame.timestamp.is_monotonic_increasing or not np.isfinite(frame[['bid', 'ask']].to_numpy()).all():
        raise ValueError(f'Invalid numeric quote order: {path}')
    return frame


def replay_exit(row, quotes, activation_r=None, fixed_rr=None, original=False):
    """Use real closer/monitor, with a unit-risk position and original fill."""
    direction = int(row.direction)
    activation = row.activation if original else row.entry + direction * activation_r * row.initial_risk if activation_r else None
    target = row.target if original and pd.notna(row.target) else row.entry + direction * fixed_rr * row.initial_risk if fixed_rr else None
    trailing = pd.notna(activation)
    position = Position(int(row.trade_id), row.symbol, OrderSide.BUY if direction > 0 else OrderSide.SELL,
                        1., row.opened_at.to_pydatetime(), row.entry, row.stop, target,
                        trailing_stop_activation_price=activation if trailing else None, trailing_stop_enabled=trailing)
    closer, monitor = TradeCloser(trailing_stop_enabled=trailing), TradeMonitor()
    closer.prepare_position(position)
    first_ms = int(row.opened_at.timestamp() * 1000)
    for ms, bid, ask in quotes:
        # Positions are monitored before fills in BacktestEngine.on_tick.
        if ms <= first_ms:
            continue
        tick = Tick(timestamp(ms), row.symbol, bid, ask, 0., 0.)
        with redirect_stdout(io.StringIO()):
            request = closer.check(position, tick, monitor)
        if request:
            return dict(r=direction * (request.price - row.entry) / row.initial_risk,
                        close_ms=int(ms), reason=request.reason.value)
    return dict(r=None, close_ms=None, reason='censored')


def process_symbol(history, symbol, trades, minutes, cache_dir, exit_variants):
    selected = set()
    for row in trades.itertuples():
        end = row.closed_at.date()
        if row.branch == 'breakout-retest' and exit_variants:
            end = max(end, (row.opened_at + pd.Timedelta(days=14)).date())
        selected.update(dates_between((row.signal_at - pd.Timedelta(days=14)).date(), end))
    files = [(day, source_path(history, symbol, day)) for day in sorted(selected)]
    files = [(day, path) for day, path in files if path.exists()]
    cache_dir.mkdir(parents=True, exist_ok=True)
    bars, paths, data_by_day = [], [], {}
    for index, (day, path) in enumerate(files):
        contract = [path.stat().st_size, path.stat().st_mtime_ns, minutes]
        cache_path = cache_dir / f'{symbol}_{day}_{minutes}m.json'
        saved = json.loads(cache_path.read_text()) if cache_path.exists() else None
        needs_ticks = any(row.submitted_at.date() == day or row.opened_at.date() <= day <= (max(row.closed_at, row.opened_at + pd.Timedelta(days=14))
                              if row.branch == 'breakout-retest' and exit_variants else row.closed_at).date()
                          for row in trades.itertuples())
        quotes = read_quotes(path, symbol) if needs_ticks or not saved or saved['contract'] != contract else None
        if not saved or saved['contract'] != contract:
            saved = {'contract': contract, 'bars': aggregate_quotes(quotes, minutes * 60_000)}
            cache_path.write_text(json.dumps(saved), encoding='utf-8')
        bars.extend(saved['bars'])
        paths.append([str(path), *contract[:2]])
        if needs_ticks:
            data_by_day[day] = quotes[['timestamp', 'bid', 'ask']].to_numpy()
        if index % 100 == 0 or index == len(files) - 1:
            print(f'{symbol}: quotes/candles {index + 1}/{len(files)}', flush=True)
    results = []
    for row in trades.itertuples():
        arrays = [data_by_day[day] for day in sorted(data_by_day) if row.opened_at.date() <= day <= row.closed_at.date()]
        quotes = np.concatenate(arrays) if arrays else np.empty((0, 3))
        start_ms, close_ms = int(row.opened_at.timestamp() * 1000), int(row.closed_at.timestamp() * 1000)
        active = quotes[(quotes[:, 0] > start_ms) & (quotes[:, 0] <= close_ms)]
        change = row.direction * (active[:, 1 if row.direction > 0 else 2] - row.entry) / row.initial_risk
        submitted_ms = int(row.submitted_at.timestamp() * 1000)
        submission = data_by_day.get(row.submitted_at.date(), np.empty((0, 3)))
        index = np.searchsorted(submission[:, 0], submitted_ms) if len(submission) else 0
        spread = submission[index, 2] - submission[index, 1] if index < len(submission) and submission[index, 0] == submitted_ms else np.nan
        baseline = replay_exit(row, active, original=True)
        result = dict(trade_id=row.trade_id, spread_at_submission=spread, spread_risk=spread / row.planned_risk,
                      mfe_r=float(max(0, change.max())) if len(change) else np.nan,
                      mae_r=float(min(0, change.min())) if len(change) else np.nan,
                      ticks_held=len(active), baseline_replay_r=baseline['r'], baseline_replay_close_ms=baseline['close_ms'],
                      baseline_replay_reason=baseline['reason'])
        if row.branch == 'breakout-retest' and exit_variants:
            horizon = (row.opened_at + pd.Timedelta(days=14)).date()
            all_quotes = np.concatenate([data_by_day[day] for day in sorted(data_by_day)
                                         if row.opened_at.date() <= day <= horizon])
            all_quotes = all_quotes[all_quotes[:, 0] > start_ms]
            for name, activation_r, fixed_rr in [('trail_1r', 1., None), ('trail_1_5r', 1.5, None),
                                                ('trail_2r', 2., None), ('fixed_2_3r', None, 2.3)]:
                alternative = replay_exit(row, all_quotes, activation_r=activation_r, fixed_rr=fixed_rr)
                result[name + '_r'] = alternative['r']
                result[name + '_close_ms'] = alternative['close_ms']
        results.append(result)
    return bars, results, paths


def reconstruct_context(engine, bars, row, minutes):
    interval = timedelta(minutes=minutes)
    signal_ms = int(row.signal_at.timestamp() * 1000) // (minutes * 60_000) * (minutes * 60_000)
    indices = {bar['start_ms']: i for i, bar in enumerate(bars)}
    end = indices.get(signal_ms)
    if end is None:
        return {'context_matches': 0, 'context_error': 'signal_candle_missing'}
    strategy, rules = engine.strategy, engine.strategy.regression_rules
    n = int(rules['regression_lookback_bars'])
    candles = {source: [Candle(timestamp(bar['start_ms']), *bar[source], bar['count'], *bar['volumes'])
                         for bar in bars[max(0, end - n - 12):end + 1]] for source in ('bid', 'ask', 'mid')}
    final = len(candles['mid']) - 1
    prefix = 'reentry' if row.branch == 'failed-breakout' else 'pullback' if row.branch in ('trend-pullback','channel-bounce') else 'breakout'
    max_bars = 1 if row.branch == 'breakout-hold' else rules['regression_'+prefix+'_max_bars']
    matches = []
    for age in range(max_bars + 1):
        event = final - age
        if event < n:
            continue
        evaluation = strategy._evaluate_regression_pipeline(row.symbol,
            candles['bid'][event-n:event], candles['ask'][event-n:event], candles['mid'][event-n:event], .00001,
            timeframe_label='local', timeframe_multiplier=1, require_fresh_swings=False, rules=rules, channel_only=True)
        if not evaluation or evaluation[1]['reasons']:
            continue
        analysis = evaluation[1]
        frozen = FrozenChannel(analysis, candles['mid'][event-1].start_time,
                               candles['bid'][event-1].close_price, candles['ask'][event-1].close_price)
        tracker = RegressionReentryTracker() if row.branch == 'failed-breakout' else RegressionPullbackTracker(row.branch) if row.branch in ('trend-pullback','channel-bounce') else RegressionImpulseTracker() if row.branch == 'breakout-hold' else RegressionBreakoutTracker()
        for i in range(event, final + 1):
            signal, details = tracker.evaluate(row.symbol, candles['bid'][i], candles['ask'][i],
                row.signal_at.to_pydatetime(), interval, rules, frozen if i == event else None)
            if signal is not None:
                if (i == final and signal.side.name.lower() == row.side
                        and abs(signal.stop_loss_price - row.stop) < 1e-9
                        and (row.branch != 'failed-breakout' or abs(signal.take_profit_price - row.target) < 1e-9)):
                    matches.append((signal.entry_context, details))
                break
    result = {'context_matches': len(matches)}
    if len(matches) != 1:
        return result
    context, analysis = matches[0]
    atr = context['atr']
    result['context'] = context
    result.update(atr=atr, stop_atr=row.planned_risk / atr,
                  channel_width_atr=(context['channel_upper'] - context['channel_lower']) / atr,
                  parallelism=analysis['parallelism'], swing_density=analysis['swing_density'],
                  filtered_pair_density=analysis['filtered_consecutive_swing_pair_density'],
                  channel_drift=context.get('channel_drift_ratio', np.nan),
                  excursion_atr=context.get('excursion_atr', np.nan),
                  reentry_bars=context.get('reentry_bars', np.nan),
                  breakout_close_atr=context.get('breakout_close_atr', np.nan),
                  confirmation_body=context.get('confirmation_body_ratio', np.nan),
                  confirmation_location=context.get('confirmation_close_location', np.nan),
                  channel_slope_atr=row.direction * analysis['shared_slope'] / atr,
                  market_volatility_ratio=analysis['market_volatility_ratio'])
    return result


def market_features(bars, row, minutes, atr):
    last_ms = int(row.signal_at.timestamp() * 1000) // (minutes * 60_000) * minutes * 60_000
    history = [bar for bar in bars if bar['start_ms'] <= last_ms][-180:]
    mid = np.array([bar['mid'] for bar in history])
    o, h, l, c = mid[-1]
    size = h - l
    direction = row.direction
    result = dict(signal_body_direction=direction * (c - o) / size if size else 0,
                  signal_close_location=((c-l) if direction > 0 else (h-c)) / size if size else .5,
                  rejection_wick=((min(o,c)-l) if direction > 0 else (h-max(o,c))) / size if size else 0,
                  signal_range_atr=size / atr,
                  signal_gap_hours=(history[-1]['start_ms'] - history[-2]['start_ms']) / 3_600_000)
    closes = mid[:, 3]
    ranges = np.maximum(mid[1:, 1], closes[:-1]) - np.minimum(mid[1:, 2], closes[:-1])
    for window in (6, 24, 72):
        if len(closes) > window:
            result[f'momentum_{window}_atr'] = direction * (closes[-1] - closes[-window-1]) / atr
            path = np.abs(np.diff(closes[-window-1:])).sum()
            result[f'efficiency_{window}'] = abs(closes[-1] - closes[-window-1]) / path if path else 0
    result['volatility_6_48'] = float(ranges[-6:].mean() / ranges[-48:].mean()) if len(ranges) >= 48 else np.nan
    return result


def candidate_masks(frame):
    failed = frame.branch.eq('failed-breakout')
    return {
        'exclude_usdchf': frame.symbol.ne('USDCHF'),
        'failed_only': failed,
        'failed_without_usdchf': failed & frame.symbol.ne('USDCHF'),
        'spread_risk_le_5pct': frame.spread_risk.le(.05),
        'spread_risk_le_10pct': frame.spread_risk.le(.10),
        'failed_rr_ge_1_5': ~failed | frame.planned_rr.ge(1.5),
        'failed_rr_ge_2': ~failed | frame.planned_rr.ge(2),
        'failed_excursion_le_1atr': ~failed | frame.excursion_atr.le(1),
        'failed_excursion_ge_0_25atr': ~failed | frame.excursion_atr.ge(.25),
        'failed_drift_le_0_25': ~failed | frame.channel_drift.le(.25),
        'failed_width_ge_3atr': ~failed | frame.channel_width_atr.ge(3),
        'failed_rejection_wick_ge_30pct': ~failed | frame.rejection_wick.ge(.3),
        'failed_stopside_wick_le_25pct': ~failed | frame.rejection_wick.le(.25),
        'failed_reentry_next_bar': ~failed | frame.reentry_bars.eq(1),
        'failed_nextbar_or_short_samebar': ~failed | frame.reentry_bars.eq(1) | (frame.reentry_bars.eq(0) & frame.rejection_wick.le(.25)),
        'failed_directional_body': ~failed | frame.signal_body_direction.gt(0),
        'failed_efficiency24_le_0_3': ~failed | frame.efficiency_24.le(.3),
        'failed_no_strong_opposing_momentum': ~failed | frame.momentum_24_atr.ge(-2),
        'no_rollover_21_23utc': ~frame.hour_utc.isin([21, 22, 23]),
        'no_volatility_burst': frame.volatility_6_48.le(1.5),
        'stop_ge_0_75atr': frame.stop_atr.ge(.75),
    }


def bootstrap_mean(values, seed=19026, repeats=10_000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    generator = np.random.default_rng(seed)
    means = generator.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def analyze(frame):
    report = {'total': stats(frame), 'groups': {}, 'candidates': []}
    for cols in (['branch'], ['symbol'], ['year'], ['branch', 'symbol'], ['branch', 'year'], ['hour_utc'], ['order_type'], ['side']):
        report['groups']['/'.join(cols)] = grouped_stats(frame, cols)
    report['total']['mean_r_iid_ci95'] = bootstrap_mean(frame.r)
    for name, mask in candidate_masks(frame).items():
        candidate = {'name': name, 'all': stats(frame.loc[mask]), 'removed': stats(frame.loc[~mask]),
                     'development_2020_2022': stats(frame.loc[mask & frame.year.le(2022)]),
                     'validation_2023_2025': stats(frame.loc[mask & frame.year.ge(2023)]),
                     'by_year': grouped_stats(frame.loc[mask], ['year']),
                     'by_symbol': grouped_stats(frame.loc[mask], ['symbol'])}
        report['candidates'].append(candidate)
    report['baseline_development'] = stats(frame.loc[frame.year.le(2022)])
    report['baseline_validation'] = stats(frame.loc[frame.year.ge(2023)])
    report['feature_comparisons'] = {}
    features = ['spread_risk', 'planned_rr', 'stop_atr', 'channel_width_atr', 'channel_drift', 'excursion_atr',
                'reentry_bars', 'parallelism', 'swing_density', 'signal_body_direction', 'rejection_wick',
                'efficiency_24', 'momentum_24_atr', 'volatility_6_48', 'mfe_r', 'mae_r']
    for branch, subset in frame.groupby('branch'):
        report['feature_comparisons'][branch] = {
            feature: {outcome: {'n': int(group[feature].notna().sum()), 'median': float(group[feature].median()),
                                'q25': float(group[feature].quantile(.25)), 'q75': float(group[feature].quantile(.75))}
                      for outcome, group in subset.groupby('outcome')} for feature in features}
    breakout = frame.loc[frame.branch.eq('breakout-retest')]
    report['exit_variants'] = {}
    for name in ['r', 'trail_1r_r', 'trail_1_5r_r', 'trail_2r_r', 'fixed_2_3r_r']:
        if name in breakout:
            valid = breakout.loc[breakout[name].notna()].copy()
            valid['r'] = valid[name]
            valid['pnl'] = valid['r'] * valid.risk_money
            report['exit_variants'][name] = {'all': stats(valid), 'by_year': grouped_stats(valid, ['year']),
                                             'censored': int(breakout[name].isna().sum())}
    return report


def json_default(value):
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(clean(value), indent=2, default=json_default, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=Path(__file__).parent.parent / 'out.log')
    parser.add_argument('--history', type=Path, default=Path('C:/jforex_history'))
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'artifacts/hybrid_trade_audit')
    parser.add_argument('--minutes', type=int, default=60)
    parser.add_argument('--skip-exit-variants', action='store_true')
    args = parser.parse_args()
    if args.minutes <= 0 or 1440 % args.minutes:
        parser.error('The candle interval must divide a UTC day into whole minutes')
    args.output.mkdir(parents=True, exist_ok=True)
    log_fingerprint = sha256(args.log.read_bytes()).hexdigest()
    text, orders, frame = read_log(args.log)
    if ('Regression entry mode: hybrid' not in text
            or 'Position exit mode: failed-breakout = fixed channel center; breakout-retest = trailing stop' not in text):
        parser.error('Branch inference requires a hybrid log with fixed-center reentries and trailing breakouts')
    frame.to_csv(args.output / 'logged_trades.csv', index=False)
    write_json(args.output / 'log_summary.json', {'total': stats(frame), 'by_branch': grouped_stats(frame, ['branch']),
                                               'by_symbol_year': grouped_stats(frame, ['symbol', 'year'])})
    print(f'Parsed {len(frame)} closed trades / {len(orders)} orders', flush=True)
    engine = engine_from_log(text, args.minutes)
    context_records, tick_records, manifest = [], [], []
    contexts = []
    for symbol, subset in frame.groupby('symbol'):
        bars, results, paths = process_symbol(args.history, symbol, subset, args.minutes, args.output / 'candle_cache', not args.skip_exit_variants)
        tick_records.extend(results)
        manifest.extend(paths)
        for row in subset.itertuples():
            recovered = reconstruct_context(engine, bars, row, args.minutes)
            context = recovered.pop('context', None)
            if context:
                contexts.append({'trade_id': row.trade_id, 'context': context})
            recovered.update(trade_id=row.trade_id)
            atr = recovered.get('atr', np.nan)
            recovered.update(market_features(bars, row, args.minutes, atr))
            context_records.append(recovered)
        print(f'{symbol}: recovered {sum(item.get("context_matches") == 1 for item in context_records)}/{len(context_records)} contexts', flush=True)
        pd.DataFrame(context_records).to_csv(args.output / 'entry_features_partial.csv', index=False)
        pd.DataFrame(tick_records).to_csv(args.output / 'tick_features_partial.csv', index=False)
    combined = frame.merge(pd.DataFrame(context_records), on='trade_id', validate='one_to_one').merge(pd.DataFrame(tick_records), on='trade_id', validate='one_to_one')
    combined.to_csv(args.output / 'trades_enriched.csv', index=False)
    report = analyze(combined)
    if sha256(args.log.read_bytes()).hexdigest() != log_fingerprint:
        raise ValueError('The source log changed while auditing; rerun against an immutable copy')
    report['provenance'] = {'log': str(args.log.resolve()), 'log_sha256': log_fingerprint,
                            'effective_rules': engine.strategy.regression_rules, 'files': manifest,
                            'entry_context_unique_matches': int(combined.context_matches.eq(1).sum()),
                            'baseline_replay_max_r_error': float((combined.r - combined.baseline_replay_r).abs().max()),
                            'baseline_replay_close_matches': int((combined.baseline_replay_close_ms == combined.closed_at.map(lambda value: int(value.timestamp()*1000))).sum())}
    write_json(args.output / 'contexts.json', contexts)
    write_json(args.output / 'analysis.json', report)
    print(json.dumps({'total':report['total'], 'exit_variants':{name:result['all'] for name,result in report['exit_variants'].items()}}, default=json_default), flush=True)


if __name__ == '__main__':
    main()
