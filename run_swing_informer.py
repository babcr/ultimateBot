"""Train/resume a direct Informer gain regressor, or evaluate its frozen test."""
import argparse
from dataclasses import asdict, fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ultimateBot.backtester.research.swing_informer import (
    InformerConfig, SwingInformerResolver, atomic_json, evaluate_predictions, fit, predict_data,
)
from ultimateBot.backtester.research.swing_informer_data import CONFIRMED, SYMBOL, SwingInformerData, bounds, extract
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol

DEFAULT_DATABASE = Path('C:/TradingBotData/swing_ts2vec/research.sqlite3')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Fit on train, select epoch and threshold on validation only')
    train.add_argument('--database', type=Path)
    train.add_argument('--run-id')
    train.add_argument('--dataset', type=Path, help='Reuse a frozen dataset.npz for a new model without rereading the DB')
    train.add_argument('--output-dir', type=Path)
    train.add_argument('--resume', type=Path, help='Run directory with request.json and intermediate checkpoints')
    train.add_argument('--symbols', nargs='+')
    train.add_argument('--no-swing-volume', action='store_true', help='Ablate volumes from a collection that has them')
    for name in ('train_start', 'train_end', 'validation_start', 'validation_end'):
        train.add_argument('--'+name.replace('_', '-'))
    defaults = InformerConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        train.add_argument('--'+field.name.replace('_', '-'), type=type(value), default=None)
    test = commands.add_parser('test', help='Evaluate one already frozen model without selecting a new threshold')
    test.add_argument('--model', type=Path, required=True)
    test.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    test.add_argument('--run-id')
    test.add_argument('--start-date', required=True)
    test.add_argument('--end-date', required=True)
    test.add_argument('--output-dir', type=Path, required=True)
    test.add_argument('--device', default='cpu')
    test.add_argument('--threads', type=int, default=2)
    return p


def train_command(args):
    defaults = InformerConfig()
    overrides = {f.name: getattr(args, f.name) for f in fields(defaults) if getattr(args, f.name) is not None}
    if args.resume:
        output = args.resume.resolve()
        if args.output_dir or args.database or args.run_id or args.dataset or args.symbols or args.no_swing_volume or any(
                getattr(args, name) for name in ('train_start', 'train_end', 'validation_start', 'validation_end')):
            raise ValueError('--resume restores the recorded data and dates; do not specify new data selectors')
        if set(overrides)-{'epochs', 'device', 'threads', 'checkpoint_batches'}:
            raise ValueError('Resume can change only epochs, device, threads and checkpoint-batches')
        request = json.loads((output/'request.json').read_text(encoding='utf-8'))
        config = InformerConfig(**{**request['config'], **overrides})
    else:
        if args.output_dir is None:
            raise ValueError('train requires --output-dir or --resume')
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError('Use an empty output directory or --resume')
        output.mkdir(parents=True, exist_ok=True)
        if args.dataset:
            if args.database or args.run_id or args.symbols or args.no_swing_volume or any(
                    getattr(args, name) for name in ('train_start', 'train_end', 'validation_start', 'validation_end')):
                raise ValueError('--dataset fixes collection, periods, symbols and volumes; omit those data selectors')
            cached = SwingInformerData.load(args.dataset)
            overrides.setdefault('seq_len', cached.metadata['seq_len'])
        if 'seq_len' in overrides and 'label_len' not in overrides:
            overrides['label_len'] = overrides['seq_len']//2
        config = InformerConfig(**{**asdict(defaults), **overrides})
        protocol = ResearchProtocol(**{name: getattr(args, name) or value for name, value in dict(
            train_start='2015-01-01', train_end='2021-12-31',
            validation_start='2022-01-01', validation_end='2023-12-31').items()})
        request = dict(config=asdict(config), database=str((args.database or DEFAULT_DATABASE).resolve()),
            run_id=args.run_id, symbols=args.symbols, include_volume=False if args.no_swing_volume else None,
            periods={'train': [protocol.train_start, protocol.train_end],
                     'validation': [protocol.validation_start, protocol.validation_end]})
        if args.dataset:
            if config.seq_len != cached.metadata['seq_len']:
                raise ValueError('Sequence length must match the cached dataset')
            request.update(dataset_path=str(args.dataset.resolve()),
                dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(), periods=cached.metadata['periods'])
        atomic_json(output/'request.json', request)
    snapshot = Path(request.get('dataset_path', output/'dataset.npz'))
    if request.get('dataset_path') and not snapshot.is_file():
        raise FileNotFoundError('The explicitly reused dataset snapshot is missing')
    if snapshot.exists():
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != request.get('dataset_sha256'):
            raise ValueError('Cached dataset hash differs; do not resume on modified inputs')
        data = SwingInformerData.load(snapshot)
    else:
        if (output/'latest.pt').exists():
            raise ValueError('Resume requires the original dataset.npz snapshot')
        data = extract(request['database'], run_id=request['run_id'], periods=request['periods'],
            seq_len=config.seq_len, symbols=request['symbols'], include_volume=request['include_volume'])
        temp = output/'.dataset.tmp.npz'
        data.save(temp)
        temp.replace(snapshot)
        request['dataset_sha256'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        atomic_json(output/'request.json', request)
    result = fit(data, output, config, resume=(output/'latest.pt').exists())
    print(json.dumps({k: result[k] for k in ('model_type', 'threshold', 'best_epoch', 'validation_qualified')}, indent=2))
    return result


def test_command(args):
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new test output directory')
    resolver = SwingInformerResolver(args.model, device=args.device, threads=args.threads)
    meta = resolver.metadata
    start, _ = bounds(args.start_date, args.end_date)
    if start < datetime.fromisoformat(meta['available_after']):
        raise ValueError('Test must begin after training and threshold validation')
    data = extract(args.database, run_id=args.run_id or meta['source']['run_id'],
        periods={'test': [args.start_date, args.end_date]}, seq_len=meta['config']['seq_len'],
        symbols=meta['scope']['symbols'], include_volume=meta['feature_schema'].endswith('_mean_quote_volume_v2'))
    if (data.metadata['opportunity_config'] != meta['opportunity_config']
            or data.metadata['feature_schema'] != meta['feature_schema']):
        raise ValueError('Test must use the original exit configuration and feature schema')
    indices = data.partition('test')
    prediction = predict_data(resolver.regressor, data, indices)
    selection = data.stop_selection(indices)
    result = evaluate_predictions(data.labels[indices], prediction, meta['threshold'], meta['normalization']['target_mean'], selection)
    times = data.points[data.contexts[indices, -1], CONFIRMED]
    years = times.astype('int64').astype('datetime64[s]').astype('datetime64[Y]').astype(str)
    result['years'] = {year: evaluate_predictions(data.labels[indices][years == year], prediction[years == year],
        meta['threshold'], meta['normalization']['target_mean'], selection.subset(years == year) if selection is not None else None) for year in sorted(set(years))}
    symbol_ids = data.points[data.contexts[indices, -1], SYMBOL].astype(int)
    result['per_symbol'] = {symbol: evaluate_predictions(data.labels[indices][symbol_ids == i], prediction[symbol_ids == i],
        meta['threshold'], meta['normalization']['target_mean'], selection.subset(symbol_ids == i) if selection is not None else None) for i, symbol in enumerate(meta['scope']['symbols'])
        if (symbol_ids == i).any()}
    result.update(model=str(args.model.resolve()), model_id=meta['resolver_id'], threshold=meta['threshold'],
        source=data.metadata, threshold_changed=False,
        interpretation='Frozen independent opportunity evaluation, not a portfolio backtest or proof of an edge.')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'test.json', result)
    np.savez_compressed(output/'test_predictions.npz', confirmed_at=times, symbol=symbol_ids,
                        prediction=prediction, actual=data.labels[indices],
                        accepted=(selection.mask(prediction, meta['threshold']) if selection is not None else prediction >= meta['threshold']))
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    return train_command(args) if args.command == 'train' else test_command(args)


if __name__ == '__main__':
    main()
