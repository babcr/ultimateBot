#!/usr/bin/env python3
"""Brute-force alternating swing opportunities -> TS2Vec -> XGBoost research."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ultimateBot.backtester.research.swing_series import VOLUME_FEATURE_SCHEMA, SwingPoint, SwingWindow
from ultimateBot.backtester.research.swing_opportunities import OpportunityConfig, collect, connect, existing_collection
from ultimateBot.backtester.research.swing_encoder import SwingEncoderConfig
from ultimateBot.backtester.research.swing_resolver import (
    ResearchProtocol, ResolverConfig, SwingResolver, TrainingSelection, frozen_test, resolver_directory, train, RESOLVER_TARGETS,
)


ROOT = Path("C:/TradingBotData/swing_ts2vec")


def duration(value):
    match = re.fullmatch(r"([1-9][0-9]*)(s|m|min|h|d|w)", str(value).lower())
    if match is None:
        raise argparse.ArgumentTypeError("Use a positive duration such as 60m, 24h or 7d")
    return int(match[1]) * {"s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400, "w": 604800}[match[2]]


def windows_for(args):
    result = []
    if args.window_mode in ("both", "duration"):
        result.extend(SwingWindow("duration", seconds) for seconds in args.durations)
    if args.window_mode in ("both", "pairs"):
        result.extend(SwingWindow("pairs", n) for n in args.pairs)
    return tuple(dict.fromkeys(result))


def opportunity_config(args):
    return OpportunityConfig(args.candle_interval, args.trailing_start_r, args.order_expiration_bars,
                             args.execution_delay_seconds, args.risk_cash, args.stop_buffer_pips, args.pip_size,
                             args.stop_variants, args.stop_step_pips, args.max_holding_seconds)


def training_configs(args):
    resolver = ResolverConfig(**{f.name: getattr(args, f.name) for f in fields(ResolverConfig)})
    encoder = SwingEncoderConfig(**{f.name: getattr(args, "ts2vec_" + f.name) for f in fields(SwingEncoderConfig)})
    return resolver, encoder


def training_selection(args):
    durations, pairs = args.durations, args.pairs
    if args.command == "run":
        durations = durations if args.window_mode in ("both", "duration") else None
        pairs = pairs if args.window_mode in ("both", "pairs") else None
    mode = args.window_mode
    if getattr(args, 'representation', 'ts2vec') == 'swing-grid':
        if mode == 'pairs' or pairs is not None:
            raise ValueError('Swing grid requires duration windows, not --pairs')
        if getattr(args, 'swing_volume', None) is True:
            raise ValueError('Swing grid is price-only; omit --swing-volume')
        mode = 'duration'
    return TrainingSelection(symbols=tuple(args.symbols) if args.symbols else None,
                             include_volume=getattr(args, 'swing_volume', None),
                             interval_seconds=args.candle_interval, window_mode=mode,
                             durations=tuple(durations) if durations else None,
                             pairs=tuple(pairs) if pairs else None)


def parse_args(argv=None):
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ("collect", "train", "run", "search", "test", "export", "score"):
        child = commands[name] = sub.add_parser(name)
        child.add_argument("--database", type=Path, default=ROOT / "research.sqlite3")
    for name in ("collect", "run"):
        child = commands[name]
        child.add_argument("--history", type=Path, default=Path("C:/jforex_history"))
        child.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDCAD", "USDCHF"])
        child.add_argument("--start-date", type=date.fromisoformat, default=date(2020, 1, 1))
        child.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
        child.add_argument("--candle-interval", type=duration, default=3600)
        child.add_argument("--window-mode", choices=("both", "duration", "pairs"), default="both")
        child.add_argument("--durations", type=duration, nargs="+", default=[43200, 86400, 172800])
        child.add_argument("--pairs", type=int, nargs="+", default=[4, 8, 16], help="Exact number of high/low couples (2*N swings)")
        child.add_argument("--trailing-start-r", type=float, default=1.)
        child.add_argument("--order-expiration-bars", type=int, default=2,
                           help="Pending order lifetime only; never time-closes a filled position")
        child.add_argument("--execution-delay-seconds", type=float, default=0.)
        child.add_argument("--risk-cash", type=float, default=100., help="Constant virtual cash risk per independent opportunity")
        child.add_argument("--stop-buffer-pips", type=float, default=0.,
                           help="Extra stop distance below ask lows for buys, above bid highs for sells")
        child.add_argument("--pip-size", type=float, default=0., help="0: infer standard FX pip size; explicit price unit per pip for other symbols")
        child.add_argument('--stop-variants', type=int, default=1, help='Trades per swing including the base structural stop')
        child.add_argument('--stop-step-pips', type=float, default=5., help='Added stop margin per successive variant')
        child.add_argument('--max-holding', dest='max_holding_seconds', type=duration, default=14*86400,
                           help='Close at the first quote after this duration from fill (default: 2w)')
        child.add_argument("--all-ticks", action="store_true", help="Disable the verified event-jump acceleration")
        child.add_argument('--store-contexts', action=argparse.BooleanOptionalAction, default=None,
                           help='Build and store legacy context blobs during collection (default: defer windowing to train/search)')
        child.add_argument('--swing-volume', action=argparse.BooleanOptionalAction, default=None,
                           help='Add mean per-tick bid volume (high) / ask volume (low) over both swing candles to TS2Vec inputs')
        child.add_argument("--dry-run", action="store_true")
    commands['collect'].add_argument('--run-id', default=None,
        help='Add new symbols to this completed collection, inheriting its dates and execution/window settings')
    for name in ("train", "run", "search"):
        child = commands[name]
        child.add_argument('--representation', choices=('ts2vec', 'swing-grid'), default='ts2vec',
                           help='TS2Vec embeddings or direct zero-filled candle grids')
        child.add_argument('--grid-normalization', choices=('swing-mean', 'candle-close-mean'), default='swing-mean',
                           help='Grid reference: mean of swings, or mean of all closed candle bid/ask midpoint closes')
        child.add_argument('--cohort-policy', choices=('common', 'per-window'), default='common',
                           help='Grid cohorts: legacy common training cohort, or stable per-window fitting with common validation ranking')
        child.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
        for field in fields(ResolverConfig):
            default = getattr(ResolverConfig(), field.name)
            options = {"choices": RESOLVER_TARGETS} if field.name == "target" else {}
            child.add_argument("--" + field.name.replace("_", "-"), type=type(default), default=default, **options)
        for field in fields(SwingEncoderConfig):
            default = getattr(SwingEncoderConfig(), field.name)
            descriptions = {
                'train_epochs': 'Complete upstream training epochs (default: 1); budget scales with selected data',
                'train_iters': 'Explicit optimizer-update budget; overrides epochs (0: use epochs)',
                'depth': 'Residual depth (0: receptive field covers the longest training window)',
                'max_train_length': 'Full-window length limit (0: unlimited); exceeding it is an error, never a crop',
                'max_train_samples': 'Explicit pretraining sample cap (0: all available training sequences)',
                'learning_rate': 'TS2Vec optimizer learning rate; a tunable baseline, not an automatic optimum'}
            child.add_argument("--ts2vec-" + field.name.replace("_", "-"), type=type(default), default=default,
                               help=descriptions.get(field.name))
        for field in fields(ResearchProtocol):
            child.add_argument("--"+field.name.replace('_','-'),default=getattr(ResearchProtocol(),field.name),
                               help="Explicit inclusive UTC calendar boundary YYYY-MM-DD")
    for name in ("train", "search"):
        child = commands[name]
        child.add_argument("--run-id", default=None, help="Defaults to latest completed collection matching the selectors")
        child.add_argument('--swing-volume', action=argparse.BooleanOptionalAction, default=None,
                           help='Use swing volume inputs when collected; --no-swing-volume trains a six-feature comparison on the same data')
        child.add_argument("--symbols", nargs="+", default=None, help="One symbol or an explicit pooled group; default: all collected symbols")
        child.add_argument("--candle-interval", type=duration, default=None, help="Must match the collection; never resamples swings")
        child.add_argument("--window-mode", choices=("both", "duration", "pairs"), default="both")
        child.add_argument("--durations", type=duration, nargs="+", default=None, help="Duration windows; search can build new sizes from stored swings")
        child.add_argument("--pairs", type=int, nargs="+", default=None, help="Exact couple counts; search can build new sizes from stored swings")
    child = commands['search']
    child.add_argument('--duration-range', type=duration, nargs=3, metavar=('START', 'STOP', 'STEP'),
                       help='Inclusive duration grid, e.g. 6h 72h 6h')
    child.add_argument('--pairs-range', type=int, nargs=3, metavar=('START', 'STOP', 'STEP'),
                       help='Inclusive couple-count grid, e.g. 2 32 2')
    child.add_argument('--repeats', type=int, default=3, help='Independent seeds per window; rank median validation R')
    child.add_argument('--max-depths', type=int, nargs='+', default=None,
                       help='XGBoost depth grid, e.g. 3 5 7 9; compare each window/depth across repeats (default: --max-depth)')
    child.add_argument('--resume', type=Path, default=None, help='Resume a search directory with exactly the same options')
    child.add_argument('--dry-run', action='store_true', help='Read-only preview of the grid and number of models')
    commands["export"].add_argument("--run-id", required=True)
    commands["export"].add_argument("--model-id", default=None, help="One swing encoder model to export; omit for opportunity outcomes only")
    commands["export"].add_argument("--output", type=Path, required=True)
    commands["score"].add_argument("--resolver", type=Path, required=True, help="Window artifact directory or experiment directory with selected.json")
    commands["score"].add_argument("--context-id", type=int, required=True)
    commands["test"].add_argument("--run-id",default=None,help="Collection containing the 2026 observations; defaults to latest completed run")
    commands["test"].add_argument("--resolver",type=Path,required=True,help="Frozen experiment directory (selected.json) or a specific frozen window")
    commands["test"].add_argument("--test-start",default="2026-01-01")
    commands["test"].add_argument("--test-end",default="2026-12-31")
    commands["test"].add_argument("--min-test-samples",type=int,default=30)
    commands["test"].add_argument("--output-dir",type=Path,default=None)
    commands["test"].add_argument("--threads",type=int,default=2)
    args = parser.parse_args(raw_arguments)
    try:
        if args.command == 'collect' and args.run_id is not None:
            supplied = {token.split('=', 1)[0] for token in raw_arguments if token.startswith('--')}
            if '--symbols' not in supplied:
                raise ValueError('Specify --symbols explicitly when adding symbols to an existing run')
            _, parameters = existing_collection(args.database, args.run_id)
            stored_contexts = parameters.get('store_contexts', True)
            if args.store_contexts is not None and args.store_contexts != stored_contexts:
                raise ValueError('Adding symbols must preserve the context storage setting')
            args.store_contexts = stored_contexts
            stored_volume = parameters['feature_schema'] == VOLUME_FEATURE_SCHEMA
            if args.swing_volume is not None and args.swing_volume != stored_volume:
                raise ValueError('Adding symbols must preserve the swing volume feature setting')
            args.swing_volume = stored_volume
            stored = OpportunityConfig(**parameters['config'])
            for field in fields(OpportunityConfig):
                attribute = 'candle_interval' if field.name == 'interval_seconds' else field.name
                flag = '--max-holding' if attribute == 'max_holding_seconds' else '--'+attribute.replace('_', '-')
                if flag not in supplied:
                    setattr(args, attribute, getattr(stored, field.name))
            for attribute, key in (('start_date', 'start'), ('end_date', 'end')):
                if '--'+attribute.replace('_', '-') not in supplied:
                    setattr(args, attribute, date.fromisoformat(parameters[key]))
            if not {'--window-mode', '--durations', '--pairs'} & supplied:
                args.window_mode = 'both'
                args.durations = [w['value'] for w in parameters['windows'] if w['mode']=='duration']
                args.pairs = [w['value'] for w in parameters['windows'] if w['mode']=='pairs']
            if '--all-ticks' not in supplied:
                args.all_ticks = not parameters.get('accelerated', True)
            if (opportunity_config(args) != stored or str(args.start_date) != parameters['start']
                    or str(args.end_date) != parameters['end']
                    or set(windows_for(args)) != {SwingWindow(**w) for w in parameters['windows']}):
                raise ValueError('Adding symbols must preserve the existing dates, candle interval, windows, stop and trailing parameters')
        if args.command in ("collect", "run"):
            if args.store_contexts is None:
                args.store_contexts = False
            opportunity_config(args)
            windows_for(args)
            if args.start_date > args.end_date:
                raise ValueError("Start date must not exceed end date")
        if args.command == 'search':
            if args.repeats < 1:
                raise ValueError('Search repeats must be positive')
            if args.max_depths is not None:
                if '--max-depth' in {token.split('=', 1)[0] for token in raw_arguments}:
                    raise ValueError('Choose --max-depth for one depth or --max-depths for a grid, not both')
                if any(depth < 1 for depth in args.max_depths):
                    raise ValueError('Search depths must be positive integers')
                args.max_depths = sorted(set(args.max_depths))
            for grid_name, values_name in (('duration_range', 'durations'), ('pairs_range', 'pairs')):
                grid = getattr(args, grid_name)
                if grid is not None:
                    start, stop, step = grid
                    if min(start, step) < 1 or stop < start:
                        raise ValueError('Search range requires 0 < start <= stop and a positive step')
                    setattr(args, values_name, list(dict.fromkeys((getattr(args, values_name) or []) + list(range(start, stop+1, step)))))
        if args.command in ('train', 'run', 'search'):
            supplied = {token.split('=', 1)[0] for token in raw_arguments if token.startswith('--')}
            if args.representation == 'swing-grid' and any(token.startswith('--ts2vec-') or token == '--min-encoder-samples' for token in supplied):
                raise ValueError('TS2Vec encoder options do not apply to --representation swing-grid')
            if args.cohort_policy != 'common' and args.representation != 'swing-grid':
                raise ValueError('--cohort-policy per-window requires --representation swing-grid')
            if args.grid_normalization != 'swing-mean' and args.representation != 'swing-grid':
                raise ValueError('--grid-normalization requires --representation swing-grid')
            if {'--ts2vec-train-iters', '--ts2vec-train-epochs'} <= supplied:
                raise ValueError('Choose TS2Vec epochs or an explicit iteration budget, not both')
        if args.command in ("train", "run", "search"):
            training_configs(args)
            training_selection(args)
            ResearchProtocol(**{f.name:getattr(args,f.name) for f in fields(ResearchProtocol)})
        if args.command=='test':
            if date.fromisoformat(args.test_start)>date.fromisoformat(args.test_end) or min(args.min_test_samples,args.threads)<1:
                raise ValueError('Invalid test date range or sample/thread count')
    except ValueError as error:
        parser.error(str(error))
    return args


def export(database, run_id, output, model_id=None):
    if not Path(database).is_file():
        raise FileNotFoundError("Swing database does not exist")
    db = connect(database)
    try:
        if db.execute("SELECT 1 FROM swing_runs WHERE run_id=?", (run_id,)).fetchone() is None:
            raise ValueError("Unknown swing run")
        if model_id is None:
            cursor = db.execute("SELECT * FROM swing_events WHERE run_id=? ORDER BY confirmed_at,symbol", (run_id,))
            columns = [c[0] for c in cursor.description]
            rows = (dict(r) for r in cursor)
        else:
            model = db.execute("SELECT * FROM swing_embedding_models WHERE model_id=? AND run_id=?", (model_id, run_id)).fetchone()
            if model is None:
                raise ValueError("Unknown swing embedding model for this run")
            dimensions = json.loads(model["metadata_json"])["config"]["output_dims"]
            cursor = db.execute("""SELECT e.*,c.context_id,c.window_key,c.information_start,c.context_end,c.n_swings,c.span_seconds,v.embedding
                FROM swing_events e JOIN swing_contexts c USING(event_id) JOIN swing_embeddings v USING(context_id)
                WHERE e.run_id=? AND v.model_id=? ORDER BY e.confirmed_at,e.symbol""", (run_id, model_id))
            columns = [c[0] for c in cursor.description if c[0] != "embedding"] + [f"embedding_{i:03d}" for i in range(dimensions)]
            def records():
                for record in cursor:
                    row = dict(record)
                    vector = np.frombuffer(row.pop("embedding"), dtype="<f4")
                    yield {**row, **{f"embedding_{i:03d}": float(v) for i, v in enumerate(vector)}}
            rows = records()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return str(output.resolve())
    finally:
        db.close()


def print_selected_threshold(result):
    """Keep the exported model's exact threshold visible after the JSON summary."""
    threshold = repr(result['selected_threshold'])
    strategy = 'swing-grid' if result.get('representation') == 'swing-grid' else 'swing-ts2vec'
    depth = f"max_depth={result['selected_max_depth']} " if 'selected_max_depth' in result else ''
    objective = result.get('selection_objective')
    score = (f"validation_metric={objective['metric']} validation_score={result['selected_validation_score']:.9g}\n"
             if objective else '')
    print(f"\n[swing-validation-selected] window={result['selected_window_validation_only']} {depth}"
          f"target={result['selected_target']} validation_qualified={result['validation_qualified']}\n"
          f"{score}"
          f"Seuil initial du modele retenu : {threshold}\n"
          f"Option backtest : --{strategy}-threshold {threshold}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'search':
        from ultimateBot.backtester.research.swing_window_search import search
        resolver, encoder = training_configs(args)
        result = search(args.database, run_id=args.run_id, output_dir=args.output_dir,
                        config=resolver, encoder_config=encoder, selection=training_selection(args),
                        protocol=ResearchProtocol(**{f.name: getattr(args, f.name) for f in fields(ResearchProtocol)}),
                        repeats=args.repeats, resume=args.resume, dry_run=args.dry_run,
                        representation=args.representation, cohort_policy=args.cohort_policy,
                        grid_normalization=args.grid_normalization, max_depths=args.max_depths)
        summary = result if args.dry_run else {k: result[k] for k in ('directory', 'selected_window_validation_only',
            'selected_directory', 'selected_threshold', 'selected_target', 'selected_max_depth',
            'selection_objective', 'selected_validation_score',
            'positive_validation', 'positive_median_validation', 'validation_qualified',
            'validation_predictive_gain', 'test_evaluated', 'leaderboard')}
        print(json.dumps(summary, indent=2), flush=True)
        if not args.dry_run:
            print_selected_threshold(result)
        return result
    if getattr(args, "dry_run", False):
        result = {"opportunities": asdict(opportunity_config(args)), "windows": [asdict(w) for w in windows_for(args)],
                  'store_contexts': args.store_contexts,
                  'windowing': 'during_collection' if args.store_contexts else 'deferred_to_train_search',
                  'swing_volume': bool(args.swing_volume),
                  "symbols": args.symbols, "start": str(args.start_date), "end": str(args.end_date), "database": str(args.database)}
        if args.command == 'collect' and args.run_id is not None:
            result.update(run_id=args.run_id, operation='append_symbols')
        if args.command == "run":
            resolver, encoder = training_configs(args)
            result.update(resolver=asdict(resolver),encoder=asdict(encoder),
                          grid_normalization=args.grid_normalization,
                          protocol=asdict(ResearchProtocol(**{f.name:getattr(args,f.name) for f in fields(ResearchProtocol)})),
                          test="Separate frozen test command; no test data is used by run/train")
        print(json.dumps(result, indent=2))
        return result
    run_id = getattr(args, "run_id", None)
    if args.command in ("collect", "run"):
        result = collect(args.database, history=args.history, symbols=args.symbols, start=args.start_date, end=args.end_date,
                         config=opportunity_config(args), windows=windows_for(args), accelerated=not args.all_ticks,
                         store_contexts=args.store_contexts,
                         run_id=run_id, include_volume=args.swing_volume)
        run_id = result["run_id"]
        print(json.dumps(result, indent=2), flush=True)
    if args.command in ("train", "run"):
        resolver, encoder = training_configs(args)
        result = train(args.database, run_id=run_id, output_dir=args.output_dir, config=resolver, encoder_config=encoder,
                       protocol=ResearchProtocol(**{f.name:getattr(args,f.name) for f in fields(ResearchProtocol)}),
                       selection=training_selection(args), representation=args.representation, cohort_policy=args.cohort_policy,
                       grid_normalization=args.grid_normalization)
        print(json.dumps({k: result[k] for k in ("run_id", "directory", "selected_window_validation_only",
                                              "selected_directory", "selected_threshold", "selected_target",
                                              "positive_validation", "validation_qualified")}, indent=2), flush=True)
        print_selected_threshold(result)
    if args.command == "export":
        result = export(args.database, args.run_id, args.output, args.model_id)
        print(result)
    if args.command=='test':
        result=frozen_test(args.database,directory=args.resolver,run_id=args.run_id,start=args.test_start,end=args.test_end,
                           output_dir=args.output_dir,min_samples=args.min_test_samples,threads=args.threads)
        print(json.dumps(result,indent=2))
    if args.command == "score":
        if not args.database.is_file():
            raise FileNotFoundError("Swing database does not exist")
        directory=resolver_directory(args.resolver)
        db = connect(args.database)
        try:
            row = db.execute("""SELECT c.*,e.symbol,e.run_id,e.stop_buffer_pips FROM swing_contexts_resolved c
                JOIN swing_events e USING(event_id) WHERE c.context_id=?""", (args.context_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown swing context")
            model_type = json.loads((directory/'metadata.json').read_text(encoding='utf-8')).get('model_type')
            if model_type == 'swing-grid':
                from ultimateBot.backtester.research.swing_grid import SwingGridResolver
                resolver = SwingGridResolver(directory)
            elif model_type == 'swing-nn':
                from ultimateBot.backtester.research.swing_neural import SwingNeuralResolver
                resolver = SwingNeuralResolver(directory)
            else:
                resolver = SwingResolver(directory)
            if row["window_key"] != resolver.metadata["window_key"]:
                raise ValueError("Context window does not match the resolver")
            parameters = json.loads(db.execute("SELECT parameters_json FROM swing_runs WHERE run_id=?", (row["run_id"],)).fetchone()[0])
            original = asdict(OpportunityConfig(**resolver.metadata["opportunity_config"]))
            current = asdict(OpportunityConfig(**parameters["config"]))
            if {k:v for k,v in original.items() if k != "risk_cash"} != {k:v for k,v in current.items() if k != "risk_cash"}:
                raise ValueError("Context candle interval, stop and trailing parameters differ from the resolver")
            points = [SwingPoint(**{k: datetime.fromisoformat(v) if k.endswith("_at") or k == "pair_start" else v for k, v in s.items()})
                      for s in json.loads(row["swings_json"])]
            extra = {}
            if resolver.metadata['opportunity_config'].get('stop_variants', 1) > 1:
                extra['stop_buffer_pips'] = row['stop_buffer_pips']
            if model_type in ('swing-grid', 'swing-nn') and resolver.grid_normalization == 'candle-close-mean':
                from ultimateBot.backtester.research.swing_candle_prices import CandleCloseIndex
                as_of = datetime.fromisoformat(row['context_end'])
                prices = CandleCloseIndex(parameters, symbols=[row['symbol']],
                    start=as_of-timedelta(seconds=resolver.window.value), before=as_of,
                    interval_seconds=resolver.metadata['interval_seconds'],
                    cache_root=Path(args.database).resolve().parent/'swing_candle_close_cache')
                extra['normalization_mean'] = prices.mean(row['symbol'], as_of=as_of, window=resolver.window)
            result = resolver.predict(points, as_of=datetime.fromisoformat(row["context_end"]), symbol=row["symbol"], **extra)
        finally:
            db.close()
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
