#!/usr/bin/env python3
"""Plan, train, search and evaluate neural pnl-r regressors on dense swing grids."""
import argparse
from contextlib import closing
from dataclasses import asdict, fields, replace
from itertools import product
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultimateBot.backtester.research.swing_grid import _read_only, _sha, collection, grid_input_width
from ultimateBot.backtester.research.swing_neural import (
    NeuralConfig, NeuralData, SwingNeuralResolver, architecture_spec, atomic_json, evaluate_predictions,
    fit, frozen_test, prepare_data, recalibrate_model, RUNTIME_FIELDS,
)
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol, TrainingSelection
from ultimateBot.backtester.research.swing_series import SwingWindow
from ultimateBot.backtester.research.swing_selection import StopSelection
from ultimateBot.run_swing_ts2vec import duration

DEFAULT_DATABASE = Path('C:/TradingBotData/swing_ts2vec/research.sqlite3')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('plan', 'train', 'search'):
        child = commands.add_parser(name)
        child.add_argument('--database', type=Path)
        child.add_argument('--run-id')
        child.add_argument('--symbols', nargs='+')
        child.add_argument('--grid-normalization', choices=('swing-mean', 'candle-close-mean'))
        child.add_argument('--grid-cache-root', type=Path, help='Reuse an existing XGBoost _swing_grid_cache directory')
        child.add_argument('--output-dir', type=Path)
        if name != 'plan':
            child.add_argument('--resume', type=Path, help='Restore the recorded dataset and configuration')
        for field in fields(ResearchProtocol):
            child.add_argument('--'+field.name.replace('_', '-'))
        for field in fields(NeuralConfig):
            default = getattr(NeuralConfig(), field.name)
            choices = {'architecture': ('auto', 'mlp', 'conv1d'), 'profile': ('compact', 'balanced', 'wide')}
            extra = dict(choices=choices[field.name]) if field.name in choices else {}
            if field.name == 'min_accepted_validation':
                extra['help'] = 'Legacy compatibility option, ignored: accepted-event counts never filter validation'
            child.add_argument('--'+field.name.replace('_', '-'), type=type(default), **extra)
        if name in ('train', 'plan'):
            child.add_argument('--duration', type=duration)
        else:
            child.add_argument('--durations', type=duration, nargs='+')
            child.add_argument('--architectures', choices=('auto', 'mlp', 'conv1d'), nargs='+')
            child.add_argument('--profiles', choices=('compact', 'balanced', 'wide'), nargs='+')
            child.add_argument('--seeds', type=int, nargs='+')
    test = commands.add_parser('test')
    test.add_argument('--model', type=Path, required=True)
    test.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    test.add_argument('--run-id')
    test.add_argument('--start-date', required=True)
    test.add_argument('--end-date', required=True)
    test.add_argument('--output-dir', type=Path, required=True)
    test.add_argument('--device', default='cpu')
    test.add_argument('--threads', type=int, default=2)
    test.add_argument('--min-samples', type=int, default=30)
    recalibrate = commands.add_parser('recalibrate', help='Recompute exact thresholds from saved validation without training')
    recalibrate.add_argument('--model', type=Path, required=True, help='One frozen model or a completed search directory')
    recalibrate.add_argument('--output-dir', type=Path, required=True, help='New output directory; originals remain frozen')
    return p


def request_for(args):
    overrides = {f.name:getattr(args, f.name) for f in fields(NeuralConfig) if getattr(args, f.name) is not None}
    if getattr(args, 'resume', None):
        if args.output_dir or any(getattr(args, k, None) is not None for k in (
                'database', 'run_id', 'symbols', 'grid_normalization', 'grid_cache_root', 'duration', 'durations',
                'architectures', 'profiles', 'seeds', 'train_start', 'train_end', 'validation_start', 'validation_end')):
            raise ValueError('Resume restores data, dates and model selection; omit new data selectors')
        mutable = RUNTIME_FIELDS if args.command == 'train' else RUNTIME_FIELDS-{'epochs'}
        if set(overrides)-mutable:
            raise ValueError('Resume can change runtime settings only (and epochs for unfinished train)')
        output = args.resume.resolve()
        request = json.loads((output/'request.json').read_text(encoding='utf-8'))
        if request['command'] != args.command:
            raise ValueError('Resume command must match the recorded run')
        if args.command == 'train' and (output/'metadata.json').exists():
            raise ValueError('Model is already frozen; use a new output directory for further training')
        request['config'] = asdict(NeuralConfig(**{**request['config'], **overrides}))
        atomic_json(output/'request.json', request)
        return output, request, True
    config = NeuralConfig(**overrides)
    protocol = ResearchProtocol(**{name:getattr(args, name) or default for name, default in dict(
        train_start='2018-01-01', train_end='2022-12-31', validation_start='2023-01-01', validation_end='2023-12-31').items()})
    durations = (args.durations or [21600, 43200, 86400]) if args.command == 'search' else [args.duration or 43200]
    database = (args.database or DEFAULT_DATABASE).resolve()
    selection = TrainingSelection(symbols=tuple(args.symbols) if args.symbols else None,
                                  window_mode='duration', durations=tuple(durations))
    with closing(_read_only(database)) as db:
        run, params, windows, symbols = collection(db, args.run_id, selection)
    request = dict(command=args.command, config=asdict(config), database=str(database), run_id=run['run_id'],
        symbols=symbols, durations=[w.value for w in windows], protocol=asdict(protocol),
        source_fingerprint=params['source_fingerprint'], interval_seconds=params['config']['interval_seconds'],
        grid_normalization=args.grid_normalization or 'candle-close-mean',
        grid_cache_root=str(args.grid_cache_root.resolve()) if args.grid_cache_root else None)
    request['opportunity_config'] = params['config']
    if args.command == 'search':
        request.update(architectures=list(dict.fromkeys(args.architectures or [config.architecture])),
            profiles=list(dict.fromkeys(args.profiles or ([config.profile] if args.profile else ['compact', 'balanced']))),
            seeds=list(dict.fromkeys(args.seeds or [config.seed])))
        for seed in request['seeds']:
            replace(config, seed=seed)
    if args.command == 'plan':
        return None, request, False
    if args.output_dir is None:
        raise ValueError('Provide --output-dir for a new run, or --resume for an interrupted run')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory or resume')
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'request.json', request)
    return output, request, False


def get_data(request, folder, duration_seconds):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'data.json').exists():
        data = NeuralData.load(folder)
        source = data.source
        if (source['run_id'] != request['run_id'] or source['symbols'] != request['symbols']
                or source['source_fingerprint'] != request['source_fingerprint']
                or source['grid_normalization'] != request['grid_normalization']
                or source['protocol'] != request['protocol'] or source['window_key'] != f'duration_{duration_seconds}'):
            data.close()
            raise ValueError('Recorded neural dataset differs from the request')
        return data
    data = prepare_data(request['database'], run_id=request['run_id'], symbols=request['symbols'],
        duration=duration_seconds, protocol=ResearchProtocol(**request['protocol']),
        grid_normalization=request['grid_normalization'], cache_root=request['grid_cache_root'])
    if data.source['source_fingerprint'] != request['source_fingerprint']:
        data.close()
        raise ValueError('Collection source changed since this run was planned')
    data.snapshot(folder)
    return data


def train_command(output, request, resume):
    if (output/'metadata.json').exists():
        raise ValueError('Model is already frozen; use a new output directory for further training')
    data = get_data(request, output, request['durations'][0])
    try:
        return fit(data, output, NeuralConfig(**request['config']), resume=(output/'latest.pt').exists())
    finally:
        data.close()


def rank_models(records):
    """Common validation events; never refit a model or retune its own threshold."""
    observations, event_sets = {}, []
    for record in records:
        file = Path(record['directory'])/'validation_predictions.npz'
        if _sha(file) != record['metadata']['validation_predictions_sha256']:
            raise ValueError('Saved neural validation predictions changed')
        with np.load(file, allow_pickle=False) as data:
            rows = {str(event):(float(y), float(pred)) for event, y, pred in zip(data['event_id'], data['actual'], data['prediction'])}
            meta = record['metadata']
            own_policy = StopSelection.from_event_ids(data['event_id'], meta['opportunity_config'])
            if own_policy is not None:
                record['own_execution_validation'] = evaluate_predictions(data['actual'], data['prediction'],
                    meta['threshold'], meta['normalization']['target_mean'], own_policy)
        observations[record['directory']] = rows
        event_sets.append(set(rows))
    common = sorted(set.intersection(*event_sets))
    minimum = max(record['metadata']['config']['min_validation_samples'] for record in records)
    if len(common) < minimum:
        raise ValueError(f'Too few common neural validation events: {len(common)} < {minimum}')
    y = np.asarray([observations[records[0]['directory']][event][0] for event in common])
    groups = {}
    for record in records:
        meta = record['metadata']
        rows = observations[record['directory']]
        if any(rows[event][0] != value for event, value in zip(common, y)):
            raise ValueError('Common neural validation outcomes differ')
        pred = np.asarray([rows[event][1] for event in common])
        selection = StopSelection.from_event_ids(common, meta['opportunity_config'])
        evaluation = evaluate_predictions(y, pred, meta['threshold'], meta['normalization']['target_mean'], selection)
        record['comparison_validation'] = evaluation
        key = (meta['window_key'], meta['architecture']['architecture'], meta['config']['profile'])
        groups.setdefault(key, []).append(record)
    leaderboard = []
    for key, values in groups.items():
        errors = [value['comparison_validation']['prediction_mse'] for value in values]
        median = float(np.median(errors))
        representative = min(values, key=lambda r:(abs(r['comparison_validation']['prediction_mse']-median), r['metadata']['config']['seed']))
        leaderboard.append(dict(window_key=key[0], architecture=key[1], profile=key[2], seeds=len(values),
            median_validation_mse=median, worst_validation_mse=max(errors),
            median_validation_r=float(np.median([v['comparison_validation']['selected']['total_r'] for v in values])),
            min_validation_r=min(v['comparison_validation']['selected']['total_r'] for v in values),
            max_validation_r=max(v['comparison_validation']['selected']['total_r'] for v in values),
            active_seeds=sum(v['comparison_validation']['selected']['opportunities'] > 0 for v in values),
            representative=representative['directory']))
    leaderboard.sort(key=lambda r:(r['median_validation_mse'], r['worst_validation_mse'], -r['median_validation_r']))
    selected = next(r for r in records if r['directory'] == leaderboard[0]['representative'])
    selected_meta = selected['metadata']
    comparison = selected['comparison_validation']
    own = selected.get('own_execution_validation')
    own_qualified = (own['improvement_over_constant'] > 0 and own['selected']['total_r'] > 0) if own else selected_meta['validation_qualified']
    qualified = (own_qualified and comparison['improvement_over_constant'] > 0
        and comparison['selected']['total_r'] > 0)
    return dict(leaderboard=leaderboard, common_validation_events=len(common),
        common_validation_swings=comparison.get('complete_swings', len(common)), selected_directory=selected['directory'],
        selected_threshold=selected_meta['threshold'], validation_qualified=bool(qualified),
        ranking='median_common_validation_mse_then_worst_mse_then_median_R',
        selected_comparison_validation=comparison, threshold_reselected=False, test_used_for_selection=False)


def search_plan(request):
    """Resolve all effective candidates without loading cached matrices or torch."""
    config = NeuralConfig(**request['config'])
    trials, names, counts = [], set(), {}
    for seconds in request['durations']:
        dims = grid_input_width(SwingWindow('duration', seconds), request['interval_seconds'], request.get('opportunity_config'))
        for architecture, profile, seed in product(request['architectures'], request['profiles'], request['seeds']):
            candidate = replace(config, architecture=architecture, profile=profile, seed=seed)
            spec = architecture_spec(dims, candidate)
            name = f"duration_{seconds}/{spec['architecture']}-{profile}-seed{seed}"
            if name in names:
                continue
            names.add(name)
            trials.append(dict(name=name, window_seconds=seconds, input_dims=dims,
                               architecture=spec['architecture'], profile=profile, seed=seed,
                               parameters=spec['parameters']))
            counts[spec['architecture']] = counts.get(spec['architecture'], 0)+1
    return dict(trial_count=len(trials), counts_by_architecture=counts,
                device=config.device, threads=config.threads,
                max_epochs_per_trial=config.epochs, patience=config.patience, trials=trials)


def search_command(output, request, resume):
    config = NeuralConfig(**request['config'])
    plan = search_plan(request)
    atomic_json(output/'plan.json', plan)
    print('[swing-nn-search-plan] '+json.dumps({k:v for k, v in plan.items() if k != 'trials'}), flush=True)
    for seconds in dict.fromkeys(request['durations']):
        trials = [t for t in plan['trials'] if t['window_seconds'] == seconds]
        print(f"[swing-nn-search-window] duration_s={seconds} input_dims={trials[0]['input_dims']} "
              f"architectures={','.join(dict.fromkeys(t['architecture'] for t in trials))} "
              f"trials={len(trials)}", flush=True)
    records, names = [], set()
    for seconds in request['durations']:
        data_started = monotonic()
        data = get_data(request, output/'datasets'/f'duration_{seconds}', seconds)
        print(f'[swing-nn-data-ready] duration_s={seconds} elapsed_s={monotonic()-data_started:.1f}', flush=True)
        try:
            for architecture, profile, seed in product(request['architectures'], request['profiles'], request['seeds']):
                candidate = replace(config, architecture=architecture, profile=profile, seed=seed)
                spec = architecture_spec(data.source['input_dims'], candidate)
                name = f"duration_{seconds}/{spec['architecture']}-{profile}-seed{seed}"
                if name in names:
                    continue
                names.add(name)
                folder = output/'trials'/name
                mode = 'reuse' if (folder/'metadata.json').exists() else 'resume' if (folder/'latest.pt').exists() else 'new'
                trial_started = monotonic()
                print(f"[swing-nn-search-trial] trial={len(names)}/{plan['trial_count']} name={name} mode={mode}", flush=True)
                if (folder/'metadata.json').exists():
                    meta = SwingNeuralResolver(folder, threads=config.threads, device=config.device).metadata
                    if (meta['source'] != data.source or any(meta['config'][k] != v for k, v in asdict(candidate).items()
                                                           if k not in RUNTIME_FIELDS)):
                        raise ValueError('Completed neural trial differs from the search request')
                else:
                    folder.mkdir(parents=True, exist_ok=True)
                    data.snapshot(folder)
                    meta = fit(data, folder, candidate, resume=(folder/'latest.pt').exists())
                records.append(dict(directory=str(folder), metadata=meta))
                atomic_json(output/'search.json', dict(status='running', completed=[r['directory'] for r in records], request=request))
                print(f"[swing-nn-search-trial-done] trial={len(names)}/{plan['trial_count']} "
                      f"elapsed_s={monotonic()-trial_started:.1f} mode={mode}", flush=True)
        finally:
            data.close()
    return publish_search(output, request, records)


def publish_search(output, request, records):
    result = rank_models(records)
    for record in records:
        meta = record['metadata']
        record['own_validation'] = record.get('own_execution_validation', meta['validation'])['selected']
        record['threshold'] = meta['threshold']
        record['selection_reason'] = meta.get('threshold_selection_diagnostics', {}).get('reason', 'legacy_saved_selection')
    result.update(directory=str(output), status='completed', request=request,
                  trials=[{k:v for k,v in r.items() if k!='metadata'} for r in records])
    atomic_json(output/'search.json', result)
    atomic_json(output/'selected.json', dict(directory=result['selected_directory'], model_type='swing-nn',
        threshold=result['selected_threshold'], validation_qualified=result['validation_qualified']))
    lines = ['# Recherche des configurations neuronales', '',
        f"**Modèle retenu : `{result['selected_directory']}`.**",
        f"**Seuil initial : `{result['selected_threshold']:.9g}` R.** Qualification : {result['validation_qualified']}.", '',
        f"Événements de validation communs : {result['common_validation_events']}.",
        f"Swings communs évaluables : {result['common_validation_swings']}.",
        'Chaque durée apprend sur sa propre cohorte. La comparaison utilise des prédictions et seuils figés.',
        'Classement par MSE médiane sur la validation commune, puis pire MSE. Les R départagent les égalités.',
        'La répétition représentative est la plus proche de la médiane, pas la meilleure graine.',
        'Cette validation choisit les configurations ; aucun test indépendant n’est utilisé.', '',
        '| Fenêtre | Architecture | Profil | Graines actives/total | MSE médiane | R médian | R minimum | R maximum |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in result['leaderboard']:
        lines.append(f"| {row['window_key']} | {row['architecture']} | {row['profile']} | {row['active_seeds']}/{row['seeds']} | "
                     f"{row['median_validation_mse']:.8g} | {row['median_validation_r']:.3f} | "
                     f"{row['min_validation_r']:.3f} | {row['max_validation_r']:.3f} |")
    lines += ['', '## Résultats individuels', '',
        'Les R médians nuls peuvent masquer des graines positives et des abstentions. La validation commune peut exclure des événements de la validation propre.', '',
        '| Modèle | Seuil | N propre | R propre | N commun | R commun | Décision du seuil |',
        '|---|---:|---:|---:|---:|---:|---|']
    for record in records:
        name = Path(record['directory']).relative_to(output/'trials')
        own, common = record['own_validation'], record['comparison_validation']['selected']
        lines.append(f"| {name} | {record['threshold']:.8g} | {own['opportunities']} | {own['total_r']:.3f} | "
                     f"{common['opportunities']} | {common['total_r']:.3f} | {record['selection_reason']} |")
    (output/'rapport.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def recalibrate_command(model, output):
    model, output = Path(model).resolve(), Path(output).resolve()
    if not (model/'search.json').is_file():
        return recalibrate_model(model, output)
    original = json.loads((model/'search.json').read_text(encoding='utf-8'))
    if original['status'] != 'completed':
        raise ValueError('Only completed neural searches can be recalibrated')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Recalibration requires a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    request = dict(command='recalibrate', source_search=str(model), method='per_model_exact_threshold_selection_v1',
                   original_request=original['request'], training_performed=False, validation_reused=True)
    atomic_json(output/'request.json', request)
    records = []
    for trial in original['trials']:
        source = Path(trial['directory'])
        relative = source.relative_to(model/'trials')
        destination = output/'trials'/relative
        meta = recalibrate_model(source, destination)
        records.append(dict(directory=str(destination), metadata=meta))
        print(f"[swing-nn-recalibrate] model={relative} accepted={meta['validation']['selected']['opportunities']} "
              f"total_r={meta['validation']['selected']['total_r']:.3f} reason={meta['threshold_selection_diagnostics']['reason']}", flush=True)
    return publish_search(output, request, records)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == 'test':
            result = frozen_test(args.database, directory=args.model, run_id=args.run_id, start=args.start_date,
                end=args.end_date, output_dir=args.output_dir, threads=args.threads, device=args.device, min_samples=args.min_samples)
        elif args.command == 'recalibrate':
            result = recalibrate_command(args.model, args.output_dir)
        else:
            output, request, resume = request_for(args)
            if args.command == 'plan':
                from ultimateBot.backtester.research.swing_series import SwingWindow
                dims = grid_input_width(SwingWindow('duration', request['durations'][0]), request['interval_seconds'], request.get('opportunity_config'))
                result = dict(request=request, architecture=architecture_spec(dims, NeuralConfig(**request['config'])))
            elif args.command == 'train':
                result = train_command(output, request, resume)
            else:
                result = search_command(output, request, resume)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    summary = result if args.command in ('test', 'plan') else {
        k:result[k] for k in ('model_type', 'best_epoch', 'threshold', 'validation_qualified',
                             'directory', 'selected_directory', 'selected_threshold', 'leaderboard') if k in result}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
