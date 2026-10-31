"""Repeated window search on a shared validation cohort; never evaluate the test set."""
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np

from ultimateBot.backtester.research.swing_contexts import ensure_windows, context_build_plan
from ultimateBot.backtester.research.swing_encoder import SwingEncoderConfig, encoder_training_plan
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, connect, json_dump
from ultimateBot.backtester.research.swing_resolver import (
    ResearchProtocol, ResolverConfig, TrainingSelection, select_collection, train,
    selection_objective, validation_score, validation_has_gain,
)
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, FEATURE_SCHEMAS, VOLUME_FEATURE_SCHEMA, SWING_SCHEMA, SwingWindow, model_feature_schema


def search_plan(database, *, run_id=None, config=ResolverConfig(), encoder_config=SwingEncoderConfig(),
                protocol=ResearchProtocol(), selection=TrainingSelection(), repeats=3, representation='ts2vec', cohort_policy='common',
                grid_normalization='swing-mean', max_depths=None):
    depths = tuple(max_depths) if max_depths is not None else (config.max_depth,)
    if not depths or any(type(depth) is not int or depth < 1 for depth in depths):
        raise ValueError('Search max_depths must contain positive integers')
    depths = tuple(sorted(set(depths)))
    config = replace(config, max_depth=depths[0])
    if grid_normalization != 'swing-mean' and representation != 'swing-grid':
        raise ValueError('Grid normalization applies only to swing-grid')
    if cohort_policy not in ('common', 'per-window') or (cohort_policy != 'common' and representation != 'swing-grid'):
        raise ValueError('Per-window cohorts are supported only for swing-grid')
    if representation not in ('ts2vec', 'swing-grid'):
        raise ValueError('Unknown swing representation')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Search repeats must be a positive integer")
    if not Path(database).is_file():
        raise FileNotFoundError("Collect the swing database before searching windows")
    # A preview must not create tables, migrate the database or write contexts.
    db = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        if representation == 'swing-grid':
            from ultimateBot.backtester.research.swing_grid import collection
            run, parameters, _, symbols = collection(db, run_id, selection)
            selection = replace(selection, window_mode='duration')
        else:
            base = replace(selection, window_mode='both', durations=None, pairs=None)
            run, parameters, _, symbols = select_collection(db, run_id, base)
    finally:
        db.close()
    if (parameters.get('feature_schema') not in FEATURE_SCHEMAS or parameters.get('swing_schema') != SWING_SCHEMA
            or parameters.get('opportunity_schema') != OPPORTUNITY_SCHEMA
            or parameters.get('risk_management_filters') is not False):
        raise ValueError("Incompatible swing collection for window search")
    explicit = selection.durations is not None or selection.pairs is not None
    windows = []
    for mode, values in (('duration', selection.durations), ('pairs', selection.pairs)):
        if selection.window_mode not in ('both', mode):
            continue
        if values is not None:
            windows.extend(SwingWindow(mode, value) for value in values)
        elif not explicit:
            windows.extend(SwingWindow(**w) for w in parameters['windows'] if w['mode'] == mode)
    if not windows:
        raise ValueError("No windows selected; provide --durations, --pairs or a range")
    # An explicit limit is a contract, not permission to silently crop a window.
    if representation == 'ts2vec' and encoder_config.max_train_length and any(w.mode == 'pairs' and 2*w.value > encoder_config.max_train_length for w in windows):
        raise ValueError('Pair window exceeds the explicit TS2Vec length limit; use --ts2vec-max-train-length 0')
    replace(config, seed=config.seed + repeats - 1)
    replace(encoder_config, seed=encoder_config.seed + repeats - 1)
    plan = {'run_id': run['run_id'], 'database': str(Path(database).resolve()),
        'source_fingerprint': parameters['source_fingerprint'], 'opportunity_config': parameters['config'],
        'symbols': symbols, 'interval_seconds': parameters['config']['interval_seconds'],
        'windows': [asdict(w) for w in windows], 'repeats': repeats,
        'model_pairs_to_train': len(windows)*repeats*len(depths),
        'config': asdict(config), 'encoder_config': asdict(encoder_config), 'protocol': asdict(protocol),
        'test_evaluated': False, 'ranking': 'median_validation_total_r_then_worst_then_coverage',
        'representative': 'closest_to_validation_median_not_best_seed'}
    # A single depth retains the old identity, including explicit --max-depths 3.
    # Multiple depths are distinct candidates, never extra repeats of one model.
    if len(depths) > 1:
        plan['max_depths'] = list(depths)
    if config.target != 'pnl-r':
        plan['validation_objective'] = selection_objective(config.target)
        plan['ranking'] = 'median_validation_mean_label_then_worst_then_coverage'
        plan['representative'] = 'closest_to_target_score_median_not_best_seed'
    effective_schema = (FEATURE_SCHEMA if representation == 'swing-grid' else
                        model_feature_schema(parameters['feature_schema'], selection.include_volume))
    # Keep old six-feature resume identities stable. New volume collections
    # distinguish full input from the price-only comparison on the same cohort.
    if parameters['feature_schema'] == VOLUME_FEATURE_SCHEMA:
        plan['feature_schema'] = effective_schema
    if representation == 'swing-grid':
        from ultimateBot.backtester.research.swing_grid import grid_schema, grid_input_width, normalization_description
        plan.update(representation=representation, grid_schema=grid_schema(grid_normalization, parameters['config']),
            grid_preflight={w.key: {'input_dims': grid_input_width(w, plan['interval_seconds'], parameters['config']),
                'normalization': 'price / mean(actual window swing prices) - 1', 'holes': 0,
                'encoder_training': False, 'dense_input': True} for w in windows})
        if grid_normalization != 'swing-mean':
            plan['grid_normalization'] = grid_normalization
            for entry in plan['grid_preflight'].values():
                entry['normalization'] = normalization_description(grid_normalization)
    # Omit the legacy default so existing saved search identities remain valid.
    if cohort_policy != 'common':
        plan['cohort_policy'] = cohort_policy
    plan['plan_sha256'] = hashlib.sha256(json_dump(plan).encode()).hexdigest()
    if representation == 'swing-grid':
        return plan
    # Workload observations can change when missing contexts are materialized.
    # They are not part of the resume identity, which contains the configuration.
    db = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        train_start, train_end, _, _ = protocol.bounds
        build_plan = context_build_plan(db, run['run_id'], windows, start=train_start,
                                        before=protocol.bounds[3], symbols=symbols)
        plan['context_build_preflight'] = build_plan
        keys = [w.key for w in windows]
        counts = dict.fromkeys(keys, 0)
        placeholders = ','.join('?' for _ in keys)
        symbol_placeholders = ','.join('?' for _ in symbols)
        # Count from the covering (event_id,window_key) index. Reading every
        # context row's length can page through tens of GB of feature blobs.
        for key, count in db.execute(f'''SELECT c.window_key,count(*) FROM swing_events e
            JOIN swing_contexts c USING(event_id) WHERE e.run_id=? AND e.symbol IN ({symbol_placeholders})
            AND c.window_key IN ({placeholders}) AND e.confirmed_at>=? AND e.confirmed_at<?
            GROUP BY c.window_key''',
            (run['run_id'], *symbols, *keys, train_start.isoformat(), train_end.isoformat())):
            counts[key] = count
        available_keys = {SwingWindow(**w).key for w in parameters['windows']}
        preflight = {}
        for window in windows:
            count = counts[window.key]
            missing_contexts = any(build_plan['windows'][window.key]['missing_ranges'].values())
            if not count or window.key not in available_keys or missing_contexts:
                status = 'pending_context_build' if window.key not in available_keys or missing_contexts else 'no_training_contexts'
                preflight[window.key] = {'status': status, 'sequence_policy': 'full_window',
                    'existing_training_contexts': count,
                    'note': 'Exact workload will be printed after contexts are built, before encoder training'}
                continue
            # At most one unambiguous swing per candle confirmation. Pairs have
            # an exact length; duration windows have this data-independent bound.
            maximum = 2*window.value if window.mode == 'pairs' else max(2, math.ceil(window.value/plan['interval_seconds']))
            if encoder_config.max_train_length:
                maximum = min(maximum, encoder_config.max_train_length)
            estimate = encoder_training_plan(encoder_config, np.full(count, maximum, dtype=np.int64),
                                             feature_schema=effective_schema)
            estimate['length_basis'] = 'exact_pair_count' if window.mode == 'pairs' else 'duration_candle_upper_bound'
            estimate['status'] = 'ready_for_exact_planning_at_fit'
            estimate['length_limit_checked_at_fit'] = bool(encoder_config.max_train_length)
            if window.mode == 'duration':
                estimate['warnings'] = [warning.replace('shorter_than_training_window', 'shorter_than_duration_upper_bound')
                                        for warning in estimate['warnings']]
                estimate['sequence_length_min'] = None
                estimate['sequence_length_max_upper_bound'] = estimate.pop('sequence_length_max')
                estimate['padded_training_length_upper_bound'] = estimate.pop('padded_training_length')
                estimate['training_buffer_bytes_upper_bound'] = estimate.pop('training_buffer_bytes')
                estimate['depth_upper_bound'] = estimate.pop('depth')
                estimate['receptive_field_swings_upper_bound'] = estimate.pop('receptive_field_swings')
            preflight[window.key] = estimate
        plan['encoder_preflight'] = preflight
    finally:
        db.close()
    return plan


def rank_windows(trials, keys):
    rows = []
    target = trials[0].get('config', {}).get('target', 'pnl-r')
    if any(trial.get('config', {}).get('target', 'pnl-r') != target for trial in trials):
        raise ValueError('Cannot rank different target labels together')
    for key in keys:
        evaluations = [trial['windows'][key]['report'].get('comparison_validation',
                       trial['windows'][key]['report']['partitions']['validation']) for trial in trials]
        values = [evaluation['selected'] for evaluation in evaluations]
        totals = [v['total_r'] for v in values]
        median = float(np.median(totals))
        scores = [validation_score(evaluation, target) for evaluation in evaluations]
        median_score = float(np.median(scores))
        representative = min(range(len(trials)), key=lambda i: (abs(scores[i]-median_score), i))
        mode, value = key.rsplit('_', 1)
        rows.append({'window_key': key, 'mode': mode, 'value': int(value), 'repeats': len(values),
            'median_validation_r': median, 'worst_validation_r': float(min(totals)),
            'best_validation_r': float(max(totals)), 'std_validation_r': float(np.std(totals)),
            'median_accepted': float(np.median([v['opportunities'] for v in values])),
            'median_coverage': float(np.median([v['coverage'] for v in values])),
            'positive_repeats': sum(v > 0 for v in totals), 'representative_trial': representative,
            'predictive_gain_repeats': sum(bool(t['windows'][key]['report'].get('learning_diagnostics', {}).get('validation_predictive_gain')) for t in trials),
            'selection_metric': selection_objective(target)['metric'], 'target': target,
            'median_validation_score': median_score, 'worst_validation_score': float(min(scores)),
            'best_validation_score': float(max(scores)), 'std_validation_score': float(np.std(scores)),
            'positive_objective_repeats': sum(score > selection_objective(target)['abstention_score'] for score in scores),
            'directory': trials[representative]['windows'][key]['directory']})
        reports = [trial['windows'][key]['report'] for trial in trials]
        if all('comparison_validation' in report for report in reports):
            own = [report['partitions']['validation'] for report in reports]
            rows[-1].update(median_own_validation_r=float(np.median([v['selected']['total_r'] for v in own])),
                median_own_validation_score=float(np.median([validation_score(v, target) for v in own])),
                median_own_accepted=float(np.median([v['selected']['opportunities'] for v in own])),
                own_validation_count=own[0]['rows'], comparison_validation_count=reports[0]['comparison_validation']['rows'],
                median_trees_used=float(np.median([r['learning_diagnostics']['trees_used'] for r in reports])))
    return sorted(rows, key=_rank_key)


def _rank_key(row):
    return (-row['median_validation_score'], -row['worst_validation_score'], -row['median_coverage'], row['window_key'])


def rank_search_trials(trials, keys, *, depth_grid=False):
    if not depth_grid:
        return rank_windows(trials, keys)
    rows = []
    for depth in sorted({trial['config']['max_depth'] for trial in trials}):
        indices = [i for i, trial in enumerate(trials) if trial['config']['max_depth'] == depth]
        for row in rank_windows([trials[i] for i in indices], keys):
            row['max_depth'] = depth
            row['representative_trial'] = indices[row['representative_trial']]
            rows.append(row)
    return sorted(rows, key=lambda row: (*_rank_key(row), row['max_depth']))


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json_dump(value), encoding='utf-8')
    temporary.replace(path)


def _save(db, directory, report, status, error=None):
    _write_json(directory/'search.json', report)
    with db:
        db.execute("UPDATE swing_window_searches SET status=?,report_json=?,error=? WHERE search_id=?",
                   (status, json_dump(report), error, report['search_id']))


def search(database, *, output_dir, run_id=None, config=ResolverConfig(), encoder_config=SwingEncoderConfig(),
           protocol=ResearchProtocol(), selection=TrainingSelection(), repeats=3, resume=None, dry_run=False,
           representation='ts2vec', cohort_policy='common', grid_normalization='swing-mean', max_depths=None):
    plan = search_plan(database, run_id=run_id, config=config, encoder_config=encoder_config,
                       protocol=protocol, selection=selection, repeats=repeats, representation=representation,
                       cohort_policy=cohort_policy, grid_normalization=grid_normalization, max_depths=max_depths)
    if dry_run:
        return plan
    windows = tuple(SwingWindow(**w) for w in plan['windows'])
    keys = [w.key for w in windows]
    config = ResolverConfig(**plan['config'])
    depths = plan.get('max_depths', [config.max_depth])
    depth_grid = len(depths) > 1
    encoder_config = SwingEncoderConfig(**plan['encoder_config'])
    fixed_selection = TrainingSelection(symbols=tuple(plan['symbols']), interval_seconds=plan['interval_seconds'],
        include_volume=selection.include_volume,
        durations=tuple(w.value for w in windows if w.mode == 'duration') or None,
        pairs=tuple(w.value for w in windows if w.mode == 'pairs') or None)
    db = connect(database)
    active = False
    try:
        if resume is not None:
            directory = Path(resume).resolve()
            report = json.loads((directory/'search.json').read_text(encoding='utf-8'))
            if config.target != 'pnl-r' and 'validation_objective' not in report['plan']:
                raise ValueError('This search used the previous PnL threshold objective; start a new search for mean target labels')
            if report['plan']['plan_sha256'] != plan['plan_sha256']:
                raise ValueError("Resume requires the same collection, windows, depths, dates, configurations and repeats")
            record = db.execute("SELECT * FROM swing_window_searches WHERE search_id=?", (report['search_id'],)).fetchone()
            if record is None or Path(record['directory']) != directory:
                raise ValueError("Search is not registered in this database")
        else:
            search_id = uuid4().hex
            directory = Path(output_dir).resolve()/f"search_{plan['run_id'][:12]}_{search_id[:12]}"
            directory.mkdir(parents=True, exist_ok=False)
            report = {'search_id': search_id, 'directory': str(directory), 'plan': plan, 'trials': [],
                      'test_evaluated': False, 'status': 'running'}
            if representation == 'swing-grid':
                report['representation'] = representation
            if cohort_policy != 'common':
                report['cohort_policy'] = cohort_policy
            with db:
                db.execute("INSERT INTO swing_window_searches(search_id,run_id,status,directory,parameters_json) VALUES(?,?,'running',?,?)",
                    (search_id, plan['run_id'], str(directory), json_dump(plan)))
        print(f"[swing-search] directory={directory} windows={len(windows)} repeats={repeats} "
              f"max_depths={depths} model_pairs={plan['model_pairs_to_train']} test=not_evaluated", flush=True)
        active = True
        report['status'] = 'running'
        _save(db, directory, report, 'running')
        if representation == 'swing-grid':
            report['context_build'] = {'method': 'read_only_scalar_grid_cache', 'context_blobs_created': False}
        else:
            report['context_build'] = ensure_windows(db, plan['run_id'], windows, start=protocol.bounds[0],
                                                    before=protocol.bounds[3], symbols=plan['symbols'])
        # Use identical seed sequences and cached cohorts at every depth.
        trials = []
        jobs = [(depth, repeat) for depth in depths for repeat in range(repeats)]
        for index, (depth, repeat) in enumerate(jobs):
            candidate = replace(config, max_depth=depth, seed=config.seed+repeat)
            candidate_encoder = replace(encoder_config, seed=encoder_config.seed+repeat)
            if index < len(report['trials']):
                saved = report['trials'][index]
                trial = json.loads(Path(saved['comparison']).read_text(encoding='utf-8'))
                if (saved['repeat'] != repeat+1 or saved.get('max_depth', depth) != depth
                        or trial['config'] != asdict(candidate)
                        or trial['encoder_config'] != asdict(candidate_encoder)
                        or set(trial['windows']) != set(keys)
                        or trial.get('cohort_policy', 'common') != cohort_policy
                        or trial.get('grid_normalization', 'swing-mean') != grid_normalization
                        or trial['run_id'] != plan['run_id'] or trial['protocol'] != asdict(protocol)
                        or config.target != 'pnl-r' and trial.get('target_definition', {}).get('threshold_selection_metric')
                           != 'validation_mean_target_label'):
                    raise ValueError("Saved trial does not match the search plan")
                print(f"[swing-search] trial={index+1}/{len(jobs)} max_depth={depth} "
                      f"repeat={repeat+1}/{repeats} reuse={trial['experiment_id']}", flush=True)
            else:
                print(f"[swing-search] trial={index+1}/{len(jobs)} max_depth={depth} repeat={repeat+1}/{repeats} "
                      f"encoder_seed={candidate_encoder.seed} resolver_seed={candidate.seed}", flush=True)
                trial = train(database, run_id=plan['run_id'], output_dir=directory/'trials',
                    config=candidate, encoder_config=candidate_encoder,
                    protocol=protocol, selection=fixed_selection,
                    **({'representation': representation} if representation != 'ts2vec' else {}),
                    **({'cohort_policy': cohort_policy} if cohort_policy != 'common' else {}),
                    **({'grid_normalization': grid_normalization} if grid_normalization != 'swing-mean' else {}))
                entry = {'repeat': repeat+1, 'experiment_id': trial['experiment_id'],
                    'comparison': str(Path(trial['directory'])/'comparison.json')}
                if depth_grid:
                    entry['max_depth'] = depth
                report['trials'].append(entry)
                _save(db, directory, report, 'running')
            trials.append(trial)
            if trial['dataset_sha256'] != trials[0]['dataset_sha256']:
                raise ValueError("Collection/cohort changed between repeats; start a new search")
            report['leaderboard'] = rank_search_trials(trials, keys, depth_grid=depth_grid)
            _save(db, directory, report, 'running')
        selected = report['leaderboard'][0]
        representative = trials[selected['representative_trial']]['windows'][selected['window_key']]
        metadata = json.loads((Path(selected['directory'])/'metadata.json').read_text(encoding='utf-8'))
        selected_validation = representative['report'].get('comparison_validation', representative['report']['partitions']['validation'])
        positive = selected_validation['selected']['total_r'] > 0
        predictive_gain = representative['report'].get('learning_diagnostics', {}).get('validation_predictive_gain')
        report.update(status='completed', selected_window_validation_only=selected['window_key'],
            selection_objective=selection_objective(config.target),
            selected_validation_score=validation_score(selected_validation, config.target),
            selected_max_depth=selected.get('max_depth', config.max_depth),
            selected_directory=selected['directory'], positive_validation=positive,
            selected_threshold=metadata['threshold'], selected_target=metadata['target'],
            validation_predictive_gain=predictive_gain,
            validation_qualified=bool(predictive_gain and validation_has_gain(selected_validation, config.target)
                and selected['median_validation_score'] > selection_objective(config.target)['abstention_score']
                and (cohort_policy == 'common' or representative['report']['validation_qualified'])),
            positive_median_validation=selected['median_validation_r'] > 0)
        _write_json(directory/'selected.json', {'window_key': selected['window_key'],
            'selection_objective': report['selection_objective'], 'selected_validation_score': report['selected_validation_score'],
            'max_depth': report['selected_max_depth'],
            'threshold': report['selected_threshold'], 'target': report['selected_target'],
            'directory': selected['directory'], 'positive_validation': positive, 'search_id': report['search_id'],
            'scope': metadata['scope'], 'test_evaluated': False,
            'validation_predictive_gain': predictive_gain, 'validation_qualified': report['validation_qualified'],
            'selection_rule': plan['ranking'], 'median_validation_r': selected['median_validation_r']})
        with (directory/'leaderboard.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected))
            writer.writeheader()
            writer.writerows(report['leaderboard'])
        lines = ['# Recherche répétée des fenêtres de swings', '',
            f"Fenêtre retenue sur validation : `{selected['window_key']}`.", '',
            f"Profondeur XGBoost retenue : **{report['selected_max_depth']}**. "
            f"Profondeurs comparées : `{depths}`.", '',
            f"Objectif de validation : `{report['selection_objective']['metric']}` pour `{config.target}`. "
            f"Score retenu : **{report['selected_validation_score']:.9g}**.", '',
            f"**Seuil initial du modèle retenu : `{report['selected_threshold']!r}`** "
            f"(cible `{report['selected_target']}`).", '',
            f"Option de backtest : `--{'swing-grid' if representation == 'swing-grid' else 'swing-ts2vec'}-threshold {report['selected_threshold']!r}`. "
            'Sans cette option, le robot utilise déjà ce seuil enregistré dans le modèle.', '',
            ('Chaque fenetre utilise directement sa grille dense dans XGBoost, sans TS2Vec. Les grilles sont mises en cache une fois et reutilisees entre repetitions.'
             if representation == 'swing-grid' else 'Chaque fenêtre possède son TS2Vec et son XGBoost. Même cohorte pour toutes les fenêtres et répétitions.'),
            ('Apprentissage, early stopping et seuil propres a chaque fenetre. Classement sur une validation commune, sans rechoisir les seuils.'
             if cohort_policy == 'per-window' else 'Meme cohorte pour toutes les fenetres et repetitions.'), '',
            *([f"Normalisation : `{grid_normalization}`.", ''] if representation == 'swing-grid' else []),
            'Classement : médiane du score de validation dans les unités de la cible, puis pire répétition et couverture médiane.',
            'Modèle exporté : répétition la plus proche de la médiane, avec son seuil propre figé.', '',
            f"Qualification sur validation (gain prédictif et score cible au-dessus du neutre) : `{report['validation_qualified']}`. Ce diagnostic n'est pas une certification de performance.", '',
            '| Fenêtre / profondeur | Répétitions | Score cible médian | Pire score cible | R médian descriptif | N accepté médian |',
            '|---|---:|---:|---:|---:|---:|']
        for row in report['leaderboard']:
            name = f"{row['window_key']} / {row.get('max_depth', config.max_depth)}"
            lines.append(f"| {name} | {row['repeats']} | {row['median_validation_score']:.9g} | "
                         f"{row['worst_validation_score']:.9g} | {row['median_validation_r']:.3f} | {row['median_accepted']:g} |")
        if cohort_policy == 'per-window':
            lines += ['', '## Resultat propre et score de classement', '',
                'Le resultat propre, les arbres et le seuil restent stables lorsque la liste de durees change. '
                'Le score commun peut changer car les evenements comparables changent. Ce score ne reentraine pas les modeles.', '',
                '| Fenetre / profondeur | N validation propre | N commun | Score cible propre median | Score cible commun median | Arbres medians |',
                '|---|---:|---:|---:|---:|---:|']
            for row in report['leaderboard']:
                name = f"{row['window_key']} / {row.get('max_depth', config.max_depth)}"
                lines.append(f"| {name} | {row['own_validation_count']} | {row['comparison_validation_count']} | "
                             f"{row['median_own_validation_score']:.9g} | {row['median_validation_score']:.9g} | {row['median_trees_used']:g} |")
        lines += ['', 'Aucune évaluation de test. La fenêtre est la meilleure parmi la grille explorée, sans garantie d’optimalité globale.',
            'Répéter les graines mesure une variabilité d’apprentissage ; ce ne sont pas des validations temporelles indépendantes.',
            'Les opportunités peuvent se chevaucher : une somme R est distincte du rendement d’un portefeuille.',
            'Multiplier les essais sur une même validation peut surajuster cette validation. Le test final reste séparé.']
        (directory/'rapport.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        _save(db, directory, report, 'completed')
        return report
    except BaseException as error:
        if active:
            report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            _save(db, directory, report, report['status'], str(error))
        raise
    finally:
        db.close()
