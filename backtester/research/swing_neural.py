"""Supervised neural regression on the exact dense swing grids used by XGBoost."""
from dataclasses import asdict, dataclass, replace
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from time import monotonic

import numpy as np

from ultimateBot.backtester.research.swing_grid import (
    GridData, _read_only, _sha, collection, grid_schema, grid_vector, grid_input_width, append_stop_input, prepare,
)
from ultimateBot.backtester.research.swing_informer import atomic_json, atomic_torch, evaluate_predictions
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, OpportunityConfig, uses_stop_distance
from ultimateBot.backtester.research.swing_resolver import ResearchProtocol, TrainingSelection, resolver_directory
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, SWING_SCHEMA, SwingWindow
from ultimateBot.backtester.research.swing_selection import StopSelection, acceptance_mask, exact_stop_threshold

MODEL_VERSION = 1


@dataclass(frozen=True)
class NeuralConfig:
    architecture: str = 'auto'
    profile: str = 'balanced'
    dropout: float = .15
    batch_size: int = 512
    epochs: int = 40
    patience: int = 8
    learning_rate: float = .001
    weight_decay: float = .001
    min_train_samples: int = 1000
    min_validation_samples: int = 200
    # Retained solely to read older requests/checkpoints. Accepted-event counts
    # are descriptive; this legacy setting never filters threshold candidates.
    min_accepted_validation: int = 0
    min_threshold: float = 0.
    checkpoint_batches: int = 100
    seed: int = 1
    threads: int = 2
    device: str = 'cpu'

    def __post_init__(self):
        if self.architecture not in ('auto', 'mlp', 'conv1d') or self.profile not in ('compact', 'balanced', 'wide'):
            raise ValueError('Choose architecture auto/mlp/conv1d and profile compact/balanced/wide')
        for name in ('batch_size', 'epochs', 'patience', 'min_train_samples', 'min_validation_samples',
                     'checkpoint_batches', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.min_accepted_validation) is not int or self.min_accepted_validation < 0:
            raise ValueError('Legacy min_accepted_validation must be a nonnegative integer')
        if (not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.min_threshold) or type(self.seed) is not int or not 0 <= self.seed < 2**31
                or not isinstance(self.device, str)
                or not (self.device == 'cpu' or self.device == 'cuda' or self.device.startswith('cuda:'))):
            raise ValueError('Invalid optimizer, dropout, seed, threshold or device')


def architecture_spec(input_dims, config):
    if type(input_dims) is not int or input_dims < 1:
        raise ValueError('Input dimensions must be a positive integer')
    architecture = ('mlp' if input_dims <= 128 else 'conv1d') if config.architecture == 'auto' else config.architecture
    factor = {'compact': 1, 'balanced': 2, 'wide': 4}[config.profile]
    if architecture == 'mlp':
        base = 32 if input_dims <= 64 else 64
        hidden = [min(base*factor, 512), min(base*factor//2, 256)]
        parameters = sum((a+1)*b for a, b in zip([input_dims, *hidden], [*hidden, 1]))
        return dict(version=MODEL_VERSION, architecture=architecture, input_dims=input_dims,
                    hidden=hidden, dropout=config.dropout, parameters=parameters)
    channels = [8*factor, 16*factor, 16*factor]
    bins, head = min(8, input_dims), 16*factor
    parameters = sum((a*5+1)*b for a, b in zip([1, *channels[:-1]], channels))
    parameters += (channels[-1]*bins+1)*head+head+1
    return dict(version=MODEL_VERSION, architecture=architecture, input_dims=input_dims, channels=channels,
                kernel_size=5, dilations=[1, 2, 4], pool_bins=bins, hidden=[head],
                dropout=config.dropout, parameters=parameters)


def build_model(spec):
    import torch
    nn = torch.nn
    layers = []
    if spec['architecture'] == 'mlp':
        previous = spec['input_dims']
        for width in spec['hidden']:
            layers.extend([nn.Linear(previous, width), nn.GELU(), nn.Dropout(spec['dropout'])])
            previous = width
    else:
        layers.append(nn.Unflatten(1, (1, spec['input_dims'])))
        previous = 1
        for width, dilation in zip(spec['channels'], spec['dilations']):
            layers.extend([nn.Conv1d(previous, width, spec['kernel_size'], dilation=dilation,
                                     padding=dilation*(spec['kernel_size']//2)), nn.GELU(), nn.Dropout(spec['dropout'])])
            previous = width
        layers.extend([nn.AdaptiveAvgPool1d(spec['pool_bins']), nn.Flatten(),
                       nn.Linear(previous*spec['pool_bins'], spec['hidden'][0]), nn.GELU(), nn.Dropout(spec['dropout'])])
        previous = spec['hidden'][0]
    output = nn.Linear(previous, 1)
    # The initial predictor is exactly the training mean, providing an explicit
    # baseline candidate at epoch zero instead of forcing any positive forecast.
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    model = nn.Sequential(*layers)
    if sum(p.numel() for p in model.parameters()) != spec['parameters']:
        raise ValueError('Neural parameter count differs from the recorded architecture')
    return model


class NeuralData:
    def __init__(self, grid, source):
        self.grid, self.source = grid, source
        key = source['window_key']
        if (grid.manifest['dataset_sha256'] != source['dataset_sha256']
                or grid.manifest['request']['grid_schema'] != source['grid_schema']
                or grid.manifest['request']['run_id'] != source['run_id']
                or grid.manifest['dimensions'] != {key: source['input_dims']}):
            raise ValueError('Neural dataset differs from its recorded grid snapshot')
        self.x = grid.matrices[key]
        self.y = np.asarray([r['pnl_r'] for r in grid.rows], dtype=np.float64)
        if not np.isfinite(self.y).all():
            raise ValueError('Neural labels must be finite realized pnl-r values')
        self.partitions = {phase: np.asarray([i for i, row in enumerate(grid.rows) if row['phase'] == phase], dtype=np.int64)
                           for phase in grid.manifest['request']['periods']}

    def close(self):
        self.grid.close()

    def snapshot(self, output):
        atomic_json(Path(output)/'data.json', dict(directory=str(self.grid.directory), source=self.source))

    @classmethod
    def load(cls, output):
        snapshot = json.loads((Path(output)/'data.json').read_text(encoding='utf-8'))
        return cls(GridData(snapshot['directory']), snapshot['source'])


def prepare_data(database, *, run_id, symbols, duration, protocol, grid_normalization='candle-close-mean', cache_root=None):
    selection = TrainingSelection(symbols=tuple(symbols) if symbols else None, window_mode='duration', durations=(duration,))
    with closing(_read_only(database)) as db:
        run, params, windows, symbols = collection(db, run_id, selection)
    window = windows[0]
    periods = dict(train=[protocol.train_start, protocol.train_end], validation=[protocol.validation_start, protocol.validation_end])
    data, params = prepare(database, run_id=run['run_id'], windows=(window,), symbols=symbols,
        periods=periods, target='pnl-r', cache_root=cache_root or Path(database).resolve().parent/'swing_grid_cache',
        grid_normalization=grid_normalization)
    source = dict(run_id=run['run_id'], source_fingerprint=params['source_fingerprint'],
        dataset_sha256=data.manifest['dataset_sha256'], window_key=window.key,
        input_dims=grid_input_width(window, params['config']['interval_seconds'], params['config']), interval_seconds=params['config']['interval_seconds'],
        grid_normalization=grid_normalization, grid_schema=grid_schema(grid_normalization, params['config']), symbols=symbols,
        opportunity_config=params['config'], protocol=asdict(protocol), target='pnl-r')
    return NeuralData(data, source)


def normalizer(data, indices, batch_size):
    squares, count = 0., 0
    for start in range(0, len(indices), batch_size):
        x = np.asarray(data.x[indices[start:start+batch_size]], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite neural training inputs')
        squares += float(np.sum(x*x))
        count += x.size
    y = data.y[indices]
    return dict(input_scale=max(math.sqrt(squares/count), 1e-6), target_mean=float(y.mean()),
                target_scale=max(float(y.std()), 1e-6), fit_partition='train_only', input_centering=False)


def predict_array(model, values, normalization, *, device='cpu', batch_size=512):
    import torch
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            x = torch.as_tensor(np.asarray(values[start:start+batch_size])/normalization['input_scale'],
                                dtype=torch.float32, device=device)
            result.append(model(x).flatten().cpu().numpy().astype(np.float64)*normalization['target_scale']
                          + normalization['target_mean'])
    output = np.concatenate(result) if result else np.empty(0, dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite neural predictions')
    return output


def predict_data(model, data, indices, normalization, config):
    parts = [predict_array(model, data.x[indices[start:start+config.batch_size]], normalization,
                           device=config.device, batch_size=config.batch_size)
             for start in range(0, len(indices), config.batch_size)]
    return np.concatenate(parts) if parts else np.empty(0)


def choose_threshold(y, predictions, config, selection=None):
    """Evaluate every distinct attainable acceptance set in O(n log n)."""
    if selection is not None:
        return exact_stop_threshold(y, predictions, config.min_threshold, selection)
    y, predictions = np.asarray(y, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    if (y.ndim != 1 or not len(y) or predictions.shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(predictions).all()):
        raise ValueError('Threshold selection requires finite aligned labels and predictions')
    minimum = config.min_threshold
    order = np.argsort(-predictions, kind='stable')
    scores, labels = predictions[order], y[order]
    sums = np.cumsum(labels)
    profit = np.cumsum(np.maximum(labels, 0))
    loss = np.cumsum(np.maximum(-labels, 0))
    wins = np.cumsum(labels > 0)
    # A score tie must be accepted as a whole, never split into an unattainable
    # top-k subset. This also handles constant predictions.
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])+1
    sets = {float(scores[n-1]):int(n) for n in ends if scores[n-1] >= minimum}
    sets[minimum] = int(np.searchsorted(-scores, -minimum, side='right'))
    none = max(minimum, float(np.nextafter(float(predictions.max()), np.inf)))
    sets[none] = 0
    candidates = []
    for threshold, n in sorted(sets.items()):
        total, positive, negative, win = (float(sums[n-1]), float(profit[n-1]), float(loss[n-1]), int(wins[n-1])) if n else (0., 0., 0., 0)
        candidates.append(dict(threshold=threshold, opportunities=n, total_r=total, mean_r=total/n if n else None,
            win_rate=win/n if n else None, profit_factor_r=positive/negative if negative else None,
            gross_profit_r=positive, gross_loss_r=negative))
    selected = max(candidates, key=lambda r: (r['total_r'], -r['opportunities']))
    active = [r for r in candidates if r['opportunities'] > 0]
    best_active = max(active, key=lambda r:r['total_r']) if active else None
    available = sets[minimum]
    reason = ('positive_validation_pnl' if selected['opportunities'] else
              'no_prediction_at_or_above_floor' if not available else
              'no_profitable_admissible_threshold')
    diagnostics = dict(method='exact_unique_scores_no_count_filter_v3', thresholds_evaluated=len(candidates),
        admissible_nonempty_thresholds=len(active), minimum_accepted=0, acceptance_count_filter=False,
        legacy_minimum_accepted_ignored=config.min_accepted_validation,
        events_at_floor=available, reason=reason, best_exploratory=best_active,
        best_admissible_nonempty=best_active,
        at_floor=next(r for r in candidates if r['threshold'] == minimum),
        selection_includes_abstention=True)
    # Keep reports bounded even for a negative floor admitting all ~50k unique
    # predictions. The optimum and diagnostic reference points are always kept.
    keep = set(np.linspace(0, len(candidates)-1, min(100, len(candidates)), dtype=int).tolist())
    mandatory = [selected, best_active, next(r for r in candidates if r['threshold'] == minimum)]
    keep.update(i for i, r in enumerate(candidates) if any(r is point for point in mandatory))
    return {**selected, 'diagnostics': diagnostics}, [candidates[i] for i in sorted(keep)]


RUNTIME_FIELDS = {'epochs', 'threads', 'device', 'checkpoint_batches', 'min_accepted_validation'}


def epoch_selection_diagnostics(metadata):
    """Separate the constant benchmark from epochs that actually trained."""
    history = metadata['history']
    trained = min(history, key=lambda row: row['validation_mse_r']) if history else None
    baseline = metadata['validation']['constant_mse']
    return dict(criterion='validation_mse_r', constant_is_candidate=True,
        selected_predictor='training_mean_constant' if metadata['best_epoch'] == 0 else 'trained_network',
        baseline_mse_r=baseline,
        best_trained_epoch=trained['epoch'] if trained else None,
        best_trained_mse_r=trained['validation_mse_r'] if trained else None,
        best_trained_improvement_over_constant=baseline-trained['validation_mse_r'] if trained else None)


def fit(data, output, config=NeuralConfig(), *, resume=False, stop_after_batches=None):
    import torch
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metadata.json').exists():
        raise ValueError('Model is already frozen; use a new output directory')
    if not resume and (output/'latest.pt').exists():
        raise ValueError('Use resume to continue an existing neural training')
    train, validation = data.partitions['train'], data.partitions['validation']
    if len(train) < config.min_train_samples or len(validation) < config.min_validation_samples:
        raise ValueError(f'Insufficient data: train={len(train)}, validation={len(validation)}')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    shuffle = torch.Generator().manual_seed(config.seed)
    spec = architecture_spec(data.source['input_dims'], config)
    if resume:
        state = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        if (state['source'] != data.source or state['architecture'] != spec or any(
                state['config'][k] != v for k, v in asdict(config).items() if k not in RUNTIME_FIELDS)):
            raise ValueError('Resume requires the same data, architecture and statistical configuration')
        normalization = state['normalization']
    else:
        normalization = normalizer(data, train, config.batch_size)
        state = dict(epoch=0, cursor=0, order=None, updates=0, stale=0, history=[],
                     best_epoch=0, best_loss=None, loss_sum=0., seen=0)
    model = build_model(spec).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5,
        patience=max(1, config.patience//3), min_lr=min(config.learning_rate, 1e-5))
    if resume:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(config.device)
        torch.set_rng_state(state['torch_rng'])
        shuffle.set_state(state['shuffle_rng'])
        if config.device.startswith('cuda') and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])

    def save_best():
        state['best_model'] = {k:v.detach().cpu().clone() for k, v in model.state_dict().items()}
        write_best()

    def write_best():
        atomic_torch(output/'best.pt', dict(model=state['best_model'], architecture=spec,
            normalization=normalization, source=data.source, config=asdict(config), version=MODEL_VERSION))

    def checkpoint():
        atomic_torch(output/'latest.pt', {**state, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'config': asdict(config), 'architecture': spec,
            'source': data.source, 'normalization': normalization, 'torch_rng': torch.get_rng_state(),
            'shuffle_rng': shuffle.get_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if config.device.startswith('cuda') else []})
        atomic_json(output/'progress.json', {k: state[k] for k in ('epoch', 'cursor', 'updates', 'best_epoch', 'best_loss', 'stale', 'history')})

    if not resume:
        state['best_loss'] = float(np.mean((data.y[validation]-normalization['target_mean'])**2))
        save_best()
        checkpoint()
    else:
        # A shutdown between best.pt and latest.pt must roll back both to the
        # same completed checkpoint, including the best validation candidate.
        write_best()
    print('[swing-nn-plan] '+json.dumps(dict(architecture=spec, train=len(train), validation=len(validation),
        device=config.device, resume_epoch=state['epoch'], resume_cursor=state['cursor'])), flush=True)
    started, updates = monotonic(), 0
    try:
        while state['epoch'] < config.epochs and state['stale'] < config.patience:
            epoch_started = monotonic()
            if state['order'] is None:
                state['order'] = torch.randperm(len(train), generator=shuffle)
            model.train()
            while state['cursor'] < len(train):
                indices = train[state['order'][state['cursor']:state['cursor']+config.batch_size].numpy()]
                x = torch.as_tensor(np.asarray(data.x[indices])/normalization['input_scale'], dtype=torch.float32, device=config.device)
                y = torch.as_tensor((data.y[indices]-normalization['target_mean'])/normalization['target_scale'],
                                    dtype=torch.float32, device=config.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x).flatten(), y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite neural loss; last checkpoint retained')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                state['cursor'] += len(indices)
                state['updates'] += 1
                state['loss_sum'] += float(loss.detach())*len(indices)
                state['seen'] += len(indices)
                updates += 1
                if state['updates'] % config.checkpoint_batches == 0:
                    checkpoint()
                    print(f"[swing-nn-batch] epoch={state['epoch']+1} samples={state['cursor']}/{len(train)} "
                          f"updates={state['updates']} elapsed_s={monotonic()-started:.1f}", flush=True)
                if stop_after_batches is not None and updates >= stop_after_batches:
                    checkpoint()
                    return dict(interrupted=True, updates=state['updates'])
            validation_started = monotonic()
            training_seconds = validation_started-epoch_started
            predictions = predict_data(model, data, validation, normalization, config)
            validation_seconds = monotonic()-validation_started
            loss_r = float(np.mean((data.y[validation]-predictions)**2))
            improved = loss_r < state['best_loss']
            if improved:
                state.update(best_loss=loss_r, best_epoch=state['epoch']+1, stale=0)
                save_best()
            else:
                state['stale'] += 1
            scheduler.step(loss_r)
            state['history'].append(dict(epoch=state['epoch']+1, train_mse_r=state['loss_sum']/state['seen']*normalization['target_scale']**2,
                validation_mse_r=loss_r, best=improved, learning_rate=optimizer.param_groups[0]['lr']))
            state.update(epoch=state['epoch']+1, cursor=0, order=None, loss_sum=0., seen=0)
            checkpoint()
            print(f"[swing-nn-epoch] epoch={state['epoch']} validation_mse_r={loss_r:.7g} "
                  f"best_epoch={state['best_epoch']} stale={state['stale']}/{config.patience} "
                  f"validation_rows={len(validation)} prediction_min_r={predictions.min():.7g} "
                  f"prediction_max_r={predictions.max():.7g} "
                  f"session_train_s={training_seconds:.1f} validation_s={validation_seconds:.1f} "
                  f"session_epoch_s={monotonic()-epoch_started:.1f}", flush=True)
    except KeyboardInterrupt:
        # An interrupt can land inside optimizer.step(). Preserve the latest
        # completed checkpoint rather than publishing a half-updated batch.
        raise
    best = torch.load(output/'best.pt', map_location=config.device, weights_only=True)
    model.load_state_dict(best['model'])
    predictions = predict_data(model, data, validation, normalization, config)
    selection = StopSelection.from_rows([data.grid.rows[i] for i in validation], data.source['opportunity_config'])
    selected, candidates = choose_threshold(data.y[validation], predictions, config, selection)
    evaluation = evaluate_predictions(data.y[validation], predictions, selected['threshold'], normalization['target_mean'], selection)
    qualified = evaluation['improvement_over_constant'] > 0 and selected['total_r'] > 0
    source = data.source
    available = datetime.fromisoformat(source['protocol']['validation_end']).replace(tzinfo=timezone.utc)+timedelta(days=1)
    digest = _sha(output/'best.pt')
    metadata = dict(version=MODEL_VERSION, model_type='swing-nn', target='pnl-r', prediction_unit='R',
        feature_schema=FEATURE_SCHEMA, swing_schema=SWING_SCHEMA, opportunity_schema=OPPORTUNITY_SCHEMA,
        grid_schema=source['grid_schema'], grid_normalization=source['grid_normalization'], input_dims=source['input_dims'],
        window_key=source['window_key'], interval_seconds=source['interval_seconds'], opportunity_config=source['opportunity_config'],
        scope=dict(symbols=source['symbols']), available_after=available.isoformat(), source=source,
        architecture=spec, config=asdict(config), normalization=normalization, threshold=selected['threshold'],
        input_feature_names=[f'candle_{i:05d}' for i in range(source['input_dims']-int(uses_stop_distance(source['opportunity_config'])))]
            + (['stop_buffer_ratio'] if uses_stop_distance(source['opportunity_config']) else []),
        min_threshold=config.min_threshold, threshold_abstains=selected['opportunities'] == 0,
        selection_policy=evaluation.get('selection_policy', 'independent_opportunities_v1'),
        acceptance_count_filter=False,
        threshold_selection_diagnostics=selected['diagnostics'],
        validation_qualified=bool(qualified), validation=evaluation, threshold_candidates=candidates,
        best_epoch=state['best_epoch'], completed_epochs=state['epoch'], history=state['history'],
        checkpoint_sha256=digest, resolver_id=digest, torch_version=str(torch.__version__),
        test_used_for_selection=False, extra_tabular_inputs=uses_stop_distance(source['opportunity_config']))
    epoch_diagnostic = metadata['epoch_selection_diagnostics'] = epoch_selection_diagnostics(metadata)
    temp = output/'.validation_predictions.tmp.npz'
    np.savez_compressed(temp, event_id=np.asarray([data.grid.rows[i]['event_id'] for i in validation]),
        prediction=predictions, actual=data.y[validation], accepted=acceptance_mask(predictions, selected['threshold'], selection))
    temp.replace(output/'validation_predictions.npz')
    metadata['validation_predictions_sha256'] = _sha(output/'validation_predictions.npz')
    write_report(output, metadata)
    atomic_json(output/'metadata.json', metadata)
    print(f"[swing-nn-selected] model={output} best_epoch={state['best_epoch']} "
          f"predictor={epoch_diagnostic['selected_predictor']} validation_rows={evaluation['rows']} "
          f"validation_mse_r={evaluation['prediction_mse']:.9g} constant_mse_r={evaluation['constant_mse']:.9g} "
          f"best_trained_epoch={epoch_diagnostic['best_trained_epoch']} "
          f"best_trained_mse_r={epoch_diagnostic['best_trained_mse_r']} "
          f"initial_threshold_r={selected['threshold']:.9g} validation_qualified={qualified} "
          f"accepted={selected['opportunities']} total_r={selected['total_r']:.3f} "
          f"reason={selected['diagnostics']['reason']} events_at_floor={selected['diagnostics']['events_at_floor']} "
          f"acceptance_count_filter=off", flush=True)
    return metadata


def write_report(output, metadata):
    val = metadata['validation']
    epoch_diagnostic = epoch_selection_diagnostics(metadata)
    lines = ['# Réseau neuronal sur la grille de swings', '',
        f"Architecture : `{json.dumps(metadata['architecture'])}`.",
        f"Normalisation de grille : `{metadata['grid_normalization']}`. Cible : pnl-r, en R.",
        f"Périodes : `{json.dumps(metadata['source']['protocol'])}`.", '',
        f"**Seuil initial : `{metadata['threshold']:.9g}` R.** Plancher : `{metadata['min_threshold']}` R.",
        f"Qualification : **{metadata['validation_qualified']}**. Abstention : **{metadata['threshold_abstains']}**.",
        f"Meilleure époque : {metadata['best_epoch']} ; époques terminées : {metadata['completed_epochs']}.", '',
        f"Exemples de validation évalués : {val['rows']} ; prédicteur retenu : `{epoch_diagnostic['selected_predictor']}`.",
        f"MSE validation : {val['prediction_mse']:.9g} ; constante apprise sur train : {val['constant_mse']:.9g}.",
        f"Prédictions : [{val['prediction_min']:.7g}, {val['prediction_max']:.7g}] R ; écart-type {val['prediction_std']:.7g}.",
        f"Événements acceptés : {val['selected']['opportunities']} ; somme : {val['selected']['total_r']:.4f} R.", '',
        'La grille est identique à celle de XGBoost. Une échelle RMS scalaire apprise sur le train conserve les zéros.',
        'La validation choisit l’époque et le seuil. Le test est séparé et ne modifie aucun réglage.',
        'Les sommes de R concernent des opportunités indépendantes et chevauchantes, pas un rendement de portefeuille.',
        'La qualification sur validation ne démontre pas un edge reproductible.']
    if epoch_diagnostic['best_trained_epoch'] is not None:
        lines += ['', f"Meilleur réseau entraîné selon la MSE : époque {epoch_diagnostic['best_trained_epoch']}, "
                  f"MSE {epoch_diagnostic['best_trained_mse_r']:.9g}."]
    if metadata['best_epoch'] == 0:
        lines += ['La constante initiale a été retenue : aucune époque entraînée ne la bat sur la MSE globale.',
                  'Zéro trade accepté ne signifie pas zéro donnée de validation. Les époques entraînées ont été évaluées,',
                  'puis leurs poids ont été écartés avant la sélection du seuil. La MSE sélectionne les poids ; le PnL sélectionne le seuil.']
    diagnostic = metadata.get('threshold_selection_diagnostics')
    if val.get('selection_policy'):
        lines += ['', 'Exécution : première distance acceptée par swing, par ordre croissant des stops.',
            f"Swings complets évaluables : {val['complete_swings']} ; incomplets écartés : {val['incomplete_swings']}.",
            'La MSE décrit toutes les prédictions ; les métriques de trading retiennent au plus un trade par swing.']
    if diagnostic:
        lines += ['', '## Sélection du seuil', '',
            f"Méthode : `{diagnostic['method']}` ; décision : `{diagnostic['reason']}`.",
            f"Événements au plancher : {diagnostic['events_at_floor']} ; aucun minimum d’événements acceptés.",
            f"Seuils évalués : {diagnostic['thresholds_evaluated']} (tous les ensembles réalisables)."]
        floor = diagnostic.get('at_floor')
        if floor:
            lines += [f"Au plancher fixé, avant optimisation du seuil : {floor['total_r']:.4f} R sur {floor['opportunities']} événements.",
                      'Le meilleur R de validation est non négatif par construction, car l’abstention fait partie des choix.']
        exploratory = diagnostic['best_exploratory']
        if exploratory:
            lines += [f"Meilleur résultat exploratoire : {exploratory['total_r']:.4f} R sur "
                      f"{exploratory['opportunities']} événements, seuil {exploratory['threshold']:.9g} R. "
                      'L’effectif est informatif et ne constitue pas un filtre.']
    (Path(output)/'rapport.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def recalibrate_model(directory, output):
    """Derive a new frozen calibration from saved validation, preserving weights."""
    original = SwingNeuralResolver(directory)
    source, meta = original.directory, dict(original.metadata)
    predictions_file = source/'validation_predictions.npz'
    if _sha(predictions_file) != meta['validation_predictions_sha256']:
        raise ValueError('Neural validation prediction checksum mismatch')
    with np.load(predictions_file, allow_pickle=False) as saved:
        y, predictions = saved['actual'].copy(), saved['prediction'].copy()
        selection = StopSelection.from_event_ids(saved['event_id'], meta['opportunity_config'])
    config = NeuralConfig(**meta['config'])
    selected, candidates = choose_threshold(y, predictions, config, selection)
    evaluation = evaluate_predictions(y, predictions, selected['threshold'], meta['normalization']['target_mean'], selection)
    calibration = dict(source_model=str(source), source_metadata_sha256=_sha(source/'metadata.json'),
        previous_threshold=meta['threshold'], previous_accepted=meta['validation']['selected']['opportunities'],
        previous_total_r=meta['validation']['selected']['total_r'], training_performed=False, validation_reused=True)
    meta.update(threshold=selected['threshold'], threshold_abstains=selected['opportunities']==0,
        selection_policy=evaluation.get('selection_policy', 'independent_opportunities_v1'),
        acceptance_count_filter=False,
        validation=evaluation, threshold_candidates=candidates, threshold_selection_diagnostics=selected['diagnostics'],
        validation_qualified=bool(evaluation['improvement_over_constant'] > 0 and selected['total_r'] > 0), calibration=calibration)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Recalibration requires a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('best.pt', 'validation_predictions.npz'):
        shutil.copy2(source/filename, output/filename)
    if selection is not None:
        with np.load(predictions_file, allow_pickle=False) as saved:
            arrays = {name:saved[name].copy() for name in saved.files}
        arrays['accepted'] = acceptance_mask(predictions, selected['threshold'], selection)
        np.savez_compressed(output/'validation_predictions.npz', **arrays)
        meta['validation_predictions_sha256'] = _sha(output/'validation_predictions.npz')
    if _sha(output/'best.pt') != meta['checkpoint_sha256']:
        raise ValueError('Copied neural weights differ from the original')
    write_report(output, meta)
    atomic_json(output/'metadata.json', meta)
    return meta


class SwingNeuralResolver:
    def __init__(self, directory, *, threads=2, device='cpu'):
        import torch
        self.directory = resolver_directory(directory)
        meta = self.metadata = json.loads((self.directory/'metadata.json').read_text(encoding='utf-8'))
        if (meta.get('model_type') != 'swing-nn' or meta.get('version') != MODEL_VERSION or meta.get('target') != 'pnl-r'
                or meta.get('feature_schema') != FEATURE_SCHEMA or meta.get('swing_schema') != SWING_SCHEMA
                or meta.get('opportunity_schema') != OPPORTUNITY_SCHEMA
                or meta.get('grid_schema') != grid_schema(meta.get('grid_normalization'), meta['opportunity_config'])):
            raise ValueError('Unsupported frozen swing neural model')
        if _sha(self.directory/'best.pt') != meta['checkpoint_sha256']:
            raise ValueError('Neural checkpoint hash mismatch')
        saved = torch.load(self.directory/'best.pt', map_location=device, weights_only=True)
        if (saved['source'] != meta['source'] or saved['normalization'] != meta['normalization']
                or saved['architecture'] != meta['architecture'] or saved['version'] != MODEL_VERSION
                or any(saved['config'][k] != v for k, v in meta['config'].items() if k not in RUNTIME_FIELDS)):
            raise ValueError('Frozen neural metadata and checkpoint differ')
        self.config = replace(NeuralConfig(**meta['config']), threads=threads, device=device)
        self.window = SwingWindow('duration', int(meta['window_key'].rsplit('_', 1)[1]))
        if (grid_input_width(self.window, meta['interval_seconds'], meta['opportunity_config']) != meta['input_dims']
                or architecture_spec(meta['input_dims'], self.config) != meta['architecture']
                or meta['source']['opportunity_config'] != meta['opportunity_config']
                or meta['source']['symbols'] != meta['scope']['symbols']
                or any(meta['source'][k] != meta[k] for k in ('input_dims', 'window_key', 'interval_seconds',
                                                            'grid_schema', 'grid_normalization'))):
            raise ValueError('Inconsistent frozen neural input configuration')
        norm = meta['normalization']
        if (not np.isfinite([norm['input_scale'], norm['target_mean'], norm['target_scale'], meta['threshold']]).all()
                or norm['input_scale'] <= 0 or norm['target_scale'] <= 0 or norm['input_centering'] is not False
                or norm['fit_partition'] != 'train_only'):
            raise ValueError('Invalid frozen neural normalizer')
        torch.set_num_threads(threads)
        devices = list(range(torch.cuda.device_count())) if device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            self.model = build_model(meta['architecture']).to(device)
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        self.grid_normalization = meta['grid_normalization']

    def predict(self, points, *, as_of, symbol=None, candles=None, normalization_mean=None, stop_buffer_pips=None):
        if as_of.tzinfo is None:
            raise ValueError('Neural decisions require an explicit timezone')
        if as_of < datetime.fromisoformat(self.metadata['available_after']):
            raise ValueError('Neural training/validation postdates this decision')
        symbols = self.metadata['scope']['symbols']
        symbol = symbol.upper() if symbol else symbols[0] if len(symbols) == 1 else None
        if symbol not in symbols:
            raise ValueError('Specify a symbol in the neural model scope')
        if self.grid_normalization == 'candle-close-mean':
            if candles is not None:
                if normalization_mean is not None:
                    raise ValueError('Provide candles or their mean, not both')
                from ultimateBot.backtester.research.swing_candle_prices import candle_close_mean
                normalization_mean = candle_close_mean(candles, as_of=as_of, window=self.window,
                    interval_seconds=self.metadata['interval_seconds'])
            if normalization_mean is None:
                raise ValueError('Neural model requires its closed candle prices')
        elif candles is not None or normalization_mean is not None:
            raise ValueError('Cannot change frozen swing-mean normalization')
        points = tuple(points)
        values = grid_vector(points, as_of=as_of, window=self.window, interval_seconds=self.metadata['interval_seconds'],
                             normalization_mean=normalization_mean)
        values = append_stop_input(values, self.metadata['opportunity_config'], symbol, points[-1].price, stop_buffer_pips)
        score = float(predict_array(self.model, values[None, :], self.metadata['normalization'],
                                   device=self.config.device)[0])
        return dict(prediction=score, target='pnl-r', threshold=self.metadata['threshold'],
                    accepted=score >= self.metadata['threshold'], side=points[-1].side,
                    resolver_id=self.metadata['resolver_id'])


def frozen_test(database, *, directory, run_id=None, start, end, output_dir, threads=2, device='cpu', min_samples=30):
    from ultimateBot.backtester.research.swing_resolver import day_start
    resolver = SwingNeuralResolver(directory, threads=threads, device=device)
    meta = resolver.metadata
    if min_samples < 1 or day_start(start) < datetime.fromisoformat(meta['available_after']) or day_start(start) > day_start(end):
        raise ValueError('Neural test must follow validation and have a positive sample minimum')
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new neural test output directory')
    data, params = prepare(database, run_id=run_id or meta['source']['run_id'], windows=(resolver.window,),
        symbols=meta['scope']['symbols'], periods=dict(test=[start, end]), target='pnl-r',
        cache_root=Path(database).resolve().parent/'swing_grid_cache', grid_normalization=meta['grid_normalization'])
    try:
        original, current = [asdict(OpportunityConfig(**p)) for p in (meta['opportunity_config'], params['config'])]
        if {k:v for k,v in original.items() if k!='risk_cash'} != {k:v for k,v in current.items() if k!='risk_cash'}:
            raise ValueError('Frozen neural test must preserve candle, stop, expiry and trailing settings')
        if len(data.rows) < min_samples:
            raise ValueError('Too few labeled neural test opportunities')
        predictions = predict_array(resolver.model, data.matrices[meta['window_key']], meta['normalization'],
                                    device=device, batch_size=resolver.config.batch_size)
        y = np.asarray([r['pnl_r'] for r in data.rows])
        selection = StopSelection.from_rows(data.rows, meta['opportunity_config'])
        result = dict(model=str(resolver.directory), start=start, end=end, threshold=meta['threshold'],
            metrics=evaluate_predictions(y, predictions, meta['threshold'], meta['normalization']['target_mean'], selection),
            fitting_performed=False, threshold_changed=False, dataset_sha256=data.manifest['dataset_sha256'])
        result['per_symbol'] = {s:evaluate_predictions(y[mask], predictions[mask], meta['threshold'],
            meta['normalization']['target_mean'], selection.subset(mask) if selection is not None else None) for s in meta['scope']['symbols']
            if (mask := np.asarray([r['symbol']==s for r in data.rows])).any()}
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/'test.json', result)
        np.savez_compressed(output/'predictions.npz', event_id=np.asarray([r['event_id'] for r in data.rows]),
                            prediction=predictions, actual=y, accepted=acceptance_mask(predictions, meta['threshold'], selection))
        return result
    finally:
        data.close()
