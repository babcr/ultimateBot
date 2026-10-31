"""Upstream Informer, supervised directly on a swing opportunity's realized R."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import builtins
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic
import types
from uuid import uuid4

import numpy as np

from ultimateBot.backtester.research.swing_informer_data import (
    CONFIRMED, SYMBOL, SwingInformerData, features, instant,
)
from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, uses_stop_distance, stop_distance_feature
from ultimateBot.backtester.research.swing_series import SWING_SCHEMA, feature_names
from ultimateBot.backtester.research.swing_selection import acceptance_mask, selection_metrics, exact_stop_threshold

UPSTREAM = Path(__file__).resolve().parents[2]/'Informer2020'
VENDOR_FILES = ('utils/masking.py', 'models/encoder.py', 'models/decoder.py',
                'models/embed.py', 'models/attn.py', 'models/model.py')


def upstream_hash():
    return hashlib.sha256(b''.join((UPSTREAM/p).read_bytes() for p in VENDOR_FILES)).hexdigest()


@lru_cache(maxsize=1)
def informer_class():
    """Isolate upstream's absolute models/utils imports; never patch the submodule."""
    if not (UPSTREAM/'models/model.py').is_file():
        raise RuntimeError('Initialize Informer2020: git submodule update --init Informer2020')
    namespace = '_ultimatebot_informer_vendor'
    package = types.ModuleType(namespace)
    package.__path__ = [str(UPSTREAM)]
    sys.modules[namespace] = package
    modules = {}
    original_import = builtins.__import__
    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return original_import(name, globals, locals, fromlist, level)
    for filename in VENDOR_FILES:
        name = filename[:-3].replace('/', '.')
        spec = importlib.util.spec_from_file_location(namespace+'.'+name, UPSTREAM/filename)
        module = importlib.util.module_from_spec(spec)
        module.__dict__['__builtins__'] = {**vars(builtins), '__import__': scoped_import}
        sys.modules[spec.name] = module
        # Loading a vendor checkout must not leave untracked __pycache__ files.
        exec(compile((UPSTREAM/filename).read_bytes(), str(UPSTREAM/filename), 'exec'), module.__dict__)
        modules[name] = module
    return modules['models.model'].Informer


@dataclass(frozen=True)
class InformerConfig:
    seq_len: int = 64
    label_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 128
    factor: int = 5
    dropout: float = .1
    attention: str = 'prob'
    batch_size: int = 64
    epochs: int = 10
    patience: int = 3
    learning_rate: float = .0003
    weight_decay: float = .0001
    min_train_samples: int = 1000
    min_validation_samples: int = 200
    min_accepted_validation: int = 100
    min_threshold: float = 0.
    checkpoint_batches: int = 100
    seed: int = 1
    threads: int = 2
    device: str = 'cpu'

    def __post_init__(self):
        for key in ('seq_len', 'label_len', 'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff', 'factor',
                    'batch_size', 'epochs', 'patience', 'min_train_samples', 'min_validation_samples',
                    'min_accepted_validation', 'checkpoint_batches', 'threads'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 8 <= self.seq_len <= 2048 or self.seq_len % 2 or not 2 <= self.label_len <= self.seq_len:
            raise ValueError('Require an even seq_len in [8,2048] and label_len in [2,seq_len]')
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError('d_model must be even and divisible by n_heads')
        if self.e_layers > int(math.log2(self.seq_len)):
            raise ValueError('Too many distillation layers for this swing sequence length')
        if self.attention not in ('prob', 'full') or not 0 <= self.dropout < 1:
            raise ValueError('Invalid attention or dropout')
        if (not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not math.isfinite(self.min_threshold) or not 0 <= self.seed < 2**31):
            raise ValueError('Invalid optimizer, threshold floor or seed')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_torch(path, value):
    import torch
    path = Path(path)
    temp = path.with_name('.'+path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def normalizer(data, train, batch_size):
    count, total, squares = 0, None, None
    for start in range(0, len(train), batch_size):
        x, _ = data.batch(train[start:start+batch_size])
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        if total is None:
            total, squares = np.zeros(flat.shape[1]), np.zeros(flat.shape[1])
        total += flat.sum(axis=0)
        squares += (flat*flat).sum(axis=0)
        count += len(flat)
    mean = total/count
    scale = np.maximum(np.sqrt(np.maximum(squares/count-mean**2, 0)), 1e-6)
    y = data.labels[train].astype(np.float64)
    return dict(mean=mean.tolist(), scale=scale.tolist(), target_mean=float(y.mean()),
                target_scale=max(float(y.std()), 1e-6), fit_partition='train_only')


class InformerRegressor:
    def __init__(self, config, normalization, input_dims):
        import torch
        self.config, self.normalization = config, normalization
        self.mean = np.asarray(normalization['mean'], dtype=np.float32)
        self.scale = np.asarray(normalization['scale'], dtype=np.float32)
        if (self.mean.shape != (input_dims,) or self.scale.shape != (input_dims,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any()
                or not math.isfinite(normalization['target_mean']) or not math.isfinite(normalization['target_scale'])
                or normalization['target_scale'] <= 0):
            raise ValueError('Invalid train-only normalizer')
        torch.set_num_threads(config.threads)
        self.model = informer_class()(enc_in=input_dims, dec_in=input_dims, c_out=1,
            seq_len=config.seq_len, label_len=config.label_len, out_len=1,
            factor=config.factor, d_model=config.d_model, n_heads=config.n_heads,
            e_layers=config.e_layers, d_layers=config.d_layers, d_ff=config.d_ff,
            dropout=config.dropout, attn=config.attention, embed='timeF', freq='t',
            activation='gelu', output_attention=False, distil=True, mix=False,
            device=torch.device(config.device)).to(config.device)

    def forward(self, x, marks):
        import torch
        config = self.config
        values = torch.as_tensor((x-self.mean)/self.scale, dtype=torch.float32, device=config.device)
        time = torch.as_tensor(marks, dtype=torch.float32, device=config.device)
        # Decoder contains observed features only, never past or future trade outcomes.
        decoder = torch.cat([values[:, -config.label_len:], torch.zeros_like(values[:, :1])], dim=1)
        decoder_time = torch.cat([time[:, -config.label_len:], time[:, -1:]], dim=1)
        return self.model(values, time, decoder, decoder_time)[:, 0, 0]

    def predict(self, x, marks):
        import torch
        self.model.eval()
        devices = list(range(torch.cuda.device_count())) if self.config.device.startswith('cuda') else []
        # Upstream ProbSparse samples even in eval(). A fixed local seed makes
        # decisions repeatable and independent of prior inference/training calls.
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(self.config.seed)
            prediction = self.forward(x, marks).cpu().numpy().astype(np.float64)
        prediction = prediction*self.normalization['target_scale']+self.normalization['target_mean']
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite Informer prediction')
        return prediction


def predict_data(model, data, indices):
    results = []
    for start in range(0, len(indices), model.config.batch_size):
        x, marks = data.batch(indices[start:start+model.config.batch_size])
        results.append(model.predict(x, marks))
    return np.concatenate(results) if results else np.empty(0, dtype=np.float64)


def outcome_metrics(labels, accepted):
    values = np.asarray(labels, dtype=np.float64)[accepted]
    profit, loss = float(np.maximum(values, 0).sum()), float(-np.minimum(values, 0).sum())
    return dict(opportunities=len(values), total_r=float(values.sum()),
                mean_r=float(values.mean()) if len(values) else None,
                win_rate=float((values > 0).mean()) if len(values) else None,
                profit_factor_r=profit/loss if loss else None, gross_profit_r=profit, gross_loss_r=loss)


def validation_threshold(labels, predictions, config, selection=None):
    if selection is not None:
        return exact_stop_threshold(labels, predictions, config.min_threshold, selection)
    # A fixed floor prevents selecting a negative expected-R threshold by default.
    thresholds = np.unique(np.r_[config.min_threshold, np.quantile(predictions, np.linspace(0, .99, 50))])
    thresholds = thresholds[thresholds >= config.min_threshold]
    candidates = [dict(threshold=float(t), **outcome_metrics(labels, predictions >= t)) for t in thresholds]
    eligible = [r for r in candidates if r['opportunities'] >= config.min_accepted_validation]
    best = max(eligible, key=lambda r: (r['total_r'], r['opportunities'])) if eligible else candidates[0]
    return best, candidates


def evaluate_predictions(labels, predictions, threshold, baseline, selection=None):
    labels = np.asarray(labels, dtype=np.float64)
    mse = float(np.mean((labels-predictions)**2))
    baseline_mse = float(np.mean((labels-baseline)**2))
    result = dict(rows=len(labels), prediction_mse=mse, constant_prediction=baseline,
                constant_mse=baseline_mse, improvement_over_constant=baseline_mse-mse,
                prediction_min=float(predictions.min()), prediction_max=float(predictions.max()),
                prediction_std=float(predictions.std()),
                all_opportunities=outcome_metrics(labels, np.ones(len(labels), dtype=bool)),
                selected=outcome_metrics(labels, predictions >= threshold))
    if selection is not None:
        result.update(selection.diagnostic(),
            all_opportunities=selection_metrics(labels, selection.mask(np.ones(len(labels)), 0.), selection),
            selected=selection_metrics(labels, selection.mask(predictions, threshold), selection))
    return result


def fit(data, output, config=InformerConfig(), *, resume=False, stop_after_batches=None):
    """Atomic checkpoints include optimizer, exact shuffle, cursor and RNG state."""
    import torch
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output/'latest.pt').exists():
        raise ValueError('Use --resume for an existing training run')
    if not resume and (output/'metadata.json').exists():
        raise ValueError('Do not overwrite a frozen model')
    train, validation = data.partition('train'), data.partition('validation')
    if len(train) < config.min_train_samples or len(validation) < config.min_validation_samples:
        raise ValueError(f'Insufficient data: train={len(train)}, validation={len(validation)}')
    if data.contexts.shape[1] != config.seq_len:
        raise ValueError('Dataset and Informer sequence lengths differ')
    source = data.metadata
    symbols_seen = set(data.points[data.contexts[train, -1], SYMBOL].astype(int))
    if symbols_seen != set(range(len(source['symbols']))):
        raise ValueError('Every selected symbol must have training observations')
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    shuffle = torch.Generator().manual_seed(config.seed)
    if resume:
        state = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        previous = state['config']
        # Epoch budget may be extended; hardware/threads may change. Statistical
        # inputs, optimizer, architecture and batch size remain frozen on resume.
        mutable = {'epochs', 'device', 'threads', 'checkpoint_batches'}
        if any(previous[k] != asdict(config)[k] for k in previous if k not in mutable):
            raise ValueError('Resume requires the same model, optimizer, batch size and threshold configuration')
        if state['source'] != source or state['upstream_sha256'] != upstream_hash():
            raise ValueError('Resume dataset or upstream source differs from the checkpoint')
        if (output/'metadata.json').exists():
            raise ValueError('Model is already frozen; use a new output directory and --dataset for further training')
        normalization = state['normalization']
    else:
        normalization = normalizer(data, train, config.batch_size)
        state = dict(epoch=0, cursor=0, order=None, loss_sum=0., seen=0, best_loss=None,
                     best_epoch=None, stale=0, updates=0, history=[])
    dims = len(normalization['mean'])
    model = InformerRegressor(config, normalization, dims)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if resume:
        model.model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(config.device)
        shuffle.set_state(state['shuffle_rng'])
        torch.set_rng_state(state['torch_rng'])
        if config.device.startswith('cuda') and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    print('[swing-informer-plan] '+json.dumps(dict(train=len(train), validation=len(validation),
        swings=config.seq_len, input_dims=dims, parameters=sum(p.numel() for p in model.model.parameters()),
        device=config.device, epochs=config.epochs, resume_epoch=state['epoch'], resume_cursor=state['cursor'])), flush=True)

    def checkpoint():
        saved = {**state, 'config': asdict(config), 'normalization': normalization, 'source': source,
                 'upstream_sha256': upstream_hash(), 'model': model.model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                 'shuffle_rng': shuffle.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if config.device.startswith('cuda') else []}
        atomic_torch(output/'latest.pt', saved)
        atomic_json(output/'progress.json', {k: state[k] for k in ('epoch', 'cursor', 'updates', 'best_epoch', 'best_loss', 'stale', 'history')})

    if not resume:
        checkpoint()
    started, this_call_updates = monotonic(), 0
    while state['epoch'] < config.epochs and state['stale'] < config.patience:
        if state['order'] is None:
            state['order'] = torch.randperm(len(train), generator=shuffle)
        model.model.train()
        while state['cursor'] < len(train):
            indices = train[state['order'][state['cursor']:state['cursor']+config.batch_size].numpy()]
            x, marks = data.batch(indices)
            y = torch.as_tensor((data.labels[indices]-normalization['target_mean'])/normalization['target_scale'],
                                dtype=torch.float32, device=config.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward(x, marks)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Informer training loss; last valid checkpoint retained')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            state['cursor'] += len(indices)
            state['seen'] += len(indices)
            state['loss_sum'] += float(loss.detach())*len(indices)
            state['updates'] += 1
            this_call_updates += 1
            if state['updates'] % config.checkpoint_batches == 0:
                checkpoint()
                print(f"[swing-informer] epoch={state['epoch']+1}/{config.epochs} "
                      f"samples={state['cursor']}/{len(train)} mse_scaled={state['loss_sum']/state['seen']:.6f} "
                      f"updates={state['updates']} elapsed_s={monotonic()-started:.1f}", flush=True)
            if stop_after_batches is not None and this_call_updates >= stop_after_batches:
                checkpoint()
                return {'interrupted': True, 'updates': state['updates']}
        prediction = predict_data(model, data, validation)
        loss_r = float(np.mean((prediction-data.labels[validation].astype(np.float64))**2))
        improved = state['best_loss'] is None or loss_r < state['best_loss']
        if improved:
            state['best_loss'], state['best_epoch'], state['stale'] = loss_r, state['epoch']+1, 0
            atomic_torch(output/'best.pt', dict(model=model.model.state_dict(), config=asdict(config),
                normalization=normalization, upstream_sha256=upstream_hash(), source=source))
        else:
            state['stale'] += 1
        state['history'].append(dict(epoch=state['epoch']+1,
            train_mse_r=state['loss_sum']/state['seen']*normalization['target_scale']**2,
            validation_mse_r=loss_r, best=improved))
        print(f"[swing-informer-epoch] epoch={state['epoch']+1} validation_mse_r={loss_r:.6f} "
              f"best_epoch={state['best_epoch']} stale={state['stale']}/{config.patience}", flush=True)
        state.update(epoch=state['epoch']+1, cursor=0, order=None, loss_sum=0., seen=0)
        checkpoint()
    best = torch.load(output/'best.pt', map_location=config.device, weights_only=True)
    model.model.load_state_dict(best['model'])
    prediction = predict_data(model, data, validation)
    selection = data.stop_selection(validation)
    selected, candidates = validation_threshold(data.labels[validation], prediction, config, selection)
    evaluation = evaluate_predictions(data.labels[validation], prediction, selected['threshold'], normalization['target_mean'], selection)
    qualified = ((selection is not None or selected['opportunities'] >= config.min_accepted_validation) and selected['total_r'] > 0
                 and selected['profit_factor_r'] is not None and selected['profit_factor_r'] >= 1.05
                 and evaluation['improvement_over_constant'] > 0)
    digest = hashlib.sha256((output/'best.pt').read_bytes()).hexdigest()
    available_after = (datetime.fromisoformat(source['periods']['validation'][1]).replace(tzinfo=timezone.utc)
                       + timedelta(days=1)).isoformat()
    metadata = dict(version=1, model_type='swing-informer', target='pnl-r', prediction_unit='R',
        feature_schema=source['feature_schema'], swing_schema=SWING_SCHEMA, opportunity_schema=OPPORTUNITY_SCHEMA,
        window_key=f'pairs_{config.seq_len//2}', interval_seconds=source['opportunity_config']['interval_seconds'],
        opportunity_config=source['opportunity_config'], available_after=available_after,
        threshold=selected['threshold'], min_threshold=config.min_threshold, validation_qualified=bool(qualified),
        selection_policy=evaluation.get('selection_policy', 'independent_opportunities_v1'),
        best_epoch=state['best_epoch'], completed_epochs=state['epoch'], config=asdict(config),
        input_features=[*feature_names(source['feature_schema']), *['symbol_'+s for s in source['symbols']]]
            + (['stop_buffer_ratio'] if uses_stop_distance(source['opportunity_config']) else []),
        input_definition='swing_features_v1_with_log1p_volume_and_symbol_one_hot',
        normalization=normalization, resolver_id=digest, checkpoint_sha256=digest,
        upstream_sha256=upstream_hash(), scope={'symbols': source['symbols']}, source=source,
        validation=evaluation, threshold_candidates=candidates, history=state['history'],
        test_used_for_selection=False,
        interpretation='Independent overlapping opportunities, not a portfolio backtest or proof of an edge.')
    np.savez_compressed(output/'validation_predictions.npz', indices=validation, prediction=prediction,
                        actual=data.labels[validation], accepted=acceptance_mask(prediction, selected['threshold'], selection))
    write_report(output, metadata)
    # Metadata is the publication marker: a resumed training never mutates an
    # already published model, and interrupted finalization can simply be retried.
    atomic_json(output/'metadata.json', metadata)
    print(f"[swing-informer-selected] model={output} best_epoch={state['best_epoch']} "
          f"initial_threshold_r={selected['threshold']:.8g} validation_qualified={qualified} "
          f"accepted={selected['opportunities']} total_r={selected['total_r']:.3f}", flush=True)
    return metadata


def write_report(output, metadata):
    val, config = metadata['validation'], metadata['config']
    chosen = val['selected']
    lines = ['# Informer sur les swings', '', f"Collecte : `{metadata['source']['run_id']}`.",
        f"Cible : gain realise en R. Contexte : {config['seq_len']} swings. Bougies : {metadata['interval_seconds']/60:g} min.",
        f"Periodes : `{json.dumps(metadata['source']['periods'])}`.", '',
        f"**Seuil initial retenu : {metadata['threshold']:.8g} R.** Plancher : {metadata['min_threshold']:g} R.",
        f"Validation qualifiee : **{metadata['validation_qualified']}**. Meilleure epoque : {metadata['best_epoch']}.", '',
        f"MSE validation : {val['prediction_mse']:.6f}. MSE constante apprise sur train : {val['constant_mse']:.6f}.",
        f"Predictions validation : [{val['prediction_min']:.6f}, {val['prediction_max']:.6f}] R ; ecart-type {val['prediction_std']:.6f}.",
        f"Opportunites retenues : {chosen['opportunities']} ; somme {chosen['total_r']:.3f} R ; PF {chosen['profit_factor_r']}.", '',
        'Le decodeur ne contient que des variables observees, jamais les labels de gain.',
        'Normalisation ajustee sur train uniquement. Labels connus avant la fin de leur periode.',
        'La validation choisit l\'epoque et le seuil ; elle ne constitue pas un test independant.',
        'Les sommes de R portent sur des opportunites chevauchantes aux sorties de la collecte, pas sur un compte.',
        'Aucune periode de test n\'est evaluee automatiquement.']
    (Path(output)/'rapport.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


class SwingInformerResolver:
    def __init__(self, directory, *, device='cpu', threads=2):
        import torch
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory/'metadata.json').read_text(encoding='utf-8'))
        meta = self.metadata
        if meta.get('model_type') != 'swing-informer' or meta.get('version') != 1 or meta.get('target') != 'pnl-r':
            raise ValueError('Not a supported frozen swing Informer')
        if meta['upstream_sha256'] != upstream_hash():
            raise ValueError('Informer upstream source changed since training')
        if hashlib.sha256((self.directory/'best.pt').read_bytes()).hexdigest() != meta['checkpoint_sha256']:
            raise ValueError('Informer checkpoint hash mismatch')
        saved = torch.load(self.directory/'best.pt', map_location=device, weights_only=True)
        if saved['normalization'] != meta['normalization'] or saved['source'] != meta['source']:
            raise ValueError('Frozen Informer metadata and checkpoint differ')
        runtime_fields = {'epochs', 'device', 'threads', 'checkpoint_batches'}
        if (saved['upstream_sha256'] != meta['upstream_sha256'] or any(
                saved['config'][key] != value for key, value in meta['config'].items() if key not in runtime_fields)):
            raise ValueError('Frozen Informer configuration differs from its checkpoint')
        config = replace(InformerConfig(**meta['config']), device=device, threads=threads)
        devices = list(range(torch.cuda.device_count())) if device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            self.regressor = InformerRegressor(config, meta['normalization'], len(meta['input_features']))
        self.regressor.model.load_state_dict(saved['model'])
        self.regressor.model.eval().requires_grad_(False)

    def predict(self, points, *, as_of, symbol=None, stop_buffer_pips=None):
        points = tuple(points)
        if as_of < instant(self.metadata['available_after']):
            raise ValueError('Informer training/validation postdates this decision')
        symbols = self.metadata['scope']['symbols']
        symbol = symbol.upper() if symbol else symbols[0] if len(symbols) == 1 else None
        if symbol not in symbols or len(points) != self.regressor.config.seq_len:
            raise ValueError('Wrong symbol or swing count for this Informer')
        if points[-1].confirmed_at != as_of or any(p.confirmed_at > as_of for p in points):
            raise ValueError('Informer context must end exactly at the decision time')
        if any(p.kind not in ('high', 'low') or not p.pair_start <= p.pivot_at < p.confirmed_at for p in points):
            raise ValueError('Invalid confirmed swing geometry')
        needs_volume = self.metadata['feature_schema'].endswith('_mean_quote_volume_v2')
        if needs_volume and any(p.mean_quote_volume is None for p in points):
            raise ValueError('This Informer requires collected quote volumes')
        raw = np.asarray([[(p.price, 1 if p.kind == 'high' else -1, p.pivot_at.timestamp(),
            p.confirmed_at.timestamp(), p.mean_quote_volume or 0., symbols.index(symbol)) for p in points]])
        x, marks = features(raw, self.metadata['interval_seconds'], self.metadata['feature_schema'], len(symbols))
        if uses_stop_distance(self.metadata['opportunity_config']):
            from ultimateBot.backtester.research.swing_informer_data import append_stop_feature
            x = append_stop_feature(x, [stop_distance_feature(self.metadata['opportunity_config'], symbol,
                                                            points[-1].price, stop_buffer_pips)])
        prediction = float(self.regressor.predict(x, marks)[0])
        return dict(prediction=prediction, target='pnl-r', threshold=self.metadata['threshold'],
                    accepted=prediction >= self.metadata['threshold'], side=points[-1].side,
                    resolver_id=self.metadata['resolver_id'])
