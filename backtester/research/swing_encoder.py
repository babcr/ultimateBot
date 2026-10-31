"""Frozen TS2Vec on irregular swing sequences, never on OHLC candles."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from uuid import uuid4

import numpy as np

from ultimateBot.backtester.embeddings.ts2vec_service import upstream_class, upstream_source_hash
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMA, FEATURE_SCHEMAS, feature_names


@dataclass(frozen=True)
class SwingEncoderConfig:
    output_dims: int = 64
    hidden_dims: int = 32
    depth: int = 0
    train_iters: int = 0
    train_epochs: int = 1
    learning_rate: float = .001
    batch_size: int = 16
    max_train_length: int = 0
    max_train_samples: int = 0
    seed: int = 1
    threads: int = 2
    device: str = "cpu"

    def __post_init__(self):
        for key in ("output_dims", "hidden_dims", "train_epochs", "batch_size", "threads"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("depth", "train_iters", "max_train_length", "max_train_samples"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if self.max_train_length == 1 or not 0 <= self.seed < 2**31:
            raise ValueError("Length limit must be zero (full windows) or at least two; seed must be in [0,2**31)")
        if self.batch_size < 2 or self.max_train_samples == 1:
            raise ValueError('Instance contrastive learning requires at least two sequences per batch and sample selection')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('TS2Vec learning_rate must be in (0,1]')


def encoder_training_plan(config, sequence_lengths, *, feature_schema=FEATURE_SCHEMA):
    """Resolve workload and receptive field from training inputs only, without fitting."""
    lengths = np.asarray(sequence_lengths, dtype=np.int64)
    if lengths.ndim != 1 or not len(lengths) or (lengths < 2).any():
        raise ValueError('Require training sequence lengths of at least two swings')
    maximum = int(lengths.max())
    if config.max_train_length and maximum > config.max_train_length:
        raise ValueError(f'Full window has {maximum} swings, exceeding max_train_length={config.max_train_length}; '
                         'use --ts2vec-max-train-length 0 or raise the limit. Windows are never silently truncated.')
    # Each of depth+1 residual blocks contains two width-3 convolutions,
    # with dilations 1,2,...,2**depth: receptive field = 1+4*(2**(depth+1)-1).
    depth = config.depth or max(1, math.ceil(math.log2((maximum+3)/4))-1)
    receptive_field = 1+4*(2**(depth+1)-1)
    indices = np.arange(len(lengths))
    if config.max_train_samples and len(indices) > config.max_train_samples:
        indices = np.sort(np.random.default_rng(config.seed).choice(indices, config.max_train_samples, replace=False))
    selected = lengths[indices]
    samples = len(selected)
    if samples < 2:
        raise ValueError('Instance contrastive learning requires at least two training sequences')
    batch = min(config.batch_size, samples)
    batches = samples//batch  # The upstream loader uses drop_last=True.
    updates = config.train_iters or config.train_epochs*batches
    presentations = updates*batch
    warnings = []
    if config.train_iters and presentations < samples:
        warnings.append('iteration_budget_below_one_data_pass')
    if receptive_field < maximum:
        warnings.append('explicit_depth_receptive_field_shorter_than_training_window')
    if samples < len(lengths):
        warnings.append('explicit_training_sample_limit')
    return {'budget_mode': 'iterations' if config.train_iters else 'epochs',
            'available_training_sequences': int(len(lengths)), 'selected_training_sequences': int(samples),
            'batch_size': batch, 'full_batches_per_epoch': batches,
            'dropped_sequences_per_epoch': samples % batch,
            'epochs_requested': None if config.train_iters else config.train_epochs,
            'optimizer_updates': updates, 'sequence_presentations': presentations,
            'equivalent_data_passes': presentations/samples,
            'sequence_length_min': int(lengths.min()), 'sequence_length_max': maximum,
            'padded_training_length': int(selected.max()),
            'training_buffer_bytes': int(samples*selected.max()*len(feature_names(feature_schema))*4),
            'depth': depth, 'receptive_field_swings': receptive_field,
            'sequence_policy': 'full_window', 'truncated_sequences': 0, 'warnings': warnings}


class SwingEncoder:
    def __init__(self, config=SwingEncoderConfig(), *, feature_schema=FEATURE_SCHEMA):
        self.config = config
        self.feature_schema = feature_schema
        self.feature_names = feature_names(feature_schema)
        self.model = None
        self.metadata = None
        self.model_id = None

    def _new_model(self):
        return upstream_class()(input_dims=len(self.feature_names), output_dims=self.config.output_dims,
                                hidden_dims=self.config.hidden_dims, depth=self.config.depth,
                                device=self.config.device, batch_size=self.config.batch_size,
                                lr=self.config.learning_rate, max_train_length=None)

    def _check(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != len(self.feature_names) or len(sequence) < 2 or not np.isfinite(sequence).all():
            raise ValueError(f"TS2Vec requires finite [swings,{len(self.feature_names)}] features and at least two alternating swings")
        return sequence

    def fit(self, sequences, *, observed_at, cutoff: datetime, checkpoint, provenance):
        if self.model is not None or Path(checkpoint).exists():
            raise ValueError("Encoder already exists; use a new output path")
        if not sequences or len(sequences) != len(observed_at) or any(t > cutoff for t in observed_at):
            raise ValueError("Encoder pretraining must contain only contexts observed by its cutoff")
        requested_config = self.config
        plan = encoder_training_plan(self.config, [len(s) for s in sequences], feature_schema=self.feature_schema)
        print('[swing-ts2vec-plan] ' + json.dumps(plan, sort_keys=True), flush=True)
        self.config = replace(self.config, depth=plan['depth'])
        import torch
        torch.set_num_threads(self.config.threads)
        torch.empty(0, device=self.config.device)
        rng = np.random.default_rng(self.config.seed)
        selected = np.arange(len(sequences))
        if self.config.max_train_samples and len(selected) > self.config.max_train_samples:
            selected = np.sort(rng.choice(selected, self.config.max_train_samples, replace=False))
        arrays = [self._check(sequences[i]) for i in selected]
        # Avoid an additional concatenated copy of all full-length inputs.
        total = sum(map(len, arrays))
        sums = sum((a.sum(axis=0, dtype=np.float64) for a in arrays), np.zeros(len(self.feature_names)))
        squares = sum((np.square(a, dtype=np.float64).sum(axis=0) for a in arrays), np.zeros(len(self.feature_names)))
        mean = sums/total
        self.mean = mean.astype(np.float32)
        self.scale = np.maximum(np.sqrt(np.maximum(squares/total-mean**2, 0)), 1e-6).astype(np.float32)
        length = max(map(len, arrays))
        train = np.full((len(arrays), length, len(self.feature_names)), np.nan, dtype=np.float32)
        for i, array in enumerate(arrays):
            train[i, :len(array)] = (array - self.mean) / self.scale
        numpy_state, python_state = np.random.get_state(), random.getstate()
        devices = list(range(torch.cuda.device_count())) if self.config.device.startswith("cuda") else []
        iteration_losses = []
        def record_iteration(model, loss):
            iteration_losses.append(float(loss))
            if model.n_iters == 1 or model.n_iters % max(1, plan['full_batches_per_epoch']) == 0:
                print(f"[swing-ts2vec] updates={model.n_iters}/{plan['optimizer_updates']} loss={loss:.6f}", flush=True)
        try:
            with torch.random.fork_rng(devices=devices):
                np.random.seed(self.config.seed)
                random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
                self.model = self._new_model()
                self.model.after_iter_callback = record_iteration
                losses = self.model.fit(train, n_iters=self.config.train_iters or None,
                                        n_epochs=None if self.config.train_iters else self.config.train_epochs)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        if not all(torch.isfinite(p).all().item() for p in self.model.net.parameters()):
            raise ValueError("Nonfinite TS2Vec weights")
        self.metadata = {"version": 2, "feature_schema": self.feature_schema, "feature_names": list(self.feature_names),
                         "config": asdict(self.config), "trained_through": max(observed_at).isoformat(),
                         "requested_config": asdict(requested_config), "training_plan": plan,
                         "optimizer_updates": self.model.n_iters, "completed_epochs": self.model.n_epochs,
                         "available_after": cutoff.isoformat(), "pretrain_samples": len(arrays),
                         "mean": self.mean.tolist(), "scale": self.scale.tolist(), "provenance": provenance,
                         "pooling": "full_series_max_exact_length_no_padding_at_inference",
                         "upstream_sha256": upstream_source_hash(), "torch_version": str(torch.__version__),
                         "numpy_version": np.__version__,
                         "epoch_losses": [float(v) for v in losses],
                         "last_loss": iteration_losses[-1] if iteration_losses else None}
        target = Path(checkpoint).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            torch.save({"metadata": self.metadata, "state_dict": self.model.net.state_dict()}, temporary)
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self.model_id = hashlib.sha256(target.read_bytes()).hexdigest()
        self.model.net.eval().requires_grad_(False)
        return self

    @classmethod
    def load(cls, checkpoint, *, device="cpu", threads=2):
        import torch
        torch.set_num_threads(threads)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        metadata = saved["metadata"]
        if metadata["version"] not in (1, 2) or metadata["feature_schema"] not in FEATURE_SCHEMAS or metadata["upstream_sha256"] != upstream_source_hash():
            raise ValueError("Incompatible swing encoder checkpoint")
        service = cls(SwingEncoderConfig(**{**metadata["config"], "device": device, "threads": threads}),
                      feature_schema=metadata['feature_schema'])
        if metadata.get('feature_names') != list(service.feature_names):
            raise ValueError('Encoder feature names do not match its schema')
        devices = list(range(torch.cuda.device_count())) if device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            service.model = service._new_model()
        service.model.net.load_state_dict(saved["state_dict"])
        service.model.net.eval().requires_grad_(False)
        service.metadata = metadata
        service.mean = np.asarray(metadata["mean"], dtype=np.float32)
        service.scale = np.asarray(metadata["scale"], dtype=np.float32)
        if (service.mean.shape != (len(service.feature_names),) or service.scale.shape != service.mean.shape
                or not np.isfinite(service.mean).all() or not np.isfinite(service.scale).all() or (service.scale <= 0).any()):
            raise ValueError("Invalid encoder scaler")
        service.model_id = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        return service

    def encode(self, sequences, *, observed_at, training_reconstruction=False):
        if self.model is None or len(sequences) != len(observed_at):
            raise ValueError("Train/load the swing encoder and provide one timestamp per context")
        cutoff = datetime.fromisoformat(self.metadata["available_after"])
        if training_reconstruction:
            if any(at >= cutoff for at in observed_at):
                raise ValueError("Training reconstruction is restricted to the encoder training period")
        elif any(at < cutoff for at in observed_at):
            raise ValueError("Encoder postdates this decision; historical training reconstruction must be explicit")
        if not sequences:
            return np.empty((0, self.config.output_dims), dtype=np.float32)
        # No padding can influence max pooling. Batch only identical sequence lengths.
        grouped = defaultdict(list)
        for i, sequence in enumerate(sequences):
            array = self._check(sequence)
            if (self.metadata['version'] >= 2 and self.config.max_train_length
                    and len(array) > self.config.max_train_length):
                raise ValueError('Full input window exceeds the frozen encoder length limit; truncation is forbidden')
            grouped[len(array)].append((i, array))
        result = np.empty((len(sequences), self.config.output_dims), dtype=np.float32)
        for group in grouped.values():
            for start in range(0, len(group), self.config.batch_size):
                batch = group[start:start + self.config.batch_size]
                inputs = np.stack([(array - self.mean) / self.scale for _, array in batch])
                vectors = self.model.encode(inputs, encoding_window="full_series", batch_size=self.config.batch_size)
                result[[i for i, _ in batch]] = vectors
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite TS2Vec representations")
        return result
