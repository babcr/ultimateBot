"""Frozen TS2Vec representations of observed, completed market candles.

The upstream submodule stays untouched. No price, scaler or model update from
after ``as_of`` is allowed into a representation used by the trading dataset.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import os
import random
import sys
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np


DEFAULT_DATA_DIRECTORY = Path("C:/TradingBotData")
DEFAULT_MODEL_PATH = DEFAULT_DATA_DIRECTORY / "models" / "ts2vec_order_creation.pt"
FEATURE_SCHEMA = "mid_ohlc_log_previous_close_bps_v1"
UPSTREAM = Path(__file__).resolve().parents[2] / "ts2vec"


def upstream_source_hash():
    files = [UPSTREAM / "ts2vec.py", UPSTREAM / "utils.py", *sorted((UPSTREAM / "models").glob("*.py"))]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


@lru_cache(maxsize=1)
def upstream_class():
    """Load absolute upstream imports in a private namespace, without sys.path changes."""
    if not (UPSTREAM / "ts2vec.py").is_file():
        raise RuntimeError("TS2Vec submodule missing: git submodule update --init --recursive")
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise RuntimeError("TS2Vec requires PyTorch: py -m pip install -r requirements-embeddings.txt") from error
    namespace = "_ultimatebot_ts2vec_vendor"
    package = types.ModuleType(namespace)
    package.__path__ = [str(UPSTREAM)]
    sys.modules[namespace] = package
    # Upstream's models package already uses relative imports internally.
    modules = {name: importlib.import_module(f"{namespace}.{name}")
               for name in ("models", "models.losses", "utils")}
    original_import = builtins.__import__

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return original_import(name, globals, locals, fromlist, level)

    spec = importlib.util.spec_from_file_location(f"{namespace}.ts2vec", UPSTREAM / "ts2vec.py")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": scoped_import}
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TS2Vec


def candle_features(candles) -> np.ndarray:
    """Four log OHLC ratios to the preceding close, in basis points (not pips)."""
    prices = np.asarray([[c.open_price, c.high_price, c.low_price, c.close_price]
                         for c in candles], dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] != 4 or len(prices) < 2:
        raise ValueError("An embedding requires at least two completed OHLC candles")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("TS2Vec requires finite, positive OHLC prices")
    previous = np.concatenate(([prices[0, 0]], prices[:-1, 3]))
    return (np.log(prices / previous[:, None]) * 10_000.0).astype(np.float32)


@dataclass(frozen=True)
class EmbeddingConfig:
    context_bars: int = 150
    output_dims: int = 320
    hidden_dims: int = 64
    depth: int = 10
    train_iters: int = 200
    seed: int = 1
    device: str = "cpu"
    threads: int = 2

    def __post_init__(self):
        if self.context_bars < 2:
            raise ValueError("TS2Vec context requires at least two candles")
        if min(self.output_dims, self.hidden_dims, self.depth, self.train_iters, self.threads) < 1:
            raise ValueError("TS2Vec dimensions, depth, iterations and threads must be positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("TS2Vec seed must be in [0, 2**32)")


@dataclass(frozen=True)
class EmbeddingSnapshot:
    model_id: str
    values: tuple[float, ...]
    context_start: str
    context_end: str
    context_bars: int


class TS2VecService:
    """Reusable encoder: call ``embed_market`` at any historical/live decision time.

    When no checkpoint exists, bootstrap once from histories already observed at
    the first request. A single persisted model is then shared by every seed.
    A separately pretrained checkpoint can be supplied with the same API.
    """

    def __init__(self, checkpoint=DEFAULT_MODEL_PATH, config=EmbeddingConfig(),
                 candle_interval=timedelta(minutes=10)):
        self.checkpoint = Path(checkpoint).resolve()
        self.config = config
        self.candle_interval = candle_interval
        if candle_interval <= timedelta(0):
            raise ValueError("Embedding candle interval must be positive")
        self.model = None
        self.model_id = None
        self.metadata = None
        self._cache = {}
        # Check dependencies immediately, before a potentially long replay.
        self._class = upstream_class()
        import torch
        torch.set_num_threads(config.threads)
        torch.empty(0, device=config.device)  # Fail immediately for an unavailable device.
        if self.checkpoint.exists():
            self._load()

    def _new_model(self):
        return self._class(input_dims=4, output_dims=self.config.output_dims,
                           hidden_dims=self.config.hidden_dims, depth=self.config.depth,
                           device=self.config.device, batch_size=8,
                           max_train_length=self.config.context_bars)

    def completed_candles(self, market, symbol, as_of):
        return [c for c in market.get_candles(symbol, include_active=False,
                    price_source="mid", limit=self.config.context_bars,
                    candle_interval=self.candle_interval)
                if utc(c.start_time) + self.candle_interval <= utc(as_of)]

    def fit_histories(self, histories: dict, *, as_of: datetime, source="observed_history_bootstrap"):
        """Train once, using only supplied completed candles before the cutoff."""
        if self.model is not None or self.checkpoint.exists():
            raise ValueError("Encoder already exists; choose a new checkpoint to train a new version")
        usable = {}
        for symbol, candles in histories.items():
            past = [c for c in candles if utc(c.start_time) + self.candle_interval <= utc(as_of)]
            if len(past) >= 2:
                usable[symbol] = past[-self.config.context_bars:]
        if not usable:
            raise ValueError("No observed completed candles available to train TS2Vec")
        # Overlapping windows provide instance negatives as well as temporal negatives.
        length = min(64, min(len(c) for c in usable.values()))
        windows = []
        for candles in usable.values():
            for end in range(length, len(candles) + 1, max(1, length // 2)):
                windows.append(candle_features(candles[end - length:end]))
            if len(candles) > length:
                windows.append(candle_features(candles[-length:]))
        train = np.stack(windows)
        self.mean = train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        self.scale = np.maximum(train.std(axis=(0, 1), dtype=np.float64), 1e-6).astype(np.float32)
        train = ((train - self.mean) / self.scale).astype(np.float32)
        import torch
        numpy_state, python_state = np.random.get_state(), random.getstate()
        cuda_devices = list(range(torch.cuda.device_count())) if str(self.config.device).startswith("cuda") else []
        print(f"[ts2vec-training] windows={len(train)} length={length} "
              f"iterations={self.config.train_iters} cutoff={utc(as_of).isoformat()}", flush=True)
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                np.random.seed(self.config.seed)
                random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
                self.model = self._new_model()
                losses = self.model.fit(train, n_iters=self.config.train_iters)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        if not all(torch.isfinite(p).all().item() for p in self.model.net.parameters()):
            raise ValueError("TS2Vec training produced non-finite weights")
        self.metadata = {
            "format_version": 1, "feature_schema": FEATURE_SCHEMA,
            "config": asdict(self.config), "candle_interval_seconds": self.candle_interval.total_seconds(),
            "trained_through": max(utc(c[-1].start_time) + self.candle_interval for c in usable.values()).isoformat(),
            "training_requested_at": utc(as_of).isoformat(), "training_source": source,
            "training_symbols": sorted(usable), "training_windows": len(train), "training_window_bars": length,
            "training_start": min(utc(c[0].start_time) for c in usable.values()).isoformat(),
            "training_iterations": self.model.n_iters,
            "last_epoch_loss": float(losses[-1]) if losses else None,
            "pooling": "full_series_max", "upstream_source_sha256": upstream_source_hash(),
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "scaler_mean": self.mean.tolist(), "scaler_scale": self.scale.tolist(),
        }
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint.with_name(f".{self.checkpoint.name}.{uuid4().hex}.tmp")
        try:
            torch.save({"metadata": self.metadata, "state_dict": self.model.net.state_dict()}, temporary)
            # Never overwrite a model created by another process while training.
            os.link(temporary, self.checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
        self.model_id = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.model.net.eval().requires_grad_(False)
        print(f"[ts2vec-ready] model_id={self.model_id} checkpoint={self.checkpoint}", flush=True)

    def _load(self):
        import torch
        saved = torch.load(self.checkpoint, map_location=self.config.device, weights_only=True)
        self.metadata = saved["metadata"]
        if self.metadata["format_version"] != 1 or self.metadata["feature_schema"] != FEATURE_SCHEMA:
            raise ValueError("Unsupported TS2Vec checkpoint or feature schema")
        if self.metadata["upstream_source_sha256"] != upstream_source_hash():
            raise ValueError("TS2Vec upstream sources differ from those used to train this checkpoint")
        for name in ("context_bars", "output_dims", "hidden_dims", "depth"):
            if self.metadata["config"][name] != getattr(self.config, name):
                raise ValueError(f"TS2Vec checkpoint has a different {name}; use its settings or a new model path")
        if self.metadata["candle_interval_seconds"] != self.candle_interval.total_seconds():
            raise ValueError("TS2Vec checkpoint uses a different candle interval")
        self.mean = np.asarray(self.metadata["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(self.metadata["scaler_scale"], dtype=np.float32)
        # Model construction must not change the bot's RNG state, including on reload.
        with torch.random.fork_rng(devices=[]):
            self.model = self._new_model()
        self.model.net.load_state_dict(saved["state_dict"])
        self.model.net.eval().requires_grad_(False)
        self.model_id = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()

    def embed_candles(self, candles, *, as_of: datetime) -> EmbeddingSnapshot:
        if self.model is None:
            raise ValueError("Train or load TS2Vec before requesting an embedding")
        if utc(as_of) < utc(datetime.fromisoformat(self.metadata["trained_through"])):
            raise ValueError("TS2Vec checkpoint was trained on data later than this decision (look-ahead)")
        candles = list(candles)[-self.config.context_bars:]
        if any(utc(c.start_time) + self.candle_interval > utc(as_of) for c in candles):
            raise ValueError("An embedding cannot include an active or future candle")
        if any(utc(a.start_time) >= utc(b.start_time) for a, b in zip(candles, candles[1:])):
            raise ValueError("Embedding candles must be in strictly increasing time order")
        values = candle_features(candles)
        values = ((values - self.mean) / self.scale).astype(np.float32)
        # Full-series pooling is causal at the decision: all input is in its past.
        # causal=True alone in upstream only affects its sliding-window mode.
        vector = self.model.encode(values[None, ...], encoding_window="full_series")[0]
        if vector.shape != (self.config.output_dims,) or not np.isfinite(vector).all():
            raise ValueError("Invalid TS2Vec embedding")
        return EmbeddingSnapshot(self.model_id, tuple(float(v) for v in vector),
                                 utc(candles[0].start_time).isoformat(),
                                 (utc(candles[-1].start_time) + self.candle_interval).isoformat(), len(candles))

    def embed_market(self, market, symbol: str, *, as_of: datetime) -> EmbeddingSnapshot:
        candles = self.completed_candles(market, symbol, as_of)
        if not candles:
            raise ValueError(f"No completed embedding context for {symbol}")
        # Check the cutoff even on cache hits when replaying or reusing a service.
        if self.metadata is None or utc(as_of) < utc(datetime.fromisoformat(self.metadata["trained_through"])):
            raise ValueError("TS2Vec is untrained or its checkpoint is from the future")
        key = (utc(candles[-1].start_time), len(candles))
        cached = self._cache.get(symbol)
        if cached is None or cached[0] != key:
            cached = (key, self.embed_candles(candles, as_of=as_of))
            self._cache[symbol] = cached
        return cached[1]
