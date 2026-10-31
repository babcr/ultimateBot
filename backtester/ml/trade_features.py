"""One feature contract shared by training and inference at order creation."""
from __future__ import annotations

import numpy as np


FEATURE_CONTRACT = "ts2vec_regression_trailing_side_order_creation_v1"


def feature_names(dimensions):
    return [f"embedding_{i:03d}" for i in range(dimensions)] + ["regression_lookback", "trailing", "side"]


def feature_vector(embedding, lookback, trailing, side, *, dimensions):
    values = np.asarray(embedding, dtype=np.float32)
    if values.shape != (dimensions,) or not np.isfinite(values).all():
        raise ValueError("Invalid embedding shape or values for trade validator")
    if int(lookback) != lookback or lookback < 2 or trailing not in (0, 1) or side not in (-1, 1):
        raise ValueError("Invalid lookback, trailing or side for trade validator")
    return np.concatenate((values, np.asarray([lookback, trailing, side], dtype=np.float32)))


def stamp_order_parameters(order, strategy):
    """Pending orders retain their generating configuration until their fill."""
    if order.strategy_lookback_bars is None:
        order.strategy_lookback_bars = strategy.get_history_lookback_bars(order.symbol)
    if order.strategy_timeframe_minutes is None:
        order.strategy_timeframe_minutes = strategy.get_analysis_interval(order.symbol).total_seconds() / 60
    if order.strategy_use_global_timeframe_validation is None:
        order.strategy_use_global_timeframe_validation = strategy.get_global_timeframe_validation_enabled(order.symbol)
