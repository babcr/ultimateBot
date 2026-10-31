"""Closed-candle reentry selection; consumes no future fill or candle."""
from dataclasses import replace
from math import isfinite

from ultimateBot.backtester.strategy.signal import SignalSide


def select_reentry(signal, mid, rules):
    if (signal is None or (signal.entry_context or {}).get("entry_mode") != "failed-breakout"
            or rules.get("regression_reentry_quality_mode", "legacy") == "legacy"):
        return signal, None
    values = (mid.open_price, mid.high_price, mid.low_price, mid.close_price)
    if (not all(isfinite(value) and value > 0 for value in values)
            or not mid.low_price <= min(mid.open_price, mid.close_price)
            <= max(mid.open_price, mid.close_price) <= mid.high_price or mid.high_price == mid.low_price):
        return None, "reentry_quality_invalid_candle"
    wick = ((min(mid.open_price, mid.close_price) - mid.low_price) if signal.side == SignalSide.BUY
            else (mid.high_price - max(mid.open_price, mid.close_price))) / (mid.high_price - mid.low_price)
    bars = signal.entry_context["reentry_bars"]
    if not (bars == 1 or bars == 0 and wick <= rules["regression_reentry_max_same_bar_stopside_wick"]):
        return None, "reentry_quality_late" if bars > 1 else "reentry_quality_same_bar_wick"
    return replace(signal, entry_context={**signal.entry_context, "reentry_quality_mode": "selective",
        "signal_stopside_wick": wick, "reentry_quality_case": "next_bar" if bars == 1 else "same_bar_short_wick"}), None
