"""Causal failed-breakout events against a channel fitted before the excursion."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.signal import Signal, SignalSide


@dataclass(frozen=True)
class FrozenChannel:
    analysis: dict[str, Any]
    fitted_through: datetime
    previous_bid_close: float
    previous_ask_close: float

    def bounds(self, index: int) -> tuple[float, float]:
        lines = {line["kind"]: line for line in (
            self.analysis["anchored_line"], self.analysis["floating_line"])}
        return tuple(float(lines[kind]["intercept"] + lines[kind]["slope"] * index)
                     for kind in ("low", "high"))


@dataclass
class Excursion:
    channel: FrozenChannel
    side: SignalSide
    started_at: datetime
    last_bar: datetime
    low_bid: float
    high_ask: float
    excursion_atr: float
    bars: int = 0


class RegressionReentryTracker:
    def __init__(self):
        self.events: dict[str, Excursion] = {}

    def clear(self, symbol: str | None = None) -> None:
        if symbol is None:
            self.events.clear()
        else:
            self.events.pop(symbol, None)

    def evaluate(
        self, symbol: str, bid: Candle, ask: Candle, timestamp: datetime,
        interval: timedelta, rules: dict[str, Any], channel: FrozenChannel | None,
    ) -> tuple[Signal | None, dict[str, Any]]:
        event = self.events.get(symbol)
        analysis = deepcopy(event.channel.analysis if event else channel.analysis if channel else {})
        analysis.update(entry_mode="failed-breakout", closed_at=bid.start_time, symbol=symbol,
                        reasons=[], direction="flat")

        def reject(reason):
            analysis["reasons"] = [reason]
            return None, analysis

        expiry = rules["regression_reentry_max_bars"]
        if event:
            if bid.start_time - event.last_bar != interval:
                self.clear(symbol)
                return reject("reentry_history_gap")
            event.bars += 1
            event.last_bar = bid.start_time
            if event.bars > expiry:
                self.clear(symbol)
                return reject("reentry_expired")
            channel = event.channel
        elif channel is None:
            return reject("reentry_channel_unavailable")
        elif bid.start_time - channel.fitted_through != interval:
            return reject("reentry_history_gap")

        n = int(channel.analysis["bar_count"])
        fit_start = int(channel.analysis.get("regression_window_start", 0))
        lookback = int(channel.analysis.get("regression_lookback_bars", n - fit_start))
        atr = float(channel.analysis["atr"])
        if not channel.analysis.get("floating_line"):
            self.clear(symbol)
            return reject("reentry_requires_two_lines")
        # Reject intersecting lines over the fitted window and detection horizon.
        widths = [upper - lower for lower, upper in
                  (channel.bounds(i) for i in (fit_start, n - 1, n + expiry))]
        if not all(isfinite(width) and width > 0 for width in widths):
            self.clear(symbol)
            return reject("reentry_invalid_channel")
        drift = max(abs(channel.analysis[key]["slope"]) for key in ("anchored_line", "floating_line"))
        drift_ratio = drift * lookback / min(widths)
        analysis["channel_drift_ratio"] = drift_ratio
        if drift_ratio > rules["regression_reentry_max_drift_ratio"]:
            self.clear(symbol)
            return reject("reentry_channel_too_directional")
        lower, upper = channel.bounds(n + (event.bars if event else 0))
        up = (bid.high_price - upper) / atr
        down = (lower - ask.low_price) / atr
        threshold = rules["regression_reentry_min_excursion_atr"]
        above = up > 0 and up >= threshold
        below = down > 0 and down >= threshold
        if above and below:
            self.clear(symbol)
            return reject("reentry_ambiguous_excursion")
        if event is None:
            previous_lower, previous_upper = channel.bounds(n - 1)
            if not (previous_lower <= channel.previous_ask_close
                    and channel.previous_bid_close <= previous_upper):
                return reject("reentry_previous_close_outside")
            if not above and not below:
                return reject("reentry_no_excursion")
            event = Excursion(channel, SignalSide.SELL if above else SignalSide.BUY,
                              bid.start_time, bid.start_time, bid.low_price, ask.high_price,
                              up if above else down)
            self.events[symbol] = event
        elif (event.side == SignalSide.SELL and below) or (event.side == SignalSide.BUY and above):
            self.clear(symbol)
            return reject("reentry_opposite_boundary_touched")
        event.low_bid = min(event.low_bid, bid.low_price)
        event.high_ask = max(event.high_ask, ask.high_price)
        event.excursion_atr = max(event.excursion_atr, up if event.side == SignalSide.SELL else down)
        analysis.update(channel_fitted_through=channel.fitted_through,
                        reentry_bars=event.bars, excursion_atr=event.excursion_atr,
                        channel_lower=lower, channel_upper=upper)
        if not (bid.close_price < upper and ask.close_price > lower):
            return reject("reentry_awaiting_close_inside")

        # Consume once, even if sizing/session/spread subsequently rejects the order.
        self.clear(symbol)
        buffer = rules["regression_reentry_stop_buffer_atr"] * atr
        stop = event.low_bid - buffer if event.side == SignalSide.BUY else event.high_ask + buffer
        target = (lower + upper) / 2
        analysis.update(direction="buy" if event.side == SignalSide.BUY else "sell",
                        reentry_stop=stop, reentry_target=target)
        entry_context = {
            "entry_mode": "failed-breakout", "channel_fitted_through": channel.fitted_through.isoformat(),
            "excursion_started_at": event.started_at.isoformat(), "confirmation_bar": bid.start_time.isoformat(),
            "regression_lookback": lookback, "atr": atr, "channel_lower": lower, "channel_upper": upper,
            "channel_drift_ratio": drift_ratio, "excursion_atr": event.excursion_atr,
            "reentry_bars": event.bars, "stop_loss": stop, "take_profit": target,
            "anchored_line": channel.analysis["anchored_line"],
            "floating_line": channel.analysis["floating_line"],
        }
        return Signal(symbol, timestamp, event.side, stop_loss_price=stop, take_profit_price=target,
                      minimum_reward_risk_ratio=rules["regression_reentry_min_rr"],
                      entry_context=entry_context), analysis
