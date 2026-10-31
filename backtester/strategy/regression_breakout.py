"""Closed-candle breakout followed by a retest of a frozen regression channel."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from ultimateBot.backtester.market.candle_builder import Candle
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel
from ultimateBot.backtester.strategy.signal import Signal, SignalSide


@dataclass
class BreakoutEvent:
    channel: FrozenChannel
    side: SignalSide
    started_at: datetime
    last_bar: datetime
    breakout_close_atr: float
    bars: int = 0
    retest_started: bool = False
    low_bid: float = float("inf")
    high_ask: float = float("-inf")
    retest_trigger: float | None = None
    breakout_body_ratio: float = 0.0
    breakout_close_location: float = 0.0
    first_retest_bar: int | None = None


def directional_candle_quality(candle: Candle, direction: int) -> tuple[float, float]:
    """Body and close location in the trade direction; dojis cannot confirm."""
    size = candle.high_price - candle.low_price
    if size <= 0:
        return 0.0, 0.0
    body = direction * (candle.close_price - candle.open_price) / size
    location = ((candle.close_price - candle.low_price) if direction > 0
                else (candle.high_price - candle.close_price)) / size
    return body, location


class RegressionBreakoutTracker:
    def __init__(self):
        self.events: dict[str, BreakoutEvent] = {}

    def clear(self, symbol: str | None = None) -> None:
        if symbol is None:
            self.events.clear()
        else:
            self.events.pop(symbol, None)

    def evaluate(self, symbol: str, bid: Candle, ask: Candle, timestamp: datetime,
                 interval: timedelta, rules: dict[str, Any], channel: FrozenChannel | None
                 ) -> tuple[Signal | None, dict[str, Any]]:
        event = self.events.get(symbol)
        confirmed_mode = rules["regression_breakout_confirmation_mode"] == "confirmed"
        confirmation_window = rules["regression_breakout_confirmation_max_bars"] if confirmed_mode else 0
        channel = event.channel if event else channel
        analysis = deepcopy(channel.analysis if channel else {})
        analysis.update(entry_mode="breakout-retest", closed_at=bid.start_time,
                        symbol=symbol, reasons=[], direction="flat")

        def reject(reason, *, clear=False):
            if clear:
                self.clear(symbol)
            analysis["reasons"] = ["breakout_" + reason]
            return None, analysis

        if channel is None:
            return reject("channel_unavailable")
        if confirmed_mode:
            for quote in (bid, ask):
                values = (quote.open_price, quote.high_price, quote.low_price, quote.close_price)
                if (not all(isfinite(value) and value > 0 for value in values)
                        or not quote.low_price <= min(quote.open_price, quote.close_price)
                        <= max(quote.open_price, quote.close_price) <= quote.high_price):
                    return reject("invalid_candle", clear=True)
        previous_bar = event.last_bar if event else channel.fitted_through
        if ask.start_time != bid.start_time or bid.start_time - previous_bar != interval:
            return reject("history_gap", clear=True)
        if event:
            event.bars += 1
            event.last_bar = bid.start_time
            deadline = rules["regression_breakout_max_bars"]
            if confirmation_window and event.first_retest_bar is not None:
                deadline = event.first_retest_bar + confirmation_window
            if event.bars > deadline:
                analysis.update(breakout_bars=event.bars, first_retest_bar=event.first_retest_bar,
                                confirmation_deadline_bar=deadline)
                if confirmation_window:
                    reason = "confirmation_expired" if event.first_retest_bar is not None else "retest_expired"
                else:
                    reason = "expired"
                return reject(reason, clear=True)

        atr = float(channel.analysis["atr"])
        if not isfinite(atr) or atr <= 0:
            return reject("invalid_atr", clear=True)
        if not channel.analysis.get("anchored_line") or not channel.analysis.get("floating_line"):
            return reject("requires_two_lines", clear=True)
        n = int(channel.analysis["bar_count"])
        start = int(channel.analysis.get("regression_window_start", 0))
        horizon = n + rules["regression_breakout_max_bars"] + confirmation_window
        if any(not all(isfinite(p) for p in bounds) or bounds[0] >= bounds[1]
               for bounds in (channel.bounds(i) for i in (start, n - 1, horizon))):
            return reject("invalid_channel", clear=True)
        lower, upper = channel.bounds(n + (event.bars if event else 0))
        analysis.update(channel_fitted_through=channel.fitted_through, channel_lower=lower,
                        channel_upper=upper, breakout_bars=event.bars if event else 0)

        if event is None:
            old_lower, old_upper = channel.bounds(n - 1)
            if not (channel.previous_ask_close >= old_lower and channel.previous_bid_close <= old_upper):
                return reject("previous_close_outside")
            if bid.high_price > upper and ask.low_price < lower:
                return reject("ambiguous_candle")
            up = (bid.close_price - upper) / atr
            down = (lower - ask.close_price) / atr
            threshold = rules["regression_breakout_min_close_atr"]
            if up > 0 and up >= threshold:
                side, extension = SignalSide.BUY, up
            elif down > 0 and down >= threshold:
                side, extension = SignalSide.SELL, down
            else:
                return reject("no_close_outside")
            direction = 1 if side == SignalSide.BUY else -1
            quote = bid if direction > 0 else ask
            body, location = directional_candle_quality(quote, direction)
            analysis.update(breakout_body_ratio=body, breakout_close_location=location)
            if confirmed_mode:
                previous_close = channel.previous_bid_close if direction > 0 else channel.previous_ask_close
                # A moving regression boundary alone must not create a price breakout.
                if direction * (quote.close_price - previous_close) <= 0:
                    return reject("no_price_impulse")
                if body <= 0 or body < rules["regression_breakout_min_breakout_body_ratio"]:
                    return reject("weak_breakout_body")
                if location < rules["regression_breakout_min_close_location"]:
                    return reject("weak_breakout_close")
            self.events[symbol] = BreakoutEvent(
                channel, side, bid.start_time, bid.start_time, extension,
                breakout_body_ratio=body, breakout_close_location=location,
            )
            # A breakout candle cannot also establish its own retest.
            return reject("awaiting_retest")

        buying = event.side == SignalSide.BUY
        boundary = upper if buying else lower
        close = bid.close_price if buying else ask.close_price
        direction = 1 if buying else -1
        penetration = -direction * (close - boundary) / atr
        if (buying and bid.low_price <= lower) or (not buying and ask.high_price >= upper):
            return reject("opposite_boundary_touched", clear=True)
        if penetration > rules["regression_breakout_invalidation_atr"]:
            return reject("close_back_inside", clear=True)
        if confirmed_mode:
            wick_depth = ((boundary - bid.low_price) if buying else (ask.high_price - boundary)) / atr
            analysis["retest_depth_atr"] = max(0.0, wick_depth)
            if wick_depth > rules["regression_breakout_max_retest_depth_atr"]:
                return reject("retest_too_deep", clear=True)
        tolerance = rules["regression_breakout_retest_tolerance_atr"] * atr
        # Require an intersection with the boundary's zone, not a candle wholly away from it.
        touched = (bid.low_price <= boundary + tolerance and bid.high_price >= boundary - tolerance
                   if buying else ask.high_price >= boundary - tolerance and ask.low_price <= boundary + tolerance)
        if touched and not event.retest_started:
            event.first_retest_bar = event.bars
        event.retest_started |= touched
        if not event.retest_started:
            return reject("awaiting_retest")
        confirmation_bars = event.bars - event.first_retest_bar
        analysis.update(first_retest_bar=event.first_retest_bar, confirmation_bars=confirmation_bars,
                        confirmation_deadline_bar=(event.first_retest_bar + confirmation_window
                                                   if confirmation_window else rules["regression_breakout_max_bars"]))
        event.low_bid = min(event.low_bid, bid.low_price)
        event.high_ask = max(event.high_ask, ask.high_price)
        previous_trigger = event.retest_trigger
        if touched:
            # Keep this closed retest candle for the NEXT candle's comparison.
            # Never compare a close with its own high/low or future extrema.
            event.retest_trigger = bid.high_price if buying else ask.low_price
        open_price = bid.open_price if buying else ask.open_price
        if direction * (close - boundary) <= 0 or direction * (close - open_price) <= 0:
            return reject("awaiting_confirmation")
        confirmation_level = boundary
        confirmation_body, confirmation_location = directional_candle_quality(bid if buying else ask, direction)
        if confirmed_mode:
            analysis.update(confirmation_body_ratio=confirmation_body,
                            confirmation_close_location=confirmation_location,
                            retest_trigger=previous_trigger)
            if previous_trigger is None:
                return reject("awaiting_reclaim")
            confirmation_level = ((max(boundary, previous_trigger) if buying else min(boundary, previous_trigger))
                                  + direction * rules["regression_breakout_confirmation_buffer_atr"] * atr)
            analysis["confirmation_level"] = confirmation_level
            if direction * (close - confirmation_level) <= 0:
                return reject("awaiting_reclaim")
            if confirmation_body < rules["regression_breakout_min_confirmation_body_ratio"]:
                return reject("weak_confirmation_body")
            if confirmation_location < rules["regression_breakout_min_close_location"]:
                return reject("weak_confirmation_close")

        # Consume confirmation even if execution, sizing, session or validation rejects it.
        self.clear(symbol)
        maximum_extension = rules["regression_breakout_max_entry_distance_atr"] * atr
        executable_close = ask.close_price if buying else bid.close_price
        if direction * (executable_close - boundary) > maximum_extension:
            return reject("entry_too_far")
        buffer = rules["regression_breakout_stop_buffer_atr"] * atr
        stop = event.low_bid - buffer if buying else event.high_ask + buffer
        analysis.update(direction="buy" if buying else "sell", breakout_stop=stop)
        entry_context = {
            "entry_mode": "breakout-retest", "channel_fitted_through": channel.fitted_through.isoformat(),
            "confirmation_mode": rules["regression_breakout_confirmation_mode"],
            "breakout_started_at": event.started_at.isoformat(), "confirmation_bar": bid.start_time.isoformat(),
            "breakout_bars": event.bars, "breakout_close_atr": event.breakout_close_atr,
            "first_retest_bar": event.first_retest_bar, "confirmation_bars": confirmation_bars,
            "confirmation_max_bars": confirmation_window,
            "breakout_body_ratio": event.breakout_body_ratio,
            "breakout_close_location": event.breakout_close_location,
            "confirmation_body_ratio": confirmation_body,
            "confirmation_close_location": confirmation_location,
            "retest_trigger": previous_trigger, "confirmation_level": confirmation_level,
            "regression_lookback": int(channel.analysis.get("regression_lookback_bars", n - start)),
            "atr": atr, "channel_lower": lower, "channel_upper": upper,
            "retest_low_bid": event.low_bid, "retest_high_ask": event.high_ask,
            "stop_loss": stop, "anchored_line": channel.analysis["anchored_line"],
            "floating_line": channel.analysis["floating_line"],
            "minimum_entry_price": confirmation_level if buying else boundary - maximum_extension,
            "maximum_entry_price": boundary + maximum_extension if buying else confirmation_level,
        }
        return Signal(symbol, timestamp, event.side, stop_loss_price=stop,
                      minimum_entry_price=confirmation_level if buying else boundary - maximum_extension,
                      maximum_entry_price=boundary + maximum_extension if buying else confirmation_level,
                      entry_context=entry_context), analysis
