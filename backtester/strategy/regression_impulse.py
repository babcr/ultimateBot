"""Compression breakout confirmed only by the immediately following outside hold."""
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

from ultimateBot.backtester.strategy.regression_breakout import directional_candle_quality
from ultimateBot.backtester.strategy.signal import Signal,SignalSide


@dataclass
class ImpulseEvent:
    channel: object
    last_bar: object
    direction: int
    close: float


class RegressionImpulseTracker:
    entry_name = "breakout-hold"

    def __init__(self):
        self.events = {}

    def clear(self,symbol=None):
        if symbol is None:
            self.events.clear()
        else:
            self.events.pop(symbol,None)

    def evaluate(self,symbol,bid,ask,timestamp,interval,rules,channel):
        event = self.events.get(symbol)
        channel = event.channel if event else channel
        analysis = deepcopy(channel.analysis if channel else {})
        analysis.update(entry_mode=self.entry_name,symbol=symbol,closed_at=bid.start_time,direction="flat",reasons=[])

        def reject(reason,clear=False):
            if clear:
                self.clear(symbol)
            analysis["reasons"] = ["impulse_"+reason]
            return None,analysis

        if channel is None:
            return reject("channel_unavailable")
        if ask.start_time != bid.start_time or bid.start_time-(event.last_bar if event else channel.fitted_through) != interval:
            return reject("history_gap",True)
        for quote in (bid,ask):
            if (not all(isfinite(value) and value > 0 for value in (quote.open_price,quote.high_price,quote.low_price,quote.close_price))
                    or not quote.low_price <= min(quote.open_price,quote.close_price)
                    <= max(quote.open_price,quote.close_price) <= quote.high_price):
                return reject("invalid_candle",True)
        if not analysis.get("anchored_line") or not analysis.get("floating_line"):
            return reject("requires_two_lines",True)
        atr,n = analysis["atr"],int(analysis["bar_count"])
        if not isfinite(atr) or atr <= 0:
            return reject("invalid_atr",True)
        bounds = [channel.bounds(index) for index in (int(analysis.get("regression_window_start",0)),n-1,n+1)]
        if any(not all(isfinite(value) for value in pair) or pair[0] >= pair[1] for pair in bounds):
            return reject("invalid_channel",True)
        lower,upper = channel.bounds(n+(1 if event else 0))
        if event is None:
            if analysis.get("market_volatility_atr",float("inf"))/atr > rules["regression_impulse_max_compression_ratio"]:
                return reject("no_compression")
            if (upper-lower)/atr > rules["regression_impulse_max_width_atr"]:
                return reject("channel_too_wide")
            previous_lower,previous_upper = channel.bounds(n-1)
            if channel.previous_bid_close > previous_upper or channel.previous_ask_close < previous_lower:
                return reject("previous_close_outside")
            if bid.high_price > upper and ask.low_price < lower:
                return reject("ambiguous_breakout")
            up,down = (bid.close_price-upper)/atr,(lower-ask.close_price)/atr
            if up > 0 and up >= rules["regression_impulse_min_close_atr"]:
                direction,extension = 1,up
            elif down > 0 and down >= rules["regression_impulse_min_close_atr"]:
                direction,extension = -1,down
            else:
                return reject("no_breakout")
            if extension > rules["regression_impulse_max_breakout_distance_atr"]:
                return reject("breakout_too_far")
            quote = bid if direction > 0 else ask
            previous = channel.previous_bid_close if direction > 0 else channel.previous_ask_close
            body,location = directional_candle_quality(quote,direction)
            slope = analysis.get("shared_slope",0.)
            if direction*(quote.close_price-previous) <= 0 or direction*slope < 0:
                return reject("opposing_impulse")
            if body < rules["regression_impulse_min_body_ratio"] or location < rules["regression_impulse_min_close_location"]:
                return reject("weak_breakout")
            self.events[symbol] = ImpulseEvent(channel,bid.start_time,direction,quote.close_price)
            return reject("awaiting_hold")
        # Exactly one later candle is allowed; no delayed rolling trigger.
        self.clear(symbol)
        direction = event.direction
        buying = direction > 0
        boundary = upper if buying else lower
        quote = bid if buying else ask
        body,location = directional_candle_quality(quote,direction)
        outside = (bid.low_price-boundary if buying else boundary-ask.high_price)/atr
        trigger = event.close+direction*rules["regression_impulse_confirmation_buffer_atr"]*atr
        if (outside < rules["regression_impulse_min_hold_low_atr"]
                or direction*(quote.close_price-trigger) <= 0
                or body < rules["regression_impulse_min_hold_body_ratio"]
                or location < rules["regression_impulse_min_hold_close_location"]):
            return reject("hold_not_confirmed")
        maximum = boundary+direction*rules["regression_impulse_max_entry_distance_atr"]*atr
        executable = ask.close_price if buying else bid.close_price
        if direction*(executable-maximum) > 0:
            return reject("entry_too_far")
        stop = boundary-direction*rules["regression_impulse_stop_buffer_atr"]*atr
        context = dict(entry_mode=self.entry_name,channel_fitted_through=channel.fitted_through.isoformat(),
            breakout_started_at=event.last_bar.isoformat(),confirmation_bar=bid.start_time.isoformat(),
            atr=atr,channel_lower=lower,channel_upper=upper,confirmation_body_ratio=body,
            confirmation_close_location=location,stop_loss=stop,
            minimum_entry_price=trigger if buying else maximum,maximum_entry_price=maximum if buying else trigger)
        analysis.update(direction="buy" if buying else "sell",reasons=[])
        return Signal(symbol,timestamp,SignalSide.BUY if buying else SignalSide.SELL,stop_loss_price=stop,
            minimum_entry_price=context["minimum_entry_price"],maximum_entry_price=context["maximum_entry_price"],entry_context=context),analysis
