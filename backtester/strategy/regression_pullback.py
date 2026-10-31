"""A touch then later reclaim of the trend-side edge of a frozen channel."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from ultimateBot.backtester.strategy.regression_breakout import directional_candle_quality
from ultimateBot.backtester.strategy.regression_reentry import FrozenChannel
from ultimateBot.backtester.strategy.signal import Signal, SignalSide


@dataclass
class PullbackEvent:
    channel: FrozenChannel
    direction: int
    started_at: datetime
    last_bar: datetime
    trigger: float
    low_bid: float
    high_ask: float
    bars: int = 0


class RegressionPullbackTracker:
    def __init__(self,entry_name="trend-pullback"):
        self.events = {}
        self.entry_name = entry_name

    def clear(self, symbol=None):
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
            analysis["reasons"] = ["pullback_"+reason]
            return None,analysis

        if channel is None:
            return reject("channel_unavailable")
        if (ask.start_time != bid.start_time
                or bid.start_time-(event.last_bar if event else channel.fitted_through) != interval):
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
        if event:
            event.bars += 1
            event.last_bar = bid.start_time
            if event.bars > rules["regression_pullback_max_bars"]:
                return reject("expired",True)
        horizon = n+rules["regression_pullback_max_bars"]
        bounds = [channel.bounds(index) for index in (int(analysis.get("regression_window_start",0)),n-1,horizon)]
        if any(not all(isfinite(value) for value in pair) or pair[0] >= pair[1] for pair in bounds):
            return reject("invalid_channel",True)
        lower,upper = channel.bounds(n+(event.bars if event else 0))
        width = upper-lower
        slopes = [analysis[key]["slope"] for key in ("anchored_line","floating_line")]
        ranging = rules["regression_pullback_regime"] == "range"
        if not ranging and slopes[0]*slopes[1] <= 0:
            return reject("not_directional",True)
        if event:
            direction = event.direction
        elif ranging:
            tolerance = rules["regression_pullback_touch_tolerance_atr"]*atr
            low_touch = bid.low_price <= lower+tolerance
            high_touch = ask.high_price >= upper-tolerance
            if low_touch == high_touch:
                return reject("ambiguous_touch" if low_touch else "no_touch")
            direction = 1 if low_touch else -1
        else:
            direction = 1 if slopes[0] > 0 else -1
        lookback = int(analysis.get("regression_lookback_bars",n))
        drift = min(abs(slope) for slope in slopes)*lookback/max(pair[1]-pair[0] for pair in bounds)
        if ranging:
            drift = max(abs(slope) for slope in slopes)*lookback/min(pair[1]-pair[0] for pair in bounds)
        analysis.update(channel_lower=lower,channel_upper=upper,pullback_drift_ratio=drift,
                        pullback_bars=event.bars if event else 0)
        if (analysis.get("parallelism",0) < rules["regression_pullback_min_parallelism"]
                or (drift > rules["regression_pullback_range_max_drift_ratio"] if ranging else drift < rules["regression_pullback_min_drift_ratio"])
                or width/atr < rules["regression_pullback_min_width_atr"]):
            return reject("weak_channel",True)
        buying = direction > 0
        boundary = lower if buying else upper
        depth = ((boundary-bid.low_price) if buying else (ask.high_price-boundary))/atr
        if ranging and depth > 0 and depth >= rules["regression_pullback_range_max_excursion_atr"]:
            return reject("reserved_for_failed_breakout",True)
        if depth > rules["regression_pullback_max_depth_atr"]:
            return reject("too_deep",True)
        if (bid.high_price >= upper if buying else ask.low_price <= lower):
            return reject("opposite_boundary_touched",True)
        if (bid.close_price <= lower or ask.close_price >= upper):
            return reject("close_outside",True)
        if event is None:
            old_lower,old_upper = channel.bounds(n-1)
            previous = channel.previous_bid_close if buying else channel.previous_ask_close
            old_boundary = old_lower if buying else old_upper
            tolerance = rules["regression_pullback_touch_tolerance_atr"]*atr
            if direction*(previous-old_boundary) <= tolerance:
                return reject("previous_close_near_edge")
            touched = (bid.low_price <= boundary+tolerance and bid.high_price >= boundary-tolerance if buying
                       else ask.high_price >= boundary-tolerance and ask.low_price <= boundary+tolerance)
            if not touched:
                return reject("no_touch")
            self.events[symbol] = PullbackEvent(channel,direction,bid.start_time,bid.start_time,
                bid.high_price if buying else ask.low_price,bid.low_price,ask.high_price)
            return reject("awaiting_reclaim")
        event.low_bid = min(event.low_bid,bid.low_price)
        event.high_ask = max(event.high_ask,ask.high_price)
        quote = bid if buying else ask
        trigger = event.trigger+direction*rules["regression_pullback_confirmation_buffer_atr"]*atr
        body,location = directional_candle_quality(quote,direction)
        if direction*(quote.close_price-trigger) <= 0:
            return reject("awaiting_reclaim")
        if body <= 0 or body < rules["regression_pullback_min_body_ratio"] or location < rules["regression_pullback_min_close_location"]:
            return reject("weak_reclaim")
        # Confirmation is consumed even if entry-space/risk/submission checks reject it.
        self.clear(symbol)
        maximum = boundary+direction*rules["regression_pullback_max_entry_fraction"]*width
        executable = ask.close_price if buying else bid.close_price
        if direction*(executable-maximum) > 0:
            return reject("entry_too_far")
        stop = event.low_bid-rules["regression_pullback_stop_buffer_atr"]*atr if buying else event.high_ask+rules["regression_pullback_stop_buffer_atr"]*atr
        distance = direction*(executable-stop)
        opposite = upper if buying else lower
        if distance <= 0 or direction*(opposite-executable)/distance < rules["regression_pullback_min_opposite_rr"]:
            return reject("insufficient_space")
        context = dict(entry_mode=self.entry_name,pullback_regime=rules["regression_pullback_regime"],channel_fitted_through=channel.fitted_through.isoformat(),
            pullback_started_at=event.started_at.isoformat(),confirmation_bar=bid.start_time.isoformat(),
            pullback_bars=event.bars,atr=atr,channel_lower=lower,channel_upper=upper,
            channel_drift_ratio=drift,confirmation_body_ratio=body,confirmation_close_location=location,
            stop_loss=stop,minimum_entry_price=trigger if buying else maximum,
            maximum_entry_price=maximum if buying else trigger)
        analysis.update(direction="buy" if buying else "sell",reasons=[])
        return Signal(symbol,timestamp,SignalSide.BUY if buying else SignalSide.SELL,stop_loss_price=stop,
            minimum_entry_price=context["minimum_entry_price"],maximum_entry_price=context["maximum_entry_price"],
            entry_context=context),analysis
