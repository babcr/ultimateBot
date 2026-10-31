from __future__ import annotations

from datetime import timedelta
from math import isfinite

from ultimateBot.backtester.config.backtest_config import DEFAULT_CANDLE_INTERVAL
from ultimateBot.backtester.strategy.swing_strategy import SwingStrategy
from ultimateBot.backtester.strategy.temporal_patterns import (
    TemporalPatternConfig, TemporalPatternTracker, confirmed_swings, structural_swings, detect_groups,
)


class TemporalPatternStrategy(SwingStrategy):
    """Opt-in swing groups; never alters the existing regression strategy."""

    minimum_lookback_bars = 14

    def __init__(self, config: TemporalPatternConfig | None = None, lookback: int | None = None,
                 analysis_interval: timedelta = DEFAULT_CANDLE_INTERVAL):
        self.config = config or TemporalPatternConfig()
        capacity = lookback if lookback is not None else max(60, self.config.max_span_bars + 2 * self.config.max_swing_gap_bars + 4)
        if capacity < self.minimum_lookback_bars:
            raise ValueError("swing-patterns requires at least 14 history candles")
        super().__init__({"level_lookback_bars": capacity, "protective_stop_buffer_atr": self.config.stop_buffer_atr},
                         analysis_interval=analysis_interval)
        self.tracker = TemporalPatternTracker(self.config)

    def set_analysis_interval(self, interval, symbol=None):
        super().set_analysis_interval(interval, symbol)
        if symbol is None:
            self.tracker.events.clear()
            self.tracker.forming.clear()
            self.tracker.consumed.clear()
        else:
            self.tracker.clear(symbol)
            self.tracker.consumed.pop(symbol, None)

    def set_history_lookback_bars(self, bars, symbol=None):
        super().set_history_lookback_bars(bars, symbol)
        if symbol is None:
            self.tracker.events.clear()
            self.tracker.forming.clear()
            self._last_closed_candle.clear()
        else:
            self.tracker.clear(symbol)
            self._last_closed_candle.pop(symbol, None)

    def on_market_data(self, batch, context):
        busy = (context.portfolio.has_open_position(batch.symbol)
                or context.pending_orders_for_symbol > 0)
        if busy:
            self.tracker.clear(batch.symbol)
        circuits = self._get_structure_candles(batch.symbol, context)
        if circuits is None:
            return []
        bid, ask = circuits["bid"], circuits["ask"]
        at = bid[-1].start_time
        interval = self.get_analysis_interval(batch.symbol)
        now = context.market_state.last_ticks[batch.symbol].timestamp
        if now < at + interval:
            # Engine finalization can expose a partial last candle as closed.
            return []
        if now - (at + interval) > interval:
            self.tracker.clear(batch.symbol)
            self.tracker.counts["stale_candle_release"] += 1
            return []
        if self._last_closed_candle.get(batch.symbol) == at:
            return []
        self._last_closed_candle[batch.symbol] = at
        if busy:
            self.tracker.counts["position_or_order_busy"] += 1
            return []
        mid = context.market_state.get_candles(batch.symbol, include_active=False, price_source="mid",
                                               limit=len(bid), candle_interval=interval)
        if (len(mid) != len(bid) or any(b.start_time != m.start_time for b, m in zip(bid, mid))
                or any(not all(isfinite(v) and v > 0 for v in
                               (c.open_price, c.high_price, c.low_price, c.close_price))
                       or not c.low_price <= min(c.open_price, c.close_price) <= max(c.open_price, c.close_price) <= c.high_price
                       for bars in (bid, ask, mid) for c in bars)):
            self.tracker.clear(batch.symbol)
            self.tracker.counts["invalid_candles"] += 1
            return []
        if bid[-1].start_time - bid[-2].start_time != interval:
            self.tracker.clear(batch.symbol)
            self.tracker.counts["decision_history_gap"] += 1
            return []
        # True MID ATR, fitted strictly before the potential breakout candle.
        ranges = [max(c.high_price - c.low_price, abs(c.high_price - p.close_price),
                      abs(c.low_price - p.close_price)) for p, c in zip(mid[:-2], mid[1:-1])][-14:]
        atr = sum(ranges) / len(ranges)
        groups = []
        if batch.symbol not in self.tracker.events:
            swings = confirmed_swings(bid[:-1], ask[:-1], interval, self.config.max_market_gap_bars)
            swings = structural_swings(swings, atr, self.config.min_swing_leg_atr)
            groups = detect_groups(swings, at, interval, atr, self.config)
            # The first observed crossing must be this candle, not an old break rediscovered later.
            groups = [g for g in groups if not any(
                bar.start_time >= g.swings[-1].confirmed_at
                and g.direction * (bar.close_price - g.level(bar.start_time)) > self.config.breakout_buffer_atr * g.atr
                for bar in (bid[:-1] if g.direction > 0 else ask[:-1]))]
            self.tracker.counts["localized_groups"] += len(groups)
            for group in groups:
                self.tracker.counts[group.name + ":group_windows"] += 1
        signal = self.tracker.evaluate(batch.symbol, bid[-1], ask[-1], mid[-1], bid[-2], ask[-2],
                                       interval, context.market_state.last_ticks[batch.symbol].timestamp, groups)
        return [signal] if signal else []

    def diagnostics(self):
        return {"temporal_patterns": dict(self.tracker.counts)}
