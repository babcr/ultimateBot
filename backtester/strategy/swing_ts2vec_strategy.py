"""Use frozen swing research models in the ordinary tick/portfolio engine."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
import json
from math import isfinite
from time import monotonic

from ultimateBot.backtester.research.swing_opportunities import OPPORTUNITY_SCHEMA, OpportunityConfig
from ultimateBot.backtester.research.swing_resolver import SwingResolver, is_binary_target, resolver_directory
from ultimateBot.backtester.research.swing_series import FEATURE_SCHEMAS, VOLUME_FEATURE_SCHEMA, SWING_SCHEMA, SwingSeries, SwingWindow
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.backtester.strategy.swing_threshold import DailyTradeTarget, validate_min_threshold


def model_configuration(path):
    directory = resolver_directory(path)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("feature_schema") not in FEATURE_SCHEMAS or metadata.get("swing_schema") != SWING_SCHEMA
            or metadata.get("opportunity_schema") != OPPORTUNITY_SCHEMA):
        raise ValueError("Use a frozen model trained with the current unrestricted swing collector")
    config = OpportunityConfig(**metadata["opportunity_config"])
    if metadata["interval_seconds"] != config.interval_seconds:
        raise ValueError("Inconsistent model candle interval")
    mode, value = metadata["window_key"].rsplit("_", 1)
    return directory, metadata, config, SwingWindow(mode, int(value))


def resolve_swing_threshold(metadata, override=None, min_threshold=None):
    threshold = float(metadata["threshold"] if override is None else override)
    if not isfinite(threshold):
        raise ValueError("Swing TS2Vec threshold must be finite")
    if is_binary_target(metadata["target"]) and not 0 <= threshold <= 1:
        raise ValueError("A probability threshold must be between 0 and 1")
    minimum = validate_min_threshold(min_threshold, probability=is_binary_target(metadata['target']))
    if minimum is not None:
        threshold = max(threshold, minimum)
    return threshold


class SwingTS2VecStrategy(Strategy):
    """Observe every closed candle, including while portfolio exposure is busy."""

    minimum_candles = 2
    fill_on_signal_release = True
    strategy_name = 'swing-ts2vec'

    def _load_resolver(self, threads):
        return SwingResolver(self.directory, threads=threads)

    def _observe_closed_candle(self, symbol, bid, ask):
        pass

    def _predict_swing(self, points, *, as_of, symbol, stop_buffer_pips=None):
        extra = {} if stop_buffer_pips is None else dict(stop_buffer_pips=stop_buffer_pips)
        return self.resolver.predict(points, as_of=as_of, symbol=symbol, **extra)

    def __init__(self, directory, *, threads=2, threshold=None, min_threshold=None, trailing_start_r=None, take_profit_r=None,
                 stop_buffer_pips=None, decision_log="off", daily_target=None):
        if decision_log not in ("off", "accepted", "all"):
            raise ValueError("Swing decision log must be off, accepted or all")
        self.decision_log = decision_log
        self.directory, self.metadata, self.opportunity_config, self.window = model_configuration(directory)
        self.trained_trailing_start_r = self.opportunity_config.trailing_start_r
        self.trained_stop_buffer_pips = self.opportunity_config.stop_buffer_pips
        if stop_buffer_pips is not None:
            self.opportunity_config = replace(self.opportunity_config, stop_buffer_pips=float(stop_buffer_pips))
        self.take_profit_r = float(take_profit_r) if take_profit_r is not None else None
        if self.take_profit_r is not None and (not isfinite(self.take_profit_r) or self.take_profit_r <= 0):
            raise ValueError('Fixed take profit must be positive and finite in R')
        self.exit_mode = 'fixed-take-profit' if self.take_profit_r is not None else 'trailing'
        if trailing_start_r is not None:
            self.opportunity_config = replace(self.opportunity_config, trailing_start_r=float(trailing_start_r))
        self.threshold = resolve_swing_threshold(self.metadata, threshold, min_threshold)
        self.min_threshold = float(min_threshold) if min_threshold is not None else None
        self.trained_threshold = float(self.metadata["threshold"])
        self.threshold_source = "model" if threshold is None else "override"
        if self.threshold > float(self.metadata['threshold'] if threshold is None else threshold):
            self.threshold_source = 'minimum'
        self.initial_threshold = self.threshold
        self.initial_threshold_source = self.threshold_source
        self.daily_target = (DailyTradeTarget(self.threshold, daily_target, probability=is_binary_target(self.metadata["target"]),
                                            min_threshold=self.min_threshold)
                             if daily_target is not None else None)
        if self.daily_target is not None:
            self.threshold_source = "adaptive"
        self.analysis_interval = timedelta(seconds=self.opportunity_config.interval_seconds)
        self.resolver = self._load_resolver(threads)
        self.available_after = datetime.fromisoformat(self.metadata["available_after"])
        self.series = {}
        self.last_start = {}
        self.counts = Counter()
        self._progress_date = None
        self._progress_phase = None
        self._progress_at = None

    def report_progress(self, tick):
        clock = monotonic()
        day = tick.timestamp.date()
        phase = "warmup_before_model" if tick.timestamp < self.available_after else "trading_period"
        if (day == self._progress_date and phase == self._progress_phase
                and self._progress_at is not None and clock-self._progress_at < 30):
            return
        self._progress_date, self._progress_phase, self._progress_at = day, phase, clock
        print(f"[{self.strategy_name}-progress] simulated_at={tick.timestamp.isoformat()} phase={phase} "
              f"model_available_after={self.available_after.isoformat()} "
              f"swings={self.counts['raw_opportunities']} scored={self.counts['scored']} "
              f"accepted={self.counts['accepted']} risk_rejected={self.counts['rejected_by_risk']} "
              f"context_not_ready={self.counts['context_not_ready_or_ambiguous']} threshold={self.threshold} "
              f"min_threshold={self.min_threshold}", flush=True)

    def on_market_data(self, batch, context):
        symbol = batch.symbol
        scope = self.metadata.get("scope", {})
        if scope and symbol not in scope["symbols"]:
            raise ValueError(f"{symbol} is outside the frozen model's symbol scope")
        tick = context.market_state.last_ticks.get(symbol)
        if tick is None:
            return []
        if self.daily_target is not None and tick.timestamp >= self.available_after:
            closed = self.daily_target.advance(tick.timestamp)
            self.threshold = self.daily_target.threshold
            if closed is not None:
                print(f"[{self.strategy_name}-threshold-period] " + json.dumps(closed), flush=True)
        self.report_progress(tick)
        bids, asks = [context.market_state.get_candles(
            symbol, include_active=False, price_source=source, candle_interval=self.analysis_interval)
            for source in ("bid", "ask")]
        if len(bids) != len(asks) or any(b.start_time != a.start_time for b, a in zip(bids, asks)):
            raise ValueError("Bid/ask closed candles must align")
        series = self.series.setdefault(symbol, SwingSeries(self.analysis_interval, (self.window,),
                                       include_volume=self.metadata['feature_schema'] == VOLUME_FEATURE_SCHEMA))
        signals = []
        for bid, ask in zip(bids, asks):
            if symbol in self.last_start and bid.start_time <= self.last_start[symbol]:
                continue
            if bid.start_time + self.analysis_interval > tick.timestamp:
                # Engine finalization can expose the last unfinished candle.
                continue
            series.append(bid, ask)
            self._observe_closed_candle(symbol, bid, ask)
            self.last_start[symbol] = bid.start_time
            self.counts["closed_candles"] += 1
            for swing in series.last_candidates:
                self.counts["raw_opportunities"] += 1
                if swing.confirmed_at < self.available_after:
                    self.counts["before_model_availability"] += 1
                    continue
                if tick.timestamp >= swing.confirmed_at + self.analysis_interval * self.opportunity_config.order_expiration_bars:
                    self.counts["already_expired_at_release"] += 1
                    continue
                points = series.context(self.window, swing.confirmed_at)
                if points is None:
                    self.counts["context_not_ready_or_ambiguous"] += 1
                    continue
                accepted_swing = False
                # Prefer the smallest admitted structural-stop margin. Each
                # swing releases at most one signal, including if risk vetoes it.
                for variant in range(self.opportunity_config.stop_variants):
                    buffer = self.opportunity_config.buffer_for(variant)
                    decision = self._predict_swing(points, as_of=swing.confirmed_at, symbol=symbol,
                        stop_buffer_pips=buffer if self.opportunity_config.stop_variants > 1 else None)
                    decision = {**decision, "threshold": self.threshold, "trained_threshold": self.trained_threshold,
                                "exit_mode": self.exit_mode, "take_profit_r": self.take_profit_r,
                                "stop_buffer_pips": buffer, "stop_variant_index": variant,
                                "trained_stop_buffer_pips": self.trained_stop_buffer_pips,
                                "trailing_start_r": self.opportunity_config.trailing_start_r if self.take_profit_r is None else None,
                                "trained_trailing_start_r": self.trained_trailing_start_r,
                                "min_threshold": self.min_threshold,
                                "threshold_source": self.threshold_source,
                                "accepted": decision["prediction"] >= self.threshold}
                    self.counts["scored"] += 1
                    self.counts["accepted" if decision["accepted"] else "rejected_by_model"] += 1
                    if self.decision_log == "all" or self.decision_log == "accepted" and decision["accepted"]:
                        print(f"[{self.strategy_name}-decision] " + json.dumps({
                            "symbol": symbol, "confirmed_at": swing.confirmed_at.isoformat(),
                            "decision_at": tick.timestamp.isoformat(), "window": self.window.key, **decision}), flush=True)
                    if not decision["accepted"]:
                        continue
                    signals.append(Signal(
                        symbol=symbol, timestamp=swing.confirmed_at,
                        side=SignalSide.BUY if swing.side == 1 else SignalSide.SELL, confidence=1.,
                        stop_loss_price=self.opportunity_config.stop_for(swing, symbol, variant),
                        entry_context={"entry_mode": self.strategy_name, "window_key": self.window.key,
                                       "resolver_id": self.metadata["resolver_id"],
                                       "embedding_model_id": self.metadata.get("embedding_model_id"),
                                       "prediction": decision["prediction"], "threshold": decision["threshold"],
                                       "min_threshold": self.min_threshold,
                                       "exit_mode": self.exit_mode, "take_profit_r": self.take_profit_r,
                                       "stop_buffer_pips": buffer, "stop_variant_index": variant,
                                       "trained_stop_buffer_pips": self.trained_stop_buffer_pips,
                                       "trailing_start_r": self.opportunity_config.trailing_start_r if self.take_profit_r is None else None,
                                       "trained_trailing_start_r": self.trained_trailing_start_r,
                                       "trained_threshold": self.trained_threshold, "threshold_source": self.threshold_source,
                                       "swing_price": swing.price, "confirmed_at": swing.confirmed_at.isoformat()}))
                    accepted_swing = True
                    break
                if self.daily_target is not None:
                    self.daily_target.record(accepted_swing)
                    self.threshold = self.daily_target.threshold
        return signals

    def prepare_filled_position(self, position):
        """Compute exit levels from actual fill risk, including delayed fills."""
        if self.opportunity_config.max_holding_seconds:
            position.close_after = position.opened_at+timedelta(seconds=self.opportunity_config.max_holding_seconds)
        distance = abs(position.entry_price - position.stop_loss)
        if distance == 0:
            position.trailing_stop_enabled = False
            position.trailing_stop_activation_price = None
            position.take_profit = None
            return
        direction = 1 if position.side.name == "BUY" else -1
        if self.take_profit_r is not None:
            position.trailing_stop_enabled = False
            position.trailing_stop_activation_price = None
            position.trailing_stop_state = None
            position.take_profit = position.entry_price + direction * distance * self.take_profit_r
            return
        position.trailing_stop_activation_price = (position.entry_price + direction * distance
                                                  * self.opportunity_config.trailing_start_r)

    def set_analysis_interval(self, interval, symbol=None):
        if interval != self.analysis_interval:
            raise ValueError("The frozen swing model requires its original candle interval")
