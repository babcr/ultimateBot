from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from ultimateBot.backtester.data.tick import Tick

@dataclass
class TickBatch:
    symbol: str
    ticks: list[Tick]

class TickBatcher:
    def __init__(
        self,
        batch_size: int,
        analysis_interval: timedelta | None = None,
        analysis_interval_provider: Callable[[str], timedelta] | None = None,
        keep_last_tick_only: bool = False,
    ):
        self.buffers: dict[str, list[Tick]] = {}
        self._analysis_windows: dict[str, datetime] = {}
        self.batch_size = max(1, int(batch_size))
        self.analysis_interval = analysis_interval if analysis_interval and analysis_interval.total_seconds() > 0.0 else None
        self.analysis_interval_provider = analysis_interval_provider
        self.keep_last_tick_only = keep_last_tick_only

    def push(self, tick: Tick) -> Optional[TickBatch]:
        if self.analysis_interval is not None or self.analysis_interval_provider is not None:
            return self._push_for_analysis_window(tick)

        buffer = self.buffers.setdefault(tick.symbol, [])
        buffer.append(tick)
        if len(buffer) < self.batch_size:
            return None
        batch = TickBatch(symbol=tick.symbol, ticks=buffer.copy())
        buffer.clear()
        return batch

    def _push_for_analysis_window(self, tick: Tick) -> Optional[TickBatch]:
        buffer = self.buffers.setdefault(tick.symbol, [])
        analysis_window = self._analysis_window_start(tick.timestamp, self._interval_for(tick.symbol))
        previous_window = self._analysis_windows.get(tick.symbol)

        if previous_window is None:
            self._analysis_windows[tick.symbol] = analysis_window
            buffer.append(tick)
            return None

        if analysis_window == previous_window:
            if self.keep_last_tick_only:
                buffer.clear()
            buffer.append(tick)
            return None

        batch = TickBatch(symbol=tick.symbol, ticks=buffer.copy())
        buffer.clear()
        buffer.append(tick)
        self._analysis_windows[tick.symbol] = analysis_window
        return batch

    def _interval_for(self, symbol: str) -> timedelta | None:
        if self.analysis_interval_provider is None:
            return self.analysis_interval
        interval = self.analysis_interval_provider(symbol)
        if interval is None or interval.total_seconds() <= 0.0:
            return self.analysis_interval
        return interval

    def _analysis_window_start(self, timestamp: datetime, interval: timedelta | None = None) -> datetime:
        if interval is None:
            return timestamp

        interval_seconds = max(1, int(interval.total_seconds()))
        normalized = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        epoch = int(normalized.timestamp())
        bucket_epoch = epoch - (epoch % interval_seconds)
        return datetime.fromtimestamp(bucket_epoch, tz=normalized.tzinfo)

    def flush(self) -> list[TickBatch]:
        batches: list[TickBatch] = []
        for symbol, buffer in self.buffers.items():
            if not buffer:
                continue
            batches.append(TickBatch(symbol=symbol, ticks=buffer.copy()))
            buffer.clear()

        batches.sort(key=lambda batch: batch.ticks[-1].timestamp)
        return batches
