from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from ultimateBot.backtester.data.tick import Tick

@dataclass
class TickBatch:
    symbol: str
    ticks: list[Tick]

class TickBatcher:
    def __init__(self, batch_size: int, analysis_interval: timedelta | None = None):
        self.buffers: dict[str, list[Tick]] = {}
        self._analysis_windows: dict[str, datetime] = {}
        self.batch_size = max(1, int(batch_size))
        self.analysis_interval = analysis_interval if analysis_interval and analysis_interval.total_seconds() > 0.0 else None

    def push(self, tick: Tick) -> Optional[TickBatch]:
        if self.analysis_interval is not None:
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
        analysis_window = self._analysis_window_start(tick.timestamp)
        previous_window = self._analysis_windows.get(tick.symbol)

        if previous_window is None:
            self._analysis_windows[tick.symbol] = analysis_window
            buffer.append(tick)
            return None

        if analysis_window == previous_window:
            buffer.append(tick)
            return None

        batch = TickBatch(symbol=tick.symbol, ticks=buffer.copy())
        buffer.clear()
        buffer.append(tick)
        self._analysis_windows[tick.symbol] = analysis_window
        return batch

    def _analysis_window_start(self, timestamp: datetime) -> datetime:
        if self.analysis_interval is None:
            return timestamp

        interval_seconds = max(1, int(self.analysis_interval.total_seconds()))
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