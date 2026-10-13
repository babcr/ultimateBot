from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import islice
from math import isfinite
from typing import Any, Literal, Optional

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.market.swing_history import ReversalSwingHistory


PriceSource = Literal["mid", "bid", "ask"]
_VALID_PRICE_SOURCES = frozenset({"mid", "bid", "ask"})


def normalize_price_source(value: str | None, default: PriceSource = "mid") -> PriceSource:
	normalized = str(value or default).strip().lower()
	if normalized in _VALID_PRICE_SOURCES:
		return normalized
	return default


@dataclass
class Candle:
	start_time: datetime
	open_price: float
	high_price: float
	low_price: float
	close_price: float
	tick_count: int = 0
	bid_volume_sum: float = 0.0
	ask_volume_sum: float = 0.0
	bid_volume_count: int = 0
	ask_volume_count: int = 0

	@property
	def size(self) -> float:
		return max(0.0, self.high_price - self.low_price)


@dataclass(frozen=True)
class CandleStats:
	sample_size: int
	max_size: float
	average_size: float


class CandleBuilder:
	def __init__(self, candle_interval: timedelta, lookback_candles: int, price_source: PriceSource = "mid"):
		self.candle_interval = candle_interval
		self.lookback_candles = max(1, int(lookback_candles))
		self.price_source = normalize_price_source(price_source)
		self._active: dict[str, Candle] = {}
		self._closed: dict[str, deque[Candle]] = defaultdict(
			lambda: deque(maxlen=self.lookback_candles)
		)
		self._swing_histories: dict[str, ReversalSwingHistory] = {}
		self._candle_counts: dict[str, int] = {}

	def update(self, tick: Tick) -> None:
		bucket_start = self._bucket_start(tick.timestamp)
		price = self._price_from_tick(tick)

		candle = self._active.get(tick.symbol)
		if candle is None or candle.start_time != bucket_start:
			if candle is not None:
				self._append_closed_candle(tick.symbol, candle)

			self._active[tick.symbol] = Candle(
				start_time=bucket_start,
				open_price=price,
				high_price=price,
				low_price=price,
				close_price=price,
				tick_count=1,
			)
			self._add_quote_volumes(self._active[tick.symbol], tick)
			return

		candle.high_price = max(candle.high_price, price)
		candle.low_price = min(candle.low_price, price)
		candle.close_price = price
		candle.tick_count += 1
		self._add_quote_volumes(candle, tick)

	@staticmethod
	def _add_quote_volumes(candle: Candle, tick: Tick) -> None:
		for side in ("bid", "ask"):
			value = getattr(tick, f"{side}_volume", None)
			if value is not None and isfinite(value) and value >= 0:
				setattr(candle, f"{side}_volume_sum", getattr(candle, f"{side}_volume_sum") + value)
				setattr(candle, f"{side}_volume_count", getattr(candle, f"{side}_volume_count") + 1)

	def get_stats(self, symbol: str, include_active: bool = True) -> Optional[CandleStats]:
		candles = self.get_candles(symbol, include_active=include_active)

		sizes = [candle.size for candle in candles if candle.tick_count > 0 and candle.size > 0.0]
		if not sizes:
			return None

		return CandleStats(
			sample_size=len(sizes),
			max_size=max(sizes),
			average_size=sum(sizes) / len(sizes),
		)

	def get_candles(self, symbol: str, include_active: bool = True, limit: int | None = None) -> list[Candle]:
		"""Project the newest candles in time order, without copying candle objects."""
		if limit is not None and limit < 0:
			raise ValueError("Candle limit must be non-negative")
		closed = self._closed.get(symbol, ())
		active = self._active.get(symbol) if include_active else None
		if limit is None:
			candles = list(closed)
		else:
			closed_limit = max(0, limit - int(active is not None))
			# Start at the newest end so reading a small window costs O(limit).
			candles = list(islice(reversed(closed), closed_limit))
			candles.reverse()
		if active is not None and (limit is None or limit > 0):
			candles.append(active)
		return candles

	def get_reversal_swings(self, symbol: str, atr: float, limit: int | None = None) -> list[dict[str, Any]]:
		"""Project raw swings from closed candles; the consumer handles normalization."""
		if limit is not None and limit < 0:
			raise ValueError("Swing candle limit must be non-negative")
		closed = self._closed.get(symbol)
		if not closed:
			return []
		history = self._swing_histories.get(symbol)
		if history is None:
			history = ReversalSwingHistory(self.lookback_candles)
			for candle in closed:
				history.append(candle)
			self._swing_histories[symbol] = history
		return history.project(atr, limit=limit)

	def _append_closed_candle(self, symbol: str, candle: Candle) -> None:
		self._closed[symbol].append(candle)
		self._candle_counts[symbol] = self._candle_counts.get(symbol, 0) + 1
		history = self._swing_histories.get(symbol)
		if history is not None:
			history.append(candle)

	def finalize_active_candles(self) -> None:
		for symbol, candle in list(self._active.items()):
			self._append_closed_candle(symbol, candle)
			del self._active[symbol]

	def _bucket_start(self, timestamp: datetime) -> datetime:
		interval_seconds = max(1, int(self.candle_interval.total_seconds()))
		normalized = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
		epoch = int(normalized.timestamp())
		bucket_epoch = epoch - (epoch % interval_seconds)
		return datetime.fromtimestamp(bucket_epoch, tz=normalized.tzinfo)

	def _price_from_tick(self, tick: Tick) -> float:
		if self.price_source == "bid":
			return float(tick.bid)
		if self.price_source == "ask":
			return float(tick.ask)
		return 0.5 * (tick.bid + tick.ask)
