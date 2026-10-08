from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from ultimateBot.backtester.data.tick import Tick


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

	def update(self, tick: Tick) -> None:
		bucket_start = self._bucket_start(tick.timestamp)
		price = self._price_from_tick(tick)

		candle = self._active.get(tick.symbol)
		if candle is None or candle.start_time != bucket_start:
			if candle is not None:
				self._closed[tick.symbol].append(candle)

			self._active[tick.symbol] = Candle(
				start_time=bucket_start,
				open_price=price,
				high_price=price,
				low_price=price,
				close_price=price,
				tick_count=1,
			)
			return

		candle.high_price = max(candle.high_price, price)
		candle.low_price = min(candle.low_price, price)
		candle.close_price = price
		candle.tick_count += 1

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

	def get_candles(self, symbol: str, include_active: bool = True) -> list[Candle]:
		candles = list(self._closed.get(symbol, ()))
		if include_active and symbol in self._active:
			candles.append(self._active[symbol])
		return candles

	def finalize_active_candles(self) -> None:
		for symbol, candle in list(self._active.items()):
			self._closed[symbol].append(candle)
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
