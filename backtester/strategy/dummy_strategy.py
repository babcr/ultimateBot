from __future__ import annotations

import hashlib
import random

from ultimateBot.backtester.data.batcher import TickBatch
from ultimateBot.backtester.strategy.signal import Signal, SignalSide
from ultimateBot.backtester.strategy.strategy import Strategy
from ultimateBot.backtester.strategy.strategy_context import StrategyContext


class DummyStrategy(Strategy):
   """Simple deterministic strategy used to smoke-test the backtester."""

   def __init__(self, seed: int):
      self.seed = seed
      self._last_signal_week: dict[str, tuple[int, int]] = {}

   def _random_side(self, symbol: str, year: int, week: int) -> SignalSide:
      value = f"{self.seed}:{symbol}:{year}:{week}"
      digest = hashlib.sha256(value.encode("utf-8")).digest()
      week_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
      rng = random.Random(week_seed)
      return rng.choice([SignalSide.BUY, SignalSide.SELL, SignalSide.HOLD])

   def on_market_data(self, batch: TickBatch, context: StrategyContext) -> list[Signal]:
      if not batch.ticks:
         return []

      last_tick = batch.ticks[-1]
      iso_calendar = last_tick.timestamp.isocalendar()
      week_key = (iso_calendar[0], iso_calendar[1])

      if self._last_signal_week.get(batch.symbol) == week_key:
         return []

      self._last_signal_week[batch.symbol] = week_key
      side = self._random_side(
         symbol=batch.symbol,
         year=iso_calendar[0],
         week=iso_calendar[1],
      )

      return [
         Signal(
            symbol=batch.symbol,
            timestamp=last_tick.timestamp,
            side=side,
            confidence=1.0,
         )
      ]