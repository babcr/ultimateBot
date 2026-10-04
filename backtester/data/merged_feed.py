import heapq
from typing import Iterator
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.data.tick_source import TickSource

class MergedTickFeed:
   """Merges multiple tick sources into a single chronologically ordered feed."""

   def __init__(self, sources: list[TickSource]):
      self.sources = sources

   def __iter__(self) -> Iterator[Tick]:
      iterators = [iter(source) for source in self.sources]
      heap = []
      for index, iterator in enumerate(iterators):
         try:
            tick = next(iterator)
            heapq.heappush(heap, (tick.timestamp, index, tick))
         except StopIteration:
            continue

      while heap:
         _, source_index, tick = heapq.heappop(heap)
         yield tick

         try:
            next_tick = next(iterators[source_index])
            heapq.heappush(heap, (next_tick.timestamp, source_index, next_tick))
         except StopIteration:
            continue