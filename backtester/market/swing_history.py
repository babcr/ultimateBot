from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ultimateBot.backtester.market.candle_builder import Candle


@dataclass(frozen=True)
class ReversalSwing:
    kind: str
    price: float
    confirmation_index: int
    pair_range: float


class ReversalSwingHistory:
    """Retain raw two-candle reversals; normalize only after selecting a window."""

    def __init__(self, candle_capacity: int):
        self.candle_capacity = max(1, int(candle_capacity))
        self._candle_count = 0
        self._previous: Candle | None = None
        self._candidates: deque[ReversalSwing] = deque()

    def append(self, candle: Candle) -> None:
        index = self._candle_count
        previous = self._previous
        if previous is not None:
            kind = None
            if previous.close_price > previous.open_price and candle.close_price < candle.open_price:
                kind = "high"
            elif previous.close_price < previous.open_price and candle.close_price > candle.open_price:
                kind = "low"
            if kind is not None:
                pair_high = max(float(previous.high_price), float(candle.high_price))
                pair_low = min(float(previous.low_price), float(candle.low_price))
                self._candidates.append(ReversalSwing(
                    kind, pair_high if kind == "high" else pair_low, index, pair_high - pair_low,
                ))
        self._previous = candle
        self._candle_count += 1
        window_start = self._candle_count - self.candle_capacity
        # A reversal needs its preceding candle as well as its confirmation.
        while self._candidates and self._candidates[0].confirmation_index <= window_start:
            self._candidates.popleft()

    def project(self, atr: float, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and limit < 0:
            raise ValueError("Swing candle limit must be non-negative")
        window_size = min(self._candle_count, self.candle_capacity,
                          self.candle_capacity if limit is None else limit)
        window_start = self._candle_count - window_size
        denominator = max(atr, 1e-9)
        swings = []
        for candidate in reversed(self._candidates):
            if candidate.confirmation_index <= window_start:
                break
            swings.append({
                "kind": candidate.kind,
                "price": candidate.price,
                "index": candidate.confirmation_index - window_start,
                "strength": float(max(candidate.pair_range / denominator, 0.0)),
            })
        swings.reverse()
        return swings
