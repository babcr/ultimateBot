from __future__ import annotations

from dataclasses import dataclass
from math import floor, isclose, isfinite, ulp
from typing import TYPE_CHECKING

from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide

if TYPE_CHECKING:
    from ultimateBot.backtester.portfolio.position import Position


@dataclass
class TrailingStopState:
    """Immutable price references and the trailing progress of one filled trade."""

    activation_price: float
    initial_stop_loss: float
    initial_stop_distance: float
    breakeven_price: float
    active: bool = False
    steps: int = 0


@dataclass(frozen=True)
class TrailingStopUpdate:
    previous_stop_loss: float
    stop_loss: float
    steps: int
    activated: bool


class TrailingStopManager:
    """Start at a separate RR threshold, then let the trade run on risk steps."""

    def prepare_order(self, order: OrderRequest) -> None:
        activation_price = order.trailing_stop_activation_price
        if (order.stop_loss is None or activation_price is None
                or not isfinite(order.stop_loss) or order.stop_loss <= 0.0
                or not isfinite(activation_price) or activation_price <= 0.0):
            raise ValueError("Trailing stop requires a positive initial SL and a separate activation price")
        self._validate_target(order.side, activation_price, order.take_profit)
        order.trailing_stop_activation_price = activation_price
        order.take_profit = None

    def prepare_position(self, position: Position) -> TrailingStopState:
        if position.trailing_stop_state is not None:
            return position.trailing_stop_state
        activation_price = position.trailing_stop_activation_price
        if activation_price is None and position.take_profit is not None:
            # Directly supplied positions have no planned risk evaluation.
            activation_price = position.entry_price + (position.take_profit - position.entry_price) / 2.0
        if (position.stop_loss is None or activation_price is None
                or not isfinite(position.stop_loss) or position.stop_loss <= 0.0
                or not isfinite(activation_price) or activation_price <= 0.0
                or not isfinite(position.entry_price) or position.entry_price <= 0.0):
            raise ValueError("Trailing stop requires a positive fill price, initial SL and activation price")
        distance = abs(position.entry_price - position.stop_loss)
        if distance <= 0.0:
            raise ValueError("Trailing stop requires a nonzero initial stop distance")
        self._validate_target(position.side, activation_price, position.take_profit)
        state = TrailingStopState(
            activation_price=activation_price,
            initial_stop_loss=position.stop_loss,
            initial_stop_distance=distance,
            # PnL uses the actual fill against the executable bid/ask; there are no extra fees.
            breakeven_price=position.entry_price,
        )
        position.trailing_stop_state = state
        position.trailing_stop_activation_price = activation_price
        position.take_profit = None
        return state

    @staticmethod
    def _validate_target(side: OrderSide, activation_price: float, take_profit: float | None) -> None:
        if take_profit is None:
            return
        direction = 1.0 if side == OrderSide.BUY else -1.0
        if not isfinite(take_profit) or take_profit <= 0.0 or direction * (take_profit - activation_price) <= 0.0:
            raise ValueError("Trailing activation must be reached before the take-profit target")

    def update(self, position: Position, tick: Tick) -> TrailingStopUpdate | None:
        if position.symbol != tick.symbol:
            return None
        state = self.prepare_position(position)
        direction = 1.0 if position.side == OrderSide.BUY else -1.0
        exit_price = tick.bid if position.side == OrderSide.BUY else tick.ask
        if not isfinite(exit_price) or exit_price <= 0.0:
            return None
        progress = direction * (exit_price - state.activation_price)
        if not state.active and progress < 0.0:
            return None

        step_ratio = max(0.0, progress / state.initial_stop_distance)
        nearest_step = round(step_ratio)
        # Decimal tick prices can land a few floating-point ULPs below an exact step.
        tolerance = 4.0 * max(ulp(exit_price), ulp(state.activation_price), ulp(state.initial_stop_distance))
        if isclose(max(0.0, progress), nearest_step * state.initial_stop_distance,
                   rel_tol=0.0, abs_tol=tolerance):
            step_ratio = float(nearest_step)
        steps = max(state.steps, floor(step_ratio))
        activated = not state.active
        if not activated and steps == state.steps:
            return None

        candidate_stop = state.breakeven_price + direction * steps * state.initial_stop_distance
        previous_stop = position.stop_loss
        stop_loss = (max(previous_stop, candidate_stop) if position.side == OrderSide.BUY
                     else min(previous_stop, candidate_stop))
        state.active = True
        state.steps = steps
        position.stop_loss = stop_loss
        return TrailingStopUpdate(previous_stop, stop_loss, steps, activated)
