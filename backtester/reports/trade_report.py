from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite, isclose
from typing import TYPE_CHECKING

from ultimateBot.backtester.monitoring.close_request import CloseReason, CloseRequest
from ultimateBot.backtester.data.tick import Tick
from ultimateBot.backtester.monitoring.trade_monitor import TradeMonitor
from ultimateBot.backtester.monitoring.trailing_stop import TrailingStopManager
from ultimateBot.backtester.orders.order import OrderRequest, OrderSide
from ultimateBot.backtester.portfolio.portfolio import Portfolio
from ultimateBot.backtester.portfolio.position import Position

if TYPE_CHECKING:
   from ultimateBot.backtester.optimization.bandit_core import PositionLearningRecord

@dataclass(frozen=True)
class TradeReport:
   trade_id: int
   symbol: str

   side: OrderSide

   opened_at: datetime
   closed_at: datetime

   entry_price: float
   exit_price: float

   quantity: float

   pnl: float
   max_favorable_pnl: float
   max_adverse_pnl: float

   close_reason: CloseReason
   trailing_stop_enabled: bool = False
   strategy_configuration_id: int | None = None
   strategy_lookback_bars: int | None = None
   strategy_timeframe_minutes: int | None = None
   strategy_use_global_timeframe_validation: bool | None = None
   learning_record: PositionLearningRecord | None = None
   is_complete: bool = True
   event_id: str | None = None
   deal_ids: tuple[str, ...] = ()
   pnl_gross: float | None = None


class TradeCloser:
   def __init__(
      self,
      trailing_stop_enabled: bool = False,
      trailing_stop_manager: TrailingStopManager | None = None,
   ):
      self.trailing_stop_enabled = bool(trailing_stop_enabled)
      self.trailing_stop_manager = (
         trailing_stop_manager if trailing_stop_manager is not None else TrailingStopManager()
      ) if self.trailing_stop_enabled else None

   def prepare_order(self, order: OrderRequest) -> None:
      if order.trailing_stop_enabled is None:
         order.trailing_stop_enabled = self.trailing_stop_enabled
      if order.trailing_stop_enabled:
         self._ensure_trailing_manager()
         self.trailing_stop_manager.prepare_order(order)
      else:
         order.trailing_stop_activation_price = None

   def prepare_position(self, position: Position) -> None:
      if position.trailing_stop_enabled is None:
         position.trailing_stop_enabled = (
            position.trailing_stop_state is not None
            or position.trailing_stop_activation_price is not None
            or self.trailing_stop_enabled
         )
      if position.trailing_stop_enabled:
         self._ensure_trailing_manager()
         self.trailing_stop_manager.prepare_position(position)

   def _ensure_trailing_manager(self) -> None:
      if self.trailing_stop_manager is None:
         self.trailing_stop_manager = TrailingStopManager()

   def set_trailing_stop_enabled(self, enabled: bool) -> None:
      # The flag applies to new orders; existing orders and positions retain their mode.
      self.trailing_stop_enabled = bool(enabled)

   def check(self, position: Position, tick: Tick, monitor: TradeMonitor) -> CloseRequest | None:
      if position.symbol != tick.symbol:
         return None
      self.prepare_position(position)
      # Honor the previously posted SL before allowing this tick to advance it.
      request = monitor.check(position, tick)
      if request is not None or not position.trailing_stop_enabled:
         return request
      update = self.trailing_stop_manager.update(position, tick)
      if update is None:
         return None
      state = position.trailing_stop_state
      print(
         f"[trailing-stop] trade_id={position.trade_id} symbol={position.symbol} "
         f"timestamp={tick.timestamp.isoformat()} side={position.side.name.lower()} "
         f"event={'activated' if update.activated else 'advanced'} steps={update.steps} "
         f"activation_price={state.activation_price} breakeven_price={state.breakeven_price} "
         f"initial_stop_distance={state.initial_stop_distance} "
         f"previous_stop_loss={update.previous_stop_loss} stop_loss={update.stop_loss}",
         flush=True,
      )
      # A fill beyond the activation level can put breakeven above/below the current quote.
      return monitor.check(position, tick)

   def close(
      self,
      portfolio: Portfolio,
      request: CloseRequest
   ) -> TradeReport:
      position = portfolio.positions[request.trade_id]
      closed_quantity = position.quantity if request.quantity is None else request.quantity
      if not isfinite(closed_quantity) or not 0 < closed_quantity <= position.quantity:
         raise ValueError("Close quantity must be finite, positive and within the open position")
      complete = isclose(closed_quantity, position.quantity, rel_tol=1e-12, abs_tol=0.0)
      if complete:
         portfolio.positions.pop(request.trade_id)
      else:
         position.quantity -= closed_quantity

      if position.side == OrderSide.BUY:
         pnl = (request.price - position.entry_price) * closed_quantity
      else:
         pnl = (position.entry_price - request.price) * closed_quantity
      pnl *= position.profit_multiplier

      portfolio.balance += pnl

      return TradeReport(
         trade_id=request.trade_id,
         symbol=position.symbol,
         side=position.side,
         opened_at=position.opened_at,
         closed_at=request.timestamp,
         entry_price=position.entry_price,
         exit_price=request.price,
         quantity=closed_quantity,
         pnl=pnl,
         max_favorable_pnl=position.max_favorable_pnl,
         max_adverse_pnl=position.max_adverse_pnl,
         close_reason=request.reason,
         trailing_stop_enabled=bool(position.trailing_stop_enabled),
         strategy_configuration_id=position.strategy_configuration_id,
         strategy_lookback_bars=position.strategy_lookback_bars,
         strategy_timeframe_minutes=position.strategy_timeframe_minutes,
         strategy_use_global_timeframe_validation=position.strategy_use_global_timeframe_validation,
         learning_record=position.learning_record,
         is_complete=complete,
         event_id=request.event_id,
         deal_ids=(request.event_id,) if request.event_id else (),
         pnl_gross=pnl,
      )
