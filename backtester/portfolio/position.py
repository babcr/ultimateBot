from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from ultimateBot.backtester.orders.order import OrderSide

if TYPE_CHECKING:
   from ultimateBot.backtester.monitoring.trailing_stop import TrailingStopState
   from ultimateBot.backtester.optimization.bandit_core import PositionLearningRecord

@dataclass
class Position:
   trade_id: int

   symbol: str
   side: OrderSide

   quantity: float

   opened_at: datetime
   entry_price: float

   stop_loss: Optional[float] = None
   take_profit: Optional[float] = None

   max_favorable_pnl: float = 0.0
   max_adverse_pnl: float = 0.0
   trailing_stop_activation_price: Optional[float] = None
   trailing_stop_state: Optional["TrailingStopState"] = None
   trailing_stop_enabled: Optional[bool] = None
   strategy_configuration_id: Optional[int] = None
   strategy_lookback_bars: Optional[int] = None
   strategy_timeframe_minutes: Optional[int] = None
   strategy_use_global_timeframe_validation: Optional[bool] = None
   learning_record: Optional["PositionLearningRecord"] = None
   profit_multiplier: float = 1.0
   account_currency: str = "engine_price_quantity"
