from datetime import timedelta

from ultimateBot.backtester.orders.order import OrderRequest, OrderType
from ultimateBot.backtester.risk.risk_manager import RiskEvaluation
from ultimateBot.backtester.strategy.signal import Signal


class OrderBuilder:
    def __init__(self, execution_delay: timedelta = timedelta(milliseconds=50)):
        self.execution_delay = execution_delay
        self._next_order_id = 1

    def build(self, signal: Signal, evaluation: RiskEvaluation) -> OrderRequest:
        limit_price = None
        stop_price = None
        if evaluation.order_type == OrderType.LIMIT:
            limit_price = evaluation.entry_price
        elif evaluation.order_type == OrderType.STOP:
            stop_price = evaluation.entry_price

        order = OrderRequest(
            order_id=self._next_order_id,
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            created_at=signal.timestamp,
            eligible_at=signal.timestamp + self.execution_delay,
            side=evaluation.side,
            quantity=evaluation.quantity,
            order_type=evaluation.order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss=evaluation.stop_loss,
            take_profit=evaluation.take_profit,
        )
        self._next_order_id += 1
        return order
