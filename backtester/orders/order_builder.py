from datetime import timedelta

from ultimateBot.backtester.config.backtest_config import OrderConfig
from ultimateBot.backtester.orders.order import OrderRequest, OrderType
from ultimateBot.backtester.risk.risk_manager import RiskEvaluation
from ultimateBot.backtester.strategy.signal import Signal


DEFAULT_ORDER_CONFIG = OrderConfig()


class OrderBuilder:
    def __init__(
        self,
        execution_delay: timedelta = DEFAULT_ORDER_CONFIG.execution_delay,
        expiration_bars: int = DEFAULT_ORDER_CONFIG.expiration_bars,
        candle_interval: timedelta = DEFAULT_ORDER_CONFIG.candle_interval,
    ):
        config = OrderConfig(execution_delay=execution_delay, expiration_bars=expiration_bars,
                             candle_interval=candle_interval)
        self.execution_delay = config.execution_delay
        self.expiration_bars = config.expiration_bars
        self.candle_interval = config.candle_interval
        self.expiration_delay = config.expiration_delay
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
            expires_at=signal.timestamp + self.expiration_delay,
            side=evaluation.side,
            quantity=evaluation.quantity,
            order_type=evaluation.order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            stop_loss=evaluation.stop_loss,
            take_profit=evaluation.take_profit,
            trailing_stop_activation_price=evaluation.trailing_stop_activation_price,
            profit_multiplier=evaluation.profit_multiplier,
            account_currency=evaluation.account_currency,
        )
        self._next_order_id += 1
        return order

    def set_expiration_interval(self, order: OrderRequest, candle_interval: timedelta) -> None:
        if candle_interval <= timedelta(0):
            raise ValueError("Order expiration candle interval must be positive")
        order.expires_at = order.created_at + candle_interval * self.expiration_bars
