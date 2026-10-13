from ultimateBot.backtester.portfolio.position import Position
from ultimateBot.backtester.orders.order import OrderSide

class Portfolio:
   def __init__(self, initial_balance: float):
      self.initial_balance = initial_balance
      self.balance = initial_balance

      self.positions: dict[int, Position] = {}

      self._next_trade_id = 1

   @property
   def equity(self) -> float:
      return self.balance

   @property
   def capital_at_risk(self) -> float:
      total = 0.0
      for position in self.positions.values():
         if position.stop_loss is None:
            continue
         loss_distance = (position.entry_price - position.stop_loss if position.side == OrderSide.BUY
                          else position.stop_loss - position.entry_price)
         total += max(0.0, loss_distance) * position.quantity * position.profit_multiplier
      return total

   @property
   def free_margin(self) -> float:
      return max(0.0, self.equity - self.capital_at_risk)

   def get_capital_source(self, source: str) -> float:
      normalized = str(source or "equity").strip().lower()
      if normalized == "balance":
         return self.balance
      if normalized == "free_margin":
         return self.free_margin
      return self.equity

   def has_open_position(self, symbol: str) -> bool:
      return any(position.symbol == symbol for position in self.positions.values())

