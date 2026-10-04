from dataclasses import dataclass
from typing import TYPE_CHECKING



if TYPE_CHECKING:
    from ultimateBot.backtester.portfolio import Portfolio
    from ultimateBot.backtester.market import MarketState

@dataclass
class StrategyContext:
   """Context information for strategy execution."""
   portfolio: "Portfolio"
   market_state: "MarketState"