"""
Mock Prosperity datamodel — matches the real competition API exactly.

In the real competition this module is provided by the platform.
We replicate it here so our trader can be tested locally.

Key types
---------
Order(symbol, price, quantity)
    positive quantity = buy order
    negative quantity = sell order

OrderDepth
    buy_orders  : dict[price, volume]  (bids, positive volumes)
    sell_orders : dict[price, volume]  (asks, negative volumes)

TradingState
    timestamp       : int
    traderData      : str  (persisted string from previous tick)
    listings        : dict[Symbol, Listing]
    order_depths    : dict[Symbol, OrderDepth]
    own_trades      : dict[Symbol, list[Trade]]
    market_trades   : dict[Symbol, list[Trade]]
    position        : dict[Symbol, int]
    observations    : Observation
"""

from dataclasses import dataclass, field
from typing import Dict, List

Symbol = str
UserId = str
Product = str


@dataclass
class Order:
    symbol: Symbol
    price: int
    quantity: int

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({self.symbol} {side} {abs(self.quantity)}@{self.price})"


@dataclass
class OrderDepth:
    # price → volume  (positive volumes on both sides)
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)


@dataclass
class Trade:
    symbol: Symbol
    price: int
    quantity: int
    buyer: UserId = ""
    seller: UserId = ""
    timestamp: int = 0


@dataclass
class Listing:
    symbol: Symbol
    product: Product
    denomination: str


@dataclass
class Observation:
    plainValueObservations: Dict[str, float] = field(default_factory=dict)
    conversionObservations: Dict[str, object] = field(default_factory=dict)


@dataclass
class TradingState:
    traderData: str
    timestamp: int
    listings: Dict[Symbol, Listing]
    order_depths: Dict[Symbol, OrderDepth]
    own_trades: Dict[Symbol, List[Trade]]
    market_trades: Dict[Symbol, List[Trade]]
    position: Dict[Symbol, int]
    observations: Observation
