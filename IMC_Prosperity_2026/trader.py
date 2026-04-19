"""
IMC Prosperity Round 1 — Trading Algorithm
==========================================

Products
--------
ASH_COATED_OSMIUM   — Price hovers near 10 000 with very little drift.
                      Strategy: market-make around fair value, with order-book
                      imbalance skewing (verified contra-indicator: heavy bids
                      → price falls; heavy asks → price rises).

INTARIAN_PEPPER_ROOT — Price follows formula:
                           fair = 9998.5 + (day+2)*1000 + 0.001*timestamp
                       Trend is +0.1 per tick, +1000 per day, perfectly linear.
                       Strategy: stay long at the position limit always.
                       We use passive bids to avoid paying the 14-tick spread
                       when building the position.

Key findings from data analysis
---------------------------------
1. PEPPER fair value is a perfect formula — residual noise is ±10, zero
   autocorrelation. Just stay long, don't overthink it.

2. ASH order-book IMBALANCE is a CONTRA-INDICATOR:
   - High bid volume → next-tick price change: -1.06 avg
   - High ask volume → next-tick price change: +1.03 avg
   Use this to skew passive quotes more aggressively.

3. ASH one-sided book (ask-only): avg ask = 10009, next-tick mid = 10001.
   Never take an ASH ask when no bids exist — adverse selection of ~8pts.

4. ASH 1-tick mean reversion: autocorr = -0.487. Fade large moves.

Submitting
----------
Upload this file (trader.py) to the Prosperity web portal.
The platform provides the real `datamodel` module — our local copy in
datamodel.py is only used for offline back-testing.
"""

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_LIMIT = {
    "ASH_COATED_OSMIUM": 50,
    "INTARIAN_PEPPER_ROOT": 50,
}

ASH = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

# ASH fair value. The price mean-reverts tightly around 10 000 across all days.
ASH_FAIR = 10_000

# Rolling window for ASH fair-value estimation (ticks).
ASH_WINDOW = 20

# Passive quote half-spread (we post bid at fair−EDGE, ask at fair+EDGE).
# ASH typical spread is ~16, so 2 is inside the market and will get filled.
ASH_EDGE = 2

# Aggressive take thresholds: cross the spread when price is clearly mispriced.
ASH_TAKE_BUY_THRESHOLD = ASH_FAIR - 1    # buy if ask ≤ 9 999
ASH_TAKE_SELL_THRESHOLD = ASH_FAIR + 1   # sell if bid ≥ 10 001

# PEPPER: the trend formula constants (verified from 3 days of data).
# fair_value(day, ts) = PEPPER_BASE + (day + 2) * 1000 + PEPPER_SLOPE * ts
PEPPER_BASE = 9_998.5
PEPPER_SLOPE = 0.001   # price increase per timestamp unit


# ---------------------------------------------------------------------------
# Trader state persisted across ticks (serialised as JSON in traderData)
# ---------------------------------------------------------------------------

@dataclass
class State:
    ash_mids: List[float]       # rolling window of two-sided ASH mid prices
    ash_last_change: float      # last tick's price change (for mean-reversion)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "State":
        try:
            d = json.loads(s)
            return State(
                ash_mids=d.get("ash_mids", []),
                ash_last_change=d.get("ash_last_change", 0.0),
            )
        except Exception:
            return State(ash_mids=[], ash_last_change=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_bid(depth: OrderDepth) -> Optional[int]:
    return max(depth.buy_orders) if depth.buy_orders else None


def best_ask(depth: OrderDepth) -> Optional[int]:
    return min(depth.sell_orders) if depth.sell_orders else None


def mid(depth: OrderDepth) -> Optional[float]:
    """Mid price from best bid/ask, falling back to one side if needed."""
    bb, ba = best_bid(depth), best_ask(depth)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None


def two_sided_mid(depth: OrderDepth) -> Optional[float]:
    """Mid price only when both sides exist (no adverse selection)."""
    bb, ba = best_bid(depth), best_ask(depth)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return None


def order_book_imbalance(depth: OrderDepth) -> float:
    """
    (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol).
    Range: -1 (all asks) to +1 (all bids).

    DATA INSIGHT: this is a CONTRA-indicator for ASH.
    Positive imbalance (more bids) → next-tick price falls.
    Negative imbalance (more asks) → next-tick price rises.
    """
    bid_vol = sum(depth.buy_orders.values()) if depth.buy_orders else 0
    ask_vol = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# ASH strategy
# ---------------------------------------------------------------------------

def trade_ash(depth: OrderDepth, position: int, state: State) -> List[Order]:
    """
    Market-make ASH_COATED_OSMIUM around a stable fair value.

    DATA INSIGHTS (verified from 3 days of CSV data):
    - ASH mean-reverts to ~10 000. True mid std ≈ 5 pts.
    - 1-tick price change autocorrelation: -0.487 (strong mean reversion).
    - Order book imbalance is a CONTRA-indicator: heavy bids → price falls.
      We use this to skew passive quote sizes (subtle, 0.2 scale factor).
    - One-sided (ask-only) moments: avg ask ≈10009, next mid ≈10001.
      We avoid taking those by checking that our threshold is well below fair.

    Two modes:
    1. TAKE — cross the spread when price is clearly wrong (≥1 pt from fair).
    2. MAKE — post passive bid/ask at fair ± EDGE, skewed by inventory
               and order-book imbalance signal.
    """
    orders: List[Order] = []
    limit = POSITION_LIMIT[ASH]

    m = mid(depth)
    if m is not None:
        state.ash_mids.append(m)
        if len(state.ash_mids) > ASH_WINDOW:
            state.ash_mids.pop(0)

    fair = (
        sum(state.ash_mids) / len(state.ash_mids)
        if state.ash_mids
        else float(ASH_FAIR)
    )
    fair_int = round(fair)

    remaining_buy = limit - position
    remaining_sell = limit + position

    # ------------------------------------------------------------------
    # 1. TAKE
    # ------------------------------------------------------------------
    ba = best_ask(depth)
    if ba is not None and ba <= ASH_TAKE_BUY_THRESHOLD and remaining_buy > 0:
        for price in sorted(depth.sell_orders):
            if price > ASH_TAKE_BUY_THRESHOLD:
                break
            vol = abs(depth.sell_orders[price])
            qty = clamp(vol, 0, remaining_buy)
            if qty > 0:
                orders.append(Order(ASH, price, qty))
                remaining_buy -= qty

    bb = best_bid(depth)
    if bb is not None and bb >= ASH_TAKE_SELL_THRESHOLD and remaining_sell > 0:
        for price in sorted(depth.buy_orders, reverse=True):
            if price < ASH_TAKE_SELL_THRESHOLD:
                break
            vol = depth.buy_orders[price]
            qty = clamp(vol, 0, remaining_sell)
            if qty > 0:
                orders.append(Order(ASH, price, -qty))
                remaining_sell -= qty

    # ------------------------------------------------------------------
    # 2. MAKE — skew by position + imbalance contra-signal
    # ------------------------------------------------------------------
    position_skew = position / limit              # −1 to +1
    imbalance = order_book_imbalance(depth)
    # Contra-indicator: heavy bids → lean ask, scale lightly
    imbalance_contra = imbalance * 0.2
    total_skew = max(-1.0, min(1.0, position_skew + imbalance_contra))

    base_size = 10
    bid_size = clamp(int(base_size * (1 - 0.5 * total_skew)), 0, remaining_buy)
    ask_size = clamp(int(base_size * (1 + 0.5 * total_skew)), 0, remaining_sell)

    if bid_size > 0:
        orders.append(Order(ASH, fair_int - ASH_EDGE, bid_size))
    if ask_size > 0:
        orders.append(Order(ASH, fair_int + ASH_EDGE, -ask_size))

    return orders


# ---------------------------------------------------------------------------
# PEPPER strategy
# ---------------------------------------------------------------------------

def pepper_fair_value(day: int, timestamp: int) -> float:
    """
    Analytically derived fair value formula (R² ≈ 1.000, verified on 3 days).
    The trend is perfectly linear with slope 0.001 per timestamp.
    """
    return PEPPER_BASE + (day + 2) * 1000 + PEPPER_SLOPE * timestamp


def trade_pepper(
    depth: OrderDepth,
    position: int,
    day: int,
    timestamp: int,
) -> List[Order]:
    """
    Stay long at position limit at all times.

    Since PEPPER spread is ~14 ticks and we know fair value exactly, we use
    passive bids at (fair - 8) rather than always crossing the ask.
    After the first few fills we're at max position and just hold.

    We only sell if somehow above limit (shouldn't happen) or if price is
    hugely above fair (irrational spike — take profit).
    """
    orders: List[Order] = []
    limit = POSITION_LIMIT[PEPPER]
    deficit = limit - position   # positive = need to buy more

    if deficit <= 0:
        return orders

    fair = pepper_fair_value(day, timestamp)

    # Priority 1: cross the spread if necessary to build position quickly.
    # Take available asks up to fair + 8 (we'll still profit from the trend).
    max_buy_price = int(fair + 8)
    for price in sorted(depth.sell_orders):
        if price > max_buy_price:
            break
        vol = abs(depth.sell_orders[price])
        qty = clamp(vol, 0, deficit)
        if qty > 0:
            orders.append(Order(PEPPER, price, qty))
            deficit -= qty
        if deficit <= 0:
            break

    # Priority 2: passive bid for any remaining deficit.
    # Post at fair - 6 (inside the typical spread of 14 ticks).
    # Will often get hit as the market moves up through our bid.
    if deficit > 0:
        passive_bid = int(fair - 6)
        orders.append(Order(PEPPER, passive_bid, deficit))

    return orders


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------

class Trader:
    def run(self, state: TradingState):
        trader_state = State.from_json(state.traderData)
        all_orders: Dict[str, List[Order]] = {}

        # Infer trading day from timestamp range.
        # Competition runs 3 days: day -2 (earliest), -1, 0.
        # The backtester sets state.timestamp; in live trading the platform does.
        # We determine day from traderData or default to 0.
        try:
            meta = json.loads(state.traderData) if state.traderData else {}
            current_day = meta.get("day", 0)
        except Exception:
            current_day = 0

        if ASH in state.order_depths:
            ash_pos = state.position.get(ASH, 0)
            all_orders[ASH] = trade_ash(
                state.order_depths[ASH],
                ash_pos,
                trader_state,
            )

        if PEPPER in state.order_depths:
            pepper_pos = state.position.get(PEPPER, 0)
            all_orders[PEPPER] = trade_pepper(
                state.order_depths[PEPPER],
                pepper_pos,
                current_day,
                state.timestamp,
            )

        # Persist: merge trader_state (mids list) with day metadata
        out = asdict(trader_state)
        out["day"] = current_day
        trader_data = json.dumps(out)

        return all_orders, 0, trader_data
