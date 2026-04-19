"""
Local back-tester for the Prosperity trader.

Reads the price CSVs provided by the competition, reconstructs an
OrderDepth on every tick, calls Trader.run(), simulates fills, and
reports profit-and-loss.

Usage
-----
    python prosperity/backtest.py              # runs all three days
    python prosperity/backtest.py --day 0      # single day
    python prosperity/backtest.py --plot       # show price + PnL chart
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

# Add parent dir so we can find the datamodel module
sys.path.insert(0, os.path.dirname(__file__))
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    Trade,
    TradingState,
)
from trader import Trader, POSITION_LIMIT

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DAYS = [-2, -1, 0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices(day: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"prices_round_1_day_{day}.csv")
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def build_order_depth(row: pd.Series) -> OrderDepth:
    """
    Convert a single row from prices CSV into an OrderDepth object.
    Columns: bid_price_1/2/3, bid_volume_1/2/3, ask_price_1/2/3, ask_volume_1/2/3
    """
    depth = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bp > 0:
            depth.buy_orders[int(bp)] = int(bv)

        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and ap > 0:
            depth.sell_orders[int(ap)] = -int(av)  # negative for asks

    return depth


# ---------------------------------------------------------------------------
# Fill simulation
# ---------------------------------------------------------------------------

def simulate_fills(
    orders: List[Order],
    depth: OrderDepth,
    position: int,
    limit: int,
) -> tuple[List[Trade], int, float]:
    """
    Match our orders against the order book.

    Returns (filled_trades, new_position, realised_cash_flow).
    Cash flow is negative when we buy (we pay) and positive when we sell.
    """
    trades: List[Trade] = []
    pos = position
    cash = 0.0

    for order in orders:
        qty = order.quantity  # + = buy, − = sell

        if qty > 0:
            # We want to buy — match against sell_orders (asks)
            for price in sorted(depth.sell_orders):
                if price > order.price:
                    break  # our bid can't reach this ask
                avail = abs(depth.sell_orders[price])
                can_buy = min(qty, avail, limit - pos)
                if can_buy <= 0:
                    break
                trades.append(Trade(order.symbol, price, can_buy, "SUBMISSION", ""))
                pos += can_buy
                cash -= price * can_buy
                qty -= can_buy
                if qty <= 0:
                    break

        elif qty < 0:
            # We want to sell — match against buy_orders (bids)
            sell_qty = -qty
            for price in sorted(depth.buy_orders, reverse=True):
                if price < order.price:
                    break  # our ask can't reach this bid
                avail = depth.buy_orders[price]
                can_sell = min(sell_qty, avail, limit + pos)
                if can_sell <= 0:
                    break
                trades.append(Trade(order.symbol, price, -can_sell, "", "SUBMISSION"))
                pos -= can_sell
                cash += price * can_sell
                sell_qty -= can_sell
                if sell_qty <= 0:
                    break

    return trades, pos, cash


# ---------------------------------------------------------------------------
# Back-test engine
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    day: int
    pnl_series: List[float] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    product_pnl: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    product_trades: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


def run_day(day: int, initial_positions=None, initial_cash=None) -> DayResult:
    df = load_prices(day)
    products = df["product"].unique().tolist()

    positions: Dict[str, int] = initial_positions or {p: 0 for p in products}
    cash: Dict[str, float] = initial_cash or {p: 0.0 for p in products}

    listings = {p: Listing(p, p, "SEASHELLS") for p in products}
    trader = Trader()
    # Pre-seed trader_data with the day so PEPPER fair value formula works
    import json as _json
    trader_data = _json.dumps({"day": day, "ash_mids": [], "ash_last_change": 0.0})
    result = DayResult(day=day)

    for ts, group in df.groupby("timestamp"):
        order_depths: Dict[str, OrderDepth] = {}
        mid_prices: Dict[str, float] = {}

        for _, row in group.iterrows():
            prod = row["product"]
            order_depths[prod] = build_order_depth(row)
            mp = row.get("mid_price")
            if pd.notna(mp) and mp > 0:
                mid_prices[prod] = float(mp)

        state = TradingState(
            traderData=trader_data,
            timestamp=int(ts),
            listings=listings,
            order_depths=order_depths,
            own_trades={p: [] for p in products},
            market_trades={p: [] for p in products},
            position=dict(positions),
            observations=Observation(),
        )

        all_orders, _conversions, trader_data = trader.run(state)

        for prod, orders in all_orders.items():
            depth = order_depths.get(prod, OrderDepth())
            limit = POSITION_LIMIT.get(prod, 50)
            trades, new_pos, delta_cash = simulate_fills(
                orders, depth, positions.get(prod, 0), limit
            )
            positions[prod] = new_pos
            cash[prod] += delta_cash
            result.product_trades[prod] += len(trades)

        # Mark-to-market PnL = cash + position * mid_price
        total_pnl = 0.0
        for prod in products:
            mp = mid_prices.get(prod, 0)
            total_pnl += cash[prod] + positions[prod] * mp
            result.product_pnl[prod] = cash[prod] + positions[prod] * mp

        result.pnl_series.append(total_pnl)
        result.timestamps.append(int(ts))

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: List[DayResult]):
    print("\n" + "=" * 55)
    print("  IMC Prosperity Round 1 — Backtest Results")
    print("=" * 55)

    grand_total = 0.0
    for r in results:
        final_pnl = r.pnl_series[-1] if r.pnl_series else 0
        grand_total += final_pnl
        print(f"\nDay {r.day:+d}  |  Total PnL: {final_pnl:>10.1f} seashells")
        for prod, pnl in r.product_pnl.items():
            n = r.product_trades[prod]
            print(f"   {prod:<28}  {pnl:>10.1f}  ({n} fills)")

    print(f"\n{'─'*55}")
    print(f"  Grand total PnL: {grand_total:>10.1f} seashells")
    print("=" * 55)


def plot_results(results: List[DayResult]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)),
                             sharex=False)
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ax.plot(r.timestamps, r.pnl_series, lw=1.5, color="#2196F3")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_title(f"Day {r.day:+d} — PnL (seashells)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("PnL")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prosperity Round 1 Backtest")
    parser.add_argument("--day", type=int, choices=[-2, -1, 0],
                        help="Run a single day (default: all days)")
    parser.add_argument("--plot", action="store_true",
                        help="Show a PnL chart after running")
    args = parser.parse_args()

    days_to_run = [args.day] if args.day is not None else DAYS

    results = []
    for day in days_to_run:
        print(f"Running day {day:+d}...", end=" ", flush=True)
        r = run_day(day)
        final = r.pnl_series[-1] if r.pnl_series else 0
        print(f"PnL = {final:,.1f}")
        results.append(r)

    print_report(results)

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
