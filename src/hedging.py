# File: src/hedging.py — replace entire file
"""
Hedging P&L accounting.

hedge_pnl() takes futures paths and ANY array of hedge holdings, and returns the
terminal P&L of a short option hedged with those holdings. Every strategy in the
project (delta, Whalley-Wilmott, Leland, the network) goes through this same function.

Later phases add the neural network and training loop to this file.
"""

import numpy as np

from src.pricing import black76_price, black76_delta, black76_vega
from src.simulate import simulate_gbm_paths


def hedge_pnl(paths, holdings, payoff, cost=0.0, premium=0.0, liquidate_at_expiry=True):
    """
    Terminal P&L of a short option hedged with futures.

    paths     array (n_paths, n_steps + 1): futures prices, column 0 = today
    holdings  array (n_paths, n_steps): futures held from step t to step t+1
    payoff    array (n_paths,): option payoff at expiry (what the short pays out)
    cost      proportional transaction cost (0.001 = 10 basis points of notional)
    premium   option premium received at time 0
    liquidate_at_expiry  if True, also pay the cost of closing the hedge at expiry

    PnL = premium - payoff + sum(delta_t * (F_{t+1} - F_t)) - sum(c * |delta_t - delta_{t-1}| * F_t)
    The position before day 0 is zero, so the first trade (buying the initial hedge) is costed.
    Interest rate is zero in the hedging simulations, so no financing terms appear.
    """
    paths = np.asarray(paths, dtype=float)
    holdings = np.asarray(holdings, dtype=float)
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    if holdings.shape != (n_paths, n_steps):
        raise ValueError(
            f"holdings must have shape {(n_paths, n_steps)}, got {holdings.shape}"
        )

    price_changes = np.diff(paths, axis=1)
    hedge_gains = np.sum(holdings * price_changes, axis=1)

    previous = np.concatenate([np.zeros((n_paths, 1)), holdings[:, :-1]], axis=1)
    trades = holdings - previous
    trading_costs = cost * np.sum(np.abs(trades) * paths[:, :-1], axis=1)
    if liquidate_at_expiry:
        trading_costs = trading_costs + cost * np.abs(holdings[:, -1]) * paths[:, -1]

    return premium - payoff + hedge_gains - trading_costs


if __name__ == "__main__":
    # Quick self-check: Black-76 delta hedge of a short ATM call, daily vs 4x daily,
    # with and without transaction costs.
    F0, K, T, r, sigma = 100.0, 100.0, 30 / 252, 0.0, 0.30
    premium = float(black76_price(F0, K, T, r, sigma, "call"))
    vega_per_unit_vol = float(black76_vega(F0, K, T, r, sigma)) * 100.0
    n_paths = 20_000

    print("Short ATM call, Black-76 delta hedge, 20,000 paths")
    print(f"  premium received = {premium:.4f}")
    print(f"  {'rebal/day':>9} {'cost':>6} {'mean P&L':>9} {'std P&L':>8} {'Derman-Kamal std':>17}")

    for n_steps in (30, 120):
        paths = simulate_gbm_paths(F0, sigma, T, n_steps, n_paths)
        dt = T / n_steps
        time_to_expiry = T - dt * np.arange(n_steps)
        holdings = black76_delta(paths[:, :-1], K, time_to_expiry, r, sigma, "call")
        payoff = np.maximum(paths[:, -1] - K, 0.0)
        predicted_std = np.sqrt(np.pi / 4.0) * vega_per_unit_vol * sigma / np.sqrt(n_steps)

        for cost in (0.0, 0.001):
            pnl = hedge_pnl(paths, holdings, payoff, cost=cost, premium=premium)
            print(
                f"  {n_steps // 30:>9} {cost * 1e4:>4.0f}bp {pnl.mean():>9.4f} "
                f"{pnl.std(ddof=1):>8.4f} {predicted_std:>17.4f}"
            )