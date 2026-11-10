# File: src/simulate.py — replace entire file
"""
Path simulation for futures prices.

Phase 0: geometric Brownian motion (GBM), simulated exactly in log space.
Later phases add Heston stochastic volatility here.
"""

import numpy as np

from src.pricing import black76_price

SEED = 42  # default seed; every experiment records the seed it used


def simulate_gbm_paths(F0, sigma, T, n_steps, n_paths, seed=SEED, mu=0.0):
    """
    Simulate futures price paths under GBM.

    Returns an array of shape (n_paths, n_steps + 1).
    Column 0 is today's price F0; column n_steps is the price at expiry.
    mu is the drift. It is 0 under the pricing measure; it stays 0 in our
    hedging experiments unless we deliberately switch it on.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return F0 * np.exp(log_paths)


def mc_price_european(F0, K, T, r, sigma, option_type="call", n_paths=200_000, seed=SEED):
    """
    Monte Carlo price of a European option on a future.

    Returns (price, standard_error).
    Only the terminal price matters for a European payoff, so we simulate it in one step.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    F_T = F0 * np.exp(-0.5 * sigma**2 * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoff = np.maximum(F_T - K, 0.0)
    else:
        payoff = np.maximum(K - F_T, 0.0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    standard_error = discounted.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(standard_error)


if __name__ == "__main__":
    # Quick self-check, run only when this file is run directly.
    F0, K, T, r, sigma = 100.0, 100.0, 30 / 252, 0.0, 0.30
    closed_form = float(black76_price(F0, K, T, r, sigma, "call"))

    # 1) Monte Carlo vs closed form, and how the error shrinks with more paths.
    print("Baseline ATM call: Monte Carlo vs Black-76 closed form")
    print(f"  closed form = {closed_form:.4f}")
    print(f"  {'paths':>8} {'MC price':>9} {'std err':>8} {'z-score':>8}")
    for n in (10_000, 40_000, 160_000, 640_000):
        price, se = mc_price_european(F0, K, T, r, sigma, "call", n_paths=n)
        z = (price - closed_form) / se
        print(f"  {n:>8} {price:>9.4f} {se:>8.4f} {z:>8.2f}")

    # 2) Path simulator: shape and the martingale check (average terminal F should be F0).
    paths = simulate_gbm_paths(F0, sigma, T, n_steps=30, n_paths=10_000)
    terminal = paths[:, -1]
    se_terminal = terminal.std(ddof=1) / np.sqrt(len(terminal))
    print("\nPath simulator (10,000 paths, 30 daily steps)")
    print(f"  array shape = {paths.shape}   (paths, time points)")
    print(f"  first path, first 5 days = {np.round(paths[0, :5], 2)}")
    print(f"  average terminal F = {terminal.mean():.3f} +/- {se_terminal:.3f}   (should be close to 100)")