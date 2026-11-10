# File: src/pricing.py — replace entire file
"""
Black-76 pricing and Greeks for European options on futures.

All functions accept single numbers or NumPy arrays (vectorised).
Inputs:
    F      futures price
    K      strike
    T      time to expiry in years (must be > 0)
    r      continuously compounded risk-free rate
    sigma  annualised volatility (0.30 = 30%)
    option_type  "call" or "put"
"""

import numpy as np
from scipy.stats import norm


def _check_option_type(option_type):
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _d1_d2(F, K, T, sigma):
    """The two standardised distances used in every Black-76 formula."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(T <= 0):
        raise ValueError("T must be strictly positive; at expiry use the payoff instead.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive.")
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def black76_price(F, K, T, r, sigma, option_type="call"):
    """Black-76 premium."""
    _check_option_type(option_type)
    d1, d2 = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))
    if option_type == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_delta(F, K, T, r, sigma, option_type="call"):
    """dV/dF: number of futures per option. This is the hedge ratio."""
    _check_option_type(option_type)
    d1, _ = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))
    if option_type == "call":
        return df * norm.cdf(d1)
    return -df * norm.cdf(-d1)


def black76_gamma(F, K, T, r, sigma):
    """d2V/dF2: how fast delta changes as F moves. Same for calls and puts."""
    d1, _ = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))
    return df * norm.pdf(d1) / (np.asarray(F, dtype=float) * sigma * np.sqrt(T))


def black76_vega(F, K, T, r, sigma):
    """dV/dsigma per 1 vol point (i.e. per 0.01 of sigma). Same for calls and puts."""
    d1, _ = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))
    return df * np.asarray(F, dtype=float) * norm.pdf(d1) * np.sqrt(T) / 100.0


def black76_theta(F, K, T, r, sigma, option_type="call", days_per_year=252):
    """Change in option value per day of time passing (usually negative for a long option)."""
    _check_option_type(option_type)
    d1, _ = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))
    price = black76_price(F, K, T, r, sigma, option_type)
    decay = -df * np.asarray(F, dtype=float) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    theta_per_year = decay + r * price
    return theta_per_year / days_per_year


if __name__ == "__main__":
    # Quick self-check, run only when this file is run directly.

    # 1) Textbook example: Hull, Black's model on a futures option.
    #    F = K = 20, r = 9%, T = 4 months, vol = 25%. Hull reports 1.12 (rounded).
    F, K, T, r, sigma = 20.0, 20.0, 4.0 / 12.0, 0.09, 0.25
    call = float(black76_price(F, K, T, r, sigma, "call"))
    put = float(black76_price(F, K, T, r, sigma, "put"))
    print("Hull example")
    print(f"  call = {call:.4f}   put = {put:.4f}   (Hull: 1.12)")

    # 2) Put-call parity: C - P should equal exp(-rT) * (F - K) exactly.
    parity_gap = (call - put) - np.exp(-r * T) * (F - K)
    print(f"  put-call parity gap = {parity_gap:.2e}   (should be ~1e-16)")

    # 3) Project baseline: ATM call, F = K = 100, 30 trading days, 30% vol, r = 0.
    F, K, T, r, sigma = 100.0, 100.0, 30 / 252, 0.0, 0.30
    print("\nBaseline ATM call (F=K=100, 30 trading days, 30% vol, r=0)")
    print(f"  price = {float(black76_price(F, K, T, r, sigma)):.4f}")
    print(f"  delta = {float(black76_delta(F, K, T, r, sigma)):.4f}")
    print(f"  gamma = {float(black76_gamma(F, K, T, r, sigma)):.4f}")
    print(f"  vega  = {float(black76_vega(F, K, T, r, sigma)):.4f}  per vol point")
    print(f"  theta = {float(black76_theta(F, K, T, r, sigma)):.4f}  per trading day")