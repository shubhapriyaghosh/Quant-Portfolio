import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


def greeks(S, K, T, r, sigma, option_type="call"):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    gamma = norm.pdf(D1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(D1) * np.sqrt(T) / 100
    rho_call = K * T * np.exp(-r * T) * norm.cdf(D2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-D2) / 100

    if option_type == "call":
        delta = norm.cdf(D1)
        theta = (
            -S * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(D2)
        ) / 365
        rho = rho_call
    else:
        delta = norm.cdf(D1) - 1
        theta = (
            -S * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-D2)
        ) / 365
        rho = rho_put

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho,
    }


def payoff_grid(K, premium, option_type="call", n=100):
    prices = np.linspace(K * 0.5, K * 1.5, n)
    if option_type == "call":
        payoff = np.maximum(prices - K, 0) - premium
    else:
        payoff = np.maximum(K - prices, 0) - premium
    return prices, payoff
