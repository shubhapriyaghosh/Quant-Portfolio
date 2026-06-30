import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_returns(prices):
    return prices.pct_change().dropna()


def annualize_inputs(returns, periods_per_year=252):
    mu = returns.mean() * periods_per_year
    cov = returns.cov() * periods_per_year
    return mu, cov


def portfolio_performance(weights, mu, cov, risk_free_rate=0.02):
    weights = np.array(weights)
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights.T @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol != 0 else np.nan
    return ret, vol, sharpe


def max_sharpe_portfolio(mu, cov, risk_free_rate=0.02):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def objective(w):
        return -portfolio_performance(w, mu, cov, risk_free_rate)[2]

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def min_volatility_portfolio(mu, cov):
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def objective(w):
        return portfolio_performance(w, mu, cov)[1]

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def monte_carlo_portfolios(mu, cov, n_portfolios=5000, risk_free_rate=0.02, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    tickers = list(mu.index)
    n = len(tickers)

    for _ in range(n_portfolios):
        weights = rng.random(n)
        weights = weights / weights.sum()
        ret, vol, sharpe = portfolio_performance(weights, mu, cov, risk_free_rate)
        row = {"Return": ret, "Volatility": vol, "Sharpe": sharpe}
        row.update({ticker: weight for ticker, weight in zip(tickers, weights)})
        rows.append(row)

    return pd.DataFrame(rows)
