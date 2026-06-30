import numpy as np
import pandas as pd
from scipy.stats import norm


def portfolio_returns(prices, positions):
    """Calculate portfolio daily returns from price data and share positions."""
    positions = pd.Series(positions)
    aligned = prices[positions.index]
    values = aligned * positions
    portfolio_value = values.sum(axis=1)
    returns = portfolio_value.pct_change().dropna()
    return portfolio_value, returns


def historical_var(returns, confidence=0.95):
    alpha = 1 - confidence
    return np.percentile(returns, alpha * 100)


def parametric_var(returns, confidence=0.95):
    z = norm.ppf(1 - confidence)
    return returns.mean() + z * returns.std()


def expected_shortfall(returns, confidence=0.95):
    var = historical_var(returns, confidence)
    return returns[returns <= var].mean()


def annualized_volatility(returns, periods=252):
    return returns.std() * np.sqrt(periods)


def max_drawdown(portfolio_value):
    drawdown = portfolio_value / portfolio_value.cummax() - 1
    return drawdown.min(), drawdown


def stress_test(portfolio_value, shock=-0.20):
    latest_value = portfolio_value.iloc[-1]
    stressed_value = latest_value * (1 + shock)
    loss = stressed_value - latest_value
    return {
        "Latest Portfolio Value": latest_value,
        "Shock": shock,
        "Stressed Value": stressed_value,
        "Loss": loss,
    }
