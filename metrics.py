import numpy as np
import pandas as pd


def calculate_performance(returns, equity_curve=None, periods_per_year=252):
    returns = pd.Series(returns).dropna()

    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(returns) / periods_per_year
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() != 0 else np.nan

    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(periods_per_year) if downside.std() != 0 else np.nan

    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
    }


def count_trades(position_series):
    changes = position_series.diff().fillna(0)
    entries = (changes > 0).sum()
    exits = (changes < 0).sum()
    return int(entries), int(exits)
