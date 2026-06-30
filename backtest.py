import pandas as pd

from metrics import calculate_performance, count_trades


def run_backtest(data, signals, initial_capital=100000, transaction_cost_bps=5):
    """Vectorized long/flat backtest.

    transaction_cost_bps=5 means 0.05% per position change.
    """
    result = signals.copy()
    result["market_return"] = data["Close"].pct_change().fillna(0)
    result["strategy_return_before_cost"] = result["position"] * result["market_return"]

    turnover = result["position"].diff().abs().fillna(0)
    cost = turnover * transaction_cost_bps / 10000

    result["strategy_return"] = result["strategy_return_before_cost"] - cost
    result["equity_curve"] = initial_capital * (1 + result["strategy_return"]).cumprod()
    result["buy_hold_curve"] = initial_capital * (1 + result["market_return"]).cumprod()
    result["drawdown"] = result["equity_curve"] / result["equity_curve"].cummax() - 1

    metrics = calculate_performance(result["strategy_return"], result["equity_curve"])
    entries, exits = count_trades(result["position"])
    metrics["Entries"] = entries
    metrics["Exits"] = exits
    metrics["Final Equity"] = result["equity_curve"].iloc[-1]

    return result, metrics
