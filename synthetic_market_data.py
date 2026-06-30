import numpy as np
import pandas as pd


def make_price_data(
    tickers=None,
    start_price=100.0,
    periods=756,
    seed=42,
):
    """Generate synthetic daily price data for offline testing.

    periods=756 roughly means 3 trading years.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "SPY", "TLT", "GLD"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    prices = {}
    for i, ticker in enumerate(tickers):
        drift = 0.00025 + i * 0.00003
        vol = 0.012 + i * 0.001
        rets = rng.normal(drift, vol, size=periods)
        series = start_price * np.exp(np.cumsum(rets))
        prices[ticker] = series

    return pd.DataFrame(prices, index=dates)


def make_ohlcv(ticker="SPY", periods=756, seed=7):
    """Generate synthetic OHLCV data for backtesting demos."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    returns = rng.normal(0.00035, 0.012, periods)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, periods))
    volume = rng.integers(1_000_000, 5_000_000, periods)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
