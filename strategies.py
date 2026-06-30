import pandas as pd


def moving_average_crossover(data, short_window=20, long_window=50):
    """Long when short MA > long MA, otherwise flat."""
    signals = pd.DataFrame(index=data.index)
    signals["close"] = data["Close"]
    signals["short_ma"] = data["Close"].rolling(short_window).mean()
    signals["long_ma"] = data["Close"].rolling(long_window).mean()
    signals["signal"] = 0
    signals.loc[signals["short_ma"] > signals["long_ma"], "signal"] = 1
    signals["position"] = signals["signal"].shift(1).fillna(0)
    return signals


def momentum_strategy(data, lookback=20):
    """Long when lookback return is positive, flat otherwise."""
    signals = pd.DataFrame(index=data.index)
    signals["close"] = data["Close"]
    signals["momentum"] = data["Close"].pct_change(lookback)
    signals["signal"] = 0
    signals.loc[signals["momentum"] > 0, "signal"] = 1
    signals["position"] = signals["signal"].shift(1).fillna(0)
    return signals


def mean_reversion_strategy(data, window=20, z_entry=-1.0):
    """Long when price is below rolling mean by z_entry standard deviations."""
    signals = pd.DataFrame(index=data.index)
    signals["close"] = data["Close"]
    rolling_mean = data["Close"].rolling(window).mean()
    rolling_std = data["Close"].rolling(window).std()
    signals["z_score"] = (data["Close"] - rolling_mean) / rolling_std
    signals["signal"] = 0
    signals.loc[signals["z_score"] < z_entry, "signal"] = 1
    signals["position"] = signals["signal"].shift(1).fillna(0)
    return signals
