import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from common.synthetic_market_data import make_price_data


def load_prices(tickers=None, start="2018-01-01", end=None):
    if tickers is None:
        tickers = ["AAPL", "MSFT", "SPY"]

    if yf is not None:
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                if "Close" in raw:
                    prices = raw["Close"]
                else:
                    prices = raw
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(tickers[0])
                return prices.dropna()
        except Exception:
            pass

    return make_price_data(tickers=tickers)
