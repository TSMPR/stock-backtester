"""Price data loading: CSV files, synthetic data, or Yahoo Finance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_prices_csv(path: str, column: str = "Close") -> pd.Series:
    """Load a price series from a CSV with a Date column and a price column."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    prices = df[column].astype(float).dropna()
    prices.name = column
    return prices


def generate_synthetic_prices(
    n_days: int = 1500,
    start: str = "2018-01-01",
    annual_return: float = 0.10,
    annual_vol: float = 0.20,
    initial_price: float = 100.0,
    seed: int | None = 42,
) -> pd.Series:
    """Simulate a price path with geometric Brownian motion (business days).

    Useful for testing without internet access.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal(
        (annual_return - 0.5 * annual_vol**2) * dt,
        annual_vol * np.sqrt(dt),
        size=n_days,
    )
    prices = initial_price * np.exp(np.cumsum(shocks))
    index = pd.bdate_range(start=start, periods=n_days, name="Date")
    return pd.Series(prices, index=index, name="Close")


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """Download adjusted close prices from Yahoo Finance (needs `yfinance`).

    Examples: "AAPL", "RELIANCE.NS", "^NSEI".
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install yfinance: pip install yfinance") from exc

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a 1-column frame
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.name = "Close"
    return close
