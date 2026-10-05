"""Trading strategies.

Each strategy takes a price Series and returns a *signal* Series:
1.0 = want to be invested, 0.0 = want to be in cash.

The signal at day t uses only data up to and including day t. The engine
shifts it forward one day before trading on it, which avoids lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Always invested. The benchmark every strategy should be compared to."""
    return pd.Series(1.0, index=prices.index, name="buy_and_hold")


def moving_average_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Invested when the fast moving average is above the slow one."""
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    signal[slow_ma.isna()] = 0.0  # not enough history yet
    signal.name = f"ma_cross_{fast}_{slow}"
    return signal


def momentum(prices: pd.Series, lookback: int = 126) -> pd.Series:
    """Invested when the return over the past `lookback` days is positive."""
    past_return = prices.pct_change(lookback)
    signal = (past_return > 0).astype(float)
    signal[past_return.isna()] = 0.0
    signal.name = f"momentum_{lookback}"
    return signal


def _rsi(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(loss.notna() & gain.notna())


def rsi_mean_reversion(
    prices: pd.Series, window: int = 14, low: float = 30.0, high: float = 70.0
) -> pd.Series:
    """Buy when RSI drops below `low` (oversold), sell when it rises above `high`."""
    rsi = _rsi(prices, window)
    signal = np.zeros(len(prices))
    holding = 0.0
    for i, value in enumerate(rsi.to_numpy()):
        if not np.isnan(value):
            if value < low:
                holding = 1.0
            elif value > high:
                holding = 0.0
        signal[i] = holding
    return pd.Series(signal, index=prices.index, name=f"rsi_{window}_{int(low)}_{int(high)}")
