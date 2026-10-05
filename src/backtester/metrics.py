"""Performance metrics computed from a Series of periodic (daily) returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    """Compounded return over the whole period."""
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = (1 + returns).prod()
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio. `risk_free_rate` is an annual rate."""
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std < 1e-12 or np.isnan(std):  # tolerance: constant returns give ~0, not exactly 0
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Like Sharpe, but only penalizes downside volatility."""
    excess = returns - risk_free_rate / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    if downside < 1e-12 or np.isnan(downside):
        return float("nan")
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage drop from the running peak of the equity curve."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline (a negative number)."""
    if len(returns) == 0:
        return float("nan")
    return float(drawdown_series(returns).min())


def summary(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """All headline metrics in one dictionary."""
    return {
        "Total return": total_return(returns),
        "CAGR": cagr(returns),
        "Annual volatility": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns, risk_free_rate),
        "Sortino": sortino_ratio(returns, risk_free_rate),
        "Max drawdown": max_drawdown(returns),
    }
