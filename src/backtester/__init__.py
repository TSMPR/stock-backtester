"""A small, readable stock strategy backtester."""

from .data import load_prices_csv, generate_synthetic_prices, download_prices
from .strategies import (
    buy_and_hold,
    moving_average_crossover,
    momentum,
    rsi_mean_reversion,
)
from .engine import run_backtest, BacktestResult
from .metrics import (
    total_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    summary,
)

__all__ = [
    "load_prices_csv",
    "generate_synthetic_prices",
    "download_prices",
    "buy_and_hold",
    "moving_average_crossover",
    "momentum",
    "rsi_mean_reversion",
    "run_backtest",
    "BacktestResult",
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "summary",
]
