"""Matplotlib charts for backtest results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt

from .engine import BacktestResult
from .metrics import drawdown_series


def plot_result(result: BacktestResult, title: str = "Backtest", path: str | None = None):
    """Equity curves on top, drawdowns below. Saves to `path` if given."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    ax1.plot(result.equity, label="Strategy", linewidth=1.6)
    ax1.plot(result.benchmark_equity, label="Buy & hold", linewidth=1.2, alpha=0.8)
    ax1.set_title(title)
    ax1.set_ylabel("Portfolio value")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.fill_between(
        result.strategy_returns.index,
        drawdown_series(result.strategy_returns) * 100,
        0,
        alpha=0.5,
        label="Strategy",
    )
    ax2.plot(drawdown_series(result.asset_returns) * 100, linewidth=1, label="Buy & hold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
