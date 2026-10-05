"""Backtest engine: turns prices + a signal into strategy returns."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import summary


@dataclass
class BacktestResult:
    prices: pd.Series
    signal: pd.Series
    position: pd.Series          # position actually held each day (signal shifted by 1)
    asset_returns: pd.Series
    strategy_returns: pd.Series  # after transaction costs
    equity: pd.Series            # strategy equity curve in currency
    benchmark_equity: pd.Series  # buy-and-hold equity curve
    n_trades: int

    def metrics(self, risk_free_rate: float = 0.0) -> pd.DataFrame:
        """Strategy vs buy-and-hold metrics side by side."""
        return pd.DataFrame(
            {
                "Strategy": summary(self.strategy_returns, risk_free_rate),
                "Buy & hold": summary(self.asset_returns, risk_free_rate),
            }
        )


def run_backtest(
    prices: pd.Series,
    signal: pd.Series,
    initial_capital: float = 10_000.0,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Simulate trading `signal` on `prices`.

    - The signal is shifted one day: a decision made at today's close earns
      tomorrow's return. This prevents lookahead bias.
    - `cost_bps` is charged per unit of position change (5 bps = 0.05%).
    """
    signal = signal.reindex(prices.index).fillna(0.0)
    position = signal.shift(1).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    costs = turnover * (cost_bps / 10_000)

    strategy_returns = position * asset_returns - costs
    equity = initial_capital * (1 + strategy_returns).cumprod()
    benchmark_equity = initial_capital * (1 + asset_returns).cumprod()

    return BacktestResult(
        prices=prices,
        signal=signal,
        position=position,
        asset_returns=asset_returns,
        strategy_returns=strategy_returns,
        equity=equity,
        benchmark_equity=benchmark_equity,
        n_trades=int((turnover > 0).sum()),
    )
