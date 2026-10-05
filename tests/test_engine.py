import pandas as pd
import pytest

from backtester import (
    buy_and_hold,
    generate_synthetic_prices,
    moving_average_crossover,
    run_backtest,
)


def test_buy_and_hold_matches_benchmark_without_costs():
    prices = generate_synthetic_prices(n_days=300, seed=1)
    result = run_backtest(prices, buy_and_hold(prices), cost_bps=0)
    # Position is 0 on day 0 (signal shifted), so compare from day 1 onwards.
    assert result.equity.iloc[-1] == pytest.approx(result.benchmark_equity.iloc[-1])


def test_signal_is_shifted_to_avoid_lookahead():
    prices = pd.Series([100.0, 110.0, 99.0, 99.0], index=pd.bdate_range("2024-01-01", periods=4))
    # Signal says "invested" only on day 1 (the +10% day).
    signal = pd.Series([0.0, 1.0, 0.0, 0.0], index=prices.index)
    result = run_backtest(prices, signal, cost_bps=0)
    # We can only earn day 2's return (-10%), NOT day 1's +10%.
    assert result.strategy_returns.iloc[1] == pytest.approx(0.0)
    assert result.strategy_returns.iloc[2] == pytest.approx(99 / 110 - 1)


def test_costs_reduce_returns():
    prices = generate_synthetic_prices(n_days=500, seed=3)
    signal = moving_average_crossover(prices, 5, 20)
    free = run_backtest(prices, signal, cost_bps=0)
    costly = run_backtest(prices, signal, cost_bps=50)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_trade_count_matches_signal_changes():
    prices = generate_synthetic_prices(n_days=200, seed=4)
    signal = pd.Series(0.0, index=prices.index)
    signal.iloc[50:100] = 1.0
    result = run_backtest(prices, signal)
    assert result.n_trades == 2  # one entry, one exit
