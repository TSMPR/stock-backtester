import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


def test_total_return_compounds():
    r = pd.Series([0.10, 0.10])
    assert total_return(r) == pytest.approx(0.21)


def test_cagr_one_year_equals_total_return():
    r = pd.Series([0.001] * 252)
    assert cagr(r) == pytest.approx(total_return(r))


def test_max_drawdown_simple():
    # equity: 1.0 -> 1.2 -> 0.6  => drawdown from peak 1.2 to 0.6 = -50%
    r = pd.Series([0.2, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_never_positive():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 500))
    assert max_drawdown(r) <= 0


def test_volatility_scales_with_sqrt_time():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 1000))
    assert annualized_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 10)))
