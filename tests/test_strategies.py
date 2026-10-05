import pytest

from backtester import (
    generate_synthetic_prices,
    momentum,
    moving_average_crossover,
    rsi_mean_reversion,
)


@pytest.fixture
def prices():
    return generate_synthetic_prices(n_days=400, seed=7)


def test_signals_are_binary_and_aligned(prices):
    for sig in (
        moving_average_crossover(prices),
        momentum(prices),
        rsi_mean_reversion(prices),
    ):
        assert sig.index.equals(prices.index)
        assert set(sig.unique()) <= {0.0, 1.0}


def test_ma_cross_flat_during_warmup(prices):
    sig = moving_average_crossover(prices, 20, 50)
    assert (sig.iloc[:49] == 0).all()


def test_ma_cross_rejects_bad_windows(prices):
    with pytest.raises(ValueError):
        moving_average_crossover(prices, 50, 20)
