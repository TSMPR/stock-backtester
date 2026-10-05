# Stock Backtester

A small, readable Python backtester for testing simple trading strategies against buy-and-hold. Built to be easy to read, extend and test.

## Features

- **Data**: Yahoo Finance (US stocks, NSE tickers like `RELIANCE.NS`, indices like `^NSEI`), local CSV, or simulated prices for offline use
- **Strategies**: moving-average crossover, momentum, RSI mean reversion, buy & hold
- **Engine**: next-day execution (no lookahead bias) and transaction costs in basis points
- **Metrics**: total return, CAGR, volatility, Sharpe, Sortino, max drawdown
- **Charts**: equity curve and drawdown plot vs buy & hold
- **Tests**: pytest suite covering metrics, engine and strategies

## Project structure

```
stock-backtester/
├── src/backtester/
│   ├── data.py         # CSV / synthetic / yfinance loaders
│   ├── strategies.py   # signal generators
│   ├── engine.py       # run_backtest + BacktestResult
│   ├── metrics.py      # performance metrics
│   └── plotting.py     # matplotlib charts
├── tests/
├── data/sample_prices.csv
├── run_backtest.py     # command-line runner
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
pip install -r requirements.txt

# offline demo with the bundled sample data
python run_backtest.py --csv data/sample_prices.csv --strategy ma_cross

# real data
python run_backtest.py --ticker AAPL --start 2018-01-01 --strategy momentum
python run_backtest.py --ticker TCS.NS --strategy rsi --cost-bps 10

# run the tests
pytest
```

Charts are saved to `results/`.

## Use as a library

```python
from backtester import download_prices, moving_average_crossover, run_backtest

prices = download_prices("RELIANCE.NS", "2015-01-01")
signal = moving_average_crossover(prices, fast=20, slow=50)
result = run_backtest(prices, signal, initial_capital=100_000, cost_bps=5)
print(result.metrics())
```

## How it works

1. A strategy looks at prices and outputs a **signal**: 1 (invested) or 0 (cash).
2. The engine **shifts the signal by one day**. A decision made at today's close can only earn tomorrow's return. Skipping this is lookahead bias and makes results unrealistically good.
3. Daily strategy return = `position × asset return − transaction costs`.
4. Metrics are computed from the daily return series.

## Pitfalls to keep in mind

- **Overfitting**: tuning parameters until a chart looks good on one price history says little about the future. Test on data the strategy was not tuned on.
- **Survivorship bias**: tickers that exist today are the survivors. Delisted losers are missing from the data.
- **Costs and slippage**: real trading costs more than the model assumes, especially for frequent trading.
- **Synthetic data**: simulated prices are a random walk, so any strategy "edge" on them is luck.

## Ideas for next steps

- Walk-forward / train-test split evaluation
- Position sizing and stop losses
- Multi-asset portfolios
- Parameter grid search with out-of-sample validation

## License

MIT
