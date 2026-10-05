"""Command-line runner.

Examples:
    python run_backtest.py --synthetic --strategy ma_cross
    python run_backtest.py --ticker AAPL --start 2018-01-01 --strategy momentum
    python run_backtest.py --ticker RELIANCE.NS --strategy rsi
    python run_backtest.py --csv data/sample_prices.csv --strategy ma_cross
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtester import (  # noqa: E402
    download_prices,
    generate_synthetic_prices,
    load_prices_csv,
    momentum,
    moving_average_crossover,
    rsi_mean_reversion,
    run_backtest,
)
from backtester.plotting import plot_result  # noqa: E402

STRATEGIES = {
    "ma_cross": moving_average_crossover,
    "momentum": momentum,
    "rsi": rsi_mean_reversion,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Simple stock strategy backtester")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", help="Yahoo Finance ticker, e.g. AAPL or TCS.NS")
    src.add_argument("--csv", help="CSV with Date and Close columns")
    src.add_argument("--synthetic", action="store_true", help="Use simulated prices")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--strategy", choices=STRATEGIES, default="ma_cross")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if args.ticker:
        prices, label = download_prices(args.ticker, args.start, args.end), args.ticker
    elif args.csv:
        prices, label = load_prices_csv(args.csv), Path(args.csv).stem
    else:
        prices, label = generate_synthetic_prices(), "synthetic"

    signal = STRATEGIES[args.strategy](prices)
    result = run_backtest(prices, signal, args.capital, args.cost_bps)

    pd_opts = {"float_format": lambda x: f"{x:,.3f}"}
    print(f"\n{label} | strategy: {args.strategy} | trades: {result.n_trades}\n")
    print(result.metrics().to_string(**pd_opts))

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{label}_{args.strategy}.png"
    plot_result(result, title=f"{label} - {args.strategy}", path=str(out_path))
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
