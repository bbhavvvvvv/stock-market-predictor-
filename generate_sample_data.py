#!/usr/bin/env python3
"""Generate synthetic OHLCV data for local testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, output: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)

    close = [100.0]
    for _ in range(rows - 1):
        daily_return = rng.normal(0.0007, 0.02)
        close.append(close[-1] * (1 + daily_return))
    close = np.array(close)

    open_prices = close * (1 + rng.normal(0, 0.004, size=rows))
    high = np.maximum(open_prices, close) * (1 + rng.uniform(0.001, 0.015, size=rows))
    low = np.minimum(open_prices, close) * (1 - rng.uniform(0.001, 0.015, size=rows))
    volume = rng.integers(1_000_000, 8_000_000, size=rows)

    df = pd.DataFrame(
        {
            "Date": dates.date.astype(str),
            "Open": open_prices.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Sample data written to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic stock data")
    parser.add_argument("--rows", type=int, default=300, help="Number of business-day rows")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_data.csv"),
        help="Where to save the generated CSV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate(rows=args.rows, output=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
