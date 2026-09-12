#!/usr/bin/env python
"""Generate a synthetic 1-minute OHLCV dataset shaped like AAPL RTH data.

This is NOT real market data. It exists so the whole project — signals,
backtester, optimizer, reporting — can be exercised end-to-end without
Alpaca credentials. Real conclusions about strategy edge require running
this same pipeline against actual historical data (see scripts/download_data.py).

Usage:
    python scripts/generate_sample_data.py --start 2025-01-01 --end 2025-08-31 \
        --out data/cache/AAPL_sample.parquet --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample(start: str, end: str, seed: int = 7, start_price: float = 220.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trading_days = pd.bdate_range(start, end)

    rows = []
    price = start_price
    for day in trading_days:
        # Small day-to-day drift (slightly positive, like a mature large-cap in an uptrend).
        day_drift = rng.normal(0.0002, 0.001)
        session_minutes = pd.date_range(
            day.replace(hour=9, minute=30), day.replace(hour=15, minute=59), freq="min", tz="America/New_York"
        )
        n = len(session_minutes)

        # Per-minute log returns: small mean-reverting noise plus the day's drift,
        # with a volatility hump near the open/close (typical intraday U-shape).
        t = np.linspace(0, 1, n)
        vol_shape = 0.00035 + 0.0004 * (np.exp(-8 * t) + np.exp(-8 * (1 - t)))
        noise = rng.normal(0, 1, n)
        mean_reversion = -0.05  # gentle pull back toward the day's open, encourages pullback/reversal patterns
        log_rets = day_drift / n + vol_shape * noise

        closes = np.empty(n)
        opens = np.empty(n)
        last = price
        for i in range(n):
            drift_adj = mean_reversion * (last / price - 1.0) * 0.01
            ret = log_rets[i] + drift_adj
            opens[i] = last
            last = last * (1 + ret)
            closes[i] = last

        intrabar_noise = np.abs(rng.normal(0, vol_shape * 0.6, n)) * opens
        highs = np.maximum(opens, closes) + intrabar_noise
        lows = np.minimum(opens, closes) - intrabar_noise
        base_volume = rng.lognormal(mean=8.5, sigma=0.5, size=n)
        volume_shape = 1.5 * (np.exp(-6 * t) + np.exp(-6 * (1 - t))) + 0.5
        volumes = (base_volume * volume_shape).astype(int)

        for i in range(n):
            rows.append({
                "timestamp": session_minutes[i],
                "open": round(float(opens[i]), 4),
                "high": round(float(highs[i]), 4),
                "low": round(float(lows[i]), 4),
                "close": round(float(closes[i]), 4),
                "volume": int(max(volumes[i], 1)),
            })
        price = closes[-1]

    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-08-31")
    parser.add_argument("--out", default="data/cache/AAPL_sample.parquet")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = generate_sample(args.start, args.end, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} synthetic 1-minute bars ({df['timestamp'].dt.date.nunique()} sessions) -> {out_path}")


if __name__ == "__main__":
    main()
