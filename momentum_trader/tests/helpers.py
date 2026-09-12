"""Shared test fixtures/helpers for building small synthetic bar DataFrames."""
from __future__ import annotations

import pandas as pd

TZ = "America/New_York"


def make_bars(date_str: str, ohlcv: list[tuple[float, float, float, float, int]], start_time: str = "09:30") -> pd.DataFrame:
    """Build a one-day DataFrame of 1-minute bars from a list of (o,h,l,c,v) tuples.

    Bars are timestamped one minute apart starting at `start_time` ET, matching
    the shape src.data_loader.load_bars() would hand to the rest of the pipeline.
    """
    start = pd.Timestamp(f"{date_str} {start_time}", tz=TZ)
    rows = []
    for i, (o, h, l, c, v) in enumerate(ohlcv):
        rows.append({
            "timestamp": start + pd.Timedelta(minutes=i),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.date
    return df


def make_multi_day_bars(days: dict[str, list[tuple[float, float, float, float, int]]]) -> pd.DataFrame:
    frames = [make_bars(date_str, ohlcv) for date_str, ohlcv in days.items()]
    return pd.concat(frames, ignore_index=True)


def flat_day(date_str: str, n: int = 30, price: float = 100.0, volume: int = 1000) -> pd.DataFrame:
    ohlcv = [(price, price + 0.01, price - 0.01, price, volume) for _ in range(n)]
    return make_bars(date_str, ohlcv)
