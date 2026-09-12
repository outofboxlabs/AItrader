"""Technical indicators used by the signal layer.

Every function here is causal: the value at row N is computed only from
rows <= N. This is what makes it safe for the backtester to compute
indicators once over the whole bar stream up front — a decision made using
row N's indicator values never actually depends on row N+1 or later, so
generating a signal at row N and filling it at row N+1's open (see
execution.py) introduces no lookahead bias.

Two different "reset" behaviors are used deliberately:
  * VWAP and rolling intraday high/low reset every session (they are
    inherently intraday concepts), computed via a groupby("date").
  * ATR / EMA / RSI / relative-volume are computed as continuous rolling
    series across the whole (RTH-only) bar stream, the same way a live
    trading system would maintain them from one session into the next.
    This only affects warm-up: after the first day, they are fully formed
    at the start of every following session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (exponential smoothing with alpha = 1/period)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def relative_volume(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Current bar volume vs. its own trailing rolling average (simplified
    proxy — a production system would compare against the same time-of-day
    bucket averaged over prior sessions)."""
    rolling_avg = volume.rolling(lookback, min_periods=1).mean()
    return volume / rolling_avg.replace(0.0, np.nan)


def session_vwap(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]
    cum_pv = pv.groupby(df[date_col]).cumsum()
    cum_vol = df["volume"].groupby(df[date_col]).cumsum()
    return cum_pv / cum_vol.replace(0.0, np.nan)


def rolling_intraday_high(df: pd.DataFrame, date_col: str = "date", exclude_current: bool = True) -> pd.Series:
    """Expanding max of `high` within each session.

    exclude_current=True returns the highest high *before* the current bar,
    which is what the reversal signal needs as its pullback reference point
    (otherwise a breakout bar's own high would immediately reset the
    reference and mask the pullback that just happened).
    """
    expanding_max = df.groupby(date_col)["high"].cummax()
    if exclude_current:
        return expanding_max.groupby(df[date_col]).shift(1)
    return expanding_max


def rolling_intraday_low(df: pd.DataFrame, date_col: str = "date", exclude_current: bool = True) -> pd.Series:
    expanding_min = df.groupby(date_col)["low"].cummin()
    if exclude_current:
        return expanding_min.groupby(df[date_col]).shift(1)
    return expanding_min


def add_all_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Return a copy of df with every indicator column appended."""
    out = df.copy()
    out["atr"] = atr(out, period=atr_period)
    out["ema_9"] = ema(out["close"], 9)
    out["ema_20"] = ema(out["close"], 20)
    out["rsi_14"] = rsi(out["close"], 14)
    out["relative_volume"] = relative_volume(out["volume"], lookback=20)
    out["vwap"] = session_vwap(out)
    out["intraday_high_ref"] = rolling_intraday_high(out)
    out["intraday_low_ref"] = rolling_intraday_low(out)
    out["is_bullish"] = out["close"] > out["open"]
    out["is_bearish"] = out["close"] < out["open"]
    return out
