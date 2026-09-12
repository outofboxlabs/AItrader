"""Entry-signal generation.

`generate_entry_signals` is pure and vectorized: given an indicator-enriched
DataFrame (one session's worth or many, already sorted ascending) and a
FiltersConfig/StrategyConfig, it returns a boolean Series `entry_signal`
where True at row N means "the base reversal pattern + all enabled filters
were satisfied using information available at row N's close".

The backtester is responsible for turning `entry_signal[N] == True` into an
order filled at row N+1's open (or wherever config.execution.entry_method
says) — this module never looks past row N itself, which is what keeps the
whole pipeline free of lookahead bias.
"""
from __future__ import annotations

import pandas as pd

from src.config import FiltersConfig, StrategyConfig


def base_reversal_pattern(df: pd.DataFrame, pullback_pct: float) -> pd.Series:
    """Core pullback + reversal pattern, using only row N and N-1.

    Conditions, all evaluated at row N (the prospective signal bar):
      1. Previous candle (N-1) is bearish.
      2. Current candle (N) is bullish.
      3. Current close (N) > previous candle's high (N-1) — reversal thrust.
      4. By the time of the previous (bearish) candle, price had already
         pulled back by at least `pullback_pct` from the pre-pullback
         intraday high (intraday_high_ref, which excludes the current bar
         by construction — see indicators.rolling_intraday_high).
    """
    prev_bearish = df["is_bearish"].shift(1).fillna(False)
    curr_bullish = df["is_bullish"]
    thrust_above_prev_high = df["close"] > df["high"].shift(1)

    ref_high = df["intraday_high_ref"]
    prev_low = df["low"].shift(1)
    pullback_at_prev_bar = (ref_high - prev_low) / ref_high
    sufficient_pullback = pullback_at_prev_bar >= pullback_pct

    pattern = prev_bearish & curr_bullish & thrust_above_prev_high & sufficient_pullback
    return pattern.fillna(False)


def _market_trend_ok(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Simplified market-trend filter: EMA20 slope over the last `lookback`
    bars must be non-negative (i.e. not fighting a falling market)."""
    slope = df["ema_20"] - df["ema_20"].shift(lookback)
    return (slope >= 0).fillna(False)


def _rsi_recovering(df: pd.DataFrame, oversold_threshold: float) -> pd.Series:
    """Previous bar's RSI was at/below the oversold threshold and RSI is
    now rising on the signal bar — i.e. recovering from weakness."""
    prev_rsi = df["rsi_14"].shift(1)
    was_weak = prev_rsi <= oversold_threshold
    rising = df["rsi_14"] > prev_rsi
    return (was_weak & rising).fillna(False)


def generate_entry_signals(
    df: pd.DataFrame,
    strategy_cfg: StrategyConfig,
    filters_cfg: FiltersConfig,
) -> pd.Series:
    """Compute the final entry_signal boolean series for the whole DataFrame."""
    signal = base_reversal_pattern(df, strategy_cfg.pullback_pct)

    if filters_cfg.vwap:
        signal &= (df["close"] > df["vwap"]).fillna(False)

    if filters_cfg.ema:
        signal &= (df["ema_9"] > df["ema_20"]).fillna(False)

    if filters_cfg.rsi:
        signal &= _rsi_recovering(df, filters_cfg.rsi_oversold_threshold)

    if filters_cfg.relative_volume:
        signal &= (df["relative_volume"] >= filters_cfg.relative_volume_threshold).fillna(False)

    if filters_cfg.min_volume:
        signal &= (df["volume"] >= filters_cfg.min_volume_threshold).fillna(False)

    if filters_cfg.market_trend:
        signal &= _market_trend_ok(df)

    # ATR must be warmed up (not NaN) before we can size a stop off it.
    signal &= df["atr"].notna()

    return signal.fillna(False)
