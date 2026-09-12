import numpy as np
import pandas as pd

from src.indicators import atr, rolling_intraday_high, session_vwap, true_range
from tests.helpers import make_bars


def test_atr_matches_manual_wilder_recursion():
    period = 5
    ohlcv = [
        (100, 101, 99, 100.5, 1000),
        (100.5, 102, 100, 101.5, 1000),
        (101.5, 101.8, 100.5, 100.8, 1000),
        (100.8, 101.0, 99.5, 99.8, 1000),
        (99.8, 100.2, 99.0, 99.5, 1000),
        (99.5, 100.5, 99.2, 100.2, 1000),
        (100.2, 100.9, 99.9, 100.7, 1000),
        (100.7, 101.5, 100.4, 101.2, 1000),
    ]
    df = make_bars("2025-03-03", ohlcv)

    tr = true_range(df)
    expected_tr = [ohlcv[0][1] - ohlcv[0][2]]  # first bar: no prev close, TR = high-low
    for i in range(1, len(ohlcv)):
        h, l = ohlcv[i][1], ohlcv[i][2]
        prev_close = ohlcv[i - 1][3]
        expected_tr.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
    np.testing.assert_allclose(tr.values, expected_tr, rtol=1e-9)

    alpha = 1.0 / period
    manual_atr = [expected_tr[0]]
    for i in range(1, len(expected_tr)):
        manual_atr.append(alpha * expected_tr[i] + (1 - alpha) * manual_atr[-1])

    computed = atr(df, period=period)
    # First `period - 1` values are masked (min_periods=period)
    assert computed.iloc[: period - 1].isna().all()
    np.testing.assert_allclose(computed.iloc[period - 1:].values, manual_atr[period - 1:], rtol=1e-9)


def test_atr_is_causal_appending_future_bars_does_not_change_past_values():
    ohlcv = [(100 + i * 0.1, 100 + i * 0.1 + 1, 100 + i * 0.1 - 1, 100 + i * 0.1 + 0.3, 1000) for i in range(20)]
    df = make_bars("2025-03-03", ohlcv)
    atr_full = atr(df, period=5)

    df_prefix = df.iloc[:12].copy()
    atr_prefix = atr(df_prefix, period=5)

    np.testing.assert_allclose(atr_prefix.values, atr_full.iloc[:12].values, rtol=1e-9)


def test_vwap_resets_each_session():
    ohlcv_day1 = [(100, 101, 99, 100, 1000), (100, 102, 100, 101, 3000)]
    ohlcv_day2 = [(200, 201, 199, 200, 1000), (200, 203, 200, 202, 1000)]
    df = pd.concat([make_bars("2025-03-03", ohlcv_day1), make_bars("2025-03-04", ohlcv_day2)], ignore_index=True)

    vwap = session_vwap(df)
    # Day 2's VWAP should be near 200, uninfluenced by day 1's much lower prices.
    assert vwap.iloc[2] == 200.0
    assert 200 <= vwap.iloc[3] <= 203


def test_rolling_intraday_high_excludes_current_bar():
    ohlcv = [(100, 105, 99, 104, 1000), (104, 110, 103, 106, 1000), (106, 108, 105, 107, 1000)]
    df = make_bars("2025-03-03", ohlcv)
    ref_high = rolling_intraday_high(df, exclude_current=True)

    assert np.isnan(ref_high.iloc[0])  # no prior bar yet
    assert ref_high.iloc[1] == 105.0   # highest high before bar 1 is bar 0's high
    assert ref_high.iloc[2] == 110.0   # highest high before bar 2 is bar 1's high (110)
