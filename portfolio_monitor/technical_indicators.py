"""Real, self-computed technical indicators from daily close prices --
SMA/EMA, RSI, and MACD -- rather than depending on a third-party "quant"
skill or library. These are well-known, standard formulas; implementing
them directly here (on top of pandas, already a dependency via yfinance)
keeps this fully testable with plain numbers and avoids taking on an
external technical-analysis library's own dependency/version churn.
"""

from __future__ import annotations

from typing import Optional


def _sma(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _ema_series(closes: list[float], period: int) -> list[Optional[float]]:
    """EMA at each index, seeded by the SMA of the first `period` values
    (the standard approach) -- None for indices before enough history
    exists to seed it."""
    if len(closes) < period:
        return [None] * len(closes)
    multiplier = 2.0 / (period + 1)
    result: list[Optional[float]] = [None] * (period - 1)
    seed = sum(closes[:period]) / period
    result.append(seed)
    prev = seed
    for price in closes[period:]:
        prev = (price - prev) * multiplier + prev
        result.append(prev)
    return result


def _rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Wilder's RSI: seed average gain/loss with a simple average of the
    first `period` changes, then smooth every change after that with
    Wilder's recursive formula. Returns None if there isn't enough
    history (a newly-listed stock, thin data, etc.)."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes: list[float]) -> Optional[dict]:
    """Standard 12/26/9 MACD: fast EMA minus slow EMA, a 9-period EMA of
    that line as the signal, and their difference as the histogram.
    Returns None if there isn't enough history for the signal line (needs
    ~35 points: 26 to seed the slow EMA, plus 9 more to seed the
    signal)."""
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = [f - s if f is not None and s is not None else None for f, s in zip(ema12, ema26)]
    macd_values = [v for v in macd_line if v is not None]
    if len(macd_values) < 9:
        return None
    signal_series = _ema_series(macd_values, 9)
    macd_latest = macd_values[-1]
    signal_latest = signal_series[-1]
    if signal_latest is None:
        return None
    return {
        "macd": macd_latest,
        "signal": signal_latest,
        "histogram": macd_latest - signal_latest,
    }


def compute_technical_snapshot(daily_history: list[dict]) -> dict:
    """Latest-value snapshot of a handful of standard indicators computed
    from daily closes (sorted oldest-first) -- not a full time series,
    since the AI agent reading this only needs "where things stand right
    now," not to re-derive the chart itself. Keys for indicators that
    don't have enough history yet are simply omitted rather than sent as
    null noise. Never raises -- returns {} on empty/insufficient input."""
    closes = [d["close"] for d in daily_history if d.get("close") is not None]
    if not closes:
        return {}

    snapshot: dict = {"latest_close": closes[-1]}

    for period in (20, 50, 200):
        sma = _sma(closes, period)
        if sma is not None:
            snapshot[f"sma_{period}"] = sma
            snapshot[f"price_vs_sma_{period}_pct"] = (closes[-1] - sma) / sma * 100.0

    rsi = _rsi(closes, 14)
    if rsi is not None:
        snapshot["rsi_14"] = rsi

    macd = _macd(closes)
    if macd is not None:
        snapshot["macd"] = macd["macd"]
        snapshot["macd_signal"] = macd["signal"]
        snapshot["macd_histogram"] = macd["histogram"]

    return snapshot
