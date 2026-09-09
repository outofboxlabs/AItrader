"""Deterministic macro gate: blends VIX level/percentile, VIX term
structure, market breadth, and credit spread into a single 0-100 score
describing the general environment the book sits in. Same data in, same
score out -- this is context, not a trade signal.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def percentile_rank(current: float, history: list[float]) -> float:
    """% of `history` at or below `current`. 100 = current is the highest
    value in the window; 0 = current is at/under the lowest."""
    if not history:
        return 50.0
    below_or_eq = sum(1 for v in history if v <= current)
    return below_or_eq / len(history) * 100.0


def score_vix_level(vix_closes: list[float]) -> dict:
    """Higher score = calmer (current VIX low relative to its own trailing history)."""
    current = vix_closes[-1]
    percentile = percentile_rank(current, vix_closes)
    return {"vix": current, "vix_percentile": percentile, "score": 100.0 - percentile}


def score_term_structure(vix: float, vix3m: float, calm_ratio: float, stress_ratio: float) -> dict:
    """VIX3M > VIX (contango) is the calm/normal state; VIX > VIX3M
    (backwardation) signals near-term stress."""
    if not vix3m:
        return {"vix": vix, "vix3m": vix3m, "ratio": None, "score": 50.0}
    ratio = vix / vix3m
    if stress_ratio == calm_ratio:
        score = 50.0
    else:
        raw = (stress_ratio - ratio) / (stress_ratio - calm_ratio) * 100.0
        score = max(0.0, min(100.0, raw))
    return {"vix": vix, "vix3m": vix3m, "ratio": ratio, "score": score}


def score_breadth(above_count: int, total_count: int) -> dict:
    """% of the basket trading above its 200-day moving average -- higher
    is healthier/calmer."""
    pct = (above_count / total_count * 100.0) if total_count else 50.0
    return {"pct_above_200dma": pct, "constituents": total_count, "score": pct}


def score_credit_spread(ratio_history: list[float]) -> dict:
    """HYG/TLT ratio: high yield outperforming treasuries is risk-on/calm.
    Scored by the ratio's own percentile so it self-calibrates over time."""
    current = ratio_history[-1]
    percentile = percentile_rank(current, ratio_history)
    return {"credit_ratio": current, "credit_percentile": percentile, "score": percentile}


def blend_macro_score(components: dict[str, float], weights: dict[str, float]) -> dict:
    """Weighted blend of the four component scores. Weights are normalized
    to sum to 1.0 (so relative magnitudes are what matter)."""
    total_weight = sum(weights.values())
    normalized = {k: (w / total_weight if total_weight else 0.0) for k, w in weights.items()}
    score = sum(components[k] * normalized[k] for k in normalized if k in components)
    return {"score": max(0.0, min(100.0, score)), "weights": normalized}


# ---- Live data pulls (network) ----


def _closes(ticker: str, period: str = "2y") -> list[float]:
    hist = yf.Ticker(ticker).history(period=period)
    return [float(c) for c in hist["Close"].dropna().tolist()]


def fetch_vix_history(lookback_days: int) -> list[float]:
    closes = _closes("^VIX")
    return closes[-lookback_days:] if len(closes) > lookback_days else closes


def fetch_vix_and_vix3m() -> tuple[float, float]:
    vix = _closes("^VIX", period="5d")[-1]
    vix3m = _closes("^VIX3M", period="5d")[-1]
    return vix, vix3m


def fetch_breadth(tickers: list[str], sma_window: int = 200) -> tuple[int, int]:
    above, total = 0, 0
    for ticker in tickers:
        try:
            closes = _closes(ticker)
            if len(closes) < sma_window:
                continue
            sma = sum(closes[-sma_window:]) / sma_window
            if closes[-1] > sma:
                above += 1
            total += 1
        except Exception:
            continue
    return above, total


def fetch_credit_ratio_history(lookback_days: int) -> list[float]:
    hyg = yf.Ticker("HYG").history(period="2y")["Close"]
    tlt = yf.Ticker("TLT").history(period="2y")["Close"]
    aligned = pd.concat([hyg, tlt], axis=1, keys=["HYG", "TLT"]).dropna()
    ratio = (aligned["HYG"] / aligned["TLT"]).tolist()
    return ratio[-lookback_days:] if len(ratio) > lookback_days else ratio


def compute_macro_gate(
    weights: dict[str, float],
    lookback_days: int,
    calm_ratio: float,
    stress_ratio: float,
    breadth_tickers: list[str],
) -> dict:
    """Pull fresh inputs and compute today's macro gate score end to end."""
    vix_history = fetch_vix_history(lookback_days)
    vix, vix3m = fetch_vix_and_vix3m()
    above, total = fetch_breadth(breadth_tickers)
    credit_history = fetch_credit_ratio_history(lookback_days)

    vix_level = score_vix_level(vix_history)
    term_structure = score_term_structure(vix, vix3m, calm_ratio, stress_ratio)
    breadth = score_breadth(above, total)
    credit_spread = score_credit_spread(credit_history)

    components = {
        "vix_level": vix_level["score"],
        "term_structure": term_structure["score"],
        "breadth": breadth["score"],
        "credit_spread": credit_spread["score"],
    }
    blended = blend_macro_score(components, weights)

    return {
        "score": blended["score"],
        "weights": blended["weights"],
        "vix_level": vix_level,
        "term_structure": term_structure,
        "breadth": breadth,
        "credit_spread": credit_spread,
    }
