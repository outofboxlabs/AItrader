"""Penny stock screen: liquid US stocks trading under a price cap.

Unlike growth_screener/nearlow_screener, no single quality filter is
imposed here -- many penny stocks have no analyst coverage at all, so
requiring one (like Near 52W Low's buy-ratio gate) would exclude most
of the universe. Instead every candidate that clears the liquidity/
market-cap floors is included, carrying three independent columns a
human can sort or judge by:
  - buy_ratio_pct / ratings_count -- analyst quality, when there is any.
  - pct_from_52w_high -- momentum (close to its own high = trending up).
  - volume -- how actively traded it actually is.

Two price thresholds ("5" and "1", config.PENNYSTOCK_THRESHOLDS) are run
as separate screens sharing the same pool/liquidity settings -- a pure
data screen, no AI/news call, no API key required, same as the other
screeners.
"""

from __future__ import annotations

import json
import numbers
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import yfinance as yf
from yfinance import EquityQuery


def _current_period_ratings(recommendations: dict) -> dict:
    """Same shape-parsing as nearlow_screener/growth_screener -- see
    those for why this is duplicated rather than shared."""
    result = {}
    for category in ("strongBuy", "buy", "hold", "sell", "strongSell"):
        col = recommendations.get(category)
        if isinstance(col, dict) and col:
            result[category] = next(iter(col.values()))
        elif isinstance(col, list) and col:
            result[category] = col[0]
    return result


def _buy_ratio_pct(ratings: dict) -> tuple[Optional[float], int]:
    total = sum(v for v in ratings.values() if isinstance(v, numbers.Real))
    if not total:
        return None, 0
    buy_like = float(ratings.get("strongBuy", 0)) + float(ratings.get("buy", 0))
    return buy_like / total * 100.0, int(total)


def _is_pre_revenue(t) -> Optional[bool]:
    """Same as nearlow_screener._is_pre_revenue -- see there for why this
    is duplicated rather than shared, why .info's totalRevenue is checked
    before the full income statement, and why None is treated the same
    as True by callers/the UI filter."""
    try:
        info = t.info or {}
    except Exception:
        info = {}
    revenue = info.get("totalRevenue")
    if isinstance(revenue, numbers.Real):
        return float(revenue) <= 0

    try:
        income = t.get_income_stmt(freq="yearly")
    except Exception:
        return None
    if income is None or income.empty:
        return None
    latest_col = sorted(income.columns, reverse=True)[0]
    for label in ("Total Revenue", "Total Revenues", "Operating Revenue"):
        if label not in income.index:
            continue
        try:
            revenue = float(income.loc[label, latest_col])
        except (TypeError, ValueError):
            continue
        if revenue == revenue:  # not NaN
            return revenue <= 0
    return None


def _enrich_candidate(q: dict) -> tuple[Optional[dict], str]:
    """Fetch one candidate's 52-week range, volume, and (if any) analyst
    data -- no hard filter beyond what the EquityQuery pool already
    applied, so a candidate is only excluded here on a genuine data
    failure. Never raises, so one bad ticker can't take down a parallel
    batch."""
    ticker = q.get("symbol")
    if not ticker:
        return None, "no_symbol"

    t = yf.Ticker(ticker)
    try:
        fast_info = t.fast_info
        year_high = fast_info.year_high
        year_low = fast_info.year_low
    except Exception as exc:
        return None, f"range_error:{type(exc).__name__}"

    current_price = q.get("regularMarketPrice")
    if not current_price or not year_low:
        return None, "missing_price_or_range"

    pct_from_52w_low = (current_price - year_low) / year_low * 100.0
    pct_from_52w_high = (current_price - year_high) / year_high * 100.0 if year_high else None

    try:
        recommendations = t.get_recommendations_summary(as_dict=True) or {}
        ratings = _current_period_ratings(recommendations)
    except Exception:
        ratings = {}
    buy_ratio_pct, ratings_count = _buy_ratio_pct(ratings)

    try:
        price_targets = t.get_analyst_price_targets() or {}
        mean_target = price_targets.get("mean")
    except Exception:
        mean_target = None
    target_upside_pct = (mean_target - current_price) / current_price * 100.0 if mean_target else None
    is_pre_revenue = _is_pre_revenue(t)

    candidate = {
        "ticker": ticker,
        "name": q.get("shortName") or q.get("longName"),
        "price": current_price,
        "year_low": year_low,
        "year_high": year_high,
        "pct_from_52w_low": pct_from_52w_low,
        "pct_from_52w_high": pct_from_52w_high,
        "volume": q.get("regularMarketVolume"),
        "target_mean": mean_target,
        "target_upside_pct": target_upside_pct,
        "analyst_ratings": ratings,
        "buy_ratio_pct": buy_ratio_pct,
        "ratings_count": ratings_count,
        "market_cap": q.get("marketCap"),
        "is_pre_revenue": is_pre_revenue,
    }
    # Same round-trip as the other screeners -- yfinance's dict
    # conversions carry numpy/pandas types that json.dumps chokes on
    # downstream (SQLite storage, jsonify, CSV export).
    return json.loads(json.dumps(candidate, default=str)), "included"


def find_pennystock_candidates(
    price_threshold: float,
    candidate_pool_size: int = 200,
    min_market_cap: float = 10_000_000,
    min_volume: int = 200_000,
    max_results: int = 100,
    max_workers: int = 20,
) -> list[dict]:
    """Screen a liquid US candidate pool priced under `price_threshold`,
    sorted by trading volume (most active first) -- there's no single
    "best" ordering here the way there is for Near 52W Low (proximity to
    the low) or Growth (upside), since penny-stock quality signals are
    genuinely mixed-bag; volume at least surfaces names that are actually
    being traded, not illiquid shells. Returns up to max_results.

    Enrichment runs on a thread pool for the same reason as the other
    screeners: each candidate is its own yfinance round trip."""
    query = EquityQuery(
        "AND",
        [
            EquityQuery("EQ", ["region", "us"]),
            EquityQuery("LT", ["intradayprice", price_threshold]),
            EquityQuery("GTE", ["intradaymarketcap", min_market_cap]),
            EquityQuery("GT", ["dayvolume", min_volume]),
        ],
    )
    response = yf.screen(query, sortField="dayvolume", sortAsc=False, size=candidate_pool_size)
    quotes = response.get("quotes", []) if response else []

    candidates = []
    reasons: Counter = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_enrich_candidate, q) for q in quotes]
        for future in as_completed(futures):
            result, reason = future.result()
            reasons[reason] += 1
            if result is not None:
                candidates.append(result)

    breakdown = ", ".join(f"{reason}={count}" for reason, count in reasons.most_common())
    print(f"[pennystock_screener] threshold={price_threshold} pool={len(quotes)} included={len(candidates)} -- {breakdown or 'no candidates in pool'}")

    candidates.sort(key=lambda c: c.get("volume") or 0, reverse=True)
    return candidates[:max_results]
