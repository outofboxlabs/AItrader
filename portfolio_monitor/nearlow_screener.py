"""Near-52-week-low screen: liquid US stocks trading close to their 52-week
low that analysts still rate favorably -- i.e. "beaten down, but not for a
reason the Street has given up on."

Two independent conditions are ANDed together:
  1. Price is within `max_pct_from_low`% of the 52-week low.
  2. At least `min_ratings_count` analyst ratings exist, and at least
     `min_buy_ratio_pct`% of them are "buy" or "strong buy".

Both thresholds are configurable in config.py. As with growth_screener,
this is a pure data screen -- no AI/news call, no API key required.
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
    """Same shape-parsing as growth_screener._current_period_ratings --
    duplicated locally rather than shared, since it's a small, self-
    contained piece of pandas .to_dict()-shape parsing and this module
    otherwise has no dependency on growth_screener."""
    result = {}
    for category in ("strongBuy", "buy", "hold", "sell", "strongSell"):
        col = recommendations.get(category)
        if isinstance(col, dict) and col:
            result[category] = next(iter(col.values()))
        elif isinstance(col, list) and col:
            result[category] = col[0]
    return result


def _buy_ratio_pct(ratings: dict) -> tuple[Optional[float], int]:
    """% of all analyst ratings that are "buy" or "strong buy", plus the
    total ratings count (needed separately since a 100% ratio off of one
    rating isn't meaningful). Uses numbers.Real, not isinstance(v, (int,
    float)) -- see growth_screener for why: real yfinance data is numpy-
    typed and would otherwise be silently excluded."""
    total = sum(v for v in ratings.values() if isinstance(v, numbers.Real))
    if not total:
        return None, 0
    buy_like = float(ratings.get("strongBuy", 0)) + float(ratings.get("buy", 0))
    return buy_like / total * 100.0, int(total)


def _enrich_candidate(
    q: dict, max_pct_from_low: float, min_buy_ratio_pct: float, min_ratings_count: int
) -> tuple[Optional[dict], str]:
    """Fetch one candidate's 52-week range and analyst data, and apply both
    filters. Returns (None, reason) if excluded, or (candidate, "included")
    -- never raises, so one bad ticker can't take down a parallel batch."""
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
    if pct_from_52w_low > max_pct_from_low:
        return None, "too_far_from_low"

    try:
        recommendations = t.get_recommendations_summary(as_dict=True) or {}
    except Exception as exc:
        return None, f"recommendations_error:{type(exc).__name__}"

    ratings = _current_period_ratings(recommendations)
    buy_ratio_pct, ratings_count = _buy_ratio_pct(ratings)
    if buy_ratio_pct is None or ratings_count < min_ratings_count:
        return None, "not_enough_ratings"
    if buy_ratio_pct < min_buy_ratio_pct:
        return None, "ratings_not_strong_enough"

    try:
        price_targets = t.get_analyst_price_targets() or {}
        mean_target = price_targets.get("mean")
    except Exception:
        mean_target = None
    target_upside_pct = (mean_target - current_price) / current_price * 100.0 if mean_target else None

    pct_from_52w_high = (current_price - year_high) / year_high * 100.0 if year_high else None

    candidate = {
        "ticker": ticker,
        "name": q.get("shortName") or q.get("longName"),
        "price": current_price,
        "year_low": year_low,
        "year_high": year_high,
        "pct_from_52w_low": pct_from_52w_low,
        "pct_from_52w_high": pct_from_52w_high,
        "target_mean": mean_target,
        "target_upside_pct": target_upside_pct,
        "analyst_ratings": ratings,
        "buy_ratio_pct": buy_ratio_pct,
        "market_cap": q.get("marketCap"),
    }
    # See growth_screener._enrich_candidate for why this round-trip is
    # required: yfinance's dict conversions carry numpy/pandas types that
    # json.dumps chokes on downstream (SQLite storage, jsonify, CSV export).
    return json.loads(json.dumps(candidate, default=str)), "included"


def find_nearlow_candidates(
    candidate_pool_size: int = 150,
    min_market_cap: float = 300_000_000,
    min_price: float = 5.0,
    min_volume: int = 100_000,
    max_pct_from_low: float = 15.0,
    min_buy_ratio_pct: float = 60.0,
    min_ratings_count: int = 3,
    max_results: int = 100,
    max_workers: int = 20,
) -> list[dict]:
    """Screen a liquid US candidate pool (sorted by trailing 52-week %
    change ascending, i.e. the most beaten-down names first), then filter
    to those within max_pct_from_low% of their 52-week low AND with at
    least min_ratings_count analyst ratings of which >= min_buy_ratio_pct%
    are buy/strong-buy. Returns up to max_results, sorted by proximity to
    the 52-week low (closest first).

    Enrichment runs on a thread pool for the same reason as
    growth_screener: each candidate is its own yfinance round trip."""
    query = EquityQuery(
        "AND",
        [
            EquityQuery("EQ", ["region", "us"]),
            EquityQuery("GTE", ["intradaymarketcap", min_market_cap]),
            EquityQuery("GTE", ["intradayprice", min_price]),
            EquityQuery("GT", ["dayvolume", min_volume]),
        ],
    )
    response = yf.screen(query, sortField="fiftytwowkpercentchange", sortAsc=True, size=candidate_pool_size)
    quotes = response.get("quotes", []) if response else []

    candidates = []
    reasons: Counter = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_enrich_candidate, q, max_pct_from_low, min_buy_ratio_pct, min_ratings_count)
            for q in quotes
        ]
        for future in as_completed(futures):
            result, reason = future.result()
            reasons[reason] += 1
            if result is not None:
                candidates.append(result)

    breakdown = ", ".join(f"{reason}={count}" for reason, count in reasons.most_common())
    print(f"[nearlow_screener] screener_pool={len(quotes)} included={len(candidates)} -- {breakdown or 'no candidates in pool'}")

    candidates.sort(key=lambda c: c["pct_from_52w_low"])
    return candidates[:max_results]
