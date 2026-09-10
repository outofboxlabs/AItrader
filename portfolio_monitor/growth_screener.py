"""Growth-candidate screen: liquid US stocks where the analyst consensus
price target implies substantial upside, alongside the analyst rating
breakdown (including what fraction are "strong buy") and how far each
sits from its 52-week high/low.

The rating breakdown is shown, not filtered on -- a strict "majority
strong buy" requirement turned out to exclude most otherwise-qualifying
names (see _strong_buy_ratio_pct), so it's surfaced as a ratio for a
human to judge instead of a hard gate.

Honesty note baked into the design: no data source predicts "50% growth
in the next month" -- that's not a published metric anywhere. The closest
real, checkable proxy is the analyst consensus price target, which is
conventionally a ~12-month view, not a 1-month one. This module is named
and labeled around that reality (target_upside_pct) rather than
pretending to a 1-month forecast that doesn't exist.
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
    """`Ticker.get_recommendations_summary(as_dict=True)` shape is
    {"period": {0: "0m", 1: "-1m", ...}, "strongBuy": {0: N, ...}, ...}
    (pandas .to_dict() over columns [period, strongBuy, buy, hold, sell,
    strongSell]). Returns the most recent ("0m") row as {category: count}."""
    result = {}
    for category in ("strongBuy", "buy", "hold", "sell", "strongSell"):
        col = recommendations.get(category)
        if isinstance(col, dict) and col:
            result[category] = next(iter(col.values()))
        elif isinstance(col, list) and col:
            result[category] = col[0]
    return result


def _strong_buy_ratio_pct(ratings: dict) -> Optional[float]:
    """% of all analyst ratings that are "strong buy". None if no ratings
    data at all (shown as such rather than treated as 0%).

    Uses numbers.Real rather than isinstance(v, (int, float)) -- real
    yfinance data comes from pandas .to_dict(), which yields numpy.int64/
    float64 values that are NOT instances of Python's builtin int/float,
    so the naive check would silently treat every real count as invalid
    and always return None."""
    total = sum(v for v in ratings.values() if isinstance(v, numbers.Real))
    if not total:
        return None
    return float(ratings.get("strongBuy", 0)) / total * 100.0


def _enrich_candidate(q: dict, target_upside_threshold: float) -> tuple[Optional[dict], str]:
    """Fetch one candidate's analyst data and 52-week range, and apply the
    upside filter. Returns (None, reason) if excluded, or (candidate,
    "included") -- never raises, so one bad ticker can't take down a
    parallel batch of these. The reason string is only for the diagnostic
    summary in find_growth_candidates; callers that don't care can ignore
    it. The analyst rating breakdown (including strong_buy_ratio_pct) is
    included on every candidate for a human to judge -- it's not filtered
    on, since a strict "majority strong buy" requirement excluded most
    otherwise-qualifying names in practice."""
    ticker = q.get("symbol")
    if not ticker:
        return None, "no_symbol"

    t = yf.Ticker(ticker)
    try:
        price_targets = t.get_analyst_price_targets() or {}
    except Exception as exc:
        return None, f"price_targets_error:{type(exc).__name__}"
    try:
        recommendations = t.get_recommendations_summary(as_dict=True) or {}
    except Exception as exc:
        return None, f"recommendations_error:{type(exc).__name__}"

    current_price = price_targets.get("current") or q.get("regularMarketPrice")
    mean_target = price_targets.get("mean")
    if not current_price or not mean_target:
        return None, "missing_price_or_target"

    target_upside_pct = (mean_target - current_price) / current_price * 100.0
    if target_upside_pct < target_upside_threshold:
        return None, "below_upside_threshold"

    ratings = _current_period_ratings(recommendations)
    strong_buy_ratio_pct = _strong_buy_ratio_pct(ratings)

    try:
        fast_info = t.fast_info
        year_high = fast_info.year_high
        year_low = fast_info.year_low
    except Exception:
        year_high = None
        year_low = None

    pct_from_52w_high = (current_price - year_high) / year_high * 100.0 if year_high else None
    pct_from_52w_low = (current_price - year_low) / year_low * 100.0 if year_low else None

    candidate = {
        "ticker": ticker,
        "name": q.get("shortName") or q.get("longName"),
        "price": current_price,
        "target_mean": mean_target,
        "target_upside_pct": target_upside_pct,
        "analyst_ratings": ratings,
        "strong_buy_ratio_pct": strong_buy_ratio_pct,
        "pct_from_52w_high": pct_from_52w_high,
        "pct_from_52w_low": pct_from_52w_low,
        "market_cap": q.get("marketCap"),
    }
    # yfinance's dict conversions can carry pandas/numpy types (int64,
    # float64, Timestamp, ...) that json.dumps chokes on downstream (SQLite
    # storage, Flask's jsonify, the CSV writer) -- round-trip through json
    # with a str fallback now so everything past this point is plain JSON
    # types. This is the same fix already applied in movers.py's
    # get_analyst_snapshot; it was missing here, which is what caused a
    # scan that found real candidates to still crash with a 502 before
    # ever reaching storage or the CSV export.
    return json.loads(json.dumps(candidate, default=str)), "included"


def find_growth_candidates(
    candidate_pool_size: int = 60,
    min_market_cap: float = 300_000_000,
    min_price: float = 5.0,
    min_volume: int = 100_000,
    target_upside_threshold: float = 50.0,
    max_results: int = 50,
    max_workers: int = 20,
) -> list[dict]:
    """Screen a liquid US candidate pool (sorted by trailing 52-week % change
    as a momentum proxy), then filter to names where the analyst consensus
    price target implies >= target_upside_threshold% upside. The analyst
    rating breakdown (strong_buy_ratio_pct and the raw counts) is included
    on every result for a human to judge, not filtered on -- returns up to
    max_results, sorted by upside descending.

    Per-candidate enrichment (analyst data, 52-week range) is I/O-bound --
    each one is its own yfinance network round trip -- so it runs on a
    thread pool rather than sequentially; candidate_pool_size=60 with
    max_workers=20 keeps a full scan to roughly the time of 3 sequential
    lookups instead of 60."""
    query = EquityQuery(
        "AND",
        [
            EquityQuery("EQ", ["region", "us"]),
            EquityQuery("GTE", ["intradaymarketcap", min_market_cap]),
            EquityQuery("GTE", ["intradayprice", min_price]),
            EquityQuery("GT", ["dayvolume", min_volume]),
        ],
    )
    response = yf.screen(query, sortField="fiftytwowkpercentchange", sortAsc=False, size=candidate_pool_size)
    quotes = response.get("quotes", []) if response else []

    candidates = []
    reasons: Counter = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_enrich_candidate, q, target_upside_threshold) for q in quotes]
        for future in as_completed(futures):
            result, reason = future.result()
            reasons[reason] += 1
            if result is not None:
                candidates.append(result)

    breakdown = ", ".join(f"{reason}={count}" for reason, count in reasons.most_common())
    print(f"[growth_screener] screener_pool={len(quotes)} included={len(candidates)} -- {breakdown or 'no candidates in pool'}")

    candidates.sort(key=lambda c: c["target_upside_pct"], reverse=True)
    return candidates[:max_results]
