"""Biggest price-drop screen: large-cap US stocks (market cap at or above
a selectable floor -- see config.BIGDROP_THRESHOLDS) ranked by their
1-day, 1-week, and 1-month price drop.

Unlike Near 52W Low, no single quality filter (analyst buy ratio,
proximity to the 52-week low) is imposed -- the point here is to surface
any big-cap hit, not just ones the Street still likes (see
pennystock_screener for the same reasoning). Every candidate instead
carries the full Near-52W-Low-style column set (52-week range, buy
ratio, target upside, pre-revenue/negative-P/E signals) so a human can
judge each drop on its own merits.

yfinance's screener has no native 1-week/1-month percent-change field
(only 1-day "percentchange" and "fiftytwowkpercentchange" -- see
EQUITY_SCREENER_FIELDS), so those two are computed here from ~3 months
of daily closes per candidate, anchored to the same live current price
used everywhere else in this app (current_price vs. a close N trading
days back), rather than sourced from any single opaque quote field.
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
    """Same shape-parsing as nearlow_screener/pennystock_screener -- see
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


def _is_pre_revenue(info: dict, t) -> Optional[bool]:
    """Same as nearlow_screener._is_pre_revenue -- see there for why this
    is duplicated rather than shared, why .info is fetched once by the
    caller and passed in (it's also where trailing_pe comes from), why
    its totalRevenue is checked before the full income statement, and why
    None is treated the same as True by callers/the UI filter."""
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


def _compute_trailing_pe(info: dict) -> Optional[float]:
    """Same as nearlow_screener._compute_trailing_pe -- see there for why
    this is duplicated: Yahoo's own "trailingPE" field is commonly
    missing/None specifically WHEN it would be negative, so this computes
    price / trailingEps ourselves whenever yfinance has both of those but
    not trailingPE directly."""
    trailing_pe = info.get("trailingPE")
    if isinstance(trailing_pe, numbers.Real):
        return float(trailing_pe)
    eps = info.get("trailingEps")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if isinstance(eps, numbers.Real) and eps != 0 and isinstance(price, numbers.Real):
        return float(price) / float(eps)
    return None


def _pct_change_from(closes: list[float], current_price: float, trading_days_back: int) -> Optional[float]:
    """% change from current_price (live) to the close `trading_days_back`
    entries before the end of `closes` (oldest-first, yfinance's own
    order) -- e.g. trading_days_back=1 is the most recent completed
    close ("since yesterday", i.e. the standard 1-day change), 6 is 5
    trading days before that (1 week), 22 is 21 trading days before that
    (1 month). Returns None if there isn't enough history yet (e.g. a
    very recently listed company)."""
    idx = len(closes) - trading_days_back
    if idx < 0:
        return None
    past_close = closes[idx]
    if not past_close:
        return None
    return (current_price - past_close) / past_close * 100.0


def _enrich_candidate(q: dict) -> tuple[Optional[dict], str]:
    """Fetch one candidate's price history, 52-week range, and analyst
    data. Never raises, so one bad ticker can't take down a parallel
    batch."""
    ticker = q.get("symbol")
    if not ticker:
        return None, "no_symbol"

    current_price = q.get("regularMarketPrice")
    if not current_price:
        return None, "missing_price"

    t = yf.Ticker(ticker)
    try:
        fast_info = t.fast_info
        year_high = fast_info.year_high
        year_low = fast_info.year_low
    except Exception as exc:
        return None, f"range_error:{type(exc).__name__}"

    try:
        hist = t.history(period="3mo", interval="1d")
    except Exception as exc:
        return None, f"history_error:{type(exc).__name__}"
    if hist is None or hist.empty:
        return None, "no_history"
    closes = [float(c) for c in hist["Close"].dropna().tolist()]
    if not closes:
        return None, "no_history"

    pct_change_1d = _pct_change_from(closes, current_price, 1)
    pct_change_1w = _pct_change_from(closes, current_price, 6)
    pct_change_1m = _pct_change_from(closes, current_price, 22)

    pct_from_52w_low = (current_price - year_low) / year_low * 100.0 if year_low else None
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

    try:
        info = t.info or {}
    except Exception:
        info = {}
    is_pre_revenue = _is_pre_revenue(info, t)
    trailing_pe = _compute_trailing_pe(info)

    candidate = {
        "ticker": ticker,
        "name": q.get("shortName") or q.get("longName"),
        "price": current_price,
        "year_low": year_low,
        "year_high": year_high,
        "pct_from_52w_low": pct_from_52w_low,
        "pct_from_52w_high": pct_from_52w_high,
        "pct_change_1d": pct_change_1d,
        "pct_change_1w": pct_change_1w,
        "pct_change_1m": pct_change_1m,
        "target_mean": mean_target,
        "target_upside_pct": target_upside_pct,
        "analyst_ratings": ratings,
        "buy_ratio_pct": buy_ratio_pct,
        "ratings_count": ratings_count,
        "market_cap": q.get("marketCap"),
        "is_pre_revenue": is_pre_revenue,
        "trailing_pe": trailing_pe,
    }
    # Same round-trip as the other screeners -- yfinance's dict
    # conversions carry numpy/pandas types that json.dumps chokes on
    # downstream (SQLite storage, jsonify, CSV export).
    return json.loads(json.dumps(candidate, default=str)), "included"


def find_bigdrop_candidates(
    min_market_cap: float,
    candidate_pool_size: int = 150,
    min_volume: int = 100_000,
    max_results: int = 100,
    max_workers: int = 20,
) -> list[dict]:
    """Screen a liquid US candidate pool at or above `min_market_cap`,
    sorted by today's % change ascending (biggest 1-day losers first,
    the closest native Yahoo sort available) to bias the pool toward
    genuinely-dropping names, then enrich each with 52-week range,
    analyst data, and the computed 1-day/1-week/1-month drop. Returns up
    to max_results sorted by 1-day drop (most negative first) -- the UI
    itself can re-sort by 1-week/1-month instead.

    Enrichment runs on a thread pool for the same reason as the other
    screeners: each candidate needs its own yfinance round trip (here,
    a history() call as well as fast_info/analyst data)."""
    query = EquityQuery(
        "AND",
        [
            EquityQuery("EQ", ["region", "us"]),
            EquityQuery("GTE", ["intradaymarketcap", min_market_cap]),
            EquityQuery("GT", ["dayvolume", min_volume]),
        ],
    )
    response = yf.screen(query, sortField="percentchange", sortAsc=True, size=candidate_pool_size)
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
    print(f"[bigdrop_screener] min_market_cap={min_market_cap} pool={len(quotes)} included={len(candidates)} -- {breakdown or 'no candidates in pool'}")

    candidates.sort(key=lambda c: c.get("pct_change_1d") if c.get("pct_change_1d") is not None else 0.0)
    return candidates[:max_results]
