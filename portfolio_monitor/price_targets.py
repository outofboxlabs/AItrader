"""Historical per-analyst price-target announcements via Financial
Modeling Prep (FMP) -- unlike yfinance's free upgrade/downgrade history
(grade changes only, no dollar figures), FMP's price-target endpoint
gives the actual target price, the stock's price when it was posted, and
the analyst/firm -- letting the Price Target Analyst persona (and,
later, the price chart) show what price was actually called for and
when, not just that a rating changed.

Requires an FMP API key (Settings tab). FMP's free "Basic" plan may or
may not include this specific endpoint -- that's unconfirmed as of
writing (FMP's own docs weren't reachable to check directly). A 401/402/
403 here just means the key's plan doesn't cover it; like every other
external call in this app, get_price_target_history() degrades to None
rather than crashing, and logs the real status code so that's easy to
tell apart from a network failure or an empty result.

Wall Street price targets are conventionally a 12-month forward call
unless a firm's own note says otherwise (which this API doesn't expose)
-- so `target_date` for each record is just `published_date + 365 days`,
labeled as a convention-based estimate, not asserted as a fact the firm
stated.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Optional

import requests

BASE_URL = "https://financialmodelingprep.com/stable/price-target-news"


def get_price_target_history(ticker: str, api_key: str, limit: int = 50) -> Optional[list[dict]]:
    """Recent individual analyst price-target announcements for `ticker`,
    most-recent-first, each annotated with a `target_date` = published_date
    + 365 days (the conventional 12-month horizon). Returns None on any
    failure (bad ticker, network error, missing/invalid/under-plan key,
    unexpected response shape) -- never raises."""
    if not api_key:
        return None

    response = None
    try:
        response = requests.get(
            BASE_URL,
            params={"symbol": ticker, "limit": limit, "apikey": api_key},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status = getattr(response, "status_code", "no response")
        body = (getattr(response, "text", "") or "")[:300]
        print(f"[price_targets] FMP request for {ticker} failed (status={status}): {type(exc).__name__}: {exc} -- body: {body}", file=sys.stderr)
        return None

    if not isinstance(payload, list):
        print(f"[price_targets] FMP returned unexpected (non-list) shape for {ticker}: {str(payload)[:300]}", file=sys.stderr)
        return None
    if not payload:
        print(f"[price_targets] FMP returned 0 price targets for {ticker}", file=sys.stderr)

    targets = []
    for item in payload:
        published_date_str = (item.get("publishedDate") or "").replace("Z", "+00:00")
        try:
            published_date = datetime.fromisoformat(published_date_str)
        except ValueError:
            continue
        price_target = item.get("priceTarget")
        if price_target is None:
            continue
        price_when_posted = item.get("priceWhenPosted")
        target_date = published_date + timedelta(days=365)
        implied_pct_change = None
        if price_when_posted:
            implied_pct_change = (price_target - price_when_posted) / price_when_posted * 100.0
        targets.append(
            {
                "published_date": published_date.date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "analyst_company": item.get("analystCompany"),
                "analyst_name": item.get("analystName"),
                "price_target": price_target,
                "price_when_posted": price_when_posted,
                "implied_pct_change": implied_pct_change,
                "news_title": item.get("newsTitle"),
                "news_url": item.get("newsURL"),
            }
        )
    return targets
