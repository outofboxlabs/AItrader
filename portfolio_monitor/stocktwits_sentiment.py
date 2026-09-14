"""Live social-media sentiment for a ticker via StockTwits' public message
stream (https://stocktwits.com) -- a real, long-running trader social
network where every message is tagged Bullish/Bearish by the person who
wrote it, not something inferred by a model. Uses StockTwits' plain
public per-symbol stream endpoint, which needs no API key or account
registration at all -- unlike StockGeist (broken signup) and Reddit
(script-app registration that proved unreliable in practice, and whose
own unauthenticated .json access Reddit blocked outright in 2026), there's
no sign-up step here for a human to get stuck on.

A single call only returns ~30 messages, so we page backward through
message IDs using the endpoint's `max` cursor (ask for messages with an
id below the oldest one we've already seen) to pull several pages of raw
posts per request instead of just one. This still isn't "every post
ever" -- it stops once a page comes back older than the requested window,
short of a full page, or empty -- but it's a real improvement in depth
over a single call, using the same no-signup endpoint.

Caveat: StockTwits' own developer program is reportedly paused for NEW
elevated/authenticated app registrations, but this basic public stream
endpoint doesn't require any registration and has worked unauthenticated
for years. If that ever changes, get_recent_messages() degrades to None
like every other external call in this app, rather than crashing.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

WINDOW_TO_TIMEDELTA = {
    "hour": timedelta(hours=1),
    "today": timedelta(hours=24),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}

PAGE_SIZE = 30
MAX_PAGES = 5  # up to ~150 raw messages per request, across up to 5 HTTP calls

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def _fetch_page(ticker: str, max_id: Optional[int]) -> Optional[list[dict]]:
    """One page of up to PAGE_SIZE raw StockTwits message dicts, or None on failure."""
    params = {"limit": PAGE_SIZE}
    if max_id is not None:
        params["max"] = max_id

    response = None
    try:
        response = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            params=params,
            timeout=15,
            headers=_HEADERS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status = getattr(response, "status_code", "no response")
        print(f"[social_sentiment] StockTwits request for {ticker} failed (status={status}): {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    raw_messages = payload.get("messages") or []
    if not raw_messages:
        print(f"[social_sentiment] StockTwits returned 0 messages for {ticker} (response keys: {sorted(payload.keys())})", file=sys.stderr)
    return raw_messages


def get_recent_messages(ticker: str, window: str, now: Optional[datetime] = None) -> Optional[list[dict]]:
    """Recent StockTwits messages mentioning `ticker`, most-recent-first,
    filtered to `window` ("hour"|"today"|"week"|"month"|"year"). Pages
    backward through up to MAX_PAGES calls (PAGE_SIZE messages each) via
    the endpoint's `max` cursor, stopping early once a page's oldest
    message falls outside the window or a page comes back short (meaning
    there's no more history). A "past year" request on a high-volume
    ticker may still only reflect the last several days of chatter if
    that's as far back as MAX_PAGES*PAGE_SIZE messages reaches -- an
    honest limitation of this free, no-signup endpoint, not a bug.
    Returns None if the very first page fails outright (bad ticker,
    network error, unexpected shape); a later page failing just stops
    pagination and returns whatever was already collected."""
    cutoff_delta = WINDOW_TO_TIMEDELTA.get(window)
    if cutoff_delta is None:
        return None
    now = now or datetime.now(timezone.utc)
    cutoff = now - cutoff_delta

    all_raw_messages: list[dict] = []
    max_id: Optional[int] = None
    for _ in range(MAX_PAGES):
        page = _fetch_page(ticker, max_id)
        if page is None:
            if not all_raw_messages:
                return None
            break
        if not page:
            break

        all_raw_messages.extend(page)

        ids = [m.get("id") for m in page if isinstance(m.get("id"), int)]
        if not ids:
            break
        max_id = min(ids) - 1

        oldest_created_at_str = (page[-1].get("created_at") or "").replace("Z", "+00:00")
        try:
            oldest_created_at = datetime.fromisoformat(oldest_created_at_str)
        except ValueError:
            oldest_created_at = None
        if oldest_created_at is not None and oldest_created_at < cutoff:
            break
        if len(page) < PAGE_SIZE:
            break

    messages = []
    seen_ids = set()
    for m in all_raw_messages:
        msg_id = m.get("id")
        if msg_id is not None:
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)

        created_at_str = (m.get("created_at") or "").replace("Z", "+00:00")
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except ValueError:
            continue
        if created_at < cutoff:
            continue
        sentiment = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        messages.append(
            {
                "body": m.get("body"),
                "username": (m.get("user") or {}).get("username"),
                "sentiment": sentiment,  # "Bullish" | "Bearish" | None (untagged)
                "created_at": created_at.isoformat(),
                "likes": (m.get("likes") or {}).get("total"),
            }
        )
    return messages
