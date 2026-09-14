"""Live social-media sentiment for a ticker via StockTwits' public message
stream (https://stocktwits.com) -- a real, long-running trader social
network where every message is tagged Bullish/Bearish by the person who
wrote it, not something inferred by a model. Uses StockTwits' plain
public per-symbol stream endpoint, which needs no API key or account
registration at all -- unlike StockGeist (broken signup) and Reddit
(script-app registration that proved unreliable in practice), there's no
sign-up step here for a human to get stuck on.

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


def get_recent_messages(ticker: str, window: str, now: Optional[datetime] = None) -> Optional[list[dict]]:
    """Recent StockTwits messages mentioning `ticker`, most-recent-first,
    filtered to `window` ("hour"|"today"|"week"|"month"|"year"). The
    underlying endpoint only ever returns its most recent ~30 messages
    (no server-side date-range filtering) -- so a "past year" request on
    a high-volume ticker may only reflect the last day or two of actual
    chatter, not a literal year. That's an honest limitation of this
    free, no-signup endpoint, not a bug: it just means "here's everything
    recent that falls in this window," and the window rarely binds for
    a quiet ticker but can for a busy one. Returns None on any failure
    (bad ticker, network error, unexpected shape)."""
    cutoff_delta = WINDOW_TO_TIMEDELTA.get(window)
    if cutoff_delta is None:
        return None
    now = now or datetime.now(timezone.utc)
    cutoff = now - cutoff_delta

    response = None
    try:
        response = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "Accept": "application/json",
            },
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
    messages = []
    for m in raw_messages:
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
