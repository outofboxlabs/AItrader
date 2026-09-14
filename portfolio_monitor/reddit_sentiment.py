"""Live social-media sentiment for a ticker via Reddit's own official API
-- free for personal/non-commercial use (see
https://www.reddit.com/wiki/api), which this local single-user tool
qualifies as. Pulls recent posts mentioning the ticker from a curated set
of retail-finance subreddits and hands their titles/engagement straight
to the AI persona to read sentiment from -- same pattern as the News
agent -- rather than this module inventing its own bullish/bearish score
on top of real people's words.

Replaces an earlier StockGeist-based version: StockGeist's own signup
dashboard turned out to be unreliable in practice, whereas Reddit's API
is a well-established, directly-verifiable path.

Auth: OAuth2 "client credentials" grant (app-only, no Reddit user login
needed). One-time setup is registering a free "script" app at
https://www.reddit.com/prefs/apps to get a client ID + secret. Reddit
added a manual-approval step for new apps in late 2025, so there can be
a short wait after registering before the credentials actually work.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import requests

import config

FINANCE_SUBREDDITS = "wallstreetbets+stocks+investing+StockMarket+options"

# window name -> Reddit's own search time-filter value ("t" query param).
WINDOW_TO_REDDIT_T = {
    "hour": "hour",
    "today": "day",
    "week": "week",
    "month": "month",
    "year": "year",
}

_cached_token: Optional[str] = None
_cached_token_expiry: float = 0.0


def _get_access_token(client_id: str, client_secret: str) -> Optional[str]:
    """App-only OAuth2 token (client_credentials grant) -- no Reddit user
    login required, just the app's own client ID/secret. Cached in memory
    until shortly before it expires (Reddit tokens last ~1 hour) so this
    doesn't re-authenticate on every single call."""
    global _cached_token, _cached_token_expiry
    if _cached_token and time.time() < _cached_token_expiry:
        return _cached_token

    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": config.REDDIT_USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    token = payload.get("access_token")
    if not token:
        return None
    _cached_token = token
    _cached_token_expiry = time.time() + payload.get("expires_in", 3600) - 60  # refresh a bit early
    return token


def search_recent_posts(
    ticker: str, window: str, client_id: str, client_secret: str, limit: int = 15
) -> Optional[list[dict]]:
    """Recent posts mentioning `ticker` across a curated set of retail-
    finance subreddits, within `window` ("hour"|"today"|"week"|"month"|
    "year"), most recent first. Returns None on any failure (bad
    credentials, network error, unexpected shape) -- the caller treats
    missing sentiment data as "can't be assessed" rather than an error."""
    reddit_t = WINDOW_TO_REDDIT_T.get(window)
    if reddit_t is None:
        return None

    token = _get_access_token(client_id, client_secret)
    if not token:
        return None

    try:
        response = requests.get(
            f"https://oauth.reddit.com/r/{FINANCE_SUBREDDITS}/search",
            params={"q": ticker, "restrict_sr": "on", "sort": "new", "t": reddit_t, "limit": limit},
            headers={"Authorization": f"Bearer {token}", "User-Agent": config.REDDIT_USER_AGENT},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    children = (payload.get("data") or {}).get("children") or []
    posts = []
    for child in children:
        post = child.get("data") or {}
        created = post.get("created_utc")
        posts.append(
            {
                "title": post.get("title"),
                "subreddit": post.get("subreddit"),
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "created_at": datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else None,
                "permalink": f"https://reddit.com{post.get('permalink')}" if post.get("permalink") else None,
            }
        )
    return posts
