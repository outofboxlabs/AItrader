"""Live social-media sentiment for a ticker via StockGeist
(https://www.stockgeist.ai) -- a real, established sentiment-data vendor
(operating since 2021, public docs and Python client on GitHub) that
aggregates social-media posts and computes message-volume/sentiment
metrics per ticker, rather than us scraping X/Reddit ourselves.

Honesty note: the exact endpoint, auth, and parameter names below come
from StockGeist's public GitHub client source (base URL, endpoint path,
`token`/`symbol`/`timeframe`/`start`/`end`/`filter` params all confirmed
from their client's source code) -- but the exact field names inside a
live response were NOT independently verified against a real account (no
key was available while building this). summarize_message_metrics() is
deliberately generic (works over whatever numeric fields actually show
up) so a field-name mismatch degrades to "fewer stats," not a crash. If
your StockGeist response looks off once you have a real key, the fix is
almost certainly just the metric names passed as `filter`.
"""

from __future__ import annotations

import numbers
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

import config

# window name -> (how far back, bucket size for the time series). Coarser
# buckets for longer windows keep the response (and therefore the prompt
# built from it) a reasonable size instead of asking for e.g. 5-minute
# bars across a whole year.
WINDOW_TO_RANGE = {
    "hour": (timedelta(hours=1), "5m"),
    "today": (timedelta(hours=24), "1h"),
    "week": (timedelta(days=7), "1h"),
    "month": (timedelta(days=30), "1d"),
    "year": (timedelta(days=365), "1d"),
}

DEFAULT_FILTER = ("total_count", "pos_index", "neg_index", "message_ratio")


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def get_message_metrics(
    ticker: str, window: str, api_key: str, now: Optional[datetime] = None
) -> Optional[dict]:
    """Raw message-metrics time series for `ticker` over `window`
    ("hour"|"today"|"week"|"month"|"year"). Returns None on any failure
    (bad ticker, bad key, network error, unexpected shape) -- the caller
    treats missing sentiment data as "can't be assessed" rather than an
    error, same convention as every other external data pull in this
    app."""
    if window not in WINDOW_TO_RANGE:
        return None
    lookback, timeframe = WINDOW_TO_RANGE[window]
    now = now or datetime.now(timezone.utc)
    start = now - lookback

    try:
        response = requests.get(
            f"{config.STOCKGEIST_BASE_URL}time-series/message-metrics",
            params={
                "token": api_key,
                "symbol": ticker.upper(),
                "timeframe": timeframe,
                "start": _iso(start),
                "end": _iso(now),
                "filter": ",".join(DEFAULT_FILTER),
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def summarize_message_metrics(raw: Optional[dict]) -> Optional[dict]:
    """Turns a raw (potentially long) time series into a compact summary
    an AI prompt can use without needing hundreds of data points: for
    every numeric field found across all entries, its average/min/max,
    plus a simple first-vs-last trend direction. Deliberately schema-
    agnostic -- it doesn't assume exact field names, since those aren't
    independently verified (see module docstring). Returns None if `raw`
    is empty/unusable."""
    if not raw:
        return None

    entries = None
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                entries = value
                break
    if not entries:
        return None

    fields: dict[str, list[float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                fields.setdefault(key, []).append(float(value))

    if not fields:
        return None

    summary = {"data_points": len(entries)}
    for key, values in fields.items():
        summary[key] = {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "first": values[0],
            "last": values[-1],
        }
    return summary
