"""Forex Factory economic calendar: high-impact scheduled news events
(rate decisions, CPI, NFP, GDP, etc.) that tend to move currency markets
sharply the moment they're released.

Forex Factory has no official public API or documented field reference
for this. This pulls their public JSON feed that powers their own
embeddable calendar widget -- unofficial and undocumented, but the
standard way the retail trading-bot community reads this data, since
scraping the HTML calendar page directly is fragile and against their
ToS. The field names below (title, country, impact, forecast, previous,
actual) were confirmed against a real pull -- they came through
correctly on the first try. The timestamp field did not: a first guess
of a "date" ISO string was wrong (every row rendered "n/a"). Later
research points to a "dateline" field carrying a Unix UTC timestamp
instead, which is what's implemented below, but this STILL hasn't been
confirmed against a real response (this session's network access can't
reach forexfactory.com or its CDN to verify directly) -- fetch_calendar_events
prints the first raw event to the terminal on every call specifically so
this can be checked/fixed quickly if it's still wrong.

This module only reads the calendar. It does not place or evaluate any
trade -- see app.py's Forex Calendar tab docstring for why.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests

DEFAULT_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

_SUFFIX_MULTIPLIERS = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _parse_numeric(value) -> Optional[float]:
    """Best-effort numeric parse of a calendar value cell, which commonly
    carries a %, comma thousand-separators, or a K/M/B/T magnitude suffix
    (e.g. "3.1%", "227K", "1,250"). Returns None rather than raising if it
    doesn't look numeric -- this is a display nicety, never load-bearing."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    multiplier = 1.0
    if s[-1].upper() in _SUFFIX_MULTIPLIERS:
        multiplier = _SUFFIX_MULTIPLIERS[s[-1].upper()]
        s = s[:-1]
    s = s.replace(",", "").replace("%", "").strip()
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _surprise_pct(actual, forecast) -> Optional[float]:
    """% deviation of the actual release from what was forecast -- the
    "surprise" that tends to drive the sharp move right after release.
    None if either side isn't cleanly numeric (text values like "Better"
    or an event with no forecast at all) or forecast is exactly zero."""
    a, f = _parse_numeric(actual), _parse_numeric(forecast)
    if a is None or f is None or f == 0:
        return None
    return (a - f) / abs(f) * 100.0


def _extract_iso_datetime(e: dict) -> Optional[str]:
    """Best-effort event timestamp as an ISO 8601 string. Prefers a
    "dateline" field (a Unix UTC timestamp, per third-party reports on
    this feed's shape) over a plain "date" string, since a first attempt
    using "date" alone came back as unparseable/missing on a real pull.
    Never raises -- falls back to whatever "date" holds (possibly None,
    rendered as "n/a" downstream) rather than blocking the rest of the
    row's real data (title/impact/forecast/etc, which came through fine)."""
    dateline = e.get("dateline")
    if dateline is not None:
        try:
            ts = float(dateline)
            if ts > 1e12:  # looks like milliseconds, not seconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            pass
    return e.get("date")


def fetch_calendar_events(feed_url: str = DEFAULT_FEED_URL, timeout: float = 15.0) -> list[dict]:
    """Pull and normalize this week's calendar from Forex Factory's feed.
    Raises on network/parse failure -- callers decide how to degrade
    (app.py falls back to the last cached copy in the db)."""
    response = requests.get(feed_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    raw_events = response.json()

    if raw_events:
        # This feed is unofficial and undocumented -- if any column ever
        # looks wrong again, this line in the terminal shows exactly what
        # Forex Factory actually sent, so the parser can be fixed against
        # real data instead of another guess.
        print(f"[forex_calendar] raw sample event (verify field names against this): {raw_events[0]}")

    events = []
    for e in raw_events:
        actual = e.get("actual")
        forecast = e.get("forecast")
        events.append(
            {
                "date": _extract_iso_datetime(e),
                "country": e.get("country"),
                "title": e.get("title"),
                "impact": e.get("impact"),
                "forecast": forecast,
                "previous": e.get("previous"),
                "actual": actual,
                "surprise_pct": _surprise_pct(actual, forecast),
            }
        )
    print(f"[forex_calendar] fetched {len(events)} event(s) from {feed_url}")
    return events
