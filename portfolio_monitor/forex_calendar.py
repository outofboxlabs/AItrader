"""Forex Factory economic calendar: high-impact scheduled news events
(rate decisions, CPI, NFP, GDP, etc.) that tend to move currency markets
sharply the moment they're released.

Forex Factory has no official public API or documented field reference
for this. This pulls their public JSON feed that powers their own
embeddable calendar widget -- unofficial and undocumented, but the
standard way the retail trading-bot community reads this data, since
scraping the HTML calendar page directly is fragile and against their
ToS. The field shape below (title, country, date, impact, forecast,
previous, actual) is now CONFIRMED against a real pull of the feed:
"date" is a plain ISO 8601 string with a UTC offset already baked in
(e.g. "2026-09-06T21:30:00-04:00"), not a separate timestamp field --
an earlier guess that it needed a Unix-timestamp fallback was chasing a
field that doesn't actually exist in this feed. (The real bug that made
every row show "n/a" turned out to be downstream in db.py, not here --
see get_forex_calendar_events.)

This module only reads the calendar. It does not place or evaluate any
trade -- see app.py's Forex Calendar tab docstring for why.
"""

from __future__ import annotations

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
                "date": e.get("date"),
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
