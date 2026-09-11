"""Forex Factory economic calendar: high-impact scheduled news events
(rate decisions, CPI, NFP, GDP, etc.) that tend to move currency markets
sharply the moment they're released.

Forex Factory has no official public API or documented field reference
for this. This pulls their public JSON feed that powers their own
embeddable calendar widget -- unofficial and undocumented, but the
standard way the retail trading-bot community reads this data, since
scraping the HTML calendar page directly is fragile and against their
ToS. The field shape is now CONFIRMED against a real pull of the full
week: "date" is a plain ISO 8601 string with a UTC offset already baked
in (e.g. "2026-09-06T21:30:00-04:00"). (An earlier guess that "date"
needed a Unix-timestamp fallback was chasing a field that doesn't
actually exist in this feed -- the real "n/a" bug turned out to be
downstream in db.py; see get_forex_calendar_events.)

IMPORTANT confirmed limitation: this feed carries "title", "country",
"date", "impact", "forecast", and "previous" ONLY -- there is no
"actual" field anywhere in it, confirmed by checking every event across
a full week including several hours/days past their release time. It is
a static forecast-only snapshot generated once for the week, not a live
feed -- unlike Forex Factory's own website, which does show actuals as
they release. _surprise_pct() and the "actual"/"surprise_pct" output
fields are kept for if a future response ever does carry one (or a
different feed is swapped in), but expect them to always be None/"n/a"
against this feed as it currently behaves.

This module only reads the calendar. It does not place or evaluate any
trade -- see app.py's Forex Calendar tab docstring for why.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import requests

from . import ai_client

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


PORTFOLIO_IMPACT_SYSTEM_PROMPT = """You are a market analyst embedded in a portfolio \
monitoring tool. You are given one scheduled or just-released economic calendar \
event (a rate decision, CPI, NFP, GDP, etc.) and a list of the user's current \
stock/option positions. Your job is ONLY to inform, never to advise:

- In 2-4 sentences, explain the plausible mechanism by which this specific event \
could affect the specific positions given -- reference the actual tickers/asset \
types provided, not generic commentary that could apply to any portfolio.
- If the event has already released (an actual value is given), factor in \
whether it beat, missed, or matched the forecast, and by how much.
- List which of the given tickers, if any, you consider most relevant to this \
event.
- If you genuinely see no plausible connection between this event and the given \
positions, say so plainly rather than forcing a connection.

You must NEVER say to buy, sell, hold, or otherwise act on any position. You are \
laying out a plausible mechanism for a human to weigh, not telling them what to \
do.

Respond with ONLY a JSON object, no other text, matching exactly this shape:
{"impact_summary": "...", "affected_tickers": ["...", "..."], \
"disclaimer": "This is not investment advice."}
"""


def build_portfolio_impact_user_message(event: dict, positions: list[dict]) -> str:
    lines = [
        f"Event: {event.get('title')} ({event.get('country')})",
        f"Scheduled: {event.get('date')}",
        f"Impact level: {event.get('impact')}",
        f"Previous: {event.get('previous') or 'n/a'}, Forecast: {event.get('forecast') or 'n/a'}, "
        f"Actual: {event.get('actual') or 'not yet released'}",
    ]
    if event.get("surprise_pct") is not None:
        lines.append(f"Surprise vs forecast: {event['surprise_pct']:+.1f}%")
    lines.append("")
    lines.append("Current positions:")
    if positions:
        for p in positions:
            ticker = p.get("ticker")
            if p.get("asset_type") == "option":
                lines.append(
                    f"- {ticker} {p.get('option_type')} option, strike {p.get('strike')}, expiry {p.get('expiry')}"
                )
            else:
                lines.append(f"- {ticker} shares, {p.get('contracts')} shares")
    else:
        lines.append("(no positions saved)")
    return "\n".join(lines)


def _parse_portfolio_impact_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            return {
                "impact_summary": text.strip()[:500] or None,
                "affected_tickers": [],
                "disclaimer": "This is not investment advice.",
                "parse_error": True,
            }
    parsed.setdefault("disclaimer", "This is not investment advice.")
    parsed.setdefault("affected_tickers", [])
    parsed.setdefault("parse_error", False)
    return parsed


def analyze_portfolio_impact(
    event: dict, positions: list[dict], provider: str, model: str, api_key: Optional[str] = None
) -> dict:
    """How one calendar event could plausibly affect the given positions,
    via the configured AI provider. Raises on API failure -- the caller
    decides how to degrade. On-demand only (one event at a time, only
    when a human asks) -- never runs automatically across a whole
    month's events, both to keep this affordable and because that's not
    what was asked for."""
    user_message = build_portfolio_impact_user_message(event, positions)
    text = ai_client.call_provider(
        provider, PORTFOLIO_IMPACT_SYSTEM_PROMPT, user_message, model, api_key=api_key, max_tokens=500
    )
    return _parse_portfolio_impact_json(text)
