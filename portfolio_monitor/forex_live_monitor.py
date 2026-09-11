"""Watches Forex Factory's LIVE calendar page (not the JSON feed, which
we've confirmed never carries actuals) for the actual value of each
scheduled event, polling in the seconds after its release time. This
answers the real question behind any later "trade on the news" idea:
how long after the scheduled time does the actual print actually show
up, and by how much did it beat/miss the forecast.

This is a research/monitoring tool only -- it logs results to a CSV.
It does not place, size, or evaluate any trade, and never will run
unattended; see app.py's Forex Calendar tab docstring for the standing
reason why.

IMPORTANT: this scrapes forexfactory.com's live HTML page directly,
which is against their Terms of Service and can break without warning
if they change their markup. That tradeoff was made explicitly, at the
user's direction, after their own browser inspector confirmed the
`calendar__cell calendar__actual` cell structure used below. The
row/title/currency selectors are additionally grounded in a real
third-party scraper's reverse-engineered selectors (not just guessed),
but this has NOT been run against the live site from this session --
this session's network access is blocked from reaching forexfactory.com
at all. Every fetch prints enough of what it found (or didn't) to
recalibrate quickly against real output.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from . import forex_calendar

LIVE_CALENDAR_URL = "https://www.forexfactory.com/calendar"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _day_query_value(d: date) -> str:
    """Forex Factory's day-view URL param, e.g. "sep11.2026" (no leading
    zero on the day, per confirmed examples)."""
    return f"{d.strftime('%b').lower()}{d.day}.{d.year}"


def fetch_live_day_html(day: date, timeout: float = 15.0) -> str:
    """Fetch the live calendar page for one specific day. Raises on
    network/HTTP failure -- callers decide how to degrade."""
    url = f"{LIVE_CALENDAR_URL}?day={_day_query_value(day)}"
    response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    print(f"[forex_live_monitor] fetched {url} -> {response.status_code}, {len(response.text)} bytes")
    return response.text


def _normalize_direction(css_class: str) -> Optional[str]:
    """"better"/"worse" map directly; anything else (a class we haven't
    seen, e.g. a "same" or neutral state) is returned as None with the
    raw class preserved separately by the caller -- never guess at a
    meaning we haven't actually confirmed."""
    if css_class in ("better", "worse"):
        return css_class
    return None


def find_actual_for_event(html: str, title: str, country: str) -> Optional[dict]:
    """Find one event's actual value on a fetched live-calendar page, by
    matching the event title text (exact) and currency. Returns None if
    the event row can't be found, or if it's found but the actual cell
    is still empty (not yet released) -- both are legitimate, expected
    "no data yet" outcomes, not errors."""
    soup = BeautifulSoup(html, "html.parser")

    for title_span in soup.select("span.calendar__event-title"):
        if title_span.get_text(strip=True) != title:
            continue
        row = title_span.find_parent("tr")
        if row is None:
            continue

        currency_cell = row.select_one("td.calendar__currency")
        if currency_cell is None or currency_cell.get_text(strip=True) != country:
            continue

        actual_cell = row.select_one("td.calendar__actual")
        if actual_cell is None:
            return None
        actual_span = actual_cell.find("span")
        actual_text = actual_span.get_text(strip=True) if actual_span else actual_cell.get_text(strip=True)
        if not actual_text:
            return None

        raw_class = " ".join(actual_span.get("class", [])) if actual_span else ""
        return {"actual": actual_text, "direction": _normalize_direction(raw_class), "raw_class": raw_class}

    return None


@dataclass
class PollResult:
    title: str
    country: str
    impact: str
    scheduled_at: str
    forecast: str
    previous: str
    actual: Optional[str]
    direction: Optional[str]
    seconds_after_scheduled: Optional[float]
    timed_out: bool


def poll_for_actual(
    title: str,
    country: str,
    event_time: datetime,
    poll_interval_seconds: float = 3.0,
    max_wait_seconds: float = 300.0,
) -> tuple[Optional[dict], Optional[float], bool]:
    """Waits until event_time (if in the future), then polls the live
    page every poll_interval_seconds until the actual appears or
    max_wait_seconds have elapsed since event_time. Returns
    (result_or_None, seconds_after_scheduled_or_None, timed_out)."""
    now = datetime.now(timezone.utc)
    if event_time > now:
        time.sleep((event_time - now).total_seconds())

    deadline = event_time.timestamp() + max_wait_seconds
    while True:
        attempt_time = datetime.now(timezone.utc)
        try:
            html = fetch_live_day_html(event_time.date())
            result = find_actual_for_event(html, title, country)
        except Exception as exc:
            print(f"[forex_live_monitor] poll error for {country} {title!r}: {exc}")
            result = None

        if result is not None:
            elapsed = (attempt_time - event_time).total_seconds()
            print(f"[forex_live_monitor] {country} {title!r} -> actual={result['actual']!r} ({elapsed:.1f}s after scheduled time)")
            return result, elapsed, False

        if attempt_time.timestamp() >= deadline:
            print(f"[forex_live_monitor] {country} {title!r} -> gave up after {max_wait_seconds:.0f}s, no actual appeared")
            return None, None, True

        time.sleep(poll_interval_seconds)


_CSV_FIELDS = [
    "title", "country", "impact", "scheduled_at", "forecast", "previous",
    "actual", "direction", "seconds_after_scheduled", "timed_out",
]

# Each qualifying event polls on its own thread (see run_week_monitor), and
# several can finish around the same moment -- this lock keeps the
# exists-check + header-write + row-append atomic across threads, so two
# results landing together can't race on "does the file exist yet" and
# corrupt the CSV with an interleaved/duplicated header.
_csv_lock = threading.Lock()


def _append_result_row(csv_path: str, result: PollResult) -> None:
    with _csv_lock:
        file_exists = os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(result))
    print(f"[forex_live_monitor] logged {result.country} {result.title!r} to {csv_path}")


def _qualifying_events(events: list[dict], include_non_us: bool) -> list[dict]:
    """High-impact only (the events that actually move markets sharply);
    USD only unless include_non_us -- matches the "otherwise US only"
    default the option was asked for."""
    now = datetime.now(timezone.utc)
    result = []
    for e in events:
        if (e.get("impact") or "").strip().lower() != "high":
            continue
        if not include_non_us and (e.get("country") or "").upper() != "USD":
            continue
        if not e.get("date"):
            continue
        try:
            event_time = datetime.fromisoformat(e["date"])
        except ValueError:
            continue
        if event_time <= now:
            continue
        result.append({**e, "_event_time": event_time})
    return result


def run_week_monitor(
    include_non_us: bool = False,
    poll_interval_seconds: float = 3.0,
    max_wait_seconds: float = 300.0,
    output_dir: str = "exports/forex_live_monitor",
) -> str:
    """Schedules a background poller for every qualifying upcoming
    High-impact event still left in this week's calendar, each starting
    at its own scheduled time and running independently on its own
    thread -- so this call covers the whole remaining week, not just one
    event, and returns once every one of them has either found an
    actual or timed out. Blocks for as long as that takes (up to days,
    for a Monday run covering a Friday event) -- run it as a long-lived
    script, not a quick one-off.

    Returns the CSV path results are logged to."""
    events = _qualifying_events(forex_calendar.fetch_calendar_events(), include_non_us)
    if not events:
        print("[forex_live_monitor] no qualifying upcoming high-impact events this week -- nothing to monitor")
        return ""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"forex_live_monitor_{timestamp}.csv")
    print(f"[forex_live_monitor] monitoring {len(events)} event(s) this week -- logging to {csv_path}")

    def _worker(e: dict) -> None:
        result, elapsed, timed_out = poll_for_actual(
            e["title"], e["country"], e["_event_time"], poll_interval_seconds, max_wait_seconds
        )
        _append_result_row(
            csv_path,
            PollResult(
                title=e["title"],
                country=e["country"],
                impact=e.get("impact") or "",
                scheduled_at=e["date"],
                forecast=e.get("forecast") or "",
                previous=e.get("previous") or "",
                actual=result["actual"] if result else None,
                direction=result["direction"] if result else None,
                seconds_after_scheduled=elapsed,
                timed_out=timed_out,
            ),
        )

    threads = [threading.Thread(target=_worker, args=(e,), daemon=True) for e in events]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"[forex_live_monitor] done -- results in {csv_path}")
    return csv_path
