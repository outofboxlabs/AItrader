#!/usr/bin/env python3
"""Local dashboard for the portfolio monitor.

A multi-tab local web app: run the portfolio analysis, review today's big
market movers with an AI rebound read, edit positions.json (by hand or
via screenshot), and manage which AI provider/API key is used.
Everything runs on your own machine -- the only network calls are to
yfinance and whichever AI provider you've configured.

Usage:
    python app.py
Opens automatically at http://127.0.0.1:5050
"""

from __future__ import annotations

import calendar
import json
import os
import shutil
import threading
import traceback
import webbrowser
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from flask import Flask, jsonify, render_template_string, request

import config
from portfolio_monitor import credentials, data, exports, forex_calendar, forex_live_monitor, growth_screener, movers, nearlow_screener, news, pipeline, scheduler, vision
from portfolio_monitor import db as db_mod
from portfolio_monitor.models import Position

app = Flask(__name__)

PROVIDERS = ["anthropic", "openai", "gemini"]
NEWS_DEFAULT_MODEL = {
    "anthropic": config.ANTHROPIC_NEWS_MODEL,
    "openai": config.OPENAI_NEWS_MODEL,
    "gemini": config.GEMINI_NEWS_MODEL,
}
VISION_DEFAULT_MODEL = {
    "anthropic": config.ANTHROPIC_VISION_MODEL,
    "openai": config.OPENAI_VISION_MODEL,
    "gemini": config.GEMINI_VISION_MODEL,
}


# --- Positions --------------------------------------------------------------


def _load_positions_raw() -> list[dict]:
    if not os.path.exists(config.POSITIONS_PATH):
        return []
    with open(config.POSITIONS_PATH) as f:
        return json.load(f)


def _save_positions_raw(positions: list[dict]) -> None:
    # Validate every row through the real model before writing anything --
    # collect ALL row errors (identified by index + ticker) rather than
    # raising on the first one, so a single bad row doesn't hide the reason
    # for the rest, and the error is actionable instead of a bare Python
    # exception string.
    errors = []
    for i, row in enumerate(positions):
        try:
            Position.from_dict(row)
        except Exception as exc:
            errors.append(f"row {i + 1} ({row.get('ticker', '?')}): {exc}")
    if errors:
        raise ValueError("; ".join(errors))

    if os.path.exists(config.POSITIONS_PATH):
        shutil.copy(config.POSITIONS_PATH, config.POSITIONS_PATH + ".bak")

    with open(config.POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


@app.route("/")
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        default_provider=config.NEWS_PROVIDER,
        growth_upside_threshold=config.GROWTH_TARGET_UPSIDE_THRESHOLD_PCT,
        nearlow_max_pct_from_low=config.NEARLOW_MAX_PCT_FROM_LOW,
        nearlow_min_buy_ratio_pct=config.NEARLOW_MIN_BUY_RATIO_PCT,
        nearlow_min_ratings_count=config.NEARLOW_MIN_RATINGS_COUNT,
    )


@app.route("/api/positions", methods=["GET"])
def get_positions():
    positions = _load_positions_raw()
    print(f"[positions] GET -> {len(positions)} position(s) from {config.POSITIONS_PATH}")
    return jsonify(positions)


@app.route("/api/positions", methods=["POST"])
def save_positions():
    positions = request.get_json(force=True)
    if not isinstance(positions, list):
        return jsonify({"error": "expected a JSON array of positions"}), 400
    try:
        _save_positions_raw(positions)
    except Exception as exc:
        print(f"[positions] POST rejected {len(positions)} row(s): {exc}")
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 400
    print(f"[positions] POST -> saved {len(positions)} position(s) to {config.POSITIONS_PATH}")
    return jsonify({"status": "ok", "count": len(positions)})


@app.route("/api/parse-screenshot", methods=["POST"])
def parse_screenshot():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no image uploaded"}), 400

    provider = request.form.get("provider", config.VISION_PROVIDER)
    if provider not in VISION_DEFAULT_MODEL:
        return jsonify({"error": f"unknown provider {provider!r}"}), 400
    model = request.form.get("model") or VISION_DEFAULT_MODEL[provider]

    api_key = credentials.resolve_api_key(provider, interactive=False)
    if not api_key:
        return jsonify({"error": f"No saved API key for {provider}. Add one in the Settings tab first."}), 400

    image_bytes = file.read()
    media_type = file.mimetype or "image/png"

    try:
        extracted = vision.extract_positions_from_image(image_bytes, media_type, provider, model, api_key=api_key)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{provider} extraction failed: {exc}"}), 502

    return jsonify({"positions": extracted})


# --- Portfolio analysis -------------------------------------------------


@app.route("/api/run-analysis", methods=["POST"])
def run_analysis():
    body = request.get_json(silent=True) or {}
    asof_str = body.get("asof")
    asof_date = datetime.strptime(asof_str, "%Y-%m-%d").date() if asof_str else None
    skip_macro = bool(body.get("skip_macro", False))
    skip_news = bool(body.get("skip_news", False))
    news_provider = body.get("news_provider", config.NEWS_PROVIDER)
    news_model = body.get("news_model") or None

    news_api_key = None
    if not skip_news:
        news_api_key = credentials.resolve_api_key(news_provider, interactive=False)
        if not news_api_key:
            skip_news = True  # degrade gracefully rather than failing the whole run

    try:
        result = pipeline.run_full_analysis(
            positions_path=config.POSITIONS_PATH,
            db_path=config.DB_PATH,
            snapshots_dir=config.SNAPSHOTS_DIR,
            asof_date=asof_date,
            skip_macro=skip_macro,
            skip_news=skip_news,
            news_provider=news_provider,
            news_model=news_model,
            news_api_key=news_api_key,
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


# --- Top movers / rebound suggestions ---------------------------------------


def _latest_macro_score() -> Optional[float]:
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        row = conn.execute("SELECT score FROM macro_gate ORDER BY asof_date DESC LIMIT 1").fetchone()
    return row["score"] if row else None


def _run_movers(provider: str, model: Optional[str], asof_date: Optional[date] = None) -> list[dict]:
    asof_date = asof_date or date.today()
    resolved_model = model or NEWS_DEFAULT_MODEL[provider]
    api_key = credentials.resolve_api_key(provider, interactive=False)
    macro_score = _latest_macro_score()
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        results = movers.run_movers_scan(
            conn,
            asof_date,
            provider,
            resolved_model,
            api_key=api_key,
            macro_score=macro_score,
            threshold_pct=config.MOVERS_DROP_THRESHOLD_PCT,
            min_market_cap=config.MOVERS_MIN_MARKET_CAP,
            min_price=config.MOVERS_MIN_PRICE,
            min_volume=config.MOVERS_MIN_VOLUME,
            max_results=config.MOVERS_MAX_RESULTS,
        )
    exports.export_to_csv(results, "top_movers", "top_movers", export_root=config.EXPORTS_DIR)
    return results


@app.route("/api/movers", methods=["GET"])
def get_movers():
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        latest_date = db_mod.get_latest_market_movers_date(conn)
        if not latest_date:
            return jsonify({"asof_date": None, "movers": []})
        drops = db_mod.get_market_movers(conn, latest_date)
        combined = []
        for d in drops:
            rebound = db_mod.get_rebound_analysis(conn, latest_date, d["ticker"]) or {}
            combined.append({**d, "rebound": rebound})
    return jsonify({"asof_date": latest_date, "movers": combined})


@app.route("/api/movers/run", methods=["POST"])
def run_movers_now():
    body = request.get_json(silent=True) or {}
    provider = body.get("provider", config.NEWS_PROVIDER)
    model = body.get("model") or None

    if provider not in PROVIDERS:
        return jsonify({"error": f"unknown provider {provider!r}"}), 400

    try:
        results = _run_movers(provider, model)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502

    return jsonify({"asof_date": date.today().isoformat(), "movers": results})


@app.route("/api/scheduler/status", methods=["GET"])
def scheduler_status():
    return jsonify({"running": scheduler.is_running(), "next_run": scheduler.get_next_run_time()})


# --- Growth screener (Top Growth tab) ---------------------------------------


def _run_growth_screen(asof_date: Optional[date] = None) -> list[dict]:
    """No news/AI call here -- this is a pure data screen (screener +
    analyst ratings + 52-week range), so it needs no API key/provider."""
    asof_date = asof_date or date.today()
    db_mod.init_db(config.DB_PATH)
    candidates = growth_screener.find_growth_candidates(
        candidate_pool_size=config.GROWTH_CANDIDATE_POOL_SIZE,
        min_market_cap=config.GROWTH_MIN_MARKET_CAP,
        min_price=config.GROWTH_MIN_PRICE,
        min_volume=config.GROWTH_MIN_VOLUME,
        target_upside_threshold=config.GROWTH_TARGET_UPSIDE_THRESHOLD_PCT,
        max_results=config.GROWTH_MAX_RESULTS,
        max_workers=config.GROWTH_MAX_WORKERS,
    )
    with db_mod.connect(config.DB_PATH) as conn:
        db_mod.save_growth_candidates(conn, asof_date.isoformat(), candidates)
    exports.export_to_csv(candidates, "top_growth", "top_growth", export_root=config.EXPORTS_DIR)
    return candidates


@app.route("/api/growth", methods=["GET"])
def get_growth():
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        latest_date = db_mod.get_latest_growth_candidates_date(conn)
        if not latest_date:
            return jsonify({"asof_date": None, "candidates": []})
        candidates = db_mod.get_growth_candidates(conn, latest_date)
    return jsonify({"asof_date": latest_date, "candidates": candidates})


@app.route("/api/growth/run", methods=["POST"])
def run_growth_now():
    try:
        candidates = _run_growth_screen()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    return jsonify({"asof_date": date.today().isoformat(), "candidates": candidates})


# --- Near-52-week-low screener (Near 52W Low tab) ----------------------


def _run_nearlow_screen(asof_date: Optional[date] = None) -> list[dict]:
    """No news/AI call here -- this is a pure data screen (screener +
    52-week range + analyst ratings), so it needs no API key/provider."""
    asof_date = asof_date or date.today()
    db_mod.init_db(config.DB_PATH)
    candidates = nearlow_screener.find_nearlow_candidates(
        candidate_pool_size=config.NEARLOW_CANDIDATE_POOL_SIZE,
        min_market_cap=config.NEARLOW_MIN_MARKET_CAP,
        min_price=config.NEARLOW_MIN_PRICE,
        min_volume=config.NEARLOW_MIN_VOLUME,
        max_pct_from_low=config.NEARLOW_MAX_PCT_FROM_LOW,
        min_buy_ratio_pct=config.NEARLOW_MIN_BUY_RATIO_PCT,
        min_ratings_count=config.NEARLOW_MIN_RATINGS_COUNT,
        max_results=config.NEARLOW_MAX_RESULTS,
        max_workers=config.NEARLOW_MAX_WORKERS,
    )
    with db_mod.connect(config.DB_PATH) as conn:
        db_mod.save_nearlow_candidates(conn, asof_date.isoformat(), candidates)
    exports.export_to_csv(candidates, "near_52w_low", "near_52w_low", export_root=config.EXPORTS_DIR)
    return candidates


@app.route("/api/nearlow", methods=["GET"])
def get_nearlow():
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        latest_date = db_mod.get_latest_nearlow_candidates_date(conn)
        if not latest_date:
            return jsonify({"asof_date": None, "candidates": []})
        candidates = db_mod.get_nearlow_candidates(conn, latest_date)
    return jsonify({"asof_date": latest_date, "candidates": candidates})


@app.route("/api/nearlow/run", methods=["POST"])
def run_nearlow_now():
    try:
        candidates = _run_nearlow_screen()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    return jsonify({"asof_date": date.today().isoformat(), "candidates": candidates})


# --- Forex Factory economic calendar (Forex Calendar tab) -------------------
#
# Informational only -- shows the scheduled high-impact news calendar so
# you can see what's coming and, once released, how far the actual print
# missed the forecast. It does NOT place, size, or evaluate any trade,
# and never will run unattended: this app will not add automatic order
# execution without a human confirming each trade. Reacting to a release
# within the same second it prints is also not a realistic goal for a
# local app polling a public, rate-limited calendar feed -- that's a
# latency race won by firms with colocated servers and direct data feeds,
# not a personal dashboard. This tab is for awareness and planning
# around events, not for winning that race.


def _enrich_with_live_actuals(events: list[dict]) -> None:
    """Mutates each event dict in place, filling in "actual"/"direction"/
    "surprise_pct" by checking Forex Factory's LIVE calendar page --
    the JSON feed above never carries these (confirmed empirically; see
    forex_calendar.py). Skips any event that already has an actual
    (e.g. one that came from _fetch_month_events' day-by-day live scrape,
    which already includes it -- no need to fetch that day twice). One
    fetch per unique remaining day covered, not per event, and only for
    days that have already started, since a future day can't have an
    actual yet. Never raises: a scrape failure for one day just leaves
    that day's events as "n/a", exactly as before this step existed.

    This scrapes forexfactory.com's live HTML page directly, the same
    ToS tradeoff as forex_monitor.py -- made explicitly, at the user's
    direction, not silently."""
    now = datetime.now(timezone.utc)
    events_by_day: dict = {}
    for e in events:
        if e.get("actual") or not e.get("date"):
            continue
        try:
            event_time = datetime.fromisoformat(e["date"])
        except ValueError:
            continue
        if event_time > now:
            continue
        events_by_day.setdefault(event_time.date(), []).append(e)

    for day, day_events in events_by_day.items():
        try:
            html = forex_live_monitor.fetch_live_day_html(day)
        except Exception as exc:
            print(f"[forex_calendar] live actual fetch failed for {day}: {exc}")
            continue
        for e in day_events:
            result = forex_live_monitor.find_actual_for_event(html, e["title"], e["country"])
            if result:
                e["actual"] = result["actual"]
                e["direction"] = result["direction"]
                e["surprise_pct"] = forex_calendar._surprise_pct(result["actual"], e.get("forecast"))


def _fetch_month_events() -> list[dict]:
    """This calendar month's events (the 1st through the last day),
    across every currency -- the country/impact filtering happens
    client-side in the UI, same as the impact filter already did,
    since narrowing server-side wouldn't reduce how many pages need
    fetching anyway (one request per day regardless of how many
    currencies are on it).

    The JSON feed only ever covers "this week" (confirmed -- Forex
    Factory doesn't publish a month variant), so every other day in the
    month is filled in by live-scraping that day's page individually;
    each of those already includes the schedule AND the actual in one
    pass (see forex_live_monitor.find_all_events_for_day), unlike the
    JSON-feed days which still need _enrich_with_live_actuals separately
    afterward. Costs one extra live request per day of the month not
    already covered by the JSON feed (up to ~24) -- gated by the same
    cooldown as the rest of "Run Now" in _run_forex_calendar_refresh, so
    this only actually runs once per cooldown window regardless of how
    many times the button is clicked."""
    today = date.today()
    month_start = today.replace(day=1)
    _, last_day_num = calendar.monthrange(today.year, today.month)
    month_end = today.replace(day=last_day_num)

    week_events = forex_calendar.fetch_calendar_events(config.FOREX_CALENDAR_FEED_URL)
    covered_days = set()
    for e in week_events:
        if not e.get("date"):
            continue
        try:
            covered_days.add(datetime.fromisoformat(e["date"]).date())
        except ValueError:
            pass

    all_events = list(week_events)
    day = month_start
    scraped_days = 0
    while day <= month_end:
        if day not in covered_days:
            try:
                html = forex_live_monitor.fetch_live_day_html(day)
                all_events.extend(forex_live_monitor.find_all_events_for_day(html, day))
                scraped_days += 1
            except Exception as exc:
                print(f"[forex_calendar] month scrape failed for {day}: {exc}")
        day += timedelta(days=1)

    print(f"[forex_calendar] month view: {len(week_events)} from the JSON feed + {scraped_days} day(s) live-scraped = {len(all_events)} total")
    return all_events


def _run_forex_calendar_refresh(force: bool = False) -> tuple[list[dict], bool]:
    """Returns (events, did_refetch). Enforces
    config.FOREX_CALENDAR_MIN_REFRESH_SECONDS between real upstream
    fetches -- Forex Factory's feed is rate-limited, so a request inside
    that window re-serves the cached copy instead of risking a block.
    The same cooldown also gates the live-actuals check and the
    whole-month scrape below, since there's no confirmed rate limit for
    those separate live pages either -- better to be conservative on
    all of it together."""
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        last_fetch = db_mod.get_latest_forex_calendar_fetch(conn)
        now = datetime.now(timezone.utc)
        if last_fetch and not force:
            elapsed = (now - datetime.fromisoformat(last_fetch)).total_seconds()
            if elapsed < config.FOREX_CALENDAR_MIN_REFRESH_SECONDS:
                return db_mod.get_forex_calendar_events(conn), False

        events = _fetch_month_events()
        _enrich_with_live_actuals(events)
        db_mod.save_forex_calendar_events(conn, events, fetched_at=now.isoformat())
        return db_mod.get_forex_calendar_events(conn), True


@app.route("/api/forex-calendar", methods=["GET"])
def get_forex_calendar():
    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        events = db_mod.get_forex_calendar_events(conn)
        last_fetch = db_mod.get_latest_forex_calendar_fetch(conn)
    return jsonify({"events": events, "last_fetched_at": last_fetch})


@app.route("/api/forex-calendar/run", methods=["POST"])
def run_forex_calendar_now():
    try:
        events, did_refetch = _run_forex_calendar_refresh()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    with db_mod.connect(config.DB_PATH) as conn:
        last_fetch = db_mod.get_latest_forex_calendar_fetch(conn)
    return jsonify({"events": events, "last_fetched_at": last_fetch, "refetched": did_refetch})


def _resolve_forex_analysis_provider(requested_provider: Optional[str]) -> tuple[str, Optional[str]]:
    """If a provider was explicitly requested, use exactly that one
    (and only that one -- an explicit choice with no key is a clear
    error, not a silent fallback). Otherwise try the app's configured
    default first, then every other provider, and use whichever
    actually has a saved key -- so this works with whatever key you've
    saved rather than only the one that happens to be the overall
    default."""
    if requested_provider:
        return requested_provider, credentials.resolve_api_key(requested_provider, interactive=False)
    for provider in [config.NEWS_PROVIDER] + [p for p in PROVIDERS if p != config.NEWS_PROVIDER]:
        api_key = credentials.resolve_api_key(provider, interactive=False)
        if api_key:
            return provider, api_key
    return config.NEWS_PROVIDER, None


@app.route("/api/forex-calendar/analyze-impact", methods=["POST"])
def analyze_forex_impact():
    """On-demand only -- one event at a time, only when a human clicks
    "Analyze" on it. Never runs automatically across a whole month's
    events (that would require an API key for a tab that otherwise
    doesn't need one, and could fire dozens of AI calls per "Run Now")."""
    body = request.get_json(force=True)
    event = body.get("event")
    if not isinstance(event, dict):
        return jsonify({"error": "expected an event object"}), 400

    requested_provider = body.get("provider")
    if requested_provider and requested_provider not in PROVIDERS:
        return jsonify({"error": f"unknown provider {requested_provider!r}"}), 400

    provider, api_key = _resolve_forex_analysis_provider(requested_provider)
    if not api_key:
        return jsonify({"error": "No saved API key for any provider. Add one in the Settings tab first."}), 400
    model = body.get("model") or NEWS_DEFAULT_MODEL[provider]

    positions = _load_positions_raw()
    try:
        result = forex_calendar.analyze_portfolio_impact(event, positions, provider, model, api_key=api_key)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    return jsonify(result)


@app.route("/api/forex-calendar/analyze-stock-impact", methods=["POST"])
def analyze_stock_calendar_impact():
    """On-demand only -- one ticker at a time, only when a human picks
    impact level(s) and clicks Analyze in the Portfolio tab's Calendar
    Impact section. Uses whatever calendar data is already cached from
    the Forex Calendar tab's "Run Now" -- does not trigger a fetch of
    its own, so this stays fast and doesn't duplicate that tab's cost."""
    body = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    impact_levels = body.get("impact_levels")
    if not isinstance(impact_levels, list) or not impact_levels:
        return jsonify({"error": "impact_levels must be a non-empty list"}), 400
    normalized_levels = {str(lvl).strip().lower() for lvl in impact_levels}

    requested_provider = body.get("provider")
    if requested_provider and requested_provider not in PROVIDERS:
        return jsonify({"error": f"unknown provider {requested_provider!r}"}), 400
    provider, api_key = _resolve_forex_analysis_provider(requested_provider)
    if not api_key:
        return jsonify({"error": "No saved API key for any provider. Add one in the Settings tab first."}), 400
    model = body.get("model") or NEWS_DEFAULT_MODEL[provider]

    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        all_events = db_mod.get_forex_calendar_events(conn)
    matching_events = [e for e in all_events if (e.get("impact") or "").lower() in normalized_levels]

    positions = [p for p in _load_positions_raw() if (p.get("ticker") or "").upper() == ticker]

    try:
        result = forex_calendar.analyze_stock_calendar_impact(
            ticker, positions, matching_events, provider, model, api_key=api_key
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    return jsonify(result)


_CHART_EVENT_MATCH_TOLERANCE_SECONDS = 2 * 3600  # 2 hours -- see _match_events_to_price_history


def _closest_bar_index(history: list[dict], target_time: datetime) -> Optional[int]:
    """Index of the price bar closest in time to target_time, or None if
    history is empty or the closest bar is more than
    _CHART_EVENT_MATCH_TOLERANCE_SECONDS away (e.g. the event fell on a
    weekend/holiday with no trading, or outside the fetched window) --
    better to omit a marker than place it hours away from the truth."""
    if not history:
        return None
    bar_times = [datetime.fromisoformat(h["time"]) for h in history]
    closest_idx = min(range(len(bar_times)), key=lambda i: abs((bar_times[i] - target_time).total_seconds()))
    if abs((bar_times[closest_idx] - target_time).total_seconds()) > _CHART_EVENT_MATCH_TOLERANCE_SECONDS:
        return None
    return closest_idx


def _center_history_on_marker(
    history: list[dict], marker_index: Optional[int]
) -> tuple[list[dict], Optional[int]]:
    """Trim history to a symmetric number of bars on each side of
    marker_index, so the event bar always lands exactly in the middle of
    the zoomed chart -- letting the user compare "before" and "after" at
    a glance instead of the event landing wherever the fetched window
    happened to start (e.g. jammed against the left edge for a
    pre-market event with few earlier bars). Trims to whichever side has
    less data; returns history unchanged if there's no marker to center
    on."""
    if marker_index is None or not history:
        return history, marker_index
    half = min(marker_index, len(history) - 1 - marker_index)
    trimmed = history[marker_index - half : marker_index + half + 1]
    return trimmed, half


def _match_events_to_price_history(history: list[dict], events: list[dict]) -> list[dict]:
    """For each past High-impact event, find the closest 5-minute price
    bar by time and attach a marker there -- this is what lets the chart
    draw a red bar at (approximately) the moment the event hit, not just
    the day."""
    if not history:
        return []
    markers = []
    for e in events:
        if not e.get("date"):
            continue
        try:
            event_time = datetime.fromisoformat(e["date"])
        except ValueError:
            continue
        closest_idx = _closest_bar_index(history, event_time)
        if closest_idx is None:
            continue
        markers.append(
            {
                "bar_index": closest_idx,
                "bar_time": history[closest_idx]["time"],
                "event_time": e["date"],
                "title": e.get("title"),
                "country": e.get("country"),
                # "better"/"worse" (Forex Factory's own actual-vs-forecast
                # read) or None if unknown/matched exactly -- lets the UI
                # color each marker green/red/blue instead of one flat color.
                "direction": e.get("direction"),
            }
        )
    return markers


@app.route("/api/portfolio/price-chart", methods=["GET"])
def get_portfolio_price_chart():
    """Intraday price history for one ticker, with markers on any PAST
    High-impact calendar event whose time lands close to a bar -- so the
    chart can draw a red bar at (approximately) the moment it hit. Only
    past events have a marker (a future one has no price reaction yet
    to show). Uses whatever calendar data is already cached from the
    Forex Calendar tab's "Run Now" -- doesn't trigger a fetch of its
    own."""
    ticker = (request.args.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    try:
        history = data.get_price_history(ticker)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502

    db_mod.init_db(config.DB_PATH)
    with db_mod.connect(config.DB_PATH) as conn:
        all_events = db_mod.get_forex_calendar_events(conn)
    now = datetime.now(timezone.utc)
    past_high_impact = []
    for e in all_events:
        if (e.get("impact") or "").lower() != "high" or not e.get("date"):
            continue
        try:
            if datetime.fromisoformat(e["date"]) <= now:
                past_high_impact.append(e)
        except ValueError:
            continue

    markers = _match_events_to_price_history(history, past_high_impact)
    return jsonify({"ticker": ticker, "history": history, "markers": markers})


@app.route("/api/portfolio/price-chart-zoom", methods=["GET"])
def get_portfolio_price_chart_zoom():
    """Zoomed-in price history around one specific event's real
    timestamp (not the nearest 5-minute bar) -- the "show me the exact
    reaction" view a legend click asks for. Only events within the last
    7 days can get true 1-minute bars (Yahoo's free-data limit for that
    interval); older ones fall back to 5-minute automatically, and the
    response says which interval was actually used so the UI can be
    honest about it rather than implying more precision than exists.
    An explicit `interval` query param (currently just "1h", for the
    "switch to hourly" zoom-out toggle) skips that guessing and widens
    the window so an hourly view actually shows useful context instead
    of a handful of bars. Also returns `marker_index`, the position in
    `history` of the bar closest to the event's real timestamp, so the
    chart can draw a marker at the exact moment rather than just showing
    a plain price line."""
    ticker = (request.args.get("ticker") or "").strip().upper()
    event_time_str = request.args.get("event_time")
    requested_interval = request.args.get("interval") or None
    if not ticker or not event_time_str:
        return jsonify({"error": "ticker and event_time required"}), 400
    try:
        center_time = datetime.fromisoformat(event_time_str)
    except ValueError:
        return jsonify({"error": "invalid event_time"}), 400

    window_hours = 24.0 if requested_interval == "1h" else 2.0
    try:
        history, interval = data.get_price_history_window(
            ticker, center_time, window_hours=window_hours, interval=requested_interval
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    marker_index = _closest_bar_index(history, center_time)
    history, marker_index = _center_history_on_marker(history, marker_index)
    return jsonify(
        {"ticker": ticker, "history": history, "interval": interval, "marker_index": marker_index}
    )


# --- Settings ----------------------------------------------------------


@app.route("/api/settings/has-key", methods=["GET"])
def has_key():
    provider = request.args.get("provider", config.NEWS_PROVIDER)
    key = credentials.resolve_api_key(provider, interactive=False)
    print(f"[settings] has-key({provider}) -> {bool(key)}, reading {os.path.abspath(config.CREDENTIALS_PATH)}")
    return jsonify({"has_key": bool(key)})


@app.route("/api/settings/api-key", methods=["POST"])
def save_api_key():
    body = request.get_json(force=True)
    provider = body.get("provider")
    api_key = body.get("api_key")
    if provider not in PROVIDERS:
        return jsonify({"error": "unknown provider"}), 400
    if not api_key:
        return jsonify({"error": "api_key required"}), 400
    credentials.save_key(provider, api_key)
    return jsonify({"status": "ok"})


@app.route("/api/models", methods=["GET"])
def get_models():
    provider = request.args.get("provider", config.NEWS_PROVIDER)
    if provider not in PROVIDERS:
        return jsonify({"error": "unknown provider"}), 400
    api_key = credentials.resolve_api_key(provider, interactive=False)
    if not api_key:
        return jsonify({"error": f"No saved API key for {provider}."}), 400
    try:
        model_ids = news.list_models(provider, api_key=api_key)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    return jsonify({"models": model_ids})


# --- Scheduled job -----------------------------------------------------


def _scheduled_movers_job() -> None:
    """Runs inside the background scheduler thread -- swallow errors so a
    bad day (e.g. a network hiccup) doesn't kill the scheduler itself."""
    try:
        _run_movers(config.NEWS_PROVIDER, None)
    except Exception as exc:
        print(f"[scheduler] daily movers scan failed: {exc}")


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Portfolio Dashboard</title>
<style>
  :root {
    --bg: #0f1115; --card: #1a1e27; --border: #2a2f3a; --text: #e6e6e6; --muted: #9fb4c7;
    --accent: #2d6cdf; --green: #4caf7d; --red: #d9615b; --amber: #d9a63a;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Arial, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
  header { padding: 1rem 1.5rem 0; }
  h1 { font-size: 1.3rem; margin: 0 0 0.75rem; }
  nav { display: flex; gap: 4px; border-bottom: 1px solid var(--border); padding: 0 1.5rem; }
  nav button {
    background: none; border: none; color: var(--muted); padding: 10px 16px; font-size: 0.92rem;
    cursor: pointer; border-bottom: 2px solid transparent;
  }
  nav button.active { color: var(--text); border-bottom-color: var(--accent); }
  main { padding: 1.25rem 1.5rem 3rem; max-width: 1100px; margin: 0 auto; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 1.1rem; }
  label { font-size: 0.8rem; color: var(--muted); }
  input, select { background: var(--card); color: var(--text); border: 1px solid var(--border); padding: 6px 8px; border-radius: 4px; font-size: 0.85rem; }
  button.action { background: var(--accent); color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 0.88rem; }
  button.action:disabled { opacity: 0.6; cursor: default; }
  button.secondary { background: var(--card); color: var(--text); border: 1px solid var(--border); padding: 8px 14px; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
  button.danger { background: #4a2323; color: #f0b0ac; border: none; padding: 3px 8px; border-radius: 4px; cursor: pointer; font-size: 0.78rem; }
  .cards-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.25rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 150px; flex: 1; }
  .card .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
  .card .value { font-size: 1.35rem; margin-top: 4px; font-weight: 600; }
  .value.pos { color: var(--green); } .value.neg { color: var(--red); }
  section.block { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-bottom: 1.1rem; }
  section.block h3 { margin: 0 0 10px; font-size: 0.95rem; color: var(--muted); font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 0.83rem; }
  th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; }
  th.sortable { cursor: pointer; user-select: none; white-space: nowrap; }
  th.sortable:hover { color: var(--text); }
  th.sortable .arrow { display: inline-block; width: 1em; opacity: 0.6; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.82rem; }
  .bar-label { width: 90px; flex-shrink: 0; color: var(--muted); }
  .bar-track { flex: 1; background: #11141b; border-radius: 3px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); }
  .bar-fill.flagged { background: var(--amber); }
  .bar-pct { width: 48px; text-align: right; flex-shrink: 0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }
  .badge.bullish, .badge.positive { background: #1f3a2c; color: var(--green); }
  .badge.bearish, .badge.negative { background: #3a2323; color: var(--red); }
  .badge.neutral, .badge.no.data { background: #2a2f3a; color: var(--muted); }
  .badge.impact-high { background: #3a2323; color: var(--red); }
  .badge.impact-medium { background: #3a3220; color: var(--amber); }
  .badge.impact-low, .badge.impact-holiday { background: #2a2f3a; color: var(--muted); }
  .actual-better { color: var(--green); font-weight: 600; }
  .actual-worse { color: var(--red); font-weight: 600; }
  .news-card, .mover-card { border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
  .news-card .ticker, .mover-card .ticker { font-weight: 600; margin-right: 8px; }
  a.ticker-link { color: inherit; text-decoration: none; border-bottom: 1px dotted var(--muted); }
  a.ticker-link:hover { border-bottom-color: var(--text); }
  .mover-card .drop-pct { color: var(--red); font-weight: 600; }
  .mover-card .rebound { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 0.85rem; }
  .mover-card .rebound h4 { margin: 8px 0 3px; font-size: 0.78rem; color: var(--muted); text-transform: uppercase; }
  .disclaimer { font-style: italic; color: var(--muted); font-size: 0.75rem; margin-top: 8px; }
  #status { margin: 10px 0; padding: 8px 12px; border-radius: 4px; display: none; font-size: 0.85rem; }
  #status.ok { background: #1f3a24; color: #8fe0a0; display: block; }
  #status.err { background: #3a1f1f; color: #e08f8f; display: block; }
  .spinner { display: inline-block; width: 13px; height: 13px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 6px; vertical-align: -2px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .muted { color: var(--muted); }
  .settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
  .key-row { display: flex; gap: 6px; margin-top: 8px; }
  .source-tag { font-size: 0.7rem; color: var(--muted); }
  .col-actions { width: 36px; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
</head>
<body>

<header><h1>Portfolio Dashboard</h1></header>
<nav>
  <button class="tab-btn active" data-tab="portfolio">Portfolio</button>
  <button class="tab-btn" data-tab="movers">Top Movers</button>
  <button class="tab-btn" data-tab="growth">Top Growth</button>
  <button class="tab-btn" data-tab="nearlow">Near 52W Low</button>
  <button class="tab-btn" data-tab="forex">Forex Calendar</button>
  <button class="tab-btn" data-tab="positions">Positions</button>
  <button class="tab-btn" data-tab="settings">Settings</button>
  <button class="tab-btn" data-tab="more">More</button>
</nav>

<main>

  <!-- ===================== PORTFOLIO TAB ===================== -->
  <div class="tab-panel active" id="tab-portfolio">
    <div class="controls">
      <div><label>As of</label><br><input type="date" id="p-asof"></div>
      <div><label>News provider</label><br>
        <select id="p-provider">
          <option value="anthropic">Anthropic (Claude)</option>
          <option value="openai">OpenAI (GPT)</option>
          <option value="gemini">Google (Gemini)</option>
        </select>
      </div>
      <div><label><input type="checkbox" id="p-skip-macro"> Skip macro gate</label></div>
      <div><label><input type="checkbox" id="p-skip-news"> Skip news</label></div>
      <button class="action" id="p-run-btn" onclick="runPortfolioAnalysis()">Run Analysis</button>
    </div>
    <div id="p-status"></div>
    <div id="p-results" style="display:none">
      <div class="cards-row">
        <div class="card"><div class="label">Total Value</div><div class="value" id="p-total-value">--</div></div>
        <div class="card"><div class="label">Net Delta</div><div class="value" id="p-net-delta">--</div></div>
        <div class="card"><div class="label">Daily Theta ($)</div><div class="value" id="p-theta">--</div></div>
        <div class="card"><div class="label">Macro Gate</div><div class="value" id="p-macro-score">--</div></div>
      </div>

      <section class="block">
        <h3>Positions</h3>
        <table><thead><tr><th>Position</th><th>Mark</th><th>Value</th><th>P&amp;L</th><th>DTE</th></tr></thead>
        <tbody id="p-positions-body"></tbody></table>
      </section>

      <section class="block">
        <h3>Allocation by Ticker</h3>
        <div id="p-alloc-ticker"></div>
      </section>
      <section class="block">
        <h3>Allocation by Sector</h3>
        <div id="p-alloc-sector"></div>
      </section>

      <section class="block">
        <h3>IV Environment</h3>
        <table><thead><tr><th>Position</th><th>IV</th><th>Rank</th><th>Status</th></tr></thead>
        <tbody id="p-iv-body"></tbody></table>
      </section>

      <section class="block">
        <h3>Upcoming Expiries</h3>
        <div id="p-expiries"></div>
      </section>

      <section class="block" id="p-macro-block" style="display:none">
        <h3>Macro Gate Detail</h3>
        <div id="p-macro-detail"></div>
      </section>

      <section class="block" id="p-news-block" style="display:none">
        <h3>News</h3>
        <div id="p-news-cards"></div>
      </section>

      <section class="block" id="p-calendar-impact-block" style="display:none">
        <h3>Calendar Impact</h3>
        <p class="muted" style="margin-top:-4px;">
          On demand only -- pick which Forex Factory impact level(s) to consider for each stock, then
          click Analyze. Uses whatever calendar data is already cached from the Forex Calendar tab's
          "Run Now" -- run that first if a stock shows no events found.
        </p>
        <div id="p-calendar-impact-cards"></div>
      </section>
    </div>
  </div>

  <!-- ===================== TOP MOVERS TAB ===================== -->
  <div class="tab-panel" id="tab-movers">
    <div class="controls">
      <div><label>Provider</label><br>
        <select id="m-provider">
          <option value="anthropic">Anthropic (Claude)</option>
          <option value="openai">OpenAI (GPT)</option>
          <option value="gemini">Google (Gemini)</option>
        </select>
      </div>
      <button class="action" id="m-run-btn" onclick="runMoversNow()">Run Now</button>
      <span class="muted" id="m-scheduler-status">Scheduler status: loading...</span>
    </div>
    <div id="m-status"></div>
    <div id="m-as-of" class="muted" style="margin-bottom:8px;"></div>
    <div id="m-cards"></div>
  </div>

  <!-- ===================== TOP GROWTH TAB ===================== -->
  <div class="tab-panel" id="tab-growth">
    <div class="controls">
      <button class="action" id="g-run-btn" onclick="runGrowthNow()">Run Now</button>
    </div>
    <p class="muted" style="max-width:640px;">
      No data source predicts "growth in the next month" -- that's not a metric anyone publishes.
      This screens for liquid US stocks where the <strong>analyst consensus price target</strong>
      implies at least {{ growth_upside_threshold }}% upside. Analyst targets are conventionally
      ~12-month views, not 1-month ones -- treat this as "analysts see a lot of upside here", not a
      monthly forecast. The <strong>Strong Buy %</strong> column shows what fraction of all analyst
      ratings are "strong buy" -- it's shown for you to judge, not filtered on.
    </p>
    <div id="g-status"></div>
    <div id="g-as-of" class="muted" style="margin-bottom:8px;"></div>
    <table>
      <thead><tr id="g-head">
        <th class="sortable" data-sort="ticker">Ticker<span class="arrow"></span></th>
        <th class="sortable" data-sort="price">Price<span class="arrow"></span></th>
        <th class="sortable" data-sort="target_mean">Target (mean)<span class="arrow"></span></th>
        <th class="sortable" data-sort="target_upside_pct">Upside<span class="arrow"></span></th>
        <th class="sortable" data-sort="strong_buy_ratio_pct">Strong Buy %<span class="arrow"></span></th>
        <th class="sortable" data-sort="pct_from_52w_high">From 52w High<span class="arrow"></span></th>
        <th class="sortable" data-sort="pct_from_52w_low">From 52w Low<span class="arrow"></span></th>
      </tr></thead>
      <tbody id="g-body"></tbody>
    </table>
  </div>

  <!-- ===================== NEAR 52W LOW TAB ===================== -->
  <div class="tab-panel" id="tab-nearlow">
    <div class="controls">
      <button class="action" id="nl-run-btn" onclick="runNearlowNow()">Run Now</button>
    </div>
    <p class="muted" style="max-width:640px;">
      Beaten-down stocks the analyst consensus still likes: within {{ nearlow_max_pct_from_low }}% of the
      52-week low, with at least {{ nearlow_min_ratings_count }} analyst ratings of which
      {{ nearlow_min_buy_ratio_pct }}% or more are "buy" or "strong buy". Both conditions are required --
      this is not investment advice, just a starting point for further research.
    </p>
    <div id="nl-status"></div>
    <div id="nl-as-of" class="muted" style="margin-bottom:8px;"></div>
    <table>
      <thead><tr id="nl-head">
        <th class="sortable" data-sort="ticker">Ticker<span class="arrow"></span></th>
        <th class="sortable" data-sort="price">Price<span class="arrow"></span></th>
        <th class="sortable" data-sort="pct_from_52w_low">From 52w Low<span class="arrow"></span></th>
        <th class="sortable" data-sort="pct_from_52w_high">From 52w High<span class="arrow"></span></th>
        <th class="sortable" data-sort="buy_ratio_pct">Buy Ratio %<span class="arrow"></span></th>
        <th class="sortable" data-sort="target_upside_pct">Target Upside<span class="arrow"></span></th>
        <th class="sortable" data-sort="market_cap">Market Cap<span class="arrow"></span></th>
      </tr></thead>
      <tbody id="nl-body"></tbody>
    </table>
  </div>

  <!-- ===================== FOREX CALENDAR TAB ===================== -->
  <div class="tab-panel" id="tab-forex">
    <div class="controls">
      <select id="fx-impact-filter" onchange="renderForexTable()">
        <option value="high" selected>High impact only</option>
        <option value="medium+">Medium + High</option>
        <option value="all">All impact levels</option>
      </select>
      <label><input type="checkbox" id="fx-include-non-us" onchange="renderForexTable()"> Include non-US events</label>
      <button class="action" id="fx-run-btn" onclick="runForexNow()">Run Now</button>
    </div>
    <ul class="muted" style="max-width:640px; margin: 0 0 12px; padding-left: 18px;">
      <li>Forex Factory's high-impact economic calendar (rate decisions, CPI, NFP, GDP...) for the whole month.</li>
      <li>USD events only by default -- check "Include non-US events" for every currency.</li>
      <li>This week's schedule comes from Forex Factory's public feed; every other day, plus Actual/Surprise once released, comes from checking their live page directly -- against their Terms of Service.</li>
      <li>Covering a whole month can take up to a minute -- watch the terminal for progress.</li>
      <li>If a value looks off, check <a href="https://www.forexfactory.com/calendar" target="_blank" rel="noopener">forexfactory.com/calendar</a> directly.</li>
      <li><strong>Informational only</strong> -- never places a trade or runs unattended.</li>
    </ul>
    <div id="fx-status"></div>
    <div id="fx-next-event" class="muted" style="margin-bottom:4px; font-weight: 600;"></div>
    <div id="fx-as-of" class="muted" style="margin-bottom:8px;"></div>
    <div id="fx-impact-result" style="display:none; margin-bottom:12px; padding:10px 12px; border:1px solid var(--border); border-radius:6px; background: var(--card); font-size:0.85rem;"></div>
    <table>
      <thead><tr id="fx-head">
        <th class="sortable" data-sort="date">Time<span class="arrow"></span></th>
        <th class="sortable" data-sort="country">Currency<span class="arrow"></span></th>
        <th class="sortable" data-sort="title">Event<span class="arrow"></span></th>
        <th class="sortable" data-sort="impact">Impact<span class="arrow"></span></th>
        <th class="sortable" data-sort="previous">Previous<span class="arrow"></span></th>
        <th class="sortable" data-sort="forecast">Forecast<span class="arrow"></span></th>
        <th class="sortable" data-sort="actual">Actual<span class="arrow"></span></th>
        <th class="sortable" data-sort="surprise_pct">Surprise<span class="arrow"></span></th>
        <th></th>
      </tr></thead>
      <tbody id="fx-body"></tbody>
    </table>
  </div>

  <!-- ===================== POSITIONS TAB ===================== -->
  <div class="tab-panel" id="tab-positions">
    <h2 style="font-size:1rem;">Upload a screenshot</h2>
    <div class="controls">
      <select id="v-provider">
        <option value="anthropic">Anthropic (Claude)</option>
        <option value="openai">OpenAI (GPT)</option>
        <option value="gemini">Google (Gemini)</option>
      </select>
      <input type="file" id="imageInput" accept="image/*" multiple>
      <button class="action" onclick="parseScreenshots()">Parse screenshot(s)</button>
    </div>
    <div id="v-status"></div>

    <h2 style="font-size:1rem;">Positions</h2>
    <table id="positionsTable">
      <thead>
        <tr>
          <th>Type</th><th>Ticker</th><th>Opt Type</th><th>Strike</th><th>Expiry</th>
          <th>Entry Price</th><th>Contracts/Shares</th><th>Entry Date</th>
          <th>Target</th><th>Stop</th><th class="col-actions"></th>
        </tr>
      </thead>
      <tbody id="positionsBody"></tbody>
    </table>
    <div class="controls" style="margin-top: 1rem;">
      <button class="secondary" onclick="addRow()">+ Add row</button>
      <button class="action" onclick="savePositions()">Save positions.json</button>
    </div>
  </div>

  <!-- ===================== SETTINGS TAB ===================== -->
  <div class="tab-panel" id="tab-settings">
    <h2 style="font-size:1rem;">AI provider API keys</h2>
    <p class="muted">Saved locally to .credentials.json on this machine -- never committed to git, never sent anywhere but the provider you choose.</p>
    <div class="settings-grid" id="settings-keys"></div>

    <h2 style="font-size:1rem; margin-top: 1.5rem;">Daily Top Movers scheduler</h2>
    <p class="muted" id="settings-scheduler-info">loading...</p>
  </div>

  <!-- ===================== MORE TAB (placeholder) ===================== -->
  <div class="tab-panel" id="tab-more">
    <p class="muted">More sections coming soon.</p>
  </div>

</main>

<script>
// ---------- Tab switching ----------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

function showStatus(elId, message, ok) {
  const el = document.getElementById(elId);
  el.textContent = message;
  el.className = ok ? "ok" : "err";
}

function fmtMoney(n) {
  if (n === null || n === undefined) return "--";
  return "$" + Number(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function fmtCap(n) {
  if (n === null || n === undefined) return "--";
  n = Number(n);
  if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
  return "$" + n.toLocaleString();
}
function fmtNum(n, digits) {
  if (n === null || n === undefined) return "--";
  return Number(n).toFixed(digits === undefined ? 1 : digits);
}

// ---------- PORTFOLIO TAB ----------
document.getElementById("p-asof").valueAsDate = new Date();
document.getElementById("p-provider").value = "{{ default_provider }}";

async function runPortfolioAnalysis() {
  const btn = document.getElementById("p-run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running...';
  document.getElementById("p-status").style.display = "none";

  const body = {
    asof: document.getElementById("p-asof").value,
    news_provider: document.getElementById("p-provider").value,
    skip_macro: document.getElementById("p-skip-macro").checked,
    skip_news: document.getElementById("p-skip-news").checked,
  };

  try {
    const res = await fetch("/api/run-analysis", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) { showStatus("p-status", "Error: " + data.error, false); return; }
    renderPortfolioResults(data);
    document.getElementById("p-results").style.display = "block";
  } catch (err) {
    showStatus("p-status", "Error: " + err, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Analysis";
  }
}

function renderPortfolioResults(data) {
  document.getElementById("p-total-value").textContent = fmtMoney(data.allocation.total_value);
  document.getElementById("p-net-delta").textContent = fmtNum(data.aggregate_greeks.net_delta_shares, 1);
  const theta = data.aggregate_greeks.total_daily_theta;
  const thetaEl = document.getElementById("p-theta");
  thetaEl.textContent = fmtMoney(theta);
  thetaEl.className = "value " + (theta < 0 ? "neg" : "pos");

  const macroEl = document.getElementById("p-macro-score");
  if (data.macro) {
    macroEl.textContent = fmtNum(data.macro.score, 1) + "/100";
    document.getElementById("p-macro-block").style.display = "block";
    renderMacroDetail(data.macro);
  } else {
    macroEl.textContent = "skipped";
    document.getElementById("p-macro-block").style.display = "none";
  }

  const posBody = document.getElementById("p-positions-body");
  posBody.innerHTML = "";
  data.valuations.forEach(v => {
    const pnlClass = v.unrealized_pnl >= 0 ? "pos" : "neg";
    posBody.innerHTML += `<tr>
      <td>${v.position_id}</td><td>${fmtMoney(v.mark)}</td><td>${fmtMoney(v.current_value)}</td>
      <td class="${pnlClass}">${fmtMoney(v.unrealized_pnl)} (${v.unrealized_pnl_pct !== null ? fmtNum(v.unrealized_pnl_pct) + "%" : "n/a"})</td>
      <td>${v.dte !== null ? v.dte : "n/a"}</td>
    </tr>`;
  });

  renderAllocationBars("p-alloc-ticker", data.allocation.by_ticker);
  renderAllocationBars("p-alloc-sector", data.allocation.by_sector);

  const ivBody = document.getElementById("p-iv-body");
  ivBody.innerHTML = "";
  data.iv_environment.forEach(r => {
    const status = r.status === "building history"
      ? `building history (${r.history_days}/20)`
      : `rank ${fmtNum(r.iv_rank, 0)} ${r.rich ? "RICH" : (r.cheap ? "CHEAP" : "")}`;
    ivBody.innerHTML += `<tr><td>${r.position_id}</td><td>${fmtNum(r.current_iv, 3)}</td><td>${r.iv_rank !== null ? fmtNum(r.iv_rank, 0) : "--"}</td><td>${status}</td></tr>`;
  });

  const expiriesEl = document.getElementById("p-expiries");
  const flagged = data.upcoming_expiries.filter(e => e.within_threshold);
  expiriesEl.innerHTML = flagged.length === 0
    ? '<span class="muted">none within the warning window</span>'
    : flagged.map(e => `<div>${e.position_id} -- expires ${e.expiry} (${e.dte}d)</div>`).join("");

  const newsBlock = document.getElementById("p-news-block");
  const newsCards = document.getElementById("p-news-cards");
  if (data.news && data.news.length > 0) {
    newsBlock.style.display = "block";
    newsCards.innerHTML = data.news.map(n => {
      if (n.status !== "ok") {
        return `<div class="news-card"><span class="ticker">${n.ticker}</span><span class="muted">skipped (${n.status})</span></div>`;
      }
      const flag = n.position_flag ? ' <span class="badge negative">AFFECTS POSITION</span>' : "";
      return `<div class="news-card">
        <span class="ticker">${n.ticker}</span><span class="badge ${n.sentiment}">${n.sentiment}</span>${flag}
        <div style="margin-top:6px;">${n.summary || ""}</div>
        ${n.position_flag && n.position_flag_reason ? `<div class="muted" style="margin-top:4px;">why: ${n.position_flag_reason}</div>` : ""}
      </div>`;
    }).join("");
  } else {
    newsBlock.style.display = "none";
  }

  renderCalendarImpactCards(data.valuations);
}

function renderCalendarImpactCards(valuations) {
  const block = document.getElementById("p-calendar-impact-block");
  const container = document.getElementById("p-calendar-impact-cards");
  const tickers = [...new Set(valuations.map(v => v.ticker))];
  if (tickers.length === 0) {
    block.style.display = "none";
    return;
  }
  block.style.display = "block";
  container.innerHTML = tickers.map(ticker => `
    <div class="news-card">
      <span class="ticker">${ticker}</span>
      <label style="margin-left:8px;"><input type="checkbox" class="cal-impact-level" data-ticker="${ticker}" value="High" checked> High</label>
      <label style="margin-left:8px;"><input type="checkbox" class="cal-impact-level" data-ticker="${ticker}" value="Medium"> Medium</label>
      <label style="margin-left:8px;"><input type="checkbox" class="cal-impact-level" data-ticker="${ticker}" value="Low"> Low</label>
      <button class="secondary" style="margin-left:8px; font-size:0.75rem; padding:3px 8px;" onclick="analyzeStockCalendarImpact('${ticker}', this)">Analyze</button>
      <button class="secondary" style="margin-left:4px; font-size:0.75rem; padding:3px 8px;" onclick="toggleStockChart('${ticker}', this)">Show Chart</button>
      <div class="cal-impact-result muted" id="cal-impact-result-${ticker}" style="display:none; margin-top:8px;"></div>
      <div id="chart-wrap-${ticker}" style="display:none; margin-top:10px; max-width:700px;">
        <canvas id="chart-canvas-${ticker}" height="220"></canvas>
        <div class="muted" id="chart-status-${ticker}" style="margin-top:4px; font-size:0.75rem;"></div>
        <div id="chart-legend-${ticker}" style="margin-top:8px; font-size:0.8rem;"></div>
      </div>
    </div>
  `).join("");
}

const stockCharts = {};
const stockZoomCharts = {};
const stockMarkersByTicker = {};
const stockZoomState = {};

const DIRECTION_COLORS = {
  better: { bar: "rgba(76, 175, 125, 0.45)", dot: "#4caf7d" },   // green -- actual more favorable than forecast
  worse: { bar: "rgba(217, 97, 91, 0.45)", dot: "#d9615b" },     // red -- actual less favorable than forecast
  neutral: { bar: "rgba(45, 108, 223, 0.45)", dot: "#2d6cdf" },  // blue -- matched forecast / direction unknown
};
function directionColors(direction) {
  return DIRECTION_COLORS[direction] || DIRECTION_COLORS.neutral;
}

// Draws "1", "2", ... above each marked bar in the "High-impact event"
// dataset, matching the numbers in the legend built alongside the chart --
// Chart.js has no built-in per-bar text label, so this is a small custom
// plugin rather than pulling in a whole datalabels library for one thing.
Chart.register({
  id: "markerNumbers",
  afterDatasetsDraw(chart) {
    const dsIndex = chart.data.datasets.findIndex(d => d.label === "High-impact event");
    if (dsIndex === -1) return;
    const meta = chart.getDatasetMeta(dsIndex);
    const ctx = chart.ctx;
    ctx.save();
    ctx.fillStyle = "#cfd3dc";
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    let n = 0;
    chart.data.datasets[dsIndex].data.forEach((v, i) => {
      if (v === null || v === undefined) return;
      n++;
      const point = meta.data[i];
      if (point) ctx.fillText(String(n), point.x, point.y - 6);
    });
    ctx.restore();
  },
});

async function toggleStockChart(ticker, btn) {
  const wrap = document.getElementById(`chart-wrap-${ticker}`);
  if (wrap.style.display === "block") {
    wrap.style.display = "none";
    btn.textContent = "Show Chart";
    return;
  }
  wrap.style.display = "block";
  btn.textContent = "Hide Chart";
  if (stockCharts[ticker]) return;  // already loaded once

  const statusEl = document.getElementById(`chart-status-${ticker}`);
  statusEl.textContent = "Loading...";
  try {
    const res = await fetch(`/api/portfolio/price-chart?ticker=${encodeURIComponent(ticker)}`);
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = "Error: " + data.error;
      return;
    }
    if (!data.history || data.history.length === 0) {
      statusEl.textContent = "No price history available for this ticker.";
      return;
    }
    const labels = data.history.map(h => new Date(h.time).toLocaleString(undefined, { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }));
    const closes = data.history.map(h => h.close);
    const maxClose = Math.max(...closes);
    const sortedMarkers = [...(data.markers || [])].sort((a, b) => a.bar_index - b.bar_index);
    const markerIndexSet = new Set(sortedMarkers.map(m => m.bar_index));
    const markerBars = data.history.map((h, i) => markerIndexSet.has(i) ? maxClose : null);
    const markerColors = data.history.map((h, i) => {
      const m = sortedMarkers.find(mm => mm.bar_index === i);
      return m ? directionColors(m.direction).bar : "transparent";
    });
    const markerByIndex = {};
    sortedMarkers.forEach(m => { markerByIndex[m.bar_index] = m; });

    const ctx = document.getElementById(`chart-canvas-${ticker}`).getContext("2d");
    if (stockCharts[ticker]) stockCharts[ticker].destroy();
    stockCharts[ticker] = new Chart(ctx, {
      data: {
        labels,
        datasets: [
          {
            type: "bar",
            label: "High-impact event",
            data: markerBars,
            backgroundColor: markerColors,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
            order: 2,
          },
          {
            type: "line",
            label: ticker + " price",
            data: closes,
            borderColor: "#2d6cdf",
            backgroundColor: "transparent",
            pointRadius: 0,
            borderWidth: 1.5,
            tension: 0.1,
            order: 1,
          },
        ],
      },
      options: {
        animation: false,
        scales: {
          x: { ticks: { maxTicksLimit: 10, autoSkip: true } },
          y: { position: "right" },
        },
        plugins: {
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const m = markerByIndex[items[0].dataIndex];
                return m ? [`High impact: ${m.country} ${m.title}`] : [];
              },
            },
          },
        },
      },
    });

    stockMarkersByTicker[ticker] = sortedMarkers;
    const legendHtml = sortedMarkers.length === 0
      ? '<span class="muted">No past high-impact events landed close enough to a price bar to mark.</span>'
      : sortedMarkers.map((m, i) => {
          const dot = directionColors(m.direction).dot;
          const when = new Date(m.event_time).toLocaleString(undefined, { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
          return `<div style="margin-top:2px;">
            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:${dot}; margin-right:4px;"></span>
            <a href="#" onclick="showEventZoom('${ticker}', ${i}, event)">${i + 1}. ${m.country} ${m.title} (${when})</a>
          </div>`;
        }).join("");
    document.getElementById(`chart-legend-${ticker}`).innerHTML = legendHtml;

    statusEl.textContent = `${data.history.length} bars (5-minute), last 60 days. Bar color: green = better than forecast, red = worse, blue = matched/unknown. Click a legend entry to zoom in.`;
  } catch (err) {
    statusEl.textContent = "Error: " + err;
  }
}

function showEventZoom(ticker, markerIdx, evt) {
  if (evt) evt.preventDefault();
  const marker = (stockMarkersByTicker[ticker] || [])[markerIdx];
  if (!marker) return;
  loadEventZoom(ticker, marker, "auto");
}

function setZoomGranularityButtons(wrap, active) {
  wrap.querySelectorAll(".zoom-gran-btn").forEach(b => {
    b.style.borderColor = b.dataset.gran === active ? "#2d6cdf" : "";
    b.style.color = b.dataset.gran === active ? "#2d6cdf" : "";
  });
}

async function loadEventZoom(ticker, marker, granularity) {
  let wrap = document.getElementById(`chart-zoom-wrap-${ticker}`);
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.id = `chart-zoom-wrap-${ticker}`;
    wrap.style.marginTop = "10px";
    wrap.style.borderTop = "1px solid var(--border)";
    wrap.style.paddingTop = "10px";
    wrap.innerHTML = `
      <div style="margin-bottom:6px;">
        <button type="button" class="secondary zoom-gran-btn" data-gran="auto" style="font-size:0.7rem; padding:2px 8px;">Minute</button>
        <button type="button" class="secondary zoom-gran-btn" data-gran="1h" style="font-size:0.7rem; padding:2px 8px; margin-left:4px;">Hourly</button>
      </div>
      <canvas id="chart-zoom-canvas-${ticker}" height="180"></canvas>
      <div class="muted" id="chart-zoom-status-${ticker}" style="margin-top:4px; font-size:0.75rem;"></div>`;
    document.getElementById(`chart-wrap-${ticker}`).appendChild(wrap);
    wrap.querySelectorAll(".zoom-gran-btn").forEach(b => {
      b.onclick = () => loadEventZoom(ticker, stockZoomState[ticker].marker, b.dataset.gran);
    });
  }
  stockZoomState[ticker] = { marker, granularity };
  setZoomGranularityButtons(wrap, granularity);

  const statusEl = document.getElementById(`chart-zoom-status-${ticker}`);
  statusEl.textContent = "Loading...";
  try {
    const params = new URLSearchParams({ ticker, event_time: marker.event_time });
    if (granularity === "1h") params.set("interval", "1h");
    const res = await fetch(`/api/portfolio/price-chart-zoom?${params.toString()}`);
    const zdata = await res.json();
    if (!res.ok) { statusEl.textContent = "Error: " + zdata.error; return; }
    if (!zdata.history || zdata.history.length === 0) { statusEl.textContent = "No price data available around this event."; return; }
    const labelFmt = zdata.interval === "1h"
      ? { month: "numeric", day: "numeric", hour: "2-digit" }
      : { hour: "2-digit", minute: "2-digit" };
    const labels = zdata.history.map(h => new Date(h.time).toLocaleString(undefined, labelFmt));
    const closes = zdata.history.map(h => h.close);
    const maxClose = Math.max(...closes);
    const markerBars = zdata.history.map((h, i) => i === zdata.marker_index ? maxClose : null);
    const barColor = directionColors(marker.direction).bar;

    if (stockZoomCharts[ticker]) stockZoomCharts[ticker].destroy();
    const ctx = document.getElementById(`chart-zoom-canvas-${ticker}`).getContext("2d");
    stockZoomCharts[ticker] = new Chart(ctx, {
      data: {
        labels,
        datasets: [
          {
            type: "bar",
            label: `${marker.country} ${marker.title}`,
            data: markerBars,
            backgroundColor: barColor,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
            order: 2,
          },
          {
            type: "line",
            label: ticker + " (zoomed)",
            data: closes,
            borderColor: "#2d6cdf",
            backgroundColor: "transparent",
            pointRadius: 0,
            borderWidth: 1.5,
            tension: 0.1,
            order: 1,
          },
        ],
      },
      options: { animation: false, scales: { y: { position: "right" } } },
    });
    const intervalNote = zdata.interval === "1m"
      ? "1-minute bars"
      : zdata.interval === "1h"
        ? "hourly bars"
        : "5-minute bars (1-minute data isn't available past 7 days for this event, so this is the finest available)";
    statusEl.textContent = `${zdata.history.length} ${intervalNote}, centered on this event.`;
  } catch (err) {
    statusEl.textContent = "Error: " + err;
  }
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// The AI is prompted to write past_summary/future_summary as one "- " bullet
// per event; renders those as a real <ul> instead of a run-on paragraph so
// each event's analysis stays visually separate. Falls back to plain text
// for the "no events" case, which is prose rather than bullets by design.
function renderBulletSummary(text) {
  if (!text) return "<div>n/a</div>";
  const bulletLines = text.split("\n").map(l => l.trim()).filter(l => l.startsWith("- "));
  if (bulletLines.length === 0) {
    return `<div>${escapeHtml(text)}</div>`;
  }
  const items = bulletLines.map(l => `<li>${escapeHtml(l.slice(2))}</li>`).join("");
  return `<ul style="margin:4px 0 0 18px; padding:0;">${items}</ul>`;
}

async function analyzeStockCalendarImpact(ticker, btn) {
  const checked = [...document.querySelectorAll(`.cal-impact-level[data-ticker="${ticker}"]:checked`)].map(cb => cb.value);
  const resultEl = document.getElementById(`cal-impact-result-${ticker}`);
  if (checked.length === 0) {
    resultEl.textContent = "Pick at least one impact level.";
    resultEl.style.display = "block";
    return;
  }
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "...";
  resultEl.style.display = "none";
  try {
    const res = await fetch("/api/forex-calendar/analyze-stock-impact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, impact_levels: checked }),
    });
    const data = await res.json();
    if (!res.ok) {
      resultEl.textContent = "Error: " + data.error;
      resultEl.style.display = "block";
      return;
    }
    const eventsLine = data.most_relevant_events && data.most_relevant_events.length
      ? `<div class="muted" style="margin-top:6px;">Most relevant: ${data.most_relevant_events.join(", ")}</div>`
      : "";
    resultEl.innerHTML =
      `<div><strong>Past events</strong>${renderBulletSummary(data.past_summary)}</div>` +
      `<div style="margin-top:8px;"><strong>Future events</strong>${renderBulletSummary(data.future_summary)}</div>` +
      eventsLine +
      `<div class="disclaimer" style="margin-top:6px;">${data.disclaimer || "This is not investment advice."}</div>`;
    resultEl.style.display = "block";
  } catch (err) {
    resultEl.textContent = "Error: " + err;
    resultEl.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

function renderMacroDetail(macro) {
  const el = document.getElementById("p-macro-detail");
  const ratio = macro.term_structure.ratio !== null ? fmtNum(macro.term_structure.ratio, 3) : "n/a";
  el.innerHTML = `
    <div>VIX level: vix=${fmtNum(macro.vix_level.vix,1)} percentile=${fmtNum(macro.vix_level.vix_percentile,0)} score=${fmtNum(macro.vix_level.score,1)}</div>
    <div>Term structure: vix3m=${fmtNum(macro.term_structure.vix3m,1)} ratio=${ratio} score=${fmtNum(macro.term_structure.score,1)}</div>
    <div>Breadth: ${fmtNum(macro.breadth.pct_above_200dma,0)}% above 200dma (${macro.breadth.constituents} names) score=${fmtNum(macro.breadth.score,1)}</div>
    <div>Credit spread: HYG/TLT=${fmtNum(macro.credit_spread.credit_ratio,4)} percentile=${fmtNum(macro.credit_spread.credit_percentile,0)} score=${fmtNum(macro.credit_spread.score,1)}</div>
  `;
}

function renderAllocationBars(elId, rows) {
  const el = document.getElementById(elId);
  el.innerHTML = rows.map(r => `
    <div class="bar-row">
      <div class="bar-label">${r.name}</div>
      <div class="bar-track"><div class="bar-fill ${r.flagged ? 'flagged' : ''}" style="width:${Math.min(r.pct_of_total, 100)}%"></div></div>
      <div class="bar-pct">${fmtNum(r.pct_of_total, 1)}%</div>
    </div>
  `).join("");
}

// ---------- TOP MOVERS TAB ----------
async function loadMovers() {
  const res = await fetch("/api/movers");
  const data = await res.json();
  renderMovers(data);
}

async function runMoversNow() {
  const btn = document.getElementById("m-run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Scanning...';
  document.getElementById("m-status").style.display = "none";
  try {
    const res = await fetch("/api/movers/run", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ provider: document.getElementById("m-provider").value }),
    });
    const data = await res.json();
    if (!res.ok) { showStatus("m-status", "Error: " + data.error, false); return; }
    renderMovers(data);
    showStatus("m-status", `Found ${data.movers.length} stock(s) down more than the threshold today.`, true);
  } catch (err) {
    showStatus("m-status", "Error: " + err, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Now";
  }
}

function renderMovers(data) {
  document.getElementById("m-as-of").textContent = data.asof_date ? `As of ${data.asof_date}` : "No scan has run yet.";
  const container = document.getElementById("m-cards");
  if (!data.movers || data.movers.length === 0) {
    container.innerHTML = '<p class="muted">No stocks matched the drop threshold. Click "Run Now" to scan today\\'s market.</p>';
    return;
  }
  container.innerHTML = data.movers.map(m => {
    const r = m.rebound || {};
    const sentimentClass = (r.analyst_sentiment || "no data").replace(" ", "-");
    const riskList = (r.risk_factors || []).map(rf => `<li>${rf}</li>`).join("");
    return `<div class="mover-card">
      <a class="ticker-link ticker" href="https://finance.yahoo.com/quote/${encodeURIComponent(m.ticker)}" target="_blank" rel="noopener">${m.ticker}</a> ${m.name ? `<span class="muted">${m.name}</span>` : ""}
      <span class="drop-pct">${fmtNum(m.pct_change, 1)}%</span> to ${fmtMoney(m.price)}
      ${r.status && r.status !== "ok" ? `<div class="muted">AI analysis skipped (${r.status})</div>` : `
      <div class="rebound">
        <span class="badge ${sentimentClass}">${r.analyst_sentiment || "no data"}</span>
        <h4>What happened</h4><div>${r.cause_summary || "n/a"}</div>
        <h4>Rebound case</h4><div>${r.rebound_case || "n/a"}</div>
        ${riskList ? `<h4>Risk factors</h4><ul>${riskList}</ul>` : ""}
        <h4>Macro context</h4><div>${r.macro_context || "n/a"}</div>
        <div class="disclaimer">${r.disclaimer || "This is not investment advice."}</div>
      </div>`}
    </div>`;
  }).join("");
}

async function loadSchedulerStatus() {
  const res = await fetch("/api/scheduler/status");
  const data = await res.json();
  const text = data.running
    ? (data.next_run ? `Scheduler running -- next scan: ${new Date(data.next_run).toLocaleString()}` : "Scheduler running")
    : "Scheduler not running (start via python app.py)";
  document.getElementById("m-scheduler-status").textContent = text;
  document.getElementById("settings-scheduler-info").textContent =
    text + ". Fires automatically at 3:30pm US/Eastern on weekdays while this app is running; use \\'Run Now\\' on the Top Movers tab any other time.";
}

// ---------- TOP GROWTH TAB ----------
let growthRows = [];
let growthSort = { field: "target_upside_pct", dir: -1 };

async function loadGrowth() {
  const res = await fetch("/api/growth");
  const data = await res.json();
  renderGrowth(data);
}

async function runGrowthNow() {
  const btn = document.getElementById("g-run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Screening...';
  document.getElementById("g-status").style.display = "none";
  try {
    const res = await fetch("/api/growth/run", { method: "POST" });
    const data = await res.json();
    if (!res.ok) { showStatus("g-status", "Error: " + data.error, false); return; }
    renderGrowth(data);
    showStatus("g-status", `Found ${data.candidates.length} candidate(s) meeting the upside + strong-buy criteria.`, true);
  } catch (err) {
    showStatus("g-status", "Error: " + err, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Now";
  }
}

function renderGrowth(data) {
  document.getElementById("g-as-of").textContent = data.asof_date ? `As of ${data.asof_date}` : "No scan has run yet.";
  growthRows = data.candidates || [];
  renderGrowthTable();
}

function sortGrowth(field) {
  if (growthSort.field === field) {
    growthSort.dir *= -1;
  } else {
    growthSort.field = field;
    growthSort.dir = field === "ticker" ? 1 : -1;
  }
  renderGrowthTable();
}

function renderGrowthTable() {
  const body = document.getElementById("g-body");
  const { field, dir } = growthSort;

  document.querySelectorAll("#g-head th.sortable").forEach(th => {
    const arrow = th.querySelector(".arrow");
    arrow.textContent = th.dataset.sort === field ? (dir === 1 ? "\\u25b2" : "\\u25bc") : "";
  });

  if (growthRows.length === 0) {
    body.innerHTML = '<tr><td colspan="7" class="muted">No candidates found. Click "Run Now" to screen today\\'s market.</td></tr>';
    return;
  }

  const sorted = [...growthRows].sort((a, b) => {
    let av = a[field], bv = b[field];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "string") { av = av.toLowerCase(); bv = bv.toLowerCase(); }
    if (av < bv) return -1 * dir;
    if (av > bv) return 1 * dir;
    return 0;
  });

  body.innerHTML = sorted.map(c => {
    const ratings = c.analyst_ratings || {};
    const ratingsStr = Object.keys(ratings).length
      ? Object.entries(ratings).map(([k, v]) => `${k}: ${v}`).join(", ")
      : "n/a";
    const strongBuyCell = c.strong_buy_ratio_pct !== null && c.strong_buy_ratio_pct !== undefined
      ? `${fmtNum(c.strong_buy_ratio_pct, 0)}% <span class="muted">(${ratingsStr})</span>`
      : `n/a <span class="muted">(${ratingsStr})</span>`;
    return `<tr>
      <td><a class="ticker-link" href="https://finance.yahoo.com/quote/${encodeURIComponent(c.ticker)}" target="_blank" rel="noopener">${c.ticker}</a>${c.name ? ` <span class="muted">(${c.name})</span>` : ""}</td>
      <td>${fmtMoney(c.price)}</td>
      <td>${fmtMoney(c.target_mean)}</td>
      <td class="pos">+${fmtNum(c.target_upside_pct, 1)}%</td>
      <td>${strongBuyCell}</td>
      <td>${c.pct_from_52w_high !== null && c.pct_from_52w_high !== undefined ? fmtNum(c.pct_from_52w_high, 1) + "%" : "n/a"}</td>
      <td>${c.pct_from_52w_low !== null && c.pct_from_52w_low !== undefined ? "+" + fmtNum(c.pct_from_52w_low, 1) + "%" : "n/a"}</td>
    </tr>`;
  }).join("");
}

document.getElementById("g-head").addEventListener("click", (e) => {
  const th = e.target.closest("th.sortable");
  if (th) sortGrowth(th.dataset.sort);
});

// ---------- NEAR 52W LOW TAB ----------
let nearlowRows = [];
let nearlowSort = { field: "pct_from_52w_low", dir: 1 };

async function loadNearlow() {
  const res = await fetch("/api/nearlow");
  const data = await res.json();
  renderNearlow(data);
}

async function runNearlowNow() {
  const btn = document.getElementById("nl-run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Screening...';
  document.getElementById("nl-status").style.display = "none";
  try {
    const res = await fetch("/api/nearlow/run", { method: "POST" });
    const data = await res.json();
    if (!res.ok) { showStatus("nl-status", "Error: " + data.error, false); return; }
    renderNearlow(data);
    showStatus("nl-status", `Found ${data.candidates.length} candidate(s) near their 52-week low with strong ratings.`, true);
  } catch (err) {
    showStatus("nl-status", "Error: " + err, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Now";
  }
}

function renderNearlow(data) {
  document.getElementById("nl-as-of").textContent = data.asof_date ? `As of ${data.asof_date}` : "No scan has run yet.";
  nearlowRows = data.candidates || [];
  renderNearlowTable();
}

function sortNearlow(field) {
  if (nearlowSort.field === field) {
    nearlowSort.dir *= -1;
  } else {
    nearlowSort.field = field;
    nearlowSort.dir = field === "ticker" ? 1 : -1;
  }
  renderNearlowTable();
}

function renderNearlowTable() {
  const body = document.getElementById("nl-body");
  const { field, dir } = nearlowSort;

  document.querySelectorAll("#nl-head th.sortable").forEach(th => {
    const arrow = th.querySelector(".arrow");
    arrow.textContent = th.dataset.sort === field ? (dir === 1 ? "\\u25b2" : "\\u25bc") : "";
  });

  if (nearlowRows.length === 0) {
    body.innerHTML = '<tr><td colspan="7" class="muted">No candidates found. Click "Run Now" to screen today\\'s market.</td></tr>';
    return;
  }

  const sorted = [...nearlowRows].sort((a, b) => {
    let av = a[field], bv = b[field];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "string") { av = av.toLowerCase(); bv = bv.toLowerCase(); }
    if (av < bv) return -1 * dir;
    if (av > bv) return 1 * dir;
    return 0;
  });

  body.innerHTML = sorted.map(c => {
    const ratings = c.analyst_ratings || {};
    const ratingsStr = Object.keys(ratings).length
      ? Object.entries(ratings).map(([k, v]) => `${k}: ${v}`).join(", ")
      : "n/a";
    const buyRatioCell = c.buy_ratio_pct !== null && c.buy_ratio_pct !== undefined
      ? `${fmtNum(c.buy_ratio_pct, 0)}% <span class="muted">(${ratingsStr})</span>`
      : `n/a <span class="muted">(${ratingsStr})</span>`;
    const upsideCell = c.target_upside_pct !== null && c.target_upside_pct !== undefined
      ? `${c.target_upside_pct >= 0 ? "+" : ""}${fmtNum(c.target_upside_pct, 1)}%`
      : "n/a";
    return `<tr>
      <td><a class="ticker-link" href="https://finance.yahoo.com/quote/${encodeURIComponent(c.ticker)}" target="_blank" rel="noopener">${c.ticker}</a>${c.name ? ` <span class="muted">(${c.name})</span>` : ""}</td>
      <td>${fmtMoney(c.price)}</td>
      <td class="neg">+${fmtNum(c.pct_from_52w_low, 1)}%</td>
      <td>${c.pct_from_52w_high !== null && c.pct_from_52w_high !== undefined ? fmtNum(c.pct_from_52w_high, 1) + "%" : "n/a"}</td>
      <td>${buyRatioCell}</td>
      <td>${upsideCell}</td>
      <td>${fmtCap(c.market_cap)}</td>
    </tr>`;
  }).join("");
}

document.getElementById("nl-head").addEventListener("click", (e) => {
  const th = e.target.closest("th.sortable");
  if (th) sortNearlow(th.dataset.sort);
});

// ---------- FOREX CALENDAR TAB ----------
let forexRows = [];
let forexSort = { field: "date", dir: 1 };
let forexCountdownTimer = null;

async function loadForex() {
  const res = await fetch("/api/forex-calendar");
  const data = await res.json();
  renderForex(data);
}

async function runForexNow() {
  const btn = document.getElementById("fx-run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Fetching month (can take up to a minute)...';
  document.getElementById("fx-status").style.display = "none";
  try {
    const res = await fetch("/api/forex-calendar/run", { method: "POST" });
    const data = await res.json();
    if (!res.ok) { showStatus("fx-status", "Error: " + data.error, false); return; }
    renderForex(data);
    showStatus(
      "fx-status",
      data.refetched
        ? `Fetched ${data.events.length} event(s) from Forex Factory.`
        : `Rate-limit cooldown active -- showing the last cached pull (${data.events.length} event(s)).`,
      true
    );
  } catch (err) {
    showStatus("fx-status", "Error: " + err, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Now";
  }
}

function renderForex(data) {
  document.getElementById("fx-as-of").textContent = data.last_fetched_at
    ? `Last fetched ${new Date(data.last_fetched_at).toLocaleString()}`
    : "No data fetched yet.";
  forexRows = data.events || [];
  renderForexTable();
  startForexCountdown();
}

function sortForex(field) {
  if (forexSort.field === field) {
    forexSort.dir *= -1;
  } else {
    forexSort.field = field;
    forexSort.dir = field === "date" ? 1 : -1;
  }
  renderForexTable();
}

function forexImpactMatches(impact, filter) {
  const lvl = (impact || "").toLowerCase();
  if (filter === "high") return lvl === "high";
  if (filter === "medium+") return lvl === "high" || lvl === "medium";
  return true;
}

function renderForexTable() {
  const body = document.getElementById("fx-body");
  const { field, dir } = forexSort;
  const filter = document.getElementById("fx-impact-filter").value;
  const includeNonUs = document.getElementById("fx-include-non-us").checked;

  document.querySelectorAll("#fx-head th.sortable").forEach(th => {
    const arrow = th.querySelector(".arrow");
    arrow.textContent = th.dataset.sort === field ? (dir === 1 ? "\\u25b2" : "\\u25bc") : "";
  });

  const filtered = forexRows.filter(e =>
    forexImpactMatches(e.impact, filter) && (includeNonUs || e.country === "USD")
  );

  if (filtered.length === 0) {
    body.innerHTML = '<tr><td colspan="9" class="muted">No events match this filter. Click "Run Now" to fetch this month\\'s calendar.</td></tr>';
    return;
  }

  const sorted = [...filtered].sort((a, b) => {
    let av = a[field], bv = b[field];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (field === "date") { av = new Date(av).getTime(); bv = new Date(bv).getTime(); }
    else if (typeof av === "string") { av = av.toLowerCase(); bv = bv.toLowerCase(); }
    if (av < bv) return -1 * dir;
    if (av > bv) return 1 * dir;
    return 0;
  });

  body.innerHTML = sorted.map(e => {
    const impactClass = (e.impact || "").toLowerCase();
    const surpriseCell = e.surprise_pct !== null && e.surprise_pct !== undefined
      ? `${e.surprise_pct >= 0 ? "+" : ""}${fmtNum(e.surprise_pct, 1)}%`
      : "--";
    const when = e.date ? new Date(e.date) : null;
    const actualClass = e.direction === "better" ? "actual-better" : e.direction === "worse" ? "actual-worse" : "";
    return `<tr>
      <td>${when && !isNaN(when) ? when.toLocaleString() : "n/a"}</td>
      <td>${e.country || "n/a"}</td>
      <td>${e.title || "n/a"}</td>
      <td><span class="badge impact-${impactClass}">${e.impact || "n/a"}</span></td>
      <td>${e.previous ?? "n/a"}</td>
      <td>${e.forecast ?? "n/a"}</td>
      <td class="${actualClass}">${e.actual ?? "n/a"}</td>
      <td>${surpriseCell}</td>
      <td><button class="secondary" style="font-size:0.75rem; padding:3px 8px;" data-event='${JSON.stringify(e).replace(/'/g, "&#39;")}' onclick="analyzeForexImpact(this)">Analyze</button></td>
    </tr>`;
  }).join("");
}

async function analyzeForexImpact(btn) {
  const event = JSON.parse(btn.getAttribute("data-event"));
  const resultEl = document.getElementById("fx-impact-result");
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "...";
  resultEl.style.display = "none";
  try {
    const res = await fetch("/api/forex-calendar/analyze-impact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event }),
    });
    const data = await res.json();
    if (!res.ok) {
      resultEl.textContent = "Error: " + data.error;
      resultEl.style.display = "block";
      return;
    }
    const tickersLine = data.affected_tickers && data.affected_tickers.length
      ? `<div class="muted" style="margin-top:6px;">Tickers referenced: ${data.affected_tickers.join(", ")}</div>`
      : "";
    resultEl.innerHTML =
      `<strong>${event.title} (${event.country})</strong> -- impact on your portfolio:<br>${data.impact_summary || "n/a"}` +
      tickersLine +
      `<div class="disclaimer" style="margin-top:6px;">${data.disclaimer || "This is not investment advice."}</div>`;
    resultEl.style.display = "block";
  } catch (err) {
    resultEl.textContent = "Error: " + err;
    resultEl.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

document.getElementById("fx-head").addEventListener("click", (e) => {
  const th = e.target.closest("th.sortable");
  if (th) sortForex(th.dataset.sort);
});

function startForexCountdown() {
  if (forexCountdownTimer) clearInterval(forexCountdownTimer);
  const tick = () => {
    const el = document.getElementById("fx-next-event");
    const now = Date.now();
    const includeNonUs = document.getElementById("fx-include-non-us").checked;
    const upcoming = forexRows
      .filter(e =>
        (e.impact || "").toLowerCase() === "high" &&
        (includeNonUs || e.country === "USD") &&
        e.date && new Date(e.date).getTime() > now
      )
      .sort((a, b) => new Date(a.date) - new Date(b.date));
    if (upcoming.length === 0) { el.textContent = ""; return; }
    const next = upcoming[0];
    const diffMs = new Date(next.date).getTime() - now;
    const h = Math.floor(diffMs / 3600000);
    const m = Math.floor((diffMs % 3600000) / 60000);
    const s = Math.floor((diffMs % 60000) / 1000);
    el.textContent = `Next high-impact event: ${next.country || ""} ${next.title || ""} in ${h}h ${m}m ${s}s`;
  };
  tick();
  forexCountdownTimer = setInterval(tick, 1000);
}

// ---------- POSITIONS TAB ----------
let rows = [];

function emptyRow(source) {
  return {
    asset_type: "shares", ticker: "", option_type: "", strike: "", expiry: "",
    entry_price: "", contracts: "", entry_date: "", target_price: "", stop_price: "",
    _source: source || "manual",
  };
}

function renderPositionsTable() {
  const body = document.getElementById("positionsBody");
  body.innerHTML = "";
  rows.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><select onchange="updateField(${i}, 'asset_type', this.value)">
            <option value="shares" ${row.asset_type === "shares" ? "selected" : ""}>shares</option>
            <option value="option" ${row.asset_type === "option" ? "selected" : ""}>option</option>
          </select></td>
      <td><input value="${row.ticker ?? ""}" onchange="updateField(${i}, 'ticker', this.value)"></td>
      <td><select onchange="updateField(${i}, 'option_type', this.value)">
            <option value="" ${!row.option_type ? "selected" : ""}></option>
            <option value="call" ${row.option_type === "call" ? "selected" : ""}>call</option>
            <option value="put" ${row.option_type === "put" ? "selected" : ""}>put</option>
          </select></td>
      <td><input value="${row.strike ?? ""}" onchange="updateField(${i}, 'strike', this.value)"></td>
      <td><input value="${row.expiry ?? ""}" placeholder="YYYY-MM-DD" onchange="updateField(${i}, 'expiry', this.value)"></td>
      <td><input value="${row.entry_price ?? ""}" onchange="updateField(${i}, 'entry_price', this.value)"></td>
      <td><input value="${row.contracts ?? ""}" onchange="updateField(${i}, 'contracts', this.value)"></td>
      <td><input value="${row.entry_date ?? ""}" placeholder="YYYY-MM-DD" onchange="updateField(${i}, 'entry_date', this.value)"></td>
      <td><input value="${row.target_price ?? ""}" onchange="updateField(${i}, 'target_price', this.value)"></td>
      <td><input value="${row.stop_price ?? ""}" onchange="updateField(${i}, 'stop_price', this.value)"></td>
      <td class="col-actions"><button class="danger" onclick="deleteRow(${i})">x</button></td>
    `;
    body.appendChild(tr);
    if (row._source && row._source !== "manual") {
      const tag = document.createElement("tr");
      tag.innerHTML = `<td colspan="11" class="source-tag">from screenshot -- please verify every field above (Entry Date is rarely visible on a brokerage screen and can be left blank)</td>`;
      body.appendChild(tag);
    }
  });
}

function updateField(i, field, value) { rows[i][field] = value; }
function addRow() { rows.push(emptyRow("manual")); renderPositionsTable(); }
function deleteRow(i) { rows.splice(i, 1); renderPositionsTable(); }

async function loadPositions() {
  const res = await fetch("/api/positions");
  const data = await res.json();
  rows = data.map(r => ({ ...emptyRow("manual"), ...r, _source: "manual" }));
  renderPositionsTable();
}

function toNumberOrNull(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
}

async function savePositions() {
  const payload = rows.map(r => {
    const row = {
      asset_type: r.asset_type, ticker: r.ticker,
      entry_price: toNumberOrNull(r.entry_price), contracts: toNumberOrNull(r.contracts),
      entry_date: r.entry_date || null, target_price: toNumberOrNull(r.target_price), stop_price: toNumberOrNull(r.stop_price),
    };
    if (r.asset_type === "option") { row.option_type = r.option_type; row.strike = toNumberOrNull(r.strike); row.expiry = r.expiry; }
    return row;
  });
  const res = await fetch("/api/positions", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (res.ok) showStatus("v-status", `Saved ${data.count} position(s) to positions.json.`, true);
  else showStatus("v-status", `Error: ${data.error}`, false);
}

async function parseScreenshots() {
  const provider = document.getElementById("v-provider").value;
  const files = document.getElementById("imageInput").files;
  if (files.length === 0) { showStatus("v-status", "Choose at least one screenshot first.", false); return; }
  showStatus("v-status", `Parsing ${files.length} screenshot(s) with ${provider}...`, true);
  for (const file of files) {
    const form = new FormData();
    form.append("image", file);
    form.append("provider", provider);
    try {
      const res = await fetch("/api/parse-screenshot", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) { showStatus("v-status", `Error parsing ${file.name}: ${data.error}`, false); return; }
      data.positions.forEach(p => rows.push({ ...emptyRow("screenshot"), ...p, _source: "screenshot" }));
    } catch (err) { showStatus("v-status", `Error parsing ${file.name}: ${err}`, false); return; }
  }
  renderPositionsTable();
  showStatus("v-status", "Screenshot(s) parsed -- review the highlighted rows below, then Save.", true);
}

// ---------- SETTINGS TAB ----------
const PROVIDER_LABELS = { anthropic: "Anthropic (Claude)", openai: "OpenAI (GPT)", gemini: "Google (Gemini)" };

async function loadSettings() {
  const container = document.getElementById("settings-keys");
  container.innerHTML = Object.keys(PROVIDER_LABELS).map(p => `
    <div class="card">
      <div class="label">${PROVIDER_LABELS[p]}</div>
      <div id="key-status-${p}" class="muted" style="margin: 6px 0;">checking...</div>
      <div class="key-row">
        <input type="password" id="key-input-${p}" placeholder="Paste API key">
        <button class="secondary" onclick="saveKey('${p}')">Save</button>
      </div>
    </div>
  `).join("");

  for (const p of Object.keys(PROVIDER_LABELS)) {
    const res = await fetch("/api/settings/has-key?provider=" + p);
    const data = await res.json();
    document.getElementById(`key-status-${p}`).textContent = data.has_key ? "Key saved" : "No key saved";
  }
}

async function saveKey(provider) {
  const input = document.getElementById(`key-input-${provider}`);
  const apiKey = input.value.trim();
  if (!apiKey) return;
  const res = await fetch("/api/settings/api-key", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  const data = await res.json();
  if (res.ok) {
    document.getElementById(`key-status-${provider}`).textContent = "Key saved";
    input.value = "";
  } else {
    document.getElementById(`key-status-${provider}`).textContent = "Error: " + data.error;
  }
}

// ---------- init ----------
loadPositions();
loadMovers();
loadGrowth();
loadNearlow();
loadForex();
loadSchedulerStatus();
loadSettings();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    scheduler.start_scheduler(_scheduled_movers_job)
    url = "http://127.0.0.1:5050"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Portfolio dashboard running at {url}")
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
