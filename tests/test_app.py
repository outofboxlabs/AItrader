import calendar
import io
import json
from datetime import date, datetime, timedelta, timezone

import pytest

import app as app_mod
import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod.config, "POSITIONS_PATH", str(tmp_path / "positions.json"))
    monkeypatch.setattr(app_mod.config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(app_mod.config, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(app_mod.config, "EXPORTS_DIR", str(tmp_path / "exports"))

    # _fetch_month_events live-scrapes every day of the month the JSON
    # feed doesn't cover (up to ~30 real HTTP calls) -- default that to
    # an instant, harmless failure so any test that doesn't specifically
    # care about the month scrape (most forex-calendar tests) can't
    # accidentally make real network calls. Tests that DO care override
    # this themselves via their own monkeypatch.setattr, which takes
    # precedence since it runs later, inside the test body.
    def _no_live_data(day):
        raise RuntimeError("live fetch not mocked for this test")

    monkeypatch.setattr(app_mod.forex_live_monitor, "fetch_live_day_html", _no_live_data)

    app_mod.app.config.update(TESTING=True)
    with app_mod.app.test_client() as c:
        yield c


# --- positions ---------------------------------------------------------


def test_get_positions_empty_when_file_missing(client):
    res = client.get("/api/positions")
    assert res.status_code == 200
    assert res.get_json() == []


def test_save_and_get_positions_roundtrip(client):
    payload = [{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 10, "entry_date": "2025-01-01"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    assert res.get_json()["count"] == 1
    assert client.get("/api/positions").get_json()[0]["ticker"] == "AAPL"


def test_save_positions_rejects_invalid_row(client):
    payload = [{"asset_type": "not-a-real-type", "ticker": "AAPL"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 400


def test_save_positions_accepts_missing_entry_date(client):
    """Regression test: a screenshot-extracted position commonly has no
    discoverable entry date (brokerage position lists don't show one), and
    that must not block saving the whole batch -- it previously did,
    because Position.from_dict required it, so a screenshot-derived save
    always silently failed with every row rejected."""
    payload = [{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 10, "entry_date": None}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    assert client.get("/api/positions").get_json()[0]["ticker"] == "AAPL"


def test_save_positions_error_identifies_bad_row(client):
    payload = [
        {"asset_type": "shares", "ticker": "GOOD", "entry_price": 1.0, "contracts": 1, "entry_date": "2025-01-01"},
        {"asset_type": "not-a-real-type", "ticker": "BAD"},
    ]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 400
    error = res.get_json()["error"]
    assert "row 2" in error
    assert "BAD" in error


def test_save_positions_backs_up_previous_file(client, tmp_path):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text('[{"asset_type": "shares", "ticker": "OLD", "entry_price": 1, "contracts": 1, "entry_date": "2025-01-01"}]')
    payload = [{"asset_type": "shares", "ticker": "NEW", "entry_price": 2, "contracts": 2, "entry_date": "2025-01-02"}]
    res = client.post("/api/positions", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    assert (tmp_path / "positions.json.bak").exists()
    assert "OLD" in (tmp_path / "positions.json.bak").read_text()


def test_parse_screenshot_requires_image(client):
    res = client.post("/api/parse-screenshot", data={"provider": "anthropic"})
    assert res.status_code == 400


def test_parse_screenshot_requires_api_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    data = {"image": (io.BytesIO(b"fake"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "No saved API key" in res.get_json()["error"]


def test_parse_screenshot_returns_extracted_positions(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        app_mod.vision,
        "extract_positions_from_image",
        lambda image_bytes, media_type, provider, model, api_key=None: [{"ticker": "AAPL"}],
    )
    data = {"image": (io.BytesIO(b"fake"), "screenshot.png"), "provider": "anthropic"}
    res = client.post("/api/parse-screenshot", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["positions"] == [{"ticker": "AAPL"}]


# --- run-analysis --------------------------------------------------------


def test_run_analysis_returns_pipeline_result(client, monkeypatch, tmp_path):
    (tmp_path / "positions.json").write_text(
        '[{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 5, "entry_date": "2025-01-01"}]'
    )
    monkeypatch.setattr(
        app_mod.pipeline,
        "run_full_analysis",
        lambda **kw: {"asof_date": "2025-05-01", "valuations": [], "allocation": {"total_value": 0}},
    )
    res = client.post("/api/run-analysis", data=json.dumps({"asof": "2025-05-01", "skip_news": True}), content_type="application/json")
    assert res.status_code == 200
    assert res.get_json()["asof_date"] == "2025-05-01"


def test_run_analysis_degrades_when_no_api_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    captured = {}

    def fake_run(**kw):
        captured.update(kw)
        return {"ok": True}

    monkeypatch.setattr(app_mod.pipeline, "run_full_analysis", fake_run)
    res = client.post("/api/run-analysis", data=json.dumps({}), content_type="application/json")
    assert res.status_code == 200
    assert captured["skip_news"] is True  # no key found -> degrades rather than failing


def test_run_analysis_handles_pipeline_failure(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(app_mod.pipeline, "run_full_analysis", boom)
    res = client.post("/api/run-analysis", data=json.dumps({"skip_news": True}), content_type="application/json")
    assert res.status_code == 502
    assert "error" in res.get_json()


# --- movers --------------------------------------------------------------


def test_get_movers_empty_when_no_scan_yet(client):
    res = client.get("/api/movers")
    assert res.status_code == 200
    assert res.get_json() == {"asof_date": None, "movers": []}


def test_run_movers_now_returns_results(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        app_mod.movers,
        "run_movers_scan",
        lambda conn, asof_date, provider, model, **kwargs: [
            {"ticker": "ACME", "pct_change": -45.0, "rebound": {"analyst_sentiment": "neutral"}}
        ],
    )
    res = client.post("/api/movers/run", data=json.dumps({"provider": "anthropic"}), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["movers"][0]["ticker"] == "ACME"


def test_run_movers_now_rejects_unknown_provider(client):
    res = client.post("/api/movers/run", data=json.dumps({"provider": "not-real"}), content_type="application/json")
    assert res.status_code == 400


def test_get_movers_after_run_reads_from_db(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")

    def fake_scan(conn, asof_date, provider, model, **kwargs):
        from portfolio_monitor import db as db_mod

        db_mod.save_market_movers(conn, asof_date.isoformat(), [{"ticker": "ACME", "name": "Acme", "pct_change": -45.0, "price": 10.0, "volume": 1, "market_cap": 1}])
        return [{"ticker": "ACME", "pct_change": -45.0, "rebound": {}}]

    monkeypatch.setattr(app_mod.movers, "run_movers_scan", fake_scan)
    client.post("/api/movers/run", data=json.dumps({"provider": "anthropic"}), content_type="application/json")

    res = client.get("/api/movers")
    data = res.get_json()
    assert data["asof_date"] == date.today().isoformat()
    assert data["movers"][0]["ticker"] == "ACME"


# --- scheduler -------------------------------------------------------------


def test_scheduler_status_not_running_by_default(client):
    app_mod.scheduler.stop_scheduler()
    res = client.get("/api/scheduler/status")
    assert res.get_json() == {"running": False, "next_run": None}


# --- settings --------------------------------------------------------------


def test_has_key_reports_false_when_none_saved(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    res = client.get("/api/settings/has-key?provider=anthropic")
    assert res.get_json() == {"has_key": False}


def test_save_api_key_and_then_has_key(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CREDENTIALS_PATH", str(tmp_path / ".credentials.json"))
    res = client.post("/api/settings/api-key", data=json.dumps({"provider": "anthropic", "api_key": "sk-abc"}), content_type="application/json")
    assert res.status_code == 200

    res = client.get("/api/settings/has-key?provider=anthropic")
    assert res.get_json() == {"has_key": True}


def test_save_api_key_rejects_unknown_provider(client):
    res = client.post("/api/settings/api-key", data=json.dumps({"provider": "nope", "api_key": "x"}), content_type="application/json")
    assert res.status_code == 400


def test_get_models_requires_saved_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    res = client.get("/api/models?provider=anthropic")
    assert res.status_code == 400


def test_get_models_returns_list(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(app_mod.news, "list_models", lambda provider, api_key=None: ["model-a", "model-b"])
    res = client.get("/api/models?provider=anthropic")
    assert res.status_code == 200
    assert res.get_json()["models"] == ["model-a", "model-b"]


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Portfolio Dashboard" in res.data
    assert b"{{" not in res.data  # no leftover unrendered Jinja


# --- growth screener -----------------------------------------------------


def test_get_growth_empty_when_no_scan_yet(client):
    res = client.get("/api/growth")
    assert res.status_code == 200
    assert res.get_json() == {"asof_date": None, "candidates": []}


def test_run_growth_now_returns_and_persists_results(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_mod.growth_screener,
        "find_growth_candidates",
        lambda **kw: [
            {
                "ticker": "STRONG",
                "name": "Strong Co",
                "price": 100.0,
                "target_mean": 170.0,
                "target_upside_pct": 70.0,
                "analyst_ratings": {"strongBuy": 9, "buy": 1},
                "pct_from_52w_high": -16.7,
                "pct_from_52w_low": 66.7,
                "market_cap": 5e9,
            }
        ],
    )
    res = client.post("/api/growth/run")
    assert res.status_code == 200
    data = res.get_json()
    assert data["candidates"][0]["ticker"] == "STRONG"
    assert data["asof_date"] == date.today().isoformat()

    # Persisted -- a fresh GET reads it back from the db.
    res2 = client.get("/api/growth")
    data2 = res2.get_json()
    assert data2["candidates"][0]["ticker"] == "STRONG"
    assert data2["candidates"][0]["analyst_ratings"]["strongBuy"] == 9

    # And a CSV landed in the configured exports dir under top_growth/.
    growth_export_dir = tmp_path / "exports" / "top_growth"
    assert growth_export_dir.exists()
    assert len(list(growth_export_dir.glob("*.csv"))) == 1


def test_run_growth_now_handles_screener_failure(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("yahoo screener down")

    monkeypatch.setattr(app_mod.growth_screener, "find_growth_candidates", boom)
    res = client.post("/api/growth/run")
    assert res.status_code == 502
    assert "error" in res.get_json()


# --- near-52-week-low screener -------------------------------------------


def test_get_nearlow_empty_when_no_scan_yet(client):
    res = client.get("/api/nearlow")
    assert res.status_code == 200
    assert res.get_json() == {"asof_date": None, "candidates": []}


def test_run_nearlow_now_returns_and_persists_results(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        app_mod.nearlow_screener,
        "find_nearlow_candidates",
        lambda **kw: [
            {
                "ticker": "BEATEN",
                "name": "Beaten Co",
                "price": 10.0,
                "year_low": 9.5,
                "year_high": 20.0,
                "pct_from_52w_low": 5.3,
                "pct_from_52w_high": -50.0,
                "target_mean": 15.0,
                "target_upside_pct": 50.0,
                "analyst_ratings": {"strongBuy": 6, "buy": 3},
                "buy_ratio_pct": 90.0,
                "market_cap": 5e9,
            }
        ],
    )
    res = client.post("/api/nearlow/run")
    assert res.status_code == 200
    data = res.get_json()
    assert data["candidates"][0]["ticker"] == "BEATEN"
    assert data["asof_date"] == date.today().isoformat()

    # Persisted -- a fresh GET reads it back from the db.
    res2 = client.get("/api/nearlow")
    data2 = res2.get_json()
    assert data2["candidates"][0]["ticker"] == "BEATEN"
    assert data2["candidates"][0]["analyst_ratings"]["strongBuy"] == 6

    # And a CSV landed in the configured exports dir under near_52w_low/.
    nearlow_export_dir = tmp_path / "exports" / "near_52w_low"
    assert nearlow_export_dir.exists()
    assert len(list(nearlow_export_dir.glob("*.csv"))) == 1


def test_run_nearlow_now_handles_screener_failure(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("yahoo screener down")

    monkeypatch.setattr(app_mod.nearlow_screener, "find_nearlow_candidates", boom)
    res = client.post("/api/nearlow/run")
    assert res.status_code == 502
    assert "error" in res.get_json()


# --- forex calendar -------------------------------------------------------


def test_get_forex_calendar_empty_when_no_fetch_yet(client):
    res = client.get("/api/forex-calendar")
    assert res.status_code == 200
    assert res.get_json() == {"events": [], "last_fetched_at": None}


def _future_event(**overrides):
    """A high-impact USD event scheduled safely in the future, so
    _enrich_with_live_actuals's "already happened" check naturally skips
    it -- no live fetch, no monkeypatching of forex_live_monitor needed,
    for tests that aren't specifically exercising that enrichment step."""
    event = {
        "date": (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),
        "country": "USD",
        "title": "Non-Farm Payrolls",
        "impact": "High",
        "forecast": "180K",
        "previous": "150K",
        "actual": None,
        "direction": None,
        "surprise_pct": None,
    }
    event.update(overrides)
    return event


def test_fetch_month_events_merges_json_feed_and_live_scraped_days(monkeypatch):
    """The JSON feed only ever covers "this week" -- every other day in
    the calendar month must come from live-scraping that day
    individually, and a day the JSON feed already returned must not
    also be live-scraped (that would be a wasted, redundant request)."""
    today = date.today()
    _, last_day_num = calendar.monthrange(today.year, today.month)

    today_midnight_utc = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    monkeypatch.setattr(
        app_mod.forex_calendar,
        "fetch_calendar_events",
        lambda feed_url: [
            {
                "date": today_midnight_utc, "country": "USD", "title": "Today's Event", "impact": "High",
                "forecast": "0.1%", "previous": "0.1%", "actual": None, "direction": None, "surprise_pct": None,
            }
        ],
    )

    scraped_days = []
    monkeypatch.setattr(
        app_mod.forex_live_monitor, "fetch_live_day_html", lambda day: (scraped_days.append(day), "<html></html>")[1]
    )
    monkeypatch.setattr(
        app_mod.forex_live_monitor,
        "find_all_events_for_day",
        lambda html, day: [
            {
                "date": f"{day.isoformat()}T09:00:00-04:00", "country": "USD", "title": f"Scraped {day}",
                "impact": "Low", "forecast": "", "previous": "", "actual": None, "direction": None, "surprise_pct": None,
            }
        ],
    )

    events = app_mod._fetch_month_events()

    assert today not in scraped_days  # already covered by the JSON feed -- must not be re-scraped
    assert len(scraped_days) == last_day_num - 1  # every other day of the month
    assert any(e["title"] == "Today's Event" for e in events)
    assert len(events) == 1 + len(scraped_days)


def test_fetch_month_events_survives_one_days_scrape_failing(monkeypatch):
    today = date.today()
    monkeypatch.setattr(app_mod.forex_calendar, "fetch_calendar_events", lambda feed_url: [])

    def boom(day):
        raise RuntimeError("forexfactory.com unreachable")

    monkeypatch.setattr(app_mod.forex_live_monitor, "fetch_live_day_html", boom)

    events = app_mod._fetch_month_events()  # must not raise
    assert events == []


def test_run_forex_calendar_now_fetches_and_persists(client, monkeypatch):
    monkeypatch.setattr(app_mod.forex_calendar, "fetch_calendar_events", lambda feed_url: [_future_event()])
    scheduled_at = _future_event()["date"]

    res = client.post("/api/forex-calendar/run")
    assert res.status_code == 200
    data = res.get_json()
    assert data["refetched"] is True
    assert data["events"][0]["title"] == "Non-Farm Payrolls"
    # Regression check: the db column is "event_date", but the API/UI
    # field is "date" -- get_forex_calendar_events must rename it back,
    # or every row renders "n/a" for its time regardless of what Forex
    # Factory itself sends.
    assert "event_date" not in data["events"][0]
    assert data["last_fetched_at"] is not None

    # Persisted -- a fresh GET reads it back from the db.
    res2 = client.get("/api/forex-calendar")
    data2 = res2.get_json()
    assert data2["events"][0]["country"] == "USD"


def test_run_forex_calendar_now_respects_cooldown(client, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(feed_url):
        calls["n"] += 1
        return [_future_event()]

    monkeypatch.setattr(app_mod.forex_calendar, "fetch_calendar_events", fake_fetch)

    res1 = client.post("/api/forex-calendar/run")
    assert res1.get_json()["refetched"] is True
    assert calls["n"] == 1

    # Immediately again -- should be within the cooldown window and re-serve cache.
    res2 = client.post("/api/forex-calendar/run")
    assert res2.get_json()["refetched"] is False
    assert calls["n"] == 1  # no second upstream call


def test_run_forex_calendar_now_handles_fetch_failure(client, monkeypatch):
    def boom(feed_url):
        raise RuntimeError("forex factory unreachable")

    monkeypatch.setattr(app_mod.forex_calendar, "fetch_calendar_events", boom)
    res = client.post("/api/forex-calendar/run")
    assert res.status_code == 502
    assert "error" in res.get_json()


def test_analyze_forex_impact_returns_ai_result(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    (tmp_path / "positions.json").write_text(
        '[{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 10, "entry_date": "2025-01-01"}]'
    )

    captured = {}

    def fake_analyze(event, positions, provider, model, api_key=None):
        captured["event"] = event
        captured["positions"] = positions
        return {"impact_summary": "Could pressure AAPL.", "affected_tickers": ["AAPL"], "disclaimer": "This is not investment advice.", "parse_error": False}

    monkeypatch.setattr(app_mod.forex_calendar, "analyze_portfolio_impact", fake_analyze)

    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    res = client.post("/api/forex-calendar/analyze-impact", data=json.dumps({"event": event}), content_type="application/json")

    assert res.status_code == 200
    data = res.get_json()
    assert data["impact_summary"] == "Could pressure AAPL."
    assert captured["event"]["title"] == "CPI m/m"
    assert captured["positions"][0]["ticker"] == "AAPL"


def test_analyze_forex_impact_requires_event_object(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    res = client.post("/api/forex-calendar/analyze-impact", data=json.dumps({"event": "not a dict"}), content_type="application/json")
    assert res.status_code == 400


def test_analyze_forex_impact_requires_saved_api_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    res = client.post("/api/forex-calendar/analyze-impact", data=json.dumps({"event": event}), content_type="application/json")
    assert res.status_code == 400
    assert "No saved API key" in res.get_json()["error"]


def test_analyze_forex_impact_auto_detects_provider_with_a_saved_key(client, monkeypatch):
    """The default provider (anthropic) has no key, but openai does --
    with no explicit provider requested, it must fall back to openai
    automatically rather than erroring on the default alone."""
    assert app_mod.config.NEWS_PROVIDER == "anthropic"
    monkeypatch.setattr(
        app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-openai-test" if provider == "openai" else None
    )
    captured = {}

    def fake_analyze(event, positions, provider, model, api_key=None):
        captured["provider"] = provider
        captured["api_key"] = api_key
        return {"impact_summary": "n/a", "affected_tickers": [], "disclaimer": "This is not investment advice.", "parse_error": False}

    monkeypatch.setattr(app_mod.forex_calendar, "analyze_portfolio_impact", fake_analyze)

    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    res = client.post("/api/forex-calendar/analyze-impact", data=json.dumps({"event": event}), content_type="application/json")

    assert res.status_code == 200
    assert captured["provider"] == "openai"
    assert captured["api_key"] == "sk-openai-test"


def test_analyze_forex_impact_explicit_provider_choice_is_not_overridden(client, monkeypatch):
    """An explicitly requested provider with no key is a clear error --
    it must not silently fall back to a different provider."""
    monkeypatch.setattr(
        app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-openai-test" if provider == "openai" else None
    )
    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    res = client.post(
        "/api/forex-calendar/analyze-impact",
        data=json.dumps({"event": event, "provider": "anthropic"}),
        content_type="application/json",
    )
    assert res.status_code == 400


def _seed_cached_forex_events(events):
    app_mod.db_mod.init_db(app_mod.config.DB_PATH)
    with app_mod.db_mod.connect(app_mod.config.DB_PATH) as conn:
        app_mod.db_mod.save_forex_calendar_events(conn, events, fetched_at=datetime.now(timezone.utc).isoformat())


def test_analyze_stock_impact_filters_cached_events_by_impact_level(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    (tmp_path / "positions.json").write_text(
        '[{"asset_type": "shares", "ticker": "AAPL", "entry_price": 200.0, "contracts": 10, "entry_date": "2025-01-01"}]'
    )
    _seed_cached_forex_events(
        [
            {"date": "2026-09-11T08:30:00-04:00", "country": "USD", "title": "CPI m/m", "impact": "High", "forecast": "0.4%", "previous": "0.1%"},
            {"date": "2026-09-12T08:30:00-04:00", "country": "USD", "title": "Consumer Credit", "impact": "Low", "forecast": "10B", "previous": "9B"},
        ]
    )

    captured = {}

    def fake_analyze(ticker, positions, events, provider, model, api_key=None):
        captured["ticker"] = ticker
        captured["positions"] = positions
        captured["events"] = events
        return {"past_summary": "n/a", "future_summary": "n/a", "most_relevant_events": [], "disclaimer": "This is not investment advice.", "parse_error": False}

    monkeypatch.setattr(app_mod.forex_calendar, "analyze_stock_calendar_impact", fake_analyze)

    res = client.post(
        "/api/forex-calendar/analyze-stock-impact",
        data=json.dumps({"ticker": "aapl", "impact_levels": ["High"]}),
        content_type="application/json",
    )

    assert res.status_code == 200
    assert captured["ticker"] == "AAPL"
    assert captured["positions"][0]["ticker"] == "AAPL"
    assert [e["title"] for e in captured["events"]] == ["CPI m/m"]  # the Low-impact one is filtered out


def test_analyze_stock_impact_requires_ticker(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    res = client.post(
        "/api/forex-calendar/analyze-stock-impact",
        data=json.dumps({"impact_levels": ["High"]}),
        content_type="application/json",
    )
    assert res.status_code == 400


def test_analyze_stock_impact_requires_nonempty_impact_levels(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    res = client.post(
        "/api/forex-calendar/analyze-stock-impact",
        data=json.dumps({"ticker": "AAPL", "impact_levels": []}),
        content_type="application/json",
    )
    assert res.status_code == 400


def test_analyze_stock_impact_requires_saved_api_key(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: None)
    res = client.post(
        "/api/forex-calendar/analyze-stock-impact",
        data=json.dumps({"ticker": "AAPL", "impact_levels": ["High"]}),
        content_type="application/json",
    )
    assert res.status_code == 400
    assert "No saved API key" in res.get_json()["error"]


def test_analyze_stock_impact_handles_ai_failure(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")

    def boom(ticker, positions, events, provider, model, api_key=None):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(app_mod.forex_calendar, "analyze_stock_calendar_impact", boom)

    res = client.post(
        "/api/forex-calendar/analyze-stock-impact",
        data=json.dumps({"ticker": "AAPL", "impact_levels": ["High"]}),
        content_type="application/json",
    )
    assert res.status_code == 502


# --- portfolio price chart --------------------------------------------


def test_match_events_to_price_history_finds_closest_bar_within_tolerance():
    history = [
        {"time": "2026-09-11T08:25:00-04:00", "close": 100.0},
        {"time": "2026-09-11T08:30:00-04:00", "close": 101.0},
        {"time": "2026-09-11T08:35:00-04:00", "close": 99.5},
    ]
    events = [{"date": "2026-09-11T08:31:00-04:00", "title": "CPI m/m", "country": "USD"}]
    markers = app_mod._match_events_to_price_history(history, events)
    assert len(markers) == 1
    assert markers[0]["bar_index"] == 1
    assert markers[0]["title"] == "CPI m/m"


def test_match_events_to_price_history_skips_event_too_far_from_any_bar():
    history = [{"time": "2026-09-11T08:30:00-04:00", "close": 100.0}]
    events = [{"date": "2026-09-15T08:30:00-04:00", "title": "Far Event", "country": "USD"}]
    assert app_mod._match_events_to_price_history(history, events) == []


def test_match_events_to_price_history_handles_empty_history():
    events = [{"date": "2026-09-11T08:30:00-04:00", "title": "x", "country": "USD"}]
    assert app_mod._match_events_to_price_history([], events) == []


def test_get_portfolio_price_chart_marks_close_past_high_impact_event(client, monkeypatch):
    now = datetime.now(timezone.utc)
    bar_time = (now - timedelta(hours=1)).isoformat()
    monkeypatch.setattr(app_mod.data, "get_price_history", lambda ticker, **kw: [{"time": bar_time, "close": 101.0}])

    event_date = (now - timedelta(hours=1, minutes=5)).isoformat()  # within the 2h match tolerance
    _seed_cached_forex_events(
        [{"date": event_date, "country": "USD", "title": "CPI m/m", "impact": "High", "forecast": "0.4%", "previous": "0.1%"}]
    )

    res = client.get("/api/portfolio/price-chart?ticker=aapl")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ticker"] == "AAPL"
    assert len(data["history"]) == 1
    assert len(data["markers"]) == 1
    assert data["markers"][0]["title"] == "CPI m/m"


def test_get_portfolio_price_chart_excludes_future_events(client, monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(app_mod.data, "get_price_history", lambda ticker, **kw: [{"time": now.isoformat(), "close": 101.0}])

    future_event = (now + timedelta(days=1)).isoformat()
    _seed_cached_forex_events(
        [{"date": future_event, "country": "USD", "title": "Future CPI", "impact": "High", "forecast": "0.4%", "previous": "0.1%"}]
    )

    res = client.get("/api/portfolio/price-chart?ticker=aapl")
    assert res.get_json()["markers"] == []


def test_get_portfolio_price_chart_excludes_non_high_impact(client, monkeypatch):
    now = datetime.now(timezone.utc)
    bar_time = (now - timedelta(minutes=30)).isoformat()
    monkeypatch.setattr(app_mod.data, "get_price_history", lambda ticker, **kw: [{"time": bar_time, "close": 101.0}])

    event_date = (now - timedelta(minutes=25)).isoformat()
    _seed_cached_forex_events(
        [{"date": event_date, "country": "USD", "title": "Low Impact Thing", "impact": "Low", "forecast": "", "previous": ""}]
    )

    res = client.get("/api/portfolio/price-chart?ticker=AAPL")
    assert res.get_json()["markers"] == []


def test_get_portfolio_price_chart_requires_ticker(client):
    res = client.get("/api/portfolio/price-chart")
    assert res.status_code == 400


def test_get_portfolio_price_chart_handles_data_failure(client, monkeypatch):
    def boom(ticker, **kw):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(app_mod.data, "get_price_history", boom)
    res = client.get("/api/portfolio/price-chart?ticker=AAPL")
    assert res.status_code == 502


def test_analyze_forex_impact_rejects_unknown_provider(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    res = client.post(
        "/api/forex-calendar/analyze-impact",
        data=json.dumps({"event": event, "provider": "not-real"}),
        content_type="application/json",
    )
    assert res.status_code == 400


def test_analyze_forex_impact_handles_ai_failure(client, monkeypatch):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")

    def boom(event, positions, provider, model, api_key=None):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(app_mod.forex_calendar, "analyze_portfolio_impact", boom)

    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    res = client.post("/api/forex-calendar/analyze-impact", data=json.dumps({"event": event}), content_type="application/json")
    assert res.status_code == 502
    assert "error" in res.get_json()


def test_enrich_with_live_actuals_skips_future_events(monkeypatch):
    """A future event can't have an actual yet -- must not even attempt
    a live fetch for it. (Unit-tested directly against
    _enrich_with_live_actuals, not through the full "Run Now" route --
    that route's month-wide schedule scrape legitimately does cover
    future days too, just not for their actual.)"""
    events = [_future_event()]
    live_fetch_called = []
    monkeypatch.setattr(app_mod.forex_live_monitor, "fetch_live_day_html", lambda day: live_fetch_called.append(day) or "<html></html>")

    app_mod._enrich_with_live_actuals(events)

    assert live_fetch_called == []
    assert events[0]["actual"] is None


def test_enrich_with_live_actuals_skips_events_that_already_have_an_actual(monkeypatch):
    """A day already live-scraped in full by _fetch_month_events (which
    gets the actual directly, in the same pass as the schedule) must
    not be fetched a second time here."""
    past_event = _future_event(
        date=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), actual="227K", direction="better"
    )
    live_fetch_called = []
    monkeypatch.setattr(app_mod.forex_live_monitor, "fetch_live_day_html", lambda day: live_fetch_called.append(day) or "<html></html>")

    app_mod._enrich_with_live_actuals([past_event])

    assert live_fetch_called == []
    assert past_event["actual"] == "227K"  # untouched, not overwritten


def test_run_forex_calendar_now_enriches_past_event_with_live_actual(client, monkeypatch):
    """The core new behavior: an already-released event gets its
    actual/direction/surprise_pct filled in from a live scrape, not left
    as "n/a" forever the way the JSON feed alone would leave it."""
    past_event = _future_event(date=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    monkeypatch.setattr(app_mod.forex_calendar, "fetch_calendar_events", lambda feed_url: [past_event])
    monkeypatch.setattr(app_mod.forex_live_monitor, "fetch_live_day_html", lambda day: "<html>fake page</html>")
    monkeypatch.setattr(
        app_mod.forex_live_monitor,
        "find_actual_for_event",
        lambda html, title, country: {"actual": "227K", "direction": "better", "raw_class": "better"},
    )

    res = client.post("/api/forex-calendar/run")
    assert res.status_code == 200
    event = res.get_json()["events"][0]
    assert event["actual"] == "227K"
    assert event["direction"] == "better"
    assert event["surprise_pct"] is not None

    # Persisted, including the new "direction" column.
    res2 = client.get("/api/forex-calendar")
    assert res2.get_json()["events"][0]["direction"] == "better"


def test_run_forex_calendar_now_survives_live_fetch_failure(client, monkeypatch):
    """A live-scrape failure for one day must not break the whole
    refresh -- the JSON-feed data (schedule/forecast/previous) is still
    valuable even if the live actual can't be checked right now."""
    past_event = _future_event(date=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    monkeypatch.setattr(app_mod.forex_calendar, "fetch_calendar_events", lambda feed_url: [past_event])

    def boom(day):
        raise RuntimeError("forexfactory.com unreachable")

    monkeypatch.setattr(app_mod.forex_live_monitor, "fetch_live_day_html", boom)

    res = client.post("/api/forex-calendar/run")
    assert res.status_code == 200
    event = res.get_json()["events"][0]
    assert event["title"] == "Non-Farm Payrolls"
    assert event["actual"] is None


def test_run_movers_now_writes_csv_export(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod.credentials, "resolve_api_key", lambda provider, interactive=False: "sk-test")
    monkeypatch.setattr(
        app_mod.movers,
        "run_movers_scan",
        lambda conn, asof_date, provider, model, **kwargs: [{"ticker": "ACME", "pct_change": -45.0}],
    )
    res = client.post("/api/movers/run", data=json.dumps({"provider": "anthropic"}), content_type="application/json")
    assert res.status_code == 200

    movers_export_dir = tmp_path / "exports" / "top_movers"
    assert movers_export_dir.exists()
    assert len(list(movers_export_dir.glob("*.csv"))) == 1
