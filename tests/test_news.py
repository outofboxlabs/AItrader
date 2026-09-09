import json
from datetime import date, datetime, timezone

import portfolio_monitor.news as news_mod
from portfolio_monitor import db
from portfolio_monitor.news import (
    _normalize_news_item,
    _parse_analysis_json,
    build_user_message,
    get_or_analyze_news,
)


def test_normalize_news_item_new_shape():
    raw = {
        "content": {
            "title": "Company beats earnings",
            "summary": "Big beat on EPS.",
            "pubDate": "2025-05-01T12:00:00Z",
            "provider": {"displayName": "Reuters"},
        }
    }
    item = _normalize_news_item(raw)
    assert item["title"] == "Company beats earnings"
    assert item["publisher"] == "Reuters"
    assert item["published_at"] == datetime(2025, 5, 1, 12, 0, tzinfo=timezone.utc)


def test_normalize_news_item_old_shape():
    epoch = int(datetime(2025, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
    raw = {"title": "Old-style headline", "publisher": "Bloomberg", "providerPublishTime": epoch}
    item = _normalize_news_item(raw)
    assert item["title"] == "Old-style headline"
    assert item["publisher"] == "Bloomberg"
    assert item["published_at"] == datetime(2025, 5, 1, 12, 0, tzinfo=timezone.utc)


def test_normalize_news_item_missing_title_returns_none():
    assert _normalize_news_item({"content": {}}) is None
    assert _normalize_news_item({}) is None


def test_fetch_recent_headlines_filters_by_window(monkeypatch):
    now = datetime(2025, 5, 10, tzinfo=timezone.utc)
    fresh_epoch = int(datetime(2025, 5, 9, tzinfo=timezone.utc).timestamp())
    stale_epoch = int(datetime(2025, 4, 1, tzinfo=timezone.utc).timestamp())

    class FakeTicker:
        def __init__(self, ticker):
            self.news = [
                {"title": "Fresh", "publisher": "A", "providerPublishTime": fresh_epoch},
                {"title": "Stale", "publisher": "B", "providerPublishTime": stale_epoch},
            ]

    monkeypatch.setattr(news_mod.yf, "Ticker", FakeTicker)
    headlines = news_mod.fetch_recent_headlines("AAPL", window_days=3, now=now)
    assert len(headlines) == 1
    assert headlines[0]["title"] == "Fresh"


def test_build_user_message_empty_headlines():
    msg = build_user_message("AAPL", [])
    assert "AAPL" in msg
    assert "No headlines" in msg


def test_parse_analysis_json_valid():
    text = json.dumps(
        {
            "summary": "Strong quarter.",
            "sentiment": "positive",
            "key_drivers": ["earnings beat"],
            "position_flag": True,
            "position_flag_reason": "earnings surprise",
        }
    )
    parsed = _parse_analysis_json(text)
    assert parsed["sentiment"] == "positive"
    assert parsed["parse_error"] is False


def test_parse_analysis_json_extracts_embedded_object():
    text = 'Sure, here it is:\n{"summary": "ok", "sentiment": "neutral", "key_drivers": [], "position_flag": false, "position_flag_reason": null}\nHope that helps.'
    parsed = _parse_analysis_json(text)
    assert parsed["sentiment"] == "neutral"
    assert parsed["parse_error"] is False


def test_parse_analysis_json_falls_back_on_garbage():
    parsed = _parse_analysis_json("not json at all")
    assert parsed["parse_error"] is True
    assert parsed["sentiment"] == "neutral"
    assert parsed["position_flag"] is False


def test_get_or_analyze_news_caches_and_skips_second_api_call(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    call_count = {"n": 0}

    def fake_analyze(ticker, headlines, api_key, model):
        call_count["n"] += 1
        return {
            "summary": "Something happened.",
            "sentiment": "positive",
            "key_drivers": ["driver"],
            "position_flag": False,
            "position_flag_reason": None,
        }

    monkeypatch.setattr(news_mod, "analyze_headlines_with_claude", fake_analyze)
    monkeypatch.setattr(news_mod, "fetch_recent_headlines", lambda ticker, window_days, now=None: [])

    asof = date(2025, 5, 1)
    with db.connect(db_path) as conn:
        first = get_or_analyze_news(conn, "AAPL", asof, window_days=3, model="fake-model")
        second = get_or_analyze_news(conn, "AAPL", asof, window_days=3, model="fake-model")

    assert call_count["n"] == 1  # second call hit the cache, not the API
    assert first["sentiment"] == "positive"
    assert second["sentiment"] == "positive"
    assert second["status"] == "ok"


def test_get_or_analyze_news_records_skip_status_on_failure(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    def failing_analyze(ticker, headlines, api_key, model):
        raise RuntimeError("no API key")

    monkeypatch.setattr(news_mod, "analyze_headlines_with_claude", failing_analyze)
    monkeypatch.setattr(news_mod, "fetch_recent_headlines", lambda ticker, window_days, now=None: [])

    with db.connect(db_path) as conn:
        result = get_or_analyze_news(conn, "AAPL", date(2025, 5, 1), window_days=3, model="fake-model")

    assert result["status"].startswith("skipped:")
    assert result["sentiment"] is None
