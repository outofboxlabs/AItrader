import json
from datetime import date, datetime, timezone

import pytest

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


def test_analyze_headlines_dispatches_to_anthropic(monkeypatch):
    calls = []
    monkeypatch.setattr(
        news_mod,
        "analyze_headlines_with_claude",
        lambda ticker, headlines, api_key, model: calls.append(("anthropic", model)) or {"sentiment": "neutral"},
    )
    result = news_mod.analyze_headlines("AAPL", [], provider="anthropic", model="claude-haiku-4-5")
    assert calls == [("anthropic", "claude-haiku-4-5")]
    assert result["sentiment"] == "neutral"


def test_analyze_headlines_dispatches_to_openai(monkeypatch):
    calls = []
    monkeypatch.setattr(
        news_mod,
        "analyze_headlines_with_openai",
        lambda ticker, headlines, api_key, model: calls.append(("openai", model)) or {"sentiment": "positive"},
    )
    result = news_mod.analyze_headlines("AAPL", [], provider="openai", model="gpt-4o-mini")
    assert calls == [("openai", "gpt-4o-mini")]
    assert result["sentiment"] == "positive"


def test_analyze_headlines_dispatches_to_gemini(monkeypatch):
    calls = []
    monkeypatch.setattr(
        news_mod,
        "analyze_headlines_with_gemini",
        lambda ticker, headlines, api_key, model: calls.append(("gemini", model)) or {"sentiment": "negative"},
    )
    result = news_mod.analyze_headlines("AAPL", [], provider="gemini", model="gemini-2.0-flash")
    assert calls == [("gemini", "gemini-2.0-flash")]
    assert result["sentiment"] == "negative"


def test_list_models_gemini_returns_live_names(monkeypatch):
    import google.genai as genai

    class FakeModel:
        def __init__(self, name):
            self.name = name

    class FakeModels:
        def list(self):
            return [FakeModel("models/gemini-2.0-flash"), FakeModel("models/gemini-2.0-pro")]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    result = news_mod.list_models("gemini", api_key="key")
    assert result == ["models/gemini-2.0-flash", "models/gemini-2.0-pro"]


def test_analyze_headlines_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown NEWS_PROVIDER"):
        news_mod.analyze_headlines("AAPL", [], provider="not-a-real-provider", model="whatever")


def test_get_or_analyze_news_with_openai_provider_caches(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    call_count = {"n": 0}

    def fake_openai_analyze(ticker, headlines, api_key, model):
        call_count["n"] += 1
        return {
            "summary": "OpenAI summary.",
            "sentiment": "negative",
            "key_drivers": ["driver"],
            "position_flag": False,
            "position_flag_reason": None,
        }

    monkeypatch.setattr(news_mod, "analyze_headlines_with_openai", fake_openai_analyze)
    monkeypatch.setattr(news_mod, "fetch_recent_headlines", lambda ticker, window_days, now=None: [])

    asof = date(2025, 5, 1)
    with db.connect(db_path) as conn:
        first = get_or_analyze_news(conn, "AAPL", asof, window_days=3, model="gpt-4o-mini", provider="openai")
        second = get_or_analyze_news(conn, "AAPL", asof, window_days=3, model="gpt-4o-mini", provider="openai")

    assert call_count["n"] == 1
    assert first["provider"] == "openai"
    assert second["provider"] == "openai"
    assert second["sentiment"] == "negative"


class _FakeModel:
    def __init__(self, id):
        self.id = id


def test_list_models_anthropic_returns_live_ids(monkeypatch):
    import anthropic

    class FakeModels:
        def list(self):
            return [_FakeModel("claude-opus-5"), _FakeModel("claude-haiku-4-5")]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    result = news_mod.list_models("anthropic", api_key="sk-test")
    assert result == ["claude-opus-5", "claude-haiku-4-5"]


def test_list_models_openai_filters_to_chat_models(monkeypatch):
    import openai

    class FakeModels:
        def list(self):
            return [
                _FakeModel("text-embedding-3-small"),
                _FakeModel("gpt-4o-mini"),
                _FakeModel("gpt-4o"),
                _FakeModel("whisper-1"),
            ]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    result = news_mod.list_models("openai", api_key="sk-test")
    assert result == ["gpt-4o", "gpt-4o-mini"]  # sorted, non-chat models filtered out


def test_list_models_openai_falls_back_to_full_list_if_no_chat_models_match(monkeypatch):
    import openai

    class FakeModels:
        def list(self):
            return [_FakeModel("text-embedding-3-small"), _FakeModel("whisper-1")]

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    result = news_mod.list_models("openai", api_key="sk-test")
    assert result == ["text-embedding-3-small", "whisper-1"]


def test_list_models_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown NEWS_PROVIDER"):
        news_mod.list_models("not-a-real-provider")
