from datetime import date

import pytest

from portfolio_monitor import db, movers


def test_find_big_drops_maps_screener_response(monkeypatch):
    captured = {}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        captured["query"] = query
        captured["size"] = size
        return {
            "quotes": [
                {
                    "symbol": "ACME",
                    "shortName": "Acme Corp",
                    "regularMarketChangePercent": -45.2,
                    "regularMarketPrice": 12.34,
                    "regularMarketVolume": 5_000_000,
                    "marketCap": 3_000_000_000,
                }
            ]
        }

    monkeypatch.setattr(movers.yf, "screen", fake_screen)
    result = movers.find_big_drops(threshold_pct=-40.0, max_results=25)

    assert captured["size"] == 25
    assert result == [
        {
            "ticker": "ACME",
            "name": "Acme Corp",
            "pct_change": -45.2,
            "price": 12.34,
            "volume": 5_000_000,
            "market_cap": 3_000_000_000,
        }
    ]


def test_find_big_drops_handles_empty_response(monkeypatch):
    monkeypatch.setattr(movers.yf, "screen", lambda *a, **kw: {"quotes": []})
    assert movers.find_big_drops() == []


def test_get_analyst_snapshot_sanitizes_and_tolerates_failures(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_analyst_price_targets(self):
            return {"current": 10.0, "mean": 15.0}

        def get_recommendations_summary(self, as_dict=False):
            raise RuntimeError("no data available")

        def get_upgrades_downgrades(self, as_dict=False):
            return [{"firm": "Big Bank", "action": "up"}]

    monkeypatch.setattr(movers.yf, "Ticker", FakeTicker)
    result = movers.get_analyst_snapshot("ACME")

    assert result["price_targets"] == {"current": 10.0, "mean": 15.0}
    assert result["recommendations"] == {}  # failed fetch degrades to empty, not a crash
    assert result["recent_actions"] == [{"firm": "Big Bank", "action": "up"}]


def test_parse_rebound_json_valid():
    import json

    text = json.dumps(
        {
            "cause_summary": "Missed earnings.",
            "rebound_case": "Oversold on a one-time miss.",
            "risk_factors": ["guidance cut"],
            "analyst_sentiment": "neutral",
            "macro_context": "Calm environment.",
        }
    )
    parsed = movers._parse_rebound_json(text)
    assert parsed["analyst_sentiment"] == "neutral"
    assert parsed["disclaimer"] == "This is not investment advice."
    assert parsed["parse_error"] is False


def test_parse_rebound_json_falls_back_on_garbage():
    parsed = movers._parse_rebound_json("not json")
    assert parsed["parse_error"] is True
    assert parsed["analyst_sentiment"] == "no data"
    assert parsed["disclaimer"] == "This is not investment advice."


def test_analyze_rebound_candidate_calls_ai_client(monkeypatch):
    calls = []

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        calls.append((provider, model))
        return '{"cause_summary": "ok", "rebound_case": "maybe", "risk_factors": [], "analyst_sentiment": "bullish", "macro_context": "calm"}'

    monkeypatch.setattr(movers.ai_client, "call_provider", fake_call_provider)

    result = movers.analyze_rebound_candidate(
        "ACME",
        {"pct_change": -45.0, "price": 10.0, "name": "Acme"},
        [],
        {"price_targets": {}},
        70.0,
        "anthropic",
        "claude-haiku-4-5",
        api_key="sk-test",
    )

    assert calls == [("anthropic", "claude-haiku-4-5")]
    assert result["analyst_sentiment"] == "bullish"


def test_get_or_analyze_rebound_caches_and_skips_second_call(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    call_count = {"n": 0}

    def fake_analyze(ticker, drop, headlines, analyst, macro_score, provider, model, api_key=None):
        call_count["n"] += 1
        return {
            "cause_summary": "Missed guidance.",
            "rebound_case": "Possible overreaction.",
            "risk_factors": ["weak sector"],
            "analyst_sentiment": "neutral",
            "macro_context": "calm enough",
            "disclaimer": "This is not investment advice.",
            "parse_error": False,
        }

    monkeypatch.setattr(movers, "analyze_rebound_candidate", fake_analyze)

    drop = {"ticker": "ACME", "pct_change": -45.0, "price": 10.0, "name": "Acme"}
    asof = date(2025, 5, 1)
    with db.connect(db_path) as conn:
        first = movers.get_or_analyze_rebound(conn, "ACME", asof, drop, [], {}, 70.0, "anthropic", "claude-haiku-4-5")
        second = movers.get_or_analyze_rebound(conn, "ACME", asof, drop, [], {}, 70.0, "anthropic", "claude-haiku-4-5")

    assert call_count["n"] == 1
    assert first["analyst_sentiment"] == "neutral"
    assert second["analyst_sentiment"] == "neutral"


def test_get_or_analyze_rebound_records_skip_status_on_failure(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    def failing_analyze(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(movers, "analyze_rebound_candidate", failing_analyze)

    drop = {"ticker": "ACME", "pct_change": -45.0, "price": 10.0, "name": "Acme"}
    with db.connect(db_path) as conn:
        result = movers.get_or_analyze_rebound(
            conn, "ACME", date(2025, 5, 1), drop, [], {}, None, "anthropic", "claude-haiku-4-5"
        )

    assert result["status"].startswith("skipped:")
    assert result["cause_summary"] is None


def test_run_movers_scan_end_to_end(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    monkeypatch.setattr(
        movers,
        "find_big_drops",
        lambda *a, **kw: [{"ticker": "ACME", "name": "Acme Corp", "pct_change": -45.0, "price": 10.0, "volume": 1, "market_cap": 3e9}],
    )
    monkeypatch.setattr(movers, "get_analyst_snapshot", lambda ticker: {"price_targets": {}})
    monkeypatch.setattr(movers.news_mod, "fetch_recent_headlines", lambda ticker, window_days, now=None: [])
    monkeypatch.setattr(
        movers,
        "analyze_rebound_candidate",
        lambda *a, **kw: {
            "cause_summary": "ok",
            "rebound_case": "maybe",
            "risk_factors": [],
            "analyst_sentiment": "neutral",
            "macro_context": "calm",
            "disclaimer": "This is not investment advice.",
            "parse_error": False,
        },
    )

    with db.connect(db_path) as conn:
        results = movers.run_movers_scan(conn, date(2025, 5, 1), "anthropic", "claude-haiku-4-5", api_key="sk-test", macro_score=65.0)

    assert len(results) == 1
    assert results[0]["ticker"] == "ACME"
    assert results[0]["rebound"]["analyst_sentiment"] == "neutral"

    with db.connect(db_path) as conn:
        stored = db.get_market_movers(conn, "2025-05-01")
    assert len(stored) == 1
    assert stored[0]["ticker"] == "ACME"

    import json

    json.dumps(results)  # must be JSON-serializable end to end
