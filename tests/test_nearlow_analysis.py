import pandas as pd
import pytest

from portfolio_monitor import nearlow_analysis as nla


def _candidate(**overrides):
    base = {
        "ticker": "ACME",
        "name": "Acme Corp",
        "price": 42.0,
        "year_low": 40.0,
        "year_high": 90.0,
        "pct_from_52w_low": 5.0,
        "pct_from_52w_high": -53.3,
        "target_mean": 60.0,
        "target_upside_pct": 42.9,
        "analyst_ratings": {"buy": 5, "hold": 2},
        "buy_ratio_pct": 71.4,
        "market_cap": 5_000_000_000,
    }
    base.update(overrides)
    return base


# --- get_rating_timeline -----------------------------------------------


def test_get_rating_timeline_finds_52w_low_and_annotates_actions(monkeypatch):
    daily = [
        {"date": "2026-01-01", "close": 100.0},
        {"date": "2026-02-01", "close": 40.0},  # the 52-week low
        {"date": "2026-03-01", "close": 60.0},
    ]
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: daily)

    ud_df = pd.DataFrame(
        {
            "Firm": ["Big Bank", "Small Shop"],
            "ToGrade": ["Buy", "Sell"],
            "FromGrade": ["Hold", "Buy"],
            "Action": ["up", "down"],
        },
        index=pd.to_datetime(["2026-01-15", "2026-02-15"]),
    )
    ud_df.index.name = "GradeDate"

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_upgrades_downgrades(self, as_dict=False):
            return ud_df

    monkeypatch.setattr(nla.yf, "Ticker", FakeTicker)

    result = nla.get_rating_timeline("ACME")

    assert result["week_52_low"] == {"date": "2026-02-01", "close": 40.0}
    assert len(result["actions"]) == 2

    # Most recent first.
    assert result["actions"][0]["date"] == "2026-02-15"
    assert result["actions"][0]["firm"] == "Small Shop"
    assert result["actions"][0]["before_52w_low"] is False  # issued after the low
    assert result["actions"][0]["price_at_rating"] == 40.0  # closest bar on/before 2026-02-15
    assert result["actions"][0]["pct_move_since_rating"] == pytest.approx(50.0)  # 40 -> 60

    assert result["actions"][1]["date"] == "2026-01-15"
    assert result["actions"][1]["firm"] == "Big Bank"
    assert result["actions"][1]["before_52w_low"] is True  # issued before the low
    assert result["actions"][1]["price_at_rating"] == 100.0


def test_get_rating_timeline_handles_no_price_history(monkeypatch):
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: [])

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_upgrades_downgrades(self, as_dict=False):
            raise RuntimeError("No upgrade/downgrade history found")

    monkeypatch.setattr(nla.yf, "Ticker", FakeTicker)

    result = nla.get_rating_timeline("ACME")
    assert result == {"week_52_low": None, "actions": []}


def test_get_rating_timeline_caps_to_10_most_recent_actions(monkeypatch):
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: [])
    dates = pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 16)])
    ud_df = pd.DataFrame(
        {"Firm": [f"Firm{i}" for i in range(15)], "ToGrade": ["Buy"] * 15, "FromGrade": ["Hold"] * 15, "Action": ["up"] * 15},
        index=dates,
    )

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_upgrades_downgrades(self, as_dict=False):
            return ud_df

    monkeypatch.setattr(nla.yf, "Ticker", FakeTicker)

    result = nla.get_rating_timeline("ACME")
    assert len(result["actions"]) == 10
    assert result["actions"][0]["date"] == "2026-01-15"  # most recent kept


# --- build_context_user_message -----------------------------------------


def test_build_context_user_message_includes_key_facts():
    candidate = _candidate()
    rating_timeline = {
        "week_52_low": {"date": "2026-02-01", "close": 40.0},
        "actions": [
            {
                "date": "2026-02-15",
                "firm": "Small Shop",
                "to_grade": "Sell",
                "from_grade": "Buy",
                "action": "down",
                "before_52w_low": False,
                "price_at_rating": 40.0,
                "pct_move_since_rating": 50.0,
            }
        ],
    }
    headlines = [{"title": "Acme misses on guidance", "published_at": None}]
    message = nla.build_context_user_message("ACME", candidate, rating_timeline, headlines, 65.0)

    assert "ACME" in message
    assert "40.0" in message  # 52-week low close appears
    assert "Small Shop" in message
    assert "AFTER the 52-week low" in message
    assert "Acme misses on guidance" in message
    assert "65.0/100" in message


def test_build_context_user_message_handles_missing_data():
    candidate = _candidate()
    rating_timeline = {"week_52_low": None, "actions": []}
    message = nla.build_context_user_message("ACME", candidate, rating_timeline, [], None)
    assert "unknown (price history unavailable)" in message
    assert "(none found)" in message
    assert "not available" in message


# --- analyze_expert_take --------------------------------------------------


def test_analyze_expert_take_calls_ai_client(monkeypatch):
    captured = {}

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        captured["provider"] = provider
        captured["user_message"] = user_message
        return '{"analysis": "About 200 words here.", "verdict": "buy_opportunity"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", fake_call_provider)

    result = nla.analyze_expert_take(
        "ACME", _candidate(), {"week_52_low": None, "actions": []}, [], None, "anthropic", "claude-haiku-4-5", api_key="sk-test"
    )

    assert captured["provider"] == "anthropic"
    assert "ACME" in captured["user_message"]
    assert result["analysis"] == "About 200 words here."
    assert result["verdict"] == "buy_opportunity"
    assert result["disclaimer"] == "This is not investment advice."
    assert result["parse_error"] is False


def test_analyze_expert_take_degrades_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(nla.ai_client, "call_provider", lambda *a, **kw: "not valid json")
    result = nla.analyze_expert_take(
        "ACME", _candidate(), {"week_52_low": None, "actions": []}, [], None, "anthropic", "claude-haiku-4-5", api_key="sk-test"
    )
    assert result["parse_error"] is True
    assert result["verdict"] == "mixed"
    assert result["analysis"] == "not valid json"


def test_analyze_expert_take_propagates_api_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(nla.ai_client, "call_provider", boom)
    with pytest.raises(RuntimeError):
        nla.analyze_expert_take(
            "ACME", _candidate(), {"week_52_low": None, "actions": []}, [], None, "anthropic", "claude-haiku-4-5", api_key="sk-test"
        )


# --- run_expert_panel -----------------------------------------------------


def test_run_expert_panel_calls_all_five_personas(monkeypatch):
    calls = []

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        calls.append(system_prompt)
        return '{"take": "Some take.", "stance": "bullish"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", fake_call_provider)

    panel = nla.run_expert_panel(
        "ACME", _candidate(), {"week_52_low": None, "actions": []}, [], None, "anthropic", "claude-haiku-4-5", api_key="sk-test"
    )

    assert len(panel) == 5
    assert len(calls) == 5
    assert {p["persona"] for p in panel} == {"technical", "fundamental", "news_sentiment", "ratings_timing", "macro_risk"}
    for p in panel:
        assert p["take"] == "Some take."
        assert p["stance"] == "bullish"
        assert p["error"] is None


def test_run_expert_panel_isolates_one_persona_failure(monkeypatch):
    def flaky_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        if "ratings auditor" in system_prompt.lower():
            raise RuntimeError("rate limited")
        return '{"take": "Fine.", "stance": "neutral"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", flaky_call_provider)

    panel = nla.run_expert_panel(
        "ACME", _candidate(), {"week_52_low": None, "actions": []}, [], None, "anthropic", "claude-haiku-4-5", api_key="sk-test"
    )

    assert len(panel) == 5  # the other 4 still came back
    failed = next(p for p in panel if p["persona"] == "ratings_timing")
    assert failed["error"] == "rate limited"
    assert failed["take"] is None
    others = [p for p in panel if p["persona"] != "ratings_timing"]
    assert all(p["error"] is None and p["take"] == "Fine." for p in others)


def test_run_expert_panel_gives_each_persona_only_its_own_tailored_data(monkeypatch):
    """Each of the 5 is a genuinely separate inquiry -- it should only see
    the slice of data relevant to its own question, not the full shared
    context repeated five times with a different persona label."""
    captured = {}

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        for key, (label, prompt) in nla.PANEL_SYSTEM_PROMPTS.items():
            if prompt == system_prompt:
                captured[key] = user_message
        return '{"take": "x", "stance": "neutral"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", fake_call_provider)

    rating_timeline = {
        "week_52_low": {"date": "2026-02-01", "close": 40.0},
        "actions": [
            {
                "date": "2026-02-15",
                "firm": "Small Shop",
                "to_grade": "Sell",
                "from_grade": "Buy",
                "action": "down",
                "before_52w_low": False,
                "price_at_rating": 40.0,
                "pct_move_since_rating": 50.0,
            }
        ],
    }
    headlines = [{"title": "Acme misses on guidance", "published_at": None}]

    nla.run_expert_panel("ACME", _candidate(), rating_timeline, headlines, 65.0, "anthropic", "claude-haiku-4-5", api_key="sk-test")

    # Technical only gets price/range -- no headlines, no ratings, no macro.
    assert "Acme misses on guidance" not in captured["technical"]
    assert "Market cap" not in captured["technical"]
    assert "macro gate score" not in captured["technical"]
    assert "52-week range" in captured["technical"]

    # News/sentiment only gets headlines -- no price range, no ratings.
    assert "Acme misses on guidance" in captured["news_sentiment"]
    assert "52-week range" not in captured["news_sentiment"]
    assert "Market cap" not in captured["news_sentiment"]

    # Ratings-timing only gets the rating timeline -- no headlines, no market cap.
    assert "Small Shop" in captured["ratings_timing"]
    assert "Acme misses on guidance" not in captured["ratings_timing"]
    assert "Market cap" not in captured["ratings_timing"]

    # Fundamental only gets market cap/target/ratings -- no headlines, no price range.
    assert "Market cap" in captured["fundamental"]
    assert "Acme misses on guidance" not in captured["fundamental"]
    assert "52-week range" not in captured["fundamental"]

    # Macro only gets the macro score + distance from low -- nothing else.
    assert "65.0/100" in captured["macro_risk"]
    assert "Acme misses on guidance" not in captured["macro_risk"]
    assert "Market cap" not in captured["macro_risk"]
    assert "Small Shop" not in captured["macro_risk"]
