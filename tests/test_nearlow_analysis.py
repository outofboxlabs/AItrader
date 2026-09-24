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


def _context(**overrides):
    ctx = {
        "candidate": _candidate(),
        "rating_timeline": {"week_52_low": None, "actions": []},
        "headlines": [],
        "macro_score": None,
    }
    ctx.update(overrides)
    return ctx


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
    assert result["actions"][0]["pct_above_low_at_rating"] == pytest.approx(0.0)  # already at the eventual low

    assert result["actions"][1]["date"] == "2026-01-15"
    assert result["actions"][1]["firm"] == "Big Bank"
    assert result["actions"][1]["before_52w_low"] is True  # issued before the low
    assert result["actions"][1]["price_at_rating"] == 100.0
    assert result["actions"][1]["pct_above_low_at_rating"] == pytest.approx(150.0)  # was performing well at the time


def test_get_rating_timeline_reuses_passed_in_daily_history(monkeypatch):
    """When the caller already fetched daily history (e.g. for the
    technical persona), get_rating_timeline should use it directly
    instead of pulling it a second time."""
    calls = []
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: calls.append(ticker) or [])
    monkeypatch.setattr(nla.yf, "Ticker", lambda ticker: type("T", (), {"get_upgrades_downgrades": lambda self, as_dict=False: None})())

    daily = [{"date": "2026-01-01", "close": 40.0}]
    result = nla.get_rating_timeline("ACME", daily_history=daily)

    assert calls == []  # never called get_daily_price_history itself
    assert result["week_52_low"] == {"date": "2026-01-01", "close": 40.0}


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


# --- get_institutional_activity ------------------------------------------


def test_get_institutional_activity_classifies_direction_and_annotates_price(monkeypatch):
    daily = [
        {"date": "2026-01-01", "close": 100.0},
        {"date": "2026-02-01", "close": 40.0},
        {"date": "2026-02-15", "close": 55.0},
    ]
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: daily)

    holders_df = pd.DataFrame(
        {
            "Date Reported": pd.to_datetime(["2026-02-01", "2026-02-01", "2026-02-01"]),
            "Holder": ["Big Fund LP", "Shrinking Fund LP", "New Fund LP"],
            "Shares": [1_000_000, 200_000, 50_000],
            "Value": [55_000_000, 11_000_000, 2_750_000],
            "pctHeld": [0.05, 0.01, 0.002],
            "pctChange": [0.12, -0.08, float("nan")],
        }
    )

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_institutional_holders(self, as_dict=False):
            return holders_df

    monkeypatch.setattr(nla.yf, "Ticker", FakeTicker)

    result = nla.get_institutional_activity("ACME")
    holders = {h["holder"]: h for h in result["holders"]}

    assert holders["Big Fund LP"]["direction"] == "increased"
    assert holders["Big Fund LP"]["pct_change"] == pytest.approx(0.12)
    assert holders["Big Fund LP"]["price_at_report"] == 40.0  # closest bar on/before 2026-02-01

    assert holders["Shrinking Fund LP"]["direction"] == "decreased"
    assert holders["New Fund LP"]["direction"] == "new"  # NaN pctChange -> treated as a new position
    assert holders["New Fund LP"]["pct_change"] is None


def test_get_institutional_activity_strips_trailing_whitespace_from_holder_name(monkeypatch):
    """Yahoo's own institutional-holders data has inconsistent trailing
    whitespace on some organization names (observed live: "Nvidia Corp "
    with a trailing space, alongside clean names like "Blackrock Inc.")
    -- it should be stripped so it doesn't show up as a stray space
    before the colon in the legend."""
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: [])
    holders_df = pd.DataFrame(
        {
            "Date Reported": pd.to_datetime(["2026-06-30"]),
            "Holder": ["Nvidia Corp "],
            "Shares": [47_213_353],
            "Value": [1_000_000],
            "pctHeld": [0.01],
            "pctChange": [0.0],
        }
    )
    monkeypatch.setattr(
        nla.yf, "Ticker", lambda ticker: type("T", (), {"get_institutional_holders": lambda self, as_dict=False: holders_df})()
    )

    result = nla.get_institutional_activity("ACME")
    assert result["holders"][0]["holder"] == "Nvidia Corp"


def test_get_institutional_activity_reuses_passed_in_daily_history(monkeypatch):
    calls = []
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: calls.append(ticker) or [])
    monkeypatch.setattr(
        nla.yf, "Ticker", lambda ticker: type("T", (), {"get_institutional_holders": lambda self, as_dict=False: None})()
    )

    result = nla.get_institutional_activity("ACME", daily_history=[{"date": "2026-01-01", "close": 40.0}])
    assert calls == []
    assert result == {"holders": []}


def test_get_institutional_activity_handles_no_holder_data(monkeypatch):
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: [])

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_institutional_holders(self, as_dict=False):
            raise RuntimeError("No holders found")

    monkeypatch.setattr(nla.yf, "Ticker", FakeTicker)

    result = nla.get_institutional_activity("ACME")
    assert result == {"holders": []}


def test_get_institutional_activity_sorts_most_recent_first(monkeypatch):
    monkeypatch.setattr(nla.data_mod, "get_daily_price_history", lambda ticker, **kw: [])
    holders_df = pd.DataFrame(
        {
            "Date Reported": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "Holder": ["Older Fund", "Newer Fund"],
            "Shares": [100, 200],
            "Value": [1000, 2000],
            "pctHeld": [0.01, 0.02],
            "pctChange": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(
        nla.yf, "Ticker", lambda ticker: type("T", (), {"get_institutional_holders": lambda self, as_dict=False: holders_df})()
    )

    result = nla.get_institutional_activity("ACME")
    assert [h["holder"] for h in result["holders"]] == ["Newer Fund", "Older Fund"]
    assert result["holders"][0]["direction"] == "unchanged"


# --- build_context_user_message -----------------------------------------


def test_build_context_user_message_includes_key_facts():
    context = _context(
        rating_timeline={
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
                    "pct_above_low_at_rating": 0.0,
                }
            ],
        },
        headlines=[{"title": "Acme misses on guidance", "published_at": None}],
        macro_score=65.0,
    )
    message = nla.build_context_user_message("ACME", context)

    assert "ACME" in message
    assert "40.0" in message  # 52-week low close appears
    assert "Small Shop" in message
    assert "AFTER the 52-week low" in message
    assert "0% above what would become its 52-week low when issued" in message
    assert "Acme misses on guidance" in message
    assert "65.0/100" in message


def test_build_context_user_message_handles_missing_data():
    message = nla.build_context_user_message("ACME", _context())
    assert "unknown (price history unavailable)" in message
    assert "(none found)" in message
    assert "not available" in message


def test_format_rating_action_line_states_pct_above_low_plainly():
    action = {
        "date": "2026-01-15",
        "firm": "Big Bank",
        "to_grade": "Buy",
        "from_grade": "Hold",
        "action": "up",
        "before_52w_low": True,
        "price_at_rating": 44.0,
        "pct_move_since_rating": -4.5,
        "pct_above_low_at_rating": 10.0,
    }
    line = nla._format_rating_action_line(action)
    assert "Big Bank" in line
    assert "BEFORE the 52-week low" in line
    assert "was 10% above what would become its 52-week low when issued" in line
    assert "moved -4.5% since" in line


def test_format_rating_action_line_omits_pct_above_low_when_missing():
    action = {"date": "2026-01-15", "firm": "Big Bank", "to_grade": "Buy", "from_grade": "Hold", "action": "up"}
    line = nla._format_rating_action_line(action)
    assert "above what would become its 52-week low" not in line


# --- analyze_expert_take --------------------------------------------------


def test_analyze_expert_take_calls_ai_client(monkeypatch):
    captured = {}

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        captured["provider"] = provider
        captured["user_message"] = user_message
        return '{"analysis": "About 200 words here.", "verdict": "buy_opportunity"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", fake_call_provider)

    result = nla.analyze_expert_take("ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test")

    assert captured["provider"] == "anthropic"
    assert "ACME" in captured["user_message"]
    assert result["analysis"] == "About 200 words here."
    assert result["verdict"] == "buy_opportunity"
    assert result["disclaimer"] == "This is not investment advice."
    assert result["parse_error"] is False


def test_analyze_expert_take_degrades_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(nla.ai_client, "call_provider", lambda *a, **kw: "not valid json")
    result = nla.analyze_expert_take("ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test")
    assert result["parse_error"] is True
    assert result["verdict"] == "mixed"
    assert result["analysis"] == "not valid json"


def test_analyze_expert_take_propagates_api_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(nla.ai_client, "call_provider", boom)
    with pytest.raises(RuntimeError):
        nla.analyze_expert_take("ACME", _context(), "anthropic", "claude-haiku-4-5", api_key="sk-test")


# --- run_expert_panel -----------------------------------------------------

ALL_PERSONAS = ["technical", "fundamental", "news", "ratings_timing", "macro_risk", "filings", "social_sentiment", "price_targets"]


def test_run_expert_panel_calls_only_selected_personas(monkeypatch):
    calls = []

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        calls.append(system_prompt)
        return '{"take": "Some take.", "stance": "bullish"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", fake_call_provider)

    panel = nla.run_expert_panel("ACME", _context(), ["technical", "news"], "anthropic", "claude-haiku-4-5", api_key="sk-test")

    assert len(panel) == 2
    assert len(calls) == 2
    assert {p["persona"] for p in panel} == {"technical", "news"}
    for p in panel:
        assert p["take"] == "Some take."
        assert p["stance"] == "bullish"
        assert p["error"] is None


def test_run_expert_panel_supports_all_seven_personas(monkeypatch):
    monkeypatch.setattr(nla.ai_client, "call_provider", lambda *a, **kw: '{"take": "x", "stance": "neutral"}')
    panel = nla.run_expert_panel("ACME", _context(), ALL_PERSONAS, "anthropic", "claude-haiku-4-5", api_key="sk-test")
    assert {p["persona"] for p in panel} == set(ALL_PERSONAS)


def test_run_expert_panel_skips_unknown_persona_keys(monkeypatch):
    monkeypatch.setattr(nla.ai_client, "call_provider", lambda *a, **kw: '{"take": "x", "stance": "neutral"}')
    panel = nla.run_expert_panel("ACME", _context(), ["technical", "not-a-real-persona"], "anthropic", "claude-haiku-4-5", api_key="sk-test")
    assert len(panel) == 1
    assert panel[0]["persona"] == "technical"


def test_run_expert_panel_isolates_one_persona_failure(monkeypatch):
    def flaky_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        if "ratings auditor" in system_prompt.lower():
            raise RuntimeError("rate limited")
        return '{"take": "Fine.", "stance": "neutral"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", flaky_call_provider)

    panel = nla.run_expert_panel("ACME", _context(), ALL_PERSONAS, "anthropic", "claude-haiku-4-5", api_key="sk-test")

    assert len(panel) == len(ALL_PERSONAS)  # the others still came back
    failed = next(p for p in panel if p["persona"] == "ratings_timing")
    assert failed["error"] == "rate limited"
    assert failed["take"] is None
    others = [p for p in panel if p["persona"] != "ratings_timing"]
    assert all(p["error"] is None and p["take"] == "Fine." for p in others)


def test_run_expert_panel_gives_each_persona_only_its_own_tailored_data(monkeypatch):
    """Each persona is a genuinely separate inquiry -- it should only see
    the slice of data relevant to its own question, not the full shared
    context repeated for every persona with a different label."""
    captured = {}

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        for key, (label, prompt) in nla.PANEL_SYSTEM_PROMPTS.items():
            if prompt == system_prompt:
                captured[key] = user_message
        return '{"take": "x", "stance": "neutral"}'

    monkeypatch.setattr(nla.ai_client, "call_provider", fake_call_provider)

    context = _context(
        rating_timeline={
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
        },
        headlines=[{"title": "Acme misses on guidance", "published_at": None}],
        macro_score=65.0,
        technical_indicators={"sma_20": 41.0, "price_vs_sma_20_pct": 2.4, "rsi_14": 55.0},
        financials={"fiscal_year_end": "2026-01-31", "revenue": 1000.0, "net_income": 100.0},
        filings=[{"form": "10-K", "filed": "2026-02-01", "report_date": "2025-12-31", "url": "https://sec.gov/x"}],
        social_sentiment=[
            {"body": "ACME to the moon", "username": "trader1", "sentiment": "Bullish", "likes": 120, "created_at": "2026-09-14T12:00:00+00:00"}
        ],
        social_sentiment_window="week",
        price_targets={
            "target_high": 70.0,
            "target_low": 50.0,
            "target_mean": 60.0,
            "target_median": 60.0,
            "target_date": "2027-08-14",
        },
    )

    nla.run_expert_panel("ACME", context, ALL_PERSONAS, "anthropic", "claude-haiku-4-5", api_key="sk-test")

    # Technical: price/range + real indicators -- no headlines, no market cap, no macro.
    assert "52-week range" in captured["technical"]
    assert "SMA(20)" in captured["technical"]
    assert "RSI(14)" in captured["technical"]
    assert "Acme misses on guidance" not in captured["technical"]
    assert "Market cap" not in captured["technical"]
    assert "macro gate score" not in captured["technical"]

    # News: only headlines.
    assert "Acme misses on guidance" in captured["news"]
    assert "52-week range" not in captured["news"]
    assert "Market cap" not in captured["news"]

    # Ratings-timing: only the rating timeline.
    assert "Small Shop" in captured["ratings_timing"]
    assert "Acme misses on guidance" not in captured["ratings_timing"]
    assert "Market cap" not in captured["ratings_timing"]

    # Fundamental: market cap/target/ratings + real financials -- no headlines, no price range.
    assert "Market cap" in captured["fundamental"]
    assert "Revenue: 1,000" in captured["fundamental"]
    assert "Acme misses on guidance" not in captured["fundamental"]
    assert "52-week range" not in captured["fundamental"]

    # Macro: only the macro score + distance from low.
    assert "65.0/100" in captured["macro_risk"]
    assert "Acme misses on guidance" not in captured["macro_risk"]
    assert "Market cap" not in captured["macro_risk"]
    assert "Small Shop" not in captured["macro_risk"]

    # Filings: only the filings list.
    assert "10-K" in captured["filings"]
    assert "https://sec.gov/x" in captured["filings"]
    assert "Acme misses on guidance" not in captured["filings"]

    # Social sentiment: only the StockTwits messages + window.
    assert "past week" in captured["social_sentiment"]
    assert "ACME to the moon" in captured["social_sentiment"]
    assert "trader1" in captured["social_sentiment"]
    assert "Market cap" not in captured["social_sentiment"]
    assert "Acme misses on guidance" not in captured["social_sentiment"]

    # Price targets: only the analyst price-target snapshot.
    assert "60.0" in captured["price_targets"]
    assert "2027-08-14" in captured["price_targets"]
    assert "Market cap" not in captured["price_targets"]
    assert "Acme misses on guidance" not in captured["price_targets"]


def test_filings_lines_reports_when_none_found():
    lines = nla._filings_lines("ACME", _context(filings=[]))
    assert any("none found" in line for line in lines)


def test_social_sentiment_lines_reports_when_no_data():
    lines = nla._social_sentiment_lines("ACME", _context(social_sentiment=None, social_sentiment_window="hour"))
    assert any("no messages found" in line for line in lines)
    assert any("past hour" in line for line in lines)


def test_social_sentiment_lines_lists_real_posts():
    messages = [{"body": "ACME earnings beat", "username": "trader2", "sentiment": "Bullish", "likes": 50, "created_at": "2026-09-14T08:00:00+00:00"}]
    lines = nla._social_sentiment_lines("ACME", _context(social_sentiment=messages, social_sentiment_window="today"))
    joined = "\n".join(lines)
    assert "ACME earnings beat" in joined
    assert "@trader2" in joined
    assert "Bullish" in joined
    assert "50 likes" in joined
    assert any("mentioning this ticker today" in line for line in lines)


def test_social_sentiment_lines_reports_totals_and_truncates_long_lists():
    messages = [
        {"body": f"msg {i}", "username": f"trader{i}", "sentiment": "Bullish" if i % 2 == 0 else "Bearish", "likes": i, "created_at": "2026-09-14T08:00:00+00:00"}
        for i in range(80)
    ]
    lines = nla._social_sentiment_lines("ACME", _context(social_sentiment=messages, social_sentiment_window="year"))
    joined = "\n".join(lines)
    assert "Totals across all 80 messages: 40 Bullish, 40 Bearish, 0 untagged." in joined
    assert "msg 0" in joined
    assert "msg 59" in joined
    assert "msg 60" not in joined  # beyond the per-prompt cap
    assert "20 more messages omitted" in joined


def test_price_target_lines_reports_when_none_found():
    lines = nla._price_target_lines("ACME", _context(price_targets=None))
    assert any("no analyst price-target data found" in line for line in lines)


def test_price_target_lines_lists_snapshot():
    snapshot = {
        "target_high": 70.0,
        "target_low": 50.0,
        "target_mean": 60.0,
        "target_median": 60.0,
        "target_date": "2027-08-14",
    }
    lines = nla._price_target_lines("ACME", _context(price_targets=snapshot))
    joined = "\n".join(lines)
    assert "$60.0" in joined
    assert "$50.0-$70.0" in joined
    assert "2027-08-14" in joined
    assert "12 months" in joined


def test_technical_lines_reports_when_indicators_unavailable():
    lines = nla._price_range_lines("ACME", _context(technical_indicators={}))
    assert any("not enough price history available" in line for line in lines)


def test_fundamental_lines_reports_when_financials_unavailable():
    lines = nla._fundamental_lines("ACME", _context(financials=None))
    assert any("not available for this ticker" in line for line in lines)


def test_fundamental_lines_never_mentions_analyst_target_or_ratings():
    """Regression guard: this persona used to include the candidate's
    analyst mean target and ratings breakdown, which let it fall back on
    "analysts say N% upside" reasoning instead of the actual business --
    observed in practice giving a pre-revenue biotech with no financials
    a "significantly undervalued" bullish call based purely on its price
    target. The candidate here still carries target_mean/buy_ratio_pct
    (as every real candidate does), so this only passes if the function
    itself never reads them."""
    lines = nla._fundamental_lines("ACME", _context(financials=None))
    text = " ".join(lines).lower()
    assert "target" not in text
    assert "rating" not in text
    assert "analyst" not in text
    assert "upside" not in text


def test_fundamental_lines_reports_balance_sheet_and_cash_flow_fields():
    financials = {
        "fiscal_year_end": "2026-01-31",
        "revenue": 0,
        "revenue_yoy_pct": None,
        "net_income": None,
        "gross_margin_pct": None,
        "cash": 40_000_000.0,
        "total_debt": 5_000_000.0,
        "operating_cash_flow": -18_000_000.0,
        "free_cash_flow": -20_000_000.0,
        "cash_runway_quarters": 8.0,
        "shares_outstanding": 110_000_000.0,
        "shares_outstanding_yoy_pct": 10.0,
    }
    lines = nla._fundamental_lines("ACME", _context(financials=financials))
    text = " ".join(lines)
    assert "$0 reported (confirmed pre-revenue)" in text
    assert "Cash & equivalents: 40,000,000" in text
    assert "Total debt: 5,000,000" in text
    assert "Free cash flow (annual): -20,000,000" in text
    assert "cash runway at current burn rate: 8.0 quarters" in text
    assert "Shares outstanding: 110,000,000, +10.0% YoY" in text


def test_fundamental_lines_never_calls_missing_revenue_pre_revenue():
    """Regression guard for the exact bug a user hit live: a real,
    revenue-generating company (a health insurer) got called
    "pre-revenue" because its income statement didn't have a "Total
    Revenue" line yfinance recognized -- i.e. missing DATA, not a
    confirmed $0. Only a confirmed 0 may be called pre-revenue."""
    financials = {"fiscal_year_end": "2026-01-31", "revenue": None, "net_income": 500_000_000.0}
    lines = nla._fundamental_lines("ALHC", _context(financials=financials))
    text = " ".join(lines)
    assert "confirmed pre-revenue" not in text.lower()
    assert "not available" in text
    assert "does NOT by itself mean pre-revenue" in text


def test_fundamental_lines_reports_positive_revenue_normally():
    financials = {"fiscal_year_end": "2026-01-31", "revenue": 2_700_000_000.0, "revenue_yoy_pct": 15.0}
    lines = nla._fundamental_lines("ALHC", _context(financials=financials))
    text = " ".join(lines)
    assert "Revenue: 2,700,000,000, +15.0% YoY" in text
    assert "pre-revenue" not in text.lower()


def test_fundamental_lines_reports_peer_comparison_when_available():
    financials = {"fiscal_year_end": "2026-01-31", "revenue": 2_700_000_000.0}
    peer_comparison = {
        "sector": "Healthcare",
        "industry": "Healthcare Plans",
        "target_pe": 18.5,
        "peer_avg_pe": 22.0,
        "peers": [
            {"ticker": "UNH", "name": "UnitedHealth Group", "market_cap": 400_000_000_000, "pe": 20.0},
            {"ticker": "HUM", "name": "Humana Inc.", "market_cap": 30_000_000_000, "pe": 24.0},
        ],
    }
    lines = nla._fundamental_lines("ALHC", _context(financials=financials, peer_comparison=peer_comparison))
    text = " ".join(lines)
    assert "Healthcare Plans" in text
    assert "trailing P/E: 18.5" in text
    assert "Average trailing P/E of the 2 largest same-industry peers" in text
    assert "UNH (UnitedHealth Group): P/E 20.0" in text
    assert "HUM (Humana Inc.): P/E 24.0" in text


def test_fundamental_lines_reports_when_peer_comparison_unavailable():
    financials = {"fiscal_year_end": "2026-01-31", "revenue": 2_700_000_000.0}
    lines = nla._fundamental_lines("ALHC", _context(financials=financials, peer_comparison=None))
    assert any("Peer/industry P/E comparison: not available" in line for line in lines)
