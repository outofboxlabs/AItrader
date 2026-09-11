from portfolio_monitor import forex_calendar as fx


def test_parse_numeric_plain_number():
    assert fx._parse_numeric("3.1") == 3.1


def test_parse_numeric_percent_sign():
    assert fx._parse_numeric("3.1%") == 3.1


def test_parse_numeric_thousand_separator():
    assert fx._parse_numeric("1,250") == 1250.0


def test_parse_numeric_k_suffix():
    assert fx._parse_numeric("227K") == 227_000.0


def test_parse_numeric_m_suffix():
    assert fx._parse_numeric("1.5M") == 1_500_000.0


def test_parse_numeric_none_or_empty():
    assert fx._parse_numeric(None) is None
    assert fx._parse_numeric("") is None


def test_parse_numeric_non_numeric_text():
    assert fx._parse_numeric("n/a") is None


def test_surprise_pct_computes_percentage():
    assert fx._surprise_pct("110", "100") == 10.0
    assert fx._surprise_pct("90", "100") == -10.0


def test_surprise_pct_none_when_not_numeric():
    assert fx._surprise_pct(None, "100") is None
    assert fx._surprise_pct("110", None) is None


def test_surprise_pct_none_when_forecast_zero():
    assert fx._surprise_pct("10", "0") is None


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_calendar_events_normalizes_shape(monkeypatch):
    raw = [
        {
            "title": "Non-Farm Payrolls",
            "country": "USD",
            "date": "2026-09-11T08:30:00-04:00",
            "impact": "High",
            "forecast": "180K",
            "previous": "150K",
            "actual": "227K",
        },
        {
            "title": "Bank Holiday",
            "country": "GBP",
            "date": "2026-09-12T00:00:00-04:00",
            "impact": "Holiday",
            "forecast": "",
            "previous": "",
            "actual": "",
        },
    ]

    def fake_get(url, timeout=None, headers=None):
        return _FakeResponse(raw)

    monkeypatch.setattr(fx.requests, "get", fake_get)

    events = fx.fetch_calendar_events()

    assert len(events) == 2
    nfp = events[0]
    assert nfp["title"] == "Non-Farm Payrolls"
    assert nfp["country"] == "USD"
    assert nfp["impact"] == "High"
    assert round(nfp["surprise_pct"], 2) == round((227_000 - 180_000) / 180_000 * 100, 2)

    holiday = events[1]
    assert holiday["surprise_pct"] is None


def test_fetch_calendar_events_date_field_is_a_ready_to_use_iso_string(monkeypatch):
    """Regression test using the actual shape confirmed from a real pull
    of the feed (pasted by a user straight from the JSON in their
    browser): "date" is already a complete ISO 8601 string with a UTC
    offset baked in -- no separate timestamp field, no extra parsing
    needed. An earlier guess that this needed a Unix-timestamp fallback
    was chasing a field ("dateline") that doesn't exist in this feed."""
    raw = [
        {
            "title": "German Industrial Production m/m",
            "country": "EUR",
            "date": "2026-09-07T02:00:00-04:00",
            "impact": "Low",
            "forecast": "0.1%",
            "previous": "0.2%",
        }
    ]

    def fake_get(url, timeout=None, headers=None):
        return _FakeResponse(raw)

    monkeypatch.setattr(fx.requests, "get", fake_get)

    events = fx.fetch_calendar_events()

    assert events[0]["date"] == "2026-09-07T02:00:00-04:00"


# --- portfolio impact analysis --------------------------------------------


def test_build_portfolio_impact_user_message_includes_event_and_positions():
    event = {
        "title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00",
        "impact": "High", "previous": "0.1%", "forecast": "0.4%", "actual": "0.4%", "surprise_pct": 0.0,
    }
    positions = [
        {"asset_type": "shares", "ticker": "AAPL", "contracts": 10},
        {"asset_type": "option", "ticker": "TSLA", "option_type": "put", "strike": 180.0, "expiry": "2025-05-16"},
    ]
    message = fx.build_portfolio_impact_user_message(event, positions)
    assert "CPI m/m" in message
    assert "AAPL" in message
    assert "TSLA" in message
    assert "put" in message


def test_build_portfolio_impact_user_message_handles_no_positions():
    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    message = fx.build_portfolio_impact_user_message(event, [])
    assert "no positions saved" in message.lower()


def test_analyze_portfolio_impact_calls_ai_client(monkeypatch):
    captured = {}

    def fake_call_provider(provider, system_prompt, user_message, model, api_key=None, max_tokens=800):
        captured["provider"] = provider
        captured["model"] = model
        captured["api_key"] = api_key
        return '{"impact_summary": "Could pressure rate-sensitive names.", "affected_tickers": ["AAPL"]}'

    monkeypatch.setattr(fx.ai_client, "call_provider", fake_call_provider)

    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    result = fx.analyze_portfolio_impact(
        event, [{"asset_type": "shares", "ticker": "AAPL", "contracts": 10}], "anthropic", "claude-haiku-4-5", api_key="sk-test"
    )

    assert captured["provider"] == "anthropic"
    assert captured["api_key"] == "sk-test"
    assert result["impact_summary"] == "Could pressure rate-sensitive names."
    assert result["affected_tickers"] == ["AAPL"]
    assert result["disclaimer"] == "This is not investment advice."
    assert result["parse_error"] is False


def test_analyze_portfolio_impact_degrades_on_unparseable_response(monkeypatch):
    monkeypatch.setattr(fx.ai_client, "call_provider", lambda *a, **kw: "not valid json at all")

    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    result = fx.analyze_portfolio_impact(event, [], "anthropic", "claude-haiku-4-5", api_key="sk-test")

    assert result["parse_error"] is True
    assert result["affected_tickers"] == []
    assert result["disclaimer"] == "This is not investment advice."


def test_analyze_portfolio_impact_propagates_api_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(fx.ai_client, "call_provider", boom)

    event = {"title": "CPI m/m", "country": "USD", "date": "2026-09-11T08:30:00-04:00", "impact": "High"}
    try:
        fx.analyze_portfolio_impact(event, [], "anthropic", "claude-haiku-4-5", api_key="sk-test")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError:
        pass
