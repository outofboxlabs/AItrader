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
