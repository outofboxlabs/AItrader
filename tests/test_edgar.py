import pytest

from portfolio_monitor import edgar


@pytest.fixture(autouse=True)
def _reset_cik_cache(monkeypatch):
    """The ticker->CIK map is cached at module level across calls --
    reset it before every test so one test's fake data can't leak into
    another's."""
    monkeypatch.setattr(edgar, "_TICKER_TO_CIK", None)


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


_TICKER_MAP_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1341439, "ticker": "ORCL", "title": "Oracle Corp"},
}


def test_get_cik_map_builds_uppercase_zero_padded_mapping(monkeypatch):
    monkeypatch.setattr(edgar.requests, "get", lambda url, timeout=None, headers=None: _FakeResponse(_TICKER_MAP_PAYLOAD))
    mapping = edgar._get_cik_map()
    assert mapping["AAPL"] == "0000320193"
    assert mapping["ORCL"] == "0001341439"


def test_get_cik_map_is_cached_across_calls(monkeypatch):
    calls = []

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        return _FakeResponse(_TICKER_MAP_PAYLOAD)

    monkeypatch.setattr(edgar.requests, "get", fake_get)
    edgar._get_cik_map()
    edgar._get_cik_map()
    assert len(calls) == 1  # second call served from the in-memory cache


def _fake_submissions_payload():
    return {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "10-K"],
                "filingDate": ["2026-08-01", "2026-06-15", "2026-02-20"],
                "reportDate": ["2026-06-30", "", "2025-12-31"],
                "accessionNumber": ["0001341439-26-000050", "0001341439-26-000030", "0001341439-26-000010"],
                "primaryDocument": ["orcl-20260630.htm", "orcl-8k.htm", "orcl-20251231.htm"],
                "primaryDocDescription": ["10-Q", "8-K", "10-K"],
            }
        }
    }


def test_get_recent_filings_returns_normalized_shape(monkeypatch):
    def fake_get(url, timeout=None, headers=None):
        if "company_tickers" in url:
            return _FakeResponse(_TICKER_MAP_PAYLOAD)
        return _FakeResponse(_fake_submissions_payload())

    monkeypatch.setattr(edgar.requests, "get", fake_get)

    filings = edgar.get_recent_filings("orcl", limit=8)
    assert len(filings) == 3
    assert filings[0]["form"] == "10-Q"
    assert filings[0]["filed"] == "2026-08-01"
    assert filings[0]["url"] == "https://www.sec.gov/Archives/edgar/data/1341439/000134143926000050/orcl-20260630.htm"


def test_get_recent_filings_respects_limit(monkeypatch):
    def fake_get(url, timeout=None, headers=None):
        if "company_tickers" in url:
            return _FakeResponse(_TICKER_MAP_PAYLOAD)
        return _FakeResponse(_fake_submissions_payload())

    monkeypatch.setattr(edgar.requests, "get", fake_get)
    filings = edgar.get_recent_filings("ORCL", limit=2)
    assert len(filings) == 2


def test_get_recent_filings_returns_empty_for_unknown_ticker(monkeypatch):
    monkeypatch.setattr(edgar.requests, "get", lambda url, timeout=None, headers=None: _FakeResponse(_TICKER_MAP_PAYLOAD))
    assert edgar.get_recent_filings("NOTREAL") == []


def test_get_recent_filings_returns_empty_on_network_failure(monkeypatch):
    def boom(url, timeout=None, headers=None):
        raise RuntimeError("network error")

    monkeypatch.setattr(edgar.requests, "get", boom)
    assert edgar.get_recent_filings("ORCL") == []


def test_get_recent_filings_handles_missing_filings_key(monkeypatch):
    def fake_get(url, timeout=None, headers=None):
        if "company_tickers" in url:
            return _FakeResponse(_TICKER_MAP_PAYLOAD)
        return _FakeResponse({})  # malformed/empty submissions response

    monkeypatch.setattr(edgar.requests, "get", fake_get)
    assert edgar.get_recent_filings("ORCL") == []
