from portfolio_monitor import price_targets as pt


class _FakeResponse:
    def __init__(self, payload, status_ok=True, status_code=200, text=""):
        self._payload = payload
        self._status_ok = status_ok
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _record(published_date="2026-08-14", price_target=250.0, price_when_posted=200.0, company="Morgan Stanley", analyst_name="Jane Doe"):
    return {
        "publishedDate": published_date,
        "priceTarget": price_target,
        "priceWhenPosted": price_when_posted,
        "analystCompany": company,
        "analystName": analyst_name,
        "newsTitle": "Some headline",
        "newsURL": "https://example.com/news",
    }


def test_get_price_target_history_returns_none_without_api_key():
    assert pt.get_price_target_history("ORCL", api_key="") is None
    assert pt.get_price_target_history("ORCL", api_key=None) is None


def test_get_price_target_history_returns_normalized_shape(monkeypatch):
    rec = _record()
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse([rec])

    monkeypatch.setattr(pt.requests, "get", fake_get)

    targets = pt.get_price_target_history("ORCL", api_key="test-key")
    assert captured["url"] == pt.BASE_URL
    assert captured["params"]["symbol"] == "ORCL"
    assert captured["params"]["apikey"] == "test-key"
    assert len(targets) == 1
    t = targets[0]
    assert t["published_date"] == "2026-08-14"
    assert t["target_date"] == "2027-08-14"
    assert t["analyst_company"] == "Morgan Stanley"
    assert t["analyst_name"] == "Jane Doe"
    assert t["price_target"] == 250.0
    assert t["price_when_posted"] == 200.0
    assert t["implied_pct_change"] == 25.0


def test_get_price_target_history_handles_missing_price_when_posted(monkeypatch):
    rec = _record(price_when_posted=None)
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse([rec]))
    targets = pt.get_price_target_history("ORCL", api_key="test-key")
    assert targets[0]["implied_pct_change"] is None


def test_get_price_target_history_skips_records_without_a_price_target(monkeypatch):
    no_target = _record()
    no_target["priceTarget"] = None
    good = _record()
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse([no_target, good]))
    targets = pt.get_price_target_history("ORCL", api_key="test-key")
    assert len(targets) == 1


def test_get_price_target_history_skips_unparseable_dates(monkeypatch):
    bad = _record(published_date="not-a-date")
    good = _record()
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse([bad, good]))
    targets = pt.get_price_target_history("ORCL", api_key="test-key")
    assert len(targets) == 1


def test_get_price_target_history_returns_none_on_network_failure(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise RuntimeError("network error")

    monkeypatch.setattr(pt.requests, "get", boom)
    assert pt.get_price_target_history("ORCL", api_key="test-key") is None


def test_get_price_target_history_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse(None, status_ok=False, status_code=403, text="plan upgrade required"))
    assert pt.get_price_target_history("ORCL", api_key="test-key") is None


def test_get_price_target_history_returns_none_on_unexpected_shape(monkeypatch):
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse({"Error Message": "Invalid API KEY"}))
    assert pt.get_price_target_history("ORCL", api_key="bad-key") is None


def test_get_price_target_history_returns_empty_list_when_none_found(monkeypatch):
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse([]))
    assert pt.get_price_target_history("ORCL", api_key="test-key") == []
