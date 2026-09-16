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


CONSENSUS_RECORD = {"symbol": "AAPL", "targetHigh": 400, "targetLow": 245, "targetConsensus": 339.35, "targetMedian": 360}
SUMMARY_RECORD = {
    "symbol": "AAPL",
    "lastMonthCount": 5,
    "lastMonthAvgPriceTarget": 345.0,
    "lastQuarterCount": 12,
    "lastQuarterAvgPriceTarget": 330.0,
    "lastYearCount": 40,
    "lastYearAvgPriceTarget": 300.0,
    "allTimeCount": 100,
    "allTimeAvgPriceTarget": 250.0,
}


def test_get_price_target_snapshot_returns_none_without_api_key():
    assert pt.get_price_target_snapshot("AAPL", api_key="") is None
    assert pt.get_price_target_snapshot("AAPL", api_key=None) is None


def test_get_price_target_snapshot_returns_normalized_shape(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.setdefault("urls", []).append(url)
        captured.setdefault("params", []).append(params)
        if url == pt.CONSENSUS_URL:
            return _FakeResponse([CONSENSUS_RECORD])
        return _FakeResponse([SUMMARY_RECORD])

    monkeypatch.setattr(pt.requests, "get", fake_get)

    snapshot = pt.get_price_target_snapshot("AAPL", api_key="test-key")
    assert pt.CONSENSUS_URL in captured["urls"]
    assert pt.SUMMARY_URL in captured["urls"]
    assert all(p["symbol"] == "AAPL" and p["apikey"] == "test-key" for p in captured["params"])

    assert snapshot["target_high"] == 400
    assert snapshot["target_low"] == 245
    assert snapshot["target_consensus"] == 339.35
    assert snapshot["target_median"] == 360
    assert snapshot["target_date"] is not None

    windows = {w["window"]: w for w in snapshot["trailing_windows"]}
    assert windows["lastMonth"]["avg_price_target"] == 345.0
    assert windows["lastMonth"]["count"] == 5
    assert windows["allTime"]["avg_price_target"] == 250.0


def test_get_price_target_snapshot_returns_none_when_consensus_fails(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url == pt.CONSENSUS_URL:
            raise RuntimeError("network error")
        return _FakeResponse([SUMMARY_RECORD])

    monkeypatch.setattr(pt.requests, "get", fake_get)
    assert pt.get_price_target_snapshot("AAPL", api_key="test-key") is None


def test_get_price_target_snapshot_still_returns_consensus_when_summary_fails(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url == pt.CONSENSUS_URL:
            return _FakeResponse([CONSENSUS_RECORD])
        raise RuntimeError("network error")

    monkeypatch.setattr(pt.requests, "get", fake_get)
    snapshot = pt.get_price_target_snapshot("AAPL", api_key="test-key")
    assert snapshot["target_consensus"] == 339.35
    assert snapshot["trailing_windows"] == []


def test_get_price_target_snapshot_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse(None, status_ok=False, status_code=403))
    assert pt.get_price_target_snapshot("AAPL", api_key="test-key") is None


def test_get_price_target_snapshot_returns_none_on_empty_list(monkeypatch):
    monkeypatch.setattr(pt.requests, "get", lambda url, params=None, timeout=None: _FakeResponse([]))
    assert pt.get_price_target_snapshot("AAPL", api_key="test-key") is None


def test_get_price_target_snapshot_handles_missing_trailing_window_fields(monkeypatch):
    partial_summary = {"symbol": "AAPL", "lastMonthCount": 5, "lastMonthAvgPriceTarget": 345.0}

    def fake_get(url, params=None, timeout=None):
        if url == pt.CONSENSUS_URL:
            return _FakeResponse([CONSENSUS_RECORD])
        return _FakeResponse([partial_summary])

    monkeypatch.setattr(pt.requests, "get", fake_get)
    snapshot = pt.get_price_target_snapshot("AAPL", api_key="test-key")
    assert len(snapshot["trailing_windows"]) == 1
    assert snapshot["trailing_windows"][0]["window"] == "lastMonth"
