from datetime import datetime, timezone

import pytest

from portfolio_monitor import social_sentiment as ss


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


# --- get_message_metrics --------------------------------------------------


def test_get_message_metrics_builds_expected_request(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({"time_series": [{"timestamp": "2026-09-14T00:00:00", "total_count": 5}]})

    monkeypatch.setattr(ss.requests, "get", fake_get)

    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    result = ss.get_message_metrics("ORCL", "hour", api_key="sk-test", now=now)

    assert result is not None
    assert captured["url"] == "https://api.stockgeist.ai/time-series/message-metrics"
    assert captured["params"]["token"] == "sk-test"
    assert captured["params"]["symbol"] == "ORCL"
    assert captured["params"]["timeframe"] == "5m"  # "hour" window -> 5-minute buckets
    assert captured["params"]["start"] == "2026-09-14T11:00:00"
    assert captured["params"]["end"] == "2026-09-14T12:00:00"


def test_get_message_metrics_uses_coarser_buckets_for_longer_windows(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["timeframe"] = params["timeframe"]
        return _FakeResponse({})

    monkeypatch.setattr(ss.requests, "get", fake_get)
    now = datetime(2026, 9, 14, tzinfo=timezone.utc)

    ss.get_message_metrics("ORCL", "year", api_key="sk-test", now=now)
    assert captured["timeframe"] == "1d"

    ss.get_message_metrics("ORCL", "week", api_key="sk-test", now=now)
    assert captured["timeframe"] == "1h"


def test_get_message_metrics_returns_none_for_unknown_window(monkeypatch):
    assert ss.get_message_metrics("ORCL", "fortnight", api_key="sk-test") is None


def test_get_message_metrics_returns_none_on_network_failure(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise RuntimeError("network error")

    monkeypatch.setattr(ss.requests, "get", boom)
    assert ss.get_message_metrics("ORCL", "today", api_key="sk-test") is None


def test_get_message_metrics_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(ss.requests, "get", lambda url, params=None, timeout=None: _FakeResponse(None, status_ok=False))
    assert ss.get_message_metrics("ORCL", "today", api_key="sk-test") is None


# --- summarize_message_metrics --------------------------------------------


def test_summarize_message_metrics_computes_stats_across_entries():
    raw = {
        "time_series": [
            {"timestamp": "t1", "total_count": 10, "pos_index": 0.2},
            {"timestamp": "t2", "total_count": 20, "pos_index": 0.6},
            {"timestamp": "t3", "total_count": 30, "pos_index": 0.4},
        ]
    }
    summary = ss.summarize_message_metrics(raw)
    assert summary["data_points"] == 3
    assert summary["total_count"]["avg"] == pytest.approx(20.0)
    assert summary["total_count"]["min"] == 10.0
    assert summary["total_count"]["max"] == 30.0
    assert summary["total_count"]["first"] == 10.0
    assert summary["total_count"]["last"] == 30.0
    assert summary["pos_index"]["avg"] == pytest.approx(0.4)


def test_summarize_message_metrics_handles_plain_list_response():
    raw = [{"total_count": 5}, {"total_count": 15}]
    summary = ss.summarize_message_metrics(raw)
    assert summary["data_points"] == 2
    assert summary["total_count"]["avg"] == pytest.approx(10.0)


def test_summarize_message_metrics_returns_none_for_empty_or_unusable_input():
    assert ss.summarize_message_metrics(None) is None
    assert ss.summarize_message_metrics({}) is None
    assert ss.summarize_message_metrics({"error": "bad token"}) is None
    assert ss.summarize_message_metrics([{"timestamp": "t1"}]) is None  # no numeric fields at all


def test_summarize_message_metrics_ignores_non_numeric_and_boolean_fields():
    raw = [{"timestamp": "t1", "total_count": 5, "symbol": "ORCL", "flagged": True}]
    summary = ss.summarize_message_metrics(raw)
    assert "total_count" in summary
    assert "symbol" not in summary
    assert "flagged" not in summary
