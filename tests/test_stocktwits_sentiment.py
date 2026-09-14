from datetime import datetime, timedelta, timezone

from portfolio_monitor import stocktwits_sentiment as sts


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


def _message(created_at, sentiment=None, body="ORCL looking strong", username="trader1", likes=3, msg_id=None):
    return {
        "id": msg_id,
        "body": body,
        "created_at": created_at,
        "user": {"username": username},
        "entities": {"sentiment": {"basic": sentiment} if sentiment else None},
        "likes": {"total": likes},
    }


def test_get_recent_messages_returns_normalized_shape(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    recent = _message("2026-09-14T11:30:00Z", sentiment="Bullish")

    captured = {}

    def fake_get(url, params=None, timeout=None, headers=None):
        captured["url"] = url
        return _FakeResponse({"messages": [recent]})

    monkeypatch.setattr(sts.requests, "get", fake_get)

    messages = sts.get_recent_messages("ORCL", "hour", now=now)
    assert captured["url"] == "https://api.stocktwits.com/api/2/streams/symbol/ORCL.json"
    assert len(messages) == 1
    assert messages[0]["body"] == "ORCL looking strong"
    assert messages[0]["username"] == "trader1"
    assert messages[0]["sentiment"] == "Bullish"
    assert messages[0]["likes"] == 3
    assert messages[0]["created_at"] == "2026-09-14T11:30:00+00:00"


def test_get_recent_messages_filters_out_messages_before_the_window(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    within_window = _message("2026-09-14T11:55:00Z")  # 5 min ago -- inside "hour"
    outside_window = _message("2026-09-14T09:00:00Z")  # 3 hours ago -- outside "hour"

    monkeypatch.setattr(sts.requests, "get", lambda url, params=None, timeout=None, headers=None: _FakeResponse({"messages": [within_window, outside_window]}))

    messages = sts.get_recent_messages("ORCL", "hour", now=now)
    assert len(messages) == 1
    assert messages[0]["created_at"] == "2026-09-14T11:55:00+00:00"


def test_get_recent_messages_handles_untagged_sentiment(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    untagged = _message("2026-09-14T11:59:00Z", sentiment=None)
    monkeypatch.setattr(sts.requests, "get", lambda url, params=None, timeout=None, headers=None: _FakeResponse({"messages": [untagged]}))

    messages = sts.get_recent_messages("ORCL", "hour", now=now)
    assert messages[0]["sentiment"] is None


def test_get_recent_messages_returns_none_for_unknown_window():
    assert sts.get_recent_messages("ORCL", "fortnight") is None


def test_get_recent_messages_returns_none_on_network_failure(monkeypatch):
    def boom(url, params=None, timeout=None, headers=None):
        raise RuntimeError("network error")

    monkeypatch.setattr(sts.requests, "get", boom)
    assert sts.get_recent_messages("ORCL", "today") is None


def test_get_recent_messages_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(sts.requests, "get", lambda url, params=None, timeout=None, headers=None: _FakeResponse(None, status_ok=False))
    assert sts.get_recent_messages("ORCL", "today") is None


def test_get_recent_messages_returns_empty_list_when_no_messages(monkeypatch):
    monkeypatch.setattr(sts.requests, "get", lambda url, params=None, timeout=None, headers=None: _FakeResponse({"messages": []}))
    assert sts.get_recent_messages("ORCL", "week") == []


def test_get_recent_messages_skips_messages_with_unparseable_timestamps(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    bad = _message("not-a-real-timestamp")
    good = _message("2026-09-14T11:59:00Z")
    monkeypatch.setattr(sts.requests, "get", lambda url, params=None, timeout=None, headers=None: _FakeResponse({"messages": [bad, good]}))

    messages = sts.get_recent_messages("ORCL", "hour", now=now)
    assert len(messages) == 1
    assert messages[0]["created_at"] == "2026-09-14T11:59:00+00:00"


def _full_page(created_at, start_id, count=30):
    return [
        _message(created_at, body=f"msg {start_id - i}", msg_id=start_id - i)
        for i in range(count)
    ]


def test_get_recent_messages_paginates_across_multiple_pages(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    page1 = _full_page("2026-09-14T11:00:00Z", start_id=1000, count=30)  # ids 971-1000
    page2 = [_message("2026-09-14T10:00:00Z", body="older msg", msg_id=900)]  # short page -> stop

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(dict(params or {}))
        if len(calls) == 1:
            return _FakeResponse({"messages": page1})
        return _FakeResponse({"messages": page2})

    monkeypatch.setattr(sts.requests, "get", fake_get)

    messages = sts.get_recent_messages("ORCL", "week", now=now)
    assert len(calls) == 2
    assert "max" not in calls[0]
    assert calls[1]["max"] == 970  # min id of page1 (971) - 1
    assert len(messages) == 31  # 30 from page1 + 1 from page2


def test_get_recent_messages_stops_once_page_is_older_than_window(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    # Full page, but its oldest message is already outside the "hour" window --
    # pagination should stop after this one page rather than fetching more.
    page1 = _full_page("2026-09-14T08:00:00Z", start_id=1000, count=30)

    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(dict(params or {}))
        return _FakeResponse({"messages": page1})

    monkeypatch.setattr(sts.requests, "get", fake_get)

    messages = sts.get_recent_messages("ORCL", "hour", now=now)
    assert len(calls) == 1
    assert messages == []  # all of page1 is older than "hour" cutoff


def test_get_recent_messages_stops_after_max_pages(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(dict(params or {}))
        start_id = 100000 - len(calls) * 1000
        return _FakeResponse({"messages": _full_page("2026-09-14T11:00:00Z", start_id=start_id, count=30)})

    monkeypatch.setattr(sts.requests, "get", fake_get)

    messages = sts.get_recent_messages("ORCL", "year", now=now)
    assert len(calls) == sts.MAX_PAGES
    assert len(messages) == sts.MAX_PAGES * 30


def test_get_recent_messages_keeps_earlier_pages_when_a_later_page_fails(monkeypatch):
    now = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
    page1 = _full_page("2026-09-14T11:00:00Z", start_id=1000, count=30)
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(dict(params or {}))
        if len(calls) == 1:
            return _FakeResponse({"messages": page1})
        raise RuntimeError("network error")

    monkeypatch.setattr(sts.requests, "get", fake_get)

    messages = sts.get_recent_messages("ORCL", "week", now=now)
    assert len(calls) == 2
    assert len(messages) == 30
