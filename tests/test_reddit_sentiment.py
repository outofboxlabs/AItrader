from datetime import datetime, timezone

import pytest

from portfolio_monitor import reddit_sentiment as rs


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _reset_token_cache(monkeypatch):
    monkeypatch.setattr(rs, "_cached_token", None)
    monkeypatch.setattr(rs, "_cached_token_expiry", 0.0)


# --- _get_access_token -----------------------------------------------------


def test_get_access_token_returns_token_on_success(monkeypatch):
    captured = {}

    def fake_post(url, auth=None, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["auth"] = auth
        captured["data"] = data
        captured["headers"] = headers
        return _FakeResponse({"access_token": "tok-123", "expires_in": 3600})

    monkeypatch.setattr(rs.requests, "post", fake_post)

    token = rs._get_access_token("client-id", "client-secret")
    assert token == "tok-123"
    assert captured["url"] == "https://www.reddit.com/api/v1/access_token"
    assert captured["auth"] == ("client-id", "client-secret")
    assert captured["data"] == {"grant_type": "client_credentials"}
    assert "User-Agent" in captured["headers"]


def test_get_access_token_caches_across_calls(monkeypatch):
    calls = []

    def fake_post(url, auth=None, data=None, headers=None, timeout=None):
        calls.append(1)
        return _FakeResponse({"access_token": "tok-123", "expires_in": 3600})

    monkeypatch.setattr(rs.requests, "post", fake_post)

    assert rs._get_access_token("id", "secret") == "tok-123"
    assert rs._get_access_token("id", "secret") == "tok-123"
    assert len(calls) == 1  # second call served from cache


def test_get_access_token_returns_none_on_failure(monkeypatch):
    def boom(url, auth=None, data=None, headers=None, timeout=None):
        raise RuntimeError("network error")

    monkeypatch.setattr(rs.requests, "post", boom)
    assert rs._get_access_token("id", "secret") is None


def test_get_access_token_returns_none_when_no_token_in_response(monkeypatch):
    monkeypatch.setattr(rs.requests, "post", lambda *a, **kw: _FakeResponse({"error": "invalid_grant"}))
    assert rs._get_access_token("id", "secret") is None


# --- search_recent_posts ----------------------------------------------------


def _fake_listing():
    created = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    return {
        "data": {
            "children": [
                {
                    "data": {
                        "title": "ORCL to the moon",
                        "subreddit": "wallstreetbets",
                        "score": 120,
                        "num_comments": 45,
                        "created_utc": created,
                        "permalink": "/r/wallstreetbets/comments/abc123/orcl_to_the_moon/",
                    }
                }
            ]
        }
    }


def test_search_recent_posts_returns_normalized_shape(monkeypatch):
    monkeypatch.setattr(rs, "_get_access_token", lambda client_id, client_secret: "tok-123")

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return _FakeResponse(_fake_listing())

    monkeypatch.setattr(rs.requests, "get", fake_get)

    posts = rs.search_recent_posts("ORCL", "week", "id", "secret")
    assert len(posts) == 1
    assert posts[0]["title"] == "ORCL to the moon"
    assert posts[0]["subreddit"] == "wallstreetbets"
    assert posts[0]["score"] == 120
    assert posts[0]["num_comments"] == 45
    assert posts[0]["created_at"] == "2026-09-14T12:00:00+00:00"
    assert posts[0]["permalink"] == "https://reddit.com/r/wallstreetbets/comments/abc123/orcl_to_the_moon/"

    assert captured["url"] == f"https://oauth.reddit.com/r/{rs.FINANCE_SUBREDDITS}/search"
    assert captured["params"]["q"] == "ORCL"
    assert captured["params"]["t"] == "week"
    assert captured["headers"]["Authorization"] == "Bearer tok-123"


def test_search_recent_posts_maps_today_window_to_day(monkeypatch):
    monkeypatch.setattr(rs, "_get_access_token", lambda client_id, client_secret: "tok-123")
    captured = {}
    monkeypatch.setattr(
        rs.requests, "get",
        lambda url, params=None, headers=None, timeout=None: captured.update(t=params["t"]) or _FakeResponse({}),
    )
    rs.search_recent_posts("ORCL", "today", "id", "secret")
    assert captured["t"] == "day"


def test_search_recent_posts_returns_none_for_unknown_window(monkeypatch):
    assert rs.search_recent_posts("ORCL", "fortnight", "id", "secret") is None


def test_search_recent_posts_returns_none_when_auth_fails(monkeypatch):
    monkeypatch.setattr(rs, "_get_access_token", lambda client_id, client_secret: None)
    assert rs.search_recent_posts("ORCL", "week", "id", "secret") is None


def test_search_recent_posts_returns_none_on_network_failure(monkeypatch):
    monkeypatch.setattr(rs, "_get_access_token", lambda client_id, client_secret: "tok-123")

    def boom(url, params=None, headers=None, timeout=None):
        raise RuntimeError("network error")

    monkeypatch.setattr(rs.requests, "get", boom)
    assert rs.search_recent_posts("ORCL", "week", "id", "secret") is None


def test_search_recent_posts_handles_empty_listing(monkeypatch):
    monkeypatch.setattr(rs, "_get_access_token", lambda client_id, client_secret: "tok-123")
    monkeypatch.setattr(rs.requests, "get", lambda *a, **kw: _FakeResponse({"data": {"children": []}}))
    assert rs.search_recent_posts("ORCL", "week", "id", "secret") == []
