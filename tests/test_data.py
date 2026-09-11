from datetime import datetime, timedelta, timezone

import pandas as pd

from portfolio_monitor import data as data_mod


def test_get_price_history_returns_time_and_close(monkeypatch):
    idx = pd.DatetimeIndex(
        [
            datetime(2026, 9, 11, 8, 30, tzinfo=timezone.utc),
            datetime(2026, 9, 11, 8, 35, tzinfo=timezone.utc),
        ]
    )
    df = pd.DataFrame({"Close": [100.0, 101.5]}, index=idx)

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    result = data_mod.get_price_history("AAPL")

    assert len(result) == 2
    assert result[0]["close"] == 100.0
    assert result[0]["time"] == idx[0].isoformat()
    assert result[1]["close"] == 101.5


def test_get_price_history_returns_empty_list_on_empty_data(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            return pd.DataFrame()

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_price_history("AAPL") == []


def test_get_price_history_returns_empty_list_on_exception(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            raise RuntimeError("network error")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_price_history("AAPL") == []


def test_get_price_history_window_uses_1m_for_recent_event(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(hours=3)  # well within the last 7 days
    idx = pd.DatetimeIndex([center])
    df = pd.DataFrame({"Close": [150.0]}, index=idx)
    captured = {}

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None):
            captured["interval"] = interval
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center)
    assert interval == "1m"
    assert captured["interval"] == "1m"
    assert history[0]["close"] == 150.0


def test_get_price_history_window_falls_back_to_5m_for_old_event(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(days=30)  # beyond the 7-day 1m limit
    idx = pd.DatetimeIndex([center])
    df = pd.DataFrame({"Close": [150.0]}, index=idx)
    captured = {"intervals": []}

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None):
            captured["intervals"].append(interval)
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center)
    assert interval == "5m"
    assert captured["intervals"] == ["5m"]  # never even tried 1m for an event this old


def test_get_price_history_window_falls_back_when_1m_pull_is_empty(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(hours=3)
    idx = pd.DatetimeIndex([center])
    df = pd.DataFrame({"Close": [150.0]}, index=idx)
    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None):
            calls.append(interval)
            if interval == "1m":
                return pd.DataFrame()  # empty -- e.g. right at the edge of the 7-day window
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center)
    assert interval == "5m"
    assert calls == ["1m", "5m"]
    assert history[0]["close"] == 150.0
