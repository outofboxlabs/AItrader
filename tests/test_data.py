from datetime import datetime, timezone

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
