from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

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


def test_get_daily_price_history_returns_date_and_close(monkeypatch):
    idx = pd.DatetimeIndex([datetime(2026, 1, 1), datetime(2026, 1, 2)])
    df = pd.DataFrame({"Close": [100.0, 105.0]}, index=idx)

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            assert interval == "1d"
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    result = data_mod.get_daily_price_history("AAPL")
    assert result == [{"date": "2026-01-01", "close": 100.0}, {"date": "2026-01-02", "close": 105.0}]


def test_get_daily_price_history_returns_empty_list_on_failure(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            raise RuntimeError("network error")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_daily_price_history("AAPL") == []


def test_get_price_history_window_uses_1m_for_recent_event(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(hours=3)  # well within the last 7 days
    idx = pd.DatetimeIndex([center])
    df = pd.DataFrame({"Close": [150.0]}, index=idx)
    captured = {}

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None, prepost=None):
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

        def history(self, start=None, end=None, interval=None, prepost=None):
            captured["intervals"].append(interval)
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center)
    assert interval == "5m"
    assert captured["intervals"] == ["5m"]  # never even tried 1m for an event this old


def test_get_price_history_window_uses_explicit_interval_without_fallback(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(hours=3)  # would normally auto-pick 1m
    idx = pd.DatetimeIndex([center])
    df = pd.DataFrame({"Close": [150.0]}, index=idx)
    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None, prepost=None):
            calls.append(interval)
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center, interval="1h")
    assert interval == "1h"
    assert calls == ["1h"]  # honored the explicit request, no 1m attempt first
    assert history[0]["close"] == 150.0


def test_get_price_history_window_explicit_interval_no_fallback_on_empty(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(hours=3)
    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None, prepost=None):
            calls.append(interval)
            return pd.DataFrame()  # empty

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center, interval="1h")
    assert history == []
    assert interval == "1h"
    assert calls == ["1h"]  # never silently tried a different interval


def test_get_price_history_window_falls_back_when_1m_pull_is_empty(monkeypatch):
    center = datetime.now(timezone.utc) - timedelta(hours=3)
    idx = pd.DatetimeIndex([center])
    df = pd.DataFrame({"Close": [150.0]}, index=idx)
    calls = []

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, start=None, end=None, interval=None, prepost=None):
            calls.append(interval)
            if interval == "1m":
                return pd.DataFrame()  # empty -- e.g. right at the edge of the 7-day window
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    history, interval = data_mod.get_price_history_window("AAPL", center)
    assert interval == "5m"
    assert calls == ["1m", "5m"]
    assert history[0]["close"] == 150.0


# --- resolve_ticker_query ------------------------------------------------


def test_resolve_ticker_query_prefers_exact_symbol_match(monkeypatch):
    quotes = [
        {"symbol": "ORCL", "shortname": "Oracle Corp", "quoteType": "EQUITY", "exchange": "NYQ"},
        {"symbol": "ORCLW", "shortname": "Oracle Warrant Co", "quoteType": "EQUITY", "exchange": "NYQ"},
    ]

    class FakeSearch:
        def __init__(self, query, max_results=8):
            self.quotes = quotes

    monkeypatch.setattr(data_mod.yf, "Search", FakeSearch)

    result = data_mod.resolve_ticker_query("orcl")
    assert result == {"ticker": "ORCL", "name": "Oracle Corp", "exchange": "NYQ"}


def test_resolve_ticker_query_resolves_company_name_to_first_equity_result(monkeypatch):
    quotes = [
        {"symbol": "^ORCLIDX", "shortname": "Some Related Index", "quoteType": "INDEX"},
        {"symbol": "ORCL", "shortname": "Oracle Corp", "quoteType": "EQUITY", "exchange": "NYQ"},
    ]

    class FakeSearch:
        def __init__(self, query, max_results=8):
            self.quotes = quotes

    monkeypatch.setattr(data_mod.yf, "Search", FakeSearch)

    result = data_mod.resolve_ticker_query("oracle")
    assert result["ticker"] == "ORCL"
    assert result["name"] == "Oracle Corp"


def test_resolve_ticker_query_returns_none_when_nothing_matches(monkeypatch):
    class FakeSearch:
        def __init__(self, query, max_results=8):
            self.quotes = []

    monkeypatch.setattr(data_mod.yf, "Search", FakeSearch)
    assert data_mod.resolve_ticker_query("zzznotarealcompanyzzz") is None


def test_resolve_ticker_query_returns_none_on_exception(monkeypatch):
    class FakeSearch:
        def __init__(self, query, max_results=8):
            raise RuntimeError("network error")

    monkeypatch.setattr(data_mod.yf, "Search", FakeSearch)
    assert data_mod.resolve_ticker_query("oracle") is None


# --- get_stock_snapshot ---------------------------------------------------


class _FakeFastInfo:
    def __init__(self, last_price, year_high, year_low, market_cap):
        self.last_price = last_price
        self.year_high = year_high
        self.year_low = year_low
        self.market_cap = market_cap


def test_get_stock_snapshot_returns_expected_shape(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.fast_info = _FakeFastInfo(150.0, 200.0, 100.0, 5_000_000_000)

        def get_recommendations_summary(self, as_dict=False):
            return {"strongBuy": {0: 3}, "buy": {0: 2}, "hold": {0: 2}, "sell": {0: 0}, "strongSell": {0: 0}}

        def get_analyst_price_targets(self):
            return {"current": 150.0, "mean": 180.0}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    snap = data_mod.get_stock_snapshot("ACME")
    assert snap["ticker"] == "ACME"
    assert snap["price"] == 150.0
    assert snap["year_low"] == 100.0
    assert snap["year_high"] == 200.0
    assert snap["pct_from_52w_low"] == pytest.approx(50.0)
    assert snap["pct_from_52w_high"] == pytest.approx(-25.0)
    assert snap["target_mean"] == 180.0
    assert snap["target_upside_pct"] == pytest.approx(20.0)
    assert snap["analyst_ratings"] == {"strongBuy": 3, "buy": 2, "hold": 2, "sell": 0, "strongSell": 0}
    assert snap["buy_ratio_pct"] == pytest.approx(5 / 7 * 100)
    assert snap["market_cap"] == 5_000_000_000


def test_get_stock_snapshot_returns_none_when_ticker_lookup_fails(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            raise RuntimeError("bad ticker")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_stock_snapshot("NOTREAL") is None


def test_get_stock_snapshot_returns_none_when_year_low_missing(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.fast_info = _FakeFastInfo(150.0, 200.0, None, 5_000_000_000)

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_stock_snapshot("ACME") is None


def test_get_stock_snapshot_degrades_gracefully_when_ratings_and_targets_fail(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.fast_info = _FakeFastInfo(150.0, 200.0, 100.0, 5_000_000_000)

        def get_recommendations_summary(self, as_dict=False):
            raise RuntimeError("no data")

        def get_analyst_price_targets(self):
            raise RuntimeError("no data")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    snap = data_mod.get_stock_snapshot("ACME")
    assert snap["analyst_ratings"] == {}
    assert snap["buy_ratio_pct"] is None
    assert snap["target_mean"] is None
    assert snap["target_upside_pct"] is None
    assert snap["price"] == 150.0
