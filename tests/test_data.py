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


def test_get_price_history_drops_nan_close_bars(monkeypatch):
    """A NaN close (yfinance occasionally returns one, e.g. a halted or
    partial session) must never reach jsonify(): Python's json module
    emits a literal `NaN` token for float('nan'), which isn't valid JSON
    and breaks JSON.parse() in the browser."""
    idx = pd.DatetimeIndex(
        [
            datetime(2026, 9, 11, 8, 30, tzinfo=timezone.utc),
            datetime(2026, 9, 11, 8, 35, tzinfo=timezone.utc),
        ]
    )
    df = pd.DataFrame({"Close": [100.0, float("nan")]}, index=idx)

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    result = data_mod.get_price_history("AAPL")
    assert len(result) == 1
    assert result[0]["close"] == 100.0


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


def test_get_daily_price_history_drops_nan_close_bars(monkeypatch):
    idx = pd.DatetimeIndex([datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3)])
    df = pd.DataFrame({"Close": [100.0, float("nan"), 105.0]}, index=idx)

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def history(self, period=None, interval=None):
            return df

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    result = data_mod.get_daily_price_history("AAPL")
    assert result == [{"date": "2026-01-01", "close": 100.0}, {"date": "2026-01-03", "close": 105.0}]


def test_get_analyst_price_target_snapshot_returns_normalized_shape(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_analyst_price_targets(self):
            return {"current": 250.0, "low": 245.0, "high": 400.0, "mean": 339.35, "median": 360.0}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    snapshot = data_mod.get_analyst_price_target_snapshot("AAPL")
    assert snapshot["target_mean"] == 339.35
    assert snapshot["target_median"] == 360.0
    assert snapshot["target_high"] == 400.0
    assert snapshot["target_low"] == 245.0
    assert snapshot["target_date"] is not None


def test_get_analyst_price_target_snapshot_returns_none_when_no_mean(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_analyst_price_targets(self):
            return {}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_analyst_price_target_snapshot("ZZZZ") is None


def test_get_analyst_price_target_snapshot_treats_nan_mean_as_missing(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_analyst_price_targets(self):
            return {"low": 245.0, "high": 400.0, "mean": float("nan"), "median": 360.0}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_analyst_price_target_snapshot("AAPL") is None


def test_get_analyst_price_target_snapshot_cleans_nan_secondary_fields(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_analyst_price_targets(self):
            return {"low": float("nan"), "high": 400.0, "mean": 339.35, "median": float("nan")}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    snapshot = data_mod.get_analyst_price_target_snapshot("AAPL")
    assert snapshot["target_mean"] == 339.35
    assert snapshot["target_low"] is None
    assert snapshot["target_median"] is None
    assert snapshot["target_high"] == 400.0


def test_get_analyst_price_target_snapshot_returns_none_on_failure(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_analyst_price_targets(self):
            raise RuntimeError("network error")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_analyst_price_target_snapshot("AAPL") is None


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


# --- get_financial_highlights ---------------------------------------------


def test_get_financial_highlights_computes_yoy_and_margin(monkeypatch):
    income = pd.DataFrame(
        {
            pd.Timestamp("2026-01-31"): {"Total Revenue": 1_100.0, "Net Income": 200.0, "Gross Profit": 550.0},
            pd.Timestamp("2025-01-31"): {"Total Revenue": 1_000.0, "Net Income": 150.0, "Gross Profit": 500.0},
        }
    )

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return income

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    highlights = data_mod.get_financial_highlights("ACME")
    assert highlights["fiscal_year_end"] == "2026-01-31"
    assert highlights["revenue"] == 1_100.0
    assert highlights["revenue_yoy_pct"] == pytest.approx(10.0)
    assert highlights["net_income"] == 200.0
    assert highlights["gross_margin_pct"] == pytest.approx(50.0)


def test_get_financial_highlights_handles_single_year_no_yoy(monkeypatch):
    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Total Revenue": 1_000.0, "Net Income": 100.0}})

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return income

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    highlights = data_mod.get_financial_highlights("ACME")
    assert highlights["revenue"] == 1_000.0
    assert highlights["revenue_yoy_pct"] is None
    assert highlights["gross_margin_pct"] is None  # no Gross Profit row at all


def test_get_financial_highlights_returns_none_when_statement_unavailable(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            raise RuntimeError("no data")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_financial_highlights("ACME") is None


def test_get_financial_highlights_returns_none_when_empty(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return pd.DataFrame()

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_financial_highlights("ACME") is None


def test_get_financial_highlights_includes_balance_sheet_and_cash_flow(monkeypatch):
    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Total Revenue": 1_000.0}})
    balance = pd.DataFrame(
        {
            pd.Timestamp("2026-01-31"): {"Cash And Cash Equivalents": 40_000.0, "Total Debt": 5_000.0, "Ordinary Shares Number": 110_000_000.0},
            pd.Timestamp("2025-01-31"): {"Ordinary Shares Number": 100_000_000.0},
        }
    )
    cashflow = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Free Cash Flow": -20_000.0}})

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return income

        def get_balance_sheet(self, freq="yearly"):
            return balance

        def get_cashflow(self, freq="yearly"):
            return cashflow

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    highlights = data_mod.get_financial_highlights("ACME")
    assert highlights["cash"] == 40_000.0
    assert highlights["total_debt"] == 5_000.0
    assert highlights["free_cash_flow"] == -20_000.0
    assert highlights["cash_runway_quarters"] == pytest.approx(40_000.0 / (20_000.0 / 4))  # 8.0 quarters
    assert highlights["shares_outstanding"] == 110_000_000.0
    assert highlights["shares_outstanding_yoy_pct"] == pytest.approx(10.0)


def test_get_financial_highlights_available_from_balance_sheet_alone(monkeypatch):
    """A pre-revenue biotech routinely has no usable income statement via
    yfinance but does have a balance sheet -- cash/debt/shares should
    still come through rather than the whole result being discarded."""
    balance = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Cash And Cash Equivalents": 15_000.0}})

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return pd.DataFrame()

        def get_balance_sheet(self, freq="yearly"):
            return balance

        def get_cashflow(self, freq="yearly"):
            raise RuntimeError("no data")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    highlights = data_mod.get_financial_highlights("ACME")
    assert highlights is not None
    assert highlights["cash"] == 15_000.0
    assert highlights["revenue"] is None
    assert highlights["cash_runway_quarters"] is None  # no cash-flow data to estimate burn from


def test_get_financial_highlights_no_runway_when_cash_flow_positive(monkeypatch):
    """Cash runway is a burn-rate estimate -- meaningless (and should stay
    None) when the company isn't actually burning cash."""
    balance = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Cash And Cash Equivalents": 40_000.0}})
    cashflow = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Free Cash Flow": 5_000.0}})

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            raise RuntimeError("no data")

        def get_balance_sheet(self, freq="yearly"):
            return balance

        def get_cashflow(self, freq="yearly"):
            return cashflow

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)

    highlights = data_mod.get_financial_highlights("ACME")
    assert highlights["free_cash_flow"] == 5_000.0
    assert highlights["cash_runway_quarters"] is None


def test_get_financial_highlights_finds_revenue_under_alternate_label(monkeypatch):
    """Some industries (insurers especially) don't populate yfinance's
    usual "Total Revenue" line -- real revenue-generating companies
    should still be recognized as having revenue via the alternates."""
    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Total Revenues": 2_700_000_000.0}})

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return income

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    highlights = data_mod.get_financial_highlights("ALHC")
    assert highlights["revenue"] == 2_700_000_000.0


def test_get_financial_highlights_falls_back_to_info_total_revenue(monkeypatch):
    """The exact bug hit live: an income statement with no revenue line
    yfinance's label list matches, but .info's totalRevenue has it."""
    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Net Income": 500_000_000.0}})  # no revenue line at all

    class FakeTicker:
        def __init__(self, ticker):
            pass

        def get_income_stmt(self, freq="yearly"):
            return income

        @property
        def info(self):
            return {"totalRevenue": 2_700_000_000.0}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    highlights = data_mod.get_financial_highlights("ALHC")
    assert highlights["revenue"] == 2_700_000_000.0
    assert highlights["revenue_yoy_pct"] is None  # .info only gives the current figure, no prior-year comparison


# --- get_peer_comparison ---------------------------------------------------


def test_compute_trailing_pe_prefers_yfinance_field_when_present():
    assert data_mod._compute_trailing_pe({"trailingPE": 18.5}) == 18.5


def test_compute_trailing_pe_falls_back_to_price_over_eps():
    """Confirmed live: Yahoo's own trailingPE is commonly missing/None
    specifically when it would be negative -- a real ticker (Evommune,
    Inc.) with a genuine -3.1 P/E on Robinhood (which computes and shows
    negative P/E) had no trailingPE via yfinance's .info at all, just
    price and trailingEps. Computing it ourselves surfaces the real
    number instead of "not available"."""
    result = data_mod._compute_trailing_pe({"trailingPE": None, "trailingEps": -2.5, "currentPrice": 7.75})
    assert result == pytest.approx(7.75 / -2.5)


def test_compute_trailing_pe_none_when_neither_field_available():
    assert data_mod._compute_trailing_pe({}) is None
    assert data_mod._compute_trailing_pe({"trailingEps": 0, "currentPrice": 10.0}) is None  # avoid a ZeroDivisionError


def test_get_peer_comparison_computes_target_pe_when_trailing_pe_missing(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        @property
        def info(self):
            return {"industry": "Biotechnology", "trailingPE": None, "trailingEps": -3.1, "currentPrice": 9.61}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": []}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(data_mod.yf, "screen", fake_screen)

    result = data_mod.get_peer_comparison("EVO")
    assert result["target_pe"] == pytest.approx(9.61 / -3.1)


def test_get_peer_comparison_returns_none_without_industry(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        @property
        def info(self):
            return {"sector": "Healthcare"}  # no "industry" key

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    assert data_mod.get_peer_comparison("ALHC") is None


def test_get_peer_comparison_ranks_peers_and_averages_pe(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            if self.ticker == "ALHC":
                return {"sector": "Healthcare", "industry": "Healthcare Plans", "trailingPE": 18.5}
            return {"trailingPE": None}  # peer info fallback, not exercised when screen already has it

    captured_query = {}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        captured_query["query"] = query
        captured_query["sortField"] = sortField
        return {
            "quotes": [
                {"symbol": "ALHC", "shortName": "Alignment Healthcare", "marketCap": 3_000_000_000, "trailingPE": 18.5},
                {"symbol": "UNH", "shortName": "UnitedHealth Group", "marketCap": 400_000_000_000, "trailingPE": 20.0},
                {"symbol": "HUM", "shortName": "Humana Inc.", "marketCap": 30_000_000_000, "trailingPE": 24.0},
                {"symbol": "MOH", "shortName": "Molina Healthcare", "marketCap": 15_000_000_000, "trailingPE": None},
            ]
        }

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(data_mod.yf, "screen", fake_screen)

    result = data_mod.get_peer_comparison("ALHC", max_peers=5)
    assert result["industry"] == "Healthcare Plans"
    assert result["target_pe"] == 18.5
    peer_tickers = [p["ticker"] for p in result["peers"]]
    assert "ALHC" not in peer_tickers  # excludes itself
    assert peer_tickers == ["UNH", "HUM", "MOH"]
    assert result["peer_avg_pe"] == pytest.approx((20.0 + 24.0) / 2)  # MOH's None excluded from the average


def test_get_peer_comparison_caps_at_max_peers(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def info(self):
            return {"industry": "Semiconductors", "trailingPE": 30.0}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": f"PEER{i}", "shortName": f"Peer {i}", "marketCap": 1e9, "trailingPE": 15.0} for i in range(10)]}

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(data_mod.yf, "screen", fake_screen)

    result = data_mod.get_peer_comparison("ACME", max_peers=5)
    assert len(result["peers"]) == 5


def test_get_peer_comparison_handles_screen_failure(monkeypatch):
    class FakeTicker:
        def __init__(self, ticker):
            pass

        @property
        def info(self):
            return {"industry": "Semiconductors", "trailingPE": 30.0}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        raise RuntimeError("screener unavailable")

    monkeypatch.setattr(data_mod.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(data_mod.yf, "screen", fake_screen)

    result = data_mod.get_peer_comparison("ACME")
    assert result["target_pe"] == 30.0
    assert result["peers"] == []
    assert result["peer_avg_pe"] is None
