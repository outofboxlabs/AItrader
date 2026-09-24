import numpy as np
import pytest

from portfolio_monitor import pennystock_screener as ps


def test_buy_ratio_pct_computes_percentage_and_count():
    ratio, count = ps._buy_ratio_pct({"strongBuy": 6, "buy": 3, "hold": 1, "sell": 0, "strongSell": 0})
    assert ratio == 90.0
    assert count == 10


def test_buy_ratio_pct_none_when_no_ratings():
    ratio, count = ps._buy_ratio_pct({})
    assert ratio is None
    assert count == 0


def test_is_pre_revenue_true_for_confirmed_zero_revenue():
    import pandas as pd

    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Total Revenue": 0.0}})

    class FakeTicker:
        def get_income_stmt(self, freq="yearly"):
            return income

    assert ps._is_pre_revenue({}, FakeTicker()) is True


def test_is_pre_revenue_false_for_real_revenue():
    import pandas as pd

    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Total Revenue": 500_000.0}})

    class FakeTicker:
        def get_income_stmt(self, freq="yearly"):
            return income

    assert ps._is_pre_revenue({}, FakeTicker()) is False


def test_is_pre_revenue_none_when_statement_unavailable():
    class FakeTicker:
        def get_income_stmt(self, freq="yearly"):
            raise RuntimeError("no data")

    assert ps._is_pre_revenue({}, FakeTicker()) is None


def test_is_pre_revenue_uses_info_total_revenue_first():
    class FakeTicker:
        def get_income_stmt(self, freq="yearly"):
            raise AssertionError("should not fall back to the income statement when .info already has revenue")

    assert ps._is_pre_revenue({"totalRevenue": 2_700_000_000.0}, FakeTicker()) is False


def _fake_ticker_factory(data_by_symbol):
    class FakeFastInfo:
        def __init__(self, year_high, year_low):
            self.year_high = year_high
            self.year_low = year_low

    class FakeTicker:
        def __init__(self, symbol):
            self._data = data_by_symbol[symbol]

        def get_analyst_price_targets(self):
            return self._data.get("price_targets", {})

        def get_recommendations_summary(self, as_dict=False):
            return self._data.get("recommendations", {})

        @property
        def fast_info(self):
            return FakeFastInfo(self._data["year_high"], self._data["year_low"])

    return FakeTicker


def test_find_pennystock_candidates_includes_candidates_with_no_analyst_coverage(monkeypatch):
    """Unlike Near 52W Low, a penny stock with zero analyst ratings must
    still be included -- most of the universe has none."""

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "NOCOVERAGE", "shortName": "No Coverage Co", "regularMarketPrice": 2.5, "marketCap": 5e7, "regularMarketVolume": 500_000}]}

    data = {
        "NOCOVERAGE": {
            "recommendations": {},
            "year_high": 4.0,
            "year_low": 1.5,
        },
    }

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", _fake_ticker_factory(data))

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert [r["ticker"] for r in results] == ["NOCOVERAGE"]
    assert results[0]["is_pre_revenue"] is None  # FakeTicker has no get_income_stmt -- degrades gracefully


def test_find_pennystock_candidates_reports_is_pre_revenue(monkeypatch):
    import pandas as pd

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "BIOTECH", "shortName": "Biotech Co", "regularMarketPrice": 2.5, "marketCap": 5e7, "regularMarketVolume": 500_000}]}

    income = pd.DataFrame({pd.Timestamp("2026-01-31"): {"Total Revenue": 0.0}})

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            return {}

        def get_recommendations_summary(self, as_dict=False):
            return {}

        def get_income_stmt(self, freq="yearly"):
            return income

        @property
        def fast_info(self):
            class FakeFastInfo:
                year_high = 4.0
                year_low = 1.5

            return FakeFastInfo()

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", FakeTicker)

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert results[0]["is_pre_revenue"] is True
    assert results[0]["trailing_pe"] is None  # FakeTicker has no .info -- degrades gracefully
    assert results[0]["buy_ratio_pct"] is None
    assert results[0]["ratings_count"] == 0


def test_find_pennystock_candidates_reports_trailing_pe(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "PROFITABLE", "shortName": "Profitable Co", "regularMarketPrice": 2.5, "marketCap": 5e7, "regularMarketVolume": 500_000}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            return {}

        def get_recommendations_summary(self, as_dict=False):
            return {}

        @property
        def info(self):
            return {"totalRevenue": 10_000_000.0, "trailingPE": 8.2}

        @property
        def fast_info(self):
            class FakeFastInfo:
                year_high = 4.0
                year_low = 1.5

            return FakeFastInfo()

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", FakeTicker)

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert results[0]["trailing_pe"] == 8.2
    assert results[0]["is_pre_revenue"] is False


def test_find_pennystock_candidates_computes_trailing_pe_when_missing_but_negative(monkeypatch):
    """Same fallback as nearlow_screener -- see there for the live case
    that motivated it (Yahoo omits trailingPE specifically when it would
    be negative)."""
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "LOSSMAKER", "shortName": "Loss Maker Co", "regularMarketPrice": 2.5, "marketCap": 5e7, "regularMarketVolume": 500_000}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            return {}

        def get_recommendations_summary(self, as_dict=False):
            return {}

        @property
        def info(self):
            return {"totalRevenue": 10_000_000.0, "trailingPE": None, "trailingEps": -0.4, "currentPrice": 2.5}

        @property
        def fast_info(self):
            class FakeFastInfo:
                year_high = 4.0
                year_low = 1.5

            return FakeFastInfo()

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", FakeTicker)

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert results[0]["trailing_pe"] == pytest.approx(2.5 / -0.4)


def test_find_pennystock_candidates_attaches_all_three_columns(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "PENNY", "shortName": "Penny Co", "regularMarketPrice": 2.0, "marketCap": 5e7, "regularMarketVolume": 1_000_000}]}

    data = {
        "PENNY": {
            "price_targets": {"mean": 3.0},
            "recommendations": {"strongBuy": {0: 2}, "buy": {0: 1}, "hold": {0: 1}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 4.0,
            "year_low": 1.0,
        },
    }

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", _fake_ticker_factory(data))

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    only = results[0]
    assert only["buy_ratio_pct"] == 75.0
    assert only["ratings_count"] == 4
    assert round(only["pct_from_52w_high"], 1) == -50.0  # (2-4)/4
    assert round(only["pct_from_52w_low"], 1) == 100.0  # (2-1)/1
    assert only["volume"] == 1_000_000
    assert only["target_mean"] == 3.0


def test_find_pennystock_candidates_handles_missing_data_gracefully(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "NODATA", "shortName": "No Data Co", "regularMarketPrice": 2.0, "marketCap": 5e7}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        @property
        def fast_info(self):
            raise RuntimeError("no data")

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", FakeTicker)

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert results == []  # no crash, just excluded for lack of data


def test_find_pennystock_candidates_sorts_by_volume_descending(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "LOWVOL", "shortName": "Low Vol Co", "regularMarketPrice": 2.0, "marketCap": 5e7, "regularMarketVolume": 200_000},
                {"symbol": "HIGHVOL", "shortName": "High Vol Co", "regularMarketPrice": 2.0, "marketCap": 5e7, "regularMarketVolume": 900_000},
            ]
        }

    data = {
        "LOWVOL": {"recommendations": {}, "year_high": 3.0, "year_low": 1.0},
        "HIGHVOL": {"recommendations": {}, "year_high": 3.0, "year_low": 1.0},
    }

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", _fake_ticker_factory(data))

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert [r["ticker"] for r in results] == ["HIGHVOL", "LOWVOL"]


def test_find_pennystock_candidates_survives_numpy_typed_data_end_to_end(monkeypatch):
    """Same regression class as nearlow_screener/growth_screener: numpy-
    typed data must survive the full pipeline and come out JSON-
    serializable."""

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "REAL", "shortName": "Real Co", "regularMarketPrice": 2.0, "marketCap": np.int64(5 * 10**7), "regularMarketVolume": np.int64(500_000)}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            return {"mean": np.float64(3.0)}

        def get_recommendations_summary(self, as_dict=False):
            return {
                "strongBuy": {0: np.int64(2)},
                "buy": {0: np.int64(1)},
                "hold": {0: np.int64(1)},
                "sell": {0: np.int64(0)},
                "strongSell": {0: np.int64(0)},
            }

        @property
        def fast_info(self):
            class FI:
                year_high = np.float64(4.0)
                year_low = np.float64(1.0)

            return FI()

    monkeypatch.setattr(ps.yf, "screen", fake_screen)
    monkeypatch.setattr(ps.yf, "Ticker", FakeTicker)

    results = ps.find_pennystock_candidates(price_threshold=5.0)
    assert len(results) == 1
    assert results[0]["buy_ratio_pct"] == 75.0

    import json

    json.dumps(results)  # must not raise
