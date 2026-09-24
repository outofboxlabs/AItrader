import numpy as np
import pandas as pd
import pytest

from portfolio_monitor import bigdrop_screener as bd


def test_buy_ratio_pct_computes_percentage_and_count():
    ratio, count = bd._buy_ratio_pct({"strongBuy": 6, "buy": 3, "hold": 1, "sell": 0, "strongSell": 0})
    assert ratio == 90.0
    assert count == 10


def test_is_pre_revenue_uses_info_total_revenue_first():
    class FakeTicker:
        def get_income_stmt(self, freq="yearly"):
            raise AssertionError("should not fall back to the income statement when .info already has revenue")

    assert bd._is_pre_revenue({"totalRevenue": 2_700_000_000.0}, FakeTicker()) is False


def test_compute_trailing_pe_falls_back_to_price_over_eps():
    result = bd._compute_trailing_pe({"trailingPE": None, "trailingEps": -2.5, "currentPrice": 7.75})
    assert result == pytest.approx(7.75 / -2.5)


def test_pct_change_from_computes_percentage():
    closes = [100.0, 105.0, 110.0, 90.0, 95.0]
    # current_price vs closes[-1] (95.0) -- "1 day" (trading_days_back=1)
    assert bd._pct_change_from(closes, 100.0, 1) == pytest.approx((100.0 - 95.0) / 95.0 * 100.0)
    # current_price vs closes[0] (100.0) -- 4 entries back from the end
    assert bd._pct_change_from(closes, 100.0, 5) == pytest.approx(0.0)


def test_pct_change_from_none_when_not_enough_history():
    assert bd._pct_change_from([100.0, 101.0], 100.0, 22) is None


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

        def history(self, period=None, interval=None):
            closes = self._data["closes"]
            return pd.DataFrame({"Close": closes})

        @property
        def fast_info(self):
            return FakeFastInfo(self._data["year_high"], self._data["year_low"])

        @property
        def info(self):
            return self._data.get("info", {})

    return FakeTicker


def test_find_bigdrop_candidates_computes_all_three_drop_windows(monkeypatch):
    # 30 uniquely-valued closes (1.0, 2.0, ..., 30.0), oldest-first, so
    # each lookback window lands on an unambiguous, distinct reference
    # point: closes[-1]=30.0 (1 day back), closes[-6]=25.0 (1 week back,
    # i.e. 5 trading days before that), closes[-22]=9.0 (1 month back).
    closes = [float(i) for i in range(1, 31)]
    current_price = 20.0
    data = {
        "BIGCO": {
            "closes": closes,
            "year_high": 150.0,
            "year_low": 15.0,
            "recommendations": {"strongBuy": {0: 6}, "buy": {0: 3}, "hold": {0: 1}, "sell": {0: 0}, "strongSell": {0: 0}},
            "price_targets": {"mean": 25.0},
            "info": {"totalRevenue": 1e9, "trailingPE": 12.0},
        },
    }

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "BIGCO", "shortName": "Big Co", "regularMarketPrice": current_price, "marketCap": 5e9}]}

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory(data))

    results = bd.find_bigdrop_candidates(min_market_cap=1e9)
    assert len(results) == 1
    only = results[0]
    assert only["ticker"] == "BIGCO"
    assert only["pct_change_1d"] == pytest.approx((current_price - 30.0) / 30.0 * 100.0)
    assert only["pct_change_1w"] == pytest.approx((current_price - 25.0) / 25.0 * 100.0)
    assert only["pct_change_1m"] == pytest.approx((current_price - 9.0) / 9.0 * 100.0)
    assert only["is_pre_revenue"] is False
    assert only["trailing_pe"] == 12.0
    assert only["buy_ratio_pct"] == 90.0
    assert only["target_upside_pct"] == pytest.approx((25.0 - current_price) / current_price * 100.0)
    assert only["pct_from_52w_low"] == pytest.approx((current_price - 15.0) / 15.0 * 100.0)
    assert only["pct_from_52w_high"] == pytest.approx((current_price - 150.0) / 150.0 * 100.0)


def test_find_bigdrop_candidates_includes_candidates_with_no_analyst_coverage(monkeypatch):
    """Same reasoning as pennystock_screener: no single quality filter is
    imposed here either, so a candidate with zero ratings must still be
    included."""
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "NOCOVERAGE", "shortName": "No Coverage Co", "regularMarketPrice": 50.0, "marketCap": 2e9}]}

    data = {
        "NOCOVERAGE": {
            "closes": [55.0] * 30,
            "year_high": 90.0,
            "year_low": 40.0,
            "recommendations": {},
        },
    }

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory(data))

    results = bd.find_bigdrop_candidates(min_market_cap=1e9)
    assert [r["ticker"] for r in results] == ["NOCOVERAGE"]
    assert results[0]["buy_ratio_pct"] is None
    assert results[0]["ratings_count"] == 0


def test_find_bigdrop_candidates_excludes_missing_price(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "NOPRICE", "shortName": "No Price Co", "marketCap": 2e9}]}

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory({}))

    results = bd.find_bigdrop_candidates(min_market_cap=1e9)
    assert results == []


def test_find_bigdrop_candidates_sorts_by_1day_drop_ascending(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "SMALLDROP", "shortName": "Small Drop Co", "regularMarketPrice": 97.0, "marketCap": 2e9},
                {"symbol": "BIGDROP", "shortName": "Big Drop Co", "regularMarketPrice": 70.0, "marketCap": 2e9},
            ]
        }

    data = {
        "SMALLDROP": {"closes": [100.0] * 30, "year_high": 120.0, "year_low": 90.0, "recommendations": {}},
        "BIGDROP": {"closes": [100.0] * 30, "year_high": 120.0, "year_low": 60.0, "recommendations": {}},
    }

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory(data))

    results = bd.find_bigdrop_candidates(min_market_cap=1e9)
    assert [r["ticker"] for r in results] == ["BIGDROP", "SMALLDROP"]


def test_find_bigdrop_candidates_rank_by_sorts_by_the_chosen_window(monkeypatch):
    """"Rank by" must actually change which candidate ends up on top --
    a stock barely down today but hammered over the past month should
    rank first when rank_by="1m", not "1d"."""
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "TODAYDROP", "shortName": "Today Drop Co", "regularMarketPrice": 70.0, "marketCap": 2e9},
                {"symbol": "MONTHDROP", "shortName": "Month Drop Co", "regularMarketPrice": 97.0, "marketCap": 2e9},
            ]
        }

    # TODAYDROP: flat for the past month (closes[-22]=70.0), crashed only
    # today: pct_change_1d = (70-100)/100 = -30%, pct_change_1m = 0%.
    # MONTHDROP: flat for the past 21 trading days (closes[-1]=97.0),
    # already well below where it was a month ago (closes[-22]=140.0):
    # pct_change_1d = 0%, pct_change_1m = (97-140)/140 = -30.7%.
    data = {
        "TODAYDROP": {"closes": [70.0] * 29 + [100.0], "year_high": 120.0, "year_low": 60.0, "recommendations": {}},
        "MONTHDROP": {"closes": [140.0] * 9 + [97.0] * 21, "year_high": 150.0, "year_low": 90.0, "recommendations": {}},
    }

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory(data))

    results = bd.find_bigdrop_candidates(min_market_cap=1e9, rank_by="1m")
    assert [r["ticker"] for r in results] == ["MONTHDROP", "TODAYDROP"]


def test_find_bigdrop_candidates_rank_by_controls_screen_sort_field(monkeypatch):
    captured = {}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        captured["sortField"] = sortField
        return {"quotes": []}

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory({}))

    bd.find_bigdrop_candidates(min_market_cap=1e9, rank_by="1d")
    assert captured["sortField"] == "percentchange"

    bd.find_bigdrop_candidates(min_market_cap=1e9, rank_by="1w")
    assert captured["sortField"] == "fiftytwowkpercentchange"

    bd.find_bigdrop_candidates(min_market_cap=1e9, rank_by="1m")
    assert captured["sortField"] == "fiftytwowkpercentchange"


def test_find_bigdrop_candidates_rejects_unknown_rank_by(monkeypatch):
    """An unrecognized rank_by degrades to the "1d" default rather than
    raising -- callers validate this at the API boundary, this is just a
    safety net."""
    captured = {}

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        captured["sortField"] = sortField
        return {"quotes": []}

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory({}))

    bd.find_bigdrop_candidates(min_market_cap=1e9, rank_by="bogus")
    assert captured["sortField"] == "percentchange"


def test_find_bigdrop_candidates_survives_numpy_typed_data_end_to_end(monkeypatch):
    """Same regression class as the other screeners: numpy-typed data
    must survive the full pipeline and come out JSON-serializable."""
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "REAL", "shortName": "Real Co", "regularMarketPrice": 80.0, "marketCap": np.int64(5 * 10**9)}]}

    data = {
        "REAL": {
            "closes": np.array([100.0] * 30),
            "year_high": np.float64(150.0),
            "year_low": np.float64(45.0),
            "recommendations": {
                "strongBuy": {0: np.int64(2)},
                "buy": {0: np.int64(1)},
                "hold": {0: np.int64(1)},
                "sell": {0: np.int64(0)},
                "strongSell": {0: np.int64(0)},
            },
            "price_targets": {"mean": np.float64(90.0)},
        },
    }

    monkeypatch.setattr(bd.yf, "screen", fake_screen)
    monkeypatch.setattr(bd.yf, "Ticker", _fake_ticker_factory(data))

    results = bd.find_bigdrop_candidates(min_market_cap=1e9)
    assert len(results) == 1
    assert results[0]["buy_ratio_pct"] == 75.0

    import json

    json.dumps(results)  # must not raise
