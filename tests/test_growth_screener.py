import time

import numpy as np

from portfolio_monitor import growth_screener as gs


def test_current_period_ratings_dict_of_dicts_shape():
    recommendations = {
        "period": {0: "0m", 1: "-1m"},
        "strongBuy": {0: 8, 1: 7},
        "buy": {0: 2, 1: 3},
        "hold": {0: 1, 1: 1},
        "sell": {0: 0, 1: 0},
        "strongSell": {0: 0, 1: 0},
    }
    ratings = gs._current_period_ratings(recommendations)
    assert ratings == {"strongBuy": 8, "buy": 2, "hold": 1, "sell": 0, "strongSell": 0}


def test_current_period_ratings_list_shape():
    recommendations = {"strongBuy": [8, 7], "buy": [2, 3], "hold": [1, 1], "sell": [0, 0], "strongSell": [0, 0]}
    ratings = gs._current_period_ratings(recommendations)
    assert ratings["strongBuy"] == 8


def test_current_period_ratings_missing_columns_are_omitted():
    assert gs._current_period_ratings({}) == {}


def test_strong_buy_ratio_pct_computes_percentage():
    ratio = gs._strong_buy_ratio_pct({"strongBuy": 8, "buy": 2, "hold": 0, "sell": 0, "strongSell": 0})
    assert ratio == 80.0


def test_strong_buy_ratio_pct_low_ratio_still_computed():
    # Not filtered on anymore -- a low ratio is still a valid, reportable number.
    ratio = gs._strong_buy_ratio_pct({"strongBuy": 2, "buy": 5, "hold": 3, "sell": 0, "strongSell": 0})
    assert ratio == 20.0


def test_strong_buy_ratio_pct_none_when_no_ratings():
    assert gs._strong_buy_ratio_pct({}) is None


def _fake_ticker_factory(data_by_symbol):
    class FakeFastInfo:
        def __init__(self, year_high, year_low):
            self.year_high = year_high
            self.year_low = year_low

    class FakeTicker:
        def __init__(self, symbol):
            self._data = data_by_symbol[symbol]

        def get_analyst_price_targets(self):
            return self._data["price_targets"]

        def get_recommendations_summary(self, as_dict=False):
            return self._data["recommendations"]

        @property
        def fast_info(self):
            return FakeFastInfo(self._data["year_high"], self._data["year_low"])

    return FakeTicker


def test_find_growth_candidates_filters_on_upside_only_not_ratings(monkeypatch):
    """A low strong-buy ratio no longer excludes a candidate -- only the
    upside threshold does. The ratio is still computed and returned."""

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "STRONG", "shortName": "Strong Co", "regularMarketPrice": 100.0, "marketCap": 5e9},
                {"symbol": "LOWRATIO", "shortName": "Low Ratio Co", "regularMarketPrice": 50.0, "marketCap": 2e9},
                {"symbol": "LOWUPSIDE", "shortName": "Low Upside Co", "regularMarketPrice": 80.0, "marketCap": 3e9},
            ]
        }

    data = {
        "STRONG": {
            "price_targets": {"current": 100.0, "mean": 170.0},  # 70% upside
            "recommendations": {"strongBuy": {0: 9}, "buy": {0: 1}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 120.0,
            "year_low": 60.0,
        },
        "LOWRATIO": {
            "price_targets": {"current": 50.0, "mean": 90.0},  # 80% upside, but only a 20% strong-buy ratio
            "recommendations": {"strongBuy": {0: 2}, "buy": {0: 5}, "hold": {0: 3}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 70.0,
            "year_low": 40.0,
        },
        "LOWUPSIDE": {
            "price_targets": {"current": 80.0, "mean": 90.0},  # only 12.5% upside -- below threshold
            "recommendations": {"strongBuy": {0: 9}, "buy": {0: 1}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 100.0,
            "year_low": 70.0,
        },
    }

    monkeypatch.setattr(gs.yf, "screen", fake_screen)
    monkeypatch.setattr(gs.yf, "Ticker", _fake_ticker_factory(data))

    results = gs.find_growth_candidates(target_upside_threshold=50.0)

    # LOWUPSIDE is excluded (below threshold); LOWRATIO is included despite
    # its weak strong-buy ratio, sorted above STRONG since its upside is higher.
    assert [r["ticker"] for r in results] == ["LOWRATIO", "STRONG"]

    strong = results[1]
    assert strong["target_upside_pct"] == 70.0
    assert strong["pct_from_52w_high"] == (100.0 - 120.0) / 120.0 * 100.0
    assert strong["pct_from_52w_low"] == (100.0 - 60.0) / 60.0 * 100.0
    assert strong["analyst_ratings"]["strongBuy"] == 9
    assert strong["strong_buy_ratio_pct"] == 90.0

    low_ratio = results[0]
    assert low_ratio["target_upside_pct"] == 80.0
    assert low_ratio["strong_buy_ratio_pct"] == 20.0


def test_find_growth_candidates_handles_missing_data_gracefully(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "NODATA", "shortName": "No Data Co", "regularMarketPrice": 10.0, "marketCap": 1e9}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            raise RuntimeError("no data")

        def get_recommendations_summary(self, as_dict=False):
            raise RuntimeError("no data")

    monkeypatch.setattr(gs.yf, "screen", fake_screen)
    monkeypatch.setattr(gs.yf, "Ticker", FakeTicker)

    results = gs.find_growth_candidates()
    assert results == []  # no crash, just excluded for lack of data


def test_find_growth_candidates_respects_max_results(monkeypatch):
    quotes = [{"symbol": f"T{i}", "shortName": f"T{i} Co", "regularMarketPrice": 10.0, "marketCap": 1e9} for i in range(5)]

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": quotes}

    data = {
        f"T{i}": {
            "price_targets": {"current": 10.0, "mean": 20.0},  # 100% upside
            "recommendations": {"strongBuy": {0: 9}, "buy": {0: 1}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 15.0,
            "year_low": 8.0,
        }
        for i in range(5)
    }

    monkeypatch.setattr(gs.yf, "screen", fake_screen)
    monkeypatch.setattr(gs.yf, "Ticker", _fake_ticker_factory(data))

    results = gs.find_growth_candidates(max_results=3)
    assert len(results) == 3


def test_find_growth_candidates_enriches_in_parallel_not_sequentially(monkeypatch):
    """Each per-candidate lookup sleeps 0.2s; with 20 candidates run
    sequentially that's >= 4s, but on a 20-worker thread pool it should
    finish in roughly one slot's worth of time. This is what actually
    fixes the multi-minute "Run Now" hang for a real candidate pool."""
    n = 20
    quotes = [{"symbol": f"T{i}", "shortName": f"T{i} Co", "regularMarketPrice": 10.0, "marketCap": 1e9} for i in range(n)]

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": quotes}

    class SlowFakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            time.sleep(0.2)
            return {"current": 10.0, "mean": 20.0}  # 100% upside

        def get_recommendations_summary(self, as_dict=False):
            return {"strongBuy": {0: 9}, "buy": {0: 1}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}}

        @property
        def fast_info(self):
            class FI:
                year_high = 15.0
                year_low = 8.0

            return FI()

    monkeypatch.setattr(gs.yf, "screen", fake_screen)
    monkeypatch.setattr(gs.yf, "Ticker", SlowFakeTicker)

    start = time.monotonic()
    results = gs.find_growth_candidates(max_results=n, max_workers=20)
    elapsed = time.monotonic() - start

    assert len(results) == n
    assert elapsed < 1.0  # would be >= 4.0s if run sequentially


def test_strong_buy_ratio_pct_handles_numpy_types():
    """Real yfinance data comes from pandas .to_dict(), which yields
    numpy.int64 -- NOT a Python int -- so a naive isinstance(v, (int, float))
    check silently treats every real count as invalid and returns None.
    This is what made the "Strong Buy %" column always blank in practice."""
    ratings = {
        "strongBuy": np.int64(9),
        "buy": np.int64(1),
        "hold": np.int64(0),
        "sell": np.int64(0),
        "strongSell": np.int64(0),
    }
    assert gs._strong_buy_ratio_pct(ratings) == 90.0


def test_find_growth_candidates_survives_numpy_typed_data_end_to_end(monkeypatch):
    """Regression test for a real production crash: numpy-typed data
    (int64/float64, exactly what get_analyst_price_targets and
    get_recommendations_summary(as_dict=True) actually return) made the
    whole scan return a 502 -- found candidates that then failed to
    serialize to JSON when saving/exporting. This drives the full
    find_growth_candidates() pipeline with numpy types end to end and
    asserts the result is plain-JSON-safe and the ratio is populated."""

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "REAL", "shortName": "Real Co", "regularMarketPrice": 10.0, "marketCap": np.int64(10**9)}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            return {"current": np.float64(10.0), "mean": np.float64(20.0)}

        def get_recommendations_summary(self, as_dict=False):
            return {
                "strongBuy": {0: np.int64(9)},
                "buy": {0: np.int64(1)},
                "hold": {0: np.int64(0)},
                "sell": {0: np.int64(0)},
                "strongSell": {0: np.int64(0)},
            }

        @property
        def fast_info(self):
            class FI:
                year_high = np.float64(25.0)
                year_low = np.float64(8.0)

            return FI()

    monkeypatch.setattr(gs.yf, "screen", fake_screen)
    monkeypatch.setattr(gs.yf, "Ticker", FakeTicker)

    results = gs.find_growth_candidates(target_upside_threshold=40.0)

    assert len(results) == 1
    assert results[0]["strong_buy_ratio_pct"] == 90.0

    import json

    json.dumps(results)  # must not raise -- this is exactly what crashed in production
