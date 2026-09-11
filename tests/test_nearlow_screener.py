import numpy as np

from portfolio_monitor import nearlow_screener as nl


def test_buy_ratio_pct_computes_percentage_and_count():
    ratio, count = nl._buy_ratio_pct({"strongBuy": 6, "buy": 3, "hold": 1, "sell": 0, "strongSell": 0})
    assert ratio == 90.0
    assert count == 10


def test_buy_ratio_pct_none_when_no_ratings():
    ratio, count = nl._buy_ratio_pct({})
    assert ratio is None
    assert count == 0


def test_buy_ratio_pct_handles_numpy_types():
    """Same real-world gotcha as growth_screener: pandas .to_dict() yields
    numpy.int64, not a Python int, so a naive isinstance(v, (int, float))
    check would silently exclude every real count."""
    ratings = {
        "strongBuy": np.int64(5),
        "buy": np.int64(4),
        "hold": np.int64(1),
        "sell": np.int64(0),
        "strongSell": np.int64(0),
    }
    ratio, count = nl._buy_ratio_pct(ratings)
    assert ratio == 90.0
    assert count == 10


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


def test_find_nearlow_candidates_filters_on_both_conditions(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "NEARSTRONG", "shortName": "Near Strong Co", "regularMarketPrice": 10.0, "marketCap": 5e9},
                {"symbol": "FARFROMLOW", "shortName": "Far From Low Co", "regularMarketPrice": 50.0, "marketCap": 2e9},
                {"symbol": "WEAKRATINGS", "shortName": "Weak Ratings Co", "regularMarketPrice": 10.5, "marketCap": 3e9},
            ]
        }

    data = {
        "NEARSTRONG": {
            "price_targets": {"current": 10.0, "mean": 15.0},
            "recommendations": {"strongBuy": {0: 6}, "buy": {0: 3}, "hold": {0: 1}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 20.0,
            "year_low": 9.5,  # (10-9.5)/9.5 = 5.3% from low
        },
        "FARFROMLOW": {
            "price_targets": {"current": 50.0, "mean": 80.0},
            "recommendations": {"strongBuy": {0: 6}, "buy": {0: 3}, "hold": {0: 1}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 60.0,
            "year_low": 20.0,  # (50-20)/20 = 150% from low -- excluded
        },
        "WEAKRATINGS": {
            "price_targets": {"current": 10.5, "mean": 12.0},
            "recommendations": {"strongBuy": {0: 1}, "buy": {0: 1}, "hold": {0: 8}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 22.0,
            "year_low": 10.0,  # (10.5-10)/10 = 5% from low, but only 20% buy ratio -- excluded
        },
    }

    monkeypatch.setattr(nl.yf, "screen", fake_screen)
    monkeypatch.setattr(nl.yf, "Ticker", _fake_ticker_factory(data))

    results = nl.find_nearlow_candidates(max_pct_from_low=15.0, min_buy_ratio_pct=60.0, min_ratings_count=3)

    assert [r["ticker"] for r in results] == ["NEARSTRONG"]
    only = results[0]
    assert only["buy_ratio_pct"] == 90.0
    assert round(only["pct_from_52w_low"], 1) == 5.3


def test_find_nearlow_candidates_requires_minimum_ratings_count(monkeypatch):
    """A 100% buy ratio off of just 1 rating shouldn't count as 'strong'."""

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "ONERATING", "shortName": "One Rating Co", "regularMarketPrice": 10.0, "marketCap": 5e9}]}

    data = {
        "ONERATING": {
            "price_targets": {"current": 10.0, "mean": 15.0},
            "recommendations": {"strongBuy": {0: 1}, "buy": {0: 0}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 20.0,
            "year_low": 9.5,
        },
    }

    monkeypatch.setattr(nl.yf, "screen", fake_screen)
    monkeypatch.setattr(nl.yf, "Ticker", _fake_ticker_factory(data))

    results = nl.find_nearlow_candidates(min_ratings_count=3)
    assert results == []


def test_find_nearlow_candidates_handles_missing_data_gracefully(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "NODATA", "shortName": "No Data Co", "regularMarketPrice": 10.0, "marketCap": 1e9}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        @property
        def fast_info(self):
            raise RuntimeError("no data")

    monkeypatch.setattr(nl.yf, "screen", fake_screen)
    monkeypatch.setattr(nl.yf, "Ticker", FakeTicker)

    results = nl.find_nearlow_candidates()
    assert results == []  # no crash, just excluded for lack of data


def test_find_nearlow_candidates_sorts_by_proximity_to_low(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "A", "shortName": "A Co", "regularMarketPrice": 11.0, "marketCap": 1e9},
                {"symbol": "B", "shortName": "B Co", "regularMarketPrice": 10.1, "marketCap": 1e9},
            ]
        }

    data = {
        "A": {
            "price_targets": {"current": 11.0, "mean": 15.0},
            "recommendations": {"strongBuy": {0: 9}, "buy": {0: 1}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 20.0,
            "year_low": 10.0,  # 10% from low
        },
        "B": {
            "price_targets": {"current": 10.1, "mean": 15.0},
            "recommendations": {"strongBuy": {0: 9}, "buy": {0: 1}, "hold": {0: 0}, "sell": {0: 0}, "strongSell": {0: 0}},
            "year_high": 20.0,
            "year_low": 10.0,  # 1% from low -- closer, should sort first
        },
    }

    monkeypatch.setattr(nl.yf, "screen", fake_screen)
    monkeypatch.setattr(nl.yf, "Ticker", _fake_ticker_factory(data))

    results = nl.find_nearlow_candidates(max_pct_from_low=15.0)
    assert [r["ticker"] for r in results] == ["B", "A"]


def test_find_nearlow_candidates_survives_numpy_typed_data_end_to_end(monkeypatch):
    """Regression test mirroring the exact production crash found in
    growth_screener: numpy-typed data must survive the full pipeline and
    come out JSON-serializable."""

    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {"quotes": [{"symbol": "REAL", "shortName": "Real Co", "regularMarketPrice": 10.0, "marketCap": np.int64(10**9)}]}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def get_analyst_price_targets(self):
            return {"current": np.float64(10.0), "mean": np.float64(15.0)}

        def get_recommendations_summary(self, as_dict=False):
            return {
                "strongBuy": {0: np.int64(6)},
                "buy": {0: np.int64(3)},
                "hold": {0: np.int64(1)},
                "sell": {0: np.int64(0)},
                "strongSell": {0: np.int64(0)},
            }

        @property
        def fast_info(self):
            class FI:
                year_high = np.float64(20.0)
                year_low = np.float64(9.5)

            return FI()

    monkeypatch.setattr(nl.yf, "screen", fake_screen)
    monkeypatch.setattr(nl.yf, "Ticker", FakeTicker)

    results = nl.find_nearlow_candidates(max_pct_from_low=15.0, min_buy_ratio_pct=60.0, min_ratings_count=3)

    assert len(results) == 1
    assert results[0]["buy_ratio_pct"] == 90.0

    import json

    json.dumps(results)  # must not raise
