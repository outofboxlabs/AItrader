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


def test_is_majority_strong_buy_true():
    assert gs._is_majority_strong_buy({"strongBuy": 8, "buy": 2, "hold": 1, "sell": 0, "strongSell": 0}) is True


def test_is_majority_strong_buy_false_when_plurality_not_majority():
    # strongBuy is the largest single bucket but not > 50% of the total
    assert gs._is_majority_strong_buy({"strongBuy": 4, "buy": 4, "hold": 3, "sell": 0, "strongSell": 0}) is False


def test_is_majority_strong_buy_false_when_empty():
    assert gs._is_majority_strong_buy({}) is False


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


def test_find_growth_candidates_filters_and_sorts(monkeypatch):
    def fake_screen(query, sortField=None, sortAsc=None, size=None):
        return {
            "quotes": [
                {"symbol": "STRONG", "shortName": "Strong Co", "regularMarketPrice": 100.0, "marketCap": 5e9},
                {"symbol": "WEAKMAJ", "shortName": "Weak Majority Co", "regularMarketPrice": 50.0, "marketCap": 2e9},
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
        "WEAKMAJ": {
            "price_targets": {"current": 50.0, "mean": 90.0},  # 80% upside but no strong-buy majority
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

    assert [r["ticker"] for r in results] == ["STRONG"]
    strong = results[0]
    assert strong["target_upside_pct"] == 70.0
    assert strong["pct_from_52w_high"] == (100.0 - 120.0) / 120.0 * 100.0
    assert strong["pct_from_52w_low"] == (100.0 - 60.0) / 60.0 * 100.0
    assert strong["analyst_ratings"]["strongBuy"] == 9


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
