from datetime import date

from portfolio_monitor import data as data_mod
from portfolio_monitor import macro as macro_mod
from portfolio_monitor import news as news_mod
from portfolio_monitor import pipeline


def _fake_chain(ticker, spot):
    if ticker == "AAPL":
        return {
            "ticker": "AAPL",
            "spot": spot,
            "expiries": {
                "2025-06-20": {
                    "calls": [
                        {"strike": 230.0, "bid": 9.0, "ask": 9.4, "last": 9.1, "iv": 0.32, "volume": 100, "open_interest": 500}
                    ],
                    "puts": [],
                }
            },
        }
    raise AssertionError(f"unexpected ticker {ticker}")


def test_run_full_analysis_end_to_end(tmp_path, monkeypatch):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(
        """
        [
          {
            "asset_type": "option", "ticker": "AAPL", "option_type": "call",
            "strike": 230.0, "expiry": "2025-06-20", "entry_price": 8.5,
            "contracts": 2, "entry_date": "2025-03-01",
            "target_price": 15.0, "stop_price": 4.0
          }
        ]
        """
    )
    db_path = str(tmp_path / "test.db")
    snapshots_dir = str(tmp_path / "snapshots")

    monkeypatch.setattr(data_mod, "get_spot_price", lambda ticker: 232.0)
    monkeypatch.setattr(data_mod, "pull_full_chain", lambda ticker: _fake_chain(ticker, 232.0))
    monkeypatch.setattr(
        macro_mod,
        "compute_macro_gate",
        lambda **kw: {
            "score": 65.0,
            "weights": {},
            "vix_level": {"vix": 15.0, "vix_percentile": 40.0, "score": 60.0},
            "term_structure": {"vix": 15.0, "vix3m": 16.0, "ratio": 0.94, "score": 80.0},
            "breadth": {"pct_above_200dma": 65.0, "constituents": 11, "score": 65.0},
            "credit_spread": {"credit_ratio": 0.8, "credit_percentile": 55.0, "score": 55.0},
        },
    )
    monkeypatch.setattr(news_mod, "fetch_recent_headlines", lambda ticker, window_days, now=None: [])
    monkeypatch.setattr(
        news_mod,
        "analyze_headlines_with_claude",
        lambda ticker, headlines, api_key, model: {
            "summary": f"{ticker} summary",
            "sentiment": "neutral",
            "key_drivers": [],
            "position_flag": False,
            "position_flag_reason": None,
        },
    )

    result = pipeline.run_full_analysis(
        positions_path=str(positions_path),
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        asof_date=date(2025, 5, 1),
        news_api_key="sk-test",
    )

    assert result["asof_date"] == "2025-05-01"
    assert len(result["valuations"]) == 1
    v = result["valuations"][0]
    assert v["ticker"] == "AAPL"
    assert v["mark"] == 9.2  # mid of 9.0/9.4
    assert v["current_value"] == 9.2 * 2 * 100

    assert result["allocation"]["total_value"] == v["current_value"]
    assert "net_delta_shares" in result["aggregate_greeks"]
    assert result["macro"]["score"] == 65.0
    assert len(result["news"]) == 1
    assert result["news"][0]["ticker"] == "AAPL"
    assert result["news"][0]["sentiment"] == "neutral"
    assert result["warnings"] == []

    # The whole thing must be JSON-serializable (this is what app.py returns as-is).
    import json

    json.dumps(result)


def test_run_full_analysis_skip_macro_and_news(tmp_path, monkeypatch):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(
        """
        [{"asset_type": "shares", "ticker": "MSFT", "entry_price": 400.0,
          "contracts": 5, "entry_date": "2025-01-01"}]
        """
    )
    monkeypatch.setattr(data_mod, "get_spot_price", lambda ticker: 410.0)

    result = pipeline.run_full_analysis(
        positions_path=str(positions_path),
        db_path=str(tmp_path / "test.db"),
        snapshots_dir=str(tmp_path / "snapshots"),
        asof_date=date(2025, 5, 1),
        skip_macro=True,
        skip_news=True,
    )

    assert result["macro"] is None
    assert result["news"] == []
    assert result["valuations"][0]["ticker"] == "MSFT"


def test_run_full_analysis_warns_on_missing_quote(tmp_path, monkeypatch):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(
        """
        [{"asset_type": "option", "ticker": "AAPL", "option_type": "call",
          "strike": 999.0, "expiry": "2025-06-20", "entry_price": 1.0,
          "contracts": 1, "entry_date": "2025-01-01"}]
        """
    )
    monkeypatch.setattr(data_mod, "get_spot_price", lambda ticker: 232.0)
    monkeypatch.setattr(data_mod, "pull_full_chain", lambda ticker: _fake_chain(ticker, 232.0))

    result = pipeline.run_full_analysis(
        positions_path=str(positions_path),
        db_path=str(tmp_path / "test.db"),
        snapshots_dir=str(tmp_path / "snapshots"),
        asof_date=date(2025, 5, 1),
        skip_macro=True,
        skip_news=True,
    )

    assert result["valuations"] == []
    assert len(result["warnings"]) == 1
    assert "999" in result["warnings"][0]
