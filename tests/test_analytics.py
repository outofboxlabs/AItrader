import sqlite3
from datetime import date

import pytest

from portfolio_monitor import analytics, db
from portfolio_monitor.data import Quote
from portfolio_monitor.models import Position
from portfolio_monitor.valuation import value_option_position, value_shares_position


def _shares_valuation(ticker, contracts, spot, entry_price=None):
    p = Position(
        id=f"{ticker}-SHARES",
        asset_type="shares",
        ticker=ticker,
        entry_price=entry_price if entry_price is not None else spot,
        contracts=contracts,
        entry_date=date(2025, 1, 1),
    )
    return value_shares_position(p, spot=spot)


def _option_valuation(ticker, strike, expiry, iv, spot, dte_asof, contracts=1):
    p = Position(
        id=f"{ticker}-C-{strike:g}-{expiry.isoformat()}",
        asset_type="option",
        ticker=ticker,
        option_type="call",
        strike=strike,
        expiry=expiry,
        entry_price=5.0,
        contracts=contracts,
        entry_date=date(2025, 1, 1),
    )
    quote = Quote(bid=4.9, ask=5.1, last=5.0, iv=iv, volume=10, open_interest=100)
    return value_option_position(p, spot=spot, quote=quote, asof_date=dte_asof, risk_free_rate=0.045)


def test_allocation_percentages_and_concentration_flags():
    v1 = _shares_valuation("AAPL", contracts=10, spot=100.0)  # 1000
    v2 = _shares_valuation("MSFT", contracts=10, spot=100.0)  # 1000
    sector_map = {"AAPL": "Technology", "MSFT": "Technology"}

    allocation = analytics.compute_allocation([v1, v2], sector_map, ticker_cap_pct=40.0, sector_cap_pct=60.0)

    assert allocation["total_value"] == 2000.0
    by_ticker = {r["name"]: r for r in allocation["by_ticker"]}
    assert abs(by_ticker["AAPL"]["pct_of_total"] - 50.0) < 1e-9
    assert by_ticker["AAPL"]["flagged"] is True  # 50% > 40% cap

    by_sector = {r["name"]: r for r in allocation["by_sector"]}
    assert abs(by_sector["Technology"]["pct_of_total"] - 100.0) < 1e-9
    assert by_sector["Technology"]["flagged"] is True  # 100% > 60% cap


def test_aggregate_greeks_sums_across_positions():
    v_shares = _shares_valuation("AAPL", contracts=10, spot=100.0)
    v_option = _option_valuation("AAPL", strike=100, expiry=date(2025, 12, 19), iv=0.3, spot=100.0, dte_asof=date(2025, 6, 1))

    agg = analytics.compute_aggregate_greeks([v_shares, v_option])
    assert agg["net_delta_shares"] == pytest.approx(v_shares.delta + v_option.delta)
    assert agg["total_daily_theta"] == pytest.approx(v_shares.theta + v_option.theta)
    assert agg["net_vega"] == pytest.approx(v_shares.vega + v_option.vega)


def test_upcoming_expiries_flags_within_threshold():
    near = _option_valuation("AAPL", strike=100, expiry=date(2025, 6, 20), iv=0.3, spot=100.0, dte_asof=date(2025, 6, 1))
    far = _option_valuation("AAPL", strike=100, expiry=date(2026, 1, 1), iv=0.3, spot=100.0, dte_asof=date(2025, 6, 1))

    results = analytics.compute_upcoming_expiries([near, far], threshold_days=45)
    by_id = {r["position_id"]: r for r in results}
    assert by_id[near.position.id]["within_threshold"] is True
    assert by_id[far.position.id]["within_threshold"] is False


def test_iv_environment_building_history_then_rank(tmp_path):
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    ticker, strike, expiry, option_type = "AAPL", 100.0, "2025-12-19", "call"

    with db.connect(db_path) as conn:
        # Seed 25 days of IV history so the lookback has enough data.
        chain_dates = [f"2025-05-{d:02d}" for d in range(1, 26)]
        for i, d in enumerate(chain_dates):
            iv = 0.20 + i * 0.01  # ramps from 0.20 to 0.44
            chain = {
                "ticker": ticker,
                "spot": 100.0,
                "expiries": {
                    expiry: {
                        "calls": [
                            {"strike": strike, "bid": 5.0, "ask": 5.2, "last": 5.1, "iv": iv, "volume": 1, "open_interest": 1}
                        ],
                        "puts": [],
                    }
                },
            }
            db.save_chain_snapshot_rows(conn, ticker, d, "2025-05-01T00:00:00Z", chain)

        # Not enough history yet (only a couple of days seeded).
        few_days_valuation = _option_valuation(
            ticker, strike, date(2025, 12, 19), iv=0.44, spot=100.0, dte_asof=date(2025, 5, 26)
        )
        results = analytics.compute_iv_environment(
            conn, [few_days_valuation], date(2025, 5, 2), lookback_days=252, min_history_days=20,
            rich_threshold=70.0, cheap_threshold=30.0,
        )
        assert results[0]["status"] == "building history"

        # With all 25 days visible and current IV at the historical max, rank should be 100 (rich).
        full_valuation = _option_valuation(
            ticker, strike, date(2025, 12, 19), iv=0.44, spot=100.0, dte_asof=date(2025, 5, 26)
        )
        results = analytics.compute_iv_environment(
            conn, [full_valuation], date(2025, 5, 25), lookback_days=252, min_history_days=20,
            rich_threshold=70.0, cheap_threshold=30.0,
        )
        assert results[0]["status"] == "ok"
        assert results[0]["iv_rank"] == pytest.approx(100.0)
        assert results[0]["rich"] is True
