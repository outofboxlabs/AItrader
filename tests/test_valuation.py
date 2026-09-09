from datetime import date

from portfolio_monitor.data import Quote
from portfolio_monitor.models import Position
from portfolio_monitor.valuation import value_option_position, value_shares_position


def _option_position(**overrides):
    defaults = dict(
        id="AAPL-C-230-2025-06-20",
        asset_type="option",
        ticker="AAPL",
        option_type="call",
        strike=230.0,
        expiry=date(2025, 6, 20),
        entry_price=8.50,
        contracts=2,
        entry_date=date(2025, 3, 1),
        target_price=15.0,
        stop_price=4.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def _shares_position(**overrides):
    defaults = dict(
        id="MSFT-SHARES",
        asset_type="shares",
        ticker="MSFT",
        entry_price=410.25,
        contracts=15,
        entry_date=date(2025, 2, 15),
        target_price=450.0,
        stop_price=380.0,
    )
    defaults.update(overrides)
    return Position(**defaults)


def test_shares_valuation():
    p = _shares_position()
    v = value_shares_position(p, spot=430.0)

    assert v.mark == 430.0
    assert v.current_value == 430.0 * 15
    assert v.unrealized_pnl == (430.0 - 410.25) * 15
    assert v.dte is None
    # progress: (430-410.25)/(450-410.25) * 100
    assert abs(v.progress_to_target_pct - (430.0 - 410.25) / (450.0 - 410.25) * 100.0) < 1e-9
    assert v.delta == 15  # 1 delta per share


def test_option_valuation_uses_mid_mark_and_scales_by_100x_contracts():
    p = _option_position()
    quote = Quote(bid=9.0, ask=9.4, last=9.1, iv=0.35, volume=100, open_interest=500)
    v = value_option_position(p, spot=232.0, quote=quote, asof_date=date(2025, 5, 1), risk_free_rate=0.045)

    assert abs(v.mark - 9.2) < 1e-9  # mid of bid/ask
    assert abs(v.current_value - 9.2 * 2 * 100) < 1e-6
    cost_basis = 8.50 * 2 * 100
    assert abs(v.unrealized_pnl - (9.2 * 2 * 100 - cost_basis)) < 1e-6
    assert v.dte == (date(2025, 6, 20) - date(2025, 5, 1)).days
    assert v.current_iv == 0.35
    # greeks should be non-trivial and scaled by contracts*100
    assert v.delta is not None and v.delta != 0


def test_option_valuation_falls_back_to_last_when_no_bid_ask():
    p = _option_position()
    quote = Quote(bid=None, ask=None, last=9.05, iv=0.35, volume=100, open_interest=500)
    v = value_option_position(p, spot=232.0, quote=quote, asof_date=date(2025, 5, 1), risk_free_rate=0.045)
    assert v.mark == 9.05
