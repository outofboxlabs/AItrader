from portfolio_monitor.models import Position


def test_option_auto_id():
    p = Position.from_dict(
        {
            "asset_type": "option",
            "ticker": "aapl",
            "option_type": "Call",
            "strike": 230,
            "expiry": "2025-06-20",
            "entry_price": 8.5,
            "contracts": 2,
            "entry_date": "2025-03-01",
        }
    )
    assert p.id == "AAPL-C-230-2025-06-20"
    assert p.ticker == "AAPL"
    assert p.is_option
    assert p.multiplier == 100


def test_shares_auto_id_and_multiplier():
    p = Position.from_dict(
        {
            "asset_type": "shares",
            "ticker": "msft",
            "entry_price": 410.25,
            "contracts": 15,
            "entry_date": "2025-02-15",
        }
    )
    assert p.id == "MSFT-SHARES"
    assert not p.is_option
    assert p.multiplier == 1
    assert p.target_price is None


def test_explicit_id_is_respected():
    p = Position.from_dict(
        {
            "id": "my-custom-id",
            "asset_type": "shares",
            "ticker": "spy",
            "entry_price": 500,
            "contracts": 1,
            "entry_date": "2025-01-01",
        }
    )
    assert p.id == "my-custom-id"
