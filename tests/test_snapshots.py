from portfolio_monitor.snapshots import diff_new_strikes_and_expiries


def test_diff_detects_new_strike_and_new_expiry():
    prior = {
        ("2025-06-20", "call", 230.0),
        ("2025-06-20", "put", 230.0),
    }
    current = {
        ("2025-06-20", "call", 230.0),
        ("2025-06-20", "put", 230.0),
        ("2025-06-20", "call", 235.0),  # new strike
        ("2025-07-18", "call", 230.0),  # new expiry
    }

    diff = diff_new_strikes_and_expiries(prior, current)
    assert diff["new_expiries"] == ["2025-07-18"]
    assert ("2025-06-20", "call", 235.0) in diff["new_strikes"]
    assert ("2025-07-18", "call", 230.0) in diff["new_strikes"]


def test_diff_empty_when_nothing_changed():
    keys = {("2025-06-20", "call", 230.0)}
    diff = diff_new_strikes_and_expiries(keys, keys)
    assert diff["new_expiries"] == []
    assert diff["new_strikes"] == []
