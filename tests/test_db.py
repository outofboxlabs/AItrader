import sqlite3

from portfolio_monitor import db as db_mod


def test_nearlow_candidates_round_trip_trailing_pe(tmp_path):
    """trailing_pe is a later addition (like is_pre_revenue before it) --
    guard against the same silent-data-loss bug: a missing column would
    make save_nearlow_candidates raise instead of quietly dropping the
    field, but this locks in the actual read-back value/type too."""
    db_path = str(tmp_path / "test.db")
    db_mod.init_db(db_path)
    with db_mod.connect(db_path) as conn:
        db_mod.save_nearlow_candidates(
            conn,
            "2026-01-01",
            [
                {"ticker": "PROFITABLE", "trailing_pe": 15.5},
                {"ticker": "UNKNOWN", "trailing_pe": None},
            ],
        )
    with db_mod.connect(db_path) as conn:
        rows = {r["ticker"]: r for r in db_mod.get_nearlow_candidates(conn, "2026-01-01")}
    assert rows["PROFITABLE"]["trailing_pe"] == 15.5
    assert rows["UNKNOWN"]["trailing_pe"] is None


def test_pennystock_candidates_round_trip_trailing_pe(tmp_path):
    db_path = str(tmp_path / "test.db")
    db_mod.init_db(db_path)
    with db_mod.connect(db_path) as conn:
        db_mod.save_pennystock_candidates(
            conn,
            "2026-01-01",
            "5",
            [{"ticker": "PROFITABLE", "trailing_pe": 8.2}],
        )
    with db_mod.connect(db_path) as conn:
        rows = db_mod.get_pennystock_candidates(conn, "2026-01-01", "5")
    assert rows[0]["trailing_pe"] == 8.2


def test_migrate_columns_survives_concurrent_duplicate_add(tmp_path):
    """Regression test for a real crash hit live: init_db runs on nearly
    every request, so two requests can both see a newly-added column
    missing and both attempt ALTER TABLE before either's result is
    visible to the other -- observed with the is_pre_revenue columns on a
    single-threaded dev server, right after they were added to
    _ADDED_COLUMNS. The loser must not raise; the column is already
    there either way."""
    db_path = str(tmp_path / "test.db")
    db_mod.init_db(db_path)  # first call adds every _ADDED_COLUMNS column normally

    with db_mod.connect(db_path) as conn:
        # Simulate a second request's ALTER TABLE landing after this
        # connection already thinks the column is missing -- by the time
        # _migrate_columns runs again here, the column already exists
        # (added above), so this call must be a silent no-op, not a crash.
        db_mod._migrate_columns(conn)

    with db_mod.connect(db_path) as conn:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(nearlow_candidates)")}
    assert "is_pre_revenue" in existing
    assert "is_pre_revenue_reason" in existing


def test_migrate_columns_reraises_other_operational_errors(tmp_path):
    """The duplicate-column race is the only OperationalError this should
    swallow -- anything else (a genuinely broken ALTER, a locked db) must
    still surface."""
    db_path = str(tmp_path / "test.db")
    db_mod.init_db(db_path)

    original_added_columns = db_mod._ADDED_COLUMNS
    db_mod._ADDED_COLUMNS = {"nearlow_candidates": [("new_col", "NOT VALID SQL SYNTAX HERE")]}
    try:
        with db_mod.connect(db_path) as conn:
            try:
                db_mod._migrate_columns(conn)
                assert False, "expected an OperationalError to propagate"
            except sqlite3.OperationalError as exc:
                assert "duplicate column name" not in str(exc)
    finally:
        db_mod._ADDED_COLUMNS = original_added_columns
