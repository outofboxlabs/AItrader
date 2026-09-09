"""Single SQLite database backing every layer: chain snapshots, valuations,
and portfolio analytics. All tables are keyed by asof_date so runs never
overwrite each other's history.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    ticker TEXT NOT NULL,
    expiry TEXT NOT NULL,
    option_type TEXT NOT NULL,
    strike REAL NOT NULL,
    asof_date TEXT NOT NULL,
    bid REAL,
    ask REAL,
    last REAL,
    iv REAL,
    volume REAL,
    open_interest REAL,
    spot REAL,
    pulled_at TEXT NOT NULL,
    PRIMARY KEY (ticker, expiry, option_type, strike, asof_date)
);

CREATE TABLE IF NOT EXISTS valuations (
    asof_date TEXT NOT NULL,
    position_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    mark REAL,
    current_value REAL,
    unrealized_pnl REAL,
    unrealized_pnl_pct REAL,
    dte INTEGER,
    progress_to_target_pct REAL,
    progress_to_stop_pct REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    PRIMARY KEY (asof_date, position_id)
);

CREATE TABLE IF NOT EXISTS allocation (
    asof_date TEXT NOT NULL,
    level TEXT NOT NULL,       -- 'ticker' or 'sector'
    name TEXT NOT NULL,
    value REAL,
    pct_of_total REAL,
    flagged INTEGER,
    cap_pct REAL,
    PRIMARY KEY (asof_date, level, name)
);

CREATE TABLE IF NOT EXISTS portfolio_greeks (
    asof_date TEXT NOT NULL PRIMARY KEY,
    net_delta_shares REAL,
    total_daily_theta REAL,
    net_vega REAL
);

CREATE TABLE IF NOT EXISTS iv_environment (
    asof_date TEXT NOT NULL,
    position_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    current_iv REAL,
    iv_rank REAL,
    iv_percentile REAL,
    history_days INTEGER,
    status TEXT,
    rich INTEGER,
    cheap INTEGER,
    PRIMARY KEY (asof_date, position_id)
);

CREATE TABLE IF NOT EXISTS upcoming_expiries (
    asof_date TEXT NOT NULL,
    position_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    expiry TEXT NOT NULL,
    dte INTEGER,
    within_threshold INTEGER,
    PRIMARY KEY (asof_date, position_id)
);

CREATE TABLE IF NOT EXISTS macro_gate (
    asof_date TEXT NOT NULL PRIMARY KEY,
    score REAL,
    weights_json TEXT,
    vix REAL,
    vix_percentile REAL,
    vix_level_score REAL,
    vix3m REAL,
    term_structure_ratio REAL,
    term_structure_score REAL,
    breadth_pct_above_200dma REAL,
    breadth_constituents INTEGER,
    breadth_score REAL,
    credit_ratio REAL,
    credit_percentile REAL,
    credit_score REAL
);

CREATE TABLE IF NOT EXISTS news_analysis (
    asof_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    status TEXT,
    summary TEXT,
    sentiment TEXT,
    key_drivers_json TEXT,
    position_flag INTEGER,
    position_flag_reason TEXT,
    headline_count INTEGER,
    window_days INTEGER,
    model TEXT,
    parse_error INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (asof_date, ticker)
);
"""


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def save_chain_snapshot_rows(conn, ticker: str, asof_date: str, pulled_at: str, chain: dict) -> None:
    spot = chain.get("spot")
    rows = []
    for expiry, sides in chain.get("expiries", {}).items():
        for option_type, key in (("call", "calls"), ("put", "puts")):
            for row in sides[key]:
                rows.append(
                    (
                        ticker,
                        expiry,
                        option_type,
                        row["strike"],
                        asof_date,
                        row["bid"],
                        row["ask"],
                        row["last"],
                        row["iv"],
                        row["volume"],
                        row["open_interest"],
                        spot,
                        pulled_at,
                    )
                )
    conn.executemany(
        """INSERT OR REPLACE INTO chain_snapshots
           (ticker, expiry, option_type, strike, asof_date, bid, ask, last, iv,
            volume, open_interest, spot, pulled_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


def get_iv_history(
    conn, ticker: str, expiry: str, option_type: str, strike: float, asof_date: str, lookback_days: int
) -> list[tuple[str, float]]:
    """IV history for one exact contract, from our own accumulated daily
    snapshots (never reconstructed -- only what we've actually pulled).
    """
    cur = conn.execute(
        """SELECT asof_date, iv FROM chain_snapshots
           WHERE ticker=? AND expiry=? AND option_type=? AND strike=?
             AND iv IS NOT NULL
             AND asof_date <= ?
             AND asof_date >= date(?, ?)
           ORDER BY asof_date""",
        (ticker, expiry, option_type, strike, asof_date, asof_date, f"-{lookback_days} days"),
    )
    return [(r["asof_date"], r["iv"]) for r in cur.fetchall()]


def get_prior_snapshot_dates(conn, ticker: str, before_date: str) -> list[str]:
    cur = conn.execute(
        "SELECT DISTINCT asof_date FROM chain_snapshots WHERE ticker=? AND asof_date < ? ORDER BY asof_date DESC",
        (ticker, before_date),
    )
    return [r["asof_date"] for r in cur.fetchall()]


def get_snapshot_keys(conn, ticker: str, asof_date: str) -> set[tuple[str, str, float]]:
    cur = conn.execute(
        "SELECT DISTINCT expiry, option_type, strike FROM chain_snapshots WHERE ticker=? AND asof_date=?",
        (ticker, asof_date),
    )
    return {(r["expiry"], r["option_type"], r["strike"]) for r in cur.fetchall()}


def save_valuations(conn, rows: list[tuple]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO valuations
           (asof_date, position_id, ticker, asset_type, mark, current_value,
            unrealized_pnl, unrealized_pnl_pct, dte, progress_to_target_pct,
            progress_to_stop_pct, delta, gamma, theta, vega)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


def save_allocation(conn, rows: list[tuple]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO allocation
           (asof_date, level, name, value, pct_of_total, flagged, cap_pct)
           VALUES (?,?,?,?,?,?,?)""",
        rows,
    )


def save_portfolio_greeks(
    conn, asof_date: str, net_delta_shares: float, total_daily_theta: float, net_vega: float
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO portfolio_greeks
           (asof_date, net_delta_shares, total_daily_theta, net_vega)
           VALUES (?,?,?,?)""",
        (asof_date, net_delta_shares, total_daily_theta, net_vega),
    )


def save_iv_environment(conn, rows: list[tuple]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO iv_environment
           (asof_date, position_id, ticker, current_iv, iv_rank, iv_percentile,
            history_days, status, rich, cheap)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


def save_upcoming_expiries(conn, rows: list[tuple]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO upcoming_expiries
           (asof_date, position_id, ticker, expiry, dte, within_threshold)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )


def save_macro_gate(conn, asof_date: str, macro: dict) -> None:
    vix_level = macro["vix_level"]
    term_structure = macro["term_structure"]
    breadth = macro["breadth"]
    credit_spread = macro["credit_spread"]
    conn.execute(
        """INSERT OR REPLACE INTO macro_gate
           (asof_date, score, weights_json, vix, vix_percentile, vix_level_score,
            vix3m, term_structure_ratio, term_structure_score,
            breadth_pct_above_200dma, breadth_constituents, breadth_score,
            credit_ratio, credit_percentile, credit_score)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            asof_date,
            macro["score"],
            json.dumps(macro["weights"]),
            vix_level["vix"],
            vix_level["vix_percentile"],
            vix_level["score"],
            term_structure["vix3m"],
            term_structure["ratio"],
            term_structure["score"],
            breadth["pct_above_200dma"],
            breadth["constituents"],
            breadth["score"],
            credit_spread["credit_ratio"],
            credit_spread["credit_percentile"],
            credit_spread["score"],
        ),
    )


def get_news_analysis(conn, asof_date: str, ticker: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM news_analysis WHERE asof_date=? AND ticker=?", (asof_date, ticker)
    ).fetchone()
    if row is None:
        return None
    return {
        "ticker": row["ticker"],
        "asof_date": row["asof_date"],
        "status": row["status"],
        "summary": row["summary"],
        "sentiment": row["sentiment"],
        "key_drivers": json.loads(row["key_drivers_json"]) if row["key_drivers_json"] else [],
        "position_flag": bool(row["position_flag"]),
        "position_flag_reason": row["position_flag_reason"],
        "headline_count": row["headline_count"],
        "window_days": row["window_days"],
        "model": row["model"],
        "parse_error": bool(row["parse_error"]),
    }


def save_news_analysis(conn, result: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO news_analysis
           (asof_date, ticker, status, summary, sentiment, key_drivers_json,
            position_flag, position_flag_reason, headline_count, window_days,
            model, parse_error, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result["asof_date"],
            result["ticker"],
            result.get("status"),
            result.get("summary"),
            result.get("sentiment"),
            json.dumps(result.get("key_drivers") or []),
            int(bool(result.get("position_flag"))),
            result.get("position_flag_reason"),
            result.get("headline_count"),
            result.get("window_days"),
            result.get("model"),
            int(bool(result.get("parse_error"))),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
