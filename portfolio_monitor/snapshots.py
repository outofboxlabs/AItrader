"""Dated JSON chain snapshots on disk, plus new-strike / new-expiry diffing
against the prior run (both backed by the SQLite mirror in db.py)."""

from __future__ import annotations

import json
import os
from datetime import date


def snapshot_path(snapshots_dir: str, ticker: str, asof_date: date) -> str:
    return os.path.join(snapshots_dir, f"{ticker}_{asof_date.isoformat()}.json")


def save_snapshot_json(snapshots_dir: str, ticker: str, asof_date: date, chain: dict) -> str:
    os.makedirs(snapshots_dir, exist_ok=True)
    path = snapshot_path(snapshots_dir, ticker, asof_date)
    with open(path, "w") as f:
        json.dump(chain, f, indent=2, sort_keys=True)
    return path


def diff_new_strikes_and_expiries(
    prior_keys: set[tuple[str, str, float]], current_keys: set[tuple[str, str, float]]
) -> dict:
    """prior_keys/current_keys are sets of (expiry, option_type, strike)."""
    new_expiries = sorted({k[0] for k in current_keys} - {k[0] for k in prior_keys})
    new_strikes = sorted(current_keys - prior_keys)
    return {"new_expiries": new_expiries, "new_strikes": new_strikes}
