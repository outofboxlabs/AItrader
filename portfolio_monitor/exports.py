"""Timestamped CSV exports for scan results (market movers, growth
screener). Each scan type gets its own subfolder so files don't mix."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime


def export_to_csv(rows: list[dict], subfolder: str, name_prefix: str, export_root: str = "exports") -> str:
    """Write `rows` to exports/<subfolder>/<name_prefix>_<timestamp>.csv and
    return the path written. Any nested list/dict value in a row is
    JSON-encoded into its cell so no data is silently dropped."""
    out_dir = os.path.join(export_root, subfolder)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(out_dir, f"{name_prefix}_{timestamp}.csv")

    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return path

    fieldnames = list({key for row in rows for key in row.keys()})
    flat_rows = []
    for row in rows:
        flat = {}
        for key in fieldnames:
            value = row.get(key)
            flat[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        flat_rows.append(flat)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return path
