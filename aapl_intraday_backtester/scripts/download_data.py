#!/usr/bin/env python
"""Download (and cache) historical 1-minute AAPL bars from Alpaca.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY (env vars or .env file) and
config.yaml's data.source set to "alpaca". Downloads are cached per
calendar month under data/cache/, so re-running only fetches missing months
unless --force is passed.

Usage:
    python scripts/download_data.py --config config.yaml
    python scripts/download_data.py --start 2025-01-01 --end 2025-08-31 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_loader import download_alpaca_bars
from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start", default=None, help="Override data.start_date")
    parser.add_argument("--end", default=None, help="Override data.end_date")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end

    if cfg.data.source != "alpaca":
        logger.error("config.yaml data.source is %r, not 'alpaca'. Set it to 'alpaca' to use this script.",
                      cfg.data.source)
        sys.exit(1)

    df = download_alpaca_bars(cfg, force_refresh=args.force)
    logger.info("Done. %d total bars cached for %s in %s..%s",
                len(df), cfg.symbol, cfg.data.start_date, cfg.data.end_date)


if __name__ == "__main__":
    main()
