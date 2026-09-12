#!/usr/bin/env python
"""Download (and cache) historical 1-minute AAPL bars from a real data source.

Two sources are supported, per config.yaml's data.source (or --source):

* "alpaca" — requires ALPACA_API_KEY / ALPACA_SECRET_KEY (env vars or .env
  file). Downloads are cached per calendar month under data/cache/.
* "yahoo"  — no credentials needed (uses the `yfinance` package), but Yahoo
  only serves 1-minute intraday history for roughly the trailing 30 calendar
  days, regardless of what start date you ask for. Downloads are cached per
  <=7-day chunk under data/cache/.

Either way, re-running only fetches whatever isn't already cached unless
--force is passed.

Usage:
    python scripts/download_data.py --config config.yaml
    python scripts/download_data.py --source yahoo --start 2026-08-13 --end 2026-09-12
    python scripts/download_data.py --source alpaca --start 2025-01-01 --end 2025-08-31 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_loader import download_alpaca_bars, download_yahoo_bars
from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, choices=["alpaca", "yahoo"], help="Override data.source")
    parser.add_argument("--start", default=None, help="Override data.start_date")
    parser.add_argument("--end", default=None, help="Override data.end_date")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.data.source = args.source
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end

    if cfg.data.source == "alpaca":
        df = download_alpaca_bars(cfg, force_refresh=args.force)
    elif cfg.data.source == "yahoo":
        df = download_yahoo_bars(cfg, force_refresh=args.force)
    else:
        logger.error("data.source is %r. Use --source alpaca or --source yahoo with this script.", cfg.data.source)
        sys.exit(1)

    logger.info("Done. %d total bars cached for %s (requested %s..%s; see warnings above for any clipping)",
                len(df), cfg.symbol, cfg.data.start_date, cfg.data.end_date)


if __name__ == "__main__":
    main()
