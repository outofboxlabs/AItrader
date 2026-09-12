#!/usr/bin/env python
"""Roll a training window forward through the data, re-optimizing on each
train window and testing, unchanged, on the following out-of-sample window.

Usage:
    python scripts/run_walk_forward.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_loader import load_bars
from src.optimizer import run_walk_forward, summarize_walk_forward
from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bars = load_bars(cfg)
    if bars.empty:
        logger.error("No bars loaded — check config.yaml data settings.")
        sys.exit(1)

    windows = run_walk_forward(cfg, bars)
    if not windows:
        logger.warning(
            "No walk-forward windows fit inside the available data range. "
            "Need at least train_months + test_months of data (see config.yaml walk_forward)."
        )
        return

    summary = summarize_walk_forward(windows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "walk_forward_results.csv", index=False)

    logger.info("=== Walk-forward summary (%d windows) ===", len(windows))
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Mean out-of-sample net return: %.2f%% | Std dev: %.2f%% | Positive windows: %d/%d",
                summary["net_return_pct"].mean(), summary["net_return_pct"].std(),
                (summary["net_return_pct"] > 0).sum(), len(summary))
    logger.info("Wrote walk_forward_results.csv -> %s", results_dir.resolve())


if __name__ == "__main__":
    main()
