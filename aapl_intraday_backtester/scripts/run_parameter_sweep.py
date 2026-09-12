#!/usr/bin/env python
"""Run the parameter grid sweep with a strict train/validation/test split.

Optimization (the grid search + ranking) only ever touches the
train/validation windows defined in config.yaml's `optimizer` block. Once a
configuration is selected, it is evaluated unchanged on the held-out test
window and that result is clearly labeled as out-of-sample.

Usage:
    python scripts/run_parameter_sweep.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data_loader import load_bars
from src.optimizer import optimize_train_val_test
from src.report import generate_all_charts, generate_html_report
from src.utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bars = load_bars(cfg)
    if bars.empty:
        logger.error("No bars loaded — check config.yaml data settings.")
        sys.exit(1)

    opt = cfg.optimizer
    logger.info("Train: %s..%s | Validation: %s..%s | Test (out-of-sample): %s..%s",
                opt.train_start, opt.train_end, opt.validation_start, opt.validation_end,
                opt.test_start, opt.test_end)

    tvt = optimize_train_val_test(cfg, bars)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_labeled = tvt.train_results.copy()
    train_labeled["period"] = "train"
    val_labeled = tvt.validation_results.copy()
    val_labeled["period"] = "validation"
    combined = pd.concat([train_labeled, val_labeled], ignore_index=True)
    combined.to_csv(results_dir / "parameter_results.csv", index=False)

    tvt.ranked_on_validation.head(args.top_n).to_csv(results_dir / "best_configs.csv", index=False)

    tvt.out_of_sample_trades.to_csv(results_dir / "trades.csv", index=False)
    tvt.out_of_sample_equity_curve.to_csv(results_dir / "equity_curve.csv", index=False)

    charts = generate_all_charts(
        tvt.out_of_sample_trades, tvt.out_of_sample_equity_curve, cfg.risk.starting_equity,
        results_dir / "charts", param_results=combined,
    )
    note = (
        f"IN-SAMPLE optimization used train ({opt.train_start}..{opt.train_end}) and validation "
        f"({opt.validation_start}..{opt.validation_end}) data only. Selected config: {tvt.best_params or 'none met the trade-count threshold'}. "
        f"The charts/metrics below are the OUT-OF-SAMPLE test-period result ({opt.test_start}..{opt.test_end}), "
        f"produced by running that configuration unchanged — it was never used to pick the parameters."
    )
    generate_html_report(
        tvt.out_of_sample_metrics, {}, charts, results_dir / "report.html",
        title="AAPL Intraday Strategy — Out-of-Sample Parameter Sweep Result", extra_note=note,
    )

    logger.info("Best (validation-ranked) params: %s", tvt.best_params)
    logger.info("Out-of-sample trades: %s", tvt.out_of_sample_metrics.get("total_trades"))
    logger.info("Out-of-sample net return: %s%%", tvt.out_of_sample_metrics.get("total_return_pct"))
    logger.info("Wrote parameter_results.csv, best_configs.csv, trades.csv (OOS), equity_curve.csv (OOS), report.html -> %s",
                results_dir.resolve())


if __name__ == "__main__":
    main()
