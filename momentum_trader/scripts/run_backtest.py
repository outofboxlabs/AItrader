#!/usr/bin/env python
"""Run a single backtest with the parameters in config.yaml and write
results/trades.csv, results/equity_curve.csv, and results/report.html.

Usage:
    python scripts/run_backtest.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtester import run_backtest
from src.config import load_config
from src.data_loader import load_bars
from src.metrics import benchmark_buy_and_hold, full_report
from src.report import generate_all_charts, generate_html_report
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

    logger.info("Running baseline backtest for %s (%s -> %s)", cfg.symbol, cfg.data.start_date, cfg.data.end_date)
    result = run_backtest(cfg, bars)
    metrics = full_report(result.trades, result.equity_curve, result.starting_equity)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(results_dir / "trades.csv", index=False)
    result.equity_curve.to_csv(results_dir / "equity_curve.csv", index=False)

    benchmark_metrics = {"AAPL buy & hold": benchmark_buy_and_hold(bars)}

    charts = generate_all_charts(result.trades, result.equity_curve, result.starting_equity, results_dir / "charts")
    note = (
        f"Baseline configuration: pullback={cfg.strategy.pullback_pct:.2%}, "
        f"take_profit={cfg.strategy.take_profit_pct:.2%}, atr_multiplier={cfg.strategy.atr_multiplier}, "
        f"max_hold={cfg.strategy.max_hold_minutes}min, intrabar={cfg.execution.intrabar_priority}, "
        f"entry={cfg.execution.entry_method}. Overnight exposure: 0% (strategy is fully intraday)."
    )
    generate_html_report(metrics, benchmark_metrics, charts, results_dir / "report.html", extra_note=note)

    logger.info("=== Results ===")
    for k in ("total_trades", "win_rate", "profit_factor", "expectancy_pct", "net_pnl",
              "total_return_pct", "max_drawdown_pct", "sharpe_ratio"):
        logger.info("%-22s %s", k, metrics.get(k))
    logger.info("Wrote trades.csv, equity_curve.csv, report.html -> %s", results_dir.resolve())


if __name__ == "__main__":
    main()
