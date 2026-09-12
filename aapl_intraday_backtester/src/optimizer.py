"""Parameter sweep, multi-metric ranking, train/validation/test split, and
walk-forward testing.

The one rule this whole module exists to enforce: **parameter selection
never touches the out-of-sample test window**. Every grid combination is
scored only on train+validation data; only after a single configuration is
chosen does it get run — unchanged — on the held-out test period (or, in
walk-forward mode, on the window immediately following its own training
window, never seen during that window's own optimization).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtester import run_backtest
from src.config import BacktestConfig
from src.metrics import full_report
from src.utils import get_logger

logger = get_logger(__name__)

RANK_METRICS = {
    # metric_name: higher_is_better
    "net_return_pct": True,
    "profit_factor": True,
    "max_drawdown_pct": True,   # less negative is better -> still "higher is better" on the raw signed value
    "sharpe_ratio": True,
    "expectancy_pct": True,
    "total_trades": True,
}


def slice_by_date(bars: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_d, end_d = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    mask = (bars["date"] >= start_d) & (bars["date"] <= end_d)
    return bars.loc[mask].reset_index(drop=True)


def _grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _score_one(cfg: BacktestConfig, bars: pd.DataFrame, overrides: dict) -> dict:
    variant_cfg = cfg.clone_with_strategy_overrides(**overrides)
    result = run_backtest(variant_cfg, bars)
    metrics = full_report(result.trades, result.equity_curve, result.starting_equity)
    row = {**overrides}
    row["net_return_pct"] = metrics.get("total_return_pct", np.nan)
    row["profit_factor"] = metrics.get("profit_factor", np.nan)
    row["max_drawdown_pct"] = metrics.get("max_drawdown_pct", np.nan)
    row["sharpe_ratio"] = metrics.get("sharpe_ratio", np.nan)
    row["sortino_ratio"] = metrics.get("sortino_ratio", np.nan)
    row["expectancy_pct"] = metrics.get("expectancy_pct", np.nan)
    row["win_rate"] = metrics.get("win_rate", np.nan)
    row["total_trades"] = metrics.get("total_trades", 0)
    row["net_pnl"] = metrics.get("net_pnl", 0.0)
    return row


def run_grid_search(cfg: BacktestConfig, bars: pd.DataFrame, grid: dict) -> pd.DataFrame:
    """Run every combination in `grid` against `bars` and return one row per combo."""
    combos = _grid_combinations(grid)
    logger.info("Running %d parameter combinations", len(combos))
    rows = [_score_one(cfg, bars, combo) for combo in combos]
    return pd.DataFrame(rows)


def rank_configs(results: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """Rank parameter combinations across several metrics at once (never by
    total return alone), penalizing configs with too few trades to trust.
    """
    if results.empty:
        return results

    df = results.copy()
    df["sufficient_sample"] = df["total_trades"] >= min_trades

    rank_cols = []
    for metric, higher_better in RANK_METRICS.items():
        if metric not in df.columns:
            continue
        pct_rank = df[metric].rank(pct=True, ascending=higher_better, na_option="bottom")
        rank_col = f"_rank_{metric}"
        df[rank_col] = pct_rank
        rank_cols.append(rank_col)

    df["composite_score"] = df[rank_cols].mean(axis=1)
    # Configs without enough trades are never allowed to rank as "best", regardless of score.
    df["composite_score"] = np.where(df["sufficient_sample"], df["composite_score"], -1.0)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


@dataclass
class TrainValTestResult:
    train_results: pd.DataFrame
    validation_results: pd.DataFrame
    ranked_on_validation: pd.DataFrame
    best_params: dict
    out_of_sample_metrics: dict
    out_of_sample_trades: pd.DataFrame
    out_of_sample_equity_curve: pd.DataFrame


def optimize_train_val_test(cfg: BacktestConfig, bars: pd.DataFrame) -> TrainValTestResult:
    """Grid search on train+validation only; evaluate the chosen config, unchanged, on test."""
    opt = cfg.optimizer
    train_bars = slice_by_date(bars, opt.train_start, opt.train_end)
    val_bars = slice_by_date(bars, opt.validation_start, opt.validation_end)
    test_bars = slice_by_date(bars, opt.test_start, opt.test_end)

    train_results = run_grid_search(cfg, train_bars, opt.grid)
    val_results = run_grid_search(cfg, val_bars, opt.grid)

    ranked = rank_configs(val_results, opt.min_trades_for_ranking)
    if ranked.empty or ranked.iloc[0]["composite_score"] < 0:
        logger.warning("No configuration met the minimum trade-count threshold on validation data")
        best_params = {}
    else:
        param_keys = list(opt.grid.keys())
        best_params = {k: ranked.iloc[0][k] for k in param_keys}

    logger.info("Selected out-of-sample config: %s", best_params)
    oos_cfg = cfg.clone_with_strategy_overrides(**best_params) if best_params else cfg
    oos_result = run_backtest(oos_cfg, test_bars)
    oos_metrics = full_report(oos_result.trades, oos_result.equity_curve, oos_result.starting_equity)

    return TrainValTestResult(
        train_results=train_results,
        validation_results=val_results,
        ranked_on_validation=ranked,
        best_params=best_params,
        out_of_sample_metrics=oos_metrics,
        out_of_sample_trades=oos_result.trades,
        out_of_sample_equity_curve=oos_result.equity_curve,
    )


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    test_metrics: dict


def _month_windows(bars: pd.DataFrame, train_months: int, test_months: int, step_months: int) -> list[tuple]:
    dates = sorted(bars["date"].unique())
    if not dates:
        return []
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    windows = []
    cur_train_start = start
    while True:
        train_end = cur_train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end:
            break
        windows.append((cur_train_start, train_end, test_start, test_end))
        cur_train_start = cur_train_start + pd.DateOffset(months=step_months)
    return windows


def run_walk_forward(cfg: BacktestConfig, bars: pd.DataFrame) -> list[WalkForwardWindow]:
    """Roll a train window forward through the data, optimizing on each train
    window and testing, unchanged, on the immediately following test window."""
    wf = cfg.walk_forward
    windows = _month_windows(bars, wf.train_months, wf.test_months, wf.step_months)
    logger.info("Walk-forward: %d windows (train=%dmo, test=%dmo, step=%dmo)",
                len(windows), wf.train_months, wf.test_months, wf.step_months)

    results = []
    for train_start, train_end, test_start, test_end in windows:
        train_bars = slice_by_date(bars, str(train_start.date()), str(train_end.date()))
        test_bars = slice_by_date(bars, str(test_start.date()), str(test_end.date()))

        grid_results = run_grid_search(cfg, train_bars, cfg.optimizer.grid)
        ranked = rank_configs(grid_results, cfg.optimizer.min_trades_for_ranking)

        if ranked.empty or ranked.iloc[0]["composite_score"] < 0:
            best_params = {}
        else:
            param_keys = list(cfg.optimizer.grid.keys())
            best_params = {k: ranked.iloc[0][k] for k in param_keys}

        test_cfg = cfg.clone_with_strategy_overrides(**best_params) if best_params else cfg
        test_result = run_backtest(test_cfg, test_bars)
        test_metrics = full_report(test_result.trades, test_result.equity_curve, test_result.starting_equity)

        results.append(WalkForwardWindow(train_start, train_end, test_start, test_end, best_params, test_metrics))
        logger.info("Window train=%s..%s test=%s..%s params=%s net_return=%.2f%% trades=%d",
                    train_start.date(), train_end.date(), test_start.date(), test_end.date(),
                    best_params, test_metrics.get("total_return_pct", float("nan")),
                    test_metrics.get("total_trades", 0))
    return results


def summarize_walk_forward(windows: list[WalkForwardWindow]) -> pd.DataFrame:
    rows = []
    for w in windows:
        rows.append({
            "train_start": w.train_start.date(), "train_end": w.train_end.date(),
            "test_start": w.test_start.date(), "test_end": w.test_end.date(),
            **w.best_params,
            "net_return_pct": w.test_metrics.get("total_return_pct"),
            "sharpe_ratio": w.test_metrics.get("sharpe_ratio"),
            "profit_factor": w.test_metrics.get("profit_factor"),
            "max_drawdown_pct": w.test_metrics.get("max_drawdown_pct"),
            "total_trades": w.test_metrics.get("total_trades"),
            "win_rate": w.test_metrics.get("win_rate"),
        })
    return pd.DataFrame(rows)
