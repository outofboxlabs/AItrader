"""Performance metrics computed from a trade log + equity curve.

Deliberately reports many angles (return, drawdown, risk-adjusted ratios,
trade-quality stats, exit-reason mix) rather than a single headline number:
section 29 of the spec is explicit that win rate and total return alone are
not sufficient grounds to call a strategy good.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _max_streak(mask: pd.Series) -> int:
    """Longest run of consecutive True values in a boolean series."""
    if mask.empty:
        return 0
    groups = (mask != mask.shift()).cumsum()
    streak_lengths = mask.groupby(groups).transform("size")
    return int(streak_lengths[mask].max()) if mask.any() else 0


def trade_metrics(trades: pd.DataFrame) -> dict:
    """Metrics derived purely from the trade log. Excludes 'skip' (discarded,
    intrabar-ambiguous) trades from every statistic — see backtester.py."""
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    n = len(valid)
    out: dict = {"total_trades": n}
    if n == 0:
        return out | {
            "winning_trades": 0, "losing_trades": 0, "win_rate": np.nan,
            "average_winner_pct": np.nan, "average_loser_pct": np.nan,
            "profit_factor": np.nan, "expectancy_pct": np.nan,
            "gross_pnl": 0.0, "net_pnl": 0.0,
            "largest_winner_pct": np.nan, "largest_loser_pct": np.nan,
            "average_holding_minutes": np.nan, "median_holding_minutes": np.nan,
            "max_holding_minutes": np.nan,
            "max_consecutive_wins": 0, "max_consecutive_losses": 0,
            "average_mfe": np.nan, "average_mae": np.nan,
            "exit_reason_pct": {},
        }

    wins = valid[valid["net_pnl"] > 0]
    losses = valid[valid["net_pnl"] <= 0]
    gross_wins = wins["net_pnl"].sum()
    gross_losses = losses["net_pnl"].sum()

    out.update({
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / n,
        "average_winner_pct": wins["net_return_pct"].mean() if len(wins) else np.nan,
        "average_loser_pct": losses["net_return_pct"].mean() if len(losses) else np.nan,
        "profit_factor": (gross_wins / abs(gross_losses)) if gross_losses < 0 else np.nan,
        "expectancy_pct": valid["net_return_pct"].mean(),
        "gross_pnl": valid["gross_pnl"].sum(),
        "net_pnl": valid["net_pnl"].sum(),
        "largest_winner_pct": valid["net_return_pct"].max(),
        "largest_loser_pct": valid["net_return_pct"].min(),
        "average_holding_minutes": valid["minutes_held"].mean(),
        "median_holding_minutes": valid["minutes_held"].median(),
        "max_holding_minutes": valid["minutes_held"].max(),
        "max_consecutive_wins": _max_streak(valid["net_pnl"] > 0),
        "max_consecutive_losses": _max_streak(valid["net_pnl"] <= 0),
        "average_mfe": valid["maximum_favorable_excursion"].mean(),
        "average_mae": valid["maximum_adverse_excursion"].mean(),
        "exit_reason_pct": (valid["exit_reason"].value_counts(normalize=True) * 100).round(2).to_dict(),
    })
    return out


def equity_metrics(equity_curve: pd.DataFrame, starting_equity: float) -> dict:
    """Metrics derived from the mark-to-market equity curve: return, drawdown,
    and risk-adjusted ratios (Sharpe/Sortino/Calmar)."""
    if equity_curve.empty:
        return {
            "total_return_pct": 0.0, "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan, "calmar_ratio": np.nan,
        }

    eq = equity_curve.copy()
    eq["date"] = pd.to_datetime(eq["timestamp"]).dt.date
    daily_close_equity = eq.groupby("date")["equity"].last()

    final_equity = daily_close_equity.iloc[-1]
    total_return = final_equity / starting_equity - 1.0

    n_days = len(daily_close_equity)
    years = n_days / TRADING_DAYS_PER_YEAR if n_days else np.nan
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    running_max = daily_close_equity.cummax()
    drawdown = (daily_close_equity - running_max) / running_max
    max_drawdown = drawdown.min()

    daily_returns = daily_close_equity.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0:
        sharpe = (daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = np.nan

    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std(ddof=0) > 0:
        sortino = (daily_returns.mean() / downside.std(ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sortino = np.nan

    calmar = (annualized_return / abs(max_drawdown)) if max_drawdown < 0 and annualized_return == annualized_return else np.nan

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": annualized_return * 100 if annualized_return == annualized_return else np.nan,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "daily_equity": daily_close_equity,
        "drawdown_series": drawdown,
    }


def full_report(trades: pd.DataFrame, equity_curve: pd.DataFrame, starting_equity: float) -> dict:
    """Combine trade-level and equity-level metrics into one dict."""
    tm = trade_metrics(trades)
    em = equity_metrics(equity_curve, starting_equity)
    return {**tm, **em, "starting_equity": starting_equity,
            "final_equity": starting_equity * (1 + em.get("total_return_pct", 0) / 100)}


def benchmark_buy_and_hold(bars: pd.DataFrame) -> dict:
    """Simple buy-and-hold return over the same bar dataset (first open to last close)."""
    if bars.empty:
        return {"total_return_pct": np.nan}
    first_price = bars.iloc[0]["open"]
    last_price = bars.iloc[-1]["close"]
    return {"total_return_pct": (last_price / first_price - 1.0) * 100}
