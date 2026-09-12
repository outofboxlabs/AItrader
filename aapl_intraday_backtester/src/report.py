"""Chart generation (matplotlib only, no seaborn) and the human-readable
results/report.html summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import equity_metrics

CHART_STYLE = {"figsize": (9, 4.5), "dpi": 110}


def _savefig(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=CHART_STYLE["dpi"])
    plt.close(fig)
    return path.name


def plot_equity_curve(equity_curve: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    if not equity_curve.empty:
        ax.plot(pd.to_datetime(equity_curve["timestamp"]), equity_curve["equity"], linewidth=1)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _savefig(fig, out_dir, "equity_curve")


def plot_drawdown(equity_curve: pd.DataFrame, starting_equity: float, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    em = equity_metrics(equity_curve, starting_equity)
    dd = em.get("drawdown_series")
    if dd is not None and len(dd):
        ax.fill_between(pd.to_datetime(dd.index), dd.values * 100, 0, color="crimson", alpha=0.5)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _savefig(fig, out_dir, "drawdown")


def plot_monthly_returns(equity_curve: pd.DataFrame, starting_equity: float, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    if not equity_curve.empty:
        eq = equity_curve.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"])
        eq = eq.set_index("timestamp")
        month_end_equity = eq["equity"].resample("ME").last()
        prev = month_end_equity.shift(1)
        prev.iloc[0] = starting_equity
        monthly_ret = (month_end_equity / prev - 1.0) * 100
        colors = ["seagreen" if v >= 0 else "crimson" for v in monthly_ret]
        ax.bar(monthly_ret.index.strftime("%Y-%m"), monthly_ret.values, color=colors)
    ax.set_title("Monthly Returns (%)")
    ax.set_ylabel("Return %")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _savefig(fig, out_dir, "monthly_returns")


def plot_trade_return_distribution(trades: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    if not valid.empty:
        ax.hist(valid["net_return_pct"] * 100, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Trade Net Return Distribution (%)")
    ax.set_xlabel("Net Return %")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    return _savefig(fig, out_dir, "trade_return_distribution")


def plot_win_loss(trades: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    wins = (valid["net_pnl"] > 0).sum() if not valid.empty else 0
    losses = (valid["net_pnl"] <= 0).sum() if not valid.empty else 0
    ax.bar(["Wins", "Losses"], [wins, losses], color=["seagreen", "crimson"])
    ax.set_title("Win / Loss Count")
    ax.grid(alpha=0.3, axis="y")
    return _savefig(fig, out_dir, "win_loss")


def plot_holding_times(trades: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    if not valid.empty:
        ax.hist(valid["minutes_held"], bins=30, color="darkorange", edgecolor="white")
    ax.set_title("Holding Time Distribution (minutes)")
    ax.set_xlabel("Minutes Held")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    return _savefig(fig, out_dir, "holding_times")


def plot_pnl_by_hour(trades: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    if not valid.empty:
        hours = pd.to_datetime(valid["entry_timestamp"]).dt.hour
        by_hour = valid.groupby(hours)["net_pnl"].sum()
        colors = ["seagreen" if v >= 0 else "crimson" for v in by_hour]
        ax.bar(by_hour.index.astype(str), by_hour.values, color=colors)
    ax.set_title("Net P&L by Hour of Day (entry hour, ET)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Net P&L ($)")
    ax.grid(alpha=0.3, axis="y")
    return _savefig(fig, out_dir, "pnl_by_hour")


def plot_pnl_by_weekday(trades: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    if not valid.empty:
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekdays = pd.to_datetime(valid["entry_timestamp"]).dt.dayofweek
        by_wd = valid.groupby(weekdays)["net_pnl"].sum().reindex(range(5)).fillna(0)
        colors = ["seagreen" if v >= 0 else "crimson" for v in by_wd]
        ax.bar([weekday_names[i] for i in by_wd.index], by_wd.values, color=colors)
    ax.set_title("Net P&L by Weekday")
    ax.set_ylabel("Net P&L ($)")
    ax.grid(alpha=0.3, axis="y")
    return _savefig(fig, out_dir, "pnl_by_weekday")


def plot_exit_reasons(trades: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    valid = trades[trades["exit_reason"] != "skip"] if not trades.empty else trades
    if not valid.empty:
        counts = valid["exit_reason"].value_counts()
        ax.bar(counts.index, counts.values, color="slateblue")
    ax.set_title("Exit Reason Counts")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")
    return _savefig(fig, out_dir, "exit_reasons")


def plot_parameter_heatmap(param_results: pd.DataFrame, x: str, y: str, value: str, out_dir: Path, name: str) -> str | None:
    """Average `value` over all other swept params, for each (x, y) pair."""
    if param_results.empty or x not in param_results or y not in param_results:
        return None
    pivot = param_results.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{value} by {x} / {y}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    return _savefig(fig, out_dir, name)


def generate_all_charts(trades: pd.DataFrame, equity_curve: pd.DataFrame, starting_equity: float,
                         out_dir: Path, param_results: pd.DataFrame | None = None) -> dict[str, str]:
    charts = {
        "equity_curve": plot_equity_curve(equity_curve, out_dir),
        "drawdown": plot_drawdown(equity_curve, starting_equity, out_dir),
        "monthly_returns": plot_monthly_returns(equity_curve, starting_equity, out_dir),
        "trade_return_distribution": plot_trade_return_distribution(trades, out_dir),
        "win_loss": plot_win_loss(trades, out_dir),
        "holding_times": plot_holding_times(trades, out_dir),
        "pnl_by_hour": plot_pnl_by_hour(trades, out_dir),
        "pnl_by_weekday": plot_pnl_by_weekday(trades, out_dir),
        "exit_reasons": plot_exit_reasons(trades, out_dir),
    }
    if param_results is not None and not param_results.empty:
        heatmap = plot_parameter_heatmap(
            param_results, "take_profit_pct", "atr_multiplier", "net_return_pct", out_dir, "heatmap_tp_atr"
        )
        if heatmap:
            charts["heatmap_tp_atr"] = heatmap
        heatmap2 = plot_parameter_heatmap(
            param_results, "pullback_pct", "max_hold_minutes", "net_return_pct", out_dir, "heatmap_pullback_hold"
        )
        if heatmap2:
            charts["heatmap_pullback_hold"] = heatmap2
    return charts


def _fmt(v, pct=False, money=False) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if pct:
        return f"{v:.2f}%"
    if money:
        return f"${v:,.2f}"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def generate_html_report(
    metrics: dict,
    benchmark_metrics: dict,
    charts: dict[str, str],
    output_path: Path,
    charts_subdir: str = "charts",
    title: str = "AAPL Intraday Pullback/Reversal Strategy — Backtest Report",
    extra_note: str = "",
) -> Path:
    """Write results/report.html summarizing metrics + embedding chart images."""
    exit_reason_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.1f}%</td></tr>" for k, v in metrics.get("exit_reason_pct", {}).items()
    )

    benchmark_rows = "".join(
        f"<tr><td>{name}</td><td>{_fmt(vals.get('total_return_pct'), pct=True)}</td></tr>"
        for name, vals in benchmark_metrics.items()
    )

    chart_imgs = "".join(
        f'<div class="chart"><h3>{name.replace("_", " ").title()}</h3>'
        f'<img src="{charts_subdir}/{fname}" /></div>'
        for name, fname in charts.items()
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 2rem; color: #1a1a1a; background:#fafafa; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ margin-top: 2rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem 0; }}
td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; font-size: 0.9rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.5rem 2rem; }}
.chart {{ margin: 1rem 0; background: white; padding: 0.5rem; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }}
.chart img {{ max-width: 100%; }}
.note {{ background: #fff3cd; border: 1px solid #ffe08a; padding: 0.75rem 1rem; border-radius: 6px; }}
</style></head>
<body>
<h1>{title}</h1>
<p>{extra_note}</p>

<h2>Summary</h2>
<div class="grid">
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total trades</td><td>{_fmt(metrics.get('total_trades'))}</td></tr>
<tr><td>Win rate</td><td>{_fmt((metrics.get('win_rate') or 0) * 100, pct=True)}</td></tr>
<tr><td>Profit factor</td><td>{_fmt(metrics.get('profit_factor'))}</td></tr>
<tr><td>Expectancy / trade</td><td>{_fmt((metrics.get('expectancy_pct') or 0) * 100, pct=True)}</td></tr>
<tr><td>Gross P&amp;L</td><td>{_fmt(metrics.get('gross_pnl'), money=True)}</td></tr>
<tr><td>Net P&amp;L</td><td>{_fmt(metrics.get('net_pnl'), money=True)}</td></tr>
</table>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total return</td><td>{_fmt(metrics.get('total_return_pct'), pct=True)}</td></tr>
<tr><td>Annualized return</td><td>{_fmt(metrics.get('annualized_return_pct'), pct=True)}</td></tr>
<tr><td>Max drawdown</td><td>{_fmt(metrics.get('max_drawdown_pct'), pct=True)}</td></tr>
<tr><td>Sharpe ratio</td><td>{_fmt(metrics.get('sharpe_ratio'))}</td></tr>
<tr><td>Sortino ratio</td><td>{_fmt(metrics.get('sortino_ratio'))}</td></tr>
<tr><td>Calmar ratio</td><td>{_fmt(metrics.get('calmar_ratio'))}</td></tr>
</table>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Avg winner</td><td>{_fmt((metrics.get('average_winner_pct') or 0) * 100, pct=True)}</td></tr>
<tr><td>Avg loser</td><td>{_fmt((metrics.get('average_loser_pct') or 0) * 100, pct=True)}</td></tr>
<tr><td>Largest winner</td><td>{_fmt((metrics.get('largest_winner_pct') or 0) * 100, pct=True)}</td></tr>
<tr><td>Largest loser</td><td>{_fmt((metrics.get('largest_loser_pct') or 0) * 100, pct=True)}</td></tr>
<tr><td>Avg holding (min)</td><td>{_fmt(metrics.get('average_holding_minutes'))}</td></tr>
<tr><td>Median holding (min)</td><td>{_fmt(metrics.get('median_holding_minutes'))}</td></tr>
<tr><td>Max holding (min)</td><td>{_fmt(metrics.get('max_holding_minutes'))}</td></tr>
<tr><td>Max consecutive wins</td><td>{_fmt(metrics.get('max_consecutive_wins'))}</td></tr>
<tr><td>Max consecutive losses</td><td>{_fmt(metrics.get('max_consecutive_losses'))}</td></tr>
<tr><td>Avg MFE</td><td>{_fmt(metrics.get('average_mfe'), money=False)}</td></tr>
<tr><td>Avg MAE</td><td>{_fmt(metrics.get('average_mae'), money=False)}</td></tr>
</table>
</div>

<h2>Exit Reason Breakdown</h2>
<table><tr><th>Reason</th><th>% of trades</th></tr>{exit_reason_rows}</table>

<h2>Benchmarks (overnight exposure: 0% — strategy is fully intraday)</h2>
<table><tr><th>Benchmark</th><th>Total Return</th></tr>{benchmark_rows}</table>

<h2>Charts</h2>
{chart_imgs}

</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
