"""Fill simulation: slippage, commissions, and intrabar ambiguity handling.

The intrabar-ambiguity problem (section 12 of the spec): with only OHLC for
each 1-minute bar, a single bar can have `low <= stop_price` AND
`high >= target_price` at the same time, and we have no way of knowing
whether price touched the stop or the target first. Assuming the
profitable outcome in that case silently inflates backtested performance.
`resolve_intrabar_exit` handles this explicitly and defaults to the
conservative assumption (stop first).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.config import ExecutionConfig

Side = Literal["buy", "sell"]
IntrabarPriority = Literal["conservative", "optimistic", "skip"]


def apply_slippage(price: float, side: Side, slippage_bps: float) -> float:
    """Buys fill worse (higher); sells fill worse (lower)."""
    factor = slippage_bps / 10_000.0
    if side == "buy":
        return price * (1.0 + factor)
    return price * (1.0 - factor)


def commission_cost(shares: float, cfg: ExecutionConfig) -> float:
    return shares * cfg.commission_per_share + cfg.commission_per_order


@dataclass
class FillResult:
    exit_reason: Optional[str]  # "take_profit" | "atr_stop" | "skip" | None
    raw_price: Optional[float]


def resolve_intrabar_exit(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    stop_price: float,
    target_price: float,
    priority: IntrabarPriority,
) -> FillResult:
    """Determine whether the stop and/or target were touched within one bar,
    and which one "wins" when both were touched (true order is unknowable).

    Returns raw (pre-slippage) execution price. Gaps are handled: if the
    bar's open is already through a level, the fill is assumed to occur at
    the open (which is worse for a stop, better for a target) rather than
    at the stale level price.
    """
    stop_hit = bar_low <= stop_price
    target_hit = bar_high >= target_price

    if not stop_hit and not target_hit:
        return FillResult(None, None)

    if stop_hit and not target_hit:
        fill = min(bar_open, stop_price)
        return FillResult("atr_stop", fill)

    if target_hit and not stop_hit:
        fill = max(bar_open, target_price)
        return FillResult("take_profit", fill)

    # Both touched in the same bar: true intrabar sequencing is unknown.
    if priority == "conservative":
        fill = min(bar_open, stop_price)
        return FillResult("atr_stop", fill)
    if priority == "optimistic":
        fill = max(bar_open, target_price)
        return FillResult("take_profit", fill)
    if priority == "skip":
        return FillResult("skip", None)
    raise ValueError(f"Unknown intrabar_priority: {priority!r}")


def entry_fill_price(
    signal_bar_close: float,
    next_bar_open: Optional[float],
    entry_method: str,
) -> Optional[float]:
    """Raw (pre-slippage) entry price per config.execution.entry_method.

    Defaults to next-bar-open: the signal is confirmed using bar N's fully
    formed OHLC, but we can't actually transact at bar N's own close in
    live trading, so the fill happens on bar N+1's open. This is what keeps
    the backtest from a hidden lookahead: computing a signal from bar N's
    close and pretending to fill at that same close bakes in perfect,
    zero-latency execution that never occurs in reality.
    """
    if entry_method == "next_bar_open":
        return next_bar_open  # None if there is no next bar (e.g. last bar of the dataset)
    if entry_method == "current_bar_close":
        return signal_bar_close
    raise ValueError(f"Unknown entry_method: {entry_method!r}")
