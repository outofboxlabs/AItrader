"""Open-position state machine: stop/target management, profit protection,
time stop and end-of-day rules. Intrabar fill mechanics live in execution.py;
this module is only concerned with *what* the exit levels should be at any
point in a trade's life, never with resolving same-bar ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import ProfitProtectionConfig


@dataclass
class Position:
    entry_timestamp: pd.Timestamp
    entry_price: float
    raw_entry_price: float
    shares: float
    atr_at_entry: float
    initial_stop: float
    target_price: float
    pullback_percent: float
    max_hold_minutes: int
    deadline: pd.Timestamp

    current_stop: float = field(init=False)
    mfe: float = 0.0  # max favorable excursion, in price terms (best close - entry)
    mae: float = 0.0  # max adverse excursion, in price terms (entry - worst close), positive = bad
    profit_protection_stage: str = "none"  # "none" -> "breakeven" -> "locked"

    def __post_init__(self) -> None:
        self.current_stop = self.initial_stop

    def update_excursion(self, bar_high: float, bar_low: float) -> None:
        self.mfe = max(self.mfe, bar_high - self.entry_price)
        self.mae = max(self.mae, self.entry_price - bar_low)

    def tighten_stop(self, new_stop: float) -> None:
        """Stops may only move up (toward/through entry), never widen."""
        if new_stop > self.current_stop:
            self.current_stop = new_stop


def apply_profit_protection(position: Position, current_price: float, cfg: ProfitProtectionConfig) -> None:
    """Ratchet the stop up as unrealized profit grows, per config thresholds.
    A no-op when cfg.enabled is False. Never widens the stop (see tighten_stop)."""
    if not cfg.enabled:
        return

    unrealized_pct = (current_price - position.entry_price) / position.entry_price

    if position.profit_protection_stage == "none" and unrealized_pct >= cfg.breakeven_trigger_pct:
        breakeven_stop = position.entry_price * (1.0 + cfg.breakeven_buffer_pct)
        position.tighten_stop(breakeven_stop)
        position.profit_protection_stage = "breakeven"

    if position.profit_protection_stage in ("none", "breakeven") and unrealized_pct >= cfg.lock_trigger_pct:
        lock_stop = position.entry_price * (1.0 + cfg.lock_pct)
        position.tighten_stop(lock_stop)
        position.profit_protection_stage = "locked"


def is_time_stop_hit(position: Position, current_timestamp: pd.Timestamp) -> bool:
    return current_timestamp >= position.deadline


def is_eod_force_close(current_timestamp: pd.Timestamp, force_close_ts: pd.Timestamp) -> bool:
    return current_timestamp >= force_close_ts


def is_past_new_entry_cutoff(current_timestamp: pd.Timestamp, stop_new_entries_ts: pd.Timestamp) -> bool:
    return current_timestamp >= stop_new_entries_ts
