"""Position sizing and daily risk-limit enforcement."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import RiskConfig


def compute_position_size(
    risk_cfg: RiskConfig,
    equity: float,
    entry_price: float,
    stop_price: float,
) -> int:
    """Return the number of shares to trade, honoring sizing mode and caps.

    Mode A ("fixed_dollar"): shares = fixed_dollar_amount / entry_price.
    Mode B ("risk"): shares = (equity * risk_per_trade_pct) / (entry_price - stop_price).

    Either way the position's notional value is capped at
    `equity * max_position_pct`, and share count is floored to a whole
    share unless `allow_fractional_shares` is set. No leverage: notional
    can never exceed available equity even if max_position_pct > 1.
    """
    per_share_risk = entry_price - stop_price
    if per_share_risk <= 0:
        return 0

    if risk_cfg.sizing_method == "fixed_dollar":
        dollars = risk_cfg.fixed_dollar_amount
    elif risk_cfg.sizing_method == "risk":
        risk_dollars = equity * risk_cfg.risk_per_trade_pct
        shares_by_risk = risk_dollars / per_share_risk
        dollars = shares_by_risk * entry_price
    else:
        raise ValueError(f"Unknown sizing_method: {risk_cfg.sizing_method!r}")

    max_dollars = min(equity, equity * risk_cfg.max_position_pct)
    dollars = min(dollars, max_dollars)

    shares = dollars / entry_price
    if not risk_cfg.allow_fractional_shares:
        shares = math.floor(shares)
    return max(shares, 0)


@dataclass
class DailyRiskState:
    """Tracks per-session counters used to enforce daily risk rules."""

    starting_equity_for_day: float
    trades_today: int = 0
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str | None = None

    def register_trade_close(self, net_pnl: float, risk_cfg: RiskConfig) -> None:
        self.trades_today += 1
        self.realized_pnl_today += net_pnl
        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self._evaluate_halt(risk_cfg)

    def mark_to_market_breach(self, hypothetical_pnl_if_closed_now: float, risk_cfg: RiskConfig) -> bool:
        """True if closing the open position right now would breach the
        daily loss cap — used to force-liquidate before the cap is exceeded."""
        total = self.realized_pnl_today + hypothetical_pnl_if_closed_now
        loss_limit = -abs(risk_cfg.max_daily_loss_pct) * self.starting_equity_for_day
        return total <= loss_limit

    def _evaluate_halt(self, risk_cfg: RiskConfig) -> None:
        loss_limit = -abs(risk_cfg.max_daily_loss_pct) * self.starting_equity_for_day
        if self.realized_pnl_today <= loss_limit:
            self.halted = True
            self.halt_reason = "max_daily_loss"
        elif self.trades_today >= risk_cfg.max_trades_per_day:
            self.halted = True
            self.halt_reason = "max_trades_per_day"
        elif self.consecutive_losses >= risk_cfg.max_consecutive_losses:
            self.halted = True
            self.halt_reason = "max_consecutive_losses"

    def can_open_new_trade(self, risk_cfg: RiskConfig) -> bool:
        if self.halted:
            return False
        if self.trades_today >= risk_cfg.max_trades_per_day:
            return False
        if self.consecutive_losses >= risk_cfg.max_consecutive_losses:
            return False
        return True
