"""Per-position valuation: mark, current value, unrealized P&L, DTE,
progress to target/stop, and position-level greeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import data as data_mod
from . import greeks as greeks_mod
from .models import Position


@dataclass
class PositionValuation:
    position: Position
    mark: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: Optional[float]
    dte: Optional[int]
    progress_to_target_pct: Optional[float]
    progress_to_stop_pct: Optional[float]
    # Position-level (already scaled by contracts * multiplier).
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    current_iv: Optional[float] = None


def _progress_pct(current: float, entry: float, target: Optional[float]) -> Optional[float]:
    """% of the way from entry to target/stop. Can exceed 100% (moved past
    the level) or go negative (moved the wrong way)."""
    if target is None or target == entry:
        return None
    return (current - entry) / (target - entry) * 100.0


def value_shares_position(position: Position, spot: float) -> PositionValuation:
    mark = spot
    current_value = mark * position.contracts
    cost_basis = position.entry_price * position.contracts
    pnl = current_value - cost_basis
    pnl_pct = (pnl / cost_basis * 100.0) if cost_basis else None
    return PositionValuation(
        position=position,
        mark=mark,
        current_value=current_value,
        unrealized_pnl=pnl,
        unrealized_pnl_pct=pnl_pct,
        dte=None,
        progress_to_target_pct=_progress_pct(mark, position.entry_price, position.target_price),
        progress_to_stop_pct=_progress_pct(mark, position.entry_price, position.stop_price),
        delta=position.contracts,  # 1 share = 1 share-equivalent of delta
        gamma=0.0,
        theta=0.0,
        vega=0.0,
        current_iv=None,
    )


def value_option_position(
    position: Position,
    spot: float,
    quote: data_mod.Quote,
    asof_date: date,
    risk_free_rate: float,
) -> PositionValuation:
    mark = quote.mark if quote.mark is not None else 0.0
    contract_multiplier = position.contracts * position.multiplier

    current_value = mark * contract_multiplier
    cost_basis = position.entry_price * contract_multiplier
    pnl = current_value - cost_basis
    pnl_pct = (pnl / cost_basis * 100.0) if cost_basis else None
    dte = (position.expiry - asof_date).days

    iv = quote.iv
    per_share_greeks = greeks_mod.black_scholes_greeks(
        spot=spot,
        strike=position.strike,
        years_to_expiry=max(dte, 0) / 365.0,
        risk_free_rate=risk_free_rate,
        iv=iv if iv else 0.0,
        option_type=position.option_type,
    )

    return PositionValuation(
        position=position,
        mark=mark,
        current_value=current_value,
        unrealized_pnl=pnl,
        unrealized_pnl_pct=pnl_pct,
        dte=dte,
        progress_to_target_pct=_progress_pct(mark, position.entry_price, position.target_price),
        progress_to_stop_pct=_progress_pct(mark, position.entry_price, position.stop_price),
        delta=per_share_greeks.delta * contract_multiplier,
        gamma=per_share_greeks.gamma * contract_multiplier,
        theta=per_share_greeks.theta * contract_multiplier,
        vega=per_share_greeks.vega * contract_multiplier,
        current_iv=iv,
    )


def to_db_row(valuation: PositionValuation, asof_date: date) -> tuple:
    p = valuation.position
    return (
        asof_date.isoformat(),
        p.id,
        p.ticker,
        p.asset_type,
        valuation.mark,
        valuation.current_value,
        valuation.unrealized_pnl,
        valuation.unrealized_pnl_pct,
        valuation.dte,
        valuation.progress_to_target_pct,
        valuation.progress_to_stop_pct,
        valuation.delta,
        valuation.gamma,
        valuation.theta,
        valuation.vega,
    )
