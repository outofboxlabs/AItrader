"""Sequential, bar-by-bar backtest engine.

Design choices that matter for correctness:

* **No lookahead.** Indicators (src/indicators.py) and signals
  (src/signals.py) are causal by construction: row N only ever uses rows
  <= N. A signal confirmed at row N is queued and filled at row N+1's open
  (config.execution.entry_method, default "next_bar_open") — never at row
  N's own close. This matters because a strategy that "sees" a bar's full
  OHLC and then transacts at that same bar's close implicitly assumes
  zero-latency, perfect-information execution that cannot exist in live
  trading; entering on the following bar's open is the standard way to
  avoid quietly overstating edge in a backtest.
* **Intrabar ambiguity** (a single 1-minute bar touching both the stop and
  the target) is resolved explicitly via execution.resolve_intrabar_exit,
  defaulting to the conservative assumption (stop first).
* **Strictly one position at a time**, always flat overnight: every open
  position is force-closed at (or before) config.session.force_close, and
  each trading day starts flat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.calendars import load_earnings_calendar, load_macro_calendar
from src.config import BacktestConfig
from src.execution import apply_slippage, commission_cost, entry_fill_price, resolve_intrabar_exit
from src.indicators import add_all_indicators
from src.risk import DailyRiskState, compute_position_size
from src.signals import generate_entry_signals
from src.strategy import Position, apply_profit_protection
from src.utils import get_logger, session_times

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    final_equity: float
    starting_equity: float
    config: BacktestConfig

    @property
    def valid_trades(self) -> pd.DataFrame:
        """Trades excluding those discarded by the 'skip' intrabar-ambiguity rule."""
        if self.trades.empty:
            return self.trades
        return self.trades[self.trades["exit_reason"] != "skip"]


@dataclass
class _PendingEntry:
    signal_timestamp: pd.Timestamp
    atr_at_signal: float
    pullback_percent: float


def _cost_basis_pullback(row) -> float:
    """Recover the realized pullback percent for the signal bar (for logging).
    `row` is a namedtuple from DataFrame.itertuples()."""
    ref_high = row.intraday_high_ref
    prev_low = row.prev_low
    if ref_high != ref_high or prev_low != prev_low:  # NaN check
        return float("nan")
    return (ref_high - prev_low) / ref_high


def run_backtest(cfg: BacktestConfig, bars: pd.DataFrame) -> BacktestResult:
    """Run the full backtest over `bars` (RTH-only, tz-aware, one symbol)."""
    if bars.empty:
        empty_trades = pd.DataFrame(columns=_TRADE_COLUMNS)
        return BacktestResult(empty_trades, pd.DataFrame(columns=["timestamp", "equity"]),
                               cfg.risk.starting_equity, cfg.risk.starting_equity, cfg)

    df = add_all_indicators(bars, atr_period=cfg.strategy.atr_period)
    df["prev_low"] = df["low"].shift(1)
    df["entry_signal"] = generate_entry_signals(df, cfg.strategy, cfg.filters)

    macro_cal = load_macro_calendar(cfg.calendars.excluded_dates_path) if cfg.filters.macro_news else None
    earnings_cal = (
        load_earnings_calendar(
            cfg.calendars.earnings_dates_path,
            cfg.filters.earnings_exclude_days_before,
            cfg.filters.earnings_exclude_days_after,
        )
        if cfg.filters.earnings
        else None
    )

    equity = cfg.risk.starting_equity
    trade_records: list[dict] = []
    equity_rows: list[dict] = []
    trade_id = 0

    for session_date, day_df in df.groupby("date", sort=True):
        day_df = day_df.reset_index(drop=True)
        times = session_times(cfg.session, session_date)

        macro_active = bool(macro_cal.is_excluded(session_date)) if macro_cal else False
        earnings_active = bool(earnings_cal.is_excluded(session_date)) if earnings_cal else False
        date_blocked = macro_active or earnings_active

        daily_state = DailyRiskState(starting_equity_for_day=equity)
        position: Optional[Position] = None
        pending: Optional[_PendingEntry] = None
        records = list(day_df.itertuples(index=False))
        n = len(records)

        for i in range(n):
            row = records[i]
            ts = row.timestamp
            o, h, l, c = row.open, row.high, row.low, row.close

            # --- A) fill a pending entry queued from the previous bar -------------
            if pending is not None and position is None:
                raw_entry = entry_fill_price(
                    signal_bar_close=None, next_bar_open=o, entry_method="next_bar_open"
                )
                candidate = _open_position(cfg, ts, raw_entry, pending, equity)
                if candidate is not None:
                    position = candidate
                pending = None

            # --- B) manage an open position ---------------------------------------
            if position is not None:
                position.update_excursion(h, l)
                closed = _manage_open_position(cfg, position, ts, o, h, l, c, times, daily_state)
                if closed is not None:
                    trade_id += 1
                    record = _build_trade_record(trade_id, position, closed, cfg, macro_active, earnings_active)
                    trade_records.append(record)
                    if record["exit_reason"] != "skip":
                        equity += record["net_pnl"]
                        daily_state.register_trade_close(record["net_pnl"], cfg.risk)
                    position = None

            # --- C) look for a new signal (only if flat, allowed, and not last bar) -
            can_open = (
                position is None
                and pending is None
                and not date_blocked
                and i < n - 1
                and ts < times["stop_new_entries"]
                and daily_state.can_open_new_trade(cfg.risk)
            )
            if can_open and bool(row.entry_signal):
                signal_pending = _PendingEntry(
                    signal_timestamp=ts,
                    atr_at_signal=row.atr,
                    pullback_percent=_cost_basis_pullback(row),
                )
                if cfg.execution.entry_method == "current_bar_close":
                    raw_entry = entry_fill_price(signal_bar_close=c, next_bar_open=None, entry_method="current_bar_close")
                    candidate = _open_position(cfg, ts, raw_entry, signal_pending, equity)
                    if candidate is not None:
                        position = candidate
                else:
                    pending = signal_pending

            # --- D) mark-to-market equity for the equity curve ---------------------
            unrealized = 0.0
            if position is not None:
                unrealized = (c - position.entry_price) * position.shares
            equity_rows.append({"timestamp": ts, "equity": equity + unrealized, "in_position": position is not None})

        # Safety net: force-flat at end of day even if force_close bar was missing from data.
        if position is not None:
            last_row = records[-1]
            raw_exit = last_row.close
            trade_id += 1
            closed = {"reason": "end_of_day", "raw_price": raw_exit, "timestamp": last_row.timestamp}
            record = _build_trade_record(trade_id, position, closed, cfg, macro_active, earnings_active)
            trade_records.append(record)
            equity += record["net_pnl"]
            equity_rows[-1]["equity"] = equity

    trades_df = pd.DataFrame(trade_records, columns=_TRADE_COLUMNS)
    equity_df = pd.DataFrame(equity_rows)
    return BacktestResult(trades_df, equity_df, equity, cfg.risk.starting_equity, cfg)


def _open_position(cfg: BacktestConfig, ts: pd.Timestamp, raw_entry: float, pending: _PendingEntry, equity: float) -> Optional[Position]:
    if raw_entry is None or pending.atr_at_signal != pending.atr_at_signal:  # None or NaN ATR
        return None
    entry_price = apply_slippage(raw_entry, "buy", cfg.execution.slippage_bps_per_side)
    atr_value = pending.atr_at_signal
    stop_price = entry_price - cfg.strategy.atr_multiplier * atr_value
    target_price = entry_price * (1.0 + cfg.strategy.take_profit_pct)
    shares = compute_position_size(cfg.risk, equity, entry_price, stop_price)
    if shares <= 0:
        return None
    deadline = ts + pd.Timedelta(minutes=cfg.strategy.max_hold_minutes)

    return Position(
        entry_timestamp=ts,
        entry_price=entry_price,
        raw_entry_price=raw_entry,
        shares=shares,
        atr_at_entry=atr_value,
        initial_stop=stop_price,
        target_price=target_price,
        pullback_percent=pending.pullback_percent,
        max_hold_minutes=cfg.strategy.max_hold_minutes,
        deadline=deadline,
    )


def _manage_open_position(cfg, position: Position, ts, o, h, l, c, times, daily_state: DailyRiskState) -> Optional[dict]:
    """Evaluate all exit rules for the current bar, in priority order. Returns a
    dict describing the exit if the position should close this bar, else None."""
    if ts >= times["force_close"]:
        return {"reason": "end_of_day", "raw_price": o, "timestamp": ts}

    apply_profit_protection(position, o, cfg.profit_protection)

    fill = resolve_intrabar_exit(o, h, l, position.current_stop, position.target_price, cfg.execution.intrabar_priority)
    if fill.exit_reason in ("atr_stop", "take_profit"):
        return {"reason": fill.exit_reason, "raw_price": fill.raw_price, "timestamp": ts}
    if fill.exit_reason == "skip":
        return {"reason": "skip", "raw_price": o, "timestamp": ts}

    if ts >= position.deadline:
        return {"reason": "time_stop", "raw_price": o, "timestamp": ts}

    hypothetical_pnl = (c - position.entry_price) * position.shares
    if daily_state.mark_to_market_breach(hypothetical_pnl, cfg.risk):
        return {"reason": "daily_risk_stop", "raw_price": o, "timestamp": ts}

    return None


_TRADE_COLUMNS = [
    "trade_id", "date", "entry_timestamp", "exit_timestamp", "entry_price", "exit_price",
    "shares", "atr_at_entry", "stop_at_entry", "profit_target", "pullback_percent",
    "minutes_held", "gross_return_pct", "net_return_pct", "gross_pnl", "net_pnl",
    "maximum_favorable_excursion", "maximum_adverse_excursion", "exit_reason",
    "macro_filter_active", "earnings_filter_active", "strategy_parameters",
]


def _build_trade_record(trade_id: int, position: Position, closed: dict, cfg: BacktestConfig,
                         macro_active: bool, earnings_active: bool) -> dict:
    raw_exit = closed["raw_price"]
    exit_ts = closed["timestamp"]
    exit_price = apply_slippage(raw_exit, "sell", cfg.execution.slippage_bps_per_side)

    entry_commission = commission_cost(position.shares, cfg.execution)
    exit_commission = commission_cost(position.shares, cfg.execution)

    raw_entry = position.raw_entry_price
    gross_pnl = (raw_exit - raw_entry) * position.shares
    gross_return_pct = (raw_exit - raw_entry) / raw_entry if raw_entry else 0.0

    net_pnl = (exit_price - position.entry_price) * position.shares - entry_commission - exit_commission
    entry_notional = position.entry_price * position.shares
    net_return_pct = net_pnl / entry_notional if entry_notional else 0.0

    minutes_held = (exit_ts - position.entry_timestamp).total_seconds() / 60.0

    return {
        "trade_id": trade_id,
        "date": position.entry_timestamp.date(),
        "entry_timestamp": position.entry_timestamp,
        "exit_timestamp": exit_ts,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "shares": position.shares,
        "atr_at_entry": position.atr_at_entry,
        "stop_at_entry": position.initial_stop,
        "profit_target": position.target_price,
        "pullback_percent": position.pullback_percent,
        "minutes_held": minutes_held,
        "gross_return_pct": gross_return_pct,
        "net_return_pct": net_return_pct,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "maximum_favorable_excursion": position.mfe,
        "maximum_adverse_excursion": position.mae,
        "exit_reason": closed["reason"],
        "macro_filter_active": macro_active,
        "earnings_filter_active": earnings_active,
        "strategy_parameters": cfg.strategy_parameters_dict(),
    }
