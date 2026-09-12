import datetime as dt
from pathlib import Path

import pandas as pd

from src.backtester import run_backtest
from src.config import BacktestConfig, FiltersConfig, RiskConfig, StrategyConfig
from src.risk import compute_position_size
from tests.helpers import make_bars

WARMUP = (150.0, 150.5, 149.5, 150.0, 1000)  # wide flat bar -> ATR converges to ~1.0


def _build_day(date_str: str, post_entry_bars: list[tuple] | None = None, n_total: int = 390) -> pd.DataFrame:
    """One RTH session (390 one-minute bars) with a clean pullback/reversal
    signal at 14:56, filled at 14:57, followed by caller-supplied bars.

    Bar 325 (14:55) is bearish, bar 326 (14:56) is the bullish signal bar,
    bar 327 (14:57) is the entry fill bar. Anything after that is
    `post_entry_bars`, right-padded with a flat bar out to `n_total`.
    """
    bars = [WARMUP] * n_total
    bars[325] = (150.0, 150.2, 149.5, 149.6, 1000)   # bearish
    bars[326] = (149.6, 150.9, 149.6, 150.9, 1000)   # bullish signal (close > prev high, pullback >= 0.5%)
    bars[327] = (150.9, 151.0, 150.8, 150.9, 1000)   # entry fills here (this bar's open)

    if post_entry_bars:
        for offset, bar in enumerate(post_entry_bars):
            idx = 328 + offset
            if idx < n_total:
                bars[idx] = bar
    else:
        for i in range(328, n_total):
            bars[i] = (150.9, 151.0, 150.8, 150.9, 1000)

    return make_bars(date_str, bars)


def _base_cfg(**strategy_overrides) -> BacktestConfig:
    cfg = BacktestConfig()
    defaults = dict(pullback_pct=0.005, take_profit_pct=0.01, atr_period=14,
                     atr_multiplier=1.5, max_hold_minutes=120)
    defaults.update(strategy_overrides)
    cfg.strategy = StrategyConfig(**defaults)
    cfg.filters = FiltersConfig(macro_news=False, earnings=False)
    return cfg


def test_forced_close_at_1555_with_no_tp_sl_hit():
    cfg = _base_cfg()
    bars = _build_day("2025-04-01")
    result = run_backtest(cfg, bars)

    trades = result.valid_trades
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "end_of_day"
    assert trade["entry_timestamp"].time() == dt.time(14, 57)
    assert trade["exit_timestamp"].time() == dt.time(15, 55)


def test_no_overnight_positions_across_multiple_days():
    cfg = _base_cfg()
    bars = pd.concat([_build_day("2025-04-01"), _build_day("2025-04-02")], ignore_index=True)
    result = run_backtest(cfg, bars)

    trades = result.valid_trades
    assert len(trades) == 2
    for _, trade in trades.iterrows():
        assert trade["entry_timestamp"].date() == trade["exit_timestamp"].date()

    # Never marked "in position" at the very first bar of a new session.
    eq = result.equity_curve
    eq["date"] = pd.to_datetime(eq["timestamp"]).dt.date
    first_bar_each_day = eq.groupby("date").first()
    assert not first_bar_each_day["in_position"].any()


def test_take_profit_exit_end_to_end():
    # Entry price ~150.9 * 1.01bps slippage; target = entry * 1.01. A bar whose
    # high pushes well above that target should exit via take_profit.
    cfg = _base_cfg(take_profit_pct=0.01)
    post = [(150.9, 153.5, 150.8, 153.0, 1000)]  # high=153.5 >> target (~152.4)
    bars = _build_day("2025-04-01", post_entry_bars=post)
    result = run_backtest(cfg, bars)

    trades = result.valid_trades
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "take_profit"


def test_atr_stop_exit_end_to_end():
    cfg = _base_cfg(atr_multiplier=1.5)
    post = [(150.9, 151.0, 145.0, 145.5, 1000)]  # low=145 well below the ATR stop
    bars = _build_day("2025-04-01", post_entry_bars=post)
    result = run_backtest(cfg, bars)

    trades = result.valid_trades
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "atr_stop"


def test_time_stop_exit_end_to_end():
    cfg = _base_cfg(max_hold_minutes=3)  # entry at 14:57 -> deadline 15:00
    bars = _build_day("2025-04-01")  # flat afterwards, never hits TP/SL
    result = run_backtest(cfg, bars)

    trades = result.valid_trades
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "time_stop"
    assert trades.iloc[0]["exit_timestamp"].time() == dt.time(15, 0)


def test_macro_excluded_date_blocks_all_trading(tmp_path: Path):
    excluded_csv = tmp_path / "excluded_dates.csv"
    excluded_csv.write_text("date,event,severity\n2025-04-01,FOMC Rate Decision,high\n")

    cfg = _base_cfg()
    cfg.filters.macro_news = True
    cfg.calendars.excluded_dates_path = str(excluded_csv)
    bars = _build_day("2025-04-01")

    result = run_backtest(cfg, bars)
    assert len(result.valid_trades) == 0


def test_earnings_excluded_date_blocks_all_trading(tmp_path: Path):
    earnings_csv = tmp_path / "earnings_dates.csv"
    earnings_csv.write_text("date,event,severity\n2025-04-01,AAPL Earnings,high\n")

    cfg = _base_cfg()
    cfg.filters.earnings = True
    cfg.calendars.earnings_dates_path = str(earnings_csv)
    bars = _build_day("2025-04-01")

    result = run_backtest(cfg, bars)
    assert len(result.valid_trades) == 0


def test_position_sizing_never_exceeds_max_position_pct():
    risk_cfg = RiskConfig(starting_equity=100_000, sizing_method="fixed_dollar",
                           fixed_dollar_amount=500_000, max_position_pct=0.20)
    shares = compute_position_size(risk_cfg, equity=100_000, entry_price=150.0, stop_price=148.0)
    notional = shares * 150.0
    assert notional <= 100_000 * 0.20 + 150.0  # +1 share tolerance from flooring


def test_risk_based_sizing_uses_risk_per_trade_and_never_leverages():
    risk_cfg = RiskConfig(starting_equity=100_000, sizing_method="risk",
                           risk_per_trade_pct=0.0025, max_position_pct=1.0)
    shares = compute_position_size(risk_cfg, equity=100_000, entry_price=150.0, stop_price=148.5)
    # risk_dollars = 100_000 * 0.0025 = 250; per-share risk = 1.5 -> ~166 shares
    assert 160 <= shares <= 167
    assert shares * 150.0 <= 100_000  # no leverage
