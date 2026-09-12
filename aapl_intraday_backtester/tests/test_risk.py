from src.config import RiskConfig
from src.risk import DailyRiskState


def _risk_cfg(**overrides) -> RiskConfig:
    base = dict(starting_equity=100_000, max_trades_per_day=10, max_daily_loss_pct=0.015, max_consecutive_losses=3)
    base.update(overrides)
    return RiskConfig(**base)


def test_trading_halts_once_max_daily_loss_reached():
    cfg = _risk_cfg(max_daily_loss_pct=0.01)
    state = DailyRiskState(starting_equity_for_day=100_000)

    state.register_trade_close(-500, cfg)
    assert state.can_open_new_trade(cfg)

    state.register_trade_close(-600, cfg)  # cumulative -1100 > 1% of 100k
    assert not state.can_open_new_trade(cfg)
    assert state.halt_reason == "max_daily_loss"


def test_trading_halts_after_max_consecutive_losses():
    cfg = _risk_cfg(max_consecutive_losses=2, max_daily_loss_pct=0.5)
    state = DailyRiskState(starting_equity_for_day=100_000)

    state.register_trade_close(-10, cfg)
    state.register_trade_close(-10, cfg)
    assert not state.can_open_new_trade(cfg)
    assert state.halt_reason == "max_consecutive_losses"


def test_a_win_resets_the_consecutive_loss_counter():
    cfg = _risk_cfg(max_consecutive_losses=2, max_daily_loss_pct=0.5)
    state = DailyRiskState(starting_equity_for_day=100_000)

    state.register_trade_close(-10, cfg)
    state.register_trade_close(50, cfg)
    assert state.consecutive_losses == 0
    assert state.can_open_new_trade(cfg)


def test_trading_halts_after_max_trades_per_day():
    cfg = _risk_cfg(max_trades_per_day=2, max_daily_loss_pct=0.5, max_consecutive_losses=100)
    state = DailyRiskState(starting_equity_for_day=100_000)

    state.register_trade_close(5, cfg)
    state.register_trade_close(5, cfg)
    assert not state.can_open_new_trade(cfg)
    assert state.halt_reason == "max_trades_per_day"


def test_mark_to_market_breach_detects_daily_loss_before_it_is_realized():
    cfg = _risk_cfg(max_daily_loss_pct=0.01)
    state = DailyRiskState(starting_equity_for_day=100_000)
    # No realized losses yet, but closing the open position now would breach -1% of equity.
    assert state.mark_to_market_breach(hypothetical_pnl_if_closed_now=-1200, risk_cfg=cfg)
    assert not state.mark_to_market_breach(hypothetical_pnl_if_closed_now=-500, risk_cfg=cfg)
