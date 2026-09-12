import pytest

from src.config import ExecutionConfig
from src.execution import apply_slippage, commission_cost, entry_fill_price, resolve_intrabar_exit


def test_apply_slippage_buy_is_worse_higher():
    price = 100.0
    filled = apply_slippage(price, "buy", slippage_bps=10)  # 10 bps = 0.10%
    assert filled == pytest.approx(100.10, rel=1e-6)


def test_apply_slippage_sell_is_worse_lower():
    price = 100.0
    filled = apply_slippage(price, "sell", slippage_bps=10)
    assert filled == pytest.approx(99.90, rel=1e-6)


def test_commission_cost_combines_per_share_and_per_order():
    cfg = ExecutionConfig(commission_per_share=0.005, commission_per_order=1.0)
    assert commission_cost(200, cfg) == pytest.approx(200 * 0.005 + 1.0)


def test_take_profit_only_hit_executes_at_target():
    fill = resolve_intrabar_exit(bar_open=100, bar_high=102, bar_low=99.5, stop_price=98, target_price=101, priority="conservative")
    assert fill.exit_reason == "take_profit"
    assert fill.raw_price == pytest.approx(101)


def test_atr_stop_only_hit_executes_at_stop():
    fill = resolve_intrabar_exit(bar_open=100, bar_high=100.5, bar_low=97, stop_price=98, target_price=101, priority="conservative")
    assert fill.exit_reason == "atr_stop"
    assert fill.raw_price == pytest.approx(98)


def test_gap_through_stop_fills_at_open_not_stale_stop_level():
    # Bar gaps down: open itself is already below the stop.
    fill = resolve_intrabar_exit(bar_open=95, bar_high=95.5, bar_low=94, stop_price=98, target_price=101, priority="conservative")
    assert fill.exit_reason == "atr_stop"
    assert fill.raw_price == pytest.approx(95)  # worse than the stop level


def test_neither_touched_returns_no_exit():
    fill = resolve_intrabar_exit(bar_open=100, bar_high=100.4, bar_low=99.6, stop_price=98, target_price=101, priority="conservative")
    assert fill.exit_reason is None


def test_entry_fill_price_next_bar_open():
    assert entry_fill_price(signal_bar_close=50, next_bar_open=51, entry_method="next_bar_open") == 51


def test_entry_fill_price_current_bar_close():
    assert entry_fill_price(signal_bar_close=50, next_bar_open=51, entry_method="current_bar_close") == 50


def test_entry_fill_price_next_bar_open_returns_none_if_no_next_bar():
    assert entry_fill_price(signal_bar_close=50, next_bar_open=None, entry_method="next_bar_open") is None
