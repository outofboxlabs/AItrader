"""Section 12 of the spec: a single 1-minute bar can touch both the stop
and the target, and we cannot know which happened first. These tests pin
down each of the three configurable resolutions for that ambiguous case."""
from src.execution import resolve_intrabar_exit

AMBIGUOUS_BAR = dict(bar_open=100, bar_high=102, bar_low=97, stop_price=98, target_price=101)


def test_conservative_default_assumes_stop_first():
    fill = resolve_intrabar_exit(**AMBIGUOUS_BAR, priority="conservative")
    assert fill.exit_reason == "atr_stop"
    assert fill.raw_price == 98


def test_optimistic_assumes_target_first():
    fill = resolve_intrabar_exit(**AMBIGUOUS_BAR, priority="optimistic")
    assert fill.exit_reason == "take_profit"
    assert fill.raw_price == 101


def test_skip_discards_the_ambiguous_bar():
    fill = resolve_intrabar_exit(**AMBIGUOUS_BAR, priority="skip")
    assert fill.exit_reason == "skip"
    assert fill.raw_price is None


def test_unambiguous_bars_are_unaffected_by_priority_setting():
    only_target = dict(bar_open=100, bar_high=102, bar_low=99.5, stop_price=98, target_price=101)
    for priority in ("conservative", "optimistic", "skip"):
        fill = resolve_intrabar_exit(**only_target, priority=priority)
        assert fill.exit_reason == "take_profit"
