import pytest

from portfolio_monitor.macro import (
    blend_macro_score,
    percentile_rank,
    score_breadth,
    score_credit_spread,
    score_term_structure,
    score_vix_level,
)


def test_percentile_rank_bounds():
    history = [10, 20, 30, 40, 50]
    assert percentile_rank(50, history) == 100.0  # highest value
    assert percentile_rank(10, history) == 20.0  # only itself is <= 10, 1/5
    assert percentile_rank(60, history) == 100.0  # above everything seen
    assert percentile_rank(5, []) == 50.0  # no history -> neutral


def test_score_vix_level_low_percentile_is_calm():
    # current VIX (last element) is the lowest in the window -> very calm -> high score
    history = [30.0, 25.0, 20.0, 15.0, 12.0]
    result = score_vix_level(history)
    assert result["vix"] == 12.0
    assert result["vix_percentile"] == 20.0  # only itself <= 12
    assert result["score"] == 80.0


def test_score_vix_level_high_percentile_is_stressed():
    history = [12.0, 15.0, 20.0, 25.0, 40.0]
    result = score_vix_level(history)
    assert result["score"] == 0.0  # current is the max -> percentile 100 -> score 0


def test_score_term_structure_contango_is_calm():
    # VIX well below VIX3M (contango) -> calm -> score near/at 100
    result = score_term_structure(vix=12.0, vix3m=16.0, calm_ratio=0.9, stress_ratio=1.1)
    assert result["ratio"] == pytest.approx(0.75)
    assert result["score"] == 100.0


def test_score_term_structure_backwardation_is_stressed():
    # VIX well above VIX3M (backwardation) -> stressed -> score near/at 0
    result = score_term_structure(vix=30.0, vix3m=20.0, calm_ratio=0.9, stress_ratio=1.1)
    assert result["ratio"] == pytest.approx(1.5)
    assert result["score"] == 0.0


def test_score_term_structure_midpoint():
    result = score_term_structure(vix=10.0, vix3m=10.0, calm_ratio=0.9, stress_ratio=1.1)
    assert result["ratio"] == 1.0
    assert result["score"] == pytest.approx(50.0)


def test_score_breadth():
    result = score_breadth(above_count=7, total_count=10)
    assert result["pct_above_200dma"] == 70.0
    assert result["score"] == 70.0
    assert score_breadth(0, 0)["score"] == 50.0  # no data -> neutral


def test_score_credit_spread_uses_percentile():
    history = [1.0, 1.1, 1.2, 1.3, 1.4]  # current (last) is the max
    result = score_credit_spread(history)
    assert result["credit_ratio"] == 1.4
    assert result["score"] == 100.0


def test_blend_macro_score_normalizes_weights():
    components = {"vix_level": 100.0, "term_structure": 0.0, "breadth": 50.0, "credit_spread": 50.0}
    # weights don't sum to 1.0 -- should be normalized before blending
    weights = {"vix_level": 2.0, "term_structure": 2.0, "breadth": 2.0, "credit_spread": 2.0}
    result = blend_macro_score(components, weights)
    assert result["weights"] == {"vix_level": 0.25, "term_structure": 0.25, "breadth": 0.25, "credit_spread": 0.25}
    assert result["score"] == pytest.approx((100.0 + 0.0 + 50.0 + 50.0) / 4)


def test_blend_macro_score_is_clamped_to_0_100():
    components = {"a": 200.0}
    weights = {"a": 1.0}
    assert blend_macro_score(components, weights)["score"] == 100.0
