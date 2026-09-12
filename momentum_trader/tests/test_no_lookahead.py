"""Two lookahead-safety guarantees the whole project depends on:

1. Entry signals at bar N depend only on bar N and earlier (src/signals.py).
2. Parameter optimization never lets the held-out test window influence which
   configuration gets selected (src/optimizer.py train/validation/test split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BacktestConfig, FiltersConfig, OptimizerConfig, StrategyConfig
from src.indicators import add_all_indicators
from src.optimizer import optimize_train_val_test
from src.signals import generate_entry_signals
from tests.helpers import make_bars


def _random_day(date_str: str, n: int, seed: int, base_price: float = 150.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = base_price
    ohlcv = []
    for _ in range(n):
        o = price
        ret = rng.normal(0, 0.001)
        c = o * (1 + ret)
        h = max(o, c) + abs(rng.normal(0, 0.05))
        l = min(o, c) - abs(rng.normal(0, 0.05))
        v = int(abs(rng.normal(2000, 500)))
        ohlcv.append((o, h, l, c, v))
        price = c
    return make_bars(date_str, ohlcv)


def test_entry_signal_at_bar_n_unaffected_by_bars_after_n():
    days = {}
    for i, day in enumerate(pd.bdate_range("2025-02-03", periods=6)):
        days[str(day.date())] = None
    frames = [_random_day(d, 60, seed=100 + i) for i, d in enumerate(days)]
    full_df = pd.concat(frames, ignore_index=True)
    full_df = add_all_indicators(full_df, atr_period=5)
    full_df["entry_signal"] = generate_entry_signals(full_df, StrategyConfig(pullback_pct=0.001), FiltersConfig(macro_news=False, earnings=False))

    # Truncate the dataset partway through and recompute from scratch.
    cutoff = len(full_df) - 15
    prefix_df = full_df.iloc[:cutoff][["timestamp", "open", "high", "low", "close", "volume", "date"]].copy()
    prefix_df = add_all_indicators(prefix_df, atr_period=5)
    prefix_df["entry_signal"] = generate_entry_signals(prefix_df, StrategyConfig(pullback_pct=0.001), FiltersConfig(macro_news=False, earnings=False))

    pd.testing.assert_series_equal(
        full_df["entry_signal"].iloc[:cutoff].reset_index(drop=True),
        prefix_df["entry_signal"].reset_index(drop=True),
        check_names=False,
    )


def _build_split_bars(test_variant: int) -> pd.DataFrame:
    frames = []
    for i, day in enumerate(pd.bdate_range("2025-01-01", periods=6)):  # train
        frames.append(_random_day(str(day.date()), 40, seed=1000 + i))
    for i, day in enumerate(pd.bdate_range("2025-01-09", periods=3)):  # validation
        frames.append(_random_day(str(day.date()), 40, seed=2000 + i))
    for i, day in enumerate(pd.bdate_range("2025-01-14", periods=3)):  # test (out-of-sample)
        # Seed depends on `test_variant` so the two calls produce DIFFERENT test-period data.
        frames.append(_random_day(str(day.date()), 40, seed=9000 + test_variant * 100 + i))
    return pd.concat(frames, ignore_index=True)


def _make_cfg() -> BacktestConfig:
    cfg = BacktestConfig()
    cfg.strategy = StrategyConfig(pullback_pct=0.001, take_profit_pct=0.005, atr_period=5, atr_multiplier=1.5, max_hold_minutes=30)
    cfg.filters = FiltersConfig(macro_news=False, earnings=False)
    cfg.optimizer = OptimizerConfig(
        train_start="2025-01-01", train_end="2025-01-08",
        validation_start="2025-01-09", validation_end="2025-01-13",
        test_start="2025-01-14", test_end="2025-01-20",
        min_trades_for_ranking=0,
        grid={"take_profit_pct": [0.004, 0.008], "atr_multiplier": [1.0, 2.0]},
    )
    return cfg


def test_optimizer_never_lets_test_window_leak_into_selection():
    cfg = _make_cfg()

    bars_a = _build_split_bars(test_variant=1)
    bars_b = _build_split_bars(test_variant=2)  # only the test-period rows differ from bars_a

    result_a = optimize_train_val_test(cfg, bars_a)
    result_b = optimize_train_val_test(cfg, bars_b)

    # Train/validation grid results must be identical: they never saw the
    # (different) test-period data in either run.
    pd.testing.assert_frame_equal(
        result_a.train_results.reset_index(drop=True), result_b.train_results.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        result_a.validation_results.reset_index(drop=True), result_b.validation_results.reset_index(drop=True)
    )
    assert result_a.best_params == result_b.best_params
