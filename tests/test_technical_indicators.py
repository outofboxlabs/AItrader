import pytest

from portfolio_monitor import technical_indicators as ti


def _bars(closes):
    return [{"close": c} for c in closes]


def test_compute_technical_snapshot_empty_input():
    assert ti.compute_technical_snapshot([]) == {}


def test_compute_technical_snapshot_omits_indicators_without_enough_history():
    # Only 5 daily bars -- nowhere near enough for SMA20, RSI14, or MACD.
    snapshot = ti.compute_technical_snapshot(_bars([10.0, 11.0, 10.5, 11.5, 12.0]))
    assert snapshot["latest_close"] == 12.0
    assert "sma_20" not in snapshot
    assert "rsi_14" not in snapshot
    assert "macd" not in snapshot


def test_compute_technical_snapshot_sma_and_price_vs_sma():
    closes = list(range(1, 41))  # 1..40, strictly increasing
    snapshot = ti.compute_technical_snapshot(_bars(closes))
    assert snapshot["latest_close"] == 40.0
    # SMA(20) = mean of the last 20 closes = mean(21..40) = 30.5
    assert snapshot["sma_20"] == pytest.approx(30.5)
    assert snapshot["price_vs_sma_20_pct"] == pytest.approx((40.0 - 30.5) / 30.5 * 100.0)


def test_compute_technical_snapshot_rsi_all_gains_is_100():
    # Strictly increasing closes -- every change is a gain, so Wilder's
    # RSI formula (avg_loss == 0) should hit exactly 100.
    closes = [float(i) for i in range(1, 30)]
    snapshot = ti.compute_technical_snapshot(_bars(closes))
    assert snapshot["rsi_14"] == pytest.approx(100.0)


def test_compute_technical_snapshot_rsi_all_losses_is_near_zero():
    closes = [float(i) for i in range(30, 0, -1)]
    snapshot = ti.compute_technical_snapshot(_bars(closes))
    assert snapshot["rsi_14"] == pytest.approx(0.0)


def test_compute_technical_snapshot_macd_present_with_enough_history():
    closes = [float(i) for i in range(1, 60)]
    snapshot = ti.compute_technical_snapshot(_bars(closes))
    assert "macd" in snapshot
    assert "macd_signal" in snapshot
    assert snapshot["macd_histogram"] == pytest.approx(snapshot["macd"] - snapshot["macd_signal"])


def test_compute_technical_snapshot_ignores_bars_missing_close():
    closes = [{"close": 10.0}, {"close": None}, {"close": 12.0}]
    snapshot = ti.compute_technical_snapshot(closes)
    assert snapshot["latest_close"] == 12.0
