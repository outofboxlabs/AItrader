import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

import src.data_loader as data_loader
from src.config import BacktestConfig, DataConfig
from src.data_loader import DataLoadError, REQUIRED_COLUMNS, download_yahoo_bars


def _cfg(tmp_path: Path, start: str, end: str) -> BacktestConfig:
    cfg = BacktestConfig()
    cfg.symbol = "AAPL"
    cfg.data = DataConfig(source="yahoo", cache_dir=str(tmp_path), start_date=start, end_date=end)
    return cfg


def test_yahoo_download_clips_range_to_trailing_30_days(tmp_path, monkeypatch):
    today = dt.date.today()
    seen_ranges = []

    def fake_chunk(symbol, start, end, max_retries=3):
        seen_ranges.append((start, end))
        idx = pd.date_range(f"{start} 09:30", periods=2, freq="min", tz="America/New_York")
        return pd.DataFrame({"timestamp": idx.tz_convert("UTC"), "open": [1, 1], "high": [1, 1],
                              "low": [1, 1], "close": [1, 1], "volume": [100, 100]})

    monkeypatch.setattr(data_loader, "_fetch_yahoo_chunk", fake_chunk)

    cfg = _cfg(tmp_path, start="2000-01-01", end="2099-12-31")  # absurdly wide on purpose
    df = download_yahoo_bars(cfg)

    earliest_allowed = today - dt.timedelta(days=data_loader.YAHOO_1M_LOOKBACK_DAYS)
    assert min(r[0] for r in seen_ranges) == earliest_allowed
    assert max(r[1] for r in seen_ranges) == today
    assert not df.empty
    assert list(df.columns) == REQUIRED_COLUMNS


def test_yahoo_download_never_caches_an_empty_chunk(tmp_path, monkeypatch):
    """Regression test: yfinance swallows connection failures and returns an
    empty frame instead of raising. If we cached that, a transient outage
    would permanently poison the cache with a false 'no data' answer."""

    def always_empty(symbol, start, end, max_retries=3):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    monkeypatch.setattr(data_loader, "_fetch_yahoo_chunk", always_empty)

    cfg = _cfg(tmp_path, start=str(dt.date.today() - dt.timedelta(days=10)), end=str(dt.date.today()))
    with pytest.raises(DataLoadError):
        download_yahoo_bars(cfg)

    assert list(tmp_path.glob("*.parquet")) == []


def test_yahoo_download_uses_cache_on_second_call(tmp_path, monkeypatch):
    call_count = {"n": 0}

    def fake_chunk(symbol, start, end, max_retries=3):
        call_count["n"] += 1
        idx = pd.date_range(f"{start} 09:30", periods=1, freq="min", tz="America/New_York")
        return pd.DataFrame({"timestamp": idx.tz_convert("UTC"), "open": [1], "high": [1],
                              "low": [1], "close": [1], "volume": [100]})

    monkeypatch.setattr(data_loader, "_fetch_yahoo_chunk", fake_chunk)

    cfg = _cfg(tmp_path, start=str(dt.date.today() - dt.timedelta(days=5)), end=str(dt.date.today()))
    download_yahoo_bars(cfg)
    first_call_count = call_count["n"]
    assert first_call_count > 0

    download_yahoo_bars(cfg)  # second call should hit the cache, not call _fetch_yahoo_chunk again
    assert call_count["n"] == first_call_count
