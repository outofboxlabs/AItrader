"""Historical 1-minute bar loading.

Two backends are supported, selected via config.yaml `data.source`:

* "alpaca" — pulls bars from the Alpaca Market Data v2 REST API, handling
  pagination and caching each calendar month to a local parquet file so
  re-running a backtest never re-downloads data it already has.
* "local"  — reads a CSV/parquet dataset with columns
  [timestamp, open, high, low, close, volume]. This lets the whole project
  run without any API credentials (e.g. against a synthetic sample dataset
  produced by scripts/generate_sample_data.py, or a dataset exported from
  another vendor).

Either way, the returned DataFrame is:
  - sorted ascending by timestamp
  - tz-aware, converted to America/New_York (or config.session.timezone)
  - filtered to regular trading hours only (09:30-16:00 ET by default)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import BacktestConfig, get_alpaca_credentials
from src.utils import filter_regular_hours, get_logger

logger = get_logger(__name__)

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataLoadError(RuntimeError):
    pass


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end] into calendar-month chunks for per-month caching."""
    chunks = []
    cur = start.replace(day=1)
    while cur <= end:
        month_end = (cur + pd.offsets.MonthEnd(0))
        chunk_start = max(cur, start)
        chunk_end = min(month_end, end)
        chunks.append((chunk_start, chunk_end))
        cur = cur + pd.offsets.MonthBegin(1)
    return chunks


def _fetch_alpaca_month(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feed: str,
    api_key: str,
    secret_key: str,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch one month of 1-minute bars from Alpaca, following pagination tokens."""
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
    all_bars: list[dict] = []
    page_token: Optional[str] = None
    url = ALPACA_BARS_URL.format(symbol=symbol)

    while True:
        params = {
            "timeframe": "1Min",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "all",
            "feed": feed,
        }
        if page_token:
            params["page_token"] = page_token

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise DataLoadError(f"Alpaca request failed after {max_retries} attempts: {exc}") from exc
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Alpaca rate limited (429); backing off %ss", wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                if attempt == max_retries:
                    raise DataLoadError(f"Alpaca server error {resp.status_code}: {resp.text[:300]}")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                raise DataLoadError(f"Alpaca request failed [{resp.status_code}]: {resp.text[:300]}")
            break
        else:
            raise DataLoadError("Alpaca request failed: exhausted retries")

        payload = resp.json()
        bars = payload.get("bars") or []
        all_bars.extend(bars)
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not all_bars:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.DataFrame(all_bars)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[REQUIRED_COLUMNS]


def download_alpaca_bars(cfg: BacktestConfig, force_refresh: bool = False) -> pd.DataFrame:
    """Download (or read from cache) 1-minute bars for the configured date range.

    Data is cached per calendar month under `data.cache_dir` as
    `<symbol>_<YYYY-MM>.parquet`. Only months not already cached (or
    explicitly force-refreshed) are downloaded, so re-running a backtest is
    cheap and network failures don't force a full re-download.
    """
    api_key, secret_key = get_alpaca_credentials()
    if not api_key or not secret_key:
        raise DataLoadError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Export them or put them in a .env file, "
            "or switch config.yaml data.source to 'local'."
        )

    symbol = cfg.symbol
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(cfg.data.start_date, tz="UTC")
    end = pd.Timestamp(cfg.data.end_date, tz="UTC") + pd.Timedelta(days=1)

    monthly_frames = []
    for chunk_start, chunk_end in _month_ranges(start, end):
        cache_path = cache_dir / f"{symbol}_{chunk_start.strftime('%Y-%m')}.parquet"
        if cache_path.exists() and not force_refresh:
            logger.info("Cache hit for %s", cache_path.name)
            monthly_frames.append(pd.read_parquet(cache_path))
            continue

        logger.info("Downloading %s bars for %s..%s", symbol, chunk_start.date(), chunk_end.date())
        month_df = _fetch_alpaca_month(symbol, chunk_start, chunk_end, cfg.data.feed, api_key, secret_key)
        month_df.to_parquet(cache_path, index=False)
        logger.info("Cached %d bars -> %s", len(month_df), cache_path)
        monthly_frames.append(month_df)

    if not monthly_frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    full = pd.concat(monthly_frames, ignore_index=True)
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return full


def load_local_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or parquet dataset with [timestamp, open, high, low, close, volume]."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(
            f"Local dataset not found at {path}. Run scripts/generate_sample_data.py "
            "to create a synthetic sample dataset, or point data.local_path at a real one."
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataLoadError(f"Local dataset {path} is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[REQUIRED_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def load_bars(cfg: BacktestConfig, force_refresh: bool = False) -> pd.DataFrame:
    """Load 1-minute bars per config, convert to session tz, and filter to RTH.

    This is the single entry point the rest of the codebase should use to
    get bar data: it hides whether the data came from Alpaca or a local file.
    """
    if cfg.data.source == "alpaca":
        raw = download_alpaca_bars(cfg, force_refresh=force_refresh)
    elif cfg.data.source == "local":
        raw = load_local_dataset(cfg.data.local_path)
    else:
        raise DataLoadError(f"Unknown data.source: {cfg.data.source!r}")

    if raw.empty:
        logger.warning("No bars loaded for %s", cfg.symbol)
        return raw

    raw = raw.copy()
    raw["timestamp"] = raw["timestamp"].dt.tz_convert(cfg.session.timezone)

    start_date = pd.Timestamp(cfg.data.start_date).date()
    end_date = pd.Timestamp(cfg.data.end_date).date()
    raw = raw[(raw["timestamp"].dt.date >= start_date) & (raw["timestamp"].dt.date <= end_date)]

    raw = filter_regular_hours(raw, cfg.session, ts_col="timestamp")
    raw["date"] = raw["timestamp"].dt.date
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close", "volume"])
    raw = raw.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d RTH bars for %s spanning %d sessions", len(raw), cfg.symbol, raw["date"].nunique())
    return raw
