"""Small shared helpers: logging setup and session/timezone utilities."""
from __future__ import annotations

import datetime as dt
import logging
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import SessionConfig

_LOG_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent, safe to call from every entry point)."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _LOG_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def session_times(session: SessionConfig, session_date: dt.date) -> dict[str, pd.Timestamp]:
    """Return tz-aware timestamps for the key session boundaries on a given date."""
    tz = ZoneInfo(session.timezone)

    def _at(hhmm: str) -> pd.Timestamp:
        h, m = (int(x) for x in hhmm.split(":"))
        return pd.Timestamp(dt.datetime.combine(session_date, dt.time(h, m)), tz=tz)

    return {
        "open": _at(session.open),
        "close": _at(session.close),
        "stop_new_entries": _at(session.stop_new_entries),
        "force_close": _at(session.force_close),
    }


def to_session_tz(series: pd.Series, timezone: str) -> pd.Series:
    """Convert a tz-aware (or UTC-naive-assumed) timestamp series to the session timezone."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    return series.dt.tz_convert(ZoneInfo(timezone))


def filter_regular_hours(df: pd.DataFrame, session: SessionConfig, ts_col: str = "timestamp") -> pd.DataFrame:
    """Keep only bars within [session.open, session.close) in the session's local timezone."""
    local_ts = df[ts_col]
    open_t = dt.datetime.strptime(session.open, "%H:%M").time()
    close_t = dt.datetime.strptime(session.close, "%H:%M").time()
    t = local_ts.dt.time
    mask = (t >= open_t) & (t < close_t)
    return df.loc[mask].reset_index(drop=True)
