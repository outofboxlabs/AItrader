"""Macro-news and earnings date exclusion lists.

Both calendars are currently backed by flat CSV files
(data/excluded_dates.csv, data/earnings_dates.csv) with columns
`date,event,severity`. The `ExcludedDateProvider` protocol below is the
seam a real economic-calendar API/provider would implement later without
touching the backtester or signal code — only this module would change.

The bundled CSVs contain illustrative example dates only. They are NOT a
verified, authoritative economic/earnings calendar — before relying on the
macro or earnings filter for real analysis, replace them with dates
sourced from an official calendar (Fed/BLS/BEA release schedules, or
AAPL's actual reported earnings dates).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class ExcludedDateProvider(Protocol):
    """Interface a future live economic-calendar provider should implement."""

    def high_severity_dates(self) -> set[dt.date]:
        ...


@dataclass
class CsvExcludedDateProvider:
    path: str | Path

    def __post_init__(self) -> None:
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            logger.warning("Excluded-dates file not found at %s; no dates will be excluded", path)
            return pd.DataFrame(columns=["date", "event", "severity"])
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def high_severity_dates(self) -> set[dt.date]:
        if self._df.empty:
            return set()
        high = self._df[self._df["severity"].str.lower() == "high"]
        return set(high["date"])

    def all_dates(self) -> set[dt.date]:
        return set(self._df["date"]) if not self._df.empty else set()


class MacroCalendar:
    """Wraps a macro-news excluded-date provider (see ExcludedDateProvider)."""

    def __init__(self, provider: ExcludedDateProvider):
        self._provider = provider
        self._dates = provider.high_severity_dates()

    def is_excluded(self, session_date: dt.date) -> bool:
        return session_date in self._dates


class EarningsCalendar:
    """AAPL earnings-date exclusion, optionally extended to days around each report."""

    def __init__(self, provider: CsvExcludedDateProvider, days_before: int = 0, days_after: int = 0):
        self._earnings_dates = provider.all_dates()
        self._excluded = set()
        for d in self._earnings_dates:
            for offset in range(-days_before, days_after + 1):
                self._excluded.add(d + dt.timedelta(days=offset))

    def is_excluded(self, session_date: dt.date) -> bool:
        return session_date in self._excluded


def load_macro_calendar(path: str | Path) -> MacroCalendar:
    return MacroCalendar(CsvExcludedDateProvider(path))


def load_earnings_calendar(path: str | Path, days_before: int = 0, days_after: int = 0) -> EarningsCalendar:
    return EarningsCalendar(CsvExcludedDateProvider(path), days_before=days_before, days_after=days_after)
