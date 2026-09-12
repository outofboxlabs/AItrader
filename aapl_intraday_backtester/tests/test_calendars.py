import datetime as dt
from pathlib import Path

from src.calendars import load_earnings_calendar, load_macro_calendar


def test_macro_calendar_only_excludes_high_severity(tmp_path: Path):
    csv = tmp_path / "excluded.csv"
    csv.write_text("date,event,severity\n2025-06-18,FOMC,high\n2025-06-11,CPI,medium\n")
    cal = load_macro_calendar(csv)
    assert cal.is_excluded(dt.date(2025, 6, 18))
    assert not cal.is_excluded(dt.date(2025, 6, 11))  # not "high" severity
    assert not cal.is_excluded(dt.date(2025, 6, 19))


def test_earnings_calendar_optionally_excludes_surrounding_days(tmp_path: Path):
    csv = tmp_path / "earnings.csv"
    csv.write_text("date,event,severity\n2025-05-01,AAPL Earnings,high\n")

    cal_exact = load_earnings_calendar(csv, days_before=0, days_after=0)
    assert cal_exact.is_excluded(dt.date(2025, 5, 1))
    assert not cal_exact.is_excluded(dt.date(2025, 4, 30))
    assert not cal_exact.is_excluded(dt.date(2025, 5, 2))

    cal_window = load_earnings_calendar(csv, days_before=1, days_after=1)
    assert cal_window.is_excluded(dt.date(2025, 4, 30))
    assert cal_window.is_excluded(dt.date(2025, 5, 1))
    assert cal_window.is_excluded(dt.date(2025, 5, 2))
    assert not cal_window.is_excluded(dt.date(2025, 4, 29))
