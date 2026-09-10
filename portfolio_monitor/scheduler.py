"""In-process daily scheduler for the market-movers scan.

This runs only while the local web app (app.py) is up -- it's a personal
desktop tool, not a 24/7 server, so "leave the app running" is the
tradeoff for not needing cloud hosting or OS-level task scheduling. A
manual "Run now" button (see app.py) covers the case where the app
wasn't open at the scheduled time.
"""

from __future__ import annotations

from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

_scheduler: Optional[BackgroundScheduler] = None

JOB_ID = "daily_movers_scan"


def start_scheduler(
    job_fn: Callable[[], None],
    hour: int = 15,
    minute: int = 30,
    timezone: str = "America/New_York",
) -> BackgroundScheduler:
    """Start (once) a background scheduler that calls `job_fn` with no
    arguments every weekday at hour:minute in `timezone`. Safe to call
    more than once -- later calls are a no-op and return the existing
    scheduler rather than registering a duplicate job."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        job_fn,
        trigger=CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute, timezone=timezone),
        id=JOB_ID,
        replace_existing=True,
    )
    _scheduler.start()
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


def get_next_run_time() -> Optional[str]:
    """ISO timestamp of the next scheduled firing, or None if not running."""
    if _scheduler is None:
        return None
    job = _scheduler.get_job(JOB_ID)
    if job is None or job.next_run_time is None:
        return None
    return job.next_run_time.isoformat()
