from portfolio_monitor import scheduler


def _reset():
    scheduler.stop_scheduler()


def test_start_scheduler_registers_job():
    _reset()
    try:
        sched = scheduler.start_scheduler(lambda: None, hour=15, minute=30, timezone="America/New_York")
        jobs = sched.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == scheduler.JOB_ID
        assert scheduler.is_running() is True
    finally:
        _reset()


def test_start_scheduler_is_idempotent():
    _reset()
    try:
        calls = []
        sched1 = scheduler.start_scheduler(lambda: calls.append(1))
        sched2 = scheduler.start_scheduler(lambda: calls.append(2))
        assert sched1 is sched2
        assert len(sched1.get_jobs()) == 1
    finally:
        _reset()


def test_stop_scheduler_clears_state():
    _reset()
    scheduler.start_scheduler(lambda: None)
    assert scheduler.is_running() is True
    scheduler.stop_scheduler()
    assert scheduler.is_running() is False
