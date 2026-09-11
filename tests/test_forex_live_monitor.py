import csv
from datetime import date, datetime, timedelta, timezone

from portfolio_monitor import forex_live_monitor as flm


def test_day_query_value_format():
    assert flm._day_query_value(date(2026, 9, 11)) == "sep11.2026"
    assert flm._day_query_value(date(2026, 9, 6)) == "sep6.2026"  # no leading zero


def test_normalize_direction_known_classes():
    assert flm._normalize_direction("better") == "better"
    assert flm._normalize_direction("worse") == "worse"


def test_normalize_direction_unknown_class_returns_none():
    assert flm._normalize_direction("some-future-class") is None


_ROW_TEMPLATE = """
<table class="calendar__table">
  <tr class="calendar__row calendar_row">
    <td class="calendar__cell calendar__currency currency">{country}</td>
    <td class="calendar__cell calendar__event event">
      <span class="calendar__event-title">{title}</span>
    </td>
    <td class="calendar__cell calendar__actual actual">{actual_html}</td>
  </tr>
</table>
"""


def test_find_actual_for_event_matches_by_title_and_country():
    html = _ROW_TEMPLATE.format(
        country="USD", title="Core CPI m/m", actual_html='<span class="better">0.3%</span>'
    )
    result = flm.find_actual_for_event(html, "Core CPI m/m", "USD")
    assert result == {"actual": "0.3%", "direction": "better", "raw_class": "better"}


def test_find_actual_for_event_worse_direction():
    html = _ROW_TEMPLATE.format(
        country="EUR", title="GDP m/m", actual_html='<span class="worse">-0.1%</span>'
    )
    result = flm.find_actual_for_event(html, "GDP m/m", "EUR")
    assert result["direction"] == "worse"


def test_find_actual_for_event_empty_cell_returns_none():
    """The event hasn't released yet -- an empty (or comment-only) actual
    cell is a legitimate "no data yet" outcome, not an error."""
    html = _ROW_TEMPLATE.format(country="USD", title="CPI y/y", actual_html="<!---->")
    assert flm.find_actual_for_event(html, "CPI y/y", "USD") is None


def test_find_actual_for_event_no_matching_title_returns_none():
    html = _ROW_TEMPLATE.format(country="USD", title="CPI y/y", actual_html='<span class="better">3.4%</span>')
    assert flm.find_actual_for_event(html, "Some Other Event", "USD") is None


def test_find_actual_for_event_title_matches_but_wrong_currency():
    """Multiple currencies can share an event title (e.g. "GDP m/m") --
    must not cross-match to the wrong country's row."""
    html = _ROW_TEMPLATE.format(country="GBP", title="GDP m/m", actual_html='<span class="better">0.4%</span>')
    assert flm.find_actual_for_event(html, "GDP m/m", "USD") is None


def test_parse_event_time_am_pm():
    assert flm._parse_event_time("8:30am", date(2026, 9, 11)) == "2026-09-11T08:30:00-04:00"
    assert flm._parse_event_time("2:00pm", date(2026, 9, 11)) == "2026-09-11T14:00:00-04:00"


def test_parse_event_time_noon_and_midnight_edge_cases():
    assert flm._parse_event_time("12:00pm", date(2026, 9, 11)) == "2026-09-11T12:00:00-04:00"
    assert flm._parse_event_time("12:00am", date(2026, 9, 11)) == "2026-09-11T00:00:00-04:00"


def test_parse_event_time_handles_dst_offset_correctly():
    """zoneinfo, not a hardcoded offset -- January is EST (-05:00), not EDT."""
    assert flm._parse_event_time("8:30am", date(2026, 1, 15)) == "2026-01-15T08:30:00-05:00"


def test_parse_event_time_non_clock_text_returns_none():
    assert flm._parse_event_time("All Day", date(2026, 9, 11)) is None
    assert flm._parse_event_time("Tentative", date(2026, 9, 11)) is None
    assert flm._parse_event_time("", date(2026, 9, 11)) is None


def test_normalize_impact_maps_known_levels():
    assert flm._normalize_impact("High Impact Expected") == "High"
    assert flm._normalize_impact("Medium Impact Expected") == "Medium"
    assert flm._normalize_impact("Low Impact Expected") == "Low"
    assert flm._normalize_impact("Non-Economic / Holiday") == "Holiday"


def test_normalize_impact_unknown_text_passed_through():
    assert flm._normalize_impact("Some New Label") == "Some New Label"


_DAY_TABLE_TEMPLATE = """
<table class="calendar__table">
  <tr class="calendar__row calendar_row">
    <td class="calendar__cell calendar__time time">8:30am</td>
    <td class="calendar__cell calendar__currency currency">USD</td>
    <td class="impact"><span title="High Impact Expected"></span></td>
    <td class="calendar__cell calendar__event event"><span class="calendar__event-title">Core CPI m/m</span></td>
    <td class="calendar__cell calendar__actual actual"><span class="better">0.3%</span></td>
    <td class="calendar__cell calendar__forecast forecast">0.2%</td>
    <td class="calendar__cell calendar__previous previous">0.2%</td>
  </tr>
  <tr class="calendar__row calendar_row">
    <td class="calendar__cell calendar__time time"></td>
    <td class="calendar__cell calendar__currency currency">USD</td>
    <td class="impact"><span title="High Impact Expected"></span></td>
    <td class="calendar__cell calendar__event event"><span class="calendar__event-title">Core CPI y/y</span></td>
    <td class="calendar__cell calendar__actual actual"><span class="worse">2.4%</span></td>
    <td class="calendar__cell calendar__forecast forecast">2.4%</td>
    <td class="calendar__cell calendar__previous previous">2.5%</td>
  </tr>
  <tr class="calendar__row calendar_row">
    <td class="calendar__cell calendar__time time">10:00am</td>
    <td class="calendar__cell calendar__currency currency">USD</td>
    <td class="impact"><span title="Medium Impact Expected"></span></td>
    <td class="calendar__cell calendar__event event"><span class="calendar__event-title">Prelim UoM Consumer Sentiment</span></td>
    <td class="calendar__cell calendar__actual actual"><!----></td>
    <td class="calendar__cell calendar__forecast forecast">51.0</td>
    <td class="calendar__cell calendar__previous previous">51.0</td>
  </tr>
</table>
"""


def test_find_all_events_for_day_parses_every_row():
    day = date(2026, 9, 11)
    events = flm.find_all_events_for_day(_DAY_TABLE_TEMPLATE, day)
    assert len(events) == 3
    assert [e["title"] for e in events] == ["Core CPI m/m", "Core CPI y/y", "Prelim UoM Consumer Sentiment"]


def test_find_all_events_for_day_forward_fills_blank_time_cells():
    """Confirmed from a real Forex Factory screenshot: the time is only
    printed on the first row of a same-time group, left blank after --
    the second row here (Core CPI y/y) has an empty time cell and must
    inherit 8:30am from the row above it, not be dropped."""
    day = date(2026, 9, 11)
    events = flm.find_all_events_for_day(_DAY_TABLE_TEMPLATE, day)
    assert events[0]["date"] == "2026-09-11T08:30:00-04:00"
    assert events[1]["date"] == "2026-09-11T08:30:00-04:00"  # forward-filled
    assert events[2]["date"] == "2026-09-11T10:00:00-04:00"


def test_find_all_events_for_day_extracts_impact_forecast_previous_and_actual():
    day = date(2026, 9, 11)
    events = flm.find_all_events_for_day(_DAY_TABLE_TEMPLATE, day)
    first = events[0]
    assert first["country"] == "USD"
    assert first["impact"] == "High"
    assert first["forecast"] == "0.2%"
    assert first["previous"] == "0.2%"
    assert first["actual"] == "0.3%"
    assert first["direction"] == "better"
    assert first["surprise_pct"] is not None


def test_find_all_events_for_day_empty_actual_cell_stays_none():
    day = date(2026, 9, 11)
    events = flm.find_all_events_for_day(_DAY_TABLE_TEMPLATE, day)
    not_yet_released = events[2]
    assert not_yet_released["actual"] is None
    assert not_yet_released["direction"] is None
    assert not_yet_released["surprise_pct"] is None


def test_find_all_events_for_day_skips_rows_with_no_parseable_time():
    """A row with no time cell text and nothing to forward-fill from yet
    (the very first row of the table) can't be placed in a time-sorted
    table or stored (the db's primary key needs a non-null date) --
    it must be skipped, not crash the whole parse."""
    html = """
    <table class="calendar__table">
      <tr class="calendar__row calendar_row">
        <td class="calendar__cell calendar__time time"></td>
        <td class="calendar__cell calendar__currency currency">USD</td>
        <td class="calendar__cell calendar__event event"><span class="calendar__event-title">Untimed Event</span></td>
        <td class="calendar__cell calendar__actual actual"></td>
      </tr>
    </table>
    """
    events = flm.find_all_events_for_day(html, date(2026, 9, 11))
    assert events == []


def test_poll_for_actual_returns_immediately_when_found(monkeypatch):
    past_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    monkeypatch.setattr(flm, "fetch_live_day_html", lambda day: "<html></html>")
    monkeypatch.setattr(flm, "find_actual_for_event", lambda html, title, country: {"actual": "0.3%", "direction": "better", "raw_class": "better"})

    result, elapsed, timed_out = flm.poll_for_actual("Core CPI m/m", "USD", past_time, poll_interval_seconds=0.01, max_wait_seconds=60)

    assert timed_out is False
    assert result["actual"] == "0.3%"
    assert elapsed is not None and elapsed >= 0


def test_poll_for_actual_retries_until_found(monkeypatch):
    past_time = datetime.now(timezone.utc) - timedelta(seconds=1)
    monkeypatch.setattr(flm, "fetch_live_day_html", lambda day: "<html></html>")
    monkeypatch.setattr(flm.time, "sleep", lambda seconds: None)  # skip real waiting

    calls = {"n": 0}

    def fake_find(html, title, country):
        calls["n"] += 1
        return None if calls["n"] < 3 else {"actual": "0.4%", "direction": "better", "raw_class": "better"}

    monkeypatch.setattr(flm, "find_actual_for_event", fake_find)

    result, elapsed, timed_out = flm.poll_for_actual("CPI y/y", "USD", past_time, poll_interval_seconds=0.01, max_wait_seconds=60)

    assert timed_out is False
    assert calls["n"] == 3
    assert result["actual"] == "0.4%"


def test_poll_for_actual_times_out_when_never_found(monkeypatch):
    long_past = datetime.now(timezone.utc) - timedelta(seconds=400)  # already past a small max_wait
    monkeypatch.setattr(flm, "fetch_live_day_html", lambda day: "<html></html>")
    monkeypatch.setattr(flm, "find_actual_for_event", lambda html, title, country: None)

    result, elapsed, timed_out = flm.poll_for_actual("CPI y/y", "USD", long_past, poll_interval_seconds=0.01, max_wait_seconds=5)

    assert timed_out is True
    assert result is None
    assert elapsed is None


def test_qualifying_events_defaults_to_us_high_impact_only_and_future():
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    events = [
        {"title": "CPI m/m", "country": "USD", "impact": "High", "date": future},
        {"title": "GDP m/m", "country": "GBP", "impact": "High", "date": future},  # non-US, excluded by default
        {"title": "Retail Sales", "country": "USD", "impact": "Medium", "date": future},  # not High, excluded
        {"title": "Old NFP", "country": "USD", "impact": "High", "date": past},  # already past, excluded
    ]
    result = flm._qualifying_events(events, include_non_us=False)
    assert [e["title"] for e in result] == ["CPI m/m"]


def test_qualifying_events_include_non_us():
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    events = [
        {"title": "CPI m/m", "country": "USD", "impact": "High", "date": future},
        {"title": "GDP m/m", "country": "GBP", "impact": "High", "date": future},
    ]
    result = flm._qualifying_events(events, include_non_us=True)
    assert {e["title"] for e in result} == {"CPI m/m", "GDP m/m"}


def test_append_result_row_writes_header_once(tmp_path):
    csv_path = str(tmp_path / "out" / "results.csv")
    row = flm.PollResult(
        title="CPI m/m", country="USD", impact="High", scheduled_at="2026-09-11T08:30:00-04:00",
        forecast="0.4%", previous="0.1%", actual="0.4%", direction="better",
        seconds_after_scheduled=12.3, timed_out=False,
    )
    flm._append_result_row(csv_path, row)
    flm._append_result_row(csv_path, row)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["title"] == "CPI m/m"
    assert rows[0]["direction"] == "better"


def test_run_week_monitor_returns_empty_string_when_nothing_qualifies(monkeypatch, tmp_path):
    monkeypatch.setattr(flm.forex_calendar, "fetch_calendar_events", lambda: [])
    result = flm.run_week_monitor(output_dir=str(tmp_path))
    assert result == ""


def test_run_week_monitor_logs_each_qualifying_event(monkeypatch, tmp_path):
    future = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(
        flm.forex_calendar,
        "fetch_calendar_events",
        lambda: [
            {"title": "CPI m/m", "country": "USD", "impact": "High", "date": future, "forecast": "0.4%", "previous": "0.1%"},
            {"title": "PPI m/m", "country": "USD", "impact": "High", "date": future, "forecast": "0.3%", "previous": "0.2%"},
        ],
    )
    monkeypatch.setattr(
        flm,
        "poll_for_actual",
        lambda title, country, event_time, poll_interval_seconds, max_wait_seconds: (
            {"actual": "0.5%", "direction": "better", "raw_class": "better"}, 2.0, False
        ),
    )

    csv_path = flm.run_week_monitor(output_dir=str(tmp_path))

    assert csv_path
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert {r["title"] for r in rows} == {"CPI m/m", "PPI m/m"}
    assert all(r["actual"] == "0.5%" for r in rows)
