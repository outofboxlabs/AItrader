#!/usr/bin/env python3
"""Watches Forex Factory's live calendar page for this week's remaining
high-impact events and logs how fast/what the actual print looked like
compared to the forecast, once each one releases.

This is a research tool for answering "how quickly can we actually see
the real number after a scheduled release" -- a prerequisite question
for any later trade-on-news idea. It does NOT place any trade.

Scrapes forexfactory.com's live HTML page directly, which is against
their Terms of Service -- see portfolio_monitor/forex_live_monitor.py's
docstring for that tradeoff. Runs for as long as this week's remaining
events take to release (potentially days) -- leave it running in a
terminal, it logs each result to a CSV as it happens.

Usage:
    python forex_monitor.py                  # US (USD) high-impact events only
    python forex_monitor.py --include-non-us  # every currency's high-impact events
"""

from __future__ import annotations

import argparse

from portfolio_monitor import forex_live_monitor


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Forex Factory's live calendar for actual releases")
    parser.add_argument(
        "--include-non-us",
        action="store_true",
        help="include every currency's high-impact events, not just USD",
    )
    parser.add_argument("--poll-interval", type=float, default=3.0, help="seconds between checks after an event's scheduled time")
    parser.add_argument("--max-wait", type=float, default=300.0, help="seconds to keep checking before giving up on one event")
    parser.add_argument("--output-dir", default="exports/forex_live_monitor")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    forex_live_monitor.run_week_monitor(
        include_non_us=args.include_non_us,
        poll_interval_seconds=args.poll_interval,
        max_wait_seconds=args.max_wait,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
