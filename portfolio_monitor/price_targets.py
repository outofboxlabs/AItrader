"""Analyst price-target data for a ticker via Financial Modeling Prep
(FMP) -- confirmed working on FMP's free "Basic" plan (no paid add-on).

Important limitation discovered while building this: FMP's individual,
per-analyst, per-date price-target history (the data that would let us
show "on this date, this firm called for $X, expected in 12 months")
lives behind a separate paid TipRanks add-on, not the base plan -- and
even that only covers 3 years of history. What the base "Analyst"
endpoints actually give us, for free, is:

- price-target-consensus: today's live high/low/median/consensus target
  across all covering analysts, no individual dates.
- price-target-summary: the average target over a few trailing windows
  (last month/quarter/year/all-time), each with how many targets went
  into it -- gives some sense of a trend (is the average rising or
  falling recently) without individual per-analyst events.

So this module reports a live snapshot, not a history. A `target_date`
label is still attached (today + 365 days, the conventional Wall Street
12-month horizon) since that's the standard way analyst targets are
framed, but it describes the snapshot as a whole, not any one firm's
specific call.

get_price_target_summary()'s trailing-window field names
(`{window}AvgPriceTarget` / `{window}Count`) are a best-effort
reconstruction from public documentation, not independently verified
against a real response -- parsed defensively so a renamed/missing field
degrades to that window just not appearing, not a crash. Every network
call here degrades to None on failure (bad ticker, network error,
invalid key, unexpected shape) and logs the real HTTP status so a
failure is easy to diagnose rather than silently swallowed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

CONSENSUS_URL = "https://financialmodelingprep.com/stable/price-target-consensus"
SUMMARY_URL = "https://financialmodelingprep.com/stable/price-target-summary"

TRAILING_WINDOWS = ["lastMonth", "lastQuarter", "lastYear", "allTime"]


def _get(url: str, ticker: str, api_key: str, label: str) -> Optional[dict]:
    response = None
    try:
        response = requests.get(url, params={"symbol": ticker, "apikey": api_key}, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status = getattr(response, "status_code", "no response")
        body = (getattr(response, "text", "") or "")[:300]
        print(f"[price_targets] FMP {label} request for {ticker} failed (status={status}): {type(exc).__name__}: {exc} -- body: {body}", file=sys.stderr)
        return None

    if not isinstance(payload, list) or not payload:
        print(f"[price_targets] FMP {label} returned no data for {ticker}: {str(payload)[:300]}", file=sys.stderr)
        return None
    return payload[0]


def get_price_target_snapshot(ticker: str, api_key: str) -> Optional[dict]:
    """Live analyst price-target snapshot for `ticker`: today's
    high/low/median/consensus target plus trailing-window averages
    (whichever of last month/quarter/year/all-time the response
    includes), each annotated with a `target_date` (today + 365 days,
    the conventional 12-month horizon -- not a date any single firm
    stated). Returns None only if the consensus call itself fails;
    the summary call failing just means no trailing-window averages,
    not a total failure, since the consensus figures alone are useful
    on their own."""
    if not api_key:
        return None

    consensus = _get(CONSENSUS_URL, ticker, api_key, "consensus")
    if consensus is None:
        return None

    target_date = (datetime.now(timezone.utc) + timedelta(days=365)).date().isoformat()
    snapshot = {
        "target_high": consensus.get("targetHigh"),
        "target_low": consensus.get("targetLow"),
        "target_consensus": consensus.get("targetConsensus"),
        "target_median": consensus.get("targetMedian"),
        "target_date": target_date,
        "trailing_windows": [],
    }

    summary = _get(SUMMARY_URL, ticker, api_key, "summary")
    if summary:
        for window in TRAILING_WINDOWS:
            avg = summary.get(f"{window}AvgPriceTarget")
            count = summary.get(f"{window}Count")
            if avg is not None:
                snapshot["trailing_windows"].append({"window": window, "avg_price_target": avg, "count": count})

    return snapshot
