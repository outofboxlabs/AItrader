"""Analyst price-target data for a ticker via Financial Modeling Prep
(FMP) -- confirmed working on FMP's free "Basic" plan, but ONLY for a
limited set of "popular" symbols. Confirmed live: AAPL returns real data
on a free key, while a smaller-cap name (BBW) returned HTTP 402 Payment
Required with the body "This value set for 'symbol' is not available
under your current subscription." So this isn't a blanket paid-vs-free
endpoint split -- it's symbol-by-symbol, and this app is specifically
built to analyze beaten-down/smaller-cap stocks near their 52-week low,
exactly the kind of ticker likely to hit that wall. Because of that,
get_price_target_snapshot() surfaces the real failure reason in the
result instead of collapsing every failure into a bare None, so a human
(and the AI persona reading it) can tell "FMP wants you to upgrade for
this ticker" apart from "there's just no data" or "the key is missing."

Separately: FMP's individual, per-analyst, per-date price-target history
(the data that would let us show "on this date, this firm called for $X,
expected in 12 months") lives behind a further paid TipRanks add-on, not
the base plan -- and even that only covers 3 years of history. What the
base "Analyst" endpoints give us, when they work for a given symbol, is:

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

get_price_target_summary's trailing-window field names
(`{window}AvgPriceTarget` / `{window}Count`) are a best-effort
reconstruction from public documentation, not independently verified
against a real response -- parsed defensively so a renamed/missing field
degrades to that window just not appearing, not a crash.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

CONSENSUS_URL = "https://financialmodelingprep.com/stable/price-target-consensus"
SUMMARY_URL = "https://financialmodelingprep.com/stable/price-target-summary"

TRAILING_WINDOWS = ["lastMonth", "lastQuarter", "lastYear", "allTime"]


def _get(url: str, ticker: str, api_key: str, label: str) -> tuple[Optional[dict], Optional[str]]:
    """Returns (data, error_message). data is the first record on success;
    error_message is a short, human-readable reason on any failure (HTTP
    error, network error, empty/unexpected response), always logged in
    full to stderr regardless."""
    response = None
    try:
        response = requests.get(url, params={"symbol": ticker, "apikey": api_key}, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        status = getattr(response, "status_code", "no response")
        body = (getattr(response, "text", "") or "")[:300]
        print(f"[price_targets] FMP {label} request for {ticker} failed (status={status}): {type(exc).__name__}: {exc} -- body: {body}", file=sys.stderr)
        if isinstance(status, int):
            return None, f"HTTP {status} from FMP: {body or exc}"
        return None, f"{type(exc).__name__}: {exc}"

    if not isinstance(payload, list) or not payload:
        print(f"[price_targets] FMP {label} returned no data for {ticker}: {str(payload)[:300]}", file=sys.stderr)
        return None, None  # genuinely empty, not an error -- ticker just has no data here
    return payload[0], None


def get_price_target_snapshot(ticker: str, api_key: str) -> Optional[dict]:
    """Live analyst price-target snapshot for `ticker`: today's
    high/low/median/consensus target plus trailing-window averages
    (whichever of last month/quarter/year/all-time the response
    includes), each annotated with a `target_date` (today + 365 days,
    the conventional 12-month horizon -- not a date any single firm
    stated).

    Returns None only when no api_key was given at all. Otherwise always
    returns a dict: on a real failure (HTTP error, network error) for the
    consensus call, that dict is just {"error": "<reason>"} so a human
    (or the AI persona reading it) can see why, rather than a bare "no
    data" that could as easily mean the ticker has none. The summary
    call failing/being empty just means no trailing-window averages --
    not a total failure, since the consensus figures alone are useful."""
    if not api_key:
        return None

    consensus, error = _get(CONSENSUS_URL, ticker, api_key, "consensus")
    if error:
        return {"error": error}
    if consensus is None:
        return None  # genuinely no data for this ticker, not an error

    target_date = (datetime.now(timezone.utc) + timedelta(days=365)).date().isoformat()
    snapshot = {
        "target_high": consensus.get("targetHigh"),
        "target_low": consensus.get("targetLow"),
        "target_consensus": consensus.get("targetConsensus"),
        "target_median": consensus.get("targetMedian"),
        "target_date": target_date,
        "trailing_windows": [],
    }

    summary, _summary_error = _get(SUMMARY_URL, ticker, api_key, "summary")
    if summary:
        for window in TRAILING_WINDOWS:
            avg = summary.get(f"{window}AvgPriceTarget")
            count = summary.get(f"{window}Count")
            if avg is not None:
                snapshot["trailing_windows"].append({"window": window, "avg_price_target": avg, "count": count})

    return snapshot
