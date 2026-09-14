"""SEC EDGAR filings: recent 10-K/10-Q/8-K/etc. for a ticker, straight
from SEC's own official, free, public API -- data.sec.gov. No API key,
no third-party "skill" or scraper in the middle. This is a real,
long-standing government API (stable for years), not a guess at an
undocumented endpoint.

Two calls are involved:
1. https://www.sec.gov/files/company_tickers.json -- SEC's own bulk
   ticker-to-CIK mapping (there's no "look up by ticker" endpoint, so
   this file is the standard, documented way every EDGAR integration
   does this lookup). Fetched once per process and cached in memory --
   it's ~1MB and changes rarely, so re-fetching it per request would be
   wasteful.
2. https://data.sec.gov/submissions/CIK##########.json -- one company's
   full filing history, most recent first.

SEC explicitly asks every caller to identify themselves with a
descriptive User-Agent header (see config.EDGAR_USER_AGENT) and to keep
request volume reasonable -- this module makes at most one call per
ticker per lookup, not a bulk crawl.
"""

from __future__ import annotations

from typing import Optional

import requests

import config

_TICKER_TO_CIK: Optional[dict[str, str]] = None


def _get_cik_map() -> dict[str, str]:
    """Ticker (uppercase) -> zero-padded 10-digit CIK string. Cached in
    memory for the life of the process after the first successful pull."""
    global _TICKER_TO_CIK
    if _TICKER_TO_CIK is not None:
        return _TICKER_TO_CIK

    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        timeout=15,
        headers={"User-Agent": config.EDGAR_USER_AGENT},
    )
    response.raise_for_status()
    raw = response.json()
    mapping = {}
    for entry in raw.values():
        ticker = entry.get("ticker")
        cik = entry.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker.upper()] = f"{int(cik):010d}"
    _TICKER_TO_CIK = mapping
    return mapping


def get_recent_filings(ticker: str, limit: int = 8) -> list[dict]:
    """Most recent SEC filings of any type for `ticker`, most-recent-first,
    as [{"form": "10-Q", "filed": "2026-08-01", "report_date": "...",
    "description": "...", "url": "https://www.sec.gov/Archives/..."}].
    Never raises -- a failed lookup (bad ticker, network error, SEC
    unavailable) just means an empty list, which the caller/AI prompt
    should treat as "no filings data available" rather than an error."""
    try:
        cik = _get_cik_map().get(ticker.upper())
        if not cik:
            return []

        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            timeout=15,
            headers={"User-Agent": config.EDGAR_USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    accession_numbers = recent.get("accessionNumber") or []
    primary_documents = recent.get("primaryDocument") or []
    descriptions = recent.get("primaryDocDescription") or []

    filings = []
    for i in range(len(forms)):
        accession_no_dashes = (accession_numbers[i] if i < len(accession_numbers) else "").replace("-", "")
        primary_doc = primary_documents[i] if i < len(primary_documents) else ""
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{primary_doc}"
            if accession_no_dashes and primary_doc
            else None
        )
        filings.append(
            {
                "form": forms[i],
                "filed": filing_dates[i] if i < len(filing_dates) else None,
                "report_date": report_dates[i] if i < len(report_dates) else None,
                "description": descriptions[i] if i < len(descriptions) else None,
                "url": url,
            }
        )

    return filings[:limit]
