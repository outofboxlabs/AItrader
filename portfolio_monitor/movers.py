"""Daily market-mover scan: find liquid US stocks that dropped hard in a
single day, pull their analyst ratings, and ask the configured AI
provider for an informational rebound read. Like the news layer, this is
explicitly NOT a trade signal -- the model lays out a case both ways and
is never allowed to say buy/sell/hold.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

import yfinance as yf
from yfinance import EquityQuery

from . import ai_client
from . import news as news_mod

REBOUND_SYSTEM_PROMPT = """You are a market analyst embedded in a portfolio \
monitoring tool. You are given one stock that dropped sharply in a single \
trading day, along with recent headlines, analyst ratings/price targets, \
and the current macro environment. Your job is ONLY to inform, never to \
advise:

- Summarize, in 1-2 sentences, what likely caused the drop.
- Lay out the case FOR this being a reasonable rebound candidate, if there \
is one.
- List the key risk factors AGAINST it, if any.
- State current analyst sentiment as "bullish", "neutral", "bearish", or \
"no data", based on the ratings/price targets given.
- Note briefly whether the current macro environment (given as a 0-100 \
"calm" score, higher = calmer) supports or argues against adding risk \
right now.

You must NEVER say to buy, sell, hold, or otherwise act on this stock. \
You are laying out a case both ways for a human to weigh, not telling \
them what to do.

Respond with ONLY a JSON object, no other text, matching exactly this \
shape:
{"cause_summary": "...", "rebound_case": "...", "risk_factors": ["...", "..."], \
"analyst_sentiment": "bullish|neutral|bearish|no data", "macro_context": "...", \
"disclaimer": "This is not investment advice."}
"""


def find_big_drops(
    threshold_pct: float = -40.0,
    min_market_cap: float = 2_000_000_000,
    min_price: float = 5.0,
    min_volume: int = 20_000,
    max_results: int = 25,
) -> list[dict]:
    """Liquid US stocks whose intraday % change is below `threshold_pct`
    (e.g. -40.0 means "dropped more than 40%"), via Yahoo's screener.
    Defaults mirror Yahoo's own "day_losers" liquidity filters so
    penny-stock/illiquid noise doesn't drown out real crashes."""
    query = EquityQuery(
        "AND",
        [
            EquityQuery("LT", ["percentchange", threshold_pct]),
            EquityQuery("EQ", ["region", "us"]),
            EquityQuery("GTE", ["intradaymarketcap", min_market_cap]),
            EquityQuery("GTE", ["intradayprice", min_price]),
            EquityQuery("GT", ["dayvolume", min_volume]),
        ],
    )
    response = yf.screen(query, sortField="percentchange", sortAsc=True, size=max_results)
    quotes = response.get("quotes", []) if response else []
    return [
        {
            "ticker": q.get("symbol"),
            "name": q.get("shortName") or q.get("longName"),
            "pct_change": q.get("regularMarketChangePercent"),
            "price": q.get("regularMarketPrice"),
            "volume": q.get("regularMarketVolume"),
            "market_cap": q.get("marketCap"),
        }
        for q in quotes
    ]


def get_analyst_snapshot(ticker: str) -> dict:
    """Analyst price targets + recommendation summary + recent rating
    changes for one ticker. Any field that fails to fetch is left empty
    rather than failing the whole snapshot."""
    t = yf.Ticker(ticker)
    price_targets: dict = {}
    recommendations: dict = {}
    recent_actions: list = []
    try:
        price_targets = t.get_analyst_price_targets() or {}
    except Exception:
        pass
    try:
        recommendations = t.get_recommendations_summary(as_dict=True) or {}
    except Exception:
        pass
    try:
        recent_actions = t.get_upgrades_downgrades(as_dict=True) or []
    except Exception:
        pass
    raw = {
        "price_targets": price_targets,
        "recommendations": recommendations,
        "recent_actions": recent_actions,
    }
    # yfinance's dict conversions can carry pandas/numpy types (Timestamp,
    # int64, ...) that json.dumps chokes on downstream (Flask's jsonify,
    # SQLite storage) -- round-trip through json with a str fallback now
    # so everything past this point is guaranteed plain JSON types.
    return json.loads(json.dumps(raw, default=str))


def build_rebound_user_message(
    ticker: str, drop: dict, headlines: list[dict], analyst: dict, macro_score: Optional[float]
) -> str:
    lines = [
        f"Ticker: {ticker} ({drop.get('name') or 'n/a'})",
        f"Today's drop: {drop.get('pct_change')}% to ${drop.get('price')}",
        f"Volume: {drop.get('volume')}, Market cap: {drop.get('market_cap')}",
        "",
        "Recent headlines:",
    ]
    if headlines:
        for h in headlines:
            when = h["published_at"].isoformat() if h.get("published_at") else "unknown date"
            lines.append(f"- [{when}] {h['title']}")
    else:
        lines.append("(none found in the lookback window)")

    lines.append("")
    lines.append(f"Analyst data (JSON): {json.dumps(analyst, default=str)}")
    lines.append("")
    if macro_score is not None:
        lines.append(f"Current macro gate score: {macro_score:.1f}/100 (higher = calmer environment)")
    else:
        lines.append("Macro gate score: not available")
    return "\n".join(lines)


def analyze_rebound_candidate(
    ticker: str,
    drop: dict,
    headlines: list[dict],
    analyst: dict,
    macro_score: Optional[float],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> dict:
    """Raises on API failure -- the caller decides how to degrade."""
    user_message = build_rebound_user_message(ticker, drop, headlines, analyst, macro_score)
    text = ai_client.call_provider(provider, REBOUND_SYSTEM_PROMPT, user_message, model, api_key=api_key, max_tokens=800)
    return _parse_rebound_json(text)


def _parse_rebound_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            return {
                "cause_summary": text.strip()[:500] or None,
                "rebound_case": None,
                "risk_factors": [],
                "analyst_sentiment": "no data",
                "macro_context": None,
                "disclaimer": "This is not investment advice.",
                "parse_error": True,
            }
    parsed.setdefault("disclaimer", "This is not investment advice.")
    parsed.setdefault("parse_error", False)
    return parsed


def get_or_analyze_rebound(
    conn,
    ticker: str,
    asof_date: date,
    drop: dict,
    headlines: list[dict],
    analyst: dict,
    macro_score: Optional[float],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> dict:
    """Cached per (ticker, asof_date): only calls the API once per name per day."""
    from . import db as db_mod

    cached = db_mod.get_rebound_analysis(conn, asof_date.isoformat(), ticker)
    if cached is not None:
        return cached

    try:
        analysis = analyze_rebound_candidate(ticker, drop, headlines, analyst, macro_score, provider, model, api_key)
        status = "ok"
    except Exception as exc:  # missing API key, network error, rate limit, etc.
        analysis = {
            "cause_summary": None,
            "rebound_case": None,
            "risk_factors": [],
            "analyst_sentiment": "no data",
            "macro_context": None,
            "disclaimer": "This is not investment advice.",
            "parse_error": False,
        }
        status = f"skipped: {exc}"

    result = {
        "ticker": ticker,
        "asof_date": asof_date.isoformat(),
        "provider": provider,
        "model": model,
        "status": status,
        **analysis,
    }
    db_mod.save_rebound_analysis(conn, result)
    return result


def run_movers_scan(
    conn,
    asof_date: date,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    threshold_pct: float = -40.0,
    min_market_cap: float = 2_000_000_000,
    min_price: float = 5.0,
    min_volume: int = 20_000,
    max_results: int = 25,
    news_window_days: int = 3,
    macro_score: Optional[float] = None,
) -> list[dict]:
    """Full pipeline for one day: screen for big drops, pull headlines and
    analyst data per name, get a cached rebound read, persist the drop
    data, and return the combined per-ticker results."""
    from . import db as db_mod

    drops = find_big_drops(threshold_pct, min_market_cap, min_price, min_volume, max_results)
    db_mod.save_market_movers(conn, asof_date.isoformat(), drops)

    results = []
    for drop in drops:
        ticker = drop["ticker"]
        headlines = news_mod.fetch_recent_headlines(ticker, window_days=news_window_days)
        analyst = get_analyst_snapshot(ticker)
        rebound = get_or_analyze_rebound(
            conn, ticker, asof_date, drop, headlines, analyst, macro_score, provider, model, api_key=api_key
        )
        results.append({**drop, "analyst": analyst, "rebound": rebound})
    return results
