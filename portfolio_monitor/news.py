"""Daily Claude-powered news analysis per held underlying.

This is NOT a trade signal: Claude is asked only to summarize, read
sentiment, and flag anything that specifically touches a held position --
never to recommend buying or selling. Each (ticker, date) result is cached
in SQLite so re-running the monitor the same day never re-bills the API;
the per-call cost is the only paid part of this system.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import yfinance as yf

SYSTEM_PROMPT = """You are a financial news analyst embedded in a portfolio \
monitoring tool. You are given recent headlines for one stock/ETF ticker \
that the user currently holds a position in. Your job is ONLY to inform, \
never to advise:

- Summarize, in 2-3 sentences, what actually happened.
- Read the overall sentiment of the news as "positive", "neutral", or \
"negative".
- List the key drivers behind the news (short phrases).
- Flag whether anything in the headlines specifically and materially \
affects a held position (e.g. earnings surprise, guidance change, \
analyst upgrade/downgrade, M&A, regulatory action, management change) \
and briefly say why.

You must NEVER say to buy, sell, hold, or otherwise act on a position. \
You are a summarizer and a flagger, not an advisor.

Respond with ONLY a JSON object, no other text, matching exactly this \
shape:
{"summary": "...", "sentiment": "positive|neutral|negative", \
"key_drivers": ["...", "..."], "position_flag": true|false, \
"position_flag_reason": "..." or null}
"""


def fetch_recent_headlines(ticker: str, window_days: int = 3, now: Optional[datetime] = None) -> list[dict]:
    """Recent headlines for `ticker` via yfinance .news, filtered to the
    last `window_days` days. Tolerates both the old and new yfinance news
    payload shapes."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:
        raw_items = []

    headlines = []
    for raw in raw_items:
        item = _normalize_news_item(raw)
        if item is None:
            continue
        if item["published_at"] is not None and item["published_at"] < cutoff:
            continue
        headlines.append(item)
    return headlines


def _normalize_news_item(raw: dict) -> Optional[dict]:
    # Newer yfinance shape: {"content": {"title", "summary", "pubDate", "provider": {...}}}
    content = raw.get("content")
    if isinstance(content, dict):
        title = content.get("title")
        if not title:
            return None
        provider = content.get("provider") or {}
        return {
            "title": title,
            "summary": content.get("summary"),
            "publisher": provider.get("displayName"),
            "published_at": _parse_iso(content.get("pubDate")),
        }

    # Older yfinance shape: top-level title/publisher/providerPublishTime (epoch seconds)
    title = raw.get("title")
    if not title:
        return None
    published_at = None
    epoch = raw.get("providerPublishTime")
    if epoch:
        published_at = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return {
        "title": title,
        "summary": raw.get("summary"),
        "publisher": raw.get("publisher"),
        "published_at": published_at,
    }


def _parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def build_user_message(ticker: str, headlines: list[dict]) -> str:
    if not headlines:
        return f"Ticker: {ticker}\nNo headlines found in the configured lookback window."
    lines = [f"Ticker: {ticker}", "Recent headlines:"]
    for h in headlines:
        when = h["published_at"].isoformat() if h["published_at"] else "unknown date"
        publisher = h["publisher"] or "unknown source"
        lines.append(f"- [{when}] ({publisher}) {h['title']}")
        if h.get("summary"):
            lines.append(f"  {h['summary']}")
    return "\n".join(lines)


def analyze_headlines_with_claude(ticker: str, headlines: list[dict], api_key: Optional[str], model: str) -> dict:
    """Call the Claude API once and return a parsed analysis dict. Raises
    on failure (missing key, network, rate limit) -- the caller decides
    how to degrade rather than crashing the whole run over one name."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(ticker, headlines)}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return _parse_analysis_json(text)


def analyze_headlines_with_openai(ticker: str, headlines: list[dict], api_key: Optional[str], model: str) -> dict:
    """Same contract as analyze_headlines_with_claude, via the OpenAI API.
    Raises on failure -- the caller decides how to degrade."""
    import openai

    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(ticker, headlines)},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    return _parse_analysis_json(text)


def analyze_headlines(
    ticker: str, headlines: list[dict], provider: str, model: str, api_key: Optional[str] = None
) -> dict:
    """Dispatch to whichever provider is configured."""
    if provider == "anthropic":
        return analyze_headlines_with_claude(ticker, headlines, api_key=api_key, model=model)
    if provider == "openai":
        return analyze_headlines_with_openai(ticker, headlines, api_key=api_key, model=model)
    raise ValueError(f"Unknown NEWS_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')")


def _parse_analysis_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {
                "summary": text.strip()[:500] or None,
                "sentiment": "neutral",
                "key_drivers": [],
                "position_flag": False,
                "position_flag_reason": None,
                "parse_error": True,
            }
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "summary": text.strip()[:500] or None,
                "sentiment": "neutral",
                "key_drivers": [],
                "position_flag": False,
                "position_flag_reason": None,
                "parse_error": True,
            }
    parsed.setdefault("parse_error", False)
    return parsed


def get_or_analyze_news(
    conn,
    ticker: str,
    asof_date: date,
    window_days: int,
    model: str,
    provider: str = "anthropic",
    api_key: Optional[str] = None,
) -> dict:
    """Cached per (ticker, asof_date): only calls the API once per name per day."""
    from . import db as db_mod

    cached = db_mod.get_news_analysis(conn, asof_date.isoformat(), ticker)
    if cached is not None:
        return cached

    headlines = fetch_recent_headlines(ticker, window_days=window_days)
    try:
        analysis = analyze_headlines(ticker, headlines, provider=provider, model=model, api_key=api_key)
        status = "ok"
    except Exception as exc:  # missing API key, network error, rate limit, etc.
        analysis = {
            "summary": None,
            "sentiment": None,
            "key_drivers": [],
            "position_flag": False,
            "position_flag_reason": None,
            "parse_error": False,
        }
        status = f"skipped: {exc}"

    result = {
        "ticker": ticker,
        "asof_date": asof_date.isoformat(),
        "headline_count": len(headlines),
        "window_days": window_days,
        "provider": provider,
        "model": model,
        "status": status,
        **analysis,
    }
    db_mod.save_news_analysis(conn, result)
    return result
