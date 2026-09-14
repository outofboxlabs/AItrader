"""Live market data pulls via yfinance: spot prices, option chains, marks."""

from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import yfinance as yf


@dataclass
class Quote:
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    iv: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]

    @property
    def mark(self) -> Optional[float]:
        """Mid of bid/ask when both are live quotes, else last trade price."""
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


def get_price_history(ticker: str, period: str = "60d", interval: str = "5m") -> list[dict]:
    """Intraday price history for a chart, as [{"time": ISO string with
    timezone, "close": float}, ...]. 5m bars cap out at 60 days back on
    Yahoo's free data (1m bars only go back 7 days -- too tight to cover
    "this month"), which is why that's the default rather than something
    finer. Returns [] rather than raising on an empty/failed pull --
    callers decide how to degrade (e.g. "no chart data for this ticker")."""
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period=period, interval=interval)
    except Exception:
        return []
    if hist.empty:
        return []
    return [{"time": idx.isoformat(), "close": float(row["Close"])} for idx, row in hist.iterrows()]


def get_daily_price_history(ticker: str, period: str = "1y") -> list[dict]:
    """Daily close history, as [{"date": "YYYY-MM-DD", "close": float}, ...],
    sorted oldest-first (yfinance's own order). Used to place events like an
    analyst rating change on a timeline relative to the stock's own price
    action over the past year -- e.g. finding the date of its 52-week low,
    or the price on the day a given rating was issued -- rather than for
    charting, so daily granularity over a year is what's needed, not
    intraday bars. Returns [] rather than raising on an empty/failed pull,
    same convention as get_price_history."""
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period=period, interval="1d")
    except Exception:
        return []
    if hist.empty:
        return []
    return [{"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])} for idx, row in hist.iterrows()]


def get_price_history_window(
    ticker: str, center_time: datetime, window_hours: float = 2.0, interval: Optional[str] = None
) -> tuple[list[dict], str]:
    """Zoomed price history around one specific timestamp -- the "show
    me the exact reaction" view. When `interval` is left as None, uses
    1-minute bars if center_time is within Yahoo's 7-day free-data limit
    for that interval, else 5-minute bars (which cover 60 days), falling
    back from 1m to 5m automatically if the 1m pull comes back empty near
    that edge. Passing an explicit `interval` (e.g. "1h" for the "switch
    to hourly" zoom-out view) skips all of that guessing and uses exactly
    what was asked for, with no fallback -- the caller made a deliberate
    choice and an empty result should say so rather than silently
    switching resolution underneath them. Returns (bars, interval_used)
    so the caller can be honest about precision instead of silently
    degrading -- older events genuinely can't get true minute-level data
    from this source, and callers should say so rather than pretend
    otherwise."""
    now = datetime.now(timezone.utc)
    start = center_time - timedelta(hours=window_hours)
    end = center_time + timedelta(hours=window_hours)
    auto_fallback = interval is None
    if interval is None:
        interval = "1m" if (now - center_time) <= timedelta(days=7) else "5m"

    t = yf.Ticker(ticker)

    def _pull(iv: str):
        try:
            # prepost=True so a pre-market release (e.g. an 8:30am economic
            # print, before the 9:30am open) still has bars on its "before"
            # side instead of the window starting empty right at the open.
            hist = t.history(start=start, end=end, interval=iv, prepost=True)
        except Exception:
            return None
        return None if hist.empty else hist

    hist = _pull(interval)
    if hist is None and auto_fallback and interval == "1m":
        interval = "5m"
        hist = _pull(interval)
    if hist is None:
        return [], interval
    return [{"time": idx.isoformat(), "close": float(row["Close"])} for idx, row in hist.iterrows()], interval


def get_spot_price(ticker: str) -> float:
    t = yf.Ticker(ticker)
    try:
        price = t.fast_info.get("lastPrice")
        if price:
            return float(price)
    except Exception:
        pass
    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"Could not determine spot price for {ticker}")
    return float(hist["Close"].iloc[-1])


def pull_full_chain(ticker: str) -> dict:
    """Pull the full option chain (every listed expiry, every strike) for a
    ticker. This is intentionally broader than just the held contracts so
    that later diffing can detect newly-listed strikes/expiries.
    """
    t = yf.Ticker(ticker)
    spot = get_spot_price(ticker)
    chain: dict = {"ticker": ticker, "spot": spot, "expiries": {}}
    for expiry in t.options:
        opt = t.option_chain(expiry)
        chain["expiries"][expiry] = {
            "calls": _frame_to_rows(opt.calls),
            "puts": _frame_to_rows(opt.puts),
        }
    return chain


def _frame_to_rows(df) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "strike": float(row.get("strike")),
                "bid": _safe_float(row.get("bid")),
                "ask": _safe_float(row.get("ask")),
                "last": _safe_float(row.get("lastPrice")),
                "iv": _safe_float(row.get("impliedVolatility")),
                "volume": _safe_float(row.get("volume")),
                "open_interest": _safe_float(row.get("openInterest")),
            }
        )
    return rows


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return None if f != f else f  # filter NaN
    except (TypeError, ValueError):
        return None


def resolve_ticker_query(query: str) -> Optional[dict]:
    """Resolve free text -- a ticker symbol or a company name -- to a
    single best-match ticker via Yahoo's own search index. This is a
    lookup problem, not a reasoning one, so it goes through Yahoo's
    search rather than asking an AI to guess a symbol (which can
    hallucinate a plausible-looking but wrong ticker). Prefers an exact
    case-insensitive symbol match if the query itself is one of the
    returned symbols, else the first equity-type result, else the first
    result of any type. Returns None if nothing matched at all."""
    from yfinance import Search

    try:
        quotes = Search(query, max_results=8).quotes
    except Exception:
        return None
    if not quotes:
        return None

    query_upper = query.strip().upper()
    exact = next((q for q in quotes if (q.get("symbol") or "").upper() == query_upper), None)
    if exact:
        best = exact
    else:
        equities = [q for q in quotes if q.get("quoteType") == "EQUITY"]
        best = (equities or quotes)[0]

    ticker = best.get("symbol")
    if not ticker:
        return None
    return {
        "ticker": ticker,
        "name": best.get("shortname") or best.get("longname"),
        "exchange": best.get("exchange"),
    }


def _current_period_ratings(recommendations: dict) -> dict:
    """Same shape-parsing as growth_screener._current_period_ratings and
    nearlow_screener._current_period_ratings -- duplicated locally rather
    than shared, since it's a small, self-contained piece of pandas
    .to_dict()-shape parsing and this module otherwise has no dependency
    on either screener."""
    result = {}
    for category in ("strongBuy", "buy", "hold", "sell", "strongSell"):
        col = recommendations.get(category)
        if isinstance(col, dict) and col:
            result[category] = next(iter(col.values()))
        elif isinstance(col, list) and col:
            result[category] = col[0]
    return result


def _buy_ratio_pct(ratings: dict) -> tuple[Optional[float], int]:
    """% of all analyst ratings that are "buy" or "strong buy", plus the
    total ratings count. See nearlow_screener._buy_ratio_pct for why
    numbers.Real is used instead of isinstance(v, (int, float))."""
    total = sum(v for v in ratings.values() if isinstance(v, numbers.Real))
    if not total:
        return None, 0
    buy_like = float(ratings.get("strongBuy", 0)) + float(ratings.get("buy", 0))
    return buy_like / total * 100.0, int(total)


def get_stock_snapshot(ticker: str) -> Optional[dict]:
    """The same "candidate" shape the growth/near-low screeners produce
    (price, 52-week range, analyst price target, ratings breakdown,
    market cap) but for one arbitrary ticker on demand -- not gated by
    any screen's filters or cached results. This is what lets a human
    type ANY ticker into the Stock Analysis tab and get it analyzed, not
    just names that happened to pass a screen. Returns None if the
    ticker doesn't resolve to real market data (bad symbol, delisted,
    etc.)."""
    try:
        t = yf.Ticker(ticker)
        fast_info = t.fast_info
        current_price = fast_info.last_price
        year_high = fast_info.year_high
        year_low = fast_info.year_low
        market_cap = fast_info.market_cap
    except Exception:
        return None
    if not current_price or not year_low:
        return None

    pct_from_52w_low = (current_price - year_low) / year_low * 100.0
    pct_from_52w_high = (current_price - year_high) / year_high * 100.0 if year_high else None

    ratings: dict = {}
    try:
        recommendations = t.get_recommendations_summary(as_dict=True) or {}
        ratings = _current_period_ratings(recommendations)
    except Exception:
        pass
    buy_ratio_pct, _ratings_count = _buy_ratio_pct(ratings)

    mean_target = None
    try:
        price_targets = t.get_analyst_price_targets() or {}
        mean_target = price_targets.get("mean")
    except Exception:
        pass
    target_upside_pct = (mean_target - current_price) / current_price * 100.0 if mean_target else None

    snapshot = {
        "ticker": ticker,
        "price": current_price,
        "year_low": year_low,
        "year_high": year_high,
        "pct_from_52w_low": pct_from_52w_low,
        "pct_from_52w_high": pct_from_52w_high,
        "target_mean": mean_target,
        "target_upside_pct": target_upside_pct,
        "analyst_ratings": ratings,
        "buy_ratio_pct": buy_ratio_pct,
        "market_cap": market_cap,
    }
    # Same round-trip as the screeners -- yfinance's dict conversions carry
    # numpy/pandas types that json.dumps chokes on downstream (jsonify).
    return json.loads(json.dumps(snapshot, default=str))


def find_quote(chain: dict, expiry: str, option_type: str, strike: float) -> Optional[Quote]:
    """Look up a specific contract's quote inside a pulled chain dict."""
    expiry_data = chain.get("expiries", {}).get(expiry)
    if not expiry_data:
        return None
    rows = expiry_data["calls"] if option_type.lower() == "call" else expiry_data["puts"]
    for row in rows:
        if abs(row["strike"] - strike) < 1e-6:
            return Quote(
                bid=row["bid"],
                ask=row["ask"],
                last=row["last"],
                iv=row["iv"],
                volume=row["volume"],
                open_interest=row["open_interest"],
            )
    return None
