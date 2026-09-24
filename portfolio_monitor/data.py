"""Live market data pulls via yfinance: spot prices, option chains, marks."""

from __future__ import annotations

import json
import math
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
    bars = []
    for idx, row in hist.iterrows():
        close = float(row["Close"])
        if math.isnan(close):
            continue
        bars.append({"time": idx.isoformat(), "close": close})
    return bars


def get_daily_price_history(ticker: str, period: str = "1y") -> list[dict]:
    """Daily close history, as [{"date": "YYYY-MM-DD", "close": float}, ...],
    sorted oldest-first (yfinance's own order). Used to place events like an
    analyst rating change on a timeline relative to the stock's own price
    action over the past year -- e.g. finding the date of its 52-week low,
    or the price on the day a given rating was issued -- rather than for
    charting, so daily granularity over a year is what's needed, not
    intraday bars. Returns [] rather than raising on an empty/failed pull,
    same convention as get_price_history. Bars with a NaN close (yfinance
    occasionally returns one, e.g. a halted or partial session) are
    dropped rather than passed through -- Python's json module emits a
    literal `NaN` token for float('nan'), which isn't valid JSON and
    breaks JSON.parse() in the browser once this reaches jsonify()."""
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period=period, interval="1d")
    except Exception:
        return []
    if hist.empty:
        return []
    bars = []
    for idx, row in hist.iterrows():
        close = float(row["Close"])
        if math.isnan(close):
            continue
        bars.append({"date": idx.strftime("%Y-%m-%d"), "close": close})
    return bars


def get_analyst_price_target_snapshot(ticker: str) -> Optional[dict]:
    """Live analyst price-target snapshot via yfinance's own free
    get_analyst_price_targets() -- the same call already used elsewhere
    in this app (get_stock_snapshot, the screeners) for target_mean/
    target_upside_pct, just exposed here in its own right for the rating
    chart's target-price star and the Price Target Analyst persona.
    Unlike FMP's free plan, yfinance has no per-symbol restriction here
    -- it works on small-caps and recent IPOs the same as on AAPL.
    `target_date` is this app's own estimate (today + 365 days, the
    conventional 12-month horizon), not something yfinance provides.
    Returns None if yfinance has no target data for this ticker."""
    try:
        pt = yf.Ticker(ticker).get_analyst_price_targets() or {}
    except Exception:
        return None

    def clean(value):
        if value is None:
            return None
        value = float(value)
        return None if math.isnan(value) else value

    mean = clean(pt.get("mean"))
    if mean is None:
        return None
    return {
        "target_high": clean(pt.get("high")),
        "target_low": clean(pt.get("low")),
        "target_mean": mean,
        "target_median": clean(pt.get("median")),
        "target_date": (datetime.now(timezone.utc) + timedelta(days=365)).date().isoformat(),
    }


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


def get_financial_highlights(ticker: str) -> Optional[dict]:
    """Real reported fundamentals from yfinance's own annual statements --
    income statement (revenue, net income, gross margin), balance sheet
    (cash, debt, shares outstanding), and cash flow (operating/free cash
    flow, used to estimate cash runway). Not an AI guess.

    The three statements are fetched and can fail independently -- a
    pre-revenue biotech or other early-stage company routinely has cash/
    debt/shares data via the balance sheet even when its income statement
    shows nothing worth reporting, and for exactly that kind of ticker
    cash runway and share dilution matter far more than revenue. Returns
    None only if ALL three statements are unavailable/empty (some tickers
    -- very new listings, some non-US filers -- don't have any of them
    via yfinance).

    Reads each DataFrame directly (not via as_dict=True) and converts
    fiscal-year columns to plain strings itself -- to_dict() on a frame
    with a DatetimeIndex/columns can leave pandas Timestamp objects as
    dict keys, which json.dumps refuses to serialize even with a
    default= fallback (default only rescues values, never keys)."""
    t = yf.Ticker(ticker)

    def _fetch(method_name):
        try:
            df = getattr(t, method_name)(freq="yearly")
        except Exception:
            return None
        return df if df is not None and not df.empty and len(df.columns) > 0 else None

    income = _fetch("get_income_stmt")
    balance = _fetch("get_balance_sheet")
    cashflow = _fetch("get_cashflow")
    if income is None and balance is None and cashflow is None:
        return None

    def _cols(df):
        return sorted(df.columns, reverse=True) if df is not None else []

    def _value(df, labels, col):
        """Tries each label in turn (yfinance's exact row names can vary
        by ticker/version) and returns the first real (non-NaN) match."""
        if df is None or col is None:
            return None
        for label in labels:
            if label not in df.index:
                continue
            v = df.loc[label, col]
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v == v:  # filter NaN
                return v
        return None

    income_cols, balance_cols, cashflow_cols = _cols(income), _cols(balance), _cols(cashflow)
    income_col = income_cols[0] if income_cols else None
    income_prior_col = income_cols[1] if len(income_cols) > 1 else None
    balance_col = balance_cols[0] if balance_cols else None
    balance_prior_col = balance_cols[1] if len(balance_cols) > 1 else None
    cashflow_col = cashflow_cols[0] if cashflow_cols else None

    fiscal_year_end = (income_col or balance_col or cashflow_col)

    latest_revenue = _value(income, ["Total Revenue"], income_col)
    prior_revenue = _value(income, ["Total Revenue"], income_prior_col)
    latest_net_income = _value(income, ["Net Income"], income_col)
    latest_gross_profit = _value(income, ["Gross Profit"], income_col)

    cash = _value(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], balance_col)
    total_debt = _value(balance, ["Total Debt"], balance_col)
    shares_outstanding = _value(balance, ["Ordinary Shares Number", "Share Issued"], balance_col)
    prior_shares_outstanding = _value(balance, ["Ordinary Shares Number", "Share Issued"], balance_prior_col)

    operating_cash_flow = _value(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], cashflow_col)
    free_cash_flow = _value(cashflow, ["Free Cash Flow"], cashflow_col)
    # Quarterly burn from whichever annual cash-flow figure is available --
    # only meaningful when the company is actually burning cash (negative).
    annual_burn = free_cash_flow if free_cash_flow is not None else operating_cash_flow
    quarterly_burn = -annual_burn / 4.0 if annual_burn is not None and annual_burn < 0 else None

    highlights = {
        "fiscal_year_end": fiscal_year_end.strftime("%Y-%m-%d") if fiscal_year_end is not None else None,
        "revenue": latest_revenue,
        "revenue_yoy_pct": (
            (latest_revenue - prior_revenue) / prior_revenue * 100.0
            if latest_revenue is not None and prior_revenue
            else None
        ),
        "net_income": latest_net_income,
        "gross_margin_pct": (
            latest_gross_profit / latest_revenue * 100.0
            if latest_gross_profit is not None and latest_revenue
            else None
        ),
        "cash": cash,
        "total_debt": total_debt,
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
        "cash_runway_quarters": (cash / quarterly_burn) if cash is not None and quarterly_burn else None,
        "shares_outstanding": shares_outstanding,
        "shares_outstanding_yoy_pct": (
            (shares_outstanding - prior_shares_outstanding) / prior_shares_outstanding * 100.0
            if shares_outstanding is not None and prior_shares_outstanding
            else None
        ),
    }
    return json.loads(json.dumps(highlights, default=str))


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
