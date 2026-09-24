"""On-demand AI stock analysis: a single ~200-word expert take with a
buy-opportunity verdict, and a roster of independent single-focus analyst
inquiries a human can pick from -- technical (real computed indicators),
fundamental (real financial-statement highlights), news, SEC filings,
analyst-ratings-timing, macro/risk, and social-media sentiment. Used both
by the Near 52W Low screen (one of its candidates) and by the free-text
Stock Analysis tab (any ticker a human types in) -- the prompts here
don't assume the stock is currently near its low, only that its 52-week
range is known.

Each selected persona is a genuinely separate call, given only the narrow
slice of data its own question needs -- not one prompt told "you are N
agents on a panel" with the full context repeated for every persona.
Everything here is informational only -- never a buy/sell/hold
instruction -- and only runs when a human clicks a button for one
specific ticker and picks which agents to run; nothing here runs
automatically across a whole screen.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd
import yfinance as yf

from . import ai_client
from . import data as data_mod


def _closest_daily_close_on_or_before(daily_history: list[dict], target_date: str) -> Optional[dict]:
    """The daily bar for target_date itself, or the closest earlier one --
    a rating can land on a day the market's closed, and it should still
    get "the price around that time" rather than being dropped. Assumes
    daily_history is sorted oldest-first."""
    candidates = [d for d in daily_history if d["date"] <= target_date]
    return candidates[-1] if candidates else None


def get_rating_timeline(ticker: str, daily_history: Optional[list[dict]] = None) -> dict:
    """Recent analyst rating changes (from yfinance's upgrade/downgrade
    history) annotated with the stock's own price at the time of each one,
    plus the date of its own 52-week low from a year of daily bars -- so a
    "ratings timing" read can say whether a given rating predates the
    slide to the low (and so may be stale) or postdates it (i.e. the
    analyst kept/gave that rating knowing the current price). Pass
    `daily_history` if the caller already fetched it (e.g. for the
    technical persona) to avoid pulling it twice. Never raises -- a
    failed pull just means an empty timeline, which callers (and the AI
    prompt) should treat as "can't be assessed" rather than an error."""
    if daily_history is None:
        daily_history = data_mod.get_daily_price_history(ticker)
    daily_history = sorted(daily_history, key=lambda d: d["date"])

    week_52_low = None
    if daily_history:
        low_bar = min(daily_history, key=lambda d: d["close"])
        week_52_low = {"date": low_bar["date"], "close": low_bar["close"]}

    try:
        df = yf.Ticker(ticker).get_upgrades_downgrades(as_dict=False)
    except Exception:
        df = None

    actions = []
    if df is not None and not df.empty:
        for grade_date, row in df.iterrows():
            grade_date_str = grade_date.strftime("%Y-%m-%d")
            action = {
                "date": grade_date_str,
                "firm": row.get("Firm"),
                "to_grade": row.get("ToGrade"),
                "from_grade": row.get("FromGrade"),
                "action": row.get("Action"),
            }
            if week_52_low:
                action["before_52w_low"] = grade_date_str < week_52_low["date"]
            price_then = _closest_daily_close_on_or_before(daily_history, grade_date_str)
            if price_then and daily_history:
                action["price_at_rating"] = price_then["close"]
                latest_close = daily_history[-1]["close"]
                action["pct_move_since_rating"] = (latest_close - price_then["close"]) / price_then["close"] * 100.0
                if week_52_low:
                    # How far above what would LATER become the 52-week low
                    # the price already was when this rating was issued --
                    # e.g. "only 10% above" means it was issued into an
                    # already-weak, sliding price; "50% above" means it was
                    # issued while the stock was still performing well,
                    # well before the decline that followed.
                    action["pct_above_low_at_rating"] = (
                        (price_then["close"] - week_52_low["close"]) / week_52_low["close"] * 100.0
                    )
            actions.append(action)

    actions.sort(key=lambda a: a["date"], reverse=True)
    # Cap to the most recent 10 -- older coverage history doesn't help
    # judge whether TODAY's ratings are stale, and would just bloat the
    # prompt for names with a long analyst history.
    actions = actions[:10]
    return json.loads(json.dumps({"week_52_low": week_52_low, "actions": actions}, default=str))


def get_institutional_activity(ticker: str, daily_history: Optional[list[dict]] = None) -> dict:
    """Top institutional holders (yfinance's own institutional-holders
    data, sourced from 13F filings) and whether each added to, trimmed, or
    newly opened their position -- the closest free equivalent to "hedge
    fund decisions". Unlike analyst ratings, 13F filings are quarterly
    with a ~45-day reporting lag, so most rows here share the same recent
    report date rather than being spread across the year the way rating
    changes are; still annotated with the stock's own price on that date
    so it can plot alongside the rating timeline. Also note "institutional
    holders" is Yahoo's own broader category (asset managers, pensions,
    banks, etc, alongside hedge funds proper) -- there's no free source
    that isolates hedge funds specifically. Never raises -- a failed pull
    just means an empty list."""
    if daily_history is None:
        daily_history = data_mod.get_daily_price_history(ticker)
    daily_history = sorted(daily_history, key=lambda d: d["date"])

    try:
        df = yf.Ticker(ticker).get_institutional_holders(as_dict=False)
    except Exception:
        df = None

    holders = []
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            date_reported = row.get("Date Reported")
            date_str = date_reported.strftime("%Y-%m-%d") if hasattr(date_reported, "strftime") else str(date_reported)
            pct_change = row.get("pctChange")
            if pd.isna(pct_change):
                pct_change = None
            if pct_change is None:
                direction = "new"
            elif pct_change > 0:
                direction = "increased"
            elif pct_change < 0:
                direction = "decreased"
            else:
                direction = "unchanged"
            holder_name = row.get("Holder")
            if isinstance(holder_name, str):
                # Yahoo's own institutional-holders data has inconsistent
                # trailing whitespace on some organization names (e.g.
                # "Nvidia Corp " vs "Alyeska Investment Group, L.p.") --
                # strip it so it doesn't show up as a stray space before
                # the colon in the legend.
                holder_name = holder_name.strip()
            holder = {
                "date": date_str,
                "holder": holder_name,
                "shares": row.get("Shares"),
                "value": row.get("Value"),
                "pct_held": row.get("pctHeld"),
                "pct_change": pct_change,
                "direction": direction,
            }
            price_then = _closest_daily_close_on_or_before(daily_history, date_str)
            if price_then:
                holder["price_at_report"] = price_then["close"]
            holders.append(holder)

    holders.sort(key=lambda h: h["date"], reverse=True)
    return json.loads(json.dumps({"holders": holders}, default=str))


def _timing_label(action: dict) -> str:
    before = action.get("before_52w_low")
    if before is True:
        return "BEFORE"
    if before is False:
        return "AFTER"
    return "unknown timing vs."


def _format_rating_action_line(a: dict) -> str:
    """One rating change as a plain factual line -- states the raw %
    above the eventual 52-week low at issuance (e.g. "was 10% above what
    would become its 52-week low") rather than pre-labeling it "stale" or
    "conviction" here; that judgment call is exactly what the AI reading
    this is asked to make."""
    price_note = f", price then ~{a['price_at_rating']:.2f}" if a.get("price_at_rating") is not None else ""
    above_low_note = (
        f", was {a['pct_above_low_at_rating']:.0f}% above what would become its 52-week low when issued"
        if a.get("pct_above_low_at_rating") is not None
        else ""
    )
    move_note = (
        f", stock has moved {a['pct_move_since_rating']:+.1f}% since"
        if a.get("pct_move_since_rating") is not None
        else ""
    )
    return (
        f"- [{a.get('date')}] {a.get('firm')}: {a.get('action')} "
        f"({a.get('from_grade')} -> {a.get('to_grade')}) -- {_timing_label(a)} the 52-week low"
        f"{price_note}{above_low_note}{move_note}"
    )


def build_context_user_message(ticker: str, context: dict) -> str:
    """Full shared context for the single ~200-word expert take -- unlike
    the single-focus personas below, this one synthesizes everything at
    once, so it gets the full picture (not the newer filings/financials/
    social-sentiment additions, to keep this specific prompt's scope and
    length stable)."""
    candidate = context["candidate"]
    rating_timeline = context.get("rating_timeline") or {}
    headlines = context.get("headlines") or []
    macro_score = context.get("macro_score")

    lines = [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"Current price: {candidate.get('price')}",
        f"52-week range: low {candidate.get('year_low')}, high {candidate.get('year_high')}",
        f"% above 52-week low: {candidate.get('pct_from_52w_low')}",
        f"Analyst mean price target: {candidate.get('target_mean')} "
        f"({candidate.get('target_upside_pct')}% upside from current price)",
        f"Current analyst ratings breakdown: {candidate.get('analyst_ratings')} "
        f"({candidate.get('buy_ratio_pct')}% buy/strong-buy)",
        f"Market cap: {candidate.get('market_cap')}",
        "",
    ]

    week_52_low = rating_timeline.get("week_52_low")
    if week_52_low:
        lines.append(f"Date of the stock's own 52-week low: {week_52_low['date']} (close {week_52_low['close']})")
    else:
        lines.append("Date of the 52-week low: unknown (price history unavailable)")

    lines.append("")
    lines.append("Recent analyst rating changes (most recent first):")
    actions = rating_timeline.get("actions") or []
    if actions:
        lines.extend(_format_rating_action_line(a) for a in actions)
    else:
        lines.append("(none found)")

    lines.append("")
    lines.append("Recent headlines:")
    if headlines:
        for h in headlines:
            when = h["published_at"].isoformat() if h.get("published_at") else "unknown date"
            lines.append(f"- [{when}] {h['title']}")
    else:
        lines.append("(none found in the lookback window)")

    lines.append("")
    if macro_score is not None:
        lines.append(f"Current macro gate score: {macro_score:.1f}/100 (higher = calmer environment)")
    else:
        lines.append("Macro gate score: not available")

    return "\n".join(lines)


# --- Single ~200-word expert take -------------------------------------------


EXPERT_SYSTEM_PROMPT = """You are a seasoned equity research analyst embedded \
in a portfolio monitoring tool. You are given one stock: its price/52-week- \
range data, current analyst consensus, recent analyst rating CHANGES with \
their dates and the stock's price at each one, recent headlines, and the \
current macro environment. Your job is ONLY to inform, never to advise.

Write ONE analysis of about 200 words (180-220 is fine) covering:
- Why the stock is trading where it is in its 52-week range, based on the \
headlines/data given.
- Whether the current analyst consensus and any recent rating changes still \
look credible given how much time has passed and how the price has moved \
since. Use the "% above what would become its 52-week low when issued" \
figure given for each rating: a Buy issued when the price was already close \
to that eventual low was made despite visible weakness (a stronger, more \
current signal); a Buy issued while the price was still well above that \
low (i.e. the stock was performing well at the time) predates the decline \
and may simply be stale.
- The strongest case FOR this being a buy opportunity, and the strongest \
case AGAINST, laid out side by side rather than staked as a single opinion.

Then classify your own analysis with "verdict": "buy_opportunity" (the case \
for clearly outweighs the case against), "not_a_buy" (the case against \
clearly wins), or "mixed" (genuinely too close to call). This is a label on \
your own analysis for the UI to display, not investment advice -- you must \
never tell the user to actually buy, sell, or hold.

Respond with ONLY a JSON object, no other text, matching exactly this shape:
{"analysis": "...", "verdict": "buy_opportunity|not_a_buy|mixed", \
"disclaimer": "This is not investment advice."}
"""


def _parse_expert_json(text: str) -> dict:
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
                "analysis": text.strip()[:1200] or None,
                "verdict": "mixed",
                "disclaimer": "This is not investment advice.",
                "parse_error": True,
            }
    parsed.setdefault("verdict", "mixed")
    parsed.setdefault("disclaimer", "This is not investment advice.")
    parsed.setdefault("parse_error", False)
    return parsed


def analyze_expert_take(
    ticker: str,
    context: dict,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> dict:
    """Raises on API failure -- the caller decides how to degrade."""
    user_message = build_context_user_message(ticker, context)
    text = ai_client.call_provider(provider, EXPERT_SYSTEM_PROMPT, user_message, model, api_key=api_key, max_tokens=700)
    return _parse_expert_json(text)


# --- On-demand AI pre-revenue check (screener candidates) -------------------
#
# The screeners' own is_pre_revenue field is a cheap numeric heuristic off
# yfinance's own revenue data, and it's been wrong in both directions in
# live use: a real revenue-generating health insurer (whose income
# statement's revenue line didn't match any label checked) got flagged
# pre-revenue, and a real pre-revenue nuclear-technology startup (whose
# yfinance data looked, incorrectly, like it had revenue) didn't. This is
# a separate, small, cheap AI call a human explicitly triggers (a "Check
# pre-revenue" button, never run automatically as part of a screen) that
# uses the model's own general knowledge of the company alongside
# whatever revenue figure yfinance has, rather than trusting that figure
# blindly.

PRE_REVENUE_CLASSIFIER_SYSTEM_PROMPT = """You are a financial classifier. Decide whether the given company is \
"pre-revenue" -- i.e. it has not yet generated meaningful, ongoing \
commercial revenue from its core business (common for clinical-stage \
biotech, pre-commercial energy/technology startups, and other early-stage \
companies still in development or regulatory approval). You are given \
whatever revenue figure yfinance has on file, which is sometimes missing, \
incomplete, or simply wrong for early-stage or non-standard-industry \
companies -- use your own general knowledge of the company and what it \
actually does/sells, especially when the given data is missing or seems \
inconsistent with what you know. Don't default to "pre-revenue" just \
because a number is missing, and don't default to "not pre-revenue" just \
because a small/incidental figure is present (e.g. interest income, a \
one-off pilot contract, grant funding) -- decide based on whether there is \
REAL, ongoing commercial revenue from the company's core business. If you \
simply don't know the company and no usable revenue figure was given, say \
so honestly in "reason" and answer your best guess.

Respond with ONLY a JSON object, no other text: \
{"is_pre_revenue": true or false, "reason": "one short sentence"}"""


def classify_pre_revenue(
    ticker: str,
    name: Optional[str],
    financials: Optional[dict],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> dict:
    """Raises on API failure -- the caller decides how to degrade (and,
    since this typically runs across many candidates at once, isolates
    one ticker's failure from the rest)."""
    revenue = (financials or {}).get("revenue") if financials else None
    if revenue is not None:
        revenue_line = f"yfinance's reported revenue (most recent fiscal year): {revenue:,.0f}"
    else:
        revenue_line = "yfinance has no usable revenue figure on file for this ticker."
    user_message = f"Ticker: {ticker} ({name or 'n/a'})\n{revenue_line}"
    text = ai_client.call_provider(
        provider, PRE_REVENUE_CLASSIFIER_SYSTEM_PROMPT, user_message, model, api_key=api_key, max_tokens=150
    )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
    is_pre_revenue = parsed.get("is_pre_revenue")
    return {
        "is_pre_revenue": bool(is_pre_revenue) if isinstance(is_pre_revenue, bool) else None,
        "reason": parsed.get("reason"),
    }


# --- Selectable independent single-focus analyst inquiries ------------------
#
# Each is a separate, independent call -- every persona gets its OWN
# narrow slice of `context` (see PANEL_CONTEXT_BUILDERS) and answers its
# own question with no idea any other persona exists. A human picks which
# of these actually run (see run_expert_panel's `selected_personas`);
# nothing here assumes all of them are selected together.


def _price_range_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    lines = [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"Current price: {candidate.get('price')}",
        f"52-week range: low {candidate.get('year_low')}, high {candidate.get('year_high')}",
        f"% above 52-week low: {candidate.get('pct_from_52w_low')}",
    ]
    week_52_low = (context.get("rating_timeline") or {}).get("week_52_low")
    if week_52_low:
        lines.append(f"Date of the stock's own 52-week low: {week_52_low['date']} (close {week_52_low['close']})")
    else:
        lines.append("Date of the 52-week low: unknown (price history unavailable)")

    indicators = context.get("technical_indicators") or {}
    lines.append("")
    if indicators:
        lines.append("Computed technical indicators (real, from daily closes):")
        for period in (20, 50, 200):
            sma = indicators.get(f"sma_{period}")
            if sma is not None:
                pct = indicators.get(f"price_vs_sma_{period}_pct")
                pct_note = f" (price is {pct:+.1f}% vs this average)" if pct is not None else ""
                lines.append(f"- SMA({period}): {sma:.2f}{pct_note}")
        if "rsi_14" in indicators:
            lines.append(f"- RSI(14): {indicators['rsi_14']:.1f} (>70 typically overbought, <30 oversold)")
        if "macd" in indicators:
            lines.append(
                f"- MACD: {indicators['macd']:.3f}, signal {indicators['macd_signal']:.3f}, "
                f"histogram {indicators['macd_histogram']:+.3f}"
            )
    else:
        lines.append("Computed technical indicators: not enough price history available.")
    return lines


def _fundamental_lines(ticker: str, context: dict) -> list[str]:
    """Deliberately excludes the analyst mean target/ratings that other
    context builders here include -- this persona's whole point is an
    independent read of the actual business, and handing it the target/
    rating consensus just invited it to reason from "analysts say X%
    upside" instead of the balance sheet (this is exactly what happened
    in practice: a pre-revenue biotech with no income-statement data got
    called "significantly undervalued" purely off its price target)."""
    candidate = context["candidate"]
    lines = [f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})", f"Market cap: {candidate.get('market_cap')}", ""]
    financials = context.get("financials")
    if not financials:
        lines.append("Financial statements: not available for this ticker.")
        return lines

    lines.append(f"Latest annual financials (fiscal year end {financials.get('fiscal_year_end')}):")
    revenue = financials.get("revenue")
    if revenue is not None and revenue > 0:
        yoy = financials.get("revenue_yoy_pct")
        yoy_note = f", {yoy:+.1f}% YoY" if yoy is not None else ""
        lines.append(f"- Revenue: {revenue:,.0f}{yoy_note}")
    elif revenue == 0:
        # A CONFIRMED $0 on the income statement -- this is the only case
        # that actually justifies calling the company pre-revenue.
        lines.append("- Revenue: $0 reported (confirmed pre-revenue).")
    else:
        # revenue is None: the income statement didn't have a usable
        # revenue line at all (missing statement, unmatched label, NaN).
        # This is a DATA GAP, not evidence of being pre-revenue -- a real,
        # revenue-generating company can hit this if yfinance's line-item
        # labels don't match for its industry (observed with an insurer).
        lines.append("- Revenue: not available (income statement data incomplete for this ticker -- this does NOT by itself mean pre-revenue).")
    if financials.get("net_income") is not None:
        lines.append(f"- Net income: {financials['net_income']:,.0f}")
    if financials.get("gross_margin_pct") is not None:
        lines.append(f"- Gross margin: {financials['gross_margin_pct']:.1f}%")
    if financials.get("cash") is not None:
        lines.append(f"- Cash & equivalents: {financials['cash']:,.0f}")
    if financials.get("total_debt") is not None:
        lines.append(f"- Total debt: {financials['total_debt']:,.0f}")
    if financials.get("operating_cash_flow") is not None:
        lines.append(f"- Operating cash flow (annual): {financials['operating_cash_flow']:,.0f}")
    if financials.get("free_cash_flow") is not None:
        lines.append(f"- Free cash flow (annual): {financials['free_cash_flow']:,.0f}")
    if financials.get("cash_runway_quarters") is not None:
        lines.append(f"- Estimated cash runway at current burn rate: {financials['cash_runway_quarters']:.1f} quarters")
    if financials.get("shares_outstanding") is not None:
        change = financials.get("shares_outstanding_yoy_pct")
        change_note = f", {change:+.1f}% YoY" if change is not None else ""
        lines.append(f"- Shares outstanding: {financials['shares_outstanding']:,.0f}{change_note}")

    peer_comparison = context.get("peer_comparison")
    lines.append("")
    if not peer_comparison:
        lines.append("Peer/industry P/E comparison: not available for this ticker.")
        return lines

    lines.append(f"Valuation vs. industry ({peer_comparison.get('industry')}):")
    target_pe = peer_comparison.get("target_pe")
    lines.append(f"- This company's trailing P/E: {target_pe:.1f}" if target_pe is not None else "- This company's trailing P/E: not available (likely unprofitable -- P/E is meaningless for a company with no earnings).")
    peer_avg_pe = peer_comparison.get("peer_avg_pe")
    if peer_avg_pe is not None:
        lines.append(f"- Average trailing P/E of the {len(peer_comparison.get('peers') or [])} largest same-industry peers below: {peer_avg_pe:.1f}")
    peers = peer_comparison.get("peers") or []
    if peers:
        lines.append("- Largest same-industry peers by market cap (yfinance's industry classification, not a curated competitor list):")
        for p in peers:
            pe_str = f"P/E {p['pe']:.1f}" if p.get("pe") is not None else "P/E n/a"
            lines.append(f"  - {p.get('ticker')} ({p.get('name') or 'n/a'}): {pe_str}")
    else:
        lines.append("- No same-industry peers with usable data found.")
    return lines


def _headline_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    headlines = context.get("headlines") or []
    lines = [f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})", "", "Recent headlines:"]
    if headlines:
        for h in headlines:
            when = h["published_at"].isoformat() if h.get("published_at") else "unknown date"
            lines.append(f"- [{when}] {h['title']}")
    else:
        lines.append("(none found in the lookback window)")
    return lines


def _rating_timeline_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    rating_timeline = context.get("rating_timeline") or {}
    lines = [f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})", ""]
    week_52_low = rating_timeline.get("week_52_low")
    if week_52_low:
        lines.append(f"Date of the stock's own 52-week low: {week_52_low['date']} (close {week_52_low['close']})")
    else:
        lines.append("Date of the 52-week low: unknown (price history unavailable)")
    lines.append("")
    lines.append("Recent analyst rating changes (most recent first):")
    actions = rating_timeline.get("actions") or []
    if actions:
        lines.extend(_format_rating_action_line(a) for a in actions)
    else:
        lines.append("(none found)")
    return lines


def _macro_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    macro_score = context.get("macro_score")
    lines = [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"% above its 52-week low: {candidate.get('pct_from_52w_low')}",
    ]
    if macro_score is not None:
        lines.append(f"Current macro gate score: {macro_score:.1f}/100 (higher = calmer environment)")
    else:
        lines.append("Macro gate score: not available")
    return lines


def _filings_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    filings = context.get("filings") or []
    lines = [f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})", "", "Recent SEC filings (most recent first):"]
    if filings:
        for f in filings:
            lines.append(
                f"- [{f.get('filed')}] {f.get('form')} (report date {f.get('report_date') or 'n/a'}): {f.get('url')}"
            )
    else:
        lines.append("(none found -- ticker may not resolve on SEC EDGAR, e.g. a non-US filer, or SEC's API was unreachable)")
    return lines


_WINDOW_PHRASE = {
    "hour": "in the past hour",
    "today": "today",
    "week": "in the past week",
    "month": "in the past month",
    "year": "in the past year",
}

# StockTwits pagination can return well over 100 raw messages for a busy
# ticker -- cap how many get spelled out in the prompt (the totals line
# below still reflects every message, so the bull/bear split is never lossy).
_MAX_SOCIAL_MESSAGES_IN_PROMPT = 60


def _social_sentiment_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    window = context.get("social_sentiment_window") or "today"
    window_phrase = _WINDOW_PHRASE.get(window, f"in the past {window}")
    messages = context.get("social_sentiment")
    lines = [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"Recent StockTwits messages mentioning this ticker {window_phrase} "
        f"(each tagged Bullish/Bearish by the trader who posted it, or untagged):",
    ]
    if messages:
        bullish = sum(1 for m in messages if m.get("sentiment") == "Bullish")
        bearish = sum(1 for m in messages if m.get("sentiment") == "Bearish")
        untagged = len(messages) - bullish - bearish
        lines.append(f"Totals across all {len(messages)} messages: {bullish} Bullish, {bearish} Bearish, {untagged} untagged.")
        shown = messages[:_MAX_SOCIAL_MESSAGES_IN_PROMPT]
        for m in shown:
            tag = m.get("sentiment") or "untagged"
            lines.append(f"- [{m.get('created_at')}] @{m.get('username')} ({tag}, {m.get('likes')} likes): \"{m.get('body')}\"")
        if len(messages) > len(shown):
            lines.append(f"...({len(messages) - len(shown)} more messages omitted here for brevity; the totals above already include them)")
    else:
        lines.append("(no messages found for this window)")
    return lines


def _price_target_lines(ticker: str, context: dict) -> list[str]:
    candidate = context["candidate"]
    snapshot = context.get("price_targets")
    lines = [f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})", ""]
    if not snapshot:
        lines.append("(no analyst price-target data found for this ticker)")
        return lines

    lines.append(
        "Live analyst price-target snapshot (NOT a history of individual ratings -- "
        "just today's aggregate across all covering analysts, via yfinance). "
        "target_date is this app's own estimate -- today + 12 months, the conventional "
        "Wall Street horizon -- not a date any firm itself stated:"
    )
    lines.append(
        f"Mean target: ${snapshot.get('target_mean')} "
        f"(median ${snapshot.get('target_median')}, range ${snapshot.get('target_low')}-"
        f"${snapshot.get('target_high')}), expected to play out by ~{snapshot.get('target_date')}"
    )
    return lines


PANEL_CONTEXT_BUILDERS = {
    "technical": _price_range_lines,
    "fundamental": _fundamental_lines,
    "news": _headline_lines,
    "ratings_timing": _rating_timeline_lines,
    "macro_risk": _macro_lines,
    "filings": _filings_lines,
    "social_sentiment": _social_sentiment_lines,
    "price_targets": _price_target_lines,
}


PANEL_SYSTEM_PROMPTS = {
    "technical": (
        "Technical Analyst",
        """You are a technical analyst. You're asked to independently assess one \
stock given its current price, 52-week high/low, the date of its own \
52-week low, and (when there's enough price history) real computed \
indicators: SMA(20/50/200), RSI(14), and MACD. Give a short (60-100 word) \
technical read using those actual numbers where given -- where the price \
sits relative to its moving averages, whether RSI suggests overbought/ \
oversold, what MACD suggests about momentum -- and what would change your \
mind either way. If indicators weren't available, say so rather than \
inventing them. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "fundamental": (
        "Fundamental Analyst",
        """You are a fundamental analyst. Assess this company using ONLY its \
own reported financial statements and valuation data given below: revenue \
and its growth trend, profitability (net income, gross margin), cash and \
debt on the balance sheet, operating/free cash flow, the estimated cash \
runway at the current burn rate, the share-count trend (rising share \
count signals dilution), and trailing P/E versus the average P/E of the \
largest same-industry peers also given. You do NOT have access to analyst \
price targets or ratings, and must not reason from them, guess at them, \
or mention them -- that is a different analyst's job on this panel; yours \
is the underlying business and its valuation against real peers, not what \
Wall Street's target price implies.

Revenue reported as a confirmed $0 means the company is genuinely \
pre-revenue -- treat that as a real data point about the business stage, \
say so explicitly, and weigh cash runway and dilution more heavily than \
revenue for it: state the actual cash runway figure if given (e.g. "~6 \
quarters of runway at the current burn rate") rather than vaguely saying \
runway "should be assessed" -- you have the number, use it. You do not \
know and must not guess a specific revenue-generation date (no earnings \
guidance is given to you) -- instead say plainly what the runway implies \
(e.g. whether it likely covers any near-term catalysts) and what would \
need to change (financing, cost cuts, a revenue inflection) to extend it. \
Revenue reported as "not available" is a different thing: a DATA GAP, not \
evidence of being pre-revenue -- a real, revenue-generating company can \
show this if the statement's line-item labels didn't match. Never call a \
company pre-revenue on the basis of missing data alone.

For P/E: if this company's own trailing P/E and the peer average are both \
given, say explicitly whether it trades at a premium or discount to \
peers and by roughly how much -- naming at least one or two of the peer \
tickers. If this company's P/E isn't available (typically because it's \
unprofitable), say so rather than comparing anyway.

Give a short (70-110 word) fundamental-only read covering both the \
balance-sheet health and the valuation-vs-peers comparison, and what \
would need to be true of the BUSINESS ITSELF (e.g. a revenue inflection, \
an extended runway, improving margins, a re-rating toward peer multiples) \
to strengthen it. If no financial statements are available at all, say so \
plainly and use "neutral" -- don't invent figures or fall back on price/ \
valuation-target reasoning when there is nothing fundamental to go on. \
Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "news": (
        "News Analyst",
        """You are a news analyst. You're asked to independently assess one \
stock given only its recent headlines. Give a short (60-100 word) read on \
what the news flow suggests is driving the price action, and whether the \
tone looks like it's improving, stabilizing, or deteriorating. If no \
headlines were given, say so plainly rather than guessing. Never say to \
buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "ratings_timing": (
        "Analyst-Ratings Auditor",
        """You are an analyst-ratings auditor. Your ONLY job is to independently \
check the TIMING of recent analyst rating changes for one stock against the \
stock's own price action -- nothing else about this stock is in scope for \
you. For each recent rating change you are given \
its date, the stock's price on/near that date, how far above what LATER \
became the stock's 52-week low that price already was at the time (e.g. \
"was 10% above what would become its 52-week low when issued"), whether the \
rating fell before or after the date of the 52-week low itself, and the % \
the price has moved since that rating.

Use the "how far above the eventual low" figure to judge WHAT THE ANALYST \
WAS SEEING when they made the call, not just whether the calendar date was \
before or after the low:
- If a Buy was issued when the price was ALREADY close to what became its \
52-week low (a small % above it), the analyst was looking at a stock \
already sliding/weak and chose to call it a buy anyway -- that's a real \
conviction call made despite visible weakness, and a fresher, more \
relevant signal today.
- If a Buy was issued while the price was still well above what became its \
52-week low (a large % above it, i.e. the stock was performing well at the \
time), the analyst had not yet seen the decline that followed -- that \
rating may simply be stale rather than a considered view of today's \
weakness.
- A rating reaffirmed or issued AFTER the low (or a recent downgrade) \
generally carries more current information than one from well before the \
decline.

Name the specific firm(s) and date(s) you're weighing, and state plainly \
how far above the eventual low the price was when each one you cite was \
issued. If no rating-change data was given, say so plainly rather than \
guessing. Give a short (80-120 word) take. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "macro_risk": (
        "Macro & Risk Analyst",
        """You are a macro/risk analyst. You're asked to independently assess one \
stock given only how far above its 52-week low it currently sits and the \
current macro "calm" score (0-100, higher = calmer) -- no company-specific \
fundamentals or news are in scope for you. Give a short (60-100 word) \
devil's-advocate read: what macro-level factors (rate environment, sector \
rotation, liquidity) could move a stock like this regardless of company- \
specifics, and whether the current macro backdrop supports or argues \
against adding risk to a name like this right now. Never say to buy, \
sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "filings": (
        "SEC Filings Analyst",
        """You are a filings analyst. You're asked to independently assess one \
stock given only a list of its most recent SEC filings (form type, filing \
date, and a link) from EDGAR. Give a short (60-100 word) read on what the \
recent filing activity suggests -- e.g. a fresh 10-K/10-Q means updated \
financials just became public, an 8-K often flags a material event worth \
checking, a cluster of filings in a short window can signal something in \
motion. You do NOT have the actual filing CONTENTS, only the list of what \
was filed and when -- be explicit about that limit, and suggest what a \
human should go read rather than inventing what's inside any filing. \
Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "social_sentiment": (
        "Social Sentiment Analyst",
        """You are a social-media sentiment analyst. You're asked to \
independently assess one stock given a totals summary (how many recent \
StockTwits messages were Bullish/Bearish/untagged) plus a sample list of \
the actual messages mentioning it, from a specific recent time window. \
Each message gives you who posted it, when, how many likes it got, its \
text, and -- importantly -- a Bullish/Bearish tag the POSTER THEMSELVES \
chose when writing it (or untagged, if they didn't pick one). Give a \
short (60-100 word) read on what these messages suggest about retail \
sentiment and interest right now: weigh the self-tagged Bullish/Bearish \
split directly, using the totals line as the authoritative count even if \
only a sample of messages is listed below it (don't just infer tone from \
wording when an explicit tag is given), and note rising or falling \
attention. If no messages were given (nothing posted in this window), \
say so plainly rather than guessing. Retail social sentiment is noisy \
and often contrarian -- do not treat volume or tag mix alone as a signal \
of where the stock is headed. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "price_targets": (
        "Price Target Analyst",
        """You are a price-target analyst. You're asked to independently assess \
one stock given a LIVE snapshot of analyst price targets -- NOT a history of \
individual ratings: today's mean/median/high/low target across all covering \
analysts, and a target_date (this app's own estimate -- today + 12 months, \
the conventional Wall Street horizon, NOT something any firm itself stated). \
Give a short (60-100 word) read on what the current mean target implies \
versus the stock's price, and how wide or narrow the high-low range is \
(a wide range means analysts disagree a lot about where this is headed). \
If no price-target data was given, say so plainly rather than inventing \
figures. A mean target is an aggregate opinion, not a guarantee -- do not \
treat it as a prediction that will necessarily come true. Never say to buy, \
sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
}


def _parse_panel_agent_json(text: str) -> dict:
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
            parsed = {"take": text.strip()[:800] or None}
    parsed.setdefault("stance", "neutral")
    return parsed


def run_expert_panel(
    ticker: str,
    context: dict,
    selected_personas: list[str],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Run each persona in `selected_personas` (in the order given), one
    at a time -- every persona gets ONLY the data relevant to its own
    question (see PANEL_CONTEXT_BUILDERS), not the full shared context,
    so each is a genuinely separate call. Unknown persona keys are
    silently skipped (lets the frontend send whatever it has checked
    without the backend needing to validate against a hardcoded list).
    Each persona's own failure (rate limit, transient network error) is
    isolated to its own entry rather than aborting the rest -- a partial
    result with N-1 good takes and one clearly-marked error is more
    useful than losing all of them over one bad call."""
    results = []
    for key in selected_personas:
        if key not in PANEL_SYSTEM_PROMPTS:
            continue
        label, system_prompt = PANEL_SYSTEM_PROMPTS[key]
        user_message = "\n".join(PANEL_CONTEXT_BUILDERS[key](ticker, context))
        try:
            text = ai_client.call_provider(
                provider, system_prompt, user_message, model, api_key=api_key, max_tokens=400
            )
            parsed = _parse_panel_agent_json(text)
            results.append(
                {
                    "persona": key,
                    "label": label,
                    "take": parsed.get("take"),
                    "stance": parsed.get("stance", "neutral"),
                    "error": None,
                }
            )
        except Exception as exc:
            results.append({"persona": key, "label": label, "take": None, "stance": None, "error": str(exc)})
    return results
