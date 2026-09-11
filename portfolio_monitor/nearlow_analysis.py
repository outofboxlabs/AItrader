"""On-demand AI analysis for the Near-52-Week-Low screen: a single ~200-word
expert take with a buy-opportunity verdict, and a 5-persona "panel
discussion" (technical / fundamental / news / analyst-ratings-timing /
macro-risk) that argues the stock from different angles. Both are
informational only -- never a buy/sell/hold instruction -- and only run
when a human clicks a button for one specific ticker; nothing here runs
automatically across a whole screen.
"""

from __future__ import annotations

import json
import re
from typing import Optional

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


def get_rating_timeline(ticker: str) -> dict:
    """Recent analyst rating changes (from yfinance's upgrade/downgrade
    history) annotated with the stock's own price at the time of each one,
    plus the date of its own 52-week low from a year of daily bars -- so a
    "ratings timing" read can say whether a given rating predates the
    slide to the low (and so may be stale) or postdates it (i.e. the
    analyst kept/gave that rating knowing the current price). Never
    raises -- a failed pull just means an empty timeline, which callers
    (and the AI prompt) should treat as "can't be assessed" rather than
    an error."""
    daily_history = sorted(data_mod.get_daily_price_history(ticker), key=lambda d: d["date"])

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
            actions.append(action)

    actions.sort(key=lambda a: a["date"], reverse=True)
    # Cap to the most recent 10 -- older coverage history doesn't help
    # judge whether TODAY's ratings are stale, and would just bloat the
    # prompt for names with a long analyst history.
    actions = actions[:10]
    return json.loads(json.dumps({"week_52_low": week_52_low, "actions": actions}, default=str))


def _timing_label(action: dict) -> str:
    before = action.get("before_52w_low")
    if before is True:
        return "BEFORE"
    if before is False:
        return "AFTER"
    return "unknown timing vs."


def build_context_user_message(
    ticker: str, candidate: dict, rating_timeline: dict, headlines: list[dict], macro_score: Optional[float]
) -> str:
    """Shared context block for every persona (the single expert take and
    all 5 panel agents) -- same underlying facts, different lens applied
    by each system prompt."""
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
        for a in actions:
            price_note = f", price then ~{a['price_at_rating']:.2f}" if a.get("price_at_rating") is not None else ""
            move_note = (
                f", stock has moved {a['pct_move_since_rating']:+.1f}% since"
                if a.get("pct_move_since_rating") is not None
                else ""
            )
            lines.append(
                f"- [{a.get('date')}] {a.get('firm')}: {a.get('action')} "
                f"({a.get('from_grade')} -> {a.get('to_grade')}) -- {_timing_label(a)} the 52-week low"
                f"{price_note}{move_note}"
            )
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
in a portfolio monitoring tool. You are given one stock trading near its \
52-week low: its price/range data, current analyst consensus, recent analyst \
rating CHANGES with their dates and the stock's price at each one, recent \
headlines, and the current macro environment. Your job is ONLY to inform, \
never to advise.

Write ONE analysis of about 200 words (180-220 is fine) covering:
- Why the stock may be down near its low, based on the headlines/data given.
- Whether the current analyst consensus and any recent rating changes still \
look credible given how much time has passed and how the price has moved \
since -- an old Buy rating from well before a big slide carries less weight \
than one reaffirmed, or issued, after it.
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
    candidate: dict,
    rating_timeline: dict,
    headlines: list[dict],
    macro_score: Optional[float],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> dict:
    """Raises on API failure -- the caller decides how to degrade."""
    user_message = build_context_user_message(ticker, candidate, rating_timeline, headlines, macro_score)
    text = ai_client.call_provider(provider, EXPERT_SYSTEM_PROMPT, user_message, model, api_key=api_key, max_tokens=700)
    return _parse_expert_json(text)


# --- 5-persona panel discussion ----------------------------------------------


PANEL_SYSTEM_PROMPTS = {
    "technical": (
        "Technical Analyst",
        """You are a technical analyst on a panel discussing one stock trading \
near its 52-week low. You're given its current price, 52-week high/low, and \
the date of its own 52-week low. Give a short (60-100 word) technical read: \
where the current price sits in its range, whether the low looks like it \
may be forming a base or still falling, and what price level would change \
your mind either way. You have no chart or indicator data beyond what's \
given -- be honest about that limit rather than inventing patterns you \
can't see. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "fundamental": (
        "Fundamental Analyst",
        """You are a fundamental analyst on a panel discussing one stock trading \
near its 52-week low. You're given its market cap, analyst mean price \
target, and the buy-ratio among current ratings. Give a short (60-100 word) \
fundamental read on whether the current price plausibly undervalues the \
business given what analysts are pricing in via their target, and what \
would need to be true about the business for the stock to re-rate higher. \
Be explicit when you lack real fundamental data (earnings, margins, balance \
sheet) to go on -- don't invent figures. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "news_sentiment": (
        "News & Sentiment Analyst",
        """You are a news/sentiment analyst on a panel discussing one stock \
trading near its 52-week low. You're given its recent headlines. Give a \
short (60-100 word) read on what the news flow suggests is driving the \
price action, and whether sentiment looks like it's stabilizing, still \
deteriorating, or already reflects a worst case. If no headlines were \
given, say so plainly rather than guessing. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "ratings_timing": (
        "Analyst-Ratings Auditor",
        """You are the analyst-ratings auditor on a panel discussing one stock \
trading near its 52-week low. Your ONE job is to check the TIMING of recent \
analyst rating changes against the stock's own price action. You are given \
each recent rating change's date, the stock's price on/near that date, \
whether it fell before or after the date of the stock's 52-week low, and \
the % the price has moved since that rating. Reason explicitly about \
staleness: a Buy issued well BEFORE the slide to the low may not reflect \
today's reality and could be stale; a Buy reaffirmed or issued AFTER the \
low (or a recent downgrade) is a stronger, more current signal either way. \
Name the specific firm(s) and date(s) you're weighing. If no rating-change \
data was given, say so plainly rather than guessing. Give a short (80-120 \
word) take. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "macro_risk": (
        "Macro & Risk Manager",
        """You are the macro/risk manager on a panel discussing one stock \
trading near its 52-week low. You're given the current macro "calm" score \
(0-100, higher = calmer) and everything else given about the stock. Give a \
short (60-100 word) devil's-advocate read: what could keep this stock down \
further regardless of company-specific factors (macro conditions, sector \
rotation, liquidity), and whether the current macro backdrop supports or \
argues against adding risk to a name like this right now. Never say to \
buy, sell, or hold.

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
    candidate: dict,
    rating_timeline: dict,
    headlines: list[dict],
    macro_score: Optional[float],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Run all 5 panelist personas against the same shared context, one at
    a time. Each persona's own failure (rate limit, transient network
    error) is isolated to its own entry rather than aborting the whole
    panel -- a partial panel with 4 good takes and one clearly-marked
    error is more useful than losing all 5 over one bad call."""
    user_message = build_context_user_message(ticker, candidate, rating_timeline, headlines, macro_score)
    results = []
    for key, (label, system_prompt) in PANEL_SYSTEM_PROMPTS.items():
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
