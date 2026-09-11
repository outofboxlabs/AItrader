"""On-demand AI analysis for the Near-52-Week-Low screen: a single ~200-word
expert take with a buy-opportunity verdict, and 5 independent single-focus
analyst inquiries (technical / fundamental / news / analyst-ratings-timing /
macro-risk) about the same stock. The 5 are genuinely separate calls, each
given only the narrow slice of data its own question needs -- not one
prompt told "you are 5 agents on a panel" with the full context repeated
five times. Both are informational only -- never a buy/sell/hold
instruction -- and only run when a human clicks a button for one specific
ticker; nothing here runs automatically across a whole screen.
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
in a portfolio monitoring tool. You are given one stock trading near its \
52-week low: its price/range data, current analyst consensus, recent analyst \
rating CHANGES with their dates and the stock's price at each one, recent \
headlines, and the current macro environment. Your job is ONLY to inform, \
never to advise.

Write ONE analysis of about 200 words (180-220 is fine) covering:
- Why the stock may be down near its low, based on the headlines/data given.
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


# --- 5 independent single-focus analyst inquiries ---------------------------
#
# These are 5 separate, independent calls -- each persona gets its OWN
# narrow slice of the data (see PANEL_CONTEXT_BUILDERS below) and answers
# its own question with no idea the other 4 exist. Earlier system prompts
# framed this as "you are one of 5 panelists discussing this stock", which
# read as one role-play prompt wearing different hats rather than 5
# genuinely separate inquiries -- these are written as standalone analyst
# briefs instead, with no reference to a panel or other agents.


def _price_range_lines(ticker: str, candidate: dict, rating_timeline: dict) -> list[str]:
    lines = [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"Current price: {candidate.get('price')}",
        f"52-week range: low {candidate.get('year_low')}, high {candidate.get('year_high')}",
        f"% above 52-week low: {candidate.get('pct_from_52w_low')}",
    ]
    week_52_low = rating_timeline.get("week_52_low")
    if week_52_low:
        lines.append(f"Date of the stock's own 52-week low: {week_52_low['date']} (close {week_52_low['close']})")
    else:
        lines.append("Date of the 52-week low: unknown (price history unavailable)")
    return lines


def _fundamental_lines(ticker: str, candidate: dict) -> list[str]:
    return [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"Market cap: {candidate.get('market_cap')}",
        f"Analyst mean price target: {candidate.get('target_mean')} "
        f"({candidate.get('target_upside_pct')}% upside from current price {candidate.get('price')})",
        f"Current analyst ratings breakdown: {candidate.get('analyst_ratings')} "
        f"({candidate.get('buy_ratio_pct')}% buy/strong-buy)",
    ]


def _headline_lines(ticker: str, candidate: dict, headlines: list[dict]) -> list[str]:
    lines = [f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})", "", "Recent headlines:"]
    if headlines:
        for h in headlines:
            when = h["published_at"].isoformat() if h.get("published_at") else "unknown date"
            lines.append(f"- [{when}] {h['title']}")
    else:
        lines.append("(none found in the lookback window)")
    return lines


def _rating_timeline_lines(ticker: str, candidate: dict, rating_timeline: dict) -> list[str]:
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


def _macro_lines(ticker: str, candidate: dict, macro_score: Optional[float]) -> list[str]:
    lines = [
        f"Ticker: {ticker} ({candidate.get('name') or 'n/a'})",
        f"% above its 52-week low: {candidate.get('pct_from_52w_low')}",
    ]
    if macro_score is not None:
        lines.append(f"Current macro gate score: {macro_score:.1f}/100 (higher = calmer environment)")
    else:
        lines.append("Macro gate score: not available")
    return lines


PANEL_CONTEXT_BUILDERS = {
    "technical": lambda ticker, candidate, rating_timeline, headlines, macro_score: _price_range_lines(
        ticker, candidate, rating_timeline
    ),
    "fundamental": lambda ticker, candidate, rating_timeline, headlines, macro_score: _fundamental_lines(
        ticker, candidate
    ),
    "news_sentiment": lambda ticker, candidate, rating_timeline, headlines, macro_score: _headline_lines(
        ticker, candidate, headlines
    ),
    "ratings_timing": lambda ticker, candidate, rating_timeline, headlines, macro_score: _rating_timeline_lines(
        ticker, candidate, rating_timeline
    ),
    "macro_risk": lambda ticker, candidate, rating_timeline, headlines, macro_score: _macro_lines(
        ticker, candidate, macro_score
    ),
}


PANEL_SYSTEM_PROMPTS = {
    "technical": (
        "Technical Analyst",
        """You are a technical analyst. You're asked to independently assess one \
stock trading near its 52-week low, given only its current price, 52-week \
high/low, and the date of its own 52-week low. Give a short (60-100 word) \
technical read: where the current price sits in its range, whether the low \
looks like it may be forming a base or still falling, and what price level \
would change your mind either way. You have no chart or indicator data \
beyond what's given -- be honest about that limit rather than inventing \
patterns you can't see. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "fundamental": (
        "Fundamental Analyst",
        """You are a fundamental analyst. You're asked to independently assess \
one stock trading near its 52-week low, given only its market cap, analyst \
mean price target, and the buy-ratio among current ratings. Give a short \
(60-100 word) fundamental read on whether the current price plausibly \
undervalues the business given what analysts are pricing in via their \
target, and what would need to be true about the business for the stock to \
re-rate higher. Be explicit when you lack real fundamental data (earnings, \
margins, balance sheet) to go on -- don't invent figures. Never say to buy, \
sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "news_sentiment": (
        "News & Sentiment Analyst",
        """You are a news/sentiment analyst. You're asked to independently assess \
one stock trading near its 52-week low, given only its recent headlines. \
Give a short (60-100 word) read on what the news flow suggests is driving \
the price action, and whether sentiment looks like it's stabilizing, still \
deteriorating, or already reflects a worst case. If no headlines were \
given, say so plainly rather than guessing. Never say to buy, sell, or hold.

Respond with ONLY a JSON object, no other text: \
{"take": "...", "stance": "bullish|bearish|neutral"}""",
    ),
    "ratings_timing": (
        "Analyst-Ratings Auditor",
        """You are an analyst-ratings auditor. Your ONLY job is to independently \
check the TIMING of recent analyst rating changes for one stock trading near \
its 52-week low against the stock's own price action -- nothing else about \
this stock is in scope for you. For each recent rating change you are given \
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
stock trading near its 52-week low, given only how far above that low it \
currently sits and the current macro "calm" score (0-100, higher = \
calmer) -- no company-specific fundamentals or news are in scope for you. \
Give a short (60-100 word) devil's-advocate read: what could keep a stock \
like this down further regardless of company-specific factors (macro \
conditions, sector rotation, liquidity), and whether the current macro \
backdrop supports or argues against adding risk to a name like this right \
now. Never say to buy, sell, or hold.

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
    """Run 5 independent single-focus analyst inquiries, one at a time --
    each gets ONLY the data relevant to its own question (see
    PANEL_CONTEXT_BUILDERS), not the full shared context, so this is 5
    genuinely separate calls rather than one prompt wearing 5 hats. Each
    persona's own failure (rate limit, transient network error) is
    isolated to its own entry rather than aborting the rest -- a partial
    result with 4 good takes and one clearly-marked error is more useful
    than losing all 5 over one bad call."""
    results = []
    for key, (label, system_prompt) in PANEL_SYSTEM_PROMPTS.items():
        user_message = "\n".join(PANEL_CONTEXT_BUILDERS[key](ticker, candidate, rating_timeline, headlines, macro_score))
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
