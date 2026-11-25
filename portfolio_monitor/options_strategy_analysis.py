"""Options-strategy suitability panel: given one ticker's real options-
chain-derived context (implied vs. realized volatility, term structure,
skew, days to next earnings, and a technical/trend snapshot -- see
data.get_options_strategy_context), judge all 13 strategies below as
favorable, neutral, or unfavorable right now, with a short rationale and
a rough suggested structure.

Unlike nearlow_analysis.run_expert_panel (where each persona gets its OWN
narrow slice of a shared context, since they're independent analysts
answering different questions), every strategy here is judged against
the IDENTICAL market snapshot -- there's one shared context-lines
builder, not 13 different ones. And unlike that panel's sequential loop
over a human-selected handful of personas, this always runs all 13 (the
UI has no subset picker for this tab), so it parallelizes across a
thread pool to keep total latency reasonable.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from . import ai_client

# key -> (display label, category). Order matches the user's own 1-13
# numbering and is preserved in run_strategy_panel's output regardless of
# which thread finishes first, so the UI can render in this same order.
STRATEGY_LABELS = {
    "long_call": ("Long Call", "Single-Leg Strategies"),
    "covered_call": ("Covered Call", "Single-Leg Strategies"),
    "long_put": ("Long Put", "Single-Leg Strategies"),
    "cash_secured_put": ("Cash-Secured Put", "Single-Leg Strategies"),
    "call_debit_spread": ("Call Debit Spread", "Vertical Spreads"),
    "call_credit_spread": ("Call Credit Spread", "Vertical Spreads"),
    "put_debit_spread": ("Put Debit Spread", "Vertical Spreads"),
    "put_credit_spread": ("Put Credit Spread", "Vertical Spreads"),
    "long_straddle": ("Long Straddle", "Straddles and Strangles"),
    "long_strangle": ("Long Strangle", "Straddles and Strangles"),
    "long_call_calendar": ("Long Call Calendar Spread", "Calendar Spreads"),
    "long_put_calendar": ("Long Put Calendar Spread", "Calendar Spreads"),
    "short_put_calendar": ("Short Put Calendar Spread", "Calendar Spreads"),
}

_JSON_INSTRUCTION = """

Judge this strategy's suitability for this ticker RIGHT NOW using the \
real data given -- not generic textbook advice -- and suggest a rough \
structure (how far out-of-the-money/in-the-money, roughly how many days \
to expiration) using the actual current price given, not a placeholder.

Respond with ONLY a JSON object, no other text: \
{"verdict": "favorable|neutral|unfavorable", "rationale": "one or two sentences", \
"suggested_structure": "one short sentence, using the real current price"}"""

STRATEGY_SYSTEM_PROMPTS = {
    "long_call": """You are an options strategist evaluating a LONG CALL (buying a call \
outright): a bullish, defined-risk (the premium paid), unlimited-upside \
bet that profits if the stock rises enough to cover that premium before \
expiration. Favorable when implied volatility is LOW-to-moderate \
relative to the stock's own realized volatility (you're not overpaying \
for the option) and there's a clear bullish thesis (trend, catalyst). \
Unfavorable when IV is already rich relative to realized vol (the \
premium itself eats into expected profit) or there's no clear bullish \
case."""
    + _JSON_INSTRUCTION,
    "covered_call": """You are an options strategist evaluating a COVERED CALL: owning (or \
being willing to buy) 100 shares and selling a call against them for \
income, with the premium collected providing mild downside cushion but \
capping the upside at the short strike. Favorable in flat-to-mildly- \
bullish, range-bound conditions, especially with elevated implied \
volatility (more premium collected). Unfavorable if a large upward \
breakout is plausible (caps most of the gain) or a sharp decline is \
expected (the premium collected rarely offsets a large drop)."""
    + _JSON_INSTRUCTION,
    "long_put": """You are an options strategist evaluating a LONG PUT (buying a put \
outright): a bearish, defined-risk (the premium paid) bet that profits \
if the stock falls enough to cover that premium before expiration. \
Favorable when implied volatility is LOW-to-moderate relative to \
realized volatility and there's a clear bearish thesis. Unfavorable \
when IV is already rich relative to realized vol, or there's no clear \
bearish case."""
    + _JSON_INSTRUCTION,
    "cash_secured_put": """You are an options strategist evaluating a CASH-SECURED PUT: selling a \
put while holding enough cash to buy 100 shares at the strike if \
assigned -- generates income and/or is a disciplined way to buy the \
stock at a discount to today's price. Favorable when implied volatility \
is elevated (more premium collected) and the setup is neutral-to-mildly- \
bullish (a genuine "I'd buy the dip here" case for owning the stock at \
the strike). Unfavorable if the stock isn't one you'd want to own at \
all, or IV is already low with no real catalyst."""
    + _JSON_INSTRUCTION,
    "call_debit_spread": """You are an options strategist evaluating a CALL DEBIT SPREAD: buying a \
lower-strike call and selling a higher-strike call, same expiration -- a \
cheaper, defined-risk, capped-upside bullish bet than a long call \
outright. Favorable when implied volatility is elevated enough that a \
long call alone would be expensive (selling the higher strike offsets \
some of that cost) but there's still a clear, BOUNDED bullish thesis. \
Unfavorable with low/cheap IV (a long call alone captures more of the \
same upside for little extra cost) or no bullish case at all."""
    + _JSON_INSTRUCTION,
    "call_credit_spread": """You are an options strategist evaluating a CALL CREDIT SPREAD (bear \
call spread): selling a lower-strike call and buying a higher-strike \
call, same expiration, for a net credit -- defined risk, profits if the \
stock stays BELOW the short strike through expiration. Favorable in \
high-IV, neutral-to-bearish, range-bound-or-down conditions. Unfavorable \
if a strong rally through the short strike is plausible, or IV is \
already low (little credit collected for the risk taken)."""
    + _JSON_INSTRUCTION,
    "put_debit_spread": """You are an options strategist evaluating a PUT DEBIT SPREAD: buying a \
higher-strike put and selling a lower-strike put, same expiration -- a \
cheaper, defined-risk, capped-downside bearish bet than a long put \
outright. Favorable when implied volatility is elevated enough that a \
long put alone would be expensive, but there's still a clear, BOUNDED \
bearish thesis. Unfavorable with low/cheap IV or no bearish case."""
    + _JSON_INSTRUCTION,
    "put_credit_spread": """You are an options strategist evaluating a PUT CREDIT SPREAD (bull put \
spread): selling a higher-strike put and buying a lower-strike put, same \
expiration, for a net credit -- defined risk, profits if the stock stays \
ABOVE the short strike. A very common "sell premium while staying \
neutral-to-bullish" strategy, favorable in high-IV, range-bound-or-up \
conditions. Unfavorable if a sharp decline through the short strike is \
plausible, or IV is already low."""
    + _JSON_INSTRUCTION,
    "long_straddle": """You are an options strategist evaluating a LONG STRADDLE: buying a \
call AND a put at the same (at-the-money) strike and expiration -- a \
direction-agnostic bet on a BIG move either way, large enough to cover \
both premiums. Needs implied volatility to be LOW relative to the move \
actually expected -- e.g. ahead of a binary catalyst (earnings) before \
the market has fully priced in the likely swing. Unfavorable if IV is \
already elevated/rich relative to realized volatility (the move is \
already priced in, so you're overpaying), or no large move is expected \
at all."""
    + _JSON_INSTRUCTION,
    "long_strangle": """You are an options strategist evaluating a LONG STRANGLE: buying an \
out-of-the-money call AND an out-of-the-money put, same expiration -- \
cheaper than a straddle but needs an even BIGGER move to profit, same \
underlying logic. Favorable in the same low-IV-relative-to-expected-move \
setups as a straddle, with a bit more tolerance for being early or wrong \
on magnitude since it costs less to put on. Unfavorable if IV is \
already elevated or no large move is expected."""
    + _JSON_INSTRUCTION,
    "long_call_calendar": """You are an options strategist evaluating a LONG CALL CALENDAR SPREAD: \
selling a near-term call and buying a longer-dated call, same strike \
(usually at or near the current price) -- profits most if the stock \
sits near that strike through the near-term expiration, benefiting from \
the near-term option losing time value faster than the longer-dated \
one. Most favorable when the NEAR-term option's implied volatility is \
elevated relative to the longer-dated one -- an inverted/backwardated \
term structure below, often because an earnings date falls between the \
two expirations and the near-term IV should crush disproportionately \
after it passes -- or in a flat, range-bound setup with no large move \
expected. Unfavorable in a strongly trending/breakout environment, or if \
the term structure shows no such near-term IV premium to exploit."""
    + _JSON_INSTRUCTION,
    "long_put_calendar": """You are an options strategist evaluating a LONG PUT CALENDAR SPREAD: \
selling a near-term put and buying a longer-dated put, same strike -- \
the same mechanics and term-structure logic as a call calendar spread, \
but with a slightly bearish-to-neutral bias (profits most if the stock \
sits near or drifts down toward the strike through the near-term \
expiration). Favorable with an inverted/backwardated term structure \
(often an earnings date between the two expirations) or a flat, \
range-bound-to-down setup. Unfavorable in a strongly trending/breakout \
environment, or with no such near-term IV premium to exploit."""
    + _JSON_INSTRUCTION,
    "short_put_calendar": """You are an options strategist evaluating a SHORT PUT CALENDAR SPREAD \
(a "reverse" calendar): buying a near-term put and selling a \
longer-dated put, same strike, for a net credit -- the OPPOSITE of a \
standard calendar spread. It wants the near-term option to GAIN value \
relative to the longer-dated one, which happens from a large, fast move \
in the underlying, or from near-term implied volatility RISING relative \
to the back month -- e.g. positioning just ahead of an anticipated \
binary event, wanting long near-term gamma/vega financed by the short \
back-month leg. Favorable ahead of an expected volatility expansion or \
sharp move with an identifiable near-term catalyst -- look for NORMAL \
(not yet inverted) term structure now, with a catalyst approaching that \
should invert it. Unfavorable in a quiet, range-bound environment with \
no such catalyst -- that setup favors a STANDARD (long) calendar spread \
instead, not this one."""
    + _JSON_INSTRUCTION,
}


def _context_lines(ticker: str, context: dict) -> list[str]:
    """Shared by every strategy -- unlike nearlow_analysis's per-persona
    narrow slices, all 13 strategies here are judged against the
    identical options-chain-derived snapshot."""
    current_price = context.get("current_price")
    near_dte = context.get("near_expiry_dte")
    far_dte = context.get("far_expiry_dte")
    atm_iv = context.get("atm_iv_pct")
    skew = context.get("skew_pct")
    term = context.get("term_structure_pct")
    hv_20 = context.get("hv_20_pct")
    hv_60 = context.get("hv_60_pct")
    days_to_earnings = context.get("days_to_earnings")

    lines = [f"Ticker: {ticker}", f"Current price: {current_price}"]

    if atm_iv is not None:
        lines.append(f"At-the-money implied volatility (~{near_dte}-day expiry): {atm_iv:.1f}%")
    else:
        lines.append("At-the-money implied volatility: not available")
    if hv_20 is not None:
        lines.append(f"20-day trailing realized (historical) volatility: {hv_20:.1f}%")
    if hv_60 is not None:
        lines.append(f"60-day trailing realized (historical) volatility: {hv_60:.1f}%")
    if atm_iv is not None and hv_20 is not None:
        premium = atm_iv - hv_20
        if premium > 5:
            read = "options pricing in MORE movement than realized -- relatively rich"
        elif premium < -5:
            read = "options pricing in LESS movement than realized -- relatively cheap"
        else:
            read = "roughly in line with realized movement"
        lines.append(f"IV vs 20-day realized vol: {premium:+.1f} points ({read})")
    if skew is not None:
        read = "puts bid up relative to calls -- typical/defensive skew" if skew > 0 else "calls bid up relative to puts -- unusual, often a bullish/breakout setup"
        lines.append(f"Put/call skew (~10% OTM put IV minus ~10% OTM call IV): {skew:+.1f} points ({read})")
    if term is not None and far_dte is not None:
        read = "inverted/backwardated -- market pricing in a near-term event (often earnings) between the two expiries" if term < 0 else "normal contango -- no unusual near-term event priced in"
        lines.append(f"Term structure (ATM IV at ~{far_dte}-day expiry minus ~{near_dte}-day expiry): {term:+.1f} points ({read})")
    else:
        lines.append("Term structure: not available (no suitable longer-dated expiry)")
    lines.append(f"Next earnings: in {days_to_earnings} day(s)" if days_to_earnings is not None else "Next earnings date: not available")

    technical = context.get("technical") or {}
    if technical:
        lines.append("")
        lines.append("Technical snapshot:")
        for period in (20, 50, 200):
            key = f"price_vs_sma_{period}_pct"
            if key in technical:
                lines.append(f"- Price vs SMA({period}): {technical[key]:+.1f}%")
        if "rsi_14" in technical:
            lines.append(f"- RSI(14): {technical['rsi_14']:.1f}")
        if "macd_histogram" in technical:
            lines.append(f"- MACD histogram: {technical['macd_histogram']:+.3f}")
    return lines


def _parse_strategy_json(text: str) -> dict:
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
            parsed = {"rationale": text.strip()[:800] or None}
    parsed.setdefault("verdict", "neutral")
    parsed.setdefault("suggested_structure", None)
    return parsed


def run_strategy_panel(
    ticker: str,
    context: dict,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    strategies: Optional[list[str]] = None,
) -> list[dict]:
    """Evaluates every strategy in STRATEGY_LABELS (or just `strategies`,
    for tests/future flexibility -- the UI itself always requests all 13)
    against the same shared context. Runs on a thread pool, unlike
    run_expert_panel's sequential loop, since this always fires off all
    13 calls at once rather than a human-selected handful -- sequential
    would make one run take noticeably longer. Each strategy's own
    failure is isolated to its own entry rather than aborting the rest,
    and results are returned in STRATEGY_LABELS' canonical order
    regardless of which thread finishes first."""
    keys = [k for k in (strategies if strategies is not None else list(STRATEGY_LABELS)) if k in STRATEGY_LABELS]
    user_message = "\n".join(_context_lines(ticker, context))

    def _run_one(key: str) -> dict:
        label, category = STRATEGY_LABELS[key]
        system_prompt = STRATEGY_SYSTEM_PROMPTS[key]
        try:
            text = ai_client.call_provider(
                provider, system_prompt, user_message, model, api_key=api_key, max_tokens=350
            )
            parsed = _parse_strategy_json(text)
            return {
                "strategy": key,
                "label": label,
                "category": category,
                "verdict": parsed.get("verdict", "neutral"),
                "rationale": parsed.get("rationale"),
                "suggested_structure": parsed.get("suggested_structure"),
                "error": None,
            }
        except Exception as exc:
            return {
                "strategy": key,
                "label": label,
                "category": category,
                "verdict": None,
                "rationale": None,
                "suggested_structure": None,
                "error": str(exc),
            }

    results = []
    with ThreadPoolExecutor(max_workers=max(1, len(keys))) as executor:
        futures = [executor.submit(_run_one, key) for key in keys]
        for future in as_completed(futures):
            results.append(future.result())

    order = {key: i for i, key in enumerate(STRATEGY_LABELS)}
    results.sort(key=lambda r: order.get(r["strategy"], len(order)))
    return results
