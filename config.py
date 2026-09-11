"""Central, editable configuration for the portfolio monitor.

All thresholds here have CLI overrides in main.py (see --help) -- edit the
defaults below if you want them to stick without passing flags every run.
"""

from __future__ import annotations

import os

# All of the relative paths below are anchored to this file's own directory
# rather than left as bare relative paths. A bare relative path resolves
# against the process's current working directory, which silently differs
# depending on how python app.py was launched (a terminal cd'ed elsewhere,
# a double-clicked shortcut, an IDE run config with its own working
# directory) -- so the exact same "positions.json" string can point at a
# different file on disk from one launch to the next, making saved data
# (positions, credentials, the db) appear to vanish for no visible reason.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _here(name: str) -> str:
    return os.path.join(_BASE_DIR, name)


# Black-Scholes
RISK_FREE_RATE = 0.045  # annual, e.g. 3-month T-bill yield

# Ticker -> sector lookup used for allocation reporting. Any ticker not
# listed here falls back to yfinance's Ticker.info["sector"].
TICKER_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "AMZN": "Consumer Cyclical",
    "TSLA": "Consumer Cyclical",
    "NVDA": "Technology",
    "META": "Technology",
    "SPY": "Diversified",
    "QQQ": "Diversified",
}

# Concentration flags (informational only -- these do not block anything).
CONCENTRATION_TICKER_CAP_PCT = 40.0
CONCENTRATION_SECTOR_CAP_PCT = 60.0

# IV rank/percentile, computed from our own accumulated daily snapshots.
IV_RANK_LOOKBACK_DAYS = 252
IV_RANK_MIN_HISTORY_DAYS = 20
IV_RICH_THRESHOLD = 70.0
IV_CHEAP_THRESHOLD = 30.0

# Upcoming-expiry visibility (surfaced, not advised on).
EXPIRY_WARNING_DAYS = 45

# --- Macro gate (deterministic, 0-100; higher = calmer environment) ---
# Weights are normalized to sum to 1.0 automatically, so relative values
# matter more than the exact numbers.
MACRO_WEIGHTS: dict[str, float] = {
    "vix_level": 0.35,
    "term_structure": 0.15,
    "breadth": 0.25,
    "credit_spread": 0.25,
}
MACRO_LOOKBACK_DAYS = 252  # trading days used for VIX and credit-ratio percentiles
VIX_TERM_CALM_RATIO = 0.90  # VIX/VIX3M at/under this = fully calm (contango)
VIX_TERM_STRESS_RATIO = 1.10  # VIX/VIX3M at/over this = fully stressed (backwardation)
# Breadth proxy: pulling all ~500 S&P constituents daily is slow, so this
# sector-ETF basket stands in unless SPY_CONSTITUENTS is populated below.
BREADTH_PROXY_TICKERS: list[str] = [
    "XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLRE", "XLU", "XLC",
]
SPY_CONSTITUENTS: list[str] = []  # optional: real constituent list overrides the proxy

# --- News analysis (the only paid part of the system) ---
# NEWS_PROVIDER picks which API does the summarizing; each provider has its
# own model default below. Swap providers without touching any code, or
# pick interactively at runtime with `python main.py --interactive`.
NEWS_PROVIDER = "anthropic"  # "anthropic", "openai", or "gemini"
ANTHROPIC_NEWS_MODEL = "claude-haiku-4-5"  # small/cheap model is enough for this
OPENAI_NEWS_MODEL = "gpt-4o-mini"  # verify against OpenAI's current model list -- not vetted by a live source here
GEMINI_NEWS_MODEL = "gemini-2.0-flash"  # UNVERIFIED -- no live Gemini reference in this session, confirm against Google's current docs
NEWS_WINDOW_DAYS = 3

# --- Vision extraction (screenshot -> positions.json draft), used by app.py ---
# Extracting numbers (strikes, prices, share counts) from an image is a
# harder task than summarizing text, so these default to each provider's
# strongest vision-capable model rather than the cheap news-analysis one.
VISION_PROVIDER = "anthropic"  # "anthropic", "openai", or "gemini"
ANTHROPIC_VISION_MODEL = "claude-opus-5"
OPENAI_VISION_MODEL = "gpt-4o"  # not vetted by a live source here
GEMINI_VISION_MODEL = "gemini-2.0-flash"  # UNVERIFIED -- see note above

# Local, gitignored cache of API keys entered interactively -- see
# portfolio_monitor/credentials.py. Never commit this file.
CREDENTIALS_PATH = _here(".credentials.json")

# --- Market movers screener (Top Movers tab) ---
# A >40% single-day drop is a genuinely rare, severe event -- realistically
# zero matches on most days across the whole US market. Lower this if you
# want to see smaller-but-still-notable drops more often.
MOVERS_DROP_THRESHOLD_PCT = -40.0
MOVERS_MIN_MARKET_CAP = 2_000_000_000
MOVERS_MIN_PRICE = 5.0
MOVERS_MIN_VOLUME = 20_000
MOVERS_MAX_RESULTS = 25

# --- Growth screener (Top Growth tab) ---
# "50% growth in a month" isn't a metric any data source publishes -- the
# closest real, checkable proxy is the analyst consensus price target,
# which is conventionally a ~12-month view, not a 1-month one. Labeled
# accordingly in the UI as "analyst target upside", not a 1-month forecast.
GROWTH_CANDIDATE_POOL_SIZE = 60  # each candidate costs a few network round trips -- kept modest so "Run Now" finishes in a reasonable time
GROWTH_MIN_MARKET_CAP = 300_000_000
GROWTH_MIN_PRICE = 5.0
GROWTH_MIN_VOLUME = 100_000
GROWTH_TARGET_UPSIDE_THRESHOLD_PCT = 40.0
GROWTH_MAX_RESULTS = 50
GROWTH_MAX_WORKERS = 20  # per-candidate lookups run in parallel on this many threads

# --- Near-52-week-low screener (Near 52W Low tab) ---
# Beaten-down stocks the analyst consensus still rates favorably: within
# NEARLOW_MAX_PCT_FROM_LOW% of the 52-week low, with at least
# NEARLOW_MIN_RATINGS_COUNT analyst ratings of which >= NEARLOW_MIN_BUY_RATIO_PCT%
# are "buy" or "strong buy". Both conditions are required.
NEARLOW_CANDIDATE_POOL_SIZE = 150
NEARLOW_MIN_MARKET_CAP = 300_000_000
NEARLOW_MIN_PRICE = 5.0
NEARLOW_MIN_VOLUME = 100_000
NEARLOW_MAX_PCT_FROM_LOW = 15.0
NEARLOW_MIN_BUY_RATIO_PCT = 60.0
NEARLOW_MIN_RATINGS_COUNT = 3
NEARLOW_MAX_RESULTS = 100
NEARLOW_MAX_WORKERS = 20

# --- Forex Factory economic calendar (Forex Calendar tab) ---
# Forex Factory has no official public API for this. It does publish a
# public JSON/XML/CSV/ICS feed that powers their own embeddable calendar
# widget -- unofficial and undocumented, but the standard way the retail
# trading-bot community reads this data, since scraping the HTML calendar
# page directly is fragile and against their ToS. Community reports put
# this feed's rate limit at roughly 2 requests per 5 minutes per IP, so
# FOREX_CALENDAR_MIN_REFRESH_SECONDS enforces a floor between real
# upstream fetches -- "Run Now" inside that window just re-serves the
# last cached pull instead of risking a block.
FOREX_CALENDAR_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FOREX_CALENDAR_MIN_REFRESH_SECONDS = 300

# Storage
DB_PATH = _here("portfolio.db")
SNAPSHOTS_DIR = _here("snapshots")
POSITIONS_PATH = _here("positions.json")
EXPORTS_DIR = _here("exports")  # timestamped CSVs from Top Movers / Top Growth scans
