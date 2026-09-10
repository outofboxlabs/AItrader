"""Central, editable configuration for the portfolio monitor.

All thresholds here have CLI overrides in main.py (see --help) -- edit the
defaults below if you want them to stick without passing flags every run.
"""

from __future__ import annotations

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
CREDENTIALS_PATH = ".credentials.json"

# --- Growth screener (Top Growth tab) ---
# "50% growth in a month" isn't a metric any data source publishes -- the
# closest real, checkable proxy is the analyst consensus price target,
# which is conventionally a ~12-month view, not a 1-month one. Labeled
# accordingly in the UI as "analyst target upside", not a 1-month forecast.
GROWTH_CANDIDATE_POOL_SIZE = 200
GROWTH_MIN_MARKET_CAP = 300_000_000
GROWTH_MIN_PRICE = 5.0
GROWTH_MIN_VOLUME = 100_000
GROWTH_TARGET_UPSIDE_THRESHOLD_PCT = 50.0
GROWTH_MAX_RESULTS = 50

# Storage
DB_PATH = "portfolio.db"
SNAPSHOTS_DIR = "snapshots"
POSITIONS_PATH = "positions.json"
EXPORTS_DIR = "exports"  # timestamped CSVs from Top Movers / Top Growth scans
