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
# own model default below. Swap providers without touching any code.
NEWS_PROVIDER = "anthropic"  # "anthropic" or "openai"
ANTHROPIC_NEWS_MODEL = "claude-haiku-4-5"  # small/cheap model is enough for this
OPENAI_NEWS_MODEL = "gpt-4o-mini"  # verify against OpenAI's current model list -- not vetted by a live source here
NEWS_WINDOW_DAYS = 3

# Storage
DB_PATH = "portfolio.db"
SNAPSHOTS_DIR = "snapshots"
POSITIONS_PATH = "positions.json"
