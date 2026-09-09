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

# Storage
DB_PATH = "portfolio.db"
SNAPSHOTS_DIR = "snapshots"
POSITIONS_PATH = "positions.json"
