"""Options + equity portfolio monitor.

Layer 1 (data & valuation): models, data, greeks, snapshots, db, valuation.
Layer 2 (portfolio analytics): analytics.

Modules are import-only (no side effects at import time) so later layers,
scripts, or notebooks can reuse the pieces directly.
"""
