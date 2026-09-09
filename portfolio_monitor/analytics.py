"""Layer 2: portfolio-level analytics built on top of Layer 1 valuations.

Allocation & concentration, aggregate greeks, IV rank/percentile per option,
and upcoming-expiry visibility. Everything here is informational -- no
trading advice is generated (e.g. rolling is surfaced as a fact, never
recommended).
"""

from __future__ import annotations

from datetime import date


def resolve_sector(ticker: str, sector_map: dict[str, str]) -> str:
    if ticker in sector_map:
        return sector_map[ticker]
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def compute_allocation(
    valuations, sector_map: dict[str, str], ticker_cap_pct: float, sector_cap_pct: float
) -> dict:
    total_value = sum(v.current_value for v in valuations)

    by_ticker: dict[str, float] = {}
    for v in valuations:
        by_ticker[v.position.ticker] = by_ticker.get(v.position.ticker, 0.0) + v.current_value

    ticker_sector: dict[str, str] = {t: resolve_sector(t, sector_map) for t in by_ticker}
    by_sector: dict[str, float] = {}
    for ticker, value in by_ticker.items():
        sector = ticker_sector[ticker]
        by_sector[sector] = by_sector.get(sector, 0.0) + value

    def _rows(bucket: dict[str, float], level: str, cap_pct: float) -> list[dict]:
        rows = []
        for name, value in bucket.items():
            pct = (value / total_value * 100.0) if total_value else 0.0
            rows.append(
                {
                    "level": level,
                    "name": name,
                    "value": value,
                    "pct_of_total": pct,
                    "cap_pct": cap_pct,
                    "flagged": pct > cap_pct,
                }
            )
        return rows

    return {
        "total_value": total_value,
        "by_ticker": _rows(by_ticker, "ticker", ticker_cap_pct),
        "by_sector": _rows(by_sector, "sector", sector_cap_pct),
        "ticker_sector_map": ticker_sector,
    }


def compute_aggregate_greeks(valuations) -> dict:
    return {
        "net_delta_shares": sum(v.delta or 0.0 for v in valuations),
        "total_daily_theta": sum(v.theta or 0.0 for v in valuations),
        "net_vega": sum(v.vega or 0.0 for v in valuations),
    }


def compute_iv_environment(
    conn,
    valuations,
    asof_date: date,
    lookback_days: int,
    min_history_days: int,
    rich_threshold: float,
    cheap_threshold: float,
) -> list[dict]:
    from . import db as db_mod

    results = []
    for v in valuations:
        p = v.position
        if not p.is_option or v.current_iv is None:
            continue

        history = db_mod.get_iv_history(
            conn,
            p.ticker,
            p.expiry.isoformat(),
            p.option_type,
            p.strike,
            asof_date.isoformat(),
            lookback_days,
        )
        ivs = [iv for _, iv in history]
        history_days = len(ivs)

        if history_days < min_history_days:
            results.append(
                {
                    "position_id": p.id,
                    "ticker": p.ticker,
                    "current_iv": v.current_iv,
                    "iv_rank": None,
                    "iv_percentile": None,
                    "history_days": history_days,
                    "status": "building history",
                    "rich": False,
                    "cheap": False,
                }
            )
            continue

        lo, hi = min(ivs), max(ivs)
        iv_rank = ((v.current_iv - lo) / (hi - lo) * 100.0) if hi > lo else 50.0
        below = sum(1 for iv in ivs if iv < v.current_iv)
        iv_percentile = below / history_days * 100.0

        results.append(
            {
                "position_id": p.id,
                "ticker": p.ticker,
                "current_iv": v.current_iv,
                "iv_rank": iv_rank,
                "iv_percentile": iv_percentile,
                "history_days": history_days,
                "status": "ok",
                "rich": iv_rank > rich_threshold,
                "cheap": iv_rank < cheap_threshold,
            }
        )
    return results


def compute_upcoming_expiries(valuations, threshold_days: int) -> list[dict]:
    results = []
    for v in valuations:
        p = v.position
        if not p.is_option or v.dte is None:
            continue
        results.append(
            {
                "position_id": p.id,
                "ticker": p.ticker,
                "expiry": p.expiry.isoformat(),
                "dte": v.dte,
                "within_threshold": v.dte <= threshold_days,
            }
        )
    return results
