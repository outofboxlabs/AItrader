"""End-to-end analysis pipeline: pulls live data, values every position,
and runs the Layer 2/3 analytics (allocation, aggregate greeks, IV
environment, upcoming expiries, macro gate, news). Returns one
JSON-serializable results dict so the CLI (main.py) and the local web app
(app.py) share the exact same logic instead of duplicating it.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import config
from . import analytics, credentials, data, db, macro, news, snapshots
from .models import load_positions
from .valuation import to_db_row, value_option_position, value_shares_position


def _valuation_to_dict(v) -> dict:
    p = v.position
    return {
        "position_id": p.id,
        "ticker": p.ticker,
        "asset_type": p.asset_type,
        "option_type": p.option_type,
        "strike": p.strike,
        "expiry": p.expiry.isoformat() if p.expiry else None,
        "entry_price": p.entry_price,
        "contracts": p.contracts,
        "entry_date": p.entry_date.isoformat() if p.entry_date else None,
        "target_price": p.target_price,
        "stop_price": p.stop_price,
        "mark": v.mark,
        "current_value": v.current_value,
        "unrealized_pnl": v.unrealized_pnl,
        "unrealized_pnl_pct": v.unrealized_pnl_pct,
        "dte": v.dte,
        "progress_to_target_pct": v.progress_to_target_pct,
        "progress_to_stop_pct": v.progress_to_stop_pct,
        "delta": v.delta,
        "gamma": v.gamma,
        "theta": v.theta,
        "vega": v.vega,
        "current_iv": v.current_iv,
    }


def run_full_analysis(
    *,
    positions_path: str,
    db_path: str,
    snapshots_dir: str,
    asof_date: Optional[date] = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
    ticker_cap_pct: float = config.CONCENTRATION_TICKER_CAP_PCT,
    sector_cap_pct: float = config.CONCENTRATION_SECTOR_CAP_PCT,
    iv_lookback_days: int = config.IV_RANK_LOOKBACK_DAYS,
    iv_min_history_days: int = config.IV_RANK_MIN_HISTORY_DAYS,
    iv_rich_threshold: float = config.IV_RICH_THRESHOLD,
    iv_cheap_threshold: float = config.IV_CHEAP_THRESHOLD,
    expiry_warning_days: int = config.EXPIRY_WARNING_DAYS,
    skip_macro: bool = False,
    macro_lookback_days: int = config.MACRO_LOOKBACK_DAYS,
    vix_term_calm_ratio: float = config.VIX_TERM_CALM_RATIO,
    vix_term_stress_ratio: float = config.VIX_TERM_STRESS_RATIO,
    skip_news: bool = False,
    news_window_days: int = config.NEWS_WINDOW_DAYS,
    news_provider: str = config.NEWS_PROVIDER,
    news_model: Optional[str] = None,
    news_api_key: Optional[str] = None,
) -> dict:
    asof_date = asof_date or date.today()
    pulled_at = datetime.now(timezone.utc).isoformat()

    positions = load_positions(positions_path)
    db.init_db(db_path)

    tickers = sorted({p.ticker for p in positions})
    option_tickers = sorted({p.ticker for p in positions if p.is_option})
    spots: dict[str, float] = {}
    chains: dict[str, dict] = {}
    diffs: list[dict] = []
    warnings: list[str] = []

    with db.connect(db_path) as conn:
        for ticker in tickers:
            spots[ticker] = data.get_spot_price(ticker)

        for ticker in option_tickers:
            prior_dates = db.get_prior_snapshot_dates(conn, ticker, asof_date.isoformat())
            prior_keys = db.get_snapshot_keys(conn, ticker, prior_dates[0]) if prior_dates else set()

            chain = data.pull_full_chain(ticker)
            chains[ticker] = chain
            snapshots.save_snapshot_json(snapshots_dir, ticker, asof_date, chain)
            db.save_chain_snapshot_rows(conn, ticker, asof_date.isoformat(), pulled_at, chain)

            if prior_keys:
                current_keys = db.get_snapshot_keys(conn, ticker, asof_date.isoformat())
                diff = snapshots.diff_new_strikes_and_expiries(prior_keys, current_keys)
                if diff["new_expiries"] or diff["new_strikes"]:
                    diffs.append(
                        {
                            "ticker": ticker,
                            "new_expiries": diff["new_expiries"],
                            "new_strikes": len(diff["new_strikes"]),
                        }
                    )

        valuations = []
        for position in positions:
            if position.is_option:
                quote = data.find_quote(
                    chains[position.ticker], position.expiry.isoformat(), position.option_type, position.strike
                )
                if quote is None:
                    warnings.append(f"no live quote found for {position.id}, skipped")
                    continue
                valuation = value_option_position(position, spots[position.ticker], quote, asof_date, risk_free_rate)
            else:
                valuation = value_shares_position(position, spots[position.ticker])
            valuations.append(valuation)

        db.save_valuations(conn, [to_db_row(v, asof_date) for v in valuations])

        allocation = analytics.compute_allocation(valuations, config.TICKER_SECTOR_MAP, ticker_cap_pct, sector_cap_pct)
        aggregate_greeks = analytics.compute_aggregate_greeks(valuations)
        iv_environment = analytics.compute_iv_environment(
            conn, valuations, asof_date, iv_lookback_days, iv_min_history_days, iv_rich_threshold, iv_cheap_threshold
        )
        upcoming_expiries = analytics.compute_upcoming_expiries(valuations, expiry_warning_days)

        db.save_allocation(
            conn,
            [
                (asof_date.isoformat(), r["level"], r["name"], r["value"], r["pct_of_total"], int(r["flagged"]), r["cap_pct"])
                for r in allocation["by_ticker"] + allocation["by_sector"]
            ],
        )
        db.save_portfolio_greeks(
            conn,
            asof_date.isoformat(),
            aggregate_greeks["net_delta_shares"],
            aggregate_greeks["total_daily_theta"],
            aggregate_greeks["net_vega"],
        )
        db.save_iv_environment(
            conn,
            [
                (
                    asof_date.isoformat(),
                    r["position_id"],
                    r["ticker"],
                    r["current_iv"],
                    r["iv_rank"],
                    r["iv_percentile"],
                    r["history_days"],
                    r["status"],
                    int(r["rich"]),
                    int(r["cheap"]),
                )
                for r in iv_environment
            ],
        )
        db.save_upcoming_expiries(
            conn,
            [
                (asof_date.isoformat(), r["position_id"], r["ticker"], r["expiry"], r["dte"], int(r["within_threshold"]))
                for r in upcoming_expiries
            ],
        )

        macro_result = None
        if not skip_macro:
            breadth_tickers = config.SPY_CONSTITUENTS or config.BREADTH_PROXY_TICKERS
            macro_result = macro.compute_macro_gate(
                weights=config.MACRO_WEIGHTS,
                lookback_days=macro_lookback_days,
                calm_ratio=vix_term_calm_ratio,
                stress_ratio=vix_term_stress_ratio,
                breadth_tickers=breadth_tickers,
            )
            db.save_macro_gate(conn, asof_date.isoformat(), macro_result)

        news_results: list[dict] = []
        if not skip_news:
            default_model = {
                "anthropic": config.ANTHROPIC_NEWS_MODEL,
                "openai": config.OPENAI_NEWS_MODEL,
                "gemini": config.GEMINI_NEWS_MODEL,
            }[news_provider]
            resolved_model = news_model or default_model
            api_key = news_api_key if news_api_key is not None else credentials.resolve_api_key(
                news_provider, interactive=False
            )
            for ticker in tickers:
                news_results.append(
                    news.get_or_analyze_news(
                        conn,
                        ticker,
                        asof_date,
                        news_window_days,
                        resolved_model,
                        provider=news_provider,
                        api_key=api_key,
                    )
                )

    return {
        "asof_date": asof_date.isoformat(),
        "valuations": [_valuation_to_dict(v) for v in valuations],
        "diffs": diffs,
        "warnings": warnings,
        "allocation": allocation,
        "aggregate_greeks": aggregate_greeks,
        "iv_environment": iv_environment,
        "upcoming_expiries": upcoming_expiries,
        "macro": macro_result,
        "news": news_results,
    }
