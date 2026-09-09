#!/usr/bin/env python3
"""Options + equity portfolio monitor CLI.

Pulls live marks/greeks for every position in positions.json, saves a full
option-chain snapshot per underlying, and prints/stores a valuation +
portfolio-analytics summary.

Usage:
    python main.py [--asof YYYY-MM-DD] [--positions positions.json] ...
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import config
from portfolio_monitor import analytics, data, db, snapshots
from portfolio_monitor.models import load_positions
from portfolio_monitor.valuation import to_db_row, value_option_position, value_shares_position


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Options + equity portfolio monitor")
    parser.add_argument("--positions", default=config.POSITIONS_PATH, help="path to positions.json")
    parser.add_argument("--db", default=config.DB_PATH, help="path to the SQLite database")
    parser.add_argument("--snapshots-dir", default=config.SNAPSHOTS_DIR)
    parser.add_argument(
        "--asof",
        default=None,
        help="YYYY-MM-DD run date used to stamp storage/diffing; marks always come "
        "from the live feed (defaults to today)",
    )
    parser.add_argument("--risk-free-rate", type=float, default=config.RISK_FREE_RATE)
    parser.add_argument("--ticker-cap-pct", type=float, default=config.CONCENTRATION_TICKER_CAP_PCT)
    parser.add_argument("--sector-cap-pct", type=float, default=config.CONCENTRATION_SECTOR_CAP_PCT)
    parser.add_argument("--iv-lookback-days", type=int, default=config.IV_RANK_LOOKBACK_DAYS)
    parser.add_argument("--iv-min-history-days", type=int, default=config.IV_RANK_MIN_HISTORY_DAYS)
    parser.add_argument("--iv-rich-threshold", type=float, default=config.IV_RICH_THRESHOLD)
    parser.add_argument("--iv-cheap-threshold", type=float, default=config.IV_CHEAP_THRESHOLD)
    parser.add_argument("--expiry-warning-days", type=int, default=config.EXPIRY_WARNING_DAYS)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    asof_date = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today()
    pulled_at = datetime.now(timezone.utc).isoformat()

    positions = load_positions(args.positions)
    db.init_db(args.db)

    tickers = sorted({p.ticker for p in positions})
    option_tickers = sorted({p.ticker for p in positions if p.is_option})
    spots: dict[str, float] = {}
    chains: dict[str, dict] = {}

    with db.connect(args.db) as conn:
        for ticker in tickers:
            spots[ticker] = data.get_spot_price(ticker)

        for ticker in option_tickers:
            prior_dates = db.get_prior_snapshot_dates(conn, ticker, asof_date.isoformat())
            prior_keys = db.get_snapshot_keys(conn, ticker, prior_dates[0]) if prior_dates else set()

            chain = data.pull_full_chain(ticker)
            chains[ticker] = chain
            snapshots.save_snapshot_json(args.snapshots_dir, ticker, asof_date, chain)
            db.save_chain_snapshot_rows(conn, ticker, asof_date.isoformat(), pulled_at, chain)

            if prior_keys:
                current_keys = db.get_snapshot_keys(conn, ticker, asof_date.isoformat())
                diff = snapshots.diff_new_strikes_and_expiries(prior_keys, current_keys)
                if diff["new_expiries"]:
                    print(f"[{ticker}] new expiries since last run: {diff['new_expiries']}")
                if diff["new_strikes"]:
                    print(f"[{ticker}] {len(diff['new_strikes'])} new strike(s) since last run")

        valuations = []
        for position in positions:
            if position.is_option:
                quote = data.find_quote(
                    chains[position.ticker], position.expiry.isoformat(), position.option_type, position.strike
                )
                if quote is None:
                    print(f"WARNING: no live quote found for {position.id}, skipping")
                    continue
                valuation = value_option_position(
                    position, spots[position.ticker], quote, asof_date, args.risk_free_rate
                )
            else:
                valuation = value_shares_position(position, spots[position.ticker])
            valuations.append(valuation)

        db.save_valuations(conn, [to_db_row(v, asof_date) for v in valuations])
        print_valuation_table(valuations)

        allocation = analytics.compute_allocation(
            valuations, config.TICKER_SECTOR_MAP, args.ticker_cap_pct, args.sector_cap_pct
        )
        aggregate_greeks = analytics.compute_aggregate_greeks(valuations)
        iv_environment = analytics.compute_iv_environment(
            conn,
            valuations,
            asof_date,
            args.iv_lookback_days,
            args.iv_min_history_days,
            args.iv_rich_threshold,
            args.iv_cheap_threshold,
        )
        upcoming_expiries = analytics.compute_upcoming_expiries(valuations, args.expiry_warning_days)

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

    print_analytics_summary(allocation, aggregate_greeks, iv_environment, upcoming_expiries, args)


def print_valuation_table(valuations) -> None:
    print(f"\n=== Positions ({len(valuations)}) ===")
    for v in valuations:
        pnl_pct = f"{v.unrealized_pnl_pct:.1f}%" if v.unrealized_pnl_pct is not None else "n/a"
        dte = v.dte if v.dte is not None else "n/a"
        print(
            f"{v.position.id:28s} mark={v.mark:8.2f} value={v.current_value:10.2f} "
            f"pnl={v.unrealized_pnl:9.2f} ({pnl_pct}) dte={dte}"
        )


def print_analytics_summary(allocation, aggregate_greeks, iv_environment, upcoming_expiries, args) -> None:
    print("\n=== Allocation ===")
    print(f"Total value: {allocation['total_value']:.2f}")
    print("By ticker:")
    for r in allocation["by_ticker"]:
        flag = " <<< CONCENTRATED" if r["flagged"] else ""
        print(f"  {r['name']:8s} {r['pct_of_total']:6.1f}%{flag}")
    print("By sector:")
    for r in allocation["by_sector"]:
        flag = " <<< CONCENTRATED" if r["flagged"] else ""
        print(f"  {r['name']:20s} {r['pct_of_total']:6.1f}%{flag}")

    print("\n=== Aggregate Greeks ===")
    print(f"  Net delta (share-equiv): {aggregate_greeks['net_delta_shares']:.1f}")
    print(f"  Total daily theta ($):   {aggregate_greeks['total_daily_theta']:.2f}")
    print(f"  Net vega ($/1 IV pt):    {aggregate_greeks['net_vega']:.2f}")

    print("\n=== IV Environment ===")
    for r in iv_environment:
        if r["status"] == "building history":
            print(
                f"  {r['position_id']:28s} iv={r['current_iv']:.3f} "
                f"building history ({r['history_days']}/{args.iv_min_history_days} days)"
            )
        else:
            tag = "RICH" if r["rich"] else ("CHEAP" if r["cheap"] else "")
            print(
                f"  {r['position_id']:28s} iv={r['current_iv']:.3f} "
                f"rank={r['iv_rank']:.0f} pct={r['iv_percentile']:.0f} {tag}"
            )

    print(f"\n=== Upcoming Expiries (<= {args.expiry_warning_days}d) ===")
    flagged = [r for r in upcoming_expiries if r["within_threshold"]]
    if not flagged:
        print("  none")
    for r in flagged:
        print(f"  {r['position_id']:28s} expiry={r['expiry']} dte={r['dte']}")


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
