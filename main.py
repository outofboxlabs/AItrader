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
from datetime import date, datetime

import config
from portfolio_monitor import credentials, news, pipeline


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

    parser.add_argument("--skip-macro", action="store_true", help="skip the deterministic macro gate")
    parser.add_argument("--macro-lookback-days", type=int, default=config.MACRO_LOOKBACK_DAYS)
    parser.add_argument("--vix-term-calm-ratio", type=float, default=config.VIX_TERM_CALM_RATIO)
    parser.add_argument("--vix-term-stress-ratio", type=float, default=config.VIX_TERM_STRESS_RATIO)

    parser.add_argument(
        "--skip-news",
        action="store_true",
        help="skip news analysis (the only paid part of the system)",
    )
    parser.add_argument("--news-window-days", type=int, default=config.NEWS_WINDOW_DAYS)
    parser.add_argument(
        "--news-provider",
        choices=["anthropic", "openai", "gemini"],
        default=config.NEWS_PROVIDER,
    )
    parser.add_argument(
        "--news-model",
        default=None,
        help="defaults to config.ANTHROPIC_NEWS_MODEL / OPENAI_NEWS_MODEL / GEMINI_NEWS_MODEL "
        "depending on --news-provider",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=None,
        help="defaults to the ANTHROPIC_API_KEY environment variable, then a saved key (see --interactive)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="defaults to the OPENAI_API_KEY environment variable, then a saved key (see --interactive)",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="defaults to the GEMINI_API_KEY environment variable, then a saved key (see --interactive)",
    )
    parser.add_argument(
        "--choose-model",
        action="store_true",
        help="prompt with a numbered list of live models from --news-provider before running "
        "(skip for unattended/scheduled runs -- it waits on terminal input)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for the news platform (Anthropic/OpenAI/Gemini), its API key if none is "
        "saved yet (offering to save it to .credentials.json), and the model to use -- implies "
        "--choose-model. Skip for unattended/scheduled runs.",
    )
    return parser.parse_args(argv)


_NEWS_PROVIDERS = ["anthropic", "openai", "gemini"]
_NEWS_PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI (GPT)",
    "gemini": "Google (Gemini)",
}


def prompt_for_provider(default: str) -> str:
    """Numbered platform picker. Returns `default` on empty/invalid input."""
    print("\nWhich AI platform should analyze news?")
    for i, key in enumerate(_NEWS_PROVIDERS, start=1):
        marker = "  (current default)" if key == default else ""
        print(f"  {i}. {_NEWS_PROVIDER_LABELS[key]}{marker}")

    choice = input(f"Select a platform [1-{len(_NEWS_PROVIDERS)}] (Enter to keep the default): ").strip()
    if not choice:
        return default
    if choice.isdigit() and 1 <= int(choice) <= len(_NEWS_PROVIDERS):
        return _NEWS_PROVIDERS[int(choice) - 1]

    print("Invalid selection; keeping the default.")
    return default


def prompt_for_model(provider: str, api_key: str | None) -> str | None:
    """Fetch live models for `provider`, print them numbered, and ask the
    user to pick one. Returns None (caller falls back to the configured
    default) on any failure, empty input, or invalid selection."""
    try:
        models = news.list_models(provider, api_key=api_key)
    except Exception as exc:
        print(f"Could not fetch model list from {provider} ({exc}); using the configured default.")
        return None

    if not models:
        print(f"No models returned for {provider}; using the configured default.")
        return None

    print(f"\nAvailable {provider} models:")
    for i, model_id in enumerate(models, start=1):
        print(f"  {i}. {model_id}")

    choice = input(f"Select a model [1-{len(models)}] (Enter to keep the configured default): ").strip()
    if not choice:
        return None

    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]

    print("Invalid selection; using the configured default.")
    return None


def run(args: argparse.Namespace) -> None:
    asof_date = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today()

    news_api_key = None
    if not args.skip_news:
        if args.interactive:
            args.news_provider = prompt_for_provider(args.news_provider)

        cli_key = {
            "anthropic": args.anthropic_api_key,
            "openai": args.openai_api_key,
            "gemini": args.gemini_api_key,
        }[args.news_provider]
        news_api_key = credentials.resolve_api_key(args.news_provider, cli_value=cli_key, interactive=args.interactive)

        if args.interactive or args.choose_model:
            chosen_model = prompt_for_model(args.news_provider, news_api_key)
            if chosen_model:
                args.news_model = chosen_model

    result = pipeline.run_full_analysis(
        positions_path=args.positions,
        db_path=args.db,
        snapshots_dir=args.snapshots_dir,
        asof_date=asof_date,
        risk_free_rate=args.risk_free_rate,
        ticker_cap_pct=args.ticker_cap_pct,
        sector_cap_pct=args.sector_cap_pct,
        iv_lookback_days=args.iv_lookback_days,
        iv_min_history_days=args.iv_min_history_days,
        iv_rich_threshold=args.iv_rich_threshold,
        iv_cheap_threshold=args.iv_cheap_threshold,
        expiry_warning_days=args.expiry_warning_days,
        skip_macro=args.skip_macro,
        macro_lookback_days=args.macro_lookback_days,
        vix_term_calm_ratio=args.vix_term_calm_ratio,
        vix_term_stress_ratio=args.vix_term_stress_ratio,
        skip_news=args.skip_news,
        news_window_days=args.news_window_days,
        news_provider=args.news_provider,
        news_model=args.news_model,
        news_api_key=news_api_key,
    )

    for d in result["diffs"]:
        if d["new_expiries"]:
            print(f"[{d['ticker']}] new expiries since last run: {d['new_expiries']}")
        if d["new_strikes"]:
            print(f"[{d['ticker']}] {d['new_strikes']} new strike(s) since last run")
    for w in result["warnings"]:
        print(f"WARNING: {w}")

    print_valuation_table(result["valuations"])
    print_analytics_summary(result["allocation"], result["aggregate_greeks"], result["iv_environment"], result["upcoming_expiries"], args)
    if result["macro"] is not None:
        print_macro_summary(result["macro"])
    if result["news"]:
        print_news_summary(result["news"])


def print_valuation_table(valuations: list[dict]) -> None:
    print(f"\n=== Positions ({len(valuations)}) ===")
    for v in valuations:
        pnl_pct = f"{v['unrealized_pnl_pct']:.1f}%" if v["unrealized_pnl_pct"] is not None else "n/a"
        dte = v["dte"] if v["dte"] is not None else "n/a"
        print(
            f"{v['position_id']:28s} mark={v['mark']:8.2f} value={v['current_value']:10.2f} "
            f"pnl={v['unrealized_pnl']:9.2f} ({pnl_pct}) dte={dte}"
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


def print_macro_summary(macro_result: dict) -> None:
    vix_level = macro_result["vix_level"]
    term_structure = macro_result["term_structure"]
    breadth = macro_result["breadth"]
    credit_spread = macro_result["credit_spread"]

    print(f"\n=== Macro Gate: {macro_result['score']:.1f}/100 (higher = calmer) ===")
    print(
        f"  VIX level:       vix={vix_level['vix']:.1f} percentile={vix_level['vix_percentile']:.0f} "
        f"score={vix_level['score']:.1f}"
    )
    ratio = term_structure["ratio"]
    ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
    print(
        f"  Term structure:  vix={term_structure['vix']:.1f} vix3m={term_structure['vix3m']:.1f} "
        f"ratio={ratio_str} score={term_structure['score']:.1f}"
    )
    print(
        f"  Breadth:         {breadth['pct_above_200dma']:.0f}% above 200dma "
        f"({breadth['constituents']} names) score={breadth['score']:.1f}"
    )
    print(
        f"  Credit spread:   HYG/TLT={credit_spread['credit_ratio']:.4f} "
        f"percentile={credit_spread['credit_percentile']:.0f} score={credit_spread['score']:.1f}"
    )


def print_news_summary(news_results: list[dict]) -> None:
    provider = news_results[0].get("provider", "?") if news_results else "?"
    print(f"\n=== News Analysis via {provider} (informational only -- not a trade signal) ===")
    for r in news_results:
        if r["status"] != "ok":
            print(f"  {r['ticker']:6s} skipped ({r['status']})")
            continue
        flag = " <<< AFFECTS POSITION" if r.get("position_flag") else ""
        print(f"  {r['ticker']:6s} [{r.get('sentiment', 'n/a')}]{flag}")
        if r.get("summary"):
            print(f"    {r['summary']}")
        if r.get("position_flag") and r.get("position_flag_reason"):
            print(f"    why: {r['position_flag_reason']}")


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
