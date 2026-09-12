# AAPL Intraday Pullback/Reversal Backtester

A rigorous backtesting framework for an intraday AAPL mean-reversion / reversal
strategy on 1-minute bars. **This project only backtests — it does not place
live orders.** The goal is to find out, with realistic costs and without
lookahead bias, whether the strategy has positive expectancy before anyone
considers trading it with real money.

## Is the baseline strategy profitable?

**On the bundled synthetic sample data: no.** The default configuration
(0.50% pullback, 1.00% target, 1.5x ATR stop, 120-minute time stop) loses
money net of costs (see "First experiment" below). That result is expected
and appropriate to report honestly — see *Limitations* for why the sample
data specifically cannot validate or refute the strategy's real edge. Running
this same pipeline against real Alpaca history is what actually answers the
question; the parameter sweep and walk-forward tooling exist to prevent
fooling yourself once real data is in.

## Project structure

```
aapl_intraday_backtester/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml                 # every strategy/execution/risk parameter lives here
├── data/
│   ├── excluded_dates.csv      # macro-news exclusion calendar (illustrative — see note below)
│   ├── earnings_dates.csv      # AAPL earnings exclusion calendar (illustrative)
│   └── cache/                  # downloaded / generated bar data (parquet, gitignored)
├── src/
│   ├── config.py                # typed config dataclasses + YAML loader
│   ├── utils.py                 # logging, timezone/session helpers
│   ├── data_loader.py           # Alpaca REST client + local CSV/parquet loader, RTH filter, caching
│   ├── calendars.py              # macro/earnings excluded-date providers
│   ├── indicators.py             # ATR, VWAP, EMA9/20, RSI14, rel-volume, rolling hi/lo (all causal)
│   ├── signals.py                 # pullback/reversal entry pattern + optional filters
│   ├── strategy.py                # position state machine: stops, targets, profit protection
│   ├── execution.py               # slippage, commissions, intrabar ambiguity resolution
│   ├── risk.py                    # position sizing + daily risk-limit tracking
│   ├── backtester.py              # bar-by-bar event loop (the engine)
│   ├── optimizer.py               # grid search, ranking, train/val/test split, walk-forward
│   ├── metrics.py                 # performance metrics
│   └── report.py                  # matplotlib charts + results/report.html
├── scripts/
│   ├── generate_sample_data.py    # synthetic 1-min OHLCV generator (no credentials needed)
│   ├── download_data.py           # Alpaca downloader (needs credentials)
│   ├── run_backtest.py            # single baseline backtest
│   ├── run_parameter_sweep.py     # train/validation/test grid search
│   └── run_walk_forward.py        # rolling walk-forward test
├── tests/                          # pytest suite (36 tests)
└── results/                        # trades.csv, equity_curve.csv, parameter_results.csv,
                                     # best_configs.csv, report.html, charts/
```

## Setup

```bash
cd aapl_intraday_backtester
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Alpaca credentials (optional — only needed for real data)

```bash
cp .env.example .env
# edit .env and set ALPACA_API_KEY / ALPACA_SECRET_KEY
```

If you don't have Alpaca credentials, leave `config.yaml`'s `data.source` set
to `local` (the default) and use the synthetic sample dataset instead.

## Commands

```bash
# 1. Get data — EITHER:
python scripts/generate_sample_data.py --start 2025-01-01 --end 2025-08-31   # synthetic, no credentials
# OR, with Alpaca credentials and data.source: alpaca in config.yaml:
python scripts/download_data.py --start 2025-01-01 --end 2025-08-31

# 2. Run the unit tests
pytest -q

# 3. Run one backtest (baseline params from config.yaml)
python scripts/run_backtest.py

# 4. Run the parameter sweep (train -> validation for selection, then out-of-sample test)
python scripts/run_parameter_sweep.py

# 5. Run walk-forward testing
python scripts/run_walk_forward.py

# 6. View the report
open results/report.html          # macOS
xdg-open results/report.html      # Linux
# or just open the file in a browser
```

Every script accepts `--config path/to/config.yaml` to point at an alternate configuration.

## First experiment (baseline)

Configuration exactly as specified: 0.50% pullback, 1.00% take-profit,
1.5× ATR(14) stop, 120-minute max hold, entries next-bar-open, conservative
intrabar handling, 1bp slippage per side, macro + earnings dates excluded,
new entries stop at 3:00pm ET, force-close at 3:55pm ET.

On the bundled synthetic sample data (2025-01-01 → 2025-08-31, 173 sessions):

| Metric | Value |
|---|---|
| Total trades | 380 |
| Win rate | 20.0% |
| Profit factor | 0.93 |
| Expectancy / trade | -0.007% |
| Net P&L | -$511 |
| Total return | -0.51% |
| Max drawdown | -1.04% |
| Sharpe ratio | -0.51 |

This is exactly what should happen on data with no genuine mean-reversion
edge (the synthetic generator is a near-random walk): the strategy loses a
small amount consistent with transaction costs and an unfavorable exit-timing
distribution, rather than showing an inflated, too-good-to-be-true return
that would suggest a lookahead-bias bug. Re-run `run_backtest.py` after
downloading real Alpaca history to get a meaningful answer about real edge.

The comparison sweep (ATR multiplier 1.0/1.5/2.0/2.5, time stop 30/60/90/120,
take-profit 0.40/0.50/0.75/1.00/1.25%, pullback 0.25/0.50/0.75/1.00%) is
wired up via `python scripts/run_parameter_sweep.py`, which selects a
configuration using only train+validation data and reports its result on the
held-out test window in `results/report.html` and `results/best_configs.csv`.

## Key implementation choices

* **No lookahead, by construction.** Every indicator (`src/indicators.py`) is
  a causal rolling/expanding computation — row N only ever depends on row
  N and earlier. A signal confirmed at row N is filled at row N+1's open
  (`execution.entry_method: next_bar_open`, the default), never at row N's
  own close, because a live strategy cannot transact at a price the instant
  it observes the bar that produced that price. `tests/test_no_lookahead.py`
  enforces this two ways: truncating the bar stream never changes a signal
  computed on the untruncated data, and re-running the optimizer with
  different (but otherwise identical) out-of-sample test data produces
  byte-identical train/validation grid-search results.

* **Intrabar ambiguity is resolved explicitly, not assumed away.** A single
  1-minute bar can have `low <= stop` and `high >= target` simultaneously;
  the true order of events is unknowable from OHLC alone. The default
  (`execution.intrabar_priority: conservative`) assumes the stop was hit
  first. `optimistic` and `skip` (discard the trade rather than guess) are
  available for sensitivity analysis — see `src/execution.py` and
  `tests/test_intrabar_logic.py`.

* **Indicator reset behavior.** VWAP and the rolling intraday high/low reset
  every session (they're inherently intraday concepts). ATR/EMA/RSI/relative
  volume are carried as continuous rolling series across session boundaries,
  the way a live system would maintain them — this only affects the first
  session's warm-up period.

* **Relative volume is a simplified proxy**: current bar volume vs. its own
  trailing 20-bar average, not a true time-of-day-bucketed comparison against
  prior sessions. A production system should replace this with a proper
  time-of-day baseline.

* **Market-trend filter is a simplified proxy**: EMA20 slope over the last 5
  bars, not a broader regime/SPY-relative measure. Documented in
  `src/signals.py`; swap in something more sophisticated before relying on it.

* **Gross vs. net returns**: "gross" uses the raw trigger/fill price with no
  slippage or commission; "net" applies slippage to both entry and exit and
  subtracts commissions on both legs. Both are recorded per trade and rolled
  up in the metrics report.

* **Costs are applied on every entry and exit**: `slippage_bps_per_side`
  (default 1bp) plus `commission_per_share`/`commission_per_order` (default
  $0, i.e. commission-free brokers). Position equity only changes on
  realized (net) P&L; the equity curve marks open positions to the current
  bar's close for reporting.

* **Daily risk halts trading, not just future signals mid-loss**: in addition
  to blocking new entries once `max_trades_per_day` / `max_daily_loss_pct` /
  `max_consecutive_losses` are hit, the backtester also force-closes an
  already-open position (`exit_reason = daily_risk_stop`) if holding it to
  the current bar's close would itself breach the daily loss cap.

* **Position sizing never uses leverage**: both fixed-dollar and risk-based
  sizing are capped at `min(equity, equity * max_position_pct)` in notional
  terms, then floored to a whole share by default.

* **Train/validation/test discipline**: `src/optimizer.py`'s grid search only
  ever touches the train and validation windows from `config.yaml`'s
  `optimizer` block; the chosen configuration is applied *unchanged* to the
  test window afterward. Walk-forward mode (`run_walk_forward.py`) repeats
  this on rolling windows and reports per-window and summary consistency
  rather than a single cherry-picked number.

* **Ranking is multi-metric, not total-return-only**: `rank_configs` averages
  percentile ranks across net return, profit factor, drawdown, Sharpe, and
  expectancy, and hard-excludes any configuration below
  `optimizer.min_trades_for_ranking` trades from ever ranking "best" —
  per section 29 of the spec, a strategy is not judged on return or win rate
  alone.

## Assumptions and limitations

* **The bundled sample data is synthetic**, generated by
  `scripts/generate_sample_data.py` as a mean-reverting random walk shaped
  like AAPL's intraday volatility profile. It exists purely so the full
  pipeline (signals → backtest → optimizer → report) can be exercised without
  Alpaca credentials. **It has no real predictive relationship to AAPL and
  must not be used to draw conclusions about the strategy's real edge** —
  only a backtest against genuine historical bars (via `download_data.py`)
  can do that.

* **`data/excluded_dates.csv` and `data/earnings_dates.csv` are illustrative
  placeholders**, not a verified economic/earnings calendar. Before relying
  on the macro or earnings filter, replace them with dates sourced from an
  official calendar (Fed/BLS/BEA release schedules for macro events; AAPL's
  actual reported earnings dates). `src/calendars.py` is architected so a
  live economic-calendar API/provider can implement the same
  `ExcludedDateProvider` interface later without touching signal or
  backtester code.

* **Trading-day calendar approximation**: the sample-data generator and the
  optimizer's date-window slicing use weekday dates (`pandas.bdate_range`),
  which does not account for U.S. market holidays. A real Alpaca download
  naturally only contains bars for days the market was actually open.

* **Relative volume and market-trend filters are simplified proxies** (see
  above) — treat them as a starting point for experimentation, not a
  finished signal.

* **ATR is a single continuous EWMA from the first available true range**,
  not the classic Wilder SMA-seeded version. The difference converges away
  after a few dozen bars and does not affect any conclusion at daily/monthly
  time scales, but is worth knowing if you're cross-checking against another
  platform's ATR values bar-for-bar.

* **This backtester assumes full, immediate fills** at the computed price
  (subject to slippage) for any size the sizing model requests. It does not
  model partial fills, resting-order queue position, or market impact beyond
  the flat slippage/bps assumption — a reasonable approximation at the
  position sizes implied by `max_position_pct`, less so at large size.

* **No walk-forward parameter stability guardrails**: `run_walk_forward.py`
  reports per-window results and summary statistics but does not itself flag
  or penalize windows that pick wildly different "best" parameters — inspect
  `results/walk_forward_results.csv` for consistency yourself before trusting
  a configuration.

* **Do not treat any result on the synthetic sample data as evidence the
  strategy works or doesn't.** The honest, correct use of this project is:
  point it at real historical data, run the train/validation/test sweep, and
  only trust the out-of-sample number — and even then, only after walk-forward
  testing shows consistency across multiple, non-overlapping windows.
