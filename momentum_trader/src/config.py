"""Typed configuration loaded from config.yaml.

All strategy/execution/risk constants live in config.yaml rather than being
scattered through the code. This module defines dataclasses that mirror the
YAML structure and a loader that fills in defaults for any missing keys.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a thin convenience only
    load_dotenv = None


@dataclass
class DataConfig:
    source: str = "local"  # "alpaca" | "local"
    local_path: str = "data/cache/AAPL_sample.parquet"
    cache_dir: str = "data/cache"
    start_date: str = "2025-01-01"
    end_date: str = "2025-08-31"
    feed: str = "iex"


@dataclass
class SessionConfig:
    timezone: str = "America/New_York"
    open: str = "09:30"
    close: str = "16:00"
    stop_new_entries: str = "15:00"
    force_close: str = "15:55"


@dataclass
class StrategyConfig:
    pullback_pct: float = 0.005
    take_profit_pct: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 1.5
    max_hold_minutes: int = 120


@dataclass
class FiltersConfig:
    macro_news: bool = True
    earnings: bool = True
    earnings_exclude_days_before: int = 0
    earnings_exclude_days_after: int = 0
    vwap: bool = False
    ema: bool = False
    rsi: bool = False
    rsi_oversold_threshold: float = 40.0
    relative_volume: bool = False
    relative_volume_threshold: float = 1.2
    min_volume: bool = False
    min_volume_threshold: float = 500
    market_trend: bool = False


@dataclass
class ProfitProtectionConfig:
    enabled: bool = False
    breakeven_trigger_pct: float = 0.005
    breakeven_buffer_pct: float = 0.0005
    lock_trigger_pct: float = 0.0075
    lock_pct: float = 0.003


@dataclass
class ExecutionConfig:
    entry_method: str = "next_bar_open"  # "next_bar_open" | "current_bar_close"
    intrabar_priority: str = "conservative"  # "conservative" | "optimistic" | "skip"
    slippage_bps_per_side: float = 1.0
    commission_per_share: float = 0.0
    commission_per_order: float = 0.0


@dataclass
class RiskConfig:
    starting_equity: float = 100_000.0
    sizing_method: str = "risk"  # "risk" | "fixed_dollar"
    risk_per_trade_pct: float = 0.0025
    fixed_dollar_amount: float = 10_000.0
    max_position_pct: float = 0.20
    allow_fractional_shares: bool = False
    max_trades_per_day: int = 5
    max_daily_loss_pct: float = 0.015
    max_consecutive_losses: int = 3


@dataclass
class CalendarsConfig:
    excluded_dates_path: str = "data/excluded_dates.csv"
    earnings_dates_path: str = "data/earnings_dates.csv"


@dataclass
class BenchmarkConfig:
    symbols: list = field(default_factory=lambda: ["AAPL", "SPY"])


@dataclass
class OptimizerConfig:
    train_start: str = "2025-01-01"
    train_end: str = "2025-04-30"
    validation_start: str = "2025-05-01"
    validation_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    test_end: str = "2025-08-31"
    min_trades_for_ranking: int = 15
    grid: dict = field(default_factory=dict)


@dataclass
class WalkForwardConfig:
    train_months: int = 3
    test_months: int = 1
    step_months: int = 1


@dataclass
class BacktestConfig:
    symbol: str = "AAPL"
    data: DataConfig = field(default_factory=DataConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    profit_protection: ProfitProtectionConfig = field(default_factory=ProfitProtectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    calendars: CalendarsConfig = field(default_factory=CalendarsConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    def clone_with_strategy_overrides(self, **overrides: Any) -> "BacktestConfig":
        """Return a deep copy of this config with strategy/execution/filter fields overridden.

        Used by the optimizer to build one config per parameter combination without
        mutating the base config that other sweep members are reading from.
        """
        new_cfg = copy.deepcopy(self)
        for key, value in overrides.items():
            if hasattr(new_cfg.strategy, key):
                setattr(new_cfg.strategy, key, value)
            elif hasattr(new_cfg.filters, key):
                setattr(new_cfg.filters, key, value)
            elif hasattr(new_cfg.profit_protection, key):
                setattr(new_cfg.profit_protection, key, value)
            else:
                raise KeyError(f"Unknown override key: {key}")
        return new_cfg

    def strategy_parameters_dict(self) -> dict:
        """Flat dict of the parameters that define a strategy variant (for trade logs)."""
        return {
            "pullback_pct": self.strategy.pullback_pct,
            "take_profit_pct": self.strategy.take_profit_pct,
            "atr_period": self.strategy.atr_period,
            "atr_multiplier": self.strategy.atr_multiplier,
            "max_hold_minutes": self.strategy.max_hold_minutes,
            "vwap_filter": self.filters.vwap,
            "ema_filter": self.filters.ema,
            "rsi_filter": self.filters.rsi,
            "relative_volume_filter": self.filters.relative_volume,
            "min_volume_filter": self.filters.min_volume,
            "market_trend_filter": self.filters.market_trend,
            "profit_protection": self.profit_protection.enabled,
            "intrabar_priority": self.execution.intrabar_priority,
            "entry_method": self.execution.entry_method,
        }


def _merge_dataclass(dc_instance: Any, values: Optional[dict]) -> Any:
    if not values:
        return dc_instance
    return replace(dc_instance, **{k: v for k, v in values.items() if hasattr(dc_instance, k)})


def load_config(path: str | Path = "config.yaml", env_path: Optional[str | Path] = None) -> BacktestConfig:
    """Load config.yaml (and optionally a .env file) into a BacktestConfig."""
    if load_dotenv is not None:
        load_dotenv(env_path) if env_path else load_dotenv()

    path = Path(path)
    raw: dict = {}
    if path.exists():
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = BacktestConfig()
    cfg.symbol = raw.get("symbol", cfg.symbol)
    cfg.data = _merge_dataclass(DataConfig(), raw.get("data"))
    cfg.session = _merge_dataclass(SessionConfig(), raw.get("session"))
    cfg.strategy = _merge_dataclass(StrategyConfig(), raw.get("strategy"))
    cfg.filters = _merge_dataclass(FiltersConfig(), raw.get("filters"))
    cfg.profit_protection = _merge_dataclass(ProfitProtectionConfig(), raw.get("profit_protection"))
    cfg.execution = _merge_dataclass(ExecutionConfig(), raw.get("execution"))
    cfg.risk = _merge_dataclass(RiskConfig(), raw.get("risk"))
    cfg.calendars = _merge_dataclass(CalendarsConfig(), raw.get("calendars"))
    cfg.benchmark = _merge_dataclass(BenchmarkConfig(), raw.get("benchmark"))
    cfg.optimizer = _merge_dataclass(OptimizerConfig(), raw.get("optimizer"))
    cfg.walk_forward = _merge_dataclass(WalkForwardConfig(), raw.get("walk_forward"))
    return cfg


def get_alpaca_credentials() -> tuple[Optional[str], Optional[str]]:
    """Read Alpaca credentials from the environment (populated via .env or shell)."""
    return os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
