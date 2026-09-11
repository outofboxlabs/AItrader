"""Position book: load positions.json into typed Position objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


@dataclass
class Position:
    id: str
    asset_type: str  # "option" or "shares"
    ticker: str
    entry_price: float  # per share (options: per-share premium)
    contracts: float  # options: number of contracts; shares: share count
    # Optional: a brokerage screenshot rarely shows when a position was
    # opened (that's usually only visible in a lot-level activity view),
    # so vision-extracted rows commonly come back with this as null --
    # it's cosmetic (see pipeline.py), never used in any valuation math.
    entry_date: Optional[date] = None
    target_price: Optional[float] = None  # per share
    stop_price: Optional[float] = None  # per share
    option_type: Optional[str] = None  # "call" / "put"
    strike: Optional[float] = None
    expiry: Optional[date] = None

    @property
    def is_option(self) -> bool:
        return self.asset_type == "option"

    @property
    def multiplier(self) -> int:
        """Shares per contract: 100 for options, 1 for shares."""
        return 100 if self.is_option else 1

    @classmethod
    def from_dict(cls, raw: dict) -> "Position":
        asset_type = raw["asset_type"]
        ticker = raw["ticker"].upper()

        if asset_type == "option":
            option_type = raw["option_type"].lower()
            if option_type not in ("call", "put"):
                raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
            strike = float(raw["strike"])
            expiry = _parse_date(raw["expiry"])
            auto_id = f"{ticker}-{option_type[0].upper()}-{strike:g}-{expiry.isoformat()}"
        elif asset_type == "shares":
            option_type = None
            strike = None
            expiry = None
            auto_id = f"{ticker}-SHARES"
        else:
            raise ValueError(f"Unknown asset_type: {asset_type!r}")

        return cls(
            id=raw.get("id") or auto_id,
            asset_type=asset_type,
            ticker=ticker,
            entry_price=float(raw["entry_price"]),
            contracts=float(raw["contracts"]),
            entry_date=_parse_date(raw["entry_date"]) if raw.get("entry_date") else None,
            target_price=_optional_float(raw.get("target_price")),
            stop_price=_optional_float(raw.get("stop_price")),
            option_type=option_type,
            strike=strike,
            expiry=expiry,
        )


def load_positions(path: str) -> list[Position]:
    with open(path) as f:
        raw_positions = json.load(f)
    return [Position.from_dict(p) for p in raw_positions]
