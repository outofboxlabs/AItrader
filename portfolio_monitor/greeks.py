"""Black-Scholes greeks, computed locally with only the stdlib `math` module."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float  # $ per share, per calendar day (already negative for long premium decay)
    vega: float  # $ per share, per 1 vol point (i.e. per 0.01 change in IV)


_ZERO_GREEKS = Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(
    spot: float,
    strike: float,
    years_to_expiry: float,
    risk_free_rate: float,
    iv: float,
    option_type: str,
) -> Greeks:
    """Per-share Black-Scholes greeks for one option contract.

    Returns all-zero greeks for degenerate inputs (expired, zero/negative
    IV, or non-positive spot/strike) rather than raising, since a position
    can legitimately reach expiry while still being displayed.
    """
    if years_to_expiry <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return _ZERO_GREEKS

    sqrt_t = math.sqrt(years_to_expiry)
    d1 = (
        math.log(spot / strike) + (risk_free_rate + 0.5 * iv * iv) * years_to_expiry
    ) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    pdf_d1 = _norm_pdf(d1)
    is_call = option_type.lower() == "call"

    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0

    gamma = pdf_d1 / (spot * iv * sqrt_t)

    discounted_strike = strike * math.exp(-risk_free_rate * years_to_expiry)
    if is_call:
        theta_annual = (
            -(spot * pdf_d1 * iv) / (2 * sqrt_t)
            - risk_free_rate * discounted_strike * _norm_cdf(d2)
        )
    else:
        theta_annual = (
            -(spot * pdf_d1 * iv) / (2 * sqrt_t)
            + risk_free_rate * discounted_strike * _norm_cdf(-d2)
        )
    theta_per_day = theta_annual / 365.0

    vega_per_point = spot * pdf_d1 * sqrt_t * 0.01

    return Greeks(delta=delta, gamma=gamma, theta=theta_per_day, vega=vega_per_point)
