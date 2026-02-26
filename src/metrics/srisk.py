"""
srisk.py — SRISK (Brownlees & Engle, 2017)

Formula:
    SRISK_i = max(0, k · Debt_i - (1 - k) · W_i · (1 - LRMES_i))

where:
    k       = prudential capital ratio (default 8%, Basel III Tier 1)
    Debt_i  = total liabilities (USD billions, from quarterly balance sheet)
    W_i     = market capitalisation (USD billions)
    LRMES_i = Long-Run Marginal Expected Shortfall (from mes.py)

Interpretation:
    SRISK is the expected capital shortfall of institution i in the event
    of a systemic crisis (market falls 40% over 6 months).
    SRISK > 0 ⟹ capital shortfall (undercapitalised in crisis).
    Sum of positive SRISK across all banks ≈ system capital gap.

SRISK share:
    SRISK_share_i = SRISK_i / Σ max(0, SRISK_j) × 100%
    Measures bank i's contribution to total systemic capital shortfall.
"""

import numpy as np
import pandas as pd

from src.config import cfg


# ---------------------------------------------------------------------------
# Point-in-time SRISK
# ---------------------------------------------------------------------------
def calc_srisk(
    market_cap_usd_bn: float,
    debt_usd_bn: float,
    lrmes: float,
    k: float | None = None,
) -> float:
    """
    Compute SRISK for a single bank at a point in time.

    Args:
        market_cap_usd_bn : Market capitalisation in USD billions.
        debt_usd_bn       : Total liabilities in USD billions.
        lrmes             : Long-Run MES (fraction, 0 ≤ lrmes ≤ 1).
        k                 : Prudential capital ratio (overrides cfg.srisk_k).

    Returns:
        SRISK in USD billions (always ≥ 0).
    """
    k = k if k is not None else cfg.srisk_k

    if any(np.isnan(v) for v in [market_cap_usd_bn, debt_usd_bn, lrmes]):
        return float("nan")
    if market_cap_usd_bn <= 0 or debt_usd_bn <= 0:
        return float("nan")

    srisk = k * debt_usd_bn - (1 - k) * market_cap_usd_bn * (1 - lrmes)
    return float(max(0.0, srisk))


# ---------------------------------------------------------------------------
# Rolling SRISK series
# ---------------------------------------------------------------------------
def calc_srisk_series(
    market_cap_series: pd.Series,
    debt_series: pd.Series,
    lrmes_series: pd.Series,
    k: float | None = None,
) -> pd.Series:
    """
    Compute a daily time series of SRISK.

    Args:
        market_cap_series : Daily market cap in USD billions (DatetimeIndex).
        debt_series       : Daily liabilities in USD billions (DatetimeIndex).
        lrmes_series      : Daily LRMES values (DatetimeIndex).
        k                 : Prudential capital ratio (overrides cfg.srisk_k).

    Returns:
        pd.Series of SRISK values indexed by date.
    """
    k = k if k is not None else cfg.srisk_k

    df = pd.DataFrame({
        "mcap": market_cap_series,
        "debt": debt_series,
        "lrmes": lrmes_series,
    }).dropna()

    if df.empty:
        return pd.Series(dtype=float)

    srisk_vals = (
        k * df["debt"]
        - (1 - k) * df["mcap"] * (1 - df["lrmes"])
    ).clip(lower=0.0)

    return srisk_vals.rename("srisk_usd_bn")


# ---------------------------------------------------------------------------
# SRISK ranking and aggregation
# ---------------------------------------------------------------------------
def calc_srisk_shares(srisk_values: dict[str, float]) -> dict[str, float]:
    """
    Compute each bank's share of total positive SRISK.

    Args:
        srisk_values : {bank_id: srisk_usd_bn}

    Returns:
        {bank_id: srisk_share_pct}  (sum of positive shares = 100%)
    """
    positives = {k: v for k, v in srisk_values.items()
                 if isinstance(v, float) and not np.isnan(v) and v > 0}
    total = sum(positives.values())
    if total == 0:
        return {k: 0.0 for k in srisk_values}

    return {k: round(v / total * 100, 4) for k, v in srisk_values.items()
            if k in positives}


def system_srisk(srisk_values: dict[str, float]) -> float:
    """Total system SRISK = sum of all positive SRISK values."""
    return sum(v for v in srisk_values.values()
               if isinstance(v, float) and not np.isnan(v) and v > 0)
