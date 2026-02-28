"""
mes.py — Marginal Expected Shortfall (MES) and Long-Run MES (LRMES)

Methodology (NYU V-Lab / Acharya et al. 2010):

MES:
    The average return of institution i on days when the market (system)
    return falls below its c-th percentile (VaR threshold).

    MES_i = E[r_i | r_m ≤ VaR_c(r_m)]

LRMES (Long-Run MES) — approximation via Brownlees & Engle (2017):
    Uses a log-normal / bivariate GBM closed-form approximation:

        LRMES_i ≈ 1 - exp(log(1 - D) * β_i)

    where:
        D   = hypothetical cumulative market decline (default 40%)
        β_i = OLS market beta of bank i vs regional index
              = Cov(r_i, r_m) / Var(r_m)  [encodes ρ · σ_i/σ_m]
        h   = implicit horizon encoded in D (default 22 trading days)

    Derivation: under bivariate GBM, E[R_i^h | R_m^h = log(1-D)] ≈ β_OLS · log(1-D),
    so LRMES = 1 - exp(β_OLS · log(1-D)).  β_OLS already incorporates
    the correlation ρ (β_OLS = ρ·σ_i/σ_m), so ρ must NOT be multiplied
    separately, and no √h scaling is applied since D represents the full
    horizon loss.

    Reference: Brownlees & Engle (2017), "SRISK: A Conditional Capital
    Shortfall Measure of Systemic Risk", Review of Financial Studies.

Inputs:
    bank_returns  : pd.Series, daily log-returns of bank i
    index_returns : pd.Series, daily log-returns of market system m

Outputs:
    MES  : float (daily)
    LRMES: float (long-run projection, as used in SRISK)
"""

import numpy as np
import pandas as pd

from src.config import cfg


# ---------------------------------------------------------------------------
# Simple quantile-based MES
# ---------------------------------------------------------------------------
def calc_mes(
    bank_returns: pd.Series,
    index_returns: pd.Series,
    tail_pct: float | None = None,
) -> float:
    """
    Compute MES: mean bank return on index tail days.

    Args:
        bank_returns  : Daily simple or log returns of the bank.
        index_returns : Daily returns of the regional market index.
        tail_pct      : Left-tail threshold (default from cfg).

    Returns:
        MES as a float (negative = loss during system stress).
    """
    tail_pct = tail_pct if tail_pct is not None else cfg.mes_tail_pct

    aligned = _align(bank_returns, index_returns)
    if aligned.empty or len(aligned) < 30:
        return float("nan")

    threshold = aligned["index"].quantile(tail_pct)
    tail_days = aligned[aligned["index"] <= threshold]

    if tail_days.empty:
        return float("nan")

    return float(tail_days["bank"].mean())


# ---------------------------------------------------------------------------
# Rolling MES series
# ---------------------------------------------------------------------------
def calc_mes_rolling(
    bank_returns: pd.Series,
    index_returns: pd.Series,
    window: int | None = None,
    tail_pct: float | None = None,
) -> pd.Series:
    """
    Return a daily time series of MES computed over a rolling window.

    Args:
        bank_returns  : Daily returns of the bank.
        index_returns : Daily returns of the regional market index.
        window        : Rolling window size in days (default: cfg.covar_window).
        tail_pct      : Left-tail threshold (default: cfg.mes_tail_pct).

    Returns:
        pd.Series indexed by date, values = MES for each day.
    """
    window = window or cfg.covar_window
    tail_pct = tail_pct if tail_pct is not None else cfg.mes_tail_pct

    aligned = _align(bank_returns, index_returns)
    if aligned.empty:
        return pd.Series(dtype=float)

    results = {}
    for i in range(window, len(aligned)):
        window_data = aligned.iloc[i - window:i]
        thresh = window_data["index"].quantile(tail_pct)
        tail = window_data[window_data["index"] <= thresh]["bank"]
        results[aligned.index[i]] = float(tail.mean()) if not tail.empty else float("nan")

    return pd.Series(results, name=f"{bank_returns.name}_mes")


# ---------------------------------------------------------------------------
# LRMES — closed-form approximation
# ---------------------------------------------------------------------------
def calc_lrmes(
    bank_returns: pd.Series,
    index_returns: pd.Series,
    h: int | None = None,
    market_drop: float | None = None,
    window: int | None = None,
) -> float:
    """
    Compute Long-Run MES approximation (Brownlees & Engle 2017).

        LRMES_i ≈ 1 - exp(log(1 - D) * β_OLS)

    where:
        D      = market_drop (40% default) — the cumulative crisis loss
                 that implicitly defines the horizon h
        β_OLS  = OLS market beta = Cov(r_b, r_m) / Var(r_m)
                 (already encodes the correlation ρ · σ_i/σ_m)

    Note: β_OLS = ρ · σ_i/σ_m, so ρ must NOT be multiplied separately.
    The horizon h is implicit in the market_drop parameter and does not
    appear as an explicit multiplier in the formula.

    Args:
        bank_returns  : Daily returns series for the bank.
        index_returns : Daily returns series for the regional index.
        h             : Horizon in trading days (documents the scenario;
                        default cfg.lrmes_h = 22). Not used in formula.
        market_drop   : Hypothetical cumulative market decline (default cfg.lrmes_market_drop).
        window        : Look-back window (default cfg.covar_window).

    Returns:
        LRMES as float (≥ 0, representing fractional loss).
    """
    h = h or cfg.lrmes_h
    market_drop = market_drop if market_drop is not None else cfg.lrmes_market_drop
    window = window or cfg.covar_window

    aligned = _align(bank_returns, index_returns)
    if len(aligned) < 30:
        return float("nan")

    # Use most recent `window` observations
    data = aligned.tail(window)
    r_b = data["bank"].values
    r_m = data["index"].values

    # OLS market beta = Cov(r_b, r_m) / Var(r_m)
    # β_OLS = ρ · σ_b/σ_m, so it already encodes correlation
    cov = np.cov(r_b, r_m)
    var_m = cov[1, 1]
    if var_m == 0:
        return float("nan")
    beta = cov[0, 1] / var_m

    # LRMES = 1 - exp(log(1-D) * β_OLS)
    # Correct B&E 2017 formula: no separate ρ factor, no √h scaling
    lrmes = 1 - np.exp(np.log(1 - market_drop) * beta)
    # Clamp to [0, 1] range — by definition a loss fraction
    return float(np.clip(lrmes, 0.0, 1.0))


def calc_lrmes_rolling(
    bank_returns: pd.Series,
    index_returns: pd.Series,
    window: int | None = None,
    h: int | None = None,
    market_drop: float | None = None,
) -> pd.Series:
    """Rolling daily LRMES time series."""
    window = window or cfg.covar_window
    h = h or cfg.lrmes_h
    market_drop = market_drop if market_drop is not None else cfg.lrmes_market_drop

    aligned = _align(bank_returns, index_returns)
    if aligned.empty:
        return pd.Series(dtype=float)

    results = {}
    for i in range(window, len(aligned)):
        sub = aligned.iloc[i - window:i]
        r_b = sub["bank"].values
        r_m = sub["index"].values
        cov = np.cov(r_b, r_m)
        var_m = cov[1, 1]
        if var_m == 0:
            results[aligned.index[i]] = float("nan")
            continue
        beta = cov[0, 1] / var_m
        # LRMES = 1 - exp(log(1-D) * β_OLS); no separate ρ, no √h
        val = 1 - np.exp(np.log(1 - market_drop) * beta)
        results[aligned.index[i]] = float(np.clip(val, 0.0, 1.0))

    return pd.Series(results, name=f"{bank_returns.name}_lrmes")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _align(bank: pd.Series, index: pd.Series) -> pd.DataFrame:
    """Inner-join bank and index return series on date."""
    df = pd.DataFrame({"bank": bank, "index": index}).dropna()
    return df
