"""
covar.py — CoVaR and ΔCoVaR (Adrian & Brunnermeier, 2011)

Methodology:
    CoVaR is estimated via quantile regression (τ = 5%):

        q_τ(r_system | r_bank = x) = α̂ + β̂ · x

    where r_system is the regional index return, r_bank is the bank return.

    VaR_i = q_τ(r_bank)  — bank's own VaR at quantile τ
    M_i   = q_0.5(r_bank) — bank's median return (normal state proxy)

    CoVaR_i   = α̂ + β̂ · VaR_i
    ΔCoVaR_i  = CoVaR_i - (α̂ + β̂ · M_i)
              = β̂ · (VaR_i - M_i)

    ΔCoVaR measures the marginal contribution of bank i to systemic risk.
    More negative ΔCoVaR ⟹ greater systemic risk contribution.

Note on system proxy:
    Each bank's regional index (defined in universe.py) is used as the
    system return, rather than a single global index. This follows the
    approach suggested by some authors for geographically diverse samples
    and is explicitly documented in output JSON for transparency.

Inputs:
    bank_returns  : pd.Series of daily returns for bank i
    index_returns : pd.Series of daily returns for the regional index (system)

Outputs:
    CoVaR  : float (system VaR conditional on bank being at its own VaR)
    ΔCoVaR : float (incremental systemic risk contribution)
"""

import numpy as np
import pandas as pd
import warnings

from src.config import cfg


# ---------------------------------------------------------------------------
# Point-in-time CoVaR / ΔCoVaR
# ---------------------------------------------------------------------------
def calc_covar(
    bank_returns: pd.Series,
    index_returns: pd.Series,
    quantile: float | None = None,
    window: int | None = None,
) -> dict:
    """
    Compute CoVaR and ΔCoVaR over the full (or windowed) sample.

    Args:
        bank_returns  : Daily returns of the bank.
        index_returns : Daily returns of the regional market index.
        quantile      : VaR quantile (default cfg.covar_quantile = 0.05).
        window        : If provided, use only the last `window` observations.

    Returns:
        dict with keys:
            var_bank    : Bank's own VaR at quantile τ
            median_bank : Bank's median return
            covar       : System CoVaR (conditional on bank at its VaR)
            delta_covar : ΔCoVaR = CoVaR - median-state CoVaR
            alpha       : Quantile regression intercept
            beta        : Quantile regression slope
    """
    quantile = quantile if quantile is not None else cfg.covar_quantile
    window = window or cfg.covar_window

    aligned = _align(bank_returns, index_returns)
    if len(aligned) < 60:
        return _nan_result()

    data = aligned.tail(window) if window else aligned
    r_b = data["bank"].values
    r_m = data["index"].values

    alpha, beta = _quantile_regression(r_b, r_m, quantile)
    if np.isnan(alpha):
        return _nan_result()

    var_bank = float(np.quantile(r_b, quantile))
    median_bank = float(np.median(r_b))

    covar = alpha + beta * var_bank
    covar_median = alpha + beta * median_bank
    delta_covar = covar - covar_median  # typically negative

    return {
        "var_bank": var_bank,
        "median_bank": median_bank,
        "covar": float(covar),
        "delta_covar": float(delta_covar),
        "alpha": float(alpha),
        "beta": float(beta),
    }


# ---------------------------------------------------------------------------
# Rolling CoVaR / ΔCoVaR time series
# ---------------------------------------------------------------------------
def calc_covar_rolling(
    bank_returns: pd.Series,
    index_returns: pd.Series,
    window: int | None = None,
    quantile: float | None = None,
) -> pd.DataFrame:
    """
    Return a daily time series of CoVaR and ΔCoVaR.

    Returns:
        pd.DataFrame with columns [covar, delta_covar, var_bank, beta]
        indexed by date.
    """
    window = window or cfg.covar_window
    quantile = quantile if quantile is not None else cfg.covar_quantile

    aligned = _align(bank_returns, index_returns)
    if aligned.empty:
        return pd.DataFrame()

    rows = []
    dates = []
    for i in range(window, len(aligned)):
        sub = aligned.iloc[i - window:i]
        r_b = sub["bank"].values
        r_m = sub["index"].values

        alpha, beta = _quantile_regression(r_b, r_m, quantile)
        if np.isnan(alpha):
            rows.append({"covar": np.nan, "delta_covar": np.nan,
                         "var_bank": np.nan, "beta": np.nan})
        else:
            var_bank = float(np.quantile(r_b, quantile))
            median_bank = float(np.median(r_b))
            covar = alpha + beta * var_bank
            delta_covar = beta * (var_bank - median_bank)
            rows.append({"covar": covar, "delta_covar": delta_covar,
                         "var_bank": var_bank, "beta": beta})
        dates.append(aligned.index[i])

    df = pd.DataFrame(rows, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Quantile regression (interior point / iteratively reweighted least squares)
# ---------------------------------------------------------------------------
def _quantile_regression(x: np.ndarray, y: np.ndarray, q: float) -> tuple[float, float]:
    """
    Fit y = α + β·x at quantile q using statsmodels QuantReg.
    Falls back to a simple pinball-loss gradient descent if statsmodels
    is unavailable.

    Returns:
        (alpha, beta) or (nan, nan) on failure.
    """
    n = len(x)
    if n < 30:
        return float("nan"), float("nan")

    X = np.column_stack([np.ones(n), x])
    try:
        from statsmodels.regression.quantile_regression import QuantReg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = QuantReg(y, X)
            res = qr.fit(q=q, max_iter=1000)
        params = res.params
        return float(params[0]), float(params[1])
    except Exception:
        pass

    # Lightweight fallback: subgradient method
    return _pinball_sgd(X, y, q)


def _pinball_sgd(X: np.ndarray, y: np.ndarray, q: float,
                 lr: float = 0.01, epochs: int = 500) -> tuple[float, float]:
    """Minimal pinball-loss SGD as fallback."""
    try:
        n, p = X.shape
        theta = np.zeros(p)
        for _ in range(epochs):
            pred = X @ theta
            residual = y - pred
            grad = np.where(residual >= 0, -q, (1 - q)) @ X / n
            theta -= lr * grad
        return float(theta[0]), float(theta[1])
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _align(bank: pd.Series, index: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"bank": bank, "index": index}).dropna()


def _nan_result() -> dict:
    return {
        "var_bank": float("nan"),
        "median_bank": float("nan"),
        "covar": float("nan"),
        "delta_covar": float("nan"),
        "alpha": float("nan"),
        "beta": float("nan"),
    }
