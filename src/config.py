"""
config.py — Centralised configuration (all parameters overridable via env vars)

Usage:
    from src.config import cfg

    k = cfg.srisk_k          # e.g. 0.08
    window = cfg.covar_window  # e.g. 252
"""

import os
from dataclasses import dataclass


def _float(env: str, default: float) -> float:
    return float(os.environ.get(env, default))


def _int(env: str, default: int) -> int:
    return int(os.environ.get(env, default))


def _str(env: str, default: str) -> str:
    return os.environ.get(env, default)


@dataclass
class Config:
    # ── SRISK ────────────────────────────────────────────────────────────
    srisk_k: float = _float("SRISK_K", 0.08)
    """Prudential capital ratio used in SRISK formula.
    Default: 0.08 (Basel III Tier 1 minimum).
    Override: set env SRISK_K=0.055 for stricter leverage ratio."""

    # ── LRMES / MES ──────────────────────────────────────────────────────
    mes_tail_pct: float = _float("MES_TAIL_PCT", 0.05)
    """Percentile threshold for MES tail days. Default 5th percentile."""

    lrmes_h: int = _int("LRMES_H", 22)
    """Horizon (trading days) for LRMES projection. Default 22 ≈ 1 month."""

    lrmes_market_drop: float = _float("LRMES_MARKET_DROP", 0.40)
    """Hypothetical market decline for LRMES scenario. Default 40%."""

    # ── CoVaR ────────────────────────────────────────────────────────────
    covar_quantile: float = _float("COVAR_QUANTILE", 0.05)
    """Quantile for VaR used in CoVaR. Default 5% (95% VaR)."""

    covar_window: int = _int("COVAR_WINDOW", 252)
    """Rolling window (trading days) for CoVaR quantile regression. Default 252."""

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir: str = _str("DATA_DIR", "data")
    """Root directory for output JSON / CSV files."""

    raw_dir: str = _str("RAW_DIR", "data/raw")
    """Directory for raw cached price data (gitignored)."""

    default_start_date: str = _str("DEFAULT_START_DATE", "2015-01-01")
    """Default historical start date for backfill."""

    yf_request_delay: float = _float("YF_REQUEST_DELAY", 1.0)
    """Seconds to sleep between Yahoo Finance requests to avoid rate-limiting."""

    # ── MCP Server ───────────────────────────────────────────────────────
    mcp_host: str = _str("MCP_HOST", "127.0.0.1")
    mcp_port: int = _int("MCP_PORT", 8000)

    # ── GitHub (for remote data URL in MCP) ─────────────────────────────
    github_repo: str = _str("GITHUB_REPO", "")
    """e.g. 'your-org/gsib-systemic-risk'. Used to build raw.githubusercontent.com URLs."""

    github_branch: str = _str("GITHUB_BRANCH", "main")

    def raw_base_url(self) -> str | None:
        """Return base URL for raw GitHub content, or None if not configured."""
        if not self.github_repo:
            return None
        return f"https://raw.githubusercontent.com/{self.github_repo}/{self.github_branch}"


cfg = Config()
