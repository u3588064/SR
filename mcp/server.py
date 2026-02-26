"""
server.py — G-SIBs Systemic Risk MCP Server (FastAPI + SSE transport)

Exposes systemic risk metrics for all 30 G-SIBs via the Model Context Protocol.

Transport:
    HTTP/SSE (Server-Sent Events) — suitable for remote deployment.
    Default: http://0.0.0.0:8000

Data source priority:
    1. Local data/ directory (when running alongside the pipeline)
    2. raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/data/
       (set GITHUB_REPO env var when deploying remotely)

Run:
    python mcp/server.py
    uvicorn mcp.server:app --host 0.0.0.0 --port 8000

MCP clients connect to:
    http://<host>:8000/sse
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import cfg
from src.universe import BANKS, BANK_BY_ID, REGIONS
from src.utils import is_valid_date

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mcp-server")

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="gsib-systemic-risk",
    description=(
        "Daily systemic risk metrics (MES, CoVaR/ΔCoVaR, SRISK) "
        "for all 30 FSB-designated Global Systemically Important Banks."
    ),
)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
DATA_DIR = Path(cfg.data_dir)


def _load_json(rel_path: str) -> dict | None:
    """Load JSON from local data dir or GitHub raw URL."""
    local = DATA_DIR / rel_path
    # Prevent path traversal: ensure the resolved path stays inside DATA_DIR
    try:
        resolved = local.resolve()
        data_root = DATA_DIR.resolve()
        resolved.relative_to(data_root)
    except ValueError:
        logger.warning(f"Blocked path traversal attempt: {rel_path!r}")
        return None
    if resolved.exists():
        with open(resolved, encoding="utf-8") as f:
            return json.load(f)

    base = cfg.raw_base_url()
    if base:
        url = f"{base}/data/{rel_path}"
        try:
            resp = httpx.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Remote fetch failed for {rel_path}: {e}")
    return None


def _load_bank_csv(bank_id: str) -> pd.DataFrame | None:
    """Load per-bank CSV time series."""
    local = DATA_DIR / "banks" / f"{bank_id}.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["date"])

    base = cfg.raw_base_url()
    if base:
        url = f"{base}/data/banks/{bank_id}.csv"
        try:
            return pd.read_csv(url, parse_dates=["date"])
        except Exception as e:
            logger.warning(f"Remote CSV fetch failed for {bank_id}: {e}")
    return None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def get_latest_metrics(
    bank_id: str | None = None,
    region: str | None = None,
) -> dict:
    """
    Get the latest available systemic risk metrics.

    Args:
        bank_id : Optional. Return metrics for a single bank (e.g. 'JPM').
                  Case-insensitive. If omitted, returns all banks.
        region  : Optional. Filter by region: US | CN | GB | EU | JP.
                  Ignored if bank_id is provided.

    Returns:
        Full latest.json payload (or filtered subset).
    """
    payload = _load_json("latest.json")
    if payload is None:
        return {"error": "latest.json not found. Has the pipeline run yet?"}

    banks_list = payload.get("banks", [])

    if bank_id:
        bid = bank_id.upper()
        match = [b for b in banks_list if b.get("bank_id") == bid]
        if not match:
            return {"error": f"Bank '{bid}' not found. Valid IDs: {[b.id for b in BANKS]}"}
        return {**{k: v for k, v in payload.items() if k != "banks"}, "banks": match}

    if region:
        reg = region.upper()
        filtered = [b for b in banks_list if b.get("region") == reg]
        if not filtered:
            return {"error": f"Region '{reg}' not found. Valid: {REGIONS}"}
        return {**{k: v for k, v in payload.items() if k != "banks"}, "banks": filtered}

    return payload


@mcp.tool()
def get_historical(
    bank_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    metric: str | None = None,
) -> dict:
    """
    Get historical time series for a specific bank.

    Args:
        bank_id    : Bank identifier (e.g. 'JPM', 'ICBC', 'HSBC'). Required.
        start_date : Start date YYYY-MM-DD (inclusive). Optional.
        end_date   : End date YYYY-MM-DD (inclusive). Optional.
        metric     : If provided, return only this metric column.
                     Options: mes | lrmes | covar | delta_covar | srisk_usd_bn |
                              srisk_share_pct | market_cap_usd_bn | debt_usd_bn

    Returns:
        Dict with bank info and 'records' list of daily metric values.
    """
    bid = bank_id.upper()
    if bid not in BANK_BY_ID:
        return {"error": f"Unknown bank_id '{bid}'."}

    bank = BANK_BY_ID[bid]
    df = _load_bank_csv(bid)
    if df is None or df.empty:
        return {"error": f"No historical data found for {bid}. Has the pipeline run?"}

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    if df.empty:
        return {"error": f"No data in specified date range for {bid}."}

    if metric:
        if metric not in df.columns:
            return {"error": f"Unknown metric '{metric}'. Valid: {list(df.columns[1:])}"}
        df = df[["date", metric]]

    records = df.to_dict(orient="records")
    for r in records:
        if hasattr(r.get("date"), "isoformat"):
            r["date"] = r["date"].isoformat()

    return {
        "bank_id": bid,
        "bank_name": bank.name,
        "region": bank.region,
        "covar_index": bank.index_yf,
        "record_count": len(records),
        "records": records,
    }


@mcp.tool()
def get_srisk_ranking(
    date: str | None = None,
    region: str | None = None,
    top_n: int = 30,
) -> dict:
    """
    Get banks ranked by SRISK (highest systemic capital shortfall first).

    Args:
        date   : Date YYYY-MM-DD to use for snapshot. Default: latest.
        region : Filter by region (US|CN|GB|EU|JP). Optional.
        top_n  : Number of top banks to return. Default: 30 (all).

    Returns:
        Ranked list of banks by SRISK with key metrics.
    """
    if date:
        if not is_valid_date(date):
            return {"error": f"Invalid date format '{date}'. Expected YYYY-MM-DD."}
        payload = _load_json(f"history/{date}.json")
        if payload is None:
            return {"error": f"No snapshot found for {date}."}
    else:
        payload = _load_json("latest.json")
        if payload is None:
            return {"error": "No data available."}

    banks = payload.get("banks", [])
    if region:
        banks = [b for b in banks if b.get("region") == region.upper()]

    ranked = sorted(
        [b for b in banks if b.get("srisk_usd_bn") is not None],
        key=lambda x: x.get("srisk_usd_bn", 0),
        reverse=True,
    )[:top_n]

    return {
        "date": payload.get("date"),
        "system_srisk_usd_bn": payload.get("system_srisk_usd_bn"),
        "region_filter": region,
        "ranking": [
            {
                "rank": i + 1,
                "bank_id": b.get("bank_id"),
                "bank_name": b.get("bank_name"),
                "region": b.get("region"),
                "srisk_usd_bn": b.get("srisk_usd_bn"),
                "srisk_share_pct": b.get("srisk_share_pct"),
                "lrmes": b.get("lrmes"),
                "market_cap_usd_bn": b.get("market_cap_usd_bn"),
            }
            for i, b in enumerate(ranked)
        ],
    }


@mcp.tool()
def get_delta_covar_ranking(
    date: str | None = None,
    region: str | None = None,
    top_n: int = 30,
) -> dict:
    """
    Get banks ranked by ΔCoVaR (most systemic first, i.e. most negative ΔCoVaR).

    Args:
        date   : Date YYYY-MM-DD. Default: latest.
        region : Filter by region. Optional.
        top_n  : Number of banks to return. Default: 30.

    Returns:
        Ranked list with ΔCoVaR and CoVaR values.
    """
    if date:
        if not is_valid_date(date):
            return {"error": f"Invalid date format '{date}'. Expected YYYY-MM-DD."}
        payload = _load_json(f"history/{date}.json")
        if payload is None:
            return {"error": f"No snapshot for {date}."}
    else:
        payload = _load_json("latest.json")
        if payload is None:
            return {"error": "No data available."}

    banks = payload.get("banks", [])
    if region:
        banks = [b for b in banks if b.get("region") == region.upper()]

    ranked = sorted(
        [b for b in banks if b.get("delta_covar") is not None],
        key=lambda x: x.get("delta_covar", 0),
    )[:top_n]  # most negative first = most systemic

    return {
        "date": payload.get("date"),
        "region_filter": region,
        "note": "More negative ΔCoVaR indicates greater systemic risk contribution.",
        "ranking": [
            {
                "rank": i + 1,
                "bank_id": b.get("bank_id"),
                "bank_name": b.get("bank_name"),
                "region": b.get("region"),
                "covar_index": b.get("covar_index"),
                "delta_covar": b.get("delta_covar"),
                "covar": b.get("covar"),
                "mes": b.get("mes"),
            }
            for i, b in enumerate(ranked)
        ],
    }


@mcp.tool()
def get_methodology() -> dict:
    """
    Return the full methodology documentation for all metrics.

    Returns:
        Dict describing MES, LRMES, CoVaR, ΔCoVaR, SRISK formulas,
        data sources, index choices, and configurable parameters.
    """
    return {
        "title": "G-SIBs Systemic Risk Metrics — Methodology",
        "version": "1.0",
        "metrics": {
            "MES": {
                "full_name": "Marginal Expected Shortfall",
                "reference": "Acharya, Pedersen, Philippon & Richardson (2010)",
                "formula": "MES_i = E[r_i | r_m ≤ VaR_τ(r_m)]",
                "description": "Mean return of bank i on days when the market falls below its τ-th percentile.",
                "default_tau": cfg.mes_tail_pct,
            },
            "LRMES": {
                "full_name": "Long-Run Marginal Expected Shortfall",
                "reference": "Brownlees & Engle (2017)",
                "formula": "LRMES ≈ 1 - exp(log(1-D) · ρ · β · √h)",
                "parameters": {
                    "D": f"Market drop scenario = {cfg.lrmes_market_drop:.0%}",
                    "h": f"Horizon = {cfg.lrmes_h} trading days",
                    "ρ": "Rolling Pearson correlation (bank, market index)",
                    "β": "Rolling OLS beta",
                },
            },
            "CoVaR": {
                "full_name": "Conditional Value-at-Risk",
                "reference": "Adrian & Brunnermeier (2011)",
                "formula": "q_τ(r_system | r_bank = x) = α + β·x  [quantile regression]",
                "description": "System VaR conditional on bank being at its own VaR.",
            },
            "DeltaCoVaR": {
                "full_name": "ΔCoVaR",
                "formula": "ΔCoVaR_i = β̂ · (VaR_i - Median_i)",
                "description": "Incremental systemic risk contribution vs normal state.",
                "note": "More negative = greater systemic importance.",
            },
            "SRISK": {
                "full_name": "Systemic Risk (Capital Shortfall)",
                "reference": "Brownlees & Engle (2017)",
                "formula": "SRISK_i = max(0, k·Debt_i - (1-k)·W_i·(1-LRMES_i))",
                "parameters": {
                    "k": f"Prudential capital ratio = {cfg.srisk_k:.0%} (configurable via SRISK_K env var)",
                    "Debt": "Total liabilities (USD bn, quarterly balance sheet)",
                    "W": "Market capitalisation (USD bn)",
                    "LRMES": "Long-Run MES (40% market drop, 22-day horizon)",
                },
            },
        },
        "data_sources": {
            "prices": "Yahoo Finance (primary); AkShare (CN A-share fallback)",
            "market_cap": "yfinance: shares_outstanding × adjusted close",
            "debt": "yfinance: quarterly balance sheet, forward-filled to daily",
            "fx_conversion": "All monetary values converted to USD via daily FX rates from Yahoo Finance",
        },
        "covar_index_by_region": {
            "US": "^GSPC (S&P 500) — domestic systemic benchmark",
            "CN": "^HSI (Hang Seng) for H-share listed CN banks — reflects offshore systemic risk",
            "GB": "^FTSE (FTSE 100) — post-Brexit separate benchmark",
            "EU": "^STOXX50E (EURO STOXX 50) — eurozone systemic reference",
            "JP": "^N225 (Nikkei 225) — domestic systemic reference",
        },
        "rolling_window_days": cfg.covar_window,
        "universe": "FSB 2023 G-SIB list (29 active institutions)",
    }


# ---------------------------------------------------------------------------
# FastAPI app + SSE transport
# ---------------------------------------------------------------------------
app = FastAPI(
    title="G-SIBs Systemic Risk MCP",
    description="MCP server for daily systemic risk metrics of all 30 G-SIBs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sse = SseServerTransport("/messages")

@app.get("/sse")
async def sse_endpoint(request):
    """MCP client connection endpoint (SSE)."""
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp._mcp_server.run(
            streams[0], streams[1], mcp._mcp_server.create_initialization_options()
        )

@app.post("/messages")
async def messages_endpoint(request):
    """MCP message posting endpoint."""
    await sse.handle_post_message(request.scope, request.receive, request._send)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "server": "gsib-systemic-risk-mcp", "version": "1.0.0"}

@app.get("/")
async def root():
    return {
        "name": "G-SIBs Systemic Risk MCP Server",
        "mcp_endpoint": "/sse",
        "health": "/health",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mcp.server:app",
        host=cfg.mcp_host,
        port=cfg.mcp_port,
        reload=False,
    )
