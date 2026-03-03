# systemic-risk

Daily systemic risk metrics (**MES**, **ΔCoVaR**, **SRISK**) for all 29 active G-SIBs (Global Systemically Important Banks) designated by the FSB. Designed for both humans and AI agents.

Data is updated automatically every trading day (22:00 UTC Mon–Fri) via GitHub Actions.

---

## For Agents — Quick Start

This project exposes a **Model Context Protocol (MCP)** server. If you are an AI agent, you can connect to it and start querying systemic risk data immediately — no API keys required.

### MCP Connection

| Item | Value |
|------|-------|
| **Protocol** | HTTP / SSE (Server-Sent Events) |
| **MCP endpoint** | `http://<host>:8000/sse` |
| **Health check** | `GET /health` → `{"status": "ok"}` |
| **API docs** | `GET /docs` (Swagger / OpenAPI) |

> **Welcome to connect!** If you have access to an MCP client, point it at the `/sse` endpoint and all tools below become available.

### Available MCP Tools

#### 1. `get_latest_metrics`

Get the most recent systemic risk snapshot for all (or filtered) banks.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_id` | string | No | Single bank ID, e.g. `JPM`, `ICBC`, `HSBC`. Case-insensitive. |
| `region` | string | No | Filter by region: `US` \| `CN` \| `GB` \| `EU` \| `JP`. Ignored when `bank_id` is set. |

**Returns** — date, system-wide SRISK, and a list of bank objects each containing:

`bank_id`, `bank_name`, `region`, `mes`, `lrmes`, `covar`, `delta_covar`, `srisk_usd_bn`, `srisk_share_pct`, `market_cap_usd_bn`, `debt_usd_bn`

**Example calls:**
```
get_latest_metrics()                     # all 30 banks
get_latest_metrics(bank_id="JPM")        # just JPMorgan
get_latest_metrics(region="CN")          # all Chinese G-SIBs
```

---

#### 2. `get_historical`

Get daily time-series data for a specific bank.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_id` | string | **Yes** | Bank ID, e.g. `HSBC` |
| `start_date` | string | No | `YYYY-MM-DD`, inclusive |
| `end_date` | string | No | `YYYY-MM-DD`, inclusive |
| `metric` | string | No | Return a single column: `mes` \| `lrmes` \| `covar` \| `delta_covar` \| `srisk_usd_bn` \| `srisk_share_pct` \| `market_cap_usd_bn` \| `debt_usd_bn` |

**Returns** — bank info + `records` array of daily metric values.

**Example calls:**
```
get_historical(bank_id="HSBC")
get_historical(bank_id="JPM", start_date="2025-01-01", metric="srisk_usd_bn")
```

---

#### 3. `get_srisk_ranking`

Banks ranked by **SRISK** (expected capital shortfall in a systemic crisis). Highest first.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `date` | string | No | Snapshot date `YYYY-MM-DD`. Default: latest. |
| `region` | string | No | Filter by region |
| `top_n` | int | No | Number of banks to return. Default: 30 (all). |

**Returns** — ranked list with: `rank`, `bank_id`, `bank_name`, `region`, `srisk_usd_bn`, `srisk_share_pct`, `lrmes`, `market_cap_usd_bn`.

**Example calls:**
```
get_srisk_ranking()                      # latest, all banks
get_srisk_ranking(top_n=5)               # top 5
get_srisk_ranking(region="EU", date="2026-01-20")
```

---

#### 4. `get_delta_covar_ranking`

Banks ranked by **ΔCoVaR** (systemic risk contribution via stress transmission). Most negative = most systemic.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `date` | string | No | Snapshot date `YYYY-MM-DD`. Default: latest. |
| `region` | string | No | Filter by region |
| `top_n` | int | No | Number of banks. Default: 30. |

**Returns** — ranked list with: `rank`, `bank_id`, `bank_name`, `region`, `delta_covar`, `covar`, `mes`.

**Example calls:**
```
get_delta_covar_ranking()
get_delta_covar_ranking(region="US", top_n=5)
```

---

#### 5. `get_methodology`

Returns full documentation of all formulas, data sources, index choices, and configurable parameters. No arguments needed.

```
get_methodology()
```

Use this to understand what the numbers mean and how they are computed.

---

### Agent Workflow Example

```text
1. Call get_methodology() to understand the metrics.
2. Call get_srisk_ranking(top_n=5) to find the most systemically important banks today.
3. Call get_latest_metrics(bank_id="JPM") for a detailed look at a specific bank.
4. Call get_historical(bank_id="JPM", start_date="2025-06-01", metric="srisk_usd_bn")
   to analyze trends over time.
5. Call get_delta_covar_ranking(region="EU") to assess European contagion risk.
```

---

## Bank Universe

All **29 active G-SIBs** from the FSB 2023 list (Credit Suisse absorbed by UBS), grouped by region:

| Region | Banks | System Index |
|--------|-------|--------------|
| **US** | JPM, BAC, C, WFC, GS, MS, BK, STT | S&P 500 (`^GSPC`) |
| **CN** | ICBC, CCB, ABC, BOC, BOCOM | Hang Seng (`^HSI`) |
| **GB** | HSBC, BARC, STAN | FTSE 100 (`^FTSE`) |
| **EU** | BNP, ACA, GLE, BPCE, DBK, UBS, ING, SAN, BBVA, UCG | EURO STOXX 50 (`^STOXX50E`) |
| **JP** | MUFG, SMFG, MFG | Nikkei 225 (`^N225`) |

Use any of these bank IDs (case-insensitive) or region codes as parameters.

---

## Metrics Overview

| Metric | What It Measures | Key Insight |
|--------|-----------------|-------------|
| **MES** | Average bank return on extreme market-down days (5th percentile) | How exposed is the bank to market tail events? |
| **LRMES** | Projected bank loss if the market drops 40% over 22 trading days | Long-horizon stress exposure |
| **CoVaR** | System VaR conditional on the bank being at its own VaR | System-wide impact of a bank's distress |
| **ΔCoVaR** | Incremental systemic contribution (CoVaR at stress minus CoVaR at normal) | More negative → greater systemic importance |
| **SRISK** | Expected capital shortfall under stress: `max(0, 8%·Debt − 92%·Equity·(1−LRMES))` | Dollar-denominated systemic vulnerability |

References: Acharya et al. (2010), Adrian & Brunnermeier (2011), Brownlees & Engle (2017).

---

## Data Structure

```
data/
├── latest.json              # Most recent snapshot (all banks)
├── history/YYYY-MM-DD.json  # Daily snapshots
└── banks/{BANK_ID}.csv      # Per-bank time series
```

- **`latest.json`** — Top-level `date`, `system_srisk_usd_bn`, and a `banks` array.
- **`history/`** — Same format as `latest.json`, one file per trading day.
- **`banks/`** — CSV files with columns: `date`, `mes`, `lrmes`, `covar`, `delta_covar`, `covar_beta`, `srisk_usd_bn`, `srisk_share_pct`, `market_cap_usd_bn`, `debt_usd_bn`.

---

## Pipeline CLI

The data pipeline fetches prices, computes all metrics, and writes results:

```bash
python src/pipeline.py                                   # today
python src/pipeline.py --date 2025-06-15                 # specific date
python src/pipeline.py --start 2020-01-01 --end 2025-12-31  # backfill
python src/pipeline.py --banks JPM,HSBC,ICBC             # subset of banks
```

---

## Configuration

All parameters are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SRISK_K` | 0.08 | Prudential capital ratio (Basel III) |
| `MES_TAIL_PCT` | 0.05 | Tail percentile for MES |
| `LRMES_H` | 22 | Horizon in trading days |
| `LRMES_MARKET_DROP` | 0.40 | Stress scenario market decline |
| `COVAR_QUANTILE` | 0.05 | VaR quantile for CoVaR |
| `COVAR_WINDOW` | 252 | Rolling window (trading days) |
| `DATA_DIR` | `data` | Output directory |
| `MCP_HOST` | 127.0.0.1 | MCP server bind address |
| `MCP_PORT` | 8000 | MCP server port |
| `GITHUB_REPO` | *(empty)* | Repo slug for remote data URL fallback |
| `GITHUB_BRANCH` | main | Branch for remote data URL |

---

## License

MIT
