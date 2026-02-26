"""
publish.py — Write computed metrics to JSON and CSV files

Output layout:
    data/
        latest.json                     ← latest full snapshot (all banks)
        history/YYYY-MM-DD.json         ← daily snapshot
        banks/{BANK_ID}.csv             ← per-bank time series
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import cfg
from src.universe import Bank

logger = logging.getLogger(__name__)


def _data_root() -> Path:
    root = Path(cfg.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def publish_latest(
    snapshot_date: date,
    bank_records: dict[str, dict],
    system_srisk_usd_bn: float,
) -> None:
    """Write data/latest.json with full snapshot for all banks."""
    payload = _build_payload(snapshot_date, bank_records, system_srisk_usd_bn)
    path = _data_root() / "latest.json"
    _write_json(payload, path)
    logger.info(f"Published {path}")


def publish_snapshot(
    snapshot_date: date | str,
    bank_records: dict[str, dict],
    system_srisk_usd_bn: float,
) -> None:
    """Write data/history/YYYY-MM-DD.json."""
    d = date.fromisoformat(str(snapshot_date)) if isinstance(snapshot_date, str) else snapshot_date
    payload = _build_payload(d, bank_records, system_srisk_usd_bn)
    path = _data_root() / "history" / f"{d.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(payload, path)


def publish_bank_csv(bank: Bank, date_metrics: dict[str, dict]) -> None:
    """
    Write / append data/banks/{BANK_ID}.csv with the bank's full time series.
    Columns: date, mes, lrmes, covar, delta_covar, srisk_usd_bn, srisk_share_pct,
             market_cap_usd_bn, debt_usd_bn, covar_beta
    """
    if not date_metrics:
        return

    rows = []
    for dt_str, m in sorted(date_metrics.items()):
        rows.append({
            "date": dt_str,
            "mes": m.get("mes"),
            "lrmes": m.get("lrmes"),
            "covar": m.get("covar"),
            "delta_covar": m.get("delta_covar"),
            "covar_beta": m.get("covar_beta"),
            "srisk_usd_bn": m.get("srisk_usd_bn"),
            "srisk_share_pct": m.get("srisk_share_pct"),
            "market_cap_usd_bn": m.get("market_cap_usd_bn"),
            "debt_usd_bn": m.get("debt_usd_bn"),
        })

    new_df = pd.DataFrame(rows)
    path = _data_root() / "banks" / f"{bank.id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_csv(path)
        combined = (
            pd.concat([existing, new_df])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
    else:
        combined = new_df.sort_values("date")

    combined.to_csv(path, index=False)
    logger.debug(f"Updated {path} ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_payload(
    snapshot_date: date,
    bank_records: dict[str, dict],
    system_srisk: float,
) -> dict:
    """Build the standard JSON payload structure."""
    return {
        "date": snapshot_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_version": "1.0",
        "parameters": {
            "srisk_k": cfg.srisk_k,
            "covar_quantile": cfg.covar_quantile,
            "covar_window_days": cfg.covar_window,
            "lrmes_horizon_days": cfg.lrmes_h,
            "lrmes_market_drop": cfg.lrmes_market_drop,
            "mes_tail_pct": cfg.mes_tail_pct,
        },
        "system_srisk_usd_bn": _round(system_srisk),
        "bank_count": len(bank_records),
        "banks": [
            {**_clean(rec), "bank_id": bid}
            for bid, rec in sorted(bank_records.items())
        ],
    }


def _clean(record: dict) -> dict:
    """Remove None values; keep NaN as null for JSON transparency."""
    return {k: v for k, v in record.items() if k != "bank_id"}


def _round(v) -> float | None:
    try:
        import math
        if math.isnan(float(v)):
            return None
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _write_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
