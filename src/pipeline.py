"""
pipeline.py — Orchestrates daily data → metrics → publish cycle

Usage:
    python src/pipeline.py                        # today
    python src/pipeline.py --date 2024-01-15      # specific date
    python src/pipeline.py --start 2020-01-01 --end 2024-12-31  # range (backfill)
    python src/pipeline.py --banks JPM,HSBC       # subset
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable when called as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import cfg
from src.universe import BANKS, BANK_BY_ID
from src.fetcher import fetch_prices, fetch_market_cap_series, fetch_debt_series
from src.metrics.mes import calc_lrmes_rolling, calc_mes_rolling
from src.metrics.covar import calc_covar_rolling
from src.metrics.srisk import calc_srisk_series, calc_srisk_shares, system_srisk
from src.publish import publish_snapshot, publish_bank_csv, publish_latest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    target_date: date,
    start_date: date,
    bank_ids: list[str] | None = None,
) -> None:
    """
    Run the full pipeline for a given date range and publish results.
    target_date is the final date whose snapshot gets written to data/latest.json.
    """
    banks = BANKS if not bank_ids else [BANK_BY_ID[i.upper()] for i in bank_ids]
    start_str = start_date.isoformat()
    end_str = target_date.isoformat()

    logger.info(f"Pipeline start: {start_str} → {end_str}, {len(banks)} banks")

    all_results: dict[str, dict] = {}  # bank_id → {date → metrics}
    failed_banks: list[str] = []

    for bank in banks:
        logger.info(f"Processing {bank.id} ({bank.name})")
        try:
            bank_data = process_bank(bank, start_str, end_str)
            if bank_data is not None:
                all_results[bank.id] = bank_data
                publish_bank_csv(bank, bank_data)
                logger.info(f"  ✓ {bank.id} completed")
            else:
                failed_banks.append(bank.id)
        except Exception as e:
            failed_banks.append(bank.id)
            logger.error(f"  ✗ {bank.id} failed: {e}", exc_info=True)

    # Publish daily snapshots
    if not all_results:
        raise RuntimeError(
            f"Pipeline produced no data: all {len(banks)} banks failed "
            f"({', '.join(failed_banks)})"
        )

    snapshot_dates = sorted(
        set().union(*[set(v.keys()) for v in all_results.values()])
    )
    for snap_date in snapshot_dates:
        day_data = {
            bid: metrics[snap_date]
            for bid, metrics in all_results.items()
            if snap_date in metrics
        }
        if day_data:
            # Add SRISK shares for the day
            srisk_vals = {bid: day_data[bid].get("srisk_usd_bn", float("nan"))
                          for bid in day_data}
            shares = calc_srisk_shares(srisk_vals)
            sys_srisk = system_srisk(srisk_vals)
            for bid in day_data:
                day_data[bid]["srisk_share_pct"] = shares.get(bid, 0.0)
            publish_snapshot(snap_date, day_data, sys_srisk)

    # Update latest.json from target_date
    target_str = target_date.isoformat()
    latest_data = {
        bid: metrics.get(target_str, {})
        for bid, metrics in all_results.items()
        if target_str in metrics
    }
    if latest_data:
        srisk_vals = {bid: v.get("srisk_usd_bn", float("nan"))
                      for bid, v in latest_data.items()}
        shares = calc_srisk_shares(srisk_vals)
        sys_srisk = system_srisk(srisk_vals)
        for bid in latest_data:
            latest_data[bid]["srisk_share_pct"] = shares.get(bid, 0.0)
        publish_latest(target_date, latest_data, sys_srisk)
        logger.info(f"Published latest.json for {target_str}")

    logger.info("Pipeline complete.")


def process_bank(bank, start_str: str, end_str: str) -> dict | None:
    """
    Full processing for one bank: fetch data → compute metrics → return dict.

    Returns:
        dict mapping date strings to metric dicts, or None on fatal error.
    """
    # ----- Price returns -----
    prices = fetch_prices(bank.yf_ticker, start_str, end_str, bank.ak_ticker)
    if prices.empty or len(prices) < cfg.covar_window + 10:
        logger.warning(f"  Insufficient price data for {bank.id}")
        return None

    index_prices = fetch_prices(bank.index_yf, start_str, end_str)
    if index_prices.empty:
        logger.warning(f"  Missing index data {bank.index_yf} for {bank.id}")
        return None

    # Log returns
    bank_rets = np.log(prices / prices.shift(1)).dropna()
    index_rets = np.log(index_prices / index_prices.shift(1)).dropna()

    # ----- Market cap & debt -----
    mcap = fetch_market_cap_series(bank, start_str, end_str)
    debt = fetch_debt_series(bank, start_str, end_str)

    # ----- Rolling metrics -----
    window = cfg.covar_window

    mes_series = calc_mes_rolling(bank_rets, index_rets, window=window)
    lrmes_series = calc_lrmes_rolling(bank_rets, index_rets, window=window)
    covar_df = calc_covar_rolling(bank_rets, index_rets, window=window)

    # ----- Rolling SRISK (requires aligned mcap, debt, lrmes) -----
    srisk_series = pd.Series(dtype=float)
    if not mcap.empty and not debt.empty and not lrmes_series.empty:
        srisk_series = calc_srisk_series(mcap, debt, lrmes_series)

    # ----- Assemble per-date records -----
    # Common date index = all dates where lrmes is available
    all_dates = lrmes_series.index if not lrmes_series.empty else mes_series.index

    result: dict[str, dict] = {}
    for dt in all_dates:
        dt_str = dt.strftime("%Y-%m-%d")
        covar_row = covar_df.loc[dt] if dt in covar_df.index else None

        record = {
            "bank_id": bank.id,
            "bank_name": bank.name,
            "region": bank.region,
            "covar_index": bank.index_yf,
            "mes": _safe_float(mes_series.get(dt)),
            "lrmes": _safe_float(lrmes_series.get(dt)),
            "covar": _safe_float(covar_row["covar"] if covar_row is not None else None),
            "delta_covar": _safe_float(covar_row["delta_covar"] if covar_row is not None else None),
            "covar_beta": _safe_float(covar_row["beta"] if covar_row is not None else None),
            "srisk_usd_bn": _safe_float(srisk_series.get(dt) if not srisk_series.empty else None),
            "market_cap_usd_bn": _safe_float(mcap.get(dt) if not mcap.empty else None),
            "debt_usd_bn": _safe_float(debt.get(dt) if not debt.empty else None),
        }
        result[dt_str] = record

    return result if result else None


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="G-SIBs Systemic Risk Pipeline")
    parser.add_argument("--date", default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--start", default=None,
                        help="Start date for historical range YYYY-MM-DD")
    parser.add_argument("--end", default=None,
                        help="End date for historical range YYYY-MM-DD (default: today)")
    parser.add_argument("--banks", default=None,
                        help="Comma-separated bank IDs to process (default: all)")
    args = parser.parse_args()

    today = date.today()

    if args.date:
        target = date.fromisoformat(args.date)
    elif args.end:
        target = date.fromisoformat(args.end)
    else:
        target = today

    if args.start:
        start = date.fromisoformat(args.start)
    else:
        # Default: covar_window + 30 buffer days before target
        start = target - timedelta(days=cfg.covar_window + 30)

    bank_ids = [b.strip().upper() for b in args.banks.split(",")] if args.banks else None

    try:
        run_pipeline(target_date=target, start_date=start, bank_ids=bank_ids)
    except RuntimeError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
