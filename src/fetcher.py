"""
fetcher.py — Price, market cap, and debt data fetching

Data sources (in priority order):
    1. Local disk cache (data/raw/) — avoids redundant API calls
    2. yfinance   — used for all non-CN-A-share tickers
    3. akshare    — fallback for CN A-share prices (601398, etc.)

Outputs:
    - prices(ticker, start, end)  → pd.Series of adjusted close
    - market_cap(bank, date)      → float (USD billions)
    - total_debt(bank, date)      → float (USD billions, forward-filled quarterly)
"""

import os
import time
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import cfg
from src.universe import Bank, BANK_BY_ID, ALL_INDICES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — avoid hard import errors if optional deps missing
# ---------------------------------------------------------------------------
def _yf():
    import yfinance as yf
    return yf


def _ak():
    import akshare as ak
    return ak


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace(".", "_").replace("/", "_")
    return Path(cfg.raw_dir) / f"{safe}_{start}_{end}.parquet"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Cache read failed for {path}: {e}")
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Core price fetching
# ---------------------------------------------------------------------------
def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    ak_ticker: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    Return daily adjusted close prices as a pd.Series indexed by date.

    Args:
        ticker     : Yahoo Finance ticker (primary)
        start      : YYYY-MM-DD start date (inclusive)
        end        : YYYY-MM-DD end date (inclusive)
        ak_ticker  : AkShare A-share code (fallback for CN banks)
        use_cache  : Whether to read/write disk cache

    Returns:
        pd.Series with DatetimeIndex, float values in local currency.
    """
    cache_path = _cache_path(ticker, start, end)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None and "close" in cached.columns:
            logger.debug(f"Cache hit: {ticker}")
            return cached["close"]

    # ── Try Yahoo Finance ─────────────────────────────────────────────────
    series = _fetch_yf(ticker, start, end)

    # ── Fallback: AkShare (CN A-shares) ──────────────────────────────────
    if (series is None or series.empty) and ak_ticker:
        logger.info(f"YF returned empty for {ticker}, trying AkShare {ak_ticker}")
        series = _fetch_ak(ak_ticker, start, end)

    if series is None or series.empty:
        logger.warning(f"No price data found for {ticker} ({start} to {end})")
        return pd.Series(dtype=float, name=ticker)

    series.name = ticker
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    if use_cache:
        _save_cache(pd.DataFrame({"close": series}), cache_path)

    time.sleep(cfg.yf_request_delay)
    return series


def _fetch_yf(ticker: str, start: str, end: str) -> pd.Series | None:
    try:
        yf = _yf()
        # Download with auto_adjust=True gives adjusted close in 'Close' column
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False, threads=False)
        if df.empty:
            return None
        col = "Close"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df[col].dropna()
    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}", exc_info=True)
        return None


def _fetch_ak(ak_ticker: str, start: str, end: str) -> pd.Series | None:
    """Fetch A-share daily close via AkShare stock_zh_a_hist."""
    try:
        ak = _ak()
        df = ak.stock_zh_a_hist(
            symbol=ak_ticker,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="hfq",  # backward-adjusted
        )
        if df.empty:
            return None
        df.index = pd.to_datetime(df["日期"])
        return df["收盘"].astype(float)
    except Exception as e:
        logger.error(f"AkShare error for {ak_ticker}: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Market Cap
# ---------------------------------------------------------------------------
def fetch_market_cap_series(bank: Bank, start: str, end: str) -> pd.Series:
    """
    Return daily market cap in USD billions.
    market_cap = adjusted_close * shares_outstanding
    Shares outstanding sourced from yfinance info (point-in-time approx).
    """
    prices = fetch_prices(bank.yf_ticker, start, end, bank.ak_ticker)
    if prices.empty:
        return pd.Series(dtype=float, name=f"{bank.id}_mcap")

    yf = _yf()
    info = yf.Ticker(bank.yf_ticker).fast_info
    shares = getattr(info, "shares", None) or getattr(info, "shares_outstanding", None)
    if not shares:
        logger.warning(f"No shares outstanding for {bank.id}")
        return pd.Series(dtype=float)

    # yfinance returns share counts for HK-listed stocks as ~100× the actual
    # number of individual shares (board-lot unit inflation).  Correct it here.
    if bank.yf_ticker.endswith(".HK"):
        shares = shares / 100

    mcap = prices * shares / 1e9  # convert to billions

    # FX conversion to USD for non-USD listed banks
    mcap_usd = _to_usd(mcap, bank.yf_ticker)
    mcap_usd.name = f"{bank.id}_mcap_usd_bn"
    return mcap_usd


def _to_usd(series: pd.Series, ticker: str) -> pd.Series:
    """
    Convert a price series to USD. Only handles common currency suffixes.
    .L → GBX (pence) → GBP → USD
    .HK → HKD → USD
    .PA .MI .AS .MC .DE → EUR → USD
    .SW → CHF → USD
    .T → JPY → USD  (uses JPYUSD=X which gives USD per JPY ≈ 0.0065)
    .SS .SZ → CNY → USD
    """
    fx_pairs = {
        ".L":  ("GBP=X",    0.01),   # GBX (pence) → GBP factor, then → USD
        ".HK": ("HKD=X",    1.0),
        ".PA": ("EURUSD=X", 1.0),
        ".MI": ("EURUSD=X", 1.0),
        ".AS": ("EURUSD=X", 1.0),
        ".MC": ("EURUSD=X", 1.0),
        ".DE": ("EURUSD=X", 1.0),
        ".SW": ("CHF=X",    1.0),
        # JPY=X returns JPY-per-USD (~155); use JPYUSD=X for USD-per-JPY (~0.0065)
        ".T":  ("JPYUSD=X", 1.0),
        ".SS": ("CNYUSD=X", 1.0),
    }
    for suffix, (fx_ticker, factor) in fx_pairs.items():
        if ticker.endswith(suffix):
            try:
                yf = _yf()
                fx = yf.download(fx_ticker, start=series.index[0].strftime("%Y-%m-%d"),
                                 end=(series.index[-1] + timedelta(days=3)).strftime("%Y-%m-%d"),
                                 auto_adjust=True, progress=False, threads=False)
                if not fx.empty:
                    if isinstance(fx.columns, pd.MultiIndex):
                        fx.columns = fx.columns.droplevel(1)
                    fx_rate = fx["Close"].reindex(series.index, method="ffill")
                    return series * factor * fx_rate
            except Exception as e:
                logger.warning(f"FX conversion failed for {ticker}: {e}")
    return series  # already USD or conversion failed


def _to_usd_bs(series: pd.Series, ticker: str) -> pd.Series:
    """
    Convert balance-sheet values (in native reporting currency, already in
    billions) to USD billions.

    Differs from _to_usd in two ways:
      1. No pence factor for .L stocks — balance sheets are in GBP, not GBX.
      2. Uses CNYUSD=X for .HK tickers — HK-listed Chinese banks report
         their financials in CNY (renminbi), not HKD.
    """
    bs_fx_pairs = {
        ".L":  ("GBP=X",    1.0),    # GBP (not pence) → USD
        ".HK": ("CNYUSD=X", 1.0),    # CN banks report in CNY
        ".PA": ("EURUSD=X", 1.0),
        ".MI": ("EURUSD=X", 1.0),
        ".AS": ("EURUSD=X", 1.0),
        ".MC": ("EURUSD=X", 1.0),
        ".DE": ("EURUSD=X", 1.0),
        ".SW": ("CHF=X",    1.0),
        ".T":  ("JPYUSD=X", 1.0),    # JPY → USD (direct rate)
        ".SS": ("CNYUSD=X", 1.0),
    }
    if series.empty:
        return series
    for suffix, (fx_ticker, factor) in bs_fx_pairs.items():
        if ticker.endswith(suffix):
            try:
                yf = _yf()
                start_dt = series.index[0].strftime("%Y-%m-%d")
                end_dt = (series.index[-1] + timedelta(days=3)).strftime("%Y-%m-%d")
                fx = yf.download(fx_ticker, start=start_dt, end=end_dt,
                                 auto_adjust=True, progress=False, threads=False)
                if not fx.empty:
                    if isinstance(fx.columns, pd.MultiIndex):
                        fx.columns = fx.columns.droplevel(1)
                    fx_rate = fx["Close"].reindex(series.index, method="ffill")
                    return (series * factor * fx_rate).rename(series.name)
            except Exception as e:
                logger.warning(f"Balance sheet FX conversion failed for {ticker}: {e}")
    return series  # already USD or conversion failed


# ---------------------------------------------------------------------------
# Debt (Total Liabilities from quarterly balance sheet)
# ---------------------------------------------------------------------------
def fetch_debt_series(bank: Bank, start: str, end: str) -> pd.Series:
    """
    Return daily total liabilities in USD billions (forward-filled from quarterly data).
    Uses yfinance quarterly balance sheet.
    """
    yf = _yf()
    tkr = yf.Ticker(bank.yf_ticker)
    bs = tkr.quarterly_balance_sheet
    if bs is None or bs.empty:
        logger.warning(f"Empty balance sheet for {bank.id}")
        return pd.Series(dtype=float)

    # Try common row names for total liabilities
    candidates = [
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
        "TotalLiabilities",
    ]
    debt_row = None
    for c in candidates:
        if c in bs.index:
            debt_row = bs.loc[c]
            break

    if debt_row is None:
        logger.warning(f"No liabilities row found for {bank.id}")
        return pd.Series(dtype=float)

    debt_row = debt_row.dropna().sort_index()
    debt_bn = debt_row / 1e9  # to local-currency billions

    # Reindex to daily frequency and forward-fill
    idx = pd.date_range(start, end, freq="B")
    daily = debt_bn.reindex(idx.union(debt_bn.index)).sort_index()
    daily = daily.ffill().reindex(idx)

    # Convert from local reporting currency to USD
    daily = _to_usd_bs(daily, bank.yf_ticker)
    daily.name = f"{bank.id}_debt_usd_bn"
    return daily


# ---------------------------------------------------------------------------
# Batch fetch: all banks + all indices
# ---------------------------------------------------------------------------
def fetch_all_prices(
    start: str,
    end: str,
    bank_ids: list[str] | None = None,
) -> dict[str, pd.Series]:
    """
    Fetch price series for all (or subset of) banks AND their regional indices.
    Returns dict: ticker → pd.Series of returns.
    """
    from src.universe import BANKS

    banks = BANKS if bank_ids is None else [BANK_BY_ID[i] for i in bank_ids]
    results: dict[str, pd.Series] = {}

    # Fetch bank prices
    for bank in banks:
        logger.info(f"Fetching prices: {bank.id} ({bank.yf_ticker})")
        p = fetch_prices(bank.yf_ticker, start, end, bank.ak_ticker)
        if not p.empty:
            results[bank.yf_ticker] = p

    # Fetch index prices (deduplicated)
    needed_indices = sorted(set(b.index_yf for b in banks))
    for idx_ticker in needed_indices:
        if idx_ticker not in results:
            logger.info(f"Fetching index: {idx_ticker}")
            p = fetch_prices(idx_ticker, start, end)
            if not p.empty:
                results[idx_ticker] = p

    return results
