"""
BTC Price Prediction — Data Collection Module
==============================================
Collects BTC price, macro-economic indicators, market data,
on-chain metrics, sentiment data, and geopolitical event dummies.
"""

import os
import time
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── project config ──────────────────────────────────────────────
import sys
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import cfg

warnings.filterwarnings("ignore")

# Static configuration from config.yaml
DATA_START = cfg.data_config.get("start_date", "2014-01-01")
DATA_END   = cfg.data_config.get("end_date", "2026-02-11")
TICKERS = cfg.data_config.get("tickers", {})
FRED_SERIES = cfg.data_config.get("fred_series", {})
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Halving dates and Events are still hardcoded or could be moved to config fully
# For now, let's keep them here or move them to config.yaml if they are there.
# Checking config.yaml... they are not in the minimal template I created.
# I will keep them here as constants for now to avoid breaking changes, 
# or I could add them to config.yaml. The user didn't explicitly ask for them in the YAML task description (A.1),
# but good practice is to have them there.
# Attempting to load them from config.py if I put them there? 
# Wait, I didn't put them in config.yaml. I'll include them here for simplicity.

HALVING_DATES = [
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19",
]

EVENTS = [
    ("2020-03-01", "2020-06-30", "covid_crash"),
    ("2021-05-19", "2021-07-20", "china_btc_ban"),
    ("2022-02-24", None,         "ukraine_war"),
    ("2022-05-01", "2022-06-30", "luna_terra_crash"),
    ("2022-11-08", "2022-12-31", "ftx_collapse"),
    ("2024-01-10", None,         "spot_btc_etf"),
    ("2024-11-05", None,         "us_election_2024"),
]

FUTURES_TICKER_MAP = {
    "oil": "CL=F",
    "gold": "GC=F",
    "corn": "ZC=F",
    "wheat": "ZW=F",
}

RATES_PROXY_TICKERS = {
    "rate_irx": "^IRX",  # 13-week T-bill proxy
    "rate_fvx": "^FVX",  # 5Y yield proxy
    "rate_tnx": "^TNX",  # 10Y yield proxy
}

GEOPOLITICAL_PROXY_TICKERS = {
    "geo_vix": "^VIX",   # equity fear gauge
    "geo_ovx": "^OVX",   # oil volatility gauge
}

# FOMC calendar used for "days_to_fomc" feature (fail-forward static baseline)
FOMC_DATES_UTC = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-11-06", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-11-05", "2026-12-16",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16", "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]

# ================================================================
#  1) yfinance helpers
# ================================================================
def _download_yf(ticker: str, label: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from yfinance and return Close + Volume."""
    print(f"  ↓ yfinance: {label} ({ticker})")
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"    ⚠ No data for {ticker}")
            return pd.DataFrame()
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        out = pd.DataFrame(index=df.index)
        out[f"{label}_close"] = df["Close"].values
        if "Volume" in df.columns:
            vol = df["Volume"].values
            # Only include volume if it has meaningful values
            if not (vol == 0).all():
                out[f"{label}_volume"] = vol
        return out
    except Exception as e:
        print(f"    ✗ Error downloading {ticker}: {e}")
        return pd.DataFrame()


def collect_yfinance_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Collect all yfinance-based data and merge on date index."""
    frames = []
    for label, ticker in TICKERS.items():
        df = _download_yf(ticker, label, start_date, end_date)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)  # polite delay

    if not frames:
        print("    ⚠ No yfinance data collected.")
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")

    merged.index.name = "date"
    merged.sort_index(inplace=True)
    return merged


# ================================================================
#  2) FRED data (macroeconomic indicators)
# ================================================================
def _fetch_fred_series(series_id: str, label: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a single FRED series via REST API."""
    if not FRED_API_KEY:
        return pd.DataFrame()
    
    print(f"  ↓ FRED: {label} ({series_id})")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            return pd.DataFrame()
        
        records = []
        for o in obs:
            val = o["value"]
            if val == ".":
                continue
            records.append({"date": o["date"], label: float(val)})
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    except Exception as e:
        print(f"    ✗ FRED error for {series_id}: {e}")
        return pd.DataFrame()


def collect_fred_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Collect all FRED macro series and merge."""
    if not FRED_API_KEY:
        print("  ⚠ No FRED API key — skipping macro data from FRED")
        return pd.DataFrame()

    frames = []
    for label, series_id in FRED_SERIES.items():
        df = _fetch_fred_series(series_id, label, start_date, end_date)
        if not df.empty:
            frames.append(df)
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    return merged


# ================================================================
#  3) On-chain data — Bitcoin Hash Rate
# ================================================================
def collect_hashrate() -> pd.DataFrame:
    """Fetch BTC hash rate from Blockchain.com API."""
    print("  ↓ Blockchain.com: hash rate")
    url = "https://api.blockchain.info/charts/hash-rate"
    params = {
        "timespan": "all",
        "format": "json",
        "sampled": "true",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        values = data.get("values", [])
        if not values:
            return pd.DataFrame()

        records = [{"date": dt.datetime.utcfromtimestamp(v["x"]), "hashrate": v["y"]}
                   for v in values]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date").set_index("date")
        return df
    except Exception as e:
        print(f"    ✗ Hash rate error: {e}")
        return pd.DataFrame()


# ================================================================
#  4) Fear & Greed Index
# ================================================================
def collect_fear_greed() -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index from Alternative.me."""
    print("  ↓ Alternative.me: Fear & Greed Index")
    url = "https://api.alternative.me/fng/"
    params = {"limit": "0", "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return pd.DataFrame()

        records = [{"date": dt.datetime.utcfromtimestamp(int(d["timestamp"])),
                     "fear_greed": int(d["value"])}
                   for d in data]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()
        return df
    except Exception as e:
        print(f"    ✗ Fear & Greed error: {e}")
        return pd.DataFrame()


# ================================================================
#  5) Optional futures/rates/geopolitical sources (feature flags)
# ================================================================
def collect_futures_term_structure_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collect optional commodity futures source series.
    Uses continuous front contracts from Yahoo as fail-forward baseline.
    """
    cfg_fut = cfg.futures_feature_config
    if not cfg_fut.get("enabled", False):
        return pd.DataFrame()

    print("  ↓ Optional: futures term-structure source data")
    assets = cfg_fut.get("assets", ["oil", "gold", "corn", "wheat"])
    frames = []
    for asset in assets:
        ticker = FUTURES_TICKER_MAP.get(asset)
        if not ticker:
            continue
        df = _download_yf(ticker, f"{asset}_fut", start_date, end_date)
        if not df.empty:
            frames.append(df)
        time.sleep(0.25)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged.sort_index(inplace=True)
    return merged


def create_fomc_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create calendar-based FOMC timing features."""
    meetings = sorted(pd.to_datetime(FOMC_DATES_UTC))
    df = pd.DataFrame(index=index)
    days_to_next = []
    is_fomc_week = []

    for d in index:
        upcoming = [m for m in meetings if m >= d]
        if upcoming:
            delta = (upcoming[0] - d).days
            days_to_next.append(delta)
            is_fomc_week.append(1 if 0 <= delta <= 7 else 0)
        else:
            days_to_next.append(np.nan)
            is_fomc_week.append(0)

    df["days_to_fomc"] = days_to_next
    df["is_fomc_week"] = is_fomc_week
    return df


def collect_rates_expectation_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collect optional rates-expectation proxy series from yfinance.
    """
    cfg_rates = cfg.rates_expectation_config
    if not cfg_rates.get("enabled", False):
        return pd.DataFrame()

    print("  ↓ Optional: rates expectation proxy data")
    frames = []
    for label, ticker in RATES_PROXY_TICKERS.items():
        df = _download_yf(ticker, label, start_date, end_date)
        if not df.empty:
            frames.append(df)
        time.sleep(0.25)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged.sort_index(inplace=True)
    return merged


def collect_geopolitical_proxy_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collect optional geopolitical risk proxies.
    """
    cfg_geo = cfg.geopolitical_feature_config
    if not cfg_geo.get("enabled", False):
        return pd.DataFrame()

    print("  ↓ Optional: geopolitical risk proxy data")
    frames = []
    for label, ticker in GEOPOLITICAL_PROXY_TICKERS.items():
        df = _download_yf(ticker, label, start_date, end_date)
        if not df.empty:
            frames.append(df)
        time.sleep(0.25)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="outer")
    merged.sort_index(inplace=True)
    return merged


# ================================================================
#  6) Event dummy variables
# ================================================================
def create_event_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create binary dummy columns for known macro/geopolitical events."""
    print("  ✓ Creating event dummy variables")
    df = pd.DataFrame(index=index)
    for start, end, label in EVENTS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end) if end else index.max()
        df[f"event_{label}"] = ((index >= start_dt) & (index <= end_dt)).astype(int)
    return df


# ================================================================
#  7) Halving features
# ================================================================
def create_halving_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create halving-related features: days since last halving, halving era."""
    print("  ✓ Creating halving features")
    halving_ts = [pd.Timestamp(h) for h in HALVING_DATES]
    df = pd.DataFrame(index=index)

    days_since = []
    era = []
    for d in index:
        past = [h for h in halving_ts if h <= d]
        if past:
            days_since.append((d - max(past)).days)
            era.append(len(past))
        else:
            days_since.append(np.nan)
            era.append(0)

    df["days_since_halving"] = days_since
    df["halving_era"] = era
    return df


# ================================================================
#  Master collection function
# ================================================================
def collect_all_data(start_date: str = DATA_START, 
                     end_date: str = DATA_END,
                     save_path: str = None) -> pd.DataFrame:
    """
    Collect all data sources, merge into a single daily DataFrame,
    and optionally save to disk.
    """
    print("=" * 60)
    print("BTC Prediction — Data Collection")
    print(f"Range: {start_date} to {end_date}")
    print("=" * 60)

    if save_path is None:
        save_path = os.path.join(cfg.processed_dir, "merged_dataset.csv")

    # 1) yfinance market data
    print("\n[1/8] Collecting market data (yfinance)...")
    yf_data = collect_yfinance_data(start_date, end_date)
    if yf_data.empty:
        print("Critical Data Missing: yfinance data is empty.")
        # We can't proceed without price data
        return pd.DataFrame()

    # 2) FRED macro data
    print("\n[2/8] Collecting macroeconomic data (FRED)...")
    fred_data = collect_fred_data(start_date, end_date)

    # 3) On-chain
    print("\n[3/8] Collecting on-chain data...")
    hashrate = collect_hashrate()

    # 4) Sentiment
    print("\n[4/8] Collecting sentiment data...")
    fng = collect_fear_greed()

    # 5) Optional futures term-structure sources
    print("\n[5/8] Collecting optional futures data...")
    futures_data = collect_futures_term_structure_data(start_date, end_date)

    # 6) Optional rates expectation sources
    print("\n[6/8] Collecting optional rates expectation data...")
    rates_data = collect_rates_expectation_data(start_date, end_date)

    # 7) Optional geopolitical proxies
    print("\n[7/8] Collecting optional geopolitical proxy data...")
    geo_data = collect_geopolitical_proxy_data(start_date, end_date)

    # 8) Event dummies, halving, calendar
    print("\n[8/8] Creating engineered features...")
    idx = yf_data.index
    events = create_event_dummies(idx)
    halving = create_halving_features(idx)
    fomc_calendar = create_fomc_calendar_features(idx)

    # ── Merge everything ──
    print("\nMerging all datasets...")
    merged = yf_data.copy()

    if not fred_data.empty:
        merged = merged.join(fred_data, how="left")

    if not hashrate.empty:
        merged = merged.join(hashrate, how="left")

    if not fng.empty:
        merged = merged.join(fng, how="left")

    if not futures_data.empty:
        merged = merged.join(futures_data, how="left")

    if not rates_data.empty:
        merged = merged.join(rates_data, how="left")

    if not geo_data.empty:
        merged = merged.join(geo_data, how="left")

    merged = merged.join(events, how="left")
    merged = merged.join(halving, how="left")
    merged = merged.join(fomc_calendar, how="left")

    # Forward-fill monthly/quarterly data and weekend gaps
    merged = merged.ffill()

    # Filter date range
    merged = merged.loc[start_date:end_date]

    # Summary
    print(f"\n✅ Final dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
    if not merged.empty:
        print(f"   Date range: {merged.index.min().date()} — {merged.index.max().date()}")
    print(f"   Columns: {list(merged.columns)}")
    missing = merged.isnull().sum()
    if missing.any():
        print(f"   Missing values:\n{missing[missing > 0]}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        merged.to_csv(save_path)
        print(f"\n💾 Saved to {save_path}")

        # Also save raw yfinance separately for reference
        raw_path = os.path.join(cfg.raw_dir, "yfinance_data.csv")
        yf_data.to_csv(raw_path)

    return merged


# ================================================================
if __name__ == "__main__":
    df = collect_all_data()
