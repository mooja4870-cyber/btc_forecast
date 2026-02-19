"""
BTC Price Prediction App — Configuration
"""
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# FRED API Key (https://fred.stlouisfed.org/docs/api/api_key.html)
# Set via environment variable or replace the default below
# ──────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ──────────────────────────────────────────────
# Date Ranges for Each Modeling Phase
# ──────────────────────────────────────────────
PHASE1_TRAIN = ("2014-01-01", "2017-12-31")
PHASE1_VAL   = ("2018-01-01", "2018-12-31")

PHASE2_TRAIN = ("2014-01-01", "2018-12-31")
PHASE2_VAL   = ("2019-01-01", "2019-12-31")

PHASE3_TRAIN = ("2014-01-01", "2019-12-31")
PHASE3_VAL   = ("2020-01-01", "2020-12-31")

PHASE4_TRAIN = ("2014-01-01", "2020-12-31")
PHASE4_VAL   = ("2021-01-01", "2022-12-31")

PHASE5_TRAIN = ("2014-01-01", "2022-12-31")
PHASE5_VAL   = ("2023-01-01", "2024-12-31")

PHASE6_TRAIN = ("2014-01-01", "2024-12-31")
PHASE6_VAL   = ("2025-01-01", "2026-12-31")

PRODUCTION_PHASE = 6

# ──────────────────────────────────────────────
# Prediction Horizon (days)
# ──────────────────────────────────────────────
PREDICTION_HORIZON = 30  # predict BTC price 30 days ahead

# ──────────────────────────────────────────────
# BTC Halving Dates
# ──────────────────────────────────────────────
HALVING_DATES = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-19",
]

# ──────────────────────────────────────────────
# Major Geopolitical / Market Events (dummy variables)
# Format: (start_date, end_date_or_None, label)
# ──────────────────────────────────────────────
EVENTS = [
    ("2020-03-01", "2020-06-30", "covid_crash"),
    ("2021-05-19", "2021-07-20", "china_btc_ban"),
    ("2022-02-24", None,         "ukraine_war"),
    ("2022-05-01", "2022-06-30", "luna_terra_crash"),
    ("2022-11-08", "2022-12-31", "ftx_collapse"),
    ("2024-01-10", None,         "spot_btc_etf"),
    ("2024-11-05", None,         "us_election_2024"),
]

# ──────────────────────────────────────────────
# yfinance Ticker Symbols
# ──────────────────────────────────────────────
TICKERS = {
    "btc":    "BTC-USD",
    "gold":   "GC=F",
    "oil":    "CL=F",
    "sp500":  "^GSPC",
    "nasdaq": "^IXIC",
    "dxy":    "DX-Y.NYB",
    "krw":    "KRW=X",
}

# ──────────────────────────────────────────────
# FRED Series IDs
# ──────────────────────────────────────────────
FRED_SERIES = {
    "fed_rate":       "FEDFUNDS",
    "cpi":            "CPIAUCSL",
    "m2":             "M2SL",
    "unemployment":   "UNRATE",
    "gdp":            "GDP",
    "treasury_10y":   "DGS10",
}
