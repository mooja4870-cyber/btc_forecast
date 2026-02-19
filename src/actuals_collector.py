"""
Actuals Collector
=================
Fetches the latest BTC/Financial data and updates the evaluation logs
by joining predictions with observed actuals.
"""

import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import yfinance as yf

from src.config import cfg

logger = logging.getLogger("actuals_collector")

def update_actuals():
    """
    1. Load predictions log
    2. Fetch latest BTC data (full history to be safe)
    3. Join and update eval_log.csv
    """
    preds_log_path = os.path.join(cfg.monitoring["logs_dir"], "predictions_log.csv")
    eval_log_path = os.path.join(cfg.monitoring["logs_dir"], "eval_log.csv")
    
    if not os.path.exists(preds_log_path):
        logger.warning("No predictions log found. Skipping actuals update.")
        return
        
    # Load Predictions
    preds_df = pd.read_csv(preds_log_path, parse_dates=["target_date", "ts"])
    
    # Fetch Actuals
    # We rely on yfinance for strictly the target variable (BTC-USD close)
    btc_ticker = cfg.data_config.get("tickers", {}).get("btc", "BTC-USD")
    start_date = preds_df["target_date"].min() - timedelta(days=5) # buffer
    end_date = datetime.now() + timedelta(days=1)
    
    logger.info(f"Fetching actuals for {btc_ticker} from {start_date.date()}")
    
    df_actuals = yf.download(btc_ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df_actuals.columns, pd.MultiIndex):
        df_actuals.columns = df_actuals.columns.get_level_values(0)
    
    # Prepare Actuals Series
    # Reset index to make date a column, ensure consistent formatting
    df_actuals = df_actuals[["Close"]].reset_index()
    df_actuals.columns = ["target_date", "actual_price"]
    df_actuals["target_date"] = pd.to_datetime(df_actuals["target_date"]).dt.tz_localize(None)
    preds_df["target_date"] = pd.to_datetime(preds_df["target_date"]).dt.tz_localize(None)

    # Merge
    # We want to keep all predictions, and attach actuals where available
    merged = pd.merge(preds_df, df_actuals, on="target_date", how="left")
    
    # Calculate Errors where actuals exist
    merged["error"] = merged["predicted_price"] - merged["actual_price"]
    merged["abs_error"] = merged["error"].abs()
    merged["error_pct"] = (merged["error"] / merged["actual_price"]) * 100
    
    # Save to eval_log
    os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)
    merged.to_csv(eval_log_path, index=False)
    
    valid_count = merged["actual_price"].notna().sum()
    logger.info(f"Updated eval log. Total rows: {len(merged)}, Rows with actuals: {valid_count}")
    
    return merged[merged["actual_price"].notna()]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_actuals()
