"""
BTC Price Prediction — Backtester Module
==========================================
Walk-forward backtesting to validate model reliability.
For each date in the validation period, predict future prices
at all horizons and compare with actual outcomes.
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import cfg
from src.feature_engineer import HORIZONS, DEFAULT_HORIZONS

def load_phase_models(phase: int, run_id: str = None):
    """Load all horizon models for a phase."""
    models = {}
    
    if run_id:
        phase_dir = os.path.join(cfg.models_dir, run_id, f"phase{phase}")
    else:
        phase_dir = os.path.join(cfg.models_dir, f"phase{phase}")
    
    # If not found, try without run_id (legacy/latest) if run_id was not provided
    if not os.path.exists(phase_dir) and not run_id:
         # Maybe it is in models/phaseX directly?
         pass
    
    if not os.path.exists(phase_dir):
        print(f"Warning: Phase directory not found: {phase_dir}")
        return {}

    # Check for manifest to get horizons, or use defaults
    manifest_path = os.path.join(phase_dir, "manifest.json")
    horizons = DEFAULT_HORIZONS
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                m = json.load(f)
                horizons = m.get("horizons", horizons)
        except:
            pass

    for h in horizons:
        h_dir = os.path.join(phase_dir, f"horizon_{h}d")
        if not os.path.exists(h_dir):
            continue
        
        model_files = [f for f in os.listdir(h_dir) if f.startswith("best_model_")]
        if not model_files:
            continue
        
        model = joblib.load(os.path.join(h_dir, model_files[0]))
        scaler = joblib.load(os.path.join(h_dir, "scaler.joblib"))
        with open(os.path.join(h_dir, "feature_names.json")) as f:
            feature_names = json.load(f)
        
        model_name = model_files[0].replace("best_model_", "").replace(".joblib", "")
        models[h] = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "model_name": model_name,
        }
    
    return models


def run_backtest(phase: int, sample_interval: int = 30, run_id: str = None):
    """
    Walk-forward backtest for a given phase.
    """
    # Load data
    path = os.path.join(cfg.processed_dir, "featured_dataset.csv")
    if not os.path.exists(path):
        print("Featured dataset not found.")
        return pd.DataFrame()
        
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Load models
    models = load_phase_models(phase, run_id)
    if not models:
        print(f"No models found for phase {phase}")
        return pd.DataFrame()
    
    # Determine validation range from config
    phase_cfg = cfg.model_config.get("phases", {}).get(f"phase{phase}", {})
    val_range = phase_cfg.get("val")
    
    if not val_range:
        print(f"Phase {phase} has no validation range defined.")
        return pd.DataFrame()
    
    val_start, val_end = val_range
    v_start = val_start if val_start is not None else df.index.min()
    v_end   = val_end if val_end is not None else df.index.max()
    
    # Get validation dates
    val_mask = (df.index >= v_start) & (df.index <= v_end)
    val_dates = df.index[val_mask]
    
    if val_dates.empty:
         print(f"No data in validation range {val_start} to {val_end}")
         return pd.DataFrame()

    # Sample every N days to keep backtesting manageable
    sample_dates = val_dates[::sample_interval]
    
    print(f"Phase {phase} Backtest")
    print(f"  Validation: {val_start} — {val_end}")
    print(f"  Sample dates: {len(sample_dates)} (every {sample_interval} days)")
    print(f"  Horizons: {sorted(models.keys())}")
    
    results = []
    
    for pred_date in sample_dates:
        btc_price_now = df.loc[pred_date, "btc_close"]
        
        for h, hm in models.items():
            # Check if we have actual future data
            target_date = pred_date + pd.Timedelta(days=h)
            
            if target_date > df.index[-1]:
                continue  # Can't validate, future date beyond data
            
            # Get nearest actual future date
            future_idx = df.index.get_indexer([target_date], method="nearest")[0]
            actual_future_date = df.index[future_idx]
            actual_future_price = df.iloc[future_idx]["btc_close"]
            actual_log_return = np.log(actual_future_price / btc_price_now)
            
            # Make prediction
            available_features = [f for f in hm["feature_names"] if f in df.columns]
            features = df.loc[pred_date, available_features].copy()
            
            # Fill missing features
            full_features = pd.Series(0.0, index=hm["feature_names"])
            for f in available_features:
                full_features[f] = features[f] if pd.notna(features[f]) else 0.0
            
            # Check shape - scaler expects 2D
            X = hm["scaler"].transform(full_features.values.reshape(1, -1))
            pred_log_return = hm["model"].predict(X)[0]
            
            pred_price = btc_price_now * np.exp(pred_log_return)
            
            # Price ratio: predicted / actual (1.0 = perfect)
            price_ratio = pred_price / actual_future_price if actual_future_price > 0 else np.nan
            
            results.append({
                "prediction_date": pred_date,
                "horizon_days": h,
                "btc_price_at_prediction": round(btc_price_now, 2),
                "predicted_log_return": round(pred_log_return, 6),
                "actual_log_return": round(actual_log_return, 6),
                "predicted_price": round(pred_price, 2),
                "actual_price": round(actual_future_price, 2),
                "price_ratio": round(price_ratio, 4),
                "actual_date": actual_future_date,
            })
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No backtest results generated.")
        return results_df

    # Print summary by horizon
    print(f"\n{'='*70}")
    print(f"{'Horizon':>8} | {'Samples':>7} | {'MAPE':>8} | {'DirAcc':>7} | "
          f"{'Avg Ratio':>9} | {'Med Ratio':>9}")
    print(f"{'-'*70}")
    
    for h in sorted(models.keys()):
        hdf = results_df[results_df["horizon_days"] == h]
        if hdf.empty:
            continue
        
        mape = np.mean(np.abs(hdf["predicted_price"] - hdf["actual_price"]) /
                       hdf["actual_price"]) * 100
        dir_acc = np.mean(
            (hdf["predicted_log_return"] > 0) == (hdf["actual_log_return"] > 0)
        )
        avg_ratio = hdf["price_ratio"].mean()
        med_ratio = hdf["price_ratio"].median()
        
        print(f"{h:>6}d | {len(hdf):>7} | {mape:>7.1f}% | {dir_acc:>6.1%} | "
              f"{avg_ratio:>9.3f} | {med_ratio:>9.3f}")
    
    print(f"{'='*70}")

    # Save results
    if run_id:
        out_dir = os.path.join(cfg.models_dir, run_id, f"phase{phase}")
    else:
        out_dir = os.path.join(cfg.models_dir, f"phase{phase}")
        
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, "backtest_results.csv"), index=False)
    
    return results_df


if __name__ == "__main__":
    run_backtest(phase=1)
