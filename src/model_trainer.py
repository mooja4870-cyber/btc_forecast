"""
BTC Price Prediction — Model Trainer Module (Multi-Horizon)
============================================================
Direct multi-horizon model training:
  For each prediction horizon (1d, 7d, 30d, 60d, 90d, 180d, 365d),
  train separate models that predict directly to that horizon.

Walk-forward phases are defined in config/config.yaml.
This module trains all configured phases in order.
"""

import os
import sys
import warnings
import json
import hashlib
import datetime as dt

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import cfg
from src.feature_engineer import DEFAULT_HORIZONS, HORIZONS  # Use constants if needed

# ── Columns to exclude from features ──
# All target columns + raw btc_close (leakage)
TARGET_PREFIXES = ["target_log_return", "target_direction", "target_future_price"]
EXCLUDE_BASE = ["btc_close"]


def get_exclude_cols():
    """Build list of all columns to exclude from features."""
    exclude = list(EXCLUDE_BASE)
    for prefix in TARGET_PREFIXES:
        exclude.append(prefix)  # backward compat target
        for h in DEFAULT_HORIZONS:
            exclude.append(f"{prefix}_{h}d")
    return exclude


# ================================================================
#  Data Preparation
# ================================================================
def prepare_data(df: pd.DataFrame, train_range: tuple,
                 val_range: tuple = None, target_col: str = "target_log_return_30d"):
    """
    Split data into train/val sets, handle NaN, scale features.
    """
    exclude = get_exclude_cols()
    feature_cols = [c for c in df.columns if c not in exclude]
    leakage_like = [c for c in feature_cols if c.startswith("target_") or "future_price" in c]
    if leakage_like:
        raise ValueError(f"Leakage-like features detected in training set: {leakage_like[:10]}")

    # Bounds handling for dynamic ranges (None in config)
    t_start = train_range[0] if train_range[0] is not None else df.index.min()
    t_end   = train_range[1] if train_range[1] is not None else df.index.max()
    train_mask = (df.index >= t_start) & (df.index <= t_end)
    train_df = df.loc[train_mask].copy()

    # Drop columns that are > 50% NaN in training data
    nan_pct = train_df[feature_cols].isnull().mean()
    valid_cols = nan_pct[nan_pct < 0.5].index.tolist()

    # Drop rows where target is NaN
    train_df = train_df.dropna(subset=[target_col])
    train_df[valid_cols] = (
        train_df[valid_cols]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0)
    )

    if cfg.features_asof_config.get("enforce_no_future_data", True):
        non_null = train_df[valid_cols].notna().sum().sum()
        if non_null == 0:
            raise ValueError("All feature values are NaN after as-of processing.")

    X_train = train_df[valid_cols].values
    y_train = train_df[target_col].values

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_val, y_val = None, None
    val_df = None
    if val_range:
        v_start = val_range[0] if val_range[0] is not None else df.index.min()
        v_end   = val_range[1] if val_range[1] is not None else df.index.max()
        val_mask = (df.index >= v_start) & (df.index <= v_end)
        val_df = df.loc[val_mask].copy()
        val_df = val_df.dropna(subset=[target_col])
        val_df[valid_cols] = (
            val_df[valid_cols]
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0)
        )
        X_val = scaler.transform(val_df[valid_cols].values)
        y_val = val_df[target_col].values

    return X_train, y_train, X_val, y_val, valid_cols, scaler, train_df, val_df


# ================================================================
#  Model Definitions
# ================================================================
def get_models(phase: int = 1, horizon: int = 30):
    """Return dict of model name → model instance.
    Longer horizons get stronger regularization.
    Hyperparams loaded from config.
    """
    reg_factor = max(1.0, horizon / 30.0)
    
    # Load hyperparams from config
    models_cfg = cfg.model_config.get("models", {})
    ridge_cfg = models_cfg.get("ridge", {})
    rf_cfg = models_cfg.get("random_forest", {})
    xgb_cfg = models_cfg.get("xgboost", {})
    lgb_cfg = models_cfg.get("lightgbm", {})
    
    seed = cfg.model_config.get("random_seed", 42)

    models = {
        "ridge": Ridge(alpha=ridge_cfg.get("alpha_base", 10.0) * reg_factor),
        "random_forest": RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 300), 
            max_depth=rf_cfg.get("max_depth", 8),
            min_samples_leaf=30,
            random_state=seed, n_jobs=-1
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=xgb_cfg.get("n_estimators", 500), 
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.03),
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=1.0 * reg_factor, reg_lambda=3.0 * reg_factor,
            min_child_weight=20,
            random_state=seed, n_jobs=-1, verbosity=0
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=lgb_cfg.get("n_estimators", 500), 
            max_depth=lgb_cfg.get("max_depth", 4),
            learning_rate=lgb_cfg.get("learning_rate", 0.03),
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=1.0 * reg_factor, reg_lambda=3.0 * reg_factor,
            min_child_weight=20,
            random_state=seed, n_jobs=-1, verbose=-1
        ),
    }

    # Phase 2+: even more regularization (manual override logic kept from original)
    if phase >= 2:
        for name in ["xgboost", "lightgbm"]:
            m = models[name]
            m.set_params(
                reg_alpha=2.0 * reg_factor,
                reg_lambda=5.0 * reg_factor,
            )

    return models


# ================================================================
#  Evaluation
# ================================================================
def evaluate_model(y_true, y_pred, label: str = ""):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    dir_true = (y_true > 0).astype(int)
    dir_pred = (y_pred > 0).astype(int)
    dir_acc = (dir_true == dir_pred).mean()

    actual_ratio = np.exp(y_true)
    pred_ratio = np.exp(y_pred)
    mape_ratio = np.mean(np.abs(actual_ratio - pred_ratio) / actual_ratio) * 100

    metrics = {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 4),
        "direction_accuracy": round(dir_acc, 4),
        "price_mape_pct": round(mape_ratio, 2),
    }

    if label:
        print(f"  [{label}] RMSE={metrics['rmse']:.6f}  MAE={metrics['mae']:.6f}  "
              f"R²={metrics['r2']:.4f}  DirAcc={metrics['direction_accuracy']:.1%}  "
              f"PriceMAPE={metrics['price_mape_pct']:.1f}%")

    return metrics


# ================================================================
#  Feature Importance
# ================================================================
def get_feature_importance(model, feature_names, model_name: str):
    """Extract feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": imp
    }).sort_values("importance", ascending=False)

    return fi


# ================================================================
#  Single Horizon Training
# ================================================================
def train_horizon(phase: int, horizon: int, df: pd.DataFrame,
                  train_range: tuple, val_range: tuple = None,
                  verbose: bool = True):
    """Train models for a single horizon in a given phase."""
    target_col = f"target_log_return_{horizon}d"

    if target_col not in df.columns:
        print(f"  ⚠ Target column {target_col} not found, skipping.")
        return None

    X_train, y_train, X_val, y_val, feature_names, scaler, train_df, val_df = \
        prepare_data(df, train_range, val_range, target_col=target_col)

    if verbose:
        print(f"\n  Horizon {horizon}d: {len(X_train)} train", end="")
        if X_val is not None:
            print(f", {len(X_val)} val", end="")
        print(f", {len(feature_names)} features")

    models = get_models(phase, horizon)
    results = {}
    best_model = None
    best_name = None
    best_score = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_metrics = evaluate_model(y_train, y_train_pred,
                                        f"{name}/train/{horizon}d" if verbose else "")

        val_metrics = None
        if X_val is not None:
            y_val_pred = model.predict(X_val)
            val_metrics = evaluate_model(y_val, y_val_pred,
                                          f"{name}/val/{horizon}d" if verbose else "")
            score = val_metrics["r2"]
        else:
            score = train_metrics["r2"]

        fi = get_feature_importance(model, feature_names, name)

        results[name] = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "feature_importance": fi,
        }

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    if verbose:
        print(f"  🏆 Best {horizon}d model: {best_name} (R²={best_score:.4f})")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "best_score": best_score,
        "scaler": scaler,
        "feature_names": feature_names,
        "results": results,
        "train_df": train_df,
        "val_df": val_df,
    }


# ================================================================
#  Phase Training (all horizons)
# ================================================================
def train_and_evaluate(phase_name: str, phase_config: dict, df: pd.DataFrame = None,
                       horizons: list = None, run_id: str = None):
    """
    Train models for all horizons in a given phase.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    print("=" * 60)
    print(f"Phase {phase_name} — Multi-Horizon Model Training")
    print("=" * 60)

    # Load data
    if df is None:
        path = os.path.join(cfg.processed_dir, "featured_dataset.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Set ranges
    train_range = tuple(phase_config.get("train", []))
    val_range = tuple(phase_config.get("val", [])) if phase_config.get("val") else None
    
    # Parse phase number (phase1 -> 1) for logic reuse
    try:
        phase_num = int(phase_name.replace("phase", ""))
    except:
        phase_num = 1

    print(f"\nTrain: {train_range[0]} — {train_range[1]}")
    if val_range:
        print(f"Val:   {val_range[0]} — {val_range[1]}")

    # Artifact directory: models/<run_id>/<phase_name>/
    if run_id:
        phase_dir = os.path.join(cfg.models_dir, run_id, phase_name)
    else:
        # Fallback to old behavior if no run_id
        phase_dir = os.path.join(cfg.models_dir, phase_name)
        
    os.makedirs(phase_dir, exist_ok=True)

    horizon_results = {}

    for horizon in horizons:
        result = train_horizon(phase_num, horizon, df, train_range, val_range)
        if result is None:
            continue

        horizon_results[horizon] = result

        # Save per-horizon artifacts
        h_dir = os.path.join(phase_dir, f"horizon_{horizon}d")
        os.makedirs(h_dir, exist_ok=True)

        # Save model
        best_name = result["best_model_name"]
        model_path = os.path.join(h_dir, f"best_model_{best_name}.joblib")
        joblib.dump(result["best_model"], model_path)

        # Save scaler
        scaler_path = os.path.join(h_dir, "scaler.joblib")
        joblib.dump(result["scaler"], scaler_path)

        # Save feature names
        features_path = os.path.join(h_dir, "feature_names.json")
        with open(features_path, "w") as f:
            json.dump(result["feature_names"], f)

        # Save metrics
        metrics_summary = {}
        for name, res in result["results"].items():
            metrics_summary[name] = {
                "train": res["train_metrics"],
                "val": res["val_metrics"],
            }
        with open(os.path.join(h_dir, "metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # Save feature importance
        best_fi = result["results"][best_name]["feature_importance"]
        if not best_fi.empty:
            best_fi.to_csv(os.path.join(h_dir, "feature_importance.csv"), index=False)

        # Save validation predictions
        val_df = result.get("val_df")
        if val_df is not None:
            target_col = f"target_log_return_{horizon}d"
            X_val_data = val_df[[c for c in result["feature_names"]
                                 if c in val_df.columns]]
            X_val_data = X_val_data.ffill().fillna(0)
            X_val_scaled = result["scaler"].transform(X_val_data.values)
            val_preds = result["best_model"].predict(X_val_scaled)

            pred_df = pd.DataFrame({
                "date": val_df.index,
                "actual_log_return": val_df[target_col].values,
                "predicted_log_return": val_preds,
                # "actual_btc_close": val_df["btc_close"].values if "btc_close" in val_df.columns else np.nan,
            })
            if "btc_close" in val_df.columns:
                 pred_df["actual_btc_close"] = val_df["btc_close"].values

            pred_df.to_csv(os.path.join(h_dir, "val_predictions.csv"), index=False)

    # ── Manifest Generation ──
    manifest = {
        "phase": phase_name,
        "run_id": run_id,
        "timestamp": dt.datetime.utcnow().isoformat(),
        "train_range": train_range,
        "val_range": val_range,
        "horizons": horizons,
        "best_models": {h: res["best_model_name"] for h, res in horizon_results.items()},
        "metrics_summary": {
            h: res["results"][res["best_model_name"]]["val_metrics"] 
            for h, res in horizon_results.items() 
            if res["results"][res["best_model_name"]]["val_metrics"]
        }
    }
    with open(os.path.join(phase_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Also save a backward-compatible metrics.json at phase level (for 30d)
    if 30 in horizon_results:
        r30 = horizon_results[30]
        compat_metrics = {}
        for name, res in r30["results"].items():
            compat_metrics[name] = {
                "train": res["train_metrics"],
                "val": res["val_metrics"],
            }
        with open(os.path.join(phase_dir, "metrics.json"), "w") as f:
            json.dump(compat_metrics, f, indent=2)

        # Save compat scaler & feature names & model at phase level too
        joblib.dump(r30["scaler"], os.path.join(phase_dir, "scaler.joblib"))
        with open(os.path.join(phase_dir, "feature_names.json"), "w") as f:
            json.dump(r30["feature_names"], f)

        best_name = r30["best_model_name"]
        joblib.dump(r30["best_model"],
                    os.path.join(phase_dir, f"best_model_{best_name}.joblib"))

        # Feature importance
        best_fi = r30["results"][best_name]["feature_importance"]
        if not best_fi.empty:
            best_fi.to_csv(os.path.join(phase_dir, "feature_importance.csv"),
                           index=False)
        
        # Val preds
        val_df = r30.get("val_df")
        if val_df is not None:
             # Re-predict for compatibility file
            target_col = "target_log_return_30d"
            X_val_data = val_df[[c for c in r30["feature_names"] if c in val_df.columns]].ffill().fillna(0)
            X_val_scaled = r30["scaler"].transform(X_val_data.values)
            val_preds = r30["best_model"].predict(X_val_scaled)
            
            pred_df = pd.DataFrame({
                "date": val_df.index,
                "actual_log_return": val_df[target_col].values,
                "predicted_log_return": val_preds,
            })
            if "btc_close" in val_df.columns:
                pred_df["actual_btc_close"] = val_df["btc_close"].values
            
            pred_df.to_csv(os.path.join(phase_dir, "val_predictions.csv"), index=False)

    print(f"\n💾 Phase {phase_name} models saved to {phase_dir}/")

    return horizon_results


# ================================================================
#  Run all configured phases (Legacy Entrypoint)
# ================================================================
def run_all_phases():
    """Run all configured phases sequentially using config phases."""
    path = os.path.join(cfg.processed_dir, "featured_dataset.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    phases = cfg.model_config.get("phases", {})
    phase_names = sorted(
        [p for p in phases.keys() if str(p).startswith("phase")],
        key=lambda x: int(str(x).replace("phase", "")) if str(x).replace("phase", "").isdigit() else 10**9,
    )
    
    results = {}
    for phase_name in phase_names:
        if phase_name in phases:
             # Use a dummy run_id for legacy execution
             run_id = f"legacy_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
             results[phase_name] = train_and_evaluate(phase_name, phases[phase_name], df, run_id=run_id)
             print("\n" + "=" * 60 + "\n")
    
    return results


# ================================================================
if __name__ == "__main__":
    run_all_phases()
