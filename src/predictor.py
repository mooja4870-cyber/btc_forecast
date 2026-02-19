"""
BTC Price Prediction — Predictor Module (Direct Multi-Horizon)
================================================================
Provides prediction functions using direct multi-horizon models.
Each horizon (1d, 2d, 3d, 5d, 7d, 30d, 60d, 90d, 180d, 365d) has a separate model
that predicts directly — no recursive error compounding.
"""

import os
import sys
import json
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODELS_DIR, PROCESSED_DIR
from src.config import cfg
from src.feature_engineer import HORIZONS


def _target_prediction_horizons():
    """Horizons to display/use for forecast tables and interpolation."""
    configured = cfg.features_config.get("horizons", HORIZONS) or HORIZONS
    base = [int(h) for h in configured]
    # Ensure "today(0d)" and short-term horizons are always present in outputs.
    return sorted(set(base + [0, 1, 2, 3, 5, 15]))


def _augment_predictions_with_interpolation(
    pred_df: pd.DataFrame,
    current_price: float,
    actual_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fill missing target horizons via interpolation/extrapolation on log-returns."""
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy().sort_values("horizon_days").reset_index(drop=True)
    have_horizons = set(out["horizon_days"].astype(int).tolist())
    want_horizons = _target_prediction_horizons()
    missing = [h for h in want_horizons if h not in have_horizons]
    if not missing:
        return out

    known_h = out["horizon_days"].astype(int).values
    known_lr = out["predicted_log_return"].astype(float).values
    first_h = int(known_h[0])
    first_lr = float(known_lr[0])

    rows = []
    for h in missing:
        if h == 0:
            est_lr = 0.0
        elif h <= first_h:
            est_lr = first_lr * (h / first_h) if first_h > 0 else first_lr
        else:
            est_lr = float(np.interp(h, known_h, known_lr))

        est_price = float(current_price) * float(np.exp(est_lr))
        est_ret = (float(np.exp(est_lr)) - 1.0) * 100.0
        rows.append(
            {
                "horizon_days": int(h),
                "target_date": actual_date + timedelta(days=int(h)),
                "predicted_log_return": round(est_lr, 6),
                "predicted_pct_return": round(est_ret, 2),
                "predicted_price": round(est_price, 2),
                "model_name": "Current" if int(h) == 0 else "transformer (interpolated)",
            }
        )

    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
        out = out.sort_values("horizon_days").drop_duplicates(subset=["horizon_days"], keep="first")
        out = out.reset_index(drop=True)

    # Force 0-day row to represent current snapshot exactly.
    if (out["horizon_days"] == 0).any():
        idx0 = out.index[out["horizon_days"] == 0][0]
        out.loc[idx0, "target_date"] = actual_date
        out.loc[idx0, "predicted_log_return"] = 0.0
        out.loc[idx0, "predicted_pct_return"] = 0.0
        out.loc[idx0, "predicted_price"] = round(float(current_price), 2)
        out.loc[idx0, "model_name"] = "Current"

    return out


def _phase_num_from_name(phase_name: str) -> int:
    if isinstance(phase_name, str) and phase_name.lower().startswith("phase"):
        tail = phase_name[5:]
        if tail.isdigit():
            return int(tail)
    return -1


def _default_production_phase() -> int:
    phases_cfg = cfg.model_config.get("phases", {})
    nums = [_phase_num_from_name(name) for name in phases_cfg.keys()]
    nums = [n for n in nums if n > 0]
    return max(nums) if nums else 3


def _resolve_phase_dir(phase: int = None) -> str:
    """Prefer promoted latest run artifacts when available."""
    if phase is None:
        phase = _default_production_phase()
    latest_phase_dir = os.path.join(MODELS_DIR, "latest", f"phase{phase}")
    if os.path.exists(latest_phase_dir):
        return latest_phase_dir
    return os.path.join(MODELS_DIR, f"phase{phase}")


def load_horizon_model(phase: int = None, horizon: int = 30):
    """Deprecated legacy loader. Transformer-only policy is enforced."""
    raise RuntimeError(
        "Legacy horizon models are disabled. Use load_transformer_model(horizon) instead."
    )


def load_transformer_model(horizon: int):
    """Load a Transformer model for a specific horizon."""
    h_dir = os.path.join(MODELS_DIR, "transformer", f"horizon_{horizon}d")
    
    if not os.path.exists(h_dir):
        return None
        
    try:
        # Load metadata
        with open(os.path.join(h_dir, "metadata.json")) as f:
            metadata = json.load(f)
            
        feature_cols = metadata["feature_cols"]
        
        # Load scaler stats
        scaler_stats = joblib.load(os.path.join(h_dir, "scaler_stats.joblib"))
        
        # Initialize model
        from src.transformer_model import TimeSformer
        import torch
        
        device = torch.device("cpu") # Use CPU for inference to avoid issues
        model = TimeSformer(num_features=len(feature_cols))
        model.load_state_dict(torch.load(os.path.join(h_dir, "model.pth"), map_location=device))
        model.to(device)
        model.eval()
        
        return {
            "model": model,
            "scaler_stats": scaler_stats,
            "feature_names": feature_cols,
            "model_name": "transformer",
            "type": "transformer",
            "seq_len": metadata.get("seq_len", 60)
        }
    except Exception as e:
        print(f"  ⚠️ Failed to load Transformer for {horizon}d: {e}")
        return None


def load_all_horizon_models(
    phase: int = None,
    model_preference: str = "transformer",
    allow_fallback: bool = False,
):
    """
    Load Transformer models for all available horizons.
    Legacy fallback is intentionally disabled.
    """
    models = {}
    _ = phase  # kept for API compatibility
    _ = allow_fallback
    if str(model_preference).lower() != "transformer":
        raise ValueError("Transformer-only policy active: model_preference must be 'transformer'.")

    for h in HORIZONS:
        h = int(h)
        if h <= 0:
            continue
        tf_model = load_transformer_model(h)
        if tf_model:
            models[h] = tf_model
    
    return models


def get_features_at_date(date: pd.Timestamp, feature_names: list,
                         df: pd.DataFrame = None) -> tuple:
    """Get features at a specific date (or nearest available)."""
    if df is None:
        path = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Find nearest date
    idx = df.index.get_indexer([date], method="nearest")[0]
    actual_date = df.index[idx]
    
    available_features = [f for f in feature_names if f in df.columns]
    features = df.iloc[idx][available_features]
    features = features.fillna(0)
    
    # Add any missing features as 0
    for f in feature_names:
        if f not in features.index:
            features[f] = 0
    
    btc_price = df.iloc[idx]["btc_close"]
    
    return features[feature_names], actual_date, btc_price, df


def predict_multi_horizon(
    phase: int = None,
    from_date: pd.Timestamp = None,
    model_preference: str = "transformer",
    allow_fallback: bool = False,
):
    """
    Predict BTC price at all available horizons using direct models.
    Each horizon model makes ONE independent prediction.
    
    Returns DataFrame with horizon, predicted return, predicted price.
    """
    if phase is None:
        phase = _default_production_phase()

    horizon_models = load_all_horizon_models(
        phase=phase,
        model_preference=model_preference,
        allow_fallback=allow_fallback,
    )
    
    if not horizon_models:
        raise ValueError(
            "No Transformer models found. Train them first with: "
            "`python -m src.train_transformer`"
        )
    
    # Use the first model's feature list to get features
    first_h = list(horizon_models.keys())[0]
    ref_model = horizon_models[first_h]
    
    if from_date is None:
        path = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        from_date = df.index[-1]
    else:
        df = None
    
    features, actual_date, current_price, df = get_features_at_date(
        from_date, ref_model["feature_names"], df
    )

    predictions = []
    
    for h in sorted(horizon_models.keys()):
        hm = horizon_models[h]
        # Get features for this specific model
        feat, _, _, _ = get_features_at_date(actual_date, hm["feature_names"], df)
        
        if hm.get("type") != "transformer":
            raise RuntimeError("Unexpected non-transformer model loaded under Transformer-only policy.")

        # Transformer Inference
        import torch

        # Scale
        stats = hm["scaler_stats"]

        # Transformer needs sequence of length seq_len
        seq_len = hm["seq_len"]

        # Get data ending at actual_date
        if df is not None:
            idx_loc = df.index.get_loc(actual_date)
            start_loc = idx_loc - seq_len + 1
            if start_loc < 0:
                start_loc = 0

            # Fetch sequence
            X_seq_df = df.iloc[start_loc : idx_loc + 1]
            X_seq = X_seq_df[hm["feature_names"]].fillna(0).values

            # If sequence is short, pad it
            if len(X_seq) < seq_len:
                pad_len = seq_len - len(X_seq)
                X_seq = np.pad(X_seq, ((pad_len, 0), (0, 0)), mode="constant")

            # Scale sequence
            X_seq_scaled = (X_seq - stats["mean"]) / stats["std"]

            # Tensor
            X_tensor = torch.tensor(X_seq_scaled, dtype=torch.float32).unsqueeze(0)  # [1, seq, feat]

            with torch.no_grad():
                pred_log_return = hm["model"](X_tensor).item()
        else:
            pred_log_return = 0.0
        
        pred_price = current_price * np.exp(pred_log_return)
        pct_return = (np.exp(pred_log_return) - 1) * 100
        
        predictions.append({
            "horizon_days": h,
            "target_date": actual_date + timedelta(days=h),
            "predicted_log_return": round(pred_log_return, 6),
            "predicted_pct_return": round(pct_return, 2),
            "predicted_price": round(pred_price, 2),
            "model_name": hm["model_name"],
        })
    
    pred_df = pd.DataFrame(predictions)
    pred_df = _augment_predictions_with_interpolation(
        pred_df=pred_df,
        current_price=current_price,
        actual_date=actual_date,
    )
    
    return pred_df, current_price, actual_date


def estimate_target_return_date(
    target_return_pct: float,
    phase: int = None,
    max_months: int = 12,
    from_date: str = None,
    model_preference: str = "transformer",
    allow_fallback: bool = False,
):
    """
    Given a target return percentage, estimate when it might be reached
    using direct multi-horizon predictions with interpolation.
    """
    phase_used = phase if phase is not None else _default_production_phase()
    pred_df, current_price, start_date = predict_multi_horizon(
        phase_used,
        from_date=from_date,
        model_preference=model_preference,
        allow_fallback=allow_fallback,
    )
    
    target_price = current_price * (1 + target_return_pct / 100)
    
    result = {
        "current_price": round(current_price, 2),
        "current_date": str(start_date.date()),
        "target_return_pct": target_return_pct,
        "target_price": round(target_price, 2),
        "model_phase": phase_used,
        "forecast_path": pred_df,
    }
    
    # Check if any horizon prediction crosses the target
    if target_return_pct >= 0:
        crossed = pred_df[pred_df["predicted_price"] >= target_price]
    else:
        crossed = pred_df[pred_df["predicted_price"] <= target_price]
    
    if not crossed.empty:
        first_cross = crossed.iloc[0]
        
        # Try to interpolate for a more precise estimate
        idx = pred_df.index.get_loc(first_cross.name)
        if idx > 0:
            prev_row = pred_df.iloc[idx - 1]
            curr_row = pred_df.iloc[idx]
            
            # Linear interpolation of price to find exact date
            price_diff = curr_row["predicted_price"] - prev_row["predicted_price"]
            target_diff = target_price - prev_row["predicted_price"]
            ratio = target_diff / price_diff if price_diff != 0 else 0
            
            days_diff = curr_row["horizon_days"] - prev_row["horizon_days"]
            add_days = days_diff * ratio
            estimated_days = prev_row["horizon_days"] + add_days
        else:
            estimated_days = first_cross["horizon_days"]
            
        estimated_date = start_date + timedelta(days=estimated_days)
        
        result["reached"] = True
        result["estimated_days"] = int(estimated_days)
        result["estimated_date"] = str(estimated_date.date())
        result["estimated_price"] = round(target_price, 2)
        
    else:
        # Not reached within max horizon (365 days usually)
        if target_return_pct >= 0:
            final = pred_df.iloc[-1]
            best_return = pred_df["predicted_pct_return"].max()
        else:
            final = pred_df.iloc[-1]
            best_return = pred_df["predicted_pct_return"].min()
            
        result["max_forecast_return_pct"] = round(best_return, 2)
        result["estimated_price"] = round(final["predicted_price"], 2)
        result["max_forecast_days"] = int(final["horizon_days"])
        result["reached"] = False
    
    return result


def estimate_return_at_date(
    holding_days: int,
    phase: int = None,
    from_date: str = None,
    model_preference: str = "transformer",
    allow_fallback: bool = False,
):
    """
    Given a holding period in days, estimate the expected return
    using the closest horizon model (with interpolation if possible).
    """
    phase_used = phase if phase is not None else _default_production_phase()
    pred_df, current_price, start_date = predict_multi_horizon(
        phase_used,
        from_date=from_date,
        model_preference=model_preference,
        allow_fallback=allow_fallback,
    )
    
    target_date = start_date + timedelta(days=holding_days)
    
    # Find the two surrounding horizons for interpolation
    horizons = pred_df["horizon_days"].values
    pred_returns = pred_df["predicted_log_return"].values
    if len(horizons) > 0 and horizons[0] == 0:
        horizons = horizons[1:]
        pred_returns = pred_returns[1:]
    if len(horizons) == 0:
        raise ValueError("No positive horizon predictions available.")
    
    if holding_days <= horizons[0]:
        # Extrapolate from first horizon
        ratio = holding_days / horizons[0]
        est_log_return = pred_returns[0] * ratio
    elif holding_days >= horizons[-1]:
        # Use the last horizon
        est_log_return = pred_returns[-1]
    else:
        # Interpolate between two horizons
        est_log_return = np.interp(holding_days, horizons, pred_returns)
    
    estimated_price = current_price * np.exp(est_log_return)
    pct_return = (np.exp(est_log_return) - 1) * 100
    
    result = {
        "current_price": round(current_price, 2),
        "current_date": str(start_date.date()),
        "holding_days": holding_days,
        "target_date": str(target_date.date()),
        "estimated_price": round(float(estimated_price), 2),
        "estimated_return_pct": round(float(pct_return), 2),
        "estimated_log_return": round(float(est_log_return), 6),
        "model_phase": phase_used,
        "forecast_path": pred_df,
    }
    
    return result


# ================================================================
# Backward-compatible wrapper
# ================================================================
def load_latest_model(phase: int = None):
    """Backward-compatible wrapper: return Transformer 30d artifacts."""
    _ = phase
    tf = load_transformer_model(30)
    if not tf:
        raise FileNotFoundError(
            "Transformer 30d model not found. Train with `python -m src.train_transformer`."
        )
    return tf["model"], tf["scaler_stats"], tf["feature_names"], tf["model_name"]


def predict_future_path(model, scaler, feature_names: list,
                        n_steps: int = 12, horizon_days: int = 30):
    """
    Backward-compatible wrapper: now uses multi-horizon predictions
    instead of recursive forecasting.
    """
    pred_df, current_price, start_date = predict_multi_horizon()
    return pred_df, current_price, start_date


# ================================================================
if __name__ == "__main__":
    print("=== Multi-Horizon Direct Predictions ===")
    pred_df, current_price, start_date = predict_multi_horizon()
    print(f"\nCurrent: ${current_price:,.0f} ({start_date.date()})")
    print(f"\nPredictions:")
    for _, row in pred_df.iterrows():
        print(f"  {row['horizon_days']:3d}d | "
              f"${row['predicted_price']:>10,.0f} | "
              f"{row['predicted_pct_return']:>+7.1f}% | "
              f"{row['model_name']}")

    print("\n\n=== Target Return → Date ===")
    r1 = estimate_target_return_date(target_return_pct=50.0)
    print(f"Current: ${r1['current_price']} ({r1['current_date']})")
    print(f"Target: +{r1['target_return_pct']}% = ${r1['target_price']}")
    if r1["reached"]:
        print(f"Estimated: ~{r1['estimated_days']} days → {r1['estimated_date']}")
    else:
        print(f"Not reached in {r1['max_forecast_days']} days. "
              f"Max return: {r1['max_forecast_return_pct']:.1f}%")

    print("\n\n=== Holding Period → Return ===")
    r2 = estimate_return_at_date(holding_days=180)
    print(f"Current: ${r2['current_price']} ({r2['current_date']})")
    print(f"After {r2['holding_days']} days ({r2['target_date']}):")
    print(f"  Estimated price: ${r2['estimated_price']}")
    print(f"  Estimated return: {r2['estimated_return_pct']:.2f}%")
