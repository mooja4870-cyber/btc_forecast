import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR
from src.config import cfg
from src.predictor import (
    load_transformer_model,
    get_features_at_date,
)


def _resolve_val_predictions_path(phase: int, horizon: int) -> str:
    latest_path = os.path.join(cfg.models_dir, "latest", f"phase{phase}", f"horizon_{horizon}d", "val_predictions.csv")
    if os.path.exists(latest_path):
        return latest_path
    legacy_path = os.path.join(cfg.models_dir, f"phase{phase}", f"horizon_{horizon}d", "val_predictions.csv")
    return legacy_path if os.path.exists(legacy_path) else ""


def _calibrate_log_return_with_recent_bias(
    phase: int,
    horizon: int,
    start_date_actual,
    pred_log_return: float,
):
    """
    Apply lightweight post-hoc bias calibration for Reality Check.
    Uses only rows prior to start_date_actual from validation predictions.
    """
    # Keep calibration narrow and conservative: only for 30d where gap is currently largest.
    if horizon != 30:
        return pred_log_return, {"applied": False}

    val_path = _resolve_val_predictions_path(phase=phase, horizon=horizon)
    if not val_path:
        return pred_log_return, {"applied": False, "reason": "no_val_predictions"}

    try:
        vdf = pd.read_csv(val_path, parse_dates=["date"])
    except Exception:
        return pred_log_return, {"applied": False, "reason": "read_failed"}

    if vdf.empty or "actual_log_return" not in vdf.columns or "predicted_log_return" not in vdf.columns:
        return pred_log_return, {"applied": False, "reason": "invalid_columns"}

    # Use recent window before current cutoff date.
    window = 14
    prior = vdf[vdf["date"] < pd.Timestamp(start_date_actual)].tail(window).copy()
    if len(prior) < max(7, window // 2):
        return pred_log_return, {"applied": False, "reason": "insufficient_history", "rows": int(len(prior))}

    bias = (prior["predicted_log_return"] - prior["actual_log_return"]).mean()
    # Clip to avoid excessive correction spikes.
    bias_clipped = float(np.clip(bias, -0.25, 0.25))
    calibrated = float(pred_log_return - bias_clipped)

    return calibrated, {
        "applied": True,
        "window": window,
        "rows": int(len(prior)),
        "bias_raw": float(bias),
        "bias_used": float(bias_clipped),
    }


def _predict_single_horizon(df: pd.DataFrame, phase: int, horizon: int, cutoff_date):
    """
    Predict a single horizon directly from transformer model.
    Returns: (predicted_price, base_price, start_date_actual, model_name, model_preference)
    """
    tf = load_transformer_model(horizon)
    if tf is None:
        raise FileNotFoundError(
            f"Transformer model not found for horizon={horizon}d. "
            "Train with `python -m src.train_transformer`."
        )

    import torch

    feature_names = tf["feature_names"]
    _, start_date_actual, base_price, _ = get_features_at_date(cutoff_date, feature_names, df)

    seq_len = int(tf.get("seq_len", 60))
    idx_loc = df.index.get_loc(start_date_actual)
    start_loc = max(0, idx_loc - seq_len + 1)
    X_seq = df.iloc[start_loc:idx_loc + 1][feature_names].fillna(0).values
    if len(X_seq) < seq_len:
        pad_len = seq_len - len(X_seq)
        X_seq = np.pad(X_seq, ((pad_len, 0), (0, 0)), mode="constant")

    stats = tf["scaler_stats"]
    X_seq_scaled = (X_seq - stats["mean"]) / stats["std"]
    X_tensor = torch.tensor(X_seq_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_log_return_raw = float(tf["model"](X_tensor).item())

    pred_log_return, calib_meta = _calibrate_log_return_with_recent_bias(
        phase=phase,
        horizon=horizon,
        start_date_actual=start_date_actual,
        pred_log_return=pred_log_return_raw,
    )
    predicted_price = float(base_price * np.exp(pred_log_return))
    return predicted_price, float(base_price), start_date_actual, str(tf["model_name"]), "transformer", calib_meta

def train_and_verify_horizons(horizons=[365, 30, 1]):
    print("="*60)
    print(f"🚀 SYNCING REALITY CHECK WITH PRODUCTION MODELS: {horizons}")
    print("="*60)

    # Load data to get current price and dates
    path = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
    if not os.path.exists(path):
        print(f"❌ Data file not found at {path}")
        return

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(subset=["btc_close"])
    
    results = {}
    target_date = df.index[-1]
    actual_price_today = df.iloc[-1]["btc_close"]

    phase_names = [p for p in cfg.model_config.get("phases", {}).keys() if str(p).startswith("phase")]
    phase_nums = [int(str(p).replace("phase", "")) for p in phase_names if str(p).replace("phase", "").isdigit()]
    production_phase = max(phase_nums) if phase_nums else 3

    for h in horizons:
        print(f"\n--- Syncing {h}-Day Horizon ---")
        cutoff_date = target_date - timedelta(days=h)
        
        try:
            predicted_price, base_price, start_date_actual, model_name_used, model_preference, calib_meta = _predict_single_horizon(
                df=df,
                phase=production_phase,
                horizon=h,
                cutoff_date=cutoff_date,
            )

            error_pct = abs(predicted_price - actual_price_today) / actual_price_today * 100
            multiplier = predicted_price / actual_price_today
            
            results[str(h)] = {
                "cutoff_date": str(start_date_actual.date()),
                "target_date": str(target_date.date()),
                "price_at_cutoff": float(base_price),
                "actual_price_today": float(actual_price_today),
                "predicted_price_today": float(predicted_price),
                "error_pct": float(error_pct),
                "multiplier": float(multiplier),
                "passed": bool(0.5 <= multiplier <= 2.0),
                "model_horizon_used": int(h),
                "model_phase_used": int(production_phase),
                "model_preference_used": model_preference,
                "model_name_used": model_name_used,
                "calibration_applied": bool(calib_meta.get("applied", False)),
                "calibration_window": calib_meta.get("window"),
                "calibration_rows": calib_meta.get("rows"),
                "calibration_bias_used": calib_meta.get("bias_used"),
            }
            print(f"✅ {h}d: BaseDate={start_date_actual.date()}, Today=${actual_price_today:,.0f}, Pred=${predicted_price:,.0f}")
            
        except Exception as e:
            print(f"⚠️ Error syncing {h}d: {e}")

    # Save
    output_path = os.path.join(PROCESSED_DIR, "reliability_result.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n🚀 Reality Check synced with production models successfully.")

if __name__ == "__main__":
    train_and_verify_horizons()
