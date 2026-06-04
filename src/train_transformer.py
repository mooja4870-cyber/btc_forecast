import os
import sys
import json
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, MODELS_DIR
from src.transformer_model import TimeSformer, CryptoSequenceDataset
from src.feature_engineer import HORIZONS
from src.config import cfg

SEQ_LEN = 90              # longer lookback won the arch sweep (mid-term signal)
BATCH_SIZE = 32

# ── Model architecture (chosen via scripts/arch_sweep.py honest hold-out) ──
# Smaller capacity than the original 64/2-layer: with 164 features and ~3900
# trainable rows, the big model overfit instantly (best_epoch=1). 32-dim /
# 1-layer + seq90 actually trains (ep≈16) and gives the best out-of-sample skill.
ARCH_D_MODEL = 32
ARCH_NHEAD = 4
ARCH_NUM_LAYERS = 1
ARCH_DIM_FF = 128

# ── Training-stability hyperparameters ──
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4        # L2 regularization to curb instant overfitting
DROPOUT = 0.3             # raised for stronger regularization
MIN_EPOCHS = 5           # don't early-stop on first-epoch validation noise
LR_SCHED_FACTOR = 0.5    # halve LR when validation loss plateaus
LR_SCHED_PATIENCE = 2

# ── Confidence / degeneracy thresholds for hold-out metrics ──
# A long-horizon validation window can be tiny and one-sided (e.g. 365d had
# 155 samples, 1.3% positive), making "direction accuracy" trivially gamed by
# always predicting the majority class. We expose skill-vs-baseline and flag
# these cases instead of presenting them as genuine accuracy.
MIN_RELIABLE_VAL_SAMPLES = 200
IMBALANCE_LOW = 0.10     # positive-ratio below this → degenerate (one-sided)
IMBALANCE_HIGH = 0.90    # positive-ratio above this → degenerate (one-sided)


def _normalize_horizons(horizons):
    if horizons is None:
        horizons = HORIZONS
    out = []
    for h in horizons:
        try:
            v = int(h)
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return sorted(set(out))


def _production_val_start():
    """Validation window start of the production (highest-numbered) phase.

    Used to carve out an honest out-of-sample hold-out: the evaluation model is
    trained strictly on data before this date, so dashboard validation numbers
    reflect true generalization rather than in-sample fit.
    """
    phases = cfg.model_config.get("phases", {})
    best = None
    for name, c in phases.items():
        if not (isinstance(name, str) and name.startswith("phase") and name[5:].isdigit()):
            continue
        val_range = c.get("val")
        if not val_range or not val_range[0]:
            continue
        n = int(name[5:])
        if best is None or n > best[0]:
            best = (n, val_range[0])
    if best is None:
        return None
    try:
        return pd.to_datetime(best[1])
    except Exception:
        return None


def _resolve_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def _train_model(X_scaled, y, num_features, epochs, device, shuffle=True,
                 val_fraction=0.0, patience=3):
    """Train a TimeSformer on pre-scaled sequences.

    When ``val_fraction`` > 0, the latest windows are held out as an inner
    validation split (temporal — never shuffled across the boundary) and the
    epoch with the lowest validation loss is restored (early stopping).

    Returns (model, final_train_loss, best_epoch).
    """
    dataset = CryptoSequenceDataset(X_scaled, y, seq_len=SEQ_LEN)
    n = len(dataset)

    val_loader = None
    if val_fraction > 0 and n > 10:
        n_val = max(1, int(n * val_fraction))
        n_tr = n - n_val
        if n_tr > 0:
            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, list(range(n_tr))),
                batch_size=BATCH_SIZE, shuffle=shuffle,
            )
            val_loader = DataLoader(
                torch.utils.data.Subset(dataset, list(range(n_tr, n))),
                batch_size=BATCH_SIZE, shuffle=False,
            )
        else:
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    else:
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    model = TimeSformer(
        num_features=num_features,
        d_model=ARCH_D_MODEL, nhead=ARCH_NHEAD, num_layers=ARCH_NUM_LAYERS,
        dim_feedforward=ARCH_DIM_FF, dropout=DROPOUT,
    ).to(device)
    criterion = nn.SmoothL1Loss()  # More stable than MSE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_SCHED_FACTOR, patience=LR_SCHED_PATIENCE,
        )

    best_val = float("inf")
    best_epoch = epochs
    best_state = None
    bad_epochs = 0
    avg_loss = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y)
            if torch.isnan(loss):
                print(f"    ⚠️ NaN loss at epoch {epoch+1}. Skipping batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        if val_loader is not None:
            model.eval()
            v_total, v_batches = 0.0, 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    v_loss = criterion(model(batch_X).squeeze(), batch_y)
                    if not torch.isnan(v_loss):
                        v_total += v_loss.item()
                        v_batches += 1
            v_avg = v_total / v_batches if v_batches > 0 else float("inf")
            if scheduler is not None:
                scheduler.step(v_avg)
            if v_avg < best_val - 1e-6:
                best_val = v_avg
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                # Guard against stopping on first-epoch validation noise.
                if bad_epochs >= patience and (epoch + 1) >= MIN_EPOCHS:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, avg_loss, best_epoch


def _predict_sequences(model, all_feat_values, all_index, eval_dates, mean, std, device):
    """Predict log-returns for each eval date using a seq_len window of history.

    History windows may extend back into the training period — that is past
    information available at prediction time, so it introduces no leakage.
    """
    preds = []
    for d in eval_dates:
        loc = all_index.get_loc(d)
        start = max(0, loc - SEQ_LEN + 1)
        x_seq = all_feat_values[start : loc + 1]
        if len(x_seq) < SEQ_LEN:
            pad_len = SEQ_LEN - len(x_seq)
            x_seq = np.pad(x_seq, ((pad_len, 0), (0, 0)), mode="constant")
        x_scaled = (x_seq - mean) / std
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds.append(float(model(x_tensor).item()))
    return np.asarray(preds, dtype=float)


def _compute_metrics(y_true, y_pred, base_prices=None, actual_future_prices=None):
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    direction_accuracy = float(np.mean((y_true > 0) == (y_pred > 0)))

    # Honest direction skill: a one-sided window inflates raw accuracy because
    # always predicting the majority class scores ~majority_ratio. Skill is the
    # accuracy *above* that trivial baseline (negative = no skill / worse).
    pos_ratio = float(np.mean(y_true > 0))
    majority_baseline_acc = float(max(pos_ratio, 1.0 - pos_ratio))
    direction_skill = float(direction_accuracy - majority_baseline_acc)

    price_mape_pct = None
    if base_prices is not None and actual_future_prices is not None:
        base = np.asarray(base_prices, dtype=float)
        actual = np.asarray(actual_future_prices, dtype=float)
        pred_future = base * np.exp(y_pred)
        mask = np.isfinite(actual) & (actual != 0)
        if mask.any():
            price_mape_pct = float(
                np.mean(np.abs((pred_future[mask] - actual[mask]) / actual[mask])) * 100.0
            )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "majority_baseline_acc": majority_baseline_acc,
        "direction_skill": direction_skill,
        "val_positive_ratio": pos_ratio,
        "price_mape_pct": price_mape_pct,
    }


def _evaluate_holdout(df, feature_cols, target_col, horizon, val_start, device, epochs):
    """Train an honest hold-out model (train < val_start, embargoed) and return
    out-of-sample validation metrics on the val window, or None if infeasible."""
    full_valid = df.dropna(subset=[target_col])

    # Embargo: a training row at date t reveals price at t+horizon. To keep the
    # validation window strictly unseen, drop training rows whose target window
    # spills into (or past) val_start.
    embargo = pd.Timedelta(days=int(horizon))
    train_eval = full_valid[full_valid.index < (val_start - embargo)]
    val_eval = full_valid[full_valid.index >= val_start]

    if len(train_eval) <= SEQ_LEN or len(val_eval) == 0:
        return None

    # Scaler fit on TRAIN ONLY — no peeking at the validation distribution.
    X_train_raw = train_eval[feature_cols].values
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0

    X_train_scaled = np.nan_to_num((X_train_raw - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(train_eval[target_col].values, nan=0.0, posinf=0.0, neginf=0.0)

    # Inner temporal validation split drives early stopping; best_epoch is then
    # reused to train the production model (which has no hold-out of its own).
    eval_model, _, best_epoch = _train_model(
        X_train_scaled, y_train, len(feature_cols), epochs, device,
        val_fraction=0.15, patience=3,
    )
    eval_model.eval()

    all_feat_values = df[feature_cols].values
    all_index = df.index
    eval_dates = val_eval.index

    y_pred = _predict_sequences(eval_model, all_feat_values, all_index, eval_dates, mean, std, device)
    y_true = val_eval[target_col].values.astype(float)

    base_prices = None
    actual_future_prices = None
    if "btc_close" in df.columns:
        base_prices = df["btc_close"].reindex(eval_dates).values.astype(float)
        actual_future_prices = df["btc_close"].shift(-horizon).reindex(eval_dates).values.astype(float)

    metrics = _compute_metrics(y_true, y_pred, base_prices, actual_future_prices)

    # Degeneracy: too few validation samples or a one-sided window means the
    # metrics can't be trusted (the 365d case). Low confidence additionally
    # covers "model shows no real skill" (negative R² and no direction edge).
    n_val = int(len(val_eval))
    pos_ratio = metrics.get("val_positive_ratio", 0.5)
    degenerate = (
        n_val < MIN_RELIABLE_VAL_SAMPLES
        or pos_ratio < IMBALANCE_LOW
        or pos_ratio > IMBALANCE_HIGH
    )
    low_confidence = bool(
        degenerate
        or (metrics.get("r2", 0.0) < 0.0 and metrics.get("direction_skill", 0.0) <= 0.0)
    )

    metrics.update({
        "horizon": int(horizon),
        "n_val_samples": n_val,
        "val_start": str(pd.Timestamp(val_start).date()),
        "val_end": str(val_eval.index[-1].date()),
        "eval_train_end": str(train_eval.index[-1].date()),
        "best_epoch": int(best_epoch),
        "degenerate": bool(degenerate),
        "low_confidence": low_confidence,
        "out_of_sample": True,
    })
    return metrics


def train_all_horizons(horizons=None, epochs: int = 15, eval_holdout: bool = True):
    horizons = _normalize_horizons(horizons)
    if not horizons:
        print("❌ No valid positive horizons to train.")
        return

    print("=" * 60)
    print("🚀 TRAINING TRANSFORMER MODELS FOR ALL HORIZONS")
    print(f"Horizons: {horizons}")
    print("=" * 60)

    # 1. Load Data
    path = os.path.join(PROCESSED_DIR, "featured_dataset.csv")
    if not os.path.exists(path):
        print(f"❌ Data file not found at {path}")
        return

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Exclude leakage
    exclude_cols = ["btc_close"] + [c for c in df.columns if "target" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Missing-value handling: forward-fill (carry last known value — preserves
    # time-series semantics) and only then zero-fill residual leading gaps.
    # A blanket fillna(0) would teach the model that "missing" == 0 (e.g. zero
    # inflation / zero unemployment), corrupting learned relationships.
    df[feature_cols] = df[feature_cols].ffill()
    df = df.fillna(0)

    val_start = _production_val_start() if eval_holdout else None
    if eval_holdout and val_start is None:
        print("  ⚠️ No production validation window configured — skipping hold-out evaluation.")

    device = _resolve_device()

    # Create base directory
    base_dir = os.path.join(MODELS_DIR, "transformer")
    os.makedirs(base_dir, exist_ok=True)

    # 2. Train for each horizon
    for horizon in horizons:
        print(f"\nTraining for Horizon: {horizon} days...")

        # Prepare Target (re-calculate to be sure)
        target_col = f"target_log_return_{horizon}d"
        future_close = df["btc_close"].shift(-horizon)
        df[target_col] = np.log(future_close / df["btc_close"])

        h_dir = os.path.join(base_dir, f"horizon_{horizon}d")
        os.makedirs(h_dir, exist_ok=True)

        # ── 2a. Honest hold-out evaluation (trained on train-only) ──
        val_metrics = None
        if val_start is not None:
            try:
                val_metrics = _evaluate_holdout(
                    df, feature_cols, target_col, horizon, val_start, device, epochs
                )
            except Exception as e:
                print(f"  ⚠️ Hold-out evaluation failed for {horizon}d: {e}")
                val_metrics = None

        val_metrics_path = os.path.join(h_dir, "val_metrics.json")
        if val_metrics is not None:
            with open(val_metrics_path, "w") as f:
                json.dump(val_metrics, f, indent=2)
            print(
                f"  📊 Hold-out val ({val_metrics['val_start']}~{val_metrics['val_end']}, "
                f"n={val_metrics['n_val_samples']}): "
                f"R²={val_metrics['r2']:.3f}, dir={val_metrics['direction_accuracy']:.1%}"
            )
        elif os.path.exists(val_metrics_path):
            # Stale honest metrics would mislead the dashboard — remove them.
            try:
                os.remove(val_metrics_path)
            except Exception:
                pass

        # ── 2b. Production model (trained on ALL valid data for live forecasting) ──
        train_valid = df.dropna(subset=[target_col])
        if len(train_valid) == 0:
            print(f"❌ Not enough data for {horizon}d horizon.")
            continue

        X_raw = train_valid[feature_cols].values
        mean = np.mean(X_raw, axis=0)
        std = np.std(X_raw, axis=0)
        std[std == 0] = 1.0

        X_scaled = np.nan_to_num((X_raw - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(train_valid[target_col].values, nan=0.0, posinf=0.0, neginf=0.0)

        # Reuse the early-stopping best epoch found on the honest hold-out so the
        # production model (which has no validation split) doesn't over-train.
        prod_epochs = int(epochs)
        if val_metrics is not None and val_metrics.get("best_epoch"):
            prod_epochs = int(val_metrics["best_epoch"])

        model, avg_loss, _ = _train_model(X_scaled, y, len(feature_cols), prod_epochs, device)
        print(f"✅ {horizon}d Production Model Trained ({prod_epochs} ep). Loss: {avg_loss:.6f}")

        # Save Artifacts
        torch.save(model.state_dict(), os.path.join(h_dir, "model.pth"))

        scaler_stats = {"mean": mean, "std": std}
        joblib.dump(scaler_stats, os.path.join(h_dir, "scaler_stats.joblib"))

        metadata = {
            "feature_cols": feature_cols,
            "seq_len": SEQ_LEN,
            "horizon": horizon,
            "model_type": "TimeSformer",
            "last_train_date": str(train_valid.index[-1].date()),
            "arch": {
                "d_model": ARCH_D_MODEL,
                "nhead": ARCH_NHEAD,
                "num_layers": ARCH_NUM_LAYERS,
                "dim_feedforward": ARCH_DIM_FF,
                "dropout": DROPOUT,
            },
        }
        with open(os.path.join(h_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print("\n🎉 All Transformer models trained and saved.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train TimeSformer models for selected horizons.")
    parser.add_argument(
        "--horizons",
        type=str,
        default="",
        help="Comma-separated horizons, e.g. 1,2,3,5. Empty = use config horizons.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per horizon.",
    )
    parser.add_argument(
        "--no-eval-holdout",
        action="store_true",
        help="Skip honest hold-out evaluation (faster; dashboard val metrics will be unavailable).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    target_horizons = None
    if args.horizons.strip():
        target_horizons = [x.strip() for x in args.horizons.split(",") if x.strip()]
    train_all_horizons(
        horizons=target_horizons,
        epochs=args.epochs,
        eval_holdout=not args.no_eval_holdout,
    )
