import os
import sys
import json
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


def train_all_horizons(horizons=None, epochs: int = 15):
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
    
    # Exclude leakage
    exclude_cols = ["btc_close"] + [c for c in df.columns if "target" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Fill NA
    df = df.fillna(0)
    
    # Create base directory
    base_dir = os.path.join(MODELS_DIR, "transformer")
    os.makedirs(base_dir, exist_ok=True)
    
    # 2. Train for each horizon
    for horizon in horizons:
        print(f"\nTraining for Horizon: {horizon} days...")
        
        # Prepare Target
        # Re-calculate target to be sure
        target_col = f"target_log_return_{horizon}d"
        future_close = df["btc_close"].shift(-horizon)
        df[target_col] = np.log(future_close / df["btc_close"])
        
        # Train Split (We use all available data up to the point where target exists)
        # For the final model to be used in the app, we train on ALL valid data.
        # Valid data = rows where target is not NaN.
        
        train_valid = df.dropna(subset=[target_col])
        
        if len(train_valid) == 0:
            print(f"❌ Not enough data for {horizon}d horizon.")
            continue
            
        # Scale Features
        X_raw = train_valid[feature_cols].values
        
        mean = np.mean(X_raw, axis=0)
        std = np.std(X_raw, axis=0)
        std[std == 0] = 1.0
        
        X_scaled = (X_raw - mean) / std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0) # Sanitize
        y = train_valid[target_col].values
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Dataset
        SEQ_LEN = 60
        BATCH_SIZE = 32
        EPOCHS = int(epochs)
        
        dataset = CryptoSequenceDataset(X_scaled, y, seq_len=SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            
        model = TimeSformer(num_features=len(feature_cols)).to(device)
        criterion = nn.SmoothL1Loss() # More stable than MSE
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch_X, batch_y in dataloader:
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
                
            avg_loss = total_loss/len(dataloader) if len(dataloader) > 0 else 0
            # print(f"  Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.6f}")

        print(f"✅ {horizon}d Model Trained. Loss: {avg_loss:.6f}")
        
        # Save Artifacts
        h_dir = os.path.join(base_dir, f"horizon_{horizon}d")
        os.makedirs(h_dir, exist_ok=True)
        
        # Save Model State
        torch.save(model.state_dict(), os.path.join(h_dir, "model.pth"))
        
        # Save Scaler Stats
        scaler_stats = {"mean": mean, "std": std}
        joblib.dump(scaler_stats, os.path.join(h_dir, "scaler_stats.joblib"))
        
        # Save Metadata
        metadata = {
            "feature_cols": feature_cols,
            "seq_len": SEQ_LEN,
            "horizon": horizon,
            "model_type": "TimeSformer",
            "last_train_date": str(train_valid.index[-1].date())
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    target_horizons = None
    if args.horizons.strip():
        target_horizons = [x.strip() for x in args.horizons.split(",") if x.strip()]
    train_all_horizons(horizons=target_horizons, epochs=args.epochs)
