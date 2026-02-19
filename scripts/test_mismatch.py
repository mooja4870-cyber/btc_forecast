
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root
sys.path.insert(0, os.getcwd())
from src.config import cfg
from src.predictor import predict_multi_horizon

def test_mismatch():
    base_date = "2025-02-13"
    print(f"--- Testing Base Date: {base_date} ---")

    phase_names = [p for p in cfg.model_config.get("phases", {}).keys() if str(p).startswith("phase")]
    phase_nums = sorted(
        [int(str(p).replace("phase", "")) for p in phase_names if str(p).replace("phase", "").isdigit()]
    )
    test_phases = phase_nums[-2:] if len(phase_nums) >= 2 else phase_nums

    for p in test_phases:
        try:
            pred_df, current_price, start_date = predict_multi_horizon(phase=p, from_date=base_date)
            row_365 = pred_df[pred_df["horizon_days"] == 365].iloc[0]
            print(f"Phase {p}: Pred=${row_365['predicted_price']:,.0f} (Target: {row_365['target_date'].date()})")
        except Exception as e:
            print(f"Phase {p}: Error: {e}")

if __name__ == "__main__":
    test_mismatch()
