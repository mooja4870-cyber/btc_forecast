"""
Architecture sweep — honest hold-out comparison of TimeSformer capacities.
Run: python3 scripts/arch_sweep.py
Diagnostic only (not part of the production pipeline). Picks the config with
best out-of-sample direction skill / val MAE on the 2025+ window.
"""
import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR
from src.transformer_model import TimeSformer, CryptoSequenceDataset

VAL_START = pd.Timestamp("2025-01-01")
EPOCHS = 20
BATCH = 32
PATIENCE = 3
MIN_EPOCHS = 5

CONFIGS = [
    {"name": "baseline",   "d_model": 64, "layers": 2, "nhead": 4, "ff": 256, "seq": 60, "dropout": 0.2},
    {"name": "small",      "d_model": 32, "layers": 1, "nhead": 4, "ff": 128, "seq": 60, "dropout": 0.3},
    {"name": "tiny",       "d_model": 16, "layers": 1, "nhead": 2, "ff": 64,  "seq": 60, "dropout": 0.3},
    {"name": "small_seq90","d_model": 32, "layers": 1, "nhead": 4, "ff": 128, "seq": 90, "dropout": 0.3},
    {"name": "small_seq30","d_model": 32, "layers": 1, "nhead": 4, "ff": 128, "seq": 30, "dropout": 0.3},
]


def build(df, feature_cols, target_col, horizon):
    df = df.copy()
    df[target_col] = np.log(df["btc_close"].shift(-horizon) / df["btc_close"])
    df[feature_cols] = df[feature_cols].ffill()
    df = df.fillna(0)
    valid = df.dropna(subset=[target_col])
    embargo = pd.Timedelta(days=horizon)
    tr = valid[valid.index < (VAL_START - embargo)]
    va = valid[valid.index >= VAL_START]
    return df, tr, va


def train_eval(df, feature_cols, target_col, horizon, cfg, device):
    df, tr, va = build(df, feature_cols, target_col, horizon)
    seq = cfg["seq"]
    Xtr = tr[feature_cols].values
    mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1.0
    Xtr_s = np.nan_to_num((Xtr - mean) / std)
    ytr = np.nan_to_num(tr[target_col].values)

    ds = CryptoSequenceDataset(Xtr_s, ytr, seq_len=seq)
    n = len(ds); nval = max(1, int(n * 0.15)); ntr = n - nval
    tl = DataLoader(Subset(ds, range(ntr)), batch_size=BATCH, shuffle=True)
    vl = DataLoader(Subset(ds, range(ntr, n)), batch_size=BATCH, shuffle=False)

    model = TimeSformer(num_features=len(feature_cols), d_model=cfg["d_model"],
                        nhead=cfg["nhead"], num_layers=cfg["layers"],
                        dim_feedforward=cfg["ff"], dropout=cfg["dropout"]).to(device)
    crit = nn.SmoothL1Loss()
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best, best_state, bad, best_ep = float("inf"), None, 0, EPOCHS
    for ep in range(EPOCHS):
        model.train()
        for bx, by in tl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx).squeeze(), by)
            if torch.isnan(loss): continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vt, vb = 0.0, 0
        with torch.no_grad():
            for bx, by in vl:
                bx, by = bx.to(device), by.to(device)
                l = crit(model(bx).squeeze(), by)
                if not torch.isnan(l): vt += l.item(); vb += 1
        v = vt / vb if vb else float("inf"); sched.step(v)
        if v < best - 1e-6:
            best, best_ep, best_state, bad = v, ep + 1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE and ep + 1 >= MIN_EPOCHS: break
    if best_state: model.load_state_dict(best_state)

    # honest out-of-sample eval on va
    allf = df[feature_cols].values; idx = df.index
    preds = []
    model.eval()
    for d in va.index:
        loc = idx.get_loc(d); start = max(0, loc - seq + 1)
        x = allf[start:loc + 1]
        if len(x) < seq: x = np.pad(x, ((seq - len(x), 0), (0, 0)))
        xs = np.nan_to_num((x - mean) / std)
        with torch.no_grad():
            preds.append(float(model(torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)).item()))
    yt = va[target_col].values.astype(float); yp = np.asarray(preds)
    mae = float(np.mean(np.abs(yt - yp)))
    dir_acc = float(np.mean((yt > 0) == (yp > 0)))
    pos = float(np.mean(yt > 0)); base = max(pos, 1 - pos)
    skill = dir_acc - base
    return {"mae": mae, "dir_acc": dir_acc, "skill": skill, "best_ep": best_ep, "n": len(va)}


def main():
    torch.manual_seed(42); np.random.seed(42)
    df0 = pd.read_csv(os.path.join(PROCESSED_DIR, "featured_dataset.csv"), index_col=0, parse_dates=True).sort_index()
    exclude = ["btc_close"] + [c for c in df0.columns if "target" in c]
    feature_cols = [c for c in df0.columns if c not in exclude]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"features={len(feature_cols)} device={device}\n")
    for horizon in [15, 90]:
        tcol = f"target_log_return_{horizon}d"
        print(f"===== Horizon {horizon}d =====")
        print(f"{'config':<13}{'skill':>8}{'dir_acc':>9}{'mae':>9}{'best_ep':>9}")
        for cfg in CONFIGS:
            r = train_eval(df0, feature_cols, tcol, horizon, cfg, device)
            print(f"{cfg['name']:<13}{r['skill']*100:>7.1f}%{r['dir_acc']*100:>8.1f}%{r['mae']:>9.4f}{r['best_ep']:>9}")
        print()


if __name__ == "__main__":
    main()
