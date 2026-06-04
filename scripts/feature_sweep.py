"""
Feature-set sweep — honest hold-out comparison of feature subsets, with the
production architecture fixed. Diagnostic only.
Run: python3 scripts/feature_sweep.py

Tests whether pruning redundant / non-stationary features improves
out-of-sample direction skill (vs the full 164-feature set).
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
SEQ = 90
ARCH = dict(d_model=32, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.3)


def is_nonstationary(col):
    if col.endswith("_close") or col.startswith("log_") or "_lag" in col:
        return True
    if "_ma" in col and col[-1].isdigit():   # btc_close_ma30 (level) but keep btc_ma30_pct
        return True
    if col in ("bb_upper", "bb_lower", "macd", "macd_signal"):
        return True
    return False


def make_subsets(df, all_feat):
    X = df[all_feat].ffill().fillna(0)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # target corr to decide which of a redundant pair to keep
    tcorr = X.corrwith(np.log(df["btc_close"].shift(-90) / df["btc_close"]).fillna(0)).abs()
    drop95 = set()
    for c in upper.columns:
        partners = upper.index[upper[c] > 0.95].tolist()
        for pp in partners:
            # drop the one with weaker target correlation
            loser = c if tcorr.get(c, 0) < tcorr.get(pp, 0) else pp
            drop95.add(loser)
    prune95 = [c for c in all_feat if c not in drop95]
    stationary = [c for c in all_feat if not is_nonstationary(c)]
    stat_prune = [c for c in stationary if c not in drop95]
    return {
        "all(164)": all_feat,
        f"prune95({len(prune95)})": prune95,
        f"stationary({len(stationary)})": stationary,
        f"stat+prune({len(stat_prune)})": stat_prune,
    }


def train_eval(df, feature_cols, horizon, device):
    tcol = f"t{horizon}"
    df = df.copy()
    df[tcol] = np.log(df["btc_close"].shift(-horizon) / df["btc_close"])
    df[feature_cols] = df[feature_cols].ffill()
    df = df.fillna(0)
    valid = df.dropna(subset=[tcol])
    embargo = pd.Timedelta(days=horizon)
    tr = valid[valid.index < (VAL_START - embargo)]
    va = valid[valid.index >= VAL_START]

    Xtr = tr[feature_cols].values
    mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1.0
    Xtr_s = np.nan_to_num((Xtr - mean) / std)
    ytr = np.nan_to_num(tr[tcol].values)

    ds = CryptoSequenceDataset(Xtr_s, ytr, seq_len=SEQ)
    n = len(ds); nval = max(1, int(n * 0.15)); ntr = n - nval
    tl = DataLoader(Subset(ds, range(ntr)), batch_size=BATCH, shuffle=True)
    vl = DataLoader(Subset(ds, range(ntr, n)), batch_size=BATCH, shuffle=False)

    model = TimeSformer(num_features=len(feature_cols), **ARCH).to(device)
    crit = nn.SmoothL1Loss()
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train()
        for bx, by in tl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); l = crit(model(bx).squeeze(), by)
            if torch.isnan(l): continue
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vt, vb = 0.0, 0
        with torch.no_grad():
            for bx, by in vl:
                bx, by = bx.to(device), by.to(device)
                ll = crit(model(bx).squeeze(), by)
                if not torch.isnan(ll): vt += ll.item(); vb += 1
        v = vt / vb if vb else float("inf"); sched.step(v)
        if v < best - 1e-6: best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 3 and ep + 1 >= 5: break
    if best_state: model.load_state_dict(best_state)

    allf = df[feature_cols].values; idx = df.index; preds = []
    model.eval()
    for d in va.index:
        loc = idx.get_loc(d); start = max(0, loc - SEQ + 1)
        x = allf[start:loc + 1]
        if len(x) < SEQ: x = np.pad(x, ((SEQ - len(x), 0), (0, 0)))
        xs = np.nan_to_num((x - mean) / std)
        with torch.no_grad():
            preds.append(float(model(torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)).item()))
    yt = va[tcol].values.astype(float); yp = np.asarray(preds)
    dir_acc = float(np.mean((yt > 0) == (yp > 0)))
    pos = float(np.mean(yt > 0)); skill = dir_acc - max(pos, 1 - pos)
    mae = float(np.mean(np.abs(yt - yp)))
    return skill, dir_acc, mae


def main():
    torch.manual_seed(42); np.random.seed(42)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "featured_dataset.csv"), index_col=0, parse_dates=True).sort_index()
    exclude = ["btc_close"] + [c for c in df.columns if "target" in c]
    all_feat = [c for c in df.columns if c not in exclude]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    subsets = make_subsets(df, all_feat)
    print(f"device={device}  subsets={list(subsets.keys())}\n")
    for horizon in [15, 30, 90]:
        print(f"===== {horizon}d =====")
        print(f"{'subset':<18}{'skill':>8}{'dir':>8}{'mae':>9}")
        for name, cols in subsets.items():
            sk, da, mae = train_eval(df, cols, horizon, device)
            print(f"{name:<18}{sk*100:>7.1f}%{da*100:>7.1f}%{mae:>9.4f}")
        print()


if __name__ == "__main__":
    main()
