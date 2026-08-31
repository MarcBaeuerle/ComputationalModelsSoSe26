#!/usr/bin/env python3
"""
LSTM for takeover quality prediction on BATON-Sample.

Feature window : [t_event - 5s, t_event)
Quality window : [t_event,      t_event + 3s]  (computed later)

Features
--------
A. Driver   - head_pose_state (0/1/2) from face_yaw/pitch/roll
B. Vehicle  - vEgo (binned to nearest 10), lateral_accel = vEgo^2 * tan(steering_rad)
C. Radar    - ttc_class (0=safe>5s, 1=warning 2-5s, 2=critical<2s)
D. Planning - laneChangeState (label-encoded)
E. IMU      - gyro_x, gyro_y, gyro_z

Quality score (post-event)
--------------------------
0.5 * ttc_score  +  0.5 * steering_reversal_score

Usage
-----
python scripts/lstm_takeover_quality.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
BATON_ROOT = Path("/Users/I776423/Downloads/CTM/BATON-Sample")
OUT_DIR    = Path("/Users/I776423/Downloads/CTM/ComputationalModelsSoSe26-main/reports/lstm_takeover_quality")

PRE_SEC  = 5.0
POST_SEC = 3.0
SEQ_LEN  = 100   # fixed LSTM input length (resample pre-window)

FEATURE_COLS = [
    "head_pose_state",   # A – categorical 0/1/2
    "vEgo_bin10",        # B – speed binned to nearest 10
    "lateral_accel",     # B – vEgo^2 * tan(steer_rad)
    "ttc_class",         # C – 0/1/2
    "laneChangeState",   # D – 0-3
    "gyro_x",            # E
    "gyro_y",            # E
    "gyro_z",            # E
]
N_FEAT = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def ns(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load(path: Path, time_col: str = "time_s") -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if time_col not in df.columns:
        return None
    df[time_col] = ns(df[time_col])
    return df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


def merge_route(route_path: Path) -> pd.DataFrame:
    base = load(route_path / "vehicle_dynamics.csv")
    if base is None:
        raise FileNotFoundError(route_path)
    others = {
        "driver_state": (load(route_path / "driver_state.csv"),   0.030),
        "planning":     (load(route_path / "planning.csv"),        0.030),
        "radar":        (load(route_path / "radar.csv"),           0.030),
        "imu":          (load(route_path / "imu.csv"),             0.015),
    }
    merged = base.copy()
    for _, (df, tol) in others.items():
        if df is None:
            continue
        overlap = sorted(set(merged.columns) & set(df.columns) - {"time_s"})
        if overlap:
            df = df.drop(columns=overlap)
        merged = pd.merge_asof(
            merged.sort_values("time_s"),
            df.sort_values("time_s"),
            on="time_s", direction="nearest", tolerance=tol,
        )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def head_pose_state(df: pd.DataFrame) -> pd.Series:
    """
    0 = Attentive   yaw<=15 & pitch<=10 & roll<=10
    1 = Mild        yaw 15-30 OR pitch 10-20 OR roll 10-20
    2 = Distracted  yaw>30 OR pitch>20 OR roll>20
    """
    yaw   = ns(df.get("face_yaw",   pd.Series(0, index=df.index))).abs().fillna(0)
    pitch = ns(df.get("face_pitch", pd.Series(0, index=df.index))).abs().fillna(0)
    roll  = ns(df.get("face_roll",  pd.Series(0, index=df.index))).abs().fillna(0)
    ys = np.where(yaw   > 30, 2, np.where(yaw   > 15, 1, 0))
    ps = np.where(pitch > 20, 2, np.where(pitch > 10, 1, 0))
    rs = np.where(roll  > 20, 2, np.where(roll  > 10, 1, 0))
    return pd.Series(np.maximum(np.maximum(ys, ps), rs), index=df.index, dtype=float)


def vego_bin10(df: pd.DataFrame) -> pd.Series:
    v = ns(df.get("vEgo", pd.Series(np.nan, index=df.index)))
    return (np.round(v / 10) * 10).fillna(0)


def lateral_accel(df: pd.DataFrame) -> pd.Series:
    v  = ns(df.get("vEgo",             pd.Series(0, index=df.index))).fillna(0)
    a  = ns(df.get("steeringAngleDeg", pd.Series(0, index=df.index))).fillna(0)
    return (v ** 2) * np.tan(np.deg2rad(a))


def ttc_class(df: pd.DataFrame) -> pd.Series:
    dRel = ns(df.get("leadOne_dRel", pd.Series(np.nan, index=df.index)))
    vRel = ns(df.get("leadOne_vRel", pd.Series(np.nan, index=df.index)))
    ttc  = (dRel / vRel.abs().replace(0, np.nan)).clip(0, 60)
    return pd.Series(
        np.where(ttc.isna(), 0, np.where(ttc < 2, 2, np.where(ttc < 5, 1, 0))),
        index=df.index, dtype=float,
    )


LANE_MAP = {"LANE_CHANGE_NONE": 0, "LANE_CHANGE_PREPING": 1,
            "LANE_CHANGE_STARTING": 2, "LANE_CHANGE_LANE_CHANGE": 3}

def lane_change_enc(df: pd.DataFrame) -> pd.Series:
    if "laneChangeState" not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df["laneChangeState"].map(lambda v: float(LANE_MAP.get(str(v), 0)))


def build_tensor(window: pd.DataFrame) -> np.ndarray:
    """Return float32 (SEQ_LEN, N_FEAT)."""
    w = window.reset_index(drop=True)
    feats = pd.DataFrame({
        "head_pose_state": head_pose_state(w),
        "vEgo_bin10":      vego_bin10(w),
        "lateral_accel":   lateral_accel(w),
        "ttc_class":       ttc_class(w),
        "laneChangeState": lane_change_enc(w),
        "gyro_x":          ns(w.get("gyro_x", pd.Series(0, index=w.index))).fillna(0),
        "gyro_y":          ns(w.get("gyro_y", pd.Series(0, index=w.index))).fillna(0),
        "gyro_z":          ns(w.get("gyro_z", pd.Series(0, index=w.index))).fillna(0),
    })
    feats = feats[FEATURE_COLS].ffill().fillna(0.0).values.astype(float)
    idx   = np.linspace(0, len(feats) - 1, SEQ_LEN)
    out   = np.stack([np.interp(idx, np.arange(len(feats)), feats[:, i])
                      for i in range(N_FEAT)], axis=1)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Quality score
# ─────────────────────────────────────────────────────────────────────────────

def reversal_count(series: pd.Series, threshold: float = 0.5) -> int:
    v = series.dropna().values
    if len(v) < 2:
        return 0
    rate = np.diff(v)
    rate = rate[np.abs(rate) >= threshold]
    if len(rate) < 2:
        return 0
    return int(np.sum(np.sign(rate[1:]) * np.sign(rate[:-1]) < 0))


def compute_quality(post: pd.DataFrame) -> float:
    dRel  = ns(post.get("leadOne_dRel", pd.Series(dtype=float)))
    vRel  = ns(post.get("leadOne_vRel", pd.Series(dtype=float)))
    ttc   = (dRel / vRel.abs().replace(0, np.nan)).clip(0, 30)
    ttc_s = float(np.clip(ttc.mean() / 30.0, 0, 1)) if ttc.notna().any() else 0.5

    steer = ns(post.get("steeringAngleDeg", pd.Series(dtype=float)))
    n_rev = reversal_count(steer)
    rev_s = float(np.clip(1.0 - n_rev / 20.0, 0, 1))

    return round(0.5 * ttc_s + 0.5 * rev_s, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    events = pd.read_csv(BATON_ROOT / "benchmark/takeover_events_or.csv")
    events = events[events["event_type"].str.lower() == "takeover"].copy()
    events["t_event"] = pd.to_numeric(events["event_time_sec"], errors="coerce")
    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    cache: Dict[str, pd.DataFrame] = {}
    X_list, y_list, ids = [], [], []
    skipped = 0

    for _, ev in events.iterrows():
        rid  = str(ev["route_id"])
        rp   = BATON_ROOT / rid
        if not (rp / "vehicle_dynamics.csv").exists():
            skipped += 1
            continue
        if rid not in cache:
            try:
                cache[rid] = merge_route(rp)
            except Exception:
                skipped += 1
                continue

        merged = cache[rid]
        t      = ns(merged["time_s"])
        tev    = float(ev["t_event"])

        pre  = merged.loc[(t >= tev - PRE_SEC)  & (t <  tev)]
        post = merged.loc[(t >= tev)            & (t <= tev + POST_SEC)]

        if pre.empty or post.empty:
            skipped += 1
            continue

        try:
            X_list.append(build_tensor(pre))
            y_list.append(compute_quality(post))
            ids.append(str(ev["event_id"]))
        except Exception as exc:
            warnings.warn(f"Skip {ev['event_id']}: {exc}")
            skipped += 1

    print(f"Events: {len(X_list)} built, {skipped} skipped")
    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, dtype=np.float32),
            ids)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

def build_model(hidden: int = 64, dropout: float = 0.3):
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEAT, hidden, num_layers=2,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1), nn.Sigmoid(),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)
    return M()


def normalise(X: np.ndarray):
    flat = X.reshape(-1, N_FEAT)
    lo, hi = flat.min(0), flat.max(0)
    denom  = np.where(hi - lo > 0, hi - lo, 1.0)
    return ((X - lo) / denom).astype(np.float32), lo, hi


def train_loo(X: np.ndarray, y: np.ndarray, ids: List[str],
              epochs: int = 60, lr: float = 1e-3, batch: int = 8):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import LeaveOneOut

    X_n, _, _ = normalise(X)
    N = len(X_n)
    preds = np.zeros(N, dtype=np.float32)

    for fold, (tr, te) in enumerate(LeaveOneOut().split(X_n)):
        Xtr = torch.tensor(X_n[tr])
        ytr = torch.tensor(y[tr])
        Xte = torch.tensor(X_n[te])

        loader = DataLoader(TensorDataset(Xtr, ytr),
                            batch_size=batch, shuffle=True)
        model  = build_model()
        opt    = torch.optim.Adam(model.parameters(), lr=lr)
        loss_f = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_f(model(xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            preds[te[0]] = model(Xte).item()

        if fold % 5 == 0 or fold == N - 1:
            print(f"  Fold {fold+1:2d}/{N}  "
                  f"true={y[te[0]]:.3f}  pred={preds[te[0]]:.3f}")

    mae  = float(np.mean(np.abs(preds - y)))
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    print(f"\nLOO-CV  MAE={mae:.4f}  RMSE={rmse:.4f}")
    return preds, mae, rmse


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(y, preds, ids, mae, rmse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # scatter: true vs pred
    ax = axes[0]
    ax.scatter(y, preds, s=55, alpha=0.8, color="#2E86AB",
               edgecolors="white", linewidth=0.5)
    lo = min(y.min(), preds.min()) - 0.05
    hi = max(y.max(), preds.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True Quality Score")
    ax.set_ylabel("Predicted Quality Score")
    ax.set_title(f"True vs Predicted (LOO-CV)\nMAE={mae:.4f}  RMSE={rmse:.4f}")

    # error distribution
    ax2 = axes[1]
    errors = preds - y
    sns.histplot(errors, kde=len(errors) > 5, bins=12, ax=ax2, color="#4F6D7A")
    ax2.axvline(0, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Prediction Error (pred − true)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")

    fig.suptitle("LSTM Takeover Quality Prediction — BATON-Sample", fontsize=12)
    fig.tight_layout()
    p = OUT_DIR / "lstm_results.png"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {p}")

    # quality score distribution
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(y, bins=12, kde=True, color="#7A9E7E", ax=ax3)
    ax3.set_xlabel("Quality Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Takeover Quality Scores")
    fig2.tight_layout()
    p2 = OUT_DIR / "quality_score_distribution.png"
    fig2.savefig(p2, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Plot saved: {p2}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building dataset ===")
    X, y, ids = build_dataset()
    print(f"X shape : {X.shape}   y shape : {y.shape}")
    print(f"Quality  min={y.min():.3f}  max={y.max():.3f}  mean={y.mean():.3f}\n")

    # save dataset
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    pd.Series(ids, name="event_id").to_csv(OUT_DIR / "event_ids.csv", index=False)

    # save feature summary
    flat = X.reshape(-1, N_FEAT)
    summary = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean":    flat.mean(0),
        "std":     flat.std(0),
        "min":     flat.min(0),
        "max":     flat.max(0),
    })
    summary.to_csv(OUT_DIR / "feature_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("\n=== Training LSTM (LOO-CV) ===")
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
    except ImportError:
        print("PyTorch not installed — run: pip install torch\nDataset saved, skipping training.")
        return

    preds, mae, rmse = train_loo(X, y, ids)

    results = pd.DataFrame({
        "event_id":     ids,
        "true_quality": y,
        "pred_quality": preds,
        "error":        preds - y,
    })
    results.to_csv(OUT_DIR / "loo_predictions.csv", index=False)
    print(f"\nPredictions saved: {OUT_DIR / 'loo_predictions.csv'}")

    save_plots(y, preds, ids, mae, rmse)


if __name__ == "__main__":
    main()
