#!/usr/bin/env python3
"""
Full ablation study for takeover quality prediction.

Ablations:
  1. Model comparison      — Mean, Linear, XGBoost, LSTM
  2. Leave-one-subset-out  — Remove one modality at a time
  3. Single-subset-only    — Each modality alone
  4. Quality score         — TTC-only, Reversal-only, Combined
  5. Pre-event window      — 3s, 5s, 7s

All evaluated with Leave-One-Out CV (N=31).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

BATON = Path("/Users/I776423/Downloads/CTM/BATON-Sample")
OUT   = Path("/Users/I776423/Downloads/CTM/ComputationalModelsSoSe26-main/reports/ablation")
OUT.mkdir(parents=True, exist_ok=True)

# ─── constants ────────────────────────────────────────────────────────────────
POST_SEC = 3.0
SEQ_LEN  = 100

ALL_FEATURE_COLS = [
    "head_pose_state",   # A driver
    "vEgo_bin10",        # B vehicle
    "lateral_accel",     # B vehicle
    "ttc_class",         # C radar
    "laneChangeState",   # D planning
    "gyro_x",            # E imu
    "gyro_y",            # E imu
    "gyro_z",            # E imu
]

SUBSETS = {
    "A_driver":   ["head_pose_state"],
    "B_vehicle":  ["vEgo_bin10", "lateral_accel"],
    "C_radar":    ["ttc_class"],
    "D_planning": ["laneChangeState"],
    "E_imu":      ["gyro_x", "gyro_y", "gyro_z"],
}

# ─── helpers ─────────────────────────────────────────────────────────────────

def ns(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load(path, time_col="time_s"):
    if not path.exists(): return None
    df = pd.read_csv(path, low_memory=False)
    if time_col not in df.columns: return None
    df[time_col] = ns(df[time_col])
    return df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


def merge_route(rp):
    base = load(rp / "vehicle_dynamics.csv")
    if base is None: raise FileNotFoundError(rp)
    tols = {"driver_state": 0.03, "planning": 0.03, "radar": 0.03, "imu": 0.015}
    merged = base.copy()
    for name, fname in [("driver_state","driver_state.csv"),("planning","planning.csv"),
                        ("radar","radar.csv"),("imu","imu.csv")]:
        df = load(rp / fname)
        if df is None: continue
        overlap = sorted(set(merged.columns) & set(df.columns) - {"time_s"})
        if overlap: df = df.drop(columns=overlap)
        merged = pd.merge_asof(merged.sort_values("time_s"), df.sort_values("time_s"),
                               on="time_s", direction="nearest", tolerance=tols[name])
    return merged

# ─── feature engineering ─────────────────────────────────────────────────────

def head_pose_state(df):
    yaw   = ns(df.get("face_yaw",   pd.Series(0,index=df.index))).abs().fillna(0)
    pitch = ns(df.get("face_pitch", pd.Series(0,index=df.index))).abs().fillna(0)
    roll  = ns(df.get("face_roll",  pd.Series(0,index=df.index))).abs().fillna(0)
    ys = np.where(yaw>30,2,np.where(yaw>15,1,0))
    ps = np.where(pitch>20,2,np.where(pitch>10,1,0))
    rs = np.where(roll>20,2,np.where(roll>10,1,0))
    return pd.Series(np.maximum(np.maximum(ys,ps),rs), index=df.index, dtype=float)

def vego_bin10(df):
    v = ns(df.get("vEgo", pd.Series(np.nan,index=df.index)))
    return (np.round(v/10)*10).fillna(0)

def lateral_accel(df):
    v = ns(df.get("vEgo",pd.Series(0,index=df.index))).fillna(0)
    a = ns(df.get("steeringAngleDeg",pd.Series(0,index=df.index))).fillna(0)
    return (v**2)*np.tan(np.deg2rad(a))

def ttc_class(df):
    d = ns(df.get("leadOne_dRel",pd.Series(np.nan,index=df.index)))
    v = ns(df.get("leadOne_vRel",pd.Series(np.nan,index=df.index)))
    ttc = (d/v.abs().replace(0,np.nan)).clip(0,60)
    return pd.Series(np.where(ttc.isna(),0,np.where(ttc<2,2,np.where(ttc<5,1,0))),
                     index=df.index, dtype=float)

LANE_MAP = {"LANE_CHANGE_NONE":0,"LANE_CHANGE_PREPING":1,
            "LANE_CHANGE_STARTING":2,"LANE_CHANGE_LANE_CHANGE":3}

def lane_enc(df):
    if "laneChangeState" not in df.columns: return pd.Series(0.0,index=df.index)
    return df["laneChangeState"].map(lambda v: float(LANE_MAP.get(str(v),0)))

def build_all_features(w):
    w = w.reset_index(drop=True)
    return pd.DataFrame({
        "head_pose_state": head_pose_state(w),
        "vEgo_bin10":      vego_bin10(w),
        "lateral_accel":   lateral_accel(w),
        "ttc_class":       ttc_class(w),
        "laneChangeState": lane_enc(w),
        "gyro_x": ns(w.get("gyro_x",pd.Series(0,index=w.index))).fillna(0),
        "gyro_y": ns(w.get("gyro_y",pd.Series(0,index=w.index))).fillna(0),
        "gyro_z": ns(w.get("gyro_z",pd.Series(0,index=w.index))).fillna(0),
    })

def to_tensor(feats_df, cols, seq_len=SEQ_LEN):
    arr = feats_df[cols].ffill().fillna(0.0).values.astype(float)
    idx = np.linspace(0, len(arr)-1, seq_len)
    out = np.stack([np.interp(idx, np.arange(len(arr)), arr[:,i])
                    for i in range(len(cols))], axis=1)
    return out.astype(np.float32)

# ─── quality scores ──────────────────────────────────────────────────────────

def reversal_count(series, thr=0.5):
    v = ns(series).dropna().values
    if len(v)<2: return 0
    rate = np.diff(v); rate = rate[np.abs(rate)>=thr]
    if len(rate)<2: return 0
    return int(np.sum(np.sign(rate[1:])*np.sign(rate[:-1])<0))

def compute_quality(post, mode="combined"):
    d = ns(post.get("leadOne_dRel", pd.Series(dtype=float)))
    v = ns(post.get("leadOne_vRel", pd.Series(dtype=float)))
    ttc = (d/v.abs().replace(0,np.nan)).clip(0,30)
    mean_ttc = float(ttc.mean()) if ttc.notna().any() else 15.0
    ttc_s = float(np.clip(mean_ttc/30.0, 0, 1))

    steer = ns(post.get("steeringAngleDeg", pd.Series(dtype=float)))
    n_rev = reversal_count(steer)
    rev_s = float(np.clip(1.0 - n_rev/20.0, 0, 1))

    if mode == "ttc_only":      return round(ttc_s, 4)
    if mode == "reversal_only": return round(rev_s, 4)
    return round(0.5*ttc_s + 0.5*rev_s, 4)

# ─── dataset builder (parameterised) ─────────────────────────────────────────

def build_dataset(pre_sec=5.0, quality_mode="combined"):
    events = pd.read_csv(BATON/"benchmark/takeover_events_or.csv")
    events = events[events["event_type"].str.lower()=="takeover"].copy()
    events["t_event"] = pd.to_numeric(events["event_time_sec"], errors="coerce")
    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    cache = {}
    rows, ids = [], []
    for _, ev in events.iterrows():
        rid = str(ev["route_id"])
        rp  = BATON/rid
        if not (rp/"vehicle_dynamics.csv").exists(): continue
        if rid not in cache:
            try: cache[rid] = merge_route(rp)
            except: continue
        merged = cache[rid]
        t   = ns(merged["time_s"])
        tev = float(ev["t_event"])
        pre  = merged.loc[(t >= tev-pre_sec) & (t <  tev)]
        post = merged.loc[(t >= tev)         & (t <= tev+POST_SEC)]
        if pre.empty or post.empty: continue
        try:
            feats = build_all_features(pre)
            q     = compute_quality(post, mode=quality_mode)
        except Exception as e:
            warnings.warn(str(e)); continue
        rows.append(feats)
        ids.append({"event_id": str(ev["event_id"]), "quality": q,
                    "feats": feats})
    return ids   # list of dicts


def make_XY(records, cols):
    X = np.stack([to_tensor(r["feats"], cols) for r in records]).astype(np.float32)
    y = np.array([r["quality"] for r in records], dtype=np.float32)
    lo = X.reshape(-1,len(cols)).min(0); hi = X.reshape(-1,len(cols)).max(0)
    denom = np.where(hi-lo>0, hi-lo, 1.0)
    return ((X-lo)/denom).astype(np.float32), y

# ─── models ──────────────────────────────────────────────────────────────────

def build_lstm(n_feat, hidden=64, dropout=0.3):
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, num_layers=2,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden,32), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(32,1), nn.Sigmoid())
        def forward(self, x):
            out,_ = self.lstm(x)
            return self.head(out[:,-1]).squeeze(-1)
    return M()


def loo_lstm(X, y, epochs=60, lr=1e-3, batch=8):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import LeaveOneOut
    preds = np.zeros(len(y), dtype=np.float32)
    for tr, te in LeaveOneOut().split(X):
        Xtr,ytr = torch.tensor(X[tr]),torch.tensor(y[tr])
        Xte     = torch.tensor(X[te])
        loader  = DataLoader(TensorDataset(Xtr,ytr),batch_size=batch,shuffle=True)
        model   = build_lstm(X.shape[-1])
        opt     = torch.optim.Adam(model.parameters(), lr=lr)
        loss_f  = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb,yb in loader:
                opt.zero_grad(); loss_f(model(xb),yb).backward(); opt.step()
        model.eval()
        import torch as _t
        with _t.no_grad(): preds[te[0]] = model(Xte).item()
    return preds


def loo_linear(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    preds = np.zeros(len(y), dtype=np.float32)
    Xf = X.reshape(len(X), -1)
    for tr, te in LeaveOneOut().split(Xf):
        m = Ridge(alpha=1.0).fit(Xf[tr], y[tr])
        preds[te[0]] = float(np.ravel(m.predict(Xf[te]))[0])
    return np.clip(preds, 0, 1).astype(np.float32)


def loo_xgb(X, y):
    from xgboost import XGBRegressor
    from sklearn.model_selection import LeaveOneOut
    preds = np.zeros(len(y), dtype=np.float32)
    Xf = X.reshape(len(X), -1)
    for tr, te in LeaveOneOut().split(Xf):
        m = XGBRegressor(n_estimators=100, max_depth=3,
                         learning_rate=0.1, verbosity=0,
                         random_state=42).fit(Xf[tr], y[tr])
        preds[te[0]] = float(np.ravel(m.predict(Xf[te]))[0])
    return np.clip(preds, 0, 1).astype(np.float32)


def metrics(y, preds):
    mae  = float(np.mean(np.abs(preds-y)))
    rmse = float(np.sqrt(np.mean((preds-y)**2)))
    return mae, rmse

# ─── run all ablations ────────────────────────────────────────────────────────

print("Building base dataset (pre=5s, quality=combined) …")
records = build_dataset(pre_sec=5.0, quality_mode="combined")
print(f"  {len(records)} events\n")

# ── 1. Model comparison ───────────────────────────────────────────────────────
print("=== Ablation 1: Model comparison ===")
X_all, y = make_XY(records, ALL_FEATURE_COLS)
mean_preds = np.full(len(y), y.mean(), dtype=np.float32)
mae_mean, rmse_mean = metrics(y, mean_preds)
print(f"  Mean baseline   MAE={mae_mean:.4f}  RMSE={rmse_mean:.4f}")

preds_lin = loo_linear(X_all, y)
mae_lin, rmse_lin = metrics(y, preds_lin)
print(f"  Linear (Ridge)  MAE={mae_lin:.4f}  RMSE={rmse_lin:.4f}")

try:
    preds_xgb = loo_xgb(X_all, y)
    mae_xgb, rmse_xgb = metrics(y, preds_xgb)
    print(f"  XGBoost         MAE={mae_xgb:.4f}  RMSE={rmse_xgb:.4f}")
except Exception as e:
    print(f"  XGBoost failed: {e}"); preds_xgb=mean_preds; mae_xgb=mae_mean; rmse_xgb=rmse_mean

preds_lstm = loo_lstm(X_all, y)
mae_lstm, rmse_lstm = metrics(y, preds_lstm)
print(f"  LSTM            MAE={mae_lstm:.4f}  RMSE={rmse_lstm:.4f}")

model_results = pd.DataFrame({
    "Model":["Mean baseline","Linear (Ridge)","XGBoost","LSTM"],
    "MAE":  [mae_mean, mae_lin, mae_xgb, mae_lstm],
    "RMSE": [rmse_mean,rmse_lin,rmse_xgb,rmse_lstm],
})

# ── 2. Leave-one-subset-out ───────────────────────────────────────────────────
print("\n=== Ablation 2: Leave-one-subset-out ===")
looso_results = []
for name, cols in SUBSETS.items():
    remaining = [c for c in ALL_FEATURE_COLS if c not in cols]
    X_sub, _ = make_XY(records, remaining)
    p = loo_lstm(X_sub, y)
    m, r = metrics(y, p)
    looso_results.append({"Removed":name,"Remaining cols":len(remaining),"MAE":m,"RMSE":r})
    print(f"  Remove {name:12s}  MAE={m:.4f}  RMSE={r:.4f}  (Δ MAE vs full: {m-mae_lstm:+.4f})")

looso_df = pd.DataFrame(looso_results)

# ── 3. Single-subset-only ─────────────────────────────────────────────────────
print("\n=== Ablation 3: Single-subset-only ===")
single_results = []
for name, cols in SUBSETS.items():
    X_sub, _ = make_XY(records, cols)
    p = loo_lstm(X_sub, y)
    m, r = metrics(y, p)
    single_results.append({"Subset":name,"N cols":len(cols),"MAE":m,"RMSE":r})
    print(f"  Only {name:12s}  MAE={m:.4f}  RMSE={r:.4f}")

single_df = pd.DataFrame(single_results)

# ── 4. Quality score components ───────────────────────────────────────────────
print("\n=== Ablation 4: Quality score components ===")
quality_results = []
for qmode in ["ttc_only","reversal_only","combined"]:
    recs = build_dataset(pre_sec=5.0, quality_mode=qmode)
    Xq, yq = make_XY(recs, ALL_FEATURE_COLS)
    p = loo_lstm(Xq, yq)
    m, r = metrics(yq, p)
    quality_results.append({"Quality mode":qmode,"MAE":m,"RMSE":r,"y_std":float(yq.std())})
    print(f"  {qmode:15s}  MAE={m:.4f}  RMSE={r:.4f}  y_std={yq.std():.4f}")

quality_df = pd.DataFrame(quality_results)

# ── 5. Pre-event window length ────────────────────────────────────────────────
print("\n=== Ablation 5: Pre-event window length ===")
window_results = []
for pre_sec in [3.0, 5.0, 7.0]:
    recs = build_dataset(pre_sec=pre_sec, quality_mode="combined")
    Xw, yw = make_XY(recs, ALL_FEATURE_COLS)
    p = loo_lstm(Xw, yw)
    m, r = metrics(yw, p)
    window_results.append({"Window (s)":pre_sec,"N events":len(recs),"MAE":m,"RMSE":r})
    print(f"  Window={pre_sec}s   N={len(recs)}  MAE={m:.4f}  RMSE={r:.4f}")

window_df = pd.DataFrame(window_results)

# ─── save tables ──────────────────────────────────────────────────────────────
model_results.to_csv(OUT/"ablation_model_comparison.csv", index=False)
looso_df.to_csv(OUT/"ablation_leave_one_subset_out.csv", index=False)
single_df.to_csv(OUT/"ablation_single_subset.csv", index=False)
quality_df.to_csv(OUT/"ablation_quality_mode.csv", index=False)
window_df.to_csv(OUT/"ablation_window_length.csv", index=False)

# ─── plots ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"axes.titleweight":"bold","figure.dpi":150})

BLUE="#2E86AB"; ORANGE="#E07A5F"; GREEN="#7A9E7E"; GREY="#4F6D7A"; PURPLE="#7B2D8B"

fig = plt.figure(figsize=(18, 22))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.38)

# ── Plot 1: Model comparison ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
colors_m = [GREY, BLUE, ORANGE, GREEN]
bars = ax1.bar(model_results["Model"], model_results["MAE"],
               color=colors_m, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, model_results["MAE"]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylabel("MAE (lower = better)")
ax1.set_title("Ablation 1: Model Comparison\n(LOO-CV, all features)")
ax1.set_ylim(0, model_results["MAE"].max()*1.25)

# ── Plot 2: Leave-one-subset-out ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
delta = looso_df["MAE"] - mae_lstm
bar_colors = [ORANGE if d > 0 else GREEN for d in delta]
bars2 = ax2.bar(looso_df["Removed"], delta, color=bar_colors, alpha=0.85, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.8)
for bar, val in zip(bars2, delta):
    ax2.text(bar.get_x()+bar.get_width()/2,
             bar.get_height() + (0.001 if val >= 0 else -0.003),
             f"{val:+.4f}", ha="center",
             va="bottom" if val >= 0 else "top", fontsize=8.5)
ax2.set_ylabel("ΔMAE vs full model\n(positive = removing this hurts)")
ax2.set_title("Ablation 2: Leave-One-Subset-Out\n(orange = subset was helpful)")
ax2.tick_params(axis="x", rotation=20)

# ── Plot 3: Single-subset-only ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
single_colors = [BLUE, ORANGE, GREEN, GREY, PURPLE]
bars3 = ax3.bar(single_df["Subset"], single_df["MAE"],
                color=single_colors, alpha=0.85, edgecolor="white")
ax3.axhline(mae_lstm, color="red", linewidth=1.2, linestyle="--",
            label=f"Full model ({mae_lstm:.4f})")
ax3.axhline(mae_mean, color="grey", linewidth=1.0, linestyle=":",
            label=f"Mean baseline ({mae_mean:.4f})")
for bar, val in zip(bars3, single_df["MAE"]):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax3.set_ylabel("MAE (lower = better)")
ax3.set_title("Ablation 3: Single Subset Only\n(how well does each modality predict alone?)")
ax3.legend(fontsize=8)
ax3.set_ylim(0, single_df["MAE"].max()*1.25)
ax3.tick_params(axis="x", rotation=20)

# ── Plot 4: Quality score components ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
qcols = [GREY, ORANGE, GREEN]
bars4 = ax4.bar(quality_df["Quality mode"], quality_df["MAE"],
                color=qcols, alpha=0.85, edgecolor="white")
for bar, val in zip(bars4, quality_df["MAE"]):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax4_twin = ax4.twinx()
ax4_twin.plot(quality_df["Quality mode"], quality_df["y_std"],
              "D--", color="red", linewidth=1.5, markersize=7, label="y std dev")
ax4_twin.set_ylabel("Std dev of quality scores", color="red", fontsize=9)
ax4_twin.tick_params(axis="y", colors="red")
ax4.set_ylabel("MAE (lower = better)")
ax4.set_title("Ablation 4: Quality Score Definition\n(red = spread of target values)")
ax4.set_ylim(0, quality_df["MAE"].max()*1.3)

# ── Plot 5: Window length ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
wcols = [BLUE, GREEN, ORANGE]
bars5 = ax5.bar(window_df["Window (s)"].astype(str)+"s",
                window_df["MAE"], color=wcols, alpha=0.85, edgecolor="white")
for bar, val in zip(bars5, window_df["MAE"]):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax5.set_ylabel("MAE (lower = better)")
ax5.set_title("Ablation 5: Pre-Event Window Length\n(how far back does context help?)")
ax5.set_ylim(0, window_df["MAE"].max()*1.25)
ax5.set_xlabel("Pre-event window duration")

# ── Plot 6: Summary table ──────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")
summary_data = [
    ["Ablation", "Best config", "MAE"],
    ["1. Model",          "LSTM",         f"{mae_lstm:.4f}"],
    ["2. Leave-one-out",  looso_df.loc[looso_df['MAE'].idxmin(),'Removed'],
                          f"{looso_df['MAE'].min():.4f}"],
    ["3. Single subset",  single_df.loc[single_df['MAE'].idxmin(),'Subset'],
                          f"{single_df['MAE'].min():.4f}"],
    ["4. Quality mode",   quality_df.loc[quality_df['MAE'].idxmin(),'Quality mode'],
                          f"{quality_df['MAE'].min():.4f}"],
    ["5. Window",         str(window_df.loc[window_df['MAE'].idxmin(),'Window (s)'])+"s",
                          f"{window_df['MAE'].min():.4f}"],
]
table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                  loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(11)
table.scale(1.2, 2.0)
for (r,c), cell in table.get_celld().items():
    if r == 0: cell.set_facecolor("#2E86AB"); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0: cell.set_facecolor("#f0f4f8")
ax6.set_title("Ablation Summary", fontweight="bold", fontsize=12, pad=15)

fig.suptitle("Ablation Study — Takeover Quality Prediction (BATON-Sample, LOO-CV)",
             fontsize=14, y=1.01)

out_path = OUT / "ablation_full.png"
fig.savefig(out_path, bbox_inches="tight", dpi=160)
plt.close(fig)
print(f"\nAll ablation plots saved: {out_path}")
print("\n=== Summary ===")
print(model_results.to_string(index=False))
print("\n", looso_df.to_string(index=False))
print("\n", single_df.to_string(index=False))
print("\n", quality_df.to_string(index=False))
print("\n", window_df.to_string(index=False))
