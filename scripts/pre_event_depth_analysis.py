#!/usr/bin/env python3
"""
Pre-event depth analysis: predictive signal across 1-second time bins.

Two analyses:
  1. Bin-only:        XGBoost LOO trained on each 1-second bin in isolation
                      [-5,-4]s, [-4,-3]s, [-3,-2]s, [-2,-1]s, [-1,0]s
  2. Progressive:     XGBoost LOO trained on growing retrospective windows
                      last 1s, last 2s, last 3s, last 4s, full 5s

Outputs
-------
reports/pre_event_depth/
  depth_bin_results.csv
  depth_progressive_results.csv
  depth_analysis.png
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent   # ComputationalModelsSoSe26-main/
BATON   = PROJECT / "dataset"
OUT     = PROJECT / "reports" / "pre_event_depth"
OUT.mkdir(parents=True, exist_ok=True)

POST_SEC = 3.0
ASOF_TOL = {"driver_state": 0.03, "planning": 0.03, "radar": 0.03, "imu": 0.015}

# 1-second bins relative to t_event (end inclusive at 0)
BINS = [
    (-5.0, -4.0, "[-5,-4]s"),
    (-4.0, -3.0, "[-4,-3]s"),
    (-3.0, -2.0, "[-3,-2]s"),
    (-2.0, -1.0, "[-2,-1]s"),
    (-1.0,  0.0, "[-1, 0]s"),
]
# progressive windows ending at 0
PROGRESSIVE = [
    (-1.0, 0.0, "last 1s"),
    (-2.0, 0.0, "last 2s"),
    (-3.0, 0.0, "last 3s"),
    (-4.0, 0.0, "last 4s"),
    (-5.0, 0.0, "full 5s"),
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def ns(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_csv(path: Path, time_col: str = "time_s") -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if time_col not in df.columns:
        return None
    df[time_col] = ns(df[time_col])
    return df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


def merge_route(route_path: Path) -> pd.DataFrame:
    base = load_csv(route_path / "vehicle_dynamics.csv")
    if base is None:
        raise FileNotFoundError(route_path)
    for name, fname in [("driver_state", "driver_state.csv"), ("planning", "planning.csv"),
                        ("radar", "radar.csv"), ("imu", "imu.csv")]:
        df = load_csv(route_path / fname)
        if df is None:
            continue
        overlap = sorted(set(base.columns) & set(df.columns) - {"time_s"})
        if overlap:
            df = df.drop(columns=overlap)
        base = pd.merge_asof(base.sort_values("time_s"), df.sort_values("time_s"),
                             on="time_s", direction="nearest", tolerance=ASOF_TOL[name])
    return base


# ── feature engineering (same as ablation_study.py) ──────────────────────────

def head_pose_state(df: pd.DataFrame) -> pd.Series:
    yaw   = ns(df.get("face_yaw",   pd.Series(0, index=df.index))).abs().fillna(0)
    pitch = ns(df.get("face_pitch", pd.Series(0, index=df.index))).abs().fillna(0)
    roll  = ns(df.get("face_roll",  pd.Series(0, index=df.index))).abs().fillna(0)
    ys = np.where(yaw > 30, 2, np.where(yaw > 15, 1, 0))
    ps = np.where(pitch > 20, 2, np.where(pitch > 10, 1, 0))
    rs = np.where(roll  > 20, 2, np.where(roll  > 10, 1, 0))
    return pd.Series(np.maximum(np.maximum(ys, ps), rs), index=df.index, dtype=float)


def lateral_accel(df: pd.DataFrame) -> pd.Series:
    v = ns(df.get("vEgo",             pd.Series(0, index=df.index))).fillna(0)
    a = ns(df.get("steeringAngleDeg", pd.Series(0, index=df.index))).fillna(0)
    return (v ** 2) * np.tan(np.deg2rad(a))


def ttc_class(df: pd.DataFrame) -> pd.Series:
    d = ns(df.get("leadOne_dRel", pd.Series(np.nan, index=df.index)))
    v = ns(df.get("leadOne_vRel", pd.Series(np.nan, index=df.index)))
    ttc = (d / v.abs().replace(0, np.nan)).clip(0, 60)
    return pd.Series(np.where(ttc.isna(), 0, np.where(ttc < 2, 2, np.where(ttc < 5, 1, 0))),
                     index=df.index, dtype=float)


LANE_MAP = {"LANE_CHANGE_NONE": 0, "LANE_CHANGE_PREPING": 1,
            "LANE_CHANGE_STARTING": 2, "LANE_CHANGE_LANE_CHANGE": 3}

FEAT_COLS = ["head_pose_state", "vEgo_bin10", "lateral_accel",
             "ttc_class", "laneChangeState", "gyro_x", "gyro_y", "gyro_z"]


def build_features(w: pd.DataFrame) -> pd.DataFrame:
    w = w.reset_index(drop=True)
    vego = (np.round(ns(w.get("vEgo", pd.Series(np.nan, index=w.index))) / 10) * 10).fillna(0)
    lcs  = w["laneChangeState"].map(lambda v: float(LANE_MAP.get(str(v), 0))) \
           if "laneChangeState" in w.columns else pd.Series(0.0, index=w.index)
    return pd.DataFrame({
        "head_pose_state": head_pose_state(w),
        "vEgo_bin10":      vego,
        "lateral_accel":   lateral_accel(w),
        "ttc_class":       ttc_class(w),
        "laneChangeState": lcs,
        "gyro_x": ns(w.get("gyro_x", pd.Series(0, index=w.index))).fillna(0),
        "gyro_y": ns(w.get("gyro_y", pd.Series(0, index=w.index))).fillna(0),
        "gyro_z": ns(w.get("gyro_z", pd.Series(0, index=w.index))).fillna(0),
    })


def window_to_feature_vector(feats: pd.DataFrame) -> np.ndarray:
    """Summarise a window into a flat feature vector: mean + std per feature."""
    arr = feats[FEAT_COLS].ffill().fillna(0.0).values
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0, ddof=0)])


# ── quality score (same as ablation) ─────────────────────────────────────────

def reversal_count(series: pd.Series, thr: float = 0.5) -> int:
    v = series.dropna().values
    if len(v) < 2:
        return 0
    rate = np.diff(v)
    rate = rate[np.abs(rate) >= thr]
    if len(rate) < 2:
        return 0
    return int(np.sum(np.sign(rate[1:]) * np.sign(rate[:-1]) < 0))


def compute_quality(post: pd.DataFrame) -> float:
    d = ns(post.get("leadOne_dRel", pd.Series(dtype=float)))
    v = ns(post.get("leadOne_vRel", pd.Series(dtype=float)))
    ttc = (d / v.abs().replace(0, np.nan)).clip(0, 30)
    ttc_s = float(np.clip(ttc.mean() / 30.0, 0, 1)) if ttc.notna().any() else 0.5

    steer = ns(post.get("steeringAngleDeg", pd.Series(dtype=float)))
    rev_s = float(np.clip(1.0 - reversal_count(steer) / 20.0, 0, 1))
    return round(0.5 * ttc_s + 0.5 * rev_s, 4)


# ── dataset builder ───────────────────────────────────────────────────────────

def build_dataset() -> List[Dict]:
    events = pd.read_csv(BATON / "benchmark/takeover_events_or.csv")
    events = events[events["event_type"].str.lower() == "takeover"].copy()
    events["t_event"] = pd.to_numeric(events["event_time_sec"], errors="coerce")
    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    cache: Dict[str, pd.DataFrame] = {}
    records: List[Dict] = []
    for _, ev in events.iterrows():
        rid = str(ev["route_id"])
        rp  = BATON / rid
        if not (rp / "vehicle_dynamics.csv").exists():
            continue
        if rid not in cache:
            try:
                cache[rid] = merge_route(rp)
            except Exception:
                continue
        merged = cache[rid]
        t   = ns(merged["time_s"])
        tev = float(ev["t_event"])
        full_pre  = merged.loc[(t >= tev - 5.0) & (t < tev)]
        post      = merged.loc[(t >= tev) & (t <= tev + POST_SEC)]
        if full_pre.empty or post.empty:
            continue
        try:
            feats = build_features(full_pre)
            q     = compute_quality(post)
        except Exception as e:
            warnings.warn(str(e))
            continue
        records.append({
            "event_id": str(ev["event_id"]),
            "t_event":  tev,
            "full_pre": full_pre,
            "feats":    feats,
            "quality":  q,
        })
    return records


# ── windowed feature extraction ───────────────────────────────────────────────

def extract_window_features(records: List[Dict], t_start: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for r in records:
        tev   = r["t_event"]
        pre   = r["full_pre"]
        feats = r["feats"]
        t_col = ns(pre["time_s"]).values
        mask  = (t_col >= tev + t_start) & (t_col <= tev + t_end)
        if mask.sum() < 2:
            continue
        sub_feats = feats.iloc[mask]
        vec = window_to_feature_vector(sub_feats)
        X_list.append(vec)
        y_list.append(r["quality"])
    if not X_list:
        return np.empty((0, 0)), np.empty(0)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ── XGBoost LOO ──────────────────────────────────────────────────────────────

def loo_xgb(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(X) < 4:
        return np.nan, np.nan
    preds = np.zeros(len(y), dtype=np.float32)
    for tr, te in LeaveOneOut().split(X):
        m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                         verbosity=0, random_state=42)
        m.fit(X[tr], y[tr])
        preds[te[0]] = float(np.ravel(m.predict(X[te]))[0])
    preds = np.clip(preds, 0, 1)
    mae  = float(np.mean(np.abs(preds - y)))
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    return mae, rmse


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    print("=== Building full dataset ===")
    records = build_dataset()
    print(f"  {len(records)} events loaded\n")

    # 1. Bin-only analysis
    print("=== Bin-only analysis ===")
    bin_rows = []
    for t_start, t_end, label in BINS:
        X, y = extract_window_features(records, t_start, t_end)
        print(f"  {label}  n={len(y)}", end="")
        if len(y) < 4:
            print("  [skip — too few events]")
            bin_rows.append({"bin": label, "n": len(y), "MAE": np.nan, "RMSE": np.nan})
            continue
        mae, rmse = loo_xgb(X, y)
        print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}")
        bin_rows.append({"bin": label, "n": len(y), "MAE": mae, "RMSE": rmse})

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(OUT / "depth_bin_results.csv", index=False)

    # 2. Progressive window analysis
    print("\n=== Progressive window analysis ===")
    prog_rows = []
    for t_start, t_end, label in PROGRESSIVE:
        X, y = extract_window_features(records, t_start, t_end)
        print(f"  {label}  n={len(y)}", end="")
        if len(y) < 4:
            print("  [skip — too few events]")
            prog_rows.append({"window": label, "duration_s": abs(t_start), "n": len(y),
                              "MAE": np.nan, "RMSE": np.nan})
            continue
        mae, rmse = loo_xgb(X, y)
        print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}")
        prog_rows.append({"window": label, "duration_s": abs(t_start), "n": len(y),
                          "MAE": mae, "RMSE": rmse})

    prog_df = pd.DataFrame(prog_rows)
    prog_df.to_csv(OUT / "depth_progressive_results.csv", index=False)

    # 3. Plots
    BLUE   = "#2E86AB"
    ORANGE = "#E07A5F"
    GREEN  = "#7A9E7E"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bin-only
    ax = axes[0]
    colors = [GREEN if not np.isnan(v) else "#cccccc" for v in bin_df["MAE"]]
    bars = ax.bar(bin_df["bin"], bin_df["MAE"].fillna(0), color=colors, alpha=0.85, edgecolor="white")
    for bar, row in zip(bars, bin_df.itertuples()):
        if not np.isnan(row.MAE):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{row.MAE:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("LOO-CV MAE (lower = better)")
    ax.set_title("Bin-only: which 1-second window predicts best?")
    ax.set_xlabel("Pre-event time bin")
    if bin_df["MAE"].notna().any():
        ax.set_ylim(0, bin_df["MAE"].max() * 1.3)

    # Progressive
    ax2 = axes[1]
    valid_prog = prog_df.dropna(subset=["MAE"])
    if not valid_prog.empty:
        ax2.plot(valid_prog["duration_s"], valid_prog["MAE"], "o-",
                 color=BLUE, lw=2, ms=8, label="MAE")
        ax2.plot(valid_prog["duration_s"], valid_prog["RMSE"], "s--",
                 color=ORANGE, lw=1.5, ms=7, label="RMSE")
        for _, row in valid_prog.iterrows():
            ax2.text(row["duration_s"], row["MAE"] + 0.002, f"{row['MAE']:.4f}",
                     ha="center", va="bottom", fontsize=8.5)
        ax2.set_xlabel("Pre-event window duration (s)")
        ax2.set_ylabel("LOO-CV error (lower = better)")
        ax2.set_title("Progressive windows: does longer context help?")
        ax2.set_xticks(valid_prog["duration_s"].tolist())
        ax2.legend()
        ax2.set_ylim(0, valid_prog[["MAE", "RMSE"]].max().max() * 1.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient data", transform=ax2.transAxes, ha="center")

    fig.suptitle("Pre-Event Depth Analysis — Takeover Quality Prediction (LOO-CV, XGBoost)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "depth_analysis.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    print(f"\n=== Results saved to {OUT} ===")
    print("\nBin-only:\n", bin_df.to_string(index=False))
    print("\nProgressive:\n", prog_df.to_string(index=False))


if __name__ == "__main__":
    main()
