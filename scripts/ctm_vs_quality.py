#!/usr/bin/env python3
"""
Compare Continuous Readiness Metric (CTM) with takeover quality scores.

Computes CTM (r_tilde ∈ [0,1]) from driver_state.csv using the formula in
continuous_readiness_metric.md:

  ready_raw  = sum(readyProb_0..3) / (sum(readyProb_0..3) + sum(notReadyProb_0..1))
  ready_mod  = ready_raw * faceDetected * (1 - occludedProb) * (1 - poorVisionProb)
  r_raw      = alpha * ready_mod + (1 - alpha) * clip(awarenessStatus, 0, 1)
  r_tilde    = EMA(r_raw, beta=0.9)   [α=0.5, β=0.9]

For each event we extract:
  - ctm_at_event:    r_tilde at t_event (the readiness at the moment of takeover)
  - ctm_mean_pre:    mean r_tilde over [t-5s, t)
  - ctm_min_pre:     minimum r_tilde over [t-5s, t)
  - ctm_slope_pre:   linear trend slope of r_tilde over [t-5s, t)

These are correlated against the composite quality score.

Outputs
-------
reports/ctm_vs_quality/
  ctm_event_features.csv      – CTM features per event
  ctm_quality_correlations.csv
  ctm_vs_quality.png
  ctm_trajectory_grid.png     – r_tilde traces around events
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
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent   # ComputationalModelsSoSe26-main/
BATON   = PROJECT / "dataset"
OUT     = PROJECT / "reports" / "ctm_vs_quality"
OUT.mkdir(parents=True, exist_ok=True)

PRE_SEC  = 5.0
POST_SEC = 3.0
ALPHA    = 0.5   # blend between softmax-derived and awarenessStatus
BETA     = 0.9   # EMA smoothing

ASOF_TOL = {"planning": 0.03, "radar": 0.03, "imu": 0.015}


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


def load_route_with_driver_state(route_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    vd = load_csv(route_path / "vehicle_dynamics.csv")
    ds = load_csv(route_path / "driver_state.csv")
    return vd, ds


# ── CTM computation ───────────────────────────────────────────────────────────

def compute_ctm(ds: pd.DataFrame, alpha: float = ALPHA, beta: float = BETA) -> pd.DataFrame:
    """Return a DataFrame with columns [time_s, r_tilde]."""
    ds = ds.copy().sort_values("time_s").reset_index(drop=True)

    # Step 1: base score from softmax probabilities
    ready_cols      = [c for c in ["readyProb_0","readyProb_1","readyProb_2","readyProb_3"] if c in ds.columns]
    not_ready_cols  = [c for c in ["notReadyProb_0","notReadyProb_1"] if c in ds.columns]

    if ready_cols and not_ready_cols:
        ready_sum    = ds[ready_cols].apply(ns).sum(axis=1)
        not_ready_sum = ds[not_ready_cols].apply(ns).sum(axis=1)
        denom        = (ready_sum + not_ready_sum).replace(0, np.nan)
        ready_raw    = (ready_sum / denom).fillna(0.5)
    else:
        ready_raw    = pd.Series(0.5, index=ds.index)

    # Step 2: visibility modulation
    face_det      = ns(ds["faceDetected"]).fillna(1.0).clip(0, 1) \
                    if "faceDetected" in ds.columns else pd.Series(1.0, index=ds.index)
    occluded      = ns(ds["occludedProb"]).fillna(0.0).clip(0, 1) \
                    if "occludedProb" in ds.columns else pd.Series(0.0, index=ds.index)
    poor_vision   = ns(ds["poorVisionProb"]).fillna(0.0).clip(0, 1) \
                    if "poorVisionProb" in ds.columns else pd.Series(0.0, index=ds.index)
    ready_mod     = ready_raw * face_det * (1.0 - occluded) * (1.0 - poor_vision)

    # Step 3: blend with awarenessStatus
    awareness     = ns(ds["awarenessStatus"]).clip(0, 1) \
                    if "awarenessStatus" in ds.columns else pd.Series(0.5, index=ds.index)
    awareness     = awareness.fillna(0.5)
    r_raw         = alpha * ready_mod + (1.0 - alpha) * awareness

    # Step 4: EMA
    r_tilde = np.zeros(len(r_raw), dtype=float)
    r_tilde[0] = float(r_raw.iloc[0])
    for i in range(1, len(r_raw)):
        r_tilde[i] = beta * r_tilde[i-1] + (1.0 - beta) * float(r_raw.iloc[i])

    return pd.DataFrame({"time_s": ds["time_s"].values, "r_tilde": r_tilde})


# ── quality score ─────────────────────────────────────────────────────────────

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

def build_dataset() -> pd.DataFrame:
    events = pd.read_csv(BATON / "benchmark/takeover_events_or.csv")
    events = events[events["event_type"].str.lower() == "takeover"].copy()
    events["t_event"] = pd.to_numeric(events["event_time_sec"], errors="coerce")
    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    vd_cache: Dict[str, pd.DataFrame] = {}
    ds_cache: Dict[str, Optional[pd.DataFrame]] = {}
    ctm_cache: Dict[str, pd.DataFrame] = {}
    rows: List[Dict] = []

    for _, ev in events.iterrows():
        rid = str(ev["route_id"])
        rp  = BATON / rid
        if not (rp / "vehicle_dynamics.csv").exists():
            continue

        # load vehicle dynamics
        if rid not in vd_cache:
            df = load_csv(rp / "vehicle_dynamics.csv")
            if df is None:
                continue
            for name, fname in [("planning", "planning.csv"), ("radar", "radar.csv")]:
                other = load_csv(rp / fname)
                if other is None:
                    continue
                overlap = sorted(set(df.columns) & set(other.columns) - {"time_s"})
                if overlap:
                    other = other.drop(columns=overlap)
                df = pd.merge_asof(df.sort_values("time_s"), other.sort_values("time_s"),
                                   on="time_s", direction="nearest",
                                   tolerance=ASOF_TOL.get(name))
            vd_cache[rid] = df

        # load driver_state and compute CTM
        if rid not in ds_cache:
            ds_cache[rid] = load_csv(rp / "driver_state.csv")
        if rid not in ctm_cache:
            if ds_cache[rid] is not None:
                ctm_cache[rid] = compute_ctm(ds_cache[rid])
            else:
                ctm_cache[rid] = None

        merged = vd_cache[rid]
        ctm_df = ctm_cache.get(rid)
        t   = ns(merged["time_s"])
        tev = float(ev["t_event"])

        post = merged.loc[(t >= tev) & (t <= tev + POST_SEC)]
        if post.empty:
            continue
        q = compute_quality(post)

        if ctm_df is None:
            continue
        ctm_t   = ctm_df["time_s"].values
        ctm_r   = ctm_df["r_tilde"].values

        pre_mask = (ctm_t >= tev - PRE_SEC) & (ctm_t < tev)
        if pre_mask.sum() < 2:
            continue
        ctm_pre_r  = ctm_r[pre_mask]
        ctm_pre_t  = ctm_t[pre_mask]

        # CTM at event: last value in driver_state before t_event
        at_event_idx = np.searchsorted(ctm_t, tev, side="right") - 1
        ctm_at_event = float(ctm_r[max(at_event_idx, 0)])

        # slope via linear regression
        if len(ctm_pre_t) >= 3:
            slope, _, _, _, _ = stats.linregress(ctm_pre_t - tev, ctm_pre_r)
        else:
            slope = np.nan

        rows.append({
            "event_id":      str(ev["event_id"]),
            "route_id":      rid,
            "t_event":       tev,
            "quality":       q,
            "ctm_at_event":  ctm_at_event,
            "ctm_mean_pre":  float(np.mean(ctm_pre_r)),
            "ctm_min_pre":   float(np.min(ctm_pre_r)),
            "ctm_slope_pre": float(slope) if pd.notna(slope) else np.nan,
            "_ctm_pre_t":    ctm_pre_t,   # for trajectory plot
            "_ctm_pre_r":    ctm_pre_r,
        })

    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_ctm_vs_quality(df: pd.DataFrame, out: Path) -> None:
    ctm_cols = ["ctm_at_event", "ctm_mean_pre", "ctm_min_pre", "ctm_slope_pre"]
    titles   = ["CTM at t_event", "Mean CTM [t-5s, t)", "Min CTM [t-5s, t)", "CTM slope [t-5s, t)"]
    xlabels  = ["r̃ at event", "mean r̃", "min r̃", "slope of r̃ (s⁻¹)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, col, title, xlabel in zip(axes.ravel(), ctm_cols, titles, xlabels):
        pair = df[["quality", col]].dropna()
        if len(pair) < 3:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue
        x, y = pair[col].values, pair["quality"].values
        r, p = stats.pearsonr(x, y)
        ax.scatter(x, y, s=60, alpha=0.8, color="#2E86AB", edgecolors="white", lw=0.5)
        try:
            slope_lr, intercept, *_ = stats.linregress(x, y)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, slope_lr * xfit + intercept, "r--", lw=1.5, alpha=0.7)
        except Exception:
            pass
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Composite quality score")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.set_title(f"{title}\nPearson r = {r:.3f}{sig}  (n={len(pair)})")

    fig.suptitle("CTM Features vs Takeover Quality Score", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "ctm_vs_quality.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_ctm_trajectories(df: pd.DataFrame, out: Path, n_events: int = 24) -> None:
    """Plot r_tilde traces in the pre-event window, coloured by quality score."""
    rows = df.dropna(subset=["quality", "ctm_at_event"]).head(n_events)
    if rows.empty:
        return

    ncols = 4
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    cmap = plt.cm.RdYlGn
    for ax, (_, row) in zip(axes_flat, rows.iterrows()):
        t = row["_ctm_pre_t"] - row["t_event"]
        r = row["_ctm_pre_r"]
        colour = cmap(float(row["quality"]))
        ax.plot(t, r, color=colour, lw=1.5)
        ax.axhline(row["ctm_at_event"], color="black", lw=0.7, linestyle=":")
        ax.set_xlim(-PRE_SEC, 0)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{row['event_id']}\nQ={row['quality']:.2f}", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(rows):]:
        ax.set_visible(False)

    fig.suptitle("CTM trajectories in pre-event window (coloured: red=low Q, green=high Q)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ctm_trajectory_grid.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    print("=== Building dataset: CTM features + quality scores ===")
    df = build_dataset()
    print(f"  Events loaded: {len(df)}")

    # save feature table (drop internal columns)
    save_cols = ["event_id", "route_id", "t_event", "quality",
                 "ctm_at_event", "ctm_mean_pre", "ctm_min_pre", "ctm_slope_pre"]
    df[save_cols].to_csv(OUT / "ctm_event_features.csv", index=False)
    print(df[save_cols].describe().to_string())

    # correlations
    ctm_cols = ["ctm_at_event", "ctm_mean_pre", "ctm_min_pre", "ctm_slope_pre"]
    corr_rows = []
    for col in ctm_cols:
        pair = df[["quality", col]].dropna()
        if len(pair) >= 3:
            r, p = stats.pearsonr(pair[col], pair["quality"])
            rs, ps = stats.spearmanr(pair[col], pair["quality"])
        else:
            r = p = rs = ps = np.nan
        corr_rows.append({"ctm_feature": col, "pearson_r": r, "pearson_p": p,
                          "spearman_r": rs, "spearman_p": ps, "n": len(pair)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "ctm_quality_correlations.csv", index=False)

    print("\nCorrelations:")
    print(corr_df.to_string(index=False))

    print("\n=== Generating plots ===")
    plot_ctm_vs_quality(df, OUT)
    plot_ctm_trajectories(df, OUT)
    print(f"  Saved to {OUT}")


if __name__ == "__main__":
    main()
