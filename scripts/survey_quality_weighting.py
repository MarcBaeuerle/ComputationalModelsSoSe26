#!/usr/bin/env python3
"""
Survey-based quality axis weighting for takeover events.

Maps 4-respondent human ratings (1–5 Likert) to the four BATON quality axes
and finds optimal composite weights via constrained optimization.

Survey events 1–32 are mapped sequentially to BATON events
TAKE_00152–TAKE_00183 (first 32 driver_97 sample events, sorted by ID).

Outputs
-------
reports/survey_quality/
  survey_human_scores.csv        – per-event averaged human rating (0-1)
  quality_axis_scores.csv        – four computed quality axes per event
  survey_weight_fit.json         – optimal axis weights + stats
  survey_vs_composite.png        – scatter: human score vs composite
  weight_comparison.png          – default vs survey weights bar chart
  axis_correlation.png           – correlation of each axis vs human score
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent   # ComputationalModelsSoSe26-main/
BATON   = PROJECT / "dataset"
SURVEY  = PROJECT / "Computation Models Survey (Responses) - Form Responses 1.csv"
OUT     = PROJECT / "reports" / "survey_quality"
OUT.mkdir(parents=True, exist_ok=True)

POST_SEC = 3.0
PRE_SEC  = 5.0
ASOF_TOL = {"driver_state": 0.03, "planning": 0.03, "radar": 0.03, "imu": 0.015}

AXIS_LABELS = [
    "reaction_smoothness",
    "steering_smoothness",
    "longitudinal_smoothness",
    "lateral_stability",
]

# Survey event index → BATON event ID (sequential, first 32 driver_97 events)
# TAKE_00152-00160 (route_21, 9), 00161-00166 (route_22, 6),
# 00167-00176 (route_23, 10), 00177-00183 (route_24, 7) = 32
SURVEY_EVENT_IDS = [f"TAKE_{(152+i):05d}" for i in range(32)]


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


# ── quality metrics (compact version) ────────────────────────────────────────

def _finite_derivative(vals: np.ndarray, times: np.ndarray) -> np.ndarray:
    valid = np.diff(times) > 0
    return (np.diff(vals) / np.diff(times))[valid]


def _steering_reversals(rate: np.ndarray, thr: float = 0.5) -> int:
    r = rate[np.isfinite(rate)]
    r = r[np.abs(r) >= thr]
    if len(r) < 2:
        return 0
    return int(np.sum(np.sign(r[1:]) * np.sign(r[:-1]) < 0))


def _first_sustained(df: pd.DataFrame, mask: pd.Series, t_event: float,
                     sustained: float = 0.20) -> float:
    work = pd.DataFrame({"t": ns(df["time_s"]), "a": mask.astype(bool)}).dropna().sort_values("t")
    if work.empty or not work["a"].any():
        return np.nan
    times, active = work["t"].values, work["a"].values
    med_dt = float(np.median(np.diff(times)[np.diff(times) > 0])) if len(times) > 1 else 0.0
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends   = np.flatnonzero(active & np.r_[~active[1:], True])
    for s, e in zip(starts, ends):
        if (times[e] - times[s] + med_dt) >= sustained:
            return max(0.0, times[s] - t_event)
    return np.nan


def compute_quality_axes(post: pd.DataFrame, t_event: float) -> Dict[str, float]:
    """Return four raw quality metrics (lower = worse, higher = better)."""
    # Reaction
    gas_mask    = ns(post.get("gas",   pd.Series(0, index=post.index))).fillna(0).gt(0.01)
    brake_mask  = ns(post.get("brake", pd.Series(0, index=post.index))).fillna(0).gt(0.01)
    steer_mask  = ns(post.get("steeringTorque", pd.Series(0, index=post.index))).abs().fillna(0).gt(1.0)
    rt = _first_sustained(post, gas_mask | brake_mask | steer_mask, t_event)

    # Steering
    if "steeringAngleDeg" in post.columns and ns(post["steeringAngleDeg"]).notna().sum() >= 2:
        t_arr = ns(post["time_s"]).values
        s_arr = ns(post["steeringAngleDeg"]).values
        mask  = np.isfinite(t_arr) & np.isfinite(s_arr)
        rate  = _finite_derivative(s_arr[mask], t_arr[mask])
        steer_std = float(np.nanstd(rate, ddof=1)) if len(rate) > 1 else np.nan
        steer_rev = _steering_reversals(rate)
    else:
        steer_std, steer_rev = np.nan, 0

    # Longitudinal
    accel = ns(post["aEgo"]) if "aEgo" in post.columns else (
            ns(post["accel_x"]) if "accel_x" in post.columns else pd.Series(dtype=float))
    peak_decel = float(np.nanmin(accel)) if accel.notna().any() else np.nan
    accel_std  = float(np.nanstd(accel, ddof=1)) if accel.notna().sum() > 1 else np.nan

    # Lateral
    if {"laneLeft_y", "laneRight_y"}.issubset(post.columns):
        lane_var = float(np.nanvar((ns(post["laneLeft_y"]) + ns(post["laneRight_y"])) / 2.0, ddof=1))
    else:
        lane_var = np.nan
    yaw_var = float(np.nanvar(ns(post["gyro_z"]), ddof=1)) if "gyro_z" in post.columns else np.nan

    return {
        "reaction_time_s":        rt,
        "steering_rate_std":      steer_std,
        "steering_reversal_count": steer_rev,
        "peak_deceleration_cost": max(0.0, -peak_decel) if pd.notna(peak_decel) else np.nan,
        "acceleration_std":       accel_std,
        "lane_center_variance":   lane_var,
        "yaw_rate_variance":      yaw_var,
    }


def _inv_minmax(cost: pd.Series, worst: Optional[float] = None) -> pd.Series:
    v = ns(cost)
    if worst is None:
        worst = float(v[np.isfinite(v)].max()) if v.notna().any() else 0.0
    denom = float(worst - 0.0)
    if denom <= 0:
        return v.where(v.isna(), 1.0)
    return (1.0 - (v / denom)).clip(0, 1)


def normalize_quality_axes(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.copy()
    m["reaction_smoothness"]    = _inv_minmax(m["reaction_time_s"].fillna(POST_SEC), worst=POST_SEC)
    m["steering_smoothness"]    = (_inv_minmax(m["steering_rate_std"]) +
                                   _inv_minmax(m["steering_reversal_count"])) / 2.0
    m["longitudinal_smoothness"] = (_inv_minmax(m["peak_deceleration_cost"]) +
                                    _inv_minmax(m["acceleration_std"])) / 2.0
    m["lateral_stability"]      = (_inv_minmax(m["lane_center_variance"]) +
                                   _inv_minmax(m["yaw_rate_variance"])) / 2.0
    return m


# ── survey parsing ────────────────────────────────────────────────────────────

def parse_survey(path: Path) -> pd.Series:
    """Return a Series indexed 1-32 with averaged human quality scores in [0,1]."""
    df = pd.read_csv(path)
    score_cols = [c for c in df.columns if "[Takeover" in c]
    if len(score_cols) != 32:
        warnings.warn(f"Expected 32 takeover columns, found {len(score_cols)}.")

    # extract leading integer from "N. Label"
    def parse_rating(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).split(".")[0].strip())
        except ValueError:
            return np.nan

    ratings = df[score_cols].map(parse_rating)  # (n_respondents, 32)
    mean_ratings = ratings.mean(axis=0)               # averaged across respondents

    # normalise 1-5 → 0-1
    normalised = (mean_ratings.values - 1.0) / 4.0
    result = pd.Series(normalised, index=range(1, 33), name="human_score")
    return result


# ── dataset builder ───────────────────────────────────────────────────────────

def build_quality_table(event_ids: List[str]) -> pd.DataFrame:
    events_all = pd.read_csv(BATON / "benchmark/takeover_events_or.csv")
    events = events_all[events_all["event_id"].isin(event_ids)].copy()
    events["t_event"] = pd.to_numeric(events["event_time_sec"], errors="coerce")
    events = events.dropna(subset=["t_event"]).set_index("event_id").reindex(event_ids)

    cache: Dict[str, pd.DataFrame] = {}
    rows = []
    for eid in event_ids:
        row = events.loc[eid]
        rid = str(row["route_id"])
        rp  = BATON / rid
        if not (rp / "vehicle_dynamics.csv").exists():
            rows.append({"event_id": eid, **{k: np.nan for k in [
                "reaction_time_s", "steering_rate_std", "steering_reversal_count",
                "peak_deceleration_cost", "acceleration_std",
                "lane_center_variance", "yaw_rate_variance"]}})
            continue
        if rid not in cache:
            try:
                cache[rid] = merge_route(rp)
            except Exception as e:
                warnings.warn(f"Route {rid}: {e}")
                rows.append({"event_id": eid, **{k: np.nan for k in [
                    "reaction_time_s", "steering_rate_std", "steering_reversal_count",
                    "peak_deceleration_cost", "acceleration_std",
                    "lane_center_variance", "yaw_rate_variance"]}})
                continue
        merged = cache[rid]
        t      = ns(merged["time_s"])
        tev    = float(row["t_event"])
        post   = merged.loc[(t >= tev) & (t <= tev + POST_SEC)].copy()
        if post.empty:
            rows.append({"event_id": eid, **{k: np.nan for k in [
                "reaction_time_s", "steering_rate_std", "steering_reversal_count",
                "peak_deceleration_cost", "acceleration_std",
                "lane_center_variance", "yaw_rate_variance"]}})
            continue
        ax = compute_quality_axes(post, tev)
        rows.append({"event_id": eid, **ax})

    return pd.DataFrame(rows)


# ── weight optimisation ───────────────────────────────────────────────────────

def fit_weights(
    axis_scores: pd.DataFrame,
    human_scores: pd.Series,
) -> Tuple[np.ndarray, float, float]:
    """Return (weights, pearson_r, default_pearson_r)."""
    cols  = AXIS_LABELS
    X = axis_scores[cols].values  # (n, 4)
    y = human_scores.values       # (n,)

    # drop rows where any axis or human score is missing
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    print(f"  Optimising on {valid.sum()} / {len(valid)} events (others have missing axes)")

    default_w = np.array([0.25, 0.25, 0.25, 0.25])
    default_comp  = X @ default_w
    default_r = float(pd.Series(default_comp).corr(pd.Series(y)))

    def neg_pearson(w):
        comp = X @ w
        if np.std(comp) < 1e-12:
            return 1.0
        return -float(np.corrcoef(comp, y)[0, 1])

    result = minimize(
        neg_pearson,
        x0=default_w,
        method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        bounds=[(0, 1)] * 4,
        options={"ftol": 1e-9, "maxiter": 2000},
    )
    opt_w = result.x / result.x.sum()
    opt_r = -result.fun
    return opt_w, opt_r, default_r


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_survey_vs_composite(
    axis_scores: pd.DataFrame,
    human_scores: pd.Series,
    opt_w: np.ndarray,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cols = AXIS_LABELS
    for ax, (w, title) in zip(axes, [
        (np.full(4, 0.25), "Equal weights (0.25 each)"),
        (opt_w,            "Survey-optimised weights"),
    ]):
        X = axis_scores[cols].values
        y = human_scores.values
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        comp  = (X @ w)[valid]
        hum   = y[valid]
        r = float(np.corrcoef(comp, hum)[0, 1])
        ax.scatter(hum, comp, s=70, alpha=0.8, color="#2E86AB", edgecolors="white", lw=0.5)
        lims = [min(hum.min(), comp.min()) - 0.05, max(hum.max(), comp.max()) + 0.05]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Human quality score (survey, 0–1)")
        ax.set_ylabel("Composite quality score (axes, 0–1)")
        ax.set_title(f"{title}\nPearson r = {r:.3f}")
    fig.suptitle("Survey Human Score vs Computed Quality Composite", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "survey_vs_composite.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_weight_comparison(default_w: np.ndarray, opt_w: np.ndarray, out: Path) -> None:
    x = np.arange(4)
    width = 0.35
    labels = ["Reaction\nSmoothness", "Steering\nSmoothness",
              "Longitudinal\nSmoothness", "Lateral\nStability"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, default_w, width, label="Equal (0.25)", color="#4F6D7A", alpha=0.85)
    ax.bar(x + width/2, opt_w,     width, label="Survey-optimised", color="#E07A5F", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1.0)
    ax.set_title("Quality Axis Weights: Default vs Survey-Optimised")
    ax.legend()
    for i, (d, o) in enumerate(zip(default_w, opt_w)):
        ax.text(i - width/2, d + 0.02, f"{d:.3f}", ha="center", fontsize=9)
        ax.text(i + width/2, o + 0.02, f"{o:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "weight_comparison.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_axis_correlations(
    axis_scores: pd.DataFrame,
    human_scores: pd.Series,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = AXIS_LABELS
    corrs, labels = [], []
    for col in cols:
        pair = pd.concat([axis_scores[col], human_scores], axis=1).dropna()
        r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) >= 3 else np.nan
        corrs.append(r)
        labels.append(col.replace("_", "\n"))
    colors = ["#2E86AB" if r >= 0 else "#E07A5F" for r in corrs]
    ax.bar(labels, corrs, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Pearson correlation with human score")
    ax.set_ylim(-1, 1)
    ax.set_title("Per-Axis Correlation with Survey Human Quality Score")
    for i, r in enumerate(corrs):
        ax.text(i, r + 0.03 * np.sign(r + 1e-9), f"{r:.3f}", ha="center", fontsize=10,
                fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "axis_correlation.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    print("=== Parsing survey ===")
    human_scores_all = parse_survey(SURVEY)
    print(f"  Survey events: {len(human_scores_all)}  |  mean score: {human_scores_all.mean():.3f}")

    # create DataFrame aligning survey idx 1-32 with event IDs
    survey_df = pd.DataFrame({
        "survey_idx": range(1, 33),
        "event_id":   SURVEY_EVENT_IDS,
        "human_score": human_scores_all.values,
    })
    print(f"\n  Event mapping (first 5):\n{survey_df.head().to_string(index=False)}")

    print("\n=== Computing quality axes for 32 events ===")
    quality_raw = build_quality_table(SURVEY_EVENT_IDS)
    quality_norm = normalize_quality_axes(quality_raw)
    quality_norm.to_csv(OUT / "quality_axis_scores.csv", index=False)
    print(quality_norm[["event_id"] + AXIS_LABELS].to_string(index=False))

    # align on event_id
    merged = survey_df.merge(quality_norm[["event_id"] + AXIS_LABELS], on="event_id", how="left")
    merged.to_csv(OUT / "survey_human_scores.csv", index=False)

    axis_for_opt = merged.set_index("event_id")[AXIS_LABELS]
    human_for_opt = merged.set_index("event_id")["human_score"]

    print("\n=== Fitting optimal weights ===")
    opt_w, opt_r, default_r = fit_weights(axis_for_opt, human_for_opt)

    print(f"\n  Default weights (equal): {np.full(4,0.25)}  Pearson r = {default_r:.4f}")
    print(f"  Optimised weights:       {np.round(opt_w,4)}  Pearson r = {opt_r:.4f}")

    result = {
        "default_weights": dict(zip(AXIS_LABELS, [0.25]*4)),
        "survey_weights":  dict(zip(AXIS_LABELS, opt_w.tolist())),
        "default_pearson_r": round(default_r, 6),
        "survey_pearson_r":  round(opt_r, 6),
        "n_events_used":     int(merged[AXIS_LABELS + ["human_score"]].notna().all(axis=1).sum()),
    }
    with open(OUT / "survey_weight_fit.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {OUT / 'survey_weight_fit.json'}")

    print("\n=== Generating plots ===")
    plot_survey_vs_composite(axis_for_opt, human_for_opt, opt_w, OUT)
    plot_weight_comparison(np.full(4, 0.25), opt_w, OUT)
    plot_axis_correlations(axis_for_opt, human_for_opt, OUT)
    print(f"  Plots saved to {OUT}")


if __name__ == "__main__":
    main()
