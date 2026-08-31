#!/usr/bin/env python3
"""Plot all takeover events in one combined figure.

Rows = features, Columns = events (sorted by composite quality score).
A vertical red dashed line marks t=0 for every panel.

Usage:
    python scripts/plot_pre_event_timeseries_combined.py \
        --dataset-root dataset \
        --events dataset/benchmark/takeover_events_or.csv \
        --quality reports/takeover_quality_eda/post_event_quality_metrics.csv \
        --output reports/pre_event_timeseries/all_events_combined.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

PRE_WINDOW_SEC  = 5.0
POST_WINDOW_SEC = 3.0

# Features to show as rows, in display order
FEATURES: List[Tuple[str, str]] = [
    ("awarenessStatus",        "Driver State"),
    ("isDistracted",           "Driver State"),
    ("notReadyProb_1",         "Driver State"),
    ("occludedProb",           "Driver State"),
    ("vEgo",                   "Vehicle"),
    ("aEgo",                   "Vehicle"),
    ("steeringAngleDeg",       "Vehicle"),
    ("steeringTorque",         "Vehicle"),
    ("brake",                  "Vehicle"),
    ("leadOne_dRel",           "Radar"),
    ("leadOne_vRel",           "Radar"),
    ("laneLeft_y",             "Planner"),
    ("laneRight_y",            "Planner"),
    ("accel_x",                "IMU"),
    ("gyro_z",                 "IMU"),
]

GROUP_COLORS = {
    "Driver State": "#2E86AB",
    "Vehicle":      "#4F6D7A",
    "Radar":        "#E07A5F",
    "Planner":      "#7A9E7E",
    "IMU":          "#F2A65A",
}

MODALITY_FILES = {
    "vehicle_dynamics": "vehicle_dynamics.csv",
    "planning":         "planning.csv",
    "radar":            "radar.csv",
    "driver_state":     "driver_state.csv",
    "imu":              "imu.csv",
}
ASOF_TOLERANCE = {
    "planning":     0.030,
    "radar":        0.030,
    "driver_state": 0.030,
    "imu":          0.015,
}


def numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_route(route_path: Path, time_col: str = "time_s") -> pd.DataFrame:
    frames: Dict[str, pd.DataFrame] = {}
    for name, fname in MODALITY_FILES.items():
        p = route_path / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if time_col not in df.columns:
            continue
        df[time_col] = numeric_series(df[time_col])
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)
        frames[name] = df

    if not frames:
        raise FileNotFoundError(f"No usable CSVs in {route_path}")

    base = frames.get("vehicle_dynamics")
    if base is None:
        base = next(iter(frames.values()))
    merged = base.copy()

    for name, df in frames.items():
        if df is base:
            continue
        overlap = sorted(set(merged.columns) & set(df.columns) - {time_col})
        if overlap:
            df = df.drop(columns=overlap)
        tol = ASOF_TOLERANCE.get(name)
        merged = pd.merge_asof(
            merged.sort_values(time_col),
            df.sort_values(time_col),
            on=time_col,
            direction="nearest",
            tolerance=tol,
        )
    return merged


def composite_score(row: pd.Series) -> float:
    cols = ["reaction_smoothness_score", "steering_smoothness_score",
            "longitudinal_smoothness_score", "lateral_stability_score"]
    vals = [float(row[c]) for c in cols if c in row.index and pd.notna(row[c])]
    return float(np.mean(vals)) if vals else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--events",   type=Path, default=None)
    p.add_argument("--quality",  type=Path, default=None)
    p.add_argument("--output",   type=Path,
                   default=Path("reports/pre_event_timeseries/all_events_combined.png"))
    p.add_argument("--max-events", type=int, default=None)
    return p.parse_args()


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 140})

    args = parse_args()
    dataset_root: Path = args.dataset_root
    events_path:  Path = args.events or dataset_root / "benchmark" / "takeover_events_or.csv"

    # --- load events ---
    events = pd.read_csv(events_path)
    if "event_type" in events.columns:
        events = events[events["event_type"].str.lower() == "takeover"]
    for col in ("event_time_sec", "t_event", "nearest_event_time"):
        if col in events.columns:
            events["t_event"] = pd.to_numeric(events[col], errors="coerce")
            break
    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    # --- load quality ---
    quality_df: Optional[pd.DataFrame] = None
    if args.quality and args.quality.exists():
        quality_df = pd.read_csv(args.quality).set_index("event_id")

    # --- sort events by composite quality (best → worst) ---
    def get_q(eid: str) -> float:
        if quality_df is not None and eid in quality_df.index:
            return composite_score(quality_df.loc[eid])
        return float("nan")

    events["composite_q"] = events["event_id"].apply(get_q)
    events = events.sort_values("composite_q", ascending=False).reset_index(drop=True)
    if args.max_events:
        events = events.head(args.max_events)

    # --- extract windows ---
    route_cache: Dict[str, pd.DataFrame] = {}
    windows: List[pd.DataFrame] = []
    event_meta: List[dict] = []

    for _, event in events.iterrows():
        route_id  = str(event["route_id"])
        route_path = dataset_root / route_id
        if not route_path.exists():
            continue
        if route_id not in route_cache:
            try:
                route_cache[route_id] = load_route(route_path)
            except FileNotFoundError:
                continue

        route_df = route_cache[route_id]
        t_event  = float(event["t_event"])
        times    = numeric_series(route_df["time_s"])
        mask     = (times >= t_event - PRE_WINDOW_SEC) & (times <= t_event + POST_WINDOW_SEC)
        w        = route_df.loc[mask].copy()
        if w.empty:
            continue
        w["rel_t"] = numeric_series(w["time_s"]) - t_event

        windows.append(w)
        q   = float(event["composite_q"]) if pd.notna(event["composite_q"]) else float("nan")
        ttype = ""
        if quality_df is not None and str(event["event_id"]) in quality_df.index:
            ttype = str(quality_df.loc[str(event["event_id"]), "takeover_type"])
        event_meta.append({
            "event_id":     str(event["event_id"]),
            "composite_q":  q,
            "takeover_type": ttype,
        })

    n_events   = len(windows)
    n_features = len(FEATURES)

    if n_events == 0:
        print("No events to plot.")
        return

    print(f"Plotting {n_events} events × {n_features} features …")

    # --- figure layout ---
    col_w    = max(1.5, min(2.2, 120 / n_events))
    row_h    = 1.5
    fig_w    = col_w * n_events + 1.5   # +1.5 for row labels
    fig_h    = row_h * n_features + 2.0  # +2.0 for column headers

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        n_features, n_events,
        figure=fig,
        hspace=0.08,
        wspace=0.05,
        left=0.08,
        right=0.99,
        top=0.93,
        bottom=0.06,
    )

    fig.suptitle(
        "Pre-Takeover Feature Time-Series — All Events\n"
        "(sorted left→right: best quality → worst quality   ·   red dashed = takeover moment)",
        fontsize=10, y=0.98,
    )

    for row_i, (feat, group) in enumerate(FEATURES):
        color = GROUP_COLORS.get(group, "#555555")
        for col_j, (w, meta) in enumerate(zip(windows, event_meta)):
            ax = fig.add_subplot(gs[row_i, col_j])

            if feat in w.columns:
                series = numeric_series(w[feat])
                ax.plot(w["rel_t"], series, color=color, linewidth=0.7, rasterized=True)

            ax.axvline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.85)
            ax.set_xlim(-PRE_WINDOW_SEC, POST_WINDOW_SEC)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False)

            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

            # Column header (top row only)
            if row_i == 0:
                q    = meta["composite_q"]
                q_s  = f"Q={q:.2f}" if pd.notna(q) else "Q=?"
                ttype = meta["takeover_type"]
                short_type = ttype[:3] if ttype else "?"
                eid_short = meta["event_id"].replace("TAKE_", "")
                ax.set_title(
                    f"{eid_short}\n{short_type} {q_s}",
                    fontsize=5.5, pad=2,
                    color=("green" if (pd.notna(q) and q >= 0.8)
                           else "orange" if (pd.notna(q) and q >= 0.6)
                           else "red"),
                )

            # Row label (leftmost column only)
            if col_j == 0:
                ax.set_ylabel(feat, fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
