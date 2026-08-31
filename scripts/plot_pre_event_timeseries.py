#!/usr/bin/env python3
"""Plot per-event pre-takeover time-series for the important features.

For each takeover event, generates a multi-panel figure showing the 5-second
pre-event window of the most informative features, grouped by modality.
A vertical red dashed line marks t=0 (takeover moment).

Usage:
    python scripts/plot_pre_event_timeseries.py \
        --dataset-root dataset \
        --events dataset/benchmark/takeover_events_or.csv \
        --quality reports/takeover_quality_eda/post_event_quality_metrics.csv \
        --output-dir reports/pre_event_timeseries
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Features to plot, grouped into named panels
# ---------------------------------------------------------------------------
PANELS: Dict[str, List[str]] = {
    "Driver State": [
        "awarenessStatus",
        "isDistracted",
        "readyProb_2",      # "notReady" dominant class
        "notReadyProb_1",
        "occludedProb",
        "poorVisionProb",
    ],
    "Vehicle Dynamics": [
        "vEgo",
        "aEgo",
        "steeringAngleDeg",
        "steeringTorque",
        "gas",
        "brake",
    ],
    "Radar": [
        "leadOne_dRel",
        "leadOne_vRel",
    ],
    "Planner": [
        "model_desiredCurvature",
        "laneLeft_y",
        "laneRight_y",
        "fcw",
    ],
    "IMU": [
        "accel_x",
        "gyro_z",
    ],
}

MODALITY_FILES = {
    "vehicle_dynamics": "vehicle_dynamics.csv",
    "planning": "planning.csv",
    "radar": "radar.csv",
    "driver_state": "driver_state.csv",
    "imu": "imu.csv",
}

ASOF_TOLERANCE = {
    "planning": 0.030,
    "radar": 0.030,
    "driver_state": 0.030,
    "imu": 0.015,
}

PANEL_COLORS = {
    "Driver State":     "#2E86AB",
    "Vehicle Dynamics": "#4F6D7A",
    "Radar":            "#E07A5F",
    "Planner":          "#7A9E7E",
    "IMU":              "#F2A65A",
}

PRE_WINDOW_SEC  = 5.0
POST_WINDOW_SEC = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def quality_label(row: pd.Series) -> str:
    scores = [row.get(c) for c in [
        "reaction_smoothness_score", "steering_smoothness_score",
        "longitudinal_smoothness_score", "lateral_stability_score"
    ] if pd.notna(row.get(c))]
    composite = float(np.mean(scores)) if scores else float("nan")
    label = f"Q={composite:.2f}" if pd.notna(composite) else "Q=?"
    return label


# ---------------------------------------------------------------------------
# Per-event plot
# ---------------------------------------------------------------------------

def plot_event(
    event: pd.Series,
    route_df: pd.DataFrame,
    quality_row: Optional[pd.Series],
    output_path: Path,
    time_col: str = "time_s",
) -> None:
    t_event = float(event["t_event"])
    times = numeric_series(route_df[time_col])
    mask = (times >= t_event - PRE_WINDOW_SEC) & (times <= t_event + POST_WINDOW_SEC)
    window = route_df.loc[mask].copy()
    window["rel_t"] = numeric_series(window[time_col]) - t_event

    if window.empty:
        warnings.warn(f"Empty window for {event['event_id']}, skipping.")
        return

    # Build list of (panel_name, col) pairs that exist in the window
    to_plot: List[tuple] = []
    for panel, cols in PANELS.items():
        for col in cols:
            if col in window.columns and numeric_series(window[col]).notna().any():
                to_plot.append((panel, col))

    if not to_plot:
        warnings.warn(f"No plottable columns for {event['event_id']}, skipping.")
        return

    n = len(to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(5.5 * ncols, 3.0 * nrows + 1.2))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    q_label = quality_label(quality_row) if quality_row is not None else "Q=?"
    takeover_type = str(event.get("takeover_type", "Unknown"))
    event_id = str(event["event_id"])
    route_id = str(event.get("route_id", ""))

    suptitle = (
        f"{event_id}  |  Route: {route_id}  |  Type: {takeover_type}  |  {q_label}\n"
        f"Pre-event window = {PRE_WINDOW_SEC}s  ·  vertical line = takeover moment (t=0)"
    )
    fig.suptitle(suptitle, fontsize=10, y=1.0)

    for idx, (panel, col) in enumerate(to_plot):
        row_idx, col_idx = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row_idx, col_idx])

        color = PANEL_COLORS.get(panel, "#555555")
        series = numeric_series(window[col])
        ax.plot(window["rel_t"], series, color=color, linewidth=1.2)
        ax.axvline(0, color="red", linewidth=1.4, linestyle="--", alpha=0.9, label="t=0")

        # Shade pre-event region
        ax.axvspan(-PRE_WINDOW_SEC, 0, alpha=0.04, color=color)

        ax.set_title(f"{col}\n[{panel}]", fontsize=8, pad=3)
        ax.set_xlabel("Time relative to takeover (s)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_xlim(-PRE_WINDOW_SEC, POST_WINDOW_SEC)

    # Hide unused axes
    total_cells = nrows * ncols
    for unused_idx in range(len(to_plot), total_cells):
        row_idx, col_idx = divmod(unused_idx, ncols)
        fig.add_subplot(gs[row_idx, col_idx]).set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-event pre-takeover time-series plots.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--quality", type=Path, default=None,
                   help="post_event_quality_metrics.csv (optional, adds quality label)")
    p.add_argument("--output-dir", type=Path, default=Path("reports/pre_event_timeseries"))
    p.add_argument("--max-events", type=int, default=None)
    return p.parse_args()


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold"})

    args = parse_args()
    dataset_root: Path = args.dataset_root
    events_path: Path = args.events or dataset_root / "benchmark" / "takeover_events_or.csv"
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path)
    if "event_type" in events.columns:
        events = events[events["event_type"].str.lower() == "takeover"]

    # Resolve event time column
    for col in ("event_time_sec", "t_event", "nearest_event_time"):
        if col in events.columns:
            events["t_event"] = pd.to_numeric(events[col], errors="coerce")
            break

    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    if args.max_events:
        events = events.head(args.max_events)

    # Load quality metrics if provided
    quality_df: Optional[pd.DataFrame] = None
    if args.quality and args.quality.exists():
        quality_df = pd.read_csv(args.quality).set_index("event_id")

    route_cache: Dict[str, pd.DataFrame] = {}
    plotted = 0
    skipped = 0

    for _, event in events.iterrows():
        route_id = str(event["route_id"])
        route_path = dataset_root / route_id

        if not route_path.exists():
            skipped += 1
            continue

        if route_id not in route_cache:
            try:
                route_cache[route_id] = load_route(route_path)
            except FileNotFoundError:
                skipped += 1
                continue

        route_df = route_cache[route_id]
        event_id = str(event["event_id"])

        quality_row = None
        if quality_df is not None and event_id in quality_df.index:
            quality_row = quality_df.loc[event_id]

        # Add takeover type if not already in event
        if "takeover_type" not in event.index or pd.isna(event.get("takeover_type")):
            if quality_row is not None and "takeover_type" in quality_row.index:
                event = event.copy()
                event["takeover_type"] = quality_row["takeover_type"]

        out_file = output_dir / f"{event_id}_pre_event_timeseries.png"
        try:
            plot_event(event, route_df, quality_row, out_file)
            plotted += 1
            print(f"  Saved: {out_file.name}")
        except Exception as exc:
            warnings.warn(f"Failed {event_id}: {exc}")
            skipped += 1

    print(f"\nDone. Plotted: {plotted}  |  Skipped: {skipped}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
