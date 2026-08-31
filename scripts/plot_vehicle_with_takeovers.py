#!/usr/bin/env python3
"""Plot speed, acceleration, steering angle, and brake over full route time,
with vertical dotted red lines marking every takeover event.

One figure per route.

Usage:
    python scripts/plot_vehicle_with_takeovers.py \
        --dataset-root dataset \
        --events dataset/benchmark/takeover_events_or.csv \
        --output-dir reports/vehicle_with_takeovers
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SIGNALS = [
    ("vEgo",           "Speed (m/s)",          "#2E86AB"),
    ("aEgo",           "Acceleration (m/s²)",  "#4F6D7A"),
    ("steeringAngleDeg", "Steering Angle (°)", "#7A9E7E"),
    ("brake",          "Brake",                "#E07A5F"),
]


def numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--output-dir", type=Path,
                   default=Path("reports/vehicle_with_takeovers"))
    return p.parse_args()


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
    })

    args = parse_args()
    dataset_root: Path = args.dataset_root
    events_path:  Path = args.events or dataset_root / "benchmark" / "takeover_events_or.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- load events ---
    events = pd.read_csv(events_path)
    if "event_type" in events.columns:
        events = events[events["event_type"].str.lower() == "takeover"]
    for col in ("event_time_sec", "t_event"):
        if col in events.columns:
            events["t_event"] = pd.to_numeric(events[col], errors="coerce")
            break
    events = events.dropna(subset=["t_event"])

    # group by route
    for route_id, route_events in events.groupby("route_id"):
        route_path = dataset_root / str(route_id)
        vd_path    = route_path / "vehicle_dynamics.csv"
        if not vd_path.exists():
            print(f"  Skipping {route_id}: no vehicle_dynamics.csv")
            continue

        vd = pd.read_csv(vd_path, low_memory=False)
        if "time_s" not in vd.columns:
            continue
        vd["time_s"] = numeric_series(vd["time_s"])
        vd = vd.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

        t = vd["time_s"]
        takeover_times = sorted(route_events["t_event"].dropna().tolist())

        fig, axes = plt.subplots(len(SIGNALS), 1,
                                 figsize=(18, 2.8 * len(SIGNALS)),
                                 sharex=True)

        for ax, (col, ylabel, color) in zip(axes, SIGNALS):
            if col in vd.columns:
                series = numeric_series(vd[col])
                ax.plot(t, series, color=color, linewidth=0.8, rasterized=True)
            else:
                ax.text(0.5, 0.5, f"{col} not available",
                        transform=ax.transAxes, ha="center", va="center",
                        color="grey", fontsize=9)

            # takeover lines
            for i, t_ev in enumerate(takeover_times):
                ax.axvline(t_ev, color="red", linewidth=1.2,
                           linestyle=":", alpha=0.85,
                           label="Takeover" if i == 0 else None)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, linewidth=0.4, alpha=0.5)

            if ax is axes[0]:
                ax.legend(loc="upper right", fontsize=8,
                          handlelength=1.2, framealpha=0.8)

        axes[-1].set_xlabel("Time (s)", fontsize=10)

        route_label = str(route_id).replace("/", " / ")
        n_ev = len(takeover_times)
        fig.suptitle(
            f"Route: {route_label}   ·   {n_ev} takeover event{'s' if n_ev != 1 else ''}"
            f"   ·   red dotted lines = takeover moments",
            fontsize=11, y=1.01,
        )

        safe_name = str(route_id).replace("/", "_")
        out = args.output_dir / f"{safe_name}_vehicle_signals.png"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"  Saved: {out.name}")

    print(f"\nDone. Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
