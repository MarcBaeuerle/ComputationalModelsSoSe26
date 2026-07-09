#!/usr/bin/env python3
"""Exploratory data analysis for takeover quality after manual disengagements.

The script supports two common layouts:

1. A single, already-merged time-series CSV containing ``time_s`` and ``t_event``.
2. A BATON-style route folder layout plus an event table such as
   ``dataset/benchmark/takeover_events_or.csv``.

Example:
    python scripts/takeover_quality_eda.py \
        --dataset-root dataset \
        --events dataset/benchmark/takeover_events_or.csv \
        --output-dir reports/takeover_quality_eda

    python scripts/takeover_quality_eda.py \
        --input merged_takeover_timeseries.csv \
        --output-dir reports/takeover_quality_eda
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError as exc:
    missing = exc.name or "an EDA dependency"
    raise SystemExit(
        f"Missing Python package: {missing}. Install the EDA stack with "
        "`python -m pip install pandas numpy matplotlib seaborn` and rerun."
    ) from exc


PRE_WINDOW_SEC = 5.0
POST_WINDOW_SEC = 3.0
SUSTAINED_INPUT_SEC = 0.20
INPUT_MIXED_TOLERANCE_SEC = 0.50
STEERING_TORQUE_THRESHOLD = 1.0
STEERING_RATE_REVERSAL_THRESHOLD = 0.5

TAKEOVER_TYPE_ORDER = [
    "Steering",
    "Brake",
    "Gas",
    "Mixed",
    "System-initiated",
]

FEATURE_SUBSETS: Dict[str, List[str]] = {
    "A_driver_state": [
        "readyProb_0",
        "readyProb_1",
        "readyProb_2",
        "readyProb_3",
        "notReadyProb_0",
        "notReadyProb_1",
        "awarenessStatus",
        "isDistracted",
        "distractedType",
        "occludedProb",
        "poorVisionProb",
        "faceDetected",
    ],
    "B_vehicle_dynamics": [
        "vEgo",
        "aEgo",
        "vEgoRaw",
        "standstill",
        "steeringAngleDeg",
        "steeringTorque",
        "gas",
        "brake",
        "blinker_state",
    ],
    "C_radar": [
        "leadOne_dRel",
        "leadOne_vRel",
        "leadOne_aRel",
        "leadOne_yRel",
        "leadTwo_dRel",
        "leadTwo_vRel",
        "leadTwo_aRel",
        "leadTwo_yRel",
    ],
    "D_planner": [
        "model_desiredCurvature",
        "model_desiredAcceleration",
        "laneLeft_prob",
        "laneRight_prob",
        "laneLeft_y",
        "laneRight_y",
        "hasLead",
        "shouldStop",
        "fcw",
    ],
    "E_imu": [
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ],
}

CONTINUOUS_FEATURES: Dict[str, List[str]] = {
    "B_vehicle_dynamics": [
        "vEgo",
        "aEgo",
        "vEgoRaw",
        "steeringAngleDeg",
        "steeringTorque",
        "gas",
        "brake",
    ],
    "C_radar": FEATURE_SUBSETS["C_radar"],
    "D_planner": [
        "model_desiredCurvature",
        "model_desiredAcceleration",
        "laneLeft_prob",
        "laneRight_prob",
        "laneLeft_y",
        "laneRight_y",
    ],
    "E_imu": FEATURE_SUBSETS["E_imu"],
}

QUALITY_SCORE_COLUMNS = [
    "reaction_smoothness_score",
    "steering_smoothness_score",
    "longitudinal_smoothness_score",
    "lateral_stability_score",
]

ROUTE_MODALITY_FILES = {
    "vehicle_dynamics": "vehicle_dynamics.csv",
    "planning": "planning.csv",
    "radar": "radar.csv",
    "driver_state": "driver_state.csv",
    "imu": "imu.csv",
}

ASOF_TOLERANCE_SECONDS = {
    "planning": 0.030,
    "radar": 0.030,
    "driver_state": 0.030,
    "imu": 0.015,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run takeover quality EDA around manual disengagement events."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Single merged time-series CSV with time_s and t_event columns.",
    )
    source.add_argument(
        "--dataset-root",
        type=Path,
        help="Root directory containing BATON-style route folders.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="Event table for --dataset-root mode. Defaults to benchmark takeover events.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/takeover_quality_eda"),
        help="Directory where figures and summary tables will be written.",
    )
    parser.add_argument("--time-col", default="time_s", help="Time column in seconds.")
    parser.add_argument(
        "--event-time-col",
        default="t_event",
        help="Event timestamp column. event_time_sec is accepted as a fallback.",
    )
    parser.add_argument(
        "--event-id-col",
        default="event_id",
        help="Event identifier column. Created when absent.",
    )
    parser.add_argument("--route-id-col", default="route_id", help="Route identifier column.")
    parser.add_argument(
        "--pre-window-sec",
        type=float,
        default=PRE_WINDOW_SEC,
        help="Feature window length before t_event.",
    )
    parser.add_argument(
        "--post-window-sec",
        type=float,
        default=POST_WINDOW_SEC,
        help="Target quality window length after t_event.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap for quick iteration.",
    )
    parser.add_argument(
        "--top-correlations",
        type=int,
        default=15,
        help="Number of feature-target correlations to show per target.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def set_plot_style() -> None:
    """Apply a consistent seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.frameon": True,
        }
    )


def numeric_series(values: pd.Series) -> pd.Series:
    """Return a numeric Series with non-numeric entries coerced to NaN."""
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    """Return columns that exist in a DataFrame, preserving order."""
    return [col for col in columns if col in df.columns]


def all_feature_columns() -> List[str]:
    """Return all configured feature columns in subset order."""
    columns: List[str] = []
    for subset_columns in FEATURE_SUBSETS.values():
        columns.extend(subset_columns)
    return columns


def ensure_blinker_state(df: pd.DataFrame) -> pd.DataFrame:
    """Create a categorical blinker_state column when left/right blinker columns exist."""
    if "blinker_state" in df.columns:
        return df
    if not {"leftBlinker", "rightBlinker"}.issubset(df.columns):
        return df

    left = numeric_series(df["leftBlinker"]).fillna(0).astype(bool)
    right = numeric_series(df["rightBlinker"]).fillna(0).astype(bool)
    state = np.full(len(df), "none", dtype=object)
    state[left & ~right] = "left"
    state[right & ~left] = "right"
    state[left & right] = "hazard"
    df = df.copy()
    df["blinker_state"] = state
    return df


def save_figure(fig: plt.Figure, output_path: Path, show: bool = False) -> None:
    """Save and optionally show a matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def annotate_empty_axis(ax: plt.Axes, message: str = "No data available") -> None:
    """Write a centered no-data note on an axis."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])


def prepare_modality_frame(path: Path, time_col: str) -> Optional[pd.DataFrame]:
    """Read and sort one modality CSV, returning None if it is unusable."""
    if not path.exists():
        return None
    frame = pd.read_csv(path, low_memory=False)
    if time_col not in frame.columns:
        warnings.warn(f"Skipping {path}: missing time column {time_col!r}.")
        return None
    frame = frame.copy()
    frame[time_col] = numeric_series(frame[time_col])
    frame = frame.dropna(subset=[time_col]).sort_values(time_col)
    frame = frame.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)
    return frame


def load_route_modalities(route_path: Path, time_col: str = "time_s") -> pd.DataFrame:
    """Load and as-of merge route-level sensor modalities onto a vehicle-dynamics grid."""
    frames: Dict[str, pd.DataFrame] = {}
    for name, file_name in ROUTE_MODALITY_FILES.items():
        frame = prepare_modality_frame(route_path / file_name, time_col)
        if frame is not None and not frame.empty:
            frames[name] = frame

    if not frames:
        raise FileNotFoundError(f"No usable modality CSV files found in {route_path}")

    base_name = "vehicle_dynamics" if "vehicle_dynamics" in frames else next(iter(frames))
    merged = frames[base_name].copy()

    for name, frame in frames.items():
        if name == base_name:
            continue
        overlapping = sorted(set(merged.columns).intersection(frame.columns) - {time_col})
        if overlapping:
            frame = frame.drop(columns=overlapping)
        tolerance = ASOF_TOLERANCE_SECONDS.get(name)
        merged = pd.merge_asof(
            merged.sort_values(time_col),
            frame.sort_values(time_col),
            on=time_col,
            direction="nearest",
            tolerance=tolerance,
        )

    return ensure_blinker_state(merged)


def standardize_event_table(
    events: pd.DataFrame,
    event_time_col: str,
    event_id_col: str,
    route_id_col: str,
) -> pd.DataFrame:
    """Normalize event table naming and create event IDs if needed."""
    events = events.copy()
    resolved_event_time_col = event_time_col
    if resolved_event_time_col not in events.columns and "event_time_sec" in events.columns:
        resolved_event_time_col = "event_time_sec"
    if resolved_event_time_col not in events.columns and "nearest_event_time" in events.columns:
        resolved_event_time_col = "nearest_event_time"
    if resolved_event_time_col not in events.columns:
        raise ValueError(
            f"Could not find event time column {event_time_col!r}, "
            "'event_time_sec', or 'nearest_event_time'."
        )

    events["t_event"] = numeric_series(events[resolved_event_time_col])
    events = events.dropna(subset=["t_event"]).reset_index(drop=True)

    if event_id_col in events.columns:
        events["event_id"] = events[event_id_col].astype(str)
    elif route_id_col in events.columns:
        route_values = events[route_id_col].astype(str)
        time_values = events["t_event"].map(lambda value: f"{value:.6f}")
        events["event_id"] = route_values + "__t_" + time_values
    else:
        events["event_id"] = "t_" + events["t_event"].map(lambda value: f"{value:.6f}")

    return events


def add_event_metadata(
    window: pd.DataFrame,
    event: pd.Series,
    takeover_type: str,
    time_col: str,
) -> pd.DataFrame:
    """Attach event metadata and relative time to a pre/post window."""
    window = ensure_blinker_state(window.copy())
    window["event_id"] = event["event_id"]
    window["t_event"] = float(event["t_event"])
    window["relative_time_s"] = numeric_series(window[time_col]) - float(event["t_event"])
    window["takeover_type"] = takeover_type

    for col in ["route_id", "driver_id", "vehicle_model", "event_type"]:
        if col in event.index:
            window[col] = event[col]
    return window


def build_windows_from_routes(
    dataset_root: Path,
    events_path: Path,
    time_col: str,
    event_time_col: str,
    event_id_col: str,
    route_id_col: str,
    pre_window_sec: float,
    post_window_sec: float,
    max_events: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build event windows from route folders and an event table."""
    events = pd.read_csv(events_path, low_memory=False)
    events = standardize_event_table(events, event_time_col, event_id_col, route_id_col)
    if "event_type" in events.columns:
        events = events[events["event_type"].astype(str).str.lower().eq("takeover")]
    if max_events is not None:
        events = events.head(max_events)

    route_cache: Dict[str, pd.DataFrame] = {}
    pre_windows: List[pd.DataFrame] = []
    post_windows: List[pd.DataFrame] = []
    skipped_missing_route = 0
    skipped_empty_window = 0

    if route_id_col not in events.columns:
        raise ValueError(f"Route mode requires an event table column named {route_id_col!r}.")

    for _, event in events.iterrows():
        route_id = str(event[route_id_col])
        route_path = dataset_root / route_id
        if not route_path.exists():
            skipped_missing_route += 1
            continue

        if route_id not in route_cache:
            route_cache[route_id] = load_route_modalities(route_path, time_col=time_col)

        route_frame = route_cache[route_id]
        t_event = float(event["t_event"])
        time_values = numeric_series(route_frame[time_col])
        pre_mask = (time_values >= t_event - pre_window_sec) & (time_values <= t_event)
        post_mask = (time_values >= t_event) & (time_values <= t_event + post_window_sec)
        pre_window = route_frame.loc[pre_mask].copy()
        post_window = route_frame.loc[post_mask].copy()

        if pre_window.empty or post_window.empty:
            skipped_empty_window += 1
            continue

        takeover_type = (
            str(event["takeover_type"])
            if "takeover_type" in event.index and pd.notna(event["takeover_type"])
            else infer_takeover_type(
                post_window,
                time_col=time_col,
                t_event=t_event,
                post_window_sec=post_window_sec,
            )
        )
        pre_windows.append(add_event_metadata(pre_window, event, takeover_type, time_col))
        post_windows.append(add_event_metadata(post_window, event, takeover_type, time_col))

    if skipped_missing_route:
        warnings.warn(f"Skipped {skipped_missing_route} events with missing route folders.")
    if skipped_empty_window:
        warnings.warn(f"Skipped {skipped_empty_window} events with empty pre/post windows.")

    if not pre_windows or not post_windows:
        raise ValueError("No usable event windows were created.")
    return pd.concat(pre_windows, ignore_index=True), pd.concat(post_windows, ignore_index=True)


def build_windows_from_merged_csv(
    input_path: Path,
    time_col: str,
    event_time_col: str,
    event_id_col: str,
    route_id_col: str,
    pre_window_sec: float,
    post_window_sec: float,
    max_events: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build event windows from a single event-expanded CSV."""
    frame = pd.read_csv(input_path, low_memory=False)
    if time_col not in frame.columns:
        raise ValueError(f"Input CSV must contain a time column named {time_col!r}.")

    frame = standardize_event_table(frame, event_time_col, event_id_col, route_id_col)
    frame = ensure_blinker_state(frame)
    frame[time_col] = numeric_series(frame[time_col])
    frame = frame.dropna(subset=[time_col, "t_event"])

    if max_events is not None:
        keep_events = frame["event_id"].drop_duplicates().head(max_events)
        frame = frame[frame["event_id"].isin(keep_events)]

    pre_windows: List[pd.DataFrame] = []
    post_windows: List[pd.DataFrame] = []
    for event_id, event_frame in frame.groupby("event_id", sort=False):
        event_frame = event_frame.copy()
        t_event = float(event_frame["t_event"].iloc[0])
        time_values = numeric_series(event_frame[time_col])
        pre_window = event_frame[
            (time_values >= t_event - pre_window_sec) & (time_values <= t_event)
        ].copy()
        post_window = event_frame[
            (time_values >= t_event) & (time_values <= t_event + post_window_sec)
        ].copy()
        if pre_window.empty or post_window.empty:
            warnings.warn(f"Skipping {event_id}: empty pre/post window.")
            continue

        if "takeover_type" in event_frame.columns and event_frame["takeover_type"].notna().any():
            takeover_type = str(event_frame["takeover_type"].dropna().iloc[0])
        else:
            takeover_type = infer_takeover_type(
                post_window,
                time_col=time_col,
                t_event=t_event,
                post_window_sec=post_window_sec,
            )
            pre_window["takeover_type"] = takeover_type
            post_window["takeover_type"] = takeover_type

        pre_window["relative_time_s"] = numeric_series(pre_window[time_col]) - t_event
        post_window["relative_time_s"] = numeric_series(post_window[time_col]) - t_event
        pre_windows.append(pre_window)
        post_windows.append(post_window)

    if not pre_windows or not post_windows:
        raise ValueError("No usable event windows were created from the merged CSV.")
    return pd.concat(pre_windows, ignore_index=True), pd.concat(post_windows, ignore_index=True)


def input_mask(
    df: pd.DataFrame,
    input_kind: str,
    steering_torque_threshold: float = STEERING_TORQUE_THRESHOLD,
) -> pd.Series:
    """Return a boolean mask for one driver input type."""
    false_mask = pd.Series(False, index=df.index)
    if input_kind == "gas":
        if "gasPressed" in df.columns:
            return numeric_series(df["gasPressed"]).fillna(0).gt(0)
        if "gas" in df.columns:
            return numeric_series(df["gas"]).fillna(0).gt(0.01)
        return false_mask
    if input_kind == "brake":
        if "brakePressed" in df.columns:
            return numeric_series(df["brakePressed"]).fillna(0).gt(0)
        if "brake" in df.columns:
            return numeric_series(df["brake"]).fillna(0).gt(0.01)
        return false_mask
    if input_kind == "steering":
        steering_pressed = (
            numeric_series(df["steeringPressed"]).fillna(0).gt(0)
            if "steeringPressed" in df.columns
            else false_mask
        )
        torque_pressed = (
            numeric_series(df["steeringTorque"]).abs().fillna(0).gt(steering_torque_threshold)
            if "steeringTorque" in df.columns
            else false_mask
        )
        return steering_pressed | torque_pressed
    raise ValueError(f"Unknown input kind: {input_kind}")


def first_sustained_time(
    df: pd.DataFrame,
    mask: pd.Series,
    time_col: str,
    t_event: float,
    sustained_sec: float = SUSTAINED_INPUT_SEC,
) -> float:
    """Return seconds from event to first sustained True segment, or NaN if absent."""
    if df.empty or time_col not in df.columns:
        return np.nan

    work = pd.DataFrame({"time": numeric_series(df[time_col]), "active": mask.astype(bool)})
    work = work.dropna(subset=["time"]).sort_values("time")
    if work.empty or not work["active"].any():
        return np.nan

    times = work["time"].to_numpy(dtype=float)
    active = work["active"].to_numpy(dtype=bool)
    positive_diffs = np.diff(times)
    positive_diffs = positive_diffs[positive_diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0

    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends = np.flatnonzero(active & np.r_[~active[1:], True])
    for start_idx, end_idx in zip(starts, ends):
        duration = times[end_idx] - times[start_idx] + median_dt
        if duration >= sustained_sec:
            return max(0.0, float(times[start_idx] - t_event))
    return np.nan


def infer_takeover_type(
    post_window: pd.DataFrame,
    time_col: str,
    t_event: float,
    post_window_sec: float,
) -> str:
    """Infer takeover type from the first sustained post-event driver input."""
    first_inputs = {
        "Steering": first_sustained_time(
            post_window,
            input_mask(post_window, "steering"),
            time_col=time_col,
            t_event=t_event,
        ),
        "Brake": first_sustained_time(
            post_window,
            input_mask(post_window, "brake"),
            time_col=time_col,
            t_event=t_event,
        ),
        "Gas": first_sustained_time(
            post_window,
            input_mask(post_window, "gas"),
            time_col=time_col,
            t_event=t_event,
        ),
    }
    finite_inputs = {
        label: value
        for label, value in first_inputs.items()
        if pd.notna(value) and 0.0 <= value <= post_window_sec
    }
    if not finite_inputs:
        return "System-initiated"

    earliest = min(finite_inputs.values())
    simultaneous = [
        label
        for label, value in finite_inputs.items()
        if value - earliest <= INPUT_MIXED_TOLERANCE_SEC
    ]
    if len(simultaneous) > 1:
        return "Mixed"
    return simultaneous[0]


def finite_derivative(values: pd.Series, times: pd.Series) -> np.ndarray:
    """Compute finite first derivative d(values)/dt for irregular time samples."""
    work = pd.DataFrame({"value": numeric_series(values), "time": numeric_series(times)})
    work = work.dropna().sort_values("time").drop_duplicates("time")
    if len(work) < 2:
        return np.array([], dtype=float)
    delta_t = np.diff(work["time"].to_numpy(dtype=float))
    delta_v = np.diff(work["value"].to_numpy(dtype=float))
    valid = delta_t > 0
    if not np.any(valid):
        return np.array([], dtype=float)
    return delta_v[valid] / delta_t[valid]


def count_steering_reversals(
    steering_rate: np.ndarray,
    threshold: float = STEERING_RATE_REVERSAL_THRESHOLD,
) -> int:
    """Count sign changes in steering rate after removing near-zero rates."""
    finite_rate = steering_rate[np.isfinite(steering_rate)]
    finite_rate = finite_rate[np.abs(finite_rate) >= threshold]
    if len(finite_rate) < 2:
        return 0
    signs = np.sign(finite_rate)
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def first_nonempty_numeric(df: pd.DataFrame, columns: Sequence[str]) -> Optional[pd.Series]:
    """Return the first numeric column with at least two valid observations."""
    for col in columns:
        if col in df.columns:
            series = numeric_series(df[col])
            if series.notna().sum() >= 2:
                return series
    return None


def compute_quality_metrics(
    post_df: pd.DataFrame,
    time_col: str,
    post_window_sec: float,
) -> pd.DataFrame:
    """Engineer post-event takeover quality metrics for each event."""
    rows: List[Dict[str, object]] = []
    for event_id, group in post_df.groupby("event_id", sort=False):
        group = group.sort_values(time_col).copy()
        t_event = float(group["t_event"].iloc[0])
        takeover_type = (
            str(group["takeover_type"].dropna().iloc[0])
            if "takeover_type" in group.columns and group["takeover_type"].notna().any()
            else infer_takeover_type(group, time_col, t_event, post_window_sec)
        )

        reaction_time = first_sustained_time(
            group,
            input_mask(group, "gas")
            | input_mask(group, "brake")
            | input_mask(group, "steering"),
            time_col=time_col,
            t_event=t_event,
        )

        steering_rate = (
            finite_derivative(group["steeringAngleDeg"], group[time_col])
            if "steeringAngleDeg" in group.columns
            else np.array([], dtype=float)
        )
        steering_rate_std = (
            float(np.nanstd(steering_rate, ddof=1)) if len(steering_rate) > 1 else np.nan
        )
        steering_reversals = count_steering_reversals(steering_rate)

        acceleration = first_nonempty_numeric(group, ["aEgo", "accel_x"])
        peak_deceleration = float(np.nanmin(acceleration)) if acceleration is not None else np.nan
        acceleration_std = (
            float(np.nanstd(acceleration, ddof=1))
            if acceleration is not None and acceleration.notna().sum() > 1
            else np.nan
        )

        lane_variances = [
            float(np.nanvar(numeric_series(group[col]), ddof=1))
            for col in ["laneLeft_y", "laneRight_y"]
            if col in group.columns and numeric_series(group[col]).notna().sum() > 1
        ]
        lane_offset_variance = float(np.nanmean(lane_variances)) if lane_variances else np.nan
        if {"laneLeft_y", "laneRight_y"}.issubset(group.columns):
            lane_center = (
                numeric_series(group["laneLeft_y"]) + numeric_series(group["laneRight_y"])
            ) / 2.0
            lane_center_variance = (
                float(np.nanvar(lane_center, ddof=1)) if lane_center.notna().sum() > 1 else np.nan
            )
        else:
            lane_center_variance = np.nan

        yaw_rate_variance = (
            float(np.nanvar(numeric_series(group["gyro_z"]), ddof=1))
            if "gyro_z" in group.columns and numeric_series(group["gyro_z"]).notna().sum() > 1
            else np.nan
        )

        rows.append(
            {
                "event_id": event_id,
                "route_id": group["route_id"].iloc[0] if "route_id" in group.columns else np.nan,
                "driver_id": group["driver_id"].iloc[0] if "driver_id" in group.columns else np.nan,
                "vehicle_model": (
                    group["vehicle_model"].iloc[0] if "vehicle_model" in group.columns else np.nan
                ),
                "t_event": t_event,
                "takeover_type": takeover_type,
                "reaction_time_s": reaction_time,
                "steering_rate_std": steering_rate_std,
                "steering_reversal_count": steering_reversals,
                "peak_deceleration": peak_deceleration,
                "peak_deceleration_cost": max(0.0, -peak_deceleration)
                if pd.notna(peak_deceleration)
                else np.nan,
                "acceleration_std": acceleration_std,
                "lane_offset_variance": lane_offset_variance,
                "lane_center_variance": lane_center_variance,
                "yaw_rate_variance": yaw_rate_variance,
            }
        )

    return pd.DataFrame(rows)


def inverse_minmax_score(
    cost: pd.Series,
    best: float = 0.0,
    worst: Optional[float] = None,
) -> pd.Series:
    """Convert a lower-is-better cost into a [0, 1] score where 1 is best."""
    values = numeric_series(cost)
    if worst is None:
        finite_values = values[np.isfinite(values)]
        worst = float(finite_values.max()) if not finite_values.empty else best
    denom = float(worst - best)
    if not np.isfinite(denom) or denom <= 0:
        return values.where(values.isna(), 1.0)
    score = 1.0 - ((values - best) / denom)
    return score.clip(lower=0.0, upper=1.0)


def rowwise_mean(series_list: Sequence[pd.Series]) -> pd.Series:
    """Compute a row-wise mean across partially missing score components."""
    if not series_list:
        return pd.Series(dtype=float)
    return pd.concat(series_list, axis=1).mean(axis=1, skipna=True)


def normalize_quality_axes(metrics: pd.DataFrame, post_window_sec: float) -> pd.DataFrame:
    """Create four normalized quality axes where 1 indicates the smoothest/best case."""
    metrics = metrics.copy()

    reaction_cost = numeric_series(metrics["reaction_time_s"]).fillna(post_window_sec)
    metrics["reaction_smoothness_score"] = inverse_minmax_score(
        reaction_cost,
        best=0.0,
        worst=post_window_sec,
    )

    steering_components = [
        inverse_minmax_score(metrics["steering_rate_std"], best=0.0),
        inverse_minmax_score(metrics["steering_reversal_count"], best=0.0),
    ]
    metrics["steering_smoothness_score"] = rowwise_mean(steering_components)

    longitudinal_components = [
        inverse_minmax_score(metrics["peak_deceleration_cost"], best=0.0),
        inverse_minmax_score(metrics["acceleration_std"], best=0.0),
    ]
    metrics["longitudinal_smoothness_score"] = rowwise_mean(longitudinal_components)

    lateral_components = [
        inverse_minmax_score(metrics["lane_offset_variance"], best=0.0),
        inverse_minmax_score(metrics["yaw_rate_variance"], best=0.0),
    ]
    metrics["lateral_stability_score"] = rowwise_mean(lateral_components)

    return metrics


def missing_summary_table(pre_df: pd.DataFrame) -> pd.DataFrame:
    """Create a missingness table for all configured feature subsets."""
    rows: List[Dict[str, object]] = []
    n_rows = len(pre_df)
    for subset, columns in FEATURE_SUBSETS.items():
        for col in columns:
            if col in pre_df.columns:
                missing_count = int(pre_df[col].isna().sum())
                available = True
            else:
                missing_count = n_rows
                available = False
            rows.append(
                {
                    "subset": subset,
                    "feature": col,
                    "available": available,
                    "missing_count": missing_count,
                    "n_rows": n_rows,
                    "missing_pct": 100.0 * missing_count / max(n_rows, 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["subset", "missing_pct", "feature"])


def summarize_pre_window_features(pre_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre-event feature windows into event-level mean, max, and std features.

    NaNs are intentionally not globally imputed at the EDA stage. Pandas aggregation
    skips NaNs within each event; model-facing imputation should be selected after
    reviewing the missingness table saved by this script.
    """
    numeric_feature_cols = [
        col
        for col in existing_columns(pre_df, all_feature_columns())
        if col != "blinker_state" and pd.to_numeric(pre_df[col], errors="coerce").notna().any()
    ]
    if not numeric_feature_cols:
        return pd.DataFrame(index=pre_df["event_id"].drop_duplicates())

    work = pre_df[["event_id"] + numeric_feature_cols].copy()
    for col in numeric_feature_cols:
        work[col] = numeric_series(work[col])

    summary = work.groupby("event_id").agg(["mean", "max", "std"])
    summary.columns = [f"{feature}__{stat}" for feature, stat in summary.columns]
    return summary


def plot_reaction_distribution(metrics: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot distribution of time to first sustained driver input."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = numeric_series(metrics["reaction_time_s"]).dropna()
    if data.empty:
        annotate_empty_axis(ax, "No sustained driver inputs found")
    else:
        sns.histplot(data, kde=data.nunique() > 1, bins=24, ax=ax, color="#2E86AB")
        ax.set_xlim(xmin=0, xmax=0.3)
        ax.set_xlabel("Time to first sustained driver input (s)")
        ax.set_ylabel("Event count")
        ax.set_title("Reaction Smoothness")
    save_figure(fig, output_dir / "target_reaction_time_distribution.png", show)


def plot_steering_smoothness(metrics: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot steering-rate standard deviation against steering reversal counts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = metrics.dropna(subset=["steering_rate_std", "steering_reversal_count"]).copy()
    if data.empty:
        annotate_empty_axis(ax)
    else:
        sns.scatterplot(
            data=data,
            x="steering_rate_std",
            y="steering_reversal_count",
            hue="takeover_type",
            hue_order=TAKEOVER_TYPE_ORDER,
            ax=ax,
            alpha=0.8,
            s=60,
        )
        ax.set_xlabel("Std. of steering rate (deg/s)")
        ax.set_ylabel("Steering reversal count")
        ax.set_title("Steering Smoothness")
        sns.move_legend(ax, "best", title="Takeover type")
    save_figure(fig, output_dir / "target_steering_smoothness_scatter.png", show)


def plot_longitudinal_smoothness(metrics: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot peak deceleration and acceleration variability distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title, xlabel in [
        (axes[0], "peak_deceleration", "Peak Deceleration", "Minimum acceleration"),
        (axes[1], "acceleration_std", "Acceleration Variability", "Std. acceleration"),
    ]:
        data = numeric_series(metrics[col]).dropna()
        if data.empty:
            annotate_empty_axis(ax)
        else:
            sns.histplot(data, kde=data.nunique() > 1, bins=24, ax=ax, color="#4F6D7A")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Event count")
            ax.set_title(title)
    save_figure(fig, output_dir / "target_longitudinal_smoothness.png", show)


def plot_lateral_stability(metrics: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot lane-offset and yaw-rate variance distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title, xlabel in [
        (axes[0], "lane_offset_variance", "Lane Offset Variance", "Mean lane boundary variance"),
        (axes[1], "yaw_rate_variance", "Yaw Rate Variance", "Variance of gyro_z"),
    ]:
        data = numeric_series(metrics[col]).dropna()
        if data.empty:
            annotate_empty_axis(ax)
        else:
            sns.histplot(data, kde=data.nunique() > 1, bins=24, ax=ax, color="#7A9E7E")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Event count")
            ax.set_title(title)
    save_figure(fig, output_dir / "target_lateral_stability.png", show)


def plot_quality_boxplot(metrics: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot normalized quality scores as side-by-side boxplots by takeover type."""
    available_scores = existing_columns(metrics, QUALITY_SCORE_COLUMNS)
    long_scores = metrics.melt(
        id_vars=["event_id", "takeover_type"],
        value_vars=available_scores,
        var_name="quality_axis",
        value_name="score",
    ).dropna(subset=["score"])
    long_scores["quality_axis"] = (
        long_scores["quality_axis"]
        .str.replace("_score", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    fig, ax = plt.subplots(figsize=(13, 5.5))
    if long_scores.empty:
        annotate_empty_axis(ax)
    else:
        sns.boxplot(
            data=long_scores,
            x="quality_axis",
            y="score",
            hue="takeover_type",
            hue_order=TAKEOVER_TYPE_ORDER,
            ax=ax,
            showfliers=False,
        )
        sns.stripplot(
            data=long_scores,
            x="quality_axis",
            y="score",
            hue="takeover_type",
            hue_order=TAKEOVER_TYPE_ORDER,
            ax=ax,
            dodge=True,
            size=2,
            alpha=0.35,
            legend=False,
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("")
        ax.set_ylabel("Normalized score (1 = best)")
        ax.set_title("Normalized Takeover Quality Axes by Takeover Type")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), title="Takeover type")
    save_figure(fig, output_dir / "target_quality_scores_boxplot.png", show)


def plot_target_quality(metrics: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Create all post-event target quality plots."""
    plot_reaction_distribution(metrics, output_dir, show)
    plot_steering_smoothness(metrics, output_dir, show)
    plot_longitudinal_smoothness(metrics, output_dir, show)
    plot_lateral_stability(metrics, output_dir, show)
    plot_quality_boxplot(metrics, output_dir, show)


def plot_histogram_grid(
    df: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    output_path: Path,
    show: bool,
    ncols: int = 3,
) -> None:
    """Plot a grid of histograms with KDE overlays for continuous features."""
    columns = [
        col
        for col in existing_columns(df, columns)
        if numeric_series(df[col]).dropna().nunique() > 0
    ]
    n_plots = len(columns)
    nrows = max(1, math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    if not columns:
        annotate_empty_axis(axes_array[0])
    for ax, col in zip(axes_array, columns):
        data = numeric_series(df[col]).dropna()
        sns.histplot(data, kde=data.nunique() > 1, bins=30, ax=ax, color="#33658A")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for ax in axes_array[n_plots:]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.01)
    save_figure(fig, output_path, show)


def plot_feature_distributions(pre_df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot pre-event distributions for continuous feature subsets B through E."""
    for subset, columns in CONTINUOUS_FEATURES.items():
        plot_histogram_grid(
            pre_df,
            columns,
            title=f"Pre-Event Continuous Feature Distributions: {subset}",
            output_path=output_dir / f"pre_event_histograms_{subset}.png",
            show=show,
        )


def plot_driver_state_distributions(pre_df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Plot pre-event categorical/probabilistic driver-state distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    prob_cols = existing_columns(
        pre_df,
        [
            "readyProb_0",
            "readyProb_1",
            "readyProb_2",
            "readyProb_3",
            "notReadyProb_0",
            "notReadyProb_1",
            "occludedProb",
            "poorVisionProb",
        ],
    )
    if prob_cols:
        means = pre_df[prob_cols].apply(numeric_series).mean().sort_values(ascending=False)
        sns.barplot(x=means.values, y=means.index, ax=axes[0], color="#2E86AB")
        axes[0].set_xlabel("Mean probability")
        axes[0].set_ylabel("")
        axes[0].set_title("Mean Driver-State Probabilities")
    else:
        annotate_empty_axis(axes[0])

    if "awarenessStatus" in pre_df.columns:
        sns.countplot(
            data=pre_df,
            x="awarenessStatus",
            ax=axes[1],
            color="#4F6D7A",
            order=pre_df["awarenessStatus"].value_counts(dropna=False).index,
        )
        axes[1].set_title("Awareness Status")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Samples")
    else:
        annotate_empty_axis(axes[1])

    if "distractedType" in pre_df.columns:
        counts = pre_df["distractedType"].fillna("missing").value_counts().head(15)
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=axes[2], color="#7A9E7E")
        axes[2].set_title("Distraction Type")
        axes[2].set_xlabel("Samples")
        axes[2].set_ylabel("")
    else:
        annotate_empty_axis(axes[2])

    binary_cols = existing_columns(pre_df, ["isDistracted", "faceDetected"])
    if binary_cols:
        means = pre_df[binary_cols].apply(numeric_series).mean().sort_values(ascending=False)
        sns.barplot(x=means.index, y=means.values, ax=axes[3], color="#F2A65A")
        axes[3].set_ylim(0, 1)
        axes[3].set_title("Binary Driver-State Flags")
        axes[3].set_xlabel("")
        axes[3].set_ylabel("Mean flag value")
    else:
        annotate_empty_axis(axes[3])

    save_figure(fig, output_dir / "pre_event_driver_state_distributions.png", show)


def plot_feature_correlation_clustermap(
    pre_df: pd.DataFrame,
    output_dir: Path,
    show: bool,
) -> None:
    """Plot clustered Pearson correlations among pre-event features in subsets B-E."""
    feature_cols: List[str] = []
    for subset in ["B_vehicle_dynamics", "C_radar", "D_planner", "E_imu"]:
        feature_cols.extend(existing_columns(pre_df, FEATURE_SUBSETS[subset]))

    numeric_cols = [
        col for col in feature_cols if pd.to_numeric(pre_df[col], errors="coerce").notna().any()
    ]
    if len(numeric_cols) < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty_axis(ax, "Need at least two numeric features")
        save_figure(fig, output_dir / "feature_feature_correlation_clustermap.png", show)
        return

    data = pre_df[numeric_cols].apply(lambda col: numeric_series(col))
    data = data.loc[:, data.nunique(dropna=True) > 1]
    if data.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty_axis(ax, "Need at least two non-constant features")
        save_figure(fig, output_dir / "feature_feature_correlation_clustermap.png", show)
        return

    corr = data.corr(method="pearson", min_periods=10)
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    corr_for_plot = corr.fillna(0.0)

    try:
        grid = sns.clustermap(
            corr_for_plot,
            cmap="vlag",
            center=0,
            linewidths=0.05,
            figsize=(14, 14),
            cbar_kws={"label": "Pearson correlation"},
        )
        grid.fig.suptitle("Clustered Feature-to-Feature Correlations", y=1.02)
        grid.savefig(output_dir / "feature_feature_correlation_clustermap.png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(grid.fig)
    except Exception as exc:
        warnings.warn(f"sns.clustermap failed ({exc}); falling back to heatmap.")
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(corr_for_plot, cmap="vlag", center=0, ax=ax)
        ax.set_title("Feature-to-Feature Correlations")
        save_figure(fig, output_dir / "feature_feature_correlation_clustermap.png", show)


def compute_feature_target_correlations(
    pre_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    methods: Sequence[str] = ("pearson", "spearman"),
) -> pd.DataFrame:
    """Correlate event-level pre-window summaries with normalized target axes."""
    target_cols = existing_columns(metrics, QUALITY_SCORE_COLUMNS)
    if pre_summary.empty or not target_cols:
        return pd.DataFrame(columns=["method", "feature", "target", "correlation", "n"])

    joined = pre_summary.join(metrics.set_index("event_id")[target_cols], how="inner")
    rows: List[Dict[str, object]] = []
    feature_cols = [col for col in pre_summary.columns if col in joined.columns]

    for method in methods:
        for feature in feature_cols:
            for target in target_cols:
                pair = joined[[feature, target]].apply(lambda col: numeric_series(col)).dropna()
                if len(pair) < 3 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
                    corr = np.nan
                else:
                    corr = pair[feature].corr(pair[target], method=method)
                rows.append(
                    {
                        "method": method,
                        "feature": feature,
                        "target": target,
                        "correlation": corr,
                        "n": len(pair),
                    }
                )
    return pd.DataFrame(rows)


def plot_feature_target_correlations(
    correlations: pd.DataFrame,
    output_dir: Path,
    show: bool,
    top_n: int,
) -> None:
    """Plot top feature-target correlations for Pearson and Spearman."""
    if correlations.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty_axis(ax)
        save_figure(fig, output_dir / "feature_target_correlations.png", show)
        return

    for method, method_df in correlations.groupby("method", sort=False):
        targets = list(method_df["target"].drop_duplicates())
        n_targets = len(targets)
        ncols = 2
        nrows = max(1, math.ceil(n_targets / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows))
        axes_array = np.atleast_1d(axes).ravel()

        for ax, target in zip(axes_array, targets):
            target_df = method_df[method_df["target"].eq(target)].dropna(subset=["correlation"])
            target_df = target_df.assign(abs_corr=target_df["correlation"].abs())
            target_df = target_df.nlargest(top_n, "abs_corr").sort_values("correlation")
            if target_df.empty:
                annotate_empty_axis(ax)
                continue
            colors = np.where(target_df["correlation"].ge(0), "#B2182B", "#2166AC")
            ax.barh(target_df["feature"], target_df["correlation"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(f"{method.title()} correlation")
            ax.set_ylabel("")
            ax.set_title(target.replace("_score", "").replace("_", " ").title())

        for ax in axes_array[n_targets:]:
            ax.set_visible(False)
        fig.suptitle(f"Feature-to-Target Correlations ({method.title()})", y=1.01)
        save_figure(fig, output_dir / f"feature_target_correlations_{method}.png", show)


def save_nan_notes(output_dir: Path) -> None:
    """Save a short note documenting NaN handling used in the EDA."""
    notes = """NaN handling and imputation placeholders

- Missingness is measured directly in pre_event_missingness.csv.
- Histograms and target plots drop NaNs only for the specific variable being plotted.
- Event-level pre-window summaries use pandas mean/max/std, which skip NaNs within event windows.
- Feature-to-target correlations use pairwise complete observations.
- The clustered heatmap computes pairwise correlations, then fills unresolved correlation cells with 0 only for plotting/clustering.
- No global imputation is applied in this EDA script. Add model-specific imputation after inspecting missingness patterns.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nan_handling_notes.txt").write_text(notes)


def run_eda(args: argparse.Namespace) -> None:
    """Run the full EDA pipeline."""
    set_plot_style()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        pre_df, post_df = build_windows_from_merged_csv(
            input_path=args.input,
            time_col=args.time_col,
            event_time_col=args.event_time_col,
            event_id_col=args.event_id_col,
            route_id_col=args.route_id_col,
            pre_window_sec=args.pre_window_sec,
            post_window_sec=args.post_window_sec,
            max_events=args.max_events,
        )
    else:
        events_path = args.events or args.dataset_root / "benchmark" / "takeover_events_or.csv"
        pre_df, post_df = build_windows_from_routes(
            dataset_root=args.dataset_root,
            events_path=events_path,
            time_col=args.time_col,
            event_time_col=args.event_time_col,
            event_id_col=args.event_id_col,
            route_id_col=args.route_id_col,
            pre_window_sec=args.pre_window_sec,
            post_window_sec=args.post_window_sec,
            max_events=args.max_events,
        )

    missing_table = missing_summary_table(pre_df)
    missing_table.to_csv(output_dir / "pre_event_missingness.csv", index=False)

    metrics = compute_quality_metrics(post_df, time_col=args.time_col, post_window_sec=args.post_window_sec)
    metrics = normalize_quality_axes(metrics, post_window_sec=args.post_window_sec)
    metrics.to_csv(output_dir / "post_event_quality_metrics.csv", index=False)

    pre_summary = summarize_pre_window_features(pre_df)
    pre_summary.to_csv(output_dir / "pre_event_feature_summary.csv")

    correlations = compute_feature_target_correlations(pre_summary, metrics)
    correlations.to_csv(output_dir / "feature_target_correlations.csv", index=False)

    plot_target_quality(metrics, output_dir, args.show)
    plot_feature_distributions(pre_df, output_dir, args.show)
    plot_driver_state_distributions(pre_df, output_dir, args.show)
    plot_feature_correlation_clustermap(pre_df, output_dir, args.show)
    plot_feature_target_correlations(correlations, output_dir, args.show, args.top_correlations)
    save_nan_notes(output_dir)

    print(f"EDA complete. Events analyzed: {metrics['event_id'].nunique()}")
    print(f"Pre-event samples: {len(pre_df):,}; post-event samples: {len(post_df):,}")
    print(f"Outputs written to: {output_dir.resolve()}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_eda(args)


if __name__ == "__main__":
    main()
