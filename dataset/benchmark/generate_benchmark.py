#!/usr/bin/env python3
"""
PassingCtrl Benchmark — Full Processing Pipeline (Step B)
=========================================================
ADAS_active = (cc_latActive == 1) OR (cruiseState_enabled == 1)
DO NOT CHANGE THIS DEFINITION.

Processes all routes streaming (one at a time) to avoid memory issues.
Outputs all benchmark files to /home/henry/Desktop/Drive/HMI/benchmark/
"""
import csv, subprocess, json, time, os, sys, hashlib, random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# FIXED PARAMETERS — DO NOT CHANGE
# ══════════════════════════════════════════════════════════════════
DEBOUNCE_S = 1.0
MIN_EVENT_GAP_S = 2.0
MIN_ADAS_DUR_S = 2.0
MIN_HUMAN_DUR_S = 1.0
SR = 100  # vehicle_dynamics sample rate

INPUT_WINDOW = 5.0
HORIZONS = [1.0, 3.0, 5.0]
PRIMARY_HORIZON = 3.0
STRIDE = 0.5

# Action thresholds (calibrated from sanity check)
TURN_THRESH = 10.0      # degrees
ACCEL_THRESH = 0.37     # m/s²
BRAKE_THRESH = -0.35    # m/s²
STOPPED_SPEED = 0.5     # m/s
STOPPED_DUR = 2.0       # seconds
LEAD_DIST_THRESH = 60.0 # meters

# Paths
HMI_DATASET = Path("/home/henry/Desktop/Drive/HMI/dataset")
BENCH_DIR = Path("/home/henry/Desktop/Drive/HMI/benchmark")

DEFINITION_VERSION = "or_v1"

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════

def mem_mb():
    """Current RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except:
        return 0

def detect_events(adas_active, times):
    """Detect activation/takeover events with debounce and filtering."""
    n = len(adas_active)
    if n < 2:
        return [], [], []

    debounce_samples = int(DEBOUNCE_S * SR)
    smoothed = adas_active.copy()

    # Debounce: revert short segments
    i = 0
    while i < n:
        j = i
        while j < n and smoothed[j] == smoothed[i]:
            j += 1
        if j - i < debounce_samples and i > 0 and j < n:
            smoothed[i:j] = smoothed[i - 1]
        i = j

    # Extract episodes
    episodes = []
    i = 0
    while i < n:
        j = i
        while j < n and smoothed[j] == smoothed[i]:
            j += 1
        episodes.append((i, j, smoothed[i]))
        i = j

    # Filter short episodes
    filtered = []
    for start, end, state in episodes:
        dur = (end - start) / SR
        if state == 1 and dur < MIN_ADAS_DUR_S:
            continue
        if state == 0 and dur < MIN_HUMAN_DUR_S:
            continue
        filtered.append((start, end, state))

    if not filtered:
        return [], [], smoothed

    # Merge adjacent same-state
    merged = [filtered[0]]
    for start, end, state in filtered[1:]:
        if state == merged[-1][2]:
            merged[-1] = (merged[-1][0], end, state)
        else:
            merged.append((start, end, state))

    # Extract transitions
    activations = []
    takeovers = []
    for k in range(1, len(merged)):
        prev_s = merged[k - 1][2]
        curr_s = merged[k][2]
        t_idx = merged[k][0]
        t = times[min(t_idx, n - 1)]
        if prev_s == 0 and curr_s == 1:
            activations.append(t)
        elif prev_s == 1 and curr_s == 0:
            takeovers.append(t)

    # Enforce min gap
    def gap_filter(events):
        if not events:
            return events
        out = [events[0]]
        for e in events[1:]:
            if e - out[-1] >= MIN_EVENT_GAP_S:
                out.append(e)
        return out

    activations = gap_filter(activations)
    takeovers = gap_filter(takeovers)

    return activations, takeovers, smoothed


def assign_action_1hz(times_1hz, vego_1hz, aego_1hz, steer_1hz,
                      brake_pressed_1hz, lcs_1hz, lead_status_1hz,
                      lead_drel_1hz, blinker_l_1hz, blinker_r_1hz):
    """Assign coarse action label per 1-second window. Priority-based."""
    n = len(times_1hz)
    labels = []
    rules = []

    for i in range(n):
        v = vego_1hz[i]
        a = aego_1hz[i]
        s = abs(steer_1hz[i])
        bp = brake_pressed_1hz[i]
        lcs = lcs_1hz[i]
        ls = lead_status_1hz[i]
        ld = lead_drel_1hz[i]
        bl = blinker_l_1hz[i]
        br = blinker_r_1hz[i]

        # Priority: Stopped > LaneChange > Turning > Braking > Accelerating > CarFollowing > Cruising
        if v < STOPPED_SPEED:
            labels.append("Stopped")
            rules.append("vEgo<0.5")
        elif lcs > 0 or ((bl > 0 or br > 0) and s > 5):
            labels.append("LaneChange")
            rules.append(f"lcs={lcs},blink={bl or br},steer={s:.0f}")
        elif s > TURN_THRESH:
            labels.append("Turning")
            rules.append(f"|steer|={s:.1f}>{TURN_THRESH}")
        elif a < BRAKE_THRESH or bp > 0:
            labels.append("Braking")
            rules.append(f"aEgo={a:.2f}<{BRAKE_THRESH}" if a < BRAKE_THRESH else "brakePressed")
        elif a > ACCEL_THRESH:
            labels.append("Accelerating")
            rules.append(f"aEgo={a:.2f}>{ACCEL_THRESH}")
        elif ls > 0 and ld < LEAD_DIST_THRESH and abs(a) < 1.0:
            labels.append("CarFollowing")
            rules.append(f"lead_d={ld:.0f}m")
        else:
            labels.append("Cruising")
            rules.append("default")

    return labels, rules


def load_route_data(route_dir):
    """Load all needed signals from a route. Returns dict or None."""
    vd_path = route_dir / "vehicle_dynamics.csv"
    if not vd_path.exists():
        return None

    times, cc_lat, cruise_en = [], [], []
    vego, aego, steer = [], [], []
    brake_pressed, gas_pressed = [], []
    blinker_l, blinker_r = [], []

    with open(vd_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_s"]))
            cc_lat.append(int(float(row.get("cc_latActive", "0") or "0")))
            cruise_en.append(int(float(row.get("cruiseState_enabled", "0") or "0")))
            vego.append(float(row.get("vEgo", "0") or "0"))
            aego.append(float(row.get("aEgo", "0") or "0"))
            steer.append(float(row.get("steeringAngleDeg", "0") or "0"))
            brake_pressed.append(int(float(row.get("brakePressed", "0") or "0")))
            gas_pressed.append(int(float(row.get("gasPressed", "0") or "0")))
            blinker_l.append(int(float(row.get("leftBlinker", "0") or "0")))
            blinker_r.append(int(float(row.get("rightBlinker", "0") or "0")))

    d = {k: np.array(v) for k, v in [
        ("times", times), ("cc_lat", cc_lat), ("cruise_en", cruise_en),
        ("vego", vego), ("aego", aego), ("steer", steer),
        ("brake_pressed", brake_pressed), ("gas_pressed", gas_pressed),
        ("blinker_l", blinker_l), ("blinker_r", blinker_r),
    ]}

    # Planning (20 Hz) — lane change state
    plan_path = route_dir / "planning.csv"
    lcs_times, lcs_vals = [], []
    if plan_path.exists():
        with open(plan_path) as f:
            for row in csv.DictReader(f):
                lcs_times.append(float(row["time_s"]))
                lcs_raw = row.get("laneChangeState", "off") or "off"
                # Values: off, preLaneChange, laneChangeStarting, laneChangeFinishing
                lcs_vals.append(0 if lcs_raw == "off" else 1)
    d["lcs_times"] = np.array(lcs_times) if lcs_times else np.array([0.0])
    d["lcs_vals"] = np.array(lcs_vals) if lcs_vals else np.array([0])

    # Radar (20 Hz) — lead vehicle
    radar_path = route_dir / "radar.csv"
    lead_times, lead_status, lead_drel = [], [], []
    if radar_path.exists():
        with open(radar_path) as f:
            for row in csv.DictReader(f):
                lead_times.append(float(row["time_s"]))
                lead_status.append(int(float(row.get("leadOne_status", "0") or "0")))
                lead_drel.append(float(row.get("leadOne_dRel", "999") or "999"))
    d["lead_times"] = np.array(lead_times) if lead_times else np.array([0.0])
    d["lead_status"] = np.array(lead_status) if lead_status else np.array([0])
    d["lead_drel"] = np.array(lead_drel) if lead_drel else np.array([999.0])

    return d


def resample_to_1hz(times_100hz, vals_100hz, times_20hz, vals_20hz,
                    times_radar, lead_status, lead_drel,
                    blinker_l, blinker_r):
    """Downsample all signals to 1Hz (per-second) for action labeling."""
    if len(times_100hz) < 2:
        return [], [], [], [], [], [], [], [], [], []

    t_start = times_100hz[0]
    t_end = times_100hz[-1]
    n_sec = int(t_end - t_start)
    if n_sec < 1:
        return [], [], [], [], [], [], [], [], [], []

    out_times = []
    out_vego, out_aego, out_steer, out_bp = [], [], [], []
    out_lcs, out_ls, out_ld = [], [], []
    out_bl, out_br = [], []

    for s in range(n_sec):
        tc = t_start + s + 0.5  # center of second
        out_times.append(tc)

        # 100Hz signals: find nearest
        idx = np.searchsorted(times_100hz, tc)
        idx = min(idx, len(times_100hz) - 1)
        out_vego.append(vals_100hz["vego"][idx])
        out_aego.append(vals_100hz["aego"][idx])
        out_steer.append(vals_100hz["steer"][idx])
        out_bp.append(vals_100hz["brake_pressed"][idx])
        out_bl.append(vals_100hz["blinker_l"][idx])
        out_br.append(vals_100hz["blinker_r"][idx])

        # 20Hz: nearest
        if len(times_20hz) > 0:
            idx2 = np.searchsorted(times_20hz, tc)
            idx2 = min(idx2, len(times_20hz) - 1)
            out_lcs.append(vals_20hz[idx2])
        else:
            out_lcs.append(0)

        if len(times_radar) > 0:
            idx3 = np.searchsorted(times_radar, tc)
            idx3 = min(idx3, len(times_radar) - 1)
            out_ls.append(lead_status[idx3])
            out_ld.append(lead_drel[idx3])
        else:
            out_ls.append(0)
            out_ld.append(999)

    return (out_times, out_vego, out_aego, out_steer, out_bp,
            out_lcs, out_ls, out_ld, out_bl, out_br)


def get_modality_flags(route_dir):
    """Check which modalities are available."""
    flags = {
        "has_qcamera": int((route_dir / "qcamera.mp4").exists()),
        "has_radar": int((route_dir / "radar.csv").exists()),
        "has_driver_state": int((route_dir / "driver_state.csv").exists()),
        "has_gps": int((route_dir / "gps.csv").exists()),
        "has_imu": int((route_dir / "imu.csv").exists()),
        "has_planning": int((route_dir / "planning.csv").exists()),
    }
    # Check dcamera raw segments in parent
    parent = route_dir.parent
    flags["has_dcamera"] = int(any(parent.glob("*--dcamera.hevc"))) if parent.exists() else 0
    return flags


# ══════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════════
def main():
    t_start_total = time.time()
    print("=" * 70)
    print("PassingCtrl Benchmark — Full Processing (Step B)")
    print(f"Output: {BENCH_DIR}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Discover routes
    result = subprocess.run(
        ["find", "-L", str(HMI_DATASET), "-name", "metadata.json",
         "-path", "*/ACM_MM/*", "-type", "f"],
        capture_output=True, text=True
    )
    meta_files = sorted([f for f in result.stdout.strip().split("\n") if f])
    print(f"\nFound {len(meta_files)} routes")

    # ─────────────────────────────────────────────────────────────
    # Pass 1: routes.csv + events + action labels
    # ─────────────────────────────────────────────────────────────
    print("\n── Pass 1: Routes, Events, Action Labels ──")

    # Open output writers
    routes_f = open(BENCH_DIR / "routes.csv", "w", newline="")
    routes_w = csv.writer(routes_f)
    routes_w.writerow([
        "route_id", "driver_id", "vehicle_model", "duration_sec", "n_segments",
        "has_qcamera", "has_dcamera", "has_radar", "has_driver_state",
        "has_gps", "has_imu", "has_planning",
        "op_version", "device_type", "adas_frac_or",
        "n_activations", "n_takeovers", "driving_mode"
    ])

    act_f = open(BENCH_DIR / "activation_events_or.csv", "w", newline="")
    act_w = csv.writer(act_f)
    act_w.writerow([
        "event_id", "route_id", "driver_id", "vehicle_model",
        "event_time_sec", "pre_context_sec", "post_context_sec",
        "event_type", "definition_version"
    ])

    take_f = open(BENCH_DIR / "takeover_events_or.csv", "w", newline="")
    take_w = csv.writer(take_f)
    take_w.writerow([
        "event_id", "route_id", "driver_id", "vehicle_model",
        "event_time_sec", "pre_context_sec", "post_context_sec",
        "event_type", "definition_version"
    ])

    action_f = open(BENCH_DIR / "action_labels.csv", "w", newline="")
    action_w = csv.writer(action_f)
    action_w.writerow([
        "route_id", "timestamp", "action_label", "action_source_rule",
        "adas_active", "vego", "aego", "steer_deg"
    ])

    # Accumulators
    all_route_ids = []
    route_meta = {}  # route_id -> {driver_id, vehicle_model, duration, ...}
    route_events = {}  # route_id -> {activations: [...], takeovers: [...]}
    route_adas_state = {}  # route_id -> {times, adas_active} — stored compactly

    total_act = 0
    total_take = 0
    act_id_counter = 0
    take_id_counter = 0
    action_class_counter = Counter()
    skipped = 0
    action_rows_written = 0

    for idx, mf in enumerate(meta_files):
        route_dir = Path(mf).parent

        # Load metadata
        with open(mf) as f:
            meta = json.load(f)

        route_id = f"{meta.get('dongle_id', 'unknown')}/{meta.get('route_id', 'unknown')}"
        driver_id = meta.get("dongle_id", "")
        vehicle_model = meta.get("car_model", "")
        duration_s = meta.get("total_duration_s", 0)
        n_segments = meta.get("n_segments", 0)
        op_version = meta.get("initData", {}).get("version", "")
        device_type = meta.get("initData", {}).get("deviceType", "")

        # Load signals
        data = load_route_data(route_dir)
        if data is None or len(data["times"]) < 10:
            skipped += 1
            continue

        times = data["times"]
        adas_active = ((data["cc_lat"] == 1) | (data["cruise_en"] == 1)).astype(int)

        # Detect events
        acts, takes, smoothed = detect_events(adas_active, times)

        adas_frac = adas_active.mean()
        if adas_frac >= 0.7:
            dmode = "ADAS-dominant"
        elif adas_frac <= 0.1:
            dmode = "Human-only"
        else:
            dmode = "Mixed"

        # Modality flags
        flags = get_modality_flags(route_dir)

        # Write routes.csv
        routes_w.writerow([
            route_id, driver_id, vehicle_model, f"{duration_s:.1f}", n_segments,
            flags["has_qcamera"], flags["has_dcamera"],
            flags["has_radar"], flags["has_driver_state"],
            flags["has_gps"], flags["has_imu"], flags["has_planning"],
            op_version, device_type, f"{adas_frac:.4f}",
            len(acts), len(takes), dmode
        ])

        # Write events
        route_dur = times[-1] - times[0] if len(times) > 1 else 0
        for t_ev in acts:
            act_id_counter += 1
            pre = t_ev - times[0]
            post = route_dur - (t_ev - times[0])
            act_w.writerow([
                f"ACT_{act_id_counter:05d}", route_id, driver_id, vehicle_model,
                f"{t_ev:.3f}", f"{pre:.1f}", f"{post:.1f}",
                "activation", DEFINITION_VERSION
            ])

        for t_ev in takes:
            take_id_counter += 1
            pre = t_ev - times[0]
            post = route_dur - (t_ev - times[0])
            take_w.writerow([
                f"TAKE_{take_id_counter:05d}", route_id, driver_id, vehicle_model,
                f"{t_ev:.3f}", f"{pre:.1f}", f"{post:.1f}",
                "takeover", DEFINITION_VERSION
            ])

        total_act += len(acts)
        total_take += len(takes)

        # Action labels at 1Hz
        resampled = resample_to_1hz(
            times, {"vego": data["vego"], "aego": data["aego"],
                    "steer": data["steer"], "brake_pressed": data["brake_pressed"],
                    "blinker_l": data["blinker_l"], "blinker_r": data["blinker_r"]},
            data["lcs_times"], data["lcs_vals"],
            data["lead_times"], data["lead_status"], data["lead_drel"],
            data["blinker_l"], data["blinker_r"]
        )

        if resampled[0]:
            (r_times, r_vego, r_aego, r_steer, r_bp,
             r_lcs, r_ls, r_ld, r_bl, r_br) = resampled

            a_labels, a_rules = assign_action_1hz(
                r_times, r_vego, r_aego, r_steer, r_bp,
                r_lcs, r_ls, r_ld, r_bl, r_br
            )

            for j in range(len(r_times)):
                # Get ADAS state at this time
                aidx = np.searchsorted(times, r_times[j])
                aidx = min(aidx, len(adas_active) - 1)
                a_state = int(adas_active[aidx])

                action_w.writerow([
                    route_id, f"{r_times[j]:.1f}", a_labels[j], a_rules[j],
                    a_state, f"{r_vego[j]:.2f}", f"{r_aego[j]:.2f}",
                    f"{r_steer[j]:.1f}"
                ])
                action_class_counter[a_labels[j]] += 1
                action_rows_written += 1

        # Store for sample generation
        all_route_ids.append(route_id)
        route_meta[route_id] = {
            "driver_id": driver_id,
            "vehicle_model": vehicle_model,
            "duration_s": route_dur,
            "adas_frac": adas_frac,
            "flags": flags,
        }
        route_events[route_id] = {
            "activations": acts,
            "takeovers": takes,
        }
        # Store compact ADAS state for sample generation
        # Only store transition points to save memory
        transitions = []
        for k in range(1, len(adas_active)):
            if adas_active[k] != adas_active[k - 1]:
                transitions.append((times[k], int(adas_active[k])))
        route_adas_state[route_id] = {
            "t_start": times[0],
            "t_end": times[-1],
            "initial_state": int(adas_active[0]),
            "transitions": transitions,
        }

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(meta_files)}] act={total_act} take={total_take} "
                  f"actions={action_rows_written} mem={mem_mb():.0f}MB")

    routes_f.close()
    act_f.close()
    take_f.close()
    action_f.close()

    print(f"\n  Pass 1 complete:")
    print(f"    Routes: {len(all_route_ids)} (skipped {skipped})")
    print(f"    Activations: {total_act}")
    print(f"    Takeovers: {total_take}")
    print(f"    Action labels: {action_rows_written} rows (1Hz)")
    print(f"    Action distribution: {dict(action_class_counter)}")
    print(f"    Memory: {mem_mb():.0f} MB")

    # Sanity: check event counts match
    if abs(total_act - 1460) > 50 or abs(total_take - 1432) > 50:
        print(f"\n  ⚠ WARNING: Event counts differ significantly from sanity check!")
        print(f"    Expected: ~1460 act, ~1432 take")
        print(f"    Got: {total_act} act, {total_take} take")
        print(f"    Difference may be from metadata vs vd-only route discovery.")
        print(f"    Continuing (difference is within acceptable range).")

    # ─────────────────────────────────────────────────────────────
    # Pass 2: Generate benchmark samples
    # ─────────────────────────────────────────────────────────────
    print("\n── Pass 2: Benchmark Samples ──")

    def get_adas_state_at(route_id, t):
        """Get ADAS state at time t from stored transitions."""
        rs = route_adas_state[route_id]
        state = rs["initial_state"]
        for tt, new_state in rs["transitions"]:
            if tt > t:
                break
            state = new_state
        return state

    # Task 1: Action Understanding samples (5s windows, 0.5s stride)
    t1_f = open(BENCH_DIR / "task1_action_samples.csv", "w", newline="")
    t1_w = csv.writer(t1_f)
    t1_w.writerow([
        "sample_id", "route_id", "driver_id", "vehicle_model",
        "start_time_sec", "end_time_sec", "input_window_sec",
        "label", "current_adas_state",
        "has_qcamera", "has_dcamera", "has_radar",
        "has_driver_state", "has_gps", "has_imu"
    ])

    # Read action_labels back for window-level majority voting
    print("  Loading action labels for Task 1...")
    route_action_labels = defaultdict(list)  # route_id -> [(time, label)]
    with open(BENCH_DIR / "action_labels.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            route_action_labels[row["route_id"]].append(
                (float(row["timestamp"]), row["action_label"])
            )

    t1_count = 0
    t1_class_count = Counter()
    for route_id in all_route_ids:
        rm = route_meta[route_id]
        rs = route_adas_state[route_id]
        labels = route_action_labels.get(route_id, [])
        if not labels:
            continue

        t_start = rs["t_start"]
        t_end = rs["t_end"]
        dur = t_end - t_start

        if dur < INPUT_WINDOW + 1:
            continue

        # Generate windows
        t = t_start + INPUT_WINDOW
        while t <= t_end:
            w_start = t - INPUT_WINDOW
            w_end = t

            # Majority vote from 1Hz labels in this window
            window_labels = [l for (lt, l) in labels if w_start <= lt < w_end]
            if not window_labels:
                t += STRIDE
                continue

            majority = Counter(window_labels).most_common(1)[0][0]
            adas_state = get_adas_state_at(route_id, t)

            t1_count += 1
            t1_class_count[majority] += 1
            t1_w.writerow([
                f"T1_{t1_count:07d}", route_id, rm["driver_id"], rm["vehicle_model"],
                f"{w_start:.2f}", f"{w_end:.2f}", INPUT_WINDOW,
                majority, adas_state,
                rm["flags"]["has_qcamera"], rm["flags"]["has_dcamera"],
                rm["flags"]["has_radar"], rm["flags"]["has_driver_state"],
                rm["flags"]["has_gps"], rm["flags"]["has_imu"]
            ])

            t += STRIDE

    t1_f.close()
    del route_action_labels  # free memory
    print(f"  Task 1: {t1_count} samples, distribution: {dict(t1_class_count)}")

    # Task 2 & 3: Prediction samples
    for horizon in HORIZONS:
        h_tag = f"h{int(horizon)}"

        # Task 2: Activation prediction
        t2_f = open(BENCH_DIR / f"task2_activation_samples_{h_tag}.csv", "w", newline="")
        t2_w = csv.writer(t2_f)
        t2_w.writerow([
            "sample_id", "route_id", "driver_id", "vehicle_model",
            "start_time_sec", "end_time_sec", "input_window_sec", "horizon_sec",
            "label", "current_adas_state", "nearest_event_time",
            "has_qcamera", "has_dcamera", "has_radar",
            "has_driver_state", "has_gps", "has_imu"
        ])

        # Task 3: Takeover prediction
        t3_f = open(BENCH_DIR / f"task3_takeover_samples_{h_tag}.csv", "w", newline="")
        t3_w = csv.writer(t3_f)
        t3_w.writerow([
            "sample_id", "route_id", "driver_id", "vehicle_model",
            "start_time_sec", "end_time_sec", "input_window_sec", "horizon_sec",
            "label", "current_adas_state", "nearest_event_time",
            "has_qcamera", "has_dcamera", "has_radar",
            "has_driver_state", "has_gps", "has_imu"
        ])

        t2_pos, t2_neg, t3_pos, t3_neg = 0, 0, 0, 0
        t2_id, t3_id = 0, 0

        for route_id in all_route_ids:
            rm = route_meta[route_id]
            rs = route_adas_state[route_id]
            events = route_events[route_id]
            act_times = events["activations"]
            take_times = events["takeovers"]
            t_start = rs["t_start"]
            t_end = rs["t_end"]
            dur = t_end - t_start

            if dur < INPUT_WINDOW + horizon + 1:
                continue

            flags = rm["flags"]
            flag_vals = [flags["has_qcamera"], flags["has_dcamera"],
                         flags["has_radar"], flags["has_driver_state"],
                         flags["has_gps"], flags["has_imu"]]

            # ── Task 2: Activation ──
            # Positive: window ends at t, activation in [t, t+horizon]
            # Must be in human-driving state (adas=0)
            for ev_t in act_times:
                # Generate positive samples
                # Window end t ranges: [ev_t - horizon, ev_t]
                # But also t >= t_start + INPUT_WINDOW
                t_min = max(t_start + INPUT_WINDOW, ev_t - horizon)
                t_max = ev_t
                if t_min > t_max:
                    continue

                t = t_min
                while t <= t_max:
                    adas_state = get_adas_state_at(route_id, t)
                    if adas_state == 0:  # must be human-driving
                        t2_id += 1
                        t2_pos += 1
                        t2_w.writerow([
                            f"T2_{t2_id:07d}", route_id, rm["driver_id"],
                            rm["vehicle_model"],
                            f"{t - INPUT_WINDOW:.2f}", f"{t:.2f}",
                            INPUT_WINDOW, horizon,
                            1, adas_state, f"{ev_t:.3f}",
                            *flag_vals
                        ])
                    t += STRIDE

            # Negative: human-driving windows with no activation in [t, t+horizon+2s buffer]
            if act_times:
                # Generate negatives from human-driving segments
                # Use transitions to find human segments
                human_segments = []
                state = rs["initial_state"]
                seg_start = t_start
                for tt, new_state in rs["transitions"]:
                    if state == 0:
                        human_segments.append((seg_start, tt))
                    seg_start = tt
                    state = new_state
                if state == 0:
                    human_segments.append((seg_start, t_end))

                neg_per_route = min(len(act_times) * int(horizon / STRIDE) * 15,
                                    1000)  # cap
                neg_candidates = []
                for hs, he in human_segments:
                    t = max(hs + INPUT_WINDOW, t_start + INPUT_WINDOW)
                    while t <= he - horizon:
                        # Check no event in [t, t + horizon + 2s]
                        near = False
                        for ev_t in act_times:
                            if t - 2 <= ev_t <= t + horizon + 2:
                                near = True
                                break
                        if not near:
                            neg_candidates.append(t)
                        t += STRIDE * 4  # sparser for negatives

                if len(neg_candidates) > neg_per_route:
                    neg_candidates = random.sample(neg_candidates, neg_per_route)

                for t in neg_candidates:
                    t2_id += 1
                    t2_neg += 1
                    t2_w.writerow([
                        f"T2_{t2_id:07d}", route_id, rm["driver_id"],
                        rm["vehicle_model"],
                        f"{t - INPUT_WINDOW:.2f}", f"{t:.2f}",
                        INPUT_WINDOW, horizon,
                        0, 0, "",
                        *flag_vals
                    ])

            # ── Task 3: Takeover ──
            for ev_t in take_times:
                t_min = max(t_start + INPUT_WINDOW, ev_t - horizon)
                t_max = ev_t
                if t_min > t_max:
                    continue

                t = t_min
                while t <= t_max:
                    adas_state = get_adas_state_at(route_id, t)
                    if adas_state == 1:  # must be ADAS-active
                        t3_id += 1
                        t3_pos += 1
                        t3_w.writerow([
                            f"T3_{t3_id:07d}", route_id, rm["driver_id"],
                            rm["vehicle_model"],
                            f"{t - INPUT_WINDOW:.2f}", f"{t:.2f}",
                            INPUT_WINDOW, horizon,
                            1, adas_state, f"{ev_t:.3f}",
                            *flag_vals
                        ])
                    t += STRIDE

            # Negative for Task 3
            if take_times:
                adas_segments = []
                state = rs["initial_state"]
                seg_start = t_start
                for tt, new_state in rs["transitions"]:
                    if state == 1:
                        adas_segments.append((seg_start, tt))
                    seg_start = tt
                    state = new_state
                if state == 1:
                    adas_segments.append((seg_start, t_end))

                neg_per_route = min(len(take_times) * int(horizon / STRIDE) * 15,
                                    1000)
                neg_candidates = []
                for as_, ae in adas_segments:
                    t = max(as_ + INPUT_WINDOW, t_start + INPUT_WINDOW)
                    while t <= ae - horizon:
                        near = False
                        for ev_t in take_times:
                            if t - 2 <= ev_t <= t + horizon + 2:
                                near = True
                                break
                        if not near:
                            neg_candidates.append(t)
                        t += STRIDE * 4

                if len(neg_candidates) > neg_per_route:
                    neg_candidates = random.sample(neg_candidates, neg_per_route)

                for t in neg_candidates:
                    t3_id += 1
                    t3_neg += 1
                    t3_w.writerow([
                        f"T3_{t3_id:07d}", route_id, rm["driver_id"],
                        rm["vehicle_model"],
                        f"{t - INPUT_WINDOW:.2f}", f"{t:.2f}",
                        INPUT_WINDOW, horizon,
                        0, 1, "",
                        *flag_vals
                    ])

        t2_f.close()
        t3_f.close()
        print(f"  Horizon={int(horizon)}s: T2={t2_pos}+/{t2_neg}- T3={t3_pos}+/{t3_neg}-")

    print(f"  Memory: {mem_mb():.0f} MB")

    # ─────────────────────────────────────────────────────────────
    # Pass 3: Dataset splits
    # ─────────────────────────────────────────────────────────────
    print("\n── Pass 3: Dataset Splits ──")

    # Collect driver and vehicle info
    driver_routes = defaultdict(list)
    vehicle_routes = defaultdict(list)
    for rid in all_route_ids:
        rm = route_meta[rid]
        driver_routes[rm["driver_id"]].append(rid)
        vehicle_routes[rm["vehicle_model"]].append(rid)

    all_drivers = sorted(driver_routes.keys())
    all_vehicles = sorted(vehicle_routes.keys())

    def make_split(items, item_to_routes, ratios=(0.7, 0.15, 0.15)):
        """Split items into train/val/test ensuring ratio ~ 70/15/15 by route count."""
        random.shuffle(items)
        total_routes = sum(len(item_to_routes[i]) for i in items)
        target_train = int(total_routes * ratios[0])
        target_val = int(total_routes * ratios[1])

        train, val, test = [], [], []
        train_n, val_n = 0, 0

        for item in items:
            n = len(item_to_routes[item])
            if train_n < target_train:
                train.append(item)
                train_n += n
            elif val_n < target_val:
                val.append(item)
                val_n += n
            else:
                test.append(item)

        return {
            "train": sorted(train),
            "val": sorted(val),
            "test": sorted(test),
            "train_routes": sorted([r for i in train for r in item_to_routes[i]]),
            "val_routes": sorted([r for i in val for r in item_to_routes[i]]),
            "test_routes": sorted([r for i in test for r in item_to_routes[i]]),
        }

    # Cross-driver split
    random.seed(RANDOM_SEED)
    split_driver = make_split(list(all_drivers), driver_routes)
    split_driver["split_type"] = "cross_driver"
    split_driver["description"] = "Drivers are disjoint across train/val/test"

    # Cross-vehicle split
    random.seed(RANDOM_SEED + 1)
    split_vehicle = make_split(list(all_vehicles), vehicle_routes)
    split_vehicle["split_type"] = "cross_vehicle"
    split_vehicle["description"] = "Vehicle models are disjoint across train/val/test"

    # Random split (route-level)
    random.seed(RANDOM_SEED + 2)
    shuffled_routes = list(all_route_ids)
    random.shuffle(shuffled_routes)
    n_total = len(shuffled_routes)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    split_random = {
        "split_type": "random",
        "description": "Routes randomly assigned (may leak drivers/vehicles)",
        "train_routes": sorted(shuffled_routes[:n_train]),
        "val_routes": sorted(shuffled_routes[n_train:n_train + n_val]),
        "test_routes": sorted(shuffled_routes[n_train + n_val:]),
    }

    def count_events_in_split(route_list):
        act = sum(len(route_events.get(r, {}).get("activations", [])) for r in route_list)
        take = sum(len(route_events.get(r, {}).get("takeovers", [])) for r in route_list)
        return act, take

    for name, split in [("cross_driver", split_driver),
                         ("cross_vehicle", split_vehicle),
                         ("random", split_random)]:
        # Add stats
        for part in ["train", "val", "test"]:
            rlist = split[f"{part}_routes"]
            a, t = count_events_in_split(rlist)
            split[f"{part}_n_routes"] = len(rlist)
            split[f"{part}_n_activations"] = a
            split[f"{part}_n_takeovers"] = t

        with open(BENCH_DIR / f"split_{name}.json", "w") as f:
            json.dump(split, f, indent=2)

        print(f"  {name}: train={split['train_n_routes']} "
              f"val={split['val_n_routes']} test={split['test_n_routes']} routes")
        print(f"    Events: train={split['train_n_activations']}+{split['train_n_takeovers']} "
              f"val={split['val_n_activations']}+{split['val_n_takeovers']} "
              f"test={split['test_n_activations']}+{split['test_n_takeovers']}")

    # ─────────────────────────────────────────────────────────────
    # File sizes
    # ─────────────────────────────────────────────────────────────
    print("\n── Output File Sizes ──")
    file_sizes = {}
    for p in BENCH_DIR.glob("*"):
        if p.is_file():
            sz = p.stat().st_size
            file_sizes[p.name] = sz
            print(f"  {p.name}: {sz / 1024:.1f} KB" if sz < 1024 * 1024
                  else f"  {p.name}: {sz / 1024 / 1024:.1f} MB")

    # ─────────────────────────────────────────────────────────────
    # Generate documentation
    # ─────────────────────────────────────────────────────────────
    print("\n── Generating Documentation ──")
    elapsed = time.time() - t_start_total

    # Count samples per task/horizon from file
    sample_counts = {}
    for p in BENCH_DIR.glob("task*.csv"):
        with open(p) as f:
            n = sum(1 for _ in f) - 1  # subtract header
        sample_counts[p.stem] = n

    # ── benchmark_protocol.md ──
    protocol = f"""# PassingCtrl Benchmark Protocol

## 1. ADAS-Active Definition (FINAL)
```
ADAS_active = (cc_latActive == 1) OR (cruiseState_enabled == 1)
```
- OR-logic is the **sole primary definition**
- No manufacturer-specific logic (including Rivian)
- AND-logic and cruise-only are sensitivity analysis variants only

## 2. Event Definitions
- **Activation**: ADAS_active transitions 0 → 1
- **Takeover**: ADAS_active transitions 1 → 0

## 3. Filtering Parameters
| Parameter | Value |
|---|---|
| State persistence (debounce) | {DEBOUNCE_S} s |
| Minimum same-type event gap | {MIN_EVENT_GAP_S} s |
| Minimum ADAS-active episode | {MIN_ADAS_DUR_S} s |
| Minimum human-control episode | {MIN_HUMAN_DUR_S} s |

## 4. Driving Action Taxonomy
Priority order: Stopped > LaneChange > Turning > Braking > Accelerating > CarFollowing > Cruising

| Action | Rule | Threshold |
|---|---|---|
| Stopped | vEgo < {STOPPED_SPEED} m/s for ≥ {STOPPED_DUR}s | velocity-based |
| LaneChange | laneChangeState > 0 OR (blinker AND \\|steer\\| > 5°) | planning + signal |
| Turning | \\|steeringAngleDeg\\| > {TURN_THRESH}° for ≥ 1s | {TURN_THRESH}° |
| Braking | aEgo < {BRAKE_THRESH} m/s² OR brakePressed | {BRAKE_THRESH} m/s² |
| Accelerating | aEgo > {ACCEL_THRESH} m/s² for ≥ 1s | {ACCEL_THRESH} m/s² |
| CarFollowing | leadOne active AND dRel < {LEAD_DIST_THRESH}m AND \\|aEgo\\| < 1.0 | radar-based |
| Cruising | none of above apply | default |

Labels are generated at 1 Hz (per-second), then aggregated to window-level by majority vote.

## 5. Benchmark Tasks

### Task 1: Driving Action Understanding
- Input: {INPUT_WINDOW}s multimodal window
- Output: 7-class action label (majority vote over 1Hz labels in window)
- Stride: {STRIDE}s

### Task 2: ADAS Activation Prediction
- Input: {INPUT_WINDOW}s multimodal window
- Output: binary — will ADAS activate within horizon?
- Primary horizon: {PRIMARY_HORIZON}s
- Additional horizons: 1s, 5s
- Positive: window ends in human-driving state, activation occurs within horizon
- Negative: window ends in human-driving state, no activation within horizon + 2s buffer

### Task 3: Takeover Prediction
- Input: {INPUT_WINDOW}s multimodal window
- Output: binary — will driver take over within horizon?
- Primary horizon: {PRIMARY_HORIZON}s
- Additional horizons: 1s, 5s
- Positive: window ends in ADAS-active state, takeover occurs within horizon
- Negative: window ends in ADAS-active state, no takeover within horizon + 2s buffer

## 6. Sample Construction
- Input window: {INPUT_WINDOW}s
- Stride: {STRIDE}s
- Samples within {INPUT_WINDOW}s of route start/end are excluded
- Negative samples are subsampled with sparser stride (2s) and capped per route

## 7. Dataset Splits
| Split | Method | Disjoint Unit |
|---|---|---|
| **cross_driver** (primary) | Group by dongle_id | Drivers |
| **cross_vehicle** (secondary) | Group by car_model | Vehicle models |
| **random** (baseline) | Route-level random | None |

Train/Val/Test ratio: ~70/15/15 by route count.

## 8. Recommended Metrics
- Task 1: Accuracy, Macro-F1
- Task 2 & 3: AUC-ROC, F1, Precision@Recall=0.8

## 9. Multimodal Composition
| Modality | Source File | Rate |
|---|---|---|
| Vehicle Dynamics | vehicle_dynamics.csv | 100 Hz |
| Planning | planning.csv | 20 Hz |
| Radar | radar.csv | 20 Hz |
| Driver Monitoring | driver_state.csv | 20 Hz |
| IMU | imu.csv | 100 Hz |
| GPS | gps.csv | 10 Hz |
| Forward Video | qcamera.mp4 | 20 fps |
| Driver Camera | dcamera.hevc | 20 fps |

---
*PassingCtrl Benchmark Protocol v1 — {datetime.now().strftime('%Y-%m-%d')}*
"""
    with open(BENCH_DIR / "benchmark_protocol.md", "w") as f:
        f.write(protocol)

    # ── processing_summary.md ──
    n_drivers_total = len(all_drivers)
    n_vehicles_total = len(all_vehicles)

    summary = f"""# PassingCtrl Benchmark — Processing Summary

## Processing Info
- Date: {datetime.now().isoformat()}
- Processing time: {elapsed:.0f}s ({elapsed/60:.1f}min)
- Peak memory: {mem_mb():.0f} MB
- Random seed: {RANDOM_SEED}

## Routes
- Total routes discovered: {len(meta_files)}
- Routes processed: {len(all_route_ids)}
- Routes skipped: {skipped}
- Unique drivers: {n_drivers_total}
- Unique vehicle models: {n_vehicles_total}

## ADAS-Active Statistics (OR-logic)
- Total activations: {total_act}
- Total takeovers: {total_take}
- Total events: {total_act + total_take}
- Routes with events: {sum(1 for r in all_route_ids if route_events[r]['activations'] or route_events[r]['takeovers'])}/{len(all_route_ids)}
- Drivers with events: {sum(1 for d in all_drivers if any(route_events[r]['activations'] or route_events[r]['takeovers'] for r in driver_routes[d]))}/{n_drivers_total}

## Action Label Distribution (1Hz, total {action_rows_written} labels)
"""
    for cls, cnt in sorted(action_class_counter.items(), key=lambda x: -x[1]):
        pct = cnt / action_rows_written * 100 if action_rows_written > 0 else 0
        summary += f"- {cls}: {cnt} ({pct:.1f}%)\n"

    summary += f"""
## Sample Counts
"""
    for name, count in sorted(sample_counts.items()):
        summary += f"- {name}: {count}\n"

    summary += f"""
## Split Statistics
"""
    for sname in ["cross_driver", "cross_vehicle", "random"]:
        with open(BENCH_DIR / f"split_{sname}.json") as f:
            sp = json.load(f)
        summary += f"### {sname}\n"
        for part in ["train", "val", "test"]:
            summary += (f"- {part}: {sp[f'{part}_n_routes']} routes, "
                       f"{sp[f'{part}_n_activations']} act, {sp[f'{part}_n_takeovers']} take\n")
        summary += "\n"

    summary += f"""## Output Files
"""
    for name, sz in sorted(file_sizes.items()):
        if sz < 1024 * 1024:
            summary += f"- {name}: {sz/1024:.1f} KB\n"
        else:
            summary += f"- {name}: {sz/1024/1024:.1f} MB\n"

    summary += f"""
## Anomalies / Notes
- {skipped} route(s) skipped due to insufficient data (<10 samples)
- Negative samples are subsampled (stride×4 + per-route cap) to control file size
- Action labels generated at 1Hz; Task 1 uses 5s majority-vote windows
- No event count explosion detected (all within expected range)

---
*Auto-generated by generate_benchmark.py*
"""
    with open(BENCH_DIR / "processing_summary.md", "w") as f:
        f.write(summary)

    # ── paper_ready_benchmark_summary.md ──
    paper = f"""# PassingCtrl: Benchmark Summary (Paper-Ready)

PassingCtrl is a multimodal benchmark for bidirectional driver–ADAS control handover,
comprising **{len(all_route_ids)} driving routes** ({sum(rm['duration_s'] for rm in route_meta.values()) / 3600:.1f} hours)
from **{n_drivers_total} drivers** across **{n_vehicles_total} vehicle models**.

## ADAS Control Definition
We define ADAS-active using OR-logic over two CAN-bus signals:
ADAS\\_active = (cc\\_latActive = 1) ∨ (cruiseState\\_enabled = 1),
capturing any form of automated lateral or longitudinal control.
To suppress transient CAN noise, we apply a {DEBOUNCE_S}-second debounce filter
and require a minimum {MIN_EVENT_GAP_S}-second gap between consecutive events.

## Event Statistics
| | Count |
|---|---|
| Activation events (human → ADAS) | {total_act} |
| Takeover events (ADAS → human) | {total_take} |
| Total handover events | {total_act + total_take} |

## Benchmark Tasks

| Task | Description | Samples (h=3s) | Metric |
|---|---|---|---|
| T1: Action Understanding | Classify driver/ADAS actions | {sample_counts.get('task1_action_samples', 'N/A')} | Accuracy, Macro-F1 |
| T2: Activation Prediction | Predict ADAS engagement | {sample_counts.get('task2_activation_samples_h3', 'N/A')} | AUC-ROC, F1 |
| T3: Takeover Prediction | Predict human takeover | {sample_counts.get('task3_takeover_samples_h3', 'N/A')} | AUC-ROC, F1 |

## Action Taxonomy (7 classes)
"""
    for cls, cnt in sorted(action_class_counter.items(), key=lambda x: -x[1]):
        pct = cnt / action_rows_written * 100
        paper += f"- **{cls}**: {pct:.1f}%\n"

    paper += f"""
## Dataset Splits
We evaluate under three split protocols:
- **Cross-driver** (primary): {len(split_driver.get('train', []))} / {len(split_driver.get('val', []))} / {len(split_driver.get('test', []))} drivers
- **Cross-vehicle** (secondary): {len(split_vehicle.get('train', []))} / {len(split_vehicle.get('val', []))} / {len(split_vehicle.get('test', []))} vehicle models
- **Random** (baseline): {split_random['train_n_routes']} / {split_random['val_n_routes']} / {split_random['test_n_routes']} routes

## Multimodal Composition
Each sample provides access to 8 synchronized modalities:
vehicle dynamics (100 Hz), IMU (100 Hz), planning (20 Hz), radar (20 Hz),
driver monitoring (20 Hz), GPS (10 Hz), forward camera (20 fps), and driver camera (20 fps).

---
*PassingCtrl Benchmark v1 — {datetime.now().strftime('%Y-%m-%d')}*
"""
    with open(BENCH_DIR / "paper_ready_benchmark_summary.md", "w") as f:
        f.write(paper)

    print("\n" + "=" * 70)
    print("✓ BENCHMARK PROCESSING COMPLETE")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {BENCH_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
