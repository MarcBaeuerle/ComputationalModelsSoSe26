# Predicting Takeover Quality in Level-2 Driving: A Proof of Concept

This repository contains the analysis code and pilot artifacts for a paper on
predicting human takeover quality during Level 2 Advanced Driver Assistance
System (ADAS) disengagements. The paper introduction motivates the safety
problem: L2 systems can control both steering and speed, but the human driver
must remain ready to supervise and resume control. This project turns that
motivation into a proof-of-concept data pipeline for quantifying and predicting
how stable, smooth, and timely a takeover is.

The work uses a sample of the BATON dataset and focuses on the moments around
human takeover events. It combines pre-event multimodal signals from the driver,
vehicle, radar, planner, and IMU streams with post-event behavior-based quality
scores.

## Research Questions

- RQ1: How can takeover quality be formally defined and quantified using
  multimodal vehicle and driver data?
- RQ2: Which driver, vehicle, and environmental feature groups are most
  predictive of takeover quality?
- RQ3: Which pre-event observation interval best balances prediction accuracy
  and actionable lead time?

## Project Scope

This is a compact empirical baseline rather than a deployment-ready ADAS
component. The repository is designed to support the paper by making the
analysis pipeline inspectable and repeatable:

- define takeover quality from post-event behavior in a 3-second window;
- summarize multimodal pre-event context from a 5-second window;
- compare regression models under leave-one-out cross-validation;
- test feature subset importance and temporal sensitivity;
- relate driver readiness estimates to subsequent takeover quality;
- fit survey-based quality-axis weights against human ratings.

## Repository Layout

```text
.
+-- dataset/                  # BATON-style sample data and benchmark event files
|   +-- benchmark/             # takeover events, activation events, splits, labels
|   +-- driver_54/             # route-level synchronized sensor CSVs
|   +-- driver_97/
+-- scripts/                  # analysis, modeling, plotting, and survey scripts
|   +-- takeover_quality_eda.py
|   +-- ablation_study.py
|   +-- pre_event_depth_analysis.py
|   +-- ctm_vs_quality.py
|   +-- survey_quality_weighting.py
|   +-- plot_*.py
+-- takeover_quality_eda/      # checked-in EDA tables and figures
+-- Video+survey/              # takeover video stimuli and survey score CSV
+-- assets/                    # paper or report figures
```

Several scripts write new outputs under `reports/` by default. The current
checked-in EDA snapshot lives in `takeover_quality_eda/`.

## Data

Each route follows the BATON-style multimodal layout, with time-aligned CSV
streams such as:

- `vehicle_dynamics.csv`: speed, acceleration, steering, pedal, and control
  state signals;
- `driver_state.csv`: face pose, readiness probabilities, attention, and
  visibility estimates;
- `radar.csv`: lead-vehicle distance and relative motion;
- `planning.csv`: lane, curvature, lead, stop, and warning-related planner
  outputs;
- `imu.csv`: acceleration and gyroscope signals;
- `gps.csv` and `localization.csv`: route and road-context signals.

The main event table is:

```text
dataset/benchmark/takeover_events_or.csv
```

Dataset is available at: https://huggingface.co/datasets/HenryYHW/BATON-Sample

## Pipeline

1. Load takeover events and route-level multimodal streams.
2. Merge streams by nearest timestamp with modality-specific tolerances.
3. Extract a pre-event observation window, typically `[t_event - 5s, t_event)`.
4. Compute takeover quality from post-event behavior over
   `[t_event, t_event + 3s]`.
5. Train and evaluate predictive models using leave-one-out cross-validation.
6. Run ablations over model class, feature subset, quality definition, and
   temporal window length.



## Setup

Create a Python environment and install the analysis stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost torch rgf-python
```

`torch` is only required for the LSTM comparison. `rgf-python` is only required
for the Regularized Greedy Forest comparison. The remaining scripts can be run
without those optional model packages.

## Reproducing Analyses

Run the exploratory takeover-quality report:

```bash
python scripts/takeover_quality_eda.py \
  --dataset-root dataset \
  --events dataset/benchmark/takeover_events_or.csv \
  --output-dir reports/takeover_quality_eda
```

Run the model and feature ablation study:

```bash
python scripts/ablation_study.py
```

Run the pre-event temporal depth analysis:

```bash
python scripts/pre_event_depth_analysis.py
```

Compare the Continuous Readiness Metric with takeover quality:

```bash
python scripts/ctm_vs_quality.py
```

Fit survey-optimized quality-axis weights:

```bash
python scripts/survey_quality_weighting.py
```

Generate per-event pre-takeover time-series plots:

```bash
python scripts/plot_pre_event_timeseries.py \
  --dataset-root dataset \
  --events dataset/benchmark/takeover_events_or.csv \
  --quality reports/takeover_quality_eda/post_event_quality_metrics.csv \
  --output-dir reports/pre_event_timeseries
```
