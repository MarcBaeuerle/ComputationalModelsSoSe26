# PassingCtrl Benchmark Protocol

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
| State persistence (debounce) | 1.0 s |
| Minimum same-type event gap | 2.0 s |
| Minimum ADAS-active episode | 2.0 s |
| Minimum human-control episode | 1.0 s |

## 4. Driving Action Taxonomy
Priority order: Stopped > LaneChange > Turning > Braking > Accelerating > CarFollowing > Cruising

| Action | Rule | Threshold |
|---|---|---|
| Stopped | vEgo < 0.5 m/s for ≥ 2.0s | velocity-based |
| LaneChange | laneChangeState > 0 OR (blinker AND \|steer\| > 5°) | planning + signal |
| Turning | \|steeringAngleDeg\| > 10.0° for ≥ 1s | 10.0° |
| Braking | aEgo < -0.35 m/s² OR brakePressed | -0.35 m/s² |
| Accelerating | aEgo > 0.37 m/s² for ≥ 1s | 0.37 m/s² |
| CarFollowing | leadOne active AND dRel < 60.0m AND \|aEgo\| < 1.0 | radar-based |
| Cruising | none of above apply | default |

Labels are generated at 1 Hz (per-second), then aggregated to window-level by majority vote.

## 5. Benchmark Tasks

### Task 1: Driving Action Understanding
- Input: 5.0s multimodal window
- Output: 7-class action label (majority vote over 1Hz labels in window)
- Stride: 0.5s

### Task 2: ADAS Activation Prediction
- Input: 5.0s multimodal window
- Output: binary — will ADAS activate within horizon?
- Primary horizon: 3.0s
- Additional horizons: 1s, 5s
- Positive: window ends in human-driving state, activation occurs within horizon
- Negative: window ends in human-driving state, no activation within horizon + 2s buffer

### Task 3: Takeover Prediction
- Input: 5.0s multimodal window
- Output: binary — will driver take over within horizon?
- Primary horizon: 3.0s
- Additional horizons: 1s, 5s
- Positive: window ends in ADAS-active state, takeover occurs within horizon
- Negative: window ends in ADAS-active state, no takeover within horizon + 2s buffer

## 6. Sample Construction
- Input window: 5.0s
- Stride: 0.5s
- Samples within 5.0s of route start/end are excluded
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
*PassingCtrl Benchmark Protocol v1 — 2026-03-28*
