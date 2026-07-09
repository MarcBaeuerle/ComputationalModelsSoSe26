---
license: cc-by-nc-4.0
task_categories:
- video-classification
- time-series-forecasting
language:
- en
tags:
- driving
- autonomous-driving
- multimodal
- handover
- benchmark
- naturalistic-driving
pretty_name: BATON-Sample
size_categories:
- 100G<n<1T
---

<div align="center">

# 🚗 BATON-Sample

### **B**ehavioral **A**nalysis of **T**ransition and **O**peration in **N**aturalistic Driving

<p>
  <a href="https://arxiv.org/abs/2604.07263">
    <img src="https://img.shields.io/badge/arXiv-2604.07263-b31b1b.svg?style=for-the-badge&logo=arxiv"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/datasets/HenryYHW/BATON">
    <img src="https://img.shields.io/badge/🤗-Full_Dataset-ffd21e?style=for-the-badge"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/datasets/HenryYHW/BATON-Sample">
    <img src="https://img.shields.io/badge/🤗-Sample_Dataset-ffd21e?style=for-the-badge"/>
  </a>
  &nbsp;
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge"/>
  </a>
</p>

*A large-scale multimodal benchmark for bidirectional human–DAS control transition in naturalistic driving*<br/>
*Submitted to ACM Multimedia 2026*

> **This is the sample release of BATON** — 43 routes covering all 9 modalities, ready for quick exploration and prototyping. For the full 565-route dataset, see [HenryYHW/BATON](https://huggingface.co/datasets/HenryYHW/BATON).

</div>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/teaser.png" width="100%"/>
</div>

---

## 🎬 Live Preview

<table width="100%">
<tr>
  <td width="38%" align="center" valign="top">
    <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/gif_fig1.gif" width="100%"/>
    <br/><sub><b>Continuous sequence</b> — cabin fisheye · front view · 5 fps</sub>
  </td>
  <td width="24%" align="center" valign="top">
    <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/gif_sensors.gif" width="100%"/>
    <br/><sub><b>8 CAN/IMU sensor streams</b><br/>cycling through all channels</sub>
  </td>
  <td width="38%" align="center" valign="top">
    <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/gif_fig3.gif" width="100%"/>
    <br/><sub><b>Daytime time-lapse</b> — front · cabin · 2-min intervals</sub>
  </td>
</tr>
</table>

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/gif_fig2.gif" width="72%"/>
  <br/>
  <sub><b>Nighttime driving</b> — time-lapse with <b>⬆ DAS Handover</b> and <b>↩ Human Takeover</b> event highlights</sub>
</div>

---

## 📦 Sample Release at a Glance

<div align="center">

| 🌍 Routes | 👤 Drivers | 🚙 Car Models | ⏱️ Duration | 🔄 Handover Events |
|:---------:|:----------:|:-------------:|:-----------:|:-----------------:|
| **43** | **43** | **~20** | **~15 h** | **~330** |

| 🤖 DAS Driving | 🧑 Human Driving | ⬆️ DAS Handover | ↩️ Human Takeover | 🌍 Coverage |
|:--------------:|:----------------:|:---------------:|:-----------------:|:-----------:|
| ~52% | ~48% | ~165 | ~165 | 5 Continents |

</div>

> **Full dataset:** 565 routes · 150 drivers · 99 car models · 162.1 h · 3,593 handover events — available at [HenryYHW/BATON](https://huggingface.co/datasets/HenryYHW/BATON)

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/DatasetOverview.jpg" width="88%"/>
  <br/><sub><i>Global distribution of participants, per-driver duration, and handover event breakdown (full dataset).</i></sub>
</div>

---

## 📁 Sample Contents

Each of the 43 routes contains all 9 synchronized modalities:

```
BATON-Sample/
└── {vehicle_model}/
    └── {driver_id}/
        └── {route_hash}/
            ├── vehicle_dynamics.csv   # Speed, accel, steering, pedals, DAS status
            ├── planning.csv           # DAS curvature, lane change intent
            ├── radar.csv              # Lead vehicle distance & relative speed
            ├── driver_state.csv       # Face pose, eye openness, awareness
            ├── imu.csv                # 3-axis accel & gyro at 100 Hz
            ├── gps.csv                # Coordinates, heading
            ├── localization.csv       # Road curvature, lane position
            ├── qcamera.mp4            # Front-view video (526×330, H.264, 20 fps)
            └── dcamera.mp4            # In-cabin fisheye video (1928×1208, HEVC, 20 fps)
```

---

## 🔬 Data Collection & Modalities

**Setup:** Non-intrusive plug-and-play OBD-II dongle + dual cameras. Drivers use their own vehicles during real daily commutes — no lab, no script.

| Component | Spec |
|-----------|------|
| 📡 OBD-II Dongle | CAN-bus at 100 Hz |
| 📷 Front camera | 526×330 · H.264 · 20 fps |
| 🎥 Cabin fisheye | 1928×1208 · HEVC · 20 fps |
| 🛰️ GPS | 10 Hz |

**9 synchronized modalities:**
- `vehicle_dynamics.csv` — speed, accel, steering, pedals, DAS status
- `planning.csv` — DAS curvature, lane change intent
- `radar.csv` — lead vehicle distance & relative speed
- `driver_state.csv` — face pose, eye openness, awareness
- `imu.csv` — 3-axis accel & gyro at 100 Hz
- `gps.csv` — coordinates, heading
- `localization.csv` — road curvature, lane position
- `qcamera.mp4` — front-view video
- `dcamera.mp4` — in-cabin fisheye video

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/experiment_method.jpg" width="88%"/>
</div>

| 📷 Front · Day | 📷 Front · Night | ⬆️ DAS Handover |
|:-:|:-:|:-:|
| ![](https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/qcamera_day.jpg) | ![](https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/qcamera_night.jpg) | ![](https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/qcamera_activation.jpg) |
| **🎥 Cabin · Day** | **🎥 Cabin · Night** | **↩️ Human Takeover** |
| ![](https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/dcamera_day.jpg) | ![](https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/dcamera_night.jpg) | ![](https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/dcamera_takeover.jpg) |

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/BenchmarkOverview.jpg" width="100%"/>
  <br/><sub><i>Aligned multimodal streams around a HANDOVER event: cabin video · front video · GPS trajectory · sensor signals.</i></sub>
</div>

---

## 🏆 Benchmark Tasks

<div align="center">
  <img src="https://raw.githubusercontent.com/OpenLKA/BATON/main/figs/TaskDistribution.jpg" width="80%"/>
</div>

<br/>

| Task | Description | Samples (full) | Labels | Primary Metric |
|------|-------------|:--------------:|--------|:--------------:|
| 🎯 **Task 1** | Driving action recognition (7-class) | 1,161,794 | Cruising · Car Following · Accelerating · Braking · Lane Change · Turning · Stopped | Macro-F1 |
| ⬆️ **Task 2** | Handover prediction (Human→DAS) | 65,223 | Handover (15.8%) · No Handover | AUPRC |
| ↩️ **Task 3** | Takeover prediction (DAS→Human) | 91,255 | Takeover (11.6%) · No Takeover | AUPRC |

> **Evaluation protocol:** Cross-driver split · 5-second input window · 3-second prediction horizon · 3 seeds (42, 123, 7)
>
> **🔁 Leak-safe evaluation (revision):** For the handover/takeover tasks we exclude four controller-internal automation signals (`cc_latActive`, `cruiseState_enabled`, `cs_longControlState`, `actuators_accel`) that trivially reveal the label. Video baselines use **V-JEPA2 ViT-L** features; multimodal fusion of video + structured signals improves leak-safe handover AUPRC over the strong tabular baseline.

---

## 🚀 Quick Start

### 1. Get the sample data

```bash
# Clone this sample dataset (~few GB, all modalities, 43 routes)
git lfs install
git clone https://huggingface.co/datasets/HenryYHW/BATON-Sample

# Or via Python
from huggingface_hub import snapshot_download
snapshot_download('HenryYHW/BATON-Sample', repo_type='dataset', local_dir='./data')
```

### 2. Get the full dataset

```bash
# Full dataset (565 routes) — requires HuggingFace account
python -c "
from huggingface_hub import snapshot_download
snapshot_download('HenryYHW/BATON', repo_type='dataset', local_dir='./data')
"
```

### 3. Extract video features

```bash
cd data_processing

# V-JEPA2 ViT-L features (used in main baselines)
python extract_vjepa2_features.py

# Legacy EfficientNet-B0 features (optional)
python extract_front_video_features.py
python extract_cabin_video_features.py
```

### 4. Train baselines (requires full dataset + benchmark files)

```bash
cd baseline

# GRU on all modalities — Task 1
python train_nn.py --task task1 --modality Full-All --model gru --seed 42

# XGBoost on structured signals — Task 2
python train_classical.py --task task2 --model xgb --seed 42

# Zero-shot VLM baseline (GPT-4o or Gemini 2.5 Flash)
python run_vlm.py --model gpt4o --task task1
```

See [GitHub — OpenLKA/BATON](https://github.com/OpenLKA/BATON) for the complete codebase.

---

## 📐 Evaluation Protocol

| Setting | Value |
|---------|-------|
| **Primary split** | Cross-driver (disjoint drivers in train / val / test) |
| **Additional splits** | Cross-vehicle, Random |
| **Input window** | 5 seconds |
| **Prediction horizon** | 1 s, 3 s, 5 s (main: **3 s**) |
| **Random seeds** | 42, 123, 7 — report 3-seed average |
| **Leak-safe input** | Task 2 / 3 exclude `cc_latActive`, `cruiseState_enabled`, `cs_longControlState`, `actuators_accel` |
| **Video features** | V-JEPA2 ViT-L (main); EfficientNet-B0 / CLIP (legacy) |
| **Task 1 metric** | Macro-F1 |
| **Task 2 / 3 metrics** | AUPRC (primary), AUC-ROC, F1 |

---

## 📡 Data Access

| Resource | Link |
|----------|------|
| 🔍 **This Sample** (43 routes) | [HuggingFace — HenryYHW/BATON-Sample](https://huggingface.co/datasets/HenryYHW/BATON-Sample) |
| 📦 Full Dataset (565 routes) | [HuggingFace — HenryYHW/BATON](https://huggingface.co/datasets/HenryYHW/BATON) |
| 💻 Code & Baselines | [GitHub — OpenLKA/BATON](https://github.com/OpenLKA/BATON) |
| 📄 arXiv Paper | [arxiv.org/abs/2604.07263](https://arxiv.org/abs/2604.07263) |

---

## 📜 Citation

```bibtex
@article{wang2026baton,
  title   = {BATON: A Multimodal Benchmark for Bidirectional Automation Transition
             Observation in Naturalistic Driving},
  author  = {Wang, Yuhang and Xu, Yiyao and Yang, Chaoyun and Li, Lingyao
             and Sun, Jingran and Zhou, Hao},
  journal = {arXiv preprint arXiv:2604.07263},
  year    = {2026}
}
```

---

## 📄 License

This dataset is released for **academic research use only** under [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/) (Creative Commons Attribution–NonCommercial 4.0 International).

**You are free to** use and redistribute the data for non-commercial research, and to adapt or build upon it for non-commercial purposes — **provided that:**

- **Attribution** — You must cite the BATON paper (see Citation above) in any publication or work that uses this dataset.
- **Non-Commercial** — Commercial use of this dataset or any derivative is **strictly prohibited**.
- **Academic Use Only** — This dataset is intended solely for academic research. Use in any commercial product, service, or application is not permitted.

For commercial licensing inquiries, please contact the authors.

---

<div align="center">
  <sub>
    🔗 <a href="https://arxiv.org/abs/2604.07263">Paper</a> &nbsp;·&nbsp;
    <a href="https://huggingface.co/datasets/HenryYHW/BATON">Full Dataset</a> &nbsp;·&nbsp;
    <a href="https://huggingface.co/datasets/HenryYHW/BATON-Sample">Sample Dataset</a> &nbsp;·&nbsp;
    <a href="https://github.com/OpenLKA/BATON">GitHub</a>
  </sub>
</div>
