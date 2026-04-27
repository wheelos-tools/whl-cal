---
audience: user
stability: stable
P26-04-27
---


# LiDAR↪Camera Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick run:
```bash
lidar2camera-calibrate --write-default-config --config config.yaml
lidar2camera-calibrate --config config.yaml
```

Outputs: `calibrated_tf.yaml`, `metrics.yaml`, `diagnostics/`

Acceptance heuristics: final_rms_px ≤ 1.0 px; pose p95 ≤ 1.5 px; L1O p95 ≤ 1.5 px

See docs/lidar2camera_design.md for details.
