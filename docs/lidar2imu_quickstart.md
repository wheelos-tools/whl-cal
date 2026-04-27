---
audience: user
stability: stable
P26-04-27
---


# LiDAR-to-IMU Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick run:
```bash
lidar2imu-calibrate --input lidar2imu_samples.yaml --output-dir outputs/lidar2imu/run01
```

Outputs: consolidated `calibrated_tf.yaml`, run `metrics.yaml` and `diagnostics/`.

See docs/lidar2imu_design.md for parameters.
