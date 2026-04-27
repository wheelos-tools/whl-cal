---
audience: user
stability: stable
P26-04-27
---


# LiDAR-to-LiDAR Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick run (automatic):
```bash
lidar2lidar-auto --record-path /path/to/record --conf-dir lidar2lidar/conf --output-dir outputs/lidar2lidar/run
```

Outputs: `calibrated_tf.yaml`, `metrics.yaml`, `diagnostics/`

See docs/lidar2lidar_design.md for tuning.
