---
audience: user
stability: stable
P26-04-27
---


# Camera Intrinsic Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy pyyaml
```

Interactive:
```bash
python camera/intrinsic.py --config camera_config.yaml
```

Headless (images dir):
```bash
python camera/intrinsic.py --config tmp_config.yaml --images-dir /path/to/images --pattern-size 4,3
```

Outputs: `calibration_YYYYmmdd_HHMMSS.yaml`, `comparison_view.png`.

Acceptance: avg reprojection error < 1.0 px. See docs/lidar2camera_quickstart.md for extrinsic workflow.
