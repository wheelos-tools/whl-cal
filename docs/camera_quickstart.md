---
audience: user
stability: stable
last_tested: 2026-04-27
---

# Camera Intrinsic Quick Start

This short guide shows interactive and headless modes for camera intrinsic calibration.

Prerequisites

- Install dependencies: `pip install opencv-python numpy pyyaml`
- Activate the project venv and install package: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`

Interactive (GUI)

- Run the tool:

```bash
python camera/intrinsic.py --config camera_config.yaml
```

- Controls: `S` save frame, `C` run calibration, `R` restart, `V` validate, `ESC` quit.

Headless (images directory)

- Generate or collect board images into a directory (jpg/png). Example synthetic test is provided in `tools/run_lidar2camera_smoke.py` but any photo set with a visible chessboard works.

- Create/modify a config (the script writes a default config when missing):

```bash
python camera/intrinsic.py --config tmp_config.yaml
# edit tmp_config.yaml if needed
```

- Run headless calibration on a directory of images:

```bash
python camera/intrinsic.py --config tmp_config.yaml --images-dir /path/to/images --pattern-size 4,3
```

Notes on results

- Output: `calibration_YYYYmmdd_HHMMSS.yaml` and `comparison_view.png` (undistorted vs distorted).
- Acceptance heuristic: average reprojection error < 1.0 px is typical for well-captured data. If error is larger, capture more varied poses and ensure the board covers many image regions.

For LiDAR↔Camera extrinsics and full evaluation, use the `lidar2camera` reference baseline: `lidar2camera-calibrate --config config.yaml` (see `docs/lidar2camera_quickstart.md`).
