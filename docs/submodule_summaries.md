---
audience: user
stability: stable
last_tested: 2026-04-27
summary: Consolidated summaries from removed subdirectory README files.
---

# Submodule README consolidation

This document preserves concise, user-facing snippets that were previously located in README.md files inside subdirectories. The original files have been removed to centralize user documentation under /docs and keep the repository root README as the canonical entrypoint.

## camera

- Purpose: Intrinsic camera calibration using a chessboard.
- Quick start:
  - Install: `pip install opencv-python numpy pyyaml`
  - Run: `python camera/intrinsic.py`
  - Controls: `S` save frame, `C` run calibration (>=20 frames), `Q` quit.
- Output: `calibration_results.yaml` (camera_matrix, distortion_coefficients, reprojection_error).
- Note: `camera/intrinsic.py` is intrinsic-only. Use `lidar2camera` for LiDAR↔Camera extrinsics and full evaluation (`lidar2camera-calibrate --config config.yaml`).

## tools

This repository provides operational helper scripts under `tools/` for data preparation and inspection. Examples:

- `tools/lidar2lidar/list_topics.py`: inspect Apollo record topics
- `tools/lidar2lidar/extract_pcd.py`: export PointCloud2 messages to PCD
- `tools/lidar2lidar/merge_pcd.py`: merge two PCD files with a transform

Design rule:
- `lidar2lidar/`: reusable library code and production entrypoints
- `tools/`: operational helpers and data preparation scripts

## lidar2lidar/conf

This directory stores standardized fallback extrinsics used by the automatic pipeline when `/tf_static` or `/tf` is incomplete.

- Naming: `<parent_frame>_<child_frame>_extrinsics.yaml`
- Schema (example):

```yaml
header:
  stamp:
    secs: 0
    nsecs: 0
  seq: 0
  frame_id: lslidar_main
transform:
  translation:
    x: 0.0
    y: 0.0
    z: 0.0
  rotation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
child_frame_id: lslidar_left
```

Bootstrap example:

```bash
lidar2lidar-auto \
  --record-path your/data/bag/ \
  --conf-dir lidar2lidar/conf \
  --bootstrap-conf \
  --output-dir outputs/lidar2lidar/bootstrap_only
```

## lidar2lidar

- Purpose: raw-record multi-LiDAR calibration workflow.
- Commands: `lidar2lidar-topics`, `lidar2lidar-auto`, `lidar2lidar-calibrate`, `lidar2lidar-extract`, `lidar2lidar-merge`.
- Docs:
  - Overview: `docs/lidar2lidar.md`
  - Quick Start: `docs/lidar2lidar_quickstart.md`
  - Design: `docs/lidar2lidar_design.md`

## lidar2imu

- Purpose: staged LiDAR-to-IMU calibration pipeline with a separate evaluation layer.
- Commands: `lidar2imu-calibrate`, `lidar2imu-convert-record`.
- Docs:
  - Overview: `docs/lidar2imu.md`
  - Quick Start: `docs/lidar2imu_quickstart.md`
  - Design: `docs/lidar2imu_design.md`

---

Files removed from subdirectories were migrated here to centralize user-facing documentation in `docs/`. For detailed developer design and parameters, consult the `docs/*_design.md` files for each package.
