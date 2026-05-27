---
audience: user
stability: stable
P26-05-27
---

# Camera-to-camera quick start

Before running this tool:

1. Calibrate **each camera intrinsic first** with
   [docs/camera_quickstart.md](camera_quickstart.md).
2. For result review, keep
   [docs/calibration_review_guide.md](calibration_review_guide.md) open.
3. For design background and practical references, see
   [docs/camera2camera_design.md](camera2camera_design.md) and
   [docs/calibration_methodology.md](calibration_methodology.md).

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| paired parent / child images | yes | matched by filename stem |
| parent intrinsic YAML | yes | use `camera/intrinsic.py` output or equivalent YAML |
| child intrinsic YAML | yes | same requirement as parent |
| checkerboard pattern size | yes | inner-corner count, for example `[11, 8]` |
| checkerboard square size | yes | meter unit |
| multi-pose board capture | yes | vary depth, tilt, and image location in both cameras |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Recommended directory layout

```text
run01/
  parent_intrinsics.yaml
  child_intrinsics.yaml
  parent/
    0001.png
    0002.png
  child/
    0001.png
    0002.png
```

The parent and child image directories must contain the **same stems**.

## Generate a starter config

```bash
camera2camera-calibrate \
  --write-default-config \
  --config camera2camera_config.yaml
```

Example config shape:

```yaml
cameras:
  parent:
    frame_id: camera_front_left
    image_directory: run01/parent
    intrinsics_path: run01/parent_intrinsics.yaml
  child:
    frame_id: camera_front_right
    image_directory: run01/child
    intrinsics_path: run01/child_intrinsics.yaml
target:
  type: checkerboard
  pattern_size: [11, 8]
  square_size_m: 0.025
extraction:
  min_bbox_area_ratio: 0.003
  min_edge_margin_px: 16.0
  max_pnp_reprojection_rms_px: 1.5
optimization:
  min_pairs: 8
  loss: huber
  f_scale: 1.0
  max_nfev: 300
metrics:
  warning_final_rms_px: 1.0
  warning_pair_rms_p95_px: 1.5
  warning_holdout_rms_px: 1.5
  warning_epipolar_p95_px: 1.0
output:
  directory: outputs/camera2camera/run01
```

## Run calibration

```bash
camera2camera-calibrate --config camera2camera_config.yaml
```

## Minimal validation

```bash
python tools/run_camera2camera_smoke.py --pairs 8
```

## Recommended review order

1. read `diagnostics/standardized_data.yaml`
2. read `diagnostics/data_quality.yaml`
3. read `metrics.yaml`
4. read `diagnostics/acceptance_report.yaml`
5. read `diagnostics/visualization_index.yaml`
6. inspect `diagnostics/per_pair_reprojection.csv`
7. inspect `diagnostics/leave_one_out_trials.csv`
8. inspect both image-coverage heatmaps and `pose_diversity_plot.png`
9. inspect `diagnostics/epipolar_previews/`

Outputs include:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `initial_guess/<parent>_<child>_extrinsics.yaml`
- `calibrated/<parent>_<child>_extrinsics.yaml`
- `diagnostics/`
  - `reference_dataset.yaml`
  - `extraction.yaml`
  - `optimization.yaml`
  - `evaluation.yaml`
  - `acceptance_report.yaml`
  - `status_summary.csv`
  - `standardized_data.yaml`
  - `data_quality.yaml`
  - `visualization_index.yaml`
  - `extraction_entries.csv`
  - `per_pair_reprojection.csv`
  - `leave_one_out_trials.csv`
  - `parent_image_coverage_heatmap.png`
  - `child_image_coverage_heatmap.png`
  - `pose_diversity_plot.png`
  - `epipolar_previews/`

## Acceptance baseline

- final RMS < `1.0 px`
- per-pair RMS p95 < `1.5 px`
- holdout RMS p95 < `1.5 px`
- epipolar error p95 < `1.0 px`
- accepted pair ratio remains healthy
- both parent and child image coverage span multiple grid cells
- pose diversity shows meaningful depth and tilt variation

Treat missing holdout evidence, weak coverage, or warning-level repeatability as
**review-only**, not release-ready.

