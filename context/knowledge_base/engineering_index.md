# Engineering index

This file is the compact index for the current engineering surfaces in the repo.

## Repository split

| Area | Data layer | Algorithm layer | Evaluation layer |
| --- | --- | --- | --- |
| `lidar2lidar` | `record_utils.py`, `scan2map_dataset.py`, `auto_calib.py` extraction paths | `auto_calib.py`, `scan2map_calib.py`, `temporal_calib.py` | `metrics.yaml`, `diagnostics/*.yaml` |
| `lidar2imu` | `record_converter.py` | `pipeline.py`, `algorithms.py` | `metrics.py`, `diagnostics/*.yaml` |
| `camera` | interactive/headless checkerboard sample collection in `camera/intrinsic.py` | chessboard intrinsic solve | calibration YAML + `*_diagnostics/` acceptance/data_quality/visualization artifacts |
| `lidar2camera` | raw image / PCD pair loading + extraction gating in `reference_pipeline.py` | `reference_pipeline.py`, `learning_based.py` | `metrics.yaml`, `diagnostics/*.yaml`, CSVs, heatmap/scatter/overlay review surfaces |

## Main commands

### lidar2lidar

- `lidar2lidar-auto`
- `lidar2lidar-calibrate`
- `lidar2lidar-scan2map-dataset`
- `lidar2lidar-scan2map`
- `lidar2lidar-temporal`

### lidar2imu

- `lidar2imu-calibrate`
- `lidar2imu-convert-record`
- `lidar2imu-tune-record`

### camera / lidar2camera

- `python camera/intrinsic.py`
- `lidar2camera-calibrate`

## Environment note

- When local `pip install -e .` stalls on the default PyPI route, prefer a fresh
  virtual environment and set the active environment to the Tsinghua mirror:
  `python -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`
- This keeps the mirror scoped to the current virtual environment instead of
  changing the whole machine.

## Stable artifact surfaces

### lidar2lidar

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/scene_sufficiency.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/visual_evaluation.yaml`
- `diagnostics/scan2map_dataset.yaml`
- `diagnostics/scan2map_optimization.yaml`
- `diagnostics/evaluation.yaml`
- `initial_guess/*.yaml`
- `calibrated/*.yaml`
- optional `visual_review/*/merged_cloud_colored.ply`

### lidar2imu

- `standardized_samples.yaml`
- `conversion_diagnostics.yaml`
- `calibration/calibrated_tf.yaml`
- `calibration/metrics.yaml`
- `calibration/diagnostics/algorithm.yaml`
- `calibration/diagnostics/evaluation.yaml`
- `calibration/diagnostics/observability.yaml`

### camera / lidar2camera

- camera intrinsic:
  - `calibration_*.yaml`
  - `calibration_*_diagnostics/acceptance_report.yaml`
  - `calibration_*_diagnostics/data_quality.yaml`
  - `calibration_*_diagnostics/per_view_reprojection.csv`
  - `calibration_*_diagnostics/image_coverage_heatmap.png`
- lidar2camera:
  - `calibrated_tf.yaml`
  - `metrics.yaml`
  - `diagnostics/extraction.yaml`
  - `diagnostics/optimization.yaml`
  - `diagnostics/evaluation.yaml`
  - `diagnostics/extraction_entries.csv`
  - `diagnostics/per_pose_reprojection.csv`
  - `diagnostics/leave_one_out_trials.csv`
  - `diagnostics/geometry_resolution.csv`
  - `diagnostics/image_coverage_heatmap.png`
  - `diagnostics/pose_diversity_plot.png`

## Current indexing design

### 1. Data indexing

- raw bag path
- topic choice
- frame choice
- initial transform source
- selected windows / samples
- skip / rejection reasons

### 2. Algorithm indexing

- solver mode
- loss / thresholds / policy switches
- constrained vs free solve
- selected registration method or candidate

### 3. Evaluation indexing

- coarse gate metrics
- fine diagnostics
- recommendation field
- drift to initial / baseline
- observability warnings

## Current lidar2lidar operating policy

Use this order for practical `lidar2lidar` work:

1. prefer direct `scan2scan` when the pair has high shared coverage
2. add loop closure only when a real, healthy loop exists
3. when no loop exists, aggregate multiple windows and keep a representative transform
4. treat seed-only oscillation across distinct solution families as a data or observability problem, not as convergence
5. keep release decisions tied to scene sufficiency, repeatability, visual geometry, and topology consistency instead of fitness alone

## Current lidar2imu window + gate design

The current motion-selection index is:

1. enumerate candidate motion pairs
2. assign candidates into timeline windows
3. summarize each window
4. gate weak windows
5. try top-k candidates inside valid windows
6. keep only windows whose selected candidate passes registration gating

This is the current preferred design because it makes invalid or abnormal segments
explicit in `conversion_diagnostics.yaml`.
