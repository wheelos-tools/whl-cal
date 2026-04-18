# Engineering index

This file is the compact index for the current engineering surfaces in the repo.

## Repository split

| Area | Data layer | Algorithm layer | Evaluation layer |
| --- | --- | --- | --- |
| `lidar2lidar` | `record_utils.py`, `scan2map_dataset.py`, `auto_calib.py` extraction paths | `auto_calib.py`, `scan2map_calib.py`, `temporal_calib.py` | `metrics.yaml`, `diagnostics/*.yaml` |
| `lidar2imu` | `record_converter.py` | `pipeline.py`, `algorithms.py` | `metrics.py`, `diagnostics/*.yaml` |
| `camera` | interactive image capture in `camera/intrinsic.py` | chessboard intrinsic solve | YAML result written directly by the script |
| `camera2lidar` | raw image / PCD pair loading inside scripts | `reference_based.py`, `learning_based.py` | script-local metrics only, no repo-wide stable artifact yet |

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

### camera / camera2lidar

- `python camera/intrinsic.py`
- `python camera2lidar/reference_based.py`
- `python camera2lidar/learning_based.py`

## Stable artifact surfaces

### lidar2lidar

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/scan2map_dataset.yaml`
- `diagnostics/scan2map_optimization.yaml`
- `diagnostics/evaluation.yaml`

### lidar2imu

- `standardized_samples.yaml`
- `conversion_diagnostics.yaml`
- `calibration/calibrated_tf.yaml`
- `calibration/metrics.yaml`
- `calibration/diagnostics/algorithm.yaml`
- `calibration/diagnostics/evaluation.yaml`
- `calibration/diagnostics/observability.yaml`

### camera / camera2lidar

- currently script-local outputs only
- no repo-wide `metrics.yaml` / `diagnostics/` convention yet
- this is the next structural gap to close

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
