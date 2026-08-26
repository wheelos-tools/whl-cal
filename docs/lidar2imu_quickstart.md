---
audience: user
stability: stable
P26-05-25
---

# LiDAR-to-IMU quick start

## Requirements

- Python 3.8 or newer
- Apollo record containing LiDAR, pose, IMU, and `/tf_static`
- a reasonable initial `lidar -> imu` transform
- motion with left/right turns, acceleration/braking, and flat-road segments

For collection guidance, see
[apollo_data_collection.md](apollo_data_collection.md). For algorithm tuning,
see [lidar2imu_design.md](lidar2imu_design.md).

## Install

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run from an Apollo record

Use the stable scan-to-scan baseline first:

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/imu \
  --output-dir outputs/lidar2imu/run01 \
  --profile baseline \
  --calibrate
```

If the bag does not contain the static transform, add:

```bash
--initial-transform /path/to/lidar_imu_extrinsics.yaml
```

## Recommended fast path

For repeated runs, build a prepared dataset once:

```bash
lidar2lidar-rig-dataset \
  --record-path /path/to/record \
  --output-dir outputs/prepared/run01 \
  --lidar-topics /apollo/sensor/your_lidar/PointCloud2 \
  --reference-topic /apollo/sensor/your_lidar/PointCloud2 \
  --export-voxel-size 0.05
```

Then calibrate from cached PCD, pose, and IMU artifacts:

```bash
lidar2imu-convert-record \
  --prepared-dataset-yaml \
    outputs/prepared/run01/diagnostics/prepared_rig_dataset.yaml \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --output-dir outputs/lidar2imu/run01 \
  --profile baseline \
  --calibrate
```

Prepared mode avoids repeatedly scanning the raw record during extraction,
registration review, trajectory visualization, and cloud-thickness evaluation.

## Run from standardized samples

```bash
lidar2imu-calibrate \
  --input outputs/lidar2imu/run01/standardized_samples.yaml \
  --output-dir outputs/lidar2imu/run01_replay
```

This is the fastest path for solver-only A/B tests.

## Check the result

Read the conclusion first:

```bash
python - <<'PY'
import yaml

path = "outputs/lidar2imu/run01/calibration/metrics.yaml"
with open(path, "r") as stream:
    metrics = yaml.safe_load(stream)

print(metrics["summary"]["final_acceptance_status"])
print(metrics["summary"]["release_ready"])
print(metrics["final_acceptance"]["recommendation"])
PY
```

Required production conditions:

- `summary.final_acceptance_status: pass`
- `summary.release_ready: true`
- Fisher eigenvalue and conditioning gates pass
- holdout cloud-thickness gate passes

Open the visual report:

```bash
xdg-open outputs/lidar2imu/run01/calibration/diagnostics/review_report.html
```

Key artifacts:

- `calibrated_tf.yaml`: calibrated transform
- `metrics.yaml`: acceptance result and metrics
- `diagnostics/data_quality.yaml`: extraction quality
- `diagnostics/registration_review.yaml`: per-window registration quality
- `diagnostics/trajectory_overlay.svg`: IMU/LiDAR trajectory comparison
- `diagnostics/trajectory_overlay_cloud.ply`: geometric overlay

Do not accept a run only because the solver converged. If
`freeze_xyyaw` was applied, only `z/roll/pitch` should be treated as usable.

## Diagnose extraction failures

Inspect:

```bash
cat outputs/lidar2imu/run01/conversion_diagnostics.yaml
```

Important fields:

- `motion_rejected_frame_gap`: candidates crossing LiDAR data gaps
- `lidar_median_frame_delta_ms`: nominal LiDAR frame period
- `motion_rejected_low_fitness`: failed registrations
- `motion_registered_candidate_count`: usable motion factors

If extraction does not meet the minimum sample count, the converter writes
`calibration_skipped.yaml` and does not run the solver.

For advanced solver, observability, temporal-offset, or submap experiments,
follow [lidar2imu_design.md](lidar2imu_design.md) and keep the baseline output
for comparison.
