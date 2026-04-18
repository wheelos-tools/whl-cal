# LiDAR-to-IMU Quick Start

This document is only for running the current pipeline quickly.

## 1. Install

```bash
cd /home/wfh/01code/whl-cal
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Run from standardized samples

If you already have `ground_samples` and `motion_samples`:

```bash
lidar2imu-calibrate \
  --input lidar2imu_samples.yaml \
  --planar-motion-policy auto \
  --output-dir outputs/lidar2imu/run01
```

Outputs:

- `outputs/lidar2imu/run01/calibrated_tf.yaml`
- `outputs/lidar2imu/run01/metrics.yaml`
- `outputs/lidar2imu/run01/calibrated/<parent>_<child>_extrinsics.yaml`
- `outputs/lidar2imu/run01/diagnostics/`

## 3. Run from Apollo record data

Convert one record directory into standardized samples and calibrate directly:

```bash
lidar2imu-convert-record \
  --record-path /home/wfh/01code/apollo-lite/data/bag/record_data_0402 \
  --output-dir outputs/lidar2imu/raw_validation \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --motion-frame-stride 5 \
  --min-registration-fitness 0.55 \
  --planar-motion-policy auto \
  --calibrate
```

If the bag does not contain `lidar -> parent` TF, add one of:

```bash
--initial-transform path/to/imu_lidar_front_extrinsics.yaml
```

or, for exploratory-only runs:

```bash
--identity-initial-transform
```

Outputs:

- `outputs/lidar2imu/raw_validation/standardized_samples.yaml`
- `outputs/lidar2imu/raw_validation/conversion_diagnostics.yaml`
- `outputs/lidar2imu/raw_validation/calibration/`

## 4. Check the most important results first

Open:

- `conversion_diagnostics.yaml`
- `calibration/metrics.yaml`

Start with these fields:

- `coarse_metrics.ground_normal_angle_p95_deg`
- `coarse_metrics.motion_rotation_residual_p95_deg`
- `coarse_metrics.motion_translation_residual_p95_m`
- `coarse_metrics.motion_registration_fitness_p05`
- `coarse_metrics.turn_balance_ratio`
- `coarse_metrics.left_turn_count`
- `coarse_metrics.right_turn_count`
- `coarse_metrics.statuses`
- `vehicle_motion_assessment.recommendation`
- `vehicle_motion_assessment.applied_solver_planar_motion_policy`
- `summary.solver_policy`

## 5. Current interpretation rule

- If ground metrics are good but motion metrics are weak, trust `z/roll/pitch`
  more than `x/y/yaw`.
- If `left_turn_count` or `right_turn_count` is missing, treat `x/y/yaw` as
  weakly observable.
- If `vehicle_motion_assessment.recommendation` is `z_roll_pitch_priority`,
  treat the run as vertically trustworthy but planarly under-excited.
- On weak-planar bags, prefer `--planar-motion-policy auto`; it will lock
  `x/y/yaw` to the initial prior when turn balance or yaw observability is weak,
  while continuing to refine `z/roll/pitch`.
- If the run needed `--identity-initial-transform`, do not accept the final
  extrinsics directly unless another trusted prior or cross-bag check agrees.
- If `motion_translation_residual_p95_m` is large, inspect motion pair quality
  in `conversion_diagnostics.yaml` before changing the solver.
- If `gravity-source imu` yields almost no accepted ground samples, fall back to
  pose-derived gravity for that bag.
