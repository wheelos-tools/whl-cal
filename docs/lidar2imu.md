# LiDAR-to-IMU calibration overview

`lidar2imu` now has three layers of documentation:

- **Overview**: this file
- **Quick Start**: [docs/lidar2imu_quickstart.md](lidar2imu_quickstart.md)
- **Current design**: [docs/lidar2imu_design.md](lidar2imu_design.md)

## What is implemented

- `lidar2imu-calibrate`: run the staged LiDAR-to-IMU solver from standardized
  `ground_samples` and `motion_samples`
- `lidar2imu-convert-record`: convert Apollo record data into those standardized
  samples, then optionally run calibration immediately

## Why the docs are split

- **Quick Start** is for running commands quickly without reading algorithm
  details.
- **Current design** is for iteration work: residual definitions, metrics,
  observability, conversion logic, and the latest real-bag validation notes.

## Current status

The full chain has been validated on:

- `/home/wfh/01code/apollo-lite/data/bag/record_data_0402`

The pipeline runs end-to-end and produces:

- `standardized_samples.yaml`
- `conversion_diagnostics.yaml`
- `calibration/calibrated_tf.yaml`
- `calibration/metrics.yaml`
- `calibration/diagnostics/*.yaml`

The latest validation shows:

- ground constraints are usable
- low-fitness motion pairs should be rejected at conversion time
- motion constraints in `record_data_0402` are still weak for `x/y/yaw` because
  the bag only excites one turning direction
- the evaluation layer correctly flags those weaknesses with warnings instead of
  hiding them, and now emits a `vehicle_motion_assessment` recommendation

Current recommendation on `record_data_0402`:

- keep pose-derived gravity as the default
- trust `z/roll/pitch` first
- do not accept `x/y/yaw` from this bag as a production-quality result without a
  second bag that contains both left and right turns

Use the Quick Start to rerun the pipeline, then use the design doc when tuning
the converter or solver.
