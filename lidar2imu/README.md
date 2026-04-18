# lidar2imu

This module implements a staged LiDAR-to-IMU calibration pipeline with a
separate evaluation layer.

## Commands

- `lidar2imu-calibrate`: run calibration from standardized samples
- `lidar2imu-convert-record`: convert Apollo record data into standardized
  samples, then optionally run calibration

## Documentation

- Overview: [`../docs/lidar2imu.md`](../docs/lidar2imu.md)
- Quick Start: [`../docs/lidar2imu_quickstart.md`](../docs/lidar2imu_quickstart.md)
- Current design: [`../docs/lidar2imu_design.md`](../docs/lidar2imu_design.md)

## Current status

- end-to-end conversion + calibration is working
- the pipeline has been validated on `record_data_0402`
- the latest validation suggests the current bag is usable for ground
  constraints, but weak for `x/y/yaw`

Use the Quick Start to rerun the pipeline and the design doc for the current
algorithm and iteration notes.
