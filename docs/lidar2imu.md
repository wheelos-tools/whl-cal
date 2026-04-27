---
audience: user
stability: stable
P26-04-27
---


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

## Fixed operating versions

The repo now keeps two explicit operating versions for `lidar2imu`:

- **baseline profile**
  - purpose: stable regression / comparison reference
  - command: `lidar2imu-convert-record --profile baseline`
  - behavior:
    - `scan_to_scan`
    - `--planar-motion-policy auto`
    - no automatic re-extraction loop
- **production profile**
  - purpose: current mass-production candidate
  - command: `lidar2imu-convert-record --profile production`
  - behavior:
    - `submap_to_map`
    - wider target local map support
    - `--planar-motion-policy auto`
    - automatic one-step re-extraction when extraction consistency warns
    - holdout generalization remains part of acceptance
    - trusted-reference / extraction checks now keep the old translation-norm +
      rotation gates **and** a separate vertical `z` gate

This follows a common industrial pattern:

1. keep a fixed conservative baseline for regression
2. expose a better-performing production candidate separately
3. judge both with the same stable acceptance surfaces

The current trusted-prior rule is intentionally stricter on vertical height than
on generic translation drift:

- `delta_to_reference.translation_norm_m` / `delta_to_extraction.translation_norm_m`
  still guard the overall transform difference
- `delta_to_reference.translation_xyz_m.z` /
  `delta_to_extraction.translation_xyz_m.z` now guard pure vertical drift that
  can hide inside a near-threshold total translation norm

The pipeline now also classifies the **user-provided TF itself**:

- `initial_prior_assessment=pass`
  - input TF is already inside the nominal production range
- `initial_prior_assessment=recoverable`
  - input TF is outside nominal range, but this bag still converged back to the
    accepted basin
- `initial_prior_assessment=warning`
  - input TF is outside nominal range and this run also has acceptance conflicts

This separates two different questions:

1. **Can the solver recover from this prior on this bag?**
2. **Should production trust this prior as a normal installation input?**

Current recommendation on `record_data_0402`:

- keep pose-derived gravity as the default
- trust `z/roll/pitch` first
- do not accept `x/y/yaw` from this bag as a production-quality result without a
  second bag that contains both left and right turns

Use the Quick Start to rerun the pipeline, then use the design doc when tuning
the converter or solver.
