---
name: gril-calib-validation
description: End-to-end validation and failure-analysis workflow for GRIL-Calib LiDAR-IMU calibration. Use this when running, debugging, or judging GRIL-Calib, especially for Apollo-to-ROS conversion, frame conventions, point timing, LiDAR-only odometry derivatives, initialization sensitivity, or result acceptance.
---

# GRIL-Calib Validation

Use this skill to validate GRIL from its input contract through its physical
holdout. Do not accept or reject a result merely because it differs from a
configured transform.

## Executable workflow

This skill is self-contained. Its `scripts/`, `resources/`, and `patches/`
directories are part of the workflow, not examples that need to be rewritten.

Prerequisites:

- this repository installed in a Python environment
- Docker
- `git`
- Apollo record files containing LiDAR, IMU, static TF, and independent
  GNSS/INS odometry

Run the complete workflow from the repository root:

```bash
python .agents/skills/gril-calib-validation/scripts/gril_pipeline.py all \
  --record-file /data/capture.record.00000 \
  --record-file /data/capture.record.00001 \
  --output-dir outputs/gril/capture01
```

The command:

1. clones the pinned GRIL revision into `.cache/gril-validation`
2. applies `patches/gril-validation.patch`
3. builds the supplied Docker image and deterministic GRIL frontend
4. converts Apollo records to a ROS1 bag
5. rejects the run if the conversion contract fails
6. runs GRIL twice
7. exports each result, trajectory, raw log, and aligned state sequence
   together with the pre-calibration `GRIL_batch_trace_v1.txt`
8. creates the LiDAR/GNSS trajectory comparison
9. runs reference-free dynamics and yaw-time holdout checks
10. builds an independent IMU/GNSS-odometry submap with point-thickness metrics

The supplied runner delays playback after topic advertisement so the frozen
reference cannot silently miss the first bag messages while subscribers connect.

Use alternate topics and frames when required:

```bash
python .agents/skills/gril-calib-validation/scripts/gril_pipeline.py all \
  --record-file RECORD \
  --output-dir OUTPUT \
  --lidar-topic /apollo/sensor/lidar/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --pose-topic /apollo/sensor/gnss/odometry \
  --tf-parent imu \
  --tf-child lidar \
  --scan-lines 16
```

Run individual phases with `setup`, `prepare`, `run`, and `diagnose`. Use
`--skip-submap` on `diagnose` when only signal diagnostics are needed.

### Packaged tools

- `scripts/export_apollo_rosbag.py`: exact Apollo-to-ROS conversion with
  finite-point filtering and relative float32 point time
- `scripts/audit_conversion.py`: counts, sampled-value, ring, gravity, TF, and
  point-time audit
- `scripts/evaluate_trajectories.py`: LiDAR odometry versus independent
  GNSS/INS odometry, aligned by SE(2) without scale
- `scripts/validate_dynamics.py`: derivative filtering, observability, window,
  bias, and synthetic checks
- `scripts/yaw_time_grid.py`: contiguous cross-validated yaw-time profile
- `scripts/build_imu_submap.py`: per-point SE(3) motion-compensated submap and
  local planar thickness
- `scripts/gril_pipeline.py`: pinned setup and end-to-end orchestration

The default sensor configuration is `resources/vanjeelidar16.yaml`. Treat it as
a reviewed Vanjee-16 profile; tune sensor height, excitation thresholds, and
translation bounds for another installation rather than silently reusing them.

### Debugging outputs

The trajectory diagnostic writes:

- `diagnostics/frontend_trajectory.png`
- `diagnostics/frontend_trajectory.yaml`

The submap diagnostic uses independent GNSS/INS poses as the IMU-side odometry,
interpolates SE(3) at every retained point timestamp, applies the estimated
LiDAR-to-IMU transform, and writes:

- `diagnostics/imu_submap/imu_extrinsic_submap.ply`
- `diagnostics/imu_submap/imu_extrinsic_submap_bev.png`
- `diagnostics/imu_submap/submap_metrics.yaml`

Compare thickness only across candidates built with exactly the same record,
pose source, time convention, scan/point strides, crop, voxel size, and
neighborhood settings.

Run it independently for A/B comparison:

```bash
python .agents/skills/gril-calib-validation/scripts/build_imu_submap.py \
  --record-file RECORD \
  --result OUTPUT/run_1/GRIL_Calib_result.txt \
  --output-dir OUTPUT/diagnostics/imu_submap
```

GRIL's native `pcd_save` is useful but is not this physical holdout. Upstream
GRIL accumulates `feats_undistort` using its LiDAR-only frontend pose and exits
immediately after calibration. It does not replay the capture using the final
extrinsic and independent IMU-side odometry.

## First-principles model

GRIL estimates different parameters from different signals:

- rotation and time use LiDAR and IMU angular velocity, plus angular acceleration
- roll, pitch, and vertical translation use ground-plane constraints
- horizontal lever arm uses IMU acceleration and LiDAR odometry derivatives:

  `a_I - a_L = (omega_x * omega_x + alpha_x) * t`

The horizontal translation therefore requires trustworthy LiDAR linear
acceleration, angular velocity, and angular acceleration. A bounded trajectory
or full-rank Jacobian alone is insufficient.

## Frame and parameter conventions

Verify these before running an optimizer:

1. `offset_R_L_I` and the reported rotation map LiDAR coordinates into IMU
   coordinates.
2. The reported `offset_T_L_I` is the LiDAR origin in the IMU frame.
3. GRIL config `trans_IL` is the IMU origin in the LiDAR frame, not the reported
   translation.
4. Convert them using:

   `T_LI = -R_LI * T_IL`

5. Treat the configured transform as an installation prior unless independently
   surveyed. Distance from it is not an accuracy metric.
6. In LiDAR-only initialization (`imu_en=false`), `trans_IL` does not initialize
   the LiDAR odometry frontend; it initializes the later calibration solve.

For a reported prior of yaw `+90 deg` and translation `[0, 0.34, 0.59] m`,
the matching GRIL config seed is approximately `[-0.34, 0, -0.59] m`.

## Mandatory input audit

### Coordinate audit

- Preserve LiDAR points in their native sensor frame.
- Preserve IMU angular velocity and acceleration in their native IMU frame.
- Verify the static LiDAR-to-IMU transform direction independently.
- Check gravity axis and sign from stationary IMU samples.
- Fit or inspect the ground plane; a z-up LiDAR should produce small ground
  roll/pitch and a physically plausible sensor height.
- Validate inferred rings using per-ring elevation distributions. Each ring
  should form a stable, ordered vertical angle.
- Never use a renamed ROS `frame_id` as evidence that numerical vectors were
  transformed.

### Timestamp audit

- Compare record timestamp, header timestamp, point minimum timestamp, and point
  maximum timestamp.
- Confirm point offsets are monotonic and have the expected scan duration.
- Confirm GRIL interprets point time in seconds before converting it to
  `curvature` milliseconds.
- Check scan-start versus scan-end semantics explicitly.
- Report LiDAR and IMU maximum gaps and never differentiate across a gap.

### GRIL source hazards to check

Do not begin calibration until these are verified in the exact GRIL revision:

1. `Forward_propagation_without_imu` must initialize `time_last_scan` when it
   processes the first frame. Otherwise the second-frame propagation interval
   is undefined and repeat runs may differ.
2. Supplied Velodyne point time is valid when the last point time is positive;
   the first point is normally exactly zero. A condition requiring both first
   and last point times to be positive silently rejects valid timing.
3. Count frames using supplied point time versus azimuth-reconstructed time.
   Do not allow unexplained per-frame switching between the two models.
4. Filter non-finite points before Patchwork++.
5. Ensure split records form a continuous sequence; reset at real gaps.
6. When LiDAR states are removed from the front during warm-up or temporal
   alignment, remove the paired ground-plane constraints from the front too.
   Trimming only the state deque silently pairs acceleration with an older
   ground attitude and corrupts gravity compensation.

## Validation ladder

### 1. Conversion validation

Record:

- input and output message counts
- scan duration p0/p50/p100
- point-time monotonic ratio
- supplied-time and fallback-time frame counts
- per-ring elevation p25/p50/p75
- stationary IMU acceleration norm and axis sign
- static transform and its inverse

Reject conversion if units, axes, rings, or timing semantics are ambiguous.

### 2. LiDAR-only frontend validation

Before judging extrinsics:

- plot LiDAR odometry and independent GNSS/INS odometry in one BEV figure
- align with SE(2) rigid alignment only; do not scale
- report ATE/RPE, path length, maximum step, and gap boundaries
- run the same input twice and compare trajectories
- inspect ground height and normal stability

A bounded trajectory is necessary but does not prove derivative quality.

### 3. Raw signal and observability validation

Export aligned LiDAR and IMU states:

- timestamps
- LiDAR pose and velocity
- LiDAR angular velocity
- LiDAR linear and angular acceleration
- IMU angular velocity and acceleration
- ground-to-LiDAR rotation

Report:

- axis-wise RMS and cross-sensor angular-rate correlation
- derivative spectra and high-frequency power
- bias-marginalized singular values
- excitation by time segment

Do not use GRIL's LiDAR-only angular-velocity Hessian as the sole sufficiency
test. Frontend noise can create false x/y excitation.

### 4. Staged solve

Run in this order:

1. coarse angular-rate time correlation
2. rotation-only solve
3. fixed-rotation, fixed-time translation solve
4. optional bounded time refinement
5. full joint solve

At every stage, log initial/final cost, residual percentiles, parameter bounds,
solver failures, and the transform.

### 5. Initialization and repeatability

Use exactly the same data and frontend states:

- same seed twice
- configured translation seed
- zero translation seed
- at least one bounded perturbation

Recommended research gates:

- repeated rotation spread <= `0.2 deg`
- repeated translation spread <= `0.03 m`
- repeated time spread <= `0.001 s`

Failure indicates undefined frontend state, nondeterminism, weak observability,
or competing solution families. Do not hide it with best-of-N selection.

### 6. Derivative necessity test

For horizontal translation, compare:

- GRIL's native differentiated states
- stronger zero-phase low-pass filtering
- a smooth trajectory derivative such as a spline

The candidate must improve more than residual magnitude. Require:

- compatible translations across derivative models
- compatible translations across contiguous time windows
- no accelerometer-bias saturation
- improved held-out acceleration prediction

If filtering reduces high-frequency power but moves the lever arm or increases
window spread, the failure is model mismatch or dynamic delay, not merely
white noise.

### 7. Reference-free first-principles check

Do not use the configured transform for this verdict.

With fixed rotation and timing:

1. fit one constant lever arm and accelerometer bias
2. compare against a bias-only/null model
3. train on contiguous segments and predict held-out segments
4. solve independent sufficiently excited windows

A physically supported rigid lever arm must:

- materially improve held-out acceleration residuals
- remain one solution family across windows
- remain stable when retaining only high-excitation samples

A full-rank, well-conditioned matrix with negligible predictive improvement is
numerically observable but practically unidentifiable.

### 8. Yaw-time ambiguity check

When yaw or time is uncertain, grid or profile them using contiguous holdout
error, not training cost.

Report:

- global cross-validated optimum
- each holdout segment's preferred yaw and time
- parameter ranges within 0.1% and 1% of optimum
- translation spread at the chosen yaw/time

A broad near-optimal valley or segment-dependent optimum is not a unique
physical calibration.

### 9. Synthetic self-consistency

Generate LiDAR acceleration exactly from the GRIL residual using measured
poses, angular states, IMU acceleration, and a known lever arm. The offline
diagnostic solver must recover that lever arm to numerical precision.

Use this to distinguish an incorrect diagnostic implementation from
measurement/model inconsistency.

### 10. Independent physical holdout

Before accepting a corrected GRIL result, require at least one:

- repeated result on another capture with independent motion
- surveyed mounting measurement
- held-out LiDAR/IMU trajectory consistency
- point-cloud motion compensation or map-sharpness improvement

Do not use the same configured transform both as initialization and truth.

## Verdicts

- **accepted**: input contract passes, frontend is repeatable, seeds converge to
  one family, derivative/segment checks generalize, and physical holdout agrees
- **review-only**: repeatable and internally coherent, but independent physical
  holdout or cross-sequence confirmation is missing
- **rejected**: input contract is wrong, runs are unstable, held-out prediction
  degrades, or distinct solution families remain
- **inconclusive**: data integrity is acceptable but the requested parameter is
  not practically identifiable

## Required artifacts

Write stable review outputs:

- `metrics.yaml`
- `diagnostics/input_contract.yaml`
- `diagnostics/frame_and_transform_audit.yaml`
- `diagnostics/frontend_trajectory.yaml`
- `diagnostics/derivative_quality.yaml`
- `diagnostics/staged_ablation.yaml`
- `diagnostics/repeatability.yaml`
- `diagnostics/holdout.yaml`
- trajectory and derivative plots
- raw GRIL logs and exact source revision

## Known 0827 lesson

In the validated Vanjee conversion:

- `ring=index%16` produced ordered beam elevations near `+13.9` to `-16.0 deg`
- scan point time spanned about `99.983 ms` and was monotonic
- the original GRIL condition rejected valid scans whose first point time was
  exactly zero
- the LiDAR-only CV model did not initialize `time_last_scan`
- fixing both issues changed same-data repeatability from multiple solution
  families to about `0.083 deg`, `0.020 m`, and `0.067 ms` maximum spread across
  repeated and zero-seed runs
- GRIL's state-trimming paths must also keep the ground-constraint deques
  synchronized; the upstream code removes LiDAR states from the front without
  removing their paired ground states

Therefore audit coordinate and timing implementation before blaming the GRIL
derivative formulation.
