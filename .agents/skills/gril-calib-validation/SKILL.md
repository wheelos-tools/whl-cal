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

### Runtime and input paths

The native GRIL algorithm accepts a validated `CanonicalDataset` and does not
require ROS. The currently registered offline data adapters are:

| Input type | Adapter | ROS runtime |
| --- | --- | --- |
| Apollo `.record` | `CoreApolloRecordAdapter` (`record`) | not required |
| ROS1 `.bag` | `Rosbag1Adapter` using `rosbags` | not required |
| canonical dataset | dataset reader; extraction is skipped | not required |

Live ROS topic subscription is not implemented. The frozen ROS-GRIL reference
workflow below is a separate A/B oracle and requires Docker to run its ROS
Noetic container. Do not make ROS a dependency of the canonical schema, native
runner, or solver when adding an input adapter.

For data-adapter development, install the Python ingress extra:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[lidar2imu]"
```

This installs offline bag parsing and data-audit tools, but does not install
ROS or compile the native GRIL executables. Apollo record ingress is provided
by the C++ `record` adapter and requires building `//:record_ingress` from
`tools/gril_record_core_reader` with Bzlmod and a local `apollo-lite/pkgs/core`
checkout. The native source-build toolchain and adapter command are documented
in [`docs/lidar2imu_quickstart.md`](../../../docs/lidar2imu_quickstart.md).
The C++ ingress must pass canonical-dataset equivalence before calibration;
the 0827 run used about 2.0 GB peak RSS. The exact
`wheelos_core` revision and regression artifacts are recorded in the
[GRIL `wheelos_core` baseline](../../../docs/lidar2imu_wheelos_core_baseline.md);
that baseline is not calibration-accuracy evidence.

### Frozen reference workflow prerequisites

The complete frozen-reference and physical-diagnostics workflow additionally
requires Docker, `git`, Apollo records containing LiDAR, IMU, static TF, and
independent GNSS/INS odometry, plus the native build/runtime for native
candidate phases.

Run the complete workflow from the repository root:

```bash
python .agents/skills/gril-calib-validation/scripts/gril_pipeline.py all \
  --record-file /data/capture.record.00000 \
  --record-file /data/capture.record.00001 \
  --output-dir outputs/gril/capture01
```

The command:

1. clones the pinned GRIL revision into a patch/config-keyed subdirectory of
   `.cache/gril-validation`
2. applies the validation and trace patches, then rejects unrecorded source
   changes rather than repairing or reusing a contaminated checkout
3. builds the supplied Docker image and deterministic GRIL frontend, recording
   source/config hashes, catkin build-output hash, Dockerfile hash, and
   immutable local image ID
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
Reference runs verify the recorded source, catkin build output, and image
identity before execution; keep the matching local Docker image and
`reference-environment-*.yaml` together to rerun the same environment.

### Data-ingress / algorithm boundary

Keep format-specific decoding inside `lidar2imu.adapters`. Register each offline
input name in `ADAPTERS`; CLI input choices are generated from that registry.
The adapter must produce a validated `CanonicalDataset`, preserving source
provenance in `source_type` and metadata without using those fields to select
or alter algorithm behavior. The canonical arrays carry:

- LiDAR frame ID, scan timestamps, scan offsets, native-frame XYZ, intensity,
  ring indices, and per-point time in seconds relative to each scan timestamp
- IMU frame ID, timestamps, angular velocity, and linear acceleration in the
  native IMU frame
- available static transforms and source metadata

Do not interpolate, filter, transform, reorder, or otherwise “repair” samples
in a format adapter unless that operation is already part of the audited
source-to-canonical contract. Reject ambiguous units, coordinate frames,
point-time references, ring mappings, and IMU clock provenance rather than
guessing. Keep extraction, algorithm, and evaluation separate.

Before a new adapter is used for calibration:

1. convert the same capture through the new adapter and a trusted adapter;
2. run `lidar2imu compare-datasets` and require matching counts, ordering,
   timestamps, frames, ring IDs, point timing, and numeric signals at justified
   tolerances;
3. inspect `input_contract.yaml` and resolve any differences as adapter or
   input-contract issues before invoking native GRIL;
4. confirm the native dataset boundary and algorithm configuration are
   unchanged; do not modify the C++ GRIL algorithm to accommodate a file format.

The Weilan 0827 C++ record ingress and its trusted canonical conversion are a
known regression fixture: both contain 1,375 scans, 35,239,034 finite points,
and 13,741 IMU samples. Their canonical arrays and static transforms compared
exactly in the validated environment. This checks input conversion only; it is
not evidence of calibration accuracy or physical acceptance.

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

For Apollo records, choose the IMU topic by timestamp provenance, not by name.
On the 2026-05-06 capture, `/apollo/sensor/gnss/imu` has no usable
`measurement_time` and its header is host-publish time; use
`/apollo/sensor/gnss/corrected_imu`, whose header is converted INS measurement
time, for calibration timing. The exporter and C++ record adapter unwrap its
nested `imu` payload. Verify this policy against each capture before use.

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
- `scripts/profile_gril_batch_timing.py`: angular-rate timing profile over
  actual GRIL state timestamps and contiguous windows
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

The submap diagnostic uses independent GNSS/INS poses as the reference
body-frame odometry, interpolates SE(3) at every retained point timestamp,
applies the estimated LiDAR-to-IMU transform, and writes:

- `diagnostics/imu_submap/imu_extrinsic_submap.ply`
- `diagnostics/imu_submap/imu_extrinsic_submap_bev.png`
- `diagnostics/imu_submap/submap_metrics.yaml`

**`ant2outposoffset` (INS output lever-arm) caution:** if GNSS/INS shifts
`/apollo/sensor/gnss/odometry` and `/apollo/localization/pose` to the rear axle
while `corrected_imu` / `raw_imu` `(w, f)` remain at the physical IMU origin,
`lidar2imu` calibration still yields `T_{IMU<-LiDAR}` at the IMU origin.
Without lever-arm compensation, naive `SE(2)` trajectory alignment and
`gnss/odometry` submaps will exhibit meter-scale turn errors (e.g. `2.63 m`
apparent ATE on Zhongce). Pass `--result` (and optionally `--imu-in-pose-xyz`
or `--imu-in-pose-yaml`) to `evaluate_trajectories.py` to inspect
`lever_arm_compensated` metrics and `ant2outposoffset_detected`, and pass
`--imu-in-pose-xyz` / `--imu-in-pose-yaml` (or `--batch-trace ...
--auto-compensate-pose-offset`) to `build_imu_submap.py`.

Compare thickness only across candidates built with exactly the same record,
pose source, `pose-to-imu` offset, time convention, scan/point strides, crop,
voxel size, and neighborhood settings.

Run it independently for A/B comparison:

```bash
python .agents/skills/gril-calib-validation/scripts/build_imu_submap.py \
  --record-file RECORD \
  --result OUTPUT/run_1/GRIL_Calib_result.txt \
  --batch-trace OUTPUT/run_1/GRIL_batch_trace_v1.txt \
  --auto-compensate-pose-offset \
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
For dual opposite-corner LiDARs mounted outward at 45° on an Apollo RFU `imu`
frame (+X right, +Y forward, +Z up) with FLU LiDAR axes, `left_front` (`+135°`
yaw in `imu`) and `right_back` (`-45°` yaw in `imu`) differ by **180°** in yaw
(not 90°), while sharing a physical common clock to `corrected_imu`.

## Mandatory input audit

### Coordinate audit

- Preserve LiDAR points in their native sensor frame.
- Preserve the source LiDAR `frame_id` in exported messages and verify it
  matches the record; a topic name or renamed frame is not a coordinate
  conversion.
- Preserve IMU angular velocity and acceleration in their native IMU frame.
- Resolve the static LiDAR-to-IMU transform through the static TF graph when it
  is represented by multiple edges, and verify its direction independently.
- Check gravity axis and sign from stationary IMU samples.
- Fit or inspect the ground plane; a z-up LiDAR should produce small ground
  roll/pitch and a physically plausible sensor height.
- Determine the sensor's actual channel count before inferring rings. Validate
  inferred rings using separated, stable, ordered per-ring elevation bands;
  ordered medians alone do not validate the mapping.
- Never use a renamed ROS `frame_id` as evidence that numerical vectors were
  transformed.

### Timestamp audit

- Compare record timestamp, header timestamp, point minimum timestamp, and point
  maximum timestamp.
- Verify IMU clock provenance. Do not treat a host-publish header as sensor
  time; use the capture's verified measurement-time topic and preserve its
  angular velocity and acceleration values.
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
7. Asynchronous `ikd-Tree` background rebuilds (`pthread_create`) cause
   macroscopic non-deterministic sub-scan map swaps across reruns, and
   `${PCL_DEFINITIONS}` (`-fopenmp`) activates Eigen 3.3.7's multi-threaded GEMM
   (`Parallelizer.h:kMinTaskSize=50000`) when `effective_feature_count >= 174`,
   introducing a 1-ULP (`~1e-16`) rounding race over multi-minute runs. Use opt-in
   `--ikd-rebuild sync` (which disables async `ikd-Tree` threads and locks
   single-threaded Eigen/OpenMP GEMM) when bitwise frontend repeatability is
   required, while leaving default `async` untouched for golden reference A/B tests.
8. Equal-point-count `cut_frame` splitting on corner LiDARs with occluded cab/body
   sectors creates irregular sub-frame durations and tiny sparse tail cuts; pair
   with `kImuHz=200` `dt=0` resets, parked vibration can trip `pos_end.norm() > 0.05`
   prematurely ("起点毛刺"). Use opt-in `--cut-mode equal_time --min-cut-points 400`,
   `--ahrs-imu-hz 100 --ahrs-dt-mode continuous`, and
   `--motion-start-min-speed-mps` / `--motion-start-min-omega-rad-s`.
9. Upstream `LidarOdometry.cpp` pseudo-ground EKF rows mix `ground_lidar` and
   world `state.rot_end` / `state.pos_end`, inducing mid-window vertical "banana"
   tilt and catastrophic Z runaway when close walls starve point-to-plane features
   ("终点毛刺"). Use opt-in `--odom-ground-mode disabled` and
   `--min-effective-features 30` (skipping degenerate updates while keeping state
   and ground deques synchronized).
10. Upstream `data_sufficiency_assess` (`pct_z > 0.99`) measures scaled angular
    velocity energy only and stops prematurely (~12–13% of long captures) or on
    corrupted Z spikes; use `--min-accum-duration-s`, `--max-accum-duration-s`,
    and `--eof-policy calibrate` for controlled window experiments, and separate
    **Data Sufficiency**, **Motion Excitation**, and **Parameter Observability**
    post-run via `lidar2imu evaluate-quality-gates`.
11. Unbounded unnormalized `xcorr_temporal_init` and unbounded `time_lag_2` can
    jump to multi-second false lags (`+48 s`, `-2.5 s`); opt-in
    `calibration.max_xcorr_lag_s` (bounded zero-mean Pearson correlation) and
    `calibration.max_time_lag_2_s` keep coarse and fine lag estimation physical.
12. Differentiated 30 Hz LiDAR `ang_vel`/`ang_acc` in
    `Linear_acc_Rot_Cost_without_Gravity` causes severe Errors-in-Variables (EIV)
    regression dilution (~90% lever-arm shrinkage when unbounded); opt-in
    `calibration.use_imu_angular_in_acc_cost: true` uses `R_IL * IMU_ang_vel` and
    `R_IL * IMU_ang_acc` while propagating exact analytical Jacobians wrt `R_IL`,
    and `calibration.skip_secondary_acc_norm: true` avoids re-scaling dynamic IMU
    acceleration from moving initial samples.
13. `Gril_Calib.h` hardcoded 4-decimal `Butterworth` coefficients (`sum(b)=0.0049`,
    `sum(a)=0.0048`) produce a `1.0421` (+4.21%) DC gain per `zero_phase_filt`
    pass and `1.0860` (+8.60%) after two passes (`Lidar.linear_acc`,
    `IMU.ang_acc`). Enabling `skip_secondary_acc_norm: true` leaves IMU gravity
    inflated by `+4.21%` (`~10.22 m/s²`), which couples into `pitch`, `T_LI.y`,
    and `±0.01 m/s²` `bias_aL` saturation.
14. `Linear_acc_Rot_Cost_without_Gravity` adds body-frame `R_GL * STD_GRAV`
    (`FullFrontend.cpp` `FromTwoVectors(e3, normal_lidar)`) without rotating by
    `R_LL0` into world frame `L_0`, and `Ground_Plane_Cost_IL` uses
    `R_IL^T * R_GI^T * R_GL * e3` despite `FusionAhrs` `R_GI` mapping `I -> G`
    (instead of `R_GI * R_IL^T * R_GL * e3`). Combined with `--odom-ground-mode
    disabled` world-tilt drift and `Gril_Calib.cpp`'s hard `±0.01 m/s²`
    `bias_aL` clamp, horizontal lever-arm estimates drift across windows even on
    continuous bidirectional trajectories.
15. Never split validation windows mid-motion with a cold frontend restart
    (`ikd-Tree` + `AHRS` from zero at `v >> 0` creates multi-m/s² startup
    transients), and verify loop-direction balance (front vs rear loops) before
    using a segment in multi-window consistency gates; do not reuse overlapping
    windows or training-included traces as independent holdouts.

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
- for the native batch trace, run `evaluate_trajectories.py --batch-trace
  GRIL_batch_trace_v1.txt` with the original records to plot the LiDAR path
  and relative yaw drift against INS odometry in four contiguous windows.
  INS odometry shares the input IMU chain: use it only to audit frontend
  motion, never as an independent extrinsic ground truth.
- Do not interpret an uncompensated LiDAR-vs-INS path-length ratio as frontend
  speed error: their origins differ, and turns contribute lever-arm motion.
  Prior to a physical trajectory verdict, account for the transform, while
  treating any configured transform only as a sensitivity prior.
- run the same input twice and compare trajectories
- inspect ground height and normal stability

A bounded trajectory is necessary but does not prove derivative quality.
The `odometry` trace's residual field is only meaningful after confirming the
binary computes the mean absolute accepted point-to-plane distance (see
`LidarOdometry.cpp`); earlier builds wrote zero or `nan` because the accumulator
was never incremented. Compare the actual executable digest, not just the
source or library build target. Treat even a corrected in-sample residual as
diagnostic: on the 2026-05-06 corrected-IMU capture, identical-input reruns of
the instrumented frontend produced different batch traces and terminal
solutions. Resolve frontend repeatability before using the residual to rank
calibrations or testing optimizer changes.

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
For the native GRIL batch trace, first use `profile_gril_batch_timing.py` to
screen whether timestamp-aware angular-rate norm correlation has a supported
peak inside the plausible clock window. Treat a weak, boundary, or
window-dependent peak as insufficient time observability, not as a measured
time offset.
Compare `--signal yaw_z` without smoothing against `--signal yaw_z
--smoothing-s 0.1` on the same frozen trace to test whether high-frequency
frontend noise explains a failure; retain the default norm run as the
three-axis baseline. The tool records each window's peak, zero-offset
correlation, signal spread, and explicit failure reasons. Smoothing can
strengthen correlation without making the estimated time or extrinsic valid.

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
- `quality_gates.yaml` (via `lidar2imu evaluate-quality-gates`, separating
  `data_sufficiency`, `motion_excitation`, `parameter_observability`,
  `repeatability`, `seed_sensitivity`, `multi_window`, and `holdout` into
  explicit `converged_and_verified` / `degraded_unobservable` / `failed` states)
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
