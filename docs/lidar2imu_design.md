# LiDAR-to-IMU current design

This document records the current implementation for future iteration work.

## 1. Design goal

Keep the system split into two independent layers:

1. **Algorithm layer**: estimate `x, y, z, yaw, roll, pitch`
2. **Evaluation layer**: keep metrics and diagnostics stable while the algorithm
   evolves

This separation is intentional: later changes should mostly modify the
conversion logic, sample selection, losses, or residual weights, without
changing how results are judged.

## 2. Implemented commands

### `lidar2imu-calibrate`

Runs the staged solver from standardized samples.

Main files:

- `lidar2imu/cli.py`
- `lidar2imu/pipeline.py`
- `lidar2imu/algorithms.py`
- `lidar2imu/metrics.py`
- `lidar2imu/io.py`

### `lidar2imu-convert-record`

Builds standardized samples from Apollo record data, then optionally runs the
solver.

Main file:

- `lidar2imu/record_converter.py`

## 3. Standardized input schema

The solver consumes:

- `parent_frame`
- `child_frame`
- `initial_transform`
- `ground_samples`
- `motion_samples`

### Ground sample meaning

- `lidar_plane_normal`
- `lidar_plane_offset`
- `imu_gravity`
- `imu_ground_height`
- `weight`
- `sync_dt_ms`

### Motion sample meaning

- `imu_delta`
- `lidar_delta`
- `weight`
- `sync_dt_ms`

The converter writes this schema to `standardized_samples.yaml`.

## 4. Algorithm stages

### Stage A: ground orientation

Estimate `roll/pitch` by aligning:

- LiDAR ground normal
- IMU gravity direction

Residual idea:

- minimize the cross-product error between predicted up and gravity-derived up

### Stage B: ground translation prior

Use `imu_ground_height` as a practical height prior.

Current role:

- mainly stabilizes `z`
- also exposes how weak the height-only geometry is through observability

### Stage C: motion rotation

Solve `yaw` with the hand-eye rotation residual:

- `R_A * R_IL = R_IL * R_B`

### Stage D: motion translation

Solve translation with the standard hand-eye translation equation:

- `(R_A - I) * t_IL = R_IL * t_B - t_A`

### Stage E: joint refinement

Run robust least squares over:

- ground normal residuals
- ground height residuals
- motion rotation residuals
- motion translation residuals

## 5. Evaluation design

The solver writes:

- `metrics.yaml`
- `diagnostics/algorithm.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/observability.yaml`
- `diagnostics/manifest.yaml`

### Coarse metrics

Used as gate metrics for iteration:

- `ground_normal_angle_p95_deg`
- `ground_height_residual_p95_m`
- `motion_rotation_residual_p95_deg`
- `motion_translation_residual_p95_m`
- `motion_angular_excitation_p95_deg`
- `motion_registration_fitness_p05`
- `motion_registration_inlier_rmse_p95`
- `left_turn_count`
- `right_turn_count`
- `turn_balance_ratio`
- `joint_condition_number`
- `statuses`
- `vehicle_motion_assessment`

### Fine metrics

Used for debugging and model iteration:

- per-stage residual distributions
- per-stage observability / singular values
- per-sample ground diagnostics
- per-sample motion diagnostics

## 6. Record conversion design

Current default topic mapping:

- LiDAR: `/apollo/sensor/lslidar_main/PointCloud2`
- Pose / IMU-side motion: `/apollo/localization/pose`
- IMU topic: `/apollo/sensor/gnss/imu`
- TF source: `/tf_static` + TF graph

### Current conversion flow

1. Resolve `T_imu_lidar` from TF, or load it from `--initial-transform`.
2. Resolve `T_localization_imu` from TF.
3. Sample LiDAR frames for ground extraction.
4. Fit a ground plane from each selected cloud.
5. Build `imu_gravity` from pose orientation by default.
6. Build motion candidates from localization-pose relative motion.
7. Assign motion candidates into timeline windows.
8. Gate weak windows before registration.
9. Inside each valid window, prefer candidates with enough angular excitation and
   score them by stride-normalized information so very long spans do not dominate.
10. Run LiDAR-to-LiDAR GICP on window-selected motion pairs and reject pairs whose
    registration fitness is too low.
11. Export normalized samples.

If a bag has no LiDAR-to-parent TF at all, the converter can also run with
`--identity-initial-transform`, but that mode is exploratory only and should not
be treated as a production prior.

### Why motion selection was changed

The first version used uniform sampling and failed on the real bag because only
one motion pair had enough angular excitation.

The current version:

- tries multiple frame strides
- groups motion candidates into windows
- gates weak windows
- prefers local candidates with enough angular excitation
- uses stride-normalized scoring so large spans do not dominate
- keeps only candidates that also pass registration gating

This follows the same practical idea used in `lidar2lidar`: evaluate candidate
pairs first, then spend optimization effort on the useful ones.

## 7. Latest real-bag validation

Validated on:

- `/home/wfh/01code/apollo-lite/data/bag/record_data_0402`

Generated artifacts:

- `/tmp/lidar2imu-realbag-v2/standardized_samples.yaml`
- `/tmp/lidar2imu-realbag-v2/conversion_diagnostics.yaml`
- `/tmp/lidar2imu-realbag-v2/calibration/metrics.yaml`

### What worked

- conversion completed successfully
- ground sample extraction was stable
- motion sample extraction ran end-to-end
- calibration completed and produced structured diagnostics

### What the metrics say

The bag is still **not a production-quality `x/y/yaw` calibration bag**, but the
data-layer diagnosis is now clearer.

Round summary on `record_data_0402`:

1. **Unfiltered baseline**
   - `ground_normal_angle_p95_deg ≈ 1.74`
   - `motion_rotation_residual_p95_deg ≈ 0.80`
   - `motion_translation_residual_p95_m ≈ 2.32`
   - `left_turn_count = 8`, `right_turn_count = 0`
   - interpretation: one low-quality motion pair can badly distort translation,
     and the bag is one-sided in yaw excitation.
2. **Filtered default (`--min-registration-fitness 0.55`)**
   - one low-fitness motion pair is rejected
   - `motion_rotation_residual_p95_deg ≈ 0.30`
   - `motion_translation_residual_p95_m ≈ 0.43`
   - `motion_registration_fitness_p05 ≈ 0.63`
   - `vehicle_motion_assessment.recommendation = z_roll_pitch_priority`
3. **Stricter filter (`--min-registration-fitness 0.70`)**
   - three motion pairs are rejected, leaving `5` motion samples
   - registration quality improves again, but `right_turn_count` is still `0`
   - recommendation remains `z_roll_pitch_priority`
4. **`gravity-source imu` trial**
   - `ground_selected = 0`
   - most candidate planes flip to about `178 deg` against expected up
   - interpretation: this bag does not provide a stable IMU-gravity extraction
     path, so pose-derived gravity remains the safer default

Interpretation:

- ground constraints are usable when gravity is derived from localization pose
- motion-pair quality filtering is necessary and should stay enabled
- `x/y/yaw` remain weak because the bag only excites left turns
- `z/roll/pitch` are the trustworthy part of the result on this bag
- the evaluation layer now exposes this directly through registration-quality
  gates, turn-balance metrics, and `vehicle_motion_assessment`

### Additional exploratory bag: `/mnt/synology/raw-data/2026-04-13-06-54-28`

This bag contains:

- `/apollo/sensor/seyond/front/PointCloud2`
- `/apollo/localization/pose`
- `/apollo/sensor/gnss/imu`
- `/apollo/sensor/gnss/corrected_imu`

but only these TF edges:

- `world -> imu`
- `world -> localization`

So it does **not** contain a LiDAR-to-IMU static transform.

Exploratory rounds:

1. **Identity fallback + default stride (`5`)**
   - only `1/8` motion pairs survived the registration-fitness gate
   - failure reason: the candidate builder picked very large stride-40 motion
     pairs first, and most registrations fell below the fitness threshold
2. **Identity fallback + local motion pairs (`--motion-frame-stride 1`, `--min-registration-fitness 0.45`)**
   - `8/8` motion pairs survived
   - registration quality became strong: `fitness_p05 ≈ 0.78`,
     `inlier_rmse_p95 ≈ 0.12`
   - joint solve completed, but the final yaw drifted to about `62.7 deg`
3. **Identity fallback + moderate motion pairs (`--motion-frame-stride 2`, `--min-registration-fitness 0.35`)**
   - `8/8` motion pairs survived
   - motion residuals stayed reasonable, but ground height degraded and final
     yaw drifted further to about `78.9 deg`
4. **Corrected IMU gravity trial**
   - `ground_selected = 0`
   - candidate plane normals again flipped to about `178 deg`

Interpretation:

- pose-derived gravity remains the usable choice
- smaller local motion pairs are the right data-layer strategy for this bag
- the best exploratory setting is currently stride `1` with fitness `0.45`
- however, because the bag lacks a true LiDAR-to-IMU prior and only excites one
  turning direction (`right_turn_count = 8`, `left_turn_count = 0`), the final
  extrinsics should **not** be accepted directly

### Weak-planar solver policy validation

The solver now supports:

- `--planar-motion-policy free`
- `--planar-motion-policy freeze_xyyaw`
- `--planar-motion-policy auto`

`auto` currently switches to `freeze_xyyaw` when:

- turn balance is one-sided, or
- the motion-yaw stage is degenerate

Validation on `record_data_0402` with `--min-registration-fitness 0.55`:

1. **Free 6DoF solve**
   - recommendation remains `z_roll_pitch_priority`
   - `delta_to_initial ≈ 2.17 m / 1.30 deg`
   - final yaw drifts to about `-0.63 deg`
2. **Auto planar policy**
   - applied policy: `freeze_xyyaw`
   - recommendation remains `z_roll_pitch_priority`
   - `delta_to_initial ≈ 0.016 m / 1.15 deg`
   - final yaw stays at the initial `0 deg`

Validation on the Synology front-LiDAR bag with the provided
`imu_lidar_front_extrinsics.yaml` prior:

1. **Free 6DoF solve**
   - recommendation remains `recollect_data`
   - `delta_to_initial ≈ 0.343 m / 2.65 deg`
   - final yaw drifts to about `92.5 deg`
2. **Auto planar policy**
   - applied policy: `freeze_xyyaw`
   - recommendation remains `recollect_data`
   - `delta_to_initial ≈ 0.010 m / 0.73 deg`
   - final yaw stays at the initial `90 deg`

Interpretation:

- weak-planar bags should not be allowed to spend `x/y/yaw` freedom just to reduce
  residuals
- the new policy makes that trade-off explicit: keep planar components near the
  trusted prior and continue refining `z/roll/pitch`

The tuning helper `lidar2imu-tune-record` now also sweeps
`--planar-motion-policy` and ranks trials by recommendation, then drift to the
initial prior, then warnings and residuals. This keeps weak-planar free solutions
from outranking conservative prior-preserving runs only because they have a smaller
motion residual.

### Window + gate validation

The converter now uses a windowed motion-selection policy instead of one global
candidate ranking.

Validation on `record_data_0402` with `--min-registration-fitness 0.55`:

- `8` motion windows
- `6` valid windows
- `5` selected motion samples
- first selected candidate uses `frame_stride=10`
- first selected candidate has `pose_rotation_deg ≈ 1.15`,
  `registration_fitness ≈ 0.787`

Validation on the Synology front-LiDAR bag with the provided prior:

- `8` motion windows
- `6` valid windows
- `6` selected motion samples
- first selected candidate uses `frame_stride=2`
- first selected candidate has `pose_rotation_deg ≈ 1.63`,
  `registration_fitness ≈ 0.956`

Interpretation:

- windowing makes the data-layer decision explicit
- gating weak windows reduces the chance that abnormal local segments enter the
  final motion set
- stride-normalized scoring avoids reintroducing the old long-span domination issue

## 8. Current limitations

1. The converter still treats localization pose as the default source of
    gravity and IMU-side motion.
2. The current auto policy is still prior-preserving rather than evidence-growing:
   it freezes `x/y/yaw` instead of proving they are observable.
3. The current bags still lack explicit left/right turn balance.
4. Bags with no LiDAR-to-parent TF still need either an explicit prior or a
    stronger acceptance rule than the current identity fallback.

## 9. Recommended next iteration

1. Keep `--min-registration-fitness 0.55` as the default conversion guard.
2. Add explicit observability gates for one-sided turning before accepting
   planar outputs.
3. Add repeatability checks across multiple bags and across perturbed initial
   transforms.
4. Validate on a bag with stronger left/right turn coverage.
5. When a bag lacks static TF, prefer `--initial-transform` over identity
   fallback before drawing any final calibration conclusion.
