# Troubleshooting

## When
Read when calibration diverges, produces inconsistent transforms, passes fitness but looks wrong, or runs too slowly.

## Rules
- Triage in order: input contract, coordinates/time, frontend trajectory, observability, staged solve, repeatability, holdout.
- Compare against a null/baseline model and independent data; do not compare only against the configured seed.
- Inspect explicit skip reasons, coarse/fine metrics, information diagnostics, and visualizations.
- For timing failures, verify acquisition/header/point timestamps and gap boundaries before tuning an optimizer.
- For GRIL, run the executable GRIL validation skill rather than recreating conversion or diagnostic scripts.
- For GRIL run-to-run divergence, isolate both asynchronous `ikd-Tree` background rebuilds and Eigen 3.3.7's `-fopenmp` multi-threaded GEMM 1-ULP race (`--ikd-rebuild sync`, which locks synchronous tree rebuilds and single-thread Eigen/OpenMP GEMM) before diagnosing correspondence or solver issues.
- For GRIL start/end trajectory spurs ("首尾毛刺"), mid-window vertical "banana" drift, or seconds-scale time-lag jumps (`xcorr_temporal_init`), inspect equal-count sub-frame cuts on blind-sector corner LiDARs, pseudo-ground EKF frame mixing, feature starvation near close walls, and unnormalized cross-correlation lags (`lidar2imu/evaluation/quality_gates.py`).
- For `raw_imu` captures (as opposed to `corrected_imu`), verify physical units/scale factors from hardware docs before trusting pre-scaled canonical arrays, check for a genuine stationary segment via LiDAR motion + IMU variance before subtracting initial gyro bias (never subtract during motion), and lock one fixed cross-epoch offset `H` from `measurement_time` vs `header`.
- When GRIL multi-window `T_LI.y` or pitch drifts despite continuous frontend states and bidirectional excitation, check for: (1) high-speed cold-restart slicing transients, (2) `Gril_Calib.h` Butterworth DC-gain inflation (`1.0421` / `1.0860`) amplified by `skip_secondary_acc_norm: true`, (3) `R_GL * STD_GRAV` missing `R_LL0` in `Linear_acc_Rot_Cost_without_Gravity` and `R_IL^T * R_GI^T * R_GL` order in `Ground_Plane_Cost_IL`, and (4) `±0.01 m/s²` `bias_aL` bound saturation (`Gril_Calib.cpp:568-569`).
- Separate **Data Sufficiency** (valid state count, continuity, tilt), **Motion Excitation** (path length, bidirectional yaw, angular/linear acceleration), and **Parameter Observability** (rotation Hessian, 2D/3D lever-arm singular values + boundary hits, time-offset angular-acceleration energy) rather than relying on GRIL's single `pct_z > 0.99` stop condition.
- When GNSS/INS enables output lever-arm offset (`ant2outposoffset`, shifting `gnss/odometry` and `localization/pose` to the rear axle while `corrected_imu` / `raw_imu` `(w, f)` stay at the IMU origin), `lidar2imu` calibration still yields `T_{IMU<-LiDAR}` at the physical IMU origin; pass `--imu-in-pose-xyz` / `--auto-compensate-pose-offset` (with `--batch-trace` / `--result`) to `build_imu_submap.py` and `evaluate_trajectories.py`, and never leave `localization -> imu = 0` in `/tf_static`.

## Do NOTs
- Do not differentiate motion across a record gap.
- Do not infer temporal bias from publish order.
- Do not slice GRIL validation windows mid-motion with a cold frontend restart, or treat overlapping/subset windows and training-inclusive segments as independent holdouts.
- Do not assume `bias_aL` is valid without checking `±0.01 m/s²` boundary saturation or `skip_secondary_acc_norm` Butterworth gain interaction.
- Do not blame scan-to-map until the input contract and LiDAR frontend are independently checked.
- Do not accept a result solely because Ceres converged, `pct_z > 0.99` triggered early stop, or a map appears bounded.
- Do not alter frozen GRIL defaults to enforce quality thresholds; keep stabilization flags opt-in and quality gates strictly post-run (`lidar2imu evaluate-quality-gates`).
- Do not treat raw `SE(2)` trajectory ATE or uncompensated `gnss/odometry` submap blur as `lidar2imu` failure when GNSS uses `ant2outposoffset` without checking `lever_arm_compensated` and `pose_origin_audit`.

## Sources
- General review ladder: `docs/calibration_review_guide.md`
- Validated repository conclusions: `.agents/knowledge/validated_conclusions.md`
- Open verification points: `.agents/notes/verification_points.md`
- Timing diagnosis: `.agents/knowledge/timing_synchronization.md`
- LiDAR-to-IMU historical baseline/staged evidence: `.agents/notes/lidar2imu/lidar2imu_history.md`
- Current LiDAR-to-IMU workflow: `.agents/skills/gril-calib-validation/SKILL.md`
- Scan-to-map history: `.agents/notes/lidar2lidar/scan2map_history.md`
