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
- Separate **Data Sufficiency** (valid state count, continuity, tilt), **Motion Excitation** (path length, bidirectional yaw, angular/linear acceleration), and **Parameter Observability** (rotation Hessian, 2D/3D lever-arm singular values + boundary hits, time-offset angular-acceleration energy) rather than relying on GRIL's single `pct_z > 0.99` stop condition.

## Do NOTs
- Do not differentiate motion across a record gap.
- Do not infer temporal bias from publish order.
- Do not blame scan-to-map until the input contract and LiDAR frontend are independently checked.
- Do not accept a result solely because Ceres converged, `pct_z > 0.99` triggered early stop, or a map appears bounded.
- Do not alter frozen GRIL defaults to enforce quality thresholds; keep stabilization flags opt-in and quality gates strictly post-run (`lidar2imu evaluate-quality-gates`).

## Sources
- General review ladder: `docs/calibration_review_guide.md`
- Validated repository conclusions: `context/knowledge_base/validated_conclusions.md`
- Open verification points: `context/knowledge_base/verification_points.md`
- Timing diagnosis: `context/timing_sync_context.md`
- LiDAR-to-IMU historical baseline/staged evidence: `context/lidar2imu_context.md`
- Current LiDAR-to-IMU workflow: `.agents/skills/gril-calib-validation/SKILL.md`
- Scan-to-map history: `context/scan2map_context.md`
