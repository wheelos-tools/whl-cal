---
audience: user
stability: stable
P26-05-25
---

# Calibration run, review, and visualization guide

This guide explains how to do a minimal validation run, how to read the output
artifacts, how to judge the metrics, and which visualization files matter for
each calibration module.

## 1. Shared review order

Across `lidar2lidar`, `lidar2imu`, and `lidar2camera`, use the same review order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `metrics.yaml`
4. `diagnostics/acceptance_report.yaml`
5. `diagnostics/visualization_index.yaml`

Why this order matters:

- `standardized_data.yaml` tells you what the run actually consumed
- `data_quality.yaml` tells you whether the bag was good enough to trust
- `metrics.yaml` tells you the numeric conclusion
- `acceptance_report.yaml` tells you which gates passed or failed
- `visualization_index.yaml` tells you which images / CSV / PLY files to open

For `camera`, the same pattern lives under `calibration_*_diagnostics/`.

## 2. Minimal validation commands by module

| Module | Minimal validation command | What it proves quickly |
| --- | --- | --- |
| `camera` | `python camera/intrinsic.py --config tmp_config.yaml --images-dir /path/to/images --pattern-size 4,3` | confirms the checkerboard dataset and intrinsic pipeline are usable |
| `lidar2camera` | `PYTHONPATH=. python3 tools/run_lidar2camera_smoke.py --poses 5` | confirms the runtime and optimizer behave on a synthetic reference case |
| `lidar2lidar` | `lidar2lidar-auto --record-path /path/to/record --conf-dir lidar2lidar/conf --output-dir outputs/lidar2lidar/run` | confirms the record can be parsed and a baseline automatic run can finish |
| `lidar2imu` | `lidar2imu-convert-record --record-path /path/to/record --output-dir outputs/lidar2imu/run01 --profile baseline --calibrate` | confirms the record can be converted and the staged solver can finish |

These are not release procedures. They are the shortest meaningful end-to-end
checks.

## 3. How to read the outputs

### Camera intrinsic

Read in this order:

1. `calibration_*/..._diagnostics/data_quality.yaml`
2. `calibration_*/..._diagnostics/per_view_reprojection.csv`
3. `calibration_*/..._diagnostics/image_coverage_heatmap.png`
4. `comparison_view.png`
5. the calibration YAML itself

Primary signals:

- average reprojection error
- per-view reprojection long tail
- image coverage breadth
- radial monotonicity

### LiDAR↔Camera

Read in this order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/extraction.yaml`
3. `diagnostics/geometry_resolution.csv`
4. `metrics.yaml`
5. `diagnostics/acceptance_report.yaml`
6. `diagnostics/image_coverage_heatmap.png`
7. `diagnostics/pose_diversity_plot.png`

Primary signals:

- accepted pair ratio
- final RMS and per-pose RMS
- holdout repeatability
- image coverage
- depth / tilt diversity
- board geometry resolution success

### LiDAR-to-LiDAR

Read in this order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `diagnostics/workflow.yaml`
4. `diagnostics/scene_sufficiency.yaml`
5. `metrics.yaml`
6. `diagnostics/acceptance_report.yaml`
7. `diagnostics/visual_evaluation.yaml`
8. colored merged clouds

Primary signals:

- relation connectivity
- scene sufficiency
- repeatability
- fitness / RMSE / conditioning
- wall and corner sharpness in the merged overlays

### LiDAR-to-IMU

Read in this order:

1. `standardized_samples.yaml`
2. `conversion_diagnostics.yaml`
3. `calibration/diagnostics/data_quality.yaml`
4. `calibration/metrics.yaml`
5. `calibration/diagnostics/acceptance_report.yaml`
6. `calibration/diagnostics/review_report.html`
7. `calibration/diagnostics/observability.yaml`
8. residual CSV files and SVG plots

Primary signals:

- ground residual quality
- motion registration quality
- turn balance
- yaw observability
- holdout generalization
- whether the solver had to freeze `x/y/yaw`

## 4. Practical acceptance baselines

| Module | Good first-pass baseline |
| --- | --- |
| `camera` | avg reprojection error < `1.0 px`; per-view reprojection p95 < `1.5 px`; image coverage spans multiple grid cells; no radial monotonicity warning |
| `lidar2camera` | `final_rms_px <= 1.0`; per-pose p95 <= `1.5 px`; holdout p95 <= `1.5 px`; accepted pair ratio healthy; image coverage / pose diversity / board geometry pass |
| `lidar2lidar` | `release_ready: true`; required relations connected; scene sufficiency and repeatability pass; overlays do not show ghosting / double edges |
| `lidar2imu` | `release_ready: true`; `vehicle_motion_assessment.recommendation: full_6dof_candidate`; motion registration and holdout are healthy; both turn directions are represented |

These are review baselines, not a substitute for project-specific release gates.

## 5. Visualization files that actually matter

| Module | Files to open | What you are checking visually |
| --- | --- | --- |
| `camera` | `comparison_view.png`, `image_coverage_heatmap.png` | undistortion sanity and board coverage |
| `lidar2camera` | `image_coverage_heatmap.png`, `pose_diversity_plot.png`, any overlay artifact listed in diagnostics | target coverage and whether the optimized extrinsic makes geometric sense |
| `lidar2lidar` | `merged_cloud_baseline_colored.ply`, `merged_cloud_loop_closure_colored.ply`, `visual_evaluation.yaml` | wall thickness, corner spread, ghosting, sensor misalignment after loop closure |
| `lidar2imu` | `review_report.html`, `ground_residuals_plot.svg`, `ground_height_residuals_plot.svg`, `motion_rotation_residuals_plot.svg`, `motion_residuals_plot.svg`, `motion_registration_fitness_plot.svg`, `trajectory_overlay.svg`, `trajectory_position_gap_plot.svg`, `imu_trajectory_cloud.ply`, `lidar_trajectory_cloud.ply`, `trajectory_overlay_cloud.ply`, `holdout_motion_residuals_plot.svg`, `yaw_cost_scan.svg`, plus the residual CSV/YAML files | browser-friendly visual triage for ground support, motion quality, IMU-vs-LiDAR trajectory consistency, stitched-keyframe geometry, holdout behavior, and yaw support |

Recommended tools:

- PLY / PCD: CloudCompare or Open3D
- CSV: spreadsheet, pandas, or Jupyter
- YAML: plain text viewer or your editor
- PNG: any image viewer
- HTML / SVG: any browser

## 6. Quick commands to surface the conclusion

### LiDAR-to-LiDAR

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2lidar/run/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
PY
```

### LiDAR-to-IMU

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2imu/run01/calibration/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
print(d["vehicle_motion_assessment"]["recommendation"])
PY
```

### LiDAR↔Camera

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2camera/run01/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
PY
```

## 7. How to interpret common warning patterns

### Metrics look good but visuals look bad

Do not release. Trust the visual contradiction and inspect the extraction layer.

### Visuals look okay but `data_quality.yaml` warns

Treat the run as review-only. The bag is likely under-excited or under-covered.

### `lidar2imu` froze `x/y/yaw`

The run may still be useful for `z/roll/pitch`, but it is not a full-6DoF
release result.

### `lidar2camera` converged but geometry resolution failed

Do not trust the result. This usually means the board interpretation is still
ambiguous or the sample set lacks pose diversity.

### `lidar2lidar` has good fitness but bad scene sufficiency

Treat it as review-only. Scene support is often the limiting factor for
production confidence, not ICP fitness alone.

## 8. What to read next

- Bag preparation: [docs/apollo_data_collection.md](apollo_data_collection.md)
- Module entry points: [docs/quickstart_index.md](quickstart_index.md)
- Method rationale and SOTA context:
  [docs/calibration_methodology.md](calibration_methodology.md)
