---
audience: user
stability: stable
P26-04-27
---


# LiDAR↪Camera Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick run:
```bash
lidar2camera-calibrate --write-default-config --config config.yaml
lidar2camera-calibrate --config config.yaml
```

Recommended minimal config shape:

```yaml
camera:
  intrinsics: [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
  distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
checkerboard:
  pattern_size: [8, 6]
  square_size: 0.05
point_cloud:
  plane_dist_thresh: 0.02
  min_plane_points: 500
extraction:
  min_bbox_area_ratio: 0.0008
  min_edge_margin_px: 8.0
  max_plane_residual_rmse_m: 0.02
  reject_board_geometry_warnings: true
optimization:
  min_poses: 5
  loss: huber
  f_scale: 1.0
  max_nfev: 200
metrics:
  warning_final_rms_px: 1.0
  warning_pose_rms_p95_px: 1.5
  warning_holdout_rms_px: 1.5
  warning_repeatability_translation_m: 0.05
  warning_repeatability_rotation_deg: 1.0
  warning_image_coverage_min_cells: 4
  warning_image_horizontal_span_ratio: 0.35
  warning_image_vertical_span_ratio: 0.35
  warning_depth_span_m: 0.3
  warning_tilt_span_deg: 8.0
  warning_plane_residual_rmse_m: 0.02
  warning_board_extent_ratio_min: 0.5
  warning_board_extent_ratio_max: 4.0
  warning_accepted_pair_ratio: 0.5
output:
  directory: outputs/lidar2camera/run01
```

Outputs: `calibrated_tf.yaml`, `metrics.yaml`, `diagnostics/`

Production-style review artifacts now include:

- `metrics.yaml.summary.final_acceptance_status`
- `metrics.yaml.summary.release_ready`
- `metrics.yaml.final_acceptance`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/visualization_index.yaml`
- `diagnostics/extraction_entries.csv`
- `diagnostics/per_pose_reprojection.csv`
- `diagnostics/leave_one_out_trials.csv` (when L1O is available)
- `diagnostics/geometry_resolution.csv` (when board-geometry hypotheses were resolved)
- `diagnostics/image_coverage_heatmap.png`
- `diagnostics/pose_diversity_plot.png`

Acceptance heuristics: final_rms_px ≤ 1.0 px; pose p95 ≤ 1.5 px; L1O p95 ≤ 1.5 px.
Release should follow `final_acceptance`, not solver convergence alone.

Release review now also expects:

- broad image-region coverage across the checkerboard poses
- enough depth / tilt diversity across accepted poses
- LiDAR-side board support that looks like a board-sized plane, not a large wall
  patch
- successful `diagnostics/extraction.yaml.geometry_resolution` so the selected
  board orientation is consistent across poses
- clean `geometry_resolution` status in `metrics.yaml.final_acceptance`

If these gates fail, the run should remain review-only even when the optimizer
converges and aggregate RMS looks good.

Recommended run/review order:

1. Prepare image + PCD pairs with matching stems in `data_directory`.
2. Confirm camera intrinsics were calibrated with the same camera mode that
   will be used in production.
3. Run `lidar2camera-calibrate --config config.yaml`.
4. Read `diagnostics/standardized_data.yaml` to confirm accepted vs rejected
   sample counts.
5. Read `diagnostics/extraction.yaml` and `diagnostics/geometry_resolution.csv`
   to confirm multi-hypothesis board geometry resolution converged cleanly.
6. Read `diagnostics/optimization.yaml` and confirm the stage summary is sane.
7. Read `diagnostics/data_quality.yaml` and `diagnostics/acceptance_report.yaml`.
8. Open `diagnostics/image_coverage_heatmap.png`,
   `diagnostics/pose_diversity_plot.png`, and the overlay artifact before
   promoting any result.

Treat the run as release-ready only when:

- `metrics.yaml.summary.release_ready: true`
- `diagnostics/data_quality.yaml.status: pass`
- accepted pair ratio is healthy
- image coverage / pose diversity / board geometry all pass
- geometry resolution completed without unresolved candidate failures
- visual overlay does not contradict the numeric metrics

See docs/lidar2camera_design.md for details.
