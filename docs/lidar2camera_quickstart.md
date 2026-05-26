---
audience: user
stability: stable
P26-05-25
---

# LiDAR↪Camera quick start

Before running this tool:

1. collect the Apollo session with
   [docs/apollo_data_collection.md](apollo_data_collection.md)
2. prepare paired `image + .pcd` files from that session
3. review results with
   [docs/calibration_review_guide.md](calibration_review_guide.md)
4. read the design details in
   [docs/lidar2camera_design.md](lidar2camera_design.md) and
   [docs/calibration_methodology.md](calibration_methodology.md)

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| paired images and `.pcd` files | yes | current tool input is a prepared directory, not a raw Apollo bag |
| camera intrinsics | yes | 3x3 intrinsics matrix |
| camera distortion | yes | 5-parameter distortion vector |
| checkerboard pattern size | yes | inner-corner count |
| checkerboard square size | yes | meter unit |
| output directory | yes | where `calibrated_tf.yaml` and diagnostics will be written |

## Apollo recording prerequisites

For a good bag, record at least:

- one camera image topic
- one LiDAR `PointCloud2` topic
- `/tf_static`
- optional `/tf`

Capture guidance:

- keep the board fully visible in both modalities
- collect 15-30 poses if possible
- change image region, distance, and tilt across poses
- avoid motion blur, severe clipping, or partial board occlusion

Current limitation: the repo does **not** yet provide a direct
`.record -> lidar2camera dataset` exporter. After recording in Apollo, export
synchronized image / point-cloud pairs with your existing dataset-preparation
flow.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick run

```bash
lidar2camera-calibrate --write-default-config --config config.yaml
lidar2camera-calibrate --config config.yaml
```

## Recommended minimal config shape

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

## Minimal runtime self-check

If you want a fast pipeline smoke test before touching real data:

```bash
PYTHONPATH=. python3 tools/run_lidar2camera_smoke.py --poses 5
```

## Outputs

Core outputs:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/`

Production-style review artifacts:

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
- `diagnostics/leave_one_out_trials.csv`
- `diagnostics/geometry_resolution.csv`
- `diagnostics/image_coverage_heatmap.png`
- `diagnostics/pose_diversity_plot.png`
- `diagnostics/checkerboard_alignment_previews/*.png`

## How to judge a result

Acceptance heuristics:

- `final_rms_px <= 1.0 px`
- pose reprojection p95 `<= 1.5 px`
- holdout reprojection p95 `<= 1.5 px`
- accepted pair ratio stays healthy
- image coverage, pose diversity, and board geometry all pass
- geometry resolution completes cleanly
- per-pose checkerboard alignment previews confirm that detected image corners and
  final projected board corners agree

Recommended review order:

1. read `diagnostics/standardized_data.yaml`
2. read `diagnostics/extraction.yaml`
3. read `diagnostics/geometry_resolution.csv`
4. read `metrics.yaml`
5. read `diagnostics/data_quality.yaml`
6. read `diagnostics/acceptance_report.yaml`
7. open `diagnostics/image_coverage_heatmap.png` and
   `diagnostics/pose_diversity_plot.png`
8. confirm the overlay / geometry evidence does not contradict the metrics

If geometry resolution, image coverage, or pose diversity fails, keep the run
review-only even when the optimizer converges.

## Single checkerboard practical note

Using only **one physical checkerboard** is acceptable, but one pose is not
enough evidence for production promotion.

What one board can prove:

- the image-side checkerboard was found
- the LiDAR-side plane support was plausible
- the final projection aligns on that pose

What one board pose cannot prove well:

- repeatability
- pose-family independence
- broad image coverage
- broad depth / tilt observability

When reviewing a one-board run, open:

1. `diagnostics/checkerboard_alignment_previews/*.png`
2. `diagnostics/reference_overlay.png`
3. `diagnostics/per_pose_reprojection.csv`
4. `diagnostics/leave_one_out_trials.csv` when available

If only one pose looks good, treat the result as **alignment evidence**, not as
full release evidence.
