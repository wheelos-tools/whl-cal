---
audience: user
stability: stable
last_tested: 2026-05-26
---

# LiDAR↔Camera customer guide

Use this page when you need a **customer-facing** guide for LiDAR↔Camera
calibration and review.

Important scope:

1. **Production calibration today**
   - use the checkerboard-based `lidar2camera-calibrate` pipeline
   - this is the current release-oriented path
2. **Targetless today**
   - use it as an **experimental review / benchmark path**
   - it is useful for visual inspection and controlled perturbation testing
   - it is **not** yet the production replacement for checkerboard calibration

If the goal is "deliver a usable extrinsic to a customer today", use the
checkerboard path first, then optionally use targetless outputs as supporting
visual evidence.

## 1. Requirements

### Production checkerboard path

You need:

- paired camera images and LiDAR `.pcd` files
- camera intrinsics
- camera distortion coefficients
- checkerboard inner-corner count
- checkerboard square size in meters

### Targetless review path

You need:

- a local nuScenes-compatible dataset root and info pickle
- or an existing benchmark output directory under `outputs/lidar2camera/...`

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Production calibration quick start

### Step 1. Generate the default config

```bash
lidar2camera-calibrate --write-default-config --config config.yaml
```

### Step 2. Edit `config.yaml`

A minimal example:

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
output:
  directory: outputs/lidar2camera/run01
```

Critical fields:

- `camera.intrinsics`
- `camera.distortion`
- `checkerboard.pattern_size`
- `checkerboard.square_size`
- `output.directory`

### Step 3. Run calibration

```bash
lidar2camera-calibrate --config config.yaml
```

### Step 4. Open the result

Important files:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/reference_overlay.png`
- `diagnostics/checkerboard_alignment_previews/*.png`

## 4. How to review the production result

Recommended review order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `metrics.yaml`
4. `diagnostics/acceptance_report.yaml`
5. `diagnostics/reference_overlay.png`
6. `diagnostics/checkerboard_alignment_previews/*.png`

### What `checkerboard_alignment_previews/*.png` means

Each preview image shows one checkerboard pose.

It visualizes:

- detected checkerboard corners in the image
- checkerboard corners projected by the initial transform
- checkerboard corners projected by the final transform

Good:

- final projected corners collapse onto detected image corners
- corner ordering is correct
- error is small across the whole board, not only one corner

Bad:

- final corners are still visibly shifted
- corner ordering is flipped / mirrored
- one side aligns while the other side drifts

### What `reference_overlay.png` means

This is the camera image with projected LiDAR points overlaid.

Good:

- projected points follow board edges and nearby scene structure
- depth-colored layers look coherent
- no obvious left/right or up/down global shift

Bad:

- points cross visible edges that should be empty
- board edge looks aligned but surrounding scene is clearly offset
- points smear into a cloud instead of following structure

## 5. Targetless benchmark quick start

Use this only when you want to review the experimental targetless path.

```bash
lidar2camera-nuscenes-benchmark \
  --info-path /mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/nuscenes_infos_val.pkl \
  --camera-names CAM_FRONT \
  --sample-limit 8 \
  --methods identity,edge_refine,silhouette_refine,batch_hybrid_refine,oracle_gt \
  --output-dir outputs/lidar2camera/nuscenes_benchmark
```

You can also audit initial-value sensitivity explicitly:

```bash
lidar2camera-nuscenes-precision-audit \
  --info-path /mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/nuscenes_infos_val.pkl \
  --camera-names CAM_FRONT \
  --sample-limit 4 \
  --translation-magnitudes-m 0.01,0.02,0.05,0.10 \
  --rotation-magnitudes-deg 0.1,0.3,0.5,1.0,2.0 \
  --output-dir outputs/lidar2camera/targetless_precision_audit
```

## 6. How to read targetless visual outputs

### A. `*_debug.png` means what?

Example:

`outputs/lidar2camera/targetless_batch_hybrid_eval/diagnostics/overlays/000_silhouette_refine_level_00_rot_0.50deg_trans_0.020m_0_debug.png`

This file is **not a mask**.

It is a 2x2 debug panel generated from the current targetless edge/silhouette
diagnostics:

1. top-left: RGB image
2. top-right: image edges
3. bottom-left: projected LiDAR structure edges from the **initial** transform
4. bottom-right: projected LiDAR structure edges from the **final** transform

How it is generated:

- the code projects LiDAR points into the image
- it rasterizes projected occupancy
- it computes LiDAR-side structure edges from:
  - projected occupancy boundary
  - projected depth discontinuity
- it draws those LiDAR edges on the image

So `*_debug.png` answers this question:

**Did the targetless objective move projected LiDAR edges closer to real image
edges?**

Good:

- bottom-right edges align with visible image contours better than bottom-left
- obvious vehicle / pole / building boundaries are closer after refinement

Bad:

- bottom-right is no better than bottom-left
- bottom-right drifts to another structure
- large green edge regions appear where the image has no matching contour

### B. `*_comparison.png` means what?

Example:

`outputs/lidar2camera/targetless_batch_hybrid_eval/diagnostics/overlays/000_identity_level_00_rot_0.50deg_trans_0.020m_0_comparison.png`

This file is the preferred targetless visual review artifact.

It contains three panels:

1. **Initial** depth-colored LiDAR projection
2. **Final** depth-colored LiDAR projection
3. **GT** depth-colored LiDAR projection

Each panel contains:

- dense projected LiDAR points
- depth color map
- projected point count
- depth range
- depth colorbar

For `identity`, Initial and Final are expected to look the same because the
method intentionally does **not** optimize.

For a real targetless candidate, the review question is:

**Did Final move closer to GT than Initial did?**

Good:

- Final is visually closer to GT than Initial
- projected points stay dense; point count should not collapse
- scene contours such as car roofs, poles, curb-like edges, or building edges
  align better
- no new global left/right or up/down shift appears

Bad:

- Final still looks like Initial
- Final moves away from GT
- Final only looks "sharper" because point count collapsed
- depth layering becomes inconsistent or smeared

### C. How to judge the targetless result overall

Do **not** trust a single image only.

Use this order:

1. `diagnostics/per_method_summary.csv`
2. `diagnostics/perturbation_summary.csv`
3. `diagnostics/acceptance_report.yaml`
4. representative `*_comparison.png`
5. representative `*_debug.png`

Important metrics:

- `mean_final_rotation_error_deg`
- `mean_final_translation_error_m`
- `accepted_update_rate`
- `mean_objective_improvement`
- `*_projected_point_count`
- `*_projected_point_ratio`
- `*_projected_bbox_area_ratio`

## 7. What is good and what is bad?

### Production checkerboard path

Good means:

- `acceptance_report.yaml` does not show required failures
- checkerboard previews align
- reference overlay agrees with the metrics

Bad means:

- optimizer converged but the image evidence is clearly wrong
- checkerboard previews still show visible corner drift
- `final_acceptance` stays warning / fail

### Targetless path

Good means:

- the candidate method clearly beats `identity`
- Final is visibly closer to GT than Initial
- projected point count stays healthy
- `projection_visibility` passes

Bad means:

- the candidate only matches `identity`
- the candidate becomes worse than `identity`
- the candidate only "improves" by moving to a boundary solution
- visual panels contradict the scalar metrics

## 8. Current customer recommendation

Use this rule today:

1. **For delivered calibration output**:
   use the checkerboard `lidar2camera-calibrate` pipeline.
2. **For targetless review and R&D evidence**:
   use the targetless benchmark outputs.

Current honest status of targetless:

- visualization is strong enough for review
- the benchmark is valid
- but the current targetless objectives are still experimental
- they should not replace the checkerboard production baseline yet
