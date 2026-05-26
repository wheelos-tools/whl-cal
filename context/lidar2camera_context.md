---
audience: dev
stability: stable
last_tested: 2026-05-26
---

# lidar2camera current context

## 1. Current production baseline

`lidar2camera` now follows the same repo-wide **data -> algorithm -> evaluation**
structure used by the other calibration modules.

Current release-oriented path:

1. **camera intrinsics first**
   - use `camera/intrinsic.py`
   - keep capture resolution separate from display window size
   - default to `capture.force_resolution: false` unless a native sensor mode is known
2. **reference-based lidar2camera**
   - use `lidar2camera-calibrate`
   - pair image and PCD files by stem
   - extract checkerboard corners plus LiDAR board plane support
   - resolve board orientation with gravity/PCA candidate hypotheses instead of a
     single fixed in-plane-axis guess
   - reject obviously weak samples before optimization
3. **evaluation / promotion**
   - read `metrics.yaml`
   - read `diagnostics/standardized_data.yaml`
   - read `diagnostics/data_quality.yaml`
   - read `diagnostics/acceptance_report.yaml`
   - inspect heatmap / pose-diversity / overlay evidence before promotion

`learning_based.py` remains experimental and should not be treated as the first
production release path.

There are now **two clearly different lidar2camera surfaces** in the repo:

1. **production / release path**
   - `lidar2camera-calibrate`
   - checkerboard-based
   - paired `image + .pcd`
   - this is the path to use when the question is:
     - "what is the current official lidar2camera method?"
     - "can this run be promoted?"
2. **experimental benchmark path**
   - `lidar2camera-nuscenes-benchmark`
   - nuScenes GT-perturbation evaluation
   - compares `identity`, `edge_refine`, `silhouette_refine`, and `oracle_gt`
   - this is for controlled recovery benchmarking, **not** for replacing the
     production checkerboard release contract

So when someone asks "目前 lidar2camera 的方案是什么", the correct short answer is:

- **production:** checkerboard reference-board calibration
- **experimental:** separate targetless-style nuScenes benchmark for evidence-based comparison

## 2. Camera intrinsic live-capture rule

The most important operational rule for the intrinsic tool is now:

- **do not use the display window size as the camera capture size**

The tool now separates:

- `window_width` / `window_height`
- `capture.force_resolution`
- `capture.width`
- `capture.height`
- `capture.fourcc`

Why this matters:

- many cameras have native 4:3 sensor modes
- forcing a 16:9 mode such as 1280x720 can trigger ISP or driver crop before the
  3x3 guidance grid is drawn
- if the live 3x3 grid already looks clipped, trust the capture diagnostics first
  and disable forced capture resolution before collecting data

The intrinsic YAML now records:

- `capture_runtime`
- `distortion_model`
- `undistortion_preview`
- `sample_quality`
- `per_view_reprojection_summary`

and its diagnostics directory now writes:

- `acceptance_report.yaml`
- `status_summary.csv`
- `standardized_data.yaml`
- `data_quality.yaml`
- `visualization_index.yaml`
- `per_view_reprojection.csv`
- `sample_records.csv`
- `image_coverage_heatmap.png`

so field debugging can distinguish acquisition crop from undistortion ROI, and
calibration review can move beyond average reprojection error.

## 3. Current lidar2camera extraction contract

The reference pipeline currently performs:

1. file pairing
2. chessboard corner extraction
3. LiDAR plane segmentation
4. multi-hypothesis board coordinate construction from plane support using:
   - gravity-projected axes
   - PCA-derived in-plane axes
   - axis swap / sign-flip candidates
   - single-pose IPPE reprojection scoring
   - cross-pose consistency refinement against the global transform seed
5. per-pose sample-quality gating

Per-pose gating now rejects obviously weak samples before optimization, including:

- board too close to the image edge
- board too small in the image
- LiDAR plane residual too high
- LiDAR board geometry warnings when configured to reject them

The extraction report now records:

- `accepted_pose_count`
- `accepted_pair_ratio`
- `rejected_pose_count`
- `skip_reason_counts`
- `geometry_resolution`
- per-entry `sample_quality`

This is important: the solver should not see obviously bad samples and then
"average them out" later.

## 4. Current lidar2camera release gates

`final_acceptance` is now the release contract.

Production promotion requires:

- enough paired samples
- enough accepted samples
- healthy accepted-pair ratio
- solver success
- final reprojection pass
- per-pose reprojection pass
- holdout reprojection pass
- leave-one-out repeatability pass
- image coverage pass
- pose diversity pass
- board geometry pass
- geometry resolution pass

The most important new industrial gates are:

### A. Extraction yield

- if too many paired samples are rejected, the workflow is not stable enough for
  production even if the surviving subset optimizes well

### B. Image coverage

- board centers should cover multiple image regions
- left / center / right and up / center / down coverage matters
- production cannot rely on center-only board captures

### C. Pose diversity

- accepted poses should span multiple depths and tilts
- otherwise the solution is too likely to be pose-family dependent

### D. Board geometry

- LiDAR support should look like a board-sized plane, not a large wall patch or
  tiny under-supported patch

## 5. Stable artifacts to expect

The reference pipeline now writes:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `initial_guess/*.yaml`
- `calibrated/*.yaml`
- `diagnostics/reference_dataset.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/optimization.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/manifest.yaml`
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

The checkerboard writer must sanitize NumPy-heavy diagnostic internals before
YAML output. This is important because extraction/metrics reports may carry
private candidate arrays used for board-geometry resolution, while public
diagnostic YAMLs should stay reviewable and serializable.

Review order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `diagnostics/extraction.yaml` + `diagnostics/geometry_resolution.csv`
4. `diagnostics/optimization.yaml`
5. `metrics.yaml.summary`
6. `diagnostics/acceptance_report.yaml`
7. heatmap / pose-diversity / overlay evidence

## 6. How to judge good vs bad

The repo intentionally does **not** judge a run by a single number.

The current practical rule is:

- a run is good only when **metrics, sample yield, repeatability, and visual
  evidence are mutually consistent**

### A. What "good" looks like

For the production checkerboard path, a healthy result usually means:

1. `final_acceptance.release_ready == true`
2. `accepted_pair_ratio` stays healthy instead of surviving on a tiny subset
3. `final_rms_px`, per-pose reprojection p95, and holdout p95 are all within thresholds
4. leave-one-out repeatability does not show multiple conflicting solution families
5. image coverage is broad enough across the image plane
6. pose diversity is broad enough across depth and tilt
7. board geometry and geometry resolution do not show unresolved ambiguity
8. the overlay visually agrees with the metrics

### B. What "bad" looks like

A run should stay review-only or be recollected when any of these happens:

1. optimizer converges but `accepted_pair_ratio` is low
2. RMS is low but board centers are all near the image center
3. RMS is low but poses all come from nearly the same depth / tilt family
4. holdout error or leave-one-out repeatability is unstable
5. geometry resolution needed many rounds or left unresolved poses
6. overlay shows board-edge misalignment, depth layering errors, or obvious drift

### C. What to trust most

Use this review order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `diagnostics/extraction.yaml`
4. `diagnostics/geometry_resolution.csv`
5. `metrics.yaml`
6. `diagnostics/acceptance_report.yaml`
7. `diagnostics/image_coverage_heatmap.png`
8. `diagnostics/pose_diversity_plot.png`
9. `diagnostics/reference_overlay.png`

If the image evidence disagrees with the scalar metrics, trust the conflict and
investigate; do not promote blindly.

Single-checkerboard note:

- one physical checkerboard is fine
- one pose is **not** enough for release confidence
- with one pose, `checkerboard_alignment_previews` + `reference_overlay` can show
  that the board was found and that this pose aligns
- but only multiple poses can really support repeatability, image coverage, and
  pose diversity claims

## 7. Visual evidence to inspect

Besides scalar metrics, the current pipeline already writes several artifacts
that should be treated as first-class review evidence.

### A. `diagnostics/reference_overlay.png`

What it is:

- a representative image with projected LiDAR points blended onto the camera image
- points are colored by depth

How to read it:

- board edges and nearby structure should align with visible image structure
- depth-colored points should form a coherent layered projection instead of a
  smeared cloud
- if the board is aligned but nearby scene structure is obviously offset, the
  transform is still suspicious

Typical good example:

- board boundary aligns
- nearby edges such as cabinet / wall / pole contours are also consistent
- no obvious global left-right or up-down shift

Typical bad example:

- a low RMS run still shows a clear systematic offset on one side of the board
- points cross image edges that should be empty
- depth layering looks inconsistent, indicating a plausible but wrong local minimum

### B. `diagnostics/image_coverage_heatmap.png`

What it is:

- a 3x3 image-grid heatmap counting where checkerboard centers appeared

How to read it:

- good runs occupy multiple cells
- horizontal and vertical span should both be healthy
- center-only collection is a warning sign even if reprojection looks good

Good example:

- observations cover left/center/right and upper/middle/lower regions

Bad example:

- almost all counts live in the center cell or only a narrow horizontal strip

### C. `diagnostics/pose_diversity_plot.png`

What it is:

- a scatter plot of board depth vs board tilt for accepted poses

How to read it:

- good runs spread across both depth and tilt
- narrow clusters mean the calibration is weakly conditioned and may not generalize

Good example:

- points span multiple depths and multiple tilt angles

Bad example:

- nearly all points stack into a small cluster, meaning the run depends on one
  pose family

### D. `diagnostics/geometry_resolution.csv`

What it is:

- a table describing which board-geometry candidate was selected per pose across
  resolution iterations

How to read it:

- good runs settle quickly with few or no changes after early rounds
- bad runs keep changing source / swap / sign decisions or leave unresolved poses

### E. `diagnostics/per_pose_reprojection.csv`

What it is:

- per-pose reprojection error table

How to read it:

- good runs do not rely on hiding a few catastrophic poses inside a good mean
- inspect p95 / max-like tails, not only the average

### F. `diagnostics/leave_one_out_trials.csv`

What it is:

- repeated solve results when each pose is held out once

How to read it:

- good runs remain close to the primary solution
- bad runs split into noticeably different translation / rotation families

### G. `diagnostics/checkerboard_alignment_previews/*.png`

What it is:

- per-pose preview images showing:
  - detected checkerboard corners
  - initial-transform projected board corners
  - final-transform projected board corners

How to read it:

- this is the most direct answer to "did we actually find the checkerboard?" and
  "did the solved board align on this pose?"
- good runs show final projected corners collapsing onto detected corners
- bad runs show systematic bias, corner ordering mistakes, or only marginal
  improvement from the initial projection

## 8. How to judge the experimental nuScenes benchmark

For `lidar2camera-nuscenes-benchmark`, the judgment logic is different from the
checkerboard production run.

Use this order:

1. `oracle_gt` must be essentially perfect
2. compare `silhouette_refine` or `edge_refine` against `identity`
3. inspect `perturbation_summary.csv`
4. inspect `success_curves.yaml`
5. inspect `*_comparison.png` depth-colored point-cloud projection panels
6. inspect overlays and `*_debug.png` edge panels

Interpretation:

- if `oracle_gt` is not perfect, the benchmark wiring is wrong
- if the targetless candidate is better than `identity`, it is helping
- if it matches `identity`, guard rails prevented a risky update or the objective
  was not informative enough
- if it is worse than `identity`, the current objective is not yet strong enough

Important current benchmark contract:

- default reference mode is now `rigid_sensor`, derived from
  `inv(cam2ego) @ lidar2ego`
- camera/LiDAR timestamp skew is tracked as `time_delta_ms`
- targetless runs record initial/final/GT projected point counts and projected
  point ratios, depth p50/p95, and projection bbox area to catch invalid
  projection or field-of-view regressions
- targetless runs emit `*_comparison.png` panels showing initial/final/GT
  dense depth-colored point clouds projected onto the image
- targetless runs emit `*_debug.png` panels showing image edges plus
  initial/final projected LiDAR structure edges
- targetless visualization uses a separate dense point set from the optimization
  point set, so review overlays are not hidden by high-intensity sparsification
- dense overlays should show thousands of projected points on nuScenes
  front-camera frames; a camera-only-looking panel indicates a visualization or
  point-filtering regression, even if the scalar projection code returns success
- `acceptance_report.yaml` has a `projection_visibility` gate; current smoke
  passes with min_projected_points=2963 and min_projected_ratio=0.0854

For visual review, prefer the `*_comparison.png` panel first. It shows whether
the final projection moved toward nuScenes GT and whether depth layering looks
coherent. Then use `*_debug.png` to understand whether the targetless edge or
silhouette objective had meaningful image evidence.

Today the honest status is:

- the benchmark is valid and reproducible
- the in-repo `edge_refine` / `silhouette_refine` baselines are still experimental
- it should not be described as SOTA unless a stronger cross-dataset comparison
  actually proves that
- a new `batch_hybrid_refine` multi-frame method now exists and uses:
  - multistart coarse hypotheses
  - coordinate-consensus refinement
  - projected-point-retention guard rails
- it now produces bounded updates and no longer returns the wrong large update
  that appeared in the first unconstrained batch attempt

Initial-value precision conclusion from the 2026-05-26 nuScenes CAM_FRONT audits:

- tested axis-aligned perturbations:
  - translations: x/y/z at 1 cm, 2 cm, 5 cm, and 10 cm
  - rotations: roll/pitch/yaw at 0.1 deg, 0.3 deg, 0.5 deg, 1.0 deg, and 2.0 deg
- final multi-scene audit:
  - path: `outputs/lidar2camera/targetless_precision_audit_final/diagnostics/`
  - 4 CAM_FRONT scenes, 27 axis-aligned perturbation cases, 108 rows
  - `edge_refine`: GT correction scored better in 48/108 cases = 44.4%
  - `silhouette_refine`: GT correction scored better in 36/108 cases = 33.3%
  - both edge and silhouette agreed in only 32/108 cases = 29.6%
- high-risk axes:
  - pitch >= 0.3 deg had 0% edge+silhouette agreement
  - roll/yaw >= 1.0 deg had 0% edge+silhouette agreement
- actual `silhouette_refine` optimizer did not safely recover the smallest
  1 cm / 0.1 deg perturbations; all 6 optimizer smoke cases were rejected by the
  guard and fell back to the initial guess
- relaxing the guard is not a valid fix: the optimizer proposed large ~2.3-2.4
  deg and ~7-8 cm updates for 1 cm cases, so the guard correctly prevents
  false-positive calibration updates

Practical conclusion: with the current single-frame targetless objective, the
initial value must already be treated as the calibration result. The benchmark
can verify and visualize centimeter/sub-degree perturbations, but the in-repo
targetless optimizer cannot yet be trusted to pull normal measurement errors
back automatically. If automatic recovery from several centimeters or >=0.5 deg
roll/pitch/yaw is required, the algorithm must move to a stronger multi-frame
objective with semantic/depth-boundary evidence and external baseline
comparison.

Batch-hybrid iteration result:

- implementation:
  - multi-frame shared-extrinsic search over same-camera contexts
  - multistart seed proposals
  - coordinate-consensus refinement
  - projected-point-retention guard
- practical speed:
  - default benchmark downscale is now `2.0`
  - current `CAM_FRONT` 4-sample / 3-bucket smoke runs in about `1m15s`
- final benchmark path:
  `outputs/lidar2camera/targetless_batch_multistart_eval_final/diagnostics/`
- result on 4 `CAM_FRONT` samples and 3 perturbation buckets:
  - mean rotation error:
    - `identity`: `1.1667 deg`
    - `batch_hybrid_refine`: `1.1406 deg`
  - mean translation error:
    - `identity`: `0.0567 m`
    - `batch_hybrid_refine`: `0.0587 m`
  - loose success:
    - `identity`: `8.3%`
    - `batch_hybrid_refine`: `16.7%`
  - accepted update rate: `16.7%`
  - `(2.0 deg, 0.10 m)` perturbations were not recovered
- practical interpretation:
  - `batch_hybrid_refine` is now a **safe warm-start refiner**
  - it can help some `0.5-1.0 deg / 0.02-0.05 m` cases
  - it is **not** yet a production-grade fully automatic targetless calibrator

Projection visualization comparison against common targetless practice:

- current artifacts now match the core review surfaces used by targetless
  lidar-camera work: depth-colored point projection, initial/final/GT
  comparison, image-edge vs projected-LiDAR-structure debug panels, and
  projection coverage statistics
- multi-view projection smoke:
  - `outputs/lidar2camera/targetless_multiview_projection_review/`
  - CAM_FRONT: min GT projected points=2975, mean=2995.3, mean bbox area=0.8057
  - CAM_FRONT_RIGHT: min GT projected points=3504, mean=3510.7, mean bbox area=0.8781
  - `outputs/lidar2camera/targetless_front_left_projection_review/`
  - CAM_FRONT_LEFT with 50 ms sync threshold: min GT projected points=3590,
    mean=3600.0, mean bbox area=0.8598
- projection visualization is now adequate for review; remaining blocker is
  optimization reliability, not whether the point cloud is visible on images

## 9. Current practical limitation

The biggest remaining risk is no longer a single hard-coded LiDAR in-plane axis.
The current residual limitation is now narrower:

- the LiDAR-side board still comes from **plane support**, not direct LiDAR board
  corners / coded target identities
- so runs still depend on clean board isolation, enough on-board points, and
  board-vs-background separation in the point cloud

The new candidate-resolution stage and release gates remove the old dominant
failure mode, but they do **not** make a weak physical target design magically
observable.

If higher release confidence is required, the next algorithm iteration should
prioritize one of:

1. reflective / coded LiDAR target upgrade
2. reflective target support
3. ChArUco / AprilTag-grid style target upgrade

## 10. Current judgment

Current status is best described as:

- **production-ready for the current reference-board workflow, assuming the data
  collection contract is followed**
- **safer against the previous false-pass modes in both sample selection and
  LiDAR board orientation**
- **still improved further by stronger physical targets, but no longer blocked on
  the old single-heuristic geometry step**

That is the correct baseline for this repo right now:

- release decisions are now much stricter
- weak collection workflows are easier to catch
- multi-hypothesis board geometry resolution is now part of the stable workflow
- CSV + image diagnostics now make sample screening / optimization iteration / promotion review easier to repeat
- future target improvements can plug into the same stable review contract
