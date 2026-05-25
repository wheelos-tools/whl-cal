---
audience: dev
stability: stable
last_tested: 2026-05-25
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
   - inspect overlay evidence before promotion

`learning_based.py` remains experimental and should not be treated as the first
production release path.

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

so field debugging can distinguish acquisition crop from undistortion ROI.

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

Review order:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `metrics.yaml.summary`
4. `diagnostics/acceptance_report.yaml`
5. overlay / visual evidence

## 6. Current practical limitation

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

## 7. Current judgment

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
- future target improvements can plug into the same stable review contract
