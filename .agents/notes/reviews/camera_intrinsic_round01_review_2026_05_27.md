---
audience: dev
stability: tested
last_tested: 2026-05-27
---

# Camera intrinsic round01 review (2026-05-27)

This note records the verified diagnosis for the AprilGrid intrinsic dataset:

- capture session: `outputs/camera_intrinsic/captures/round01_aprilgrid/accepted`
- primary offline run: `outputs/camera_intrinsic/runs/20260527_095411_round01_aprilgrid`
- target: AprilGrid 6x6
- image size: `1920x1080`
- config: `camera_config.yaml`

## 1. Validated baseline result

The round01 dataset is **usable for pipeline execution** but **not acceptable as a
release-ready intrinsic result**.

Verified facts:

- AprilGrid detection succeeds reliably on the collected images.
- The accepted dataset contains 18 valid images.
- `sample_count` passes.
- `image_coverage` passes on the full dataset.
- The intrinsic solve still fails quality review.

Full-dataset result from `20260527_095411_round01_aprilgrid`:

- `release_ready: false`
- `avg_reprojection_error_px = 7.9833`
- `per_view_rms_p95_px = 100.7066`
- `min_radial_derivative = -19435.6149`

Interpretation:

- this is not a small tuning miss
- the fitted distortion model is physically implausible
- the dataset should not be promoted as the final intrinsic calibration

## 2. Weak-view diagnosis

Per-view reprojection identifies the worst samples as:

1. `sample_012.jpg` — `rms_px = 100.86`
2. `sample_003.jpg` — `rms_px = 100.68`
3. `sample_008.jpg` — `rms_px = 97.97`
4. `sample_016.jpg` — `rms_px = 73.79`
5. `sample_015.jpg` — `rms_px = 67.77`

Visual review of the worst images shows a consistent risk pattern:

- several weak samples are **close-board / large-footprint** views
- several weak samples are **hand-held** at the top edge, so mild board flex is plausible
- some weak samples are collected under strong window backlight / glare
- the dataset overuses left-right span more than strong top-bottom or controlled tilt span

Important nuance:

- the problem is **not** "coverage is missing"
- the problem is **not** "AprilGrid detection is failing"
- the problem is more likely a mixture of:
  - non-rigid target support in close views
  - weak geometric consistency across poses
  - too many large, near-frontal or unstable views relative to the final solve

## 3. Salvage iterations

Several pruning experiments were run to test whether round01 can be rescued by
removing obviously weak images.

### A. Full dataset

Run:

- `outputs/camera_intrinsic/runs/20260527_095411_round01_aprilgrid`

Result:

- samples: `18`
- `release_ready: false`
- `avg_reprojection_error_px = 7.9833`
- `per_view_rms_p95_px = 100.7066`
- `radial_monotonicity = fail`

### B. Drop top-3 worst views (`sample_003`, `sample_008`, `sample_012`)

Run:

- `outputs/camera_intrinsic/runs/20260527_104441_drop_top3_aprilgrid`

Result:

- samples: `15`
- `release_ready: false`
- `avg_reprojection_error_px = 6.3232`
- `per_view_rms_p95_px = 69.5012`
- `radial_monotonicity = pass`

Interpretation:

- pruning the worst three views helps substantially
- even after that, reprojection quality is still far from acceptable
- this indicates round01 is not rescued by removing only a few obvious outliers

### C. Drop top-5 worst views (`sample_003`, `sample_008`, `sample_012`, `sample_015`, `sample_016`)

Run:

- `outputs/camera_intrinsic/runs/20260527_104446_drop_top5_aprilgrid`

Result:

- samples: `13`
- `release_ready: false`
- `avg_reprojection_error_px = 6.1861`
- `per_view_rms_p95_px = 66.8731`
- `radial_monotonicity = warning`

Interpretation:

- extra pruning gives only small additional benefit
- the dataset remains unsuitable for promotion

### D. Drop top-3 plus center-heavy large views

Run:

- `outputs/camera_intrinsic/runs/20260527_104450_drop_top3_plus_center_heavy_aprilgrid`

Result:

- samples: `13`
- `release_ready: false`
- `avg_reprojection_error_px = 6.1830`
- `per_view_rms_p95_px = 69.9980`
- `image_coverage = warning`
- `radial_monotonicity = warning`

Interpretation:

- aggressive pruning starts to damage coverage without solving the main quality problem

### E. Drop only the largest board-footprint views

Run:

- `outputs/camera_intrinsic/runs/20260527_104822_drop_large_footprint_aprilgrid`

Result:

- samples: `16`
- `release_ready: false`
- `avg_reprojection_error_px = 7.7298`

Interpretation:

- simply removing the largest close-up views is not enough
- large-footprint instability is part of the problem, not the whole problem

### F. Curated stable-view subset

Kept views:

- `sample_001`, `002`, `006`, `007`, `010`, `011`, `013`, `014`, `017`, `018`

Run:

- `outputs/camera_intrinsic/runs/20260527_105028_curated_stable_views_aprilgrid`

Result:

- samples: `10`
- `release_ready: false`
- `avg_reprojection_error_px = 6.0005`
- `per_view_rms_p95_px = 58.9460`
- `image_coverage = warning`
- `radial_monotonicity = warning`

Interpretation:

- even a manually curated subset of the most stable-looking views does not pass
- round01 should be treated as a **diagnostic round**, not a promotable run

## 4. Final conclusion for round01

The validated conclusion is:

- round01 is **not salvageable to release-ready quality by simple sample pruning**
- sample pruning improves the solve, but not enough
- the next correct step is **round02 recollection**, not more ad hoc filtering of round01

Round01 is still useful because it exposed the real operational failure modes.

## 5. Round02 collection checklist

The next collection round should be run with a stricter capture protocol.

### Target handling

- mount the AprilGrid on a **rigid flat backing** before capture
- do not rely on hand-held top-corner pinching for close views
- if hand support is unavoidable, keep the board small in the image and avoid bending torque
- verify the printed target is truly flat before capture

### Pose distribution

Aim for `12-16` accepted views, even though the minimum target is `9`.

Required structure:

1. `2-3` medium-size center views
   - board footprint roughly `6% - 12%` of the image
   - use only as anchor views, not the majority
2. `4` corner / edge views
   - top-left, top-right, bottom-left, bottom-right
   - keep the full board visible
   - avoid clipping and keep some margin from the frame boundary
3. `4` strong tilt views
   - left tilt
   - right tilt
   - upward tilt
   - downward tilt
   - target roughly `10-20` degrees of out-of-plane tilt
4. `2-3` distance-change views
   - one farther view with smaller footprint
   - one closer view with larger footprint
   - only keep the close view if the board is rigid and still fully visible

### Explicit things to avoid

- avoid collecting too many near-frontal views with similar scale
- avoid letting the board fill more than about `18% - 20%` of the image unless it is rigidly mounted
- avoid strong board flex in close views
- avoid scenes where the board is backlit by the window if an equivalent non-backlit pose is possible
- avoid using people / hands to partially occlude the border or distort the paper

### Operational acceptance during capture

During live collection, trust the following rule:

- when coverage is full, keep collecting until the UI says the required novel-pose count is satisfied
- do not restart just because coverage is full
- restart only if the accepted images themselves are visibly weak: blur, crop, severe glare, or clear board flex

### Immediate post-run review

Before treating round02 as usable:

1. inspect `capture_session.yaml`
2. confirm the accepted images are still full-resolution `1920x1080`
3. run offline calibration from the accepted directory
4. inspect:
   - `acceptance_report.yaml`
   - `per_view_reprojection.csv`
   - `image_coverage_heatmap.png`
   - `comparison_view.png`
5. reject the run if radial monotonicity or reprojection gates are still warning-level

## 6. Headless RTSP probe on 2026-05-27

A forced headless live probe was run against `camera_config.yaml` using the
configured RTSP source and AprilGrid target type.

Observed result:

- `camera-intrinsic-calibrate --capture-only --headless-live-max-seconds 25`
  reached the stream, but no AprilGrid was detected in the scene
- the app saved a valid first frame at
  `outputs/camera_intrinsic/captures/hevc_headless_probe_aprilgrid/debug/headless_first_frame.jpg`
- a direct raw-frame probe saved 5 additional valid frames under
  `outputs/camera_intrinsic/hevc_probe_frames/`
- all saved frames were full-resolution `1920x1080`
- the decoded frames were not blank; frame statistics were stable across the
  five saved samples

HEVC decoder behavior:

- OpenCV/FFmpeg emitted repeated `hevc` warnings during startup:
  `Could not find ref with POC ...` and `Error constructing the frame RPS.`
- the warnings were concentrated during warm-up and did not prevent valid frames
  from being delivered
- after warm-up, the stream reported `ready: true`, `total_valid_frames: 26`,
  and `total_delivered_frames: 5`

Interpretation:

- the RTSP path is usable, but the HEVC stream is fragile at startup
- the current failure mode looks like decoder reference-frame loss or stream
  startup synchronization rather than a total capture outage
- for future debugging, compare this stream against a known-good H.264/H.265
  source or inspect whether the encoder starts mid-GOP without a clean keyframe

## 6. Practical implication for algorithm and UI work

The sampling changes made in this iteration were still correct:

- coverage completion must remain separate from sample-count completion
- post-coverage sampling must be novelty-gated
- the UI must tell the operator what kind of pose is still missing

But operator guidance alone is not enough when the board itself may be unstable.
For this camera / target setup, **capture protocol quality dominates solver quality**.

## 7. Current recollection review (same round01 directory, 2026-05-27)

The current `outputs/camera_intrinsic/captures/round01_aprilgrid` directory is a
mixed dataset, not a single clean batch.

Validated facts:

- `accepted/` currently contains `27` images.
- `capture_session.yaml` records the latest completed capture-only batch as
   `sample_010.jpg` through `sample_018.jpg`.
- file timestamps show those `010-018` images are the newest contiguous batch in
   the manifest-backed run.
- one additional older image in the full directory (`sample_019.jpg`) does not
   meet the AprilGrid minimum-tag gate during offline replay, so the full replay
   uses `26` valid images out of `27` files.

### A. Full current accepted directory (`27` files, `26` valid)

Run:

- `outputs/camera_intrinsic/runs/20260527_143014_round01_aprilgrid`

Result:

- `release_ready: false`
- `avg_reprojection_error_px = 7.2351`
- `per_view_rms_p95_px = 74.6731`
- `min_radial_derivative = -9377.8223`

Interpretation:

- the current full directory is still not promotable
- compared with the earlier `18`-image attempt, coverage is denser, but solver
   quality is still poor
- adding more images did not fix the fundamental geometric inconsistency

### B. Latest manifest batch only (`sample_010` to `sample_018`)

Run:

- `outputs/camera_intrinsic/runs/20260527_143040_current_manifest_batch_aprilgrid`

Result:

- `release_ready: false`
- `avg_reprojection_error_px = 7.1477`
- `per_view_rms_p95_px = 72.2444`
- `min_radial_derivative = -3.9447`

Interpretation:

- the newest `9`-image batch alone is also not good enough
- this means the current recollection still fails even before it is mixed with
   older accepted images

### C. Why the heatmap is meaningful but still insufficient

The full-directory heatmap from `20260527_143014_round01_aprilgrid` is:

- top row: `10, 13, 6`
- middle row: `13, 17, 8`
- bottom row: `9, 11, 6`

This is meaningful because it shows:

- the center is still oversampled
- the right column is still relatively weak
- coverage counts are now much more balanced than the earlier failed rounds

But the run still fails. So the validated conclusion is:

- **heatmap is necessary for operator guidance, but not sufficient for intrinsic quality**
- heatmap measures spatial distribution only
- it does not catch board flex, unstable hand-held support, glare, or inconsistent
   pose geometry

### D. Current weak-view pattern

The worst current samples in the full replay include:

1. `sample_002.jpg` — `rms_px = 77.10`
2. `sample_016.jpg` — `rms_px = 76.87`
3. `sample_017.jpg` — `rms_px = 67.10`
4. `sample_005.jpg` — `rms_px = 68.09`

Shared pattern:

- large board footprint
- hand-held support or visually unstable support
- several views are strong enough to satisfy coverage, but still too inconsistent
   to support a trustworthy distortion fit

Representative contrast:

- `sample_011.jpg` is smaller, calmer, and more stable; it lands near `45 px`
   RMS in the current replay
- `sample_016.jpg` is a large hand-held oblique view; it lands near `77 px` RMS

### E. UI / interaction implication

The post-run heatmap should be treated as an operator-facing signal, not only a
diagnostic artifact.

This iteration therefore promotes the same idea into live capture:

- the live 3x3 overlay now shows **per-cell counts** instead of only binary fill
- the least-covered cells are highlighted during stage 1
- stage guidance now explicitly names the sparsest cells, matching the meaning of
   `image_coverage_heatmap.png`

This improves collection behavior, but it does **not** change the final judgment:

- the current recollection is still not release-ready
- round02 must still use a rigid flat board mount and a stricter pose plan

### F. Final implementation outcome from this iteration

Two operator-tooling changes were validated after the dataset review above:

1. the live heatmap/count overlay was repaired and revalidated with a mock frame
2. live capture now warns when a session reuses a non-empty `accepted/` directory

Practical implication:

- the heatmap concept from `image_coverage_heatmap.png` is now available during
   collection, not only after replay
- repeated use of the same `--session-name` is now called out explicitly, so a
   mixed capture directory is less likely to be misread as one clean round

This does not change the data verdict. It only makes the next collection round
easier to execute correctly.
