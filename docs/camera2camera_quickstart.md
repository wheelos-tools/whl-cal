---
audience: user
stability: stable
P26-05-27
---

# Camera-to-camera quick start

Before running this tool:

1. Calibrate **each camera intrinsic first** with
   [docs/camera_quickstart.md](camera_quickstart.md).
2. For result review, keep
   [docs/calibration_review_guide.md](calibration_review_guide.md) open.
3. For design background and practical references, see
   [docs/camera2camera_design.md](camera2camera_design.md) and
   [docs/calibration_methodology.md](calibration_methodology.md).

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| paired parent / child images | yes | matched by filename stem for offline mode |
| parent intrinsic YAML | yes | use `camera/intrinsic.py` output or equivalent YAML |
| child intrinsic YAML | yes | same requirement as parent |
| calibration target config | yes | `checkerboard`, `aprilgrid`, or `charuco` |
| checkerboard / aprilgrid geometry | yes | meter unit |
| multi-pose board capture | yes | vary depth, tilt, and image location in both cameras |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Recommended directory layout

```text
run01/
  parent_intrinsics.yaml
  child_intrinsics.yaml
  parent/
    0001.png
    0002.png
  child/
    0001.png
    0002.png
```

The parent and child image directories must contain the **same stems**.

## Generate a starter config

```bash
camera2camera-calibrate \
  --write-default-config \
  --config camera2camera_config.yaml
```

Example config shape:

```yaml
cameras:
  parent:
    frame_id: camera_front_left
    image_directory: run01/parent
    intrinsics_path: run01/parent_intrinsics.yaml
  child:
    frame_id: camera_front_right
    image_directory: run01/child
    intrinsics_path: run01/child_intrinsics.yaml
target:
  type: checkerboard
  pattern_size: [11, 8]
  square_size_m: 0.025
extraction:
  min_bbox_area_ratio: 0.003
  min_edge_margin_px: 16.0
  max_pnp_reprojection_rms_px: 1.5
optimization:
  min_pairs: 8
  loss: huber
  f_scale: 1.0
  max_nfev: 300
metrics:
  warning_final_rms_px: 1.0
  warning_pair_rms_p95_px: 1.5
  warning_holdout_rms_px: 1.5
  warning_epipolar_p95_px: 1.0
output:
  directory: outputs/camera2camera/run01
```

## Run calibration

```bash
camera2camera-calibrate --config camera2camera_config.yaml
```

## Live stereo capture

Enable `live_capture.enabled: true` in the config, then point each camera source to
an RTSP URI or device index:

```yaml
cameras:
  parent:
    frame_id: camera_left
    intrinsics_path: run01/parent_intrinsics.yaml
    source:
      uri: rtsp://192.168.1.10/stream
      codec: h265
  child:
    frame_id: camera_right
    intrinsics_path: run01/child_intrinsics.yaml
    source:
      uri: rtsp://192.168.1.11/stream
      codec: h265
target:
  type: aprilgrid
  aprilgrid:
    dictionary: DICT_APRILTAG_36h11
    grid_cols: 6
    grid_rows: 6
    tag_size: 0.04
    tag_spacing_ratio: 0.3
live_capture:
  enabled: true
  provisional_eval_interval: 1
  auto_stop_on_release_ready: true
auto_capture_settings:
  min_total_samples: 12
```

Then run:

```bash
camera2camera-calibrate \
  --config camera2camera_config.yaml \
  --session-name round01_factory \
  --headless-live-max-seconds 300
```

Useful live flags:

1. `--capture-only` saves accepted stereo pairs and skips final calibration.
2. `--require-release-ready` returns non-zero if the final run is still review-only.
3. `--headless-live-max-seconds` is useful on servers without a display.

Live capture saves paired images under
`outputs/camera2camera/captures/<session>/parent` and `child`, keeps per-camera
coverage/diversity guidance during capture, and runs provisional stereo reviews as
the pair set grows. The GUI now shows:

1. parent / child views side by side
2. per-camera live diagnostics and heatmaps
3. the current stereo relative-pose panel from the last accepted / provisional result

## Pair-by-pair live interaction

The live workflow is now intentionally **single-pair incremental**:

1. Both cameras must detect the same target.
2. Both views must pass the stability gate.
3. Both views must pass coverage / novelty checks.
4. The pair must pass per-camera pose solve checks:
   - bbox not too small
   - margin from image border not too small
   - `solvePnP` succeeds
   - reprojection RMS stays below the extraction threshold
5. Only then is the pair saved.
6. Immediately after saving, the system writes `debug/sample_<N>_review.yaml` and
   tells the operator what to do next.

Before the dataset reaches the stereo minimum pair count, the review explains:

- whether the current pair itself is geometrically valid
- how many more pairs are needed before full stereo review
- whether to prioritize uncovered image regions or stronger pose diversity

After the minimum pair count is reached, every newly accepted pair also triggers a
provisional stereo review.

## Operator guidance and debugging

The live UI / headless logs now translate common failures into direct actions:

| live symptom | meaning | operator action |
| --- | --- | --- |
| target lost | board not reliably detected in one or both cameras | keep the full board visible, reduce glare, reduce motion blur, move slightly closer |
| board too small | target bbox area is too small for stable pose | move the board closer or use a larger target |
| near image edge | target margin is too small | move the board away from the image boundary |
| pose solve failed | detector succeeded but PnP is unstable | keep the board flatter, sharper, and fully visible |
| reprojection too high | image points are noisy or board is blurred / extreme-angle | hold steadier, improve focus/exposure, avoid grazing angles |
| pose not novel | this sample is too similar to prior accepted views | change depth, tilt, and lateral offset |
| inconsistent relative transform | current pair disagrees with the current stereo consensus | avoid unsynchronized board motion; keep both views stable on the same pose |

For factory collection, the recommended operator loop is:

1. Start with the board centered and medium distance so both cameras detect it easily.
2. Collect a few easy fronto-parallel pairs first to establish a stable stereo seed.
3. Then deliberately expand:
   - left / right image coverage
   - top / bottom coverage
   - near / far depth
   - low / high tilt
4. Stop only when the provisional stereo review turns `release_ready`, not merely
   when detection succeeds.

## Recommended targets

1. `aprilgrid` is the recommended live target because tag IDs resolve orientation
   ambiguity and partial visibility better than checkerboard.
2. `checkerboard` remains supported, but live collection uses conservative per-pair
   pose gating and then relies on the stereo provisional/final review to resolve
   ordering robustly across multiple pairs.
3. `charuco` is also supported when you want marker IDs plus chessboard-style
   corner refinement.

For AprilGrid, the detector now prefers `pupil_apriltags` and falls back to
OpenCV ArUco/AprilTag when that package is unavailable.

## Minimal validation

```bash
python tools/run_camera2camera_smoke.py --pairs 8
```

## Recommended review order

1. read `diagnostics/standardized_data.yaml`
2. read `diagnostics/data_quality.yaml`
3. read `metrics.yaml`
4. read `diagnostics/acceptance_report.yaml`
5. read `diagnostics/visualization_index.yaml`
6. inspect `diagnostics/per_pair_reprojection.csv`
7. inspect `diagnostics/leave_one_out_trials.csv`
8. inspect both image-coverage heatmaps and `pose_diversity_plot.png`
9. inspect `diagnostics/epipolar_previews/`

Outputs include:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `initial_guess/<parent>_<child>_extrinsics.yaml`
- `calibrated/<parent>_<child>_extrinsics.yaml`
- `diagnostics/`
  - `reference_dataset.yaml`
  - `extraction.yaml`
  - `optimization.yaml`
  - `evaluation.yaml`
  - `acceptance_report.yaml`
  - `status_summary.csv`
  - `standardized_data.yaml`
  - `data_quality.yaml`
  - `visualization_index.yaml`
  - `extraction_entries.csv`
  - `per_pair_reprojection.csv`
  - `leave_one_out_trials.csv`
  - `parent_image_coverage_heatmap.png`
  - `child_image_coverage_heatmap.png`
  - `pose_diversity_plot.png`
  - `epipolar_previews/`

## Acceptance baseline

- final RMS < `1.0 px`
- per-pair RMS p95 < `1.5 px`
- holdout RMS p95 < `1.5 px`
- epipolar error p95 < `1.0 px`
- accepted pair ratio remains healthy
- both parent and child image coverage span multiple grid cells
- pose diversity shows meaningful depth and tilt variation

Treat missing holdout evidence, weak coverage, or warning-level repeatability as
**review-only**, not release-ready.
