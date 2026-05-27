---
audience: user
stability: stable
P26-05-25
---

# Camera intrinsic quick start

Before running this tool:

1. If you are collecting through Apollo, read
   [docs/apollo_data_collection.md](apollo_data_collection.md).
2. For result review, keep
   [docs/calibration_review_guide.md](calibration_review_guide.md) open.
3. For design background and references, see
   [docs/calibration_methodology.md](calibration_methodology.md).

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| calibration board images | yes | either from live capture or an exported image directory |
| target type | yes | `chessboard` / `aprilgrid` / `charuco` |
| board geometry parameters | yes | chessboard: `pattern_size`+`square_size`; aprilgrid/charuco: target-specific dimensions |
| fixed camera mode | yes | keep resolution / exposure / focus stable during capture |
| Apollo bag | optional | useful for traceability, but the tool itself consumes live frames or image files |

## Recommended directory layout

The intrinsic tool now separates capture datasets from calibration outputs:

```text
outputs/camera_intrinsic/
  captures/
    round01_aprilgrid/
      accepted/
        sample_001.jpg
        sample_002.jpg
      debug/
        headless_first_frame.jpg
      capture_session.yaml
  runs/
    20260526_190000_round01_aprilgrid/
      calibration.yaml
      comparison_view.png
      calibration_diagnostics/
```

Do not use the repo root `.` as `--images-dir`. Use a dedicated dataset directory such as
`outputs/camera_intrinsic/captures/round01_aprilgrid/accepted`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy pyyaml
```

## Recommended capture-first config

```yaml
camera_index: 0
window_name: Industrial Calibration Tool
window_width: 1280
window_height: 720
capture:
  buffersize: 1
  force_resolution: false
  width: null
  height: null
  fourcc: null
  warmup_frames: 12
  reconnect_bad_frame_burst: 30
  latest_frame_read_timeout_s: 0.25
  initial_ready_timeout_s: 10.0
  reconnect_sleep_s: 0.5
distortion_model: plumb_bob
pattern_size: [11, 8]
square_size: 0.025
optimization:
  resize_factor: 0.5
  detection_interval: 2
undistortion_preview:
  alpha: 1.0
  center_principal_point: false
auto_capture_settings:
  grid_shape: [3, 3]
  min_total_samples: 9
  pose_novelty_center_distance_ratio: 0.08
  pose_novelty_area_delta: 0.02
  pose_novelty_aspect_delta: 0.12
  samples_per_grid: 1
  delay_between_captures: 1.0
  stability_frames: 5
  stability_threshold: 2.0
  stability_threshold_ratio: 0.02
```

## AprilGrid config (production recommendation)

```yaml
target_type: aprilgrid
cameras:
  - { uri: "rtsp://admin:***@192.168.1.68:554/live", codec: "h265", width: 1920, height: 1080, fps: 25 }
capture:
  buffersize: 1
  force_resolution: false
  width: null
  height: null
  fourcc: null
  warmup_frames: 12
  reconnect_bad_frame_burst: 30
  latest_frame_read_timeout_s: 0.25
  initial_ready_timeout_s: 10.0
  reconnect_sleep_s: 0.5
aprilgrid:
  dictionary: DICT_APRILTAG_36h11
  grid_cols: 6
  grid_rows: 6
  tag_size: 0.04
  tag_spacing_ratio: 0.3
  min_tags_per_frame: 6
optimization:
  resize_factor: 0.5
  detection_interval: 2
workflow:
  root_dir: outputs/camera_intrinsic
  save_live_accepted_frames: true
auto_capture_settings:
  grid_shape: [3, 3]
  min_total_samples: 9
  pose_novelty_center_distance_ratio: 0.08
  pose_novelty_area_delta: 0.02
  pose_novelty_aspect_delta: 0.12
  samples_per_grid: 1
  delay_between_captures: 1.0
  stability_frames: 5
  stability_threshold: 2.0
  stability_threshold_ratio: 0.02
```

Notes:

- `tag_size` and `tag_spacing_ratio` must match your physically printed board.
- `target_type: aprilgrid` switches detection and calibration away from chessboard corners.
- Keep AprilGrid fully visible in enough poses; partial visibility is supported but too few tags per frame will be skipped.
- Auto-capture now works in two stages instead of a single `3x3 == done` rule.
- Stage 1 is spatial coverage: fill the image plane so the board appears across different image regions.
- Stage 2 starts after coverage is full: keep the same session and collect additional novel poses until `min_total_samples` is reached.
- `min_total_samples` is the minimum sample floor, not a synonym for coverage. A large close board can fill multiple cells in one shot, so coverage completion alone must not terminate collection.
- Once coverage is complete, additional stable samples are accepted only when pose novelty passes the center / scale / aspect thresholds above. The UI now explicitly shows `collect N more novel poses` and the next recommended action.
- When Stage 2 says you still need more novel poses, do not restart by default. Continue the same session and intentionally change one of these dimensions:
- change distance: move closer or farther so the board size changes by roughly 15-25%
- change tilt: tilt left/right or up/down by roughly 10-20 degrees
- change center only after changing distance or tilt; translation alone is often too weak once coverage is already complete
- Restart the session only if the accepted images are blurred / cropped / poorly detected, or if calibration finishes but the final quality gates still fail.
- `9` is no longer the whole logic by itself. The current completion rule is: coverage complete, stable frames, novelty-gated acceptance for post-coverage samples, and sample count at or above `min_total_samples`; final trust still comes from the post-calibration quality gates.
- If live RTSP preview feels delayed, keep `capture.buffersize: 1`, increase `optimization.detection_interval`, or lower `optimization.resize_factor` before touching the display window size.
- For AprilGrid, `optimization.resize_factor` is treated as a fast first-pass candidate; the detector still falls back to native resolution so lowering it should not silently disable detection on medium-size tags.
- For unstable RTSP/H265 streams, keep `capture.warmup_frames` and `capture.reconnect_bad_frame_burst` enabled so startup corruption and short decode bursts do not abort the session.

## ChArUco config (second board option)

```yaml
target_type: charuco
charuco:
  dictionary: DICT_4X4_100
  squares_x: 6
  squares_y: 8
  square_length: 0.04
  marker_length: 0.02
  min_corners_per_frame: 12
optimization:
  resize_factor: 1.0
  detection_interval: 1
auto_capture_settings:
  grid_shape: [3, 3]
  samples_per_grid: 1
  delay_between_captures: 1.0
  stability_frames: 5
  stability_threshold: 2.0
```

Notes:

- `marker_length` must be smaller than `square_length`.
- ChArUco corner IDs are used for calibration, not plain checker corners.

If the live 3x3 acquisition grid already looks clipped before calibration, check
the capture config first. The tool separates **capture resolution** from
**display window size**:

```yaml
camera_index: 0
window_width: 1280
window_height: 720
capture:
  force_resolution: false
  width: null
  height: null
  fourcc: null
```

Recommended practice:

- keep `capture.force_resolution: false` unless you have verified the camera's
  native mode
- if you must force a mode, prefer a native sensor aspect ratio instead of
  forcing `1280x720` blindly
- confirm native modes with `v4l2-ctl --list-formats-ext` before overriding
- use the same mode you will deploy in production
- the `1280x720` window is only a padded preview canvas; accepted samples stay at
  the original capture size and `capture_session.yaml.capture_runtime.display_rendering`
  records the exact mapping from capture pixels to display pixels

## Quick runs

Preferred entrypoint after `pip install -e .`:

```bash
camera-intrinsic-calibrate --config camera_config.yaml
```

`python -m camera.cli` provides the same command during source-tree development.
`python camera/intrinsic.py` is kept as a compatibility shim, not the primary CLI.

### Round 1: collect a reusable dataset from live camera

Use `--capture-only` when you want a clean collection pass first. Accepted samples
are exported automatically.

```bash
camera-intrinsic-calibrate \
  --config camera_config.yaml \
  --session-name round01 \
  --capture-only
```

If the machine is headless:

```bash
camera-intrinsic-calibrate \
  --config camera_config.yaml \
  --session-name round01 \
  --capture-only \
  --headless-live-max-seconds 180
```

Look for accepted images under:

```text
outputs/camera_intrinsic/captures/round01_aprilgrid/accepted
```

Before trusting that live collection, open the capture manifest and check:

- `outputs/camera_intrinsic/captures/round01_aprilgrid/capture_session.yaml`
- `capture_runtime.actual_capture_resolution` matches the camera mode you intended to use
- `capture_runtime.display_rendering.aspect_ratio_preserved: true`
- `capture_runtime.display_rendering.render_width` / `render_height` plus `pad_x` / `pad_y` explain exactly how the native frame was letterboxed into the preview window
- accepted sample images still have the native capture size instead of the preview size

### Round 1: calibrate from the captured dataset

```bash
camera-intrinsic-calibrate \
  --config camera_config.yaml \
  --images-dir outputs/camera_intrinsic/captures/round01_aprilgrid/accepted \
  --require-release-ready
```

### One-shot interactive live capture + calibration

```bash
camera-intrinsic-calibrate --config camera_config.yaml
```

### Headless validation from an image directory

```bash
camera-intrinsic-calibrate \
  --config tmp_config.yaml \
  --images-dir /path/to/dataset/accepted
```

After any calibration run, review these artifacts in order:

- `outputs/camera_intrinsic/runs/<session>/calibration_diagnostics/acceptance_report.yaml`
- `outputs/camera_intrinsic/runs/<session>/calibration_diagnostics/per_view_reprojection.csv`
- `outputs/camera_intrinsic/runs/<session>/calibration_diagnostics/image_coverage_heatmap.png`
- `outputs/camera_intrinsic/runs/<session>/comparison_view.png`
- `outputs/camera_intrinsic/runs/<session>/calibration.yaml`

`comparison_view.png` should show a real distorted source frame on the left and the undistorted preview on the right. If the left side is blank, the replay or capture path is not preserving the source frame correctly and the run is not reviewable.

For `target_type: chessboard`, you can still use `--pattern-size W,H` as an override.

### Production strict release gate

```bash
camera-intrinsic-calibrate \
  --config camera_config.yaml \
  --images-dir /path/to/images \
  --require-release-ready
```

When `--require-release-ready` is set, the command exits non-zero if diagnostics
`release_ready` is false.

## Iteration loop

Use the same loop every round:

1. Collect one named dataset with `--capture-only --session-name roundNN`.
2. Check `capture_session.yaml` and make sure at least 9 accepted images were exported.
3. In the same `capture_session.yaml`, confirm `capture_runtime.actual_capture_resolution`,
   `display_rendering`, and `stream_health` match expectations.
3. Re-run calibration from `captures/<session>/accepted` with `--require-release-ready`.
4. Review `runs/<timestamp>_<session>/calibration_diagnostics/data_quality.yaml` and `comparison_view.png`.
5. If `release_ready: false`, recollect another round with wider image coverage, stronger tilt variation, and a larger board footprint near image edges.

If your images were originally recorded in Apollo, export them with your normal
image-extraction workflow first. The intrinsic tool does not consume `.record`
files directly.

## Recommended review order

1. read `runs/<session>/calibration_diagnostics/data_quality.yaml`
2. read `runs/<session>/calibration_diagnostics/per_view_reprojection.csv`
3. inspect `runs/<session>/calibration_diagnostics/image_coverage_heatmap.png`
4. inspect `runs/<session>/comparison_view.png`
5. inspect `runs/<session>/calibration.yaml`

Outputs include:

- `outputs/camera_intrinsic/captures/<session>/accepted/*.jpg`
- `outputs/camera_intrinsic/captures/<session>/capture_session.yaml`
- `outputs/camera_intrinsic/runs/<timestamp>_<session>/calibration.yaml`
- `outputs/camera_intrinsic/runs/<timestamp>_<session>/comparison_view.png`
- `outputs/camera_intrinsic/runs/<timestamp>_<session>/calibration_diagnostics/`
  - `acceptance_report.yaml`
  - `status_summary.csv`
  - `standardized_data.yaml`
  - `data_quality.yaml`
  - `visualization_index.yaml`
  - `per_view_reprojection.csv`
  - `sample_records.csv`
  - `image_coverage_heatmap.png`

The YAML also records `distortion_model` plus an `undistortion_preview` section
with `alpha`, `optimized_camera_matrix`, the valid ROI, and explicit input/output
image-size fields so you can verify that undistortion preserves the capture resolution.

For RTSP/H265, the managed reader now waits through the initial startup grace window
for the first valid keyframe instead of immediately reconnecting on every early decode
error. That keeps low-latency preview without the reconnect storm that can prevent the
stream from ever reaching a stable GOP.

## Acceptance baseline

- average reprojection error < `1.0 px`
- per-view reprojection p95 < `1.5 px`
- image coverage spans multiple grid cells, not only the center
- radial distortion remains monotonic over the image radius

Treat `radial_monotonicity: warning` as calibration failure, not as a cosmetic
issue.

## Next docs

- LiDAR↔Camera extrinsic workflow:
  [docs/lidar2camera_quickstart.md](lidar2camera_quickstart.md)
- shared review guide:
  [docs/calibration_review_guide.md](calibration_review_guide.md)
