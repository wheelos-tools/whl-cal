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
| pattern size | yes | inner-corner count, for example `[11, 8]` |
| square size | yes | meter unit |
| fixed camera mode | yes | keep resolution / exposure / focus stable during capture |
| Apollo bag | optional | useful for traceability, but the tool itself consumes live frames or image files |

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
  force_resolution: false
  width: null
  height: null
  fourcc: null
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
  samples_per_grid: 1
  delay_between_captures: 1.0
  stability_frames: 5
  stability_threshold: 2.0
```

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

## Quick runs

### Interactive live capture

```bash
python camera/intrinsic.py --config camera_config.yaml
```

### Headless validation from an image directory

```bash
python camera/intrinsic.py \
  --config tmp_config.yaml \
  --images-dir /path/to/images \
  --pattern-size 4,3
```

If your images were originally recorded in Apollo, export them with your normal
image-extraction workflow first. The intrinsic tool does not consume `.record`
files directly.

## Recommended review order

1. read `*_diagnostics/data_quality.yaml`
2. read `*_diagnostics/per_view_reprojection.csv`
3. inspect `*_diagnostics/image_coverage_heatmap.png`
4. inspect `comparison_view.png`
5. inspect the calibration YAML's `capture_runtime` and `undistortion_preview`

Outputs include:

- `calibration_YYYYmmdd_HHMMSS.yaml`
- `comparison_view.png`
- `calibration_YYYYmmdd_HHMMSS_diagnostics/`
  - `acceptance_report.yaml`
  - `status_summary.csv`
  - `standardized_data.yaml`
  - `data_quality.yaml`
  - `visualization_index.yaml`
  - `per_view_reprojection.csv`
  - `sample_records.csv`
  - `image_coverage_heatmap.png`

The YAML also records `distortion_model` plus an `undistortion_preview` section
with `alpha`, `optimized_camera_matrix`, and the valid ROI.

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
