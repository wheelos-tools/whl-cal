---
audience: user
stability: stable
P26-04-27
---


# Camera Intrinsic Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy pyyaml
```

Interactive:
```bash
python camera/intrinsic.py --config camera_config.yaml
```

Recommended capture-first config:

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

If the live 3x3 acquisition grid already looks clipped before calibration,
check the capture config first. The tool now separates **capture resolution**
from **display window size**:

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

- Keep `capture.force_resolution: false` unless you have verified the camera's
  native mode.
- If you must force a mode, prefer a native sensor aspect ratio (commonly 4:3)
  instead of forcing 1280x720 blindly.
- Confirm native modes with `v4l2-ctl --list-formats-ext` before overriding
  capture resolution.
- The app overlays live capture diagnostics so you can distinguish **sensor
  crop during acquisition** from **letterboxing in the display window**.

Recommended review order:

1. Start with `capture.force_resolution: false`.
2. Confirm the live 3x3 grid is fully visible before collecting any samples.
3. Only if needed, switch to a verified native capture mode and retry.
4. After calibration, inspect `comparison_view.png` plus the YAML's
   `capture_runtime` and `undistortion_preview` fields.

Headless (images dir):
```bash
python camera/intrinsic.py --config tmp_config.yaml --images-dir /path/to/images --pattern-size 4,3
```

Outputs: `calibration_YYYYmmdd_HHMMSS.yaml`, `comparison_view.png`.

The YAML now also records `distortion_model` plus an `undistortion_preview`
section with `alpha`, `optimized_camera_matrix`, and the valid ROI. This helps
distinguish "full-FOV undistortion preview" from the all-valid crop window when
the display would otherwise look clipped.

Acceptance: avg reprojection error < 1.0 px. See docs/lidar2camera_quickstart.md for extrinsic workflow.
