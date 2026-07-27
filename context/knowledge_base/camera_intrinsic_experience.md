---
audience: dev
stability: stable
last_updated: 2026-07-27
---

# Camera intrinsic calibration experience

This note is the compact, non-duplicated knowledge base for `camera` intrinsic calibration.
Use it as the single context entry before reading historical run notes.

## 1. Baseline and scope

- Entry point: `camera-intrinsic-calibrate` (or `python -m camera.cli`)
- Pipeline split stays fixed:
  1. data capture / screening
  2. intrinsic solve
  3. diagnostics / acceptance
- Supported targets: `chessboard`, `aprilgrid`, `charuco`

## 2. Lens model policy

- `distortion_model: plumb_bob`:
  - aliases: `pinhole`, `opencv`, `radtan`
  - uses OpenCV pinhole model (`cv2.calibrateCamera`)
- `distortion_model: fisheye`:
  - alias: `equidistant`
  - uses OpenCV fisheye model (`cv2.fisheye.calibrate`)
- Do not mix models across rounds when comparing quality trends.

## 3. Capture and config rules

- Keep capture resolution separate from display window:
  - `window_width/window_height` are UI only
  - preserve native capture mode unless you have verified a required forced mode
- Keep one stable camera mode per run (resolution/focus/exposure).
- Use staged auto-capture behavior:
  - first complete coverage
  - then collect novel poses until sample target is met

Recommended chessboard baseline config is tracked in `conf/camera_config_chess.yaml`.

## 4. Solver/evaluation contract

- Reprojection metric is **2D RMS per view**, then averaged across valid views.
- Do not use `L2 / N`; correct 2D RMS is `L2 / sqrt(N)`.
- Quality gates should always include:
  - sample count
  - image coverage
  - average reprojection
  - per-view reprojection p95
  - radial monotonicity
  - sample-image-size consistency

## 5. Artifacts to trust

Check in this order:

1. `calibration_diagnostics/acceptance_report.yaml`
2. `calibration_diagnostics/data_quality.yaml`
3. `calibration_diagnostics/per_view_reprojection.csv`
4. `calibration_diagnostics/image_coverage_heatmap.png`
5. `comparison_view.png`
6. `calibration.yaml`

Critical fields:

- `calibration.yaml`:
  - `distortion_model`
  - `camera_matrix`
  - `distortion_coefficients`
  - `capture_runtime`
  - `undistortion_preview`
- diagnostics:
  - `status_summary.csv`
  - `sample_records.csv`
  - `visualization_index.yaml`

## 6. Common failure patterns

- Low reprojection with center-heavy samples can still fail downstream robustness.
- Mixed capture resolutions in one run invalidates the intrinsic result.
- Incorrect lens model selection can present as unstable or non-monotonic distortion.
- A visually broken `comparison_view.png` means the run is not reviewable.
