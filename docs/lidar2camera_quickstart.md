---
audience: user
stability: stable
---

# LiDAR→Camera quick start

Calibrate the rigid transform between a LiDAR and a camera from a checkerboard
waved in front of both sensors. This documents the `lidar2camera/extract/`
pipeline (point-to-plane bundle). The legacy `reference_based.py` is deprecated
(~100 px error on real data); do not use it.

Full reference: [`lidar2camera/README.md`](../lidar2camera/README.md).

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| camera video | yes | H.264/H.265 `.mkv` |
| per-frame sidecar | yes | `*.frames.csv` with a `clk_unix_ns` column (NTP wall clock) |
| LiDAR PCDs | yes | extracted from Apollo Cyber bags, named `<prefix>_<seq>_<lidar_ts_ns>.pcd` |
| camera intrinsics | yes | 3x3 matrix + 5-param distortion, in `config.yaml` |
| checkerboard pattern / square | yes | inner-corner count `(cols, rows)` and square size (m) |

Sync key: camera `clk_unix_ns` ↔ LiDAR `header.lidar_timestamp` — both NTP unix
ns, directly comparable. No `/tf` needed.

## Recording prerequisites

- Record a Cyber bag containing the LiDAR `PointCloud2` channel, and a camera
  `.mkv` + `.frames.csv` for the camera that overlaps that LiDAR's view.
- Hold the board **upright** and **still** (~1–2 s) at each placement.
- Span the image (incl. corners), multiple distances (2–6 m) and tilts (±30°).
- Keep the board away from wall-mounted boards / reflective clutter.
- Cameras and the compute box must share NTP.

## Per-run layout

Each capture is self-contained under `lidar2camera/runs/<id>/`:

```
runs/<id>/
  inputs/             video + frames.csv + lidar PCDs
  cam_candidates/     step-1 detections
  calibration_data/   step-2 paired NNNN.png / NNNN.pcd
  calibration_output/ step-3 extrinsic + overlays
```

`config.yaml` selects the active run with `run_dir:`; all scripts resolve paths
through `extract/runpaths.py`, so the steps take no path arguments.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install opencv-python            # plus ffmpeg on PATH
```

## Quick run

Step 0 — extract LiDAR PCDs from the bags (run where the bags live, e.g. the
Orin; needs only the pure-python reader, no Apollo runtime):

```bash
pip install --user cyber_record protobuf==3.19.4
python3 lidar2camera/extract/extract_pcd_from_bag.py \
  --bags '/path/to/run/bag/all_*' \
  --channel /apollo/sensor/livox/front/PointCloud2 \
  --out lidar2camera/runs/<id>/inputs/livox_front_pcd --prefix livox_front
```

Steps 1–4 — point `run_dir` and the `camera:` block at this capture in
`config.yaml`, then:

```bash
python3 lidar2camera/extract/detect_camera.py      # 1  video -> candidates.json
python3 lidar2camera/extract/pair_livox.py         # 2  select + sync -> calibration_data/
python3 lidar2camera/extract/calibrate_p2plane.py  # 3  EM match + point-to-plane bundle
python3 lidar2camera/extract/overlay_undistort.py  # 4  rectified verification overlay (optional)
```

**Misaligned camera/LiDAR pairs.** `calibrate_p2plane.py` assumes the camera and
LiDAR look roughly the same way. If they sit at different yaws (a side camera +
a differently-mounted side lidar) the solve diverges or returns an implausible
translation (|t| ≳ 0.5 m). Use the yaw-search wrapper instead — it tries
relative yaws, keeps the lowest-reprojection physically-plausible result, and
the bundle still refines full 6-DoF:

```bash
python3 lidar2camera/extract/calibrate_auto.py     # 3'  for misaligned pairs
```

## Recommended minimal config shape

```yaml
run_dir: runs/20260610_171231          # active capture under runs/
camera:
  intrinsics: [[1377.06, 0.0, 951.30], [0.0, 1377.46, 531.46], [0.0, 0.0, 1.0]]
  distortion: [-0.4184, 0.2198, -0.000132, -0.000313, -0.0679]   # plumb_bob
checkerboard:
  pattern_size: [8, 11]                # interior corners (cols, rows)
  square_size: 0.045                   # metres
data:
  camera_video:  inputs/cam2_192.168.1.64_00000.mkv
  camera_csv:    inputs/cam2_192.168.1.64.frames.csv
  livox_pcd_dir: inputs/livox_front_pcd
  detect_stride: 3
  max_frame: 0                         # 0 = whole video
  n_poses: 24
  sync_tol_ms: 60
```

(The `camera:` block is per-camera — update it together with `run_dir` when you
switch runs. `point_cloud:` / `optimization:` keys are legacy, used only by the
deprecated `reference_based.py`.)

## Outputs

Written to `runs/<id>/calibration_output/`:

- `lidar2camera_extrinsic.yaml` — `extrinsic_lidar_to_camera` /
  `extrinsic_camera_to_lidar` (4x4), plus `point_to_plane_rms_mm`,
  `board_center_reproj_px_median/mean`, `n_poses`, `kept_poses`.
- `verification_overlay.png` — LiDAR board points (red) + full cloud projected on
  the distorted image.
- `verification_overlay_undistorted.png` — same on the rectified image.

## How to judge a result

1. **Open the overlay first.** The red board points must sit on the held board,
   and the projected floor/structure must follow the real scene. This is the
   strongest evidence — trust it over a single number.
2. **Metrics** (capture-quality dependent, not hard gates):
   - `point_to_plane_rms_mm` — how flat board points sit on the camera plane;
     ~30 mm is typical, well-determined (rotation + range).
   - `board_center_reproj_px_median` — in-plane accuracy. A clean forward,
     fronto-parallel capture reaches ~15 px; oblique side captures land ~30–40 px
     and the translation wobbles run-to-run.
   - sane baseline: `|t|` should match the physical sensor separation
     (cm–dm), not metres.
3. **If the result is poor**, it is almost always the capture, not the method:
   sparse/oblique board, ~30 ms motion blur, or the board staying center-front
   and fronto-parallel. Recapture with more image coverage, ranges, and tilts.

## CAD cross-check (optional)

If you have the sensors' CAD-model positions, compare the baseline distance to
`|t|` from the extrinsic (frame-independent). A full rotation check needs the
sensors' CAD orientations, not just positions.
