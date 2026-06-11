# LiDAR → Camera Extrinsic Calibration

Estimate the rigid transform between a LiDAR and a camera using a checkerboard
that is moved freely in front of both sensors. The camera gives a precise board
plane per frame (sub-pixel corners + known intrinsics); the LiDAR gives the
board point cloud. We align the two across many poses.

The current method lives in **`extract/`** and is driven by **`config.yaml`**.
`reference_based.py` and `learning_based.py` are **deprecated** (see bottom).

---

## Method (why this one)

A handheld board in a real scene defeats naïve target calibration:

- **Cluttered LiDAR** — RANSAC on the full cloud locks onto a wall/floor, not the
  0.4 × 0.5 m board.
- **Static confounders** — calibration boards mounted on walls sit at a fixed
  spot and get mistaken for the moving board.
- **Sparse board** — a Livox board at ~3–4 m has only a few hundred points, so a
  single plane-normal estimate is noisy.

`extract/calibrate_p2plane.py` handles all three:

1. **EM / ICP matching** — the camera PnP predicts where the board should be in
   the LiDAR; we crop a tight box there and RANSAC the board plane, then update
   the transform and repeat. A static object cannot follow the camera's moving
   trajectory, so it is rejected.
2. **Point-to-plane bundle** — optimise the 6-DoF extrinsic so every LiDAR board
   point lands on its (accurate) camera-derived board plane. Uses *all* board
   points, so it is robust to the noisy per-board LiDAR normal.
3. **Center anchor** — fronto-parallel boards leave lateral/vertical translation
   unconstrained by point-to-plane alone, so a board-center-to-center term pins
   the in-plane translation.

Output is verified by reprojecting the LiDAR board points onto the image.

---

## Inputs

- **Camera**: an H.264/H.265 `.mkv` plus a `*.frames.csv` sidecar that maps each
  frame to `clk_unix_ns` (NTP wall clock).
- **LiDAR**: Apollo Cyber RT record bags (`all_*`) containing a
  `apollo.drivers.PointCloud` channel.

Cameras and the compute share NTP, so `clk_unix_ns` (camera) and
`header.lidar_timestamp` (LiDAR) are directly comparable — that is the sync key.

> The board must be held **upright** and **still** for ~1–2 s at each pose.

---

## Pipeline

Edit the `data:` and `camera:`/`checkerboard:` sections of `config.yaml` first.
Dependencies: `pip install -e .` plus `opencv-python`, and `ffmpeg` on PATH.

### Step 0 — extract LiDAR PCDs from the bags  (run where the bags live, e.g. the Orin)

Needs only the pure-Python reader, **no Apollo runtime**:

```bash
pip install --user cyber_record protobuf==3.19.4
python3 extract/extract_pcd_from_bag.py \
    --bags '/path/to/run/bag/all_*' \
    --channel /apollo/sensor/livox/front/PointCloud2 \
    --out /path/to/livox_front_pcd --prefix livox_front
tar -czf livox_front_pcd.tar.gz -C /path/to livox_front_pcd   # copy to workstation
```

Each cloud is written as `livox_front_<seq>_<lidar_ts_ns>.pcd`; the timestamp in
the filename is what Step 2 syncs on.

### Step 1 — detect the checkerboard in the video

```bash
python3 extract/detect_camera.py
```

Samples every `detect_stride`-th frame, runs sub-pixel corner detection, and
writes `extract/cam_candidates/candidates.json` (+ a PNG per detection).

### Step 2 — pair camera poses with LiDAR clouds

```bash
python3 extract/pair_livox.py
```

Selects `n_poses` diverse, stationary poses (farthest-point sampling over image
position + board size) and matches each to the nearest cloud within
`sync_tol_ms`. Writes paired `NNNN.png` / `NNNN.pcd` into `calibration_data/`.

### Step 3 — calibrate

```bash
python3 extract/calibrate_p2plane.py
```

Runs the EM match + point-to-plane bundle and writes:

- `calibration_output/lidar2camera_extrinsic.yaml` — extrinsic (both directions)
  + quality metrics + kept poses.
- `calibration_output/verification_overlay.png` — LiDAR board points (red) and
  the full cloud projected onto the **distorted** image.

### Step 4 — undistorted overlay (optional)

```bash
python3 extract/overlay_undistort.py
```

Same overlay on the **rectified** image (`cv2.undistort`, points projected with
zero distortion) → `calibration_output/verification_overlay_undistorted.png`.

---

## Outputs & quality

`lidar2camera_extrinsic.yaml` reports:

- `extrinsic_lidar_to_camera` / `extrinsic_camera_to_lidar` — 4×4 matrices.
- `point_to_plane_rms_mm` — how flat board points sit on the camera plane.
- `board_center_reproj_px_median/mean` — board-center reprojection error.
- `kept_poses` — poses that survived outlier rejection.

Always eyeball the verification overlay: the red points must sit on the held
board and the projected floor must follow the real floor.

### Reference result (cam2 / 192.168.1.64 ↔ livox_front, warehouse capture)

`point_to_plane RMS ≈ 32 mm`, `board-center reprojection ≈ 14 px median` over 9
clean poses — a usable rough/moderate extrinsic. Precision was limited by the
**capture**, not the method: a sparse handheld Livox board, ~30 ms residual
sync, and the board staying mostly center-front and fronto-parallel.

### For a precise calibration, recapture with

- the board held **still** (~1–2 s) at each pose;
- **many distinct image positions** (incl. the four corners), multiple ranges
  (2–6 m), and varied **tilts** (±30° pitch/yaw/roll);
- the board **away** from any wall-mounted boards / clutter.

---

## Files

```
config.yaml                      sensor intrinsics, board, data paths, params
extract/
  extract_pcd_from_bag.py        step 0  bag  -> PCD  (pure-python cyber_record)
  detect_camera.py               step 1  video -> candidates.json
  pair_livox.py                  step 2  pose select + sync -> calibration_data/
  calibrate_p2plane.py           step 3  EM match + point-to-plane bundle
  overlay_undistort.py           step 4  rectified verification overlay
  experimental/                  superseded stepping-stones (see its README)
reference_based.py               DEPRECATED — plane-only, wrong corner corresp.
learning_based.py                DEPRECATED — learned method, not maintained
```

`reference_based.py` is kept only for reference; it gave ~100 px error on real
data because it fabricates board corners with an arbitrary in-plane rotation.
Use `extract/calibrate_p2plane.py`.
