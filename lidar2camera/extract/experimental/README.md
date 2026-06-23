# experimental/ — superseded stepping-stones

These scripts were stages on the way to `extract/calibrate_p2plane.py`. They are
kept for reference but are **not** part of the supported pipeline.

- **`extract_pairs.py`** — early all-in-one that read Cyber bags directly and
  paired images with the nearest cloud, using a coarse pose-grouping selector.
  Superseded by `extract_pcd_from_bag.py` + `pair_livox.py` (timestamp filenames
  + farthest-point pose selection).
- **`find_board.py`** — standalone board detector (geometry + intensity stripe
  score). Found board-sized planes but could not reliably tell the real board
  from clutter/wall boards without a transform prior.
- **`crop_boards.py`** — PnP-bearing-gated board crop that overwrote the PCDs for
  use with `reference_based.py`. The bearing-only gate still caught confounders;
  replaced by the EM matcher inside `calibrate_p2plane.py`.
- **`auto_init.py`** — RANSAC over (camera centre <-> LiDAR board candidate)
  correspondences to derive the prior with no axis assumption. Sound in theory,
  but the board often isn't a reliable candidate in cluttered/oblique side
  scans, so it found too few inliers. The yaw-search in `calibrate_auto.py`
  (which scores by the full-bundle reprojection) proved far more robust.
- **`em_calibrate.py`** — first EM attempt using board *centers* (Kabsch) and
  normal-based rotation. Centers alone span mostly one axis (underconstrained
  rotation) and the sparse LiDAR normals were too noisy; replaced by the
  point-to-plane + center-anchor objective.
