# lidar2camera current context

## 1. Current codebase state

The repository already has camera-related code, but it is not yet organized into
the same repo-wide calibration framework used by `lidar2lidar` and `lidar2imu`.

Current files:

- `camera/intrinsic.py`
  - interactive chessboard-based camera intrinsic calibration
- `camera/README.md`
  - quick start for the intrinsic tool
- `camera2lidar/reference_based.py`
  - checkerboard / reference-based LiDAR-camera extrinsic calibration
- `camera2lidar/learning_based.py`
  - targetless LiDAR-camera calibration experiment

## 2. Gap versus the current repo standard

Compared with `lidar2lidar` and `lidar2imu`, camera-related calibration is still
missing a repo-level split into:

1. **data layer**
   - explicit dataset artifact
   - pairing / synchronization diagnostics
   - sample or window gating
2. **algorithm layer**
   - stable command entrypoints
   - comparable reference-based vs targetless paths
3. **evaluation layer**
   - stable `metrics.yaml`
   - stable `diagnostics/*.yaml`
   - explicit acceptance recommendation

## 3. Recommended next direction

The next repo-level calibration target should be **lidar2camera**.

Recommended pattern:

### A. Data layer

- define an explicit dataset artifact for image + LiDAR pairs
- add window + gate logic:
  - sync quality
  - board detection quality or feature coverage quality
  - image blur / visibility gates
  - point-cloud support / plane quality gates

### B. Algorithm layer

Keep two branches visible:

1. **reference-based**
   - checkerboard / target-based
   - stronger acceptance path
2. **targetless**
   - scene / feature-based
   - comparison and diagnostic path first

### C. Evaluation layer

Use stable outputs such as:

- reprojection error
- holdout reprojection error
- board-plane consistency
- LiDAR-to-image overlay quality
- transform drift to initial prior
- repeatability across poses / windows
- recommendation field

## 4. Repo-level rule for this module

`lidar2camera` should follow the same principle as the other modules:

- algorithms and metrics keep iterating
- conclusions are driven by tested data
- validated conclusions and open verification points must stay separate

## 5. Immediate next step

Before changing algorithms, first document and stabilize:

1. current command entrypoints
2. expected inputs / outputs
3. evaluation surfaces
4. what counts as an accepted run vs a diagnostic run
