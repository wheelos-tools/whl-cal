# lidar2camera current context

## 1. Current codebase state

The repository already has camera-related code, but it is not yet organized into
the same repo-wide calibration framework used by `lidar2lidar` and `lidar2imu`.

Current files:

- `camera/intrinsic.py`
  - interactive chessboard-based camera intrinsic calibration (supports headless --images-dir)
- `docs/camera_quickstart.md`
  - quick start for the intrinsic tool and headless usage
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

## 6. 2026-04-24 industrial lidar2camera plan

The next industrial lidar2camera direction is now clearer after reviewing both the
existing scripts and repo-wide practice in `lidar2lidar` / `lidar2imu`.

### 6.1 Repo-level baseline decision

For this repo, the industrial baseline should be:

1. **reference-based / target-based**
   - checkerboard / ChArUco / AprilTag-grid style workflow
   - multi-pose joint optimization
   - stable metrics and diagnostics
   - this is the release path
2. **targetless / learning-based**
   - comparison / initialization / drift-monitoring path first
   - do not treat it as the first production release path

This matches common industry practice:

- use explicit observable targets for the first high-precision offline baseline
- keep targetless methods as a secondary path until they show comparable stability,
  repeatability, and uncertainty under tested data

### 6.2 Industrial architecture for this module

`lidar2camera` should now follow the same repo-wide split:

1. **data extraction**
   - pair image / PCD files explicitly
   - detect 2D board corners
   - detect LiDAR board plane / board support
   - record skip reasons and quality diagnostics
2. **algorithm**
   - choose an initial transform from a best single-pose candidate or explicit prior
   - optimize one shared LiDAR->camera transform jointly over all accepted poses
   - keep the targetless path separate
3. **evaluation**
   - stable `metrics.yaml`
   - stable `diagnostics/`
   - repeatability and uncertainty instead of one-shot promotion

### 6.3 Current industrial recommendation

The first industrial-grade baseline for this repo should use:

- camera intrinsic already calibrated first
- target-based LiDAR-camera extrinsic
- multiple diverse board poses
- robust least-squares over all poses
- leave-one-pose-out repeatability
- uncertainty summary

The minimum evaluation surfaces should be:

- final reprojection RMS
- per-pose reprojection distribution
- leave-one-pose-out holdout reprojection
- transform repeatability under leave-one-pose-out
- parameter uncertainty
- recommendation field
- extraction skip reasons and board support diagnostics

### 6.4 Known current limitation from the old script

The legacy `camera2lidar/reference_based.py` had a useful multi-pose structure, but
it was still missing repo-wide industrial scaffolding:

- no stable CLI entrypoint
- no stable `metrics.yaml`
- no stable `diagnostics/*.yaml`
- no explicit extraction artifact
- no leave-one-pose-out repeatability gate
- no repo-level recommendation field

It also used a heuristic LiDAR board orientation construction from plane normal plus
gravity-aligned in-plane axes. That is acceptable as a **starting baseline**, but it
must remain visible as a limitation until stronger board-side geometry extraction is
validated.

### 6.5 Industrial refactor started in this round

This round starts the actual repo-level refactor:

- new package: `lidar2camera/`
- new stable CLI baseline:
  - `lidar2camera-calibrate`
- new layered implementation:
  - `lidar2camera/models.py`
  - `lidar2camera/reference_pipeline.py`
  - `lidar2camera/metrics.py`
  - `lidar2camera/io.py`
  - `lidar2camera/cli.py`
- old `camera2lidar/reference_based.py` becomes a compatibility entrypoint to the
  new baseline

The new baseline now writes stable artifacts:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `initial_guess/*.yaml`
- `calibrated/*.yaml`
- `diagnostics/reference_dataset.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/optimization.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/manifest.yaml`

### 6.6 First industrial metrics now adopted

The new baseline now evaluates:

- pose count
- initial RMS
- final RMS
- per-pose reprojection RMS
- leave-one-pose-out holdout reprojection
- leave-one-pose-out transform repeatability
- uncertainty summary from leave-one-pose-out spread
- delta to initial transform

Current recommendation meanings:

- `accepted_reference_candidate`
  - current reference-based run passes the current release-oriented checks
- `repeatability_review`
  - leave-one-pose-out stability is not good enough yet
- `reference_quality_review`
  - reprojection quality is still weak even if the solver converged
- `recollect_data`
  - not enough accepted poses

### 6.7 Next industrial tasks after this baseline

After the baseline scaffolding, the next high-value work is:

1. validate the new CLI on real lidar2camera calibration data
2. improve LiDAR-side board geometry extraction beyond the current heuristic
3. add better pose-coverage metrics:
   - left / center / right image coverage
   - near / mid / far depth coverage
   - board tilt / yaw coverage
4. decide whether the current board should remain checkerboard-only or move to a
   more industrial target such as ChArUco / AprilTag-grid for stronger corner
   identity and better in-plane orientation observability
5. keep `learning_based.py` as experimental until it can match the same repeatability
   and uncertainty surfaces
