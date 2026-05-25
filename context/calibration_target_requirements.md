---
audience: dev
stability: stable
last_reviewed: 2026-05-25
---

# Calibration target / board requirements

This note answers a practical question for the current repository:

- **Which calibration workflows really require a physical board or target?**
- **What are the hard requirements on that board?**
- **Which recommendations are upgrades rather than current must-haves?**

The conclusion is intentionally split by module, because the answer is **not**
the same across the repo.

## 1. Repo-level summary

| Workflow | Physical board required by current repo workflow? | Current baseline | Main requirement |
| --- | --- | --- | --- |
| `camera` intrinsic | **Yes** | planar checkerboard | flat, rigid, measured board + broad image coverage |
| `lidar2camera` | **Yes** | checkerboard visible in camera + planar board visible in LiDAR | known image pattern + LiDAR-observable plane with clean support |
| `lidar2lidar` | **No** | scene-based scan registration | overlap, static structure, scene sufficiency |
| `lidar2imu` | **No** | motion / observability-based calibration | excitation quality, turn balance, trustworthy motion samples |

That split is also the current codebase reality:

- `camera/intrinsic.py` is explicitly checkerboard-based.
- `lidar2camera/reference_pipeline.py` is explicitly checkerboard + LiDAR plane support based.
- `lidar2lidar` and `lidar2imu` are not built around a board workflow; they rely
  on scene geometry and motion observability instead.

## 2. External references behind the judgment

This note is aligned with the following external baselines:

1. **OpenCV camera calibration guidance**
   - chessboard / ChArUco board based intrinsic calibration
   - at least about 10 good pattern views
   - broad pose and image-region coverage
   - source: <https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html>
2. **OpenCV ArUco / ChArUco guidance**
   - ChArUco is better than marker-only calibration
   - partial visibility and occlusion handling are stronger than plain checkerboards
   - source: <https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html>
3. **Kalibr target guidance**
   - prefer rigid, flat targets
   - re-measure printed dimensions
   - Aprilgrid is preferred when target ID / partial visibility matters
   - keep a white border around the target
   - source: <https://github.com/ethz-asl/kalibr/wiki/calibration-targets>
4. **Autoware-style camera-lidar target calibration practice**
   - board / correspondence quality matters more than raw solver convergence
   - manual correspondence methods require well-spread points and enough geometry
   - representative open-source reference: <https://github.com/vmyakovlev/CPFL-Autoware/blob/470b04b95eee435d2bb529d978d4c91bf57f34b0/ros/src/sensing/fusion/packages/autoware_camera_lidar_calibrator/README.md>
5. **JointCalib / related target-based lidar-camera calibration literature**
   - LiDAR-camera calibration is stronger when the target provides more explicit
     geometry / identity than a plain unstructured plane
   - checkerboard + circle / coded features improve observability
   - representative references:
     - <https://arxiv.org/abs/2202.13708>
     - <https://github.com/OpenCalib/JointCalib>

These references are used here as **design justification**, not as a claim that
the repo has already implemented every possible upgrade.

## 3. Camera intrinsic calibration (`camera`)

### Current repo baseline

The current intrinsic tool is a **planar checkerboard** workflow:

- `pattern_size`
- `square_size`
- image-space checkerboard corner extraction
- coverage and per-view reprojection diagnostics

Relevant repo surfaces:

- `camera/intrinsic.py`
- `docs/camera_quickstart.md`

### Hard requirements

For the current repo workflow, the checkerboard must satisfy all of these:

1. **Planar and rigid**
   - the board cannot bend, curl, or wave
   - paper-only prints without a rigid backing are risky
2. **Known and trustworthy square size**
   - printed dimensions must match the configured `square_size`
   - if the print scaling is wrong, the solution is structurally wrong
3. **High contrast and clean corners**
   - the board must be easy to detect reliably
   - blurred edges, reflections, and low-light corners create long-tail bad views
4. **Broad image coverage**
   - not just center-only captures
   - the board should visit multiple image regions and tilts
5. **No acquisition crop**
   - the live 3x3 guide must be fully visible before collection
   - forcing a non-native sensor mode can crop before the board is even seen

### Why these are hard requirements

OpenCV’s standard camera calibration guidance is fundamentally based on:

- correct object geometry
- reliable corner detection
- multiple views from different poses

The current repo now enforces that same idea through:

- `sample_records.csv`
- `per_view_reprojection.csv`
- `image_coverage_heatmap.png`
- `radial_monotonicity` review

### Failure modes caused by a bad intrinsic board

1. warped board -> biased focal length / distortion fit
2. wrong square size -> physically wrong scale interpretation
3. clipped board during capture -> weak image-edge observability
4. center-only captures -> unstable distortion near image boundaries
5. glossy reflections / low contrast -> false or noisy corner localization

### Upgrade recommendations

These are good improvements, but **not current hard requirements** for this repo:

1. **ChArUco**
   - better partial-view robustness
   - stronger corner identity
2. **Aprilgrid**
   - similar advantage when coded target identity matters
3. **dimension re-measurement and board QA**
   - especially for mass-production lines using printed boards

## 4. LiDAR-to-camera calibration (`lidar2camera`)

### Current repo baseline

The current release baseline is still a **target-based checkerboard reference**
workflow:

- the camera sees checkerboard corners
- the LiDAR sees a planar board patch
- the pipeline resolves board orientation from plane support using
  gravity/PCA/IPPE candidate hypotheses

Relevant repo surfaces:

- `lidar2camera/reference_pipeline.py`
- `docs/lidar2camera_quickstart.md`
- `docs/lidar2camera_design.md`
- `context/lidar2camera_context.md`

### Hard requirements

For the current repo workflow, the target must satisfy all of these:

1. **The camera-visible side must be a known checkerboard**
   - correct `pattern_size`
   - correct `square_size`
   - corners must be fully detectable
2. **The LiDAR-visible side must behave like one clean plane**
   - enough inlier points
   - limited plane residual
   - not merged with the background wall / ground / clutter
3. **The board must be rigid and flat**
   - otherwise image geometry and LiDAR plane geometry disagree
4. **The board must be large enough for both sensors**
   - large enough in image space to avoid `image_board_too_small`
   - large enough in LiDAR space to avoid weak plane support
5. **The board must be isolated enough from surrounding surfaces**
   - otherwise the plane fit may lock onto a wall patch instead of the board

### Why these are hard requirements

The current code is not using a coded LiDAR target or explicit LiDAR corners.
It still depends on:

1. image-side checkerboard corner geometry
2. LiDAR-side plane support geometry
3. multi-pose consistency to resolve board orientation ambiguity

So the physical target must support **both** modalities. That is why the repo now
checks:

- image edge margin
- image bbox area ratio
- plane residual RMSE
- board extent ratio
- geometry-resolution consistency

### Failure modes caused by a bad lidar2camera board

1. **board too small**
   - weak image and/or LiDAR observability
2. **board merged with wall**
   - plane segmentation fits a facade, not the board
3. **board not rigid**
   - corners and plane disagree
4. **insufficient LiDAR returns**
   - orientation resolution becomes unstable
5. **uncoded symmetric board ambiguity**
   - still manageable with current candidate resolution, but weaker than coded targets

### Upgrade recommendations

These are strong future improvements, but **not yet hard requirements** in the
current repo:

1. **ChArUco / AprilTag-grid / Aprilgrid**
   - adds target identity
   - helps with partial visibility and ambiguity
2. **reflective / retroreflective LiDAR target treatment**
   - improves LiDAR-side observability
3. **JointCalib-style richer target geometry**
   - stronger than a plain board plane

Current practical judgment:

- **plain planar checkerboard board is still acceptable for the current repo**
- **coded / reflective targets are the best next upgrade path**

## 5. LiDAR-to-LiDAR calibration (`lidar2lidar`)

### Current repo baseline

The current repo baseline is **not board-based**.

The production path is:

- raw record extraction
- pairwise / workflow-planned registration
- scene sufficiency checks
- visual evaluation on merged point clouds

Relevant repo surfaces:

- `docs/lidar2lidar.md`
- `docs/lidar2lidar_design.md`
- `context/lidar2lidar_advanced_strategy.md`

### Practical requirement

For `lidar2lidar`, the “target” is really:

1. **enough overlap**
2. **static structure**
3. **good spatial geometry**
4. **repeatable visual evidence**

This means:

- walls, corners, poles, facades, and other stable structure matter
- a dedicated calibration board is **not** a requirement of the current workflow

### Why a board is not the current requirement

The current repo is designed around:

- scan-to-scan / scan-to-map registration
- loop closure
- visual ghosting / thickness review

So the workflow’s main bottlenecks are:

- overlap
- dynamic contamination
- weak geometry
- topology and loop consistency

not “what board should be printed”.

### When a board could still be useful

A board can still help in **bench diagnostics** or unit-style experiments, but it
is not the current production baseline and should not be confused with the
repo’s main workflow.

## 6. LiDAR-to-IMU calibration (`lidar2imu`)

### Current repo baseline

`lidar2imu` is also **not board-based** in this repository.

The effective “target” is **trajectory excitation and observability quality**:

- enough ground constraints
- enough rotation / motion diversity
- enough left/right turning balance
- trustworthy extraction windows

Relevant repo surfaces:

- `docs/lidar2imu.md`
- `docs/lidar2imu_design.md`
- `context/lidar2imu_context.md`

### Practical requirement

For `lidar2imu`, the hard requirements are:

1. **good motion excitation**
   - especially if you want trustworthy `x/y/yaw`
2. **balanced turning / yaw observability**
   - one-sided-turn bags remain weak
3. **usable ground / motion sample extraction**
   - low-fitness motion pairs should be rejected

### Why a board is not the current requirement

The solver is built on:

- standardized motion samples
- ground constraints
- priors
- holdout / basin / extraction consistency

So a physical checkerboard or board target is simply not part of the current
pipeline design.

### What “target quality” means here

For `lidar2imu`, “target quality” should be read as:

- **bag quality**
- **trajectory quality**
- **observability quality**

not board quality.

## 7. Final judgment

If someone asks “what calibration board do we need in this repo?”, the precise
answer is:

1. **camera intrinsic**
   - yes, a good planar checkerboard is required today
2. **lidar2camera**
   - yes, a rigid planar checkerboard-style board that is also cleanly visible to
     LiDAR as a plane is required today
3. **lidar2lidar**
   - no dedicated board is required by the current production workflow
4. **lidar2imu**
   - no dedicated board is required by the current production workflow

And if the question is “where should we upgrade first?”:

1. first upgrade **lidar2camera** target observability
   - coded / reflective target improvements have the highest value
2. then optionally upgrade **camera intrinsic** from plain checkerboard toward
   ChArUco / Aprilgrid if partial-view robustness becomes important
3. do **not** force a board into `lidar2lidar` or `lidar2imu` just for symmetry;
   their main production constraints are scene quality and motion quality, not
   target print design
