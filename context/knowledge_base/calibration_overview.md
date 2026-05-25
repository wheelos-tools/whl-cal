# Calibration overview

This file is the top-level summary of the current calibration status in the repo.

## Guiding principle

- **算法 + 评测指标持续迭代，以测试数据为准**

This means:

- a solver result is not automatically an accepted calibration result
- the acceptance level is only as strong as the tested data behind it
- the knowledge base must separate validated conclusions from open validation points

## Current module status

| Module | Current state | Trusted conclusion | Current role |
| --- | --- | --- | --- |
| `lidar2lidar` | metrics-first, real-bag validated | `scan2scan` is the production baseline; `scan2map` is conditional refinement | active and usable |
| `lidar2imu` | metrics-first, real-bag validated | weak-planar bags should use `--planar-motion-policy auto`; trust `z/roll/pitch` before `x/y/yaw` on one-sided-turn bags | active and usable |
| `camera` | metrics-first intrinsic workflow with diagnostics | capture-mode review + per-view reprojection + coverage + radial-monotonicity should drive intrinsic acceptance | active and usable |
| `lidar2camera` | reference pipeline with screening / staged review artifacts | target-based reference calibration is the release baseline; learning-based remains experimental until repeatability is validated | active and usable |

## lidar2lidar summary

### What is already established

- keep the workflow split into:
  - extraction
  - algorithm
  - evaluation
- `scan2scan` remains the production default
- `scan2map` should be compared against `scan2scan`, not replace it blindly
- vehicle-rig judgment must split:
  - planar: `x/y/yaw`
  - vertical-attitude: `z/pitch/roll`

### Current tested conclusion

- `record_data_0402`:
  - `left -> main`: scan2map can be accepted as a refinement candidate
  - `right -> main`: unconstrained scan2map remains diagnostic because its gain is mainly driven by `z/pitch/roll` drift

## lidar2imu summary

### What is already established

- keep the workflow split into:
  - record conversion / sample extraction
  - staged solver
  - stable evaluation
- pose-derived gravity is currently the default
- one-sided-turn bags should not be used to promote full `x/y/yaw`

### Current tested conclusion

- `record_data_0402`:
  - trustworthy: `z/roll/pitch`
  - weak: `x/y/yaw`
- Synology front-LiDAR bag:
  - diagnostic-only
  - usable to test data-layer and prior-preserving behavior

### New tested controls

- weak-planar solver policy:
  - `free`
  - `freeze_xyyaw`
  - `auto`
- window + gate motion selection:
  - window the timeline
  - gate weak windows
  - select rotation-qualified local candidates
  - reject low-fitness registrations

## Next module: lidar2camera

The active repository-level target is now to keep camera-related calibration in
the same structure already used by `lidar2lidar` and `lidar2imu`, and continue
iterating on physical target observability rather than script structure.

### Current codebase state

- `camera/intrinsic.py`
  - chessboard-based intrinsic calibration with stable diagnostics / acceptance artifacts
- `lidar2camera/reference_pipeline.py`
  - checkerboard / reference-based LiDAR-camera calibration
- `lidar2camera/learning_based.py`
  - targetless LiDAR-camera calibration experiment

### Current engineering goal

Keep `camera` and `lidar2camera` inside the same repo-wide pattern:

1. **data layer**
   - image / LiDAR pair ingestion
   - pose / board / scene window selection
   - window + gate for invalid or abnormal samples
2. **algorithm layer**
   - reference-based path
   - targetless path
   - optional learned priors
3. **evaluation layer**
    - stable reprojection / alignment metrics
    - leave-one-pose-out repeatability
    - uncertainty summary
    - recommendation field

Current baseline decision:

1. **reference-based / target-based**
   - release baseline
2. **targetless / learning-based**
   - experimental comparison / diagnostic branch

Current tested workflow decision:

1. **camera intrinsics**
   - release only when capture mode, coverage, per-view error, and radial monotonicity all look sane
2. **lidar2camera**
   - release only when data screening, geometry resolution, reprojection, holdout, repeatability, and visual diagnostics all pass

## Read next

1. `validated_conclusions.md`
2. `engineering_index.md`
3. `verification_points.md`
4. `../lidar2imu_context.md`
5. `../scan2map_context.md`
