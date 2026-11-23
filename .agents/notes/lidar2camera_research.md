# LiDAR-to-Camera calibration research and roadmap

## Purpose and decision

This note compares the repository's reference-board LiDAR-to-camera
calibration with established target-based workflows and targetless research
directions. Preserve the checkerboard reference pipeline as the release-oriented
baseline. Keep targetless methods as bounded research/diagnostic candidates
until they win on independent accuracy evidence and representative data.

Current decision: **INCONCLUSIVE for production accuracy beyond the target
reference workflow**. The repository has a substantive target-based pipeline
and a GT-perturbation benchmark framework for targetless methods, but no
lidar2camera run artifacts are present in the current workspace to support a
new local accuracy claim.

## System boundary and current baseline

The production-oriented path is `lidar2camera-calibrate`:

1. Load fixed camera intrinsics and distortion from configuration.
2. Pair image and point-cloud files by identical filename stem, or export
   nearest timestamp pairs from an Apollo record.
3. Detect checkerboard image corners with OpenCV.
4. Fit a dominant LiDAR plane with RANSAC, infer board axes/origin from
   gravity/PCA hypotheses and robust point extents, then resolve board
   orientation candidates with PnP and cross-pose consistency.
5. Optimize one rigid six-DoF LiDAR-camera transform over all accepted board
   poses by robust pixel reprojection least squares.
6. Emit per-pose residuals, leave-one-pose-out (L1O) trials, image/pose
   coverage, geometry diagnostics, uncertainty proxies, overlays, and
   acceptance artifacts.

This is a reasonable baseline architecture: known target geometry provides
cross-modal correspondences, repeated board poses constrain a shared
transform, and extraction is separated from optimization and review. It is
more auditable than a single targetless objective and is the preferred path for
a controlled customer calibration today.

The separate `lidar2camera.learning_based.LearningBasedCalibrator` is not a
production alternative. It converts a monocular depth estimate into a scaled
3D cloud, allows a Sim(3)-like scale adjustment, and applies feature/RANSAC and
colored-ICP registration. Monocular depth scale and shape errors are
confounded with rigid extrinsics, so its current objective does not establish a
physically valid LiDAR-camera transform.

The nuScenes benchmark in `lidar2camera.nuscenes_benchmark` is a different,
experimental track. It compares unchanged perturbed extrinsics (`identity`)
against local edge, direct-visual/NID-style, silhouette, line-feature, and
multi-frame hybrid refinements, with nuScenes ground truth used to measure
recovery. It includes update guards, projection-retention checks, perturbation
levels, per-sample results, and visual overlays. This is useful research
infrastructure, not evidence of success by itself; the `oracle_gt` path is
only a wiring sanity check. No matching output directory was found under
`outputs/lidar2camera` in the inspected workspace, so do not infer local
benchmark performance from method names or code comments.

## Comparison with mature practice

| Practice area | Current repository | Practical comparison and opportunity |
| --- | --- | --- |
| Target-based correspondences | Checkerboard corners in the image; a fitted LiDAR plane and inferred board geometry. | Mature workflows use rigid, dimensionally verified boards/tags, multiple poses, known intrinsics, careful frame/time handling, and visual review. Tier IV/Autoware calibration tools provide target/tag-based and interactive sensor calibration workflows. |
| Board geometry | Dominant-plane support extent and gravity/PCA hypotheses are used to recover board axes and center. | Stronger methods directly identify physical board boundaries/corners or coded fiducials in both modalities. A background plane accidentally selected by RANSAC or a board support contaminated by nearby geometry can bias the inferred 3D corners. |
| Optimization | Fixed intrinsics; one rigid 6-DoF transform; robust least squares on image corner residuals. | This is appropriate when intrinsics, target dimensions, timing and corner identities are trustworthy. Add joint variables only when the data can observe them and the independent validation can separate their errors. |
| Data-quality gates | Corner detection, board bbox/margins, plane residual and extent checks, coverage/depth/tilt span, accepted ratio. | Keep these, but validate gates against manually reviewed positives and negatives. Image coverage and sample count are not substitutes for extrinsic observability or board metrology. |
| Validation | Per-pose training residuals, L1O holdout reprojection, transform clustering/repeatability, image coverage, geometry resolution, overlays. | L1O is useful sensitivity evidence, but poses from one capture session/board are correlated. Add capture-session holdout, independent repeat captures, and a physical or independently surveyed reference when making accuracy claims. |
| Temporal behavior | Record export selects nearest image/point-cloud messages by record-message timestamps; the default maximum gap is 80 ms. PCD export retains XYZ, not per-point acquisition times. | Timestamp provenance must be verified per sensor. A large pairing threshold is acceptable only for a stationary target/rig; otherwise it can pair different board poses or moving scenes. Deskew and temporal calibration are separate from rigid extrinsic refinement. |
| Camera model | Projection uses `cv2.projectPoints` with configured distortion. | This is a standard pinhole/RADTAN-style projection. The camera intrinsic package's fisheye model is not automatically honored by this path; do not feed fisheye coefficients into `projectPoints` as if they were pinhole coefficients. Add a model-aware projection path before claiming fisheye LiDAR-camera support. |
| Targetless branch | Several targetless candidate objectives and a GT-perturbation benchmark are implemented. | Targetless edge/direct methods can be useful for initialization, monitoring or a production candidate after broad evidence. Their non-convex objectives can reward wrong edge/texture matches, visibility changes, or point-count loss unless guarded and evaluated against GT/holdout. |

### Specific implementation risks to measure

1. **Plane support is not necessarily the board.** The extractor chooses a
   dominant plane in a PCD, estimates robust 5th/95th percentile extents, and
   compares them to the board-template extent. It does not directly observe
   each LiDAR board corner. A large wall, partial board, nearby coplanar
   surface, board flex, or incorrect square size can make the inferred
   object-points systematically wrong while still producing a plausible
   reprojection fit.
2. **Fixed intrinsics transfer directly into extrinsic error.** Intrinsic
   calibration must use the same camera, lens/focus, resolution, crop/resize,
   and distortion model as the extrinsic images. The extrinsic path does not
   jointly estimate or propagate uncertainty in intrinsics.
3. **Checkerboard ordering and planar geometry need observability.** The code
   enumerates axis/sign/swap hypotheses and resolves them using PnP and
   cross-pose consistency, which is a useful safeguard. Nevertheless, nearly
   repeated board poses, symmetric patterns, or weak board extent can preserve
   multiple solution families. Do not count the iterative candidate resolver
   itself as independent truth.
4. **The objective is image-corner reprojection, not full-scene fusion
   correctness.** A good corner fit can coexist with poor scene projection if
   intrinsics, timing, camera/LiDAR frame convention, or board geometry is
   wrong. Review both checkerboard overlays and independent scene overlays.
5. **L1O can overstate generalization.** Each trial removes one pose but uses
   other poses from the same acquisition and target. It may not reveal
   session-specific target flex, mounting movement, temperature/focus change,
   or timestamp bias.
6. **Timing is especially important for moving captures.** Exported images and
   scans are paired by nearest record-message time with an 80 ms default
   threshold. The point-cloud conversion drops point timestamps. For a moving
   board or vehicle, this can introduce motion-dependent reprojection bias;
   a static board/rig capture is the safer first calibration dataset.
7. **Current uncertainty is a proxy.** The code derives a covariance-like
   quantity from the optimizer Jacobian and leave-one-out transform spread.
   It is useful for sensitivity review but must not be interpreted as a
   statistically calibrated confidence interval without validating the noise
   model, residual weighting, parameter scaling and model adequacy.

## Research comparison

### Target-based, release-oriented calibration

This is the recommended production baseline for `whl-cal`:

- Calibrate camera intrinsics independently first and freeze their exact model,
  resolution and calibration file.
- Use a rigid, flat, dimensionally measured target that is visible and has
  enough LiDAR returns at the selected range/incidence angles. Coded
  AprilTag/ChArUco-style targets can reduce identity/ordering ambiguity and
  tolerate partial visibility, but only if LiDAR target geometry is also
  reliably extracted.
- Capture multiple non-coplanar/tilted board poses at varied depth and image
  position, with sharp images and a stationary target/rig during each paired
  observation.
- Estimate one shared rigid transform; inspect all accepted and rejected
  observations, not only the aggregate RMS.
- Validate with capture-session holdout, independent repeat calibration, and
  scene overlays not used to solve.

Checkerboard remains acceptable when its pattern is rigid, measured accurately,
clearly detected, and board pose diversity is adequate. Switching to AprilGrid
alone will not fix a LiDAR plane/target-boundary error or weak timing.

### Targetless candidates

Targetless methods are attractive where board setup is costly or unavailable,
but method families make different assumptions:

- **Direct pixel/photometric or mutual-information alignment** compares image
  appearance with projected LiDAR attributes. It can use many pixels and avoid
  explicit board extraction, but is sensitive to intensity-camera appearance
  mismatch, exposure, texture, occlusion, dynamic objects, and local minima.
- **Edge/line/silhouette alignment** can exploit road, building and object
  boundaries. It can be robust to absolute intensity scale, but different
  sensing physics, beam divergence, occlusion, and image/LiDAR edge displacement
  mean coincident edges are not guaranteed. Use diverse scenes and conservative
  projection-retention guards.
- **Learned or multimodal feature matching** may improve initialization in
  weakly textured/structured cases, but adds model/domain/version dependencies
  and still needs a metric/physical acceptance path independent of learned
  confidence.
- **Motion/continuous-time methods** can estimate time offset and extrinsics
  jointly when sensor timing and motion excitation justify it. They require
  accurate per-sensor timestamps, a correct continuous-time motion model, and
  independent holdout; they are not a substitute for fixing wrong timestamps.

Relevant external anchors:

- **Autoware / Tier IV CalibrationTools**: practical target/tag-based and
  interactive workflows; useful engineering comparison for capture, frame
  wiring, and review, not evidence that one implementation wins on this repo's
  data.
- **AIST direct visual LiDAR-camera calibration toolbox** (ICRA 2023,
  arXiv:2302.05094): a general targetless toolbox and useful reference for
  automatic initialization and refinement. Its assumptions, sensor models,
  and dataset results must be verified before treating it as a drop-in.
- **MFCalib** (IROS 2024, arXiv:2409.00992): targetless multi-feature edge
  alignment with a LiDAR beam model; relevant because it explicitly models
  LiDAR edge/beam effects instead of assuming projected boundaries match
  camera edges exactly. It is a research candidate, not a local benchmark
  result.
- **Automatic targetless LiDAR-camera calibration survey** (Artificial
  Intelligence Review, 2022, DOI:10.1007/s10462-022-10317-y): useful taxonomy
  across target-based, feature/edge, information-theoretic, motion-based and
  learning-based methods. Use it to scope candidates, not to select a winner.
- **Zhang (2000)** and OpenCV calibration documentation: camera model and
  reprojection baseline references.

Paper benchmark numbers are not comparable unless the target, camera model,
LiDAR, initialization, perturbation, split and error metric match. The local
nuScenes benchmark's GT perturbation protocol is a stronger selection tool
than an unreferenced real-scene overlay, but its data and outputs must be
present and all failures retained.

## Hypothesis and iteration roadmap

### Round 0 — Freeze and audit target baseline

**Hypothesis:** the largest avoidable errors are more likely to come from
input/board geometry, intrinsics transfer, timing or board-pose diversity than
from the six-DoF least-squares solver.

Lock camera model and image mode, measured board dimensions/flatness, input
hashes, record timestamp source, image/scan pair delta, frames, seed, accepted
and rejected pose IDs, software versions and exact command. Manually review a
stratified set of accepted/rejected samples and checkerboard/scene overlays.

**Decision evidence:** accepted/rejected accounting; board-plane extent and
fit per pose; intrinsics identity; timestamp skew; per-pose pixel residual
vectors; distinct solution families; independent session or surveyed
extrinsic where available.

**Abort condition:** unknown image resize/crop, questionable board scale/flatness,
wrong or unverified frame direction, or uncontrolled motion invalidates an
accuracy claim before algorithm comparison.

### Round 1 — Improve reference target/extraction before optimizer

**Hypothesis:** directly identifying the board boundary/corners or using a
coded target with an explicit LiDAR-visible boundary will reduce pose-dependent
board-coordinate bias versus dominant-plane extent inference.

Keep the same image detections, optimizer, intrinsic calibration and fixed
evaluation; compare the current plane/PCA target geometry with one bounded
alternative on the same poses. Add board metrology and per-pose 3D-to-2D
residual review.

**Acceptance:** lower independent transform error and per-pose tail residual,
stable results after excluding any one pose, no competing board-orientation
family, and improved/unchanged scene overlays on held-out captures. A lower
training RMS alone is not a win.

### Round 2 — Add independent holdout and perturbation recovery

Split by complete capture session/route, not random or single-pose L1O alone.
Perturb a trusted/measured transform across predeclared rotation and
translation offsets; report every solve/failure and parameter recovery.

**Acceptance:** error improves against independent reference over the
declared perturbation envelope and stays stable across new captures, without
loss of observability or geometry. If no independent reference exists, label
the result `INCONCLUSIVE` even if repeatability is high.

### Round 3 — Close camera-model and temporal gaps

Validate camera intrinsics at the exact image mode and implement an explicitly
model-matched projection path before supporting fisheye data. For dynamic
captures, preserve per-point times and camera exposure/reference time; first
test on controlled moving board / ego-motion data and compare static-target
results with and without deskew/time-offset estimation.

**Acceptance:** reduced range/time-dependent residual pattern and improved
independent extrinsic error; no claim based only on fitting a temporal offset
on the same sequence used to tune it.

### Round 4 — Benchmark one targetless candidate

Use the in-repo nuScenes GT-perturbation benchmark with a fixed dataset split,
same initial perturbations, sensor-time gate, evaluation and overlays. Start
with one candidate, preferably direct/edge or MFCalib-style edge modeling,
not a simultaneous rewrite of extraction, optimizer and acceptance gates.
Preserve `identity` and `oracle_gt` sanity paths.

**Acceptance:** beats identity on extrinsic rotation/translation recovery over
multiple scenes and perturbation magnitudes, preserves projected-point
coverage, reports no-results, and transfers to a second sensor/data regime.
Otherwise keep it as diagnostic/initializer-only.

## Immediate first-data validation plan

Before changing the solver or starting a targetless comparison, run one
controlled target-based calibration with a rigid checkerboard whose dimensions
and flatness have been measured. Keep the camera intrinsics, image mode,
board, frame convention, and software revision fixed throughout the run.

1. **Capture:** keep the board and sensor rig stationary during each paired
   image/scan observation. Collect varied board distances, tilts, and image
   locations; avoid a set of nearly identical front-facing poses. Record the
   image and point-cloud timestamp values and their source, along with the
   camera configuration and board measurements. Do not rely on the default
   80 ms pairing threshold as evidence of synchronization.
2. **Extract and review:** preserve a manifest of every input pair and its
   pairing delta, acceptance/rejection decision, and reason. Review overlays
   and board-plane geometry for both accepted and rejected observations before
   solving; investigate background-plane selection, partial board returns,
   and poor image corner detections explicitly.
3. **Hold out:** reserve a complete capture session for evaluation rather than
   relying only on leave-one-pose-out trials from the solve session. Do not
   tune thresholds or select the final result using the held-out session.
4. **Evaluate:** report accepted/rejected counts, per-pose reprojection
   residuals and tails, transform stability, board/image coverage, timestamp
   deltas, and held-out checkerboard plus independent scene overlays. Compare
   to a surveyed/physical reference when available. If no independent
   reference exists, classify physical accuracy as `INCONCLUSIVE`, regardless
   of solver convergence or repeatability.
5. **Decide:** only after this baseline is reviewable should one extraction
   change be tested against it on identical inputs. Keep the existing solver
   and evaluation contract fixed for that comparison; do not change board
   extraction, camera projection model, timing correction, and optimizer
   together.

Do not invent universal numeric sufficiency thresholds for pose count, board
tilt, timestamp delta, or residuals. Set and record thresholds for the actual
camera, LiDAR, target, motion conditions, and application accuracy requirement
before evaluating candidate results.

## Release rule

Classify runs as `ACCEPT`, `REJECT`, `INCONCLUSIVE`, or `NEEDS_MORE_DATA`.
Release requires:

1. verified intrinsics/model/resolution and transform/frame direction;
2. dimensionally trustworthy board geometry or an independently validated
   targetless method;
3. adequate board/scene coverage, orientation and depth diversity;
4. timestamp handling appropriate to sensor motion and acquisition duration;
5. low per-pose and held-out residuals without systematic spatial/time trends;
6. repeatability across independent captures and a single supported solution
   family;
7. independent physical/survey evidence or a justified application-specific
   acceptance threshold; and
8. human review of checkerboard alignment and held-out scene projections.

Do not infer physical accuracy from solver convergence, low training RMS,
L1O alone, a targetless paper result, or a single attractive overlay.

## References

- [LiDAR↔Camera design](../../docs/lidar2camera_design.md)
- [LiDAR↔Camera quick start](../../docs/lidar2camera_quickstart.md)
- [LiDAR↔Camera benchmark guide](../../docs/lidar2camera_nuscenes_benchmark.md)
- [Calibration review guide](../../docs/calibration_review_guide.md)
- [Calibration methodology](../../docs/calibration_methodology.md)
- [Autoware LiDAR-camera calibration](https://docs.autoware.org/1.9.0/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/calibrating-sensors/lidar-camera-calibration/)
- [Tier IV CalibrationTools](https://github.com/tier4/CalibrationTools)
- [AIST direct visual calibration toolbox](https://github.com/koide3/direct_visual_lidar_calibration)
- Koide et al., [General, Single-shot, Target-less, and Automatic
  LiDAR-Camera Extrinsic Calibration Toolbox](https://arxiv.org/abs/2302.05094)
- [MFCalib repository](https://github.com/Es1erda/MFCalib)
- Lin et al., [MFCalib](https://arxiv.org/abs/2409.00992)
- [Automatic targetless LiDAR-camera calibration survey](https://doi.org/10.1007/s10462-022-10317-y)
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
