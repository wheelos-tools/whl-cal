# Camera intrinsic calibration research and roadmap

## Purpose and decision

This is the decision contract for improving the repository's monocular camera
intrinsic calibration. Keep the existing OpenCV solver as the baseline until a
candidate improves independent validation on representative data. A low
training reprojection error, successful solve, or a visually plausible
undistorted preview is not sufficient evidence for promotion.

Current judgment: **INCONCLUSIVE for production-level accuracy claims**. The
pipeline already has a useful capture, solve, and evaluation foundation, but
does not yet establish generalization and repeatability strongly enough to
justify a solver replacement or additional camera models.

## Current baseline

- Entry point: `camera-intrinsic-calibrate` / `camera.cli`.
- Pipeline: capture and sample screening → intrinsic solve → diagnostics and
  acceptance.
- Targets: chessboard, AprilGrid, and ChArUco.
- Models: `plumb_bob` (including pinhole/RADTAN aliases) via
  `cv2.calibrateCamera`, and OpenCV fisheye via `cv2.fisheye.calibrate`.
- Capture guidance includes image-grid coverage, stability, and pose novelty.
- Current review includes average and per-view reprojection error, image
  coverage, sample-size consistency, radial monotonicity, and a before/after
  undistortion preview.

The model choice must match the lens projection. Do not expand to omni, Double
Sphere, EUCM, or other models by default. Evaluate them only for a demonstrated
wide-angle/omnidirectional need and against the same frozen data and evaluation.

## Practice comparison and gaps

| Area | Existing behavior | Improvement opportunity |
| --- | --- | --- |
| Lens model | Standard pinhole/RADTAN and OpenCV fisheye are available. | Add controlled model comparison only when lens characteristics warrant it; review validation and edge behavior, not just training fit. |
| Capture | Multiple target types, spatial coverage and basic stability/novelty guidance. | Better encode pose diversity, target rigidity/flatness, sharpness, glare, and capture-mode consistency; area and aspect-ratio novelty are only proxies for geometric observability. |
| Evaluation | Training-set average/per-view reprojection, coverage, monotonicity, and size checks. | Add a held-out-view evaluation with intrinsics fixed, spatial/radial residual summaries, and parameter stability across independent captures. |
| Review | One undistortion comparison image and a coverage heatmap. | Add representative held-out views, residual-vector plots/heatmaps, and explicit center-versus-edge behavior. |
| Release evidence | Existing quality gates and concise customer summary. | Separate in-sample fit from generalization and repeatability; make missing evidence explicit and avoid treating a pixel threshold as universal. |

There is no universal reprojection-error threshold independent of image
resolution, lens field of view, target quality, and downstream error budget.
The existing pixel gates are useful project baselines, not a standalone proof
of parameter accuracy.

## Repository evidence

The AprilGrid round01 review documented a failed result despite sample count and
coverage passing: 18 images produced about 7.98 px average reprojection error,
about 100.7 px per-view RMS P95, and a failed radial-monotonicity check. Several
views had potential board-flex, hand-held instability, or backlight issues.
Pruning weak views did not make the dataset release-ready. This is evidence that
coverage and sample count alone are insufficient; it is not evidence for
replacing the solver.

References:

- [Camera intrinsic experience](../../context/knowledge_base/camera_intrinsic_experience.md)
- [AprilGrid round01 review](../../context/camera_intrinsic_round01_review_2026_05_27.md)
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV fisheye camera model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [Kalibr supported camera models](https://github.com/ethz-asl/kalibr/wiki/supported-models)

External references describe available models and common reporting practice;
they are not local benchmark results. Error thresholds should be selected from
the application's pixel/geometric error budget and verified empirically.

## Hypothesis

The highest-value first improvement is a stronger evidence path, not a new
optimizer: frozen capture conditions plus independent held-out views, spatial
residual diagnostics, and repeat calibrations will identify whether failures
come from capture/target quality, model mismatch, or solver behavior. Only
after that diagnosis should one change model or optimization behavior.

## Experiment sequence

### P0 — Freeze a reproducible baseline dataset

- Record camera/lens identity when known, resolution and sensor mode, focus,
  zoom, exposure, target type/dimensions, source image hashes, configuration,
  software/OpenCV version, and exact command.
- Keep each run to one stable image mode and fixed focus/zoom/exposure state.
- Use a rigid, flat target and collect sharp views spanning image position,
  target scale, and out-of-plane tilt, including useful edge/corner coverage.
- Preserve accepted and rejected samples with explicit reasons; do not combine
  old sessions unless their capture identities and conditions are verified.

**Acceptance:** dataset and run identity are reproducible; target and camera
mode are documented; failures and exclusions remain accounted for.

### P1 — Add holdout and spatial-residual evaluation

- Split by capture session or deliberate view groups before fitting. Do not
  tune thresholds or model choices on the holdout.
- Fit intrinsics on training views. For each holdout view, keep intrinsics
  fixed, estimate only that view's target pose, then report point/view error
  distributions separately from training error.
- Report residual vectors and error by image region/radius, per-view tails, and
  representative holdout undistortion previews. Retain current artifacts and
  add diagnostics without breaking the existing output contract.
- Explicitly note that per-view pose fitting means holdout reprojection is a
  generalization check, not independent physical ground truth.

**Acceptance:** held-out views show no unexplained systematic edge/radial
pattern, thresholds are tied to the intended application, and all views and
failure cases are included in the report.

### P2 — Test repeatability and parameter stability

- Repeat capture/calibration independently under the same fixed camera mode;
  compare focal lengths, principal point, distortion parameters, and projected
  pixel differences over the usable field of view.
- If data volume permits, use view resampling as a sensitivity diagnostic;
  preserve all trials and do not select only the best result.

**Acceptance:** parameter/projection variation is within predeclared
application-specific tolerances and no materially different solution family
appears across runs.

### P3 — Controlled lens-model comparison

- Compare `plumb_bob` and fisheye only where both are plausible for the actual
  lens. For stronger wide-angle lenses, consider another model only after
  documenting its assumptions, runtime/deployment cost, and implementation
  provenance.
- Use identical images, splits, detector outputs where compatible, and
  evaluation code. Compare holdout errors, residual field patterns,
  monotonic/valid projection behavior, repeatability, and downstream projection
  impact.

**Acceptance:** promote a candidate only if it improves held-out and
repeatability evidence across representative captures without regressions in
valid image field, output compatibility, or operational cost. Otherwise retain
the baseline or mark the result inconclusive.

### P4 — Improve capture gates based on observed failures

Add blur, glare, target-flatness, or observability gates only when P0–P2
diagnostics show those failure modes and the new gate can be validated on both
accepted and rejected examples. Preserve explicit rejection reasons and avoid
using coverage or sample count as a proxy for calibration quality.

## Release rule

Classify each result as `ACCEPT`, `REJECT`, `INCONCLUSIVE`, or
`NEEDS_MORE_DATA`. Release requires:

1. sufficient and documented target/camera data with accounted exclusions;
2. acceptable application-specific held-out error without systematic
   image-field residuals;
3. stable parameters/projections across independent captures;
4. physically plausible distortion/projection behavior over the claimed field
   of view; and
5. human review of representative original/undistorted images and residual
   diagnostics.

If independent captures, holdout evidence, or a downstream error budget are
missing, report the result as review-only/inconclusive rather than inferring
accuracy from solver convergence or training RMS.
