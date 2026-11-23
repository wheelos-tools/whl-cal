# Camera intrinsic decisions

- Keep the OpenCV pinhole/RADTAN and fisheye solvers as the baseline; select a
  model that matches the lens rather than adding models speculatively.
- Use consistent capture mode, focus, exposure, and target geometry within a
  run. Sample count and image coverage do not prove parameter accuracy.
- Current evidence does not establish generalization or repeatability across
  independent captures. Keep results inconclusive when holdout or repeat
  capture evidence is missing.

See the [camera quick start](../../docs/camera_quickstart.md) and
[review guide](../../docs/calibration_review_guide.md). Dated capture evidence
and model-comparison plans are in
[research notes](../notes/camera_intrinsic_research.md) and the
[round01 review](../notes/reviews/camera_intrinsic_round01_review_2026_05_27.md).
