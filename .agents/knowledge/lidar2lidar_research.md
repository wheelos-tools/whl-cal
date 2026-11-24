# LiDAR-to-LiDAR decisions

- Keep direct `scan2scan` as the production baseline when overlap and scene
  geometry are sufficient.
- Treat `scan2map` as conditional refinement/validation, not a blanket
  replacement.
- Use topology-aware, healthy edges and multi-window evidence; a loop residual,
  registration fitness, or one merged-cloud image is not sufficient by itself.
- Review scene sufficiency, repeatability, transform consistency, and visual
  geometry before acceptance.

## Shared geometry, association, and information

- Inspect the geometry actually matched by both sensors, not all geometry
  detected in either cloud. Opposite normals of parallel walls do not add a
  new constraint direction. Extra windows can complement one another even
  when each sees only one wall; evaluate the complete window set.
- Distinguish local shape detection, cross-sensor physical identity, and
  independent information. A PCA line is not a confirmed pole; a reflective
  stripe is not automatically a set of uniquely identified 3D point landmarks.
  Distinct target indices are not necessarily independent objects.
- Residuals must be reported with retained support and rejection counts.
  Lower residual after losing correspondences is not an improvement.
- Fixed-correspondence, fixed-normal Gauss-Newton information is conditional
  on those assumptions, not automatically a calibrated extrinsic covariance.
  Check continuous nuisance coupling, shared plane/scan errors, and empirical
  re-extraction stability before interpreting its inverse as uncertainty.
- A partial measurement's unobservable pose coordinates must not determine
  compatibility with another measurement. Use nullspace-representative
  invariance as a theoretical control. Component-wise robust losses in a
  whitening basis also need an explicit noise/outlier interpretation.
- Sensor-internal intensity normalization is not cross-sensor physical
  reflectivity calibration. Test background/viewpoint changes separately
  from geometric identity and nearest-neighbor distance.
- Qualify the entire static stack, not only its center state. A frontend
  initialization gap is not proof of stationarity. Low gyro cannot prove zero
  translation, and LiDAR-only odometry is not independent motion truth.
  Distinguish inter-scan compensation from per-point deskew.

## Data expansion before estimator proliferation

Do not stop at diagnosing weak geometry or wait for a perfect model before
testing additional qualified data. Freeze the estimator and compare:

- A: the existing windows;
- B: additional reliable windows with comparatively redundant geometry;
- C: the same number of additional reliable windows that complement the
  complete set's weak directions.

Match motion, stack duration, timing, preprocessing, and initialization where
possible; disclose unmatched conditions and missing controls. Do not equate
more frames with independent constraints or quietly relax quality/weighting
rules to make C win. Keep invariant controls alongside the data experiment.

## Active Zhongce research baseline

The current three-window **research group A**, its replay command, source
snapshot, measured limitations, and pending B/C experiment are recorded in
[the 2026-11-24 baseline note](../notes/lidar2lidar/2026-11-24_zhongce_baseline.md).
It is review-only and does not replace the production `scan2scan` / GICP
baseline. A joint excitation verdict is not L2L release approval.

See the [quick start](../../docs/lidar2lidar_quickstart.md) and
[design](../../docs/lidar2lidar_design.md) for supported behavior. Historical
playbooks and detailed candidate comparisons are in
[research notes](../notes/lidar2lidar_research.md).
