# LiDAR-to-IMU decisions

- Native GRIL is the current algorithm baseline, but remains research-only; a
  completed run is not an accuracy claim.
- Weilan 0827 corrected-IMU is a provisional regression fixture, not ground
  truth. Its physical validation rejected the reported candidate, and the
  native/frozen rotation comparison still misses the documented equivalence
  gate.
- Preserve the frozen algorithm defaults. Adapters must produce equivalent
  canonical datasets; quality evaluation stays post-run.
- Require full-frontend repeatability, observability, independent holdout, and
  physical evidence before promoting an extrinsic.

Use the [LiDAR-to-IMU quick start](../../docs/lidar2imu_quickstart.md) for
commands, [GRIL baseline](../../docs/lidar2imu/07_gril_baseline.md) for the
current input/algorithm contract, and the
[validation skill](../skills/gril-calib-validation/SKILL.md) for executable
checks. Detailed method comparisons are retained in
[research notes](../notes/lidar2imu_research.md).
