# Timing synchronization

- Prefer sensor measurement timestamps and per-point times over record-write
  or host-publish time when the input contract identifies them.
- For GRIL replay across clock domains, derive one fixed clock offset from the
  audited publication/measurement relationship, lock it before synchronization,
  and merge aligned measurement times. Do not re-anchor individual samples to
  message headers.
- Timestamp field meanings are source- and capture-specific. Audit each topic
  and dataset before reusing a timing conclusion.

See the [LiDAR-to-IMU time contract](../../docs/lidar2imu_quickstart.md) and the
[detailed timing investigation](../notes/timing_synchronization_findings.md).
The capture-specific topic measurements are in
[timing notes](../notes/timing_topic_table.md).
