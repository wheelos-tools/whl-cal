# C++ GRIL Apollo record ingress

This Bzlmod workspace builds `record_ingress`, the C++ Apollo `.record`
adapter for GRIL. It uses `wheelos_core::RecordReader` and the record-provided
protobuf descriptors to decode LiDAR, IMU, and static TF data. The output is a
versioned canonical interchange file; the Python adapter loads it into the
existing canonical model and writes the established NPZ/YAML artifacts. GRIL
preprocessing, synchronization, frontend, solver, and native dataset boundary
are unchanged.

## Build

Build against the local `apollo-lite/pkgs/core` checkout:

```bash
cd tools/gril_record_core_reader
bazel build \
  --override_module=wheelos_core=/path/to/apollo-lite/pkgs/core \
  //:record_ingress
```

## Prepare an Apollo record dataset

From the repository root:

```bash
gril-migrate prepare \
  --input /path/to/capture \
  --input-type record \
  --core-record-ingress tools/gril_record_core_reader/bazel-bin/record_ingress \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16 \
  --output-dir outputs/gril/capture01/dataset
```

Compare the result with a trusted canonical conversion before calibration.
The pinned 0827 input, exact `wheelos_core` revision, hashes, timings, and
review-only result are recorded in
[`docs/lidar2imu_wheelos_core_baseline.md`](../../docs/lidar2imu_wheelos_core_baseline.md).

The C++ adapter is the sole Apollo record ingress registered for GRIL; there is
no Python record-decoding fallback. ROS1 `.bag` remains a separate offline
adapter. Neither path starts or requires a ROS runtime.
