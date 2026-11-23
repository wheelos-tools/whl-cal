# C++ GRIL Apollo record ingress

This Bzlmod workspace builds `record_ingress`, the C++ Apollo `.record`
adapter for GRIL. It uses `wheelos_core::RecordReader` and the record-provided
protobuf descriptors to decode protobuf LiDAR, IMU, and static TF data, and
includes bounds-checked FlatBuffer decoding for
`::apollo::drivers::flat::PointCloud` (`modules/common_msgs/sensor_msgs/pointcloud.fbs`
wire layout). The output is a
versioned canonical interchange file; the Python adapter loads it into the
existing canonical model and writes the established NPZ/YAML artifacts. GRIL
preprocessing and optimization are not part of this adapter.

Interchange v2 appends three signed-int64 nanosecond arrays after IMU values
and before transforms: LiDAR `measurement_time` (scan end), LiDAR publication
header, IMU independent publication header. Zero means unavailable; an IMU
header used as measurement fallback is not independent publication evidence.
Canonical scan starts, point offsets and IMU measurement signals are unchanged.
The Python reader also accepts v1 (without these arrays); canonical schema 1
stores them as optional NPZ fields. See the
[clock contract](../../docs/lidar2imu_quickstart.md#measurement-clocks-and-publication-time)
before replaying different sensor epochs.

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
lidar2imu prepare \
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
