# ROS-free GRIL native — research source build only

This directory contains the ROS-free, normal-CMake GRIL-Calib implementation.
It preserves the upstream calibration residuals, filtering, synchronization,
temporal initialization, and Ceres solve order while removing ROS/catkin, ROS
message types, Python, and matplotlib dependencies.

## Packaging status

This is a **developer/research source-build surface**, not a distributable
runtime package or a production calibration release. CMake deliberately has no
installable artifacts or CPack configuration. Its sole `install()` rule fails
so that `cmake --install` cannot accidentally turn the current experimental
tree into an installer.

The repository's Python wheel only supplies the `gril-migrate` migration and
evidence CLI; it neither builds nor bundles this native executable. Build the
native targets directly with CMake only for controlled reproduction and A/B
work.

The complete native executable requires a C++14 toolchain, CMake 3.16+, Eigen
**3.3.7 exactly**, PCL with compatible OpenMP runtime support, Ceres 2.0+,
and pthreads. It has no ROS, catkin, rosbag, TF, or generated ROS-message
runtime dependency. The source CMake build makes PCL/Eigen/Threads explicit;
Ceres conditionally enables the full executable and batch target. Inspect the
linked PCL/OpenMP runtime on the target host rather than assuming the
development machine's ABI is portable.

Distribution remains blocked until all of the following are recorded and
approved:

1. the frozen full native-vs-reference A/B gate passes (the recorded rotation
   discrepancy is `0.406964 deg`, above the `0.2 deg` gate);
2. independent physical validation accepts the candidate (the frozen reference
   is rejected and the native physical verdict is unknown);
3. a formal license/provenance review resolves the upstream `package.xml` BSD
   declaration versus the bundled LI-Init GPLv2 provenance, and records
   redistribution obligations for GRIL, LI-Init, PCL, Eigen, Ceres, and their
   transitive dependencies.

**Scope:** this contains the batch calibration core, isolated trace runners,
and the integrated VELO/Velodyne canonical event path through FIFO
synchronization, IMU-free constant-velocity propagation, Patchwork++, Fusion
AHRS, LiDAR-only odometry/EKF, data sufficiency, and live `LI_Calibration`.
Feature extraction and other LiDAR types have **not** been ported.

The source was adapted from
[`Taeyoung96/GRIL-Calib`](https://github.com/Taeyoung96/GRIL-Calib.git) at
pinned revision `c09b01a05ec83bc0a361941acf897109aaecf0a6`.
The native adaptation was made on 2026-08-28. It retains the validated
front-of-deque synchronization fixes from
`.agents/skills/gril-calib-validation/patches/gril-validation.patch`.

## Provenance and intentional deviations

| Area | Intentional native difference |
| --- | --- |
| Build | Catkin is replaced by a standalone CMake library, trace runner, tests, and smoke executable. |
| Dependencies | ROS, generated messages, Python, and matplotlib are absent. Exact frontend reproduction uses Eigen 3.3.7 and PCL 1.10-compatible accumulation; the calibration solve additionally uses Ceres. |
| Common types | Upstream aliases, gravity constant, skew matrix, and Euler conversion are defined locally with the same formulas. |
| IMU ingestion | `sensor_msgs::Imu` is replaced by angular velocity, linear acceleration, timestamp, and normalization scalar arguments; normalization math is unchanged. |
| Calibration weights | The three upstream globals are owned here and default to the upstream ROS defaults of `1.0`; callers may assign them before calibration. |
| Logs | The same diagnostic filenames are opened under the CMake build tree's `Log/` directory instead of the ROS package source tree. |
| Plot/timing/UI | `plot_result()` is a no-op and matplotlib is not linked; the scope timer and color-only console decoration are omitted. |
| Smoke API | Four read-only count accessors support ingestion verification without entering the insufficient-data calibration path. |
| Batch replay API | `set_batch_inputs()` restores trace records directly into the exact pre-`LI_Calibration` deques, avoiding a second normalization or reconstruction step. |
| Velodyne API | ROS `PointCloud2` conversion is replaced by typed `{x,y,z,intensity,time_s,ring}` input. Point order, millisecond curvature, filters, fallback timing, sort, and cut loop match the patched upstream VELO cut path. |
| Velodyne boundary | Empty scans, invalid configuration, rings outside the upstream 128-entry timing arrays, and non-finite timestamps are rejected before the algorithm. Non-finite XYZ values are filtered at their original indices; this extends upstream's NaN check to infinities without changing finite-input behavior. |
| Validated fix | Every audited front-trimming site pops all paired ground-constraint deques exactly with each LiDAR state. |
| LiDAR odometry | The pinned ikd-tree, PCL 1.10 voxel order, local-map movement, five-neighbor plane fit, LiDAR-only Jacobian, ground rows, iterated EKF, convergence/rematch schedule, covariance update, and map insertion order are retained. |
| Full input | `GRIL_NATIVE_DATASET 1` is a streamed, versioned binary event boundary generated from the canonical dataset without ROS. |
| Hard timing | The one-shot hard offset, compensated IMU enqueue timestamp, rollback queue clearing, exact FIFO end comparisons, and full-scan ground state paired to each cut retain the audited callback semantics. |
| Forward gaps | Golden mode adds no reset. The separate reset mode requires an explicit threshold and recreates all frontend/calibration state before derivatives resume. |
| Ground plane fit | Patchwork++ and the integrated plane fit retain PCL 1.10's original float accumulation order instead of inheriting the shifted PCL 1.15 covariance accumulator. |

All other batch sequence lengths, loop bounds, deque operations, filtering,
residual construction, parameter initialization, solver options, and solve
order retain upstream behavior. In particular, this core does not add input
guards, exceptions, shortened loop bounds, or alternative deque trimming.

## Build

After installing the prerequisite development packages through the host's
normal package manager, use a build directory inside this repository:

```bash
cmake -S third_party/gril_native -B third_party/gril_native/build
cmake --build third_party/gril_native/build
ctest --test-dir third_party/gril_native/build --output-on-failure
```

Eigen 3.3.7 exactly and PCL are required for the frontend targets. When the
system package is not 3.3.7, pass reviewed headers explicitly:

```bash
cmake -S third_party/gril_native -B third_party/gril_native/build \
  -DGRIL_PINNED_EIGEN_INCLUDE_DIR=/path/to/eigen-3.3.7
```

Host-specific SIMD flags exported by PCL are not inherited by the native
targets because the frozen Noetic build used baseline x86 floating-point
semantics. Ceres Solver 2.0+ enables the separate batch-calibration targets.
This command is deliberately not followed by `cmake --install`: the explicit
distribution gate rejects installation until the prerequisites above are met.

The smoke executable checks construction and primitive data ingestion only.
It deliberately does not run calibration on insufficient synthetic data.

The complete executable is:

```bash
third_party/gril_native/build/gril_native_full_frontend \
  --input native_dataset_v1.bin \
  --config gril_native_full.conf \
  --output GRIL_Calib_result.txt \
  --trace GRIL_full_frontend_trace_v1.txt \
  --batch-trace GRIL_batch_trace_v1.txt \
  --batch-executable third_party/gril_native/build/gril_native_batch \
  --batch-config gril_native.conf \
  --gap-policy golden
```

The full executable hands its live queues to the clean batch executable as soon
as `data_sufficiency_assess` passes. This preserves the exact versioned batch
input while isolating Ceres from the active ikd-tree rebuild worker.

`GRIL_NATIVE_DATASET 1` begins with its ASCII magic line and a little-endian
`uint64` event count. Events are sorted by `(timestamp_ns, type, index)`, with
IMU before LiDAR at an equal timestamp. An IMU event is a `uint8` tag (`0`),
an `int64` timestamp, and six little-endian `float64` values (angular velocity,
then linear acceleration). A LiDAR event is a `uint8` tag (`1`), a `uint32`
one-based scan number, an `int64` timestamp, a `uint64` point count, then
per-point little-endian `{float32 x, y, z, intensity, time_s; uint16 ring}`.
All canonical IMU samples are retained even when `--scan-count` limits LiDAR
scans: fallback point timing and a one-shot hard clock offset can require IMU
records after the final selected raw LiDAR timestamp.

## Native VELO/Velodyne preprocessing

Include `Gril_Calib/VelodynePreprocess.h` and link `GRIL::velodyne`.

```cpp
VelodynePreprocessConfig config;
config.blind = 1.0;
config.point_filter_num = 2;
config.n_scans = 16;
config.required_frame_num = 4;
config.scan_count = 20;

VelodynePreprocessResult result =
    preprocess_velodyne_scan(points, scan_timestamp_s, config);
```

Input point time is in seconds. Output `PointType::curvature` and
`cut_timestamps_ms` are in milliseconds, matching upstream. `surface` is the
sorted `pl_surf` after the cut loop's in-place per-cut time rebasing;
`cut_clouds` and `cut_timestamps_ms` correspond to upstream `pcl_out` and
`time_lidar`.

Behavior retained from patched upstream:

- supplied time is selected when the **last** point time is positive, so a
  normal first-point offset of exactly zero is accepted;
- otherwise per-ring time is reconstructed from azimuth at `3.61 deg/ms`, and
  the first accepted point of each ring is omitted;
- blind filtering uses squared range against `blind * blind`;
- stride uses the original input index before ring acceptance;
- rings must be below `n_scans` to enter the surface cloud;
- the surface cloud is sorted by curvature before cutting;
- cut iteration begins at sorted point index 1;
- scans with `scan_count < 20` force one cut; scan 20 and later use
  `required_frame_num`.

Feature extraction is intentionally not part of this layer.

The `gril_native_preprocess_trace` executable is migration-only evidence. It
replays selected canonical scans and writes every sorted surface and cut-cloud
point. Scans 1, 20, and 21 from the full 0827 dataset are byte-identical to the
frozen ROS reference trace.

## Native synchronization and CV propagation

Link `GRIL::frontend` for ROS-free FIFO synchronization and GRIL's original
IMU-free constant-velocity propagation. The implementation preserves the
upstream timestamp inequalities, first-frame `dt=0.1`, `time_last_scan`
initialization, covariance block operations, and reverse point-undistortion
loop. Migration trace executables verify canonical events through synchronized
packages and propagated states; they are evidence tools, not production gates.

## Native LiDAR-only odometry

Link `GRIL::odometry` for the ROS-free ikd-tree/PCL odometry and iterated EKF
layer. `LidarOdometryCore::process()` consumes one CV-propagated undistorted
cloud plus the current LiDAR ground rotation/normal and updates the same
24-state layout in place. The default native configuration explicitly repeats
the frozen Velodyne launch-only values `max_iteration=5` and
`cube_side_length=1000`; the reviewed Vanjee YAML records both values.

The vendored `include/ikd-Tree/ikd_Tree.{h,cpp}` files are byte-identical to
revision `c09b01a05ec83bc0a361941acf897109aaecf0a6`. The implementation retains
the original order: local-map segmentation, surface voxel filtering, initial
tree build, correspondence/plane evaluation, point and ground measurement
rows, iterated EKF update, rematch decision, covariance update, and incremental
map insertion.

The upstream ikd-tree still starts its background rebuild worker with
`pthread_create`. Tree contents observed by later packages can therefore
depend on foreground/worker scheduling; the ROS-free port does not disable or
serialize that worker because doing so would change the pinned algorithm.

`gril_native_odometry_trace` defines the deterministic
`GRIL_ODOMETRY_INPUT 1` -> `GRIL_ODOMETRY_TRACE 1` boundary for a later frozen
frontend A/B. It records every propagated/updated state, downsampled and
effective cloud, final correspondence set, thresholds, iteration counters, and
map counters. This layer does not add quality gates or change solver decisions.

The input header contains one `config` record in this order:
`max_iteration`, `cube_side_length`, `filter_size_surf`, `filter_size_map`,
`det_range`, and `ground_cov`. Each frame then contains its index/timestamp,
the propagated 24-state record, ground quaternion/normal, and undistorted
cloud. State fields use the existing frontend trace order followed by the
row-major 24x24 covariance. Cloud records contain
`x y z intensity curvature normal_x normal_y normal_z`.

The output repeats the complete boundary inputs before writing update metrics,
the updated state, downsampled body/world clouds, effective body/normal clouds,
and the final selected flag, residual, and nearest points for every
downsampled point. Numbers use 17 significant digits.

## Batch trace contract

`GRIL_BATCH_TRACE 1` is a strict, versioned, whitespace-delimited text format.
The reference patch
`gril/reference_patches/gril-batch-trace-v1.patch` applies after
`gril-validation.patch` to upstream revision
`c09b01a05ec83bc0a361941acf897109aaecf0a6`. It writes
`result/GRIL_batch_trace_v1.txt` immediately before `LI_Calibration` and does
not modify any calibration state or computation.

Header and section order are fixed:

```text
GRIL_BATCH_TRACE 1
orig_odom_freq INTEGER
cut_frame_num INTEGER
timediff_imu_wrt_lidar FLOAT
move_start_time FLOAT
imu_states COUNT
imu STATE_FIELDS
...
lidar_states COUNT
lidar STATE_FIELDS
...
ground_constraints COUNT
ground GROUND_FIELDS
...
END
```

`STATE_FIELDS` are:

1. timestamp
2. row-major `rot_end` (9 values)
3. `pos_end` (3)
4. `ang_vel` (3)
5. `linear_vel` (3)
6. `ang_acc` (3)
7. `linear_acc` (3)

The `imu` records are the complete normalized `IMU_state_group_ALL` states
used by GRIL. The `lidar` records are the complete raw
`Lidar_state_group` states.

`GROUND_FIELDS` are LiDAR-ground quaternion `w x y z`, IMU-ground quaternion
`w x y z`, LiDAR-frame normal `x y z`, and LiDAR-ground distance. Ground
records are index-paired one-to-one with LiDAR records. Numbers are written
with 17 significant digits. The parser rejects unknown versions, non-finite
numbers, missing/extra fields, trailing content, and mismatched paired counts.

### Reference patch usage

```bash
git clone https://github.com/Taeyoung96/GRIL-Calib.git
cd GRIL-Calib
git checkout c09b01a05ec83bc0a361941acf897109aaecf0a6
git apply /path/to/whl-cal/.agents/skills/gril-calib-validation/patches/gril-validation.patch
git apply /path/to/whl-cal/gril/reference_patches/gril-batch-trace-v1.patch
```

### Native replay usage

Every config key is mandatory so a replay cannot silently inherit a default.
`config/gril_native.conf.example` lists the complete native configuration.

```bash
third_party/gril_native/build/gril_native_batch \
  --trace GRIL_batch_trace_v1.txt \
  --config third_party/gril_native/config/gril_native.conf.example \
  --output GRIL_Calib_result.txt
```

The runner restores all trace deques exactly, assigns every calibration
parameter and residual weight from the config, calls the unchanged
`LI_Calibration` sequence, and writes the upstream
`GRIL_Calib_result.txt` text shape. The file's time field follows upstream
behavior and uses `get_time_result()`.

The runner rejects traces that cannot safely reach upstream's fixed trimming
and two filter passes. This validation is isolated at the executable boundary;
the calibration core retains upstream behavior.

## License and provenance

GRIL-Calib is by TaeYoung Kim and contributors and is heavily adapted from
LI-Init by Fangcheng Zhu and contributors. This derivative keeps the upstream
GPL version 2 text in `LICENSE` and is marked GPL-2.0-only. See the change
notices at the top of the adapted source files.
