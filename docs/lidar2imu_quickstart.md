---
audience: customer
stability: review-only
---

# LiDAR-to-IMU calibration

Native GRIL consumes a validated canonical dataset. Apollo `.record` ingress
uses the C++ `wheelos_core` adapter; offline ROS1 `.bag` uses the `rosbags`
adapter. Neither requires a ROS runtime. Real-time ROS topic subscription is
not implemented. The native frontend and solver are shared across input
formats.

The `gril.adapters` registry is the data-ingress extension point. A new format
must produce the existing `CanonicalDataset` contract and pass an input
equivalence audit before it is used for calibration; do not modify GRIL
preprocessing, synchronization, frontend, or solver to accommodate a container
format.

This is a developer/research source build, not a published customer installer
or registry image. The 0827 result remains review-only; successful execution
does not prove calibration accuracy.

## 1. Install the data adapters

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[lidar2imu]"
```

The extra provides Python-side offline bag and validation tools. It does not
install ROS or build the native GRIL executable or C++ record ingress.

## 2. Build the native GRIL runtime

The complete frontend requires C++14, CMake 3.16+, Eigen 3.3.7, PCL with
compatible OpenMP support, Ceres 2.0+, and pthreads. Install these from a
reviewed toolchain, then run:

```bash
cmake -S third_party/gril_native -B third_party/gril_native/build
cmake --build third_party/gril_native/build
ctest --test-dir third_party/gril_native/build --output-on-failure
```

If the host Eigen version is not 3.3.7, provide the reviewed headers:

```bash
cmake -S third_party/gril_native -B third_party/gril_native/build \
  -DGRIL_PINNED_EIGEN_INCLUDE_DIR=/path/to/eigen-3.3.7
cmake --build third_party/gril_native/build
```

Do not run `cmake --install`; the native install gate remains closed pending
algorithm-equivalence, physical-validation, and license/provenance approval.

## 3. Prepare and validate input

For Apollo record, build the C++ ingress once using Bzlmod and the local
`apollo-lite/pkgs/core` checkout:

Set the capture directory, then use the reviewed Vanjee-16 defaults:

```text
LiDAR: /apollo/sensor/vanjeelidar/up/PointCloud2
IMU:   /apollo/sensor/gnss/imu
```

```bash
cd tools/gril_record_core_reader
bazel build \
  --override_module=wheelos_core=/path/to/apollo-lite/pkgs/core \
  //:record_ingress
cd ../..
```

Pass one record shard or the directory containing all continuous split shards:

```text
capture/
├── capture.record.00000
├── capture.record.00001
└── capture.record.00002
```

Create a canonical dataset without running the algorithm:

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

For any capture, compare the C++ output against trusted canonical data before
running GRIL. The Weilan 0827 C++ ingress baseline and its exact Core revision,
array hashes, native result, and timings are documented in
[`lidar2imu_wheelos_core_baseline.md`](lidar2imu_wheelos_core_baseline.md).
Re-run the equivalence gate after changing the ingress code, Core revision,
topics, or sensor profile:

```bash
gril-migrate prepare \
  --input /mnt/synology/weilan/lidar2imu/0827 \
  --input-type record \
  --core-record-ingress tools/gril_record_core_reader/bazel-bin/record_ingress \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16 \
  --output-dir outputs/gril/0827_wheelos_core_baseline/dataset

gril-migrate compare-datasets \
  --left outputs/gril/0827_wheelos_core_baseline/dataset \
  --right outputs/gril/0827_data_layer_record \
  --output outputs/gril/0827_wheelos_core_baseline/canonical_comparison.yaml
```

The comparison is an ingress check, not a calibration-accuracy verdict. It
passes only when counts, ordering, timestamps, frames, rings, point times, and
numeric sensor data satisfy the comparison contract.

For a ROS1 bag, use `--input-type bag`, the bag path, and its actual LiDAR and
IMU topics. This is offline bag decoding, not live ROS integration. Check IMU
timestamp provenance for every capture before selecting the topic. Audit a new
adapter against trusted canonical arrays with `gril-migrate compare-datasets`;
do not use the configured extrinsic as conversion truth.

## 4. Run native GRIL

```bash
gril-migrate run-native \
  --input outputs/gril/capture01/dataset \
  --input-type canonical \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/capture01/run
```

Using `--input-type canonical` skips extraction only. It does not change or skip
the GRIL frontend, preprocessing, synchronization, or solver.

## 5. Review outputs

```text
outputs/gril/capture01/run/
├── customer_summary.yaml
├── GRIL_Calib_result.txt
├── manifest.yaml
├── dataset/
├── GRIL_full_frontend_trace_v1.txt
└── GRIL_batch_trace_v1.txt
```

Review `customer_summary.yaml`, then the native result and traces. The current
verdict is intentionally `review_required`; solver completion does not prove
horizontal lever-arm, yaw/time, or physical accuracy.

## Release status

There is no public installer or customer registry image. External distribution
remains blocked until native/reference equivalence, independent physical
holdout, and GRIL/LI-Init plus transitive dependency license provenance are
approved. The 0827 record/bag equivalence and the 20 native CTest / 29 Python
migration tests establish input-path and component evidence only; they do not
close the frozen-reference rotation gate or independent physical-accuracy gate.

## Optional local Docker review

The Dockerfile uses ignored files under `docker/lidar2imu/runtime/` and the
local `whl-cal-lidar2imu:slim` dependency-cache image. It is a controlled local
review path, not a published installer. See [container instructions](../docker/README.md)
for bundle preparation and image build steps. Rebuild and test native GRIL
after any C++ source change.

For a research run that produces `GRIL_batch_trace_v1.txt`, inspect timing
support before using the reported time offset:

```bash
python .agents/skills/gril-calib-validation/scripts/profile_gril_batch_timing.py \
  --input lidar2imu-output/GRIL_batch_trace_v1.txt \
  --output-dir lidar2imu-output/diagnostics --signal yaw_z
```

The script writes a per-window YAML report and plot. Peaks at the search
boundary, weak correlation, or disagreement between windows mean the offset
remains inconclusive; a filtered signal or successful native solver cannot
override independent physical validation. This is a development diagnostic,
not part of the delivered container interface.

To check whether the LiDAR-only frontend is the bottleneck, run
`evaluate_trajectories.py` with `--batch-trace` and all record shards. It
writes `frontend_trajectory.yaml`, `frontend_trajectory.png`, and
`frontend_yaw_drift.png`; the aligned paths, per-window yaw changes, and
path lengths must be reviewed before attempting time/translation optimization.
GNSS/INS odometry is coupled to the input IMU and is not independent
extrinsic truth. The sensor origins differ, so uncorrected path length and
ATE/RPE alone are not a frontend failure verdict.

For correspondence diagnosis, `GRIL_full_frontend_trace_v1.txt` records
`odometry` rows with the effective feature count and the mean absolute
point-to-plane distance of accepted correspondences at the final EKF
iteration (meters). A zero-feature frame reports `nan`, not a successful
zero residual. This in-sample residual is diagnostic only: lower values do
not establish correct LiDAR motion or extrinsic accuracy. Older native binaries
reported zeros or `nan` because the residual sum was never populated; identify
the executable by its digest. On a 2026-05-06 corrected-IMU capture, two
identical-input runs of the instrumented binary produced different frontend
traces and final solutions. Do not use their residual differences to rank
calibrations until repeated full-frontend states agree; compare windowed yaw
behavior and held-out point-cloud geometry as separate checks.

For developer comparisons, follow the
[LiDAR-to-IMU research decision contract](../.agents/knowledge/lidar2imu_research.md).
It separates dataset qualification, same-input GRIL/LI-Init experiments and
independent extrinsic review; the Weilan 0827 example is inconclusive and is
not a release or accuracy benchmark.
