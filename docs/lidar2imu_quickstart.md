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

Use the single `lidar2imu` command for LiDAR-to-IMU data preparation,
calibration, and review.

The `lidar2imu.adapters` registry is the data-ingress extension point. A new format
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

For the current Weilan 0827 working profile, use:

```text
LiDAR: /apollo/sensor/vanjeelidar/up/PointCloud2
IMU:   /apollo/sensor/gnss/corrected_imu
```

The IMU topic above is verified for this capture's measurement-time header.
For other captures, inspect timestamp provenance and select the correct topic;
do not infer sensor time from a topic name. The historical 0827
`/apollo/sensor/gnss/imu` ingress baseline has a different signal/time identity.

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
lidar2imu prepare \
  --input /mnt/synology/weilan/lidar2imu/0827 \
  --input-type record \
  --core-record-ingress tools/gril_record_core_reader/bazel-bin/record_ingress \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/corrected_imu \
  --scan-lines 16 \
  --output-dir outputs/gril/0827_corrected_imu_baseline/dataset
```

For the validated shard set, expect 1,375 scans, 35,239,034 finite points, and
13,741 IMU samples. Inspect `dataset/input_contract.yaml` for the selected
topics, frames, ring/point-time semantics, continuity, static TF, and corrected
IMU timestamp provenance before running the solver.

The corrected-IMU 0827 profile has a trusted ROSbag canonical comparison
recorded in the [research decision contract](../.agents/knowledge/lidar2imu_research.md).
The result proves input equivalence only; the extrinsic's independent physical
verdict is rejected. For any new adapter or changed input path, compare the C++
output against trusted canonical data before running GRIL. The historical
raw-IMU Weilan 0827 C++ ingress baseline and its exact Core revision,
array hashes, native result, and timings are documented in
[`lidar2imu_wheelos_core_baseline.md`](lidar2imu_wheelos_core_baseline.md).
Re-run the equivalence gate after changing the ingress code, Core revision,
topics, or sensor profile:

```bash
# Historical raw-IMU adapter regression only; do not compare it to corrected_imu.
lidar2imu prepare \
  --input /mnt/synology/weilan/lidar2imu/0827 \
  --input-type record \
  --core-record-ingress tools/gril_record_core_reader/bazel-bin/record_ingress \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16 \
  --output-dir outputs/gril/0827_wheelos_core_baseline/dataset

lidar2imu compare-datasets \
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
adapter against trusted canonical arrays with `lidar2imu compare-datasets`;
do not use the configured extrinsic as conversion truth.

The GRIL method and the reason 0827 is only a working fixture are documented
in [GRIL baseline method](lidar2imu/07_gril_baseline.md). Do not use the
historical raw-IMU canonical dataset as the corrected-IMU run input or as its
comparison target.

### Measurement clocks and publication time

Canonical measurement timestamps are never replaced with host headers:
Apollo point timestamps define the scan start and relative point times;
`measurement_time` defines the LiDAR **scan end** and, when valid, raw IMU
measurement time. Optional NPZ arrays preserve the original LiDAR scan-end
`measurement_timestamps_ns` and independent `dispatch_timestamps_ns` for both
streams (zero means unavailable). IMU headers used as measurement-time fallback
are not also claimed as independent dispatch evidence. The corrected-IMU topic
selection above is unchanged.

With complete independent publication evidence in a shared host clock and
overlapping publication ranges, native replay first computes one coarse offset
candidate, even when the original measurement ranges overlap:

```text
candidate = median(IMU measurement - IMU publication)
          - median(LiDAR scan_end measurement - LiDAR publication)
H = candidate if abs(candidate) > 1 second, otherwise 0
native IMU time = original IMU measurement - H - configured C
reported total IMU-to-LiDAR lag = H + C + GRIL tau1 + GRIL tau2
```

Events merge on fixed-H aligned measurement time; raw payload timestamps,
point intervals and IMU signals remain unchanged. Backward host headers are
retained and contribute only to the one-shot median, never to per-message
synchronization or a drift model. Publication latency asymmetry can bias this
coarse initializer: it is not a calibrated time offset or accuracy evidence.
H is locked before either stream enters native synchronization and survives
explicit forward-gap resets. Runtime trace/manifest record H and C separately;
the existing batch-trace `timediff_imu_wrt_lidar` field carries their sum.
The one-second threshold retains GRIL's hard-offset semantics and leaves
same-clock/small-offset replay unchanged. A ten-second capture with a five-second
clock offset still needs H, despite overlapping raw timestamp ranges.

Schema-1 canonical datasets and old C++ interchange-v1 files still load.
Without complete independent evidence, legacy canonical and ROS bags retain
H=0 compatibility when their measurement ranges are not separated by more than
one second; this is not proof of a shared clock. Larger range separation fails
explicitly: re-extract the original record instead of guessing an offset.
Bags do not establish independent publication timing from storage timestamps.

`compare-datasets` compares the three optional time-provenance arrays exactly,
without retiming either input, and reports each side's presence and comparison
status. Differing values or a field present on only one side produce
`different`, even if all measurement/signal arrays match. Both sides lacking
a field retain legacy equivalence checks, with explicit `both_absent` status;
this is not evidence that missing publication times are equivalent. A new
record extraction compared against an older canonical/bag dataset can therefore
require re-extraction of the trusted provenance rather than a full-equivalence
claim based only on matching measurement arrays.

## 4. Run native GRIL

```bash
for run in 01 02 03; do
  lidar2imu run-native \
    --input outputs/gril/0827_corrected_imu_baseline/dataset \
    --input-type canonical \
    --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
    --executable third_party/gril_native/build/gril_native_full_frontend \
    --output-dir "outputs/gril/0827_corrected_imu_baseline/run_${run}"
done
```

Using `--input-type canonical` skips extraction only. It does not change or skip
the GRIL frontend, preprocessing, synchronization, or solver.
Keep all three full runs and use the same canonical dataset, config,
executable, and default `golden` gap policy. The native executable is built
from the `gril_native_full_frontend_exec` CMake target.

For an explicitly identified research initialization experiment, the YAML may
include `calibration.rotation_LI_xyzw: [qx, qy, qz, qw]`, a finite unit
quaternion mapping native LiDAR coordinates into native IMU coordinates.
Omitting it preserves the frozen identity rotation seed. This initializes
the batch rotation-only solve; it does not rotate input measurements, fix
the estimated rotation, or initialize the LiDAR-only odometry frontend.
The full frontend passes this option through its generated batch config.
Use a rebuilt batch executable that supports the option.

Compose the actual TF chain first. Its reported translation `t_LI` and
rotation `R_LI` require `trans_IL = -R_LI.T @ t_LI` in the calibration config.
Do not substitute a mechanical mounting angle for a coordinate transform:
an RFU (right/forward/up) IMU and FLU (forward/left/up) reference differ by
90 degrees even before considering mounting. TF is an initialization prior,
not accuracy ground truth; keep the unseeded run and all quality gates.

## 5. Review outputs

```text
outputs/gril/0827_corrected_imu_baseline/
├── dataset/
│   ├── dataset.yaml
│   ├── input_contract.yaml
│   ├── lidar.npz
│   └── imu.npz
├── run_01/
│   ├── native_dataset_v1.bin
│   ├── gril_native_full.conf
│   ├── gril_native.conf
│   ├── customer_summary.yaml
│   ├── GRIL_Calib_result.txt
│   ├── manifest.yaml
│   ├── GRIL_full_frontend_trace_v1.txt
│   └── GRIL_batch_trace_v1.txt
├── run_02/
├── run_03/
└── diagnostics/
```

Review `customer_summary.yaml`, then the native result and traces. The current
verdict is intentionally `review_required`; solver completion does not prove
horizontal lever-arm, yaw/time, or physical accuracy.
The runner does not create root-level `metrics.yaml` or `diagnostics/`
automatically; place independent repeatability, holdout, trajectory and
visualization reports there without overwriting raw run artifacts.

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
independent extrinsic review. Historical raw-IMU 0827 evidence is inconclusive;
the corrected-IMU working fixture's physical verdict is rejected. Neither is
a release or accuracy benchmark.
