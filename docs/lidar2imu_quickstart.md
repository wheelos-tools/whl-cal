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
`prepare-segment` rejects any known nonzero replay H, including offsets hidden
by overlapping raw ranges; review the full capture rather than cropping two
different physical windows with common absolute nanosecond boundaries.

`compare-datasets` compares the three optional time-provenance arrays exactly,
without retiming either input, and reports each side's presence and comparison
status. Differing values or a field present on only one side produce
`different`, even if all measurement/signal arrays match. Both sides lacking
a field retain legacy equivalence checks, with explicit `both_absent` status;
this is not evidence that missing publication times are equivalent. A new
record extraction compared against an older canonical/bag dataset can therefore
require re-extraction of the trusted provenance rather than a full-equivalence
claim based only on matching measurement arrays.

### Research segments

`prepare-segment` extracts a half-open absolute nanosecond interval from an
already audited canonical dataset. Choose boundaries and gap limits from the
capture's clock, scan rate and motion review; these example numbers are not
recommended thresholds for other captures.

```bash
lidar2imu prepare-segment \
  --input /path/to/canonical-dataset \
  --output-dir outputs/gril/segment_01/dataset \
  --start-ns START_NS --end-ns END_NS \
  --max-lidar-gap-ns 120000000 --max-imu-gap-ns 20000000
```

The output contains complete scans, IMU samples strictly inside the same
interval, an input contract and parent-manifest digest. Scans crossing the end
boundary are excluded. Extraction fails if IMU samples do not bracket all
retained points, timestamps repeat, or gaps exceed the supplied limits. It
does not check motion observability, clock provenance, GRIL warm-up or data
sufficiency. Reserve independent holdout data and ensure training and holdout
intervals, including their context, do not overlap. Run each segment separately
with `run-native --input-type canonical` and retain no-result runs.

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

### Opt-in stabilization and post-run quality gates

Default `lidar2imu run-native` and `run-native-batch` preserve frozen GRIL
behavior. For corner LiDARs, blind-sector scans, or multi-window/holdout
research runs, explicit opt-in flags are available without changing golden
defaults:

```bash
lidar2imu run-native \
  --input /path/to/canonical-dataset \
  --input-type canonical \
  --config path/to/stabilized_config.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/stabilized/run_01 \
  --start-scan 0 --scan-count 6800 \
  --ikd-rebuild sync \
  --cut-mode equal_time --min-cut-points 400 \
  --odom-ground-mode disabled --min-effective-features 30 \
  --ahrs-imu-hz 100 --ahrs-dt-mode continuous \
  --motion-start-min-speed-mps 0.35 --motion-start-min-omega-rad-s 0.05 \
  --min-accum-duration-s 640 --max-accum-duration-s 700 --eof-policy calibrate
```

Optional batch solver stabilization keys under `calibration:` in the YAML config
(`max_xcorr_lag_s`, `max_time_lag_2_s`, `rot_init_max_iters`,
`use_imu_angular_in_acc_cost`, `skip_secondary_acc_norm`, `tail_trim_states`)
are omitted when unset.

Evaluate post-run acceptance gates strictly on completed artifacts (never inside
the solver):

```bash
lidar2imu evaluate-quality-gates \
  --run-dir outputs/gril/stabilized/run_01 \
  --repeat-run-dir outputs/gril/stabilized/run_02 \
  --repeat-run-dir outputs/gril/stabilized/run_03 \
  --seed-run-dir outputs/gril/stabilized/seed_blind_identity_zero \
  --window-run-dir outputs/gril/stabilized/window_w1 \
  --window-run-dir outputs/gril/stabilized/window_w2 \
  --holdout-batch-trace outputs/gril/stabilized/holdout/GRIL_batch_trace_v1.txt \
  --output outputs/gril/stabilized/quality_gates.yaml
```

`evaluate-quality-gates` separates `data_sufficiency`, `motion_excitation`,
`parameter_observability` (rotation Hessian, 2D/3D translation lever-arm
singular values and boundary-hit check, time-offset angular-acceleration energy
and Pearson correlation), `repeatability`, `seed_sensitivity`, `multi_window`,
and `holdout` into `converged_and_verified` (`accepted`),
`degraded_unobservable` (`review_only`), or `failed` (`rejected`). See
`context/knowledge_base/validated_conclusions.md` and
`.agents/knowledge/lidar2imu_research.md` for rationale and validation evidence.

### Dual-sensor cross-validation and colored PCD diagnostics

When a platform carries two non-fused LiDARs calibrated independently against
the same IMU (for example `left_front` and `right_back` corner LiDARs), a static
dual-color point-cloud merge in the IMU frame is necessary to expose relative
ghosting (`T_{A<-B} = inv(T_{I<-A}) @ T_{I<-B}`), **but not sufficient on its
own** because any common-mode IMU bias `T_err` cancels out in static dual-LiDAR
overlays (`inv(T_err @ T_I_A) @ (T_err @ T_I_B) == inv(T_I_A) @ T_I_B`). Run
the 5-pillar post-run cross-diagnostic command to validate both relative
geometry and absolute IMU coupling:

```bash
lidar2imu evaluate-dual-diagnostics \
  --dataset-a /path/to/left_front/dataset \
  --run-dir-a /path/to/left_front/run_01 \
  --dataset-b /path/to/right_back/dataset \
  --run-dir-b /path/to/right_back/run_01 \
  --label-a left_front --label-b right_back \
  --output-dir outputs/gril/dual_sensor_diagnostics
```

This read-only evaluation writes `dual_diagnostics.yaml`, canonical extrinsic
YAMLs (`extrinsics/`), four binary RGB `.pcd` point clouds (`pointclouds/`,
coral-red vs cyan-blue for CloudCompare/Open3D inspection of raw static merge,
`L2L` loop-aligned reference, `100 Hz` pure-IMU turn submap, and common-mode IMU
bias control), and three diagnostic figures (`diagnostics/fig1..fig3`) covering:
1. Static structured dual-color PCD + orthogonal ground/wall 2D slices.
2. High-yaw-rate turn pure-IMU strapdown submap sharpness & common-mode bias
   ablation (breaking the static gauge blind spot and expanding diagonal corner
   FoV overlap across time).
3. `2 × lidar2imu` vs direct static `lidar2lidar` GICP loop-closure residual
   and 6×6 information conditioning.
4. Multi-heading ground plane normal vs IMU resting/quasi-uniform gravity
   vector alignment (cancelling road slope via opposite headings).
5. Three-curves-in-one angular-velocity alignment and opposite-corner
   tangential lever-arm acceleration cancellation.

To evaluate a candidate joint or externally refined extrinsic pair (using the
same 5-pillar diagnostic contract on an unseen holdout window), pass
`--extrinsic-a-yaml` and `--extrinsic-b-yaml`.

### Joint `2 × lidar2imu + lidar2lidar` spatiotemporal calibration (`calibrate-dual-joint`)

When dual corner LiDARs exhibit EKF lever-arm shrinkage in single-sensor
`lidar2imu`, run the staged joint solver documented in
[docs/dual_lidar_imu_joint_calibration_design.md](dual_lidar_imu_joint_calibration_design.md):

```bash
lidar2imu calibrate-dual-joint \
  --dataset-a /path/to/left_front/dataset \
  --run-dir-a /path/to/left_front/run_01 \
  --dataset-b /path/to/right_back/dataset \
  --run-dir-b /path/to/right_back/run_01 \
  --output-dir outputs/gril/joint_dual_calib
```

By default (`--mode v2`, `~4.5 s`), this runs `v1` -> `v2` and applies the
post-run Common-Mode Excitation Gate (`accepted` vs `needs_more_excitation`
plus explicit `z_cm_observable: false` planar nullspace reporting). Pass
`--mode all` to also run the experimental `v3` ablation:
- **`v1`**: Anisotropic $\mathrm{SE}(3)^2$ factor-graph fusion of L2I info, multi-window
  GICP L2L, multi-heading gravity-ground alignment, and explicit $z_{\text{cm}}$
  nullspace prior.
- **`v2` (promoted baseline)**: Upgrades L2L edges using
  `lidar2lidar/semantic_registration.py` (4-channel ground/facade/pole/calibrated-intensity
  primitives, self-occlusion 3D shared FoV, X-ICP localizability, MAD adaptive
  consensus, and low-speed submap fallback) + decoupled Roll/Pitch vs Yaw prior.
- **`v3` (experimental, LOTO-guarded)**: Turn-window `100 Hz` IMU-preintegrated
  cross-time multi-LiDAR plane BA with LOTO rollback protection.

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
