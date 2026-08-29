---
audience: developer
stability: experimental
---

# GRIL ROS-free migration

## Release decision — 2026-08-29

**Verdict: developer/research build only; not a production release and not
cleared for distribution.** The complete Apollo record/canonical dataset ->
native C++ GRIL execution path exists and its normal runtime does not start or
require ROS. That execution capability is distinct from approval to use,
redistribute, or promote its calibration result.

| Release-gate evidence | Current conclusion |
| --- | --- |
| complete native execution | passed: record or canonical input reaches the native frontend and batch solve without a ROS runtime |
| proven frontend components | passed: preprocessing, FIFO synchronization, IMU-free constant-velocity propagation, Patchwork++ ground segmentation, and isolated LiDAR-only odometry/EKF have focused equivalence evidence |
| frozen-reference final result | **not review-ready**: rotation/yaw discrepancy is **0.406964 deg**, above the **0.2 deg** migration gate; translation, time-offset, and native repeatability checks pass |
| exact long-run frontend state identity | not required for acceptance: upstream ikd-tree `pthread` rebuild scheduling prevents it; this does **not** waive the final-result gate |
| physical calibration validation | blocked: the frozen reference physical verdict is rejected; the native independent physical holdout is unknown |
| redistribution and installation | blocked pending formal GRIL/LI-Init licensing provenance review and the explicit native CMake install gate |

Do not present `gril_native` as a production calibration tool, use the
candidate result as an accepted physical calibration, or distribute a native
binary/source package from this repository. The frozen ROS implementation is a
migration oracle, not a fallback production release.

## Current status

| Layer | Status |
| --- | --- |
| frozen deterministic ROS reference | retained and executable |
| Apollo record adapter | implemented |
| ROS1 bag adapter without ROS runtime | implemented |
| canonical record/bag equivalence | passed on the 0827 capture |
| GRIL batch optimizer | migrated to normal CMake without ROS |
| reference batch trace replay | passed twice with identical six-decimal results |
| Velodyne preprocessing | migrated; real-frame trace is byte-identical |
| synchronization and IMU-free CV propagation | migrated and trace-equivalent |
| Patchwork++ ground segmentation | migrated and trace-equivalent across 21 sequential frames |
| ikd-tree LiDAR odometry and LiDAR-only EKF | integrated into the live native event loop |
| complete record/canonical dataset -> native GRIL result path | implemented without ROS runtime; full frozen-reference review evidence is tracked separately |
| complete ROS-free GRIL release | blocked: the current frozen-reference rotation discrepancy is above its `0.2 deg` migration gate |

The 0827 reference remained repeatable within `0.107 deg`, `0.014 m`, and
`0.188 ms`, but its complete physical verdict was still rejected by dynamics
and yaw-time holdout. Migration equivalence does not upgrade that calibration
verdict.

## Decision

The production target is the original GRIL computational pipeline without a ROS
runtime. The current `lidar2imu` solvers are not an algorithm candidate. This
repository reuses only their extraction -> algorithm -> evaluation separation
and stable review-artifact approach.

Two GRIL implementations remain during migration:

- `gril_ros_reference`: frozen ROS Noetic reference, used only as the migration
  oracle
- `gril_native`: ROS-free C++ candidate and eventual user-facing implementation

Run the frozen reference through its existing containerized validation workflow:

```bash
gril-migrate run-reference \
  --record-file capture.record.00000 \
  --output-dir outputs/gril/reference
```

This command intentionally requires Docker and uses ROS inside the reference
container. It is an A/B oracle, not part of the final user runtime.

## Frozen reference

Run `gril-migrate reference-info` for the machine-readable revision and patch
identity. Distribution of migrated GRIL sources or binaries is blocked until
the upstream licensing provenance is reviewed: `package.xml` declares BSD, but
the repository includes the GPLv2 LI-Init license and describes GRIL as derived
from LI-Init.

## Canonical input

Both Apollo record and ROS1 bag inputs are converted without starting ROS:

```bash
gril-migrate prepare \
  --input capture.record.00000 \
  --input-type record \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --output-dir outputs/gril/datasets/record

gril-migrate prepare \
  --input gril_input.bag \
  --input-type bag \
  --lidar-topic /velodyne_points \
  --imu-topic /imu/data \
  --output-dir outputs/gril/datasets/bag
```

The output contains `dataset.yaml`, `lidar.npz`, `imu.npz`, and
`input_contract.yaml`. Array hashes, frame IDs, point-time semantics, source
paths, rejected counts, and transform metadata are recorded in the manifest.

Run the complete native frontend directly from an Apollo record:

```bash
gril-migrate run-native-frontend \
  --input capture.record.00000 \
  --input-type record \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/native
```

`run-native` is an alias. Use `--input-type canonical` with an existing
`dataset.yaml` or dataset directory to skip extraction. A successful run writes
`GRIL_Calib_result.txt`, `GRIL_batch_trace_v1.txt`,
`GRIL_full_frontend_trace_v1.txt`, and `manifest.yaml`. It does not apply
quality gates or replace GRIL's data-sufficiency decision.

At the sufficiency boundary the frontend stops the ikd-tree rebuild worker,
writes the exact live `GRIL_BATCH_TRACE 1` queues, and immediately executes the
clean native batch target. That target calls the unchanged
interpolation -> first-10 discard -> filtering/differentiation -> rotation ->
alignment -> joint-solve `LI_Calibration` sequence. The process boundary keeps
the active frontend allocator/thread state out of Ceres without reconstructing
or filtering the batch inputs.

Golden-equivalence mode is the default. Production forward-gap reset behavior
is separate and must be requested explicitly:

```bash
gril-migrate run-native \
  --input outputs/gril/datasets/record \
  --input-type canonical \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/native_reset \
  --gap-policy reset \
  --forward-gap-s 1.0
```

Reset mode recreates synchronization, propagation, Patchwork++, AHRS,
odometry/EKF, sufficiency, and calibration state, so derivatives do not bridge
the declared gap. It is intentionally outside golden equivalence.

The earlier partial evidence command remains available under an accurate name:

```bash
gril-migrate run-native-frontend-trace \
  --input capture.record.00000 \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --frontend-event-executable build/gril_native_frontend_event_trace \
  --ground-executable build/gril_native_ground_trace \
  --output-dir outputs/gril/native_frontend_trace
```

Compare equivalent record and bag conversions before comparing algorithms:

```bash
gril-migrate compare-datasets \
  --left outputs/gril/datasets/record \
  --right outputs/gril/datasets/bag \
  --output outputs/gril/input_equivalence.yaml
```

## Migration order

1. Freeze the patched deterministic ROS reference and golden outputs.
2. Export reference frontend and aligned-calibration traces.
3. Extract a normal-CMake batch calibration core that consumes saved LiDAR,
   IMU, and plane states.
4. Match optimizer traces and final parameters.
5. Replace ROS IMU and point-cloud types with native structs.
6. Reproduce callback ordering in a deterministic event replay loop.
7. Port preprocessing, synchronization, LiDAR odometry, and ground constraints
   one boundary at a time.
8. Compare the complete native run against the frozen reference.

Do not refactor GRIL mathematics during migration. Preserve preprocessing,
state propagation, sample ordering, filtering, differentiation, residuals,
parameterization, solver settings, and transform conventions.

Quality thresholds are post-run diagnostics. They must not filter samples,
change solver parameters, stop GRIL execution, or otherwise feed back into the
algorithm-equivalence run. Migration is established first by matching the
frozen reference's intermediate states and outputs on identical inputs.

### Native batch replay

After applying `gril/reference_patches/gril-batch-trace-v1.patch` on top of the
frozen validation patch, the reference writes
`result/GRIL_batch_trace_v1.txt` immediately before `LI_Calibration`.

Replay that exact optimizer input without ROS:

```bash
gril-migrate run-native-batch \
  --executable /path/to/gril_native_batch \
  --trace GRIL_batch_trace_v1.txt \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --output-dir outputs/gril/native_batch
```

This proves the optimizer migration only. It does not yet prove that the
ROS-free preprocessing, LiDAR odometry, motion undistortion, or ground
segmentation frontend is equivalent.

### Native Velodyne preprocessing

The frozen reference and native implementation trace scans 1, 20, and 21,
covering the warm-up-to-steady cut transition. On the full 0827 canonical
dataset, all sorted surface points, cut timestamps, and cut-cloud points are
byte-identical across 50,641 trace lines (SHA256
`b1a29dc476ec86a4b3bb935f6bb17b43bee9f8f8e00d716e2e2fa40bd47c315e`).
Synthetic tests separately cover fallback azimuth timing and invalid points.
The reference launch's `point_filter_num=3` is repeated explicitly in the
algorithm YAML so the native frontend cannot silently choose another default.

### Patchwork++ ground segmentation

On the 0827 canonical bag, the frozen ROS reference at
`c09b01a05ec83bc0a361941acf897109aaecf0a6` and the ROS-free
`PatchworkppNative.h` replayed sequential frames 1--21 with preserved adaptive
state. Their ground/non-ground traces were byte-identical (SHA256
`4bca9139c4e229d37ef43c3fa675f2ff25d4c99db6a3faf390df8406b27e02f7`);
there were zero numerical or point-order differences. The review artifact is
`outputs/gril/ground_equivalence_0827/equivalence.yaml`.

### Synchronization and IMU-free propagation

Canonical LiDAR/IMU events from the first 21 scans reproduce the first 25 GRIL
cut packages byte-for-byte, including package timestamps, consumed IMU
timestamps, and every input point (177,504 lines; SHA256
`06bcf249bb21ed8f5014a994258850ebc010d53c577672b02c8b123dde50149a`).

Replaying those packages through the native constant-velocity propagation
matches 1,802,946 traced values. All clouds and non-covariance states are exact.
Eigen evaluation order produces 680 covariance last-bit differences, bounded
by `1.11e-16` absolute and `2.04e-15` relative error. The comparison uses a
fixed 16-machine-epsilon roundoff bound, not a calibration-quality threshold.

### Native LiDAR-only odometry and EKF

The pinned ikd-tree sources and the voxel-filter/scan-to-map update now
build as `GRIL::odometry`. The native layer preserves the frozen LiDAR-only
order and thresholds, including five neighbors, search bound `5`, plane
threshold `0.1`, score threshold `0.9`, point weight `1000`, convergence
thresholds `0.01 deg` and `0.015 cm`, rematch schedule, ground rows, covariance
update, local-cube movement, and incremental map insertion. The frozen
Velodyne launch-only values are explicit in the reviewed YAML:
`max_iteration=5`, `cube_side_length=1000`.

Exact frontend reproduction requires Eigen `3.3.7`. Eigen `3.4` changes the
iterated matrix expressions by last bits, which eventually changes float
voxel identities. The standalone build therefore rejects another Eigen
version unless `GRIL_PINNED_EIGEN_INCLUDE_DIR` names reviewed `3.3.7` headers.
It also removes architecture-specific AVX flags inherited from a host PCL
build. PCL `1.15` changed both voxel accumulation order and mean/covariance
accumulation; the native odometry and Patchwork++ paths explicitly retain the
PCL `1.10` operations used by the frozen Noetic reference.

`gril_native_odometry_trace` accepts propagated state, undistorted cloud, and
ground rotation/normal records through a versioned deterministic boundary and
emits states, clouds, correspondences, iteration/rematch counts, and map
counters for focused odometry A/B. Synthetic CMake tests cover the pinned
configuration, first-frame tree construction, and the second-frame iterated
EKF/map update.

The integrated executable adds the deterministic
`GRIL_NATIVE_DATASET 1` binary input and `GRIL_FULL_FRONTEND_TRACE 1` output
boundaries. Nanosecond timestamps are converted with ROS `Time::toSec`
arithmetic, Fusion AHRS is updated before the odometry ground update, and every
cut retains the Patchwork++ state from its own source full scan. Thus all cuts
from one source scan reuse one ground estimate even if a later scan arrives
before the final cut synchronizes. A LiDAR rollback clears these paired pending
ground states with its FIFO. The trace uses the reference FNV-1a cloud identity
and records the complete covariance rather than diagonal-only state.

The safe compatibility fixes make package metadata exact, make all propagated
and updated states exact through package 21, and preserve exact downsample
identities through package 250. Replaying the first 25 reference propagated
states through the native odometry boundary produces 25 byte-identical updated
states, proving the EKF equations and order are migrated correctly.

The remaining difference is the upstream ikd-tree background rebuild worker.
The pinned and native `ikd_Tree.{h,cpp}` sources are byte-identical, and both
start the worker with the unconfigured
`pthread_create(&rebuild_thread, NULL, multi_thread_ptr, (void*) this)`.
The worker polls with the upstream `usleep(100)` interval, while foreground
`Add_Points`, `Delete_Point_Boxes`, `Nearest_Search`, and `Search` behavior
branches on whether `Rebuild_Ptr` is active. There is no upstream configuration
for worker start order, CPU affinity, priority, or completion timing.

With identical focused odometry inputs, the reference reported tree size
`6252` at package 10. Four ordinary native runs reported
`6255, 6269, 6269, 6262`; four runs constrained to the still-parallel CPU set
`0-3` reported `6270, 6258, 6273, 6271`; and four runs with inherited
`nice -n 10` reported `6270, 6241, 6262, 6268`. The output traces differed
between runs. This rules out common process affinity and priority settings as
general equivalence mechanisms; they change scheduling conditions but do not
specify the upstream interleaving.

The full trace first reports a tree-size difference at package 10
(`6252` reference versus `6266` in the recorded native run); states remain
exact through package 21, differ by last bits at package 22, and the first
downsample identity changes at package 251. The current full A/B result has a
`0.406964 deg` rotation discrepancy, beyond the frozen `0.2 deg` comparison
threshold. Translation and time-offset comparisons, plus the native
repeatability comparison, pass their respective gates; they do not compensate
for the rotation failure. Disabling, waiting for, adding ordering around, or
otherwise pacing the rebuild would change the upstream asynchronous semantics,
so exact byte-identical long-run state reproduction is not a safe migration
target. This exact-state limitation does **not** waive the final-result
comparison: the candidate remains not review-ready until the frozen-reference
rotation gate passes.

Run `gril-migrate review-full` only after the candidate and repeat executions
finish. It reads completed manifests, traces, result files, component evidence,
and physical-holdout evidence to write `full_ab_review.yaml` and
`equivalence.yaml`; it never invokes GRIL or changes its inputs, configuration,
quality thresholds, or scheduling.

```bash
gril-migrate review-full \
  --reference-trace-archive outputs/gril/reference/reference_frontend_trace_manifest.yaml \
  --reference-result outputs/gril/reference/run_1/GRIL_Calib_result.txt \
  --reference-config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --candidate-manifest outputs/gril/native/manifest.yaml \
  --repeat-manifest outputs/gril/native_repeat/manifest.yaml \
  --evidence outputs/gril/review_evidence.yaml \
  --output-dir outputs/gril/full_review
```

`review_evidence.yaml` is a read-only `GRIL_FULL_REVIEW_EVIDENCE 1` input. It
must explicitly identify the five focused components above and separately
record physical validation; `review-full` does not manufacture either kind of
evidence. Its output is a migration review, not an acceptance certificate.

### Frozen full-frontend trace

`gril/reference_patches/gril-full-frontend-reference-trace-v2.patch` is applied
after the validation, batch, preprocessing, propagation, and ground trace
patches. It does not alter gates or mathematics. The ROS runner archives
`GRIL_full_frontend_reference_trace_v2.txt` for every run and writes
`reference_frontend_trace_manifest.yaml`, which records the pinned source
revision/archive SHA256, every patch SHA256, input-bag SHA256, and per-run
trace SHA256 plus event counts for direct trace-completeness comparison.

The versioned trace provides one package per synchronized cut, with its source
scan header timestamp, propagation state, ordered-downsample-cloud FNV-1a
identity, map initialization/statistics, per-EKF-iteration correspondence
identity and residual summary, each updated full state, motion-start event, and
the exact calibration queue-push payload. It is the comparison artifact for
native `FullFrontend`/`LidarOdometry`; its FNV-1a cloud and correspondence
identities hash the ordered float bit patterns stated in the trace header.

## Result comparison

```bash
gril-migrate compare-results \
  --reference outputs/reference/GRIL_Calib_result.txt \
  --candidate outputs/native/GRIL_Calib_result.txt \
  --output outputs/gril/result_equivalence.yaml
```

The initial research gates are 0.2 degrees, 0.03 m, and 1 ms. Final tolerances
must also account for the measured repeatability envelope of the frozen
reference. Final extrinsics alone are insufficient: frontend states, aligned
states, stage costs, repeatability, trajectory diagnostics, and independent
physical holdout must also agree.

Build the combined migration report after both implementations have run:

```bash
gril-migrate abtest \
  --reference-dataset outputs/gril/datasets/bag \
  --candidate-dataset outputs/gril/datasets/bag \
  --reference-result outputs/reference/GRIL_Calib_result.txt \
  --candidate-result outputs/native/GRIL_Calib_result.txt \
  --reference-config reference_gril.yaml \
  --candidate-config native_gril.yaml \
  --reference-trace outputs/reference/GRIL_batch_trace_v1.txt \
  --candidate-trace outputs/native/GRIL_batch_trace_v1.txt \
  --output-dir outputs/gril/abtest
```

The two dataset arguments are deliberately explicit. A result comparison is
invalid unless the input-equivalence and algorithm-configuration gates also
pass. Batch replay additionally requires byte-identical trace hashes.
Runtime-only topic, publication, PCD-save, and trajectory-output settings are
excluded from the configuration identity.

## Native dependencies

The migrated batch core retains Eigen and Ceres. The complete native frontend
additionally retains PCL, OpenMP/pthreads, and Patchwork++. ROS,
catkin, rosbag playback, TF, generated ROS messages, Livox ROS drivers, and ROS
visualization publishers are removed from the production target.

## Packaging gate

The CMake tree is currently a **developer/research source build only**. It
does not define a distributable install or CPack artifact, and its explicit
`cmake --install` gate fails until release approval. `pip install` only
installs the Python `gril-migrate` migration/evidence CLI; it does not compile
or bundle `gril_native_full_frontend`. There is no native-runtime Docker image,
installer, or wheel available or implied by this documentation.

Do not create a native installer, publish a binary, or describe a calibration
result as a production release until all of these are complete:

1. full frozen-reference A/B passes the `0.2 deg` rotation gate (the current
   discrepancy is `0.406964 deg`);
2. the native physical validation verdict is accepted (the reference is
   rejected and native remains unknown);
3. formal licensing/provenance review resolves the GRIL package metadata BSD
   declaration against its LI-Init GPLv2 lineage and records all redistribution
   obligations, including native dependencies.

This gate does not change frontend or calibration mathematics and must remain
separate from the post-run validation gates.
