---
audience: user
stability: stable
P26-05-25
---

# LiDAR-to-IMU quick start

## Requirements

- Python 3.8 or newer
- Apollo record containing LiDAR, pose, IMU, and `/tf_static`
- a reasonable initial `lidar -> imu` transform
- motion with left/right turns, acceleration/braking, and flat-road segments

For collection guidance, see
[apollo_data_collection.md](apollo_data_collection.md). For algorithm tuning,
see [lidar2imu_design.md](lidar2imu_design.md).

## Install

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run from an Apollo record

There are now two LiDAR-to-IMU paths in this repository:

- **`lidar2imu` staged solver**: the existing Python pipeline and current
  regression/production profiles.
- **GRIL native route**: the ROS-free migration of the original GRIL pipeline.
  Use this when you specifically want GRIL behavior from Apollo records.

Use the stable `lidar2imu` scan-to-scan baseline first when you are comparing
against the existing in-repo solver:

Current recommendation: **keep the staged solver**. GRIL is the desired
algorithm family for reproduction work, but the available same-repository
evidence does not prove it is a stronger production replacement yet.

| Method | Current evidence | Decision |
| --- | --- | --- |
| `lidar2imu --profile baseline` / `production` | Runs end-to-end, but current real-bag artifacts still report `warning` and `release_ready: false` for full 6-DoF acceptance. | Keep as the regression/reference surface. |
| `lidar2imu --solver-family gril_staged` / `gril_prob*` | Candidate staged variants also report `warning` / `release_ready: false` on the available artifacts. | Keep candidate-only; do not promote by name. |
| `gril-migrate run-native` | Reproduces GRIL components and emits a full native result without ROS; full A/B review is not review-ready on the current 0827 evidence. | Use for GRIL reproduction and A/B, not as a production replacement. |

Do not delete `lidar2imu` staged/baseline code until GRIL wins on the same
dataset matrix with the same post-run review contract: final result comparison,
repeatability, trajectory/holdout evidence, and physical validation. A single
friendly run or solver convergence is not enough.

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/imu \
  --output-dir outputs/lidar2imu/run01 \
  --profile baseline \
  --calibrate
```

If the bag does not contain the static transform, add:

```bash
--initial-transform /path/to/lidar_imu_extrinsics.yaml
```

## Run the ROS-free GRIL route from an Apollo record

Use this path when the target algorithm is **GRIL**, not the current
`lidar2imu` staged solver. The native GRIL implementation lives in
`third_party/gril_native`; Python only converts the record into the canonical
dataset, writes native inputs/configs, invokes the C++ executable, and assembles
review artifacts.

Build the native GRIL executable first:

```bash
cmake -S third_party/gril_native -B third_party/gril_native/build
cmake --build third_party/gril_native/build
ctest --test-dir third_party/gril_native/build --output-on-failure
```

If the host Eigen is not the reviewed `3.3.7` frontend version, configure with:

```bash
cmake -S third_party/gril_native -B third_party/gril_native/build \
  -DGRIL_PINNED_EIGEN_INCLUDE_DIR=/path/to/eigen-3.3.7
```

Run GRIL directly from one or more Apollo records:

```bash
gril-migrate run-native-frontend \
  --input /path/to/capture.record.00000 \
  --input /path/to/capture.record.00001 \
  --input-type record \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16 \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/native_run01
```

`gril-migrate run-native` is an alias for the same complete native command.
For repeated runs, split extraction from execution:

```bash
gril-migrate prepare \
  --input /path/to/capture.record.00000 \
  --input-type record \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16 \
  --output-dir outputs/gril/datasets/run01

gril-migrate run-native \
  --input outputs/gril/datasets/run01/dataset.yaml \
  --input-type canonical \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/native_run01
```

The native run writes:

- `GRIL_Calib_result.txt`
- `GRIL_batch_trace_v1.txt`
- `GRIL_full_frontend_trace_v1.txt`
- `manifest.yaml`
- `dataset/input_contract.yaml` when the input was a record or bag

This command does **not** apply quality gates during algorithm execution. GRIL's
own `data_sufficiency_assess` still decides when to call `LI_Calibration`; all
review gates are post-run evidence.

### Review GRIL A/B evidence

When a frozen ROS reference run and a repeated native run are available, assemble
the migration review:

```bash
gril-migrate review-full \
  --reference-trace-archive outputs/gril/reference/reference_frontend_trace_manifest.yaml \
  --reference-result outputs/gril/reference/run_1/GRIL_Calib_result.txt \
  --reference-config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --candidate-manifest outputs/gril/native_run01/manifest.yaml \
  --repeat-manifest outputs/gril/native_run01_repeat/manifest.yaml \
  --evidence outputs/gril/native_run01/validation_evidence.yaml \
  --output-dir outputs/gril/native_run01_review
```

Interpret the review in this order:

1. `input.verdict` and `canonical_array_hashes_verified`
2. component evidence for preprocessing, sync, CV propagation, Patchwork++, and
   isolated odometry/EKF
3. `result_equivalence` for final rotation, translation, and time offset
4. `native_repeatability`
5. physical holdout verdict

The current 0827 evidence is intentionally strict:

- component-level GRIL reproduction passes
- final native output is generated without ROS
- native repeatability passes
- full A/B is **not review-ready** because the rotation/yaw delta is
  `0.406964 deg`, above the `0.2 deg` migration gate
- translation and time offset pass their gates
- the original GRIL/reference path is also not a production-quality physical
  calibration on that capture, so do not treat a failed `<0.2 deg` A/B as proof
  that native alone is wrong

This means the current conclusion is **inconclusive for replacement**:
GRIL-native is operationally useful for reproducing and studying GRIL, but it
has not beaten the staged solver strongly enough to remove the staged solver
from the repository.

The remaining full-state mismatch is caused by upstream ikd-tree's unconfigured
pthread background rebuild scheduling. Do not hide it with fixed sleeps,
thread serialization, or disabling rebuilds: that would create a deterministic
candidate, not an equivalent GRIL reproduction. If production needs a
deterministic GRIL-derived candidate, gate it as a new method and compare it
against the frozen GRIL reference and independent physical holdouts.

## Recommended fast path

For repeated runs, build a prepared dataset once:

```bash
lidar2lidar-rig-dataset \
  --record-path /path/to/record \
  --output-dir outputs/prepared/run01 \
  --lidar-topics /apollo/sensor/your_lidar/PointCloud2 \
  --reference-topic /apollo/sensor/your_lidar/PointCloud2 \
  --export-voxel-size 0.05
```

Then calibrate from cached PCD, pose, and IMU artifacts:

```bash
lidar2imu-convert-record \
  --prepared-dataset-yaml \
    outputs/prepared/run01/diagnostics/prepared_rig_dataset.yaml \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --output-dir outputs/lidar2imu/run01 \
  --profile baseline \
  --calibrate
```

Prepared mode avoids repeatedly scanning the raw record during extraction,
registration review, trajectory visualization, and cloud-thickness evaluation.

## Run from standardized samples

```bash
lidar2imu-calibrate \
  --input outputs/lidar2imu/run01/standardized_samples.yaml \
  --output-dir outputs/lidar2imu/run01_replay
```

This is the fastest path for solver-only A/B tests.

## Check the result

Read the conclusion first:

```bash
python - <<'PY'
import yaml

path = "outputs/lidar2imu/run01/calibration/metrics.yaml"
with open(path, "r") as stream:
    metrics = yaml.safe_load(stream)

print(metrics["summary"]["final_acceptance_status"])
print(metrics["summary"]["release_ready"])
print(metrics["final_acceptance"]["recommendation"])
PY
```

Required production conditions:

- `summary.final_acceptance_status: pass`
- `summary.release_ready: true`
- Fisher eigenvalue and conditioning gates pass
- holdout cloud-thickness gate passes

Open the visual report:

```bash
xdg-open outputs/lidar2imu/run01/calibration/diagnostics/review_report.html
```

Key artifacts:

- `calibrated_tf.yaml`: calibrated transform
- `metrics.yaml`: acceptance result and metrics
- `diagnostics/data_quality.yaml`: extraction quality
- `diagnostics/registration_review.yaml`: per-window registration quality
- `diagnostics/trajectory_overlay.svg`: IMU/LiDAR trajectory comparison
- `diagnostics/trajectory_overlay_cloud.ply`: geometric overlay

Do not accept a run only because the solver converged. If
`freeze_xyyaw` was applied, only `z/roll/pitch` should be treated as usable.

## Diagnose extraction failures

Inspect:

```bash
cat outputs/lidar2imu/run01/conversion_diagnostics.yaml
```

Important fields:

- `motion_rejected_frame_gap`: candidates crossing LiDAR data gaps
- `lidar_median_frame_delta_ms`: nominal LiDAR frame period
- `motion_rejected_low_fitness`: failed registrations
- `motion_registered_candidate_count`: usable motion factors

If extraction does not meet the minimum sample count, the converter writes
`calibration_skipped.yaml` and does not run the solver.

For advanced solver, observability, temporal-offset, or submap experiments,
follow [lidar2imu_design.md](lidar2imu_design.md) and keep the baseline output
for comparison.
