---
audience: customer
stability: review-only
---

# LiDAR-to-IMU calibration

The delivered container runs the ROS-free native GRIL pipeline. It includes the
record adapter, reviewed Vanjee-16 configuration, native frontend, and batch
solver. The customer does not install ROS, Python, Ceres, Eigen, or PCL.

## 1. Get the image

Use the exact version supplied with the release:

```bash
docker pull <registry>/whl-cal-lidar2imu:<version>
export WHL_LIDAR2IMU_IMAGE=<registry>/whl-cal-lidar2imu:<version>
```

For an offline delivery package:

```bash
docker load -i whl-cal-lidar2imu-<version>.tar
export WHL_LIDAR2IMU_IMAGE=whl-cal-lidar2imu:<version>
```

Record the immutable image digest together with every calibration result.

## 2. Prepare data

Place one Apollo record or all continuous split records from the same capture in
one directory:

```text
capture/
├── capture.record.00000
├── capture.record.00001
└── capture.record.00002
```

The default topics are:

```text
LiDAR: /apollo/sensor/vanjeelidar/up/PointCloud2
IMU:   /apollo/sensor/gnss/imu
```

The default sensor profile expects a 16-line Vanjee LiDAR. Do not silently use
it for another sensor model or installation.

## 3. Run calibration

```bash
mkdir -p "$PWD/lidar2imu-output"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD/capture",dst=/data,readonly \
  --mount type=bind,src="$PWD/lidar2imu-output",dst=/output \
  "$WHL_LIDAR2IMU_IMAGE" \
  --input /data \
  --input-type record \
  --output-dir /output
```

For alternate topics:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD/capture",dst=/data,readonly \
  --mount type=bind,src="$PWD/lidar2imu-output",dst=/output \
  "$WHL_LIDAR2IMU_IMAGE" \
  --input /data \
  --input-type record \
  --output-dir /output \
  --lidar-topic /apollo/sensor/lidar/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16
```

ROS1 bag and canonical datasets are selected with `--input-type bag` and
`--input-type canonical`.

## 4. Read the result

```text
lidar2imu-output/
├── customer_summary.yaml
├── GRIL_Calib_result.txt
├── manifest.yaml
├── dataset/
├── GRIL_full_frontend_trace_v1.txt
└── GRIL_batch_trace_v1.txt
```

Customer review order:

1. `customer_summary.yaml`: complete rotation, translation, time offset, and
   current verdict.
2. `GRIL_Calib_result.txt`: native solver result.
3. Published trajectory and accumulated-point-cloud review artifacts when the
   release workflow supplies them.

The current native GRIL customer verdict is intentionally `review_required`.
Solver completion does not prove horizontal lever-arm or yaw/time accuracy.

## Release status

The container is ready for local and controlled review. External customer
distribution remains blocked until:

1. native/reference rotation equivalence is accepted;
2. independent physical trajectory and point-cloud holdout accepts the result;
3. GRIL/LI-Init and transitive dependency license provenance is approved.

## Developer build

The Docker build consumes the validated bundle under
`docker/lidar2imu/runtime/`; it does not download or rebuild native
dependencies. Maintainers create that ignored bundle from the approved cache:

```bash
python docker/lidar2imu/prepare_cached_runtime.py \
  --frontend /path/to/cache/gril_native_full_frontend \
  --batch /path/to/cache/gril_native_batch \
  --library-prefix /path/to/cache/runtime-prefix

docker build -f docker/lidar2imu/Dockerfile \
  --network=none \
  -t whl-cal-lidar2imu:<version> .
```

Maintainers must rebuild and test native GRIL before creating an image after C++
source changes. Detailed metrics remain documented in
[`calibration_metric_layers.md`](calibration_metric_layers.md).

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
