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
