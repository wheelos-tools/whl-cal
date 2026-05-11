---
audience: user
stability: stable
P26-04-27
---


# LiDAR-to-IMU Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick run:
```bash
lidar2imu-calibrate --input lidar2imu_samples.yaml --output-dir outputs/lidar2imu/run01
```

Outputs: consolidated `calibrated_tf.yaml`, run `metrics.yaml` and `diagnostics/`.

When one bag needs to feed both `lidar2imu` and `lidar2lidar`, first prepare a
shared raw-LiDAR-only dataset:
```bash
lidar2lidar-rig-dataset \
  --record-path /path/to/record \
  --output-dir outputs/prepared/rig_run \
  --lidar-topics \
    /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
    /apollo/sensor/vanjeelidar/right_front/PointCloud2 \
    /apollo/sensor/vanjeelidar/right_back/PointCloud2 \
    /apollo/sensor/vanjeelidar/left_back/PointCloud2 \
  --reference-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --sync-threshold-ms 40 \
  --frame-stride 2 \
  --export-voxel-size 0.10
```

Then run conversion / calibration from that prepared dataset:
```bash
lidar2imu-convert-record \
  --prepared-dataset-yaml outputs/prepared/rig_run/diagnostics/prepared_rig_dataset.yaml \
  --lidar-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --output-dir outputs/lidar2imu/rig_left_front \
  --profile baseline \
  --calibrate
```

See docs/lidar2imu_design.md for parameters.
