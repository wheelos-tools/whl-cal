# whl-cal

Calibration tools for Apollo `.record` data and related image / point-cloud
workflows.

## Current status

| Module | Current status | Practical recommendation |
| --- | --- | --- |
| `lidar2lidar` | real-bag validated | keep `scan2scan` as production baseline; use `scan2map` as conditional refinement |
| `gril` / LiDAR-to-IMU | sole LiDAR-to-IMU algorithm; research-only ROS-free source build | iterate on native GRIL only; its 0827 full A/B rotation discrepancy is `0.406964 deg` versus a `0.2 deg` gate, so production release is not yet approved ([release gate](docs/gril_ros_free_migration.md)) |
| `camera` | standalone intrinsic tool exists | usable as a local intrinsic calibrator |
| `camera2camera` | target-based stereo baseline exists | use the checkerboard paired-image pipeline as the current production baseline; add ChArUco next |
| `lidar2camera` | target-based industrial baseline exists | use the target-based pipeline as the current production baseline; keep targetless paths experimental |

## Recommended documentation path

If you are new to the repo, follow the docs in this order:

1. **Apollo-side prerequisites and recording**:
   [docs/apollo_data_collection.md](docs/apollo_data_collection.md)
2. **Module quick starts**:
   [docs/quickstart_index.md](docs/quickstart_index.md)
3. **How to review metrics and visualization**:
   [docs/calibration_review_guide.md](docs/calibration_review_guide.md)
4. **Advanced design / method / SOTA context**:
   [docs/calibration_methodology.md](docs/calibration_methodology.md)

## Which calibration needs what

| Module | Current tool input | Apollo-side raw data that should be recorded | Extra information you must know in advance |
| --- | --- | --- | --- |
| `camera` | live camera or exported image directory | camera image topic if you want to archive the session in Apollo; direct live capture is also supported | board pattern size, square size, fixed camera mode / exposure |
| `camera2camera` | paired image directories | two camera image topics if you want Apollo traceability; the current tool itself consumes exported image pairs | parent / child intrinsics, board pattern size, square size, multi-pose board plan |
| `lidar2camera` | paired `image + .pcd` files | camera image topic, LiDAR `PointCloud2`, `/tf_static`, optional `/tf` | camera intrinsics, distortion, checkerboard size, square size |
| `lidar2lidar` | Apollo `.record` or prepared dataset | all raw LiDAR `PointCloud2` topics, `/tf_static`, optional `/tf` | sensor topic list, approximate TF tree / initial extrinsics, scene plan |
| `gril` / LiDAR-to-IMU | Apollo `.record`, ROS1 bag, or canonical GRIL dataset | one raw LiDAR topic, raw IMU topic, `/tf_static`, optional independent GNSS/INS odometry for holdout | LiDAR topic, IMU topic, scan-line count, reviewed sensor configuration |

## Install

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The base package installs only the shared numerical and YAML dependencies. Install
the extra matching the calibration workflow you need:

```bash
pip install -e ".[camera]"
pip install -e ".[camera2camera]"
pip install -e ".[lidar2lidar]"
pip install -e ".[lidar2imu]"
pip install -e ".[lidar2camera]"
```

To install every calibration workflow in one environment, combine all extras:

```bash
pip install -e ".[all]"
```

If the default PyPI route is slow or unstable in this environment, switch pip to
the Tsinghua mirror inside the active virtual environment first:

```bash
python -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e ".[all]"
```

## Common workflow

### 1. Collect a bag in Apollo

See [docs/apollo_data_collection.md](docs/apollo_data_collection.md) for the full
checklist. The short version is:

```bash
# examples from /home/humble/01code/apollo-base
bash scripts/transform.sh start
bash scripts/gps.sh start
bash scripts/localization.sh start
bash scripts/camera.sh start
cyber_launch start /apollo/modules/drivers/lidar/launch/lidar_with_fusion_and_compensator.launch

# record all channels from the target output directory
cyber_recorder record -a -i 60 -m 2048
```

If you prefer Apollo's wrapper script, `scripts/record_bag.sh start` in
`apollo-base` launches the same recorder command.

### 2. Inspect the bag quickly

```bash
RECORD_DIR=/path/to/record_dir
lidar2lidar-topics "$RECORD_DIR"
```

### 3. Run the module you need

- LiDAR-to-LiDAR:
  [docs/lidar2lidar_quickstart.md](docs/lidar2lidar_quickstart.md)
- LiDAR-to-IMU:
  [docs/lidar2imu_quickstart.md](docs/lidar2imu_quickstart.md)
- GRIL ROS-free migration:
  [docs/gril_ros_free_migration.md](docs/gril_ros_free_migration.md)
- Camera intrinsic:
  [docs/camera_quickstart.md](docs/camera_quickstart.md)
- Camera-to-camera:
  [docs/camera2camera_quickstart.md](docs/camera2camera_quickstart.md)
- LiDAR↔Camera:
  [docs/lidar2camera_quickstart.md](docs/lidar2camera_quickstart.md)
- LiDAR↔Camera nuScenes benchmark:
  [docs/lidar2camera_nuscenes_benchmark.md](docs/lidar2camera_nuscenes_benchmark.md)

### 4. Review results in the same order every time

Across modules, the stable review order is:

1. `diagnostics/standardized_data.yaml`
2. `diagnostics/data_quality.yaml`
3. `metrics.yaml`
4. `diagnostics/acceptance_report.yaml`
5. `diagnostics/visualization_index.yaml`

Use [docs/calibration_review_guide.md](docs/calibration_review_guide.md) for the
module-specific thresholds and visualization files.

## High-value example commands

### LiDAR-to-LiDAR

```bash
lidar2lidar-auto \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/auto_calib_review \
  --sync-threshold-ms 10 \
  --min-overlap 0.30 \
  --methods 2 \
  --max-samples 1
```

For a four-LiDAR raw rig, prepare a reusable raw-only dataset first:

```bash
lidar2lidar-rig-dataset \
  --record-path "$RECORD_DIR" \
  --output-dir outputs/prepared/run-eight-raw4 \
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

### LiDAR-to-IMU

```bash
gril-migrate run-native \
  --input "$RECORD_DIR" \
  --input-type record \
  --config .agents/skills/gril-calib-validation/resources/vanjeelidar16.yaml \
  --executable third_party/gril_native/build/gril_native_full_frontend \
  --output-dir outputs/gril/run01 \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16
```

### Camera-to-camera

```bash
camera2camera-calibrate --write-default-config --config camera2camera_config.yaml
camera2camera-calibrate --config camera2camera_config.yaml
```

### LiDAR↔Camera

```bash
lidar2camera-calibrate --write-default-config --config config.yaml
lidar2camera-calibrate --config config.yaml
```

### LiDAR↔Camera nuScenes benchmark

```bash
lidar2camera-nuscenes-benchmark \
  --info-path /mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/nuscenes_infos_val.pkl \
  --camera-names CAM_FRONT \
  --sample-limit 8
```

### Camera intrinsic

```bash
camera-intrinsic-calibrate --config camera_config.yaml
```

### Containers

Camera intrinsic and ROS-free GRIL use separate images and the same read-only
input / writable output mount convention. See
[docker/README.md](docker/README.md). Live camera capture requires a GUI;
headless camera execution is supported only for offline image datasets.

## Common output artifacts

The repo keeps the final review surface stable on purpose. Depending on the
module, look for:

- `customer_summary.yaml` for the concise customer verdict and key metrics
- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`
- `initial_guess/*.yaml`
- `calibrated/*.yaml`

For `camera`, the equivalent outputs live under
`outputs/camera_intrinsic/runs/<session>/calibration_diagnostics/`, while live
accepted samples are archived under
`outputs/camera_intrinsic/captures/<session>/accepted/`.

Detailed metric layering is documented in
[docs/calibration_metric_layers.md](docs/calibration_metric_layers.md).

## Knowledge base and deeper docs

The durable project knowledge base lives under [`context/`](context/).

Recommended entry points:

- [`context/index.md`](context/index.md)
- [`context/knowledge_base/calibration_overview.md`](context/knowledge_base/calibration_overview.md)
- [`context/knowledge_base/validated_conclusions.md`](context/knowledge_base/validated_conclusions.md)
- [`context/knowledge_base/verification_points.md`](context/knowledge_base/verification_points.md)

Module docs:

- LiDAR-to-LiDAR overview: [docs/lidar2lidar.md](docs/lidar2lidar.md)
- LiDAR-to-LiDAR Quick Start: [docs/lidar2lidar_quickstart.md](docs/lidar2lidar_quickstart.md)
- LiDAR-to-LiDAR current design: [docs/lidar2lidar_design.md](docs/lidar2lidar_design.md)
- LiDAR-to-IMU quick start: [docs/lidar2imu_quickstart.md](docs/lidar2imu_quickstart.md)
- LiDAR-to-IMU Quick Start: [docs/lidar2imu_quickstart.md](docs/lidar2imu_quickstart.md)
- Camera-to-camera Quick Start: [docs/camera2camera_quickstart.md](docs/camera2camera_quickstart.md)
- Camera-to-camera design: [docs/camera2camera_design.md](docs/camera2camera_design.md)
- ROS-free GRIL design and release gate: [docs/gril_ros_free_migration.md](docs/gril_ros_free_migration.md)
- Camera intrinsic quick start: [docs/camera_quickstart.md](docs/camera_quickstart.md)
- LiDAR↔Camera Quick Start: [docs/lidar2camera_quickstart.md](docs/lidar2camera_quickstart.md)
- LiDAR↔Camera current design: [docs/lidar2camera_design.md](docs/lidar2camera_design.md)
