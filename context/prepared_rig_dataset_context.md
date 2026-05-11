---
audience: dev
stability: experimental
last_tested: 2026-05-11
---

# prepared rig dataset context

## Goal

For the `run-eight` vehicle bag, the reusable extraction surface should be built
from the four **raw** LiDARs only:

- `/apollo/sensor/vanjeelidar/left_front/PointCloud2`
- `/apollo/sensor/vanjeelidar/right_front/PointCloud2`
- `/apollo/sensor/vanjeelidar/right_back/PointCloud2`
- `/apollo/sensor/vanjeelidar/left_back/PointCloud2`

Fusion / perception topics are not the calibration source of truth for this rig.

## Implemented shared artifact

`lidar2lidar-rig-dataset` now writes:

- `diagnostics/prepared_rig_dataset.yaml`
- `cache/pointclouds/**/*.pcd`
- `cache/state.npz`

The artifact stores:

1. raw-LiDAR-only topic list
2. synchronized snapshot metadata
3. Open3D-readable cached point clouds
4. cached pose / IMU state
5. cached TF edges

`lidar2lidar-auto` and `lidar2imu-convert-record` can now both consume
`--prepared-dataset-yaml`.

## run-eight sync finding

On `run-eight`, a four-way synchronized snapshot threshold of `20 ms` is too
tight for the raw rig because:

- `left_front -> right_front` nearest timestamp gap is about `32-33 ms`
- `left_front -> left_back` is usually about `18-20 ms`
- `left_front -> right_back` is usually about `1-8 ms`

For this bag, `40 ms` is the practical four-way snapshot threshold.

This is different from the pairwise `lidar2lidar-auto` case, where a lower
threshold can still work if only one pair is synchronized at a time.

## validated subset run

Prepared dataset built on:

- `20260507092334.record.00000`

Command:

```bash
lidar2lidar-rig-dataset \
  --record-path <subset_dir> \
  --output-dir outputs/prepared/run-eight-subset-00000-raw4-dataset \
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

Outputs:

- `90` synchronized snapshots
- `1819` pose samples
- `1814` IMU samples
- effective raw-LiDAR cache rate about `5.0 Hz` per topic from an original `~10 Hz`

## lidar2imu validation on prepared dataset

Run:

```bash
lidar2imu-convert-record \
  --prepared-dataset-yaml outputs/prepared/run-eight-subset-00000-raw4-dataset/diagnostics/prepared_rig_dataset.yaml \
  --lidar-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --output-dir outputs/lidar2imu/prepared-subset-left-front \
  --profile baseline \
  --motion-frame-stride 1 \
  --calibrate
```

Observed result:

- final `imu -> left_front` remained near the trusted prior
- solver still protected weak planar DOFs
- recommendation remained `reextract_review`
- the bag still supports `z/roll/pitch` diagnosis better than free `x/y/yaw`

## lidar2lidar validation on prepared dataset

Run:

```bash
lidar2lidar-auto \
  --prepared-dataset-yaml outputs/prepared/run-eight-subset-00000-raw4-dataset/diagnostics/prepared_rig_dataset.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/prepared-subset-raw4-loop \
  --target-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --source-topics \
    /apollo/sensor/vanjeelidar/right_front/PointCloud2 \
    /apollo/sensor/vanjeelidar/right_back/PointCloud2 \
    /apollo/sensor/vanjeelidar/left_back/PointCloud2 \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --max-samples 1 \
  --save-merged-pcd \
  --loop-closure
```

Observed result on the prepared subset:

- pairwise baseline average fitness: about `0.0666`
- loop-closure translation residual p95: `4.69 m -> 2.53 m`
- loop-closure rotation residual p95: `9.70 deg -> 5.42 deg`
- wall signed-span mean: `0.109 m -> 0.100 m`
- sensor wall-offset spread mean: `0.0165 m -> 0.0401 m`

Interpretation:

- prepared dataset reuse works mechanically
- loop closure improved graph consistency and one wall-thickness proxy on this subset
- sensor-to-sensor wall spread still worsened, so visual acceptance must still
  inspect the colored merged clouds instead of relying on one scalar metric
