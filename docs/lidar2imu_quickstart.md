---
audience: user
stability: stable
P26-05-25
---

# LiDAR-to-IMU quick start

Before running this tool:

1. collect the Apollo bag with
   [docs/apollo_data_collection.md](apollo_data_collection.md)
2. confirm the bag contains LiDAR, pose, IMU, and TF channels
3. review results with
   [docs/calibration_review_guide.md](calibration_review_guide.md)
4. use [docs/lidar2imu_design.md](lidar2imu_design.md) and
   [docs/calibration_methodology.md](calibration_methodology.md) for tuning

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| Apollo `.record` directory or file | yes | or a prepared dataset manifest / standardized samples |
| LiDAR topic | yes | one LiDAR topic to calibrate against IMU |
| pose topic | yes | usually `/apollo/localization/pose` |
| IMU topic | yes | default pipeline expects `/apollo/sensor/gnss/imu` |
| `lidar -> imu` initial transform | recommended | required if the bag does not contain the static TF |

Motion guidance:

- include both left and right turns when possible
- include acceleration and braking, not only cruising
- include flat-road segments so ground extraction remains reliable

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick run

```bash
lidar2imu-calibrate \
  --input lidar2imu_samples.yaml \
  --output-dir outputs/lidar2imu/run01
```

## Outputs follow the shared calibration paradigm

1. **Data**: normalized samples, metadata, and data quality
2. **Algorithm**: staged solver result
3. **Evaluation**: conclusion, detailed metrics, and visual-review tables

Core outputs:

- `standardized_samples.yaml` when using `lidar2imu-convert-record`
- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/manifest.yaml`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`
- `diagnostics/ground_residuals.csv`
- `diagnostics/motion_residuals.csv`
- `diagnostics/holdout_motion_residuals.csv`
- `diagnostics/review_report.html`
- `diagnostics/ground_residuals_plot.svg`
- `diagnostics/motion_residuals_plot.svg`
- `diagnostics/trajectory_overlay.svg`
- `diagnostics/trajectory_position_gap_plot.svg`
- `diagnostics/imu_trajectory_cloud.ply`
- `diagnostics/lidar_trajectory_cloud.ply`
- `diagnostics/trajectory_overlay_cloud.ply`
- `diagnostics/holdout_motion_residuals_plot.svg`
- `diagnostics/yaw_cost_scan.svg`

## Recommended Apollo bag contents

- LiDAR `PointCloud2`
- `/apollo/localization/pose`
- `/apollo/sensor/gnss/imu`
- `/tf_static`
- optional `/tf`

For traceability and localization auditing, it is usually better to keep all
GNSS / IMU related channels by recording with `cyber_recorder record -a`.

## Convert directly from an Apollo record

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --output-dir outputs/lidar2imu/run01 \
  --profile baseline \
  --calibrate
```

If the bag uses different topics, specify them explicitly:

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --output-dir outputs/lidar2imu/run01 \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/imu \
  --profile baseline \
  --calibrate
```

## Reuse a shared raw-LiDAR prepared dataset

When one bag needs to feed both `lidar2imu` and `lidar2lidar`, first prepare the
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

## How to judge a result

Start from the conclusion layer:

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2imu/rig_left_front/calibration/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
print(d["vehicle_motion_assessment"]["recommendation"])
PY
```

Then inspect data quality:

```bash
cat outputs/lidar2imu/rig_left_front/calibration/diagnostics/data_quality.yaml
cat outputs/lidar2imu/rig_left_front/calibration/diagnostics/status_summary.csv
```

Finally inspect visual-review tables:

```bash
cat outputs/lidar2imu/rig_left_front/calibration/diagnostics/visualization_index.yaml
```

For the fastest visual review, open:

```bash
xdg-open outputs/lidar2imu/rig_left_front/calibration/diagnostics/review_report.html
```

Plot or review:

- `ground_residuals.csv`: `normal_angle_deg`, `height_residual_m`
- `motion_residuals.csv`: `rotation_residual_deg`, `translation_residual_m`,
  `registration_fitness`, `sync_dt_ms`
- `trajectory_overlay.svg`: relative IMU vs LiDAR odometry overlay from the
  selected motion factors
- `trajectory_position_gap_plot.svg`: cumulative position disagreement between
  the IMU motion chain and the LiDAR motion chain
- `trajectory_overlay_cloud.ply`: stitched selected keyframes using both
  trajectories; open in CloudCompare or Open3D for 3D geometry review
- `observability.yaml`: yaw cost-scan flatness and plateau width

For production full-6DoF release, require:

- `release_ready: true`
- `vehicle_motion_assessment.recommendation: full_6dof_candidate`
- balanced left/right turns
- acceptable motion registration
- acceptable holdout / repeatability when enabled
- no trusted-reference or extraction-geometry conflict

If the solver applies `freeze_xyyaw`, the run can still be useful for
`z/roll/pitch`, but it is not a full-6DoF production result.
