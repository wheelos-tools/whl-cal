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

Outputs follow the shared calibration paradigm:

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

Plot or review:

- `ground_residuals.csv`: `normal_angle_deg`, `height_residual_m`
- `motion_residuals.csv`: `rotation_residual_deg`, `translation_residual_m`,
  `registration_fitness`, `sync_dt_ms`
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
