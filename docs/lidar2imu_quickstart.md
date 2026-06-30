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
- `diagnostics/cloud_thickness_window_frames.csv`
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

For large bags, you can also switch the record backend to a dedicated
`pycyber` environment:

```bash
export WHL_CAL_RECORD_BACKEND=pycyber
```

That keeps the record reader on the `pycyber` path while the converter still
uses a single-pass bundle scan and the prepared-dataset cache to avoid repeated
rescans.

The converter canonicalizes topic timestamps before matching: LiDAR uses
`measurement_time` when present, pose and IMU use `measurement_time` first,
and GNSS best_pose / heading converts `measurement_time` from GPS epoch to
Unix time.

If you want the temporal candidate, add `--estimate-pose-time-offset`.
To compare estimators explicitly, use:

```bash
--pose-time-offset-estimator nearest_median
# or
--pose-time-offset-estimator xcorr_angular_speed
```

Otherwise keep the timestamp-normalized baseline.

To compare candidate staged solvers from standardized samples, add one of:

```bash
--solver-family gril_staged
# or
--solver-family gril_prob
# or
--solver-family gril_prob_nhc
```

`baseline` remains the default.

- `gril_staged`: screened staged candidate
- `gril_prob`: `gril_staged` + information-weighted motion residuals
- `gril_prob_nhc`: `gril_prob` + weak-motion NHC prior gating

For the current stronger front-end candidate, use:

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --output-dir outputs/lidar2imu/run01 \
  --profile production \
  --calibrate
```

That keeps the staged solver unchanged but switches motion extraction to the
larger `submap_to_map` path with the dense local scan-to-map submap builder.

To test candidate staged solvers on the same front-end, append for example:

```bash
--solver-family gril_staged
```

For probabilistic candidates, swap to `gril_prob` or `gril_prob_nhc`.

To enable extraction-stage GRIL-style observability screening explicitly:

```bash
--motion-observability-screening gril_fisher \
--motion-observability-window-sec 10 \
--motion-observability-min-window-sec 6
```

This runs a sliding Fisher-information screen in conversion, greedily merges
segments by **incremental total information-matrix capacity**
(`H_total <- H_total + H_i`), and then applies the existing global-diversity
selector on that screened pool.

If needed, tune the hard gates explicitly:

```bash
--motion-observability-min-rotation-lambda 1e-4 \
--motion-observability-min-planar-lambda 1e-3 \
--motion-observability-max-condition-number 2000 \
--motion-observability-max-merged-segments 2
```

To test the new IMU preintegration translation residual in solver stages:

```bash
--solver-family gril_prob \
--imu-preintegration-translation-weight 0.6 \
--imu-preintegration-translation-scale-m 0.08
```

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
- `cloud_thickness_window_frames.csv`: loaded/failed frame records for the
  holdout 5s stitched-cloud thickness check
- `trajectory_overlay.svg`: BEV review trajectory built from window-best
  registered motion candidates across the sequence; nodes come from a dense
  raw-scan review chain around each selected window
- `trajectory_position_gap_plot.svg`: BEV position disagreement between the IMU
  review chain and the LiDAR review chain
- `registration_review.yaml` / `registration_review.csv`: per-window overlap
  ratios, nearest-neighbor tails, registration-object builder modes, and dense
  local-map refinement stats for the actual registered objects
- `trajectory_overlay_cloud.ply`: registered-object overlay where gray is the
  target geometry, blue is the calibrated IMU-predicted source geometry, and
  red is the LiDAR-registered source geometry; each review scan is seeded from
  sparse anchor-relative poses and refined against a growing dense local map;
  open in CloudCompare
  or Open3D for direct ghosting review
- `observability.yaml`: yaw cost-scan flatness and plateau width

Physical gate:

- `metrics.yaml.coarse_metrics.cloud_thickness_holdout_p95_m`
  - `< 0.03 m`: production-grade geometric consistency
  - `> 0.05 m`: translation quality warning (review-only)

For production full-6DoF release, require:

- `summary.release_ready: true`
- `summary.final_acceptance_status: pass`
- `summary.required_gate_statuses.fisher_min_eigenvalue: pass`
- `summary.required_gate_statuses.fisher_conditioning: pass`
- `summary.required_gate_statuses.holdout_cloud_thickness: pass`

If the solver applies `freeze_xyyaw`, the run can still be useful for
`z/roll/pitch`, but it is not a full-6DoF production result.
