---
audience: user
stability: stable
P26-05-25
---

# Quickstart index — whl-cal

Use this page as the navigation hub for the repo's calibration workflows.

## Recommended reading order

1. Apollo preparation and bag recording:
   [docs/apollo_data_collection.md](apollo_data_collection.md)
2. Module quickstart from the list below
3. Metrics / visualization review:
   [docs/calibration_review_guide.md](calibration_review_guide.md)
4. Advanced design / method / SOTA context:
   [docs/calibration_methodology.md](calibration_methodology.md)

## Quickstarts

- LiDAR-to-LiDAR (scan2scan / scan2map):
  [docs/lidar2lidar_quickstart.md](lidar2lidar_quickstart.md)
- LiDAR-to-IMU (record conversion + staged solver):
  [docs/lidar2imu_quickstart.md](lidar2imu_quickstart.md)
- Camera intrinsic:
  [docs/camera_quickstart.md](camera_quickstart.md)
- LiDAR↔Camera (target-based baseline):
  [docs/lidar2camera_quickstart.md](lidar2camera_quickstart.md)

## Which quickstart should I open?

| Need | Open this doc first |
| --- | --- |
| I have an Apollo bag with multiple LiDARs and need inter-LiDAR extrinsics | `docs/lidar2lidar_quickstart.md` |
| I have an Apollo bag and need LiDAR↔IMU extrinsics | `docs/lidar2imu_quickstart.md` |
| I need camera intrinsic calibration from live capture or exported images | `docs/camera_quickstart.md` |
| I need LiDAR↔Camera extrinsics from paired image / PCD files | `docs/lidar2camera_quickstart.md` |

## Common output artifacts

Most calibration runs expose the same top-level review surface:

- `calibrated_tf.yaml` — consolidated extrinsics
- `metrics.yaml` — main quantitative result
- `diagnostics/standardized_data.yaml` — normalized input summary
- `diagnostics/data_quality.yaml` — quality / gating summary
- `diagnostics/acceptance_report.yaml` — release-review gate results
- `diagnostics/status_summary.csv` — tabular review summary
- `diagnostics/visualization_index.yaml` — where to find visual / CSV evidence

For `camera`, the equivalent files live under
`calibration_YYYYmmdd_HHMMSS_diagnostics/`.

## Fast first-pass review

1. Open the module quickstart and run the shortest meaningful command.
2. Open `standardized_data.yaml` and `data_quality.yaml`.
3. Read `metrics.yaml` and `acceptance_report.yaml`.
4. Open the visualization files listed in `visualization_index.yaml`.

Use [docs/calibration_review_guide.md](calibration_review_guide.md) for the
module-specific thresholds and visual review checklist.
