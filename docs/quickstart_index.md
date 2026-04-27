# Quickstart index — whl-cal

This page provides quick entry points to the repository's calibration tools and shows where to find the immediate results.

Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quickstarts

- LiDAR-to-LiDAR (scan2scan / scan2map): docs/lidar2lidar_quickstart.md
- LiDAR-to-IMU (sample conversion + solver): docs/lidar2imu_quickstart.md
- Camera intrinsic: camera/README.md
- Camera↔LiDAR (target-based baseline + experimental targetless): docs/camera2lidar_quickstart.md

Common output artifacts (where to look first)

- calibrated_tf.yaml — consolidated extrinsics for the run
- metrics.yaml — coarse and fine evaluation metrics
- diagnostics/ — per-stage diagnostics and recommended inspection files
- initial_guess/*.yaml and calibrated/*.yaml — per-edge or per-pose extrinsics

Quick "what to check" (first pass)

1. Open metrics.yaml and read coarse_metrics (calibrated edges, average_fitness, min_overlap, max_condition_number).
2. Open diagnostics/extraction.yaml and diagnostics/topology.yaml to confirm extraction and why edges were skipped.
3. For LiDAR↔Camera runs, open reprojection summaries and overlay visualizations.
4. Use review helpers (lidar2imu-review-runs, lidar2lidar-scan2map candidate outputs) for multi-run comparison.

If an artifact looks suspicious, follow the module quickstart for the relevant package (links above) to dig into diagnostics and parameters.
