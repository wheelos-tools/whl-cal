---
audience: dev
stability: stable
P26-04-27
---


# lidar2camera — smoke test and simplified YAML usage

This page shows a minimal YAML config and a lightweight smoke-test to verify the
lidar2camera reference pipeline on synthetic data. The smoke test is intended to
be runnable without any recorded inputs and to demonstrate the metric surface.

1) Minimal YAML config example (for real data)

```yaml
camera:
  intrinsics: [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
  distortion: [0,0,0,0,0]
checkerboard:
  pattern_size: [4, 3]
  square_size: 0.05
point_cloud:
  plane_dist_thresh: 0.02
  min_plane_points: 50
optimization:
  min_poses: 3
  loss: huber
  max_nfev: 200
output:
  directory: outputs/lidar2camera/smoke_test
```

Write this YAML to `smoke_config.yaml`, edit `data_directory` to your paired
image/pcd directory (if using real data), then run:

```bash
# writes default if missing
lidar2camera-calibrate --write-default-config --config smoke_config.yaml
# run calibration
lidar2camera-calibrate --config smoke_config.yaml
```

2) Lightweight synthetic smoke test (no dataset required)

A quick script generates a small synthetic board-pose dataset, runs the
optimizer, and prints simple delta metrics. It is useful for sanity-checking the
runtime environment and the optimizer.

Run it from the repo root so imports resolve correctly:

```bash
# from repo root
PYTHONPATH=. python3 tools/run_lidar2camera_smoke.py --poses 5
```

Sample output (successful run):

[INFO] Running optimizer...
[RESULT] translation_norm_m= 0.0
[RESULT] rotation_deg= 0.00000
[PASS] Smoke test passed — recovered transform is close to ground truth.

3) Notes

- Prefer YAML config for real runs: it keeps the CLI invocation simple and
  documents the parameters used for each run.
- The smoke test uses SciPy. If SciPy is not available the script will exit with
  a short message — install SciPy to use the full smoke runner.
- After a real run, check `metrics.yaml` and `diagnostics/` under the configured
  output directory. Key fields: `final_rms_px`, `leave_one_out_repeatability`,
  and `coarse_metrics`.
