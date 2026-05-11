---
audience: user
stability: stable
P26-04-27
---


# LiDAR-to-LiDAR Quick Start

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Quick run (automatic):
```bash
lidar2lidar-auto --record-path /path/to/record --conf-dir lidar2lidar/conf --output-dir outputs/lidar2lidar/run
```

Outputs: `calibrated_tf.yaml`, `metrics.yaml`, `diagnostics/`

Generic no-loop workflow driven by TF adjacency:
```bash
lidar2lidar-auto \
  --record-path /path/to/record \
  --workflow-yaml lidar2lidar/conf/workflow_tf_tree_example.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/run_tf_tree \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --max-samples 2
```

Prepare a reusable raw-only rig dataset first when the same bag will feed both
`lidar2lidar` and `lidar2imu`:
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

Prepared dataset outputs:

- `diagnostics/prepared_rig_dataset.yaml`
- `cache/pointclouds/**/*.pcd`
- `cache/state.npz`

Four-LiDAR rig run with explicit relations and loop refinement:
```bash
lidar2lidar-auto \
  --prepared-dataset-yaml outputs/prepared/rig_run/diagnostics/prepared_rig_dataset.yaml \
  --workflow-yaml lidar2lidar/conf/workflow_raw4_loop_example.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/rig_run \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --max-samples 2
```

Extra comparison outputs:

- workflow plan: `diagnostics/workflow.yaml`
- scene sufficiency: `diagnostics/scene_sufficiency.yaml`
- baseline star result: `calibrated_tf.yaml`
- loop-closed result: `loop_closed_tf.yaml`
- graph comparison: `diagnostics/loop_closure.yaml`
- wall / corner / slice summary: `diagnostics/visual_evaluation.yaml`
- colored sensor overlays:
  - `diagnostics/merged_cloud_baseline_colored.ply`
  - `diagnostics/merged_cloud_loop_closure_colored.ply`

See docs/lidar2lidar_design.md for tuning.
