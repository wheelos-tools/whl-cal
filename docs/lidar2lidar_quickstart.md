---
audience: user
stability: stable
P26-05-25
---

# LiDAR-to-LiDAR quick start

Before running this tool:

1. collect the Apollo bag with
   [docs/apollo_data_collection.md](apollo_data_collection.md)
2. inspect the bag with `lidar2lidar-topics`
3. review outputs with
   [docs/calibration_review_guide.md](calibration_review_guide.md)
4. use [docs/lidar2lidar_design.md](lidar2lidar_design.md) and
   [docs/calibration_methodology.md](calibration_methodology.md) for tuning

## What this tool needs

| Item | Required | Notes |
| --- | --- | --- |
| Apollo `.record` directory or file | yes | or a prepared dataset manifest |
| LiDAR topic set | yes | one target LiDAR plus one or more source LiDARs |
| `/tf_static` | strongly recommended | used for initial graph / fallback extrinsics |
| initial extrinsic priors | recommended | can come from TF or `lidar2lidar/conf/*.yaml` |
| static scene geometry | yes | walls, corners, facades, poles are ideal |

For four-LiDAR rigs, prefer the rectangle perimeter edges as primary constraints
and use diagonals as consistency checks.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick run

```bash
lidar2lidar-auto \
  --record-path /path/to/record \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/run
```

## Outputs follow the shared calibration paradigm

1. **Data**: normalized run metadata and data quality
2. **Algorithm**: calibrated transforms and solver diagnostics
3. **Evaluation**: conclusion, detailed metrics, and visual review artifacts

Core outputs:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/manifest.yaml`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`

## Apollo bag checklist

Recommended bag contents:

- all LiDAR raw `PointCloud2` topics to be calibrated
- `/tf_static`
- optional `/tf`

Scene guidance:

- prefer walls, corners, poles, facades, and curb geometry
- avoid long feature-poor segments
- avoid bags dominated by moving traffic

## Generic no-loop workflow driven by TF adjacency

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

## Prepare a reusable raw-only rig dataset first

When one bag will feed both `lidar2lidar` and `lidar2imu`:

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

## Four-LiDAR rig run with explicit relations and loop refinement

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

- `diagnostics/workflow.yaml`
- `diagnostics/scene_sufficiency.yaml`
- `calibrated_tf.yaml`
- `loop_closed_tf.yaml`
- `diagnostics/loop_closure.yaml`
- `diagnostics/visual_evaluation.yaml`
- `diagnostics/merged_cloud_baseline_colored.ply`
- `diagnostics/merged_cloud_loop_closure_colored.ply`
- `diagnostics/edge_metrics.csv`
- `diagnostics/skipped_edges.csv`

## How to judge a result

Start from the conclusion layer:

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2lidar/rig_run/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
PY
```

Then inspect the data-quality layer:

```bash
cat outputs/lidar2lidar/rig_run/diagnostics/data_quality.yaml
cat outputs/lidar2lidar/rig_run/diagnostics/status_summary.csv
```

Finally inspect visual evidence:

```bash
cat outputs/lidar2lidar/rig_run/diagnostics/visualization_index.yaml
```

For production release, require:

- `release_ready: true`
- all required gates in `diagnostics/acceptance_report.yaml` are `pass`
- required relations are connected
- scene sufficiency and repeatability are `pass`
- visual overlays do not show wall color fringing, double edges, or corner spread

If visual geometry is missing or `warning`, keep the run review-only even if the
optimizer converged.
