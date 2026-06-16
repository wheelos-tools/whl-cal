---
audience: user
stability: stable
P26-06-16
---

# LiDAR-to-LiDAR quick start

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 1. 采集

Recommended bag contents:

- all LiDAR raw `PointCloud2` topics to be calibrated. For raw4 rigs:
  - `/apollo/sensor/vanjeelidar/left_back/PointCloud2`
  - `/apollo/sensor/vanjeelidar/left_front/PointCloud2`
  - `/apollo/sensor/vanjeelidar/right_front/PointCloud2`
  - `/apollo/sensor/vanjeelidar/right_back/PointCloud2`
- `/tf_static`
- optional `/tf`

Scene guidance:

- prefer walls, corners, poles, facades, and curb geometry
- avoid long feature-poor segments
- avoid bags dominated by moving traffic

Before calibration, inspect available topics:

```bash
lidar2lidar-topics /path/to/record_or_record_dir
```

## 2. 运行

For four-LiDAR perimeter-only topology (rectangle 4 edges, no diagonals in
solution), run with `workflow_raw4_perimeter_loop.yaml`. The initial
transforms are taken from `tf_static` (and fallback TF graph), and loop-closure
stage is enabled by `--loop-closure`.

```bash
lidar2lidar-auto \
  --record-path /path/to/record_or_record_dir \
  --workflow-yaml lidar2lidar/conf/workflow_raw4_perimeter_loop.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/raw4_loop_run \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --loop-closure \
  --save-merged-pcd
```

Reference run on this repository data:

```bash
lidar2lidar-auto \
  --record-path /mnt/synology/中集/2026-06-16 \
  --workflow-yaml lidar2lidar/conf/workflow_raw4_perimeter_loop.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/2026-06-16_raw4_perimeter_loop_sync40 \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --loop-closure \
  --save-merged-pcd
```

## 3. 评价

Start from the conclusion layer:

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2lidar/raw4_loop_run/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
PY
```

Then inspect the data-quality layer:

```bash
cat outputs/lidar2lidar/raw4_loop_run/diagnostics/data_quality.yaml
cat outputs/lidar2lidar/raw4_loop_run/diagnostics/status_summary.csv
```

Finally inspect visual evidence:

```bash
cat outputs/lidar2lidar/raw4_loop_run/diagnostics/visualization_index.yaml
```

Key output files:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/edge_metrics.csv`
- `diagnostics/skipped_edges.csv`
- `diagnostics/loop_closure.yaml`
- `diagnostics/merged_cloud_solution_colored.ply`
- `diagnostics/visualization_index.yaml`

If visual geometry is missing or `warning`, keep the run review-only even if the
optimizer converged.
