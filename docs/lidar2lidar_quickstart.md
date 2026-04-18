# LiDAR-to-LiDAR Quick Start

This document is only for running the current pipeline quickly.

## 1. Install

```bash
cd /home/wfh/01code/whl-cal
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Inspect record topics

```bash
RECORD_DIR=../apollo-lite/data/bag/record_data
lidar2lidar-topics "$RECORD_DIR"
```

## 3. Run the automatic pipeline

```bash
OUTPUT_DIR=outputs/lidar2lidar/auto_calib_review

lidar2lidar-auto \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --bootstrap-conf \
  --output-dir "$OUTPUT_DIR" \
  --sync-threshold-ms 10 \
  --min-overlap 0.30 \
  --min-fitness 0.0 \
  --max-condition-number 1000000 \
  --methods 2 \
  --max-samples 1 \
  --save-merged-pcd
```

Outputs:

- `$OUTPUT_DIR/calibrated_tf.yaml`
- `$OUTPUT_DIR/metrics.yaml`
- `$OUTPUT_DIR/initial_guess/*.yaml`
- `$OUTPUT_DIR/calibrated/*.yaml`
- `$OUTPUT_DIR/diagnostics/`

The most useful diagnostics are now:

- `$OUTPUT_DIR/diagnostics/extraction.yaml`
- `$OUTPUT_DIR/diagnostics/topology.yaml`
- `$OUTPUT_DIR/diagnostics/calibration.yaml`
- `$OUTPUT_DIR/diagnostics/tf_tree.yaml`

Use stricter gates when you only want high-confidence edges:

```bash
lidar2lidar-auto \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --bootstrap-conf \
  --output-dir outputs/lidar2lidar/auto_calib_gated \
  --sync-threshold-ms 10 \
  --min-overlap 0.30 \
  --min-fitness 0.98 \
  --max-condition-number 1500 \
  --methods 2 \
  --max-samples 1
```

## 4. Optional helpers

Run the temporal hand-eye comparison pipeline:

```bash
lidar2lidar-temporal \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --bootstrap-conf \
  --output-dir outputs/lidar2lidar/temporal_compare \
  --sync-threshold-ms 50 \
  --min-overlap 0.30 \
  --methods 2 \
  --max-samples 1 \
  --temporal-window-strides 4 8 12 16 \
  --max-temporal-windows 8 \
  --max-motion-pairs 6 \
  --min-motion-pairs 3 \
  --comparison-samples 2
```

This writes:

- `metrics.yaml` with temporal-vs-pairwise comparison
- `diagnostics/pairwise_baseline.yaml`
- `diagnostics/temporal_dataset.yaml`
- `diagnostics/temporal_calibration.yaml`

Current note: on `record_data_0402`, the temporal branch is useful as an evaluation surface, but its quality gate still rejects the final hand-eye solution because the accepted same-sensor motions are too weak.

## 5. Optional helpers

Export one topic to PCD:

```bash
lidar2lidar-extract \
  --input-dir "$RECORD_DIR" \
  --output-dir outputs/lidar2lidar/pcd_export \
  -c /apollo/sensor/lslidar_main/PointCloud2
```

Refine one pair manually:

```bash
lidar2lidar-calibrate \
  --source-pcd source.pcd \
  --target-pcd target.pcd \
  --initial-transform lidar2lidar/conf/lslidar_main_lslidar_left_extrinsics.yaml \
  --output-transform result.yaml
```

Merge for visualization:

```bash
lidar2lidar-merge \
  --source-pcd source.pcd \
  --target-pcd target.pcd \
  --transform result.yaml \
  --output-pcd merged_output.pcd
```

## 6. Check these outputs first

Open:

- `metrics.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/topology.yaml`
- `diagnostics/calibration.yaml`
- `diagnostics/tf_tree.yaml`

Start with:

- `coarse_metrics.calibrated_edges`
- `coarse_metrics.average_fitness`
- `coarse_metrics.average_inlier_rmse`
- `coarse_metrics.min_overlap_ratio`
- `coarse_metrics.max_condition_number`
- `coarse_metrics.statuses`
- `skipped_edges`

## 7. Current interpretation rule

- If overlap pre-checks are weak, fix data selection before tuning registration.
- If `information_matrix.degenerate` is true or condition numbers are large,
  treat the edge as weakly constrained even if fitness looks acceptable.
- Use `topology.yaml` to understand why an edge was skipped before changing ICP
  parameters.
- For temporal hand-eye, first inspect `diagnostics/temporal_calibration.yaml` and
  check whether motion pairs were rejected because they were low-excitation or
  stagnant same-sensor solutions.
