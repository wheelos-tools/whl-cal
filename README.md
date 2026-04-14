# whl-cal

LiDAR extrinsic calibration tools for Apollo `.record` data.

Detailed documentation lives in [docs/lidar2lidar.md](docs/lidar2lidar.md).

## Install

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Set your Apollo record directory:

```bash
RECORD_DIR=../apollo-lite/data/bag/record_data
OUTPUT_DIR=outputs/lidar2lidar/auto_calib_review
```

Inspect topics:

```bash
lidar2lidar-topics "$RECORD_DIR"
```

Bootstrap fallback extrinsics from `/tf_static` and run automatic calibration:

```bash
lidar2lidar-auto \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --bootstrap-conf \
  --output-dir "$OUTPUT_DIR" \
  --sync-threshold-ms 10 \
  --min-overlap 0.30 \
  --methods 2 \
  --max-samples 1 \
  --save-merged-pcd
```

Results:

- Final consolidated extrinsics: `$OUTPUT_DIR/calibrated_tf.yaml`
- Example final extrinsics path: `outputs/lidar2lidar/auto_calib_review/calibrated_tf.yaml`
- Evaluation metrics: `$OUTPUT_DIR/metrics.yaml`
- Initial extrinsics snapshot: `$OUTPUT_DIR/initial_guess/*.yaml`
- Per-edge calibrated extrinsics: `$OUTPUT_DIR/calibrated/*.yaml`
- Refreshed fallback extrinsics: `lidar2lidar/conf/*.yaml`
- Detailed diagnostics: `$OUTPUT_DIR/diagnostics/`

Optional helpers:

```bash
lidar2lidar-extract --input-dir "$RECORD_DIR" --output-dir outputs/lidar2lidar/pcd_export -c /apollo/sensor/lslidar_main/PointCloud2
lidar2lidar-calibrate --source-pcd source.pcd --target-pcd target.pcd --initial-transform lidar2lidar/conf/lslidar_main_lslidar_left_extrinsics.yaml --output-transform result.yaml
lidar2lidar-merge --source-pcd source.pcd --target-pcd target.pcd --transform result.yaml --output-pcd merged_output.pcd
```
