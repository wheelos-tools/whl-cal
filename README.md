# whl-cal

Calibration tools for Apollo `.record` data and related image / point-cloud workflows.

## Current status

| Module | Current status | Practical recommendation |
| --- | --- | --- |
| `lidar2lidar` | real-bag validated | keep `scan2scan` as production baseline; use `scan2map` as conditional refinement |
| `lidar2imu` | real-bag validated | use `--planar-motion-policy auto` on weak-planar bags; trust `z/roll/pitch` before weak `x/y/yaw` |
| `camera` | standalone intrinsic tool exists | usable as a local intrinsic calibrator |
| `camera2lidar` | scripts exist but not yet under repo-wide metrics framework | next target for repo-level cleanup |

## Knowledge base

The durable project knowledge base lives under [`context/`](context/).

Recommended entry points:

- [`context/index.md`](context/index.md)
- [`context/knowledge_base/calibration_overview.md`](context/knowledge_base/calibration_overview.md)
- [`context/knowledge_base/validated_conclusions.md`](context/knowledge_base/validated_conclusions.md)
- [`context/knowledge_base/verification_points.md`](context/knowledge_base/verification_points.md)

Documentation:

- LiDAR-to-LiDAR overview: [docs/lidar2lidar.md](docs/lidar2lidar.md)
- LiDAR-to-LiDAR Quick Start: [docs/lidar2lidar_quickstart.md](docs/lidar2lidar_quickstart.md)
- LiDAR-to-LiDAR current design: [docs/lidar2lidar_design.md](docs/lidar2lidar_design.md)
- LiDAR-to-IMU overview: [docs/lidar2imu.md](docs/lidar2imu.md)
- LiDAR-to-IMU Quick Start: [docs/lidar2imu_quickstart.md](docs/lidar2imu_quickstart.md)
- LiDAR-to-IMU current design: [docs/lidar2imu_design.md](docs/lidar2imu_design.md)
- Camera intrinsic quick start: [camera/README.md](camera/README.md)

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

Build the scan2map dataset artifact on the same bag:

```bash
lidar2lidar-scan2map-dataset \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/scan2map_dataset_review \
  --pose-topic /apollo/localization/pose
```

This produces:

- `diagnostics/scan2map_dataset.yaml`: aligned scans, keyframes, holdout split, and submap definitions
- `diagnostics/manifest.yaml`: extraction artifact manifest

Run the scan2map candidate calibration on that dataset artifact:

```bash
lidar2lidar-scan2map \
  --record-path "$RECORD_DIR" \
  --dataset-yaml outputs/lidar2lidar/scan2map_dataset_review/diagnostics/scan2map_dataset.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/scan2map_candidate_review \
  --scan2scan-baseline-tf outputs/lidar2lidar/auto_calib_review/calibrated_tf.yaml
```

This produces:

- `calibrated_tf.yaml`: accepted scan2map candidate extrinsics
- `metrics.yaml`: coarse and fine scan2map metrics
- `diagnostics/scan2map_optimization.yaml`: per-edge optimization and holdout diagnostics
- `diagnostics/evaluation.yaml`: summarized comparison outputs

For vehicle-rig verification, you can also lock selected transform components to the
scan2scan baseline while testing one edge. This is useful when a scan2map candidate
improves holdout fitness mainly through vertical-attitude drift:

```bash
lidar2lidar-scan2map   --record-path "$RECORD_DIR"   --dataset-yaml outputs/lidar2lidar/scan2map_dataset_review/diagnostics/scan2map_dataset.yaml   --conf-dir lidar2lidar/conf   --output-dir outputs/lidar2lidar/scan2map_candidate_review   --source-topics /apollo/sensor/lslidar_right/PointCloud2   --scan2scan-baseline-tf outputs/lidar2lidar/auto_calib_review/calibrated_tf.yaml   --constraint-reference scan2scan_baseline   --lock-components z pitch roll
```

Optional helpers:

```bash
lidar2lidar-extract --input-dir "$RECORD_DIR" --output-dir outputs/lidar2lidar/pcd_export -c /apollo/sensor/lslidar_main/PointCloud2
lidar2lidar-calibrate --source-pcd source.pcd --target-pcd target.pcd --initial-transform lidar2lidar/conf/lslidar_main_lslidar_left_extrinsics.yaml --output-transform result.yaml
lidar2lidar-merge --source-pcd source.pcd --target-pcd target.pcd --transform result.yaml --output-pcd merged_output.pcd
```

## LiDAR-to-IMU calibration

`lidar2imu-calibrate` consumes curated feature samples instead of raw records.
The implementation keeps **algorithm stages** and **evaluation metrics**
separate, so the solver can evolve without changing the reporting layout.

```bash
lidar2imu-calibrate \
  --input lidar2imu_samples.yaml \
  --planar-motion-policy auto \
  --output-dir outputs/lidar2imu/run01
```

Results:

- Final extrinsics: `outputs/lidar2imu/run01/calibrated/<parent>_<child>_extrinsics.yaml`
- Consolidated output: `outputs/lidar2imu/run01/calibrated_tf.yaml`
- Evaluation metrics: `outputs/lidar2imu/run01/metrics.yaml`
- Detailed diagnostics: `outputs/lidar2imu/run01/diagnostics/`

Documentation:

- Overview: [docs/lidar2imu.md](docs/lidar2imu.md)
- Quick Start: [docs/lidar2imu_quickstart.md](docs/lidar2imu_quickstart.md)
- Current design: [docs/lidar2imu_design.md](docs/lidar2imu_design.md)

To bootstrap those standardized samples from an Apollo record bag, use:

```bash
lidar2imu-convert-record \
  --record-path /path/to/record_dir \
  --output-dir outputs/lidar2imu/raw_validation \
  --min-registration-fitness 0.55 \
  --planar-motion-policy auto \
  --calibrate
```

If the bag does not contain a static LiDAR-to-parent TF, provide an explicit
prior with `--initial-transform path/to/extrinsics.yaml`, or use
`--identity-initial-transform` for exploratory-only runs.

This produces:

- `standardized_samples.yaml`: normalized ground and motion samples
- `conversion_diagnostics.yaml`: conversion-layer diagnostics
- `calibration/`: final extrinsics, metrics, and diagnostics from the staged solver,
  including motion registration quality, turn-balance warnings, and a
  `vehicle_motion_assessment` recommendation

For weak-planar bags, `--planar-motion-policy auto` keeps `x/y/yaw` near the
initial prior when turn balance or yaw observability is weak, while still letting
the solver refine `z/roll/pitch`.

## Current calibration summary

### lidar2lidar

- The repo-level conclusion is:
  - **baseline**: `scan2scan`
  - **refinement path**: `scan2map`
- For vehicle rigs, always judge:
  - planar: `x/y/yaw`
  - vertical-attitude: `z/pitch/roll`
- On the current tested bag:
  - `left -> main` scan2map can be accepted as a refinement candidate
  - `right -> main` unconstrained scan2map remains diagnostic

### lidar2imu

- The repo-level conclusion is:
  - pose-derived gravity stays the default
  - weak-planar bags should prefer `--planar-motion-policy auto`
  - acceptance must be driven by tested data, not only by solver convergence
- The current data-layer policy now uses:
  - **window + gate** motion selection
  - registration-fitness gating
  - weak-planar solver freeze when needed

## Next module

The next repo-level cleanup target is **lidar2camera / camera2lidar**.

The goal is to bring camera-related calibration into the same pattern already used
by `lidar2lidar` and `lidar2imu`:

1. data layer
2. algorithm layer
3. stable evaluation layer

That includes:

- explicit dataset artifacts
- window + gate for invalid samples
- stable metrics / diagnostics
- separating validated conclusions from open verification points
