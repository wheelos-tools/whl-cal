# whl-cal Detailed Docs

This document contains the detailed operating notes for the `lidar2lidar`
workflow. The top-level [README.md](../README.md) is intentionally kept short.

## Commands

- `lidar2lidar-topics`: inspect Apollo record topics
- `lidar2lidar-auto`: automatic multi-LiDAR calibration from Apollo record data
- `lidar2lidar-calibrate`: refine one source-target pair manually
- `lidar2lidar-extract`: export `PointCloud2` messages to PCD files
- `lidar2lidar-merge`: merge two PCD files with a transform
- `lidar2lidartool`: backward-compatible alias of `lidar2lidar-calibrate`

## Recommended Workflow

1. Run `lidar2lidar-topics` to confirm the available `PointCloud2` topics.
2. Run `lidar2lidar-auto --bootstrap-conf` to refresh fallback extrinsics under `lidar2lidar/conf`.
3. Run `lidar2lidar-auto` for the actual calibration output.
4. If needed, export PCD files with `lidar2lidar-extract` and refine one pair with `lidar2lidar-calibrate`.
5. Use `lidar2lidar-merge` only for visualization or manual inspection of the final transform.

## Standard Extrinsics Format

Fallback and calibrated extrinsics use one YAML schema:

```yaml
header:
  stamp:
    secs: 0
    nsecs: 0
  seq: 0
  frame_id: lslidar_main
transform:
  translation:
    x: 0.0
    y: 0.0
    z: 0.0
  rotation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
child_frame_id: lslidar_left
```

Naming rule:

- `lidar2lidar/conf/<parent_frame>_<child_frame>_extrinsics.yaml`

The transform maps points from `child_frame_id` into `header.frame_id`.

## Auto Pipeline Notes

The automatic pipeline does the following:

1. Read `PointCloud2`, `/tf`, and `/tf_static` from Apollo record files.
2. Merge record TF edges with fallback extrinsics from `lidar2lidar/conf`.
3. Prefer the unique `tf_static` root frame when selecting the base topic.
4. Match synchronized point clouds under `--sync-threshold-ms`.
5. Estimate overlap and calibrate only pairs above `--min-overlap`.
6. Export normalized outputs under `outputs/lidar2lidar/...`.

## Output Files

The main automatic run writes everything under `--output-dir`.

For example, if you run with `--output-dir outputs/lidar2lidar/auto_calib_review`,
the final consolidated extrinsics file is:

- `outputs/lidar2lidar/auto_calib_review/calibrated_tf.yaml`

The recommended production-facing outputs are:

- `calibrated_tf.yaml`: consolidated final extrinsics
- `metrics.yaml`: concise evaluation metrics and skipped-edge reasons
- `initial_guess/*.yaml`: normalized initial extrinsics snapshot
- `calibrated/*.yaml`: calibrated extrinsics in the same directory layout as `initial_guess`

Detailed debug artifacts are grouped under `diagnostics/`:

- `diagnostics/manifest.yaml`: run metadata and selected target topic
- `diagnostics/tf_tree.yaml`: TF edges and root-frame analysis
- `diagnostics/topology.yaml`: candidate pairs, overlap checks, selected edges
- `diagnostics/calibration.yaml`: full per-edge registration details
- `diagnostics/merged_cloud.pcd`: optional merged cloud for visualization

## Related Docs

- [lidar2lidar/conf/README.md](../lidar2lidar/conf/README.md): fallback extrinsics directory
- [tools/README.md](../tools/README.md): helper script boundaries

## Current Limitations

- No pose-graph or global bundle optimization yet.
- Dynamic `/tf` lookup is not yet time-aware per synchronized frame.
- Merged-cloud export uses a looser sync window for visualization than for calibration.
