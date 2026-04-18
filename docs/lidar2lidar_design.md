# LiDAR-to-LiDAR current design

This document records the current implementation for future iteration work.

## 1. Design goal

Treat `lidar2lidar` as three separable layers:

1. **Data extraction**: discover topics, TF edges, synchronized frames, and
   overlap candidates from raw record data
2. **Algorithm**: choose a target topic, select source edges, and refine
   transforms with ICP / GICP
3. **Evaluation**: keep concise metrics and rich diagnostics stable across
   iteration

This matches the current preferred calibration pattern in the repo.

## 2. Implemented commands

### `lidar2lidar-auto`

Main automatic pipeline from raw Apollo record data.

Main file:

- `lidar2lidar/auto_calib.py`

### `lidar2lidar-calibrate`

Manual single-pair refinement from PCD files.

Main file:

- `lidar2lidar/cli.py`

### Record helpers

Record parsing and TF helpers live in:

- `lidar2lidar/record_utils.py`

## 3. Data extraction layer

The current automatic pipeline already includes a raw-data conversion layer.

### What it extracts

- record file list
- point cloud topics and counts
- point cloud `frame_id`s
- `/tf` and `/tf_static` transform edges
- synchronized point cloud pairs
- initial source-to-target transforms from the TF graph
- voxel overlap probes for candidate pair scoring

### Main helper functions

- `discover_record_files`
- `list_topics`
- `infer_pointcloud_topics`
- `get_topic_frame_ids`
- `collect_pointcloud_metadata`
- `extract_tf_edges`
- `build_transform_graph`
- `lookup_transform`
- `find_synchronized_pairs`
- `voxel_overlap_ratio`

### Current extraction logic

1. Read raw record files.
2. Identify all `PointCloud2` topics.
3. Merge record-derived TF edges with fallback extrinsics from `lidar2lidar/conf`.
4. Probe synchronized pairs at a loose pre-check stage.
5. Estimate overlap before running expensive registration.

This means `lidar2lidar` already has a built-in equivalent of a conversion
layer, even though it does not currently serialize a standalone
`standardized_samples.yaml` like `lidar2imu`.

## 4. Algorithm layer

### Stage A: topic discovery and target selection

Implemented in `auto_calib.py`:

- infer candidate point cloud topics
- analyze TF roots
- choose a target topic automatically if none is provided

Current strategy:

- prefer the unique `tf_static` root frame when available
- otherwise fall back to overlap-guided topic preference

### Stage B: candidate-pair screening

For each topic pair:

- verify frame IDs exist
- verify a TF path exists
- find synchronized frames
- estimate overlap from one synchronized probe

This produces `candidate_pairs` and skip reasons before calibration.

### Stage C: edge selection

For each source topic:

- map it to the target frame using TF
- choose a proxy registration target
- reject edges below `--min-overlap`

This is the main “pre-check before optimization” layer.

### Stage D: registration and method comparison

For each selected edge:

- load synchronized source / target clouds
- run one or more registration methods
- compute information-matrix diagnostics
- summarize runs per method
- choose the best method / run

Current supported methods:

- point-to-plane ICP
- GICP
- point-to-point ICP

### Stage E: output consolidation

After per-edge optimization:

- write normalized YAML extrinsics
- build `calibrated_tf.yaml`
- optionally export a merged cloud
- write `metrics.yaml`
- write diagnostics files

## 5. Evaluation layer

### Concise metrics

`metrics.yaml` currently summarizes:

- calibrated edge count
- skipped edge count
- average fitness
- average inlier RMSE
- minimum overlap ratio
- maximum condition number

Per-edge metrics include:

- chosen method
- fitness
- inlier RMSE
- overlap ratio
- sync time delta
- transform delta to initial guess
- information-matrix eigenvalues / condition number / degeneracy

### Detailed diagnostics

The pipeline writes:

- `diagnostics/manifest.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/tf_tree.yaml`
- `diagnostics/topology.yaml`
- `diagnostics/calibration.yaml`
- optional `diagnostics/merged_cloud.pcd`

These files already form a good iteration surface:

- `extraction.yaml`: raw-data extraction and candidate-pair summary
- `tf_tree.yaml`: extraction / frame graph problems
- `topology.yaml`: candidate and skip reasoning
- `calibration.yaml`: per-edge optimization details
- `metrics.yaml`: quick run summary

## 6. What already matches the lidar2imu pattern

`lidar2lidar` already has:

- raw-data extraction integrated into the pipeline
- an explicit candidate-screening stage before optimization
- a stable evaluation output (`metrics.yaml`)
- richer debug outputs separated under `diagnostics/`

So compared with `lidar2imu`, `lidar2lidar` is not missing the pattern; it is
mostly missing the **explicit documentation framing** of that pattern.

The current update also makes that pattern more explicit in artifacts:

- `diagnostics/extraction.yaml` isolates the extraction stage
- `metrics.yaml` now exposes both `coarse_metrics` and `fine_metrics`
- degenerate or poor-quality edges can now be rejected by quality gates

## 7. Current gaps for future iteration

1. The extraction layer is still implicit inside `lidar2lidar-auto`; there is no
   standalone serialized “standardized sample” artifact for offline iteration.
2. Candidate-pair selection currently uses one overlap probe per pair; richer
   pair scoring could improve robustness.
3. Method comparison is per-edge but not yet treated as a reusable experiment
   layer with fixed evaluation slices.
4. Coarse and fine metrics are now explicit, but repeated-run stability
   summaries can still be improved.

## 8. Recommended next iteration

1. Add a documented extraction artifact or report that makes pre-registration
   candidate generation easier to inspect offline.
2. Add stronger edge-quality gates around low overlap or degenerate information
   matrices.
3. Add repeated-run or multi-sample stability summaries for each edge.
4. Keep `topology.yaml` and `calibration.yaml` stable so algorithm changes can
   be compared across runs.

## 9. Current real-bag validation snapshot

Validated on:

- `/home/wfh/01code/apollo-lite/data/bag/record_data_0402`

Default run characteristics:

- selected target topic: `/apollo/sensor/lslidar_main/PointCloud2`
- calibrated edges: left, right
- `average_fitness`: about `0.9747`
- `average_inlier_rmse`: about `0.0065`
- `min_overlap_ratio`: about `0.8425`
- `max_condition_number`: about `2073.8`

Strict-gate validation:

- with `--min-fitness 0.98 --max-condition-number 1500`
- left edge remains accepted
- right edge is rejected with `quality_gate_failed`
- reported reasons include:
  - `condition_number_above_threshold`
  - `fitness_below_threshold`

This is the intended behavior of the current pattern:

1. extraction stays stable
2. algorithm still computes candidate solutions
3. evaluation decides which edges are trustworthy enough to keep

## 10. Temporal hand-eye branch

`lidar2lidar` now also has an experimental temporal branch:

- command: `lidar2lidar-temporal`
- idea: estimate `left_t -> left_{t+k}` and `main_t -> main_{t+k}` motions, then
  solve hand-eye instead of relying only on one synchronized `left -> main`
  registration pair
- comparison: the run also computes the current pairwise ICP baseline

Temporal artifacts:

- `diagnostics/temporal_dataset.yaml`
- `diagnostics/temporal_calibration.yaml`
- `diagnostics/pairwise_baseline.yaml`

Current temporal design:

1. data layer builds reusable temporal windows from raw record timestamps
2. algorithm layer estimates per-sensor motions and filters low-quality windows
3. evaluation layer rejects hand-eye solutions with weak excitation or large
   residuals

Current validation on `record_data_0402`:

- pairwise ICP baseline still succeeds for left and right edges
- temporal motion extraction now builds dense synchronized anchor candidates
- however, most same-sensor windows collapse to near-identity or low-excitation
  solutions, so the temporal quality gate rejects the final hand-eye result

Interpretation:

- the temporal branch is implemented and reusable
- the current bag is still stronger for direct cross-sensor ICP than for
  motion-based hand-eye
- next iteration should improve same-sensor odometry quality, likely with
  scan-to-submap or local-map motion estimation instead of direct scan-to-scan
