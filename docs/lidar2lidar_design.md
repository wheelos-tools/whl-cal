---
audience: dev
stability: stable
P26-04-27
---


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

Current orchestration supports:

- direct CLI-driven target-star runs
- `--workflow-yaml` runs with:
  - `mode: target_star`
  - `mode: tf_tree`
  - `mode: explicit`
  - `mode: complete`

### `lidar2lidar-rig-dataset`

Shared raw-rig extraction surface.

Main files:

- `lidar2lidar/prepared_dataset.py`
- `tools/lidar2lidar/prepare_rig_dataset.py`

Current optional extension:

- `--loop-closure`
  - calibrate additional pairwise graph edges among the selected raw LiDARs
  - keep the original pairwise star baseline
  - solve a graph-level global-consistency refinement
  - write `loop_closed_tf.yaml` and loop-closure diagnostics for side-by-side comparison

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

For multi-LiDAR vehicle rigs, the extraction layer can now also be frozen into
`diagnostics/prepared_rig_dataset.yaml` and reused later via
`--prepared-dataset-yaml`.

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

The pipeline now supports two edge-planning modes:

1. **legacy target-star**
   - calibrate each selected source directly to one target topic
2. **workflow-planned relations**
   - load `--workflow-yaml`
   - resolve explicit or TF-derived relations
   - mark each relation as:
     - `primary`
     - `supporting`
     - `check`
   - reject relations below `--min-overlap`

This is now the main “pre-check before optimization” layer.

### Stage D: registration and method comparison

For each selected edge:

- load synchronized source / target clouds
- run one or more registration methods
- compute information-matrix diagnostics
- summarize runs per method
- choose the best method / run
- summarize multi-window repeatability relative to the chosen reference run

Current supported methods:

- point-to-plane ICP
- GICP
- point-to-point ICP

### Stage E: output consolidation

After per-edge optimization:

- compose relation results back into the chosen target frame
- write normalized YAML extrinsics
- build `calibrated_tf.yaml`
- optionally export a merged cloud
- write `metrics.yaml`
- write diagnostics files

### Stage F: optional rig loop closure

When graph refinement is enabled (either by `--loop-closure` on the legacy path
or by `planner.enable_global_optimization: true` in a workflow YAML):

1. build additional pairwise measurement edges among the selected raw LiDARs
2. choose a spanning tree plus extra loop edges
3. keep the pairwise baseline as the initial rig pose set
4. solve a graph-consistency refinement with the chosen target LiDAR fixed
5. write:
   - baseline `calibrated_tf.yaml`
   - loop-closed `loop_closed_tf.yaml`
   - `loop_closed/*.yaml`
   - `diagnostics/loop_closure.yaml`
   - optional visual artifacts under `diagnostics/`

## 5. Evaluation layer

### Concise metrics

`metrics.yaml` currently summarizes:

- calibrated edge count
- skipped edge count
- average fitness
- average inlier RMSE
- minimum overlap ratio
- maximum condition number
- scene sufficiency status
- relation connectivity status
- edge repeatability p95
- visual geometry status

Per-edge metrics include:

- chosen method
- fitness
- inlier RMSE
- overlap ratio
- sync time delta
- transform delta to initial guess
- repeatability across multiple synchronized windows
- information-matrix eigenvalues / condition number / degeneracy

### Detailed diagnostics

The pipeline writes:

- `diagnostics/manifest.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/tf_tree.yaml`
- `diagnostics/workflow.yaml`
- `diagnostics/topology.yaml`
- `diagnostics/calibration.yaml`
- `diagnostics/scene_sufficiency.yaml`
- optional `diagnostics/merged_cloud.pcd`
- optional `diagnostics/loop_closure.yaml`
- optional `diagnostics/visual_evaluation.yaml`
- optional colored merged clouds for baseline / loop-closure comparison

These files already form a good iteration surface:

- `extraction.yaml`: raw-data extraction and candidate-pair summary
- `tf_tree.yaml`: extraction / frame graph problems
- `workflow.yaml`: resolved workflow topics, relations, and thresholds
- `topology.yaml`: candidate and skip reasoning
- `calibration.yaml`: per-edge optimization details
- `scene_sufficiency.yaml`: windowed overlap / skew / wall support / suggestions
- `metrics.yaml`: quick run summary
- `loop_closure.yaml`: graph-level before/after consistency
- `visual_evaluation.yaml`: wall / corner / slice summaries for human review

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
2. Scene sufficiency is now present, but still uses lightweight geometric
   proxies rather than full dynamic-object masking or map support modeling.
3. Method comparison is per-edge and includes repeatability, but holdout-window
   validation can still be strengthened.
4. Visual metrics are now concrete, but wall / corner / slice thresholds still
   need more bag-level validation.

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
