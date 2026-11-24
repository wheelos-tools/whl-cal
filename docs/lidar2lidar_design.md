---
audience: dev
stability: stable
P26-04-27
---


# LiDAR-to-LiDAR current design

This document records the current implementation for future iteration work.

Related docs:

- bag preparation: [docs/apollo_data_collection.md](apollo_data_collection.md)
- run / review flow: [docs/calibration_review_guide.md](calibration_review_guide.md)
- methodology / SOTA / references:
  [docs/calibration_methodology.md](calibration_methodology.md)

## 1. Design goal

Treat `lidar2lidar` as three separable layers:

1. **Data extraction**: discover topics, TF edges, synchronized frames, and
   overlap candidates from raw record data
2. **Algorithm**: choose a target topic, select source edges, and refine
   transforms with ICP / GICP
3. **Evaluation**: keep concise metrics and rich diagnostics stable across
   iteration

This matches the current preferred calibration pattern in the repo.

### Semantic and joint numerical contract

- Semantic refinement uses the right retraction
  `R = R_base @ Exp(phi)`, `t = t_base + R_base @ dt`.
  This is not the coupled SE(3) exponential. Additive `phi` derivatives include
  the SO(3) right Jacobian; translation derivatives use **R_base**, not R.
- Freeze each stage's correspondences, then compute residuals, Huber weights,
  noise scale and information at the **optimized transform**, relinearized
  with zero local increment. `info_matrix_6d` has `[rotation, translation]`
  ordering and `information_tangent: right_source`.
- Joint factors use the actual coupled logarithm
  `Log(inv(T_edge) @ T_pred)`. Open3D information is a left-target,
  point-to-point proxy: convert with
  `H_right = Ad(T_edge).T @ H_left @ Ad(T_edge)` before whitening.
  It is not the exact GICP covariance.
- Before common/differential-mode spectral decomposition, convert the joint
  optimizer's additive parameters to left SE(3) tangents in the shared IMU
  frame (`observability.tangent: left_imu_se3`). A shared rig rotation includes
  `delta_t = -skew(t) @ delta_rotation + delta_translation`. L2L factors alone
  cannot observe this common rigid motion.
- Weak-direction fields ending in `_target` are rotated from the source tangent
  into the target frame. The existing block-spectrum test is **not full X-ICP**:
  it does not marginalize rotation/translation coupling, account for correlated
  samples or prove global correspondence uniqueness.
- Semantic failures are retained in `v2_semantic_failures`. If all semantic
  windows fail, keep the supported GICP fallback but log it explicitly; do not
  relabel GICP edges as successful semantic edges.
- Edge `eigenvalues_6d` describes the effective information **after** gravity,
  weak-support and consensus scaling; `registration_eigenvalues_6d` preserves the
  unscaled registration spectrum. `localizability` remains the registration
  diagnostic, not the joint factor's effective strength.
- Joint `verdict` retains its common-mode excitation-gate meaning, with
  `verdict_scope: common_mode_excitation_only`. It is not an independent
  extrinsic-accuracy or release-readiness result.
- `channel_matches` counts source-to-target associations;
  `channel_unique_targets` counts distinct matched target feature indices at
  the last optimized stage. Many source points may share one target. Neither
  count represents independent objects, especially for a shared ground plane
  or repeated returns from one reflector. These counts do not change weights.
- Joint v2 exports `diagnostics/l2l_window_influence.yaml` after solving.
  It removes each window's L2L factors in turn and reports changes in all three
  transforms, with L2I/gravity/height priors, remaining factors/weights and
  optimizer initialization fixed. It does not rerun associations or consensus
  and is not a full-pipeline jackknife or calibrated uncertainty.
  Fewer than two undownweighted windows is reported explicitly; a lone window
  cannot be deletion-tested. Non-converged deletion solves do not produce a
  success-shaped maximum. The diagnostic never selects an extrinsic, changes
  a weight or overrides the excitation verdict; non-v2 selected stages are
  explicitly `not_evaluated`.

Focused regression:
`python -m tools.run_lidar2lidar_feature_smoke` verifies additive derivatives
against finite differences, solved-pose residual/information consistency,
SE(3) logarithms against matrix exponentials, Open3D tangent conversion,
joint whitening and common-mode gauge, failure/fallback accounting and optional
trace equivalence.

### Semantic L2L review trace

Local vertical-line PCA candidates are checked against non-ground support in a
radius of two voxel widths. If that context satisfies the existing facade
planarity/normal tests **and the candidate lies on that plane**, the candidate
is a wall sampling strip rather than an
independent point-to-line constraint and is excluded from the pole channel.
Coplanarity uses the existing plane-inlier distance tolerance
(`ground_inlier_dist_m`, 0.08 m by default); a separate pole near a wall must
not be removed merely because the background dominates its neighborhood.
No correspondence schedule or channel weight is changed. The trace preserves
all checked candidates, context normals/planarity, rejection flags and raw
support CSR under `source/target.line_context`; these arrays are included in
`features.npz`. Counts are reported as `planar_line_rejected_voxels`.
`line_context.png` shows excluded planar-context candidates and retained line
candidates on the same raw scene, before any matching or optimization.
Sparse or non-planar context is not proof of a physical cylinder; sub-tolerance
pole/wall separation and true wall boundaries remain ambiguous. Retained candidates are
shown before count caps, not as confirmed objects or matches. This check does
not resolve cross-window disagreement, common-mode IMU observability or
correlated plane samples.

The joint `lidar2imu calibrate-dual-joint` entry can export feature diagnostics
with `--save-l2l-feature-debug` for semantic modes. The algorithm stays unchanged:

- `semantic_registration.py` records extraction support indices and the actual
  candidates selected at each outer stage, without replacing correspondence search.
- `feature_trace.py` snapshots first rejection reasons, evaluated gates, frozen
  solver pairs and post-solve geometric residuals / Huber component weights.
- `feature_visualization.py` renders headless projections and offline interactive
  galleries from those snapshots, after joint optimization has completed.
- Static and motion-compensated stacks retain canonical scan / point indices.
  Failed registration windows and screening rejects remain reviewable; rejects
  are never sent to the optimizer merely to produce an image.

Diagnostic plane/line grouping is separate from the solver's local primitives.
Objects are not matched one-to-one: their associations are summaries of real
local correspondences. Vertical line candidates are not confirmed cylinders.
The final match view refers to the last *optimized* stage, not a failed trailing
stage or a post-hoc Open3D nearest-neighbor evaluation.

Artifact definitions and usage are in the
[feature-debug quick start](lidar2lidar_quickstart.md#4-面元柱棱线与实际匹配调试).

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

This means `lidar2lidar` already has a built-in extraction layer. Its
production review path serializes normalized input and quality reports under
`diagnostics/standardized_data.yaml` and `diagnostics/data_quality.yaml`.

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
- final acceptance status / release readiness

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

### Shared release-review artifacts

`lidar2lidar` now follows the same final review contract as `lidar2imu`.
Every completed run should expose:

- `metrics.yaml.summary.final_acceptance_status`
- `metrics.yaml.summary.release_ready`
- `metrics.yaml.final_acceptance`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`
- `diagnostics/edge_metrics.csv`
- `diagnostics/skipped_edges.csv`

The final acceptance decision combines coverage, overlap, registration fitness,
conditioning, degeneracy, scene sufficiency, relation connectivity,
repeatability, visual geometry, and loop-closure evidence when available. A
solver result with warning visual geometry, weak scene sufficiency, unresolved
relations, or poor repeatability is therefore explicitly a review candidate
rather than a release-ready calibration.

This keeps the production review order stable:

1. `standardized_data.yaml` confirms the normalized input representation.
2. `data_quality.yaml` confirms whether the data is eligible for optimization.
3. `metrics.yaml` and `acceptance_report.yaml` state the final conclusion.
4. `visualization_index.yaml` lists the visual / tabular evidence needed for
   human review.

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

## 7. Semantic / Intensity / 3D-FoV Upgrade & Joint L2I+L2L Integration

To address raw-cloud ICP corridor degeneracy, ground-point dominance, and
sensor-gain intensity mismatch on diagonal corner LiDARs,
`lidar2lidar/semantic_registration.py` provides:

- **4-channel primitive extraction (`extract_geometric_primitives`)**: balanced
  ground planes, structural vertical facades, vertical poles/edges (`2D`
  point-to-line residuals), and range/incidence-compensated robust-normalized
  reflectivity landmarks (`normalize_reflectivity`), while rejecting
  high-sphericity foliage/clutter.
- **3D visibility & self-occlusion gating (`compute_3d_shared_visibility_mask`)**:
  ego-chassis AABB ray intersection (`ray_intersects_aabb_mask`),
  uncertainty-buffered 3D azimuth/elevation/range frustum checks, and same-side
  normal co-visibility.
- **X-ICP localizability diagnostics**: `6×6` physical Fisher matrix and
  translational/rotational eigen-decomposition (`corridor_degenerate`,
  `weakest_translation_direction_target`).

For dual-LiDAR + IMU rigs, these upgraded `L2L` factors are fused with `2 × L2I`
and turn-window `100 Hz` IMU-preintegrated BA in
[docs/dual_lidar_imu_joint_calibration_design.md](dual_lidar_imu_joint_calibration_design.md).

## 8. Current gaps for future iteration

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
