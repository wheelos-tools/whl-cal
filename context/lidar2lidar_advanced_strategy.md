---
audience: dev
stability: experimental
last_tested: 2026-05-11
---

# LiDAR-to-LiDAR advanced strategy

## Goal

The next-stage `lidar2lidar` target is not just "run loop closure" but a
**prior-aware, topology-aware, scene-aware, and metric-driven** rig calibration
system.

## Implemented scaffolding (2026-05-11)

The current code now includes the first concrete version of that strategy:

- `lidar2lidar-auto --workflow-yaml <yaml>`
  - supports:
    - `mode: target_star`
    - `mode: tf_tree`
    - `mode: explicit`
    - `mode: complete`
- `diagnostics/workflow.yaml`
  - records the resolved topic set, relation roles, and thresholds
- `diagnostics/scene_sufficiency.yaml`
  - records windowed overlap, timestamp skew, wall support, dynamic unmatched
    ratio, and suggestions
- per-edge repeatability in `metrics.yaml`
  - translation / rotation p95 across multiple synchronized windows
- richer `diagnostics/visual_evaluation.yaml`
  - wall thickness
  - corner spread
  - slice sharpness

This means the repo now has the **control surface** needed for advanced
algorithm work, even though the thresholds still need more bag-level validation.

For the current four-corner Vanjee rig, the intended production interpretation is:

1. use the four raw corner LiDARs only
2. treat the rectangle perimeter as the primary loop structure
3. treat diagonals as consistency checks, not primary drivers
4. accept results only when numeric, geometric, and visual evidence agree

## Current validated starting point

### Dynamic-bag lesson

The dynamic `run-eight` bag showed that a generic graph can still be pulled by
weak or motion-contaminated edges.

The current conservative loop-closure mode is already better than the original
generic graph solve because it:

- gates low-quality edges
- compares each measurement to the trusted TF prior
- regularizes the global solve toward the trusted rig geometry
- refuses to report `pass` when there is no effective loop evidence

But that is still a **safety layer**, not the final production algorithm.

### Static-bag lesson

On `/mnt/synology/REDACTED/2026-05-07-REDACTED_USER/bag/20260508032341.record.00000`,
the four raw corner LiDARs support a valid single-component loop under static
conditions.

Observed pattern:

- the perimeter edges are all usable:
  - `left_front <-> right_front`
  - `right_front <-> right_back`
  - `right_back <-> left_back`
  - `left_back <-> left_front`
- one diagonal can still be useful as a secondary check
- the weakest diagonal is better treated as a consistency check only

This is the strongest current evidence that the right production direction is a
**rectangle-ring solver**, not a fully generic free-form graph.

## Priors the algorithm should consume

### 1. TF prior

The TF tree should be treated as a first-class prior, not just an initializer.

Recommended use:

- use record TF edges first when available
- fall back to checked-in rig extrinsics only when the record is missing a path
- track prior provenance per edge:
  - `record_tf`
  - `config_tf`
  - `baseline_estimate`
- track a trust level per prior:
  - `trusted`
  - `usable`
  - `weak`

The solver should use this prior in three ways:

1. initialize the edge solve
2. gate obviously implausible measurements
3. regularize the final graph solve

### 2. Topology prior

The rig topology should also be explicit.

For the current rectangle vehicle rig:

- primary edges: perimeter only
- secondary edges: diagonals only
- unsupported edges: any edge outside the known rig topology

Recommended rule:

- a production `pass` must require all perimeter edges to be sufficiently
  supported
- a diagonal may improve confidence, but a diagonal must not rescue a missing
  perimeter edge

### 3. Data-condition prior

The pipeline should know whether it is operating on:

- static windows
- slow-motion windows
- dynamic driving windows

That prior should directly change:

- synchronization thresholds
- allowed edge families
- edge weights
- acceptance policy

For example:

- static windows can support rectangle-ring solve and wall-thickness checks
- dynamic windows should down-weight edges with strong time skew or strong
  dynamic-object contamination

### 4. Scene prior

The solver should estimate what the scene can actually support before attempting
to optimize.

Recommended scene descriptors:

- plane count and plane normal diversity
- presence of long vertical walls
- corner / facade intersection count
- range coverage balance across sensors
- dynamic-object ratio
- occlusion asymmetry across the pair
- effective overlap on each perimeter edge

## Data-layer upgrades

### 1. Move from one-shot pair probing to windowed evidence

The current pipeline still relies too much on one synchronized pair per edge.

The next step should be:

1. build candidate windows first
2. summarize each window
3. gate weak windows
4. estimate each edge from multiple windows
5. aggregate edge evidence with repeatability statistics

That gives the algorithm:

- repeatability
- uncertainty
- better bad-window rejection
- better operator guidance

### 2. Add a scene-sufficiency report before optimization

Before solving, the pipeline should answer:

- is the current bag sufficient for pairwise calibration?
- is it sufficient for a full rectangle loop?
- which perimeter edge is weakest?
- is the scene mainly wall-dominant, corner-rich, or open-space weak?

Recommended output fields in a future `diagnostics/scene_sufficiency.yaml`:

- `perimeter_edge_overlap`
- `diagonal_edge_overlap`
- `valid_static_window_count`
- `dynamic_object_ratio`
- `plane_normal_diversity`
- `vertical_wall_support`
- `corner_support`
- `timestamp_skew_ms`
- `scene_class`
- `recommendation`
- `suggestions`

Example operator-facing suggestions:

- `left_back <-> left_front edge is weak; add a longer static hold near a left-side wall`
- `scene lacks orthogonal structure; do not trust yaw refinement`
- `diagonal disagreement is high; likely dynamic-object contamination`

### 3. Keep prepared datasets as the default iteration surface

The current prepared raw-rig dataset is already the right extraction surface for
serious algorithm work.

The next iteration should extend it with:

- per-window metadata
- optional per-window plane extraction
- optional per-window dynamic masks
- downsample-rate variants for repeatable ablations

## Algorithm roadmap

### Stage A: rectangle-ring edge planner

Replace the current "generic graph first" mindset with a topology-aware planner.

Recommended behavior:

1. define the perimeter chain explicitly
2. solve perimeter edges first
3. solve diagonals only if the perimeter is already healthy
4. reject a production `pass` if any perimeter edge fails

### Stage B: multi-window edge estimation

Each edge should be estimated from multiple windows, not one window.

For each edge:

1. solve per-window relative transforms
2. compute repeatability statistics
3. compute drift to prior
4. reject unstable windows
5. aggregate the surviving windows into one measurement plus uncertainty

This makes edge weights data-driven instead of hand-tuned.

### Stage C: prior-aware global bundle solve

The final solve should be a weighted graph optimization where each factor carries:

- edge measurement
- measurement uncertainty
- scene sufficiency score
- prior consistency score
- window repeatability score

Recommended factor families:

1. perimeter edge factors
2. diagonal consistency factors
3. TF prior factors
4. cycle-consistency factors
5. optional rigidity / symmetry priors when justified by the rig

### Stage D: fail-safe modes

The solver should explicitly degrade its ambition when evidence is weak.

Recommended modes:

1. `full_loop`
   - all perimeter edges healthy
   - diagonals consistent
   - repeatability good
2. `perimeter_only`
   - all perimeter edges healthy
   - diagonals ignored or only used as checks
3. `pairwise_safe`
   - one or more perimeter edges weak
   - keep trusted baseline and only allow small prior-safe corrections
4. `diagnostic_only`
   - scene insufficient for a production update

This is better than a single algorithm pretending all bags are equally good.

### Stage E: optional global-environment refinement

Once the rectangle-ring mode is stable, the next advanced layer can use the
environment itself as an additional consistency reference:

- scan-to-submap checks
- wall-plane bundle adjustment
- repeated static poses from different timestamps

But this should be additive, not a replacement for the rig-topology solve.

## Metric roadmap

## Coarse metrics

These should decide status and recommendation.

Recommended additions to `metrics.yaml`:

1. `perimeter_edge_count`
2. `perimeter_edge_success_rate`
3. `perimeter_complete`
4. `diagonal_consistency_status`
5. `scene_sufficiency_status`
6. `edge_repeatability_translation_p95_m`
7. `edge_repeatability_rotation_p95_deg`
8. `cycle_translation_residual_p95_m`
9. `cycle_rotation_residual_p95_deg`
10. `delta_to_prior_p95`
11. `holdout_window_success_rate`
12. `wall_thickness_p95_m`
13. `corner_sharpness_status`
14. `rate_ablation_status`
15. `recommendation`

### Fine metrics

These should explain why the run passed or failed.

Recommended additions:

1. per-edge repeatability distributions
2. per-window scene-sufficiency scores
3. per-edge delta-to-prior distributions
4. wall-plane signed-span by plane and by sensor
5. sensor-offset spread by plane family
6. diagonal disagreement by component (`x/y/z`, `yaw/pitch/roll`)
7. corner thickness / facade double-edge score
8. normal-consistency dispersion
9. cloud-to-cloud residual distributions after loop solve
10. stability across voxel sizes and sample rates

## Visual evaluation roadmap

The current wall-thickness note is useful but not yet enough.

The next visual layer should produce review surfaces that answer the same
questions a human engineer asks in CloudCompare or Open3D:

1. are walls thin?
2. are corners sharp?
3. is there ghosting or doubling?
4. do different sensors agree on the same facade?

Recommended artifacts:

1. colored merged cloud for baseline vs optimized
2. per-plane cropped views
3. BEV and side-slice overlays
4. wall cross-sections with signed thickness plots
5. corner crops with local spread metrics
6. pointwise residual heatmaps
7. before/after snapshots for the same static window

Recommended derived visual metrics:

1. wall signed-span p50 / p95
2. wall double-edge separation
3. facade normal disagreement
4. corner spread radius
5. per-sensor color-bleed width
6. slice sharpness score in BEV / XZ / YZ

## Acceptance policy

The advanced system should not promote a result from one improved scalar.

A production `pass` should require all of the following:

1. the perimeter loop is complete
2. scene sufficiency is healthy
3. repeatability is within threshold
4. cycle residuals improve or remain low
5. drift to trusted prior stays plausible
6. wall / corner visual metrics do not regress
7. manual review artifacts look consistent with the numeric result

Recommended downgrade rules:

- missing perimeter edge -> at most `warning`
- scene insufficient for yaw -> do not promote yaw
- wall thickness improves but corner sharpness worsens -> keep `warning`
- numeric consistency improves while visual artifacts worsen -> keep baseline

## Recommended next experiments

### Highest priority

1. implement `scene_sufficiency.yaml`
2. implement a true rectangle-ring mode
3. add multi-window repeatability per edge
4. validate wall / ghosting / corner metrics against manual review on static bags
5. run 10 Hz / 5 Hz / 2 Hz prepared-dataset ablations

### Second priority

1. add per-window dynamic-object masking
2. add holdout-window evaluation for loop closure
3. compare different voxel sizes and registration methods under the same metric surface
4. add optional scan-to-submap consistency checks

## Practical recommendation

If the goal is "the best possible production `lidar2lidar` system", the next
correct order is:

1. **scene sufficiency first**
2. **rectangle-ring solver second**
3. **repeatability and holdout metrics third**
4. **visual wall / corner / ghosting validation fourth**
5. **global-environment refinement last**

That order gives the best chance of building a system that is not only stronger,
but also safer, more interpretable, and easier to accept in production.
