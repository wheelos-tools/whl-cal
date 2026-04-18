# scan2map context

## Repository findings

- There are no leftover `plan*` or `todo*` files inside the repository tree.
- The current persistent planning state lives in the session `plan.md` and SQL todos.
- The repo already follows a layered calibration pattern:
  1. data extraction
  2. algorithm
  3. evaluation

This is explicit in:

- `docs/lidar2lidar_design.md`
- `docs/lidar2imu_design.md`

## Current project understanding

The current `lidar2lidar` branch should be understood as:

1. **scan2scan baseline**
   - single synchronized frame or pairwise ICP / GICP
   - useful as a practical baseline
   - already has stable extraction and evaluation outputs
2. **temporal hand-eye branch**
   - estimates same-sensor motion over time
   - solves an `AX=XB`-style hand-eye problem
   - useful as an experimental comparison branch
   - can fail when the bag has weak displacement or weak motion excitation

For the current bag, the scan2scan part is not the problem. The limiting factor is
that temporal hand-eye depends on enough motion diversity and displacement, which
is weak here, so the next main direction should be **scan2map**.

## Bag-specific observations

Record:

- `/mnt/synology/raw-data/2026-04-13-06-54-28/20260413065428.record.00000`

Observed topics / frames:

- point cloud topic: `/apollo/sensor/seyond/front/PointCloud2`
- LiDAR frame: `lidar_front`
- pose topic available: `/apollo/localization/pose`
- TF edges observed:
  - `world -> imu`
  - `world -> localization`

This means the bag is not really a multi-LiDAR bag for `lidar2lidar-auto`.
Instead, it is a better fit for a **single-LiDAR scan2map workflow** driven by
localization/world poses.

## Recommended scan2map direction

Do **not** treat this as a pure rename-only task.

The better structure is:

1. keep the current logic as **scan2scan baseline**
2. add a new **scan2map mainline**
3. compare the two in a stable evaluation layer

That preserves the existing baseline while giving the new algorithm a fair and
repeatable comparison target.

## Proposed scan2map architecture

### 1. Data extraction layer

Create a scan2map extraction artifact from raw record data.

Inputs:

- one LiDAR topic
- localization pose topic
- TF graph
- optional initial extrinsic guess

Extraction outputs should include:

- record files
- chosen LiDAR topic and frame
- chosen map/base frame (`world` or `localization`)
- sampled keyframes
- pose-aligned timestamps
- motion / path coverage statistics
- keyframe windows
- submap definitions
- initial transform source
- skip reasons for unusable windows

Recommended artifact:

- `diagnostics/scan2map_dataset.yaml`

Optional heavy artifact:

- cached map-frame or submap PCDs under a dedicated debug directory

### Current dataset-layer defaults

The current extraction defaults follow common scan-to-map practice from systems
such as LOAM / LIO-SAM / FAST-LIO, adapted to this repo's offline calibration
workflow:

- choose the preferred **raw/main LiDAR topic**, not a fused topic
- align scans to `/apollo/localization/pose`
- keep every 3rd aligned scan as **holdout**
- choose optimization keyframes by:
  - translation >= `0.5 m`, or
  - rotation >= `5 deg`, or
  - time fallback >= `2.0 s` **with** at least `0.2 m` translation or `1.0 deg`
    rotation
- define each local submap from up to `20` optimization keyframes within
  `15 m` map-frame radius

That keeps the extraction generic while avoiding the common failure mode of
adding time-only keyframes during near-stationary segments.

### Current `record_data_0402` dataset snapshot

Using `lidar2lidar-scan2map-dataset` on
`/home/wfh/01code/apollo-lite/data/bag/record_data_0402` currently yields:

- selected LiDAR topic: `/apollo/sensor/lslidar_main/PointCloud2`
- raw scans: `716`
- pose-aligned scans: `710`
- skipped scans: `6`
- optimization scans: `474`
- holdout scans: `236`
- keyframes: `71`
- submaps: `71`
- pose sync p95: about `26.8 ms`
- keyframe translation mean: about `0.589 m`
- submap support size: fixed at `20`
- submap support radius mean: about `6.69 m`

This is the current data-layer reference for the next scan2map algorithm step.

### 2. Algorithm layer

The algorithm should be explicitly staged.

#### Stage A: baseline preservation

Keep the current scan2scan result as the baseline:

- single-frame pairwise ICP/GICP
- temporal hand-eye if useful

This should remain visible in outputs, not be overwritten.

#### Stage B: keyframe and submap building

Using localization/world poses plus the current extrinsic guess:

- transform scans into map frame
- build local submaps around anchor timestamps
- filter low-motion or redundant windows
- prefer windows with better spatial coverage

#### Stage C: scan-to-map registration

For each anchor:

- register the anchor scan or local scan-submap bundle to the map/submap
- use localization-projected pose as initialization
- record registration quality and residuals

#### Stage D: joint extrinsic refinement

Use multiple scan-to-map constraints to refine the LiDAR extrinsic.

The optimization should be designed so it can evolve:

- residual weights can change
- correspondence strategy can change
- submap construction can change
- robust loss can change

But the evaluation outputs should stay stable.

### 3. Evaluation layer

Evaluation should compare:

1. initial guess
2. scan2scan baseline
3. scan2map result

Recommended stable metrics:

- accepted keyframe count
- accepted submap count
- registration fitness mean / p95
- registration RMSE mean / p95
- map overlap ratio
- residual p95
- transform delta to initial
- transform delta to scan2scan baseline
- observability / condition number
- path coverage metrics
- turn balance or motion diversity summary
- status / quality gates

Recommended diagnostics:

- `diagnostics/extraction.yaml`
- `diagnostics/scan2scan_baseline.yaml`
- `diagnostics/scan2map_dataset.yaml`
- `diagnostics/scan2map_optimization.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/manifest.yaml`

Stable top-level outputs:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/`

## Iteration rule

The project should continue to follow this rule:

- extraction stays generic and inspectable
- algorithm can iterate aggressively
- evaluation layout stays stable and decides whether a run is trustworthy

For this branch specifically:

- **scan2scan remains the baseline**
- **scan2map becomes the next main algorithm**
- metrics should make it easy to compare them bag by bag
