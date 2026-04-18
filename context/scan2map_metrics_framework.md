# scan2map metrics and experiment framework

## Scope

This note defines the **evaluation framework first**, before implementing a new
scan2map algorithm.

Target bag:

- `/home/wfh/01code/apollo-lite/data/bag/record_data_0402`

Current available comparison surfaces:

1. **initial TF / config guess**
2. **scan2scan baseline** (`lidar2lidar-auto`)
3. **temporal hand-eye comparison branch** (`lidar2lidar-temporal`)
4. **future scan2map mainline**

The key rule is:

- algorithms may change
- metrics and diagnostics layout should stay stable

## Data observations from `record_data_0402`

Observed point-cloud topics:

- `/apollo/sensor/lslidar_main/PointCloud2`
- `/apollo/sensor/lslidar_left/PointCloud2`
- `/apollo/sensor/lslidar_right/PointCloud2`
- `/apollo/sensor/lidar/fusion/PointCloud2`

Observed supporting topics:

- `/apollo/localization/pose`
- `/tf`
- `/tf_static`

Observed frame structure:

- `imu -> lslidar_main` (`/tf_static`)
- `localization -> imu` (`/tf_static`)
- `lslidar_main -> lslidar_left` (`/tf_static`)
- `lslidar_main -> lslidar_right` (`/tf_static`)
- `world -> localization` (`/tf`)

This bag therefore supports:

- scan2scan baseline evaluation
- scan2map using localization / world as the global reference
- cross-checking scan2map against a strong multi-LiDAR baseline

## Literature and best-practice guidance

The metric design should follow the common pattern seen in recent calibration
and scan-to-map evaluation work:

- **global transform accuracy**
  - rotation error
  - translation error
- **local registration quality**
  - inlier RMSE
  - overlap / correspondence quality
- **trajectory / map consistency**
  - ATE
  - RPE
  - map-to-map or cloud-to-map consistency
- **robustness**
  - success rate
  - sensitivity to initialization
  - repeatability across slices / runs
- **observability**
  - condition number
  - singular values
  - motion / turn coverage

Reference directions used for this summary:

- BLCC benchmark style transform-error reporting
- recent target-free calibration work such as GMMCalib / FAST-Calib
- scan-to-map / SLAM evaluation norms such as ATE, RPE, overlap, and map consistency

## Recommended metric split

### A. Coarse metrics

These are the **gate metrics**. They should be stable and easy to compare across
runs and algorithms.

Recommended fields:

1. `accepted_edges` / `accepted_keyframes` / `accepted_submaps`
2. `skipped_edges` / `skipped_windows`
3. `success_rate`
4. `average_fitness`
5. `fitness_p95`
6. `average_inlier_rmse`
7. `inlier_rmse_p95`
8. `min_overlap_ratio`
9. `average_overlap_ratio`
10. `max_condition_number`
11. `degenerate_count`
12. `rotation_error_deg`
13. `translation_error_m`
14. `delta_to_initial`
15. `delta_to_scan2scan_baseline`
16. `ATE_rmse_m`
17. `RPE_translation_m`
18. `RPE_rotation_deg`
19. `runtime_sec`
20. `statuses`

### B. Fine metrics

These are the **debug and iteration metrics**. They help explain *why* one
algorithm is better or worse.

Recommended fields:

1. per-edge fitness / RMSE / overlap distributions
2. per-window residual distributions
3. per-keyframe registration quality
4. per-submap support size and point count
5. information-matrix eigenvalues
6. observability singular values
7. motion excitation distributions
8. left/right turn balance
9. path coverage statistics
10. holdout registration quality
11. map-to-map Chamfer / C2C distance
12. map overlap at fixed thresholds
13. initialization perturbation sensitivity
14. repeated-run stability
15. skipped-item reasons
16. component-wise delta to the baseline (`x/y/z`, `yaw/pitch/roll`)
17. planar vs vertical split (`planar_translation_norm_m`, `pitch_roll_norm_deg`)
18. holdout deltas against the scan2scan baseline
19. condition-number ratio against the baseline
20. vehicle-rig assessment (`accept_candidate`, `planar_only_or_diagnostic`, `keep_baseline`)

## Vehicle-oriented metric additions

For vehicle-mounted rigid LiDARs, the most useful extra split is:

- **planar refinement**
  - `x/y` translation drift
  - `yaw` drift
- **vertical-attitude refinement**
  - `z` drift
  - `pitch/roll` drift

This repo now treats these as first-class debug metrics because road-driving bags often
make planar alignment look better than vertical observability really is. A scan2map
candidate that improves holdout fitness but does so mainly through `z/pitch/roll` drift
should not be treated the same way as a candidate that improves holdout metrics while
keeping all six components close to the scan2scan baseline.

Recommended additional fields in `metrics.yaml` for scan2map validation:

1. `delta_to_initial_components`
2. `delta_to_scan2scan_baseline_components`
3. `holdout_delta_to_initial`
4. `holdout_delta_to_scan2scan_baseline`
5. `vehicle_rig_assessment.thresholds`
6. `vehicle_rig_assessment.statuses`
7. `vehicle_rig_assessment.recommendation`

## How to interpret these metrics for this project

### Coarse metrics answer:

- did the algorithm succeed?
- is the result globally acceptable?
- is it better than the current baseline?
- is the problem sufficiently observable?

### Fine metrics answer:

- where did the algorithm gain or fail?
- were errors caused by low overlap, weak excitation, or poor map support?
- is the solution stable or only good on one slice / one initialization?

## Recommended metrics for each algorithm family

### 1. scan2scan baseline

Keep the current metrics and add:

- per-edge repeatability
- baseline runtime
- delta to static TF guess

### 2. temporal hand-eye

Keep:

- motion pair count
- motion excitation
- residual p95
- rotation-axis rank
- translation condition number

Use it primarily as:

- a comparison surface
- an observability warning surface

### 3. scan2map

Add:

- keyframe count
- submap count
- holdout scan-to-map fitness / RMSE
- map consistency
- ATE / RPE against localization as the reference trajectory
- robustness to initial extrinsic perturbation

## Experiment protocol on `record_data_0402`

### Fixed comparison targets

Every experiment should report:

1. `initial`
2. `scan2scan_baseline`
3. `temporal_handeye` (if it converges)
4. `scan2map_candidate`

### Fixed data split

Use one bag, but split frames by role:

1. **selection slice**
   - used to choose candidate windows or keyframes
2. **optimization slice**
   - used by the algorithm itself
3. **holdout slice**
   - never used during fitting
   - only used for comparison metrics

Recommended first implementation:

- every 3rd synchronized frame goes to holdout
- remaining frames are used for selection / optimization

### Fixed perturbation test

For robustness, test each algorithm with small perturbations around the initial guess:

- translation perturbation: `0.05 m`, `0.10 m`
- rotation perturbation: `0.5 deg`, `1.0 deg`, `2.0 deg`

## Current empirical reference on `record_data_0402`

These runs are the current baseline reference before adding scan2map:

1. **scan2scan default**
   - runtime: `25.43 s`
   - calibrated edges: `2`
   - average fitness: `0.9622`
   - average inlier RMSE: `0.0121`
   - left edge fitness: `0.9785`
   - right edge fitness: `0.9460`
2. **scan2scan strict gate**
   - gate: `min_fitness=0.98`, `max_condition_number=1500`
   - result: `0` accepted edges
   - interpretation: current right sensor is below the intended production-quality threshold, and even the left edge is slightly below a hard `0.98` fitness gate
3. **scan2scan fast-practice variant**
   - config: `voxel_size=0.06`, `remove_ground`, `max_height=2.5`
   - runtime: `25.73 s`
   - average fitness: `0.9638`
   - average inlier RMSE: `0.0141`
   - interpretation: left improves slightly, right degrades slightly, and runtime does not materially improve, so the current default scan2scan path remains the stronger reference
4. **temporal hand-eye smoke**
   - runtime: `25.07 s`
   - accepted edges: `0`
   - reason: lite configuration still produces `0` usable motion pairs on this bag slice
   - interpretation: temporal remains useful as an observability warning surface, but not as the mainline algorithm
5. **scan2map exploratory candidate**
   - target topic: `/apollo/sensor/lslidar_main/PointCloud2`
   - local-submap dataset: `8` support keyframes, radius `6 m`
   - optimization / holdout sampling: `12` optimization submaps, `6` holdout frames, sync threshold `50 ms`
   - accepted edges: `1` (`lslidar_left -> lslidar_main`)
   - chosen method: `3` (point-to-point ICP)
   - left holdout fitness: `0.7829`
   - left holdout RMSE: `0.0864`
   - left holdout overlap: `0.6111`
   - left delta to scan2scan baseline: about `0.038 m`, `0.768 deg`
   - right edge: no stable consensus across optimization submaps
   - interpretation: scan2map can improve left-edge holdout registration on local submaps, but it is not yet a drop-in replacement for the current scan2scan baseline across all sensors

This means scan2map should be judged against:

- the **default scan2scan** result as the primary baseline
- the **strict gate** as the future acceptance target
- the **temporal branch** as a warning / observability comparison only
- the **current exploratory scan2map result** as the first algorithmic reference for submap-based refinement

## Current method comparison inside scan2map

On the accepted `lslidar_left -> lslidar_main` edge, the current scan2map candidate shows:

1. **initial / static TF guess on holdout submaps**
   - fitness: `0.7379`
   - RMSE: `0.1015`
   - overlap: `0.5052`
2. **scan2scan baseline extrinsic on the same holdout submaps**
   - fitness: `0.7379`
   - RMSE: `0.1015`
   - overlap: `0.5053`
3. **scan2map method 2 (GICP)**
   - consensus keyframes: `3`
   - holdout fitness: `0.7822`
   - holdout RMSE: `0.0871`
   - holdout overlap: `0.6085`
4. **scan2map method 3 (point-to-point ICP)**
   - consensus keyframes: `10`
   - holdout fitness: `0.7829`
   - holdout RMSE: `0.0864`
   - holdout overlap: `0.6111`

Current interpretation:

- method `3` is the more stable scan2map choice on this bag because it reaches a much larger consensus set
- method `2` is close on holdout quality, but less stable across optimization submaps
- the right sensor still fails before holdout comparison, so the main blocker is optimization consistency, not final holdout scoring

## Vehicle-oriented judgment after constrained experiments

Using the newer default-submap + ground-filtered comparison on `record_data_0402`:

1. **left -> main**
   - unconstrained scan2map improves holdout metrics over scan2scan
   - delta to scan2scan baseline is about `0.032 m`, `0.779 deg`
   - planar drift stays small and vertical drift stays inside a plausible vehicle-rig range
   - current recommendation: `accept_candidate`
2. **right -> main**
   - unconstrained scan2map improves holdout metrics strongly, but only by drifting about `0.10 m` and `3.87 deg` from scan2scan
   - the drift is concentrated in `z/pitch/roll`, not `x/y/yaw`
   - holdout condition number also rises to about `1.53x` the scan2scan baseline
   - current recommendation: `planar_only_or_diagnostic`, not direct acceptance
3. **right -> main with constrained DoF (`lock z,pitch,roll`)**
   - drift drops to about `0.023 m`, `0.217 deg`
   - holdout metrics collapse back near the scan2scan baseline
   - interpretation: the previous right-edge scan2map gain was mostly coming from vertical-attitude bias rather than trustworthy extrinsic improvement

This means the current repo-level method recommendation is:

- keep **scan2scan** as the production default and primary acceptance baseline
- use **scan2map** as a secondary validation / refinement layer
- accept unconstrained scan2map only when planar drift, vertical drift, and conditioning all stay inside reasonable bounds relative to scan2scan
- treat large right-edge `z/pitch/roll` changes as a warning that the bag lacks enough vertical observability for free 6DoF scan2map acceptance

Report:

- convergence rate
- coarse metric degradation
- best / worst / median outcome

## Initial baseline result already observed

Current `scan2scan` baseline on `record_data_0402` with the existing pipeline:

- calibrated edges: `2`
- skipped edges: `1`
- average fitness: about `0.9622`
- average inlier RMSE: about `0.0121`
- min overlap ratio: about `0.7802`
- max condition number: about `2074.8`

Per-edge snapshot:

- left -> main: fitness about `0.9785`, RMSE about `0.0116`
- right -> main: fitness about `0.9460`, RMSE about `0.0127`

This is the first comparison anchor for scan2map.

## Recommended output layout

Top-level:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/`

Diagnostics:

- `diagnostics/extraction.yaml`
- `diagnostics/scan2scan_baseline.yaml`
- `diagnostics/temporal_baseline.yaml`
- `diagnostics/scan2map_dataset.yaml`
- `diagnostics/scan2map_optimization.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/manifest.yaml`

## Next algorithm step

Only after this framework is fixed:

1. implement scan2map extraction
2. implement scan2map optimization
3. run against the same holdout split
4. compare against the scan2scan baseline using the same metrics
