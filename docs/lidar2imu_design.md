---
audience: dev
stability: stable
P26-04-27
---


# LiDAR-to-IMU current design

This document records the current implementation for future iteration work.

Related docs:

- bag preparation: [docs/apollo_data_collection.md](apollo_data_collection.md)
- run / review flow: [docs/calibration_review_guide.md](calibration_review_guide.md)
- methodology / SOTA / references:
  [docs/calibration_methodology.md](calibration_methodology.md)

## 1. Design goal

Keep the system split into two independent layers:

1. **Algorithm layer**: estimate `x, y, z, yaw, roll, pitch`
2. **Evaluation layer**: keep metrics and diagnostics stable while the algorithm
   evolves

This separation is intentional: later changes should mostly modify the
conversion logic, sample selection, losses, or residual weights, without
changing how results are judged.

## 2. Implemented commands

### `lidar2imu-calibrate`

Runs the staged solver from standardized samples.

Main files:

- `lidar2imu/cli.py`
- `lidar2imu/pipeline.py`
- `lidar2imu/algorithms.py`
- `lidar2imu/metrics.py`
- `lidar2imu/io.py`

### `lidar2imu-convert-record`

Builds standardized samples from Apollo record data, then optionally runs the
solver.

Main file:

- `lidar2imu/record_converter.py`

The converter now supports two extraction inputs:

- `--record-path`
- `--prepared-dataset-yaml`

The prepared-dataset path is intended for multi-LiDAR vehicle bags where the
same raw data must feed both `lidar2imu` and `lidar2lidar` without rescanning
the record every time.

## 3. Standardized input schema

The solver consumes:

- `parent_frame`
- `child_frame`
- `initial_transform`
- `ground_samples`
- `motion_samples`

### Ground sample meaning

- `lidar_plane_normal`
- `lidar_plane_offset`
- `imu_gravity`
- `imu_ground_height`
- `weight`
- `sync_dt_ms`

### Motion sample meaning

- `imu_delta`
- `lidar_delta`
- `weight`
- `sync_dt_ms`

The converter writes this schema to `standardized_samples.yaml`.

In prepared-dataset mode, the converter reuses:

- cached raw-LiDAR `pcd` frames
- cached pose / IMU state
- cached TF edges

## 4. Algorithm stages

### Stage A: ground orientation

Estimate `roll/pitch` by aligning:

- LiDAR ground normal
- IMU gravity direction

Residual idea:

- minimize the cross-product error between predicted up and gravity-derived up

### Stage B: ground translation prior

Use `imu_ground_height` as a practical height prior.

Current role:

- mainly stabilizes `z`
- also exposes how weak the height-only geometry is through observability

### Stage C: motion rotation

Solve `yaw` with the hand-eye rotation residual:

- `R_A * R_IL = R_IL * R_B`

### Stage D: motion translation

Solve translation with the standard hand-eye translation equation:

- `(R_A - I) * t_IL = R_IL * t_B - t_A`

### Stage E: joint refinement

Run robust least squares over:

- ground normal residuals
- ground height residuals
- motion rotation residuals
- motion translation residuals

## 5. Evaluation design

The solver writes:

- `metrics.yaml`
- `diagnostics/algorithm.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/observability.yaml`
- `diagnostics/manifest.yaml`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`
- `diagnostics/ground_residuals.csv`
- `diagnostics/motion_residuals.csv`
- `diagnostics/holdout_motion_residuals.csv`

### Coarse metrics

Used as gate metrics for iteration:

- `ground_normal_angle_p95_deg`
- `ground_height_residual_p95_m`
- `motion_rotation_residual_p95_deg`
- `motion_translation_residual_p95_m`
- `motion_angular_excitation_p95_deg`
- `motion_registration_fitness_p05`
- `motion_registration_inlier_rmse_p95`
- `left_turn_count`
- `right_turn_count`
- `turn_balance_ratio`
- `joint_condition_number`
- `statuses`
- `vehicle_motion_assessment`
- `final_acceptance`

### Fine metrics

Used for debugging and model iteration:

- per-stage residual distributions
- per-stage observability / singular values
- yaw cost sharpness (`max_cost_ratio`, `within_5pct_span_deg`)
- selected-motion axis / heading diversity
- per-sample ground diagnostics
- per-sample motion diagnostics

### Shared release-review artifacts

`lidar2imu` now follows the same final review contract as `lidar2lidar`.
Every completed run should expose:

- `metrics.yaml.summary.final_acceptance_status`
- `metrics.yaml.summary.release_ready`
- `metrics.yaml.final_acceptance`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`

The final acceptance decision combines ground support, motion registration,
motion residuals, turn balance, observability, yaw cost-scan health,
extraction/reference consistency, holdout generalization, and the existing
`vehicle_motion_assessment.recommendation`. Non-`full_6dof_candidate`
recommendations remain valid diagnostics, but they are not release-ready full
6DoF calibration results.

This keeps the production review order stable:

1. `standardized_data.yaml` confirms normalized sample counts, frames, metadata,
   and transform provenance.
2. `data_quality.yaml` confirms whether the samples are eligible for full
   optimization.
3. `metrics.yaml` and `acceptance_report.yaml` state the final conclusion.
4. `visualization_index.yaml` points to residual CSVs, IMU-vs-LiDAR trajectory
   overlays, stitched-keyframe point clouds, and observability outputs that
   should be plotted or inspected.

### Observability semantics

Observability is now evaluated against the **expected rank of each stage**, not
against a blanket "at least two singular values" rule.

- 2-parameter ground orientation expects rank `2`
- 3-parameter ground translation expects rank `3`
- 1-parameter yaw solve expects rank `1`
- locked-axis translation and joint refinement expect the number of currently
  free parameters

This matters because an `N x 1` Jacobian is valid for single-parameter problems.
The old logic structurally marked those stages as degenerate even when scalar
sensitivity was nonzero.

### Yaw-stage gate

The yaw stage now uses two layers:

1. expected-rank observability on the local Jacobian
2. a periodic cost scan over `yaw ∈ [-180 deg, 180 deg]`

The cost scan writes:

- `best_yaw_deg`
- `best_cost`
- `max_cost`
- `max_cost_ratio`
- `within_5pct_span_deg`

Yaw is treated as weak when the scalar stage exists but the global cost surface
is too flat or too wide around the optimum. This avoids the earlier false
positive on all `N x 1` stages while still catching real weak-yaw bags.

## 6. Record conversion design

Current default topic mapping:

- LiDAR: `/apollo/sensor/lslidar_main/PointCloud2`
- Pose / IMU-side motion: `/apollo/localization/pose`
- IMU topic: `/apollo/sensor/gnss/imu`
- TF source: `/tf_static` + TF graph

### Current conversion flow

1. Resolve `T_imu_lidar` from TF, or load it from `--initial-transform`.
2. Resolve `T_localization_imu` from TF.
3. Sample LiDAR frames for ground extraction.
4. Fit a ground plane from each selected cloud.
5. Build `imu_gravity` from pose orientation by default.
6. Build motion candidates from localization-pose relative motion.
7. Assign motion candidates into timeline windows.
8. Gate weak windows before registration.
9. Inside each valid window, keep a **multi-scale preselection** instead of only
   one local best candidate.
10. Run LiDAR-to-LiDAR GICP on those candidates and keep only the registration-
    passing motion factors.
11. Run a **global diversity selection** across the registered pool so final
    factors preserve stride diversity, turn diversity, and horizontal heading
    diversity.
12. Export normalized samples.

If a bag has no LiDAR-to-parent TF at all, the converter can also run with
`--identity-initial-transform`, but that mode is exploratory only and should not
be treated as a production prior.

When a bag **does** provide a trusted TF, `--initial-transform` stays solver-only.
If you need to intentionally rebuild record-side geometry from a different seed
(for example, pass-2 of `--auto-reextract-if-needed`), use
`--extraction-transform` so `reference_transform` and `extraction_transform`
remain explicitly separated.

### Why motion selection was changed

The first version used uniform sampling and failed on the real bag because only
one motion pair had enough angular excitation.

The current industrialized version:

- tries multiple frame strides
- groups motion candidates into windows
- gates weak windows
- keeps multi-scale candidates inside each valid window
- runs registration first, then selects final samples from the **registered pool**
- preserves stride and heading diversity in the final set instead of collapsing to
  a single short local mode

This follows the same practical idea used in `lidar2lidar`: evaluate candidate
pairs first, then spend optimization effort on the useful ones. The current goal
is to recover more of the bag's long-horizon structure without breaking the
stable output / evaluation surfaces.

### Current map-based prototype and remaining limitation

Global-diversity selection improves the data layer, but pairwise scan-to-scan
registration still leaves some `8`-shape bags too flat in yaw. To bridge that
gap, the converter now supports:

- `--motion-registration-mode scan_to_scan`
- `--motion-registration-mode submap_to_submap`
- `--motion-registration-mode submap_to_map`

The map-side modes keep the same selection / metrics framework:

- `submap_to_submap`: symmetric pose-anchored local submap pair
- `submap_to_map`: source local submap against a larger target local map

The latest selector also uses a **coverage-aware global score** instead of only
following the strongest local factor in every window. In practice this means:

- reward new stride coverage
- reward new heading-bin coverage
- reward underrepresented turn direction
- penalize repeated same-stride / same-heading / near-pure-z-axis local factors

This matches the common industrial pattern of keyframe / factor selection by
coverage, not just by per-factor strength.

### Fixed baseline and current production candidate

To keep iteration disciplined, `lidar2imu` now distinguishes:

- **baseline**
  - fixed comparison reference
  - `scan_to_scan`
  - `--planar-motion-policy auto`
  - no automatic re-extraction loop
- **production**
  - current release candidate
  - `submap_to_map`
  - target local map widened enough to preserve long-horizon structure
  - `--planar-motion-policy auto`
  - automatic one-step re-extraction when extraction consistency warns
  - acceptance still gated by:
    - trusted-reference consistency
    - planar basin stability
    - extraction consistency
    - holdout generalization

This is the current repo-level operating rule:

- compare new ideas against `baseline`
- ship only through `production`
- keep both on the same stable metric surfaces

Real-bag validation shows the expected intermediate behavior:

- On the current `8`-shape bag, submap factors improve yaw curvature and motion
  factor quality:
  - yaw cost ratio `≈ 1.15 -> 1.57`
  - yaw `5%` plateau `≈ 143 deg -> 68.5 deg`
  - motion translation residual `p95 ≈ 0.45 m -> 0.13 m`
- With the coverage-aware selector on top of submaps, the same bag improves again:
  - selected strides `40 -> {20, 40}`
  - yaw cost ratio `≈ 1.57 -> 1.75`
  - yaw `5%` plateau `≈ 68.5 deg -> 59 deg`
- With `submap_to_map`, the same weak bag improves again:
  - default target local map: `≈ 1.75 -> 2.26`, plateau `≈ 59 deg -> 45.5 deg`
  - wider target local map: `≈ 2.26 -> 4.52`, plateau `≈ 45.5 deg -> 26.5 deg`
  - solver policy changes from `freeze_xyyaw` to `free`
- On `record0402`, submap factors improve registration quality and residuals
  without changing the main diagnosis:
  - registered candidate pool `24 -> 33`
  - motion translation residual `p95 ≈ 0.54 m -> 0.30 m`
  - yaw remains strong, but planar DOFs still stay locked because turn balance is
    one-sided
- On `record0402`, `submap_to_map` also improves sharply:
  - yaw cost ratio `≈ 7.83 -> 28.97`
  - yaw `5%` plateau `≈ 19 deg -> 9 deg`
  - diagnosis still correctly remains `turn_imbalance_only`

So the current evidence is:

- `submap_to_submap` is a useful intermediate step
- `submap_to_map` is the first prototype that can push the weak `8`-shape bag
  into `full_6dof_candidate` territory when the target local map is large enough
- but industrial acceptance still needs repeatability, holdout validation, and
  sensitivity checks, not only one successful run
- repeatability must now include **trusted-reference consistency**, not only
  residuals and yaw curvature: a map-based run can achieve `free` with strong
  internal metrics while still converging to the wrong planar basin
- when a bag already provides trusted TF, record-side extraction should use that
  trusted transform even if calibration is intentionally started from a perturbed
  initial guess; otherwise the standardized samples themselves inherit the wrong
  basin before optimization even starts
- evaluation now also tracks **extraction consistency**:
  - `delta_to_extraction`
  - `coarse_metrics.statuses.extraction_geometry`
  - `vehicle_motion_assessment.extraction_consistency`
  - `vehicle_motion_assessment.extraction_consistency_recommendations`
  - `translation_xyz_m.z` and vertical-threshold failure reasons, so a pure
    lidar-height mismatch no longer hides inside the total translation norm
- evaluation now also tracks **trusted-reference vertical consistency**:
  - `delta_to_reference.translation_xyz_m.z`
  - `delta_to_reference.vertical_error_ratio`
  - `vehicle_motion_assessment.reference_consistency_details.failure_reasons`
  - `fine_metrics.reference_consistency`
- evaluation now also tracks **user-prior recoverability**:
  - `coarse_metrics.statuses.initial_prior_nominal_range`
  - `vehicle_motion_assessment.initial_prior_assessment`
  - `vehicle_motion_assessment.initial_prior_assessment_details`
  - `fine_metrics.initial_prior_assessment`
  - status meaning:
    - `pass`: input TF already sits inside the nominal production range
    - `recoverable`: input TF is outside nominal range, but the bag still converges
      to the accepted basin
    - `warning`: input TF is outside nominal range and the run also conflicts with
      other acceptance surfaces
- evaluation now also tracks **full 6DoF prior robustness**:
  - keep the existing `planar_basin_stability` as the narrow `x/y/yaw` multistart
    surface
  - add `full_prior_robustness` as the wider multistart surface over:
    - planar perturbations
    - vertical `z` perturbation
    - roll / pitch perturbation
    - the current input/reference seed family
  - write:
    - `coarse_metrics.statuses.full_prior_robustness`
    - `vehicle_motion_assessment.full_prior_robustness`
    - `vehicle_motion_assessment.full_prior_robustness_details`
    - `fine_metrics.full_prior_robustness`
  - interpretation:
    - `pass`: nearby prior families collapse to one stable basin
    - `warning`: at least one nearby prior family reaches another basin, so one
      accepted solve is not yet industrially stable
- evaluation now also tracks **holdout generalization**:
  - split selected motion factors into:
    - calibration subset
    - holdout subset
  - current default rule: keep every 3rd selected motion sample as holdout when
    enough motion samples remain
  - write:
    - `summary.dataset_partition`
    - `coarse_metrics.statuses.holdout_generalization`
    - `vehicle_motion_assessment.holdout_generalization`
    - `vehicle_motion_assessment.holdout_validation_details`
    - `fine_metrics.holdout_validation`
  - this separates two different failure modes:
    - wrong final basin relative to a reference transform
    - stale exported samples that need a second extraction pass
    - map objectives that look strong in-sample but degrade on held-out motion
      factors
- evaluation now also tracks **repeated holdout stability**:
  - keep the existing deterministic holdout split as the first generalization check
  - replay additional holdout offsets over the same `every_n` scheme
  - write:
    - `coarse_metrics.statuses.holdout_repeatability`
    - `vehicle_motion_assessment.holdout_repeatability`
    - `vehicle_motion_assessment.holdout_repeatability_details`
    - `fine_metrics.holdout_repeatability`
    - `fine_metrics.uncertainty_summary`
  - interpretation:
    - `pass`: multiple holdout offsets keep one stable solution family and stable
      holdout ratios
    - `warning`: either different offsets degrade materially or they land in
      different solution families
    - `unknown`: the bag is too small for repeated split replay
- controlled replay on the current wide-map bag now fixes the practical rule:
  - `initial z +10%` still converges back to the trusted solution on this bag, so
    initial-guess tolerance is materially wider than the trusted-reference gate
  - `reference z +5%` can keep strong internal residuals while conflicting in
    vertical height, so trusted reference checks must use absolute + relative
    vertical thresholds, not translation norm alone
  - rough 6DoF initial error (`~0.77 m / ~16 deg`) is still recoverable on this
    bag, so the algorithm itself is not currently failing from initial-value
    sensitivity on validated samples
  - the more structural remaining risk is conversion-side trust in raw-bag TF for
    extraction geometry, not optimizer instability after samples are already fixed
- the converter now also supports an optional **single-step re-extraction loop**
  (`--auto-reextract-if-needed`):
  - trigger only when pass-1 reports `reextract_review` /
    `extraction_consistency=warning`
  - rebuild the standardized samples once using the pass-1 calibrated extrinsics
    as the new extraction seed
  - rerun calibration, preserve both passes, and publish the stronger pass back to
    the stable top-level outputs
  - this closes the loop from metric warning to action without forcing every bag
    into iterative re-extraction by default
- `full_6dof_candidate` now also requires the holdout surface to avoid a warning
  when a holdout split is available; otherwise the run is downgraded to
  `holdout_review`
- repeated holdout now provides the first bag-local uncertainty surface:
  - strong wide-map bag:
    - repeated holdout remains `pass`
    - repeated final-transform spread stays very small
  - weak baseline smoke bag:
    - repeated holdout is `unknown` because the bag is too small
    - this is itself useful industrial evidence: tiny bags should not be mistaken
      for uncertainty-qualified releases
- the immediate crossover from `lidar2lidar` is now explicit:
  - keep stable concise metrics and rich diagnostics separate
  - add repeated-run stability summaries instead of trusting one best run
  - use quality gates to decide promotion, not only optimizer success
- controlled replay after adding `full_prior_robustness` now gives the intended
  industrial separation:
  - strong wide-map bag:
    - `run_initial_equal_reference`
    - `run_initial_rough`
    - both remain `full_6dof_candidate`
    - both have `full_prior_robustness = pass`
  - pure reference-conflict bag:
    - `run_reference_rough_pure`
    - `full_prior_robustness = pass`
    - `trusted_reference_consistency = warning`
    - interpretation: the optimizer basin is stable, but it still conflicts with the
      trusted reference surface
  - weak baseline smoke bag:
    - `run_baseline_smoke_recheck`
    - `planar_basin_stability = warning`
    - `full_prior_robustness = warning`
    - `full_prior_robustness_details.primary_cause = planar_prior_sensitivity`
    - interpretation: the bag is genuinely multi-basin under nearby priors and
      should not be promoted from one lucky solve

### How `yaw_rotation_degenerate` should be interpreted

`yaw_rotation_degenerate` is now only the first layer of the decision. The
metrics output also writes:

- `vehicle_motion_assessment.yaw_diagnostic.evaluation_reliability`
- `vehicle_motion_assessment.yaw_diagnostic.trusted_for_planar_decision`
- `vehicle_motion_assessment.yaw_diagnostic.reliability_limiters`
- `vehicle_motion_assessment.yaw_diagnostic.primary_cause`
- `vehicle_motion_assessment.yaw_diagnostic.recommendations`

This separates two different questions:

1. **Can the current run really judge yaw?**
2. **If yaw is weak, what is the most likely reason and what should be changed?**

The intended interpretation is:

- `evaluation_reliability = high` means the LiDAR motion factors are stable enough
  that yaw weak/strong judgments are meaningful for planar decision making
- `evaluation_reliability != high` means the yaw result is still provisional; keep
  planar DOFs locked, but first improve factor quality before over-interpreting
  the warning
- `primary_cause` distinguishes common cases such as:
  - `repetitive_local_motion`
  - `turn_imbalance_only`
  - `local_pair_objective_too_weak`
  - `local_submap_objective_too_weak`
  - `factor_quality_or_sample_count_limited`

For bags that remain weak after submap accumulation, the next step is a **true
map-based objective**:

- scan-to-map or larger submap-to-map LiDAR factors
- long-horizon accumulation across the full trajectory
- joint optimization where extrinsics are constrained by accumulated structure,
  not just independent local factors

## 7. Latest real-bag validation

Validated on:

- `/home/wfh/01code/apollo-lite/data/bag/record_data_0402`

Generated artifacts:

- `/tmp/lidar2imu-realbag-v2/standardized_samples.yaml`
- `/tmp/lidar2imu-realbag-v2/conversion_diagnostics.yaml`
- `/tmp/lidar2imu-realbag-v2/calibration/metrics.yaml`

### What worked

- conversion completed successfully
- ground sample extraction was stable
- motion sample extraction ran end-to-end
- calibration completed and produced structured diagnostics

### What the metrics say

The bag is still **not a production-quality `x/y/yaw` calibration bag**, but the
data-layer diagnosis is now clearer.

Round summary on `record_data_0402`:

1. **Unfiltered baseline**
   - `ground_normal_angle_p95_deg ≈ 1.74`
   - `motion_rotation_residual_p95_deg ≈ 0.80`
   - `motion_translation_residual_p95_m ≈ 2.32`
   - `left_turn_count = 8`, `right_turn_count = 0`
   - interpretation: one low-quality motion pair can badly distort translation,
     and the bag is one-sided in yaw excitation.
2. **Filtered default (`--min-registration-fitness 0.55`)**
   - one low-fitness motion pair is rejected
   - `motion_rotation_residual_p95_deg ≈ 0.30`
   - `motion_translation_residual_p95_m ≈ 0.43`
   - `motion_registration_fitness_p05 ≈ 0.63`
   - `vehicle_motion_assessment.recommendation = z_roll_pitch_priority`
3. **Stricter filter (`--min-registration-fitness 0.70`)**
   - three motion pairs are rejected, leaving `5` motion samples
   - registration quality improves again, but `right_turn_count` is still `0`
   - recommendation remains `z_roll_pitch_priority`
4. **`gravity-source imu` trial**
   - `ground_selected = 0`
   - most candidate planes flip to about `178 deg` against expected up
   - interpretation: this bag does not provide a stable IMU-gravity extraction
     path, so pose-derived gravity remains the safer default

Interpretation:

- ground constraints are usable when gravity is derived from localization pose
- motion-pair quality filtering is necessary and should stay enabled
- `x/y/yaw` remain weak because the bag only excites left turns
- `z/roll/pitch` are the trustworthy part of the result on this bag
- the evaluation layer now exposes this directly through registration-quality
  gates, turn-balance metrics, and `vehicle_motion_assessment`

### Additional exploratory bag: `/mnt/synology/REDACTED/raw-data/2026-04-13-06-54-28`

This bag contains:

- `/apollo/sensor/seyond/front/PointCloud2`
- `/apollo/localization/pose`
- `/apollo/sensor/gnss/imu`
- `/apollo/sensor/gnss/corrected_imu`

but only these TF edges:

- `world -> imu`
- `world -> localization`

So it does **not** contain a LiDAR-to-IMU static transform.

Exploratory rounds:

1. **Identity fallback + default stride (`5`)**
   - only `1/8` motion pairs survived the registration-fitness gate
   - failure reason: the candidate builder picked very large stride-40 motion
     pairs first, and most registrations fell below the fitness threshold
2. **Identity fallback + local motion pairs (`--motion-frame-stride 1`, `--min-registration-fitness 0.45`)**
   - `8/8` motion pairs survived
   - registration quality became strong: `fitness_p05 ≈ 0.78`,
     `inlier_rmse_p95 ≈ 0.12`
   - joint solve completed, but the final yaw drifted to about `62.7 deg`
3. **Identity fallback + moderate motion pairs (`--motion-frame-stride 2`, `--min-registration-fitness 0.35`)**
   - `8/8` motion pairs survived
   - motion residuals stayed reasonable, but ground height degraded and final
     yaw drifted further to about `78.9 deg`
4. **Corrected IMU gravity trial**
   - `ground_selected = 0`
   - candidate plane normals again flipped to about `178 deg`

Interpretation:

- pose-derived gravity remains the usable choice
- smaller local motion pairs are the right data-layer strategy for this bag
- the best exploratory setting is currently stride `1` with fitness `0.45`
- however, because the bag lacks a true LiDAR-to-IMU prior and only excites one
  turning direction (`right_turn_count = 8`, `left_turn_count = 0`), the final
  extrinsics should **not** be accepted directly

### Weak-planar solver policy validation

The solver now supports:

- `--planar-motion-policy free`
- `--planar-motion-policy freeze_xyyaw`
- `--planar-motion-policy auto`

`auto` currently switches to `freeze_xyyaw` when:

- turn balance is one-sided, or
- the motion-yaw stage is degenerate

Validation on `record_data_0402` with `--min-registration-fitness 0.55`:

1. **Free 6DoF solve**
   - recommendation remains `z_roll_pitch_priority`
   - `delta_to_initial ≈ 2.17 m / 1.30 deg`
   - final yaw drifts to about `-0.63 deg`
2. **Auto planar policy**
   - applied policy: `freeze_xyyaw`
   - recommendation remains `z_roll_pitch_priority`
   - `delta_to_initial ≈ 0.016 m / 1.15 deg`
   - final yaw stays at the initial `0 deg`

Validation on the Synology front-LiDAR bag with the provided
`imu_lidar_front_extrinsics.yaml` prior:

1. **Free 6DoF solve**
   - recommendation remains `recollect_data`
   - `delta_to_initial ≈ 0.343 m / 2.65 deg`
   - final yaw drifts to about `92.5 deg`
2. **Auto planar policy**
   - applied policy: `freeze_xyyaw`
   - recommendation remains `recollect_data`
   - `delta_to_initial ≈ 0.010 m / 0.73 deg`
   - final yaw stays at the initial `90 deg`

Interpretation:

- weak-planar bags should not be allowed to spend `x/y/yaw` freedom just to reduce
  residuals
- the new policy makes that trade-off explicit: keep planar components near the
  trusted prior and continue refining `z/roll/pitch`

The tuning helper `lidar2imu-tune-record` now also sweeps
`--planar-motion-policy` and ranks trials by recommendation, then drift to the
initial prior, then warnings and residuals. This keeps weak-planar free solutions
from outranking conservative prior-preserving runs only because they have a smaller
motion residual.

### Window + gate validation

The converter now uses a windowed motion-selection policy instead of one global
candidate ranking.

Validation on `record_data_0402` with `--min-registration-fitness 0.55`:

- `8` motion windows
- `6` valid windows
- `5` selected motion samples
- first selected candidate uses `frame_stride=10`
- first selected candidate has `pose_rotation_deg ≈ 1.15`,
  `registration_fitness ≈ 0.787`

Validation on the Synology front-LiDAR bag with the provided prior:

- `8` motion windows
- `6` valid windows
- `6` selected motion samples
- first selected candidate uses `frame_stride=2`
- first selected candidate has `pose_rotation_deg ≈ 1.63`,
  `registration_fitness ≈ 0.956`

Interpretation:

- windowing makes the data-layer decision explicit
- gating weak windows reduces the chance that abnormal local segments enter the
  final motion set
- stride-normalized scoring avoids reintroducing the old long-span domination issue

## 8. Current limitations

1. The converter still treats localization pose as the default source of
    gravity and IMU-side motion.
2. The current auto policy is still prior-preserving rather than evidence-growing:
   it freezes `x/y/yaw` instead of proving they are observable.
3. The current bags still lack explicit left/right turn balance.
4. Bags with no LiDAR-to-parent TF still need either an explicit prior or a
    stronger acceptance rule than the current identity fallback.

## 9. Recommended next iteration

1. Keep `--min-registration-fitness 0.55` as the default conversion guard.
2. Add explicit observability gates for one-sided turning before accepting
   planar outputs.
3. Add repeatability checks across multiple bags and across perturbed initial
   transforms.
4. Validate on a bag with stronger left/right turn coverage.
5. When a bag lacks static TF, prefer `--initial-transform` over identity
   fallback before drawing any final calibration conclusion.
