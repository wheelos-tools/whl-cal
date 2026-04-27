---
audience: user
stability: stable
last_tested: 2026-04-27
---

# LiDAR-to-IMU Quick Start

This document is only for running the current pipeline quickly.

## 1. Install

```bash
cd /home/wfh/01code/whl-cal
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Run from standardized samples

If you already have `ground_samples` and `motion_samples`:

```bash
lidar2imu-calibrate \
  --input lidar2imu_samples.yaml \
  --planar-motion-policy auto \
  --output-dir outputs/lidar2imu/run01
```

Outputs:

- `outputs/lidar2imu/run01/calibrated_tf.yaml`
- `outputs/lidar2imu/run01/metrics.yaml`
- `outputs/lidar2imu/run01/calibrated/<parent>_<child>_extrinsics.yaml`
- `outputs/lidar2imu/run01/diagnostics/`

## 3. Run from Apollo record data

Convert one record directory into standardized samples and calibrate directly:

**Fixed baseline profile** (stable comparison reference):

```bash
lidar2imu-convert-record \
  --profile baseline \
  --record-path /home/wfh/01code/apollo-lite/data/bag/record_data_0402 \
  --output-dir outputs/lidar2imu/raw_validation_baseline \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --calibrate
```

**Current production profile** (current mass-production candidate):

```bash
lidar2imu-convert-record \
  --profile production \
  --record-path /home/wfh/01code/apollo-lite/data/bag/record_data_0402 \
  --output-dir outputs/lidar2imu/raw_validation \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --calibrate
```

If the bag does not contain `lidar -> parent` TF, add one of:

```bash
--initial-transform path/to/imu_lidar_front_extrinsics.yaml
```

or, for exploratory-only runs:

```bash
--identity-initial-transform
```

Outputs:

- `outputs/lidar2imu/raw_validation/standardized_samples.yaml`
- `outputs/lidar2imu/raw_validation/conversion_diagnostics.yaml`
- `outputs/lidar2imu/raw_validation/calibration/`
- `outputs/lidar2imu/raw_validation/reextract_summary.yaml` when
  `--auto-reextract-if-needed` is enabled

Profile meaning:

- `baseline`
  - `scan_to_scan`
  - `--planar-motion-policy auto`
  - no automatic second-pass re-extraction
  - use this as the stable regression / comparison reference
- `production`
  - `submap_to_map`
  - `submap_half_window=2`
  - `submap_support_stride=5`
  - `submap_min_support_frames=3`
  - `map_half_window=6`
  - `map_support_stride=10`
  - `map_min_support_frames=5`
  - `--planar-motion-policy auto`
  - `--auto-reextract-if-needed`
  - use this as the current map-side production candidate

## 4. Check the most important results first

Open:

- `conversion_diagnostics.yaml`
- `calibration/metrics.yaml`

Start with these fields:

- `coarse_metrics.ground_normal_angle_p95_deg`
- `coarse_metrics.motion_rotation_residual_p95_deg`
- `coarse_metrics.motion_translation_residual_p95_m`
- `coarse_metrics.motion_registration_fitness_p05`
- `coarse_metrics.turn_balance_ratio`
- `coarse_metrics.left_turn_count`
- `coarse_metrics.right_turn_count`
- `coarse_metrics.statuses`
- `vehicle_motion_assessment.recommendation`
- `vehicle_motion_assessment.extraction_consistency`
- `vehicle_motion_assessment.full_prior_robustness`
- `vehicle_motion_assessment.holdout_repeatability`
- `vehicle_motion_assessment.holdout_generalization`
- `vehicle_motion_assessment.initial_prior_assessment`
- `vehicle_motion_assessment.applied_solver_planar_motion_policy`
- `vehicle_motion_assessment.yaw_observability_reasons`
- `summary.solver_policy`
- `summary.run_profile`
- `summary.dataset_partition`
- `summary.delta_to_extraction`
- `summary.delta_to_reference`
- `summary.delta_to_initial`
- `fine_metrics.holdout_validation`
- `fine_metrics.reference_consistency`
- `fine_metrics.extraction_consistency`
- `fine_metrics.full_prior_robustness`
- `fine_metrics.initial_prior_assessment`
- `fine_metrics.holdout_repeatability`
- `fine_metrics.uncertainty_summary`
- `fine_metrics.algorithm_stages.motion_rotation.observability.cost_scan.max_cost_ratio`
- `fine_metrics.algorithm_stages.motion_rotation.observability.cost_scan.within_5pct_span_deg`
- `fine_metrics.motion.translation_heading_span_deg`
- `fine_metrics.motion.imu_rotation_axis_abs_mean_xyz`
- `fine_metrics.motion.selected_frame_strides`
- `conversion_diagnostics.motion_selection.strategy`
- `conversion_diagnostics.motion_selection.registered_candidate_count`
- `conversion_diagnostics.motion_selection.selected_translation_heading_span_deg`

For multi-run comparison, generate a fixed review report:

```bash
lidar2imu-review-runs \
  outputs/lidar2imu/run01 \
  outputs/lidar2imu/run02 \
  --output-dir outputs/lidar2imu/review_report
```

This writes:

- `review_summary.csv`
- `review_report.html`

The HTML report fixes the same industrial acceptance surfaces in one place:

- recommendation / solver policy
- extraction consistency
- trusted-reference consistency
- planar basin stability
- full 6DoF prior robustness
- repeated holdout stability
- holdout generalization
- yaw cost ratio / plateau
- residual and registration quality

## 5. Current interpretation rule

- If ground metrics are good but motion metrics are weak, trust `z/roll/pitch`
  more than `x/y/yaw`.
- If `left_turn_count` or `right_turn_count` is missing, treat `x/y/yaw` as
  weakly observable.
- If `vehicle_motion_assessment.recommendation` is `z_roll_pitch_priority`,
  treat the run as vertically trustworthy but planarly under-excited.
- On weak-planar bags, prefer `--planar-motion-policy auto`; it will lock
  `x/y/yaw` to the initial prior when turn balance or yaw observability is weak,
  while continuing to refine `z/roll/pitch`.
- If `yaw_observability_reasons` contains `flat_cost_scan` or
  `wide_cost_plateau`, treat `x/y/yaw` as weak **even if** turn balance looks
  acceptable.
- If `translation_heading_span_deg` is very small and
  `imu_rotation_axis_abs_mean_xyz.z` is near `1.0`, the selected motion windows
  are collapsing to near-single-direction planar snippets; improving the driving
  pattern alone may not help unless the selected windows become more diverse.
- If `conversion_diagnostics.motion_selection.strategy` is `global_diversity`,
  read `registered_candidate_count` and `selected_translation_heading_span_deg`
  first. They tell you whether the converter recovered a richer long-horizon
  motion pool before the solver even starts.
- If global-diversity selection already gives wide stride / heading coverage but
  `max_cost_ratio` is still low and `within_5pct_span_deg` is still wide, do not
  keep tuning short-window thresholds. That is the signal to move to a map-based
  objective.
- If `motion_registration_mode` is `submap_to_submap`, compare it against the
  scan-to-scan baseline using the same metrics. A useful submap run should
  usually increase `registered_candidate_count`, reduce motion residuals, and
  shrink the yaw cost plateau.
- If `motion_registration_mode` is `submap_to_map`, treat `map_half_window`,
  `map_support_stride`, and `map_min_support_frames` as the main knobs that
  control how much long-horizon structure reaches the yaw stage.
- If `conversion_diagnostics.motion_selection.selected_frame_strides` collapses to
  only one stride, or `selected_heading_bin_count` stays very small, the selector
  is still falling into one dominant local mode even if registration quality
  looks good.
- If `submap_to_submap` improves `max_cost_ratio` and residuals but
  `weak_planar_reasons` still contains `yaw_rotation_degenerate`, treat that as
  progress, not failure. It means the bag benefits from local accumulation, but
  still needs a more global map objective before `x/y/yaw` can be trusted.
- Read `vehicle_motion_assessment.yaw_diagnostic` before deciding whether a
  `yaw_rotation_degenerate` warning is a real geometric conclusion or just a
  provisional warning from weak factor quality.
- If `yaw_diagnostic.trusted_for_planar_decision` is `true`, the current bag is
  informative enough to trust the weak/strong yaw judgment itself.
- If `yaw_diagnostic.trusted_for_planar_decision` is `false`, read
  `yaw_diagnostic.reliability_limiters` first and fix those before acting on the
  yaw result.
- Use `yaw_diagnostic.primary_cause` + `yaw_diagnostic.recommendations` as the
  shortest path to the next action:
  - `repetitive_local_motion`: widen heading diversity or move from local pairs
    to submaps / maps
  - `turn_imbalance_only`: collect both left and right turns
  - `local_pair_objective_too_weak`: move from scan-to-scan to submap-to-submap
  - `local_submap_objective_too_weak`: move to scan-to-map / submap-to-map
- If `motion_registration_mode=submap_to_map` and the run becomes
  `full_6dof_candidate`, treat that as a **candidate promotion**, not final
  acceptance. Re-check repeatability under nearby map-size settings, drift
  relative to the trusted prior, and behavior on at least one comparison bag.
- Use `--profile baseline` as the fixed regression reference when comparing new
  map-side ideas. It is intentionally conservative and should stay stable.
- Use `--profile production` for the current release candidate. It keeps the
  same extraction/reference/basin/holdout acceptance framework but uses the
  current validated `submap_to_map` settings.
- If `summary.dataset_partition.holdout_enabled=true`, the solver kept every 3rd
  selected motion sample as a holdout check by default and solved only on the
  calibration subset.
- In that case, read `vehicle_motion_assessment.holdout_generalization` and
  `fine_metrics.holdout_validation` before accepting any free-planar result.
- If `holdout_generalization=warning`, do not promote the run as production-ready
  even if extraction/reference/basin surfaces pass. That means the planar result
  still degrades materially on held-out motion factors.
- Read `vehicle_motion_assessment.full_prior_robustness` together with
  `fine_metrics.full_prior_robustness`:
  - `pass`: nearby planar + vertical + roll/pitch prior perturbations still converge
    to one stable accepted basin
  - `warning`: at least one nearby prior family reaches a different basin, so do not
    treat one accepted solve as an industrially stable result yet
- Read `vehicle_motion_assessment.holdout_repeatability` together with
  `fine_metrics.holdout_repeatability`:
  - `pass`: multiple holdout offsets keep one stable solution family and stable
    held-out ratios
  - `warning`: different holdout offsets either degrade materially or converge to
    different solution families
  - `unknown`: the current bag has too few motion samples for repeated holdout
- Read `fine_metrics.uncertainty_summary` as the first released uncertainty surface:
  - it summarizes translation / rotation spread from repeated holdout replays
  - use it as a bag-local stability estimate, not as a final cross-bag production
    confidence interval
- If `planar_basin_stability=pass` but `full_prior_robustness=warning`, the run is
  no longer failing on small `x/y/yaw` perturbations only; it is still sensitive to
  a wider 6DoF prior neighborhood.
- If `holdout_generalization=pass` but `holdout_repeatability=warning`, do not trust
  one deterministic holdout split; that pattern means the map-side result is still
  split-sensitive.
- If `full_prior_robustness_details.primary_cause=planar_prior_sensitivity`, prefer
  stronger map-side constraints or trusted-reference-aware basin selection before
  widening free 6DoF release.
- Read `vehicle_motion_assessment.initial_prior_assessment` together with
  `summary.delta_to_initial`:
  - `pass`: the provided TF is already inside the nominal production range for this
    run
  - `recoverable`: the provided TF is outside the nominal range, but this bag still
    converged back to the accepted basin
  - `warning`: the provided TF is outside the nominal range and this run also has
    an acceptance conflict; treat that TF as suspicious
- If `extraction_consistency=warning`, the current final solve has already moved
  too far from the transform used to export the ground/motion samples. Treat that
  as a **re-extraction signal**, not as a solver-only issue.
- For that case, read `delta_to_extraction` and
  `extraction_consistency_recommendations`, then re-run record conversion with the
  refined transform as the extraction transform before making a final decision.
- Read `delta_to_extraction.translation_xyz_m.z` as well as the total translation
  norm. A pure height error can now trigger extraction review even when the
  total translation norm is not the dominant signal.
- `lidar2imu-convert-record --calibrate --auto-reextract-if-needed` now automates
  exactly one such second pass, stores both passes under
  `reextract_pass1/` and `reextract_pass2/`, writes the comparison to
  `reextract_summary.yaml`, and promotes the stronger pass back to the stable
  top-level outputs.
- If `initial_transform_source` and `extraction_transform_source` differ, the run
  used a separate trusted transform for record-side extraction and the provided
  transform only as the calibration initial guess. Prefer this pattern whenever
  the bag already contains a trusted in-record TF.
- If `trusted_reference_consistency=warning`, do **not** accept the run as a
  successful free-planar result even when yaw observability, registration, and
  solver policy all look good. That pattern means the optimizer found an
  internally consistent basin that still conflicts with the trusted in-bag TF.
- For that case, read `delta_to_reference` and
  `reference_consistency_recommendations` before making any calibration decision.
- In particular, check `delta_to_reference.translation_xyz_m.z`. Controlled replay
  on the current wide-map bag shows that a wrong trusted `z` can keep strong
  internal residuals, so vertical drift must be judged explicitly instead of only
  by total translation norm.
- Do not use percentage-only trusted-height rules. The repo now keeps both an
  absolute vertical threshold and a relative vertical ratio because a pure `z`
  bias can sit near a 5% heuristic while still slipping under a norm-only gate.
- If `initial_prior_assessment=recoverable`, do not immediately trust the same user
  TF on weaker bags. It means this bag recovered, not that the prior is generally
  safe for production.
- If the run needed `--identity-initial-transform`, do not accept the final
  extrinsics directly unless another trusted prior or cross-bag check agrees.
- If `motion_translation_residual_p95_m` is large, inspect motion pair quality
  in `conversion_diagnostics.yaml` before changing the solver.
- If `gravity-source imu` yields almost no accepted ground samples, fall back to
  pose-derived gravity for that bag.
