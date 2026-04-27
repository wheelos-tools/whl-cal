# lidar2imu current context

## 1. Working pattern

`lidar2imu` should continue to follow the same split used elsewhere in this repo:

1. **Data layer**: convert record data into standardized `ground_samples` and
   `motion_samples`
2. **Algorithm layer**: staged solve for `roll/pitch`, `z`, `yaw`, translation,
   then joint refinement
3. **Evaluation layer**: keep coarse/fine metrics stable while iterating the
   converter and solver

## 2. Current findings on `record_data_0402`

### Round 1: unfiltered baseline

- `12` ground samples, `8` motion samples
- `motion_translation_residual_p95_m ≈ 2.32`
- `motion_rotation_residual_p95_deg ≈ 0.80`
- `left_turn_count = 8`, `right_turn_count = 0`

Interpretation:

- the bag runs end-to-end
- one low-fitness motion pair can strongly distort translation
- the bag is one-sided in planar excitation

### Round 2: filtered default (`--min-registration-fitness 0.55`)

- rejected `1` low-fitness motion pair
- kept `7` motion samples
- `motion_registration_fitness_p05 ≈ 0.634`
- `motion_translation_residual_p95_m ≈ 0.431`
- `motion_rotation_residual_p95_deg ≈ 0.299`
- `vehicle_motion_assessment.recommendation = z_roll_pitch_priority`

Interpretation:

- the data-layer filter is necessary and should remain on by default
- `z/roll/pitch` are usable
- `x/y/yaw` are still weak because turn balance is still zero

### Round 3: IMU gravity trial

- `ground_selected = 0`
- most candidate ground planes flip to about `178 deg` against expected up

Interpretation:

- this bag is not suitable for `gravity-source imu`
- pose-derived gravity should stay the default for now

### Round 4: stricter filter (`--min-registration-fitness 0.70`)

- rejected `3` motion pairs
- kept `5` motion samples
- registration quality improves again
- recommendation is still `z_roll_pitch_priority`

Interpretation:

- stricter filtering cleans the motion set
- but filtering alone cannot recover `x/y/yaw` without right-turn excitation

## 3. Current recommendation

- keep `--min-registration-fitness 0.55` as the default converter guard
- keep pose-derived gravity as the default
- on `record_data_0402`, trust `z/roll/pitch` first
- do not accept `x/y/yaw` from this bag as a production conclusion

## 4. Additional bag: `/mnt/synology/raw-data/2026-04-13-06-54-28`

### Bag structure

- single LiDAR topic: `/apollo/sensor/seyond/front/PointCloud2`
- pose + IMU topics are present
- TF only contains `world -> imu` and `world -> localization`
- there is **no** `lidar_front -> imu` static transform in the bag

### Round 1: identity fallback + default stride

- needed `--identity-initial-transform` just to start
- with `motion-frame-stride 5` and `min-registration-fitness 0.55`, only
  `1/8` motion pairs survived

Interpretation:

- the default motion-pair ranking was too aggressive for this bag
- large stride-40 pairs produced weak LiDAR-LiDAR registration

### Round 2: identity fallback + stride 1 + fitness 0.45

- `8/8` motion pairs survived
- `motion_registration_fitness_p05 ≈ 0.781`
- `motion_translation_residual_p95_m ≈ 0.048`
- `joint_condition_number ≈ 121`
- but `right_turn_count = 8`, `left_turn_count = 0`
- final yaw drifted to about `62.7 deg`

Interpretation:

- this is the best **data-layer** configuration found so far
- but the final extrinsics are still not trustworthy because the bag has no
  trusted prior and the motion is one-sided

### Round 3: identity fallback + stride 2 + fitness 0.35

- `8/8` motion pairs survived
- motion residuals remained acceptable
- ground height residual degraded
- final yaw drifted further to about `78.9 deg`

Interpretation:

- stride 2 is worse than stride 1 for this bag

### Round 4: corrected IMU gravity

- `ground_selected = 0`
- ground normals flipped to about `178 deg`

Interpretation:

- corrected IMU gravity is not usable on this bag
- pose-derived gravity remains the safer choice

## 5. Current recommendation

- keep `record_data_0402` as the main lidar2imu reference bag for now
- for `/mnt/synology/raw-data/2026-04-13-06-54-28`, the best exploratory setup is:
  - `--identity-initial-transform`
  - `--motion-frame-stride 1`
  - `--min-registration-fitness 0.45`
- but treat that run as **diagnostic only**, not a final accepted extrinsic
- if a trusted `imu_lidar_front` prior becomes available, rerun the same bag
  with `--initial-transform` before using it for acceptance

## 6. Next iteration

1. Re-run the same pipeline on a bag with both left and right turns
2. Compare repeatability of `z/roll/pitch` across multiple bags before changing
   defaults again
3. Add explicit perturbation / repeatability tests around the initial transform
4. For bags like the Synology front-LiDAR set, prefer a real
   `--initial-transform` file over identity fallback

## 7. 2026-04-18 weak-planar solver policy

`lidar2imu` now has a solver-side planar-motion policy:

- `--planar-motion-policy free`: keep the previous fully free 6DoF solve
- `--planar-motion-policy freeze_xyyaw`: lock `x/y/yaw` to the initial prior
- `--planar-motion-policy auto`: if the bag shows one-sided turning or degenerate
  yaw observability, automatically switch to `freeze_xyyaw`

The current auto trigger is:

- `turn_imbalance`, or
- `yaw_rotation_degenerate`

### Validation on `record_data_0402`

- `free`:
  - recommendation remains `z_roll_pitch_priority`
  - `delta_to_initial ≈ 2.17 m / 1.30 deg`
  - final yaw drifts to about `-0.63 deg`
  - `motion_translation_residual_p95_m ≈ 0.420`
- `auto`:
  - applied solver policy: `freeze_xyyaw`
  - recommendation remains `z_roll_pitch_priority`
  - `delta_to_initial ≈ 0.016 m / 1.15 deg`
  - final yaw stays at the initial `0 deg`
  - `motion_translation_residual_p95_m ≈ 0.443`

Interpretation:

- on this bag, auto mode gives up a small amount of motion residual quality but
  prevents the solver from inventing a large planar offset on a one-sided-turn bag

### Validation on `/mnt/synology/raw-data/2026-04-13-06-54-28`

Using:

- `--initial-transform lidar2lidar/conf/imu_lidar_front_extrinsics.yaml`
- `--motion-frame-stride 1`
- `--min-registration-fitness 0.45`
- pose-derived gravity

Results:

- `free`:
  - recommendation remains `recollect_data`
  - `delta_to_initial ≈ 0.343 m / 2.65 deg`
  - final yaw drifts to about `92.5 deg`
- `auto`:
  - applied solver policy: `freeze_xyyaw`
  - recommendation remains `recollect_data`
  - `delta_to_initial ≈ 0.010 m / 0.73 deg`
  - final yaw stays at the initial `90 deg`

Interpretation:

- the Synology bag is still diagnostic-only because turn balance is still zero
- but auto mode now keeps the weakly observable planar components near the trusted
  prior instead of drifting to a prettier-but-less-trustworthy free solution

### Tuning framework update

`lidar2imu-tune-record` now sweeps `--planar-motion-policy` and ranks trials by:

1. status
2. recommendation
3. drift to the initial prior
4. warning count
5. residual / registration quality

This prevents the tuning summary from selecting a weak-planar free solution only
because it has a smaller residual while drifting much farther from the prior.

## 8. 2026-04-18 motion window + gate

The motion data layer now uses **window + gate** instead of one global candidate
ranking.

### Current strategy

1. enumerate motion candidates from multiple strides
2. assign candidates into timeline windows
3. summarize each window by candidate count and best pose-side excitation
4. gate weak windows before registration
5. inside each valid window:
   - prefer candidates whose pose-side rotation meets the minimum motion-rotation threshold
   - normalize candidate score by stride so very long spans do not dominate
   - try top-k candidates and keep the first one that passes registration fitness

This is meant to avoid two old failure modes:

- one very long span dominating all selected motion pairs
- locally bad windows leaking abnormal data into the final motion set

### Validation on `record_data_0402`

- `8` motion windows
- `6` valid windows
- `5` selected motion samples
- first selected window uses `frame_stride=10`
- first selected candidate: `pose_rotation_deg ≈ 1.153`, `registration_fitness ≈ 0.787`
- final auto solve still gives `z_roll_pitch_priority`
- `delta_to_initial ≈ 0.017 m / 1.22 deg`

Interpretation:

- compared with the earlier long-span preference, the new windowed logic now keeps
  the accepted motion set local enough to preserve angular excitation without
  letting large overlapping stride-40 pairs dominate selection

### Validation on `/mnt/synology/raw-data/2026-04-13-06-54-28`

- `8` motion windows
- `6` valid windows
- `6` selected motion samples
- first selected window uses `frame_stride=2`
- first selected candidate: `pose_rotation_deg ≈ 1.631`, `registration_fitness ≈ 0.956`
- final auto solve still gives `recollect_data`
- `delta_to_initial ≈ 0.008 m / 0.56 deg`

Interpretation:

- the bag is still diagnostic-only because turn balance is still one-sided
- but window + gate now selects more local and cleaner motion windows, which keeps
  the final solve tighter to the trusted prior

## 9. 2026-04-23 8-shape bag investigation: `/mnt/synology/raw-data/lidar2imu`

This bag was explicitly collected as an `8`-shape run, so the repeated
`x/y/yaw` weakness needed a deeper check than the earlier "not enough turn
balance" explanation.

### Verified bag-level result

- the bag contains `/tf_static`, and the trusted in-bag prior is:
  - `imu -> lslidar_main`
  - translation `≈ [0.0, 1.508, 1.941]`
  - rotation `≈ identity`
- this is **not** the same as the checked-in `lidar2lidar/conf/imu_lslidar_main_extrinsics.yaml`
  (`[0.0, 1.05, 0.8]`), which differs by about `1.229 m`
- using pose-derived gravity and the in-bag TF:
  - `auto` stays within about `0.005 m / 0.36 deg` of the in-bag TF
  - `free` drifts by about `2.07 m / 20.48 deg`

Interpretation:

- for this bag, the in-bag TF is self-consistent and should stay the planar prior
- the free 6DoF solve is inventing a planar correction that the bag does not
  support strongly enough

### Why an 8-shape bag still looked weak in yaw

The key finding is that the **global trajectory** is an `8`, but the
**motion samples that actually reach the solver** are still almost pure
single-axis planar motions.

Validated on the selected `11` motion samples from
`outputs/lidar2imu/20260422_lidar2imu_direct_pose_auto/standardized_samples.yaml`:

- pose-derived motion rotation axes:
  - `mean_abs_axis_xyz ≈ [0.0076, 0.0103, 0.99986]`
  - `min_abs_axis_z ≈ 0.99949`
  - `max_abs_axis_xy ≈ 0.031`
- the selected pose deltas have:
  - yaw range `≈ [-8.68 deg, 7.42 deg]`
  - pitch range `≈ [-0.185 deg, 0.094 deg]`
  - roll range `≈ [-0.152 deg, 0.093 deg]`
- the selected local translation directions in the horizontal plane collapse to:
  - angle range `≈ [-95.8 deg, -86.9 deg]`
  - meaning the chosen samples are almost all "forward local motion" with very
    similar direction

Cross-check on raw IMU and corrected IMU over the **same selected windows**:

- raw `/apollo/sensor/gnss/imu` integrated angular increments:
  - `mean_abs_axis_xyz ≈ [0.0075, 0.0120, 0.99985]`
- corrected `/apollo/sensor/gnss/corrected_imu` integrated angular increments:
  - `mean_abs_axis_xyz ≈ [0.0075, 0.0120, 0.99985]`

Interpretation:

- this is **not only** a localization-pose smoothing artifact
- on the motion windows actually selected for calibration, all three sources
  (`pose`, raw IMU, corrected IMU) see nearly the same "almost pure z-axis"
  rotation pattern
- an `8`-shape on flat ground improves turn balance, but it still does not
  automatically produce the non-coaxial rotation family that makes hand-eye yaw
  strongly observable

### Motion selection bias that matters for this bag

The current motion data layer does contain larger-span candidates:

- stride `5`: `rot_max ≈ 8.68 deg`, `score_max ≈ 17.45`
- stride `10`: `rot_max ≈ 17.11 deg`, `score_max ≈ 17.20`
- stride `20`: `rot_max ≈ 33.47 deg`, `score_max ≈ 16.82`
- stride `40`: `rot_max ≈ 62.18 deg`, `score_max ≈ 15.62`

But the current window ranking still selects:

- `12/12` windows with **best stride = 5**

because candidate score is currently:

- `(rotation_deg * 10.0 + translation_m) / stride`

Interpretation:

- the bag does contain longer and more excited arcs
- but the current data layer strongly prefers shorter local windows once the
  score is normalized by stride
- on this bag, that means the solver is fed a clean but geometrically repetitive
  set of local planar snippets instead of a more diverse motion family

### Yaw cost scan result

For the selected motion set, the motion-rotation objective was scanned across
`yaw ∈ [-180 deg, 180 deg]`.

Observed result:

- best cost at about `139.5 deg`
- `max_cost / min_cost ≈ 1.0186`
- **every sampled yaw** stayed within `5%` of the best value

Interpretation:

- under the current motion set and current formulation, the yaw cost is almost
  flat
- this is direct evidence that yaw is weakly constrained by the selected motion
  equations, even though turn balance is no longer one-sided

### Important implementation issue in the current observability logic

There is also a **logic problem** in the current yaw observability gate:

- `solve_yaw_from_motion()` optimizes a **single scalar** yaw parameter
- `observability_from_matrix()` marks a matrix as degenerate whenever fewer than
  two positive singular values are available, because the condition number is
  then set to `inf`
- for any nonzero `N x 1` Jacobian, this means:
  - `rank = 1`
  - `condition_number = inf`
  - `degenerate = true`

Validated directly:

- random `(10 x 1)` matrix -> `rank=1`, `condition=inf`, `degenerate=true`
- random `(20 x 1)` matrix -> `rank=1`, `condition=inf`, `degenerate=true`
- random `(10 x 2)` matrix -> finite condition number, `degenerate=false`

Interpretation:

- the current `yaw_rotation_degenerate` warning is at least partly structural:
  the yaw stage Jacobian is single-column by design, so the current
  observability helper will mark it degenerate even when the scalar sensitivity
  is nonzero
- this likely makes `full_6dof_candidate` effectively unreachable in the current
  lidar2imu evaluation path
- the repeated weak-yaw conclusion on recent bags is therefore a mix of:
  1. genuinely weak selected motion geometry
  2. an overly strict / mismatched single-parameter observability test

### Current recommendation after this investigation

- keep pose-derived gravity as the default on this bag
- keep the in-bag `/tf_static` transform as the planar prior
- accept that this bag can still refine `z/roll/pitch`
- do **not** interpret the repeated `yaw_rotation_degenerate` warning as purely
  a data-collection failure until the yaw observability logic is fixed
- next iteration should focus on:
  1. revising the yaw observability check for single-parameter problems
  2. adding diagnostics for selected motion-axis diversity and yaw cost sharpness
  3. testing a motion selection variant that preserves more stride diversity on
     bags with clean long-arc registration

### 2026-04-23 implementation update and multi-bag validation

The observability logic has now been revised and revalidated against real bags.

Implemented changes:

- `observability_from_matrix()` is now **expected-rank aware**
- single-parameter stages (`yaw` solve, and locked-axis translation / joint stages)
  are no longer marked degenerate purely because their Jacobian is `N x 1`
- `solve_yaw_from_motion()` now also computes a periodic yaw cost scan:
  - `max_cost_ratio`
  - `within_5pct_span_deg`
  - `scalar_sensitivity`
  - `reasons`
- motion diagnostics now surface:
  - `imu_rotation_axis_abs_mean_xyz`
  - `translation_heading_span_deg`
  - `selected_frame_strides`

Validated on three datasets:

1. Current `8`-shape bag: `outputs/lidar2imu/20260423_pose_auto_observability`

- `motion_rotation.observability.rank = 1`, `expected_rank = 1`,
  `degenerate = true`
- `scalar_sensitivity ≈ 0.00246`
- `max_cost_ratio ≈ 1.0189`
- `within_5pct_span_deg = 359.5`
- `reasons = [flat_cost_scan, wide_cost_plateau]`
- `auto` still applies `freeze_xyyaw`
- final result stays close to the in-bag prior:
  `delta_to_initial ≈ 0.0123 m / 0.362 deg`

Interpretation:

- after removing the structural `N x 1` false positive, this bag is **still**
  weak in yaw
- the reason is now explicit and data-backed: the yaw cost surface is nearly flat
- this confirms the root cause is mainly **selected motion geometry**, not only
  implementation logic

2. `record0402`: `outputs/lidar2imu/20260423_record0402_auto_observability`

- `motion_rotation.observability.rank = 1`, `expected_rank = 1`,
  `degenerate = false`
- `scalar_sensitivity ≈ 0.0111`
- `max_cost_ratio ≈ 10.26`
- `within_5pct_span_deg ≈ 16 deg`
- `reasons = []`
- `auto` still applies `freeze_xyyaw`, but now the reason is
  `turn_imbalance`, not a structural yaw-observability failure

3. Synology front bag:
   `outputs/lidar2imu/20260423_synology_round2_observability`

- `motion_rotation.observability.rank = 1`, `expected_rank = 1`,
  `degenerate = false`
- `scalar_sensitivity ≈ 0.0213`
- `max_cost_ratio ≈ 8.78`
- `within_5pct_span_deg ≈ 18 deg`
- `reasons = []`
- `auto` still applies `freeze_xyyaw` because the bag remains one-sided in turn
  balance

Updated conclusion:

- the earlier `yaw_rotation_degenerate` pattern was partly real and partly a
  single-parameter observability bug
- that structural bug is now removed
- after the fix, the current `8`-shape bag still fails yaw observability because
  its selected motion set produces a nearly flat yaw objective
- the comparison bags now separate cleanly:
  - **flat yaw cost + wide plateau** -> genuine weak yaw
  - **sharp yaw cost but one-sided turns** -> yaw stage itself is informative, but
    the bag is still not safe for free planar release under current policy

### 2026-04-23 global-diversity motion selection validation

To preserve more of the global `8`-shape information, the record-conversion layer
was updated from "pick one best local candidate per window" to a two-stage policy:

1. window-level **multi-scale preselection**
2. registration-gated **global diversity selection**

The new conversion diagnostics now write:

- `motion_selection.strategy = global_diversity`
- `motion_registered_candidate_count`
- `motion_selection.selected_frame_strides`
- `motion_selection.selected_translation_heading_span_deg`

#### Current `8`-shape bag

Validated on:

- `outputs/lidar2imu/20260423_pose_auto_globaldiv_v2`

Observed changes relative to the previous short-window selection:

- selected strides changed from only `5` to `{10, 20, 40}`
- selected heading span increased from `≈ 8.9 deg` to `≈ 42.3 deg`
- turn balance improved from `≈ 0.57` to `0.8`
- motion angular excitation `p95` increased from `≈ 8.47 deg` to `≈ 52.62 deg`
- registered candidate pool now contains `19` motion factors before final selection
- yaw cost ratio improved from `≈ 1.0189` to `≈ 1.1455`
- yaw `5%` plateau shrank from `359.5 deg` to `143 deg`

Interpretation:

- the new data layer is **meaningfully better**
- it preserves long-span motion factors and wider horizontal direction coverage
- but the resulting yaw cost is still too flat for production-grade free planar
  solving
- this means the `8`-shape bag was not only suffering from local-window collapse;
  even after recovering more global diversity, the current scan-to-scan objective
  is still not strong enough to fully constrain yaw

#### `record0402` comparison bag

Validated on:

- `outputs/lidar2imu/20260423_record0402_globaldiv_v2`

Observed behavior:

- selected strides became `{10, 20, 40}`
- selected heading span is `≈ 95.6 deg`
- registered candidate pool contains `24` motion factors before final selection
- yaw cost ratio remains strong at `≈ 11.39`
- yaw `5%` plateau remains narrow at `≈ 15.5 deg`
- final solution stays close to the trusted prior:
  `delta_to_initial ≈ 0.0154 m / 1.11 deg`
- `auto` still freezes planar DOFs because the bag remains one-sided in turn
  balance, not because yaw is structurally weak

Interpretation:

- the new global-diversity selection does **not** break the reference bag behavior
- the improved conversion logic preserves the earlier distinction:
  - strong yaw stage + poor turn balance -> keep planar lock
  - weak yaw stage even after diversity recovery -> current objective is still not
    enough

#### Industrial conclusion after this round

- global-diversity motion selection is a valid and useful production improvement
- however, it is still an **intermediate step**, not the final industrial answer
- for the current `8`-shape bag, industrial-grade `x/y/yaw` calibration now
  clearly requires a **long-horizon map-based objective**
- the next main path should be:
  1. scan-to-map / submap-to-submap lidar factors
  2. long-horizon joint optimization of extrinsics against accumulated structure
  3. validation by repeatability, map thickness, and trusted-prior drift rather
     than only local pair residuals

### 2026-04-23 submap-to-submap prototype validation

To validate whether the remaining weakness was caused by single-scan factor
noise, the converter was extended with a new motion registration mode:

- `motion_registration_mode = submap_to_submap`

For each motion candidate, the converter now:

1. takes the start and end LiDAR frames as anchors
2. uses pose + current lidar-to-imu prior to place nearby scans into each anchor
   LiDAR frame
3. builds two local submaps
4. registers submap-to-submap instead of scan-to-scan

This keeps the same output surfaces (`standardized_samples.yaml`,
`conversion_diagnostics.yaml`, `metrics.yaml`) while changing only the LiDAR
factor construction.

#### Current `8`-shape bag

Validated on:

- `outputs/lidar2imu/20260423_pose_auto_submap_v1`

Observed changes relative to `outputs/lidar2imu/20260423_pose_auto_globaldiv_v2`:

- registered candidate pool increased from `19` to `36`
- selected motion samples increased from `10` to `12`
- selected heading span increased from `≈ 42.3 deg` to `≈ 58.6 deg`
- motion registration fitness `p05` improved from `≈ 0.623` to `≈ 0.673`
- motion rotation residual `p95` improved from `≈ 1.02 deg` to `≈ 0.63 deg`
- motion translation residual `p95` improved from `≈ 0.45 m` to `≈ 0.13 m`
- yaw cost ratio improved from `≈ 1.1455` to `≈ 1.5692`
- yaw `5%` plateau shrank from `143 deg` to `68.5 deg`
- final solution still stayed close to the trusted prior:
  `delta_to_initial ≈ 0.0100 m / 0.293 deg`

Important nuance:

- the final selected factors collapsed to stride `40`, which means the stronger
  submap factors dominated the global selector
- turn balance dropped from `0.8` to `≈ 0.57`
- solver policy still applied `freeze_xyyaw`
- weak planar reason remained `yaw_rotation_degenerate`

Interpretation:

- this is a **real improvement**, not noise
- local accumulation reduces registration noise and increases yaw curvature on
  the real `8`-shape bag
- but even after submap accumulation, the current formulation still does not
  produce a strong enough yaw stage to justify free planar release
- therefore the remaining gap is no longer "better local pair choice"; it is
  "more global map structure"

#### `record0402` comparison bag

Validated on:

- `outputs/lidar2imu/20260423_record0402_submap_v1`

Observed changes relative to `outputs/lidar2imu/20260423_record0402_globaldiv_v2`:

- registered candidate pool increased from `24` to `33`
- selected motion samples increased from `9` to `10`
- selected heading span increased from `≈ 95.6 deg` to `≈ 135.1 deg`
- motion registration fitness `p05` improved from `≈ 0.681` to `≈ 0.740`
- motion rotation residual `p95` improved from `≈ 0.44 deg` to `≈ 0.24 deg`
- motion translation residual `p95` improved from `≈ 0.54 m` to `≈ 0.30 m`
- yaw stayed strongly observable:
  - cost ratio `≈ 11.39 -> 7.73`
  - `5%` plateau `≈ 15.5 deg -> 19.5 deg`
- final solution remained close to the trusted prior:
  `delta_to_initial ≈ 0.0164 m / 1.18 deg`

Interpretation:

- `submap_to_submap` improves factor quality on the reference bag too
- it does not change the main diagnosis
- planar DOFs still stay locked because this bag is one-sided in turn balance,
  not because yaw is structurally weak

#### Updated industrial conclusion

- `submap_to_submap` is a useful industrial **intermediate mode**
- it is better than pure scan-to-scan on both the weak `8`-shape bag and the
  reference bag
- however, it is still not the final industrial answer for free `x/y/yaw`
- the next step should be a true long-horizon map objective where extrinsics are
  optimized against accumulated trajectory structure, not only against
  independently registered local factors

### 2026-04-23 coverage-aware submap selection validation

After `submap_to_submap` was working, the next practical issue was that the
current `8`-shape bag still selected only stride `40` factors. That meant the
submap factor construction had improved, but the global selector was still
collapsing to one dominant local mode.

To follow more industrial keyframe-selection practice, the global score was
updated to reward:

- new stride coverage
- new heading-bin coverage
- underrepresented turn direction
- non-repetitive rotation-axis patterns

and to penalize repeated same-stride / same-heading local factors.

#### Current `8`-shape bag

Validated on:

- `outputs/lidar2imu/20260423_pose_auto_submap_v2_balanced`

Observed changes relative to `20260423_pose_auto_submap_v1`:

- selected strides changed from only `{40}` to `{20, 40}`
- selected turn counts became `left=4`, `right=7`
- selected heading bins are still only `2`
- heading span stayed `≈ 58.6 deg`
- yaw cost ratio improved from `≈ 1.569` to `≈ 1.754`
- yaw `5%` plateau shrank from `68.5 deg` to `59 deg`
- motion translation residual `p95` stayed stable at `≈ 0.127 m`
- final trusted-prior drift improved from `≈ 0.293 deg` to `≈ 0.268 deg`

Interpretation:

- this is a real incremental gain
- the selection layer is no longer always collapsing to the strongest stride-40
  factor
- but horizontal coverage is still narrow compared with the healthy comparison bag
  (`selected_heading_bin_count = 2` here vs `3` on `record0402`)
- however, the bag still remains in the `repetitive_local_motion` regime
- therefore the next bottleneck is no longer simple selector collapse; it is the
  absence of a stronger map-level objective

#### `record0402` comparison bag

Validated on:

- `outputs/lidar2imu/20260423_record0402_submap_v2_balanced`

Observed behavior relative to `20260423_record0402_submap_v1`:

- selected strides remained `{10, 20, 40}`
- heading span remained `≈ 135.1 deg`
- yaw cost ratio improved slightly from `≈ 7.726` to `≈ 7.827`
- yaw `5%` plateau improved slightly from `19.5 deg` to `19 deg`
- residuals stayed effectively unchanged

Interpretation:

- the new coverage-aware selector does not damage the healthy reference bag
- it remains a safe production-side improvement
- the dominant diagnosis on this bag is still `turn_imbalance_only`

#### Refined next optimization directions after this round

The current evidence now supports a more specific roadmap:

1. **Move from local submap factors to true map factors**
   - current weak bag after the latest improvements:
     - `yaw cost ratio ≈ 1.754`
     - `5% plateau ≈ 59 deg`
     - `selected_heading_bin_count = 2`
   - healthy comparison bag:
     - `yaw cost ratio ≈ 7.827`
     - `5% plateau ≈ 19 deg`
     - `selected_heading_bin_count = 3`
   - interpretation:
     - local submap-to-submap factors help, but they are still not accumulating
       enough global horizontal structure to separate yaw cleanly
   - next action:
     - implement scan-to-map / submap-to-map factors where each motion factor is
       constrained against a larger accumulated local map rather than an
       independently registered partner submap

2. **Keep coverage-aware selection, but treat it as a front-end safety layer**
   - the latest selector is already a production-worthy front-end improvement
   - do not spend the next main iteration only on more score tuning
   - its role should now be:
     - preserve stride / heading / turn coverage
     - prevent factor collapse
     - feed a stronger back-end map objective

3. **Evaluate the next map objective with map-specific metrics, not only pair metrics**
   - keep the current stable metrics:
     - trusted-prior drift
     - yaw cost ratio
     - yaw plateau width
     - turn balance
     - residual diagnostics
   - add map-objective metrics:
     - submap / local-map thickness
     - holdout scan-to-map residuals
     - repeatability across bags and parameter settings
     - sensitivity of free planar release to the initial prior

4. **Use current diagnostics to decide when not to trust free planar release**
   - if `yaw_diagnostic.trusted_for_planar_decision = true` and
     `primary_cause = repetitive_local_motion`, the next move is map objective,
     not more solver retuning
   - if `primary_cause = turn_imbalance_only`, the next move is improved data
     collection, not changing the optimizer

### 2026-04-23 submap-to-map prototype validation

To move closer to industry-style scan-to-map while keeping the existing
conversion/calibration surface stable, a new motion registration mode was added:

- `motion_registration_mode = submap_to_map`

Current implementation:

- source factor: local submap around the start anchor
- target factor: larger local map around the end anchor
- factor selection still uses the same global-diversity / coverage-aware front end

This is intentionally a transition step between `submap_to_submap` and a future
full scan-to-map / submap-to-map back end.

#### `record0402` comparison bag

Validated on:

- `outputs/lidar2imu/20260423_record0402_submap2map_v1`

Observed changes relative to `20260423_record0402_submap_v2_balanced`:

- motion registration fitness `p05` improved from `≈ 0.740` to `≈ 0.839`
- registration inlier RMSE `p95` improved from `≈ 0.184` to `≈ 0.172`
- motion rotation residual `p95` improved from `≈ 0.238 deg` to `≈ 0.168 deg`
- motion translation residual `p95` improved from `≈ 0.301 m` to `≈ 0.259 m`
- yaw cost ratio improved from `≈ 7.83` to `≈ 28.97`
- yaw `5%` plateau shrank from `19 deg` to `9 deg`
- trusted-prior drift remained stable:
  `delta_to_initial ≈ 0.0161 m / 1.16 deg`

Interpretation:

- `submap_to_map` is strongly beneficial on the healthy reference bag
- this is not only a small residual improvement; it dramatically sharpens yaw
  curvature
- the main diagnosis still stays `turn_imbalance_only`, which is the correct
  behavior

#### Current `8`-shape bag

Validated on:

- `outputs/lidar2imu/20260423_pose_auto_submap2map_v1`

Observed changes relative to `20260423_pose_auto_submap_v2_balanced`:

- motion registration fitness `p05` improved from `≈ 0.673` to `≈ 0.767`
- motion translation residual `p95` improved from `≈ 0.127 m` to `≈ 0.110 m`
- motion rotation residual `p95` improved from `≈ 0.616 deg` to `≈ 0.521 deg`
- yaw cost ratio improved from `≈ 1.754` to `≈ 2.260`
- yaw `5%` plateau shrank from `59 deg` to `45.5 deg`
- trusted-prior drift improved from `≈ 0.268 deg` to `≈ 0.230 deg`

Interpretation:

- this is the strongest validated improvement so far on the weak `8`-shape bag
- importantly, the plateau nearly reaches the current production threshold
  (`45.5 deg` vs configured `45 deg`)
- however, the run is still just above threshold, so the current automatic policy
  correctly continues to lock planar DOFs
- this means the map-side objective is working, but current local-map scale is
  still only barely sufficient

#### Updated optimization direction after submap-to-map v1

- the evidence now supports shifting the main effort from:
  - selector tuning
  - local submap pairing
  to:
  - larger target local maps
  - scan-to-map / submap-to-map objective design
  - map-quality validation metrics
- the current engineering question is no longer "whether map-side structure
  helps" — that is already validated
- the next question is "how much map context is enough before the weak `8`-shape
  bag crosses from near-threshold to reliably non-degenerate yaw"

#### Wider local-map validation on the current `8`-shape bag

To test whether the remaining gap after `submap_to_map v1` was simply target map
scale, a wider local-map setting was evaluated:

- `outputs/lidar2imu/20260423_pose_auto_submap2map_v2_widemap`
- parameters:
  - `map_half_window = 6`
  - `map_support_stride = 10`
  - `map_min_support_frames = 5`

Observed changes relative to `submap_to_map v1`:

- motion registration fitness `p05` improved from `≈ 0.767` to `≈ 0.969`
- registration inlier RMSE `p95` improved from `≈ 0.189` to `≈ 0.128`
- motion rotation residual `p95` improved from `≈ 0.521 deg` to `≈ 0.178 deg`
- motion translation residual `p95` improved from `≈ 0.110 m` to `≈ 0.016 m`
- yaw cost ratio improved from `≈ 2.260` to `≈ 4.517`
- yaw `5%` plateau shrank from `45.5 deg` to `26.5 deg`
- solver policy changed from `freeze_xyyaw` to `free`
- `vehicle_motion_assessment.recommendation` changed from
  `z_roll_pitch_priority` to `full_6dof_candidate`
- trusted-prior drift stayed very small:
  `delta_to_initial ≈ 0.0027 m / 0.197 deg`

Interpretation:

- this is the first validated run where the current weak `8`-shape bag crosses
  the present yaw observability thresholds
- the decisive factor was not more local-window heuristics, but a larger
  map-side target context
- this strongly supports the underlying theory:
  - the bag did contain enough global information
  - the earlier pipeline simply was not preserving enough of it

#### Updated engineering conclusion after wide local-map validation

- the repo now has a validated progression:
  1. scan-to-scan
  2. `global_diversity` scan-to-scan
  3. `submap_to_submap`
  4. coverage-aware `submap_to_submap`
  5. `submap_to_map`
  6. wider `submap_to_map`
- the current weak `8`-shape bag can now become a `full_6dof_candidate` under a
  sufficiently large local-map objective
- therefore the next industrial task is no longer proving feasibility; it is
  industrializing the map-side path:
  - repeatability across nearby map-size settings
  - holdout / map-quality validation
  - cross-bag validation
  - cleaner scan-to-map / submap-to-map implementation

#### Practical optimization direction after the breakthrough

The current best validated direction is now:

1. **Keep source submaps compact, expand target maps first**
   - validated source settings stayed small:
     - `submap_half_window = 2`
     - `submap_support_stride = 5`
   - the decisive gains came from enlarging the target local map:
     - `map_half_window = 4` gave near-threshold yaw
     - `map_half_window = 6`, `map_support_stride = 10` crossed the threshold
   - practical meaning:
     - keep the query factor local enough to remain sharp
     - inject global structure primarily through the target map

2. **Promote map-size settings to first-class tuning knobs**
   - for weak bags, the main knobs are now:
     - `map_half_window`
     - `map_support_stride`
     - `map_min_support_frames`
   - not:
     - more short-window heuristics
     - more local pair score tuning

3. **Treat `full_6dof_candidate` as a candidate state, not final acceptance**
   - the weak `8`-shape bag now reaches:
     - `solver_policy.applied = free`
     - `recommendation = full_6dof_candidate`
     - `yaw cost ratio ≈ 4.52`
     - `5% plateau ≈ 26.5 deg`
   - before calling this industrial-ready, the next validations should be:
     - nearby map-size repeatability
     - cross-bag repeatability
     - holdout scan-to-map quality
     - sensitivity to the initial trusted prior

### 2026-04-24 trusted-reference consistency validation

After `submap_to_map v2_widemap` first crossed the yaw threshold, the next
question was whether that result was stable to nearby planar initial priors.

Two controlled runs were compared:

- baseline trusted-TF run:
  - `outputs/lidar2imu/20260423_pose_auto_submap2map_v3_refcheck_baseline`
  - replayed metrics:
    `outputs/lidar2imu/20260424_refcheck_baseline_metrics/metrics.yaml`
- perturbed-prior run:
  - `outputs/lidar2imu/20260423_pose_auto_submap2map_v3_refcheck_perturbed`
  - replayed metrics:
    `outputs/lidar2imu/20260424_refcheck_perturbed_metrics/metrics.yaml`
  - perturbation:
    - translation shift about `0.32 m`
    - yaw offset about `10 deg`

Observed behavior:

- baseline trusted-TF run:
  - `trusted_reference_consistency = pass`
  - `recommendation = full_6dof_candidate`
  - `delta_to_reference ≈ 0.0028 m / 0.203 deg`
- perturbed-prior run:
  - still reaches:
    - `solver_policy.applied = free`
    - `yaw_observability = pass`
    - `motion_registration = pass`
  - but now explicitly reports:
    - `trusted_reference_consistency = warning`
    - `recommendation = reference_conflict_review`
    - `delta_to_reference ≈ 0.527 m / 10.00 deg`

Interpretation:

- this validates a new industrial failure mode:
  - the map-side objective can be locally strong enough to support `free`
  - yet still stay inside the wrong planar basin if started from a nearby but bad
    initial prior
- so "high-quality internal metrics" and "free planar policy" are **necessary but
  not sufficient**
- the acceptance surface must also check consistency against a trusted reference
  when the bag provides one

Updated engineering conclusion:

- `full_6dof_candidate` is now best interpreted as:
  - "the internal LiDAR/IMU geometry is strong enough"
  - not automatically:
  - "the final planar basin is the trusted one"
- therefore repeatability validation must include both:
  - map-size sensitivity
  - initial-prior basin sensitivity
- the metrics layer now exposes that distinction directly through:
  - `coarse_metrics.statuses.trusted_reference`
  - `vehicle_motion_assessment.trusted_reference_consistency`
  - `vehicle_motion_assessment.delta_to_reference`
  - `vehicle_motion_assessment.reference_consistency_recommendations`

### 2026-04-24 extraction-side initial-transform sensitivity and fix

The new multi-start basin check exposed a more important upstream issue than
expected.

When replaying only the **calibration** stage on previously exported datasets:

- the trusted-reference dataset stayed in the trusted basin for all nearby starts
- the perturbed-prior dataset stayed in the wrong basin for all nearby starts

This proved the problem was not only solver-side basin sensitivity. The
standardized samples themselves had already been biased by the initial transform
used during record conversion.

#### Root cause

Before the fix, `record_converter.py` used `initial_transform` for:

- expected-up projection during ground extraction
- IMU ground-height projection
- pose-to-LiDAR alignment for local submap / local map construction
- IMU-to-LiDAR registration initial guesses

So when a bag already had a trusted in-record TF but the user passed a perturbed
`--initial-transform`, the exported motion/ground samples could be pulled into the
wrong planar family before calibration started.

#### Implemented fix

The converter now separates:

- `initial_transform`: calibration starting guess
- `reference_transform`: trusted transform from `/tf_static` / merged TF graph
- `extraction_transform`: transform used for record-side geometry construction

Current rule:

- if the bag provides a trusted reference transform, use it for extraction
- still preserve the user-provided `initial_transform` as the solver initial

This is now written explicitly in standardized sample metadata:

- `initial_transform_source`
- `extraction_transform_source`
- `reference_transform_source`

#### Validation after the extraction fix

Validated on:

- `outputs/lidar2imu/20260424_pose_auto_submap2map_v4_extractref_perturbed`

This run used:

- perturbed calibration initial:
  `initial_transform_source = imu_lslidar_main_perturbed_plus.yaml`
- trusted extraction geometry:
  `extraction_transform_source = /tf_static or merged tf graph`

Observed result:

- final solution returned to the trusted basin:
  - `delta_to_reference ≈ 0.0029 m / 0.222 deg`
- despite the same perturbed initial, the run now reports:
  - `trusted_reference_consistency = pass`
  - `planar_basin_stability = pass`
  - `recommendation = full_6dof_candidate`
- multi-start trials also collapse to one reference-consistent basin:
  - `reference_consistent_trial_count = 4 / 4`
  - `distinct_solution_count = 1`

#### Updated industrial conclusion after the fix

- the earlier wrong-basin behavior was not only a solver acceptance issue
- it was also an extraction-layer coupling issue
- decoupling extraction geometry from calibration initial guess is therefore a
  real method improvement, not only a metric improvement
- for bags with trusted TF, the preferred industrial workflow is now:
  - use trusted TF for extraction
  - allow perturbed or exploratory calibration initials
  - judge acceptance by:
    - trusted-reference consistency
    - planar basin stability
    - residual / observability metrics

#### Evaluation-layer update after the extraction fix

The metrics surface now exposes extraction-side validity directly through:

- `summary.delta_to_extraction`
- `coarse_metrics.statuses.extraction_geometry`
- `vehicle_motion_assessment.extraction_consistency`
- `vehicle_motion_assessment.extraction_consistency_recommendations`

Meaning:

- `trusted_reference_consistency` answers:
  - does the final solve agree with the chosen trusted reference?
- `planar_basin_stability` answers:
  - do nearby starts land in the same final basin?
- `extraction_consistency` answers:
  - is the final solve still close enough to the transform used to build the
    exported samples, or should the bag be re-extracted and re-evaluated?
- this is now closed into the converter workflow itself:
  - `lidar2imu-convert-record --calibrate --auto-reextract-if-needed`
  - keeps pass-1 as the baseline
  - triggers exactly one pass-2 re-extraction only when
    `recommendation=reextract_review` or `extraction_consistency=warning`
  - stores both passes under `reextract_pass1/` and `reextract_pass2/`
  - writes `reextract_summary.yaml`
  - if pass-2 is stronger on recommendation / consistency / residual surfaces, it
    is copied back to the stable top-level output layout

#### Validation of the automatic re-extraction loop

Validated on two local `record0402` smoke cases:

- `outputs/lidar2imu/20260424_reextract_smoketest_local`
  - trusted-TF normal path
  - `reextract_summary.yaml` reports:
    - `triggered = false`
    - `chosen_pass = pass1`
  - confirms the new workflow does not inject a second pass into ordinary accepted
    runs
- `outputs/lidar2imu/20260424_reextract_trigger_local_free_recheck`
  - controlled trigger case using:
    - fake child frame to remove trusted reference lookup
    - `--identity-initial-transform`
    - `--planar-motion-policy free`
  - pass-1 reports:
    - `recommendation = reextract_review`
    - `delta_to_extraction ≈ 2.73 m / 2.13 deg`
  - pass-2 reports:
    - `delta_to_extraction ≈ 0.63 m / 0.38 deg`
    - `chosen_pass = pass2`
  - confirms the loop really preserves both passes, improves extraction
    consistency, and promotes the stronger pass back to the stable top-level
    output layout

#### Holdout generalization surface

The next missing industrial acceptance surface was whether a map-side result keeps
its quality on motion factors that were **not** used by the solver itself.

Implemented:

- deterministic motion holdout split inside `run_calibration()`
- current default:
  - keep every 3rd selected motion sample as holdout
  - only enable holdout when enough calibration and holdout samples remain
- write:
  - `summary.dataset_partition`
  - `coarse_metrics.statuses.holdout_generalization`
  - `vehicle_motion_assessment.holdout_generalization`
  - `vehicle_motion_assessment.holdout_validation_details`
  - `fine_metrics.holdout_validation`

Current holdout decision rule:

- compare holdout vs calibration subset on:
  - motion rotation residual `p95`
  - motion translation residual `p95`
  - registration fitness `p05`
  - registration inlier RMSE `p95`
- if the holdout degradation ratios stay within configured bounds, mark
  `holdout_generalization = pass`
- otherwise downgrade to `holdout_review`

Validated on:

- `outputs/lidar2imu/20260424_holdout_replay_v4`

Observed result:

- dataset partition:
  - calibration motion samples = `8`
  - holdout motion samples = `4`
- holdout status:
  - `holdout_generalization = pass`
  - `recommendation = full_6dof_candidate`
- holdout ratios:
  - rotation residual `≈ 0.73`
  - translation residual `≈ 0.56`
  - registration fitness `≈ 1.00`
  - registration RMSE `≈ 1.00`

Updated conclusion:

- the current wide-map `8`-shape replay is now supported by:
  - trusted-reference consistency
  - planar basin stability
  - extraction consistency
  - held-out motion-factor consistency
- this is still not the end state, but it is closer to an industrial acceptance
  rule than a single in-sample optimization result

#### Final scheme review: fixed baseline and current production version

After the latest review, the operating split is now:

- **baseline**
  - goal: fixed regression / comparison reference
  - command path:
    - `lidar2imu-convert-record --profile baseline`
  - policy:
    - `scan_to_scan`
    - `--planar-motion-policy auto`
    - keep the extraction/reference evaluation surfaces
    - do **not** inject automatic second-pass re-extraction
- **production**
  - goal: current mass-production candidate
  - command path:
    - `lidar2imu-convert-record --profile production`
  - policy:
    - `submap_to_map`
    - widened local-map support
    - `--planar-motion-policy auto`
    - automatic one-step re-extraction when extraction consistency warns
    - holdout validation stays active

Why this matches industrial practice:

1. keep one conservative baseline fixed for regression
2. let the stronger map-side path evolve as the production candidate
3. judge both with the same stable acceptance surfaces instead of changing the
   acceptance rule every time the back end changes

Current production acceptance stack:

- trusted-reference consistency
- extraction consistency
- planar basin stability
- holdout generalization
- then residual / observability / solver-policy evidence

Current repo-level conclusion:

- `scan_to_scan` remains the lidar2imu baseline
- `submap_to_map` is the current production candidate
- the production candidate is no longer justified only by one successful solve;
  it is justified by passing the same extraction/reference/basin/holdout review
  stack on the current validated wide-map bag

#### Ceiling review on the current `8`-shape bag

Current judgment:

- the current bag has **probably reached its practical engineering ceiling**
  under the present motion pattern and sensor stack
- but it has **not** proved the system-level ceiling of lidar2imu

Why this is the right split:

1. **Bag-level gains are already in the diminishing-returns regime**
   - scan-to-scan global-diversity:
     - yaw cost ratio `≈ 1.15`
     - yaw `5%` plateau `≈ 143 deg`
     - motion translation residual `p95 ≈ 0.45 m`
   - submap-to-submap:
     - yaw cost ratio `≈ 1.57`
     - yaw `5%` plateau `≈ 68.5 deg`
     - motion translation residual `p95 ≈ 0.13 m`
   - current validated wide-map submap-to-map:
     - yaw cost ratio `≈ 3.34`
     - yaw `5%` plateau `≈ 33.5 deg`
     - motion translation residual `p95 ≈ 0.016 m`
     - trusted-reference drift `≈ 0.0029 m / 0.222 deg`
   - compared with the earlier stages, the production candidate already extracts
     almost all of the obvious remaining planar information from this bag

2. **The current best run already passes the industrial acceptance stack for one bag**
   - trusted-reference consistency: `pass`
   - planar basin stability: `pass`
   - extraction consistency: `pass`
   - holdout generalization: `pass`
   - recommendation: `full_6dof_candidate`

3. **The remaining limitations are now mostly data-side, not solver-side**
   - translation heading span is still only about `58.6 deg`
   - mean IMU rotation axis is still strongly z-dominant (`|z| ≈ 0.96 ~ 0.98`)
   - the bag is still one flat-ground `8`-shape route, not a release pack with
     multiple route archetypes
   - this means more tuning on the same bag is unlikely to create a large new
     accuracy jump; the more likely result is only small metric movement

Why this is **not yet** the system ceiling:

- there is still no cross-bag repeatability matrix
- holdout is currently one deterministic split, not repeated split stability
- there is still no uncertainty summary / confidence interval for the released
  extrinsics
- there is still no release rule proving that the current production profile is
  stable across bag families, not only on the current best bag

Industrial conclusion:

- for the **current bag**, the next return from pure algorithm retuning is likely
  low
- for the **system**, the next return from batch repeatability / uncertainty /
  release qualification is likely high

4. **Keep diagnostics stable while the back end evolves**
   - continue to compare every new map-side variant using the same stable surfaces:
     - `metrics.yaml`
     - `vehicle_motion_assessment.yaw_diagnostic`
     - trusted-prior drift
     - residual summaries
   - this keeps theory + experiment aligned while the map back end continues to iterate

### 2026-04-23 validating when `yaw_rotation_degenerate` is truly trustworthy

The next question was whether `yaw_rotation_degenerate` itself can be trusted as
a real judgment, instead of only being a conservative lock signal.

To answer that, the metrics layer was extended with:

- `vehicle_motion_assessment.yaw_diagnostic.evaluation_reliability`
- `vehicle_motion_assessment.yaw_diagnostic.trusted_for_planar_decision`
- `vehicle_motion_assessment.yaw_diagnostic.reliability_limiters`
- `vehicle_motion_assessment.yaw_diagnostic.primary_cause`
- `vehicle_motion_assessment.yaw_diagnostic.recommendations`

#### High-reliability validation cases

Replayed on:

- `outputs/lidar2imu/20260423_pose_auto_globaldiv_v2_yawguide`
- `outputs/lidar2imu/20260423_pose_auto_submap_v1_yawguide`
- `outputs/lidar2imu/20260423_record0402_globaldiv_v2_yawguide`
- `outputs/lidar2imu/20260423_record0402_submap_v1_yawguide`

Observed behavior:

- current `8`-shape bag, scan-to-scan:
  - `evaluation_reliability = high`
  - `trusted_for_planar_decision = true`
  - `primary_cause = repetitive_local_motion`
  - recommendation points to:
    - keep planar lock
    - widen heading / axis diversity
    - move to `submap_to_submap`
- current `8`-shape bag, submap-to-submap:
  - `evaluation_reliability = high`
  - `trusted_for_planar_decision = true`
  - `primary_cause = repetitive_local_motion`
  - recommendation points to:
    - keep planar lock
    - accept that submaps helped but are still too local
    - move next to scan-to-map / submap-to-map
- `record0402`, scan-to-scan and submap-to-submap:
  - `evaluation_reliability = high`
  - `trusted_for_planar_decision = true`
  - `primary_cause = turn_imbalance_only`
  - recommendation correctly points to collecting both turn directions, not to
    chasing a false yaw-degenerate diagnosis

Interpretation:

- once LiDAR motion factor quality is good enough, the code can now separate:
  - **true weak yaw geometry** on the current `8`-shape bag
  - **turn imbalance only** on `record0402`
- so `yaw_rotation_degenerate` is now meaningful when
  `evaluation_reliability = high`

#### Medium-reliability validation cases

Replayed on:

- `outputs/lidar2imu/raw_validation_round1_yawguide`
- `outputs/lidar2imu/raw_validation_round2_filter_yawguide`

Observed behavior:

- `evaluation_reliability = medium`
- `trusted_for_planar_decision = false`
- `reliability_limiters = [motion_registration_not_pass]`
- recommendations begin with:
  - do not use this run alone for a free-planar decision
  - improve factor quality or sample support first

Interpretation:

- this is the intended behavior for marginal bags
- even when the local yaw stage looks non-degenerate, the system now avoids
  over-trusting that result if LiDAR motion factor quality is still weak

#### Practical rule after this round

- Trust `yaw_rotation_degenerate` as a **real geometric conclusion** only when:
  - `yaw_diagnostic.trusted_for_planar_decision = true`
  - and `yaw_diagnostic.evaluation_reliability = high`
- If not, treat it as a conservative safety signal, not as a final geometric
  diagnosis
- Use `yaw_diagnostic.primary_cause` and `recommendations` to decide the next
  action instead of manually reverse-engineering the raw metrics every time

### 2026-04-24 validating wrong trusted `z` vs wrong initial `z`

The next production question was whether the system can distinguish:

- a bad **initial guess** in `z`
- a bad **trusted reference** in `z`

This was tested with controlled replays on the current wide-map `8`-shape bag
using the validated standardized samples from:

- `outputs/lidar2imu/20260424_pose_auto_submap2map_v4_extractref_perturbed/standardized_samples.yaml`

#### Controlled replays

Generated:

- `outputs/lidar2imu/20260424_zbias_experiments/initial_z_plus_5pct.yaml`
- `outputs/lidar2imu/20260424_zbias_experiments/initial_z_plus_10pct.yaml`
- `outputs/lidar2imu/20260424_zbias_experiments/reference_z_plus_5pct.yaml`
- `outputs/lidar2imu/20260424_zbias_experiments/reference_z_plus_10pct.yaml`

Observed behavior:

1. **Wrong initial `z` is recoverable on this bag**
   - `initial z +5%`:
     - final `z ≈ 1.94254`
     - trusted-reference consistency: `pass`
     - extraction consistency: `pass`
     - recommendation: `full_6dof_candidate`
   - `initial z +10%`:
     - final `z ≈ 1.94254`
     - trusted-reference consistency: `pass`
     - extraction consistency: `pass`
     - recommendation: `full_6dof_candidate`
   - interpretation:
     - on this strong map-side bag, the optimizer basin for vertical height is wide
       enough to recover from at least `+10%` initial `z` bias
     - so **initial-guess tolerance** is materially wider than the
       trusted-reference acceptance gate

2. **Wrong trusted `z` can keep strong internal metrics**
   - before hardening the metric, `reference z +5%` produced:
     - `delta_to_reference.translation_norm_m ≈ 0.0971 m`
     - trusted-reference consistency: `pass`
     - even though the run was wrong mainly in vertical height
   - this showed the old norm-only rule had a blind spot: a pure `z` error could
     sit just below the `0.10 m` global threshold

3. **After hardening, the same `reference z +5%` case is now detected**
   - replay:
     - `outputs/lidar2imu/20260424_zbias_experiments/run_reference_z_plus_5pct_axischeck`
   - observed:
     - `delta_to_reference.translation_xyz_m.z ≈ -0.0955 m`
     - `delta_to_reference.vertical_error_ratio ≈ 0.0469`
     - `vehicle_motion_assessment.trusted_reference_consistency = warning`
     - `reference_consistency_details.failure_reasons = [vertical_translation]`
   - interpretation:
     - **percentage-only rules are not enough**
     - even a nominal `+5%` reference perturbation can show `<5%` relative error at
       evaluation time because the denominator is the wrong reference height
     - this is why the repo now uses:
       - total translation norm
       - total rotation
       - absolute vertical `z` threshold
       - relative vertical ratio

#### Practical production rule after this experiment

- For **initial `z` bias** on the current strong wide-map bag:
  - at least `+10%` is recoverable
  - so a moderately wrong initial height is **not** a reason by itself to reject
    calibration
- For **trusted reference `z`**:
  - do not rely on translation norm alone
  - always inspect:
    - `delta_to_reference.translation_xyz_m.z`
    - `delta_to_reference.vertical_error_ratio`
    - `vehicle_motion_assessment.reference_consistency_details.failure_reasons`
- For **trusted extraction geometry**:
  - inspect the same vertical fields under `delta_to_extraction`
  - if the main conflict is vertical, treat it as a sample-generation / mount-height
    issue before tuning the solver

Industrial conclusion:

- a wrong trusted `z` is a **data / prior validation** problem
- a wrong initial `z` is a **basin / solver robustness** problem
- they should not share the same acceptance rule

### 2026-04-24 checking whether the current algorithm itself is wrong

The deeper question after the `z` experiment was whether the optimizer itself is
currently broken, or whether the main weakness is still upstream in the data /
trust policy.

#### Code-path inspection

The current pipeline is structurally split as:

1. `solve_roll_pitch_from_ground()`
   - uses ground plane normals against gravity
   - identifies `roll/pitch`
2. `solve_ground_translation()`
   - uses ground height priors
   - returns a full translation vector, but on flat-ground bags this stage is much
     stronger in `z` than in `x/y`
3. `solve_yaw_from_motion()`
   - uses rotation-only motion consistency for `yaw`
4. `solve_translation_from_motion()`
   - uses motion translation consistency
5. `refine_joint_solution()`
   - jointly refines all active components

This decomposition is not showing a first-order optimizer bug on the validated
wide-map bag.

#### Controlled replay result: initial-value robustness is currently strong

Using the same validated standardized samples, the following were replayed:

- `initial z = 1.5 m`
- rough 6DoF initial TF:
  - `x = 0.0`
  - `y = 1.0`
  - `z = 1.5`
  - `yaw = 15 deg`
  - `roll = 4 deg`
  - `pitch = -3 deg`

Observed:

- all of them converged back to the same final solution:
  - `x ≈ 0.00887`
  - `y ≈ 1.50929`
  - `z ≈ 1.94254`
  - `yaw ≈ -0.0803 deg`
  - `roll ≈ 0.0598 deg`
  - `pitch ≈ 0.2170 deg`
- so, on validated samples, the current optimizer basin is **not** showing an
  obvious bug around initial-value sensitivity

#### Controlled replay result: reference-only perturbation does not move the final solve

To separate reference checking from extraction geometry, another replay fixed
`extraction_transform` explicitly and perturbed only `reference_transform`.

Observed:

- even large pure-reference perturbations (`z = 1.5 m`, rough 6DoF reference) did
  **not** move the final solution
- they only changed the acceptance result to `reference_conflict_review`

Interpretation:

- `reference_transform` is currently an **evaluation surface**
- it is not the main driver of the final solve once the dataset has already been
  exported correctly

#### The real structural weakness: extraction trust policy

The current code path in `record_converter.py` still does this when the raw bag
contains lidar -> parent TF:

- `initial_transform = record_reference_transform` if the user does not override it
- `extraction_transform = record_reference_transform`

And `extraction_transform` is then used directly to build:

- expected LiDAR up direction for ground extraction
- `imu_ground_height`
- LiDAR motion registration initial guesses
- local submap / local map alignment geometry

This means:

- if the raw-bag TF is wrong, the exported standardized samples can already be
  geometrically biased before calibration starts
- after that point, calibration may still be very stable — but stable around the
  wrong data geometry

#### Algorithm review conclusion after this round

- the current **solver core** is not the main suspected problem on the validated
  wide-map bag
- the current **system-level** weakness is:
  - too much trust in raw-bag TF for extraction geometry
  - no built-in A/B raw re-extraction comparison from multiple candidate TFs yet
  - basin / holdout acceptance still stronger for planar motion than for
    `z/roll/pitch`
- therefore the next industrial path should be:
  1. keep the current fixed baseline / production split
  2. keep stable acceptance surfaces
  3. add explicit user-prior classification
  4. add raw-level candidate extraction comparison
  5. extend acceptance from planar-only to full 6DoF prior robustness

### 2026-04-24 adding `initial_prior_assessment`

To make this behavior operational instead of manual, the metrics layer now
publishes:

- `coarse_metrics.statuses.initial_prior_nominal_range`
- `vehicle_motion_assessment.initial_prior_assessment`
- `vehicle_motion_assessment.initial_prior_assessment_details`
- `fine_metrics.initial_prior_assessment`

Current meaning:

- `pass`
  - the provided TF already lies inside the nominal production range
- `recoverable`
  - the TF is outside the nominal range, but this bag still converged back to the
    accepted basin
- `warning`
  - the TF is outside the nominal range and the run also conflicts with other
    acceptance surfaces

Current thresholds:

- nominal:
  - translation `<= 0.5 m`
  - rotation `<= 10 deg`
- max recoverable:
  - translation `<= 1.0 m`
  - rotation `<= 20 deg`

Validated examples on the current wide-map bag:

1. `pass`
   - `run_initial_equal_reference`
   - `delta_to_initial ≈ 0.0034 m / 0.239 deg`
2. `recoverable`
   - `run_initial_rough`
   - `delta_to_initial ≈ 0.768 m / 16.0 deg`
   - final acceptance still `full_6dof_candidate`
3. `warning`
   - `run_reference_rough_6dof_pure`
   - the provided TF is outside nominal range and the run lands in
     `reference_conflict_review`

Industrial interpretation:

- `recoverable` means **this bag recovered**, not that the user TF is acceptable as
  a normal production prior
- `warning` means the provided TF should be treated as suspicious and checked
  against frame definitions / installation measurements / extraction-candidate A/B
  comparisons

### 2026-04-24 extending basin checks to `full_prior_robustness`

The next industrial gap after `initial_prior_assessment` was that
`planar_basin_stability` still only tested nearby `x/y/yaw` perturbations.

That was good enough to detect weak planar basins, but not enough to answer the
production question:

- does the accepted basin stay stable under a wider nearby prior neighborhood?
- or is it only stable for one narrow planar seed family?

To cover that, the metrics layer now also publishes:

- `coarse_metrics.statuses.full_prior_robustness`
- `vehicle_motion_assessment.full_prior_robustness`
- `vehicle_motion_assessment.full_prior_robustness_details`
- `fine_metrics.full_prior_robustness`

Current intent:

- keep `planar_basin_stability` as the narrow planar-only surface
- add `full_prior_robustness` as the wider multistart surface over:
  - input/reference seed family
  - planar perturbations
  - vertical `z` perturbation
  - roll / pitch perturbation

#### Validation set

Replayed with the new metric:

1. strong nominal run
   - `outputs/lidar2imu/20260424_industrial_eval/run_initial_equal_reference`
2. strong recoverable-prior run
   - `outputs/lidar2imu/20260424_industrial_eval/run_initial_rough`
3. pure trusted-reference conflict run
   - `outputs/lidar2imu/20260424_industrial_eval/run_reference_rough_pure`
4. weak baseline smoke run
   - `outputs/lidar2imu/20260424_industrial_eval/run_baseline_smoke_recheck`

Observed:

1. **Strong nominal run**
   - `recommendation = full_6dof_candidate`
   - `initial_prior_assessment = pass`
   - `planar_basin_stability = pass`
   - `full_prior_robustness = pass`
   - `holdout_generalization = pass`
   - `full_prior_robustness_details.primary_cause = single_full_prior_basin`

2. **Strong recoverable-prior run**
   - `recommendation = full_6dof_candidate`
   - `initial_prior_assessment = recoverable`
   - `delta_to_initial ≈ 0.768 m / 16.0 deg`
   - `planar_basin_stability = pass`
   - `full_prior_robustness = pass`
   - interpretation:
     - this bag still converges to one stable basin even when the provided TF is
       materially outside the nominal production range

3. **Pure trusted-reference conflict run**
   - `recommendation = reference_conflict_review`
   - `initial_prior_assessment = warning`
   - `trusted_reference_consistency = warning`
   - `planar_basin_stability = pass`
   - `full_prior_robustness = pass`
   - interpretation:
     - the optimizer basin itself is stable
     - the run is rejected because it conflicts with the trusted reference surface,
       not because the local basin is fragile

4. **Weak baseline smoke run**
   - `recommendation = basin_sensitivity_review`
   - `yaw_observability = warning`
   - `planar_basin_stability = warning`
   - `full_prior_robustness = warning`
   - `full_prior_robustness_details.primary_cause = planar_prior_sensitivity`
   - `full_prior_robustness_details.distinct_solution_count = 3`
   - interpretation:
     - the bag is genuinely multi-basin under nearby priors
     - this is the pattern where one accepted solve must **not** be promoted as an
       industrially stable free-planar result

#### Practical industrial meaning

- `planar_basin_stability` answers:
  - "Do nearby `x/y/yaw` seeds collapse to one basin?"
- `full_prior_robustness` answers:
  - "Does that conclusion survive a wider nearby prior neighborhood?"
- `trusted_reference_consistency` answers:
  - "Even if the basin is stable, does it still agree with the trusted bag-side
    TF?"

So the production judgment is now cleaner:

- **stable + trusted**
  - `planar_basin_stability = pass`
  - `full_prior_robustness = pass`
  - `trusted_reference_consistency = pass`
- **stable but conflicts with trusted TF**
  - `full_prior_robustness = pass`
  - `trusted_reference_consistency = warning`
- **not yet industrially stable**
  - `full_prior_robustness = warning`
  - or `planar_basin_stability = warning`

This is the current industrial acceptance stack for a promoted free-planar run:

1. extraction consistency
2. trusted-reference consistency
3. planar basin stability
4. full prior robustness
5. holdout generalization
6. initial-prior assessment as a user-input interpretation layer

### 2026-04-24 repeated holdout stability, uncertainty, and lidar2lidar crossover

After fixing extraction/reference/basin/full-prior surfaces, the next highest-value
gap was no longer "one more solve on the same best bag" but:

- repeated stability across holdout splits
- a first uncertainty surface for the released extrinsics
- a clearer release workflow borrowed from `lidar2lidar`

#### Why this is the right next step

The current strong wide-map bag already passes the industrial acceptance stack for
one run, but that still leaves three system-level gaps:

1. holdout was only one deterministic split
2. there was no bag-local uncertainty summary for the final 6DoF result
3. there was still no release habit equivalent to `lidar2lidar`'s stable
   `metrics.yaml + diagnostics/` separation plus repeated-run thinking

#### What was borrowed from `lidar2lidar`

Reviewing `lidar2lidar` suggests three reusable industrial patterns:

1. **Keep stable concise metrics and rich diagnostics separate**
   - short decision surface in `metrics.yaml`
   - richer investigation surface under `diagnostics/`
2. **Treat quality gates as promotion logic, not only optimizer success**
   - just because one optimization converged does not mean the run is releasable
3. **Add repeated-run stability summaries**
   - `lidar2lidar` already points toward repeated-run / multi-sample stability as
     the next iteration gap
   - `lidar2imu` should do the same for holdout offsets and later for cross-bag
     release packs

#### Implemented in this round

The metrics layer now also publishes:

- `coarse_metrics.statuses.holdout_repeatability`
- `vehicle_motion_assessment.holdout_repeatability`
- `vehicle_motion_assessment.holdout_repeatability_details`
- `fine_metrics.holdout_repeatability`
- `fine_metrics.uncertainty_summary`

Current behavior:

- keep the existing deterministic holdout split
- replay additional holdout offsets over the same `every_n` scheme
- summarize:
  - repeated holdout pass / warning / unknown
  - whether different offsets converge to one basin or multiple basins
  - a first bag-local uncertainty summary from repeated final-transform spread

#### Validation

1. strong wide-map run
   - `outputs/lidar2imu/20260424_repeatability_eval/run_initial_equal_reference`
   - observed:
     - `holdout_repeatability = pass`
     - `holdout_generalization = pass`
     - `trial_count = 3`
     - `distinct_solution_count = 1`
     - max pairwise final delta `≈ 0.0010 m / 0.052 deg`
     - uncertainty:
       - yaw std `≈ 0.0118 deg`
       - z std `≈ 0.00046 m`
   - interpretation:
     - this is the first useful bag-local uncertainty signal for a promoted run

2. weak baseline smoke run
   - `outputs/lidar2imu/20260424_repeatability_eval/run_baseline_smoke`
   - observed:
     - `holdout_repeatability = unknown`
     - `holdout_generalization = unknown`
   - interpretation:
     - the bag is too small for repeated holdout
     - that should be treated as missing evidence, not as a silent pass

#### Updated next industrial roadmap

With this round, the immediate priority order is now:

1. **batch repeatability matrix**
   - repeat across bag families and nearby map settings
2. **uncertainty summary**
   - extend the new bag-local repeated-holdout summary into cross-bag release
     confidence
3. **freeze production release**
   - define explicit release criteria on top of:
     - extraction consistency
     - trusted-reference consistency
     - planar basin stability
     - full prior robustness
     - holdout generalization
     - holdout repeatability / uncertainty
4. **raw-level candidate extraction comparison**
   - because the larger remaining structural risk is still extraction-side trust in
     bag TF
5. **true scan-to-map continuation**
   - improve the algorithm ceiling, but not at the cost of destabilizing the now
     fixed evaluation surfaces
