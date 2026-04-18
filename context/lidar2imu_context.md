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
