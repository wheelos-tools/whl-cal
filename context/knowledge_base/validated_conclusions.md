# Validated conclusions

This file lists the current conclusions that are supported by tested data.

## lidar2lidar

### Production baseline

- `scan2scan` remains the production default.
- It is the main acceptance baseline for vehicle-rig calibration.

### scan2map

- `scan2map` is a secondary validation / refinement path, not a blanket replacement for `scan2scan`.
- On `record_data_0402`:
  - `left -> main` can be accepted as an unconstrained scan2map candidate
  - `right -> main` must remain diagnostic when unconstrained because its gain is driven mainly by `z/pitch/roll` drift

### Vehicle-rig interpretation

- Metrics must be split into:
  - planar: `x/y/yaw`
  - vertical-attitude: `z/pitch/roll`

## lidar2imu

### Gravity source

- pose-derived gravity is the current default
- `gravity-source imu` is not currently trustworthy on the tested bags

### `record_data_0402`

- This bag is usable end-to-end.
- It is **not** a production-quality `x/y/yaw` acceptance bag.
- Current trustworthy level:
  - `z/roll/pitch`: usable
  - `x/y/yaw`: weak due to one-sided turning

### Synology front-LiDAR bag

- `/mnt/synology/raw-data/2026-04-13-06-54-28` is useful for diagnostics.
- Without a trusted prior, it should not be used to directly accept final `lidar2imu` extrinsics.
- With the user-provided prior, it is still diagnostic-only because turn balance remains one-sided.

### Weak-planar solver policy

- `lidar2imu` now supports:
  - `--planar-motion-policy free`
  - `--planar-motion-policy freeze_xyyaw`
  - `--planar-motion-policy auto`
- `auto` is the current recommended policy for weak-planar bags.
- Tested result:
  - on `record_data_0402`, `auto` reduces planar drift from about `2.17 m / 1.30 deg` to about `0.017 m / 1.22 deg`
  - on the Synology bag with the user prior, `auto` reduces drift from about `0.343 m / 2.65 deg` to about `0.008 m / 0.56 deg`

### Window + gate data selection

- `lidar2imu` motion extraction now follows **window + gate** instead of pure global candidate ranking.
- Current tested behavior:
  - `record_data_0402`: `8` windows, `6` valid windows, `5` selected motion samples
  - Synology bag: `8` windows, `6` valid windows, `6` selected motion samples
- Current strategy:
  - split the motion timeline into windows
  - prefer candidates with enough angular excitation
  - normalize candidate score by stride to avoid over-preferring very long spans
  - gate weak windows and low-fitness registrations
