# Validated conclusions

This file lists the current conclusions that are supported by tested data.

## lidar2lidar

### Production baseline

- `scan2scan` remains the production default.
- It is the main acceptance baseline for vehicle-rig calibration.

### Four-corner raw-rig loop closure

- On `/mnt/synology/REDACTED/2026-05-07-REDACTED_USER/bag/20260508032341.record.00000`,
  the four raw corner LiDARs support a single-component rectangle loop under
  static conditions.
- The perimeter edges (`LF-RF`, `RF-RB`, `RB-LB`, `LB-LF`) can all be retained
  under the current conservative graph gates.
- The weakest diagonal (`LB-RF`) is better treated as a consistency check than a
  primary production constraint.

### scan2map

- `scan2map` is a secondary validation / refinement path, not a blanket replacement for `scan2scan`.
- On `record_data_0402`:
  - `left -> main` can be accepted as an unconstrained scan2map candidate
  - `right -> main` must remain diagnostic when unconstrained because its gain is driven mainly by `z/pitch/roll` drift

### Vehicle-rig interpretation

- Metrics must be split into:
  - planar: `x/y/yaw`
  - vertical-attitude: `z/pitch/roll`

### Zhongce joint v2 research baseline

- The refreshed three-static-window baseline A and its exact scope are recorded
  in the [2026-11-24 handoff](../notes/lidar2lidar/2026-11-24_zhongce_baseline.md).
  It remains review-only and does not replace the production scan2scan baseline.
- Native joint `accepted` is scoped to common-mode excitation, not independent
  L2L accuracy. Additional moving-window groups B/C have not been promoted.
- The post-gap validation data have been reused across iterations; they are not
  a fresh blind test. Keep data-expansion effects separate from estimator changes.

## lidar2imu

### GRIL/LI-Init research dataset lessons (screened through 2026-09-30)

Do not use a single `input_contract: accepted` verdict as evidence that a
capture supports differentiation, all calibration parameters, or extrinsic
accuracy. Use the following **per-capture** distinctions; pending entries are
not validated datasets. The Apollo INS pose and the input INS IMU share a
fusion chain, so their agreement is a motion check, not independent truth.

| Capture | Verified input / data limitation | Algorithm outcome / use |
| --- | --- | --- |
| Weilan 0612 | 1405 LiDAR scans; first LiDAR precedes IMU by 0.994 s. One split-family entry, not all three members as separate expanding inputs, fixes a false non-monotonic-time error. The overlap crop passes the conversion contract; sampled raw `/gnss/imu.measurement_time` is zero. | GRIL has no result after cropping: Z-rotation sufficiency only 2%. Use as an insufficient-excitation / initial-overlap regression case, not an accuracy benchmark. [Timing audit](../../outputs/gril/research_0612_preflight/record_timing_audit.yaml). |
| Weilan 0827 | 1375 scans and 13741 IMU samples; raw `/gnss/imu.measurement_time` is zero in a sampled source message. Corrected-IMU has a regular 10 ms header clock, but its values differ from raw IMU; never silently substitute it into a prior run. | Same-bag LI-Init produced no transform (Z excitation 37%); GRIL full-run translation spread 0.04029 m exceeds 0.03 m, whereas replaying one frozen batch twice gives identical results. On that batch, four yaw-rate timing windows peak at -28/-30/-42/-42 ms, a 14 ms spread above the 10 ms gate despite correlations >0.985. Retain as weak-motion / full-frontend repeatability regression, **not** an extrinsic accuracy winner. [Signal audit](../../outputs/gril/research_0827_preflight/diagnostics/record_signal_audit.yaml), [repeatability](../../outputs/gril/research_0827_repeatability/repeatability.yaml), [time profile](../../outputs/gril/research_0827_repeatability/diagnostics/gril_batch_time_profile_yaw_z_sigma_0s.yaml), [research decision contract](../../.agents/knowledge/lidar2imu_research.md). |
| Zhongji 05-06 calibration | 1591 scans, 32 rings, 15913 INS corrected-IMU samples with measurement-time header. Raw `/gnss/imu` has zero measurement time and about 10.67 ms median host-publish delay versus the matching corrected samples. | Generic GRIL sufficiency passes, but corrected-time runs return -4.45 s / -31.62 s offsets and change translation 2.38 m with the seed. Later LiDAR-only frontend yaw is unstable. Same-input diagnostic-binary reruns also diverge; this is **not** solely a data rejection. No independent extrinsic truth. [Iteration report](../../outputs/gril/research_20260506_canonical_32line_corrected_imu/iteration_report.yaml). |
| Zhongji 05-07 short recordings | 44 scans / 4.30 s and 101 scans / 9.99 s with almost no turn excitation. | Reject for calibration excitation; no algorithm accuracy run. [Matrix](../../outputs/gril/research_multicapture_20260929/dataset_matrix.yaml). |
| Zhongji 05-07 figure eight, first five shards | 900 scans, 8490 corrected-IMU samples, 32 separated elevation bands, both turn directions. LiDAR is continuous but INS IMU has five gaps of 0.57-1.56 s, one also confirmed in source shard `00001`. | Reject the **combined 90 s window** for uninterrupted motion differentiation, not the whole 27-shard family or each segment. A passing canonical format contract missed the gap gate. Neither solver was run across these gaps. [Input contract](../../outputs/gril/research_20260507_eight_first5/canonical/input_contract.yaml), [matrix](../../outputs/gril/research_multicapture_20260929/dataset_matrix.yaml). |
| Ruantong U-turn / parking 04 | U-turn family has missing split indices; parking 04 has four sequential shards. A **full first-shard pass only** finds 577 `lslidar_main` scans and 5724 raw/corrected IMU samples over about 58 s; raw `measurement_time` is populated, both IMU streams have no gap over 50 ms, and Z angular velocity exceeds 0.04 rad/s in each direction (630/595 samples). First point cloud spans about 99 ms and `/tf_static` connects `imu -> base_link -> lslidar_main`. **Neither `index % 16` nor `index % 32` produces separated elevation bands** in the first scan; its flat point decoder exposes no ring field. | Most promising *new* data candidate by clock and turn screening, **not yet GRIL-compatible**: investigate the sensor-specific channel ordering/ring decoding before conversion. Other three shards, acceleration, static startup, and independent reference remain unchecked. No algorithm runs yet. [Matrix](../../outputs/gril/research_multicapture_20260929/dataset_matrix.yaml). |

**Reusable screening order:** select one family entry and verify missing
shards; check each topic's actual measurement/header/record clock and IMU
values; reject or split at source gaps before differentiating; validate point
times, native frame, sensor-specific rings and TF direction; check left/right
turns, acceleration and startup-static support; only then freeze canonical
inputs and run the unmodified baseline. Account for no-result and failed runs.
Do not conflate a source limitation (overlap, gaps, weak motion) with a method
or build failure (frontend instability, missing dependencies). A known
configured transform is a seed, not independent truth.

**If an additional capture is needed after 0827:** determine the Ruantong
parking-04 LiDAR's real
channel ordering and check the remaining three shards for timing gaps and
excitation **before** any GRIL run. Do not apply the Vanjee
`point_index % scan_lines` ring convention to `lslidar_main`. If the LiDAR
point contract cannot be established, search a contiguous gap-free segment in
the remaining Zhongji 05-07 figure-eight shards; if none qualifies, consider
an independently documented public dataset or targeted recapture. Keep
Weilan 0612/0827 as negative regression controls. No captured set above
currently establishes a multi-method extrinsic-accuracy winner.

**Current four-method focus (Weilan 0827):** retain 0827 as the user's
working comparison data; do not imply another capture is required before
testing adapters or reference-free diagnostics. P0 native GRIL remains a
research baseline and P1 LI-Init a related runnable no-result control.
For P1, reconstructing its original LiDAR-only 800-sample-scaled Jacobian
gate gives pair scores 0.02356/0.02374/0.37011, all below 0.99; a
single-factor 0.05→0.10 m surf-voxel rerun eliminates PCL overflow warnings
but still gives only 0.02712/0.02714/0.36661 and no result. The 100 Hz
IMU warning is advisory, not its stop gate; do not weaken sufficiency.
See [P1 reconstruction](../../outputs/gril/research_0827_repeatability/diagnostics/li_init_gate_audit.yaml).
P2
LI-Calib's PointCloud2 reader reconstructs VLP-16 firing time instead of
using the 0827 Vanjee point `time/ring` fields; the first unorganized scan
has 25,291 points, exceeding its 1,824-firing lookup. P3 AFLI-Calib's
Velodyne branch does not append scans; its Hesai parser requires absolute
float64 per-point time rather than 0827's relative float32 field. These are
**verified input-adapter blockers**, not accuracy failures. See
[method matrix and point-cloud review](../../.agents/knowledge/lidar2imu_research.md).

### GRIL 2026-05-06 corrected-IMU research capture

- The verified 32-line input uses the INS measurement-time
  `/apollo/sensor/gnss/corrected_imu` stream. The original `/gnss/imu`
  header is host-publish time on this capture.
- The GRIL LiDAR-only frontend loses effective correspondences in later
  windows. Reducing `filter_size_surf` from 0.5 m to 0.3 m increased some
  correspondence counts but worsened yaw in a fixed 33–49.5 s window
  (about 0.17° to 19.96° relative to same-chain INS motion). Reject this
  one-factor candidate; neither configuration establishes extrinsic accuracy.
- Older native odometry traces' zero mean residual was an instrumentation
  error, not high registration quality. Corrected instrumentation reports
  nonzero mean absolute accepted point-to-plane distances, but two
  identical-input runs produced different frontend traces, durations, and
  calibration results. Stop algorithm ranking until repeatability is resolved.
- Evidence and binary identities:
  `outputs/gril/research_20260506_canonical_32line_corrected_imu/iteration_report.yaml`.
  INS odometry is coupled to the input IMU and cannot establish independent
  extrinsic ground truth.

### GRIL 2026-11-17 Zhongce dual-corner FlatBuffer LiDAR + corrected-IMU stabilization

- Capture `/mnt/synology/中策/2026-11-17/bag` provides two raw 32-line corner
  FlatBuffer LiDARs (`left_front`, `right_back`, `10,912` scans each; fused
  cloud excluded) and `108,621` `/apollo/sensor/gnss/corrected_imu` samples at
  `100 Hz` on `measurement_time` (`H = 0`), with a continuous `[0, 680 s]`
  pre-gap segment and a temporally disjoint `[785, 1005 s]` post-gap validation
  segment (reused in later L2L iterations, so no longer a fresh blind test).
- Unmodified GRIL failed repeatability and physical checks due to: (1) async
  `ikd-Tree` rebuild threads + Eigen 3.3.7 `-fopenmp` multi-threaded GEMM 1-ULP
  races, (2) equal-point-count `cut_frame=3` + `kImuHz=200` `dt=0` triggering a
  stationary start spur at `t ≈ 3.7 s`, (3) frame-inconsistent pseudo-ground EKF
  rows + close-wall feature starvation causing vertical "banana" drift and a
  `+244.4 m` Z blow-up / premature ~13% stop, and (4) unbounded `xcorr` lag
  jumps (`+48.4 s` / `-2.5 s`) + EIV lever-arm dilution from 30 Hz LiDAR
  angular derivatives.
- With strictly opt-in frontend/batch stabilization controls (preserving 100%
  golden defaults when omitted) and post-run Stage 3 gating
  (`lidar2imu evaluate-quality-gates`), both corners reach
  `state: converged_and_verified` (`verdict: accepted`) on `[0, 680 s]`:
  - `left_front`: `R_LI = [+0.036°, -0.075°, +132.687°]`,
    `T_LI = [-0.1513, +2.0156, +0.4767] m`, `tau = -22.109 ms` (3x SHA-256
    bit-exact `0.0° / 0.0 m / 0.0 s`; 4-seed spread `0.46° / 5.9 cm / 0.002 ms`
    including blind `I + [0,0,0]`; holdout acc RMSE `0.6442 -> 0.6118 m/s²`).
  - `right_back`: `R_LI = [-0.006°, -0.295°, -46.376°]`,
    `T_LI = [+1.6166, -3.7449, +0.4565] m`, `tau = -22.792 ms` (3x SHA-256
    bit-exact `0.0° / 0.0 m / 0.0 s`; 4-seed spread `0.53° / 2.2 cm / 0.035 ms`
    including blind `I + [0,0,0]`; holdout acc RMSE `0.8249 -> 0.7113 m/s²`).
  - Independent dual-corner cross-checks agree to `ΔYaw = 179.06°` (opposite
    45° RFU/FLU corners), `Δz = 2.0 cm`, and `Δtau = 0.68 ms` on the shared
    `corrected_imu` clock. Evidence:
    `outputs/gril/stage2b_stabilization_20261120/exp2_windows_holdout/` and
    `outputs/gril/stage2b_stabilization_20261120/diagnostics/`.

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

- `/mnt/synology/REDACTED/raw-data/2026-04-13-06-54-28` is useful for diagnostics.
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
