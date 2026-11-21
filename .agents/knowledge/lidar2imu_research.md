# LiDAR-to-INS-IMU research: data, iteration, evaluation

## When

Read before selecting a LiDAR-to-IMU capture, comparing methods, changing
GRIL, or calling an extrinsic accurate. Use
[`gril-calib-validation`](../skills/gril-calib-validation/SKILL.md) for the
executable workflow; this document is the **decision contract**, not a
replacement decoder or solver. Keep extraction, algorithm, and evaluation
separate. See [dataset lessons](../../context/knowledge_base/validated_conclusions.md)
for other captures.

## Weilan 0827: controlled system-level comparison

**Decision: INCONCLUSIVE; keep GRIL as a research baseline, promote neither
method on 0827.** GRIL and LI-Init descend from related work; this is a
comparison of runnable systems under their own sufficiency gates, not proof of
independent algorithmic superiority.

| Layer | Frozen evidence | Meaning |
| --- | --- | --- |
| Data | Three Apollo shards; the *same* ROS bag provided 1,375 Vanjee-16 scans, 35,239,034 finite points and 13,741 IMU samples over ~137.4 s to GRIL and LI-Init. 16 ordered ring elevation bands, ~100 ms scan point times, IMU gaps at most 20.125 ms. [Input contract](../../outputs/gril/research_0827_preflight/input_contract.yaml), [LI-Init bag inventory](../../outputs/gril/research_0827_liinit_user/bag_info.txt). | Format passes, **not** an external accuracy benchmark. |
| Clock | Existing bag used `/apollo/sensor/gnss/imu`: sampled raw `measurement_time=0`, so header time is not independently verified acquisition time. Corrected-IMU headers are regularly spaced at 10 ms, but raw/corrected values differ (gyro RMSE 0.00987 rad/s, acceleration RMSE 0.582 m/s² under nearest-20-ms pairing). [Signal audit](../../outputs/gril/research_0827_preflight/diagnostics/record_signal_audit.yaml). | Do not relabel previous estimates as corrected-time results or infer an offset solely from publish time. A corrected-IMU run is a **new dataset experiment**. |
| Motion / visualization | INS BEV shows repeated crossing turns, not a closed loop. [Time-colored plot](../../outputs/gril/research_0827_preflight/diagnostics/ins_trajectory_bev_time.png). INS pose uses the same fusion chain as the calibration IMU. | A figure-eight shape is not evidence of 3-axis excitation or independent extrinsic truth. |
| GRIL | Three complete runs reported with the same dataset/config/executable identity (executable SHA-256 `cf8c5585f8da7ef6b1428cf58157cb7b3d54cf338715dd4ff0a2faca501cce34`): max pairwise rotation 0.05387°, translation **0.040287 m**, time 0.00008 s. Gates: ≤0.2°, ≤0.03 m, ≤0.001 s. Frozen batch replay twice produced identical result bytes. [Repeatability](../../outputs/gril/research_0827_repeatability/repeatability.yaml). | **Result exists, translation repeatability fails.** Batch replay localizes spread upstream of the frozen solver, not to a proven specific frontend cause. |
| GRIL timing diagnostic | On the original full-run batch trace, `yaw_z` correlation peaks for four contiguous windows: 0.986/0.985/0.992/0.992, at **-28/-30/-42/-42 ms** (`imu(t+offset)` against `lidar(t)`). Peak spread is 14 ms, above the predeclared 10 ms diagnostic gate; three-axis norm also fails its window-offset gate. [YAML](../../outputs/gril/research_0827_repeatability/diagnostics/gril_batch_time_profile_yaw_z_sigma_0s.yaml), [plot](../../outputs/gril/research_0827_repeatability/diagnostics/gril_batch_time_profile_yaw_z_sigma_0s.png). | High correlation does **not** validate time offset or spatial extrinsics; do not equate the profile convention with GRIL's solver time-lag sign. This is a screening check on one run, not a calibrated clock. |
| LI-Init | The same bag yielded **no transform**; final displayed X/Y/Z rotational sufficiency was 2%/2%/37%. It warned that ~100 Hz IMU is below its recommended 150 Hz. [Run log](../../outputs/gril/research_0827_liinit_user/liinit_run.log), [empty result](../../outputs/gril/research_0827_liinit_user/src/lidar_imu_init/result/Initialization_result.txt). | Record as `no_result_insufficient_excitation`, not zero error and not proof that GRIL is more accurate. The 100 Hz warning is advisory; do not lower the gate to force a result. |

**Win/loss:** Neither method meets an independent-accuracy decision rule.
GRIL emits a candidate but fails translation repeatability; LI-Init emits no
candidate. No matched extrinsic error, independent physical holdout,
candidate-wise deskewed map comparison, or runtime-cost comparison is
established. Keep the no-result in the denominator when reporting coverage.
The repeat runs' `dataset/dataset.yaml` files are absent: the equal input
identity comes from their archived report, not a fresh hash of those two
files. The original GRIL executable digest is an *archived run identity*; a
later instrumented executable is different and must not silently replace it
in an A/B test.

## Weilan 0827 corrected-IMU working fixture

**Decision: use this as the current GRIL working input profile, not as an
accuracy benchmark.** This is a separate dataset identity from the historical
`/apollo/sensor/gnss/imu` experiment above. It uses
`/apollo/sensor/gnss/corrected_imu`, whose header was audited as converted INS
measurement time for this capture. Do not combine result rows, canonical
hashes, or repeatability claims across the two topics.

The corrected-IMU Apollo record and trusted ROSbag canonical datasets matched
exactly in the validated comparison: 1,375 scans, 35,239,034 finite points,
13,741 IMU samples, plus ordering, timestamps, frames, rings, point times,
numeric LiDAR/IMU arrays, and static transforms. This verifies extraction
equivalence only. The frozen GRIL reference produced identical result bytes
in two runs, but the independent physical diagnostics rejected the candidate:
held-out lever-arm residual improved only 1.13%, and contiguous yaw/time
holdouts did not support a unique generalizing solution. Verdict:
`rejected` for the reported extrinsic; `review-only` as a reproducible
conversion/reference fixture. Artifacts are stored outside the repository at
`/mnt/synology/whl-cal-gril-validation-20261114/weilan_0827_gril_reference/`.

The prior `wheelos_core` ingress baseline and its array hashes use the raw IMU
topic and remain useful for their original ingress/runtime regression purpose.
They are not the corrected-IMU comparison target. The current native ROS-free
implementation has a separate migration gate: its full A/B rotation
discrepancy is 0.406964 deg versus a 0.2 deg gate. It must not be described as
validated by either frozen-reference run above.

## P0-P3 method matrix and next experiment

These priorities are **experiment order, not an accuracy ranking**. A
method's paper result is not an observation on our INS-fed vehicle captures.
0827 is the chosen working capture even though it is not yet an
extrinsic-accuracy benchmark; keep its failures in the comparison rather than
silently replacing it with another dataset. Upstream source checks below
refer to LI-Calib revision `4c75b8d2f60fb22d9b8ee99c7d4e32b53781a787`
and AFLI-Calib revision `2fc27e4dab0ef43302a27ff49b4e1f1735a19414`.

| Priority / method | Weilan 0827 evidence and status | Admission gate / smallest next experiment |
| --- | --- | --- |
| P0 native GRIL | Historical raw-IMU runs fail the 3 cm translation-repeatability gate and disagree in time windows. Separately, corrected-IMU frozen-reference runs match each other but fail independent physical checks; native full A/B remains above its rotation gate. | Keep these experiment identities separate. Pin corrected-IMU canonical/config/executable digests, repeat the native **full** frontend, pass frozen-reference equivalence, then evaluate an independent physical holdout. |
| P1 LI-Init | `no_result`, not a spatial-error value: displayed X/Y/Z progress 2%/2%/37%, IMU ~100 Hz versus its suggested >150 Hz. The [gate reconstruction](../../outputs/gril/research_0827_repeatability/diagnostics/li_init_gate_audit.yaml) reproduces the failure from its own LiDAR-only state log; a one-factor voxel rerun removes frontend warnings but still yields no result. Related to GRIL, so no independent-family win is implied. | **0827 failure diagnosis complete, calibration incomplete**: retain the unmodified sufficiency gate and the no-result. Independently assess LiDAR odometry motion quality before claiming the physical input rather than its odometry is insufficient; do not inflate frame count or resample IMU to manufacture sufficiency. |
| P2 LI-Calib | `blocked_input_adapter`, not run, accuracy/runtime unknown. The actual 0827 `/velodyne_points` first message is **unorganized** `PointCloud2`, `height=1`, `width=25291`, with `x/y/z/intensity/time(float32 relative seconds)/ring`. LI-Calib's [reader](https://github.com/APRIL-ZJU/lidar_IMU_calib/blob/master/include/utils/dataset_reader.h) accepts PointCloud2 (the default launch selects `/velodyne_packets`), but its [PointCloud2 path](https://github.com/APRIL-ZJU/lidar_IMU_calib/blob/master/include/utils/vlp_common.h) ignores `time`/`ring` and calls `getExactTime(h,w)` from a VLP-16 table `[1824][16]`: on this first scan `h=0, w>=1824` is out of range. | Implement/test a sensor-aware PointCloud2 input path preserving each point's measured time/ring and require source-to-decoded timestamp/geometry agreement **before** optimization. Do not relabel Vanjee as VLP-16 or fabricate packets; after adapter acceptance check motion sufficiency. |
| P3 AFLI-Calib | `blocked_input_adapter`, not run, accuracy/runtime unknown. Official [selector](https://github.com/DCSI2022/AFLI_Calib/blob/main/Parameter_Descrip.md) lists Livox, Velodyne, Ouster, Hesai. The [Velodyne and Ouster branches](https://github.com/DCSI2022/AFLI_Calib/blob/main/include/io.cpp) currently only print a label, without appending scans. Its Hesai parser expects `time` as **float64 absolute timestamp**, whereas 0827 has float32 relative seconds. Its intended capture is fast portable/non-repetitive motion. | Check physical sensor and motion fit; only then provide a validated Vanjee parser mapping relative time and ring, and independently audit IMU stream selection/still start. Never select `HESAI` solely because it loads PointCloud2; else record `not_applicable`, **not** algorithm failure. |

**Dataset matrix:** 0827 has separate raw-IMU historical negative/unstable
evidence and a corrected-IMU working fixture whose physical verdict is
rejected; neither is an accuracy benchmark for P2/P3. Weilan 0612 lacks usable Z-turn
excitation after overlap cropping. Zhongji 05-06 has verified corrected
INS-IMU time and 32 rings, but GRIL timing/seed sensitivity and frontend
instability; Zhongji 05-07 short runs lack turns and its first-five-shard
eight-shaped run has IMU gaps, so do not differentiate across that window.
Ruantong parking-04 has promising first-shard turns and IMU continuity but
`lslidar_main` ring decoding is unresolved. See
[screened captures](../../context/knowledge_base/validated_conclusions.md)
and the [capture matrix](../../outputs/gril/research_multicapture_20260929/dataset_matrix.yaml).
None of these constitutes a shared, sufficiently excited, independently
referenced four-method benchmark.

**P1 failure localization on 0827:** The same ROS bag's first/middle/last
scans have valid relative point time ending at 99.983 ms and 16 rings;
respectively 8,431/8,422/8,660 points survive a *sampled* check of the
configured `point_filter_num=3`, `blind=2 m` and ring gate. These three
samples do not establish an all-frame acceptance ratio. The 100 Hz IMU
message is a warning in `laserMapping.cpp`, not a stop condition. LI-Init
begins accumulating once LiDAR-only position exceeds 0.05 m (logged state
index 181, cumulative traveled distance 0.561 m). Its
`LI_init.cpp:data_sufficiency_assess` appends the skew of its **CV angular
velocity** (`state.bias_g` in LiDAR-only mode, `mat_out.txt` columns 16–18)
to a 3-column Jacobian. The final 5,261 accumulated states (of 5,442
logged LiDAR states) end at 137.231 m **cumulative LiDAR odometry distance**;
the last column in `mat_out.txt` is distance, **not elapsed time**.
Reconstructing `J.T @ J`, with eigenvalues
31.098/484.788/488.611, and testing all three eigenvalue-pair products
divided by `800²` yields **0.02356/0.02374/0.37011**, all below the
unaltered **0.99** gate. The final 2%/2%/37% progress is therefore
explained by the actual gate, not by the final zero-valued shutdown print
or the IMU-rate warning. Source: [LI-Init run log](../../outputs/gril/research_0827_liinit_user/liinit_run.log),
[numerical reconstruction](../../outputs/gril/research_0827_repeatability/diagnostics/li_init_gate_audit.yaml).
This identifies a **LiDAR-motion/frontend observability blocker**, not
proven physical lack of motion: the raw IMU gyro RMS is only
0.019/0.022/0.284 rad/s in sensor x/y/z, and the raw IMU uses an unverified
publish-time header. The archived run received SIGINT after playback and
did not write an initialization result; the state log does not itself
provide elapsed acquisition time. The
progress labels are assigned by an eigenvector heuristic, not independent
ground-truth excitation measurements for the named axes. No P1 extrinsic
can be accepted from this run; a corrected-IMU run would be a separately
identified input experiment for P0 **and** P1, not a retroactive correction
of this raw-IMU comparison.

**One-factor frontend ablation (same 0827 bag SHA-256 `f6bfc346...`, same
LI-Init executable SHA-256 `2dcc972d...`):** unchanged 800/0.99 gate and
all other algorithm settings, `filter_size_surf` 0.05 → 0.10 m in an
[isolated launch](../../outputs/gril/research_0827_liinit_p1_voxel/voxel_ablation.launch).
The candidate uses cached image SHA-256 `7643f8d1...` with ROS Noetic,
Eigen 3.3.7, Ceres 1.14.0; the baseline's image digest was not archived,
so cross-image numerical identity is not proven.
The baseline has nine `VoxelGrid` overflow warnings; the candidate has
zero. In sampled scans, 0.05 m yields an approximately 2.22–5.09 billion
voxel *index span* product (above the signed 32-bit limit), explaining
the warning; this is not the actual number of occupied cells. Both runs
log 5,442 states, and the candidate's final pair scores are
0.02712/0.02714/0.36661 versus 0.02356/0.02374/0.37011 at baseline.
**P1 still produces no transform**: removing the warning does not
remedy weak three-axis motion support. Its odometry distance also changes
(137.231 → 140.320 m), so the frontend is not invariant to voxelization.
Keep 0.10 m as a diagnostic candidate only, not a validated improvement
or a reason to weaken the unchanged gate. [Reconstructed comparison](../../outputs/gril/research_0827_repeatability/diagnostics/li_init_gate_audit.yaml).

**0827 same-settings visual diagnostic (P0 only):** applying two archived GRIL
solutions to all three shards with the same per-point SE(3) INS-odometry
interpolation, stride 20 scans/32 points, 0.1 m voxel size and 2000 thickness
samples produced [full-run](../../outputs/gril/research_0827_repeatability/diagnostics/holdout_full/submap_metrics.yaml)
and [repeat-run](../../outputs/gril/research_0827_repeatability/diagnostics/holdout_repeat/submap_metrics.yaml)
maps ([full BEV](../../outputs/gril/research_0827_repeatability/diagnostics/holdout_full/imu_extrinsic_submap_bev.png),
[repeat BEV](../../outputs/gril/research_0827_repeatability/diagnostics/holdout_repeat/imu_extrinsic_submap_bev.png)).
Both use 69 scans and 50,021 sampled source points. Full-run median/p95
local thickness: 0.126/0.375 m (405 fitted neighborhoods); repeat-run:
0.125/0.308 m (429 neighborhoods). The evaluated neighborhoods differ after
candidate-dependent voxelization and planar selection: **these figures do
not rank extrinsic accuracy**. The pose is derived from the same INS chain,
not an independent holdout. These are inspection artifacts, not a P0 pass.

**Gate for the next comparison:** qualify one continuous, clock-audited
capture per claimed regime and a separate independent physical holdout;
freeze common LiDAR/IMU *values* and timestamps, frame direction and split.
For any admitted method, retain its source revision, build/adapter digest,
configuration, no-result status and cost. Compare identical train/holdout
windows, three complete runs, seed perturbations, per-parameter errors where
truth exists, and same-crop deskewed point clouds. Require P0/P1 to pass
their sufficiency/repeatability gates before interpreting the spatial
comparison. P2/P3 must first pass sensor/clock/motion compatibility; until
then report `blocked`/`not_applicable`, not zero error or a loss. Do not
reprocess 0827's raw-IMU historical result as corrected-IMU evidence.

## 1. Data evaluation (before running a solver)

- Inventory **one split-family entry** (avoid expanding the same family
  repeatedly), sensor identity, topics, frames, static TF direction and
  overlap. Record point count, scan lines validated by per-ring geometry,
  point time origin/span, finite ratio and monotonicity.
- Compare record/header/measurement times **and IMU values per topic**.
  Preserve INS measurement-time provenance. Check gap boundaries on source
  LiDAR and IMU; split into continuous windows, never differentiate across
  gaps. Distinguish `format_accepted`, `time_accepted`, and
  `parameter_observable` rather than one generic pass.
- Quantify startup stationary duration, left/right turn counts and
  accelerations, ground/structure visibility, and each method's own
  sufficiency. A weak capture remains in the matrix as a negative control;
  it is not discarded from no-result statistics.
- Freeze source paths, hashes, crop, sensor/TF conventions, topics, canonical
  input and quality report. A clock or corrected-IMU change requires a **new
  input identity for both methods**, not an apparent method improvement.

## 2. Algorithm iteration (one falsifiable change)

- **Round N baseline:** frozen input + config + actual executable/container
  digest + three complete runs + frontend/batch traces. Do not mistake the
  CMake `gril_native_full_frontend` library for the runnable
  `gril_native_full_frontend_exec` target.
- **Round N+1 candidate:** state one hypothesis and one changed factor
  (screening, correspondences, initialization, objective, or solver) with
  predicted gain and failure mode. Keep the original executable and
  thresholds available; log crashes, no-results and time/operator cost.
- **Comparison:** same sensor samples, train/holdout windows, hardware and
  evaluation definitions. Compare complete-run repeatability **before**
  optimizing on a fixed batch: deterministic batch replay cannot vindicate
  unstable frontend states. Perturb seeds and thin data only as labeled
  separate trials. If the candidate changes time or IMU values, rerun both
  systems on that new frozen input.
- **Abort/fallback:** rollback if early windows regress, frontend changes
  solution family, no-result rate grows, or holdout loses; never cherry-pick
  the best run. On 0827, fixing frontend repeatability and verifying the clock
  are prerequisites to promoting a GRIL change; LI-Init needs suitable
  multi-axis excitation and runtime before a spatial-accuracy A/B is possible.

## 3. Algorithm evaluation (separate from solver convergence)

Use `ACCEPT`, `REJECT`, `INCONCLUSIVE`, `NEEDS_MORE_DATA` with explicit
per-parameter reasons. Each method/capture row records `produced_result`,
`no_result` or `blocked`, run identity, quality gates, and missing evidence.
Report rotation and **x/y/z translation** separately from time; do not let
good roll/pitch/z hide weak yaw or horizontal lever arm.

1. **Primary:** surveyed extrinsic error if independently measured; otherwise
   label accuracy unknown. Same-INS odometry and configured TF are *not*
   independent reference. Use physical cross-capture holdout and same-crop
   candidate-wise deskewed map sharpness/ghosting as supporting evidence,
   with their common-INS limitations stated.
2. **Robustness:** at least three full-pipeline runs per capture; same-data
   rotation ≤0.2°, translation ≤0.03 m, time ≤0.001 s **as research gates**,
   plus seed/window spread and explicit no-result rate. Correlation peaks
   must be interior, sufficiently strong and within the declared offset
   spread; their magnitude alone is not time accuracy.
3. **Review:** show time-colored BEV with gaps/turns, LiDAR-only versus INS
   relative yaw (motion check only), equivalent cropped point-cloud views,
   and raw/matched correspondence counts. The trace's old zero residual
   was an instrumentation error: do not compare across executable digests.
4. **Promotion:** require equal or better **independent** accuracy on multiple
   qualified captures, stable repeatability/holdout, no hidden failures and
   manageable runtime. If either method does not produce an estimate, score
   operational coverage, **not** extrinsic error; never declare an accuracy
   winner on 0827 alone.

Reuse the existing `dataset.yaml` / `input_contract.yaml`, `manifest.yaml`,
`GRIL_full_frontend_trace_v1.txt`, `GRIL_batch_trace_v1.txt`, `metrics.yaml`
and `diagnostics/` where supplied; keep result and review artifacts separate.
See the [benchmark matrix](../../outputs/gril/research_multicapture_20260929/dataset_matrix.yaml)
for other captures and the [GRIL iteration report](../../outputs/gril/research_20260506_canonical_32line_corrected_imu/iteration_report.yaml)
for a rejected one-factor frontend ablation. Reuse their schemas; do not
introduce a second record-decoding stack.

## Zhongce 2026-11-17 dual-corner FlatBuffer LiDAR + corrected-IMU stabilization

**Decision: `converged_and_verified` (`verdict: accepted` under post-run Stage 3
quality gates) on the `[0, 680 s]` pre-gap window for both `left_front` and
`right_back` 32-line corner LiDARs against `/apollo/sensor/gnss/corrected_imu`
(`measurement_time`, `H = 0`), while preserving frozen GRIL defaults.**

- **Source & canonical ingress:** `/mnt/synology/中策/2026-11-17/bag` (`10,912`
  FlatBuffer corner scans each on `left_front` and `right_back`, `108,621`
  `corrected_imu` samples at `100 Hz`, 32 elevation rings, continuous `[0, 680 s]`
  pre-gap calibration segment and `[785, 1005 s]` post-gap unseen holdout; fused
  LiDAR excluded). Canonical datasets live at
  `/mnt/synology/whl-cal-gril-validation-20261119/{left_front,right_back}/dataset`.
- **Diagnosed failure mechanisms in unmodified baseline GRIL:**
  1. **Dual-layer frontend non-determinism:** asynchronous `ikd-Tree` background
     `pthread_create` swaps cause immediate trajectory divergence across reruns,
     and `-fopenmp` from `${PCL_DEFINITIONS}` triggers Eigen 3.3.7's multi-threaded
     `GeneralMatrixMatrix.h` 1-ULP (`~1e-16`) race when `effective_feature_count >= 174`,
     compounding over `~20,000` EKF updates. Opt-in `--ikd-rebuild sync` disables
     async tree threads and locks single-thread Eigen/OpenMP GEMM, achieving
     **3/3 bit-exact SHA-256** across both `[0, 160 s]` and `[0, 680 s]` runs.
  2. **Start spur ("起点毛刺"):** `cut_frame=3` equal-point-count splitting on
     occluded corner FOV + hardcoded `kImuHz=200` `dt=0` resets trip
     `pos_end.norm() > 0.05 m` at `t ≈ 3.7 s` while parked (actual departure
     `t ≈ 19.2 s`). Fixed opt-in via `--cut-mode equal_time --min-cut-points 80`,
     `--ahrs-imu-hz 100 --ahrs-dt-mode continuous`, and speed/omega motion gates.
  3. **Mid-window vertical "banana" drift & end blow-up ("终点毛刺"):**
     frame-inconsistent pseudo-ground EKF rows in `LidarOdometry.cpp` + close-wall
     feature starvation (`< 1.5 m` wall starving `cut3` to `0..12` features ->
     `+244.4 m` Z spike and premature `pct_z > 0.99` stop at ~13% of the bag).
     Fixed opt-in via `--odom-ground-mode disabled` and
     `--min-effective-features 30` (`max_step`: `15.24 m -> 0.21 m`; `p95` tilt:
     `3.36°` / `5.68°`).
  4. **Batch lag & lever-arm EIV dilution:** unbounded unnormalized
     `xcorr_temporal_init` (`+48.37 s` / `-2.54 s`) and noisy 30 Hz differentiated
     LiDAR `ang_vel`/`ang_acc` in `Linear_acc_Rot_Cost_without_Gravity` (~90%
     lever-arm shrinkage when unbounded). Fixed opt-in via bounded Pearson
     `max_xcorr_lag_s`, `max_time_lag_2_s`, `rot_init_max_iters`,
     `use_imu_angular_in_acc_cost: true`, and `skip_secondary_acc_norm: true`.
- **Stage 3 post-run quality-gate evidence (`lidar2imu evaluate-quality-gates`):**
  - **`left_front` (`[0, 680 s]`, unbounded):** `R_LI` XYZ `[+0.036°, -0.075°, +132.687°]`,
    `T_LI = [-0.1513, +2.0156, +0.4767] m`, `tau = -22.109 ms`. 3x repeatability:
    `0.0° / 0.0 m / 0.0 s` (`SHA-256 241fdeba...`); 4-seed basin (including blind
    `I + [0,0,0]`): `0.46° / 5.9 cm / 0.002 ms`; 3-window spread:
    `2.09° / 8.3 cm XY / 2.19 ms`; unseen `[785, 1005 s]` holdout (`6,519` states):
    gyro RMSE `0.0252 -> 0.0141 rad/s`, acc RMSE `0.6442 -> 0.6118 m/s²`.
  - **`right_back` (`[0, 680 s]`, unbounded):** `R_LI` XYZ `[-0.006°, -0.295°, -46.376°]`,
    `T_LI = [+1.6166, -3.7449, +0.4565] m`, `tau = -22.792 ms`. 3x repeatability:
    `0.0° / 0.0 m / 0.0 s` (`SHA-256 7d479f06...`); 4-seed basin (including blind
    `I + [0,0,0]`): `0.53° / 2.2 cm / 0.035 ms`; 3-window spread:
    `1.64° / 31.3 cm XY / 4.92 ms`; unseen `[785, 1005 s]` holdout (`6,520` states):
    gyro RMSE `0.0171 -> 0.0146 rad/s`, acc RMSE `0.8249 -> 0.7113 m/s²` (`-13.8%`).
  - **Dual-corner physical agreement & 5-pillar cross-diagnostic (`lidar2imu evaluate-dual-diagnostics`):**
    `Yaw_LF - Yaw_RB = +179.05°` (`0.95°` from nominal `180°` opposite-corner
    geometry), `Δz = 2.0 cm`, and independent shared-IMU clock offset agreement
    `Δtau = 0.68 ms` (`-22.11 ms` vs `-22.79 ms`). Running the 5-pillar
    dual-sensor diagnostic (`outputs/gril/stage2b_stabilization_20261120/dual_sensor_diagnostics/`)
    established two durable validation lessons:
    1. **Turn pure-IMU submap confirms crisp single-sensor `R_LI` & `tau` while breaking the static gauge blind spot:**
       over `[508.5, 511.1 s]` (`Δψ = 54.0°`), `100 Hz` IMU-deskewed wall
       thickness is `5.60 cm` (`1.85x` sharper than zero lever/lag `10.34 cm` and
       `1.32x` sharper than an injected common-mode IMU bias that is invisible in
       static dual-PCDs); road-slope-cancelled IMU tilt bias across 12 headings
       is `0.51°`; 3-curve yaw-rate RMSE is `0.0129 / 0.0185 rad/s` and active-turn
       opposite-corner acceleration discrepancy drops `-63.3%` (`1.138 -> 0.417 m/s²`).
    2. **Colored dual-PCD + orthogonal wall slices + direct `L2L` GICP expose ~13.5% loose-coupled EKF lever-arm scale shrinkage (`verdict: review_only` for joint cloud fusion):**
       although single-sensor `lidar2imu` repeatability (`0.0 m`), seed basin
       (`2–6 cm`), and EKF holdout (`-13.8%`) all pass, direct 3D wall overlays
       and 5 independent stationary GICP stops (`t = 8.7..1080 s`, `±2.5 cm`
       consistency) prove the physical sensor baseline in IMU RFU is
       `[+2.16, -6.65, -0.13] m` vs `lidar2imu`-induced `[+1.77, -5.76, -0.02] m`
       (`Δx = +38.3 cm`, `Δy = -89.0 cm`, `Δyaw = -0.95°`). Finite-bandwidth
       30 Hz LiDAR EKF velocity differentiation attenuates tangential `Δa_L`
       peaks relative to `α_IMU × T` on low-acceleration trucks (`|ω_z| <= 0.4 rad/s`),
       shrinking `T_LI` equally in training and holdout.        Therefore, for multi-LiDAR point-cloud fusion, `lidar2imu` provides the
       shared IMU gravity/yaw/timing gauge while relative `T_{A<-B}` must be
       cross-checked (or locked) with structured `lidar2lidar` registration.

### Joint `2 × L2I + upgraded L2L` calibration validation (`calibrate-dual-joint`)

Implemented and validated on Zhongce `2026-11-17` (`[0, 680 s]` calibration,
`[0, 340 s]` vs `[340, 680 s]` split-half, and unseen `[785, 1005 s]` holdout;
see [design doc](../../docs/dual_lidar_imu_joint_calibration_design.md)):
- **Unseen holdout `[785, 1005 s]` comparison (`evaluate-dual-diagnostics`)**:
  - Baseline independent GRIL (`review_only`, `2/6` checks): static ground
    Z-gap `13.79 cm` (`1.59°`), static wall gap `49.42 cm`, turn cross-sensor
    wall gap `46.68 cm`, L2L loop residual `58.50 cm / 1.77°`.
  - Joint `v1` (anisotropic `SE(3)^2` + GICP L2L + gravity): static ground Z-gap
    `0.87 cm`, wall gap `0.23 cm`, turn A-B gap `9.17 cm`, semantic L2L residual
    `14.92 cm / 0.42°`, gravity normal alignment `0.46°`.
  - Joint `v2` (upgraded 4-channel semantic + dense ground + intensity +
    3D-FoV L2L + dynamic-traffic window filtering + multi-window consensus):
    static ground Z-gap `4.53 cm`, wall gap **`0.40 cm`**, semantic L2L loop
    residual drops **24× vs baseline** to **`2.37 cm / 0.219°`** (vs bare GICP
    corridor slip `35.9 cm`), gravity tilt bias **`0.272°`**, and split-half
    `w1 [0, 340 s]` vs `w2 [340, 680 s]` `T_A_B` repeatability improves **5.8×**
    over initial `v1` (`67.56 cm -> 11.62 cm`, `0.373°`; `v1` with dynamic filter
    also improves to `8.04 cm / 0.161°`).
  - Joint `v3` (`100 Hz` turn-IMU preintegrated dual-LiDAR BA + velocity-level
    kinematics + `0.8 ms` shared-clock constraint + frozen `z_cm` +
    Leave-One-Turn-Out cross-validation guard): achieves `6/6` in-sample on
    `[0, 680 s]`, while LOTO out-of-sample CV detects `4.388 cm -> 4.513 cm`
    and **automatically reverts final selection to `v2`** (no out-of-sample
    overfitting leaked into exported `extrinsics/`).
- **Full 23-shard (`1092.2 s`) empirical validation, alternative-paradigm
  elimination, and Common-Mode Excitation Gate (2026-11-21):**
  - Across all 4 independent parking locations in the full `1092 s` capture
    (`t = 5 s, 520 s, 1035 s, 1080 s`), `semantic_l2l` achieves **`1.75 cm`**
    median spread (`max = 3.96 cm`, per-axis std `1.86 / 1.09 / 0.91 cm`).
  - Across 4 non-overlapping driving segments (`S1: 0-220s`, `S2: 220-450s`,
    `S3: 450-680s`, `S4: 785-1005s`), `Joint v2` keeps `v2_yawA + straight_slip_angle`
    invariant at **`134.42° ± 0.08°`**, proving that `~2° / 40 cm` common-mode
    drift in `S1` (`28°` left vs `270°` right turns) and `S4` (`400°` left vs
    `53°` right turns) originates from one-sided/low-acceleration L2I excitation
    rather than sensor instability.
  - Three heavier coupled alternatives were falsified by real-data A/B tests and
    retired from the production path: (a) `Old v3` (`2.6 s` turn BA: `18.6 s`,
    100% LOTO rollback due to `2.4%` Sagitta CRLB), (b) `Scheme 3` (`L2L` ->
    `360°` merged cloud into `GRIL`: `478.9 s`, still suffers `GRIL EKF`
    shrinkage to `1.46–1.86 m`), and (c) `Paradigm 7` (`6.5 s` through-turn
    vectorized `BALM2` BA without `lid_v`: `8.6 s`, `0.5° * 0.5*g*dt^2 = 1.81 m`
    gravity coupling swamps cm lever-arm curvature and degrades Holdout
    `grav_bias` `0.280° -> 0.773°`).
  - **Deployed algorithmic gate (`evaluate_common_mode_excitation_gate`)**:
    post-run gate requiring `min(left_deg, right_deg) >= 90°`,
    `turn_symmetry_ratio >= 0.25`, `p95_planar_acc >= 0.50 m/s²`,
    `max_abs_wz >= 0.32 rad/s`, and `healthy_l2l_edges >= 1`, plus explicit
    `z_cm_observable: false` under planar motion. Properly rejects `S1/S2/S4` as
    `needs_more_excitation` and accepts `S3` and `Full_0_680s` (`Yaw` spread
    `0.25°`, baseline `7.060 m` vs `7.125 m`).
  - **Only high-ROI remaining improvements**: stop stacking backend preintegrated
    BA on `<= 14 km/h` logs; focus exclusively on (1) capture excitation upgrade
    (`20–25 km/h` braking/accel `>= 1.5 m/s²` + symmetric figure-8s) and/or
    (2) physical CAD/laser-tape `(t_x, t_y, t_z)` priors.


