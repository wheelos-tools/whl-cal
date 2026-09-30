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
| P0 native GRIL | `result`, but three full runs fail the 3 cm translation-repeatability gate; time-offset windows disagree. Independent x/y/yaw accuracy unknown. | Retain the archived baseline. On the frozen 0827 input, pin executable/config/input digests, repeat the **full** frontend and evaluate an independent physical holdout; first isolate the source of frontend dispersion without tuning the solver to 0827. |
| P1 LI-Init | `no_result`, not a spatial-error value: displayed X/Y/Z progress 2%/2%/37%, IMU ~100 Hz versus its suggested >150 Hz. The [gate reconstruction](../../outputs/gril/research_0827_repeatability/diagnostics/li_init_gate_audit.yaml) reproduces the failure from its own LiDAR-only state log; a one-factor voxel rerun removes frontend warnings but still yields no result. Related to GRIL, so no independent-family win is implied. | **0827 failure diagnosis complete, calibration incomplete**: retain the unmodified sufficiency gate and the no-result. Independently assess LiDAR odometry motion quality before claiming the physical input rather than its odometry is insufficient; do not inflate frame count or resample IMU to manufacture sufficiency. |
| P2 LI-Calib | `blocked_input_adapter`, not run, accuracy/runtime unknown. The actual 0827 `/velodyne_points` first message is **unorganized** `PointCloud2`, `height=1`, `width=25291`, with `x/y/z/intensity/time(float32 relative seconds)/ring`. LI-Calib's [reader](https://github.com/APRIL-ZJU/lidar_IMU_calib/blob/master/include/utils/dataset_reader.h) accepts PointCloud2 (the default launch selects `/velodyne_packets`), but its [PointCloud2 path](https://github.com/APRIL-ZJU/lidar_IMU_calib/blob/master/include/utils/vlp_common.h) ignores `time`/`ring` and calls `getExactTime(h,w)` from a VLP-16 table `[1824][16]`: on this first scan `h=0, w>=1824` is out of range. | Implement/test a sensor-aware PointCloud2 input path preserving each point's measured time/ring and require source-to-decoded timestamp/geometry agreement **before** optimization. Do not relabel Vanjee as VLP-16 or fabricate packets; after adapter acceptance check motion sufficiency. |
| P3 AFLI-Calib | `blocked_input_adapter`, not run, accuracy/runtime unknown. Official [selector](https://github.com/DCSI2022/AFLI_Calib/blob/main/Parameter_Descrip.md) lists Livox, Velodyne, Ouster, Hesai. The [Velodyne and Ouster branches](https://github.com/DCSI2022/AFLI_Calib/blob/main/include/io.cpp) currently only print a label, without appending scans. Its Hesai parser expects `time` as **float64 absolute timestamp**, whereas 0827 has float32 relative seconds. Its intended capture is fast portable/non-repetitive motion. | Check physical sensor and motion fit; only then provide a validated Vanjee parser mapping relative time and ring, and independently audit IMU stream selection/still start. Never select `HESAI` solely because it loads PointCloud2; else record `not_applicable`, **not** algorithm failure. |

**Dataset matrix:** 0827 is the same-bag P0/P1 negative/unstable case,
not an accuracy benchmark for P2/P3; Weilan 0612 lacks usable Z-turn
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
