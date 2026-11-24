---
audience: dev
stability: evidence
date: 2026-11-24
---

# Zhongce L2L baseline A and the next data-expansion experiment

## Current decision

**Freeze group A as review-only; test qualified additional data before adding
another registration algorithm.** This refresh reproduces the current
estimator, not an accuracy improvement or promotion of new moving windows.
Production direct `scan2scan` / GICP remains unchanged.

Active local artifact directory:
`outputs/lidar2lidar/zhongce_20261117_baseline_20261124/`.
Previous reference:
`outputs/lidar2lidar/zhongce_20261117_window_audit/mainline/`.

Open the [baseline metrics](../../../outputs/lidar2lidar/zhongce_20261117_baseline_20261124/metrics.yaml),
[freeze contract](../../../outputs/lidar2lidar/zhongce_20261117_baseline_20261124/baseline_contract.yaml)
and [actual feature/match gallery](../../../outputs/lidar2lidar/zhongce_20261117_baseline_20261124/run/diagnostics/l2l_features/index.html).
These local output links are unavailable in a clean checkout.

The actual source tree contains earlier uncommitted changes. The refreshed
baseline pins those Python files and a source archive; Git HEAD alone does
not identify its implementation.

## Dataset and estimator contract

- Original capture: `/mnt/synology/中策/2026-11-17`.
- Target A: `left_front`; source B: `right_back`.
- Transform: `p_A = R_A_B @ p_B + t_A_B`; information uses the right-source
  `[rotation, translation]` tangent. Do not assume native IMU X is vehicle
  forward; infer/verify the display frame independently.
- Training: frozen GRIL results from `window_full_0_680s` under
  `outputs/gril/stage2b_stabilization_20261120/exp2_windows_holdout/`.
- Group A: the existing three static stacks, approximately 8.7, 530.1 and
  292.4 seconds, with unchanged screening, extraction, weights, priors and
  correspondence schedules.
- Joint v1 initializes v2. V1/v2 optimize two spatial extrinsics, not time
  offsets; training time offsets and GRIL outputs are reused.
- Fresh bounded validation replay uses only the two previously extracted
  low-speed stacks at approximately 827.2 and 864.3 seconds. It does **not**
  reevaluate every scan in 785-1005 seconds or rerun GRIL.
- These validation data have been reused in earlier iterations. There is no
  new blind sequence or independently surveyed extrinsic ground truth.

## Refreshed results

| Check | Current baseline A |
| --- | --- |
| V1/V2 transforms versus previous baseline | agree at matrix `atol=1e-6`, `rtol=0`; not bitwise equivalence |
| Training static / gravity windows | 3 / 12 |
| Windows not weak-support-downweighted | only `static_1` |
| Maximum relative seed change from nominal | 2.73 mm / 0.0168 degrees |
| Maximum fixed-factor window-deletion change | 7.29 cm / 0.1424 degrees |
| Maximum pairwise training-window disagreement | 11.51 cm / 0.6040 degrees |
| Operational status | review-only; no absolute-accuracy claim |

Seed trials perturb only the semantic seed by yaw +/-1 degree, with source
translations `[0.1, -0.1, 0]` and `[-0.1, 0.1, 0]` m respectively.
L2I priors, gravity and graph initialization stay fixed; semantic
registration and consensus rerun. These are not new GRIL initializations or
arbitrary full-pipeline perturbations. All three graph trials succeeded.

Window deletion holds the remaining factors, weights and initialization fixed.
It does not rerun screening, registration or consensus and is not full-pipeline
jackknife or calibrated uncertainty. Training-window disagreement compares
local pair estimates; it is not error against truth.

The native result's `verdict: accepted` has
`verdict_scope: common_mode_excitation_only`. It does not override L2L
review-only status, establish the common IMU datum, or observe prior-anchored
common vertical height.

## Reproduce and inspect

Run from the repository root with the pinned source/dependencies and a fresh
output directory:

```bash
.venv/bin/python -m lidar2imu.cli calibrate-dual-joint \
  --dataset-a /mnt/synology/whl-cal-gril-validation-20261119/left_front/dataset/dataset.yaml \
  --run-dir-a outputs/gril/stage2b_stabilization_20261120/exp2_windows_holdout/left_front/window_full_0_680s \
  --dataset-b /mnt/synology/whl-cal-gril-validation-20261119/right_back/dataset/dataset.yaml \
  --run-dir-b outputs/gril/stage2b_stabilization_20261120/exp2_windows_holdout/right_back/window_full_0_680s \
  --output-dir outputs/lidar2lidar/zhongce_baseline_replay \
  --mode v2 --save-l2l-feature-debug

.venv/bin/python -m tools.run_lidar2lidar_feature_smoke
```

The existing feature/math/diagnostic smoke passed on this refresh. It covers
wall-strip filtering, Jacobians, tangents, fixed-factor deletion and trace
exports; it does not mean the separately documented structural counterexamples
have been fixed.

| Artifact under the refreshed directory | Role |
| --- | --- |
| `baseline_contract.yaml` | command, input/source hashes, actual loaded Open3D extension, environment and limitations |
| `source_snapshot.tar.gz` | actual dirty-tree Python source, package metadata and refresh runner |
| `metrics.yaml` | replay comparison, windows, seed sensitivity, deletion, pair dispersion and timing scope |
| `seed_trials.yaml` | every seed trial, solver status, failure list and effective edges |
| `holdout_replay.yaml`, `holdout_inputs/` | bounded reused inputs and fixed-transform NN fit/overlap evaluation |
| `holdout_overlays.png` | identical crop/stride initial-versus-v2 overlays in target-A coordinates |
| `run/diagnostics/l2l_features/index.html` | actual feature, match, rejection, raw-support and residual gallery |
| `run/diagnostics/l2l_window_influence.yaml` | native fixed-factor deletion diagnostic |
| `verification.yaml` | artifact checksums and replay checks |

The NN validation uses fixed 0.15 m voxelization and a 0.35 m correspondence
distance for both old and refreshed transforms. Fitness and inlier RMS remain
fit/overlap diagnostics, not accuracy or sufficient physical acceptance.
Timing excludes record ingress and GRIL solving; do not call it full
record-to-calibration runtime. Canonical manifests, training results/traces
and bounded extracted inputs are pinned; huge canonical point archives were
not copied or rehashed in full.

## Lessons supported by the investigations

### Geometry and actual shared support

Original matched-target horizontal normal min/max ratios are approximately
0.0422, 0.0000142 and 0.00246 for `static_1/2/3`. They describe directional
coverage, not independent wall count or calibration precision. Detected
whole-scene geometry is richer than the actual shared matches.

In `static_2`, shifting along the wall by 10 cm leaves facade RMS near
1.10 cm with 12/13 nominal sources retained; shifting along its normal by
10 cm raises RMS to about 10.39 cm. Always report retained support with RMS.
Ground plus parallel walls, repeated same-direction observations, or a curb
parallel to the weak direction do not recover the missing translation.
Nonparallel shared directions can complement across windows.

### Identity and information are different from detection

Matched target reuse and line-scale changes are warnings, not proofs of false
objects. A true pole near a wall can have a planar wider neighborhood; do not
blindly delete every such candidate. Previous target-budget and strict
intensity-mutual-NN experiments had mixed global effects and were not promoted.

Disjoint first-three versus last-three scan re-extractions produced relative
translation differences of 2.64, 1.00 and 3.22 cm. The first and third windows
are inconsistent with interpreting their raw conditional information as an
already calibrated unbiased Gaussian covariance. This rejects a combined
model interpretation, not one uniquely identified physical cause.

### Proven theoretical counterexamples, not yet product fixes

- Native full-pose Euclidean consensus can suppress a correct complementary
  partial measurement because its unobservable pose representative differs:
  weight 0.00346 and about 2.07 cm error in a known-truth factor control.
  **No actual Zhongce training edge triggered that consensus gate**, so this
  is a generalization defect, not the demonstrated current error source.
- Native cloud-internal intensity normalization changes when the background
  distribution changes. In a controlled descriptor/gate test, sixteen
  unchanged known landmarks went from sixteen accepted matches to zero.
  This is not a measured real-data count or complete sensor simulation.
- Treating uniform reflective-line samples as unique 3D point landmarks can
  fabricate tangential information. Local PCA class and high intensity do
  not establish cross-sensor physical identity.
- Continuous nuisance marginalization and correlated plane/scan errors matter.
  Toy shared-noise values are illustrative, not fitted real covariances.

Equivalent whitening roots changed the actual frozen-graph relative translation
by only about 0.088 mm. Uniform L2L information scales 0.1/1/10 changed it by at
most about 4.2 mm from nominal. Neither identifies a root explanation for
8-12 cm cross-window disagreement on this capture.

### Static labels and motion reference

Current code seeds the initial static candidate from the pre-batch startup
gap and selects later candidates using a quiet center state. This is not a
verification of every point in the full stack. Raw static stacking does not
deskew; low-speed stacking compensates between scans, not every raw point.

The initial 8.7 s stack precedes saved batch LiDAR odometry; this does not prove
it moved. Direct gyro indicates little angular movement but cannot establish
zero translation. In later near-stop stacks, reported LiDAR-only angular
excursion reached about 0.229 degrees over 0.6 s versus about 0.014 degrees
from gyro integration. Neither should be silently declared independent motion
truth or used to certify compensation.

## Next experiment: more data, same estimator

**B/C are planned, not run or promoted by this baseline refresh.** Do not wait
for every theoretical problem to be solved before testing qualified data.

1. A remains the three-window frozen baseline.
2. B adds K reliable but comparatively directionally redundant windows.
3. C adds the same K reliable windows chosen to complement the **complete
   set's** weak directions. Match speed, stack duration, point-time/deskew,
   sampling, initial seeds, priors, evaluator and window counts where possible.
   Record missing controls and unmatched conditions.

The existing training-only inventory found opportunities near 140 and 334 s,
but both are moving. At 334 s directional coverage was less stack-sensitive;
this is not evidence that its estimated extrinsic is more accurate. Qualify
fixed structure, actual point times, synchronization and compensation first.
The current low-speed fallback only runs when fewer than two static windows
exist, so three weak static stops can prevent these candidates from being
considered. Do not silently relax gates or increase their weights to force an
improvement.

Keep nullspace/identity/background controls alongside A/B/C. Compare seed
sensitivity, window removal, real match support, fixed-transform held-out
geometry and failure cases together, not just training residual or normal
spectra. Require independent new-scene evidence before release. If existing
qualified data cannot supply complementary directions, collect targeted static
nonparallel shared structures or verified independent vertical landmarks.
Do not add an NDT branch merely to compensate for missing physical geometry.

## Earlier evidence (local, gitignored)

- `outputs/lidar2lidar/zhongce_20261117_math_fix_comparison/`: numerical and
  tangent corrections.
- `outputs/lidar2lidar/zhongce_20261117_global_fix/`: planar-context line filtering,
  seed/deletion tradeoffs and fixed-parameter physical checks.
- `outputs/lidar2lidar/zhongce_20261117_window_audit/`: target reuse, influence,
  non-promoted association candidates.
- `outputs/lidar2lidar/zhongce_20261117_geometry_audit/`: shared normal coverage,
  sliding profiles, nine ideal geometry controls, twelve-bin training inventory.
- `outputs/lidar2lidar/zhongce_20261117_root_cause_audit/`: native consensus
  counterexample, descriptor/noise controls, disjoint-scan fits and gyro checks.

These are evidence paths, not guaranteed files in a clean checkout. Revalidate
against code and hashes before reuse; no artifact establishes absolute
centimeter-level accuracy or an independently benchmarked SOTA comparison.
