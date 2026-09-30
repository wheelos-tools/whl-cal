# LiDAR-to-LiDAR calibration research and scan2scan roadmap

## Purpose and decision

This note compares the repository's targetless LiDAR-to-LiDAR calibration
baseline with practical registration practice and defines controlled upgrade
experiments. Keep direct `scan2scan` as the baseline where overlap and scene
geometry support it. Do not promote an optimizer based only on convergence,
fitness, inlier RMSE, or one merged-cloud image.

Current decision: **INCONCLUSIVE for independent physical accuracy claims**.
The repository has a substantial production-oriented pipeline, but evidence
still needs to be strengthened with frozen holdout windows and independent
extrinsic truth where available. Semantic and overlap-aware registration are
reasonable candidates, not established improvements on repository data.

## Current baseline

The current architecture keeps separate data extraction, algorithm, and
evaluation stages:

- `scan2scan` is the direct pairwise baseline for strong overlap and a plausible
  initial transform.
- Pairwise methods include point-to-point ICP, point-to-plane ICP, and GICP;
  NDT is a documented candidate.
- Candidate screening considers synchronized pairs, overlap, time skew, scene
  support, and transform priors.
- Multi-window repeatability, information-matrix/conditioning diagnostics,
  scene sufficiency, topology planning, optional loop closure, and geometric
  visual evaluation are already represented in the workflow.
- `scan2map` is a separate refinement/validation path, not a blanket
  replacement for direct `scan2scan`.

For a multi-LiDAR rig, the evidence-backed policy is topology-aware: use healthy
perimeter edges as primary constraints and treat weaker diagonals as checks;
when there is no actual loop, use multi-window consensus rather than repeated
seed-only reruns.

## Comparison with practical registration practice

| Area | Repository baseline | Improvement opportunity |
| --- | --- | --- |
| Local registration | Mature ICP family with coarse-to-fine refinement and several estimators. | Compare candidates using identical windows, initialization, preprocessing, and fixed evaluation code. |
| Partial overlap | Candidate-pair overlap screening exists. | Make correspondence selection explicitly overlap-aware and measure directed overlap; do not assume all points in either cloud should match. |
| Dynamic contamination | Scene sufficiency and dynamic-related checks exist. | Test static/dynamic masking or semantic weighting on labeled examples; account for false labels and useful stationary objects. |
| Observability | Information matrix, condition number, and scene features are reported. | Validate these proxies against per-DoF transform spread and known degenerate scenes before using them as hard release gates. |
| Multi-sensor graph | Topology planning and optional graph consistency are implemented. | Require individually healthy primary edges; use loop residuals as consistency evidence, not as proof of truth. |
| Validation | Repeatability and wall/corner/slice geometry diagnostics are available. | Add predeclared holdout windows, controlled initialization perturbations, and independent measured extrinsics or physical checks. |

Open3D/PCL practice supports coarse-to-fine local registration and robust
correspondence handling. GICP changes the local surface error model; it does
not cure insufficient overlap, poor initialization, dynamic objects, or
geometric degeneracy. Pose-graph information weighting similarly helps combine
edges but cannot make a weak edge informative.

## Semantic matching and overlap-region registration

### Semantic information

Semantics can help, but the most defensible initial use is **screening or
weighting**, not replacing geometric evidence:

- Mask likely moving classes (for example, moving vehicles or pedestrians)
  when the segmentation source is reliable.
- Prefer stable structural support such as road boundaries, walls, poles, or
  building surfaces only when those classes are available and cross-sensor
  labels are consistent.
- Use semantic compatibility as a soft correspondence cost/weight or a
  conservative prefilter; retain geometric distance, normal, and robust-loss
  checks.
- Keep a geometry-only path as the baseline and fallback.

Risks: segmentation domain shift, different returns and occlusions between
LiDARs, inconsistent point labels, sparse class coverage, and accidentally
removing stationary but useful geometry (such as parked vehicles). Hard
same-class matching can reject valid correspondences or introduce class-level
false matches. Semantic ICP is therefore not automatically more accurate for
sensor extrinsics; it must demonstrate improvement on dynamic and static
holdouts without harming geometry-rich scenes.

Semantic odometry literature often uses labels to remove dynamic objects from
motion estimation. That supports a targeted dynamic-mask hypothesis, but it is
not direct evidence that semantic constraints improve fixed-rig extrinsic
calibration.

### Overlap-only correspondence and evaluation

Yes: explicitly focusing registration on shared geometry is a well-established
direction for partial-overlap point clouds. Trimmed ICP and related robust
registration methods limit the influence of unmatched points; learned
overlap-aware methods estimate likely shared regions and correspondence
confidence. For this repository, a simple overlap-aware correspondence
candidate should be tested before learned models.

Important distinctions:

1. **Overlap screening** asks whether a sensor pair/window is worth solving.
2. **Overlap-restricted optimization** limits correspondences to plausible
   shared support.
3. **Overlap-aware evaluation** scores a transform on common visible geometry
   without rewarding missing/non-overlapping returns.

The pipeline already has overlap screening; this does not mean the local
optimizer is guaranteed to use only the true overlap. A bounded candidate can
add reciprocal/one-to-one correspondence checks, robust residual weighting,
and/or trimmed correspondences with a measured overlap fraction.

Risks and safeguards:

- Cropping both clouds once using the initial TF can remove valid shared points
  if the prior is biased. Update or recompute the overlap region as the
  transform changes, or use a conservative margin.
- Trimming too aggressively can produce a low residual from a small, ambiguous
  subset. Report retained correspondence count/fraction, spatial distribution,
  and directional overlap, not only residual.
- Directed overlap can differ by source/target density and field of view; report
  both directions or a clearly defined symmetric measure.
- A transform-dependent overlap mask can favor its own estimate. Keep
  independent holdout windows and full-scene visual review.
- A large planar overlap may still leave some degrees of freedom unobservable.
  Overlap quantity is not geometric diversity.

Learned overlap predictors (for example, OverlapNet) are primarily useful for
scan-pair/loop-candidate detection and coarse orientation cues. They are not a
drop-in replacement for local extrinsic calibration or its acceptance tests.
Learned partial-registration methods such as PREDATOR or GeoTransformer are
higher-cost candidates for weak initialization/low overlap and should be
benchmarked only if the simpler candidate demonstrably fails.

## Evidence from this repository

The `record_data_0402` run documented in the design material reported average
fitness around 0.9747, average inlier RMSE around 0.0065, and minimum overlap
around 0.8425. Under stricter gates, one edge was rejected for condition number
and fitness. These values demonstrate the current pipeline's ability to screen
some weak edges; they do not establish extrinsic error against independent
truth.

The 2026-05-31 static-data review records that direct `scan2scan` produced
reviewable candidates, but high fitness was misleading in wall-dominant scenes
and seed-only full-dataset solves could switch between local solution families.
The justified next step from that evidence is better scene/window selection and
representative-transform consensus, not blind ICP retuning.

## Shared empirical-FOV ablation: first data runs

The reproducible runner is
`tools/lidar2lidar/run_shared_fov_ablation.py`. It compares full-view ICP with
the same ICP, same synchronized scan pairs, and one common source-to-target
initial transform after applying a fixed shared-azimuth crop. Per-window
results, train-only angular support, point-retention fractions, fixed-transform
holdout metrics, and colored PLY overlays are written to a fresh output
directory. The estimated angular interval describes observed returns in
training scans, not a rated sensor specification.

Reproduction commands for the two measured pairs:

```bash
python tools/lidar2lidar/run_shared_fov_ablation.py \
  --record-path /mnt/synology/中集/2026-05-07-xiaomogang/bag/20260508032341.record.00000 \
  --source-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --target-topic /apollo/sensor/vanjeelidar/right_front/PointCloud2 \
  --output-dir outputs/lidar2lidar/shared_fov_ablation/zhongji_left_front_right_front \
  --max-pairs 24 --sync-threshold-ms 100

python tools/lidar2lidar/run_shared_fov_ablation.py \
  --record-path /mnt/synology/ruantong/2026-05-06-uturn/20260430065732.record.00001 \
  --source-topic /apollo/sensor/lslidar_main/PointCloud2 \
  --target-topic /apollo/sensor/rslidar_left/PointCloud2 \
  --output-dir outputs/lidar2lidar/shared_fov_ablation/ruantong_uturn_main_left \
  --max-pairs 24 --sync-threshold-ms 50
```

The Zhongji perimeter used four adjacent sensor pairs, 22 synchronized pairs
per edge (15 train, 7 holdout), and a 100 ms maximum pairing threshold; actual
per-edge maximum scan skew ranged from 31.86 to 36.12 ms. Record TF supplied
each pair's shared initial transform.

| Dataset / pair | Data and seed | Result |
| --- | --- | --- |
| Zhongji `20260508032341.record.00000`, `left_front` → `right_front` | 22 synchronized pairs (15 train, 7 holdout); record TF used as the common seed. | Observed spans 267.8° / 267.6°; median retained points 67.5% / 67.1%. Candidate holdout spread worsened: 0.513 m / 11.69° versus baseline 0.335 m / 4.01°. Fixed-shared-support holdout RMSE was 0.269 m versus 0.247 m baseline, although fitness was higher (0.404 versus 0.365). |
| Zhongji, `right_front` → `right_back` | 22 pairs (15 train, 7 holdout); record TF seed. | Median retained points 63.8% / 70.3%. Candidate improved holdout spread (0.102 m / 0.97° versus 0.124 m / 1.54°) and fixed-support RMSE (0.242 m versus 0.261 m); fitness also increased (0.810 versus 0.771). |
| Zhongji, `right_back` → `left_back` | 22 pairs (15 train, 7 holdout); record TF seed. | Median retained points 62.8% / 62.1%. Both were highly repeatable; candidate changed median spread only from 0.007 m / 0.069° to 0.005 m / 0.061°. Fixed-support RMSE was 0.226 m for both, with fitness 0.660 versus 0.659. |
| Zhongji, `left_back` → `left_front` | 22 pairs (15 train, 7 holdout); record TF seed. | Median retained points 63.8% / 65.4%. Both were highly repeatable; baseline spread was 0.011 m / 0.121° versus 0.015 m / 0.139° for candidate. Fixed-support RMSE was 0.164 m for both; fitness was 0.797 versus 0.796. |
| Ruantong U-turn `20260430065732.record.00001`, `lslidar_main` → `rslidar_left` | 24 sampled synchronized pairs (16 train, 8 holdout) from 293 available; median skew 33.37 ms, maximum 34.04 ms. No sensor-to-sensor TF seed was available, so one full-view FPFH/RANSAC seed was shared by both variants. | Empirical spans were 358.6° and 122.1°; holdout crop retained a median 65.4% / 99.8% of source / target points. The coarse seed itself was weak (fitness 0.116, RMSE 0.573 m), and train-to-holdout transform deviations were very large (baseline median 4.95 m / 16.16°; candidate 7.23 m / 51.42°). Treat this as **no calibration result**: it demonstrates initialization failure/instability, not evidence for or against shared-FOV optimization. |

Artifacts from these runs are under
`outputs/lidar2lidar/shared_fov_ablation/`. On Zhongji, three edges are tied or
slightly improved and one edge clearly regresses, so the outcome is
pair-dependent rather than a general win. These are only 22 scans over a short
capture, and the comparison has no independent extrinsic truth. Ruantong's
shared initialization did not produce a stable solution family. No method is
promoted. Next, use a defensible seed for Ruantong or improve initialization as
a separate controlled iteration, then repeat on longer independent windows.
Keep the current baseline and report any failed initialization as no-result.

## Failure isolation: FOV estimate versus overlap model

The current results do **not** support the claim that the circular-angle
calculation has a gross wraparound or source/target transform-direction bug:

- `lookup_transform(source_frame, target_frame)` returns the source-to-target
  point transform, and the crop applies it to source points and its inverse to
  target points before measuring each bearing in the receiving sensor frame.
- For Zhongji's four Vanjee sensors, independently estimated per-training-scan
  support was very stable within each sensor: starts varied by at most about
  0.7 degrees and spans by at most about 0.6 degrees. Aggregate spans were
  about 265–268 degrees.
- This only establishes repeatable **observed return support** in this one
  capture. A return-derived azimuth envelope is not a verified hardware FOV:
  sectors with no return may be caused by scene geometry, occlusion, range,
  vehicle blockage, filtering, or actual sensor limits.

The more important limitation is semantic: the code calls this a shared FOV,
but its mask is only an angular gate. It keeps a point when its bearing from
the other sensor lies in that sensor's observed sector. It does not test whether
the other sensor could see the same surface along that ray (range and
occlusion), whether the two point clouds contain reciprocal geometric matches,
or whether retained geometry constrains all six DoF. Thus it is not an exact
common-visible-surface mask even if each angular envelope were physically
correct. It can remove useful corners or planes, retain non-overlapping
surfaces, and make ICP's local minimum more/less likely depending on the edge.
Correct FOV alone therefore does not guarantee lower extrinsic error.

The dominant Zhongji failure is weak/inconsistent pairwise registration, not
clearly a bad angle wrap:

- The `left_front -> right_front` full-view baseline already had highly
  unstable holdout solutions (median 0.335 m / 4.01 degrees from its training
  medoid; p95 0.938 m / 12.82 degrees). Shared-FOV worsened these to 0.513 m /
  11.69 degrees (p95 2.831 m / 39.68 degrees). The baseline is therefore
  already outside a trustworthy basin/solution family on this edge.
- Across the four adjacent perimeter edges, composing the independently
  estimated training medoids leaves a loop residual of about 1.19 m / 32.8
  degrees for full-view and 1.17 m / 22.9 degrees for shared-FOV. The small
  candidate improvement in loop residual is not enough to establish accuracy;
  both are grossly inconsistent.
- The other three edges were stable and showed ties or modest pair-specific
  gains/losses. This is consistent with geometry-dependent ICP behavior, not a
  uniform effect from cropping.
- Zhongji localization moved only about 0.08 mm over the capture, so vehicle
  ego-motion during the roughly 32–36 ms inter-LiDAR scan skew is unlikely to
  explain its main instability. Dynamic objects and rolling-scan effects can
  still contribute. By contrast, the Ruantong U-turn is moving and its shared
  full-view FPFH/RANSAC seed was already weak (fitness 0.116, RMSE 0.573 m), so
  that run diagnoses initialization/timing risk and cannot adjudicate FOV.

The holdout RMSE/fitness values are fixed-transform nearest-neighbor scene
scores on the same selected points, not independent extrinsic truth. Fitness
can rise while RMSE or transform consistency worsens; neither establishes that
the retained points are true shared surfaces. Do not conclude “FOV failed” from
one edge, or “FOV helped” from a higher fitness/one better edge.

### Next discriminating experiment

Before retuning the optimizer, establish mask validity against an independent
reference:

1. Verify actual sensor angular limits/boresight from configuration or a
   controlled capture; keep the observed-return envelope labeled empirical.
2. At a trusted transform, compare the angular crop against a ray/range-image
   visibility mask (including sensor origin, range and occlusion) and
   reciprocal geometric correspondences. Report retained-point precision,
   recall, spatial spread, and per-DoF information, not just point fraction.
3. On static scenes with low scan skew, perturb a trusted seed by predeclared
   translation/yaw offsets. Compare full-view, current angular crop, and
   visibility/overlap-aware crop using identical ICP, train/holdout windows,
   loop closure, transform error to the trusted reference, and no-result rates.
4. Keep the candidate only if it improves independent transform error and
   holdout stability without losing geometry or breaking perimeter consistency
   across the difficult edges. Otherwise retain baseline and fix initialization,
   scene selection, or timing first.

## Interpreting small pillar and pedestrian misalignment

When the broad scene looks aligned but poles/pillars do not quite close and
pedestrians show a small offset, do not treat both observations as one
calibration error. They have different diagnostic value:

- **Static poles/pillars are useful extrinsic checks**, but a small mismatch
  still needs classification. A consistent offset of static landmarks across
  several scans usually indicates residual extrinsic error, a biased edge
  estimate, or sensor measurement/beam-model differences. A cylinder is
  geometrically useful for its centerline and radial distance, but its
  rotational symmetry does not constrain rotation about the pole axis; one
  pole alone is not a full 6-DoF reference.
- **Pedestrians are not static extrinsic references.** Their apparent offset is
  expected when the two LiDARs observe them at different times, especially
  during a scan assembled over a rotating sweep. It can coexist with a good
  static extrinsic.

### Evidence from the available Zhongji record

The checked timing analysis for
`20260508032341.record.00000` found approximately 99.984 ms of point-time span
per Vanjee scan, with about 34–36 ms inter-LiDAR phase difference on the
affected pairs (the LF↔RF nearest-scan median is 35.985 ms; LF↔LB is
34.296 ms). At pedestrian speeds of 1–2 m/s, 34–36 ms alone corresponds to
roughly 3.4–7.2 cm of target motion. A single 100 ms sweep can span 10–20 cm
of pedestrian motion across its point timestamps. Thus centimeter-scale
pedestrian ghosting is consistent with acquisition timing and target motion,
not proof of a bad extrinsic.

The same Zhongji static capture's localization moved only about 0.08 mm
through the recorded interval. That makes vehicle motion during inter-LiDAR
skew an unlikely explanation for static-pillar mismatch in this particular
record, although it does not rule out scan-phase, return-sampling, calibration,
or object-motion effects.

The current LiDAR registration path converts each cloud to XYZ and discards
per-point timestamps (`pointcloud_message_to_open3d` in
`lidar2lidar/record_utils.py`). Consequently, it neither deskews each
approximately 100 ms scan nor aligns moving-object returns to a common instant.
Frame-level `measurement_time` synchronization alone cannot make pedestrian
surfaces simultaneous. Ego-motion deskew can correct distortion caused by the
vehicle moving during a scan; it cannot deskew independently moving pedestrians.
Dynamic points must be masked/down-weighted or excluded from static-extrinsic
acceptance.

No per-pillar centerline residuals or pedestrian motion tracks were saved in
the initial shared-FOV A/B artifacts, so those artifacts cannot quantify the
reported visual offset. The current `compute_visual_plane_metrics` measures
large wall planes, plane intersections, and axis-aligned slice thickness; it
does not calculate cylinder/pole centerline error or dynamic-object residuals.
Treat the small pillar mismatch as a hypothesis to measure, not a proven
remaining error inferred from the FOV A/B.

### Diagnose the mismatch by its spatial and temporal signature

| Observed signature | Most likely explanation | Discriminating check |
| --- | --- | --- |
| The same static poles, walls, and corners shift coherently in the same direction in every scan | Residual translation/yaw/tilt or incorrect transform composition | Compare multiple static landmarks at near/mid/far range; fit a small rigid correction and see whether all landmarks improve together. |
| Near structures align but far structures separate increasingly with range | Small angular error (often yaw for horizontal displacement, roll/pitch for vertical/height trend) | Plot signed landmark residual against range and azimuth; estimate the angular component rather than globally widening ICP correspondence thresholds. |
| A pole has a stable lateral double centerline, while walls/facades also show a consistent signed offset | Extrinsic translation component, or range-dependent angular component | Fit a cylinder/axis per sensor and compare its centerline in the rig frame across at least two poles/ranges. |
| Only one pole/one azimuth differs; other static geometry closes | Different occlusion, incidence angle, beam pattern, or sparse cylinder sampling | Compare raw returns and range/intensity distributions; do not force the transform to fit a single pole. |
| Static geometry closes but only pedestrians/vehicles are doubled, and the offset changes with scan pair/time | Dynamic target motion plus inter-LiDAR and within-sweep time difference | Compare the object's displacement with its velocity times per-point acquisition-time difference; mask the object for static calibration. |
| The entire scene shifts by a nearly constant amount between successive frames | Time-varying transform, wrong frame/time association, or uncompensated vehicle motion | Recheck point measurement-time provenance, TF direction, and pose at each scan time. |

For scale, a yaw error of 0.1° creates about 1.7 cm of lateral displacement
at 10 m; 0.5° creates about 8.7 cm. A 2 cm transverse translation produces
approximately the same absolute shift at all ranges. This makes range-stratified
pillar/plane residuals much more diagnostic than a single merged-cloud score.

### Smallest corrective sequence

1. **Separate static from dynamic evidence.** Estimate extrinsics using static
   walls, corners, and multiple poles. Remove pedestrians/vehicles with
   perception labels if trustworthy, or with cross-window/scan-to-map
   consistency. Do not tune static extrinsics to align one moving person.
2. **Measure static structure directly.** Add a review artifact that fits
   vertical cylinders or robust pole centerlines separately per sensor/window
   and reports axis-angle error, centerline distance, range, azimuth, and
   residual direction. Pair it with existing wall signed offsets and corner
   spreads. Use several landmarks at distinct ranges and non-collinear
   directions.
3. **Check point timing before adding deskew.** Preserve per-point timestamps
   through extraction and report each cloud's time range and inter-sensor
   overlap. For moving-vehicle captures, deskew using interpolated ego poses to
   a shared reference time, then compare static-structure residuals before and
   after. Keep dynamic-object residuals separate; ego deskew does not correct
   independent target motion.
4. **Only then refine the calibration objective.** If stable static landmarks
   show a coherent remaining rigid bias, estimate a small bounded correction
   from diverse static geometry and validate it on held-out landmarks/windows.
   If only one object/one pole disagrees, classify it as local sampling or
   occlusion rather than moving the whole rig transform.

**Acceptance rule:** a visually “mostly good” scene may be review-only until
static landmark residuals are measured. Promote a correction only when multiple
independent static structures improve across held-out windows, no other
perimeter edge or plane regresses, the loop remains consistent, and
per-point-time effects are controlled. Pedestrian alignment is a timing /
dynamic-scene diagnostic, not an extrinsic accuracy gate.

References:

- [Current scan2scan playbook](../../context/lidar2lidar_scan2scan_playbook.md)
- [Sensor timing and clock-source context](../../context/timing_sync_context.md)
- [Per-topic timing measurements](../../context/timing_topic_table.md)
- [LiDAR-to-LiDAR design](../../docs/lidar2lidar_design.md)
- [Validated conclusions](../../context/knowledge_base/validated_conclusions.md)
- [Open3D ICP registration](https://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html)
- [Open3D multiway registration](https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html)
- Segal et al., [Generalized-ICP](https://doi.org/10.15607/RSS.2009.V.021)
- Chetverikov et al., [The Trimmed Iterative Closest Point Algorithm](https://doi.org/10.1109/ICPR.2002.1047997)
- Yang et al., [TEASER: Fast and Certifiable Point Cloud Registration](https://arxiv.org/abs/2001.07715)
- Das et al., [Observability-Aware Online Multi-Lidar Extrinsic Calibration](https://arxiv.org/abs/2212.09579)
- [Multi-LiCa](https://arxiv.org/abs/2501.11088)
- [OverlapNet](https://arxiv.org/abs/2105.11344)
- [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)

These references describe methods and reported research results, not local
benchmark outcomes. Method families may target different inputs and tasks;
verify runnable implementations, provenance, and data compatibility before
claiming a fair comparison.

## Iteration ladder

### Round N — Frozen baseline

Pin the prepared dataset/input hashes, selected record windows, TF/prior
provenance, synchronization rules, configuration, executable/library versions,
and evaluation artifacts. Include at least:

- high-overlap feature-rich static data;
- wall-dominant/degenerate data;
- partial-overlap and/or dynamic-contamination data.

Record every edge/window outcome, including no-result and rejection reasons.
Separate scenes used for solving from predeclared holdout windows.

### Round N+1 — Overlap-aware classical correspondences

**Hypothesis:** restricting/down-weighting non-overlapping points while
requiring a minimum retained count and spatial spread will reduce false
correspondences and improve held-out transform stability in partial-overlap
cases.

**Minimal candidate:** keep the same initializer and ICP estimator; add one
overlap-aware mechanism (reciprocal correspondences or trimmed correspondences)
behind a flag. Do not simultaneously change estimator, voxel sizes, and gates.

**Decision metrics:** held-out per-window transform consistency; independent
extrinsic error when available; per-DoF spread; retained correspondence
fraction and spatial distribution; residual tails; failure/no-result rate;
same-crop wall/corner ghosting; runtime.

**Abort condition:** candidate wins only by collapsing to a tiny or spatially
concentrated subset, worsens rich-overlap cases, increases solution-family
flips, or improves training residual but not holdout/physical evidence.

### Round N+2 — Semantic dynamic filtering/weighting

**Hypothesis:** suppressing correctly identified dynamic returns improves
holdout stability on dynamic scenes while preserving accuracy on static scenes.

**Minimal candidate:** use one existing, reproducible semantic source if
available; first test only conservative dynamic-class exclusion or weighting.
Do not train a new segmentation model as part of the first experiment.

**Decision metrics:** same as N+1, plus per-class point retention, label
coverage/quality, and static-scene regression.

**Abort condition:** labels are unavailable/incompatible, useful geometry is
removed, static scenes regress, or gains disappear on holdout. Record label
source/model/version and preprocessing identity.

### Round N+3 — Stronger initialization / learned partial registration

Only if N+1 demonstrates that initialization/outlier rate remains the dominant
failure mode: boundedly test TEASER++ or a learned overlap/correspondence model
as an initializer, then hand off to the existing local refinement and
evaluation. Keep the baseline runnable and compare the added install/runtime
and failure cost.

### Round N+4 — Topology-level refinement

After individual edges pass their own sufficiency gates, compare the current
graph solve with a topology-aware weighted pose graph using uncertainty-aware
edge weights. Require per-edge accuracy/repeatability, cycle consistency,
holdout edges/windows, and geometry checks. Do not let a loop rescue a weak
primary edge.

## Acceptance and release rule

Use `ACCEPT`, `REJECT`, `INCONCLUSIVE`, or `NEEDS_MORE_DATA`. Release requires:

1. adequate time/sensor/overlap data and scene geometry for each claimed DoF;
2. stable per-edge estimates across independent windows and reasonable
   initialization perturbations;
3. holdout performance and, where available, independently measured
   extrinsic error;
4. topology/cycle consistency when a real loop exists;
5. full-scene plus overlap-region visual review with residual/ghosting evidence;
6. explicit accounting of all failed, rejected, and no-result cases.

Fitness, inlier RMSE, overlap ratio, information-matrix condition, cycle error,
or merged-cloud appearance alone is insufficient. Without independent truth,
describe evidence as repeatability and geometric consistency, not proven
physical accuracy.
