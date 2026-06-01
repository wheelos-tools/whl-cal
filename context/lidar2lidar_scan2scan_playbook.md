---
audience: dev
stability: stable
last_tested: 2026-05-31
---

# LiDAR-to-LiDAR scan2scan playbook

## Purpose

This file is the current production-minded playbook for **targetless
LiDAR-to-LiDAR** calibration in this repository.

It is intentionally narrower than a generic "SOTA survey":

- prefer methods that can be reviewed, benchmarked, and gated
- keep `scan2scan` as the trusted baseline when direct overlap is strong
- add loop closure when the rig or capture graph truly contains a healthy loop
- when there is no loop, prefer **multi-window consensus** over repeated
  seed-only full-graph optimization

## Baseline decision ladder

Use this decision ladder before proposing a heavier method.

### 1. Direct high-coverage pairwise overlap exists

**Recommended baseline:** direct `scan2scan`

Use direct `scan2scan` when:

- the two LiDARs share enough real geometry
- the shared geometry is not only one large plane
- multiple static or near-static windows are available
- the seed is already roughly plausible from TF or a prior run

This is the first choice because mature tooling in Open3D and PCL still treats
ICP-family registration as the conservative local-registration baseline, not as
an obsolete fallback.

### 2. A real loop exists in the topic / rig graph

**Recommended upgrade:** add loop closure on top of pairwise `scan2scan`

Use loop closure only when:

- the topology actually contains a loop
- the loop edges themselves are healthy enough to register
- the loop is used as an additional consistency source, not as a way to hide a
  weak edge

Open3D's multiway registration tutorial is the practical anchor here: keep
strong neighboring edges as higher-trust constraints and mark loop edges as more
uncertain, then optimize a pose graph rather than trusting one edge at a time.

### 3. No loop exists, but several good windows exist

**Recommended upgrade:** no-loop multi-window consensus

When the graph has no loop, do **not** pretend that repeated seed-only reruns
create the missing constraint. Instead:

1. solve multiple windows for the same direct edge
2. compute repeatability and seed sensitivity
3. cluster or rank solution families
4. keep the representative transform (medoid / consensus winner)
5. run at most a small local refinement around that representative

This is the correct fallback for two-edge or star-topology rigs.

### 4. Overlap is weak or geometry is wall-dominant

**Verdict:** keep the run as review-only or diagnostic-only

If the scene is dominated by one wall, repeated seed-only refinement usually
does not improve observability. It only chooses among several plausible local
optima.

In that situation:

- freeze weak DoFs when justified
- keep prior-safe corrections small
- improve scene selection before changing the optimizer

## Concrete repo guidance

### Extraction and screening

For `lidar2lidar`, extraction should answer **whether the bag can support the
claimed solve** before optimization begins.

Prefer this order:

1. resolve the time source
   - use acquisition or header time when available
   - do not rely on publish order
2. build candidate static windows
3. score each window for:
   - overlap
   - dynamic contamination
   - plane count and plane-normal diversity
   - corner or facade-intersection support
   - range balance across the pair
4. reject weak windows explicitly
5. keep window summaries in stable diagnostics

For static bags, publish-time skew can often be tolerated if the real sampling
time is consistent. For true temporal-bias estimation, static bags are not
enough; motion excitation is required.

### Algorithm stages

#### Stage A: initialization

Use the strongest low-risk initializer available:

1. TF or checked-in extrinsics
2. previous accepted or review-only candidate
3. optional global initializer behind a flag

Global registration should remain **initialization only**, not the release
verdict:

- Open3D global registration
- TEASER++

These are worthwhile bounded spikes when the seed is uncertain or the outlier
rate is high, but they should still hand off to local refinement and the
standard evaluation stack.

#### Stage B: pairwise local solve

For direct overlap, use local scan registration as the baseline:

- point-to-point ICP
- point-to-plane ICP
- GICP
- optional NDT behind a flag

The correct default is data-dependent, not ideology-dependent. On some
wall-dominant bags, point-to-point can be more stable than GICP or point-to-
plane even when the latter look numerically stronger on a single run.

#### Stage C: topology handling

When a loop exists:

1. solve pairwise edges first
2. compute information / uncertainty proxies per edge
3. add loop edges as explicit, possibly lower-trust constraints
4. solve a pose graph
5. judge both pairwise quality and cycle consistency

When no loop exists:

1. solve the direct edge in multiple windows
2. keep per-window transforms
3. reject unstable windows
4. choose a representative transform
5. optionally refine once around that representative

#### Stage D: fail-safe downgrade

Every `lidar2lidar` method should degrade safely:

1. `full_loop`
   - loop exists and is healthy
2. `pairwise_consensus`
   - no loop, but multi-window direct evidence is stable
3. `pairwise_safe`
   - only partial DoFs are trustworthy
4. `diagnostic_only`
   - the scene does not support a production update

## Validation ladder

Never accept a run because one scalar improved.

Validate in this order:

1. **input sufficiency**
   - valid windows
   - overlap
   - scene class
2. **repeatability**
   - window-to-window transform spread
   - rerun stability
3. **solution-family stability**
   - one family vs multiple basins
4. **topology consistency**
   - loop residuals when a loop exists
5. **visual geometry**
   - wall thickness
   - corner spread
   - slice sharpness
   - ghosting / double-edge separation
6. **prior plausibility**
   - drift to trusted prior stays explainable

For no-loop workflows, repeatability and solution-family stability carry more
weight than raw fitness.

## What current evidence says

The 2026-05-31 static dataset established a concrete repo-level lesson:

- `left_front -> right_front` and `right_back -> right_front` are not impossible
- direct `scan2scan` can recover reviewable candidates
- high fitness alone is misleading on wall-dominant scenes
- seed-only full-dataset reruns can oscillate between distinct local optima
- therefore, the next justified upgrade is:
  - better scene-level pairing
  - representative-transform selection
  - clearer time-source handling
- not more blind ICP retuning

## Open-source and literature anchors

Map references to roles, not hype:

| Reference | Role | Practical takeaway |
| --- | --- | --- |
| Open3D ICP docs | baseline justification | ICP remains the conservative local-registration baseline and needs a plausible initializer. |
| Open3D global registration docs | initialization idea | Global registration is useful as initialization, not as the final acceptance criterion. |
| Open3D multiway registration docs | loop-closure design | Distinguish high-trust neighboring edges from uncertain loop edges and optimize them jointly. |
| PCL GICP / NDT docs | robustification idea | GICP and NDT are candidate local refiners, but should be compared against the same metric contract. |
| TEASER++ | robust initialization idea | Worth a bounded spike when correspondences are noisy or the seed is weak; still needs local refinement and normal repo gating. |
| Kalibr | release-gating caution | Strong calibration workflows are staged, report-driven, and explicit about temporal assumptions and diagnostics. |

## Candidate method status

Use these labels when discussing upgrades:

| Candidate | Status | Why |
| --- | --- | --- |
| direct `scan2scan` with stable metrics | baseline-ready | Mature, debuggable, and already aligned with repo outputs. |
| pairwise `scan2scan` + loop closure pose graph | mature enough to challenge the baseline | Strong when the graph truly contains healthy loops. |
| multi-window no-loop consensus | worth adding behind a flag | High value for star or chain topologies with several good static windows. |
| NDT local refinement | worth a bounded spike | Useful candidate on large clouds, but still data- and initialization-sensitive. |
| TEASER++ initialization | worth a bounded spike | Strong for outlier-heavy initialization, but not a replacement for evaluation and local refinement. |
| continuous-time or joint spatiotemporal optimization | not worth integrating first | Too much complexity unless motion distortion or real asynchronous sensing is proven to be the main bottleneck. |

## Anti-patterns

Avoid these failure-prone moves:

- replacing `scan2scan` with `scan2map` just because `scan2map` is newer
- claiming improvement from fitness alone
- using loop closure to rescue an edge that is individually untrustworthy
- repeating seed-only full-dataset solves until one run looks good
- hiding multi-basin behavior behind a single best run
- estimating temporal bias from only static windows

## Operator-facing rule of thumb

If overlap is high, geometry is rich, and the pair is directly observable:

- start with direct `scan2scan`

If a real loop exists and the loop edges are healthy:

- add loop closure

If no loop exists but multiple good static windows exist:

- use multi-window consensus, not repeated full-graph reseeding

If the scene is wall-dominant, low-overlap, or solution families flip:

- keep the result review-only and improve the data / pairing policy first
