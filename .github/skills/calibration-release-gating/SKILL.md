---
name: calibration-release-gating
description: Release-gating guide for deciding whether a calibration method is production-ready, review-only, or still experimental. Use this when asked whether a calibration algorithm should replace the baseline, be exposed to users, or stay as a research path.
---

# Calibration Release Gating

Use this skill to decide whether a calibration method is safe to trust operationally.

## Core policy

Production promotion requires more than better average accuracy.

A candidate should not replace the baseline unless it is:

1. **accurate enough**
2. **robust enough**
3. **diagnosable enough**
4. **operationally simple enough**
5. **clear about when it should refuse or fall back**

## Gating ladder

Always make the decision in this order:

1. **Data sufficiency gate**
   - can the method detect when the input is too weak?
2. **Correctness gate**
   - does it beat or match the baseline on trusted metrics?
3. **Robustness gate**
   - does it survive seed changes, holdout data, and harder sequences?
4. **Observability and uncertainty gate**
   - does the run show one stable solution family and acceptable uncertainty, or is it underconstrained?
5. **Diagnostics gate**
   - can a reviewer tell why it passed or failed?
6. **Operations gate**
   - is runtime, dependency burden, and operator workflow acceptable?

## Verdict classes

Classify the method as:

- **baseline-ready**
- **candidate behind a flag**
- **review-only**
- **experimental only**
- **reject**

## Required evidence

Require evidence such as:

- side-by-side benchmark against the baseline
- repeatability and holdout checks
- observability or uncertainty evidence when available
- failure-case accounting
- stable artifacts and diagnostics
- clear fallback behavior

## Promotion criteria

Promote only when the candidate:

- wins on representative data, not just easy cases
- shows one stable solution family rather than several competing explanations
- keeps failure modes observable
- does not require hidden manual rescue
- preserves or improves reviewability
- remains comparable to the previous baseline

## LiDAR-to-LiDAR release rule

For `lidar2lidar`, the release question depends on topology:

- **loop-capable case**
  - promote only when pairwise edges are healthy and loop consistency agrees with them
- **no-loop case**
  - promote only when multi-window consensus is stable and the run shows one dominant solution family

If the method only looks good after repeated reseeding, or if different seeds land in materially different extrinsics, keep it at `review-only` or below.

## Reference canon

Use production-minded references:

- mature open-source workflows like **Kalibr**
- operational toolchains such as **Autoware** calibration-related practices
- conservative geometry baselines from **OpenCV**
- optimization discipline from **Ceres Solver**

These references matter because they emphasize trust, debugging, and repeatability rather than novelty alone.

## Anti-patterns

Avoid these release mistakes:

- promoting a method because it is newer
- accepting a method with no fallback or refusal mode
- shipping a result that cannot explain bad cases
- hiding evaluation detail behind one summary score

## Output style

Lead with the release verdict, then list the evidence that justified it and the evidence still missing.
