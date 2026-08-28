---
name: calibration-algorithm-validation
description: Validation and visualization guide for sensor calibration and registration pipelines. Use this when asked to verify a calibration result, define metrics, add diagnostics, judge run quality, or decide whether an algorithm is trustworthy beyond solver convergence.
---

# Calibration Algorithm Validation

Use this skill to validate calibration results with evidence, not optimism.

## When to use

Use this skill when the task involves:

- deciding whether a calibration result is acceptable
- defining metrics, acceptance gates, or release criteria
- adding diagnostics, plots, overlays, or failure reports
- comparing baseline and candidate methods on the same artifact contract
- reviewing a run that "converged" but may still be wrong

## Core policy

Never treat these as sufficient on their own:

- solver success
- low final cost
- high registration fitness
- a single good-looking visualization

Validation must combine:

1. **Data quality**
   - accepted sample count, accepted ratio, skip reasons, coverage, synchronization quality, motion or pose diversity
2. **Numerical fit**
   - residual summaries, inlier or overlap stats, transform stability, conditioning or information indicators
3. **Generalization**
   - holdout checks, repeat runs, perturbation recovery, seed sensitivity, sequence-to-sequence consistency
4. **Observability and uncertainty**
   - transform spread, distinct solution families, observability status, or uncertainty summary when available
5. **Human review**
   - overlays, residual plots, trajectory or cloud visualizations, per-sample failure galleries

## Validation ladder

Always validate in this order:

1. **Input sufficiency**
   - Was the data set rich enough to support the claimed estimate?
2. **Per-sample quality**
   - Which samples drove the result, and which were rejected?
3. **Aggregate metrics**
   - Did the solver improve the right metrics, not just its own loss?
4. **Holdout and repeatability**
   - Does the result persist on unseen or withheld samples and across reruns?
5. **Observability and uncertainty**
   - Is the solution well-constrained, or does the run only look stable because the data is weak?
6. **Visual confirmation**
   - Do overlays and residual maps agree with the numeric story?
7. **Operational acceptance**
   - Is the result robust enough for release or only good enough for research?

## Required validation output

When validating an algorithm or run, include:

1. Data-quality summary
2. Primary metrics
3. Holdout or repeatability checks
4. Observability or uncertainty summary
5. Visualization plan
6. Acceptance rule
7. Reasons a run may still be rejected

## Recommended metric families

Choose the metric family that matches the sensing problem:

- **Camera / reprojection**
  - reprojection RMSE and percentile tails
  - image coverage and corner distribution
  - per-view reprojection breakdown
- **LiDAR / registration**
  - overlap-aware residuals
  - coarse and fine registration metrics
  - transform consistency across pairs or submaps
- **Hand-eye / trajectory**
  - rotation and translation consistency
  - observability indicators
  - solution-family stability or transform spread
  - trajectory alignment on held-out motion segments
- **Cross-modal**
  - geometric alignment overlays
  - temporal consistency
  - sensitivity to sparse or weak correspondences

## Visualization checklist

Require visual surfaces that make failure obvious:

- before and after overlays
- residual histograms and percentile summaries
- per-sample review panels
- coverage heatmaps or viewpoint distribution plots
- trajectory or point-cloud alignment views
- failure-case index with skip reasons and examples

If geometry matters, the result is not reviewable without visualization.

## Reference canon

Prefer these references when defining validation:

- **Kalibr** for report-driven calibration review
- **OpenCV calibration docs** for coverage, reprojection, and pattern-quality sanity checks
- **Ceres Solver docs** for cost interpretation, robust losses, and covariance-related caution
- **Open3D registration evaluation docs** for side-by-side registration scoring and visualization
- **PCL ICP / GICP / NDT docs** for practical registration metrics
- **Hartley and Zisserman** for geometric error interpretation

## Anti-patterns

Avoid these validation mistakes:

- using the training objective as the only metric
- changing metrics between baseline and candidate runs
- showing only a cherry-picked visualization
- averaging away catastrophic failure cases
- accepting a result without repeatability or holdout evidence
- hiding rejected samples instead of reporting them

## Repo-specific guidance

Align validation to this repository's stable review surfaces:

- preserve `metrics.yaml` and `diagnostics/`
- keep standardized data-quality outputs and acceptance summaries stable
- include explicit skip reasons and accepted ratios
- require observability or uncertainty evidence when the module provides it
- prefer coarse and fine metrics plus human-visible overlays over one scalar score

## LiDAR-to-LiDAR validation rule

For `lidar2lidar`, validate in two different ways depending on topology:

- **loop available**
  - require healthy pairwise edges, cycle consistency, repeatability, and visual geometry
- **no loop**
  - require multi-window repeatability, one stable solution family, and representative-transform agreement

In both cases, overlap, scene sufficiency, and visual geometry must stay visible. High fitness alone is never enough, and seed sensitivity must be reported whenever the method depends on iterative refinement.

## Output style

Lead with the verdict:

- accepted
- review-only
- rejected
- inconclusive

Then justify the verdict with data quality, metrics, holdout behavior, and visuals.
