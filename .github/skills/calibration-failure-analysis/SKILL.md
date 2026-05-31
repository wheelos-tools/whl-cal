---
name: calibration-failure-analysis
description: Failure analysis guide for unstable, inconsistent, or misleading calibration results. Use this when a calibration run looks wrong, diverges, flips between solutions, or passes metrics without being trustworthy.
---

# Calibration Failure Analysis

Use this skill to localize failure to the right stage and propose the smallest credible fix.

## Failure taxonomy

Classify the problem before proposing changes:

1. **Extraction failure**
   - bad sync, poor detections, weak overlap, low coverage, insufficient motion diversity
2. **Observability failure**
   - geometry cannot constrain the unknowns well enough
3. **Initialization failure**
   - the optimizer starts outside the basin of attraction
4. **Optimization failure**
   - objective mismatch, poor weighting, outliers, degeneracy, bad parameterization
5. **Evaluation failure**
   - metrics hide the error or overfit the training samples
6. **Deployment failure**
   - the result only works on one sequence or one operating regime

## Diagnostic workflow

Always debug in this order:

1. Check accepted ratio and skip reasons
2. Inspect a handful of accepted and rejected samples
3. Review residuals or alignment error per sample
4. Check seed sensitivity and rerun stability
5. Examine observability or conditioning indicators
6. Compare baseline vs candidate on the same failure cases

## What to inspect

Require concrete evidence such as:

- accepted vs rejected sample counts
- per-sample residual tables
- transform dispersion across runs or pairs
- conditioning or information summaries
- before and after overlays
- failure-case gallery with comments on likely cause

## Diagnosis patterns

Common interpretations:

- **low cost but bad geometry** usually means the objective is incomplete
- **high variance across seeds** usually means poor initialization or weak observability
- **good training metrics, bad holdout** usually means overfitting or selection bias
- **few accepted samples** usually means the extraction stage is the real bottleneck
- **candidate only wins on easy scenes** usually means the claimed upgrade is fragile

## Fix order

Prefer the smallest fix that matches the diagnosed cause:

1. Improve data screening or sample diversity
2. Improve initialization
3. Improve robustification and weighting
4. Improve objective completeness
5. Only then consider a heavier optimizer or stronger model

## Reference canon

Useful anchors include:

- **OpenCV calibration docs** for common pattern-quality and viewpoint issues
- **Ceres Solver docs** for parameterization and residual debugging guidance
- **Kalibr** for staged debugging of calibration inputs and outputs
- **Open3D / PCL** for correspondence and registration failure visualization
- **TEASER++** when outlier-ridden initialization is the actual issue

## Anti-patterns

Avoid these failure-analysis mistakes:

- proposing a rewrite before locating the failing stage
- using only aggregate averages
- diagnosing optimization before checking data sufficiency
- blaming noise when the problem is unobservable geometry
- hiding instability behind a best-of-N rerun

## Output style

Lead with:

1. probable failing stage
2. evidence for that diagnosis
3. smallest corrective action
4. how to prove the fix actually addressed the root cause
