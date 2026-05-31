---
name: calibration-algorithm-iteration
description: Iteration playbook for improving calibration and registration pipelines without losing baseline comparability. Use this when asked to optimize a calibration algorithm, propose next iterations, run ablations, or improve quality step by step rather than making a risky one-shot rewrite.
---

# Calibration Algorithm Iteration

Use this skill to improve calibration algorithms in controlled, evidence-backed steps.

## When to use

Use this skill when the task involves:

- improving an existing calibration baseline
- planning iterative upgrades instead of a rewrite
- choosing among candidate enhancements
- refining one concrete hypothesis at a time

Use `calibration-benchmark-and-ablation` for multi-method comparisons and `calibration-sota-upgrade` for paper-driven method scouting.

## Core policy

Iteration must be:

1. **Baseline-preserving**
   - keep the trusted baseline runnable and comparable
2. **Single-hypothesis**
   - change one major idea at a time
3. **Measurement-locked**
   - keep metrics and artifacts stable while iterating
4. **Failure-aware**
   - record when the change helps, hurts, or only shifts failure modes

## The iteration loop

For each round, always produce:

1. **Hypothesis**
   - what is expected to improve and why
2. **Minimal change**
   - the smallest implementation that exercises the idea
3. **Comparison**
   - baseline vs candidate on the same inputs and outputs
4. **Failure review**
   - what new breakage or brittleness appeared
5. **Decision**
   - keep, revert, or gate behind an option

## Preferred upgrade order

Prefer this ladder unless strong evidence says otherwise:

1. Better data screening and sample weighting
2. Better initialization or correspondence pruning
3. Better robust loss and coarse-to-fine scheduling
4. Better local refinement or bundle adjustment style solve
5. Better global optimization or joint optimization
6. More ambitious modeling assumptions such as continuous-time or learned priors

This order usually yields more reliable gains than jumping straight to a sophisticated optimizer.

## Common upgrade levers

Good iteration ideas include:

- stricter acceptance gates for weak samples
- observability-aware sample selection
- robust loss tuning
- multi-stage initialization
- coarse-to-fine registration
- holdout-based early rejection
- better uncertainty or conditioning diagnostics
- explicit fallback from strong candidate to trusted baseline

## Required output

When proposing or reviewing an iteration, include:

1. Current baseline
2. Proposed change
3. Expected upside
4. Expected new failure modes
5. Comparison plan
6. Rollback or fallback rule

## Reference canon

Use concrete references such as:

- **Ceres Solver docs** for iterative nonlinear optimization hygiene
- **Kalibr** for staged calibration rather than one-shot magic
- **Open3D / PCL** for coarse-to-fine registration pipelines
- **TEASER++** for strong initialization under heavy outliers
- **GICP** and **NDT** literature for alternative refinement behavior
- continuous-time calibration papers when motion distortion or asynchronous sensing is the actual bottleneck

## Anti-patterns

Avoid these iteration mistakes:

- replacing the baseline before the candidate wins repeatedly
- changing the optimizer and the metric at the same time
- merging several experimental ideas into one patch
- accepting a candidate because one demo looks better
- removing diagnostics to make the new method look cleaner

## Output style

Present iterations as a ladder:

- **Round N baseline**
- **Round N+1 candidate**
- **Expected gain**
- **Decision metric**
- **Abort condition**

Make the next step obvious and falsifiable.
