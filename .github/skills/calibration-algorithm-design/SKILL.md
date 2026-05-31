---
name: calibration-algorithm-design
description: Design guide for calibration, registration, estimation, and other iterative algorithms that need a baseline-first plan. Use this when asked to design or redesign a sensor calibration or optimization pipeline, especially when the request mentions data filtering, baseline, metrics, visualization, papers, open-source references, or incremental algorithm improvement.
---

# Calibration Algorithm Design

Use this skill to produce designs that are reviewable, iterative, and evidence-driven.

Pair it with:

- `calibration-capture-design`
- `calibration-algorithm-validation`
- `calibration-algorithm-iteration`
- `calibration-benchmark-and-ablation`
- `calibration-failure-analysis`
- `calibration-sota-upgrade`
- `calibration-release-gating`

## When to use

Use this skill when the task involves:

- sensor calibration, registration, or state-estimation algorithm design
- restructuring an existing pipeline into clearer stages
- comparing a conservative baseline against a stronger candidate
- defining metrics, acceptance gates, diagnostics, or visualization
- grounding a proposal in industry practice, open-source tools, and papers

If the request is ambiguous, state assumptions explicitly or ask for the missing deployment constraints before finalizing the design.

## Core policy

Always structure the design in three stages:

1. **Data screening / extraction**
   - Define raw inputs, synchronization, normalization, sample acceptance rules, skip reasons, and sufficiency thresholds.
   - Count what was accepted, what was rejected, and why.
   - Emit stable extraction artifacts so downstream runs remain comparable.
2. **Algorithm iteration**
   - Start with the simplest defensible baseline before proposing stronger variants.
   - Separate initialization, core solve, robustification, refinement, and optional global optimization.
   - Keep failure modes observable; do not hide data problems inside one giant optimizer.
3. **Metric validation and visualization**
   - Define numeric metrics, holdout or repeatability checks, acceptance gates, and visual review surfaces.
   - Require visualization when geometry or alignment matters.
   - Keep artifact contracts stable so baseline and improved methods can be compared directly.

## Baseline-first ladder

When asked to design a new algorithm, present the plan in this order:

1. **Problem statement**
   - Name the sensors, coordinate frames, target outputs, runtime constraints, and likely failure modes.
2. **Baseline**
   - Choose the most conservative method that is already trusted in industry or in this repository.
   - For board-friendly workflows, prefer target-based baselines unless strong evidence says otherwise.
   - For targetless scene-registration workflows such as `lidar2lidar`, prefer the repository's trusted scene-based baseline instead of forcing a target-based formulation.
   - Prefer staged or pairwise solves before joint or global optimization.
   - Define exactly what the minimal end-to-end baseline run produces.
3. **Improvement path**
   - Propose one to three incremental upgrades with explicit hypotheses.
   - For each upgrade, state what new signal or assumption it uses, which metric should improve, which failure mode it may introduce, and how it remains comparable to the baseline.
4. **Decision rule**
   - Do not replace the baseline just because a new method converges or matches a paper result.
   - Upgrade only if the new path improves accuracy and debuggability or operational robustness.
   - Keep the baseline alive until the new path wins on repeated evidence.

## Required design output

When producing an algorithm design, include these sections:

1. Goal and scope
2. Inputs and assumptions
3. Three-stage pipeline
4. Baseline algorithm
5. Improvement roadmap
6. Metrics and acceptance gates
7. Visualization and diagnostics
8. References and prior art
9. Minimal validation plan

## Repo-specific guidance

Align designs to this repository's architecture:

- Preserve the **extraction -> algorithm -> evaluation** split.
- Keep stable top-level outputs such as `metrics.yaml`, `diagnostics/`, `calibrated_tf.yaml`, and module-specific standardized data artifacts.
- For `lidar2lidar`, keep `scan2scan` as the practical baseline and position `scan2map` as additive refinement.
- Prefer decisions based on coarse and fine metrics, conditioning or information diagnostics, repeatability, explicit skip reasons, and human-visible overlays rather than a single fitness number.

## Reference expectations

Before proposing a stronger method, check at least these categories:

- **Industry practice**: what production systems actually release with
- **Open-source tools**: for example Kalibr, Autoware calibration tooling, and mature target-based pipelines
- **Canonical papers and texts**: for example Zhang, Hartley and Zisserman, Tsai-Lenz, Park-Martin, ICP, GICP, and continuous-time calibration references
- **In-repo precedent**: existing modules, artifact contracts, review order, and diagnostics

Use references to justify:

- why the baseline is credible
- what assumption changes the improved method makes
- which metrics should reveal success or regression
- which visualization is needed before trusting the result

## Named references to prefer

Prefer concrete prior art over vague "SOTA" language. Good anchors include:

- **Kalibr** for staged sensor calibration workflows and report-like outputs
- **OpenCV calibration docs** for conservative camera baseline design and coverage guidance
- **Ceres Solver docs** for robust loss design, parameterization, and optimization hygiene
- **Open3D / PCL registration docs** for ICP, GICP, NDT, evaluation, and visualization patterns
- **TEASER++** for outlier-robust global initialization ideas when correspondences are noisy
- **Autoware calibration tooling/docs** for operational constraints and release-facing expectations
- **Zhang 2000**, **Tsai-Lenz 1989**, **Park-Martin 1994**, **Besl-McKay 1992**, **Segal et al. 2009**, **Magnusson 2009**, and **Hartley-Zisserman** for canonical baseline justification

## Anti-patterns

Avoid these design mistakes:

- skipping data-quality screening and trusting solver convergence
- proposing only a final sophisticated method with no baseline
- mixing acceptance logic into solver internals
- changing metrics or artifacts so baseline and candidate runs cannot be compared
- using a paper result as release evidence without holdout or visualization

## Output style

When using this skill:

- lead with the recommended baseline
- then present the three-stage pipeline
- then give an incremental upgrade ladder
- cite concrete references or projects, not vague "SOTA" claims
- explicitly call out what to measure and what to visualize

## Example framing

For a calibration task, answer in a structure like:

```text
Baseline:
- Conservative method ...
- Why it is the right first release candidate ...

Stage 1: data screening
- accepted and rejected rules
- sufficiency thresholds
- extraction artifacts

Stage 2: algorithm
- initialization
- robust solve
- optional refinement

Stage 3: validation
- metrics
- holdout or repeatability
- visual diagnostics

Upgrade path:
- v1 baseline
- v2 robustification
- v3 stronger global refinement

References:
- paper
- open-source project
- repo precedent
```

## In this repository, start from these precedents

- `.github/copilot-instructions.md`
- `docs/calibration_methodology.md`
- `docs/calibration_review_guide.md`
