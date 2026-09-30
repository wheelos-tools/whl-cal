---
name: calibration-sota-upgrade
description: Guide for turning papers, blogs, and SOTA-style ideas into safe calibration upgrades. Use this when asked to apply a paper, assess a novel method, mine open-source projects for ideas, or translate research into a practical roadmap.
---

# Calibration SOTA Upgrade

Use this skill to turn research ideas into practical, reviewable upgrades instead of speculative rewrites.

## Core policy

Treat every paper or blog idea as a candidate, not a verdict.

Before adopting a method, ask:

1. What assumption lets this method win?
2. Does our data actually satisfy that assumption?
3. What trusted baseline should it be compared against?
4. What minimal spike would test it without destabilizing the pipeline?
5. What evidence is required before it can influence release decisions?

## Source ranking

Prefer sources in this order:

1. mature open-source implementation with docs and examples
2. paper plus maintained code
3. strong engineering blog or technical write-up with reproducible detail
4. paper without code
5. hype without reproducibility

## How to use references

Map each reference to one of these roles:

- baseline justification
- initialization idea
- robustification idea
- validation or diagnostic idea
- benchmark or ablation idea
- release-gating caution

Do not cite a paper unless it changes the concrete plan.

## Practical review checklist

For each candidate method, explicitly score:

- reproducibility
- data-assumption fit
- implementation complexity
- latency or runtime cost
- diagnostic visibility
- fallback path to baseline

## Reference canon

Strong recurring anchors include:

- **Zhang 2000** for conservative calibration baselines
- **Tsai-Lenz 1989** and **Park-Martin 1994** for hand-eye grounding
- **Hartley and Zisserman** for geometric sanity
- **ICP**, **GICP**, and **NDT** literature for registration families
- **TEASER++** for robust global initialization under heavy outliers
- **Kalibr** for practical multi-sensor calibration workflows
- **Ceres Solver docs** for optimization craftsmanship
- **OpenCV**, **Open3D**, and **PCL** docs for practical implementation patterns

## Upgrade pattern

When proposing a paper-driven improvement, structure it as:

1. baseline to beat
2. paper or project idea
3. assumption it relies on
4. minimal implementation spike
5. expected gain
6. likely new failure mode
7. acceptance evidence required

## LiDAR-to-LiDAR upgrade ladder

For `lidar2lidar`, prefer these judgments:

- **direct `scan2scan` with strong overlap**: baseline-ready
- **pairwise `scan2scan` plus loop closure pose graph**: mature enough to challenge the baseline when loops really exist
- **multi-window no-loop consensus**: worth adding behind a flag
- **NDT or GICP replacements**: worth a bounded spike, not an automatic upgrade
- **TEASER++ or feature-based global registration**: initialization ideas, not release criteria
- **continuous-time or joint spatiotemporal optimization**: not worth integrating first unless real async timing or motion distortion is proven to dominate

## Anti-patterns

Avoid these mistakes:

- saying "SOTA" without defining the benchmark
- copying a paper objective without its data assumptions
- changing the benchmark so the candidate looks better
- shipping a candidate that cannot explain its own failures
- removing the baseline because the paper is more sophisticated

## Output style

Lead with one of:

- not worth integrating
- worth a bounded spike
- worth adding behind a flag
- mature enough to challenge the baseline

Then justify with evidence quality and operational fit.
