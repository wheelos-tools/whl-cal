---
name: calibration-benchmark-and-ablation
description: Benchmark and ablation playbook for comparing calibration methods fairly. Use this when asked to compare baselines, evaluate a new candidate, design experiments, or decide whether an algorithm improvement is real.
---

# Calibration Benchmark and Ablation

Use this skill to compare calibration methods fairly and decide what actually improved.

## Core policy

Benchmarking must answer:

- does the candidate beat the baseline?
- where does it beat the baseline?
- where does it lose?
- is the gain robust enough to matter operationally?

Do not benchmark only on a friendly sequence or a single scalar metric.

## Benchmark design

Always define:

1. **Method set**
   - trusted baseline, candidate, and optionally a stronger oracle-like upper bound
2. **Dataset matrix**
   - easy, nominal, and failure-prone cases
   - if possible, both in-repo data and a public or standardized reference set
3. **Perturbation plan**
   - seed offsets, weak-data cases, or synthetic perturbations
4. **Metric set**
   - accuracy, robustness, repeatability, runtime, diagnostic completeness
5. **Decision rule**
   - what counts as a real win

## Required experiment types

Prefer to include:

- nominal comparison
- holdout comparison
- perturbation recovery
- seed sensitivity
- data thinning or sample-reduction stress
- runtime or operator-cost comparison

## Ablation policy

When a candidate has multiple changes, isolate them:

- screening change alone
- initialization change alone
- objective change alone
- refinement change alone

This prevents accidental credit assignment to the wrong idea.

## Required outputs

When writing a benchmark plan or result summary, include:

1. Methods compared
2. Data matrix
3. Metrics table
4. Win-loss summary
5. Notable failure cases
6. Recommendation

## Good comparison habits

Prefer:

- the same artifact contract for all methods
- fixed acceptance metrics across methods
- explicit "no result" accounting when a method fails
- per-case tables, not only mean scores
- visual review on representative wins and losses

## LiDAR-to-LiDAR benchmark matrix

For `lidar2lidar`, include at least these case families when possible:

1. high-overlap direct pairwise scenes
2. loop-capable rigs with healthy perimeter edges
3. no-loop rigs with several static windows
4. wall-dominant or low-corner scenes
5. seed-perturbed reruns
6. static publish-skew cases vs truly motion-rich timing cases

Report whether the candidate:

- improves pairwise quality
- improves loop consistency when loops exist
- reduces or increases solution-family instability
- keeps the same diagnostics and artifact contract as the baseline

## Reference canon

Good anchors include:

- **Open3D registration evaluation examples**
- **PCL registration examples**
- **Kalibr** for report-style quality review
- public datasets and community benchmarks where relevant, but only when the sensor setup is meaningfully comparable
- engineering practice from production autonomy stacks where robustness beats flashy average metrics

## Anti-patterns

Avoid these benchmarking mistakes:

- averaging away catastrophic failures
- evaluating on only one recording
- changing metrics between methods
- silently dropping failed cases
- claiming SOTA from internal-only, unreproducible wins

## Output style

Lead with the recommendation:

- keep baseline
- promote candidate behind a flag
- replace baseline
- inconclusive

Then justify it with win-loss structure, not just a mean metric.
