---
name: algorithm-research
description: Evidence-first research workflow for turning algorithm questions into reproducible engineering decisions. Use when comparing algorithms, designing controlled experiments, iterating on a baseline, or planning validation.
---

# Algorithm Research

Act as an **Algorithm Research Lead**. Turn the engineering problem into a
reproducible, evidence-backed decision:

> First principles → Evidence → Experiment → Iteration

Prefer simple, maintainable solutions over unnecessary complexity. A paper
result, solver convergence, or one successful run is not evidence that a method
is accurate or ready to replace the baseline.

## Research loop

Repeat these stages whenever evidence changes the problem understanding:

1. **Decompose** — define the objective, inputs and outputs, system boundary,
   constraints, metrics, baseline, and acceptance criteria. Split the system
   into replaceable modules.
2. **Research** — inspect relevant papers, industry practice, open-source
   implementations, benchmarks, datasets, and in-repository precedent. For
   each candidate, record its target problem, assumptions, complexity,
   implementation and deployment constraints, and known limitations.
3. **Hypothesize** — state a falsifiable prediction:
   “If X changes to Y, metric Z should improve while constraints C remain
   satisfied.” Choose the smallest change that tests it and keep a strong
   baseline.
4. **Experiment** — run controlled A/B tests using the same data, evaluation,
   hardware, and important configuration. Change one major factor at a time.
   Measure algorithm effectiveness and engineering cost; record failures.
5. **Critique** — try to disprove the result. Check fairness, baseline validity,
   confounding changes, dataset representativeness, alternative explanations,
   failure cases, simpler solutions, and missing evidence. Use PROPOSAL versus
   CRITIC debate when useful, ending with evidence or a concrete experiment.
6. **Review** — classify the result as `ACCEPT`, `REJECT`, `INCONCLUSIVE`, or
   `NEEDS_MORE_DATA`. State what is known, what evidence supports it, and what
   remains uncertain. Separate paper claims, experiment results, and engineering
   conclusions.
7. **Update** — feed conclusions into problem understanding, architecture,
   algorithm choice, known risks, open questions, and the research roadmap.
   Give every next step a measurable objective and acceptance criterion.

## Calibration workflow

Preserve the repository's **data screening → algorithm → evaluation** split.
For sensor calibration, use this order:

### 1. Data inspection and extraction

- Inventory candidate captures before selecting a benchmark set. Record source,
  duration, sensor and firmware identity when known, record splits, topics,
  sample rates, gaps, and available motion / reference signals.
- Verify actual message schemas, timestamp semantics, units, frames, axes,
  calibration state, and sensor provenance. Do not infer these from filenames
  or topic names.
- Explicitly determine whether an IMU topic contains raw gyro / accelerometer
  measurements, corrected measurements, or INS-fused outputs. Determine
  whether any INS pose used for validation is independent of that IMU solution.
- Define sample acceptance and rejection rules before optimization. Count
  accepted and rejected samples with reasons; emit a stable extraction manifest
  and reusable standardized dataset.
- Treat a figure-eight route as a motion-design goal, not proof of calibration
  accuracy. Verify both turn directions, angular and linear excitation,
  acceleration / braking, flat-road sections, synchronization, and gaps.

### 2. Baseline and algorithm iteration

- Establish the exact runnable baseline and pin source revision, configuration,
  executable / container digest, compiler and numerical-library versions
  (especially Eigen/Ceres/PCL), input hashes, and commands. For native solvers,
  replay a small or frozen batch trace before spending time on a full capture.
- Verify that every named comparison method is independently runnable and
  document its provenance. Related or derivative implementations are not
  independent algorithm families merely because they have different names.
- Convert each method's input to the same canonical data contract where
  possible. Keep extraction artifacts and metrics fixed across methods.
- Run the unmodified baseline first. Then change one hypothesis at a time:
  screening, initialization, objective / robustification, refinement, or model.
- Log failures and invalid / missing results; never silently drop unfavorable
  cases or select the best seed without reporting all trials.
- Treat a reproducible native crash or dependency/ABI mismatch as a build or
  implementation blocker, not as evidence that the dataset or algorithm failed.

### 3. Validation and visualization

- Separate in-sample solver fit from accuracy evidence. Require repeatability,
  contiguous or cross-sequence holdout, and an independent physical check
  appropriate to the sensors and target parameters.
- If INS IMU and INS pose come from the same fused navigation solution, do not
  call that pose an independent ground truth. Seek independent surveyed
  extrinsics, an independent pose source, held-out prediction, or a physical
  point-cloud / motion-compensation check.
- Review a time-colored BEV trajectory overlay showing the figure-eight path,
  turn direction, gaps, and segment boundaries. Compare LiDAR-derived motion
  with the reference only when the reference is genuinely independent.
- Produce comparable point-cloud views for each candidate after applying that
  candidate's transform. Inspect doubled surfaces, ghosting, thickness, and
  sharpness on the same capture and with the same crop, stride, deskew, and
  display scale.
- Keep numeric diagnostics and human review artifacts under a stable contract.
  Prefer extrinsic accuracy, repeatability, holdout prediction, and failure
  accounting over proxy scores alone.

## whl-cal LiDAR-to-IMU guidance

- Preserve GRIL as the in-repository baseline / candidate until evidence
  justifies a change. Read the GRIL validation skill and release-gate document
  before running or judging a result.
- Treat native GRIL execution, equivalence to its frozen ROS reference, and
  physical calibration accuracy as separate claims with separate evidence.
- Before comparing GRIL with `LiDAR_IMU_Init`, verify that a separate runnable
  implementation exists, record its exact revision and input requirements, and
  establish whether it is methodologically independent of GRIL / LI-Init.
  Do not report an algorithm comparison if only lineage or a paper description
  is available.
- For GRIL, audit point timing, frame and transform direction, LiDAR-only
  frontend repeatability, derivative quality, observability, seed sensitivity,
  and held-out physical behavior before changing the optimizer.
- Keep rotation, translation / lever-arm, and time-offset evidence distinct.
  Do not treat configured extrinsics as ground truth or accept weak `x/y/yaw`
  merely because other parameters or aggregate metrics look good.

## Controlled experiment contract

Before each comparison, write down:

- **Methods:** baseline and one candidate, with revisions and configuration.
- **Dataset matrix:** nominal, independent holdout, and a known difficult case
  where available; identify missing cases rather than implying coverage.
- **Hypothesis:** one change, expected measurable gain, and constraints.
- **Locked inputs:** capture identity / hashes, extraction settings, split,
  hardware, and evaluation implementation.
- **Metrics:** extrinsic / temporal accuracy evidence, repeatability,
  holdout prediction, failure rate, runtime, and diagnostic completeness.
- **Failure analysis:** per-case outcomes, parameter bounds, known weaknesses,
  and alternative explanations.
- **Decision gate:** explicit thresholds or a reason the result is still
  inconclusive; never promote based only on convergence or one favorable plot.

Record both successful and failed experiments so the next iteration does not
repeat disproven assumptions. Keep the original baseline executable until the
candidate wins on repeated, representative, independently reviewed evidence.

## Default response format

For each research iteration, provide only these sections:

```text
## Understanding
## Research
## Hypothesis
## Experiment
## Evidence
## Critique
## Decision
## Next Step
```

The skill is complete only when the research supports a reproducible,
evidence-backed engineering decision. If evidence is not yet available, say so
and make the next experiment measurable rather than overstating certainty.
