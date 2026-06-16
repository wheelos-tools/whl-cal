---
name: calibration-capture-design
description: Dataset and capture-design guide for sensor calibration pipelines. Use this when asked what data to collect, how to recollect a failing calibration dataset, how to improve observability, or whether a solver problem is really a data problem.
---

# Calibration Capture Design

Use this skill to design calibration data collection before blaming the solver.

## When to use

Use this skill when the task involves:

- planning a calibration recording session
- deciding whether poor results require recollection instead of optimizer changes
- improving coverage, excitation, observability, or scene diversity
- defining module-specific capture checklists
- designing a dataset that can support benchmarking and holdout evaluation

## Core policy

In calibration work, the highest-leverage fix is often better data rather than a more complex algorithm.

Always design capture around:

1. **Required raw signals**
   - topics, frames, timestamps, static transforms, and supporting metadata
2. **Coverage and excitation**
   - viewpoint, depth, tilt, motion direction, turn balance, and scene diversity
3. **Observability**
   - the data must actually constrain the unknowns of interest
4. **Auditability**
   - record enough context to explain failures later

## Capture design workflow

For any module, specify:

1. **what must be recorded**
2. **what is strongly recommended**
3. **what motion or scene pattern is needed**
4. **what weak-data pattern to avoid**
5. **what fast post-record checks should be run before optimization**

## Module-specific heuristics

Prefer guidance like:

- **camera**
  - use the exact production camera mode
  - cover center, corners, and multiple tilts
  - avoid autofocus or auto-exposure drift
- **lidar2camera**
  - keep the target fully visible in both modalities
  - vary depth, horizontal position, and tilt
  - avoid collecting all poses from one image region or one depth
- **lidar2lidar**
  - prefer static scenes with walls, corners, poles, and facades
  - reduce moving traffic and open-sky-only segments
  - include geometry that supports repeatable registration
- **lidar2imu**
  - include both left and right turns
  - include acceleration and braking, not only cruising
  - keep enough flat-road support for ground and motion assessment

## Recollection triggers

Recommend recollection before solver redesign when you see:

- low accepted-pair ratio
- weak pose or motion diversity
- poor image or scene coverage
- unstable holdout behavior
- multiple solution families
- geometry that obviously underconstrains the requested degrees of freedom

## Required output

When using this skill, include:

1. Capture goal
2. Required signals
3. Planned excitation and coverage
4. Expected observability
5. Fast post-record checks
6. Recollection criteria

## Reference canon

Prefer:

- in-repo capture guidance such as `docs/apollo_data_collection.md`
- **OpenCV calibration docs** for camera-board coverage practice
- **Kalibr** workflows for staged collection and validation discipline
- production autonomy workflows that privilege stable data collection over brittle solver heroics

## Anti-patterns

Avoid these capture mistakes:

- collecting many redundant frames instead of diverse ones
- recording only nominal straight-line motion for motion-based calibration
- letting auto-exposure or autofocus change during a session
- using scenes with poor geometric structure for targetless registration
- trying to fix an underconstrained dataset with a heavier optimizer

## Output style

Lead with whether the current dataset is:

- sufficient
- sufficient for review only
- needs targeted recollection
- fundamentally underconstrained

Then provide the smallest useful recollection plan.
