---
audience: dev
stability: stable
last_tested: 2026-04-27
---

# Calibration knowledge base index

This `context/` directory is the durable knowledge base for this repository.

The organizing rule is:

- **tested data wins**
- algorithms and metrics are expected to keep iterating
- conclusions are only valid at the level supported by real-bag evidence

## Recommended reading order

1. `principles/iteration_rules.md`
2. `knowledge_base/calibration_overview.md`
3. `knowledge_base/validated_conclusions.md`
4. `knowledge_base/engineering_index.md`
5. module detail notes:
   - `lidar2imu_context.md`
   - `prepared_rig_dataset_context.md`
   - `lidar2lidar_advanced_strategy.md`
   - `scan2map_context.md`
   - `scan2map_metrics_framework.md`
   - `lidar2camera_context.md`
6. `knowledge_base/verification_points.md`

## Directory structure

### Principles

- `principles/iteration_rules.md`
  - repo-level calibration iteration rules
  - data -> algorithm -> evaluation separation
  - acceptance rules driven by tested bags

### Knowledge base

- `knowledge_base/calibration_overview.md`
  - top-level repo calibration summary
  - current status of lidar2lidar / lidar2imu / lidar2camera
- `knowledge_base/validated_conclusions.md`
  - current accepted conclusions
  - what is already trustworthy
- `knowledge_base/verification_points.md`
  - open validation items
  - next experiments and unresolved questions
- `knowledge_base/engineering_index.md`
  - repo structure
  - commands
  - stable artifacts
  - current indexing / diagnostic surfaces

### Module detail notes

- `lidar2imu_context.md`
  - detailed lidar2imu iteration history
  - bag-by-bag findings
  - current weak-planar strategy
- `prepared_rig_dataset_context.md`
  - raw-LiDAR-only reusable extraction surface
  - run-eight bag findings for shared lidar2imu/lidar2lidar preprocessing
- `lidar2lidar_advanced_strategy.md`
  - next-stage production strategy for four-corner rig calibration
  - priors, topology-aware solving, scene sufficiency, and advanced metrics
- `scan2map_context.md`
  - detailed lidar2lidar scan2map direction note
- `scan2map_metrics_framework.md`
  - scan2map coarse/fine metric framework
- `lidar2camera_context.md`
  - initial camera / lidar2camera module note
  - current state and next repo-level target

### Working backlog

- `todos.md`
  - active experimental backlog and intermediate handoff notes

## Usage rule

Use the knowledge-base files for the current source of truth.

Use the module detail notes when you need the history behind a conclusion.
