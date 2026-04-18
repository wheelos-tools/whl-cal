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
- `scan2map_context.md`
  - detailed lidar2lidar scan2map direction note
- `scan2map_metrics_framework.md`
  - scan2map coarse/fine metric framework
- `lidar2camera_context.md`
  - initial camera / camera2lidar module note
  - current state and next repo-level target

### Working backlog

- `todos.md`
  - active experimental backlog and intermediate handoff notes

## Usage rule

Use the knowledge-base files for the current source of truth.

Use the module detail notes when you need the history behind a conclusion.
