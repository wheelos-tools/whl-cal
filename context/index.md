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
5. `calibration_paradigm.md`
6. `calibration_target_requirements.md`
7. module detail notes:
   - `lidar2imu_context.md`
   - `lidar2imu_customer_runbook.md`
   - `prepared_rig_dataset_context.md`
   - `lidar2lidar_scan2scan_playbook.md`
   - `lidar2lidar_advanced_strategy.md`
   - `calibration_dataset_2026_05_06.md`
   - `calibration_algorithm_review_2026_05_17.md`
  - `camera_intrinsic_round01_review_2026_05_27.md`
   - `timing_sync_context.md`
   - `timing_topic_table.md`
   - `scan2map_context.md`
   - `scan2map_metrics_framework.md`
   - `lidar2camera_context.md`
8. `knowledge_base/verification_points.md`

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

- `calibration_paradigm.md`
  - shared data -> algorithm -> evaluation and visualization contract
- `calibration_target_requirements.md`
  - which modules require a physical target board
  - current hard requirements vs upgrade recommendations
  - external references behind the judgment
- `lidar2imu_context.md`
  - detailed lidar2imu iteration history
  - bag-by-bag findings
  - current weak-planar strategy
- `lidar2imu_customer_runbook.md`
  - customer-facing lidar2imu runbook
  - automatic workflow, result viewing, and good-vs-bad visualization examples
- `prepared_rig_dataset_context.md`
  - raw-LiDAR-only reusable extraction surface
  - run-eight bag findings for shared lidar2imu/lidar2lidar preprocessing
- `lidar2lidar_scan2scan_playbook.md`
  - current production-minded LiDAR-to-LiDAR decision ladder
  - high-overlap scan2scan baseline, loop-closure policy, and no-loop consensus fallback
- `lidar2lidar_advanced_strategy.md`
  - next-stage production strategy for four-corner rig calibration
  - priors, topology-aware solving, scene sufficiency, and advanced metrics
- `calibration_dataset_2026_05_06.md`
  - metadata, topic inventory, and calibration implications for `/mnt/synology/REDACTED/2026-5-6-标定/`
- `calibration_algorithm_review_2026_05_17.md`
  - lidar2lidar/lidar2imu algorithm review, shared acceptance contract, and target-bag validation
- `camera_intrinsic_round01_review_2026_05_27.md`
  - AprilGrid intrinsic round01 failure analysis
  - weak-view pruning experiments and round02 recollection checklist
- `timing_sync_context.md`
  - clock-source and timestamp-chain analysis for Vanjee / Huace / MSF
  - bag-level timing measurements and synchronization recommendations
- `timing_topic_table.md`
  - per-topic timestamp source table and field-selection guidance
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
