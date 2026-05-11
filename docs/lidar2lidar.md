---
audience: user
stability: stable
P26-04-27
---


# LiDAR-to-LiDAR calibration overview

`lidar2lidar` now has three layers of documentation:

- **Overview**: this file
- **Quick Start**: [docs/lidar2lidar_quickstart.md](lidar2lidar_quickstart.md)
- **Current design**: [docs/lidar2lidar_design.md](lidar2lidar_design.md)

## What is implemented

- `lidar2lidar-topics`: inspect available `PointCloud2` topics in Apollo record data
- `lidar2lidar-auto`: run the automatic multi-LiDAR pipeline from raw record files
- `lidar2lidar-auto --workflow-yaml <yaml>`: run a workflow-planned pipeline with explicit or TF-derived relations
- `lidar2lidar-auto --loop-closure`: compare the pairwise star baseline against a graph-consistent rig solution
- `lidar2lidar-calibrate`: refine one source-target pair manually
- `lidar2lidar-extract`: export `PointCloud2` messages to PCD
- `lidar2lidar-merge`: visualize a merged result

## Why the docs are split

- **Quick Start** is for rerunning the current pipeline quickly.
- **Current design** is for future iteration on data extraction, calibration
  logic, and metrics / diagnostics.

## Current status

`lidar2lidar` already matches much of the iteration pattern used by `lidar2imu`:

- raw-data extraction from record files is built in
- the automatic pipeline is explicitly staged
- outputs already separate concise metrics from detailed diagnostics

The remaining work is mostly about making that structure easier to reason about
and iterate on, not replacing the implementation.

Use the Quick Start to run it, then use the design doc to inspect extraction,
candidate-pair selection, registration, and evaluation details.

For four-LiDAR vehicle rigs, the practical workflow is now:

1. define the relation plan in a workflow YAML
2. choose either:
   - `mode: tf_tree` for no-loop calibration driven by TF adjacency
   - `mode: explicit` for a user-defined chain / loop / check-edges plan
3. run `lidar2lidar-auto --workflow-yaml ...`
3. compare:
   - baseline `calibrated_tf.yaml`
   - loop-closed `loop_closed_tf.yaml`
   - `diagnostics/workflow.yaml`
   - `diagnostics/scene_sufficiency.yaml`
   - `diagnostics/loop_closure.yaml`
   - `diagnostics/visual_evaluation.yaml`
   - colored merged clouds for human inspection

The visual evaluation layer is meant for the common engineering questions that
numeric ICP metrics alone do not settle:

- are long walls thinner after global consistency?
- do corners and poles show less double-edge ghosting?
- does one sensor stay visibly offset from the others on flat facades?
- are BEV / XZ / YZ slices sharper after refinement?
