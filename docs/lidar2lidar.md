# LiDAR-to-LiDAR calibration overview

`lidar2lidar` now has three layers of documentation:

- **Overview**: this file
- **Quick Start**: [docs/lidar2lidar_quickstart.md](lidar2lidar_quickstart.md)
- **Current design**: [docs/lidar2lidar_design.md](lidar2lidar_design.md)

## What is implemented

- `lidar2lidar-topics`: inspect available `PointCloud2` topics in Apollo record data
- `lidar2lidar-auto`: run the automatic multi-LiDAR pipeline from raw record files
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
