# Tools

This directory contains operational scripts layered on top of the core
`lidar2lidar` package.

Contents:

- `tools/lidar2lidar/list_topics.py`: inspect Apollo record topics
- `tools/lidar2lidar/extract_pcd.py`: export `PointCloud2` messages to PCD
- `tools/lidar2lidar/merge_pcd.py`: merge two PCD files with a transform

Design rule:

- `lidar2lidar/`: reusable library code and production entrypoints
- `tools/`: operational helpers and data preparation scripts