# Architecture

## When
Read before changing package boundaries, pipeline stages, data extraction, or output contracts.

## Rules
- Keep extraction, algorithm, and evaluation as separate responsibilities.
- Put reusable workflow logic in library packages; keep `tools/` as operational entrypoints.
- Treat scan-to-scan, temporal calibration, and scan-to-map as distinct algorithm roles.
- Route GRIL Apollo record decoding through the Bzlmod `wheelos_core` C++ adapter; keep Python format-specific work at the canonical-dataset boundary.

## Do NOTs
- Do not hide extraction or evaluation inside an optimizer.
- Do not rename an experimental candidate into the production baseline.
- Do not create a second record-decoding stack without first checking the supported adapter.

## Sources
- Package and CLI inventory: `pyproject.toml`
- Repository overview: `README.md`
- Shared pipeline model: `context/calibration_paradigm.md`
- LiDAR-to-LiDAR design: `docs/lidar2lidar_design.md`
- LiDAR-to-IMU GRIL design: `docs/gril_ros_free_migration.md`
- LiDAR-to-camera design: `docs/lidar2camera_design.md`
- GRIL record ingress: `gril/adapters/core_record.py`, `tools/gril_record_core_reader/record_ingress.cc`
- Supported Apollo messages: `lidar2lidar/apollo_record_messages.py`
