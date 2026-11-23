# Agent knowledge

This directory contains concise, durable engineering facts for agents and maintainers. Supported behavior and user-facing instructions belong in [`docs/`](../../docs/quickstart_index.md); dated experiments and intermediate findings belong in [`../notes/`](../notes/README.md). Treat code as authoritative when sources conflict.

## Core guidance

- [Architecture](architecture.md): package boundaries and data flow.
- [Conventions](conventions.md): artifacts, schemas, style, and documentation.
- [Troubleshooting](troubleshooting.md): authoritative diagnostics and failure triage.
- [Calibration paradigm](calibration_paradigm.md): data, algorithm, and evaluation responsibilities.
- [Target requirements](calibration_target_requirements.md): module capture/target requirements.
- [Validated conclusions](validated_conclusions.md): current evidence-backed findings.
- [Timing synchronization](timing_synchronization.md): measurement-time and GRIL clock rules.

## Module research

- [LiDAR-to-IMU](lidar2imu_research.md)
- [LiDAR-to-LiDAR](lidar2lidar_research.md)
- [LiDAR-to-camera](lidar2camera_research.md)
- [Camera intrinsic](camera_intrinsic_research.md)

## Working and historical material

See [notes](../notes/README.md) for detailed module research, dated
investigations, experiment reports, and open verification items. Notes are
evidence records, not guaranteed descriptions of the current implementation.

Evidence paths under `outputs/` point to gitignored local run artifacts and may
not exist in a clean checkout.
