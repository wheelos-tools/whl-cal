# Agent notes

This directory contains dated investigations, experiment records, historical
runbooks, and temporary follow-ups. These files preserve evidence and context;
they are not the source of truth for current behavior. Verify commands,
interfaces, defaults, and conclusions against the current code and maintained
documentation before using them.

## Contents

- `datasets/`: dataset-specific inventories and screening.
- `reviews/`: dated algorithm and camera reviews.
- `lidar2imu/`, `lidar2lidar/`, `lidar2camera/`: module investigations.
- Root-level research files: cross-module studies, requirements, and timing
  evidence.
- `todos.md`, `verification_points.md`: open or historical follow-up lists.

For concise, current engineering guidance, use [`../knowledge/README.md`](../knowledge/README.md).

Paths under the gitignored `outputs/` directory refer to local run artifacts
and may not exist in a clean checkout.

## Current L2L experiment handoff

- [2026-11-24 Zhongce baseline A and data-expansion plan](lidar2lidar/2026-11-24_zhongce_baseline.md):
  refreshed three-static-window joint v2 research baseline, measured failure
  modes, structural counterexamples, and pending A/B/C data experiments.
