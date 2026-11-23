---
audience: user
stability: stable
---

# LiDAR-to-camera customer guide

## Choose a workflow

| Need | Use | Status |
| --- | --- | --- |
| Deliver LiDAR-to-camera extrinsics | [Target-based quick start](lidar2camera_quickstart.md) | Current release-oriented baseline |
| Review targetless methods on nuScenes | [Benchmark guide](lidar2camera_nuscenes_benchmark.md) | Experimental; not a replacement for target-based calibration |

For production runs, the quick start owns installation, configuration, commands,
and artifact names. This guide focuses on interpreting visual evidence and
making a cautious customer decision.

## Review a target-based result

Follow the [calibration review guide](calibration_review_guide.md) for the
complete artifact order. In addition, inspect:

- `diagnostics/checkerboard_alignment_previews/*.png`
- `diagnostics/reference_overlay.png`

Each checkerboard preview compares detected image corners with projections
from the initial and final transforms. A useful result aligns the final
projection across the whole board, not just at one corner. A mirrored corner
order, remaining drift, or alignment on only one side indicates a problem.

The reference overlay projects LiDAR points onto the camera image. Check that
points follow the board boundary and nearby scene structure. Points crossing
visible empty regions, a globally shifted scene, or smeared geometry contradict
the numeric metrics.

A single good-looking pose is alignment evidence, not proof of repeatability,
pose-family independence, or broad depth/tilt observability. Keep such a run
review-only when the holdout or acceptance evidence is missing.

## Review a targetless benchmark

Use the benchmark guide for commands, options, and report schemas. Review the
per-method and perturbation summaries before opening representative overlays.

- `*_debug.png` compares image edges with projected LiDAR structure edges from
  the initial and final transforms. The final panel should align better without
  adding unmatched edge regions.
- `*_comparison.png` compares initial, final, and ground-truth projections.
  For `identity`, initial and final should match. For an optimizing candidate,
  final should move toward ground truth without collapsing projected point
  count or damaging depth layering.

Check `diagnostics/per_method_summary.csv`,
`diagnostics/perturbation_summary.csv`, and
`diagnostics/acceptance_report.yaml` together with those images. A single
improved overlay is not sufficient evidence.

## Customer decision

Use the target-based pipeline for delivered calibration today. Treat targetless
results as experimental review evidence until they demonstrate repeatability
and independent accuracy against the reference path.
