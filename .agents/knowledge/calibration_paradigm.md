---
audience: dev
stability: stable
---

# Calibration data, algorithm, and evaluation boundaries

Keep calibration workflows separated into:

1. **Data** — decode inputs, preserve timestamps/frames/provenance, normalize
   samples, and report data-quality limitations before optimization.
2. **Algorithm** — estimate the calibration without hiding data-selection or
   observability decisions.
3. **Evaluation** — report metrics, limitations, and review/release guidance
   from completed outputs; evaluation gates must not silently alter solver
   execution.

For motion- or scene-dependent calibration, select and qualify bounded windows
at the data layer when needed; do not let an optimizer silently hide weak or
abnormal input segments.

The stages are shared design responsibilities, not a promise that every module
uses the same files or schema. Follow each module's current implementation and
quick start.

## Current artifact contracts

- `lidar2lidar`, `camera2camera`, and `lidar2camera` write `metrics.yaml` and, on their
  production review paths, shared diagnostics such as `standardized_data.yaml`,
  `data_quality.yaml`, `acceptance_report.yaml`, and `visualization_index.yaml`.
- `camera` writes a calibration YAML, `customer_summary.yaml`, and a
  `<calibration-stem>_diagnostics/` directory with acceptance, data-quality,
  standardized-data, visualization-index, and per-view artifacts. It does not
  use the shared `metrics.yaml` contract.
- Native GRIL (`lidar2imu run-native`) writes `manifest.yaml`,
  `customer_summary.yaml`, `GRIL_Calib_result.txt`,
  `GRIL_full_frontend_trace_v1.txt`, and `GRIL_batch_trace_v1.txt`, plus the
  native config. Post-run evaluation commands produce their own reports. This
  runner does not emit the shared `metrics.yaml` or `diagnostics/` contract.

These are implementation-specific contracts and can vary by command. Check
the module quick start and output manifest before assuming a file exists.

## Review rule

Solver convergence, registration fitness, or a successful process exit does
not establish calibration accuracy. Assess input provenance, data sufficiency,
observability, repeatability, holdout/physical evidence where available, and
visual diagnostics. Keep rejected and no-result runs visible.

For maintained workflow details, use [the quickstart index](../../docs/quickstart_index.md)
and [the review guide](../../docs/calibration_review_guide.md). Historical
evidence belongs in [agent notes](../notes/README.md).
