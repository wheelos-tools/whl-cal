---
audience: dev
stability: stable
last_tested: 2026-05-17
---

# Production calibration paradigm

All calibration modules should follow the same production review lifecycle:

1. **Data**
   - normalize raw inputs into a standard representation
   - write metadata and transform provenance
   - evaluate data quality before optimization
2. **Algorithm**
   - run the solver only after data quality is known
   - preserve weak-observability decisions in the output
   - allow algorithms to iterate without changing the review contract
3. **Evaluation**
   - conclusion layer: release / review / reject
   - detail layer: coarse and fine metrics
   - visual layer: clouds, plots, CSVs, or visualization indices for manual review

## Required artifacts

Every production-style calibration run should write:

| artifact | layer | purpose |
| --- | --- | --- |
| `diagnostics/standardized_data.yaml` | data | normalized input representation and metadata |
| `diagnostics/data_quality.yaml` | data | pre/post extraction quality gates and recommendation |
| `calibrated_tf.yaml` | algorithm | consolidated final transform output |
| `metrics.yaml` | evaluation | stable machine-readable summary |
| `metrics.yaml.summary.final_acceptance_status` | evaluation | top-level conclusion |
| `metrics.yaml.summary.release_ready` | evaluation | boolean release gate |
| `metrics.yaml.final_acceptance` | evaluation | full gate list and actions |
| `diagnostics/acceptance_report.yaml` | evaluation | human-readable release review |
| `diagnostics/status_summary.csv` | evaluation | spreadsheet-friendly gate summary |
| `diagnostics/visualization_index.yaml` | visualization | how to inspect visual / tabular evidence |

Module-specific visual evidence should also be exported:

- `lidar2lidar`
  - colored merged clouds when available
  - `diagnostics/visual_evaluation.yaml`
  - `diagnostics/edge_metrics.csv`
  - `diagnostics/skipped_edges.csv`
- `lidar2imu`
  - `diagnostics/ground_residuals.csv`
  - `diagnostics/motion_residuals.csv`
  - `diagnostics/holdout_motion_residuals.csv`
  - `diagnostics/observability.yaml`

## Review order

1. Read `diagnostics/standardized_data.yaml`.
   - confirm topics / frames / sample counts / transform provenance
2. Read `diagnostics/data_quality.yaml`.
   - confirm data is good enough to optimize
3. Read `metrics.yaml.summary`.
   - check `final_acceptance_status` and `release_ready`
4. Read `diagnostics/acceptance_report.yaml`.
   - review failed or warning gates
5. Read `diagnostics/visualization_index.yaml`.
   - open clouds or plot CSV residuals before promoting a result

## Production rule

Solver convergence alone is never sufficient. A result is production-ready only
when:

- the standardized data representation is complete
- required data-quality gates pass
- algorithm observability supports the released degrees of freedom
- final acceptance is `pass`
- visual / tabular evidence does not contradict the metrics

If any required visualization is missing, mark the run as review-only until
equivalent evidence is provided.
