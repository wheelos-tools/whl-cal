# Calibration metric layers

Calibration artifacts are split by audience. The split reduces customer-facing
noise without removing evidence needed to debug or reject a calibration.

## Customer layer

Open `customer_summary.yaml` first.

| Module | Customer fields |
| --- | --- |
| Camera intrinsic | verdict, release readiness, accepted/required samples, average and P95 reprojection error, occupied image cells, visual-review paths |
| GRIL LiDAR-to-IMU | review verdict, complete rotation/translation/time result, LiDAR scan count, IMU sample count |

The GRIL summary remains `review_required` until independent trajectory,
point-cloud thickness, repeatability, and holdout gates are integrated and pass.
Solver completion never becomes a customer acceptance claim by itself.

## Developer layer

Developer diagnostics retain the complete evidence:

- Camera: acceptance gates, data quality, per-view reprojection CSV, sample CSV,
  coverage heatmap, capture runtime, and visualization index.
- GRIL: canonical dataset contract, native run manifest, frontend trace, batch
  trace, A/B comparison, repeatability, trajectory, dynamics, yaw/time, and
  independent submap diagnostics when those workflows are run.

Do not delete developer artifacts to simplify the customer report. Add new
diagnostics under the developer layer and promote only stable decision values
to `customer_summary.yaml`.
