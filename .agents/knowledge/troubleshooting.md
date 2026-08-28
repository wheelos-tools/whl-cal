# Troubleshooting

## When
Read when calibration diverges, produces inconsistent transforms, passes fitness but looks wrong, or runs too slowly.

## Rules
- Triage in order: input contract, coordinates/time, frontend trajectory, observability, staged solve, repeatability, holdout.
- Compare against a null/baseline model and independent data; do not compare only against the configured seed.
- Inspect explicit skip reasons, coarse/fine metrics, information diagnostics, and visualizations.
- For timing failures, verify acquisition/header/point timestamps and gap boundaries before tuning an optimizer.
- For GRIL, run the executable GRIL validation skill rather than recreating conversion or diagnostic scripts.

## Do NOTs
- Do not differentiate motion across a record gap.
- Do not infer temporal bias from publish order.
- Do not blame scan-to-map until the input contract and LiDAR frontend are independently checked.
- Do not accept a result solely because Ceres converged or a map appears bounded.

## Sources
- General review ladder: `docs/calibration_review_guide.md`
- Validated repository conclusions: `context/knowledge_base/validated_conclusions.md`
- Open verification points: `context/knowledge_base/verification_points.md`
- Timing diagnosis: `context/timing_sync_context.md`
- LiDAR-to-IMU history: `context/lidar2imu_context.md`
- Scan-to-map history: `context/scan2map_context.md`
- GRIL workflow: `.agents/skills/gril-calib-validation/SKILL.md`
