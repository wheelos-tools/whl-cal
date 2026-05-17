---
audience: dev
stability: experimental
last_tested: 2026-05-17
---

# Calibration algorithm review and evaluation-contract update

This note records the review of `lidar2lidar` and `lidar2imu` against
industry-style calibration practice, using:

- dataset: `/mnt/synology/REDACTED/2026-5-6-标定/`
- metadata note: `calibration_dataset_2026_05_06.md`
- timing notes:
  - `timing_sync_context.md`
  - `timing_topic_table.md`

## Dataset-specific conclusion

The 2026-05-06 dataset is **not** a full raw4 corner-rig LiDAR package. The
record exposes raw:

- `/apollo/sensor/vanjeelidar/left_front/PointCloud2`
- `/apollo/sensor/vanjeelidar/right_back/PointCloud2`

and fused:

- `/apollo/sensor/lidar/fusion/PointCloud2`

It does not expose raw `right_front` or `left_back` topics. Therefore this bag
can validate timing, data-quality reporting, lidar2imu extraction, and a
two-LiDAR raw-pair lidar2lidar diagnostic, but it cannot prove four-corner
rectangle-loop lidar2lidar production quality.

## Shared best-practice calibration paradigm

Both lidar2lidar and lidar2imu should keep the same top-level lifecycle:

1. **Data quality before optimization**
   - topic availability
   - timestamp provenance
   - sync jitter
   - scene / motion / ground support
   - explicit reject reasons
2. **Algorithm**
   - solver may evolve quickly
   - weak components must be locked or marked diagnostic
   - priors and reference transforms must be reported separately
3. **Evaluation**
   - stable `metrics.yaml`
   - detailed `diagnostics/*.yaml`
   - visual / tabular artifacts
   - explicit final release decision

The implementation now adds a shared final review contract:

- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `metrics.yaml.summary.final_acceptance_status`
- `metrics.yaml.summary.release_ready`
- `metrics.yaml.final_acceptance`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`

The shared helper lives in:

- `calibration_common/evaluation.py`

## lidar2lidar review

### What is reasonable

The current `lidar2lidar` architecture is broadly reasonable:

- extraction / algorithm / evaluation are separated
- topic, TF, sync, overlap, and skip reasons are reported
- multi-method registration and information-matrix diagnostics exist
- workflow YAML supports planned topology
- loop closure provides a rig-level consistency check
- visual evaluation already includes wall / corner / slice metrics

This matches the correct industry direction: do not trust ICP fitness alone;
combine geometry, repeatability, graph consistency, and visual evidence.

### Remaining limitations

Important gaps remain:

1. scene sufficiency is still more of a report than a hard front-door gate
2. there is no true holdout-window validation for pairwise LiDAR edges
3. visualization should eventually add BEV / XZ / YZ slices, residual heatmaps,
   and per-plane crops
4. full raw4 production evaluation requires all four raw LiDAR topics
5. generic relation planning is not yet a dedicated rectangle-ring solver

### 2026-05-06 minimal validation run

Command shape:

```bash
lidar2lidar-auto \
   --record-path /mnt/synology/REDACTED/2026-5-6-标定/ \
   --conf-dir lidar2lidar/conf \
  --target-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --source-topics /apollo/sensor/vanjeelidar/right_back/PointCloud2 \
  --output-dir outputs/lidar2lidar/2026_05_06_review_pair \
  --sync-threshold-ms 40 \
  --min-overlap 0.05 \
  --methods 2 \
  --max-samples 1
```

Result summary:

| field | value |
| --- | --- |
| calibrated edges | `1` |
| average fitness | `0.04128529698149951` |
| average inlier RMSE | `0.03570456531507494` |
| min overlap ratio | `0.05233012236396772` |
| max condition number | `1965.4476073108776` |
| final acceptance | `warning` |
| release ready | `false` |

The warning is expected and correct because:

- only one raw pair is present
- scene sufficiency is warning
- repeatability is weak
- visual geometry is unavailable
- loop closure is unavailable

This is a diagnostic pair run, not a production rig calibration result.

## lidar2imu review

### What is reasonable

The current `lidar2imu` architecture is also broadly reasonable:

- record conversion exports standardized samples
- ground and motion samples are separated
- solver stages reflect the physical problem:
  - ground orientation
  - ground translation
  - yaw from motion
  - translation from motion
  - joint refinement
- weak planar motion is handled with `--planar-motion-policy auto`
- observability is expected-rank aware
- yaw uses cost-scan flatness / plateau evidence
- holdout, repeatability, reference consistency, extraction consistency, and
  basin stability are already present as evaluation concepts

### Remaining limitations

Important gaps remain:

1. pre-calibration gates need to become harder and more explicit
2. sync jitter is recorded but should become a blocking gate when excessive
3. residual plots / cost-scan plots / holdout summaries should become standard
   visual artifacts
4. final release status must be separate from solver convergence
5. non-`full_6dof_candidate` outputs should remain valid diagnostics but not be
   promoted as production full-6DoF calibration

### 2026-05-06 minimal validation run

Command shape:

```bash
lidar2imu-convert-record \
   --record-path /mnt/synology/REDACTED/2026-5-6-标定/ \
   --output-dir outputs/lidar2imu/2026_05_06_review_lf \
  --lidar-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/imu \
  --max-ground-samples 4 \
  --max-motion-samples 4 \
  --motion-frame-stride 10 \
  --min-registration-fitness 0.20 \
  --planar-motion-policy auto \
  --identity-initial-transform \
  --calibrate
```

Result summary:

| field | value |
| --- | --- |
| ground samples | `4` |
| motion samples | `4` |
| final translation | `x=-1.055, y=2.665, z=0.31229393644918885` |
| final euler | `yaw=135.00038388114902, roll=0.19021858259643493, pitch=0.08513187529699791` |
| solver policy | `auto -> freeze_xyyaw` |
| weak planar reasons | `yaw_rotation_degenerate` |
| yaw reasons | `flat_cost_scan`, `wide_cost_plateau` |
| final acceptance | `warning` |
| release ready | `false` |

The warning is expected and correct because:

- motion registration is weak
- motion rotation / translation residuals are weak
- yaw is not observable enough for release
- holdout generalization is unavailable in the minimal run
- the solver correctly locks `yaw/x/y`

This bag can support partial review of `z/roll/pitch` behavior and conversion
quality, but the minimal run should not be promoted as a release-ready full
6DoF lidar2imu result.

## Implementation summary

Code changes:

- added `calibration_common/evaluation.py`
- added `final_acceptance` to lidar2lidar metrics
- added `final_acceptance` to lidar2imu metrics
- both pipelines now write:
  - `diagnostics/standardized_data.yaml`
  - `diagnostics/data_quality.yaml`
  - `diagnostics/acceptance_report.yaml`
  - `diagnostics/status_summary.csv`
  - `diagnostics/visualization_index.yaml`
- updated design docs to document the shared release-review artifacts

This makes lidar2lidar and lidar2imu consistent at the evaluation-contract
level while still letting their algorithms evolve independently.

## Next algorithm iterations

### lidar2lidar

1. add holdout-window validation for pairwise edges
2. add BEV/XZ/YZ and residual heatmap artifacts
3. make scene sufficiency a true pre-optimization gate for required edges
4. add rectangle-ring-specific constraints for full raw4 rigs

### lidar2imu

1. promote sync jitter, ground inlier ratio, and motion diversity into hard
   pre-calibration gates
2. add residual / yaw-cost / holdout-repeatability plots or CSV tables
3. add explicit partial-DOF acceptance modes:
   - `z_roll_pitch_release`
   - `full_6dof_release`
   - `diagnostic_only`
4. require cross-run or holdout evidence before releasing planar components
