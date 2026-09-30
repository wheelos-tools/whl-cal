---
audience: user
stability: experimental
last_tested: 2026-05-26
---

# LiDAR↔Camera nuScenes benchmark

This workflow benchmarks **experimental targetless lidar↔camera calibration**
on nuScenes using the dataset's own camera intrinsics and lidar-to-camera
ground-truth extrinsics.

It does **not** replace the production reference-board workflow in
[docs/lidar2camera_quickstart.md](lidar2camera_quickstart.md).

Use this page when you want to answer questions like:

1. if I perturb the ground-truth extrinsic, can my method recover it?
2. is the method better than the perturbed-input baseline?
3. does the benchmark wiring itself make sense?

## What this benchmark currently runs

Current in-repo methods:

| Method | Role |
| --- | --- |
| `identity` | baseline that keeps the perturbed input unchanged |
| `edge_refine` | experimental targetless edge-based refinement path |
| `direct_visual_refine` | experimental targetless direct-visual (NID-style) local refinement path |
| `sensorscalib_line_refine` | experimental SensorsCalibration-style line-feature refinement path |
| `silhouette_refine` | experimental targetless structure/silhouette edge refinement path |
| `batch_hybrid_refine` | experimental multi-frame shared-extrinsic consensus search |
| `oracle_gt` | sanity path that returns nuScenes ground truth exactly |

The benchmark is intentionally conservative:

- `edge_refine` is allowed to optimize only inside a bounded local search region
- `direct_visual_refine` uses image↔projected-LiDAR direct information objectives
  (NID-style) inside the same bounded local search region
- `sensorscalib_line_refine` uses line-feature alignment inspired by
  SensorsCalibration CRLF-style road-scene calibration
- `silhouette_refine` adds LiDAR projected structure/silhouette edge comparison
- `batch_hybrid_refine` uses multiple same-camera frames with a shared extrinsic
  update, a multistart coarse-search stage, and a coordinate-consensus refinement
  stage, but it now falls back to the initial guess unless the update is both
  multi-scene-consistent, preserves projected-point visibility, and stays inside
  the configured uncertainty range
- if the update drifts too far or fails the guard checks, it falls back to the
  initial perturbed extrinsic instead of forcing a risky update

## Input assumptions

The current implementation reads the OpenDataLab-style layout directly from the
local filesystem, for example:

```text
/mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/
  nuscenes_infos_val.pkl
  samples/
  sweeps/
```

The loader currently expects the new `data_list` schema in
`nuscenes_infos_val.pkl` and resolves:

- images from `samples/<CAM_NAME>/...`
- lidar from `samples/LIDAR_TOP/...` with a `sweeps/LIDAR_TOP/...` fallback

## Minimal command

```bash
lidar2camera-nuscenes-benchmark \
  --info-path /mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/nuscenes_infos_val.pkl \
  --camera-names CAM_FRONT \
  --sample-limit 8 \
  --rotation-perturb-deg 0.5,1.0,2.0 \
  --translation-perturb-m 0.02,0.05,0.10 \
  --perturbations-per-level 2 \
  --output-dir outputs/lidar2camera/nuscenes_benchmark
```

## Initial-value precision audit

Use this command when the question is "how accurate does my initial extrinsic
need to be?"

```bash
lidar2camera-nuscenes-precision-audit \
  --info-path /mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/nuscenes_infos_val.pkl \
  --camera-names CAM_FRONT \
  --sample-limit 4 \
  --translation-magnitudes-m 0.01,0.02,0.05,0.10 \
  --rotation-magnitudes-deg 0.1,0.3,0.5,1.0,2.0 \
  --output-dir outputs/lidar2camera/targetless_precision_audit
```

It writes:

- `diagnostics/objective_landscape.csv`
- `diagnostics/objective_summary.csv`
- `diagnostics/precision_audit_summary.yaml`

The audit does not use GT as an optimizer target. It asks a stricter diagnostic
question: if the current initial value is perturbed, does the targetless
edge/silhouette objective at the true GT correction score better than staying at
the perturbed initial value?

## Useful options

| Option | Meaning |
| --- | --- |
| `--camera-names` | comma-separated cameras, e.g. `CAM_FRONT,CAM_FRONT_LEFT` |
| `--sample-limit` | cap on selected camera samples |
| `--sample-tokens` | explicit sample-token filter |
| `--methods` | comma-separated method list |
| `--reference-transform-mode` | `rigid_sensor` or `sample_pair`; default is `rigid_sensor` |
| `--max-sensor-time-delta-ms` | reject camera/LiDAR pairs with too much timestamp skew |
| `--search-rotation-deg` | local search bound around the initial guess |
| `--search-translation-m` | local search bound around the initial guess |
| `--image-downscale` | speed/quality trade-off for the experimental method |
| `--intensity-percentile` | LiDAR intensity percentile used by the targetless objective |
| `--max-points` | maximum points used by the targetless objective |
| `--visualization-max-range-m` | independent dense point range used only for visual overlays |
| `--visualization-max-points` | maximum dense points used only for visual overlays |
| `--overlay-point-radius-px` | rendered point radius for depth-colored overlays |
| `--methods` | include `direct_visual_refine` to evaluate the new direct-visual candidate |

## Split-directory comparison run (baseline vs SensorsCalibration-style)

If you want two clean output directories for side-by-side review:

```bash
lidar2camera-nuscenes-split-benchmark \
  --info-path /mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/nuscenes_infos_val.pkl \
  --camera-names CAM_FRONT \
  --sample-limit 4 \
  --output-dir outputs/lidar2camera/nuscenes_split_benchmark
```

This writes:

- `outputs/lidar2camera/nuscenes_split_benchmark/baseline_algorithms/`
- `outputs/lidar2camera/nuscenes_split_benchmark/sensorscalibration_algorithms/`
- `outputs/lidar2camera/nuscenes_split_benchmark/split_compare_summary.yaml`

## Outputs

The benchmark writes:

- `metrics.yaml`
- `diagnostics/benchmark_manifest.yaml`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/per_sample_results.csv`
- `diagnostics/per_method_summary.csv`
- `diagnostics/perturbation_summary.csv`
- `diagnostics/success_curves.yaml`
- `diagnostics/visualization_index.yaml`
- `diagnostics/overlays/*.png`
- `diagnostics/overlays/*_comparison.png` for side-by-side initial/final/GT
  depth-colored point-cloud projection review
- `diagnostics/overlays/*_debug.png` for targetless edge-debug panels

## How to read the result

### 1. Check the sanity path first

`oracle_gt` should be essentially perfect:

- rotation error ~= `0`
- translation error ~= `0`

If not, the benchmark wiring is wrong.

### 2. Compare `edge_refine` against `identity`

This is the core benchmark question:

- if `edge_refine` is **worse** than `identity`, the current objective is not yet
  strong enough for that slice
- if it is **equal**, the guard rails protected the run and the method chose not
  to move
- if it is **better**, the method recovered part of the perturbation

### 3. Use the overlays only as supporting evidence

Do **not** treat a single pretty overlay as success.

Prefer:

1. `per_method_summary.csv`
2. `perturbation_summary.csv`
3. `success_curves.yaml`
4. `metrics.yaml`
5. `acceptance_report.yaml`
6. representative overlays

Pay special attention to:

- `accepted_update_rate`
- `mean_objective_improvement`
- `initial_projected_point_count`, `final_projected_point_count`,
  `gt_projected_point_count`
- `initial_projected_point_ratio`, `final_projected_point_ratio`,
  `gt_projected_point_ratio`
- `*_projected_depth_p50_m`, `*_projected_depth_p95_m`
- `*_projected_bbox_area_ratio`
- targetless `*_comparison.png` panels that compare initial, final, and
  nuScenes-GT depth-colored point-cloud projections on the same RGB image
- targetless `*_debug.png` panels that compare image edges with initial/final
  projected LiDAR structure edges

The `*_comparison.png` panels are the preferred visual sanity check for demo and
debugging: initial/final/GT projections are rendered side by side with the same
TURBO depth color map, black point outlines, a depth legend, and per-panel point
counts. A correct update should move the final projection toward the GT
projection without losing projected point count or producing obvious depth-layer
drift.

The benchmark intentionally separates **optimization points** from
**visualization points**:

- optimization can stay conservative and use high-intensity / bounded points
- visualization uses a denser range-limited cloud so reviewers can actually see
  whether the LiDAR projection lands on vehicles, lane boundaries, poles, and
  building edges

If a visual panel still looks like a plain camera image, check
`*_projected_point_count` first. For nuScenes front-camera review, a healthy
single-frame overlay should usually have hundreds to thousands of projected
points, not only a few dozen.

The `projection_visibility` gate in `acceptance_report.yaml` fails or warns if
the initial/final/GT visual panels do not contain enough projected LiDAR points.
This catches the common failure mode where the math runs but the review image is
not actually useful.

Current initial-value conclusion:

- final audit run:
  `outputs/lidar2camera/targetless_precision_audit_final/diagnostics/`
- tested 4 CAM_FRONT scenes, 27 axis-aligned perturbation cases, 108 rows total
- perturbations covered x/y/z translations at 1, 2, 5, and 10 cm and
  roll/pitch/yaw at 0.1, 0.3, 0.5, 1.0, and 2.0 degrees
- GT correction scored better than the perturbed initial value in:
  - `edge_refine`: 48/108 = 44.4%
  - `silhouette_refine`: 36/108 = 33.3%
  - both objectives together: 32/108 = 29.6%
- pitch is especially weak: pitch >= 0.3 degrees had 0% edge+silhouette
  agreement on this slice
- roll/yaw are also not production-safe: roll/yaw >= 1.0 degree had 0%
  edge+silhouette agreement on this slice

Therefore the current single-frame `edge_refine` / `silhouette_refine`
objectives are **not reliable enough to recover normal measurement errors**.
Do not lower guard rails just to force a visible update; if the candidate does
not beat `identity`, keep the initial guess and treat the run as review-only.

Current multi-frame conclusion:

- `batch_hybrid_refine` is now implemented as:
  - shared-extrinsic multi-frame search
  - multistart coarse hypotheses
  - coordinate-consensus refinement
  - projected-point-retention guard
- the benchmark defaults now bias toward practical speed:
  - `image_downscale=2.0`
  - smaller batch search budget than the first batch prototype
- final benchmark slice:
  `outputs/lidar2camera/targetless_batch_multistart_eval_final/diagnostics/`
- tested 4 `CAM_FRONT` samples with perturbation buckets:
  - `(0.5 deg, 0.02 m)`
  - `(1.0 deg, 0.05 m)`
  - `(2.0 deg, 0.10 m)`
- runtime for `identity,batch_hybrid_refine,oracle_gt` on that slice:
  about `1m15s`
- measured effect on that slice:
  - mean rotation error improved from `1.1667 deg` (`identity`) to `1.1406 deg`
  - loose success improved from `8.3%` to `16.7%`
  - accepted update rate was `16.7%`
  - mean translation did **not** improve overall:
    `0.0567 m` (`identity`) -> `0.0587 m`
  - `(2.0 deg, 0.10 m)` perturbations were **not** recovered

Practical input requirement for this implementation:

- use an initial extrinsic that is already good enough for projection review
- expect the current targetless path to behave as a **safe warm-start refiner**,
  not as a production-grade automatic calibrator
- it can help some `0.5-1.0 deg / 0.02-0.05 m` warm-start cases, but it does not
  yet robustly correct larger installation error
- if automatic recovery from several centimeters or >=0.5 degree roll/pitch/yaw
  is required with production confidence, this implementation still needs a
  stronger multi-frame/semantic/depth-boundary optimizer before production use

## Practical interpretation

This benchmark is designed for **evidence-based comparison**, not for claiming
automatic SOTA by default.

In the current repo, nuScenes plays a very specific role:

- it is the **validation bed** for experimental targetless lidar↔camera methods
- it is **not** the production release surface for the checkerboard pipeline
- it tells you whether a targetless candidate can recover known GT perturbations
  under a reproducible public-data protocol

A serious comparison still requires:

1. a held-out nuScenes subset
2. the same perturbation schedule for every method
3. the same camera split and success thresholds
4. comparison against strong external targetless baselines

Current status:

- the repository now has a reproducible nuScenes GT-perturbation benchmark
- the in-repo targetless path is an **experimental baseline**, not a production
  winner
- the reference-board pipeline remains the production release baseline
