---
audience: user
stability: experimental
last_tested: 2026-05-25
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
| `oracle_gt` | sanity path that returns nuScenes ground truth exactly |

The benchmark is intentionally conservative:

- `edge_refine` is allowed to optimize only inside a bounded local search region
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

## Useful options

| Option | Meaning |
| --- | --- |
| `--camera-names` | comma-separated cameras, e.g. `CAM_FRONT,CAM_FRONT_LEFT` |
| `--sample-limit` | cap on selected camera samples |
| `--sample-tokens` | explicit sample-token filter |
| `--methods` | comma-separated method list |
| `--search-rotation-deg` | local search bound around the initial guess |
| `--search-translation-m` | local search bound around the initial guess |
| `--image-downscale` | speed/quality trade-off for the experimental method |

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

## Practical interpretation

This benchmark is designed for **evidence-based comparison**, not for claiming
automatic SOTA by default.

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
