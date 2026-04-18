# Copilot instructions

## Build, test, and lint commands

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Lint / format

The checked-in CI workflow installs these tools:

```bash
python -m pip install black==24.4.2 flake8==7.0.0 isort==5.13.2
```

Use them directly when you need local lint checks:

```bash
black --check .
isort --check-only .
flake8 .
```

### Validation runs

There is no checked-in `pytest` / `unittest` suite. Validate changes by running the smallest relevant CLI entrypoint:

```bash
# Inspect one bag quickly
lidar2lidar-topics "$RECORD_DIR"

# Single-pair LiDAR registration
lidar2lidar-calibrate \
  --source-pcd source.pcd \
  --target-pcd target.pcd \
  --initial-transform lidar2lidar/conf/<parent>_<child>_extrinsics.yaml \
  --output-transform result.yaml

# Minimal lidar2lidar automatic run
lidar2lidar-auto \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/auto_calib_review \
  --sync-threshold-ms 10 \
  --min-overlap 0.30 \
  --methods 2 \
  --max-samples 1

# Build scan2map dataset artifact
lidar2lidar-scan2map-dataset \
  --record-path "$RECORD_DIR" \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/scan2map_dataset_review \
  --pose-topic /apollo/localization/pose

# Run scan2map candidate against that dataset
lidar2lidar-scan2map \
  --record-path "$RECORD_DIR" \
  --dataset-yaml outputs/lidar2lidar/scan2map_dataset_review/diagnostics/scan2map_dataset.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/scan2map_candidate_review

# Run lidar2imu from standardized samples
lidar2imu-calibrate \
  --input lidar2imu_samples.yaml \
  --output-dir outputs/lidar2imu/run01
```

## High-level architecture

- `lidar2lidar` is the main raw-record calibration package. Treat it as three layers:
  1. **Extraction** from Apollo `.record` data (`record_utils.py`, `auto_calib.py`, `scan2map_dataset.py`, `temporal_calib.py`)
  2. **Algorithm** (`auto_calib.py` for scan-to-scan baseline, `temporal_calib.py` for hand-eye comparison, `scan2map_calib.py` for scan-to-map candidate)
  3. **Evaluation** through stable `metrics.yaml` and `diagnostics/*.yaml`
- `lidar2imu` is intentionally split between:
  - record conversion to `standardized_samples.yaml` (`record_converter.py`)
  - calibration / metrics (`pipeline.py`, `algorithms.py`, `metrics.py`, `io.py`)
- Apollo record decoding does **not** use `cyber_record` or `record_msg`. The supported stack is:
  - `lidar2lidar/record_adapter.py` wrapping `pycyber.record.RecordReader`
  - `lidar2lidar/apollo_record_messages.py` providing minimal in-repo protobuf definitions for the message types this repo actually consumes
- `tools/` contains operational helper scripts layered on top of the library packages. Reusable workflow logic belongs in `lidar2lidar/` or `lidar2imu/`, not in `tools/`.

## Key conventions

- Preserve the repo’s **extraction → algorithm → evaluation** split. The algorithm may change aggressively, but output layout and judgment surfaces should stay stable.
- Calibration pipelines are expected to write stable top-level artifacts:
  - `calibrated_tf.yaml`
  - `metrics.yaml`
  - `diagnostics/`
  - and, when relevant, per-edge extrinsics under `initial_guess/` and `calibrated/`
- Use `lidar2lidar/extrinsic_io.py` helpers for reading and writing extrinsics. The canonical YAML schema is:
  - `header.frame_id`
  - `child_frame_id`
  - `transform.translation`
  - `transform.rotation`
- When working on `lidar2lidar`, keep the current algorithm roles clear:
  - `lidar2lidar-auto` is the practical **scan-to-scan baseline**
  - `lidar2lidar-temporal` is an **experimental comparison / observability branch**
  - `lidar2lidar-scan2map-dataset` and `lidar2lidar-scan2map` are the **scan-to-map path**
- `scan2map` is additive, not a rename of `scan2scan`. Keep the baseline visible in outputs and comparisons.
- `scan2map` currently assumes a dataset artifact first, then optimization:
  - `diagnostics/scan2map_dataset.yaml` is the extraction surface
  - `diagnostics/scan2map_optimization.yaml` and `diagnostics/evaluation.yaml` are the optimization/evaluation surfaces
  - the current default dataset rule is to keep every 3rd aligned scan as holdout
- Prefer decisions based on `coarse_metrics`, `fine_metrics`, information-matrix diagnostics, and explicit skip reasons. Do not treat registration fitness alone as sufficient evidence.
- Update the matching overview / quick-start / design docs when a pipeline, artifact, or command changes. The repo keeps these docs separate on purpose.
