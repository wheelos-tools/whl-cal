# lidar2camera Quick Start

Industrial reference-based LiDAR↔Camera calibration quick start.

1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Create a default config and inspect

```bash
# writes config.yaml with defaults if missing
lidar2camera-calibrate --write-default-config --config config.yaml
# edit config.yaml: set data_directory (paired image/.pcd), output.directory, camera intrinsics
```

3. Run calibration

```bash
lidar2camera-calibrate --config config.yaml
```

4. Outputs (output.directory)

- calibrated_tf.yaml
- metrics.yaml
- diagnostics/reference_dataset.yaml
- diagnostics/extraction.yaml
- diagnostics/optimization.yaml
- diagnostics/reference_overlay.png (visual overlay)
- calibrated/*.yaml (per-pose or final)

5. Quick checks

- `metrics.yaml` → final RMS, per-pose RMS, leave-one-pose-out repeatability
- `diagnostics/reference_dataset.yaml` → accepted poses and skip reasons
- `diagnostics/reference_overlay.png` → visual sanity check

6. Acceptance guidance

- Require final RMS and holdout (L1O) reproducibility to be within repository warning thresholds (see config.metrics in default config).
- If repeatability shows multiple distinct solution families, recollect with more diverse poses before promoting.

See also: context/lidar2camera_context.md for design rationale and recommended metrics.
