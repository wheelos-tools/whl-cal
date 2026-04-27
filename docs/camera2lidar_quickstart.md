# Camera ↔ LiDAR Quick Start

This quickstart covers the two available paths in this repo:

- Reference-based (target / checkerboard) — industrial baseline and recommended release path
- Learning-based (targetless) — experimental, diagnostic and visualization path

Reference-based (recommended baseline)

1. Prepare dataset: synchronized image + point-cloud pairs with diverse board poses. Ensure camera intrinsics are pre-calibrated.
2. Create a config (compatibility helper will create a default):

```bash
python camera2lidar/reference_based.py
# If no config.yaml exists, the script writes a default template and exits.
# Edit config.yaml: set data_directory to synchronized pairs, set output.directory.
python camera2lidar/reference_based.py
```

3. Expected outputs (config-controlled output directory):

- calibrated_tf.yaml
- metrics.yaml
- diagnostics/reference_dataset.yaml
- diagnostics/extraction.yaml
- diagnostics/optimization.yaml
- calibrated/*.yaml (per-pose or final)

Checks and interpretation

- Start with `metrics.yaml` — check final RMS / per-pose reprojection RMS.
- Check `diagnostics/reference_dataset.yaml` for accepted poses and skip reasons.
- Run leave-one-pose-out (L1PO) repeatability if available: evaluate transform spread and per-pose reprojection spread.
- Recommendation field (accepted_reference_candidate / repeatability_review / recollect_data) indicates whether the run is production-ready.

Learning-based (experimental)

Run the packaged learning-based demo (creates a sample dataset if none exists):

```bash
python camera2lidar/learning_based.py
```

Outputs:

- learning_output_expert/final_calibration.yaml (scale + transform + metrics)
- learning_output_expert/final_alignment_overlay.png (visual overlay)

Checks and interpretation

- Inspect `final_calibration.yaml` for `metrics.fitness` and `metrics.rmse`.
- Open `final_alignment_overlay.png` for a visual overlay of LiDAR points onto the image.
- Treat this method as experimental: compare with reference-based runs and use repeatability/holdout checks before trusting for production.

Where to go next

- For design rationale and recommended industrial metrics, read: context/lidar2camera_context.md
- For implementing a repo-level baseline, follow the recommended pattern: data extraction → algorithm → evaluation (see context file above).

