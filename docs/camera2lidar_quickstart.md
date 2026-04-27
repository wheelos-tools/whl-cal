# Camera ↔ LiDAR Quick Start (UNIFIED)

The project standard name is now `lidar2camera`. The old `camera2lidar` package remains as small compatibility wrappers that delegate to `lidar2camera`.

Canonical commands:

- Reference-based (recommended baseline):

```bash
# writes default config if missing and exits
lidar2camera-calibrate --write-default-config --config config.yaml
# then run
lidar2camera-calibrate --config config.yaml
```

- Experimental learning-based demo (still experimental; kept for diagnostics):

```bash
# compatibility wrapper — delegates to lidar2camera.learning_based
python camera2lidar/learning_based.py
# or directly run the module implementation
python -m lidar2camera.learning_based
```

What changed

- `lidar2camera/` is the canonical package. New implementations and metrics live there (see lidar2camera/cli.py and lidar2camera/reference_pipeline.py).
- `camera2lidar/` files are compatibility shims to preserve older invocation patterns.

Where to look for outputs

- calibrated_tf.yaml, metrics.yaml, diagnostics/, calibrated/*.yaml under the configured output directory.

See also: docs/lidar2camera_quickstart.md and context/lidar2camera_context.md for design rationale and acceptance guidance.
