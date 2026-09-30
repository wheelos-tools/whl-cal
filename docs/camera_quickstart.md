---
audience: customer
stability: stable
---

# Camera intrinsic calibration

The delivered container includes the camera calibration program and all runtime
dependencies. Live capture always opens a visible preview; only an offline image
dataset may run without a display.

## 1. Get the image

Use the version supplied with the release:

```bash
docker pull <registry>/whl-cal-camera:<version>
export WHL_CAMERA_IMAGE=<registry>/whl-cal-camera:<version>
```

For an offline delivery package:

```bash
docker load -i whl-cal-camera-<version>.tar
export WHL_CAMERA_IMAGE=whl-cal-camera:<version>
```

Do not use an unversioned `latest` tag for a calibration record.

## 2. Live USB camera calibration

Create a writable output directory:

```bash
mkdir -p "$PWD/camera-output"
```

Allow the container process to use the current X11 display, then run:

```bash
xhost +si:localuser:"$(id -un)"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device /dev/video0:/dev/video0 \
  --env DISPLAY \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
  --mount type=bind,src="$PWD/camera-output",dst=/output \
  "$WHL_CAMERA_IMAGE" \
  --output-dir /output
```

The default target is an `11 x 8` inner-corner chessboard with `0.025 m`
squares. Keep the board visible, sharp, and distributed across the image. The
UI shows coverage and capture progress.

To collect reusable images without solving immediately:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device /dev/video0:/dev/video0 \
  --env DISPLAY \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
  --mount type=bind,src="$PWD/camera-output",dst=/output \
  "$WHL_CAMERA_IMAGE" \
  --output-dir /output \
  --session-name round01 \
  --capture-only
```

Live capture refuses to start when neither `DISPLAY` nor `WAYLAND_DISPLAY` is
available. It never collects customer data invisibly.

## 3. Offline image calibration

Mount the image directory read-only:

```bash
mkdir -p "$PWD/camera-output"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD/images",dst=/data/images,readonly \
  --mount type=bind,src="$PWD/camera-output",dst=/output \
  "$WHL_CAMERA_IMAGE" \
  --images-dir /data/images \
  --output-dir /output \
  --require-release-ready
```

All images in one run must use the same resolution, focus, zoom, exposure mode,
and distortion model.

## 4. Custom target or network camera

Copy and edit the supplied YAML configuration, then mount it:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --network host \
  --env DISPLAY \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
  --mount type=bind,src="$PWD/camera.yaml",dst=/config/camera.yaml,readonly \
  --mount type=bind,src="$PWD/camera-output",dst=/output \
  "$WHL_CAMERA_IMAGE" \
  --config /config/camera.yaml \
  --output-dir /output
```

The configuration supports:

- `chessboard`: `pattern_size`, `square_size`
- `aprilgrid`: grid dimensions, tag size, spacing, minimum visible tags
- `charuco`: board dimensions, square length, marker length
- `plumb_bob` and `fisheye` distortion models
- USB index or configured RTSP camera URI

Board dimensions must match the physical printed target.

## 5. Read the result

Each completed calibration run writes:

```text
camera-output/
├── captures/<session>/accepted/
└── runs/<session>/
    ├── customer_summary.yaml
    ├── calibration.yaml
    ├── comparison_view.png
    └── calibration_diagnostics/
```

Customer review order:

1. `customer_summary.yaml`: verdict and five key quality values.
2. `comparison_view.png`: distorted and undistorted images must both be real.
3. `calibration.yaml`: final camera matrix and distortion coefficients.

When `verdict` is not `accepted`, keep the diagnostics directory and recollect
images according to `next_action`; do not release the intrinsic parameters.

The release gate requires adequate sample/grid/outer-quadrant coverage, a
consistent native image size, global corner-weighted reprojection RMS no
greater than `1.0 px`, and the 95th percentile of per-view RMS no greater than
`1.5 px`. The solver-reported RMS must agree with the recomputed residual RMS
within `max(0.05 px, 10%)`, and the distortion projection must remain valid
over the image field. These are review gates, not a substitute for checking
capture quality and downstream camera performance.

## Developer build

Container maintainers build the image from the repository root:

```bash
docker build -f docker/camera/Dockerfile \
  -t whl-cal-camera:<version> .
```

Detailed metrics remain documented in
[`calibration_metric_layers.md`](calibration_metric_layers.md).
