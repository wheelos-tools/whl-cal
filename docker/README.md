# whl-cal containers

The release surfaces are split by runtime role:

| Image | User workflow | Runtime |
| --- | --- | --- |
| `whl-cal-camera` | Visible live capture or offline image calibration | Python, OpenCV GUI |
| `whl-cal-lidar2imu` | Apollo record/ROS1 bag to native GRIL result | Python adapter, ROS-free native GRIL |

Both images use read-only `/data` input and writable `/output` artifacts.
Customer-facing results are written to `customer_summary.yaml`; detailed
developer evidence remains in diagnostics, manifests, and traces.

## LiDAR-to-IMU

The publish build uses the validated local runtime bundle under
`docker/lidar2imu/runtime/`. This ignored directory is assembled from cached
native binaries and their exact `ldd` dependency closure; the build does not
download or rebuild native dependencies.

Prepare it from the validated cache before building:

```bash
python docker/lidar2imu/prepare_cached_runtime.py \
  --frontend /path/to/cache/gril_native_full_frontend \
  --batch /path/to/cache/gril_native_batch \
  --library-prefix /path/to/cache/runtime-prefix
```

The Dockerfile also copies the pinned Python runtime from the local
`whl-cal-lidar2imu:slim` cache image. A release build can therefore be proven
offline:

```bash
docker build --network=none \
  -f docker/lidar2imu/Dockerfile \
  -t whl-cal-lidar2imu:<version> .
```

```bash
docker build -f docker/lidar2imu/Dockerfile -t whl-cal-lidar2imu:local .

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src=/path/to/records,dst=/data,readonly \
  --mount type=bind,src="$PWD/output",dst=/output \
  whl-cal-lidar2imu:local \
  --input /data \
  --input-type record \
  --output-dir /output
```

The image does not contain ROS. Its fixed entrypoint supplies the native
executable and reviewed Vanjee-16 configuration.

The native GRIL source remains `review-only`. Do not push this image to an
external registry until the physical-accuracy and license/provenance gates in
`third_party/gril_native/README.md` are approved.

## Camera intrinsic

Live capture always requires a visible GUI. On Linux/X11:

```bash
docker build -f docker/camera/Dockerfile -t whl-cal-camera:local .

xhost +si:localuser:"$(id -un)"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --device /dev/video0:/dev/video0 \
  --env DISPLAY \
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
  --mount type=bind,src="$PWD/output",dst=/output \
  whl-cal-camera:local \
  --output-dir /output
```

An offline image dataset may run without a display:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD/images",dst=/data/images,readonly \
  --mount type=bind,src="$PWD/output",dst=/output \
  whl-cal-camera:local \
  --images-dir /data/images \
  --output-dir /output
```

Override the built-in configuration by mounting a file and passing
`--config /config/camera.yaml`.
