---
audience: user
stability: stable
P26-05-25
---

# Apollo data collection guide for calibration

This guide explains how to prepare Apollo, start the required hardware modules,
record a calibration bag with `cyber_recorder`, and verify that the bag contains
the information needed by each `whl-cal` pipeline.

## 1. Before going on vehicle

Recommended baseline:

1. Apollo host / container can start normally.
2. Sensor clocks are already synchronized by your vehicle integration flow.
3. Static transforms are loaded and `transform` can publish `/tf_static`.
4. Storage is sufficient for full-rate recording.
5. You know which LiDAR vendor launch matches the vehicle.

For Apollo host/container startup, refer to the quick-start flow in
`/home/humble/01code/apollo-base/README.md`.

## 2. Start the Apollo modules needed for calibration

The exact LiDAR launch depends on the installed sensor vendor, but the common
calibration stack is:

```bash
cd /home/humble/01code/apollo-base

# static TF
bash scripts/transform.sh start

# GNSS / IMU side
bash scripts/gps.sh start

# localization output (/apollo/localization/pose)
bash scripts/localization.sh start

# cameras
bash scripts/camera.sh start
# or, when your deployment uses camera + video together:
bash scripts/camera_and_video.sh start
```

LiDAR examples from the same Apollo tree:

```bash
# Velodyne wrapper
bash scripts/velodyne.sh start

# Vendor-specific direct launches
cyber_launch start /apollo/modules/drivers/lidar/lslidar/launch/lslidar.launch
cyber_launch start /apollo/modules/drivers/lidar/vanjeelidar/launch/vanjeelidar.launch
cyber_launch start /apollo/modules/drivers/lidar/hesai/launch/hesai.launch
cyber_launch start /apollo/modules/drivers/lidar/livox/launch/livox.launch
cyber_launch start /apollo/modules/drivers/lidar/rslidar/launch/rslidar.launch

# Integrated multi-lidar example (Vanjee + RSLiDAR + fusion + compensator)
cyber_launch start /apollo/modules/drivers/lidar/launch/lidar_with_fusion_and_compensator.launch
```

Use the launch that matches your deployed hardware and topic layout.

## 3. Verify that the channels are alive before recording

Check the published channels first:

```bash
cyber_channel list | grep -E 'camera|PointCloud2|/apollo/localization/pose|/apollo/sensor/gnss|/tf'
```

For the key channels, also verify the live rate:

```bash
cyber_channel hz /apollo/localization/pose
cyber_channel hz /apollo/sensor/gnss/imu
cyber_channel hz /apollo/sensor/your_lidar/PointCloud2
cyber_channel hz /apollo/sensor/camera/your_camera/image
```

If a calibration bag is going to be used for `lidar2imu`, do not skip the rate
check on `pose` and IMU channels.

## 4. Record with `cyber_recorder`

Recommended direct command:

```bash
mkdir -p /path/to/calibration_bags/session_01
cd /path/to/calibration_bags/session_01
cyber_recorder record -a -i 60 -m 2048
```

This is the same recorder setup used by Apollo's `scripts/record_bag.sh`.

If you prefer Apollo's wrapper:

```bash
cd /home/humble/01code/apollo-base
bash scripts/record_bag.sh start
```

Recommended recording practice:

1. Start all required modules.
2. Wait 5-10 seconds for topics and TF to stabilize.
3. Start the recorder.
4. Run the planned calibration route or target-capture session.
5. Keep recording for another 5-10 seconds before stopping.

## 5. What each calibration pipeline needs

| Module | Mandatory raw information | Strongly recommended extra information | Practical capture advice |
| --- | --- | --- | --- |
| `camera` | images containing a calibration board | fixed focus, fixed exposure, stable resolution | direct live capture is simplest; Apollo recording is optional |
| `lidar2camera` | camera images, LiDAR point clouds, checkerboard visible in both | `/tf_static`, repeated poses with depth / tilt diversity | collect 15-30 poses; avoid motion blur and partial target occlusion |
| `lidar2lidar` | all relevant raw `PointCloud2` topics | `/tf_static`, `/tf`, approximate initial extrinsics | prefer walls, corners, poles, facades; reduce heavy traffic / pedestrians |
| `lidar2imu` | one LiDAR topic, `/apollo/localization/pose`, IMU, TF | `/apollo/sensor/gnss/best_pose`, `/apollo/sensor/gnss/corrected_imu`, `/apollo/sensor/gnss/heading` | include both left and right turns, acceleration, braking, and flat-road segments |

## 6. Topic checklist by module

### Camera intrinsic

The current tool consumes a live camera or an exported image directory, not a raw
Apollo bag directly. If you still record in Apollo for traceability, keep at
least the camera image topic.

### LiDAR↔Camera

Recommended bag contents:

- camera image topic, for example `/apollo/sensor/camera/front_6mm/image`
- LiDAR `PointCloud2` topic
- `/tf_static`
- optional `/tf`

Current limitation: `lidar2camera-calibrate` still consumes paired `image + .pcd`
files, so after recording you must export synchronized image / point-cloud pairs
with your existing dataset-preparation flow. `whl-cal` does not yet provide a
direct Apollo-record-to-lidar2camera exporter.

### LiDAR-to-LiDAR

Recommended bag contents:

- all LiDAR raw `PointCloud2` topics to be calibrated
- `/tf_static`
- optional `/tf`

For four-LiDAR raw rigs, record all four raw topics even if the vehicle also
publishes fused clouds.

### LiDAR-to-IMU

Recommended bag contents:

- one LiDAR `PointCloud2` topic
- `/apollo/localization/pose`
- IMU topic used by your pipeline
- `/tf_static`
- optional `/tf`

In Apollo deployments, it is usually safer to record all GNSS/IMU-related topics
with `-a` so localization can be audited later.

## 7. Motion / scene design by module

### Camera intrinsic

- Use the same camera mode that will be used in production.
- Cover the center, four corners, and multiple tilts.
- Avoid autofocus or auto-exposure changes during the session.

### LiDAR↔Camera

- Keep the checkerboard fully visible in both modalities.
- Change distance, horizontal position, and tilt between poses.
- Avoid collecting all poses from one depth and one image region.

### LiDAR-to-LiDAR

- Prefer static environments with walls, corners, poles, and facade edges.
- Avoid long periods with only moving traffic or open sky.
- For rig calibration, include straight driving plus turns and curb-side geometry.

### LiDAR-to-IMU

- Include both left and right turns whenever possible.
- Include acceleration and braking, not only constant-speed cruising.
- Keep enough flat-road segments so the ground plane can be extracted reliably.

## 8. Fast post-record checks

Before running any solver:

```bash
RECORD_DIR=/path/to/record_dir
cd /home/humble/01code/whl-cal
source .venv/bin/activate
lidar2lidar-topics "$RECORD_DIR"
```

Then verify:

1. the expected LiDAR topics exist
2. `/apollo/localization/pose` exists for `lidar2imu`
3. the IMU topic exists for `lidar2imu`
4. `/tf_static` exists for any extrinsic workflow
5. image topics exist if the session was collected for `lidar2camera`

## 9. Typical field mistakes to avoid

Do not release a calibration bag when any of these are true:

- the target sensor topic appears only intermittently
- `/tf_static` is missing or obviously incomplete
- localization output is absent for a `lidar2imu` run
- the vehicle only turned in one direction
- LiDAR bags were collected in feature-poor or highly dynamic scenes
- camera target sessions used the wrong resolution / focus / exposure mode

## 10. What to read next

After data collection:

1. run the module quick start from [docs/quickstart_index.md](quickstart_index.md)
2. review outputs with [docs/calibration_review_guide.md](calibration_review_guide.md)
3. read [docs/calibration_methodology.md](calibration_methodology.md) when you
   need the design rationale or method trade-offs
