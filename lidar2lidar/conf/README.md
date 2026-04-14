# LiDAR Extrinsics Conf

This directory stores standardized fallback extrinsics used by the automatic
pipeline when `/tf_static` or `/tf` is incomplete.

Naming rule:

- `<parent_frame>_<child_frame>_extrinsics.yaml`

Schema:

```yaml
header:
  stamp:
    secs: 0
    nsecs: 0
  seq: 0
  frame_id: lslidar_main
transform:
  translation:
    x: 0.0
    y: 0.0
    z: 0.0
  rotation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
child_frame_id: lslidar_left
```

The matrix encoded here maps points from `child_frame_id` into
`header.frame_id`.

You can bootstrap this directory from a record file with:

```bash
lidar2lidar-auto \
  --record-path your/data/bag/ \
  --conf-dir lidar2lidar/conf \
  --bootstrap-conf \
  --output-dir outputs/lidar2lidar/bootstrap_only
```
