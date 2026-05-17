---
audience: dev
stability: experimental
last_tested: 2026-05-17
---

# 2026-05-06 calibration dataset meta

Dataset path:

- `/mnt/synology/REDACTED/2026-5-6-标定/`

Trigger metadata from `record-msg.json`:

- `event_time`: `1778037168444`
- local time: `2026-05-06T11:12:48.444+08:00`
- `desc`: `手动触发`

## Files

| file | size bytes | mtime +08:00 | note |
| --- | ---: | --- | --- |
| `20260506031248.record.00000` | `2152061181` | `2026-05-06 11:22:25` | Apollo record shard |
| `20260506031248.record.00001` | `2151787744` | `2026-05-06 11:21:27` | Apollo record shard |
| `20260506031248.record.00002` | `2151807473` | `2026-05-06 11:23:54` | Apollo record shard |
| `20260506031248.record.00003` | `648897770` | `2026-05-06 11:19:15` | Apollo record shard |
| `2026-05-06-11-12-48_s.zip` | `2520652540` | `2026-05-06 12:32:01.846` | archive copy containing the four records plus `record-msg.json` |
| `record-msg.json` | `59` | `2026-05-06 11:22:25` | trigger metadata |

The zip archive does **not** contain extra sensor files. It contains:

- `20260506031248.record.00000`
- `20260506031248.record.00001`
- `20260506031248.record.00002`
- `20260506031248.record.00003`
- `record-msg.json`

## Topic inventory

Command:

```bash
lidar2lidar-topics /mnt/synology/REDACTED/2026-5-6-标定/
```

Observed topics:

| topic | count | role |
| --- | ---: | --- |
| `/apollo/sensor/gnss/raw_data` | `100252` | raw GNSS stream |
| `/tf` | `31806` | dynamic TF |
| `/apollo/sensor/gnss/corrected_imu` | `15913` | GNSS/INS corrected IMU, preferred timing reference |
| `/apollo/sensor/gnss/ins_stat` | `15913` | GNSS status |
| `/apollo/sensor/gnss/best_pose` | `15912` | GNSS best pose |
| `/apollo/sensor/gnss/imu` | `15912` | raw GNSS IMU topic; do not use as strict sensor-time reference |
| `/apollo/sensor/gnss/odometry` | `15912` | GNSS odometry, preferred timing reference |
| `/apollo/sensor/gnss/heading` | `15912` | GNSS heading |
| `/apollo/localization/slam/pose` | `15912` | SLAM/localization pose |
| `/apollo/localization/msf_status` | `15912` | MSF status |
| `/apollo/localization/pose` | `15912` | localization pose |
| `/apollo/localization/fusion/status` | `15894` | localization status |
| `/apollo/sensor/vanjeelidar/right_back/PointCloud2` | `1593` | raw Vanjee LiDAR |
| `/apollo/sensor/vanjeelidar/left_front/PointCloud2` | `1592` | raw Vanjee LiDAR |
| `/apollo/localization/endpoint/status` | `1592` | localization endpoint status |
| `/apollo/sensor/lidar/fusion/PointCloud2` | `1592` | fused LiDAR output, not a raw independent sensor |
| `/apollo/hmi/status` | `282` | HMI status |
| `/apollo/monitor` | `250` | monitor |
| `/apollo/sensor/gnss/rtcm_data` | `169` | RTK data |
| `/apollo/sensor/gnss/rtk_obs` | `159` | RTK observation |
| `/apollo/sensor/gnss/ins_status` | `158` | low-rate GNSS INS status |
| `/apollo/monitor/system_status` | `122` | system status |
| `/tf_static` | `1` | static TF |

## Calibration implications

### lidar2lidar

This dataset is **not** a full four-corner raw-rig dataset. The record exposes:

- raw `left_front`
- raw `right_back`
- fused `/apollo/sensor/lidar/fusion/PointCloud2`

It does **not** expose raw `right_front` or raw `left_back` topics in the record
inventory. Therefore:

1. Do not use this bag to claim four-LiDAR rectangle-loop calibration quality.
2. Use it for:
   - left-front to right-back raw pair diagnostics
   - raw-to-fusion sanity checks
   - timing / data-quality / visualization pipeline validation
3. Full raw4 loop-closure evaluation still requires a bag with all four raw
   Vanjee topics.

### lidar2imu

This dataset is suitable for lidar2imu record conversion review because it has:

- localization pose at about 100 Hz
- GNSS/INS topics at about 100 Hz
- raw LiDAR at about 10 Hz
- `/tf_static` plus `/tf`

For timestamp interpretation, keep using the policy from
`timing_sync_context.md`:

- Vanjee LiDAR: use `measurement_time`
- GNSS odometry / corrected_imu: use `header.timestamp_sec`
- GNSS best_pose / heading: use `measurement_time`
- GNSS IMU header is not a strict sensor-time reference

## Required pre-calibration checks

Before accepting calibration output from this dataset, every pipeline should
report:

1. sensor/topic availability
2. time-span and message-rate summary
3. timestamp provenance and sync jitter
4. ground / plane support for lidar2imu
5. motion excitation, turn balance, and registration quality for lidar2imu
6. shared-scene overlap, scene sufficiency, repeatability, and visual geometry
   for lidar2lidar
7. a final `final_acceptance` decision in `metrics.yaml`
8. `diagnostics/acceptance_report.yaml`
9. `diagnostics/status_summary.csv`

The last three items are the shared review surface introduced so that
`lidar2lidar` and `lidar2imu` follow the same release-review pattern.
