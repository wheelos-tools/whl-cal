---
audience: dev
stability: experimental
last_tested: 2026-05-11
---

# sensor timing and clock-source context

## Scope

This note records the current timing-chain analysis for:

-- bag:
  - `/mnt/synology/REDACTED/2026-05-07-REDACTED_USER/bag/20260508032341.record.00000`
- LiDAR stack:
  - `modules/drivers/lidar/vanjeelidar`
- GNSS stack:
  - `modules/drivers/gnss/parser/huace`
- localization stack:
  - `modules/localization/msf`

The durable question is not just "how many milliseconds apart are the topics"
but **which clock each topic is using**:

1. device / sensor measurement time
2. converted UNIX time
3. host publish time
4. record write time

## Measured bag-level findings

Generated artifact:

-- `outputs/analysis/REDACTED_USER_timing_summary.yaml`

### 1. Vanjee LiDAR time chain

On this bag, each raw Vanjee point cloud behaves like:

1. point-level timestamps span about one scan period
2. `measurement_time` matches the **last point timestamp**
3. `header.timestamp_sec` is written a few milliseconds later
4. the record write timestamp follows a little later again

Observed medians:

| topic | scan span | `header - measurement` | `record - header` | `record - measurement` |
| --- | --- | --- | --- | --- |
| `left_front` | `99.984 ms` | `5.039 ms` | `2.097 ms` | `7.143 ms` |
| `right_front` | `99.984 ms` | `5.057 ms` | `2.985 ms` | `8.041 ms` |
| `right_back` | `99.984 ms` | `5.049 ms` | `2.575 ms` | `7.622 ms` |
| `left_back` | `99.984 ms` | `5.037 ms` | `2.401 ms` | `7.443 ms` |

Interpretation:

- the point clouds are effectively **10 Hz scans**
- `measurement_time` is the usable scan time
- `header.timestamp_sec` is **not** the raw LiDAR device time here; it is a
  later host-side publish timestamp

### 2. Inter-LiDAR skew pattern on the bag

Using LiDAR `measurement_time` as the time axis:

| pair | nearest delta median |
| --- | --- |
| `LF <-> RB` | `2.764 ms` |
| `LF <-> RF` | `35.985 ms` |
| `LF <-> LB` | `34.296 ms` |

This confirms the same pattern already seen in calibration work:

- one opposite sensor pair is almost aligned
- the other two are delayed by about `34-36 ms`
- a four-way static snapshot therefore needs a tolerance around `40 ms`, not
  `10-20 ms`

### 3. LiDAR vs GNSS / localization nearest deltas

Using each topic’s most meaningful available time field:

- LiDAR: `measurement_time`
- GNSS odometry / corrected_imu: `header.timestamp_sec` (already converted to
  UNIX from INS measurement time)
- GNSS best_pose / heading: `measurement_time` converted from GPS epoch to UNIX
- localization pose: `measurement_time`

Nearest-neighbor medians from each LiDAR to reference topics:

| source | `gnss.odometry` | `gnss.corrected_imu` | `gnss.best_pose` | `gnss.heading` | `localization.pose` | `localization.slam.pose` |
| --- | --- | --- | --- | --- | --- | --- |
| `LF` | `4.521 ms` | `4.521 ms` | `4.521 ms` | `4.521 ms` | `2.982 ms` | `4.521 ms` |
| `RF` | `0.674 ms` | `0.674 ms` | `0.674 ms` | `0.674 ms` | `3.284 ms` | `0.674 ms` |
| `RB` | `2.604 ms` | `2.604 ms` | `2.604 ms` | `2.604 ms` | `0.381 ms` | `2.604 ms` |
| `LB` | `1.122 ms` | `1.122 ms` | `1.122 ms` | `1.122 ms` | `1.666 ms` | `1.122 ms` |

Interpretation:

- the bag is **not globally desynchronized**
- the LiDARs are close to the GNSS / localization measurement timeline
- the dominant timing problem is **inter-LiDAR phase offset**, not GNSS/localization drift

### 4. GNSS / localization publish-chain delays

Observed medians:

| topic | `record - header` | `header - measurement` | interpretation |
| --- | --- | --- | --- |
| `gnss.best_pose` | `0.127 ms` | `2.183 ms` | measurement is sensor time; header is later host publish time |
| `gnss.heading` | `0.126 ms` | `2.265 ms` | same pattern as best_pose |
| `gnss.odometry` | `2.369 ms` | n/a | header already carries converted measurement time |
| `gnss.corrected_imu` | `2.332 ms` | n/a | header already carries converted measurement time |
| `localization.pose` | `0.350 ms` | `0.000 ms` | header and measurement time are identical on this bag |
| `localization.slam.pose` | `0.200 ms` | `2.512 ms` | publish time is later than pose measurement time |
| `localization.msf_status` | `0.260 ms` | `0.316 ms` median, long tail to `10 ms+` | status topic is less timing-stable than pose |

## Code-path findings

### 1. Vanjee LiDAR current time source

The checked-in Vanjee driver path currently writes point-cloud header time with
host/system time:

- `modules/drivers/lidar/common/lidar_component_base_impl.h`
  - `point_cloud->mutable_header()->set_timestamp_sec(cyber::Time().Now().ToSecond());`

At the same time, the Vanjee decoder config exposes:

- `use_lidar_clock`
- `ts_first_point`
- `use_offset_timestamp`

and the checked-in default online config is:

- `modules/drivers/lidar/vanjeelidar/conf/vanjeelidar.pb.txt`
  - `use_lidar_clock: false`

So the current checked-in Vanjee path should be understood as:

1. **measurement_time / per-point timestamps** from the decoder path
2. **header.timestamp_sec** from host publish time

This matches the bag statistics above: the header is about `5 ms` later than the
measurement time.

### 2. Huace GNSS current time source

The Huace parser reads:

- `gps_week`
- `seconds_in_gps_week`

and sets sensor measurement time in **GPS epoch seconds**:

- `modules/drivers/gnss/parser/huace/huace_parser.cc`
  - `bestpos->set_measurement_time(...)`
  - `imu->set_measurement_time(...)`
  - `heading->set_measurement_time(...)`
  - `ins->mutable_header()->set_timestamp_sec(gps_time_sec);`
  - `ins->set_measurement_time(gps_time_sec);`

Then `DataParser` republishes those messages differently:

1. `best_pose`
   - `common::util::FillHeader("gnss", bestpos_ptr.get());`
   - result: **header becomes host publish time**
2. `heading`
   - `common::util::FillHeader("gnss", heading_ptr.get());`
   - result: **header becomes host publish time**
3. raw `gnss/imu`
   - a new IMU message is created and `FillHeader("gnss", ...)` is called
   - measurement time is **not preserved**
4. `gnss/odometry`
   - `gps->mutable_header()->set_timestamp_sec(GpsToUnixSeconds(ins->measurement_time()));`
   - result: header is **converted UNIX measurement time**
5. `gnss/corrected_imu`
   - `imu->mutable_header()->set_timestamp_sec(GpsToUnixSeconds(ins->measurement_time()));`
   - result: header is **converted UNIX measurement time**

So the current GNSS stack is internally mixed:

- some topics expose **sensor measurement time**
- some expose **host publish time**
- some convert sensor time into **UNIX header time**

### 3. MSF localization current time source

The MSF fusion path writes:

- `localization->set_measurement_time(ins_pva_.time);`
- `headerpb_loc->set_timestamp_sec(apollo::cyber::Clock::NowInSeconds());`

in:

- `modules/localization/msf/local_integ/localization_integ_process.cc`

So the fusion output should conceptually be:

- `measurement_time`: fused pose time
- `header.timestamp_sec`: system publish time

But the bag shows:

- `/apollo/localization/pose`
  - `header.timestamp_sec == measurement_time`

This implies the runtime branch used for the bag is effectively publishing
fusion pose on the measurement timeline, at least for `/apollo/localization/pose`.

The lidar-localization branch is explicitly measurement-timed:

- `modules/localization/msf/local_integ/localization_lidar_process.cc`
  - `lidar_local_msg->set_measurement_time(pre_location_time_);`
  - `headerpb->set_timestamp_sec(pre_location_time_);`

The MSF broadcaster also has an explicit switch:

- `modules/localization/msf/msf_localization_component.cc`
  - TF can use either `localization.measurement_time()` or
    `localization.header().timestamp_sec()`

### 4. Checked-in config mismatch to watch

The checked-in GNSS config still says:

- `modules/drivers/gnss/conf/gnss_conf.pb.txt`
  - `format: NOVATEL_BINARY`

while the parser factory already supports:

- `HUACE_TEXT`
- `FORSENSE_TEXT`
- `FORSENSE_BINARY`

So for a Huace-equipped vehicle, the checked-in config should not be assumed to
match the real runtime hardware without verification.

## Current system state

The current stack should be understood as:

1. **Vanjee LiDAR**
   - measurement time and per-point timestamps are meaningful
   - header time is host publish time
   - checked-in default config uses `use_lidar_clock: false`
2. **Huace GNSS**
   - parser time root is GPS epoch from the device
   - downstream Apollo topics are not fully normalized to one convention
3. **Localization**
   - some outputs align header with measurement time
   - some status / side topics still carry additional publish delay

This means the system is **usable**, but it is not yet a single explicit
clock-provenance architecture.

## Industry reference direction

The current best-practice direction is:

1. GNSS-disciplined absolute time root
2. PTP/gPTP or PPS + disciplined PHC/system clock
3. LiDAR measurement time from sensor/PTP clock
4. per-point timestamps preserved for deskew
5. host publish time kept as metadata, not confused with measurement time
6. explicit timestamp provenance per topic

For static multi-LiDAR calibration, the practical conclusion is:

- if inter-LiDAR phase offsets are around `34-36 ms`, use a snapshot threshold
  around `40 ms`
- if you want to tighten below that, first fix the clock discipline, not just
  the calibration code

## Recommended actions

### Immediate

1. treat **LiDAR `measurement_time`** as the calibration / localization timebase,
   not LiDAR `header.timestamp_sec`
2. for GNSS comparison, prefer:
   - `/apollo/sensor/gnss/odometry`
   - `/apollo/sensor/gnss/corrected_imu`
   because they are already converted to UNIX measurement time in the header
3. do **not** use `/apollo/sensor/gnss/imu` header time as a trusted sensor-time
   reference; the current publish path overwrites it with host time
4. treat `best_pose` / `heading` header time as publish time and
   `measurement_time` as the real sensor time

### System cleanup

1. unify GNSS topic timestamp policy:
   - either all key GNSS outputs expose measurement time in header (UNIX)
   - or all expose measurement time separately and never overload header
2. expose clock provenance in diagnostics:
   - `sensor_time`
   - `publish_time`
   - `record_time`
   - `clock_source`
3. verify the real runtime GNSS config for Huace and update the checked-in
   `gnss_conf.pb.txt` accordingly

### Production-grade target

1. use a GNSS-disciplined time root
2. move Vanjee to a clearly verified clock mode:
   - `use_lidar_clock=true` only if the LiDAR clock is actually disciplined to
     the same time root
   - otherwise keep the current mode, but rely on `measurement_time` rather than
     header time
3. add a timing-health report before calibration:
   - inter-LiDAR nearest delta
   - LiDAR-to-GNSS nearest delta
   - LiDAR `header - measurement`
   - GNSS `header - measurement`
   - localization `header - measurement`
4. gate promotion of tighter synchronization thresholds on verified timing
   health, not on one good bag alone
