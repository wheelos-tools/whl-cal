---
audience: dev
stability: experimental
last_tested: 2026-05-11
---

# 每个 Topic 的时间出处与当前包实测值（便于后续分析）

生成时间（UTC+8）: 2026-05-11T19:27:28.448+08:00

说明：本表基于 `outputs/analysis/REDACTED_USER_timing_summary.yaml` 的统计和仓内源码追踪（Vanjee driver / Huace parser / MSF localization）。建议后续分析严格按“应看字段”执行，不要混用 header 与 measurement。

| Topic | 时间出处（代码） | 后续分析应看 | 本包实测（中位数/说明） | 风险/备注 |
| --- | --- | --- | --- | --- |
| /apollo/sensor/vanjeelidar/left_front/PointCloud2 | lidar_component_base_impl.h: header 写入 cyber::Time().Now(); pointcloud decoder 提供 measurement_time & per-point timestamps | measurement_time | header - measurement ≈ 5.039 ms；record - measurement ≈ 7.143 ms；scan span ≈ 99.984 ms | header 不是设备采样时刻；用 header 会引入 ~5 ms 偏差 |
| /apollo/sensor/vanjeelidar/right_front/PointCloud2 | 同上 | measurement_time | header - measurement ≈ 5.057 ms；record - measurement ≈ 8.041 ms；scan span ≈ 99.984 ms | 同上 |
| /apollo/sensor/vanjeelidar/right_back/PointCloud2 | 同上 | measurement_time | header - measurement ≈ 5.049 ms；record - measurement ≈ 7.622 ms；scan span ≈ 99.984 ms | 同上 |
| /apollo/sensor/vanjeelidar/left_back/PointCloud2 | 同上 | measurement_time | header - measurement ≈ 5.037 ms；record - measurement ≈ 7.443 ms；scan span ≈ 99.984 ms | 同上 |
| /apollo/sensor/gnss/odometry | data_parser.cc: 用 GpsToUnixSeconds(ins->measurement_time()) 写入 header.timestamp_sec | header.timestamp_sec (UNIX measurement) | record - header ≈ 2.369 ms | 适合对齐 LiDAR 的参考时间 |
| /apollo/sensor/gnss/corrected_imu | data_parser.cc: 同上，header 写入转换后的 UNIX measurement | header.timestamp_sec (UNIX measurement) | record - header ≈ 2.332 ms | 适合对齐 LiDAR 的参考时间 |
| /apollo/sensor/gnss/best_pose | huace_parser.cc: bestpos.measurement_time = GPS epoch；data_parser 使用 FillHeader 覆盖 header | measurement_time（需转 UNIX） | header - measurement ≈ 2.183 ms；record - measurement ≈ 2.307 ms | header 为 publish time，分析时应用 measurement_time |
| /apollo/sensor/gnss/heading | 同上 | measurement_time（需转 UNIX） | header - measurement ≈ 2.265 ms；record - measurement ≈ 2.385 ms | header 为 publish time |
| /apollo/sensor/gnss/imu | data_parser::PublishImu 新建消息并 FillHeader，measurement_time 未保留 | 不建议当主时间基准（只能用 header） | record - header ≈ 0.135 ms | measurement_time 被丢失，不能作为 sensor-time 参考 |
| /apollo/sensor/gnss/ins_stat | FillHeader 后发布 | header | record - header ≈ 0.116 ms | 状态消息，低频，非精同步依据 |
| /apollo/localization/pose | localization_integ_process.cc: 理论上 measurement_time=ins_pva_.time，header 写 Clock::Now()；但本包 header==measurement | measurement_time（本包等于 header） | header - measurement = 0；record - measurement ≈ 0.350 ms | 本包里表现干净，可用作参考，但勿假设所有 localization topic 都一致 |
| /apollo/localization/slam/pose | localization_lidar_process.cc: lidar-local 化 explicit 使用 pre_location_time_ | measurement_time | header - measurement ≈ 2.512 ms；record - measurement ≈ 2.720 ms | header 晚约 2.5 ms |
| /apollo/localization/msf_status | msf status 有 measurement/header，但更抖 | 不建议当高精同步基准 | header - measurement median ≈ 0.316 ms，但 p95 ≈ 10.35 ms（长尾） | 长尾明显，稳定性差，谨慎使用 |


## 总结与建议（快速版）

- 标定 / 配准均以 **LiDAR measurement_time** 为准。
- 对齐时优先使用 `/apollo/sensor/gnss/odometry` 与 `/apollo/sensor/gnss/corrected_imu`（它们的 header 已是 UNIX 测量时刻）。
- 不要把 `/apollo/sensor/gnss/imu` 的 header 当作设备测时基准——测时在发布链路里丢失。
- 对于四路静态标定，当前 bag 的 **inter-LiDAR 相位偏差** 是主问题：LF↔RB≈2.8ms，LF↔RF≈36.0ms，LF↔LB≈34.3ms，建议 snapshot 容限设为 ~40ms。
