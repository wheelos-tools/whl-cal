---
audience: user
stability: stable
P26-04-27
---


# LiDAR2IMU 用户说明与操作文档

本文档面向直接使用 `whl-cal` 进行 `lidar2imu` 标定的用户，目标是用**一个文档**说明完整操作流程：准备数据、检查 bag、执行标定、查看结果、处理常见失败。

配套文档：

- Apollo 侧准备与录包： [docs/apollo_data_collection.md](apollo_data_collection.md)
- 通用指标 / 可视化复核： [docs/calibration_review_guide.md](calibration_review_guide.md)
- 方法设计 / SOTA / 参考资料：
  [docs/calibration_methodology.md](calibration_methodology.md)

## 1. 标定目标

`lidar2imu` 用于估计：

- `x, y, z`
- `yaw, roll, pitch`

当前流程分成两层：

1. **数据转换层**：从 Apollo `.record` 中提取 `ground_samples` 和 `motion_samples`
2. **求解与评估层**：执行 staged solver，并输出稳定的 `metrics.yaml` 与 `diagnostics/`

这意味着你可以先把原始数据转成标准样本，再单独重跑求解；也可以一步完成。

## 2. 适用数据

推荐输入数据具备以下内容：

- LiDAR 点云 topic，例如 `/apollo/sensor/lslidar_main/PointCloud2`
- 位姿 topic，例如 `/apollo/localization/pose`
- IMU topic，例如 `/apollo/sensor/gnss/imu`
- TF，至少能提供 `lidar -> imu` 与 `localization -> imu` 的关系

更适合做完整 6DoF 标定的数据特征：

- 地面清晰，能稳定提取地面平面
- 车辆有足够运动激励
- 最好同时包含左转和右转

如果只有单侧转向或 yaw 激励很弱，当前流程通常仍可给出较稳定的 `z / roll / pitch`，但 `x / y / yaw` 只能作为弱结论。

## 3. 环境准备

```bash
cd /home/wfh/01code/whl-cal
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 4. 先检查 bag 是否可用

先看 record 里有哪些 topic：

```bash
lidar2lidar-topics /path/to/record_dir_or_record_file
```

至少确认：

- LiDAR 点云 topic 存在
- `/apollo/localization/pose` 存在
- `/apollo/sensor/gnss/imu` 或可替代 IMU topic 存在
- `/tf_static` 和 `/tf` 存在

如果 bag 里没有 `lidar -> imu` 静态外参，需要额外提供：

```bash
--initial-transform path/to/imu_lidar_extrinsics.yaml
```

如果 bag 自带 `lidar -> parent` TF，但你要**按新的几何重新导出样本**，额外提供：

```bash
--extraction-transform path/to/imu_lidar_extrinsics.yaml
```

这会保留 bag 内 TF 作为 `reference_transform`，同时把你提供的外参用于地面提取、局部地图构造和运动因子重建。`--auto-reextract-if-needed` 的第二轮会自动走这条路径。

如果只是探索数据可用性，也可以临时使用：

```bash
--identity-initial-transform
```

但这种模式只适合诊断，不适合直接接受最终外参。

## 5. 最常用的一步跑通方式

这是最推荐的入口：从 Apollo record 直接转换并标定。

```bash
lidar2imu-convert-record \
  --record-path /path/to/record_dir_or_record_file \
  --output-dir outputs/lidar2imu/run01 \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --motion-frame-stride 5 \
  --min-registration-fitness 0.55 \
  --planar-motion-policy auto \
  --calibrate
```

默认 topic 映射是：

- LiDAR: `/apollo/sensor/lslidar_main/PointCloud2`
- Pose: `/apollo/localization/pose`
- IMU: `/apollo/sensor/gnss/imu`

如果你的 bag 使用别的 topic，显式指定：

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --output-dir outputs/lidar2imu/run01 \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/imu \
  --initial-transform path/to/imu_lidar_extrinsics.yaml \
  --min-registration-fitness 0.55 \
  --planar-motion-policy auto \
  --calibrate
```

## 6. 两阶段运行方式

如果你想先固定数据层，再反复调 solver，建议拆成两步。

### 6.1 只做 record 转换

```bash
lidar2imu-convert-record \
  --record-path /path/to/record \
  --output-dir outputs/lidar2imu/record_conversion \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --motion-frame-stride 5 \
  --min-registration-fitness 0.55
```

这一步会生成：

- `standardized_samples.yaml`
- `conversion_diagnostics.yaml`

### 6.2 基于标准样本单独求解

```bash
lidar2imu-calibrate \
  --input outputs/lidar2imu/record_conversion/standardized_samples.yaml \
  --planar-motion-policy auto \
  --output-dir outputs/lidar2imu/calibration_only
```

## 7. 关键参数怎么选

### `--min-registration-fitness`

用于过滤 LiDAR-LiDAR motion pair 的配准质量。

- 推荐起点：`0.55`
- 如果 motion pair 很少、经常被拒绝，可以试 `0.45`
- 如果数据质量很好、想更严格，可以试 `0.70`

### `--motion-frame-stride`

控制 motion sample 的候选跨度。

- 推荐起点：`5`
- 如果长跨度配准不稳定，可降到 `1` 或 `2`
- 如果 bag 很长且局部运动太小，可尝试更大值

### `--min-motion-rotation-deg`

控制 motion stage 对角激励的最低要求。

- 默认思路：`1.0`
- 如果提示有效角激励样本不够，可以降到 `0.5` 或 `0.2`
- 降低该值后，要更谨慎看 `x / y / yaw`

### `--planar-motion-policy`

- `auto`：推荐默认值；平面可观性弱时自动冻结 `x / y / yaw`
- `free`：完全自由求解 6DoF
- `freeze_xyyaw`：手动锁定 `x / y / yaw`

在真实车载数据上，若存在单侧转向或 yaw 可观性弱，优先使用：

```bash
--planar-motion-policy auto
```

### `--gravity-source`

- `pose`：当前默认，通常更稳定
- `imu`：只有在原始 IMU 重力方向可靠时才建议尝试

如果 `gravity-source imu` 导致几乎没有 ground sample，被判定为 `plane_not_ground_like`，应回退到：

```bash
--gravity-source pose
```

## 8. 输出文件说明

执行成功后，重点看这些文件：

- `standardized_samples.yaml`：标准化后的 ground / motion 样本
- `conversion_diagnostics.yaml`：数据提取、窗口选择、motion gate 细节
- `calibration/calibrated_tf.yaml`：汇总后的最终外参
- `calibration/metrics.yaml`：主要评估结果
- `calibration/diagnostics/`：算法与观测性细节

稳定产物路径通常是：

```text
outputs/lidar2imu/run01/
├── standardized_samples.yaml
├── conversion_diagnostics.yaml
└── calibration/
    ├── calibrated_tf.yaml
    ├── metrics.yaml
    ├── calibrated/<parent>_<child>_extrinsics.yaml
    └── diagnostics/
```

## 9. 结果先看什么

打开 `calibration/metrics.yaml`，优先看：

- `coarse_metrics.ground_normal_angle_p95_deg`
- `coarse_metrics.ground_height_residual_p95_m`
- `coarse_metrics.motion_rotation_residual_p95_deg`
- `coarse_metrics.motion_translation_residual_p95_m`
- `coarse_metrics.motion_registration_fitness_p05`
- `coarse_metrics.left_turn_count`
- `coarse_metrics.right_turn_count`
- `coarse_metrics.turn_balance_ratio`
- `vehicle_motion_assessment.recommendation`
- `vehicle_motion_assessment.applied_solver_planar_motion_policy`

推荐的判读方式：

1. **地面指标好、motion 指标弱**：优先相信 `z / roll / pitch`
2. **左右转不平衡**：把 `x / y / yaw` 视为弱可观结果
3. **recommendation = `z_roll_pitch_priority`**：表示垂向与姿态更可信，平面分量偏弱
4. **solver policy = `freeze_xyyaw`**：说明系统已判断当前 bag 平面可观性不足

## 10. 常见失败与处理

### 情况 1：`Need at least 3 motion samples with angular excitation`

说明用于 yaw stage 的有效角激励样本不够。

处理顺序：

1. 降低 `--min-motion-rotation-deg`，例如从 `1.0` 调到 `0.5` 或 `0.2`
2. 降低 `--motion-frame-stride`，例如改成 `1` 或 `2`
3. 降低 `--min-registration-fitness`，例如从 `0.55` 调到 `0.45`
4. 如果仍然不足，说明这包本身 yaw 激励弱，应只接受 `z / roll / pitch`

### 情况 2：`Need at least 3 ground samples`

说明 ground stage 没有得到足够有效地面样本。

优先检查：

1. LiDAR 数据是否包含稳定路面
2. 车辆是否处于坡道、路肩、复杂非平地场景
3. 是否使用了 `--gravity-source imu` 且 IMU 重力方向不可靠

常见修复：

```bash
--gravity-source pose
```

### 情况 3：motion 样本很多，但 `x / y / yaw` 漂移大

这通常不是“求解收敛更好”，而是 bag 的平面可观性不足。

优先看：

- `left_turn_count`
- `right_turn_count`
- `turn_balance_ratio`
- `vehicle_motion_assessment`

若是弱平面 bag，保留：

```bash
--planar-motion-policy auto
```

不要只因为自由解的残差更小，就直接接受漂移更大的 6DoF 结果。

## 11. 批量调参方法

当单次运行失败，或者你想系统比较多个参数组合时，用：

```bash
lidar2imu-tune-record \
  --record-path /path/to/record \
  --output-dir outputs/lidar2imu/tuning \
  --gravity-sources pose,imu \
  --motion-frame-strides 1,2,5 \
  --min-registration-fitness-values 0.45,0.55 \
  --min-motion-rotation-values 0.2,0.5,1.0 \
  --planar-motion-policies auto,free \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --calibrate
```

重点查看：

- `tuning_summary.yaml`
- `tuning_summary.csv`

这个工具会综合比较：

1. 是否成功
2. recommendation 是否更可信
3. 相对初值漂移是否过大
4. warning 数量
5. residual 与 registration 质量

## 12. 一个推荐的日常使用流程

对一个新 bag，建议按下面顺序操作：

1. `lidar2lidar-topics` 看 topic 是否齐全
2. 用 `lidar2imu-convert-record --calibrate` 跑默认配置
3. 查看 `conversion_diagnostics.yaml` 与 `calibration/metrics.yaml`
4. 如果 motion 样本不足，调 `stride / fitness / min_motion_rotation_deg`
5. 如果 ground 不稳定，优先用 `pose` gravity
6. 如果 turn balance 明显失衡，只接受 `z / roll / pitch`
7. 只有当多包结果稳定、且左右转激励充分时，再把 `x / y / yaw` 当作最终接受结论

## 13. 0421 实测 bag 示例

对以下 bag：

```text
/home/wfh/01code/apollo-lite/data/bag/0421/20260420083748.record.00000
```

已验证可用的一组参数是：

```bash
lidar2imu-convert-record \
  --record-path /home/wfh/01code/apollo-lite/data/bag/0421/20260420083748.record.00000 \
  --output-dir outputs/lidar2imu/0421_example \
  --max-ground-samples 12 \
  --max-motion-samples 8 \
  --motion-frame-stride 5 \
  --min-registration-fitness 0.55 \
  --min-motion-rotation-deg 0.2 \
  --planar-motion-policy auto \
  --calibrate
```

该 bag 的结论是：

- `z / roll / pitch` 可用
- `x / y / yaw` 仍然偏弱
- `gravity-source imu` 不适合这个 bag
- `pose-derived gravity` 更稳

如果你只是想先复现实测结果，直接从这一组参数开始即可。
