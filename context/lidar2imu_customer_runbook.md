---
audience: user
stability: experimental
last_tested: 2026-05-26
---

# LiDAR2IMU 客户交付版自动标定说明

## 需求

1. 工业级的、全自动的 LiDAR-to-IMU 标定解决方案
2. 精确计算 LiDAR 相对 IMU 的 6DoF 外参：
   - 平移：`x, y, z`
   - 旋转：`yaw, roll, pitch`

## 项目地址

<https://github.com/wheelos-tools/whl-cal>

## 这套方案的特点

- **不需要标定板**
- **不需要人工逐帧挑数据**
- 自动完成：
  - bag 可用性检查
  - 地面样本提取
  - 运动样本构造
  - 低质量样本剔除
  - 外参优化
  - 指标评估
  - HTML / SVG / PLY 可视化输出

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/wfh/01code/whl-cal
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 准备标定数据

`lidar2imu` **不需要棋盘格标定板**，需要的是一段质量足够好的车载录包。

至少需要以下内容：

- 1 路 LiDAR 点云 topic
- 1 路位姿 topic，通常是 `/apollo/localization/pose`
- 1 路 IMU topic，通常是 `/apollo/sensor/gnss/imu` 或 `/apollo/sensor/gnss/corrected_imu`
- `TF`，至少要能解析出 `lidar -> imu` 和 `localization -> imu`

推荐采集方式：

- 尽量包含**左转 + 右转**
- 尽量有**加速 / 减速**
- 尽量有**平地直行段**
- 推荐接近**“8 字形”** 或者至少是**双向转弯 + heading 覆盖充分**的轨迹

### 3. 先检查 bag 是否可用

```bash
lidar2lidar-topics /path/to/record_or_record_dir
```

至少确认：

- LiDAR topic 存在
- `/apollo/localization/pose` 存在
- IMU topic 存在
- `/tf_static` 存在

### 4. 运行标定程序

推荐直接使用当前更接近量产的 `production` profile：

```bash
lidar2imu-convert-record \
  --record-path /path/to/record_or_record_dir \
  --lidar-topic /apollo/sensor/your_lidar/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/corrected_imu \
  --parent-frame imu \
  --profile production \
  --output-dir outputs/lidar2imu/customer_run01 \
  --calibrate
```

如果 bag 里没有 `lidar -> imu` 静态外参，需要额外提供：

```bash
--initial-transform path/to/imu_lidar_extrinsics.yaml
```

---

## 自动标定流程（程序会自动完成）

执行命令后，系统会自动完成下面这些步骤：

1. **解析 record 与 TF**
   - 自动识别 LiDAR frame、parent frame、参考 TF
2. **自动筛选 ground samples**
   - 提取地面法向和地面高度信息
3. **自动筛选 motion samples**
   - 从 LiDAR 局部配准和 IMU / pose 运动中构造 hand-eye 因子
4. **自动剔除坏样本**
   - 自动拒绝低 `registration_fitness` 的局部运动片段
5. **自动判断平面可观性**
   - 如果 `x / y / yaw` 可观性不足，系统会自动冻结这些自由度
6. **自动生成结果**
   - `metrics.yaml`
   - `diagnostics/review_report.html`
   - `diagnostics/*.svg`
   - `diagnostics/*.ply`

也就是说，客户不需要自己手工挑帧或人工配准，只需要准备一段质量足够好的动态数据。

---

## 推荐的客户执行命令

### 通用命令模板

```bash
lidar2imu-convert-record \
  --record-path /data/calib_record \
  --lidar-topic /apollo/sensor/lidar/main/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/corrected_imu \
  --parent-frame imu \
  --profile production \
  --output-dir outputs/lidar2imu/customer_run01 \
  --calibrate
```

### 本次真实数据示例

```bash
lidar2imu-convert-record \
  --record-path /mnt/synology/中集/2026-5-6-标定 \
  --lidar-topic /apollo/sensor/vanjeelidar/left_front/PointCloud2 \
  --pose-topic /apollo/localization/pose \
  --imu-topic /apollo/sensor/gnss/corrected_imu \
  --parent-frame imu \
  --profile production \
  --output-dir outputs/lidar2imu/20260506_left_front_production \
  --calibrate
```

---

## 结果文件在哪里看

一次成功运行后，最重要的文件在这里：

```text
outputs/lidar2imu/customer_run01/
├── standardized_samples.yaml
├── conversion_diagnostics.yaml
└── calibration/
    ├── calibrated_tf.yaml
    ├── metrics.yaml
    └── diagnostics/
        ├── acceptance_report.yaml
        ├── status_summary.csv
        ├── visualization_index.yaml
        ├── review_report.html
        ├── ground_residuals_plot.svg
        ├── motion_residuals_plot.svg
        ├── trajectory_overlay.svg
        ├── trajectory_position_gap_plot.svg
        ├── yaw_cost_scan.svg
        └── trajectory_overlay_cloud.ply
```

说明：

- 上面这个目录结构对应的是**客户实际最常用的**
  `lidar2imu-convert-record --calibrate` 输出
- 如果你后续是基于 `standardized_samples.yaml` 单独运行
  `lidar2imu-calibrate`，那么结果通常会直接写到输出目录根下，而不是再包一层
  `calibration/`
- `*.ply` 点云产物依赖原始 record 仍然可访问；如果只拷贝了
  `standardized_samples.yaml` 到另一台机器重新求解，可能没有这些点云文件

### 最快查看方式

1. 打开总览报告：

```bash
xdg-open outputs/lidar2imu/customer_run01/calibration/diagnostics/review_report.html
```

2. 打开点云重叠结果：

```bash
CloudCompare \
  outputs/lidar2imu/customer_run01/calibration/diagnostics/trajectory_overlay_cloud.ply
```

3. 看最终结论：

```bash
cat outputs/lidar2imu/customer_run01/calibration/diagnostics/acceptance_report.yaml
cat outputs/lidar2imu/customer_run01/calibration/diagnostics/status_summary.csv
```

---

## 如何判断标定结果是否“好”

### 最终结论先看这 4 个字段

打开 `metrics.yaml` 或 `acceptance_report.yaml`，优先看：

- `summary.final_acceptance_status`
- `summary.release_ready`
- `final_acceptance.recommendation`
- `vehicle_motion_assessment.recommendation`

### 量产可接受的目标状态

- `release_ready: true`
- `final_acceptance.recommendation` 不再是 review-only
- `vehicle_motion_assessment.recommendation: full_6dof_candidate`
- 没有 `trusted_reference` / `extraction_geometry` 冲突
- `holdout_generalization` 不报警

### 当前系统中几个关键阈值

| 指标 | 倾向于“好” | 倾向于“坏” |
| --- | --- | --- |
| `motion_registration_fitness_p05` | `> 0.55`，越高越好 | 低于 `0.55` |
| `motion_rotation_residual_p95_deg` | `< 0.3 deg` | 持续高于 `0.3 deg` |
| `motion_translation_residual_p95_m` | `< 0.05 m` | 持续高于 `0.05 m` |
| `yaw max_cost_ratio` | `> 1.5`，越高越好 | 接近 `1.0`，说明曲线很平 |
| `yaw within_5pct_span_deg` | `< 45 deg`，越小越好 | 很宽，说明很多 yaw 都差不多 |
| `turn_balance_ratio` | 越接近 `1.0` 越好 | 一侧转弯明显不足 |
| `release_ready` | `true` | `false` |

---

## 可视化结果怎么看

下面这些可视化是客户最应该学会看的。

### 1. `review_report.html`

这是**总入口**。建议先看它，再下钻到具体 SVG / CSV / PLY。

它会集中展示：

- 最终结论
- 主要指标
- 残差曲线
- yaw 可观性曲线
- IMU / LiDAR 轨迹对比
- 可直接打开的点云文件

### 2. `motion_residuals_plot.svg`

**表示什么：**

- 每个被选中的运动样本，其平移残差有多大

**什么表示好：**

- 曲线整体较低
- 没有明显离群尖峰

**什么表示坏：**

- 多个尖峰
- 某些样本残差明显高于其他样本
- 曲线整体偏高

### 3. `trajectory_overlay.svg`

**表示什么：**

- 用同一组 motion samples 累积出来的两条相对轨迹：
  - **蓝色虚线粗轨迹**：IMU / pose 侧轨迹
  - **红色实线细轨迹**：LiDAR 侧轨迹

**什么表示好：**

- 两条曲线整体形状一致
- 相对偏差稳定，没有明显发散

**什么表示坏：**

- 轨迹越走越分开
- 转弯处明显不一致
- 整体形状明显不同

### 4. `trajectory_position_gap_plot.svg`

**表示什么：**

- IMU 轨迹与 LiDAR 轨迹在累积过程中的位置差
- 图上预期就是**一条紫色曲线**
- 横轴：trajectory node index
- 纵轴：position gap (m)

**什么表示好：**

- 整体较低
- 波动小
- 不持续单调增大

**什么表示坏：**

- 逐步发散
- 某些区间突然跳大

### 5. `yaw_cost_scan.svg`

**表示什么：**

- 在 `[-180°, 180°]` 范围内扫描 yaw，查看目标函数是不是有清晰最优值
- 当前图使用的是 **`cost / best_cost` 归一化比值**
- 图上的**绿色虚线**表示 `+5%` 阈值
- 图上的**灰色竖线**表示当前最优 yaw

**什么表示好：**

- 曲线有明显尖锐低谷
- `max_cost_ratio` 高
- `within_5pct_span_deg` 小

**什么表示坏：**

- 曲线很平
- 低谷很宽
- 多个 yaw 看起来都差不多

### 6. `trajectory_overlay_cloud.ply`

**表示什么：**

- 用选中的 keyframe 片段分别按 IMU 和 LiDAR 轨迹拼接出的点云重叠结果

**什么表示好：**

- 墙面、路沿、地面边界基本重合
- 没有明显双层轮廓
- 局部结构一致

**什么表示坏：**

- 看到明显“双墙”“双边线”“重影”
- 同一结构有较大平移错位
- 某些转弯段错位尤其严重

---

## 真实示例：什么叫“好”，什么叫“坏”

下面示例来自这次真实数据：

- 较强候选：`outputs/lidar2imu/20260506_left_front_production_visualized_plus`
- 较弱 baseline：`outputs/lidar2imu/20260506_left_front_baseline_visualized_plus`

说明：

- 这些示例图来自我们基于现有 `standardized_samples.yaml` 重新生成的
  `*_visualized_plus` 目录
- 所以示例路径是 `.../diagnostics/...`
- 客户按本文推荐方式执行 `lidar2imu-convert-record --calibrate` 时，请优先查看：
  `outputs/lidar2imu/<run_name>/calibration/diagnostics/...`

### 示例 A：较好的 yaw 可观性

> 这张图来自 production 结果。它说明 yaw 目标函数已经明显尖锐，已经不是“很多 yaw 都一样好”。

![较好的 yaw cost scan](../outputs/lidar2imu/20260506_left_front_production_visualized_plus/diagnostics/yaw_cost_scan.svg)

这类图通常表示：

- yaw 可观性比较健康
- 更接近可接受的 full-6DoF 候选

### 示例 B：较差的 yaw 可观性

> 这张图来自 baseline 结果。它比较平，说明选中的局部运动片段太重复，yaw 支持不足。

![较差的 yaw cost scan](../outputs/lidar2imu/20260506_left_front_baseline_visualized_plus/diagnostics/yaw_cost_scan.svg)

这类图通常表示：

- yaw 支持不足
- 需要更丰富的双向转弯和更长尺度结构
- 或需要从 `scan_to_scan` 升级到 `submap_to_map`

### 示例 C：IMU / LiDAR 轨迹对比

> 这张图来自 production 结果。两条轨迹形状接近，说明 motion factors 已明显改善。

![IMU 与 LiDAR 轨迹对比](../outputs/lidar2imu/20260506_left_front_production_visualized_plus/diagnostics/trajectory_overlay.svg)

### 示例 D：累计位置差曲线

> 这张图用来判断 IMU 与 LiDAR 轨迹差是否持续扩大。

![IMU 与 LiDAR 累计位置差](../outputs/lidar2imu/20260506_left_front_production_visualized_plus/diagnostics/trajectory_position_gap_plot.svg)

### 示例 E：对比报告

> 同一份数据，baseline 与 production 的差异可以直接从对比报告里看出来。

- 对比 HTML：
  `outputs/lidar2imu/20260506_left_front_visualized_iteration_review/review_report.html`
- 其中 production 的结果明显优于 baseline：
  - `yaw max_cost_ratio: 1.21 -> 8.08`
  - `yaw 5% plateau: 115.5 deg -> 18.5 deg`
  - `motion translation residual p95: 0.882 m -> 0.021 m`

### 示例 F：点云重叠

当前示例点云在这里：

- `../outputs/lidar2imu/20260506_left_front_production_visualized_plus/diagnostics/imu_trajectory_cloud.ply`
- `../outputs/lidar2imu/20260506_left_front_production_visualized_plus/diagnostics/lidar_trajectory_cloud.ply`
- `../outputs/lidar2imu/20260506_left_front_production_visualized_plus/diagnostics/trajectory_overlay_cloud.ply`

建议直接用 CloudCompare 打开 `trajectory_overlay_cloud.ply`。

如果你看到：

- 蓝红两层结构基本重合：**好**
- 墙面和边界出现双层：**坏**

---

## 一个非常重要的现实判断

> **“看起来很漂亮” 不等于 “已经可以量产放行”。**

本次真实 production 示例虽然已经做到：

- `motion_registration = pass`
- `motion_rotation = pass`
- `motion_translation = pass`
- `yaw_observability = pass`
- `planar_basin_stability = pass`
- `trusted_reference = pass`

但它仍然 **不是 release-ready**，因为：

- `ground_orientation = warning`
- `holdout_generalization = warning`
- `motion_assessment_recommendation = warning`
- `final_acceptance.recommendation = review_or_partial_dof_only`

这说明：

- 单次解已经很好，尤其是 yaw 与 motion 因子质量已经明显改善
- 但地面方向稳定性和跨切分泛化还没有完全达到量产放行标准

因此客户在验收时，一定不要只看一张好看的图，还要看：

1. `release_ready`
2. `holdout_generalization`
3. `recommendation`

---

## 常见问题

### 1. 为什么我没有得到 6DoF，而是只得到部分可信结果？

因为当前 bag 的 `x / y / yaw` 可观性不足。最常见原因：

- 只有单侧转弯
- heading 覆盖太小
- 局部 motion snippets 太重复

### 2. 为什么 HTML 看起来不错，但结论还是 warning？

因为系统不仅看残差，还看：

- yaw 可观性
- basin stability
- trusted reference consistency
- holdout / repeatability

### 3. 什么样的数据最适合客户现场一次成功？

优先满足：

- 左右转都存在
- 有一段接近 figure-eight 的运动
- 地面清晰
- 周围静态结构丰富
- 不要全程匀速直行

---

## 推荐客户交付流程

1. 先用 `lidar2lidar-topics` 检查 bag
2. 直接运行 `lidar2imu-convert-record --profile production --calibrate`
3. 打开 `review_report.html`
4. 打开 `trajectory_overlay_cloud.ply`
5. 最后再看 `acceptance_report.yaml`
6. 只有 `release_ready: true` 时，才把该结果当作量产可接受结果

---

## 配套参考

- `docs/lidar2imu_quickstart.md`
- `docs/lidar2imu_user_guide.md`
- `docs/calibration_review_guide.md`
- `context/calibration_dataset_2026_05_06.md`
- `context/lidar2imu_context.md`
