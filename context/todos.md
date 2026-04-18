# Project todos (context)

## improve-right-edge-scan2map (pending)

目标：提高 `record_data_0402` 中右侧传感器（lslidar_right）在 scan2map 优化中的跨子图一致性，使其与 left-edge 在 holdout 指标上可比。

主要动作项：
1. 检查 diagnostics/scan2map_optimization.yaml 中右边失败运行的子图几何、重叠和同步误差分布。
2. 尝试不同局部子图参数：submap radius（6m, 8m, 10m, 15m）与 support keyframes（6, 8, 12, 20）。
3. 为 lslidar_right 添加传感器专用预处理试验：
   - voxel_downsample（0.02, 0.04, 0.06）
   - ground removal（基于高度阈值或地面分割）
   - max_height 限制（如 2.5m, 3.5m）
4. 优化配对选择：先列出所有 source->submap 同步对，再按 sync_error、时间覆盖和估计重叠排序并挑选子集。
5. 运行 `lidar2lidar-scan2map`（使用 dataset artifact），记录每轮 `metrics.yaml` 与 `diagnostics/scan2map_optimization.yaml` 以便对比。
6. 若仍无法形成共识，尝试更方向性子图（窄扇形）或增加子图数量以覆盖更多视角。

预期产出：
- context/（本文件）里记录每轮参数与结论
- diagnostics/scan2map_optimization_{trial}.yaml
- metrics.yaml（每轮）
- 若成功：calibrated_tf_right.yaml 与 diagnostics/evaluation_right.yaml

负责人：你（或下次接手的人）
计划状态：pending

---

保存位置：`context/todos.md`（此文件）

时间戳：2026-04-17T14:46:53.555Z


## 2026-04-18 constrained-DoF findings

right -> main 在 `record_data_0402` 上的新增结论：

- unconstrained + method3 + loose gate + ground removal:
  - holdout fitness ≈ 0.862
  - holdout rmse ≈ 0.096
  - overlap ≈ 0.636
  - 但相对 scan2scan baseline 漂移 ≈ 0.10 m / 3.9 deg
  - 偏差集中在 `z` / `roll`
- lock `z, roll` to scan2scan baseline:
  - 漂移显著减小到 ≈ 0.021 m / 0.324 deg
  - 但 holdout 指标几乎退回 baseline（fitness ≈ 0.571, overlap ≈ 0.291）
- lock `z, pitch, roll` to scan2scan baseline:
  - 结果与 lock `z, roll` 类似，说明 scan2map 的主要“收益”基本来自垂向姿态偏移，而不是平面分量优化
- lock `x, y, yaw` to scan2scan baseline:
  - holdout 仍高（fitness ≈ 0.862, overlap ≈ 0.611）
  - 但仍保持 ≈ 0.099 m / 3.87 deg 的整体漂移，说明问题确实主要在 `z/pitch/roll`

当前判断：
- right edge 的 scan2map gain 主要由 `z/pitch/roll` 偏置驱动，不宜直接作为最终外参接受。
- 下一轮优先方向不是继续放松 gate，而是：
  1. 为 right edge 增加 production-safe profile：默认锁 `z/pitch/roll` 或至少锁 `z/roll`，只把 unconstrained 作为诊断模式
  2. 重新设计 right-edge keyframe gate，不再沿用全局 `min_keyframe_fitness=0.90`
  3. 若要继续释放垂向自由度，需要增加额外约束，例如更强的 vertical prior、map support 过滤，或分层代价函数


## 2026-04-18 lidar2lidar final recommendation

当前最终建议：

- `scan2scan` 继续作为 `lidar2lidar` 的生产默认方法。
- `scan2map` 作为二级验证 / refinement，而不是直接替代 `scan2scan`。
- 对车辆场景，必须把指标拆成：
  - 平面分量：`x/y/yaw`
  - 垂向姿态分量：`z/pitch/roll`
- `left -> main` 当前可以接受 unconstrained scan2map；
- `right -> main` 当前只能把 unconstrained scan2map 视为诊断模式，因为其 gain 主要来自 `z/pitch/roll` 偏置。
- 若继续在 right edge 上使用 scan2map，推荐先以 `scan2scan` 为基线，并锁定 `z/pitch/roll`，只将自由 6DoF 结果作为观测性分析，不直接写回最终外参。

## 2026-04-18 lidar2imu first iteration

当前 `lidar2imu` 的第一轮结论：

- `record_data_0402` 可以完成端到端标定，但它不是强 `x/y/yaw` 数据包。
- 转换层默认加入 `--min-registration-fitness 0.55` 之后，可以稳定剔除明显坏的 motion pair。
- 过滤掉低质量 motion pair 后，`motion_translation_residual_p95_m` 从约 `2.32` 降到约 `0.43`，说明主问题先在数据层而不是求解器本身。
- 即便 motion pair 质量改善，`left_turn_count > 0` 且 `right_turn_count = 0` 仍然成立，所以当前包只能支持 `z/roll/pitch` 优先。
- `gravity-source imu` 在该包上会导致 `ground_selected = 0`，因此目前继续使用 pose-derived gravity 更稳妥。

下一步建议：

1. 在 solver 层加入 `z_roll_pitch_priority` 下的平面分量降权 / 锁定。
2. 换一包同时包含左转和右转的数据，验证 `x/y/yaw` 是否恢复稳定。
3. 对多个 bag 比较 `z/roll/pitch` 的重复性，再决定是否调整默认阈值。

## 2026-04-18 synology front-lidar bag

`/mnt/synology/raw-data/2026-04-13-06-54-28` 的新增结论：

- 这个包只有单个前向 LiDAR，且 bag 内没有 `lidar_front -> imu` 静态 TF。
- 因此给 `lidar2imu-convert-record` 增加了两种入口：
  - `--initial-transform <yaml>`
  - `--identity-initial-transform`
- identity fallback + 默认 stride 会失败：`1/8` motion pairs 可用。
- 调成更局部的 motion pair 之后：
  - `--motion-frame-stride 1 --min-registration-fitness 0.45` 是当前最好的数据层配置；
  - `8/8` motion pairs 可用，registration fitness p05 约 `0.78`；
  - 但由于没有真实先验外参，且只有单侧转向（`right_turn_count=8`），最终解会漂到不可信的 yaw（约 `62.7 deg`）。
- `gravity-source imu` / `corrected_imu` 在该包上仍然会导致 `ground_selected = 0`。

当前判断：

- 该包适合做 **数据层 / 弱观测诊断**；
- 不适合在没有真实初值的前提下直接产出最终 `lidar2imu` 外参；
- 如果后续拿到 `imu_lidar_front` 初值文件，应优先用 `--initial-transform` 重跑这包。

## 2026-04-18 lidar2imu weak-planar policy

本轮已完成：

- 在 `lidar2imu` solver 中加入 `--planar-motion-policy`：
  - `free`
  - `freeze_xyyaw`
  - `auto`
- `auto` 会在 `turn_imbalance` 或 `yaw_rotation_degenerate` 时自动锁定
  `x/y/yaw`，只继续优化 `z/roll/pitch`
- `metrics.yaml` 现在会显式写出 `summary.solver_policy` 和
  `vehicle_motion_assessment.applied_solver_planar_motion_policy`
- `lidar2imu-tune-record` 也已经支持 sweep `planar_motion_policy`，且排序规则
  改成优先看 recommendation 和 drift，再看 residual

本轮验证结论：

1. `record_data_0402`
   - `free`: `delta_to_initial ≈ 2.17 m / 1.30 deg`
   - `auto`: `delta_to_initial ≈ 0.016 m / 1.15 deg`
   - 说明：在弱平面观测下，auto 能明显抑制不可信的平面漂移
2. Synology + 用户初值
   - `free`: `delta_to_initial ≈ 0.343 m / 2.65 deg`
   - `auto`: `delta_to_initial ≈ 0.010 m / 0.73 deg`
   - 说明：该包仍然只能做诊断，但 auto 更符合“守住先验，不乱漂”的原则
3. `lidar2imu-tune-record` smoke
   - 在 `pose + stride1 + fitness0.45 + auto/free` 的最小 sweep 中，
     调整后的 ranking 会优先选 `policy_auto`

下一步优先级：

1. 做初值扰动 / repeatability 实验，量化 `auto` 与 `free` 在弱观测 bag 上的方差
2. 换一包同时包含左转和右转的数据，验证 `auto` 是否能自动退出 freeze 模式
3. 再决定是否把 `auto` 提升为更强的默认 acceptance 策略

## 2026-04-18 lidar2camera next target

下一阶段目标：

- 把 `camera` / `camera2lidar` 纳入和 `lidar2lidar`、`lidar2imu` 一样的
  **data -> algorithm -> evaluation** 结构

当前已知现状：

- `camera/intrinsic.py` 已经是一个可用的相机内参标定脚本
- `camera2lidar/reference_based.py` 是基于棋盘板的 LiDAR-camera 外参路径
- `camera2lidar/learning_based.py` 是 targetless 的实验路径
- 但当前仍缺少：
  - 统一的数据集 artifact
  - 窗口 + 门控的数据筛选
  - 稳定的 `metrics.yaml` / `diagnostics/`
  - 明确的 acceptance recommendation

下一步建议：

1. 先梳理 `camera` 和 `camera2lidar` 的输入 / 输出 / 当前命令入口
2. 先定义 repo 级的评测面，再动算法
3. 把 reference-based 作为初始 acceptance baseline
4. 把 targetless 路径先作为 comparison / diagnostic branch
