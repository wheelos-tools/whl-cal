---
audience: dev
stability: proposal
---

# LiDAR-IMU 标定方案

本文档是 GV-MultiCalib 方案索引。详细设计按数据、算法、评估和实施计划拆分；原提案各章已完整归档到对应子文档，避免单文件维护和评审负担。

## 当前决策

- **基线优先**：GRIL 是 whl-cal LiDAR-to-IMU 的首个算法基线；Weilan 0827 corrected-IMU 是临时工作回归数据集，不是外参真值。Frozen-reference 物理诊断 rejected，native migration 仍未通过最终旋转等价门，详见 [GRIL 基线方法](lidar2imu/07_gril_baseline.md)。
- **方案定位**：GV-MultiCalib 是后续研究候选，目标为地面车辆、多 Session、共享 LiDAR→IMU 外参的连续时间标定。先证明坐标/残差/可观测性，再实现单段与多段优化。
- **三段流水线**：数据提取与质量筛选 → 算法求解 → 指标验证与可视化。质量门限用于解释、拒收和发布决策，不改写求解器行为。
- **数据边界**：Weilan 0827 corrected-IMU profile 已通过 Apollo/ROSbag canonical 等价检查，可作为工作输入 fixture；物理外参仍 rejected。中集及其他 Weilan capture 仍须分别审计时钟、连续性、点时间和传感器通道。
- **时钟边界**：测量时间与独立发布时间分别持久化；跨历元仅在回放入口一次初始化固定 H，绝不逐消息 header 重锚。旧同钟数据兼容，缺独立发布时间的跨历元数据明确拒绝；约定与限制见 [quick start](lidar2imu_quickstart.md#measurement-clocks-and-publication-time)。
- **精度结论**：0827 raw-IMU 历史结果与 corrected-IMU 新工作 profile 是不同输入身份；二者及 native/frozen migration 证据不可合并。缺少独立物理验证时结果只能是研究候选。

## 子文档

1. [目标、系统边界与数学模型](lidar2imu/01_scope_and_model.md)
2. [数据筛选、地面约束与几何特征](lidar2imu/02_data_and_geometry.md)
3. [初始化与连续时间联合优化](lidar2imu/03_initialization_and_solver.md)
4. [可观测性、验证与结果报告](lidar2imu/04_validation_and_artifacts.md)
5. [工程边界与实施计划](lidar2imu/05_implementation_plan.md)
6. [参考实现对比与复用决策](lidar2imu/06_reference_projects.md)
7. [GRIL 基线方法与 Weilan 0827 工作数据集](lidar2imu/07_gril_baseline.md)
8. [LiDAR-to-IMU 模块目录与命名](lidar2imu/08_module_layout.md)

## 推荐阅读顺序

先读目标/模型与数据方案，再读求解器和验证要求，最后按实施计划推进。参考项目文档是设计依据，不代表它们都能直接处理本项目的 Apollo/Vanjee 输入或可作为精度真值。
