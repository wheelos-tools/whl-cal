---
audience: dev
stability: proposal
---

# 参考实现对比与复用决策

[返回 LiDAR-IMU 方案索引](../lidar2imu.md)

本地源码位于 `~/01code/`。下列比较基于当前本地 checkout 的 README 与实现入口；上游算法论文结果不能代替本项目相同输入上的复现、独立验证或发布证据。

| 项目 | 实际定位与可复用经验 | 对本项目的限制 / 决策 |
| --- | --- | --- |
| [LI-Init](https://github.com/zfc-zfc/LiDAR_IMU_Init) | FAST-LO 风格 LiDAR 里程计上的实时初始化；包含时间偏移、外参、重力和 IMU Bias 估计。可借鉴激励诊断、初始化流程和在线前端组织。见 `~/01code/LiDAR_IMU_Init/README.md`、`include/LI_init/LI_init.cpp`。 | 与 GRIL 有研究脉络关系，不作为独立算法家族的胜出证据。仓库已有 Weilan 0827 的 no-result 记录；保留原始充分性门限，不为强制产出结果而降门限。传感器配置、时间戳和点时间必须先核对。 |
| [LI-Calib](https://github.com/APRIL-ZJU/lidar_IMU_calib) | 连续时间批量优化，联合 IMU 残差和 LiDAR point-to-surfel 距离，并可优化时间偏移。适合作为连续时间轨迹、关联及局部几何因子的参考。见 `~/01code/lidar_IMU_calib/README.md`、`src/core/`。 | 本地 README 标注当前仅支持 VLP-16。Weilan 0827 的非组织 PointCloud2 不能直接按该 reader 假设解析；数据适配未通过前只能标记 `blocked_input_adapter`，不应把它描述为算法失败或精度比较。 |
| [GRIL-Calib](https://github.com/Taeyoung96/GRIL-Calib) | 面向地面机器人的 targetless 标定，以 Ground Plane Motion 约束从平面车辆运动估计外参。是地面先验与地面车辆基线最直接的参考。论文：[arXiv:2312.14035](https://arxiv.org/abs/2312.14035)。本地实现：`~/01code/GRIL-Calib/include/Gril_Calib/Gril_Calib.cpp`。 | 上游 ROS 实现与本仓库 ROS-free native 移植不是同一个可执行物；保留本仓库冻结输入/二进制和等价性要求。地面高度等先验不能冒充数据独立提供的信息。当前运行结果仍须按本仓库研究契约判为 review-only，不能以收敛或论文报告替代物理验证。 |
| [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) | 快速、自适应的地面分割器，提供地面 / 非地面分类及局部几何的上游特征。论文：[arXiv:2207.11919](https://arxiv.org/abs/2207.11919)。本地源码：`~/01code/patchwork-plusplus/`。 | 它不是 LiDAR-IMU 标定器。只作为几何预处理候选；须在目标传感器、地面坡度、点密度和动态物体条件下评估误分割，并与 GRIL 地面处理保持可比较输入。 |
| [Kalibr](https://github.com/ethz-asl/kalibr) | 成熟的相机-IMU / IMU-IMU 空间与时间标定工具，提供连续时间批量估计、轨迹表示和报告工作流；相关基础文献包括 Furgale、Rehder、Siegwart，IROS 2013。见 `~/01code/kalibr/README.md`。 | Kalibr 的主问题不是 LiDAR-IMU，也不提供本方案所需地面运动 / LiDAR surfel 残差。借鉴连续时间和时钟建模，不把它当作 LiDAR-IMU 直接基线。 |
| SensorsCalibration (`~/01code/SensorsCalibration/lidar2imu/README.md`) | 本地仓库含手动 LiDAR-to-pose-sensor 对齐工具；可借鉴人工检查交互和点云可视化。 | 手动调节不是可重复的自动估计器，也不是数值精度证据。仅用于辅助可视化，不参与自动方法胜负判定。 |
| [OA-LICalib](https://github.com/APRIL-ZJU/OA-LICalib) | 连续时间 LiDAR-IMU 批量标定，并提出基于信息指标的数据选择与可观测方向更新。适合作为可观测性设计参考。见 `~/01code/OA-LICalib/README.md`、`src/trajectory/`、`src/calib/`。 | 本地 README 说明其可观测模块和仿真代码将另行发布；因此当前 checkout 不可直接作为该能力的可执行基准。先在本项目用合成数据验证边缘化信息、先验区分和退化方向，再考虑同输入比较。 |

## 复用边界

1. **首要对照**：仓库现有 ROS-free native GRIL。冻结输入、配置和可执行文件后比较完整前端重复性；不要另建一份记录解码链。
2. **算法候选**：先复现一个与数据格式兼容的 LI-Calib / LI-Init / OA-LICalib 组件或做最小实验；不可因方法不兼容而把 `blocked` 记为零误差或算法失败。
3. **几何组件**：Patchwork++ 仅负责地面候选特征；通过同一 canonical 点云、固定参数和可视化评估其效果。
4. **时序与轨迹**：参考 Kalibr 的连续时间建模理念，但为 LiDAR 点时间、车辆地面约束和每 Session 时钟建立本项目明确的残差及约定。
5. **许可证 / 版本**：任何代码复用前，记录 commit、依赖版本、许可证和修改范围；论文引用与代码许可证分别审核。

## 仓库内优先依据

- [GRIL ROS-free migration](../gril_ros_free_migration.md)：输入适配、native 执行、等价性与发布边界。
- [LiDAR-to-IMU research contract](../../.agents/knowledge/lidar2imu_research.md)：数据筛选、对照实验和独立精度判定。
- [Frozen `wheelos_core` baseline](../lidar2imu_wheelos_core_baseline.md)：Weilan 0827 的输入和 native 执行回归基线。
- [Calibration review guide](../calibration_review_guide.md)：稳定输出和人工审阅顺序。
