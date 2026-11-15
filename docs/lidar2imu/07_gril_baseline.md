---
audience: dev
stability: review-only
---

# GRIL 基线方法与 Weilan 0827 工作数据集

[返回 LiDAR-IMU 方案索引](../lidar2imu.md) ·
[快速标定](../lidar2imu_quickstart.md) ·
[GRIL ROS-free 迁移与发布门限](../gril_ros_free_migration.md)

## 1. 决策与范围

GRIL 是 whl-cal LiDAR-to-IMU 的**首个算法基线**：优先复现并保持其
ground-plane motion、LiDAR-only 前端和分阶段标定行为，不以 GV-MultiCalib
研究方案替换它。Weilan 0827 暂定为工作回归数据集，用于固定输入契约、
复现流程和比较后续改动；它不是标定精度真值，也不代表算法或产品已验收。

本基线保留三段边界：

1. **Extraction**：Apollo record / bag 适配器生成唯一的 `CanonicalDataset`，
   并记录 topic、frame、timestamp、ring、point-time、TF 和来源。
2. **Algorithm**：运行未被质量门限改写的 GRIL native full frontend 与 batch
   solver；冻结配置、输入、可执行文件和 gap policy。
3. **Evaluation**：独立检查输入、完整前端重复性、时间与杆臂 holdout、
   轨迹/点云效果，并给出接受状态。评估失败不能被 solver 收敛覆盖。

GV-MultiCalib 仍是后续研究候选。新初始化或残差只能在同一 canonical
数据、GRIL 配置和对照产物上做独立实验；不得暗中改变此基线。

## 2. 当前工作数据身份

| 项目 | Weilan 0827 工作 profile |
| --- | --- |
| Apollo record | `/mnt/synology/weilan/lidar2imu/0827`，三个连续 shards |
| LiDAR | `/apollo/sensor/vanjeelidar/up/PointCloud2`，Vanjee-16 |
| IMU | `/apollo/sensor/gnss/corrected_imu`；本次审计使用其 measurement-time header |
| Scan lines | 16；ring 为 0–15 |
| Canonical 数量 | 1,375 scans、35,239,034 finite points、13,741 IMU samples |
| 证据状态 | 数据转换等价通过；标定物理结果 rejected，整体 review-only |

corrected-IMU frozen-reference 实验将同一 capture 通过 Apollo record 与
可信 ROSbag canonical 路径比较，样本顺序、时间戳、ring、点时间、XYZ、
intensity、IMU 信号和静态变换逐项一致。该证据只证明输入一致，不证明
外参精度。实验记录位于
`/mnt/synology/whl-cal-gril-validation-20261114/weilan_0827_gril_reference/`。

此实验的 frozen GRIL 两次结果字节一致；独立物理诊断却未支持该外参：
held-out 杆臂残差只改善 1.13%，不足以支持单一泛化杆臂；yaw/time
分段 profile 也没有唯一泛化解。因此该输出仅是**可重复的 review fixture**，
不得发布为准确外参。后续 native ROS-free 结果还须单独通过 frozen-reference
等价门和物理验证。

仓库中的 [`wheelos_core` ingress baseline](../lidar2imu_wheelos_core_baseline.md)
使用 `/apollo/sensor/gnss/imu`，是历史 raw-IMU 输入身份，用于记录适配器、
二进制和执行耗时。它与本节 corrected-IMU profile 的时间/信号不同，不可
互换或作为其 canonical compare 目标；旧数据与结论保留作历史回归案例。
不同 capture 必须按 measurement-time provenance 重新选择 IMU topic，
不能仅凭 topic 名称套用 0827 配置。

## 3. 基线算法职责

GRIL 的完整执行由 LiDAR-only 前端与 batch 标定组成。前端按冻结的
Velodyne/Vanjee 配置完成点云预处理、事件同步、无 IMU 恒速传播、
Patchwork++ 地面分割、LiDAR odometry / EKF 和地面约束；达到数据充分性
边界后，将原始状态队列交给独立 batch 进程。Batch 端执行 GRIL 原有的
状态插值、前段丢弃、滤波/微分、旋转估计、对齐与联合求解。

实现边界：

- 数据格式差异只放在 `lidar2imu.adapters`；算法接收相同 canonical arrays。
- `lidar2imu/algorithms/gril/full_runner.py` 是 record/bag/canonical 到 ROS-free full-run 的执行边界；
  原始 topic adapter 的 `source_type` 只描述来源，不应决定算法行为。
- 以 `gap_policy=golden` 作为 baseline，保持冻结语义；切换 `reset` 属于显式
  算法运行语义变化，必须另建实验身份。
- 外部输入质量评估、结果比较和诊断只能给出准入/拒绝结论；保留 GRIL
  自带的数据充分性判断，不得静默改写前端队列、solver 参数或停止条件。
- `rotation_LI_xyzw` 是可选的研究初始化，不属于默认基线；默认保持冻结的
  identity seed。带 TF seed 的运行必须另存，且 TF 不能充当外参真值。

## 4. 坐标与配置约定

GRIL 输出的旋转 `R_LI` 将 LiDAR 向量映射到 IMU 坐标系；输出平移 `t_LI`
是 LiDAR 原点在 IMU 坐标系中的位置。配置中的 `trans_IL` 则是 IMU 原点
在 LiDAR 系中的位置，转换为：

```text
trans_IL = -R_LI.T @ t_LI
```

不要把 LiDAR 高度、IMU 离地高度和两传感器间的垂直杆臂混为一谈。
`resources/vanjeelidar16.yaml` 是 Vanjee-16 参数 profile，其中的安装变换、
sensor height、激励阈值和边界是初始化/模型参数，不是独立测量真值。
目标安装、LiDAR 通道数或地面条件不同，必须审阅并调整配置，但应将新配置
标记为独立 baseline 变体。

## 5. 固定产物目录

研究数据和大体积运行文件放在被 `.gitignore` 忽略的 `outputs/gril/`；
设计文档放在 `docs/lidar2imu/`，入口索引为 `docs/lidar2imu.md`。建议将
capture、IMU clock profile 和方法写入运行目录名，避免 `run01` 等名称丢失身份：

```text
outputs/gril/0827_corrected_imu_baseline/
├── dataset/
│   ├── dataset.yaml
│   ├── input_contract.yaml
│   ├── lidar.npz
│   └── imu.npz
├── canonical_comparison.yaml  # when a trusted comparison is rerun locally
├── run_01/
│   ├── native_dataset_v1.bin
│   ├── gril_native_full.conf
│   ├── gril_native.conf
│   ├── GRIL_Calib_result.txt
│   ├── GRIL_full_frontend_trace_v1.txt
│   ├── GRIL_batch_trace_v1.txt
│   ├── manifest.yaml
│   └── customer_summary.yaml
├── run_02/
├── run_03/
└── diagnostics/
    ├── repeatability.yaml
    ├── holdout.yaml
    ├── frontend_trajectory.yaml
    └── ...
```

`lidar2imu prepare` 写出 `dataset.yaml`、两个 NPZ 和 `input_contract.yaml`；
每个 `run-native` 输出独立的 manifest、result 和 trace。原生 runner 不会自动
生成根级 `metrics.yaml` 或把诊断放进 `diagnostics/`，应由相应验证工具写入；
不要把不同运行的 artifact 覆盖到同一个目录。每份报告都应保存输入、
配置和 executable SHA-256、source revision、topic/frame/time profile、
运行状态、指标、失败原因与可视化链接。

## 6. 最小执行与验收

准备环境、record ingress 和完整命令见
[LiDAR-to-IMU quick start](../lidar2imu_quickstart.md)。基线流程是：

1. 用明确指定的 corrected-IMU topic 提取 canonical 数据，检查
   `input_contract.yaml` 的 counts、连续性、point-time、ring、frames、
   静态 TF 和 measurement-time provenance。输入有变化时重新做可信 adapter
   equivalence；不能比较不同 IMU topic 的 canonical 数据。
2. 对同一 dataset、配置和 executable 完整运行至少三次，分别保存到
   `run_01/`、`run_02/`、`run_03/`。记录每次 config/dataset/executable
   digest、结果参数、完整前端轨迹和 batch trace；不要挑选“最好的一次”。
3. 先检查重复运行和初始化扰动，再看时间窗口、导数质量及 holdout。
   研究参考 spread gate 为 rotation `<= 0.2 deg`、translation `<= 0.03 m`、
   time `<= 0.001 s`；超限须归因为前端不稳定、弱可观测或多解，不能仅靠
   重复 solver batch 掩盖 full frontend 的差异。
4. 对水平杆臂做 bias-only 对照和连续时间段 holdout；时间/yaw 用连续窗口
   检查最优解是否稳定。进一步通过独立轨迹或运动补偿点云验证物理效果。
   不使用同一配置外参同时作 seed 和 truth。
5. 报告 verdict：`accepted` 需输入合约、完整前端、重复性、holdout 和物理
   证据全通过；缺少独立物理验证为 `review-only`；数据完整但多解/信息不足
   为 `inconclusive`；输入错误、失稳或 holdout 退化为 `rejected`。

另外，当前 native migration 的 frozen-reference 完整结果旋转差为
`0.406964 deg`，高于 `0.2 deg` 比较门限，详见
[迁移门限](../gril_ros_free_migration.md)。因此 native GRIL 仍是 research
build，不是可发布标定器。该 migration 差值、历史 raw-IMU 重复性和本节
corrected-IMU frozen-reference 物理结论属于不同实验，报告时必须分别注明。

## 7. 后续迭代规则

先固定本基线的输入、算法配置、可执行文件和产物契约，再一次只验证一个
假设：例如 TF rotation seed、导数滤波或另一种 LiDAR odometry。候选必须
使用完全相同的 canonical input 和 holdout 划分，并保留基线所有失败样本。
只有在独立物理验证、重复性和可解释性上持续胜过 GRIL，才讨论替代基线。

参考：[GRIL-Calib](06_reference_projects.md)、[研究决策契约](../../.agents/knowledge/lidar2imu_research.md)、
[验证技能](../../.agents/skills/gril-calib-validation/SKILL.md)。
