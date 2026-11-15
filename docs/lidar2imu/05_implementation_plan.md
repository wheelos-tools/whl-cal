---
audience: dev
stability: proposal
---

# 工程边界与实施计划

[返回 LiDAR-IMU 方案索引](../lidar2imu.md)

## 十二、开源项目复用及自研边界
这一部分决定开发成本和商业产品边界。

我建议采用：

一个地面车辆标定基线、一个连续时间优化参考、一个初始化参考、一个几何预处理模块，以及一个自主开发的多段联合和质量控制引擎。

## 十三、工程目录与模块接口
以下目录仅作为模块职责草图，**不是立即新建独立工程的决定**。第一版先沿用本仓库的包边界；P0 / P1 通过后，再评估是否需要单独的 C++ 子系统及 ROS2 / WheelOS adapter。
gv_multicalib/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── config/
│   ├── sensors.yaml
│   ├── calibration.yaml
│   ├── motion_gating.yaml
│   └── quality_thresholds.yaml
│
├── include/gv_multicalib/
│   │
│   ├── common/
│   │   ├── types.hpp
│   │   ├── coordinate_frames.hpp
│   │   ├── calibration_state.hpp
│   │   └── covariance.hpp
│   │
│   ├── data/
│   │   ├── session.hpp
│   │   ├── bag_reader.hpp
│   │   ├── timestamp_model.hpp
│   │   └── cache_manager.hpp
│   │
│   ├── preprocessing/
│   │   ├── ground_segmentation.hpp
│   │   ├── ground_plane.hpp
│   │   ├── surfel_extraction.hpp
│   │   └── dynamic_filter.hpp
│   │
│   ├── motion/
│   │   ├── motion_profiler.hpp
│   │   ├── excitation_analysis.hpp
│   │   └── segment_selector.hpp
│   │
│   ├── initialization/
│   │   ├── gravity_initializer.hpp
│   │   ├── yaw_initializer.hpp
│   │   ├── time_initializer.hpp
│   │   └── lever_arm_initializer.hpp
│   │
│   ├── trajectory/
│   │   ├── se3_spline.hpp
│   │   └── spline_initializer.hpp
│   │
│   ├── factors/
│   │   ├── imu_gyro_factor.hpp
│   │   ├── imu_accel_factor.hpp
│   │   ├── lidar_surfel_factor.hpp
│   │   ├── ground_tangent_factor.hpp
│   │   ├── ground_height_factor.hpp
│   │   └── prior_factor.hpp
│   │
│   ├── optimizer/
│   │   ├── single_session_optimizer.hpp
│   │   ├── multi_session_optimizer.hpp
│   │   └── parameter_manager.hpp
│   │
│   ├── observability/
│   │   ├── information_matrix.hpp
│   │   ├── svd_analyzer.hpp
│   │   ├── marginal_covariance.hpp
│   │   └── sensitivity_analyzer.hpp
│   │
│   ├── validation/
│   │   ├── consistency_checker.hpp
│   │   ├── holdout_validator.hpp
│   │   └── calibration_report.hpp
│   │
│   └── export/
│       ├── yaml_exporter.hpp
│       └── ros2_adapter.hpp
│
├── src/
│   └── ...
│
├── apps/
│   ├── gv_calib.cpp
│   ├── gv_inspect.cpp
│   └── gv_validate.cpp
│
├── tests/
│   ├── test_coordinate_frames.cpp
│   ├── test_imu_jacobian.cpp
│   ├── test_spline_jacobian.cpp
│   ├── test_ground_factor.cpp
│   ├── test_lever_arm.cpp
│   ├── test_time_offset.cpp
│   └── simulation/
│
├── third_party/
│
└── docs/
    ├── mathematical_model.md
    ├── coordinate_convention.md
    ├── factor_design.md
    ├── observability.md
    └── validation_protocol.md


核心 API 建议
先定义几个稳定的数据结构，避免后期算法模块互相依赖具体实现。
struct SessionMetadata {
    std::string session_id;

    std::string lidar_frame;
    std::string imu_frame;

    double lidar_frequency;
    double imu_frequency;

    double timestamp_start;
    double timestamp_end;

    std::string lidar_serial;
    std::string imu_serial;

    ClockModel clock_model;
};

struct CalibrationResult {
    Eigen::Quaterniond q_IL;
    Eigen::Vector3d t_IL;

    double time_offset;

    Eigen::MatrixXd covariance;

    ObservabilityReport observability;
    CalibrationQuality quality;
};

struct CalibrationReport {
    bool converged;
    bool accepted;

    CalibrationResult result;

    std::vector<SessionReport> sessions;
    std::vector<ValidationResult> validations;
};


注意：最终结果中的 time_offset 应与时间模型对应。如果各 Session 的有效时间偏移不同，应输出每段偏移以及公共参数，不能强行合并成一个标量。

## 十四、实施路线与各阶段验收
最终开发顺序调整为 P0～P7。

这里刻意把数学仿真提前，避免直接修改开源项目、开始多段优化，最后才发现模型不成立。

## 当前执行约束与数据状态

本计划是后续算法研究路线，不代表这些开发阶段已经完成。当前先按仓库既有 **extraction → algorithm → evaluation** 边界工作；不立即创建独立的 `gv_multicalib/` 工程，不重写 record ingress，也不替换 ROS-free native GRIL。先把模型、数据契约与最小实验在现有仓库中验证，再决定是否需要新的 C++ 子系统。

用户指定的数据根目录及已知状态：

| 数据源 | 现状 / 计划用途 | 进入求解前必须完成 |
| --- | --- | --- |
| `/mnt/synology/中集/2026-5-6-标定` | 四个连续命名的 Apollo record shards；候选 Zhongji 05-06 标定采集。仓库已有同日期、32-ring、corrected-IMU 的研究记录，但在将其证据套用到此路径前，必须先以 shard 哈希和 topic inventory 确认是同一份输入。已有研究发现 GRIL 时间偏移/初值敏感且完整 LiDAR-only frontend 不稳定；当前不是精度基准。 | 核对全部 shard、topic/TF、LiDAR channel/ring/point-time、corrected IMU 时间来源和值、连续性和缺口；对齐现有 `input_contract` / iteration report 的输入哈希。 |
| `/mnt/synology/weilan/lidar2imu/0612` | 已有研究的低激励 / 初始时间重叠回归案例，不作准确度基准。 | 复用旧 input identity；若输入有变则重新审计，不因 no-result 降低充分性门限。 |
| `/mnt/synology/weilan/lidar2imu/0827` | 暂定为 GRIL 工作回归数据集。corrected-IMU frozen-reference 输入转换等价且两次参考结果一致，但独立物理诊断 rejected；raw-IMU 的既有实验仍是独立历史输入身份。二者都不是外参真值。 | 明确使用 `/apollo/sensor/gnss/corrected_imu` 并核对 shard/topic/hash；raw/corrected 数据不得互比或合并。native full-run 仍需三次重复、migration 等价和独立物理 holdout。 |
| `/mnt/synology/weilan/lidar2imu/bag` | 两个无扩展名分片，当前文档未确认它们对应的采集、格式或时钟。 | 先查 manifest、记录头和 topic，识别采集身份后再纳入；确认前标记 `unclassified`。 |

相关的已存研究记录位于 `.agents/knowledge/lidar2imu_research.md` 和 `context/knowledge_base/validated_conclusions.md`。重跑或比较前仍须核对这次本地文件哈希；目录名或已有结果不能证明输入相同。现有这些数据都没有独立物理外参真值。

### 依赖顺序

| 阶段 | 依赖 | 退出条件 |
| --- | --- | --- |
| P0 数据与参考基线 | 无 | 至少一个确切冻结的 canonical 输入；不合格样本有原因码；GRIL 完整运行的可执行文件、配置、输入 identity 可追溯。若没有合格数据，转为补采设计，不启动真实数据精度对比。 |
| P1 模型与仿真 | 可与 P0 并行 | 坐标变换、时间符号、主要残差与 Jacobian 通过合成真值和退化用例。 |
| P2 单 Session 算法 | P0 + P1 | 合格数据上可重复运行；与 frozen GRIL 对照；每个参数能区分数据观测、先验主导、不可观测和失败。 |
| P3 多 Session 联合 | P2 | 合成互补 Session 能提升对应参数信息；世界系 Gauge 与各 Session Bias / 时钟不互相污染。 |
| P4 信息分析 / 选择 | P2，P3 后扩展 | 输出边缘化数据 Jacobian 信息、弱方向、先验敏感度及 Session 增益，且不更改优化执行结果。 |
| P5 独立验证 | P2–P4 与独立 holdout / 物理测量 | 重复性、holdout、参数级指标和几何可视化均通过；否则保持 research/review-only。 |
| P6 工具链 | P5 的候选结果契约稳定 | extraction/cache、CLI、诊断和配置追溯可重复；不改变已验证算法语义。 |
| P7 平台集成 | P6 + 发布审查 | 配置方向、时间约定、下游回归和依赖许可全部审核。 |
### P0：开源基线复现
可立即实施

目标： 不开发新优化器，先获得可信的对照结果。

工作：
- 核对仓库 native GRIL 冻结基线及已有运行记录；仅在输入、二进制或配置 identity 不同或需要复现时重跑，不用上游 ROS 版悄悄替换基线。
- 盘点 LI-Init、LI-Calib、OA-LICalib 的实现版本、数据支持范围和已有运行结果；已有 no-result / adapter-blocked 状态原样保留。
- 评估 Patchwork++ 的地面分割输出，不把它作为标定器或自动默认前端。
- 比较 Kalibr 的时序 / 连续时间建模与 SensorsCalibration 的人工可视化边界。
- 审计两个用户指定数据根目录的 hashes、record topics、时间戳来源、点时间 / ring、TF、缺失分片及 gap；输出 frozen canonical candidate 和数据质量报告。
- 确认 Ceres / Eigen / Sophus 等版本。
- 建立第三方许可证清单。
验收：
- 参考项目职责和输入适配边界明确；不能运行的项目标记 `blocked` / `not_applicable` 并注明原因。
- 同一输入、配置和 executable identity 的 GRIL 结果可重现；前端不稳定时记录为 blocker，不排名算法精度。
- 至少一个候选输入通过格式、时钟、连续性、点时间和传感器通道验证；否则转入补采计划，不进入 P2。
- 原始输入和 canonical artifacts 有可复核哈希、topic / frame / time provenance 及 skip reason。
- 记录不同项目的坐标约定。
- 确认自有传感器是否在支持范围内。

### P1：数学模型与仿真验证
可与 P0 并行；通过后才能进入 P2

目标： 证明核心数学因子正确。

工作：
- 固定 FLU 和 SE(3) 变换规范。
- 搭建可配置的平面车辆运动仿真器。
- 注入已知外参、时间偏移和 Bias。
- 实现 IMU、Surfel、Ground、Prior 因子。
- 使用数值差分验证解析 Jacobian。
- 建立数据可观测性分析。
验收：
- 已知参数可以恢复。
- 纯直线工况正确识别杆臂退化。
- 无有效高度参考时正确标记 \(p_z\)。
- 静态或恒定动态下能够识别时间偏移退化。
- 坡道数据不会被全局平坦度先验强行拉平。

### P2：单段 GV 标定器
核心算法开发

目标： 不跨 Bag，完成第一版地面车辆专用标定。

工作：
- 单段 SE(3) B-Spline。
- IMU 连续时间残差。
- Ground Motion。
- Ground Height。
- 非地面 Surfel。
- 两阶段外参优化。
验收：
- 与 frozen native GRIL 在完全相同的 canonical input / holdout / evaluation protocol 上进行对照。
- 仅在传感器、时钟和输入 adapter 通过时与 LI-Calib / LI-Init / OA-LICalib 比较；否则明确标记 blocked。
- 输出独立参数及协方差。
- 误差、精度和失败条件可解释。

### P3：多段联合优化
项目核心差异化

目标： 建立真正的多段共享外参图。

工作：
- 每段独立轨迹。
- 每段独立 Bias。
- 有效 Session 时间偏移。
- 全局共享外参。
- 多段 Ground / Surfel 因子。
- 多段信息融合。
验收：
- 单段不足、多段互补时能够提升参数置信度。
- 新增数据不会无条件拉偏外参。
- 不同世界系 Gauge 不会交叉污染。
- 相同硬件配置重复采集得到一致结果。

### P4：可观测性及自动数据选择
核心产品能力

目标： 让系统自动知道哪些数据有价值。

工作：
- Session 信息矩阵。
- SVD 退化方向。
- 参数边缘协方差。
- 先验敏感性分析。
- 运动激励评分。
- 自动数据段筛选。
验收：
- 能说明哪个参数不可观。
- 能说明加入哪一段数据改善了哪个方向。
- 能区分数据主导和先验主导。

### P5：实车重复性及独立验证
量产验证前置

目标： 避免只在优化数据上表现好。

工作：
- 独立 Bag 或独立物理测量；当前指定数据没有独立外参真值时，必须另行取得，不得用配置 TF / INS 融合轨迹充当真值。
- 重复采集。
- 外参敏感性。
- LiDAR 运动补偿。
- IMU 残差。
- 下游里程计对比。
- 真值验证。
验收：
- 参数重复性满足要求。
- 验证集残差符合要求。
- 不存在先验掩盖的不可观测参数。
- 能可靠拒绝不合格数据。

### P6：量产工具链
产品化

目标： 自动运行、缓存、诊断和版本追踪。

工作：
- .calib_cache。
- 自动数据质量分析。
- CLI 工具。
- YAML 配置输出。
- 标定报告。
- 失败回滚。
- 第三方依赖治理。

### P7：平台集成
最终交付

目标： 接入 ROS2 / WheelOS / Apollo。
- 离线标定程序。
- 传感器外参配置导出。
- 时间同步参数导出。
- 结果审核。
- 标定版本管理。
- 下游配置回归。
不需要将标定优化器强行放进自动驾驶实时感知进程。第一版应当是独立的离线工具，后续再决定是否提供在线漂移监测。

## 十七、最终决策与开发优先级
将整个方案归纳为以下决策表。
设计问题	最终决策
轨迹使用 SE(2) 还是 SE(3)	保留完整 SE(3)
是否采用世界系 \(v_z=0\)	废除
是否引入局部地面切向约束	是
是否需要地面高度因子	是，但必须具备独立高度参考
Yaw 如何初始化	平面速度对应或手眼标定
能否用 \(\omega_z\) 估计 Yaw	不能
水平杆臂如何估计	转弯运动学 + 几何约束
是否强制要求角加速度激励	不强制，依据实际信息矩阵
\(p_z\) 如何处理	人工测量先验 + 可用的地面高度约束
地面与 Surfel 是否分流	是
是否认为 Hessian 天然正交	否
多段轨迹是否共享世界系	不共享
外参是否全局共享	同一安装配置下共享
Bias 是否全局共享	默认按 Session 独立
时间偏移是否全局共享	按实际时钟机制决定
是否采用 SPARSE_SCHUR	第一版不采用
第一版 Ceres 求解器	SPARSE_NORMAL_CHOLESKY
是否使用单一 Hessian 绝对阈值	否
可观测性如何判断	数据 Jacobian + SVD + 边缘化信息
精度如何验收	后验协方差 + 先验敏感性 + 独立验证
是否自动选择数据段	第一版规则筛选，后续信息增益选择

最终工程结论
GV-MultiCalib v3.0
作为后续研究候选；须按 P0 / P1 的数据、数学和实现准入门槛逐阶段进入工程验证

核心基础：
- Ground Vehicle SE(3) 连续时间轨迹。
- GRIL Ground Plane Motion。
- LI-Calib 风格的 IMU / Surfel 联合优化。
- 多 Session 共享外参。
- Session 时钟与 Bias 管理。
- 可观测性及后验不确定度分析。
核心自研：
- 多段状态管理。
- 车辆激励分析。
- 多段参数联合优化。
- 先验敏感性。
- 自动数据质量评估。
- 独立标定验收。

我认为最终需要坚持的一个设计思想是：

GV-MultiCalib 不是通过不断增加因子，让所有参数最终都能得到一个数值，而是通过不同来源的信息，确定每个参数究竟被什么约束。

例如：
- Yaw 由运动对应和环境几何共同确定。
- 水平杆臂由转弯运动和几何信息确定。
- Roll/Pitch 由可靠的重力、旋转和地面参考共同确定。
- \(p_z\) 由独立物理尺寸及地面高度参考确定。
- 时间偏移由动态时序对应确定。
- 多段联合负责增加信息，而不是掩盖单段不可观测性。
最终允许输出完整的 6-DoF 外参，但必须同时说明其每个自由度的信息来源和可信度。

这才是从论文复现走向真正车载标定工程的关键。
