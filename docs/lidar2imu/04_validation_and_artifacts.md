---
audience: dev
stability: proposal
---

# 可观测性、验证与结果报告

[返回 LiDAR-IMU 方案索引](../lidar2imu.md)

## 十、可观测性分析与结果可信度系统
这是 GV-MultiCalib 最值得投入自研能力的模块。

不仅要知道解出来什么，更要知道每个参数是被什么信息解出来的。
### 10.1 必须区分两种信息矩阵
第一种：仅数据提供的信息。
\[
H_{\mathrm{data}}=J_{\mathrm{data}}^TJ_{\mathrm{data}}
\]

它不包含人工测量先验、强制参数固定和人为正则化。

第二种：加入先验后的信息矩阵。
\[
H_{\mathrm{post}}=
H_{\mathrm{data}}+H_{\mathrm{prior}}
\]

如果只有第二种满秩，并不意味着传感器数据已经独立确定所有参数。

例如：
- \(p_z\) 只有人工测量先验。
- 平面运动无法提供新的 \(p_z\) 信息。
- 最终仍然可以有一个数值稳定的完整 6-DoF 外参。
但必须标记：

p_z = PRIOR_DOMINATED

而不能输出成：

p_z = DATA_CALIBRATED
### 10.2 必须消除轨迹和其他干扰状态
设：
\[
\theta=
[R_{IL},t_{IL},\delta]^T
\]

为最终需要关注的标定参数；
\[
\eta=
[C_k,b_{g,k},b_{a,k},g_k,\ldots]
\]

为其他轨迹、Bias 和辅助状态。

将 Hessian 分块：
\[
H=
\begin{bmatrix}
H_{\theta\theta}&H_{\theta\eta}\\
H_{\eta\theta}&H_{\eta\eta}
\end{bmatrix}
\]

标定参数的边缘化信息矩阵：
\[
\boxed{
H_{\mathrm{marg}}=
H_{\theta\theta}
-
H_{\theta\eta}
H_{\eta\eta}^{\dagger}
H_{\eta\theta}
}
\]

其中需要正确处理 Gauge Freedom 和奇异子空间。

如果存在复杂 Gauge 或 \(H_{\eta\eta}\) 秩亏，应使用投影、秩揭示 QR 或带阈值的 SVD，而不是简单对病态矩阵求逆。

这样才能避免将轨迹本身提供的信息误认为外参本身具有独立可观测性。
### 10.3 无量纲归一化
对于标定参数：
\[
\Delta\theta=
D\Delta\xi
\]

其中 \(D\) 为预先定义的目标误差尺度：
\[
D=
\operatorname{diag}
(
\epsilon_R,\epsilon_R,\epsilon_R,
\epsilon_p,\epsilon_p,\epsilon_p,
\epsilon_t
)
\]

测量白化后：
\[
\tilde J=J_{\mathrm{white}}D
\]

\[
\tilde H=\tilde J^T\tilde J
\]

进行特征值分解：
\[
\tilde H=V\Lambda V^T
\]

重点分析：
- 是否存在零空间。
- 最弱可观测方向。
- 最小有效奇异值。
- 条件数。
- 外参参数之间的相关性。
- 不同数据段的信息贡献。
不再使用没有统一参数尺度依据的：
\[
\lambda_{\min}(H)>10^3
\]

作为硬性验收规则。
### 10.4 后验协方差
使用加先验后的信息矩阵计算局部后验协方差：
\[
\Sigma_{\mathrm{post}}
\approx H_{\mathrm{post}}^\dagger
\]

在消除 Gauge、边缘化 nuisance states 后，输出：
\[
\sigma_{\mathrm{roll}},
\sigma_{\mathrm{pitch}},
\sigma_{\mathrm{yaw}},
\sigma_{p_x},
\sigma_{p_y},
\sigma_{p_z},
\sigma_{\delta}
\]

这里是局部高斯近似下的不确定度，不应视为所有鲁棒损失和模型误差下都严格准确的统计置信区间。

必须配合重复采集、独立验证和必要的 Bootstrap 进行检查。
### 10.5 统一参数状态
每一个参数都有四种状态：
DATA_OBSERVED
数据自身提供了足够的信息。

PRIOR_DOMINATED
结果主要依赖物理测量或人工先验。

UNOBSERVABLE
当前数据无法确定该参数或其特定方向。

FAILED
模型、数据或优化过程不满足基本质量要求。


这比简单输出一个 6-DoF 变换矩阵更有工程意义。
### 10.6 初始量产精度目标
原方案中的精度目标可以保留为第一版候选指标，但统一采用 \(3\sigma\) 定义：
参数	初始目标
Yaw 标准误差	\(3\sigma \le 0.1^\circ\)
水平杆臂	\(3\sigma \le 1\,cm\)
时间偏移	\(3\sigma \le 1\,ms\)
Roll / Pitch	根据系统精度预算制定
垂直杆臂	根据独立高度测量精度及先验贡献判定

这里的数值是项目目标，不是通用行业标准。

特别是 \(p_z\)：假设人工测量本身的标准差为 2 cm，且没有更强的独立高度约束，就不应要求系统凭空输出 1 cm 的可信精度。

## 十五、必须通过的系统级测试
这些测试是进入量产验收前的准入条件。
编号	测试工况	期望系统行为
T01	纯直线匀速	不虚构杆臂和时间偏移观测
T02	直线加减速	利用有效动态信号估计时间偏移
T03	恒定转弯	正确利用角速度信息估计水平杆臂
T04	S 弯	检查多方向运动信息和联合精度
T05	多段互补运动	比较联合与单段信息改善
T06	坡道	不使用错误的全局 \(v_z=0\) 约束
T07	错误 \(p_z\) 先验	识别水平参数敏感性和先验主导
T08	地面法向错误	Ground Factor 降权或拒绝
T09	时间戳重置	每段时间模型正确处理
T10	非地面特征稀缺	正确报告几何退化
T11	重复数据段	避免统计信息被重复累加
T12	外参反向输入	坐标转换和结果导出测试失败，而不是静默通过

尤其建议增加三组对照：
实验 A：人工 \(p_z\) 敏感性
分别输入：
\[
p_z-\Delta,\quad p_z,\quad p_z+\Delta
\]

建议先测试 \(\Delta=1cm,2cm,3cm\)。

记录：
\[
\Delta p_x,\Delta p_y,\Delta\psi
\]

并检查：
- 联合外参是否稳定。
- 地面残差是否发生异常变化。
- 后验协方差是否明显依赖先验。
实验 B：多段信息消融
分别运行：
- 只使用直线数据。
- 只使用转弯数据。
- 只使用地面约束。
- 只使用 Surfel。
- 直线 + 转弯。
- 直线 + 转弯 + 地面高度。
- 全部数据联合。
输出每个参数的信息矩阵与边缘标准差。

这可以证明多段联合优化到底带来了什么。
实验 C：独立采集验证
使用一组数据标定，再用不同时间采集的 Bag 验证。

重点比较：
- 外参变化。
- 时间偏移变化。
- IMU 运动学残差。
- 点云去畸变前后效果。
- 里程计轨迹一致性。
闭环 ATE 可以是其中一个指标，但不能单独作为通过条件。

## 十六、最终标定报告规范
GV-MultiCalib 最终不能只输出：
rotation:
  x: ...
  y: ...
  z: ...

translation:
  x: ...
  y: ...
  z: ...

time_offset: ...


应该输出完整的结果追溯信息。

建议：
calibration:
  version: "3.0"

  status: "ACCEPTED"

  transform:
    parent_frame: "imu_link"
    child_frame: "lidar_link"

    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0

    translation:
      x: 0.0
      y: 0.0
      z: 0.0

  timing:
    model: "per_session_offset"
    sessions:
      session_001:
        offset: 0.0
      session_002:
        offset: 0.0

  uncertainty:
    rotation_std: [0.0, 0.0, 0.0]
    translation_std: [0.0, 0.0, 0.0]
    time_offset_std: 0.0

  observability:
    roll: "DATA_OBSERVED"
    pitch: "DATA_OBSERVED"
    yaw: "DATA_OBSERVED"
    px: "DATA_OBSERVED"
    py: "DATA_OBSERVED"
    pz: "PRIOR_DOMINATED"
    time_offset: "DATA_OBSERVED"

  validation:
    independent_dataset: true
    parameter_consistency: true
    pointcloud_consistency: true

  metadata:
    session_count: 3
    lidar_serial: "..."
    imu_serial: "..."
    source_hash: "..."
    config_hash: "..."


上面的数值只是格式示例，不代表标定结果。

最终状态必须支持：
- ACCEPTED：满足完整交付要求。
- ACCEPTED_WITH_PRIOR：部分参数依赖物理先验，但先验来源和精度均符合要求。
- DEGRADED：仅部分参数可信。
- REJECTED：数据或模型不足以提供可用结果。
对于 DEGRADED，应明确哪些参数可以使用，哪些参数必须继续采用已知机械测量值或旧标定值。
