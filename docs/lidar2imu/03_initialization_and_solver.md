---
audience: dev
stability: proposal
---

# 初始化与连续时间联合优化

[返回 LiDAR-IMU 方案索引](../lidar2imu.md)

## 八、初始化模块设计
初始化的职责是给非线性优化提供可靠起点，并不负责在信息不足时强行生成完整标定结果。
### 8.1 Roll / Pitch 初始化
优先利用：
- 静止 IMU 的重力观测。
- 可靠且经过验证的水平地面法向。
- 有效的车辆旋转轴对应关系。
对于已知水平地面：
\[
\frac{f_I}{\|f_I\|}
\]

与 LiDAR 地面法向经过坐标变换后存在明确的对齐关系，可求取两个方向的对齐旋转。

但这里有一个重要条件：

只有当地面已知近似水平时，才能把地面法向直接与重力对齐。

在坡道或横坡下，地面法向与重力方向并不相同。

因此，对于道路坡度未知的场景，不允许通过一次静止重力与地面法向对齐，直接锁定 Roll/Pitch。
### 8.2 Yaw 初始化
取消原先的：
\[
\omega_{z,I}
\longrightarrow
\psi
\]

这一错误路径。

推荐两种方式。

方式 A：多时刻平面速度对应

有可信的同参考点速度时：
\[
v_I\simeq R_z(\psi)v_L
\]

但需要处理杆臂速度、时间偏移和侧滑。

方式 B：平面相对运动手眼标定

通过多时间间隔的相对位姿：
\[
A_iX=XB_i
\]

构建平面手眼问题，联合考虑旋转与平移。

当有多方向运动、不同转弯曲率以及足够的平移信息时，才能稳定求解 Yaw 和水平杆臂。

优先使用多段联合估计，而不是只依赖某个时间点的方向差。
### 8.3 时间偏移初始化
先进行粗同步。

可使用：
- 陀螺仪角速度模长。
- 经过初步旋转对齐后的三维角速度。
- 变化的加速度信号。
对于角速度：
\[
c(\tau)=
\operatorname{corr}
\left(
\|\omega_I(t)\|,
\|\omega_L(t+\tau)\|
\right)
\]

寻找相关性高的时移作为初值。

需要强调：
- 恒定角速度不能提供有效的时间偏移辨识。
- 恒定线加速度通常也不能单独确定时间偏移。
- 时间偏移必须有时变信号。
- 互相关结果是初值，不是最终精确标定结果。
### 8.4 平移杆臂初始化
初值来源：
1. CAD 或人工尺寸测量。
2. 转弯段刚体运动学。
3. 多段联合几何优化。
杆臂加速度关系可以作为线性初值计算。

但正式估计应使用样条产生的角速度和角加速度，在连续时间图优化中完成，而不是对高噪声的离散角速度直接做一次数值微分后就认为已经得到精确杆臂。
### 8.5 B-Spline 初始化
平移：
\[
\min_C\|BC-P\|_W^2
\]

使用 QR 或 SVD 求解，不直接通过求逆：
\[
(B^TB)^{-1}B^TP
\]

来实现。

旋转：
- 使用 SO(3) 局部参数化。
- 对数映射和旋转插值。
- 在流形上拟合或迭代更新。
禁止把普通三维线性最小二乘直接应用于四元数控制点。

## 九、Ceres 多段联合优化器
### 9.1 优化目标
最终优化：
\[
\boxed{
\min_{\Theta}
E_{\mathrm{total}}(\Theta)
}
\]

其中：
\[
\begin{aligned}
E_{\mathrm{total}}=&
E_{\mathrm{gyro}}+
E_{\mathrm{acc}}+
E_{\mathrm{surfel}}\\
&+
E_{\mathrm{ground,tangent}}+
E_{\mathrm{ground,height}}\\
&+
E_{\mathrm{extrinsic,prior}}+
E_{\mathrm{bias,prior}}+
E_{\mathrm{time,prior}}
\end{aligned}
\]

其中不同因子按照各自测量噪声进行白化，鲁棒损失用于控制异常观测。
### 9.2 参数组织
建议采用：
Global Parameters
│
├── q_IL             # LiDAR -> IMU rotation
├── t_IL             # LiDAR origin in IMU
│
└── Clock Parameters # 按硬件时钟架构定义
    ├── delta_global
    └── optional clock scale

Session 001
│
├── SE3 B-Spline Controls
├── gyro_bias
├── accel_bias
├── gravity / local world definition
└── effective_time_offset

Session 002
│
├── SE3 B-Spline Controls
├── gyro_bias
├── accel_bias
├── gravity / local world definition
└── effective_time_offset

...

Residual Blocks
│
├── IMU Gyroscope Factor
├── IMU Accelerometer Factor
├── LiDAR Surfel Factor
├── Ground Tangential Factor
├── Ground Height Factor
├── Extrinsic Measurement Prior
└── Bias / Clock Prior


这里不要通过因子名称预先规定某个因子只能修改某几个外参参数。

所有因子都应根据实际物理模型产生完整 Jacobian，允许真实的参数交叉影响。
### 9.3 两阶段参数求解
保留原方案先固定 \(p_z\)，再释放 \(p_z\) 的思路，但重新定位它的意义。
Stage A：稳定水平参数的初步估计
- 使用人工测量的 \(p_z\) 初值。
- 暂时固定 \(p_z\)，用于数值稳定和敏感性对照。
- 使用 IMU、Surfel、时间和地面运动约束。
- 优化旋转、水平平移和允许估计的时间参数。
- 不使用未经验证的世界系平坦度约束。
获得初步水平外参和旋转结果

Stage B：垂直参数及完整外参精化
- 释放 \(p_z\)。
- 加入独立物理测量先验。
- 具备可靠离地高度参考时加入 Ground Height Factor。
- 使用完整 Jacobian 联合精化。
- 重新计算协方差和数据可观测性。
- 对 \(p_z\) 初值与测量误差进行敏感性试验。
获得最终候选标定参数及可信度

注意：

这两阶段只是一种优化初始化和数值稳定策略，不构成数学上的绝对隔离。

不能因为第一阶段固定了 \(p_z\)，就声称水平外参已经不受垂直参数误差影响。

如果第二阶段结果对人工 \(p_z\) 初值过度敏感，最终必须输出先验主导状态，不能静默接受联合优化结果。
### 9.4 Ceres 求解器
第一版统一采用：
ceres::Solver::Options options;

options.minimizer_type =
    ceres::TRUST_REGION;

options.linear_solver_type =
    ceres::SPARSE_NORMAL_CHOLESKY;

options.sparse_linear_algebra_library_type =
    ceres::SUITE_SPARSE;

options.linear_solver_ordering_type =
    ceres::AMD;


根据实际稀疏矩阵结构，后续可以测试 NESDIS 排序。

不再采用原方案：
SPARSE_SCHUR


并把所有 B-Spline 控制点放入同一个消元组的做法。

原因是一个样条残差通常同时涉及多个相邻控制点，无法满足该消元组需要独立集的条件。Ceres 官方文档明确规定了这一要求。对于一般稀疏问题，官方也推荐使用 SPARSE_NORMAL_CHOLESKY。
GitHub+2


后续如果需要更进一步优化：
- 测试自动排序。
- 分析带状稀疏结构。
- 采用局部消元策略。
- 必要时实现专用稀疏线性求解器。
但第一版不为性能而提前牺牲正确性。
