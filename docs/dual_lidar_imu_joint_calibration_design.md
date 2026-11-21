---
audience: dev
stability: stable
last_tested: 2026-11-20
---

# 双雷达-IMU (`2 × L2I + L2L`) 联合标定与升级版 `L2L` 设计方案

[返回 Quickstart 索引](quickstart_index.md) ·
[LiDAR-to-IMU Quickstart](lidar2imu_quickstart.md) ·
[LiDAR-to-LiDAR 设计](lidar2lidar_design.md) ·
[标定方法论](calibration_methodology.md)

## 1. 问题背景与核心动机

在双角雷达 + 单 IMU 的典型商用车构型（如中策 `2026-11-17` 数据：`left_front` 与 `right_back` 32 线角雷达 + `corrected_imu`）上，通过 [五维交叉诊断](lidar2imu_quickstart.md#5-pillar-dual-sensor-diagnostics) 发现了单路标定无法察觉的结构性互补与短板：

1. **单路 `lidar2imu` (`L2I`) 的能力与极限**：
   - 对 **重力对齐的 Roll/Pitch**（多航向坡度对消后 `~0.51°`）、**传感器相对 IMU 3D 旋转角速度对齐**（陀螺 RMSE `0.013~0.019 rad/s`）、**硬件时钟偏置 $\tau_k$**（`-22.1 ms / -22.8 ms`）具有高精度可观性。
   - **核心短板**：在重卡低角加速度机动（$|\omega_z| \le 0.4\text{ rad/s}$）下，松耦合 LiDAR EKF 的 `30 Hz` 恒速平滑与速度数值微分会衰减切向加速度峰值，造成水平杆臂向 IMU 原点**径向收缩 ~13.5%**——双路外参诱导出的雷达间相对基线与直连 L2L 存在 `0.977 m`（`Δx=+0.383 m, Δy=-0.890 m, Δyaw=-0.95°`）的显著残差；同时平路行驶时垂直杆臂 $t_z$ 物理不可观。
2. **现有 `lidar2lidar` (`L2L`) 的能力与短板**：
   - 在静止多帧叠加下，直连几何可将两雷达之间的 **6-DoF 相对位姿** $T_{AB} = T_{IA}^{-1} T_{IB}$ 钉死到 `±2 cm / ±0.08°`。
   - **现有实现短板**（详见 `lidar2lidar/lidar2lidar.py`, `fov.py`, `loop_closure.py`, `record_utils.py`）：
     1. 入口丢弃了 FlatBuffer/PointCloud2 已解析的 `intensity` 与 `ring`；
     2. 未区分地面、建筑立面、竖直杆/墙角与散射噪点（树冠/杂波），地面点过量淹没水平约束，走廊场景信息矩阵条件数高达 `822~914`；
     3. `fov.py` 仅做基于固定初值的 1D 方位角扇区裁剪，既未剔除本车车体遮挡射线，也未做薄墙法向同侧门控，初值偏差大时反而裁掉真共视墙面；
     4. `loop_closure.py` 用单一标量权重等权缩放 6 个自由度，未按各向异性信息矩阵分解可观测子空间。

---

## 2. 文献范式调研与假设适配评估

| 角色 | 代表文献 / 开源范式 | 核心数学机制 | 在本方案中的作用 |
| :--- | :--- | :--- | :--- |
| **结构基元 + 退化感知** | **M-LOAM** (Jiao et al., *IEEE T-RO* 2021) / **X-ICP** (Tuna et al., *IEEE T-RO* 2024) | 分离地面/立面/边缘线基元；对 $6\times 6$ 海森矩阵 $\mathbf{H}$ 做特征值-特征向量本征分解，在退化方向显式降权或施加先验 | **采纳并落地为 `v2`**：免深度学习提取 4 类几何语义基元，并以 X-ICP 特征分解量化每个窗口的可定位自由度 |
| **物理归一化反射率配准** | **Colored-ICP** (Park et al., *ICCV* 2017) / **Intensity-ICP** | 切向光度/强度残差辅助平坦表面切向滑移约束 | **适配改进为 `v2`**：先做距离/入射角补偿与传感器内分位数归一化（消除不同雷达硬件增益差），再用高反射率边缘点约束标线/反光牌 |
| **流形因子图融合** | **Kalibr** / **Lie-Group $\mathrm{SE}(3)$ Factor Graph** (Barfoot) | 在 $\mathrm{SE}(3)$ 切空间按完整 $6\times 6$ Fisher 信息矩阵融合异构观测并对零空间正则 | **核心底座 `v1`**：正交分解共模/差模自由度，按真实各向异性信息融合 `2×L2I + L2L + 重力` |
| **纯 IMU 预积分 + 多雷达 BA** | **Forster Preintegration** (*T-RO* 2017) / **BALM2 & MLCC** (HKU MARS, *T-RO* 2022) / **iKalibr** (2024) | 直接用原始 `100 Hz` IMU 预积分驱动跨时间多雷达点云平面锐度（最小特征值/点到面 BA），不经由里程计低通微分 | **创造性破局 `v3`**：在 `2.5 s` 急转弯窗口绕开 `30 Hz` EKF 带宽收缩，消除共模杆臂残余偏差与跨传感器时延差 |

---

## 3. 升级版 `L2L` 设计（`lidar2lidar/semantic_registration.py`）

### 3.1 四通道物理语义过滤与基元提取
输入保留 `(x, y, z, intensity, ring)`，首先按体素（默认 `0.35 m`）计算局部协方差特征值 $\lambda_1 \ge \lambda_2 \ge \lambda_3$：
- **散射噪点剔除**：球度 $\lambda_3 / \lambda_1 > 0.28$ 的树冠/灌木杂点直接剔除；
- **通道 1 — 地面基元 $\mathcal{P}_{\text{ground}}$**：RANSAC 拟合主地面并筛选 $|n_z| \ge 0.90$、平面度 $(\lambda_2-\lambda_3)/\lambda_1 \ge 0.45$ 的面元，设置均衡上限点数（如 `400` 点），仅提供法向点到面约束 $[\Delta z, \Delta\text{roll}, \Delta\text{pitch}]$；
- **通道 2 — 结构立面 $\mathcal{P}_{\text{facade}}$**：$|n_z| < 0.30$ 且平面度 $\ge 0.45$ 的墙体/箱体面元，构建高精度点到面残差 $r_{\mathcal{P}} = \mathbf{n}_q^\top(\mathbf{R}_{AB}\mathbf{p} + \mathbf{t}_{AB} - \mathbf{q})$；
- **通道 3 — 竖直杆/墙角棱线 $\mathcal{E}_{\text{pole}}$**：线性度 $(\lambda_1-\lambda_2)/\lambda_1 \ge 0.55$ 且主方向 $|v_{1,z}| \ge 0.85$ 的立柱/路灯杆/墙角，构建 2D 水平点到线正交残差：
  $$\mathbf{r}_{\mathcal{E}} = \left(\mathbf{I}_3 - \mathbf{v}_q\mathbf{v}_q^\top\right)\left(\mathbf{R}_{AB}\mathbf{p} + \mathbf{t}_{AB} - \mathbf{q}\right)$$
  单根竖杆即可打破长直走廊的轴向平移退化；
- **通道 4 — 距离/入射角补偿与传感器内归一化反射率特征 $\mathcal{M}_{\text{intensity}}$**：
  $$\tilde{I}(\mathbf{p}) = I(\mathbf{p}) \cdot \frac{\max(\|\mathbf{p}\|, 1.0)^\gamma}{\max(|\hat{\mathbf{p}}\cdot\mathbf{n}|, \cos 75^\circ)^\beta}, \quad z_I = \frac{\tilde{I} - \mathrm{median}(\tilde{I})}{\max(\mathrm{IQR}(\tilde{I}), 10^{-3})}$$
  取 $z_I \ge \tau_{\text{retro}}$ 的稳定高反光点（车道线、反光牌轮廓），抑制饱和拖尾后与目标高反光近邻匹配。

### 3.2 三重 3D 物理共视域门控
1. **本车包围盒自遮挡射线剔除（Ego AABB Occlusion）**：检测传感器原点到目标点的射线是否穿过卡车车体包围盒 $\mathcal{B}_{\text{truck}}$，防止“穿透车厢匹配对侧点”。
2. **初值协方差自适应 3D 视锥交集（Uncertainty-Buffered 3D Frustum）**：同时覆盖水平方位角与垂直俯仰角 $[\phi_{\min}, \phi_{\max}]$、有效距离 $[r_{\min}, r_{\max}]$，并随外参不确定度 $(\sigma_\theta^{(k)}, \sigma_t^{(k)})$ 动态缩放角裕量 $\Delta\alpha = 3\sigma_\theta^{(k)} + \frac{3\sigma_t^{(k)}}{\max(\|\mathbf{p}\|, 2.0)}$，避免早期有偏初值误删真实共视墙面。
3. **法向同侧一致性门控**：要求 $\mathbf{n}_A^\top(\hat{\mathbf{R}}_{AB}\mathbf{n}_B) > \cos 35^\circ$，根除薄墙正反面误吸合。

### 3.3 X-ICP 可定位性诊断
在最优解处构建加权 Gauss-Newton 信息矩阵 $\mathbf{H}_{\text{L2L}} = \frac{1}{\hat{\sigma}^2}\mathbf{J}^\top\mathbf{W}\mathbf{J} \in \mathbb{R}^{6\times 6}$ 并做特征分解 $\mathbf{H}_{\text{L2L}} = \mathbf{V}\operatorname{diag}(\lambda_1,\dots,\lambda_6)\mathbf{V}^\top$，输出：
- 条件数 $\kappa = \lambda_6 / \max(\lambda_1, 10^{-9})$；
- 平移与旋转子空间的最小/最大特征值及对应的**退化方向特征向量**（例如长直走廊沿道路切线方向平移可定位性弱）。

---

## 4. 创造性联合建模：共模 / 差模正交分解与 `2 × L2I + L2L` 统一优化

待求时空外参状态为 `14` 维（参考系为 IMU 坐标系 $I$）：
$$\mathcal{X}_{\text{calib}} = \left\{\mathbf{T}_{IA} \in \mathrm{SE}(3),\; \mathbf{T}_{IB} \in \mathrm{SE}(3),\; \tau_A \in \mathbb{R},\; \tau_B \in \mathbb{R}\right\}$$
其中点变换定义为 $\mathbf{p}_I = \mathbf{R}_{Ik}\mathbf{p}_{L_k} + \mathbf{t}_{Ik},\; k \in \{A, B\}$，诱导相对外参 $\mathbf{T}_{AB}(\mathcal{X}) = \mathbf{T}_{IA}^{-1}\mathbf{T}_{IB}$。

### 4.1 规范自由度（Gauge Freedom）与可观测性正交分解定理
将 `12` 维空间外参 $(\delta\boldsymbol{\xi}_{IA}, \delta\boldsymbol{\xi}_{IB})$ 分解为 **差模子空间** 与 **共模子空间**：

| 子空间 | 自由度 | 物理含义 | 主导约束来源 | 可观测性结论 |
| :--- | :--- | :--- | :--- | :--- |
| **差模** $\delta\boldsymbol{\xi}_{\text{diff}}$ | `6 DoF` | 两雷达相对位姿 $\mathbf{T}_{AB} = \mathbf{T}_{IA}^{-1}\mathbf{T}_{IB}$ | 多窗口语义 `L2L` | **全 6-DoF 强可观**（`±1~2 cm, ±0.08°`），彻底治愈 EKF 造成的 `0.98 m` 相对基线收缩 |
| **共模倾角** $(\bar{\phi}, \bar{\theta})$ | `2 DoF` | 双雷达骨架整体相对 IMU 水平倾角 | 多航向重力-地面对齐 + 3D 陀螺 | **强可观**（`< 0.3°`），且相反航向自动对消路面坡度 |
| **共模航向** $\bar{\psi}$ | `1 DoF` | 整体相对 IMU 航向角 | 双路陀螺对齐 + 转弯动力学 | **强可观**（双传感器独立里程计方差减半） |
| **共模水平杆臂** $\bar{\mathbf{t}}_{xy}$ | `2 DoF` | IMU 原点在双雷达连线中的水平偏置 | `v1/v2`: 双路 L2I 平均；`v3`: 转弯纯 IMU 预积分双雷达 BA | **`v3` 可直接用三维墙面弯弧矢高绕开 EKF 低通收缩**，恢复真实杆臂 |
| **共模垂直安装高度** $\bar{t}_z$ | `1 DoF` | 双雷达整体相对 IMU 的安装高度 | **平路运动物理不可观**（$\omega_x,\omega_y\approx 0$） | 显式投影至零空间 $\mathbf{P}_{\mathcal{N}}$，由物理量测/先验 $z_{\text{prior}}$ 锚定并标记为不可观 |
| **时延** $(\tau_A, \tau_B)$ | `2 DoF` | 各自相对 IMU 的时间偏置 | `L2I` 互相关 + `v3` 转弯点云去畸变锐度 | **强可观**，且 $\Delta\tau=\tau_A-\tau_B$ 由跨传感器转弯墙面对齐进一步锁定 |

### 4.2 统一优化目标与三步演进阶梯 (`v1 → v2 → v3`)

$$\min_{\mathcal{X}_{\text{calib}},\, \mathcal{X}_{\text{aux}}} \sum_{k\in\{A,B\}} J_{\text{L2I},k} + \sum_{w\in\mathcal{W}_{\text{static}}} J_{\text{L2L},w}\!\left(\mathbf{T}_{IA}^{-1}\mathbf{T}_{IB}\right) + \sum_{h\in\mathcal{W}_{\text{grav}}} J_{\text{grav},h}(\mathbf{R}_{IA},\mathbf{R}_{IB}) + \sum_{u\in\mathcal{W}_{\text{turn}}} J_{\text{IMU-BA},u}(\mathcal{X}_{\text{calib}}, \mathbf{v}_{u,0}, \mathbf{g}_{u,0}) + \left\|\mathbf{P}_{\mathcal{N}}(\mathcal{X}\ominus\mathcal{X}_0)\right\|_{\boldsymbol{\Lambda}_{\text{prior}}}^2$$

1. **`v1`（首版挑战基线）— 各向异性 $\mathrm{SE}(3)$ 因子图融合**：
   - 在流形上联合融合双路 `L2I` 旋转/水平平移（令平路不可观 $z$ 轴与大方差收缩方向降权）、多窗口 `GICP L2L` 完整 $6\times 6$ 信息矩阵、多航向坡度对消重力-地面法向约束与 $\bar{t}_z$ 零空间先验。
   - **效果**：直接消除双雷达对角基线 `0.98 m` 闭环残差，保留重力水平精度。
2. **`v2`（特征与视野升级）— 语义基元 + 归一化反射率 + 3D 共视域 `L2L`**：
   - 将 `v1` 中的裸 GICP 边替换为 `semantic_registration` 输出的 4 通道基元 + 归一化强度 + 车体自遮挡 3D 共视域 + X-ICP 信息矩阵边，解决走廊退化与弱重叠漂移。
3. **`v3`（紧耦合精化）— 急转弯 `100 Hz` 纯 IMU 预积分 + 双雷达跨时间 BA**：
   - 以 `v2` 结果为初值，在多个大角速度转弯窗口（`~2.6 s`）内用 `100 Hz` IMU 预积分 $(\Delta\mathbf{R}_I(t), \Delta\mathbf{p}_I(t), \Delta\mathbf{v}_{\text{spec}}(t))$ 将双雷达结构化点云（立面/地面/竖杆）统一投影到窗口首帧 $I_{u,0}$：
     $$\mathbf{p}_{k,j}^{I_{u,0}} = \Delta\mathbf{p}_I(t_j+\tau_k) + \mathbf{v}_{u,0}(t_j+\tau_k-t_0) + \frac{1}{2}\mathbf{g}_{u,0}(t_j+\tau_k-t_0)^2 + \Delta\mathbf{R}_I(t_j+\tau_k)\left(\mathbf{R}_{Ik}\mathbf{p}_{k,j} + \mathbf{t}_{Ik}\right)$$
   - 采用外层双雷达联合体素平面/边缘重拟合 + 内层精确解析雅可比高斯牛顿/TRF 联合优化 $(\mathbf{T}_{IA}, \mathbf{T}_{IB}, \tau_A, \tau_B, \{\mathbf{v}_{u,0}, \mathbf{g}_{u,0}\})$，并叠加静态 `L2L`、重力对齐和陀螺旋转因子防止过拟合。内置锐度回退保护：若转弯窗口墙厚未改善或恶化超过阈值则回退 `v2`。

---

## 5. 三阶段工程流水线与 CLI 命令

严格保持本仓 **Extraction → Algorithm → Evaluation** 解耦：

1. **Stage 1（数据筛选与多模态提取）**：`screen_joint_calibration_windows` 提取静止共视窗口 $\mathcal{W}_{\text{static}}$、多航向重力窗口 $\mathcal{W}_{\text{grav}}$、高角速度转弯窗口 $\mathcal{W}_{\text{turn}}$，并显式记录每个候选窗口的 `skip_reason`。
2. **Stage 2（联合求解）**：`lidar2imu calibrate-dual-joint` 支持 `--mode v1 | v2 | v3 | all`（默认 `v2` 核心联合图，`< 5 s`；传 `--mode all` 可同时输出 `v1/v2/v3` 三档消融对比与标准 `extrinsics/*.yaml`）。
3. **Stage 3（交叉诊断与未见 Holdout 验证）**：使用 `lidar2imu evaluate-dual-diagnostics --extrinsic-a-yaml ... --extrinsic-b-yaml ...` 在**独立未见 Holdout 片段**（如 `[785, 1005 s]`）运行完全一致的五维定量与双色 `.pcd`/切片图评估。

### 典型运行命令

```bash
# 1. 运行 2xL2I + L2L 联合标定（输出 v1/v2/v3 对比及最终标准外参 YAML）
lidar2imu calibrate-dual-joint \
  --dataset-a /path/to/left_front/dataset/dataset.yaml \
  --run-dir-a /path/to/left_front/window_full_0_680s \
  --dataset-b /path/to/right_back/dataset/dataset.yaml \
  --run-dir-b /path/to/right_back/window_full_0_680s \
  --output-dir outputs/gril/joint_dual_calib \
  --mode all

# 2. 在未见 Holdout 片段上用同一套五维诊断验收联合标定外参
lidar2imu evaluate-dual-diagnostics \
  --dataset-a /path/to/left_front/dataset/dataset.yaml \
  --run-dir-a /path/to/left_front/holdout_785_1005s \
  --dataset-b /path/to/right_back/dataset/dataset.yaml \
  --run-dir-b /path/to/right_back/holdout_785_1005s \
  --extrinsic-a-yaml outputs/gril/joint_dual_calib/extrinsics/imu_left_front_extrinsics.yaml \
  --extrinsic-b-yaml outputs/gril/joint_dual_calib/extrinsics/imu_right_back_extrinsics.yaml \
  --output-dir outputs/gril/joint_dual_calib_holdout_diag
```

---

## 6. 全量 23 分片实测经验总结、路线淘汰结论与门禁后正式标定值

在中策 `2026-11-17` 全量 23 个 `.record` 分片（`1092.2 s / 10,912` 帧，划分 4 个独立 `~220 s` 片段 `S1..S4` + 全段 `Full_0_680s` + 全段 4 处跨 18 分钟停车点 + 未见 Holdout `[785, 1005 s]`）上的系统级 A/B 实测确立了以下工程结论：

### 6.1 `v1 -> v2` 核心联合图胜出与三大更重路线淘汰实证

| 对比路线 | `Full` 耗时 | Holdout 墙隙 / L2L 闭环 | Holdout 重力偏置 / 转弯墙厚 | LOTO / 跨段结论 |
| :--- | :--- | :--- | :--- | :--- |
| 单路基线 `GRIL` | `~260 s` | `49.42 cm` / `58.59 cm, 1.74°` | `0.460°` / `6.67 cm` | 基线内缩 ~`13.5%`（`6.20 m` vs `7.12 m`） |
| **`Joint v2`（当前主线：`v1 GICP -> v2 语义 L2L + 各向异性图`）** | **`4.46 s`** | **`0.51 cm`** / **`2.10 cm, 0.212°`** | **`0.280°`** / **`5.89 cm`** | **全指标最优**；全包 4 停车点中位偏差 **`1.75 cm`**；速度级动力学降幅 **`90.8%`** |
| `Old v3`（`2.6 s` 短转弯 BA） | `18.60 s` | 同 `v2`（被回退） | 同 `v2` | **淘汰**：自由 $\mathbf{v}_{u,0}$ 下二阶弧高灵敏度仅 `2.4%`，**100% LOTO 回退** |
| `Scheme 3`（合帧 `360°` 再进 `GRIL EKF`） | `478.9 s` | `172.31 cm`（无 `v1` 漏斗时） / `4.42 cm` | `0.573°` / — | **淘汰**：慢 107× 且仍受 `GRIL EKF` 杆臂收缩（`1.46~1.86 m`） |
| `Paradigm 7`（`6.5 s` 贯通 `BALM2` BA） | `8.6 s` | 同 `v2`（被回退） | `0.773°` / `6.24 cm`（未回退前） | **淘汰**：长窗 $\frac{1}{2}g\Delta t^2$（`0.5°` 即 `1.81 m`）淹没 cm 级杆臂，LOTO 回退 |

核心工程教训：
1. **`v1` GICP 粗漏斗不可跳过**：若直接用收缩后的 `GRIL T_AB_init ([-5.43, +2.61])` 跳过 `v1` 跑语义 ICP，会在重复杆/墙道路上落入 `[-5.95, +2.50]` 假极小（正交墙错位 `172 cm`）；`v1 -> v2` 级联总能收敛到全局真解 `[-6.45, +3.02]`（结构内点率高 `9.5%~12.8%`）。
2. **平路低速（`<= 14 km/h`）车体航向不可由 3D 陀螺独立保证**：实测 `S1..S4` 跨段 `v2_yawA + straight_slip_angle` 恒为 `134.42° ± 0.08°`，证明弱段（`S1` 零左转、`S4` 偏左转且加速少）的 `2°` 航向漂移源于前端单边激励不足，而非传感器松动。
3. **无局部重力联合的 `dv_spec` 长窗速度积分会放大路面坡度**：`1°` 坡度在 `4 s` 内产生 `0.68 m/s` 漏项，直接把航向拉偏 `6.5°`。

### 6.2 共模激励门禁（`evaluate_common_mode_excitation_gate`）与门禁后正式标定值

在 `joint_dual_calib.py` 中新增后验激励门禁（永不改动核心求解器内部状态），仅当同时满足以下条件时输出 `verdict: accepted`，否则标定值仅作诊断参考并标记 `needs_more_excitation`：
- 双向转弯平衡：`min(left_turn_deg, right_turn_deg) >= 90°` 且 `turn_symmetry_ratio >= 0.25`（积分阈 `|\omega_z| > 0.08 rad/s`）；
- 水平动态加速：`p95_planar_acc >= 0.50 m/s²`（`0.3 s` 平滑去均值）；
- 峰值横摆角速度：`max_abs_wz >= 0.32 rad/s` 且健康 `L2L` 边 $\ge 1$；
- 垂直高度 $z_{\text{cm}}$ 单独评估 `p99(|\omega_{xy}|) >= 0.30 rad/s`，未达标明确报告 `z_cm_observable: false (prior_anchored_planar_nullspace)`。

#### 中策 `2026-11-17` 门禁筛选实测与正式接受标定值

- **被门禁准确拦截的弱激励/单边转向片段**：
  - `S1 [0, 220 s]` -> `needs_more_excitation`（左转 `28.1°` vs 右转 `270.2°`，对称比 `0.104`，`p95_acc = 0.486 m/s²`）
  - `S2 [220, 450 s]` -> `needs_more_excitation`（左转 `72.4°` vs 右转 `335.4°`，对称比 `0.216`，`p95_acc = 0.484 m/s²`）
  - `S4 [785, 1005 s]` -> `needs_more_excitation`（左转 `399.6°` vs 右转 `52.7°`，对称比 `0.132`，`max_wz = 0.315`）
- **通过门禁的正式接受片段**：
  - **`S3 [450, 680 s]`**（`accepted`：左转 `471.3°` / 右转 `250.4°` / 对称比 `0.531` / `p95_acc = 0.591 m/s²`）：
    - `left_front`: `rpy = [-0.2380°, +0.9237°, +132.8455°]`, `t_LI = [-0.2720, +2.3395, +0.4504] m`, `tau = -18.587 ms`
    - `right_back`: `rpy = [+0.0385°, -0.5096°, -47.1765°]`, `t_LI = [+1.9292, -4.3691, +0.4462] m`, `tau = -23.551 ms`
    - `baseline = 7.0604 m`, `z_cm_observable = false`
  - **`Full [0, 680 s]`（主标定输出，`accepted`：左转 `583.8°` / 右转 `850.0°` / 对称比 `0.687` / `p95_acc = 0.524 m/s²`）**：
    - `left_front`: `rpy = [-0.1419°, +0.7041°, +133.1000°]`, `t_LI = [-0.3675, +2.5235, +0.4897] m`, `tau = -22.109 ms`
    - `right_back`: `rpy = [-0.3507°, -0.2785°, -46.8998°]`, `t_LI = [+1.8328, -4.2527, +0.4435] m`, `tau = -22.792 ms`
    - `T_A_B` 平移 `[-6.4501, +3.0238, -0.1180] m`, `baseline = 7.1246 m`, `z_cm_observable = false`
    - 两通过片段 `Yaw` 一致性仅差 **`0.25°~0.28°`**，基线差 **`6.4 cm`**（门禁前全段离散达 `45–50 cm / 2.5°`）。

### 6.3 唯一高 ROI 的剩余改进方向

停止在低速（`<= 14 km/h`）数据上继续堆砌后端预积分 BA 优化器。在 `Joint v2 + 共模激励门禁` 已实现 `L2L ~1–2 cm`、倾角 `0.28°` 的基础上，唯一值得投入的高 ROI 后续动作仅剩两项：
1. **采集端激励升级（低成本、高收益）**：专门采标定包时将车速提至 **`20–25 km/h`**，包含 **3–4 次明显急加速与重刹（`>= 1.5 m/s²`）** 及 **2 组对称“8”字转向（`|\omega_z| >= 0.45 rad/s`）**，可直接拉高单路 `L2I` 航向与水平杆臂信噪比，抑制 `30 Hz` EKF 带宽衰减。
2. **硬件 CAD / 激光测距先验锚定（最高确定性）**：利用整车机械 CAD 图纸或激光测距仪实测两雷达相对 IMU 的 `(t_x, t_y, t_z)` 安装尺寸，对平路物理不可观的 $z_{\text{cm}}$ 及低速弱激励下的共模水平杆臂 $\bar{\mathbf{t}}_{xy}$ 施加硬先验（`z_cm_prior_m` / `t_cm_prior`）。

