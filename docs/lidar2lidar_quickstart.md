---
audience: user
stability: stable
P26-06-16
---

# LiDAR-to-LiDAR quick start

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[lidar2lidar]"
```

## 1. 采集

Recommended bag contents:

- all LiDAR raw `PointCloud2` topics to be calibrated. For raw4 rigs:
  - `/apollo/sensor/vanjeelidar/left_back/PointCloud2`
  - `/apollo/sensor/vanjeelidar/left_front/PointCloud2`
  - `/apollo/sensor/vanjeelidar/right_front/PointCloud2`
  - `/apollo/sensor/vanjeelidar/right_back/PointCloud2`
- `/tf_static`
- optional `/tf`

Scene guidance:

- prefer walls, corners, poles, facades, and curb geometry
- avoid long feature-poor segments
- avoid bags dominated by moving traffic

Before calibration, inspect available topics:

```bash
lidar2lidar-topics /path/to/record_or_record_dir
```

## 2. 运行

### 当前算法与入口

| 类别 | 当前实现 | 入口与边界 |
| --- | --- | --- |
| 点到面 ICP | `method=1`，多尺度局部配准 | `lidar2lidar-calibrate --method 1` / `lidar2lidar-auto --methods 1` |
| GICP | `method=2`，局部协方差配准；当前几何基线 | 同上，选 `2` |
| 点到点 ICP | `method=3`，欧氏距离局部配准 | 同上，选 `3` |
| Semantic L2L | `method=4`，地面/立面点到面、柱棱线点到线 | 同上，选 `4`；旧 PCD/auto 入口只有 XYZ，没有 intensity/ring |
| 运动手眼标定 | LiDAR 里程计相对运动 `AX=XB`；旋转 SVD、平移最小二乘及非线性精化 | `lidar2lidar-temporal`；需要充分的旋转/平移激励，不是另一种 ICP |
| 条件性 scan2map | 将上述配准用于局部地图、多窗口共识与 holdout | `lidar2lidar-scan2map`；不是独立配准算法，也不默认替代 scan2scan |
| Joint v1/v2/v3 | v1: GICP + 双路 L2I 各向异性图；v2: 用 v1 图结果初始化语义边再融合；v3: 追加转弯 IMU 预积分 BA | `lidar2imu calibrate-dual-joint --mode ...` |

FPFH + RANSAC 是未提供初值时的**粗初始化**，语义路径的 BEV FFT 是弱支持时的
粗初始化回退；loop closure 是多边融合，不应与精配准方法重复计数。
本仓没有可直接切换的 NDT、TEASER++ 或 MLCC 实现。
`--methods 4` 的普通 Open3D fitness/对应集是解后的统一几何评估，
**不是语义求解器的真实匹配**。需要完整四通道及特征调试时使用下述 joint v2 入口；
反射率通道还取决于输入是否实际提供有效 intensity。
Joint v1/v2 仅优化两个空间外参（`12` 维），时间偏置沿用已有 L2I；
不是在 v2 中重新联合估计时延。

For four-LiDAR perimeter-only topology (rectangle 4 edges, no diagonals in
solution), run with `workflow_raw4_perimeter_loop.yaml`. The initial
transforms are taken from `tf_static` (and fallback TF graph), and loop-closure
stage is enabled by `--loop-closure`.

```bash
lidar2lidar-auto \
  --record-path /path/to/record_or_record_dir \
  --workflow-yaml lidar2lidar/conf/workflow_raw4_perimeter_loop.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/raw4_loop_run \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --loop-closure \
  --save-merged-pcd
```

Reference run on this repository data:

```bash
lidar2lidar-auto \
  --record-path /mnt/synology/中集/2026-06-16 \
  --workflow-yaml lidar2lidar/conf/workflow_raw4_perimeter_loop.yaml \
  --conf-dir lidar2lidar/conf \
  --output-dir outputs/lidar2lidar/2026-06-16_raw4_perimeter_loop_sync40 \
  --sync-threshold-ms 40 \
  --min-overlap 0.15 \
  --methods 2 \
  --loop-closure \
  --save-merged-pcd
```

## 3. 评价

Start from the conclusion layer:

```bash
python - <<'PY'
import yaml
d = yaml.safe_load(open("outputs/lidar2lidar/raw4_loop_run/metrics.yaml"))
print(d["summary"]["final_acceptance_status"])
print(d["summary"]["release_ready"])
print(d["final_acceptance"]["recommendation"])
PY
```

Then inspect the data-quality layer:

```bash
cat outputs/lidar2lidar/raw4_loop_run/diagnostics/data_quality.yaml
cat outputs/lidar2lidar/raw4_loop_run/diagnostics/status_summary.csv
```

Finally inspect visual evidence:

```bash
cat outputs/lidar2lidar/raw4_loop_run/diagnostics/visualization_index.yaml
```

Key output files:

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/edge_metrics.csv`
- `diagnostics/skipped_edges.csv`
- `diagnostics/loop_closure.yaml`
- `diagnostics/merged_cloud_solution_colored.ply`
- `diagnostics/visualization_index.yaml`

If visual geometry is missing or `warning`, keep the run review-only even if the
optimizer converged.

## 4. 面元、柱/棱线与实际匹配调试

双雷达联合标定的语义 L2L 路径支持可选
`--save-l2l-feature-debug`（`--mode v2/v3/all`）。安装绘图依赖：

```bash
pip install -e ".[lidar2lidar,lidar2imu,diagnostics]"
```

复用已经提取的规范化数据和 L2I 运行结果，不需要重新运行 GRIL：

```bash
lidar2imu calibrate-dual-joint \
  --dataset-a /path/to/left_front/dataset/dataset.yaml \
  --run-dir-a /path/to/left_front/window_full_0_680s \
  --dataset-b /path/to/right_back/dataset/dataset.yaml \
  --run-dir-b /path/to/right_back/window_full_0_680s \
  --output-dir outputs/lidar2lidar/joint_feature_debug \
  --mode v2 --save-l2l-feature-debug
```

用浏览器直接打开
`diagnostics/l2l_features/index.html`，点击窗口进入离线交互页面。
选择 XY/XZ/YZ、特征通道和匹配阶段；点击源或目标特征查看 ID、
原始支撑点、匹配对象、门限、淘汰原因及 Huber 分量权重。
不需要服务器、显示器或外部网络。

每个窗口输出：

| 文件 | 内容 |
| --- | --- |
| `features.png` / `features_facade_pole.png` | 全部特征 / 仅立面和竖直线候选，标注部分 ID、法向、轴向 |
| `matches_initial.png` / `matches_stage_*.png` | 第一个匹配阶段 / 各阶段的真实对应关系 |
| `matches_final.png` / `matches_final_facade.png` / `matches_final_pole.png` | 最后实际完成求解阶段的冻结匹配，全通道 / 立面 / 柱棱线 |
| `overlay_before_after.png` / `overlay_3d.png` | 同坐标系、尺度和采样的标定前后叠加 |
| `residuals.png` | 各阶段匹配数量、优化后几何残差、软权重 |
| `features.npz` | 特征、支撑点索引、点云与扫描来源、逐阶段变换和匹配数组 |
| `correspondences.csv` | 所有源特征的候选目标、淘汰原因和是否实际参与求解 |
| `objects.yaml` / `object_matches.csv` | 诊断面片/线段聚合，以及底层匹配汇总出的关联覆盖率 |
| `metadata.yaml` | 同步差、提取前后数量、配置、残差、可观测性、联合图降权状态 |

`source` 为 sensor B，`target` 为 sensor A；
变换约定是 `p_target = R_target_source @ p_source + t_target_source`。
PNG 连线按空间与残差抽样，图上标注展示数/总数，CSV/NPZ 保留全部关系。
`matches_initial` 是粗初始化后的 stage 0，并不虚构粗初始化之前尚未建立的语义匹配；
输入初值及粗初始化后的变换分别记录在 `T_initial` 和 `T_coarse`。

当前检测的是局部面元和竖直线状特征，**不是完整圆柱识别**。
`Plane_*`、`PoleCandidate_*` 是仅供人读图的聚合，不改变求解器；
圆柱确认字段始终为 false。特征数量不等于实体面/柱数量。
门控 `-1` 表示未执行；Huber 是软降权，不能当作二值内点标签。
未完成优化阶段的候选匹配不算“参与了求解”。

失败求解窗口仍输出图集；筛选拒绝且已读取点云的窗口输出检测图但不运行配准；
因同步问题未读取的候选只列出原因，不伪造特征图。
总索引保留失败、弱支持、重力降权和多窗口共识异常状态，以及窗口间外参差异。
图集默认 `review-only`：还需独立窗口、初值敏感性和几何检查才能接受标定结果。

开发回归烟测：

```bash
python -m tools.run_lidar2lidar_feature_smoke
```
