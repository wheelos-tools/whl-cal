---
audience: dev
stability: proposal
---

# LiDAR-to-IMU 模块目录与命名

[返回方案索引](../lidar2imu.md) ·
[GRIL 基线方法](07_gril_baseline.md) ·
[快速标定](../lidar2imu_quickstart.md)

## 目标与已采用的边界

`whl-cal` 是 Python distribution/repository 名称；在现有仓库中，各标定能力
直接作为 Python 包存在。LiDAR-to-IMU 的业务包因此统一为 `lidar2imu/`，
GRIL 仅作为该模块的算法实现目录，不再占用仓库根目录的 `gril/` 包名。
没有新建不对称的 `whl_cal/` 包装层；若未来要把所有标定能力统一迁到
`whl_cal.*` namespace，应作为独立的大版本迁移处理。

```text
lidar2imu/
├── adapters/                 # record / bag 解码，输出 canonical 数据
├── algorithms/
│   └── gril/                 # GRIL frontend、配置、batch/full runner
├── evaluation/               # input contract、比较、A/B review、摘要
├── reference/
│   └── gril/                 # frozen ROS reference 身份、runner、patches
├── models.py                 # CanonicalDataset 及数据模型
├── dataset_io.py             # canonical 文件读写与 hash
└── cli.py                    # CLI 解析与流水线编排，不承载算法实现

tools/
├── lidar2imu/                # 操作脚本，例如容器运行入口
└── gril_record_core_reader/  # Bzlmod wheelos_core Apollo record ingress

third_party/gril_native/      # ROS-free C++ GRIL 源码、构建与测试
outputs/gril/<experiment>/    # 忽略跟踪的 dataset、runs 和 review artifacts
```

职责约束：

- `adapters/` 只处理格式差异，不做算法性滤波、坐标修复或样本重排。
- `models.py` 与 `dataset_io.py` 定义稳定 canonical 边界，算法和评估共享它。
- `algorithms/gril/` 保持冻结算法行为；不导入 ROSbag / Apollo 解码实现，也不
  自行决定物理结果是否 accepted。
- `evaluation/` 读取稳定输入和运行结果，写诊断/判定，不反馈修改求解器。
- `reference/gril/` 是迁移 oracle，不是第二套生产算法。
- C++ ROS-free GRIL 留在 `third_party/gril_native/`；Python runner 只做参数、
  canonical 序列化、进程调用与 artifact manifest。
- 用户可见的独立操作脚本放在 `tools/`；复用的 Python API 留在包内。

## 文件与命令命名

- 用户只使用单一入口 `lidar2imu`；各操作通过子命令区分，避免额外别名。
- 数据文件名使用 `dataset.yaml`、`lidar.npz`、`imu.npz` 等 artifact 类型名；
  运行结果用现有 `GRIL_*_v1` schema 名，schema 变更时递增版本。
- 算法实现保留 `config.py`、`frontend_runner.py`、`full_runner.py`、
  `native_runner.py` 等文件名；它们位于 `algorithms/gril/` 后职责已明确，
  不在此次纯目录迁移中引入无行为收益的批量重命名。
- 泛化的 `validation.py` 已改为 `evaluation/migration_review.py`；
  frozen reference 文件归入 `reference/gril/identity.py` 和 `runner.py`；
  容器脚本归入 `tools/lidar2imu/run_gril_container.sh`。
- 运行目录名包含数据集和输入身份，例如
  `outputs/gril/0827_corrected_imu_baseline/run_01/`。`outputs/gril/` 是结果存储
  分类，不是 Python 包位置。

## 后续建议

这次移动不改变 canonical schema、GRIL 数学行为、原生可执行文件、CLI
子命令或 artifact schema。先用现有迁移测试确认包移动无行为变化；若之后
发现 `full_runner.py` 或 `native_runner.py` 的边界需要拆分，再根据具体职责
将它们改为 `full_calibration_runner.py`、`batch_solver_runner.py`，并单独验证
trace、manifest 和 repeatability 不变。避免一次性搬目录、改 API、改算法和改
产物格式，以便把回归归因到单一变化。
