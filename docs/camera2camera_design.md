---
audience: dev
stability: stable
P26-05-27
---

# Camera-to-camera design and parameters

目标：给开发者和工程师一条**量产可复核**的 camera↔camera 基线链路，并保持与仓库其它标定模块一致的输出面。

## 1. 整体架构

遵循仓库统一模式：**Extraction → Algorithm → Evaluation**。

- `camera2camera/cli.py` — CLI 入口
- `camera2camera/reference_pipeline.py` — 主流程：配置读取、配对、提取、初始化、联合优化、留一法评估
- `camera2camera/metrics.py` — 指标与门禁
- `camera2camera/io.py` — 稳定输出面
- `camera2camera/models.py` — dataclass
- `tools/run_camera2camera_smoke.py` — 合成 smoke

## 2. 当前 baseline 的方法选择

当前实现先落地 **checkerboard + 固定内参 + 全局 bundle-adjustment 风格联合优化**：

1. 每对图像检测棋盘角点
2. 每个相机各自 `solvePnP`
3. 用每对样本生成 parent→child 相对位姿候选
4. 对 child 角点顺序做有限离散解析（`identity / flip_x / flip_y / flip_xy`）
5. 选共识初猜
6. 联合优化：
   - 一个全局 camera→camera 外参
   - 每对样本一个 board→parent pose
7. 做 leave-one-out repeatability 与 holdout 误差评估

这样做的原因：

- 比只调用一次 `stereoCalibrate` 更容易接入仓库现有的质量门禁
- 能显式输出每对样本的 residual / holdout / repeatability
- 支持未来平滑演进到 ChArUco / AprilGrid，而不打破 review surface

## 3. Extraction 细节

- 输入：`cameras.parent.image_directory` 与 `cameras.child.image_directory`
- 配对：按文件 stem 匹配
- 棋盘检测：
  - 优先 `findChessboardCornersSB`
  - 回退 `findChessboardCorners + cornerSubPix`
- 样本前置门控：
  - bbox 面积占比过小拒绝
  - 距边界 margin 过小拒绝
  - 任一相机 `solvePnP` 失败拒绝
  - 任一相机 `solvePnP` reprojection 过大拒绝
- 棋盘顺序解析：
  - 对 child 角点尝试四种离散翻转
  - 以跨样本相对位姿共识选择最终顺序
  - 偏离共识过大的样本拒绝

## 4. Algorithm 细节

优化变量：

- 6 DoF 全局 `parent -> child` 外参
- 每对样本 6 DoF `board -> parent` pose

求解器：

- `scipy.optimize.least_squares`
- `method='trf'`
- `jac='2-point'`
- 默认鲁棒损失 `huber`

残差：

- parent 图像 reprojection residual
- child 图像 reprojection residual

额外迭代：

- 第一轮优化后统计每对 `combined_rms_px`
- 超出 `optimization.outlier_pair_rms_px` 的样本可被剔除
- 再优化一轮，形成更稳定的 production baseline

## 5. Evaluation / 门禁

当前主门禁：

- `final_rms_px`
- `per_pair_reprojection p95`
- `holdout_reprojection p95`
- `repeatability delta`
- `epipolar_error p95`
- `accepted_pair_ratio`
- `parent_image_coverage`
- `child_image_coverage`
- `pose_diversity`

最终放行仍然看：

- `metrics.yaml.summary.final_acceptance_status`
- `metrics.yaml.summary.release_ready`
- `final_acceptance.recommendation`

不是只看“优化收敛成功”。

## 6. 输出契约

保持仓库统一 review surface：

- `calibrated_tf.yaml`
- `metrics.yaml`
- `diagnostics/standardized_data.yaml`
- `diagnostics/data_quality.yaml`
- `diagnostics/acceptance_report.yaml`
- `diagnostics/status_summary.csv`
- `diagnostics/visualization_index.yaml`

以及 camera2camera 专有输出：

- `diagnostics/reference_dataset.yaml`
- `diagnostics/extraction.yaml`
- `diagnostics/optimization.yaml`
- `diagnostics/evaluation.yaml`
- `diagnostics/per_pair_reprojection.csv`
- `diagnostics/leave_one_out_trials.csv`
- `diagnostics/parent_image_coverage_heatmap.png`
- `diagnostics/child_image_coverage_heatmap.png`
- `diagnostics/pose_diversity_plot.png`
- `diagnostics/epipolar_previews/`

## 7. 工程建议

- 量产建议先固定内参，不要在 cam2cam 阶段重新漂移 intrinsic
- 同一批采样保持曝光、焦距、分辨率固定
- 棋盘要覆盖两路相机的不同图像区域，而不是只在中心抖动
- 深度、tilt、左右偏移都要变化
- 若 checkerboard 顺序解析频繁翻转，优先升级到 **ChArUco**

## 8. 后续演进方向

按照业界/开源优秀实践，下一步优先级建议是：

1. 增加 `ChArUco` target 支持，作为更稳的量产默认
2. 增加多相机图优化，把 stereo baseline 扩展到 rig baseline
3. 增加跨 session repeatability 聚合
4. 增加 rectification / overlap 视觉复核图

