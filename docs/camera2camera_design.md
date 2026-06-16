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

当前实现是 **固定内参 + target-aware stereo extraction + 全局 bundle-adjustment 风格联合优化**，
并同时支持：

- `checkerboard`
- `aprilgrid`
- `charuco`

其中 live capture 的推荐默认是 **AprilGrid**，离线 paired-image baseline 仍然完整保留。

核心流程：

1. 每对图像检测 target feature
2. 每个相机各自 `solvePnP`
3. 用每对样本生成 parent→child 相对位姿候选
4. 对 `checkerboard` 的 child 角点顺序做有限离散解析（`identity / flip_x / flip_y / flip_xy`）
5. 选共识初猜
6. 联合优化：
   - 一个全局 camera→camera 外参
   - 每对样本一个 board→parent pose
7. 做 leave-one-out repeatability 与 holdout 误差评估

这样做的原因：

- 比只调用一次 `stereoCalibrate` 更容易接入仓库现有的质量门禁
- 能显式输出每对样本的 residual / holdout / repeatability
- target detector 已经统一，后续继续演进 target 不需要打破 review surface

## 3. Extraction 细节

- 输入：`cameras.parent.image_directory` 与 `cameras.child.image_directory`
- 配对：按文件 stem 匹配
- target 检测：
  - `checkerboard`：优先 `findChessboardCornersSB`，回退 `findChessboardCorners + cornerSubPix`
  - `aprilgrid`：优先 `pupil_apriltags`，缺失时回退 OpenCV ArUco/AprilTag
  - `charuco`：OpenCV ChArUco detector
- 样本前置门控：
  - bbox 面积占比过小拒绝
  - 距边界 margin 过小拒绝
  - 任一相机 `solvePnP` 失败拒绝
  - 任一相机 `solvePnP` reprojection 过大拒绝
- 排序/匹配策略：
  - `checkerboard`：对 child 角点尝试四种离散翻转，并用跨样本相对位姿共识决策
  - `aprilgrid`：使用共同 tag ID 对齐，两路图像天然消除棋盘翻转歧义
  - `charuco`：使用共同 feature ID 对齐

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
- live 模式下优先使用 **AprilGrid**；它对部分遮挡、斜视角和姿态歧义更稳
- 若 checkerboard 顺序解析频繁翻转，优先升级到 **AprilGrid / ChArUco**

## 8. Live capture 工作流

当前 `camera2camera-calibrate` 已支持 live stereo capture：

1. 同时打开两路 camera source（RTSP 或 device）
2. 两路都检测到 target 后做稳定度判定
3. 仅在两路都稳定且都通过 coverage / novelty 门控时保存一对图像
4. 采集中持续输出：
   - 左右相机 side-by-side 对比画面
   - parent 热力图 / 多样性进度
   - child 热力图 / 多样性进度
   - 当前相机相对位姿面板（last accepted / provisional / final）
   - 当前这 1 对样本的 pose / 几何质量 review
   - provisional stereo review 结果
5. 当 provisional stereo 达到 `release_ready` 时，可自动停止并输出最终外参

这条链路复用了 `camera` 模块已有的：

- stability-gated auto capture
- coverage heatmap / diversity guidance
- GUI / headless live fallback

因此交互方式与单目 intrinsic capture 保持一致，不会出现两套风格完全不同的采集 UX。

### live 模式的逐对门控

为了更适合工厂操作员，live stereo 不再是“先盲采很多张再统一看结果”，而是每接收
1 对就立刻判断这 1 对是否值得留下：

1. target detection 必须成功
2. 双路稳定度必须达标
3. 双路 coverage / novelty 门控必须达标
4. 双路 pose solve 必须达标：
   - bbox 足够大
   - edge margin 足够安全
   - `solvePnP` 成功
   - reprojection RMS 低于阈值
5. 若已经有若干 accepted pairs，则当前 pair 还要和当前 stereo consensus
   一致，避免把明显错误或不同步的 pair 混进来

注：

- 对 `aprilgrid / charuco`，live 模式可以更早使用相对位姿共识做门控
- 对 `checkerboard`，live 模式会更保守：先确保单对 pose 质量，再把稳健的
  排序决策留给多对样本的 stereo provisional/final review

这样做的好处是：

- 采集员知道“这 1 对为什么被拒绝”
- debug 信息天然贴近操作动作，而不是只给离线工程师看
- 在数据还不够多时，也能先保证每一对都是高质量 seed

### live 调试提示的设计原则

调试提示直接围绕操作动作展开，而不是只暴露算法名词：

- `target lost` → 提醒用户调可见性、距离、眩光、模糊
- `bbox too small` → 提醒用户把板移近
- `near image edge` → 提醒用户把板移回画面内部
- `pose solve failed / reprojection high` → 提醒用户改善清晰度、姿态、稳定度
- `pose not novel` → 提醒用户改变深度 / tilt / 横向位置
- `inconsistent relative transform` → 提醒用户避免两路不一致或板子在两路间不同步运动

这些提示同时写入 live manifest 和 `debug/sample_<N>_review.yaml`，便于现场和事后复盘。

## 9. 后续演进方向

按照业界/开源优秀实践，下一步优先级建议是：

1. 增加跨 session repeatability 聚合，让多轮采集的稳定性也能自动量化
2. 增加多相机图优化，把 stereo baseline 扩展到 rig baseline
3. 增加 rectification / overlap 视觉复核图
4. 增加 live 采集阶段的 stereo-specific overlap / baseline observability 提示
