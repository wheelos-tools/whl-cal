# LiDAR↔Camera 快速上手（用户文档）

目标：提供一份面向客户的简洁上手指南，包含快速运行、关键指标说明与验收建议。

先决条件

- 已安装 Python 3.8+。
- 已安装依赖（建议在虚拟环境中）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

快速运行（最少步骤）

1. 生成默认配置模板：

```bash
lidar2camera-calibrate --write-default-config --config config.yaml
```

2. 编辑 config.yaml：将 data_directory 指向同步的 image/.pcd 对（按文件 stem 配对），可选调整 output.directory 与 camera.intrinsics（若已知）。

3. 运行标定：

```bash
lidar2camera-calibrate --config config.yaml
# 可通过 --output-dir 覆盖输出目录
```

主要输出（在 output.directory）：

- calibrated_tf.yaml — 最终合并的外参
- metrics.yaml — 标准化评估（摘要 + coarse / fine 指标）
- diagnostics/reference_dataset.yaml — 提取到的 pose 列表与 skip 理由
- diagnostics/extraction.yaml / optimization.yaml / evaluation.yaml — 分阶段诊断
- diagnostics/reference_overlay.png — 视觉覆盖图（LiDAR 投影到图像）
- calibrated/*.yaml — 每个 pose（或最终）外参文件

首先检查（首轮）

1. 打开 metrics.yaml.summary.final_rms_px：像素级全局 RMS（越低越好）。
2. 打开 metrics.yaml.coarse_metrics.pose_reprojection_rms_p95_px：每个 pose reprojection 的 95% 百分位，反映边缘误差。 
3. 检查 camera_calibration_assessment.recommendation：
   - accepted_reference_candidate → 建议通过（库内默认门限）；
   - repeatability_review / reference_quality_review / recollect_data → 需按建议复采或调整。
4. 如有 leave-one-out（L1O）结果，查看 fine_metrics.leave_one_out_repeatability 与 camera_calibration_assessment.pose_repeatability。
5. 查看 diagnostics/reference_overlay.png 做目视检验（点云投影是否覆盖棋盘角点位置）。

默认验收门限（可在 config.metrics 中调整）

- final_rms_px ≤ 1.0 px （默认 warning_final_rms_px）
- pose_reprojection_rms_p95_px ≤ 1.5 px （默认 warning_pose_rms_p95_px）
- holdout（L1O）p95 ≤ 1.5 px（默认 warning_holdout_rms_px）
- 重复性阈值：translation ≤ 0.05 m，rotation ≤ 1.0°（用于判定解族聚类）

说明：这些是“告警/建议”阈值（warning），最终是否推广还需结合工程可接受度与业务场景。

常见失败原因与快速修复

- 报错 “Insufficient valid poses”：至少采集配置的 min_poses（默认 5）且保证多样化的板位（左右、远近、倾斜）。
- image_corners_not_found：检查棋盘格是否完整、曝光/模糊或遮挡，调整拍摄或增加样张。
- plane_segmentation_failed / insufficient_plane_points：检查点云中棋盘板是否可见并有足够点支持；靠近板或提高点云密度。
- final RMS 大：确认 camera intrinsics 是否正确；用 overlay 图检验配准方向，再决定是否重采集或调参。
- L1O 重复性差：增加 pose 多样性（视角、深度、倾斜），或改进 LiDAR 侧板面提取质量。

进阶：如何调整

- 在 config.yaml 中修改 metrics.* 阈值以匹配你的验收标准。
- 调整 point_cloud.plane_dist_thresh / min_plane_points 控制平面分割的严格度。
- 调整 optimization.loss（huber / linear / cauchy）与 f_scale 用于鲁棒性控制。

快速自检（无真实数据）

- 仓库提供轻量合成 smoke 测试：

```bash
# 在仓库根目录运行（确保 PYTHONPATH=.）
PYTHONPATH=. python3 tools/run_lidar2camera_smoke.py --poses 5
```

成功样例输出：

[INFO] Running optimizer...
[RESULT] translation_norm_m= 0.0
[RESULT] rotation_deg= 0.00000
[PASS] Smoke test passed — recovered transform is close to ground truth.

更多信息

- 设计与参数说明：docs/lidar2camera_design.md
- 设计背景与策略：context/lidar2camera_context.md
