---
audience: dev
stability: stable
last_tested: 2026-04-27
---

# LiDAR↔Camera 设计文档与参数说明

目标：为开发者和工程师描述实现细节、参数含义、预期行为与调参建议。

一、整体架构概览

遵循仓库统一模式：数据层（Extraction）→ 算法层（Optimization）→ 评估层（Evaluation）。

主要模块（文件）

- lidar2camera/cli.py — 命令行入口（--config、--write-default-config、--output-dir）。
- lidar2camera/reference_pipeline.py — 主流程：配对、提取、初猜、优化、L1O、构建指标、写输出。
- lidar2camera/io.py — 结果写出（calibrated_tf.yaml、metrics.yaml、diagn断文件、extrinsics 文件）。
- lidar2camera/metrics.py — 指标计算与门控逻辑（recommendation 决策）。
- lidar2camera/models.py — 数据/配置的 dataclass。
- lidar2camera/learning_based.py — 实验性 targetless 基线（保留为对比）。

二、数据提取（Extraction）细节

1. 配对：按文件 stem 匹配 image (*.png/*.jpg...) 与 *.pcd，生成 pose 列表。
2. 图像角点检测：cv2.findChessboardCorners + cornerSubPix（flags 包含 ADAPTIVE_THRESH 与 FAST_CHECK）。
3. LiDAR 侧板面提取：使用 Open3D 的 pcd.segment_plane(distance_threshold=config.plane_distance_threshold_m, ransac_n=3, num_iterations=1000)。
4. 板位坐标构建：通过板模板(_build_board_template)映射到分割平面 centroid 与投影 x/y 轴（以 gravity 投影作为 in-plane 方向假设）。
5. 提取诊断：记录 plane_inlier_count、plane_residual_rmse_m、skip_reason（如 plane_segmentation_failed、insufficient_plane_points、image_corners_not_found 等）。

三、初始变换选择

- 优先使用 config.initial_transform（若提供）。
- 否则对每个 pose 运行 cv2.solvePnP(..., flags=cv2.SOLVEPNP_IPPE) 生成单帧候选解，选取使全局 reprojection RMS 最小的候选解作为初猜。

四、优化器（Algorithm）

- 变量：3 个旋转矢量分量（rotvec）+ 3 个平移分量（tx,ty,tz），参数向量长度为 6。
- 求解器：scipy.optimize.least_squares(..., method='trf', jac='2-point')。
- 损失函数：可选（config.optimization.loss），默认 'huber'（鲁棒化离群点）。f_scale 控制 loss 的尺度。
- 控制：optimization_max_nfev 限制函数评估次数，防止过长收敛。
- 输出：包含 success/status/message/nfev/cost/jacobian_rank/jacobian_shape。IO 层尝试对雅可比矩阵求逆估计参数不确定性；失败时返回 None。

五、留一法（Leave-one-out）重复性评估

- 对每个 holdout pose，使用剩余 pose 优化得到子解并将其用于预测 holdout pose 的 reprojection RMS（holdout_rms_px）。
- 汇总所有 holdout trials，计算 holdout_rms_summary（mean/p95/max）和 delta_to_primary（candidate vs primary 的 translation_norm_m 与 rotation_deg）。
- 通过聚类计数（translation_threshold_m 与 rotation_threshold_deg）判定 distinct_solution_count，若 >1 则表明存在多解家族（warning）。
- 返回结构包含 trials、uncertainty_summary（translation_std、rotation_vector_std）与 recommendations。

六、指标与推荐逻辑（metrics.py）

输出结构（关键字段）

- summary: final_translation_m、final_euler_deg、initial_rms_px、final_rms_px、pose_count、delta_to_initial
- coarse_metrics: pose_count、initial_rms_px、final_rms_px、pose_reprojection_rms_p95_px、holdout_reprojection_rms_p95_px、statuses
- camera_calibration_assessment: recommendation（accepted_reference_candidate / repeatability_review / reference_quality_review / recollect_data）、各项 statuses、leave_one_out_details
- fine_metrics: per_pose_reprojection、leave_one_out_repeatability、uncertainty_summary、extraction、optimization、artifacts

决定推荐的核心规则（代码实现）

- statuses:
  - pose_count: pass if pose_count >= min_pose_count
  - reprojection: pass if final_rms_px <= metrics_warning_final_rms_px
  - pose_reprojection: pass if per-pose p95 <= metrics_warning_pose_rms_p95_px
  - holdout_pose: pass/unknown/warning 根据 L1O 的 holdout p95 与 metrics_warning_holdout_rms_px
  - pose_repeatability: 由 L1O 返回的 status

- recommendation:
  - 若 pose_count、reprojection、pose_reprojection 均为 pass，且 L1O 状态为 pass 或 unknown → accepted_reference_candidate
  - 若 pose_count 告警 → recollect_data
  - 若 L1O 为 warning → repeatability_review
  - 否则 → reference_quality_review

七、配置项（config.yaml）及默认值（来自 reference_pipeline.default_reference_config_payload）

示例（重要项与含义）：

- camera.intrinsics: 3x3 相机内参矩阵（必需或在 config 中指定）
- camera.distortion: 5 元径向/切向失真参数
- checkerboard.pattern_size: [cols, rows]（内角点数）
- checkerboard.square_size: 方格边长（单位：米）
- point_cloud.plane_dist_thresh: 平面分割距离阈值（米，默认 0.02）
- point_cloud.min_plane_points: 平面内点最小数量（默认 500）
- frames.parent / frames.child: 坐标系命名
- data_directory: 同步 image/.pcd 对的目录（默认 "calibration_data"）
- optimization.min_poses: 最少 accepted pose 数（默认 5）
- optimization.loss: 优化损失类型（默认 'huber'）
- optimization.f_scale: 损失尺度（默认 1.0）
- optimization.max_nfev: 最大评估次数（默认 200）
- metrics.warning_final_rms_px: final RMS 的警告阈值（默认 1.0）
- metrics.warning_pose_rms_p95_px: pose reprojection p95 警告阈值（默认 1.5）
- metrics.warning_holdout_rms_px: L1O holdout p95 警告阈值（默认 1.5）
- metrics.warning_repeatability_translation_m: 重复性 translation 警告阈值（默认 0.05 m）
- metrics.warning_repeatability_rotation_deg: 重复性 rotation 警告阈值（默认 1.0 deg）
- output.directory: 输出目录（默认 "calibration_output"）

八、可调参数与工程建议

- plane_dist_thresh：若分割不到板，适当放宽（增大）阈值；若误分割周围结构，收紧阈值。
- min_plane_points：在远距离或稀疏点云时降低该值，但会降低板位几何可靠性。
- optimization.loss / f_scale：对异常值敏感时使用 huber 或 cauchy，并调小 f_scale 以提高鲁棒性。
- metrics.* 阈值：作为工程门控，建议按相机分辨率、像素尺度与项目容忍度调整。

九、扩展点与注意事项

- 支持其他 target（ChArUco / AprilTag-grid）可提升角点稳定性与 pose 身份标识，建议作为后续演进。
- 若数据包含可信 in-record TF（例如车载 TF），在提取阶段优先使用可信 TF 做 extraction geometry。
- 对 targetless 方案（lidar2camera.learning_based）保留为实验路径，不应直接用于生产发布，直到经过重复性与不确定度评估。

十、测试与验收

- 使用 tools/run_lidar2camera_smoke.py 做快速合成数据自检（无需真实样本）。
- 对真实数据，建议至少在 3 个不同 session / 视角上重复执行以验证 cross-bag repeatability。

参考实现位置：
- 代码入口： lidar2camera/reference_pipeline.py::run_reference_calibration_from_config
- 指标： lidar2camera/metrics.py::build_metrics_output
- IO： lidar2camera/io.py::write_outputs


