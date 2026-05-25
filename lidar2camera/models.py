from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferencePoseObservation:
    pose_id: str
    image_path: str
    pcd_path: str
    image_size_wh: tuple[int, int]
    image_points: np.ndarray
    object_points: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceCalibrationDataset:
    parent_frame: str
    child_frame: str
    camera_matrix: np.ndarray
    camera_distortion: np.ndarray
    observations: list[ReferencePoseObservation]
    initial_transform: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceCalibrationConfig:
    data_directory: str = "calibration_data"
    parent_frame: str = "camera"
    child_frame: str = "lidar"
    board_pattern_size: tuple[int, int] = (8, 6)
    board_square_size_m: float = 0.05
    plane_distance_threshold_m: float = 0.02
    min_plane_points: int = 500
    extraction_min_bbox_area_ratio: float = 0.0008
    extraction_min_edge_margin_px: float = 8.0
    extraction_max_plane_residual_rmse_m: float = 0.02
    extraction_reject_board_geometry_warnings: bool = True
    min_pose_count: int = 5
    optimization_loss: str = "huber"
    optimization_f_scale: float = 1.0
    optimization_max_nfev: int = 200
    metrics_warning_final_rms_px: float = 1.0
    metrics_warning_pose_rms_p95_px: float = 1.5
    metrics_warning_holdout_rms_px: float = 1.5
    metrics_warning_repeatability_translation_m: float = 0.05
    metrics_warning_repeatability_rotation_deg: float = 1.0
    metrics_min_leave_one_out_pose_count: int = 5
    metrics_warning_image_coverage_min_cells: int = 4
    metrics_warning_image_horizontal_span_ratio: float = 0.35
    metrics_warning_image_vertical_span_ratio: float = 0.35
    metrics_warning_depth_span_m: float = 0.3
    metrics_warning_tilt_span_deg: float = 8.0
    metrics_warning_plane_residual_rmse_m: float = 0.02
    metrics_warning_board_extent_ratio_min: float = 0.5
    metrics_warning_board_extent_ratio_max: float = 4.0
    metrics_warning_accepted_pair_ratio: float = 0.5
