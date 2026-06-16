from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StereoCalibrationObservation:
    pose_id: str
    parent_image_path: str
    child_image_path: str
    parent_image_size_wh: tuple[int, int]
    child_image_size_wh: tuple[int, int]
    parent_image_points: np.ndarray
    child_image_points: np.ndarray
    object_points: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StereoCalibrationDataset:
    parent_frame: str
    child_frame: str
    parent_camera_matrix: np.ndarray
    parent_camera_distortion: np.ndarray
    child_camera_matrix: np.ndarray
    child_camera_distortion: np.ndarray
    observations: list[StereoCalibrationObservation]
    initial_transform: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StereoCalibrationConfig:
    parent_image_directory: str = "calibration_data/parent"
    child_image_directory: str = "calibration_data/child"
    parent_frame: str = "camera_parent"
    child_frame: str = "camera_child"
    target_type: str = "checkerboard"
    board_pattern_size: tuple[int, int] = (11, 8)
    board_square_size_m: float = 0.025
    extraction_min_bbox_area_ratio: float = 0.003
    extraction_min_edge_margin_px: float = 16.0
    extraction_max_pnp_reprojection_rms_px: float = 1.5
    extraction_max_candidate_translation_delta_m: float = 0.25
    extraction_max_candidate_rotation_delta_deg: float = 3.0
    min_pair_count: int = 8
    optimization_loss: str = "huber"
    optimization_f_scale: float = 1.0
    optimization_max_nfev: int = 300
    optimization_max_refinement_rounds: int = 2
    optimization_outlier_pair_rms_px: float = 2.5
    metrics_warning_final_rms_px: float = 1.0
    metrics_warning_pair_rms_p95_px: float = 1.5
    metrics_warning_holdout_rms_px: float = 1.5
    metrics_warning_epipolar_p95_px: float = 1.0
    metrics_warning_repeatability_translation_m: float = 0.02
    metrics_warning_repeatability_rotation_deg: float = 0.3
    metrics_min_leave_one_out_pair_count: int = 6
    metrics_warning_image_coverage_min_cells: int = 4
    metrics_warning_image_horizontal_span_ratio: float = 0.35
    metrics_warning_image_vertical_span_ratio: float = 0.35
    metrics_warning_depth_span_m: float = 0.3
    metrics_warning_tilt_span_deg: float = 8.0
    metrics_warning_accepted_pair_ratio: float = 0.5
