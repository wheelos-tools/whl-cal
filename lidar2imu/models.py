#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created Date: 2026-02-09
# Author: daohu527

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroundSample:
    timestamp_ns: int
    lidar_plane_normal: np.ndarray
    lidar_plane_offset: float
    imu_gravity: np.ndarray
    imu_ground_height: float | None = None
    weight: float = 1.0
    sync_dt_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionSample:
    start_timestamp_ns: int
    end_timestamp_ns: int
    imu_delta_rotation: np.ndarray
    imu_delta_translation: np.ndarray
    lidar_delta_rotation: np.ndarray
    lidar_delta_translation: np.ndarray
    weight: float = 1.0
    sync_dt_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationDataset:
    parent_frame: str
    child_frame: str
    ground_samples: list[GroundSample]
    motion_samples: list[MotionSample]
    initial_transform: np.ndarray
    extraction_transform: np.ndarray | None = None
    reference_transform: np.ndarray | None = None
    motion_candidate_pool: list[MotionSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationConfig:
    min_ground_samples: int = 3
    min_motion_samples: int = 3
    min_motion_rotation_deg: float = 1.0
    loss: str = "huber"
    solver_family: str = "baseline"
    planar_motion_policy: str = "auto"
    probabilistic_weight_min: float = 0.25
    probabilistic_weight_max: float = 4.0
    probabilistic_observability_weight_min: float = 0.25
    probabilistic_observability_weight_max: float = 3.0
    imu_preintegration_translation_weight: float = 0.0
    imu_preintegration_translation_scale_m: float = 0.08
    imu_preintegration_min_confidence: float = 0.1
    nhc_prior_mode: str = "off"
    nhc_prior_weight: float = 1.0
    nhc_prior_lateral_scale_m: float = 0.08
    nhc_prior_vertical_scale_m: float = 0.05
    nhc_prior_min_forward_translation_m: float = 0.1
    solver_screen_min_registration_fitness: float | None = None
    solver_screen_max_registration_inlier_rmse_m: float | None = None
    ground_normal_scale_rad: float = 0.02
    ground_height_scale_m: float = 0.05
    motion_rotation_scale_rad: float = 0.02
    motion_translation_scale_m: float = 0.05
    metrics_warning_rotation_deg: float = 0.3
    metrics_warning_translation_m: float = 0.05
    metrics_warning_height_m: float = 0.03
    metrics_fisher_min_eigenvalue: float = 1.0
    metrics_fisher_max_condition_number: float = 1e4
    metrics_warning_registration_fitness: float = 0.55
    metrics_warning_registration_inlier_rmse_m: float = 0.25
    metrics_min_turn_count_per_direction: int = 1
    metrics_min_yaw_cost_ratio: float = 1.5
    metrics_max_yaw_5pct_span_deg: float = 45.0
    metrics_extraction_warning_translation_m: float = 0.1
    metrics_extraction_warning_rotation_deg: float = 2.0
    metrics_extraction_warning_vertical_m: float = 0.05
    metrics_extraction_warning_vertical_ratio: float = 0.05
    metrics_reference_warning_translation_m: float = 0.1
    metrics_reference_warning_rotation_deg: float = 2.0
    metrics_reference_warning_vertical_m: float = 0.05
    metrics_reference_warning_vertical_ratio: float = 0.05
    metrics_initial_prior_nominal_translation_m: float = 0.5
    metrics_initial_prior_nominal_rotation_deg: float = 10.0
    metrics_initial_prior_max_recoverable_translation_m: float = 1.0
    metrics_initial_prior_max_recoverable_rotation_deg: float = 20.0
    metrics_multistart_translation_perturbation_m: float = 0.25
    metrics_multistart_yaw_perturbation_deg: float = 10.0
    metrics_multistart_vertical_perturbation_m: float = 0.25
    metrics_multistart_roll_pitch_perturbation_deg: float = 3.0
    metrics_holdout_every_n: int = 3
    metrics_holdout_repeat_count: int = 3
    metrics_holdout_min_repeat_evaluations: int = 2
    metrics_holdout_min_motion_samples: int = 3
    metrics_holdout_max_rotation_residual_ratio: float = 1.5
    metrics_holdout_max_translation_residual_ratio: float = 1.5
    metrics_holdout_min_registration_fitness_ratio: float = 0.85
    metrics_holdout_max_registration_inlier_rmse_ratio: float = 1.5
    metrics_repeatability_warning_translation_m: float = 0.05
    metrics_repeatability_warning_rotation_deg: float = 0.3
    metrics_cloud_thickness_target_duration_sec: float = 5.0
    metrics_cloud_thickness_max_frames: int = 24
    metrics_cloud_thickness_voxel_size_m: float = 0.2
    metrics_cloud_thickness_production_m: float = 0.03
    metrics_cloud_thickness_warning_m: float = 0.05
