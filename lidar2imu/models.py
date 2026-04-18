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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationConfig:
    min_ground_samples: int = 3
    min_motion_samples: int = 3
    min_motion_rotation_deg: float = 1.0
    loss: str = "huber"
    planar_motion_policy: str = "auto"
    ground_normal_scale_rad: float = 0.02
    ground_height_scale_m: float = 0.05
    motion_rotation_scale_rad: float = 0.02
    motion_translation_scale_m: float = 0.05
    metrics_warning_rotation_deg: float = 0.3
    metrics_warning_translation_m: float = 0.05
    metrics_warning_height_m: float = 0.03
    metrics_warning_condition_number: float = 1e4
    metrics_warning_registration_fitness: float = 0.55
    metrics_warning_registration_inlier_rmse_m: float = 0.25
    metrics_min_turn_count_per_direction: int = 1
