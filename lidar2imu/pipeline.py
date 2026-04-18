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

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import (refine_joint_solution,
                                  solve_ground_translation,
                                  solve_roll_pitch_from_ground,
                                  solve_translation_from_motion,
                                  solve_yaw_from_motion,
                                  transform_from_components,
                                  yaw_roll_pitch_from_matrix)
from lidar2imu.metrics import build_metrics_output
from lidar2imu.models import CalibrationConfig, CalibrationDataset


def _count_turns(dataset: CalibrationDataset) -> tuple[int, int]:
    signed_yaw_deg = []
    for sample in dataset.motion_samples:
        try:
            signed_yaw_deg.append(
                float(
                    np.degrees(
                        R.from_matrix(sample.imu_delta_rotation).as_euler(
                            "ZYX", degrees=False
                        )[0]
                    )
                )
            )
        except ValueError:
            signed_yaw_deg.append(0.0)
    left_turn_count = int(sum(1 for value in signed_yaw_deg if value > 0.5))
    right_turn_count = int(sum(1 for value in signed_yaw_deg if value < -0.5))
    return left_turn_count, right_turn_count


def _resolve_planar_motion_policy(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    motion_rotation_stage: dict,
) -> dict:
    left_turn_count, right_turn_count = _count_turns(dataset)
    weak_planar_reasons = []
    if (
        min(left_turn_count, right_turn_count)
        < config.metrics_min_turn_count_per_direction
    ):
        weak_planar_reasons.append("turn_imbalance")
    if motion_rotation_stage.get("observability", {}).get("degenerate", False):
        weak_planar_reasons.append("yaw_rotation_degenerate")

    requested_policy = config.planar_motion_policy
    applied_policy = requested_policy
    if requested_policy == "auto":
        applied_policy = "freeze_xyyaw" if weak_planar_reasons else "free"

    return {
        "requested": requested_policy,
        "applied": applied_policy,
        "left_turn_count": left_turn_count,
        "right_turn_count": right_turn_count,
        "weak_planar_reasons": weak_planar_reasons,
        "locked_components": (
            ["yaw", "x", "y"] if applied_policy == "freeze_xyyaw" else []
        ),
    }


def run_calibration(
    dataset: CalibrationDataset,
    config: CalibrationConfig | None = None,
    output_dir: str | None = None,
) -> dict:
    config = config or CalibrationConfig()
    initial_transform = np.asarray(dataset.initial_transform, dtype=float)
    initial_yaw, initial_roll, initial_pitch = yaw_roll_pitch_from_matrix(
        initial_transform
    )
    initial_translation = np.asarray(initial_transform[:3, 3], dtype=float)

    ground_rotation, ground_orientation_stage = solve_roll_pitch_from_ground(
        dataset.ground_samples,
        initial_roll=initial_roll,
        initial_pitch=initial_pitch,
        config=config,
    )
    ground_translation, ground_translation_stage = solve_ground_translation(
        dataset.ground_samples,
        rotation=ground_rotation,
        initial_translation=initial_translation,
    )
    yaw, motion_rotation_stage = solve_yaw_from_motion(
        dataset.motion_samples,
        fixed_roll=ground_orientation_stage["parameters"]["roll_rad"],
        fixed_pitch=ground_orientation_stage["parameters"]["pitch_rad"],
        initial_yaw=initial_yaw,
        config=config,
    )
    motion_rotation = transform_from_components(
        yaw=yaw,
        roll=ground_orientation_stage["parameters"]["roll_rad"],
        pitch=ground_orientation_stage["parameters"]["pitch_rad"],
        translation=np.zeros(3, dtype=float),
    )[:3, :3]
    solver_policy = _resolve_planar_motion_policy(
        dataset=dataset,
        config=config,
        motion_rotation_stage=motion_rotation_stage,
    )
    translation_locked_axes = (
        (True, True, False)
        if solver_policy["applied"] == "freeze_xyyaw"
        else (False, False, False)
    )
    motion_translation_initial = np.asarray(ground_translation, dtype=float).reshape(3)
    if solver_policy["applied"] == "freeze_xyyaw":
        motion_translation_initial[:2] = initial_translation[:2]
    motion_translation, motion_translation_stage = solve_translation_from_motion(
        dataset.motion_samples,
        rotation=motion_rotation,
        initial_translation=motion_translation_initial,
        locked_axes=translation_locked_axes,
    )
    joint_yaw = initial_yaw if solver_policy["applied"] == "freeze_xyyaw" else yaw

    joint_initial = np.array(
        [
            joint_yaw,
            ground_orientation_stage["parameters"]["roll_rad"],
            ground_orientation_stage["parameters"]["pitch_rad"],
            motion_translation[0],
            motion_translation[1],
            motion_translation[2],
        ],
        dtype=float,
    )
    final_transform, joint_stage = refine_joint_solution(
        dataset.ground_samples,
        dataset.motion_samples,
        initial_params=joint_initial,
        config=config,
        locked_mask=(
            solver_policy["applied"] == "freeze_xyyaw",
            False,
            False,
            solver_policy["applied"] == "freeze_xyyaw",
            solver_policy["applied"] == "freeze_xyyaw",
            False,
        ),
    )
    stages = {
        "ground_orientation": ground_orientation_stage,
        "ground_translation": ground_translation_stage,
        "motion_rotation": motion_rotation_stage,
        "motion_translation": motion_translation_stage,
        "solver_policy": solver_policy,
        "joint": joint_stage,
    }
    metrics_output, evaluation_diagnostics = build_metrics_output(
        dataset=dataset,
        final_transform=final_transform,
        initial_transform=initial_transform,
        stages=stages,
        config=config,
        output_dir=output_dir or "",
    )

    return {
        "initial_transform": initial_transform,
        "final_transform": final_transform,
        "stages": stages,
        "metrics": metrics_output,
        "evaluation": evaluation_diagnostics,
    }
