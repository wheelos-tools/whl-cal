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

# isort: off
import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import (
    normalize_vector,
    summarize_values,
    transform_delta_metrics,
    yaw_roll_pitch_from_matrix,
)
from lidar2imu.models import CalibrationConfig, CalibrationDataset

# isort: on


def _coarse_status(value: float | None, warning_threshold: float) -> str:
    if value is None:
        return "unknown"
    if value <= warning_threshold:
        return "pass"
    return "warning"


def _coarse_status_min(value: float | None, minimum_threshold: float) -> str:
    if value is None:
        return "unknown"
    if value >= minimum_threshold:
        return "pass"
    return "warning"


def _ground_diagnostics(
    dataset: CalibrationDataset, transform: np.ndarray
) -> tuple[list[dict], dict]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    per_sample = []
    angle_errors = []
    height_residuals = []
    sync_deltas = []

    for sample in dataset.ground_samples:
        predicted_up = rotation @ sample.lidar_plane_normal
        target_up = -normalize_vector(sample.imu_gravity)
        cosine = np.clip(np.dot(predicted_up, target_up), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cosine)))
        angle_errors.append(angle_deg)

        record = {
            "timestamp_ns": int(sample.timestamp_ns),
            "normal_angle_deg": angle_deg,
            "weight": float(sample.weight),
        }
        if sample.sync_dt_ms is not None:
            sync_deltas.append(float(sample.sync_dt_ms))
            record["sync_dt_ms"] = float(sample.sync_dt_ms)
        if sample.imu_ground_height is not None:
            plane_offset_imu = sample.lidar_plane_offset - float(
                predicted_up @ translation
            )
            height_residual = plane_offset_imu - float(sample.imu_ground_height)
            height_residuals.append(float(height_residual))
            record["height_residual_m"] = float(height_residual)
        per_sample.append(record)

    return per_sample, {
        "sample_count": len(dataset.ground_samples),
        "height_prior_count": sum(
            1
            for sample in dataset.ground_samples
            if sample.imu_ground_height is not None
        ),
        "normal_angle_deg": summarize_values(angle_errors),
        "height_residual_m": summarize_values(np.abs(height_residuals)),
        "sync_dt_ms": summarize_values(sync_deltas),
    }


def _motion_diagnostics(
    dataset: CalibrationDataset, transform: np.ndarray
) -> tuple[list[dict], dict]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    per_sample = []
    rotation_errors = []
    translation_errors = []
    excitation_rotations = []
    excitation_translations = []
    signed_yaw_imu = []
    sync_deltas = []

    for sample in dataset.motion_samples:
        rotation_error = (
            sample.imu_delta_rotation
            @ rotation
            @ sample.lidar_delta_rotation.T
            @ rotation.T
        )
        rotation_error_deg = float(
            np.degrees(np.linalg.norm(R.from_matrix(rotation_error).as_rotvec()))
        )
        translation_error = (
            sample.imu_delta_rotation - np.eye(3, dtype=float)
        ) @ translation - (
            rotation @ sample.lidar_delta_translation - sample.imu_delta_translation
        )
        translation_error_norm = float(np.linalg.norm(translation_error))
        excitation_rot = float(
            np.degrees(
                max(
                    np.linalg.norm(
                        R.from_matrix(sample.imu_delta_rotation).as_rotvec()
                    ),
                    np.linalg.norm(
                        R.from_matrix(sample.lidar_delta_rotation).as_rotvec()
                    ),
                )
            )
        )
        excitation_trans = float(
            max(
                np.linalg.norm(sample.imu_delta_translation),
                np.linalg.norm(sample.lidar_delta_translation),
            )
        )
        try:
            yaw_imu = float(
                np.degrees(
                    R.from_matrix(sample.imu_delta_rotation).as_euler(
                        "ZYX", degrees=False
                    )[0]
                )
            )
        except ValueError:
            yaw_imu = 0.0

        rotation_errors.append(rotation_error_deg)
        translation_errors.append(translation_error_norm)
        excitation_rotations.append(excitation_rot)
        excitation_translations.append(excitation_trans)
        signed_yaw_imu.append(yaw_imu)

        record = {
            "start_timestamp_ns": int(sample.start_timestamp_ns),
            "end_timestamp_ns": int(sample.end_timestamp_ns),
            "rotation_residual_deg": rotation_error_deg,
            "translation_residual_m": translation_error_norm,
            "angular_excitation_deg": excitation_rot,
            "translation_excitation_m": excitation_trans,
            "imu_signed_yaw_deg": yaw_imu,
            "weight": float(sample.weight),
        }
        registration_fitness = sample.metadata.get("registration_fitness")
        if registration_fitness is not None:
            record["registration_fitness"] = float(registration_fitness)
        registration_inlier_rmse = sample.metadata.get("registration_inlier_rmse")
        if registration_inlier_rmse is not None:
            record["registration_inlier_rmse"] = float(registration_inlier_rmse)
        if sample.sync_dt_ms is not None:
            sync_deltas.append(float(sample.sync_dt_ms))
            record["sync_dt_ms"] = float(sample.sync_dt_ms)
        per_sample.append(record)

    left_turn_count = int(sum(1 for value in signed_yaw_imu if value > 0.5))
    right_turn_count = int(sum(1 for value in signed_yaw_imu if value < -0.5))
    registration_fitness_values = [
        float(sample.metadata["registration_fitness"])
        for sample in dataset.motion_samples
        if sample.metadata.get("registration_fitness") is not None
    ]
    registration_inlier_rmse_values = [
        float(sample.metadata["registration_inlier_rmse"])
        for sample in dataset.motion_samples
        if sample.metadata.get("registration_inlier_rmse") is not None
    ]
    turn_balance_ratio = 0.0
    if max(left_turn_count, right_turn_count) > 0:
        turn_balance_ratio = float(
            min(left_turn_count, right_turn_count)
            / max(left_turn_count, right_turn_count)
        )
    return per_sample, {
        "sample_count": len(dataset.motion_samples),
        "rotation_residual_deg": summarize_values(rotation_errors),
        "translation_residual_m": summarize_values(translation_errors),
        "angular_excitation_deg": summarize_values(excitation_rotations),
        "translation_excitation_m": summarize_values(excitation_translations),
        "registration_fitness": summarize_values(registration_fitness_values),
        "registration_inlier_rmse": summarize_values(registration_inlier_rmse_values),
        "sync_dt_ms": summarize_values(sync_deltas),
        "left_turn_count": left_turn_count,
        "right_turn_count": right_turn_count,
        "turn_balance_ratio": turn_balance_ratio,
    }


def _motion_registration_status(motion_summary: dict, config: CalibrationConfig) -> str:
    fitness_status = _coarse_status_min(
        motion_summary["registration_fitness"]["p05"],
        config.metrics_warning_registration_fitness,
    )
    rmse_status = _coarse_status(
        motion_summary["registration_inlier_rmse"]["p95"],
        config.metrics_warning_registration_inlier_rmse_m,
    )
    if "unknown" in (fitness_status, rmse_status):
        return "unknown"
    if fitness_status == "pass" and rmse_status == "pass":
        return "pass"
    return "warning"


def _turn_balance_status(motion_summary: dict, config: CalibrationConfig) -> str:
    if motion_summary["sample_count"] <= 0:
        return "unknown"
    if (
        min(motion_summary["left_turn_count"], motion_summary["right_turn_count"])
        >= config.metrics_min_turn_count_per_direction
    ):
        return "pass"
    return "warning"


def _build_motion_assessment(coarse_metrics: dict, stages: dict) -> dict:
    ground_support = "warning"
    if (
        coarse_metrics["ground_normal_angle_p95_deg"] is not None
        and coarse_metrics["ground_normal_angle_p95_deg"] <= 2.0
        and coarse_metrics["statuses"]["ground_height"] == "pass"
    ):
        ground_support = "pass"
    yaw_observability = "pass"
    motion_rotation_observability = stages["motion_rotation"].get("observability") or {}
    if motion_rotation_observability.get("degenerate", False):
        yaw_observability = "warning"
    if coarse_metrics["statuses"]["turn_balance"] != "pass":
        yaw_observability = "warning"

    if (
        ground_support == "pass"
        and coarse_metrics["statuses"]["motion_registration"] == "pass"
        and coarse_metrics["statuses"]["motion_rotation"] == "pass"
        and coarse_metrics["statuses"]["motion_translation"] == "pass"
        and yaw_observability == "pass"
        and coarse_metrics["statuses"]["observability"] == "pass"
    ):
        recommendation = "full_6dof_candidate"
    elif ground_support == "pass":
        recommendation = "z_roll_pitch_priority"
    else:
        recommendation = "recollect_data"

    assessment = {
        "ground_support": ground_support,
        "motion_registration_quality": coarse_metrics["statuses"][
            "motion_registration"
        ],
        "turn_balance": coarse_metrics["statuses"]["turn_balance"],
        "yaw_observability": yaw_observability,
        "joint_observability": coarse_metrics["statuses"]["observability"],
        "recommendation": recommendation,
    }
    solver_policy = stages.get("solver_policy", {})
    if solver_policy:
        assessment["requested_solver_planar_motion_policy"] = solver_policy.get(
            "requested"
        )
        assessment["applied_solver_planar_motion_policy"] = solver_policy.get("applied")
        assessment["solver_locked_components"] = solver_policy.get(
            "locked_components", []
        )
        assessment["weak_planar_reasons"] = solver_policy.get("weak_planar_reasons", [])
    return assessment


def build_metrics_output(
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    initial_transform: np.ndarray,
    stages: dict,
    config: CalibrationConfig,
    output_dir: str,
) -> tuple[dict, dict]:
    ground_per_sample, ground_summary = _ground_diagnostics(dataset, final_transform)
    motion_per_sample, motion_summary = _motion_diagnostics(dataset, final_transform)
    yaw_deg, roll_deg, pitch_deg = yaw_roll_pitch_from_matrix(final_transform)
    delta_to_initial = transform_delta_metrics(initial_transform, final_transform)

    coarse_metrics = {
        "ground_sample_count": int(ground_summary["sample_count"]),
        "ground_height_prior_count": int(ground_summary["height_prior_count"]),
        "motion_sample_count": int(motion_summary["sample_count"]),
        "ground_normal_angle_p95_deg": ground_summary["normal_angle_deg"]["p95"],
        "ground_height_residual_p95_m": ground_summary["height_residual_m"]["p95"],
        "motion_rotation_residual_p95_deg": motion_summary["rotation_residual_deg"][
            "p95"
        ],
        "motion_translation_residual_p95_m": motion_summary["translation_residual_m"][
            "p95"
        ],
        "motion_angular_excitation_p95_deg": motion_summary["angular_excitation_deg"][
            "p95"
        ],
        "motion_registration_fitness_p05": motion_summary["registration_fitness"][
            "p05"
        ],
        "motion_registration_inlier_rmse_p95": motion_summary[
            "registration_inlier_rmse"
        ]["p95"],
        "left_turn_count": int(motion_summary["left_turn_count"]),
        "right_turn_count": int(motion_summary["right_turn_count"]),
        "turn_balance_ratio": float(motion_summary["turn_balance_ratio"]),
        "joint_condition_number": stages["joint"]["observability"]["condition_number"],
        "statuses": {
            "ground_orientation": _coarse_status(
                ground_summary["normal_angle_deg"]["p95"],
                config.metrics_warning_rotation_deg,
            ),
            "ground_height": _coarse_status(
                ground_summary["height_residual_m"]["p95"],
                config.metrics_warning_height_m,
            ),
            "motion_rotation": _coarse_status(
                motion_summary["rotation_residual_deg"]["p95"],
                config.metrics_warning_rotation_deg,
            ),
            "motion_translation": _coarse_status(
                motion_summary["translation_residual_m"]["p95"],
                config.metrics_warning_translation_m,
            ),
            "motion_registration": _motion_registration_status(motion_summary, config),
            "turn_balance": _turn_balance_status(motion_summary, config),
            "observability": _coarse_status(
                stages["joint"]["observability"]["condition_number"],
                config.metrics_warning_condition_number,
            ),
        },
    }

    metrics_output = {
        "summary": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
            "final_translation_m": {
                "x": float(final_transform[0, 3]),
                "y": float(final_transform[1, 3]),
                "z": float(final_transform[2, 3]),
            },
            "final_euler_deg": {
                "yaw": float(np.degrees(yaw_deg)),
                "roll": float(np.degrees(roll_deg)),
                "pitch": float(np.degrees(pitch_deg)),
            },
            "delta_to_initial": delta_to_initial,
            "solver_policy": stages.get("solver_policy", {}),
        },
        "coarse_metrics": coarse_metrics,
        "vehicle_motion_assessment": _build_motion_assessment(coarse_metrics, stages),
        "fine_metrics": {
            "ground": ground_summary,
            "motion": motion_summary,
            "algorithm_stages": stages,
            "artifacts": {
                "output_dir": output_dir,
                "diagnostics_dir": f"{output_dir}/diagnostics",
            },
        },
    }
    diagnostics = {
        "ground_per_sample": ground_per_sample,
        "motion_per_sample": motion_per_sample,
        "observability": {
            stage_name: stage_payload.get("observability")
            for stage_name, stage_payload in stages.items()
        },
    }
    return metrics_output, diagnostics
