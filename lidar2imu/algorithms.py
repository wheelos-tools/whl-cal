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

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from lidar2imu.models import CalibrationConfig, GroundSample, MotionSample


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(array)
    if norm <= 1e-12:
        raise ValueError("Vector norm must be positive.")
    return array / norm


def normalize_plane(normal: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    unit_normal = normalize_vector(normal)
    scale = np.linalg.norm(np.asarray(normal, dtype=float).reshape(3))
    unit_offset = float(offset) / scale
    if unit_offset < 0.0:
        unit_normal = -unit_normal
        unit_offset = -unit_offset
    return unit_normal, unit_offset


def rotation_from_yaw_roll_pitch(yaw: float, roll: float, pitch: float) -> np.ndarray:
    return R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def yaw_roll_pitch_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    yaw, pitch, roll = R.from_matrix(matrix[:3, :3]).as_euler("ZYX", degrees=False)
    return float(yaw), float(roll), float(pitch)


def transform_from_components(
    yaw: float, roll: float, pitch: float, translation: np.ndarray
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_from_yaw_roll_pitch(yaw, roll, pitch)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def rotation_angle_degrees(rotation_matrix: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def transform_delta_metrics(
    initial_transform: np.ndarray, refined_transform: np.ndarray
) -> dict[str, float]:
    delta = refined_transform @ np.linalg.inv(initial_transform)
    return {
        "translation_norm_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": rotation_angle_degrees(delta[:3, :3]),
    }


def summarize_values(values: list[float] | np.ndarray) -> dict:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "p05": None,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "p05": float(np.percentile(array, 5)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def observability_from_matrix(matrix: np.ndarray, threshold: float = 1e-9) -> dict:
    if matrix.size == 0:
        return {
            "rank": 0,
            "condition_number": None,
            "singular_values": [],
            "degenerate": True,
        }
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    positive = singular_values[singular_values > threshold]
    if positive.size >= 2:
        condition_number = float(positive[0] / positive[-1])
    else:
        condition_number = float("inf")
    return {
        "rank": int(positive.size),
        "condition_number": condition_number,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "degenerate": bool(
            positive.size < min(matrix.shape) or not np.isfinite(condition_number)
        ),
    }


def _imu_up_vector(sample: GroundSample) -> np.ndarray:
    return -normalize_vector(sample.imu_gravity)


def _sample_weight(weight: float) -> float:
    return math.sqrt(max(float(weight), 1e-12))


def solve_roll_pitch_from_ground(
    samples: list[GroundSample],
    initial_roll: float,
    initial_pitch: float,
    config: CalibrationConfig,
) -> tuple[np.ndarray, dict]:
    if len(samples) < config.min_ground_samples:
        raise ValueError(
            f"Need at least {config.min_ground_samples} ground samples, got {len(samples)}."
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        roll, pitch = params
        rotation = rotation_from_yaw_roll_pitch(0.0, roll, pitch)
        values = []
        for sample in samples:
            predicted_up = rotation @ sample.lidar_plane_normal
            target_up = _imu_up_vector(sample)
            values.extend(
                _sample_weight(sample.weight) * np.cross(predicted_up, target_up)
            )
        return np.asarray(values, dtype=float)

    result = least_squares(
        residuals,
        x0=np.array([initial_roll, initial_pitch], dtype=float),
        loss=config.loss,
        f_scale=config.ground_normal_scale_rad,
    )
    roll, pitch = result.x
    rotation = rotation_from_yaw_roll_pitch(0.0, roll, pitch)
    angle_errors = []
    for sample in samples:
        predicted_up = rotation @ sample.lidar_plane_normal
        target_up = _imu_up_vector(sample)
        cosine = np.clip(np.dot(predicted_up, target_up), -1.0, 1.0)
        angle_errors.append(float(np.degrees(np.arccos(cosine))))

    return rotation, {
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
        "parameters": {
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
        },
        "residuals": {
            "normal_angle_deg": summarize_values(angle_errors),
        },
        "observability": observability_from_matrix(result.jac),
    }


def solve_ground_translation(
    samples: list[GroundSample], rotation: np.ndarray, initial_translation: np.ndarray
) -> tuple[np.ndarray, dict]:
    height_samples = [
        sample for sample in samples if sample.imu_ground_height is not None
    ]
    if not height_samples:
        return np.asarray(initial_translation, dtype=float).reshape(3), {
            "success": False,
            "skipped": True,
            "reason": "no_height_priors",
            "parameters": {
                "translation": [
                    float(value)
                    for value in np.asarray(initial_translation, dtype=float)
                    .reshape(3)
                    .tolist()
                ],
            },
        }

    design = []
    values = []
    for sample in height_samples:
        normal_imu = rotation @ sample.lidar_plane_normal
        weight = _sample_weight(sample.weight)
        design.append(weight * normal_imu)
        values.append(
            weight * (sample.lidar_plane_offset - float(sample.imu_ground_height))
        )

    design_matrix = np.asarray(design, dtype=float)
    value_vector = np.asarray(values, dtype=float)
    translation, _, _, _ = np.linalg.lstsq(design_matrix, value_vector, rcond=None)
    residuals = design_matrix @ translation - value_vector

    return translation, {
        "success": True,
        "skipped": False,
        "parameters": {
            "translation": [float(value) for value in translation.tolist()],
        },
        "residuals": {
            "height_linear_residual_m": summarize_values(np.abs(residuals)),
        },
        "observability": observability_from_matrix(design_matrix),
    }


def solve_yaw_from_motion(
    samples: list[MotionSample],
    fixed_roll: float,
    fixed_pitch: float,
    initial_yaw: float,
    config: CalibrationConfig,
) -> tuple[float, dict]:
    filtered_samples = []
    min_excitation_rad = np.radians(config.min_motion_rotation_deg)
    for sample in samples:
        imu_rot = np.linalg.norm(R.from_matrix(sample.imu_delta_rotation).as_rotvec())
        lidar_rot = np.linalg.norm(
            R.from_matrix(sample.lidar_delta_rotation).as_rotvec()
        )
        if max(imu_rot, lidar_rot) >= min_excitation_rad:
            filtered_samples.append(sample)

    if len(filtered_samples) < config.min_motion_samples:
        raise ValueError(
            f"Need at least {config.min_motion_samples} motion samples with angular excitation, "
            f"got {len(filtered_samples)}."
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        yaw = float(params[0])
        rotation = rotation_from_yaw_roll_pitch(yaw, fixed_roll, fixed_pitch)
        values = []
        for sample in filtered_samples:
            rotation_error = (
                sample.imu_delta_rotation
                @ rotation
                @ sample.lidar_delta_rotation.T
                @ rotation.T
            )
            values.extend(
                _sample_weight(sample.weight)
                * R.from_matrix(rotation_error).as_rotvec()
            )
        return np.asarray(values, dtype=float)

    result = least_squares(
        residuals,
        x0=np.array([initial_yaw], dtype=float),
        loss=config.loss,
        f_scale=config.motion_rotation_scale_rad,
    )
    yaw = float(result.x[0])
    rotation = rotation_from_yaw_roll_pitch(yaw, fixed_roll, fixed_pitch)
    angle_errors = []
    for sample in filtered_samples:
        rotation_error = (
            sample.imu_delta_rotation
            @ rotation
            @ sample.lidar_delta_rotation.T
            @ rotation.T
        )
        angle_errors.append(
            float(np.degrees(np.linalg.norm(R.from_matrix(rotation_error).as_rotvec())))
        )

    return yaw, {
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
        "used_samples": len(filtered_samples),
        "parameters": {
            "yaw_rad": yaw,
            "yaw_deg": float(np.degrees(yaw)),
        },
        "residuals": {
            "rotation_residual_deg": summarize_values(angle_errors),
        },
        "observability": observability_from_matrix(result.jac),
    }


def solve_translation_from_motion(
    samples: list[MotionSample],
    rotation: np.ndarray,
    initial_translation: np.ndarray,
    locked_axes: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, dict]:
    initial_translation = np.asarray(initial_translation, dtype=float).reshape(3)
    if not samples:
        return initial_translation, {
            "success": False,
            "skipped": True,
            "reason": "no_motion_samples",
            "parameters": {
                "translation": [float(value) for value in initial_translation.tolist()],
                "locked_axes": {
                    axis: bool(locked)
                    for axis, locked in zip(("x", "y", "z"), locked_axes)
                },
            },
        }

    design_rows = []
    value_rows = []
    for sample in samples:
        weight = _sample_weight(sample.weight)
        design_rows.append(
            weight * (sample.imu_delta_rotation - np.eye(3, dtype=float))
        )
        value_rows.append(
            weight
            * (rotation @ sample.lidar_delta_translation - sample.imu_delta_translation)
        )

    design_matrix = np.concatenate(design_rows, axis=0)
    value_vector = np.concatenate(value_rows, axis=0)
    translation = initial_translation.copy()
    locked_indices = [index for index, locked in enumerate(locked_axes) if locked]
    free_indices = [index for index, locked in enumerate(locked_axes) if not locked]
    if free_indices:
        reduced_design = design_matrix[:, free_indices]
        reduced_value = value_vector.copy()
        if locked_indices:
            reduced_value = reduced_value - (
                design_matrix[:, locked_indices] @ initial_translation[locked_indices]
            )
        solved_values, _, _, _ = np.linalg.lstsq(
            reduced_design, reduced_value, rcond=None
        )
        translation[np.asarray(free_indices, dtype=int)] = solved_values
    residual_vector = design_matrix @ translation - value_vector
    residual_norms = np.linalg.norm(residual_vector.reshape(-1, 3), axis=1)
    observability_matrix = (
        design_matrix[:, free_indices]
        if free_indices
        else np.zeros((design_matrix.shape[0], 0), dtype=float)
    )

    return translation, {
        "success": True,
        "skipped": False,
        "parameters": {
            "translation": [float(value) for value in translation.tolist()],
            "locked_axes": {
                axis: bool(locked) for axis, locked in zip(("x", "y", "z"), locked_axes)
            },
        },
        "residuals": {
            "translation_residual_m": summarize_values(residual_norms),
        },
        "observability": observability_from_matrix(observability_matrix),
    }


def refine_joint_solution(
    ground_samples: list[GroundSample],
    motion_samples: list[MotionSample],
    initial_params: np.ndarray,
    config: CalibrationConfig,
    locked_mask: tuple[bool, bool, bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
        False,
        False,
    ),
) -> tuple[np.ndarray, dict]:
    if not ground_samples:
        raise ValueError("Joint refinement requires at least one ground sample.")
    if not motion_samples:
        raise ValueError("Joint refinement requires at least one motion sample.")

    component_names = ("yaw", "roll", "pitch", "x", "y", "z")
    reference_params = np.asarray(initial_params, dtype=float).reshape(6)
    active_indices = [index for index, locked in enumerate(locked_mask) if not locked]

    def expand_params(params: np.ndarray) -> np.ndarray:
        full = reference_params.copy()
        if active_indices:
            full[np.asarray(active_indices, dtype=int)] = np.asarray(
                params, dtype=float
            )
        return full

    def residuals(params: np.ndarray) -> np.ndarray:
        yaw, roll, pitch, tx, ty, tz = expand_params(params)
        translation = np.array([tx, ty, tz], dtype=float)
        rotation = rotation_from_yaw_roll_pitch(yaw, roll, pitch)
        values = []

        for sample in ground_samples:
            predicted_up = rotation @ sample.lidar_plane_normal
            target_up = _imu_up_vector(sample)
            values.extend(
                _sample_weight(sample.weight)
                * np.cross(predicted_up, target_up)
                / config.ground_normal_scale_rad
            )
            if sample.imu_ground_height is not None:
                plane_offset_imu = sample.lidar_plane_offset - float(
                    predicted_up @ translation
                )
                values.append(
                    _sample_weight(sample.weight)
                    * (plane_offset_imu - float(sample.imu_ground_height))
                    / config.ground_height_scale_m
                )

        for sample in motion_samples:
            rotation_error = (
                sample.imu_delta_rotation
                @ rotation
                @ sample.lidar_delta_rotation.T
                @ rotation.T
            )
            values.extend(
                _sample_weight(sample.weight)
                * R.from_matrix(rotation_error).as_rotvec()
                / config.motion_rotation_scale_rad
            )
            translation_error = (
                sample.imu_delta_rotation - np.eye(3, dtype=float)
            ) @ translation - (
                rotation @ sample.lidar_delta_translation - sample.imu_delta_translation
            )
            values.extend(
                _sample_weight(sample.weight)
                * translation_error
                / config.motion_translation_scale_m
            )

        return np.asarray(values, dtype=float)

    optimize_initial = (
        reference_params[np.asarray(active_indices, dtype=int)]
        if active_indices
        else np.zeros(0, dtype=float)
    )
    result = least_squares(
        residuals,
        x0=optimize_initial,
        loss=config.loss,
        f_scale=1.0,
    )
    yaw, roll, pitch, tx, ty, tz = expand_params(result.x)
    transform = transform_from_components(
        yaw, roll, pitch, np.array([tx, ty, tz], dtype=float)
    )
    return transform, {
        "success": bool(result.success),
        "message": result.message,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
        "parameters": {
            "yaw_rad": float(yaw),
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "yaw_deg": float(np.degrees(yaw)),
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
            "translation": [float(tx), float(ty), float(tz)],
            "locked_components": [
                name for name, locked in zip(component_names, locked_mask) if locked
            ],
        },
        "observability": observability_from_matrix(result.jac),
    }
