from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import circular_span_deg, normalize_vector, summarize_values
from lidar2imu.evaluation.consistency import coarse_status, coarse_status_min
from lidar2imu.models import CalibrationConfig, CalibrationDataset


def ground_diagnostics(
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


def motion_diagnostics(
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
    axis_components_abs = []
    translation_heading_deg = []
    sync_deltas = []
    frame_strides = []

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
        translation_error = (sample.imu_delta_rotation - np.eye(3, dtype=float)) @ (
            translation
        ) - (rotation @ sample.lidar_delta_translation - sample.imu_delta_translation)
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

        imu_rotvec = R.from_matrix(sample.imu_delta_rotation).as_rotvec()
        imu_rotvec_norm = float(np.linalg.norm(imu_rotvec))
        imu_axis_abs = (
            np.abs(imu_rotvec / imu_rotvec_norm)
            if imu_rotvec_norm > 1e-12
            else np.zeros(3, dtype=float)
        )
        if excitation_trans > 1e-12:
            translation_heading_deg.append(
                float(
                    np.degrees(
                        np.arctan2(
                            sample.imu_delta_translation[1],
                            sample.imu_delta_translation[0],
                        )
                    )
                )
            )

        rotation_errors.append(rotation_error_deg)
        translation_errors.append(translation_error_norm)
        excitation_rotations.append(excitation_rot)
        excitation_translations.append(excitation_trans)
        signed_yaw_imu.append(yaw_imu)
        axis_components_abs.append(imu_axis_abs)

        record = {
            "start_timestamp_ns": int(sample.start_timestamp_ns),
            "end_timestamp_ns": int(sample.end_timestamp_ns),
            "rotation_residual_deg": rotation_error_deg,
            "translation_residual_m": translation_error_norm,
            "angular_excitation_deg": excitation_rot,
            "translation_excitation_m": excitation_trans,
            "imu_signed_yaw_deg": yaw_imu,
            "imu_rotation_axis_abs": {
                "x": float(imu_axis_abs[0]),
                "y": float(imu_axis_abs[1]),
                "z": float(imu_axis_abs[2]),
            },
            "weight": float(sample.weight),
        }
        if excitation_trans > 1e-12:
            record["imu_translation_heading_deg"] = float(translation_heading_deg[-1])
        registration_fitness = sample.metadata.get("registration_fitness")
        if registration_fitness is not None:
            record["registration_fitness"] = float(registration_fitness)
        registration_inlier_rmse = sample.metadata.get("registration_inlier_rmse")
        if registration_inlier_rmse is not None:
            record["registration_inlier_rmse"] = float(registration_inlier_rmse)
        if sample.sync_dt_ms is not None:
            sync_deltas.append(float(sample.sync_dt_ms))
            record["sync_dt_ms"] = float(sample.sync_dt_ms)
        frame_stride = sample.metadata.get("frame_stride")
        if frame_stride is not None:
            frame_stride = int(frame_stride)
            frame_strides.append(frame_stride)
            record["frame_stride"] = frame_stride
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
    axis_abs_mean = (
        np.mean(np.asarray(axis_components_abs, dtype=float), axis=0)
        if axis_components_abs
        else None
    )
    return per_sample, {
        "sample_count": len(dataset.motion_samples),
        "rotation_residual_deg": summarize_values(rotation_errors),
        "translation_residual_m": summarize_values(translation_errors),
        "angular_excitation_deg": summarize_values(excitation_rotations),
        "translation_excitation_m": summarize_values(excitation_translations),
        "translation_heading_deg": summarize_values(translation_heading_deg),
        "translation_heading_span_deg": circular_span_deg(translation_heading_deg),
        "imu_rotation_axis_abs_mean_xyz": (
            {
                "x": float(axis_abs_mean[0]),
                "y": float(axis_abs_mean[1]),
                "z": float(axis_abs_mean[2]),
            }
            if axis_abs_mean is not None
            else None
        ),
        "frame_stride": summarize_values(frame_strides),
        "selected_frame_strides": sorted(set(frame_strides)),
        "registration_fitness": summarize_values(registration_fitness_values),
        "registration_inlier_rmse": summarize_values(registration_inlier_rmse_values),
        "sync_dt_ms": summarize_values(sync_deltas),
        "left_turn_count": left_turn_count,
        "right_turn_count": right_turn_count,
        "turn_balance_ratio": turn_balance_ratio,
    }


def motion_registration_status(motion_summary: dict, config: CalibrationConfig) -> str:
    fitness_status = coarse_status_min(
        motion_summary["registration_fitness"]["p05"],
        config.metrics_warning_registration_fitness,
    )
    rmse_status = coarse_status(
        motion_summary["registration_inlier_rmse"]["p95"],
        config.metrics_warning_registration_inlier_rmse_m,
    )
    if "unknown" in (fitness_status, rmse_status):
        return "unknown"
    if fitness_status == "pass" and rmse_status == "pass":
        return "pass"
    return "warning"


def turn_balance_status(motion_summary: dict, config: CalibrationConfig) -> str:
    if motion_summary["sample_count"] <= 0:
        return "unknown"
    if (
        min(motion_summary["left_turn_count"], motion_summary["right_turn_count"])
        >= config.metrics_min_turn_count_per_direction
    ):
        return "pass"
    return "warning"
