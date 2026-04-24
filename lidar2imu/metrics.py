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
    circular_span_deg,
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


def _reference_consistency_status(
    delta_to_reference: dict | None, config: CalibrationConfig
) -> str:
    return _transform_consistency_status(
        delta=delta_to_reference,
        translation_warning_m=config.metrics_reference_warning_translation_m,
        rotation_warning_deg=config.metrics_reference_warning_rotation_deg,
        vertical_warning_m=config.metrics_reference_warning_vertical_m,
        vertical_warning_ratio=config.metrics_reference_warning_vertical_ratio,
    )


def _extraction_consistency_status(
    delta_to_extraction: dict | None, config: CalibrationConfig
) -> str:
    return _transform_consistency_status(
        delta=delta_to_extraction,
        translation_warning_m=config.metrics_extraction_warning_translation_m,
        rotation_warning_deg=config.metrics_extraction_warning_rotation_deg,
        vertical_warning_m=config.metrics_extraction_warning_vertical_m,
        vertical_warning_ratio=config.metrics_extraction_warning_vertical_ratio,
    )


def _transform_consistency_failure_reasons(
    delta: dict | None,
    *,
    translation_warning_m: float,
    rotation_warning_deg: float,
    vertical_warning_m: float,
    vertical_warning_ratio: float,
) -> list[str]:
    if delta is None:
        return []
    reasons = []
    if float(delta["translation_norm_m"]) > float(translation_warning_m):
        reasons.append("translation_norm")
    if float(delta["rotation_deg"]) > float(rotation_warning_deg):
        reasons.append("rotation")
    translation_xyz = delta.get("translation_xyz_m") or {}
    vertical_error = translation_xyz.get("z")
    if vertical_error is not None and abs(float(vertical_error)) > float(
        vertical_warning_m
    ):
        reasons.append("vertical_translation")
    vertical_error_ratio = delta.get("vertical_error_ratio")
    if vertical_error_ratio is not None and float(vertical_error_ratio) > float(
        vertical_warning_ratio
    ):
        reasons.append("vertical_translation_ratio")
    return reasons


def _transform_consistency_status(
    *,
    delta: dict | None,
    translation_warning_m: float,
    rotation_warning_deg: float,
    vertical_warning_m: float,
    vertical_warning_ratio: float,
) -> str:
    if delta is None:
        return "unknown"
    if not _transform_consistency_failure_reasons(
        delta,
        translation_warning_m=translation_warning_m,
        rotation_warning_deg=rotation_warning_deg,
        vertical_warning_m=vertical_warning_m,
        vertical_warning_ratio=vertical_warning_ratio,
    ):
        return "pass"
    return "warning"


def _reference_consistency_recommendations(
    delta_to_reference: dict | None,
    status: str,
    config: CalibrationConfig,
) -> list[str]:
    if delta_to_reference is None or status != "warning":
        return []
    recommendations = [
        "This run fits the internal motion objective but diverges from the trusted reference transform; review basin sensitivity before accepting it.",
        (
            "Re-run from the trusted reference and from nearby perturbed priors. "
            "If the optimizer lands in different planar basins, do not treat this as a production-ready free-planar result."
        ),
        (
            "Require cross-bag agreement, holdout map evidence, or an external measurement before overriding a trusted in-bag TF by "
            f">{config.metrics_reference_warning_translation_m:.2f} m / "
            f">{config.metrics_reference_warning_rotation_deg:.1f} deg."
        ),
    ]
    failure_reasons = _transform_consistency_failure_reasons(
        delta_to_reference,
        translation_warning_m=config.metrics_reference_warning_translation_m,
        rotation_warning_deg=config.metrics_reference_warning_rotation_deg,
        vertical_warning_m=config.metrics_reference_warning_vertical_m,
        vertical_warning_ratio=config.metrics_reference_warning_vertical_ratio,
    )
    if (
        "vertical_translation" in failure_reasons
        or "vertical_translation_ratio" in failure_reasons
    ):
        recommendations.append(
            "The conflict is materially vertical (`z`); treat this as a mount-height / trusted-TF check first, not just a generic translation mismatch."
        )
        recommendations.append(
            "Verify the trusted sensor height against tape-measure / CAD / installation data before allowing the optimizer to override it."
        )
    return recommendations


def _extraction_consistency_recommendations(
    delta_to_extraction: dict | None,
    status: str,
    config: CalibrationConfig,
) -> list[str]:
    if delta_to_extraction is None or status != "warning":
        return []
    recommendations = [
        "The final solve diverges from the transform used to build the exported ground and motion samples; this run may need a second extraction pass.",
        "Re-run record conversion using the refined transform as the extraction transform, then compare residuals, basin stability, and reference consistency.",
        (
            "If the second-pass extraction keeps or improves internal metrics while staying within "
            f"{config.metrics_extraction_warning_translation_m:.2f} m / "
            f"{config.metrics_extraction_warning_rotation_deg:.1f} deg across nearby starts, prefer the re-extracted result."
        ),
    ]
    failure_reasons = _transform_consistency_failure_reasons(
        delta_to_extraction,
        translation_warning_m=config.metrics_extraction_warning_translation_m,
        rotation_warning_deg=config.metrics_extraction_warning_rotation_deg,
        vertical_warning_m=config.metrics_extraction_warning_vertical_m,
        vertical_warning_ratio=config.metrics_extraction_warning_vertical_ratio,
    )
    if (
        "vertical_translation" in failure_reasons
        or "vertical_translation_ratio" in failure_reasons
    ):
        recommendations.append(
            "The extraction mismatch is materially vertical (`z`); rebuild samples with a corrected lidar height before trusting any refined result."
        )
    return recommendations


def _basin_stability_status(basin_stability: dict | None) -> str:
    if not basin_stability:
        return "unknown"
    return str(basin_stability.get("status", "unknown"))


def _safe_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    baseline = float(baseline)
    if abs(baseline) <= 1e-12:
        if abs(float(value)) <= 1e-12:
            return 1.0
        return None
    return float(value) / baseline


def _initial_prior_nominal_status(
    delta_to_initial: dict | None, config: CalibrationConfig
) -> str:
    if delta_to_initial is None:
        return "unknown"
    if float(delta_to_initial["translation_norm_m"]) <= float(
        config.metrics_initial_prior_nominal_translation_m
    ) and float(delta_to_initial["rotation_deg"]) <= float(
        config.metrics_initial_prior_nominal_rotation_deg
    ):
        return "pass"
    return "warning"


def _build_initial_prior_assessment(
    delta_to_initial: dict | None,
    *,
    recommendation: str,
    extraction_consistency: str,
    trusted_reference_consistency: str,
    basin_stability_status: str,
    holdout_generalization: str,
    config: CalibrationConfig,
) -> dict:
    if delta_to_initial is None:
        return {
            "status": "unknown",
            "primary_cause": "missing_initial_transform",
            "recommendations": [],
        }

    translation = float(delta_to_initial["translation_norm_m"])
    rotation = float(delta_to_initial["rotation_deg"])
    failure_reasons = []
    if translation > float(config.metrics_initial_prior_nominal_translation_m):
        failure_reasons.append("translation_outside_nominal_range")
    if rotation > float(config.metrics_initial_prior_nominal_rotation_deg):
        failure_reasons.append("rotation_outside_nominal_range")

    hard_failure_reasons = []
    if translation > float(config.metrics_initial_prior_max_recoverable_translation_m):
        hard_failure_reasons.append("translation_outside_recovery_range")
    if rotation > float(config.metrics_initial_prior_max_recoverable_rotation_deg):
        hard_failure_reasons.append("rotation_outside_recovery_range")

    if not failure_reasons:
        return {
            "status": "pass",
            "primary_cause": "within_nominal_range",
            "delta_to_initial": delta_to_initial,
            "failure_reasons": [],
            "thresholds": {
                "nominal_translation_m": float(
                    config.metrics_initial_prior_nominal_translation_m
                ),
                "nominal_rotation_deg": float(
                    config.metrics_initial_prior_nominal_rotation_deg
                ),
                "max_recoverable_translation_m": float(
                    config.metrics_initial_prior_max_recoverable_translation_m
                ),
                "max_recoverable_rotation_deg": float(
                    config.metrics_initial_prior_max_recoverable_rotation_deg
                ),
            },
            "recommendations": [],
        }

    stable_acceptance = (
        recommendation == "full_6dof_candidate"
        and extraction_consistency != "warning"
        and trusted_reference_consistency != "warning"
        and basin_stability_status != "warning"
        and holdout_generalization != "warning"
    )

    if not hard_failure_reasons and stable_acceptance:
        status = "recoverable"
        primary_cause = "recovered_from_poor_initial"
        recommendations = [
            "The provided TF sits outside the nominal production range, but this bag still converged to the accepted basin.",
            "Treat this as a recoverable prior, not as a generally safe installation prior; weaker bags may not recover the same way.",
            "When this status appears repeatedly, compare against the fixed baseline and a trusted/external measurement before promoting the user TF as acceptable.",
        ]
    else:
        status = "warning"
        primary_cause = (
            "outside_recovery_range"
            if hard_failure_reasons
            else "outside_nominal_range_with_acceptance_conflict"
        )
        recommendations = [
            "The provided TF is outside the nominal range and should be treated as suspicious on this run.",
            "Check frame definitions, installation measurements, and whether record-side extraction geometry was already built from a wrong TF.",
            "Do not trust a single calibration replay here; compare baseline, trusted reference, and alternative extraction candidates before accepting the result.",
        ]

    return {
        "status": status,
        "primary_cause": primary_cause,
        "delta_to_initial": delta_to_initial,
        "failure_reasons": failure_reasons + hard_failure_reasons,
        "thresholds": {
            "nominal_translation_m": float(
                config.metrics_initial_prior_nominal_translation_m
            ),
            "nominal_rotation_deg": float(
                config.metrics_initial_prior_nominal_rotation_deg
            ),
            "max_recoverable_translation_m": float(
                config.metrics_initial_prior_max_recoverable_translation_m
            ),
            "max_recoverable_rotation_deg": float(
                config.metrics_initial_prior_max_recoverable_rotation_deg
            ),
        },
        "recommendations": recommendations,
    }


def _build_holdout_validation(
    train_motion_summary: dict,
    holdout_motion_summary: dict | None,
    holdout_plan: dict | None,
    config: CalibrationConfig,
) -> dict:
    holdout_plan = dict(holdout_plan or {})
    if not holdout_plan.get("enabled") or holdout_motion_summary is None:
        return {
            "status": "unknown",
            "enabled": False,
            "strategy": holdout_plan.get("strategy"),
            "every_n": holdout_plan.get("every_n"),
            "calibration_motion_samples": holdout_plan.get(
                "calibration_motion_samples",
                train_motion_summary.get("sample_count", 0),
            ),
            "holdout_motion_samples": holdout_plan.get("holdout_motion_samples", 0),
            "reason": holdout_plan.get("reason", "holdout_not_available"),
            "recommendations": [],
        }

    train_rotation_p95 = train_motion_summary["rotation_residual_deg"]["p95"]
    train_translation_p95 = train_motion_summary["translation_residual_m"]["p95"]
    train_fitness_p05 = train_motion_summary["registration_fitness"]["p05"]
    train_rmse_p95 = train_motion_summary["registration_inlier_rmse"]["p95"]
    holdout_rotation_p95 = holdout_motion_summary["rotation_residual_deg"]["p95"]
    holdout_translation_p95 = holdout_motion_summary["translation_residual_m"]["p95"]
    holdout_fitness_p05 = holdout_motion_summary["registration_fitness"]["p05"]
    holdout_rmse_p95 = holdout_motion_summary["registration_inlier_rmse"]["p95"]

    rotation_ratio = _safe_ratio(holdout_rotation_p95, train_rotation_p95)
    translation_ratio = _safe_ratio(holdout_translation_p95, train_translation_p95)
    fitness_ratio = _safe_ratio(holdout_fitness_p05, train_fitness_p05)
    rmse_ratio = _safe_ratio(holdout_rmse_p95, train_rmse_p95)

    checks = {}
    if rotation_ratio is not None:
        checks["rotation_residual_ratio"] = (
            rotation_ratio <= config.metrics_holdout_max_rotation_residual_ratio
        )
    if translation_ratio is not None:
        checks["translation_residual_ratio"] = (
            translation_ratio <= config.metrics_holdout_max_translation_residual_ratio
        )
    if fitness_ratio is not None:
        checks["registration_fitness_ratio"] = (
            fitness_ratio >= config.metrics_holdout_min_registration_fitness_ratio
        )
    if rmse_ratio is not None:
        checks["registration_inlier_rmse_ratio"] = (
            rmse_ratio <= config.metrics_holdout_max_registration_inlier_rmse_ratio
        )

    if not checks:
        status = "unknown"
        primary_cause = "holdout_metrics_unavailable"
        recommendations: list[str] = []
    elif all(checks.values()):
        status = "pass"
        primary_cause = "stable_holdout_residuals"
        recommendations = []
    elif not checks.get("registration_fitness_ratio", True) or not checks.get(
        "registration_inlier_rmse_ratio", True
    ):
        status = "warning"
        primary_cause = "holdout_registration_gap"
        recommendations = [
            "Holdout motion factors register materially worse than the calibration subset; do not treat one split as production-ready free-planar evidence.",
            "Increase map support, strengthen holdout overlap, or improve motion-factor quality before trusting planar release.",
        ]
    else:
        status = "warning"
        primary_cause = "holdout_residual_gap"
        recommendations = [
            "Holdout residuals degrade materially relative to the calibration subset; require repeatability across splits or bags before accepting free x/y/yaw.",
            "Prefer map settings that keep holdout residual ratios near the calibration subset instead of optimizing only the in-sample objective.",
        ]

    return {
        "status": status,
        "enabled": True,
        "strategy": holdout_plan.get("strategy"),
        "every_n": holdout_plan.get("every_n"),
        "reason": holdout_plan.get("reason"),
        "primary_cause": primary_cause,
        "calibration_motion_samples": int(train_motion_summary["sample_count"]),
        "holdout_motion_samples": int(holdout_motion_summary["sample_count"]),
        "calibration": {
            "rotation_residual_p95_deg": train_rotation_p95,
            "translation_residual_p95_m": train_translation_p95,
            "registration_fitness_p05": train_fitness_p05,
            "registration_inlier_rmse_p95": train_rmse_p95,
            "selected_frame_strides": train_motion_summary.get(
                "selected_frame_strides"
            ),
            "translation_heading_span_deg": train_motion_summary.get(
                "translation_heading_span_deg"
            ),
        },
        "holdout": {
            "rotation_residual_p95_deg": holdout_rotation_p95,
            "translation_residual_p95_m": holdout_translation_p95,
            "registration_fitness_p05": holdout_fitness_p05,
            "registration_inlier_rmse_p95": holdout_rmse_p95,
            "selected_frame_strides": holdout_motion_summary.get(
                "selected_frame_strides"
            ),
            "translation_heading_span_deg": holdout_motion_summary.get(
                "translation_heading_span_deg"
            ),
        },
        "ratios": {
            "rotation_residual_p95": rotation_ratio,
            "translation_residual_p95": translation_ratio,
            "registration_fitness_p05": fitness_ratio,
            "registration_inlier_rmse_p95": rmse_ratio,
        },
        "thresholds": {
            "max_rotation_residual_ratio": float(
                config.metrics_holdout_max_rotation_residual_ratio
            ),
            "max_translation_residual_ratio": float(
                config.metrics_holdout_max_translation_residual_ratio
            ),
            "min_registration_fitness_ratio": float(
                config.metrics_holdout_min_registration_fitness_ratio
            ),
            "max_registration_inlier_rmse_ratio": float(
                config.metrics_holdout_max_registration_inlier_rmse_ratio
            ),
        },
        "checks": checks,
        "recommendations": recommendations,
    }


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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_yaw_diagnostic(
    dataset: CalibrationDataset,
    coarse_metrics: dict,
    motion_summary: dict,
    stages: dict,
    config: CalibrationConfig,
) -> dict:
    motion_rotation_stage = stages.get("motion_rotation", {})
    observability = motion_rotation_stage.get("observability") or {}
    reason_codes = list(observability.get("reasons", []))
    cost_scan = observability.get("cost_scan") or {}
    motion_registration_status = coarse_metrics["statuses"]["motion_registration"]
    turn_balance_status = coarse_metrics["statuses"]["turn_balance"]
    motion_mode = dataset.metadata.get("motion_registration_mode", "scan_to_scan")
    selection_strategy = dataset.metadata.get(
        "motion_selection_strategy", "uniform_or_unknown"
    )
    used_samples = int(motion_rotation_stage.get("used_samples", 0))
    sample_count = int(motion_summary["sample_count"])
    heading_span_deg = motion_summary.get("translation_heading_span_deg")
    axis_abs_mean = motion_summary.get("imu_rotation_axis_abs_mean_xyz") or {}
    axis_z_mean = axis_abs_mean.get("z")
    stride_values = list(motion_summary.get("selected_frame_strides", []))
    degenerate = bool(observability.get("degenerate", False))
    reliability_limiters = []
    if not motion_rotation_stage.get("success"):
        reliability_limiters.append("motion_rotation_stage_unsuccessful")
    if motion_registration_status != "pass":
        reliability_limiters.append("motion_registration_not_pass")
    if sample_count < config.min_motion_samples:
        reliability_limiters.append("motion_sample_count_below_minimum")
    if used_samples < config.min_motion_samples:
        reliability_limiters.append("used_motion_sample_count_below_minimum")

    if (
        motion_rotation_stage.get("success")
        and motion_registration_status == "pass"
        and sample_count >= config.min_motion_samples
        and used_samples >= config.min_motion_samples
    ):
        evaluation_reliability = "high"
    elif (
        motion_rotation_stage.get("success")
        and sample_count >= config.min_motion_samples
    ):
        evaluation_reliability = "medium"
    else:
        evaluation_reliability = "low"

    narrow_heading_span = (
        heading_span_deg is not None
        and heading_span_deg < config.metrics_max_yaw_5pct_span_deg
    )
    z_axis_dominant = axis_z_mean is not None and axis_z_mean >= 0.95
    flat_cost = "flat_cost_scan" in reason_codes
    wide_plateau = "wide_cost_plateau" in reason_codes
    zero_sensitivity = "zero_scalar_sensitivity" in reason_codes

    if degenerate:
        if evaluation_reliability != "high":
            primary_cause = "factor_quality_or_sample_count_limited"
        elif zero_sensitivity:
            primary_cause = "near_zero_yaw_information"
        elif narrow_heading_span or z_axis_dominant:
            primary_cause = "repetitive_local_motion"
        elif motion_mode == "scan_to_scan":
            primary_cause = "local_pair_objective_too_weak"
        elif motion_mode == "submap_to_submap":
            primary_cause = "local_submap_objective_too_weak"
        else:
            primary_cause = "yaw_objective_too_weak"
    elif turn_balance_status != "pass":
        primary_cause = "turn_imbalance_only"
    else:
        primary_cause = "yaw_supported"

    recommendations = []
    if evaluation_reliability != "high":
        recommendations.append(
            "Do not use this run alone for a free-planar decision; improve factor quality or sample support first."
        )
    if degenerate:
        recommendations.append(
            "Keep planar DOFs locked in auto mode; do not trust free x/y/yaw on this run."
        )
        if evaluation_reliability != "high":
            recommendations.extend(
                [
                    "Treat this yaw warning as provisional until LiDAR factor quality is stable enough to support a real decision.",
                    "First improve factor quality: tighten sync, reduce dynamic clutter, or use stronger submap factors before re-judging yaw.",
                ]
            )
        elif primary_cause == "near_zero_yaw_information":
            recommendations.extend(
                [
                    "Add motion segments with larger heading change between samples so the yaw stage has non-zero scalar sensitivity.",
                    "Do not rely on long straight segments or tiny per-window rotations when collecting calibration data.",
                ]
            )
        elif primary_cause == "repetitive_local_motion":
            recommendations.extend(
                [
                    "Increase horizontal diversity in the selected motion snippets; the current set is still too close to a single local mode.",
                    "Target wider translation heading coverage and preserve non-z-dominant rotation snippets instead of repeating near-pure-yaw local windows.",
                ]
            )
            if motion_mode == "scan_to_scan":
                recommendations.append(
                    "Try --motion-registration-mode submap_to_submap so longer-span structure survives registration."
                )
            else:
                recommendations.append(
                    "If repetitive local motion persists after submaps, move to scan-to-map or submap-to-map factors."
                )
        elif primary_cause == "local_pair_objective_too_weak":
            recommendations.extend(
                [
                    "Pairwise scan-to-scan factors are still too local for this bag; switch to --motion-registration-mode submap_to_submap first.",
                    "If yaw is still flat after submaps, move to a true map-based objective instead of tuning more short-window thresholds.",
                ]
            )
        elif primary_cause == "local_submap_objective_too_weak":
            recommendations.extend(
                [
                    "Even submap-to-submap is still too local for this bag; move next to scan-to-map or larger submap-to-map optimization.",
                    "Validate future free-planar runs by repeatability and trusted-prior drift, not by a single optimization result.",
                ]
            )
        else:
            recommendations.append(
                "Escalate from local factors to a more global map objective before attempting free planar release."
            )
        if flat_cost:
            recommendations.append(
                "A low yaw cost ratio means many yaw values fit almost equally well; treat that as weak information, not as a good optimum."
            )
        if wide_plateau:
            recommendations.append(
                "A wide 5% yaw plateau means the objective stays nearly flat around many candidate yaws; prefer stronger geometry over solver retuning."
            )
    elif primary_cause == "turn_imbalance_only":
        recommendations.extend(
            [
                "Yaw itself is locally supported, but planar release is still unsafe because the bag is one-sided in turn direction.",
                "Collect comparable left and right turns before trusting free x/y/yaw.",
            ]
        )
    else:
        recommendations.extend(
            [
                "Yaw evidence is strong enough for comparison runs on this bag.",
                "Before accepting free planar release, still check repeatability across bags and drift relative to the trusted prior.",
            ]
        )

    return {
        "degenerate": degenerate,
        "evaluation_reliability": evaluation_reliability,
        "trusted_for_planar_decision": evaluation_reliability == "high",
        "reliability_limiters": reliability_limiters,
        "primary_cause": primary_cause,
        "reason_codes": reason_codes,
        "evidence": {
            "motion_registration_status": motion_registration_status,
            "turn_balance_status": turn_balance_status,
            "motion_sample_count": sample_count,
            "used_motion_samples": used_samples,
            "motion_registration_mode": motion_mode,
            "motion_selection_strategy": selection_strategy,
            "selected_frame_strides": stride_values,
            "translation_heading_span_deg": heading_span_deg,
            "imu_rotation_axis_abs_mean_z": axis_z_mean,
            "max_cost_ratio": cost_scan.get("max_cost_ratio"),
            "within_5pct_span_deg": cost_scan.get("within_5pct_span_deg"),
            "scalar_sensitivity": observability.get("scalar_sensitivity"),
        },
        "recommendations": _dedupe_preserve_order(recommendations),
    }


def _build_motion_assessment(
    dataset: CalibrationDataset,
    coarse_metrics: dict,
    motion_summary: dict,
    stages: dict,
    config: CalibrationConfig,
    delta_to_initial: dict | None,
    delta_to_extraction: dict | None,
    delta_to_reference: dict | None,
    basin_stability: dict | None,
    full_prior_robustness: dict | None,
    holdout_validation: dict | None,
) -> dict:
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
    yaw_diagnostic = _build_yaw_diagnostic(
        dataset=dataset,
        coarse_metrics=coarse_metrics,
        motion_summary=motion_summary,
        stages=stages,
        config=config,
    )
    extraction_consistency = _extraction_consistency_status(delta_to_extraction, config)
    trusted_reference_consistency = _reference_consistency_status(
        delta_to_reference, config
    )
    extraction_consistency_recommendations = _extraction_consistency_recommendations(
        delta_to_extraction, extraction_consistency, config
    )
    reference_consistency_recommendations = _reference_consistency_recommendations(
        delta_to_reference, trusted_reference_consistency, config
    )
    extraction_failure_reasons = _transform_consistency_failure_reasons(
        delta_to_extraction,
        translation_warning_m=config.metrics_extraction_warning_translation_m,
        rotation_warning_deg=config.metrics_extraction_warning_rotation_deg,
        vertical_warning_m=config.metrics_extraction_warning_vertical_m,
        vertical_warning_ratio=config.metrics_extraction_warning_vertical_ratio,
    )
    reference_failure_reasons = _transform_consistency_failure_reasons(
        delta_to_reference,
        translation_warning_m=config.metrics_reference_warning_translation_m,
        rotation_warning_deg=config.metrics_reference_warning_rotation_deg,
        vertical_warning_m=config.metrics_reference_warning_vertical_m,
        vertical_warning_ratio=config.metrics_reference_warning_vertical_ratio,
    )
    basin_stability_status = _basin_stability_status(basin_stability)
    full_prior_robustness_status = _basin_stability_status(full_prior_robustness)
    holdout_generalization = "unknown"
    if holdout_validation is not None:
        holdout_generalization = str(holdout_validation.get("status", "unknown"))

    if (
        ground_support == "pass"
        and coarse_metrics["statuses"]["motion_registration"] == "pass"
        and coarse_metrics["statuses"]["motion_rotation"] == "pass"
        and coarse_metrics["statuses"]["motion_translation"] == "pass"
        and yaw_observability == "pass"
        and coarse_metrics["statuses"]["observability"] == "pass"
        and extraction_consistency != "warning"
        and trusted_reference_consistency != "warning"
        and basin_stability_status != "warning"
        and full_prior_robustness_status != "warning"
        and holdout_generalization != "warning"
    ):
        recommendation = "full_6dof_candidate"
    elif extraction_consistency == "warning":
        recommendation = "reextract_review"
    elif trusted_reference_consistency == "warning":
        recommendation = "reference_conflict_review"
    elif basin_stability_status == "warning":
        recommendation = "basin_sensitivity_review"
    elif full_prior_robustness_status == "warning":
        recommendation = "prior_sensitivity_review"
    elif holdout_generalization == "warning":
        recommendation = "holdout_review"
    elif ground_support == "pass":
        recommendation = "z_roll_pitch_priority"
    else:
        recommendation = "recollect_data"

    initial_prior_assessment = _build_initial_prior_assessment(
        delta_to_initial,
        recommendation=recommendation,
        extraction_consistency=extraction_consistency,
        trusted_reference_consistency=trusted_reference_consistency,
        basin_stability_status=basin_stability_status,
        holdout_generalization=holdout_generalization,
        config=config,
    )

    assessment = {
        "ground_support": ground_support,
        "motion_registration_quality": coarse_metrics["statuses"][
            "motion_registration"
        ],
        "turn_balance": coarse_metrics["statuses"]["turn_balance"],
        "yaw_observability": yaw_observability,
        "joint_observability": coarse_metrics["statuses"]["observability"],
        "extraction_consistency": extraction_consistency,
        "trusted_reference_consistency": trusted_reference_consistency,
        "planar_basin_stability": basin_stability_status,
        "full_prior_robustness": full_prior_robustness_status,
        "holdout_generalization": holdout_generalization,
        "initial_prior_assessment": initial_prior_assessment.get("status"),
        "recommendation": recommendation,
        "yaw_diagnostic": yaw_diagnostic,
    }
    if delta_to_initial is not None:
        assessment["delta_to_initial"] = delta_to_initial
        assessment["initial_prior_assessment_details"] = initial_prior_assessment
    if delta_to_extraction is not None:
        assessment["delta_to_extraction"] = delta_to_extraction
        assessment["extraction_transform_source"] = dataset.metadata.get(
            "extraction_transform_source"
        )
        assessment["extraction_consistency_details"] = {
            "failure_reasons": extraction_failure_reasons,
            "thresholds": {
                "translation_norm_m": float(
                    config.metrics_extraction_warning_translation_m
                ),
                "rotation_deg": float(config.metrics_extraction_warning_rotation_deg),
                "vertical_translation_m": float(
                    config.metrics_extraction_warning_vertical_m
                ),
                "vertical_translation_ratio": float(
                    config.metrics_extraction_warning_vertical_ratio
                ),
            },
        }
        assessment["extraction_consistency_recommendations"] = (
            extraction_consistency_recommendations
        )
    if delta_to_reference is not None:
        assessment["delta_to_reference"] = delta_to_reference
        assessment["reference_transform_source"] = dataset.metadata.get(
            "reference_transform_source"
        )
        assessment["reference_consistency_details"] = {
            "failure_reasons": reference_failure_reasons,
            "thresholds": {
                "translation_norm_m": float(
                    config.metrics_reference_warning_translation_m
                ),
                "rotation_deg": float(config.metrics_reference_warning_rotation_deg),
                "vertical_translation_m": float(
                    config.metrics_reference_warning_vertical_m
                ),
                "vertical_translation_ratio": float(
                    config.metrics_reference_warning_vertical_ratio
                ),
            },
        }
        assessment["reference_consistency_recommendations"] = (
            reference_consistency_recommendations
        )
    if basin_stability is not None:
        assessment["planar_basin_stability_details"] = basin_stability
    if full_prior_robustness is not None:
        assessment["full_prior_robustness_details"] = full_prior_robustness
    if holdout_validation is not None:
        assessment["holdout_validation_details"] = holdout_validation
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
        assessment["yaw_observability_reasons"] = solver_policy.get(
            "yaw_observability_reasons", []
        )
    return assessment


def build_metrics_output(
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    initial_transform: np.ndarray,
    stages: dict,
    config: CalibrationConfig,
    output_dir: str,
    basin_stability: dict | None = None,
    full_prior_robustness: dict | None = None,
    full_dataset: CalibrationDataset | None = None,
    holdout_dataset: CalibrationDataset | None = None,
    holdout_plan: dict | None = None,
) -> tuple[dict, dict]:
    ground_per_sample, ground_summary = _ground_diagnostics(dataset, final_transform)
    motion_per_sample, motion_summary = _motion_diagnostics(dataset, final_transform)
    holdout_motion_per_sample = []
    holdout_motion_summary = None
    if holdout_dataset is not None:
        holdout_motion_per_sample, holdout_motion_summary = _motion_diagnostics(
            holdout_dataset, final_transform
        )
    holdout_validation = _build_holdout_validation(
        motion_summary,
        holdout_motion_summary,
        holdout_plan,
        config,
    )
    yaw_deg, roll_deg, pitch_deg = yaw_roll_pitch_from_matrix(final_transform)
    delta_to_initial = transform_delta_metrics(initial_transform, final_transform)
    delta_to_extraction = None
    if dataset.extraction_transform is not None:
        delta_to_extraction = transform_delta_metrics(
            dataset.extraction_transform, final_transform
        )
    delta_to_reference = None
    if dataset.reference_transform is not None:
        delta_to_reference = transform_delta_metrics(
            dataset.reference_transform, final_transform
        )

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
            "initial_prior_nominal_range": _initial_prior_nominal_status(
                delta_to_initial, config
            ),
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
            "extraction_geometry": _extraction_consistency_status(
                delta_to_extraction, config
            ),
            "trusted_reference": _reference_consistency_status(
                delta_to_reference, config
            ),
            "planar_basin_stability": _basin_stability_status(basin_stability),
            "full_prior_robustness": _basin_stability_status(full_prior_robustness),
            "holdout_generalization": str(holdout_validation.get("status", "unknown")),
        },
    }

    motion_assessment = _build_motion_assessment(
        dataset=dataset,
        coarse_metrics=coarse_metrics,
        motion_summary=motion_summary,
        stages=stages,
        config=config,
        delta_to_initial=delta_to_initial,
        delta_to_extraction=delta_to_extraction,
        delta_to_reference=delta_to_reference,
        basin_stability=basin_stability,
        full_prior_robustness=full_prior_robustness,
        holdout_validation=holdout_validation,
    )

    metrics_output = {
        "summary": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
            "run_profile": dataset.metadata.get("run_profile"),
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
            "delta_to_extraction": delta_to_extraction,
            "delta_to_reference": delta_to_reference,
            "solver_policy": stages.get("solver_policy", {}),
            "dataset_partition": {
                "total_motion_samples": int(
                    len((full_dataset or dataset).motion_samples)
                ),
                "calibration_motion_samples": int(len(dataset.motion_samples)),
                "holdout_motion_samples": int(
                    0
                    if holdout_dataset is None
                    else len(holdout_dataset.motion_samples)
                ),
                "holdout_every_n": (
                    None if holdout_plan is None else holdout_plan.get("every_n")
                ),
                "holdout_enabled": bool(holdout_plan and holdout_plan.get("enabled")),
            },
        },
        "coarse_metrics": coarse_metrics,
        "vehicle_motion_assessment": motion_assessment,
        "fine_metrics": {
            "ground": ground_summary,
            "motion": motion_summary,
            "holdout_motion": holdout_motion_summary,
            "algorithm_stages": stages,
            "extraction_consistency": {
                "delta_to_extraction": delta_to_extraction,
                "status": _extraction_consistency_status(delta_to_extraction, config),
                "failure_reasons": _transform_consistency_failure_reasons(
                    delta_to_extraction,
                    translation_warning_m=config.metrics_extraction_warning_translation_m,
                    rotation_warning_deg=config.metrics_extraction_warning_rotation_deg,
                    vertical_warning_m=config.metrics_extraction_warning_vertical_m,
                    vertical_warning_ratio=config.metrics_extraction_warning_vertical_ratio,
                ),
                "thresholds": {
                    "translation_norm_m": float(
                        config.metrics_extraction_warning_translation_m
                    ),
                    "rotation_deg": float(
                        config.metrics_extraction_warning_rotation_deg
                    ),
                    "vertical_translation_m": float(
                        config.metrics_extraction_warning_vertical_m
                    ),
                    "vertical_translation_ratio": float(
                        config.metrics_extraction_warning_vertical_ratio
                    ),
                },
            },
            "reference_consistency": {
                "delta_to_reference": delta_to_reference,
                "status": _reference_consistency_status(delta_to_reference, config),
                "failure_reasons": _transform_consistency_failure_reasons(
                    delta_to_reference,
                    translation_warning_m=config.metrics_reference_warning_translation_m,
                    rotation_warning_deg=config.metrics_reference_warning_rotation_deg,
                    vertical_warning_m=config.metrics_reference_warning_vertical_m,
                    vertical_warning_ratio=config.metrics_reference_warning_vertical_ratio,
                ),
                "thresholds": {
                    "translation_norm_m": float(
                        config.metrics_reference_warning_translation_m
                    ),
                    "rotation_deg": float(
                        config.metrics_reference_warning_rotation_deg
                    ),
                    "vertical_translation_m": float(
                        config.metrics_reference_warning_vertical_m
                    ),
                    "vertical_translation_ratio": float(
                        config.metrics_reference_warning_vertical_ratio
                    ),
                },
            },
            "initial_prior_assessment": _build_initial_prior_assessment(
                delta_to_initial,
                recommendation=motion_assessment["recommendation"],
                extraction_consistency=motion_assessment["extraction_consistency"],
                trusted_reference_consistency=motion_assessment[
                    "trusted_reference_consistency"
                ],
                basin_stability_status=motion_assessment["planar_basin_stability"],
                holdout_generalization=motion_assessment["holdout_generalization"],
                config=config,
            ),
            "planar_basin_stability": basin_stability,
            "full_prior_robustness": full_prior_robustness,
            "holdout_validation": holdout_validation,
            "artifacts": {
                "output_dir": output_dir,
                "diagnostics_dir": f"{output_dir}/diagnostics",
            },
        },
    }
    diagnostics = {
        "ground_per_sample": ground_per_sample,
        "motion_per_sample": motion_per_sample,
        "holdout_motion_per_sample": holdout_motion_per_sample,
        "observability": {
            stage_name: stage_payload.get("observability")
            for stage_name, stage_payload in stages.items()
        },
    }
    return metrics_output, diagnostics
