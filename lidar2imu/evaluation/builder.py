from __future__ import annotations

from typing import Any

import numpy as np

from lidar2imu.algorithms import transform_delta_metrics, yaw_roll_pitch_from_matrix
from lidar2imu.evaluation.assessment import (
    build_final_acceptance,
    build_motion_assessment,
)
from lidar2imu.evaluation.cloud_thickness import evaluate_cloud_thickness
from lidar2imu.evaluation.consistency import (
    basin_stability_status,
    coarse_status,
    coarse_status_min,
    extraction_consistency_status,
    initial_prior_nominal_status,
    reference_consistency_status,
    transform_consistency_failure_reasons,
)
from lidar2imu.evaluation.diagnostics import (
    ground_diagnostics,
    motion_diagnostics,
    motion_registration_status,
    turn_balance_status,
)
from lidar2imu.evaluation.holdout import (
    build_holdout_validation,
    combined_holdout_generalization_status,
)
from lidar2imu.models import CalibrationConfig, CalibrationDataset


def _joint_fisher_metrics(stages: dict) -> dict:
    observability = (stages.get("joint") or {}).get("observability") or {}
    singular_values = [
        float(value)
        for value in observability.get("singular_values", [])
        if float(value) > 1e-9
    ]
    min_singular_value = None if not singular_values else float(min(singular_values))
    return {
        "rank": int(observability.get("rank", len(singular_values))),
        "expected_rank": int(observability.get("expected_rank", len(singular_values))),
        "condition_number": observability.get("condition_number"),
        "min_singular_value": min_singular_value,
        "min_eigenvalue": (
            None
            if min_singular_value is None
            else float(min_singular_value * min_singular_value)
        ),
        "degenerate": bool(observability.get("degenerate", False)),
    }


def _fisher_observability_status(
    fisher_metrics: dict,
    config: CalibrationConfig,
) -> dict:
    rank = int(fisher_metrics["rank"])
    expected_rank = int(fisher_metrics["expected_rank"])
    rank_warning = rank < expected_rank
    min_eigen_status = coarse_status_min(
        fisher_metrics["min_eigenvalue"],
        config.metrics_fisher_min_eigenvalue,
    )
    condition_status = coarse_status(
        fisher_metrics["condition_number"],
        config.metrics_fisher_max_condition_number,
    )
    if rank_warning:
        min_eigen_status = "warning"
        condition_status = "warning"
    if min_eigen_status == "pass" and condition_status == "pass":
        combined = "pass"
    elif "warning" in (min_eigen_status, condition_status):
        combined = "warning"
    else:
        combined = "unknown"
    return {
        "min_eigenvalue": min_eigen_status,
        "conditioning": condition_status,
        "combined": combined,
        "rank_warning": rank_warning,
    }


def build_metrics_output(
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    initial_transform: np.ndarray,
    stages: dict,
    config: CalibrationConfig,
    output_dir: str,
    basin_stability: dict | None = None,
    full_prior_robustness: dict | None = None,
    holdout_repeatability: dict | None = None,
    full_dataset: CalibrationDataset | None = None,
    holdout_dataset: CalibrationDataset | None = None,
    holdout_plan: dict | None = None,
) -> tuple[dict, dict]:
    ground_per_sample, ground_summary = ground_diagnostics(dataset, final_transform)
    motion_per_sample, motion_summary = motion_diagnostics(dataset, final_transform)
    holdout_motion_per_sample: list[dict[str, Any]] = []
    holdout_motion_summary = None
    if holdout_dataset is not None:
        holdout_motion_per_sample, holdout_motion_summary = motion_diagnostics(
            holdout_dataset, final_transform
        )
    holdout_validation = build_holdout_validation(
        motion_summary,
        holdout_motion_summary,
        holdout_plan,
        config,
    )
    fisher_metrics = _joint_fisher_metrics(stages)
    fisher_statuses = _fisher_observability_status(fisher_metrics, config)
    cloud_thickness_summary, cloud_thickness_window_frames = evaluate_cloud_thickness(
        calibration_dataset=dataset,
        holdout_dataset=holdout_dataset,
        final_transform=final_transform,
        config=config,
        enable_expensive_metrics=bool(output_dir),
    )
    holdout_generalization = combined_holdout_generalization_status(
        holdout_validation, holdout_repeatability
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
        "cloud_thickness_holdout_p95_m": cloud_thickness_summary.get(
            "representative_thickness_p95_m"
        ),
        "cloud_thickness_ground_p95_m": (
            None
            if cloud_thickness_summary.get("ground_plane") is None
            else cloud_thickness_summary.get("ground_plane", {})
            .get("support", {})
            .get("thickness_p95_m")
        ),
        "cloud_thickness_wall_p95_m": (
            None
            if cloud_thickness_summary.get("wall_plane") is None
            else cloud_thickness_summary.get("wall_plane", {})
            .get("support", {})
            .get("thickness_p95_m")
        ),
        "fisher_rank": int(fisher_metrics["rank"]),
        "fisher_expected_rank": int(fisher_metrics["expected_rank"]),
        "fisher_min_singular_value": fisher_metrics["min_singular_value"],
        "fisher_min_eigenvalue": fisher_metrics["min_eigenvalue"],
        "fisher_condition_number": fisher_metrics["condition_number"],
        "left_turn_count": int(motion_summary["left_turn_count"]),
        "right_turn_count": int(motion_summary["right_turn_count"]),
        "turn_balance_ratio": float(motion_summary["turn_balance_ratio"]),
        "statuses": {
            "initial_prior_nominal_range": initial_prior_nominal_status(
                delta_to_initial, config
            ),
            "ground_orientation": coarse_status(
                ground_summary["normal_angle_deg"]["p95"],
                config.metrics_warning_rotation_deg,
            ),
            "ground_height": coarse_status(
                ground_summary["height_residual_m"]["p95"],
                config.metrics_warning_height_m,
            ),
            "motion_rotation": coarse_status(
                motion_summary["rotation_residual_deg"]["p95"],
                config.metrics_warning_rotation_deg,
            ),
            "motion_translation": coarse_status(
                motion_summary["translation_residual_m"]["p95"],
                config.metrics_warning_translation_m,
            ),
            "motion_registration": motion_registration_status(motion_summary, config),
            "turn_balance": turn_balance_status(motion_summary, config),
            "fisher_min_eigenvalue": fisher_statuses["min_eigenvalue"],
            "fisher_conditioning": fisher_statuses["conditioning"],
            "fisher_observability": fisher_statuses["combined"],
            "extraction_geometry": extraction_consistency_status(
                delta_to_extraction, config
            ),
            "trusted_reference": reference_consistency_status(
                delta_to_reference, config
            ),
            "planar_basin_stability": basin_stability_status(basin_stability),
            "full_prior_robustness": basin_stability_status(full_prior_robustness),
            "holdout_repeatability": (
                "unknown"
                if holdout_repeatability is None
                else str(holdout_repeatability.get("status", "unknown"))
            ),
            "holdout_generalization": holdout_generalization,
            "cloud_thickness": str(cloud_thickness_summary.get("status", "unknown")),
        },
    }

    motion_assessment = build_motion_assessment(
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
        holdout_repeatability=holdout_repeatability,
    )
    final_acceptance = build_final_acceptance(coarse_metrics)

    metrics_output = {
        "summary": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
            "run_profile": dataset.metadata.get("run_profile"),
            "solver_family": (stages.get("solver_family") or {}).get("name"),
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
                "holdout_offset": (
                    None if holdout_plan is None else holdout_plan.get("offset")
                ),
            },
            "final_acceptance_status": final_acceptance["status"],
            "release_ready": final_acceptance["release_ready"],
        },
        "coarse_metrics": coarse_metrics,
        "final_acceptance": final_acceptance,
        "vehicle_motion_assessment": motion_assessment,
        "fine_metrics": {
            "ground": ground_summary,
            "motion": motion_summary,
            "holdout_motion": holdout_motion_summary,
            "algorithm_stages": stages,
            "extraction_consistency": {
                "delta_to_extraction": delta_to_extraction,
                "status": extraction_consistency_status(delta_to_extraction, config),
                "failure_reasons": transform_consistency_failure_reasons(
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
                "status": reference_consistency_status(delta_to_reference, config),
                "failure_reasons": transform_consistency_failure_reasons(
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
            "initial_prior_assessment": motion_assessment.get(
                "initial_prior_assessment_details"
            ),
            "planar_basin_stability": basin_stability,
            "full_prior_robustness": full_prior_robustness,
            "holdout_validation": holdout_validation,
            "holdout_repeatability": holdout_repeatability,
            "cloud_thickness": cloud_thickness_summary,
            "fisher_observability": {
                **fisher_metrics,
                "statuses": fisher_statuses,
                "thresholds": {
                    "min_eigenvalue": float(config.metrics_fisher_min_eigenvalue),
                    "max_condition_number": float(
                        config.metrics_fisher_max_condition_number
                    ),
                },
            },
            "uncertainty_summary": (
                None
                if holdout_repeatability is None
                else holdout_repeatability.get("uncertainty_summary")
            ),
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
        "cloud_thickness_window_frames": cloud_thickness_window_frames,
        "cloud_thickness": cloud_thickness_summary,
        "observability": {
            stage_name: stage_payload.get("observability")
            for stage_name, stage_payload in stages.items()
            if isinstance(stage_payload, dict)
        },
    }
    return metrics_output, diagnostics
