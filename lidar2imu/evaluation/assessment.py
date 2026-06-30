from __future__ import annotations

from calibration_common.evaluation import (
    build_final_acceptance as _build_final_acceptance,
)
from lidar2imu.evaluation.consistency import (
    basin_stability_status,
    build_initial_prior_assessment,
    extraction_consistency_recommendations,
    extraction_consistency_status,
    reference_consistency_recommendations,
    reference_consistency_status,
    transform_consistency_failure_reasons,
)
from lidar2imu.evaluation.holdout import combined_holdout_generalization_status
from lidar2imu.models import CalibrationConfig, CalibrationDataset


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_yaw_diagnostic(
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
        "recommendations": dedupe_preserve_order(recommendations),
    }


def build_motion_assessment(
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
    holdout_repeatability: dict | None,
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
    yaw_diagnostic = build_yaw_diagnostic(
        dataset=dataset,
        coarse_metrics=coarse_metrics,
        motion_summary=motion_summary,
        stages=stages,
        config=config,
    )
    extraction_consistency = extraction_consistency_status(delta_to_extraction, config)
    trusted_reference_consistency = reference_consistency_status(
        delta_to_reference, config
    )
    extraction_consistency_recs = extraction_consistency_recommendations(
        delta_to_extraction, extraction_consistency, config
    )
    reference_consistency_recs = reference_consistency_recommendations(
        delta_to_reference, trusted_reference_consistency, config
    )
    extraction_failure_reasons = transform_consistency_failure_reasons(
        delta_to_extraction,
        translation_warning_m=config.metrics_extraction_warning_translation_m,
        rotation_warning_deg=config.metrics_extraction_warning_rotation_deg,
        vertical_warning_m=config.metrics_extraction_warning_vertical_m,
        vertical_warning_ratio=config.metrics_extraction_warning_vertical_ratio,
    )
    reference_failure_reasons = transform_consistency_failure_reasons(
        delta_to_reference,
        translation_warning_m=config.metrics_reference_warning_translation_m,
        rotation_warning_deg=config.metrics_reference_warning_rotation_deg,
        vertical_warning_m=config.metrics_reference_warning_vertical_m,
        vertical_warning_ratio=config.metrics_reference_warning_vertical_ratio,
    )
    basin_status = basin_stability_status(basin_stability)
    full_prior_status = basin_stability_status(full_prior_robustness)
    holdout_repeatability_status = "unknown"
    if holdout_repeatability is not None:
        holdout_repeatability_status = str(
            holdout_repeatability.get("status", "unknown")
        )
    holdout_generalization = combined_holdout_generalization_status(
        holdout_validation, holdout_repeatability
    )
    cloud_thickness = str(coarse_metrics["statuses"].get("cloud_thickness", "unknown"))
    fisher_min_eigenvalue = str(
        coarse_metrics["statuses"].get("fisher_min_eigenvalue", "unknown")
    )
    fisher_conditioning = str(
        coarse_metrics["statuses"].get("fisher_conditioning", "unknown")
    )
    fisher_observability = str(
        coarse_metrics["statuses"].get("fisher_observability", "unknown")
    )

    if cloud_thickness == "warning" or holdout_generalization == "warning":
        recommendation = "holdout_review"
    elif fisher_observability == "warning":
        recommendation = "recollect_data"
    elif (
        ground_support == "pass"
        and fisher_observability == "pass"
        and cloud_thickness == "pass"
    ):
        recommendation = "full_6dof_candidate"
    elif ground_support == "pass":
        recommendation = "z_roll_pitch_priority"
    else:
        recommendation = "recollect_data"

    initial_prior_assessment = build_initial_prior_assessment(
        delta_to_initial,
        recommendation=recommendation,
        extraction_consistency=extraction_consistency,
        trusted_reference_consistency=trusted_reference_consistency,
        basin_stability_status=basin_status,
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
        "extraction_consistency": extraction_consistency,
        "trusted_reference_consistency": trusted_reference_consistency,
        "planar_basin_stability": basin_status,
        "full_prior_robustness": full_prior_status,
        "holdout_repeatability": holdout_repeatability_status,
        "holdout_generalization": holdout_generalization,
        "cloud_thickness": cloud_thickness,
        "fisher_min_eigenvalue": fisher_min_eigenvalue,
        "fisher_conditioning": fisher_conditioning,
        "fisher_observability": fisher_observability,
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
            extraction_consistency_recs
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
        assessment["reference_consistency_recommendations"] = reference_consistency_recs
    if basin_stability is not None:
        assessment["planar_basin_stability_details"] = basin_stability
    if full_prior_robustness is not None:
        assessment["full_prior_robustness_details"] = full_prior_robustness
    if holdout_validation is not None:
        assessment["holdout_validation_details"] = holdout_validation
    if holdout_repeatability is not None:
        assessment["holdout_repeatability_details"] = holdout_repeatability
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


def build_final_acceptance(coarse_metrics: dict) -> dict:
    statuses = coarse_metrics["statuses"]
    gates = [
        {
            "name": "ground_sample_count",
            "status": "pass" if coarse_metrics["ground_sample_count"] > 0 else "fail",
            "severity": "required",
            "evidence": f"{coarse_metrics['ground_sample_count']} ground samples",
            "action": "Collect or extract ground-support windows before trusting z/roll/pitch.",
        },
        {
            "name": "motion_sample_count",
            "status": "pass" if coarse_metrics["motion_sample_count"] > 0 else "fail",
            "severity": "required",
            "evidence": f"{coarse_metrics['motion_sample_count']} motion samples",
            "action": "Collect or select motion windows before trusting x/y/yaw.",
        },
        {
            "name": "fisher_min_eigenvalue",
            "status": statuses.get("fisher_min_eigenvalue", "unknown"),
            "severity": "required",
            "evidence": (
                "fisher_min_eigenvalue="
                f"{coarse_metrics.get('fisher_min_eigenvalue')}, "
                f"rank={coarse_metrics.get('fisher_rank')}/"
                f"{coarse_metrics.get('fisher_expected_rank')}"
            ),
            "action": (
                "Reject degenerate runs where the weakest observable dimension "
                "falls below the Fisher minimum threshold."
            ),
        },
        {
            "name": "fisher_conditioning",
            "status": statuses.get("fisher_conditioning", "unknown"),
            "severity": "required",
            "evidence": (
                "fisher_condition_number="
                f"{coarse_metrics.get('fisher_condition_number')}"
            ),
            "action": (
                "Reject ill-conditioned runs where information is highly "
                "unbalanced across dimensions."
            ),
        },
        {
            "name": "holdout_cloud_thickness",
            "status": statuses.get("cloud_thickness", "unknown"),
            "severity": "required",
            "evidence": (
                "cloud_thickness_holdout_p95_m="
                f"{coarse_metrics.get('cloud_thickness_holdout_p95_m')}"
            ),
            "action": (
                "Use holdout stitched-cloud plane thickness as a physical gate: "
                "<3cm is production-grade, >5cm indicates translation misalignment."
            ),
        },
    ]
    return _build_final_acceptance(
        module="lidar2imu",
        gates=gates,
        pass_recommendation="release_full_6dof",
        review_recommendation="review_or_partial_dof_only",
        fail_recommendation="reject_and_recollect",
    )
