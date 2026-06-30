from __future__ import annotations

from lidar2imu.models import CalibrationConfig


def coarse_status(value: float | None, warning_threshold: float) -> str:
    if value is None:
        return "unknown"
    if value <= warning_threshold:
        return "pass"
    return "warning"


def coarse_status_min(value: float | None, minimum_threshold: float) -> str:
    if value is None:
        return "unknown"
    if value >= minimum_threshold:
        return "pass"
    return "warning"


def transform_consistency_failure_reasons(
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


def transform_consistency_status(
    *,
    delta: dict | None,
    translation_warning_m: float,
    rotation_warning_deg: float,
    vertical_warning_m: float,
    vertical_warning_ratio: float,
) -> str:
    if delta is None:
        return "unknown"
    if not transform_consistency_failure_reasons(
        delta,
        translation_warning_m=translation_warning_m,
        rotation_warning_deg=rotation_warning_deg,
        vertical_warning_m=vertical_warning_m,
        vertical_warning_ratio=vertical_warning_ratio,
    ):
        return "pass"
    return "warning"


def reference_consistency_status(
    delta_to_reference: dict | None, config: CalibrationConfig
) -> str:
    return transform_consistency_status(
        delta=delta_to_reference,
        translation_warning_m=config.metrics_reference_warning_translation_m,
        rotation_warning_deg=config.metrics_reference_warning_rotation_deg,
        vertical_warning_m=config.metrics_reference_warning_vertical_m,
        vertical_warning_ratio=config.metrics_reference_warning_vertical_ratio,
    )


def extraction_consistency_status(
    delta_to_extraction: dict | None, config: CalibrationConfig
) -> str:
    return transform_consistency_status(
        delta=delta_to_extraction,
        translation_warning_m=config.metrics_extraction_warning_translation_m,
        rotation_warning_deg=config.metrics_extraction_warning_rotation_deg,
        vertical_warning_m=config.metrics_extraction_warning_vertical_m,
        vertical_warning_ratio=config.metrics_extraction_warning_vertical_ratio,
    )


def reference_consistency_recommendations(
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
    failure_reasons = transform_consistency_failure_reasons(
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


def extraction_consistency_recommendations(
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
    failure_reasons = transform_consistency_failure_reasons(
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


def basin_stability_status(basin_stability: dict | None) -> str:
    if not basin_stability:
        return "unknown"
    return str(basin_stability.get("status", "unknown"))


def initial_prior_nominal_status(
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


def build_initial_prior_assessment(
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
