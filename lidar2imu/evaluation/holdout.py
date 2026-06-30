from __future__ import annotations

from lidar2imu.models import CalibrationConfig


def safe_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    baseline = float(baseline)
    if abs(baseline) <= 1e-12:
        if abs(float(value)) <= 1e-12:
            return 1.0
        return None
    return float(value) / baseline


def build_holdout_validation(
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

    rotation_ratio = safe_ratio(holdout_rotation_p95, train_rotation_p95)
    translation_ratio = safe_ratio(holdout_translation_p95, train_translation_p95)
    fitness_ratio = safe_ratio(holdout_fitness_p05, train_fitness_p05)
    rmse_ratio = safe_ratio(holdout_rmse_p95, train_rmse_p95)

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


def combined_holdout_generalization_status(
    holdout_validation: dict | None,
    holdout_repeatability: dict | None,
) -> str:
    statuses = [
        str(payload.get("status", "unknown"))
        for payload in (holdout_validation, holdout_repeatability)
        if payload is not None
    ]
    if not statuses:
        return "unknown"
    if any(status == "warning" for status in statuses):
        return "warning"
    if any(status == "pass" for status in statuses):
        return "pass"
    return "unknown"
