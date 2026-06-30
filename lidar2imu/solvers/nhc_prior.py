from __future__ import annotations

from typing import Any

from lidar2imu.models import CalibrationConfig


def resolve_nhc_prior(
    config: CalibrationConfig,
    *,
    solver_policy: dict[str, Any],
    screening_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    mode = str(config.nhc_prior_mode or "off")
    if mode not in {"off", "auto", "force"}:
        mode = "off"
    if mode == "off":
        return None

    weak_reasons: list[str] = []
    if str(solver_policy.get("applied", "")) == "freeze_xyyaw":
        weak_reasons.append("freeze_xyyaw_policy")

    left_turn_count = int(solver_policy.get("left_turn_count", 0))
    right_turn_count = int(solver_policy.get("right_turn_count", 0))
    if min(left_turn_count, right_turn_count) < int(
        config.metrics_min_turn_count_per_direction
    ):
        weak_reasons.append("turn_imbalance")

    sufficiency = (
        {}
        if screening_summary is None
        else dict(screening_summary.get("sufficiency") or {})
    )
    if sufficiency and not bool(sufficiency.get("ready_for_free_planar", True)):
        weak_reasons.append("global_motion_sufficiency")

    enabled = mode == "force" or bool(weak_reasons)
    return {
        "enabled": bool(enabled),
        "mode": mode,
        "activation_reasons": weak_reasons,
        "weight": float(config.nhc_prior_weight),
        "lateral_scale_m": float(config.nhc_prior_lateral_scale_m),
        "vertical_scale_m": float(config.nhc_prior_vertical_scale_m),
        "min_forward_translation_m": float(config.nhc_prior_min_forward_translation_m),
    }
