from __future__ import annotations

import math
from typing import Any, Mapping


def _safe_float(value: Any, default: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(candidate):
        return float(default)
    return float(candidate)


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(float(value), float(minimum)), float(maximum)))


def motion_information_components(
    candidate: Mapping[str, Any], *, base_stride: int | None = None
) -> dict[str, float]:
    pose_rotation_deg = max(_safe_float(candidate.get("pose_rotation_deg"), 0.0), 0.0)
    pose_translation_m = max(_safe_float(candidate.get("pose_translation_m"), 0.0), 0.0)
    registration_fitness = _safe_float(candidate.get("registration_fitness"), 0.85)
    registration_inlier_rmse = max(
        _safe_float(candidate.get("registration_inlier_rmse"), 0.12), 1e-3
    )
    overlap_quality = _safe_float(
        candidate.get("registered_overlap_quality_score"), 1.0
    )
    overlap_within_ratio = _safe_float(
        candidate.get("registered_overlap_within_0p4m_ratio"), 0.90
    )
    overlap_nn_mean_m = max(
        _safe_float(candidate.get("registered_overlap_nn_mean_m"), 0.25), 1e-3
    )
    sample_weight = max(_safe_float(candidate.get("weight"), 1.0), 1e-6)

    stride = max(int(round(_safe_float(candidate.get("stride"), 1.0))), 1)
    if base_stride is None:
        base_stride = stride
    base_stride = max(int(base_stride), 1)

    excitation_rotation = _bounded(pose_rotation_deg / 8.0, 0.25, 2.5)
    excitation_translation = _bounded(pose_translation_m / 1.2, 0.25, 2.5)

    fitness_confidence = _bounded(registration_fitness / 0.85, 0.25, 1.5)
    rmse_confidence = _bounded(0.15 / registration_inlier_rmse, 0.25, 1.5)
    registration_confidence = math.sqrt(fitness_confidence * rmse_confidence)

    overlap_quality_confidence = _bounded(overlap_quality, 0.25, 1.5)
    overlap_ratio_confidence = _bounded(overlap_within_ratio / 0.90, 0.25, 1.5)
    overlap_distance_confidence = _bounded(0.25 / overlap_nn_mean_m, 0.25, 1.5)
    overlap_confidence = (
        overlap_quality_confidence
        * overlap_ratio_confidence
        * overlap_distance_confidence
    ) ** (1.0 / 3.0)

    uncertainty_scale = math.sqrt(registration_confidence * overlap_confidence)
    rotation_confidence = sample_weight * uncertainty_scale * excitation_rotation
    translation_confidence = sample_weight * uncertainty_scale * excitation_translation

    base_information_score = max(pose_rotation_deg, 0.1) * 10.0 + max(
        pose_translation_m, 0.01
    )
    probabilistic_information_score = base_information_score * uncertainty_scale
    probabilistic_window_score = probabilistic_information_score / float(stride)

    stride_bonus = min(
        math.log2(float(stride) / float(base_stride) + 1.0),
        2.0,
    )
    probabilistic_stride_multiplier = 1.0 + (0.08 * float(stride_bonus))

    return {
        "pose_rotation_deg": float(pose_rotation_deg),
        "pose_translation_m": float(pose_translation_m),
        "stride": float(stride),
        "sample_weight": float(sample_weight),
        "excitation_rotation": float(excitation_rotation),
        "excitation_translation": float(excitation_translation),
        "registration_confidence": float(registration_confidence),
        "overlap_confidence": float(overlap_confidence),
        "uncertainty_scale": float(uncertainty_scale),
        "rotation_confidence": float(rotation_confidence),
        "translation_confidence": float(translation_confidence),
        "base_information_score": float(base_information_score),
        "probabilistic_information_score": float(probabilistic_information_score),
        "probabilistic_window_score": float(probabilistic_window_score),
        "probabilistic_stride_multiplier": float(probabilistic_stride_multiplier),
    }


def normalize_confidence_weights(
    raw_weights: list[float], *, min_weight: float, max_weight: float
) -> list[float]:
    if not raw_weights:
        return []
    safe_weights = [max(float(weight), 1e-12) for weight in raw_weights]
    mean_weight = float(sum(safe_weights) / len(safe_weights))
    if mean_weight <= 1e-12:
        return [1.0 for _ in safe_weights]
    return [
        _bounded(weight / mean_weight, float(min_weight), float(max_weight))
        for weight in safe_weights
    ]
