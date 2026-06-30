from __future__ import annotations

# isort: off
import math
from typing import Any

import numpy as np

from lidar2imu.models import CalibrationConfig, CalibrationDataset, MotionSample
from lidar2imu.motion_information import motion_information_components
from lidar2imu.motion_information import normalize_confidence_weights

# isort: on


def _window_id(sample: MotionSample) -> int | None:
    raw_window_id = sample.metadata.get("window_id")
    return None if raw_window_id is None else int(raw_window_id)


def _value_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=float).reshape(-1)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(candidate):
        return float(default)
    return float(candidate)


def _observability_covariance_weight(
    sample: MotionSample,
    config: CalibrationConfig,
) -> float:
    metadata = sample.metadata
    lambda_value = max(
        _safe_float(metadata.get("observability_combined_min_eigenvalue"), 0.0),
        0.0,
    )
    condition_number = max(
        _safe_float(metadata.get("observability_combined_condition_number"), 1.0),
        1.0,
    )
    gain_value = max(
        _safe_float(metadata.get("observability_min_eigenvalue_gain"), 0.0), 0.0
    )
    gain_ratio = max(
        _safe_float(metadata.get("observability_min_eigenvalue_gain_ratio"), 0.0),
        0.0,
    )
    if lambda_value <= 0.0 and gain_value <= 0.0 and gain_ratio <= 0.0:
        return 1.0

    lambda_term = min(math.log1p(lambda_value * 10.0), 4.0) / max(
        math.log1p(10.0), 1e-9
    )
    condition_term = 1.0 / (1.0 + math.log10(max(condition_number, 1.0)))
    gain_term = 1.0 + min(gain_ratio, 2.0) * 0.40 + min(gain_value * 10.0, 1.0) * 0.20
    raw_weight = lambda_term * condition_term * gain_term
    return float(
        min(
            max(raw_weight, float(config.probabilistic_observability_weight_min)),
            float(config.probabilistic_observability_weight_max),
        )
    )


def _preintegration_confidence(sample: MotionSample) -> float:
    if "imu_preintegration_confidence" not in sample.metadata:
        return 1.0
    return max(
        _safe_float(sample.metadata.get("imu_preintegration_confidence"), 1.0), 1e-3
    )


def build_information_weighted_motion_samples(
    motion_samples: list[MotionSample],
    config: CalibrationConfig,
) -> tuple[list[MotionSample], dict[str, Any]]:
    if not motion_samples:
        return [], {
            "enabled": True,
            "sample_count": 0,
            "rotation_weight": _value_summary([]),
            "translation_weight": _value_summary([]),
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    rotation_raw_weights: list[float] = []
    translation_raw_weights: list[float] = []
    observability_weights: list[float] = []
    preintegration_confidences: list[float] = []
    for sample in motion_samples:
        candidate = {
            "pose_rotation_deg": sample.metadata.get("pose_rotation_deg"),
            "pose_translation_m": sample.metadata.get("pose_translation_m"),
            "registration_fitness": sample.metadata.get("registration_fitness"),
            "registration_inlier_rmse": sample.metadata.get("registration_inlier_rmse"),
            "registered_overlap_quality_score": sample.metadata.get(
                "registered_overlap_quality_score"
            ),
            "registered_overlap_within_0p4m_ratio": sample.metadata.get(
                "registered_overlap_within_0p4m_ratio"
            ),
            "registered_overlap_nn_mean_m": sample.metadata.get(
                "registered_overlap_nn_mean_m"
            ),
            "stride": sample.metadata.get("frame_stride"),
            "weight": sample.weight,
        }
        components = motion_information_components(candidate)
        observability_weight = _observability_covariance_weight(sample, config)
        preintegration_confidence = _preintegration_confidence(sample)
        rotation_raw = float(components["rotation_confidence"]) * float(
            observability_weight
        )
        translation_raw = (
            float(components["translation_confidence"])
            * float(observability_weight)
            * float(max(preintegration_confidence, 0.25))
        )
        rotation_raw_weights.append(rotation_raw)
        translation_raw_weights.append(translation_raw)
        observability_weights.append(float(observability_weight))
        preintegration_confidences.append(float(preintegration_confidence))
        rows.append(
            {
                "sample": sample,
                "window_id": _window_id(sample),
                "start_timestamp_ns": int(sample.start_timestamp_ns),
                "end_timestamp_ns": int(sample.end_timestamp_ns),
                "rotation_raw_weight": rotation_raw,
                "translation_raw_weight": translation_raw,
                "uncertainty_scale": float(components["uncertainty_scale"]),
                "observability_covariance_weight": float(observability_weight),
                "preintegration_confidence": float(preintegration_confidence),
            }
        )

    rotation_weights = normalize_confidence_weights(
        rotation_raw_weights,
        min_weight=float(config.probabilistic_weight_min),
        max_weight=float(config.probabilistic_weight_max),
    )
    translation_weights = normalize_confidence_weights(
        translation_raw_weights,
        min_weight=float(config.probabilistic_weight_min),
        max_weight=float(config.probabilistic_weight_max),
    )

    weighted_samples: list[MotionSample] = []
    serialized_rows: list[dict[str, Any]] = []
    for row, rotation_weight, translation_weight in zip(
        rows, rotation_weights, translation_weights
    ):
        sample = row["sample"]
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "rotation_weight": float(rotation_weight),
                "translation_weight": float(translation_weight),
                "information_uncertainty_scale": float(row["uncertainty_scale"]),
                "observability_covariance_weight": float(
                    row["observability_covariance_weight"]
                ),
                "preintegration_confidence": float(row["preintegration_confidence"]),
            }
        )
        weighted_samples.append(
            MotionSample(
                start_timestamp_ns=int(sample.start_timestamp_ns),
                end_timestamp_ns=int(sample.end_timestamp_ns),
                imu_delta_rotation=np.asarray(sample.imu_delta_rotation, dtype=float),
                imu_delta_translation=np.asarray(
                    sample.imu_delta_translation, dtype=float
                ),
                lidar_delta_rotation=np.asarray(
                    sample.lidar_delta_rotation, dtype=float
                ),
                lidar_delta_translation=np.asarray(
                    sample.lidar_delta_translation, dtype=float
                ),
                weight=float(sample.weight),
                sync_dt_ms=sample.sync_dt_ms,
                metadata=metadata,
            )
        )
        serialized_rows.append(
            {
                "window_id": row["window_id"],
                "start_timestamp_ns": int(row["start_timestamp_ns"]),
                "end_timestamp_ns": int(row["end_timestamp_ns"]),
                "rotation_raw_weight": float(row["rotation_raw_weight"]),
                "translation_raw_weight": float(row["translation_raw_weight"]),
                "rotation_weight": float(rotation_weight),
                "translation_weight": float(translation_weight),
                "uncertainty_scale": float(row["uncertainty_scale"]),
                "observability_covariance_weight": float(
                    row["observability_covariance_weight"]
                ),
                "preintegration_confidence": float(row["preintegration_confidence"]),
            }
        )

    return weighted_samples, {
        "enabled": True,
        "sample_count": int(len(weighted_samples)),
        "rotation_weight": _value_summary(rotation_weights),
        "translation_weight": _value_summary(translation_weights),
        "observability_covariance_weight": _value_summary(observability_weights),
        "preintegration_confidence": _value_summary(preintegration_confidences),
        "rows": serialized_rows,
    }


def build_information_weighted_dataset(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
) -> tuple[CalibrationDataset, dict[str, Any]]:
    weighted_motion_samples, selected_summary = (
        build_information_weighted_motion_samples(
            dataset.motion_samples,
            config,
        )
    )
    weighted_candidate_pool, pool_summary = build_information_weighted_motion_samples(
        dataset.motion_candidate_pool,
        config,
    )
    weighted_dataset = CalibrationDataset(
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        ground_samples=list(dataset.ground_samples),
        motion_samples=list(weighted_motion_samples),
        initial_transform=np.asarray(dataset.initial_transform, dtype=float),
        extraction_transform=(
            None
            if dataset.extraction_transform is None
            else np.asarray(dataset.extraction_transform, dtype=float)
        ),
        reference_transform=(
            None
            if dataset.reference_transform is None
            else np.asarray(dataset.reference_transform, dtype=float)
        ),
        motion_candidate_pool=list(weighted_candidate_pool),
        metadata=dict(dataset.metadata),
    )
    return weighted_dataset, {
        "enabled": True,
        "weight_bounds": {
            "min": float(config.probabilistic_weight_min),
            "max": float(config.probabilistic_weight_max),
        },
        "active_selection": selected_summary,
        "candidate_pool": pool_summary,
    }
