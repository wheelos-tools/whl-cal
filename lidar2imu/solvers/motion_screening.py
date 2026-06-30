from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.models import CalibrationConfig, MotionSample

_SUPPORT_ROTATION_SCALE_DEG = 0.6
_SUPPORT_TRANSLATION_SCALE_M = 0.05
_SUPPORT_MEDIAN_RATIO = 0.60
_SUPPORT_ROTATION_MEDIAN_FACTOR = 1.10
_SUPPORT_TRANSLATION_MEDIAN_FACTOR = 1.10


def count_turns(motion_samples: list[MotionSample]) -> tuple[int, int]:
    signed_yaw_deg = []
    for sample in motion_samples:
        try:
            signed_yaw_deg.append(
                float(
                    np.degrees(
                        R.from_matrix(sample.imu_delta_rotation).as_euler(
                            "ZYX", degrees=False
                        )[0]
                    )
                )
            )
        except ValueError:
            signed_yaw_deg.append(0.0)
    left_turn_count = int(sum(1 for value in signed_yaw_deg if value > 0.5))
    right_turn_count = int(sum(1 for value in signed_yaw_deg if value < -0.5))
    return left_turn_count, right_turn_count


def motion_turn_sign(sample: MotionSample) -> str | None:
    signed_yaw_deg = float(sample.metadata.get("imu_signed_yaw_deg") or 0.0)
    if signed_yaw_deg > 0.5:
        return "left"
    if signed_yaw_deg < -0.5:
        return "right"
    return None


def _heading_deg(sample: MotionSample) -> float | None:
    heading = sample.metadata.get("imu_translation_heading_deg")
    return None if heading is None else float(heading)


def _circular_distance_deg(angle_a: float, angle_b: float) -> float:
    return float(abs(((angle_a - angle_b) + 180.0) % 360.0 - 180.0))


def _window_id(sample: MotionSample) -> int | None:
    window_id = sample.metadata.get("window_id")
    return None if window_id is None else int(window_id)


def screen_motion_samples(
    motion_samples: list[MotionSample],
    config: CalibrationConfig,
) -> tuple[list[MotionSample], dict[str, Any]]:
    min_rotation_rad = np.radians(float(config.min_motion_rotation_deg))
    fitness_threshold = config.solver_screen_min_registration_fitness
    if fitness_threshold is None:
        fitness_threshold = config.metrics_warning_registration_fitness
    rmse_threshold = config.solver_screen_max_registration_inlier_rmse_m
    if rmse_threshold is None:
        rmse_threshold = config.metrics_warning_registration_inlier_rmse_m

    selected: list[MotionSample] = []
    skip_reason_counts: dict[str, int] = {}
    for sample in motion_samples:
        reasons = []
        imu_rot = np.linalg.norm(R.from_matrix(sample.imu_delta_rotation).as_rotvec())
        lidar_rot = np.linalg.norm(
            R.from_matrix(sample.lidar_delta_rotation).as_rotvec()
        )
        if max(imu_rot, lidar_rot) < min_rotation_rad:
            reasons.append("below_min_motion_rotation")
        registration_fitness = sample.metadata.get("registration_fitness")
        if registration_fitness is not None and float(registration_fitness) < float(
            fitness_threshold
        ):
            reasons.append("registration_fitness_below_threshold")
        registration_inlier_rmse = sample.metadata.get("registration_inlier_rmse")
        if registration_inlier_rmse is not None and float(
            registration_inlier_rmse
        ) > float(rmse_threshold):
            reasons.append("registration_inlier_rmse_above_threshold")
        if reasons:
            for reason in reasons:
                skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            continue
        selected.append(sample)

    return selected, {
        "input_sample_count": int(len(motion_samples)),
        "selected_sample_count": int(len(selected)),
        "skipped_sample_count": int(len(motion_samples) - len(selected)),
        "min_motion_rotation_deg": float(config.min_motion_rotation_deg),
        "min_registration_fitness": float(fitness_threshold),
        "max_registration_inlier_rmse_m": float(rmse_threshold),
        "skip_reason_counts": skip_reason_counts,
    }


def resolve_planar_motion_policy(
    motion_samples: list[MotionSample],
    config: CalibrationConfig,
    motion_rotation_stage: dict[str, Any],
    *,
    screening_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_turn_count, right_turn_count = count_turns(motion_samples)
    yaw_observability = motion_rotation_stage.get("observability", {})
    yaw_observability_reasons = list(yaw_observability.get("reasons", []))
    sufficiency_summary = (
        {}
        if screening_summary is None
        else dict(screening_summary.get("sufficiency") or {})
    )
    sufficiency_reasons = list(sufficiency_summary.get("free_planar_reasons", []))
    weak_planar_reasons = []
    if (
        min(left_turn_count, right_turn_count)
        < config.metrics_min_turn_count_per_direction
    ):
        weak_planar_reasons.append("turn_imbalance")
    if yaw_observability.get("degenerate", False):
        weak_planar_reasons.append("yaw_rotation_degenerate")
    if screening_summary is not None and int(
        screening_summary.get("selected_sample_count", 0)
    ) < int(config.min_motion_samples):
        weak_planar_reasons.append("screened_motion_sample_count_below_minimum")
    if sufficiency_summary and not bool(
        sufficiency_summary.get("ready_for_free_planar", True)
    ):
        weak_planar_reasons.append("global_motion_sufficiency")

    requested_policy = config.planar_motion_policy
    applied_policy = requested_policy
    if requested_policy == "auto":
        applied_policy = "freeze_xyyaw" if weak_planar_reasons else "free"

    return {
        "requested": requested_policy,
        "applied": applied_policy,
        "left_turn_count": left_turn_count,
        "right_turn_count": right_turn_count,
        "weak_planar_reasons": weak_planar_reasons,
        "yaw_observability_reasons": yaw_observability_reasons,
        "sufficiency_reasons": sufficiency_reasons,
        "locked_components": (
            ["yaw", "x", "y"] if applied_policy == "freeze_xyyaw" else []
        ),
        "screening": screening_summary,
    }


def _rotation_residual_deg(sample: MotionSample, rotation: np.ndarray) -> float:
    rotation_error = (
        sample.imu_delta_rotation
        @ rotation
        @ sample.lidar_delta_rotation.T
        @ rotation.T
    )
    return float(np.degrees(np.linalg.norm(R.from_matrix(rotation_error).as_rotvec())))


def _translation_residual_m(
    sample: MotionSample, rotation: np.ndarray, translation: np.ndarray
) -> float:
    translation_error = (
        sample.imu_delta_rotation - np.eye(3, dtype=float)
    ) @ translation - (
        rotation @ sample.lidar_delta_translation - sample.imu_delta_translation
    )
    return float(np.linalg.norm(translation_error))


def _support_score(
    sample: MotionSample,
    *,
    rotation_residual_deg: float,
    translation_residual_m: float,
) -> float:
    overlap_quality = float(
        sample.metadata.get("registered_overlap_quality_score") or 1.0
    )
    return float(
        overlap_quality
        / (
            1.0
            + (rotation_residual_deg / _SUPPORT_ROTATION_SCALE_DEG)
            + (translation_residual_m / _SUPPORT_TRANSLATION_SCALE_M)
        )
    )


def _rescue_score(sample: MotionSample, support_score: float) -> float:
    pose_rotation_deg = max(float(sample.metadata.get("pose_rotation_deg") or 0.0), 1.0)
    return float(support_score * np.sqrt(pose_rotation_deg))


def _sample_support_row(
    sample: MotionSample, *, rotation: np.ndarray, translation: np.ndarray
) -> dict[str, Any]:
    rotation_residual_deg = _rotation_residual_deg(sample, rotation)
    translation_residual_m = _translation_residual_m(sample, rotation, translation)
    support_score = _support_score(
        sample,
        rotation_residual_deg=rotation_residual_deg,
        translation_residual_m=translation_residual_m,
    )
    return {
        "sample": sample,
        "window_id": _window_id(sample),
        "frame_stride": (
            None
            if sample.metadata.get("frame_stride") is None
            else int(sample.metadata["frame_stride"])
        ),
        "turn_sign": motion_turn_sign(sample),
        "heading_deg": _heading_deg(sample),
        "pose_rotation_deg": float(sample.metadata.get("pose_rotation_deg") or 0.0),
        "registration_fitness": (
            None
            if sample.metadata.get("registration_fitness") is None
            else float(sample.metadata["registration_fitness"])
        ),
        "registered_overlap_quality_score": (
            None
            if sample.metadata.get("registered_overlap_quality_score") is None
            else float(sample.metadata["registered_overlap_quality_score"])
        ),
        "rotation_residual_deg": float(rotation_residual_deg),
        "translation_residual_m": float(translation_residual_m),
        "support_score": float(support_score),
        "rescue_score": float(_rescue_score(sample, support_score)),
    }


def prune_inconsistent_motion_samples(
    motion_samples: list[MotionSample],
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    config: CalibrationConfig,
) -> tuple[list[MotionSample], dict[str, Any]]:
    if not motion_samples:
        return [], {
            "input_sample_count": 0,
            "selected_sample_count": 0,
            "pruned_sample_count": 0,
            "pruned": False,
            "pruned_window_ids": [],
            "rows": [],
        }

    rows = []
    for index, sample in enumerate(motion_samples):
        row = _sample_support_row(
            sample,
            rotation=rotation,
            translation=translation,
        )
        row["index"] = int(index)
        rows.append(row)

    rotation_median = float(np.median([row["rotation_residual_deg"] for row in rows]))
    translation_median = float(
        np.median([row["translation_residual_m"] for row in rows])
    )
    support_median = float(np.median([row["support_score"] for row in rows]))
    max_prunable = max(int(len(rows) - max(int(config.min_motion_samples), 1)), 0)
    prunable_rows = []
    for row in rows:
        low_support = row["support_score"] < (support_median * _SUPPORT_MEDIAN_RATIO)
        high_residual = row["rotation_residual_deg"] > max(
            rotation_median * _SUPPORT_ROTATION_MEDIAN_FACTOR,
            _SUPPORT_ROTATION_SCALE_DEG,
        ) or row["translation_residual_m"] > max(
            translation_median * _SUPPORT_TRANSLATION_MEDIAN_FACTOR,
            _SUPPORT_TRANSLATION_SCALE_M,
        )
        row["low_support"] = bool(low_support)
        row["high_residual"] = bool(high_residual)
        row["keep"] = True
        if low_support and high_residual:
            prunable_rows.append(row)

    pruned_indices: set[int] = set()
    for row in sorted(prunable_rows, key=lambda item: item["support_score"]):
        if len(pruned_indices) >= max_prunable:
            break
        pruned_indices.add(int(row["index"]))
        row["keep"] = False

    selected_samples = []
    serialized_rows = []
    pruned_window_ids = []
    for row in rows:
        keep = bool(row["keep"])
        if keep:
            selected_samples.append(row["sample"])
        else:
            if row["window_id"] is not None:
                pruned_window_ids.append(int(row["window_id"]))
        serialized_rows.append(
            {key: value for key, value in row.items() if key not in {"index", "sample"}}
        )

    return selected_samples, {
        "input_sample_count": int(len(rows)),
        "selected_sample_count": int(len(selected_samples)),
        "pruned_sample_count": int(len(rows) - len(selected_samples)),
        "pruned": bool(len(selected_samples) != len(rows)),
        "pruned_window_ids": sorted(set(pruned_window_ids)),
        "rules": {
            "support_score_median_ratio": float(_SUPPORT_MEDIAN_RATIO),
            "rotation_residual_median_factor": float(_SUPPORT_ROTATION_MEDIAN_FACTOR),
            "translation_residual_median_factor": float(
                _SUPPORT_TRANSLATION_MEDIAN_FACTOR
            ),
            "rotation_residual_scale_deg": float(_SUPPORT_ROTATION_SCALE_DEG),
            "translation_residual_scale_m": float(_SUPPORT_TRANSLATION_SCALE_M),
        },
        "medians": {
            "rotation_residual_deg": float(rotation_median),
            "translation_residual_m": float(translation_median),
            "support_score": float(support_median),
        },
        "rows": serialized_rows,
    }


def _fill_novelty_score(
    row: dict[str, Any], selected_rows: list[dict[str, Any]]
) -> float:
    heading_deg = row.get("heading_deg")
    existing_headings = [
        float(existing["heading_deg"])
        for existing in selected_rows
        if existing.get("heading_deg") is not None
    ]
    if heading_deg is None or not existing_headings:
        heading_bonus = 1.0
    else:
        min_heading_distance = min(
            _circular_distance_deg(float(heading_deg), existing_heading)
            for existing_heading in existing_headings
        )
        heading_bonus = 0.5 + (min_heading_distance / 10.0)
    return float(row["rescue_score"] * heading_bonus)


def reselect_motion_samples_from_pool(
    current_motion_samples: list[MotionSample],
    candidate_pool: list[MotionSample],
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    config: CalibrationConfig,
    target_sample_count: int,
) -> tuple[list[MotionSample], dict[str, Any]]:
    if not candidate_pool or target_sample_count <= 0:
        return list(current_motion_samples), {
            "enabled": False,
            "reason": "empty_candidate_pool",
            "final_window_ids": [
                window_id
                for window_id in (
                    _window_id(sample) for sample in current_motion_samples
                )
                if window_id is not None
            ],
        }

    screened_pool, pool_screening = screen_motion_samples(candidate_pool, config)
    selected_rows = [
        _sample_support_row(
            sample,
            rotation=rotation,
            translation=translation,
        )
        for sample in current_motion_samples
    ]
    current_window_ids = {
        int(row["window_id"])
        for row in selected_rows
        if row.get("window_id") is not None
    }
    available_rows = [
        _sample_support_row(
            sample,
            rotation=rotation,
            translation=translation,
        )
        for sample in screened_pool
        if _window_id(sample) not in current_window_ids
    ]

    filled_window_ids: list[int] = []
    while len(selected_rows) < int(target_sample_count) and available_rows:
        backup_row = max(
            available_rows,
            key=lambda item: _fill_novelty_score(item, selected_rows),
        )
        selected_rows.append(backup_row)
        available_rows = [
            item
            for item in available_rows
            if item.get("window_id") != backup_row.get("window_id")
        ]
        if backup_row.get("window_id") is not None:
            filled_window_ids.append(int(backup_row["window_id"]))

    swap_history: list[dict[str, Any]] = []
    while available_rows:
        turn_counts: dict[str | None, int] = {}
        for row in selected_rows:
            sign = row.get("turn_sign")
            turn_counts[sign] = turn_counts.get(sign, 0) + 1
        replaceable_rows = [
            row
            for row in selected_rows
            if row.get("turn_sign") is None
            or turn_counts.get(row.get("turn_sign"), 0) > 1
        ]
        if not replaceable_rows:
            break
        weakest_row = min(replaceable_rows, key=lambda item: item["rescue_score"])
        remaining_selected = [
            row
            for row in selected_rows
            if row.get("window_id") != weakest_row.get("window_id")
        ]
        backup_row = max(
            available_rows,
            key=lambda item: _fill_novelty_score(item, remaining_selected),
        )
        if (
            float(backup_row["rescue_score"])
            <= float(weakest_row["rescue_score"]) * 1.10
        ):
            break
        selected_rows = remaining_selected + [backup_row]
        available_rows = [
            item
            for item in available_rows
            if item.get("window_id") != backup_row.get("window_id")
        ]
        if (
            weakest_row.get("window_id") is not None
            and backup_row.get("window_id") is not None
        ):
            swap_history.append(
                {
                    "out_window_id": int(weakest_row["window_id"]),
                    "in_window_id": int(backup_row["window_id"]),
                    "out_rescue_score": float(weakest_row["rescue_score"]),
                    "in_rescue_score": float(backup_row["rescue_score"]),
                }
            )

    selected_rows.sort(
        key=lambda item: (
            int(item["sample"].start_timestamp_ns),
            int(item["sample"].end_timestamp_ns),
        )
    )
    selected_samples = [row["sample"] for row in selected_rows]
    serialized_rows = [
        {key: value for key, value in row.items() if key != "sample"}
        for row in selected_rows
    ]
    return selected_samples, {
        "enabled": True,
        "target_sample_count": int(target_sample_count),
        "pool_screening": pool_screening,
        "filled_window_ids": filled_window_ids,
        "swap_history": swap_history,
        "initial_window_ids": [
            int(window_id)
            for window_id in sorted(
                {
                    window_id
                    for window_id in (
                        _window_id(sample) for sample in current_motion_samples
                    )
                    if window_id is not None
                }
            )
        ],
        "final_window_ids": [
            int(window_id)
            for window_id in sorted(
                {
                    window_id
                    for window_id in (_window_id(sample) for sample in selected_samples)
                    if window_id is not None
                }
            )
        ],
        "rows": serialized_rows,
    }
