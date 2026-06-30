from __future__ import annotations

from typing import Any

from lidar2imu.models import CalibrationConfig, CalibrationDataset, MotionSample
from lidar2imu.solvers.motion_objective import (
    evaluate_motion_subset_objective,
    motion_sample_window_ids,
    window_id,
)
from lidar2imu.solvers.motion_screening import motion_turn_sign


def anchor_window_ids(motion_samples: list[MotionSample]) -> set[int]:
    anchors_by_sign: dict[str | None, MotionSample] = {}
    for sample in motion_samples:
        sign = motion_turn_sign(sample)
        current_anchor = anchors_by_sign.get(sign)
        sample_rotation_deg = float(sample.metadata.get("pose_rotation_deg") or 0.0)
        if current_anchor is None:
            anchors_by_sign[sign] = sample
            continue
        anchor_rotation_deg = float(
            current_anchor.metadata.get("pose_rotation_deg") or 0.0
        )
        if sample_rotation_deg > anchor_rotation_deg:
            anchors_by_sign[sign] = sample
    return {
        int(raw_window_id)
        for raw_window_id in (window_id(sample) for sample in anchors_by_sign.values())
        if raw_window_id is not None
    }


def improve_motion_samples_with_local_search(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    ground_stage: dict[str, Any],
    initial_transform,
    motion_samples: list[MotionSample],
    *,
    locked_window_ids: set[int] | None = None,
) -> tuple[list[MotionSample], dict[str, Any]]:
    current_samples = sorted(
        list(motion_samples),
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if not dataset.motion_candidate_pool:
        return current_samples, {"enabled": False, "reason": "empty_candidate_pool"}
    locked_window_ids = set() if locked_window_ids is None else set(locked_window_ids)

    current_score, current_metrics = evaluate_motion_subset_objective(
        dataset,
        config,
        ground_stage,
        initial_transform,
        current_samples,
    )
    search_history = [
        {
            "window_ids": motion_sample_window_ids(current_samples),
            "objective": float(current_score),
            **current_metrics,
        }
    ]

    improved = True
    while improved:
        improved = False
        selected_window_ids = {
            int(raw_window_id)
            for raw_window_id in (window_id(sample) for sample in current_samples)
            if raw_window_id is not None
        }
        turn_counts: dict[str | None, int] = {}
        for sample in current_samples:
            sign = motion_turn_sign(sample)
            turn_counts[sign] = turn_counts.get(sign, 0) + 1
        replaceable_indices = [
            index
            for index, sample in enumerate(current_samples)
            if (window_id(sample) not in locked_window_ids)
            and (
                motion_turn_sign(sample) is None
                or turn_counts.get(motion_turn_sign(sample), 0) > 1
            )
        ]
        backup_samples = [
            sample
            for sample in dataset.motion_candidate_pool
            if window_id(sample) not in selected_window_ids
        ]
        best_swap = None
        best_score = current_score
        best_metrics = None
        for replace_index in replaceable_indices:
            for backup_sample in backup_samples:
                candidate_samples = list(current_samples)
                candidate_samples[replace_index] = backup_sample
                candidate_samples.sort(
                    key=lambda sample: (
                        sample.start_timestamp_ns,
                        sample.end_timestamp_ns,
                    )
                )
                candidate_score, candidate_metrics = evaluate_motion_subset_objective(
                    dataset,
                    config,
                    ground_stage,
                    initial_transform,
                    candidate_samples,
                )
                if candidate_score + 1e-6 >= best_score:
                    continue
                best_score = candidate_score
                best_metrics = candidate_metrics
                best_swap = {
                    "out_window_id": window_id(current_samples[replace_index]),
                    "in_window_id": window_id(backup_sample),
                    "samples": candidate_samples,
                }
        if best_swap is None:
            break
        current_samples = best_swap["samples"]
        current_score = float(best_score)
        search_history.append(
            {
                "window_ids": motion_sample_window_ids(current_samples),
                "objective": float(current_score),
                **(best_metrics or {}),
            }
        )
        improved = True

    return current_samples, {
        "enabled": True,
        "search_history": search_history,
        "final_window_ids": motion_sample_window_ids(current_samples),
    }
