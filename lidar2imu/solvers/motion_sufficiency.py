from __future__ import annotations

from typing import Any

import numpy as np

from lidar2imu.models import CalibrationConfig, MotionSample
from lidar2imu.solvers.motion_objective import motion_sample_window_ids, window_id
from lidar2imu.solvers.motion_screening import motion_turn_sign

_HEADING_BIN_WIDTH_DEG = 45.0
_MIN_HEADING_SPAN_DEG = 20.0


def _heading_deg(sample: MotionSample) -> float | None:
    heading = sample.metadata.get("imu_translation_heading_deg")
    return None if heading is None else float(heading)


def _heading_bin(heading_deg: float | None) -> int | None:
    if heading_deg is None:
        return None
    return int(np.floor((float(heading_deg) + 180.0) / _HEADING_BIN_WIDTH_DEG)) % int(
        round(360.0 / _HEADING_BIN_WIDTH_DEG)
    )


def _circular_span_deg(angles_deg: list[float]) -> float:
    if len(angles_deg) <= 1:
        return 0.0
    normalized = sorted((float(angle) + 360.0) % 360.0 for angle in angles_deg)
    gap_sizes = []
    for index, value in enumerate(normalized):
        next_value = normalized[(index + 1) % len(normalized)]
        if index == len(normalized) - 1:
            next_value += 360.0
        gap_sizes.append(next_value - value)
    return float(360.0 - max(gap_sizes))


def _sample_row(sample: MotionSample) -> dict[str, Any]:
    return {
        "window_id": window_id(sample),
        "turn_sign": motion_turn_sign(sample),
        "heading_deg": _heading_deg(sample),
        "heading_bin": _heading_bin(_heading_deg(sample)),
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
    }


def _anchor_window_ids(rows: list[dict[str, Any]]) -> list[int]:
    anchors_by_sign: dict[str | None, dict[str, Any]] = {}
    for row in rows:
        sign = row["turn_sign"]
        current_anchor = anchors_by_sign.get(sign)
        if current_anchor is None or float(row["pose_rotation_deg"]) > float(
            current_anchor["pose_rotation_deg"]
        ):
            anchors_by_sign[sign] = row
    return sorted(
        {
            int(row["window_id"])
            for row in anchors_by_sign.values()
            if row.get("window_id") is not None
        }
    )


def _selection_metrics(motion_samples: list[MotionSample]) -> dict[str, Any]:
    rows = [_sample_row(sample) for sample in motion_samples]
    headings = [
        float(row["heading_deg"]) for row in rows if row.get("heading_deg") is not None
    ]
    heading_bins = sorted(
        {int(row["heading_bin"]) for row in rows if row.get("heading_bin") is not None}
    )
    turn_counts = {"left": 0, "right": 0, "neutral": 0}
    for row in rows:
        sign = row["turn_sign"]
        if sign == "left":
            turn_counts["left"] += 1
        elif sign == "right":
            turn_counts["right"] += 1
        else:
            turn_counts["neutral"] += 1
    return {
        "sample_count": int(len(motion_samples)),
        "unique_window_count": int(len(motion_sample_window_ids(motion_samples))),
        "window_ids": motion_sample_window_ids(motion_samples),
        "turn_counts": turn_counts,
        "heading_count": int(len(headings)),
        "heading_bins": heading_bins,
        "heading_bin_count": int(len(heading_bins)),
        "heading_span_deg": float(_circular_span_deg(headings)),
        "total_pose_rotation_deg": float(
            sum(float(row["pose_rotation_deg"]) for row in rows)
        ),
        "max_pose_rotation_deg": float(
            max((float(row["pose_rotation_deg"]) for row in rows), default=0.0)
        ),
        "anchor_window_ids": _anchor_window_ids(rows),
        "rows": rows,
    }


def assess_motion_sufficiency(
    active_motion_samples: list[MotionSample],
    candidate_pool: list[MotionSample],
    config: CalibrationConfig,
) -> dict[str, Any]:
    active_metrics = _selection_metrics(active_motion_samples)
    pool_source = candidate_pool if candidate_pool else active_motion_samples
    candidate_pool_metrics = _selection_metrics(pool_source)

    min_total_rotation_deg = max(
        float(config.min_motion_rotation_deg)
        * float(max(config.min_motion_samples, 1))
        * 2.0,
        10.0,
    )
    min_heading_bin_count = 2
    thresholds = {
        "min_motion_samples": int(config.min_motion_samples),
        "min_turn_count_per_direction": int(
            config.metrics_min_turn_count_per_direction
        ),
        "min_total_pose_rotation_deg": float(min_total_rotation_deg),
        "min_heading_bin_count": int(min_heading_bin_count),
        "min_heading_span_deg": float(_MIN_HEADING_SPAN_DEG),
    }

    free_planar_reasons: list[str] = []
    advisory_reasons: list[str] = []
    if active_metrics["unique_window_count"] < int(config.min_motion_samples):
        free_planar_reasons.append("active_motion_window_count_below_minimum")
    if min(
        active_metrics["turn_counts"]["left"],
        active_metrics["turn_counts"]["right"],
    ) < int(config.metrics_min_turn_count_per_direction):
        free_planar_reasons.append("active_turn_imbalance")
    if active_metrics["total_pose_rotation_deg"] < min_total_rotation_deg:
        free_planar_reasons.append("active_total_rotation_below_threshold")
    if active_metrics["heading_count"] >= 2:
        if active_metrics["heading_bin_count"] < min_heading_bin_count:
            free_planar_reasons.append("active_heading_bin_count_below_threshold")
        if active_metrics["heading_span_deg"] < _MIN_HEADING_SPAN_DEG:
            advisory_reasons.append("active_heading_span_below_threshold")

    local_search_reasons = list(free_planar_reasons)
    if (
        candidate_pool_metrics["unique_window_count"]
        <= active_metrics["unique_window_count"]
    ):
        local_search_reasons.append("candidate_pool_has_no_extra_windows")
    if min(
        candidate_pool_metrics["turn_counts"]["left"],
        candidate_pool_metrics["turn_counts"]["right"],
    ) < int(config.metrics_min_turn_count_per_direction):
        local_search_reasons.append("candidate_pool_turn_imbalance")
    if candidate_pool_metrics["total_pose_rotation_deg"] < min_total_rotation_deg:
        local_search_reasons.append("candidate_pool_total_rotation_below_threshold")
    if candidate_pool_metrics["heading_count"] >= 2:
        if candidate_pool_metrics["heading_bin_count"] < min_heading_bin_count:
            local_search_reasons.append(
                "candidate_pool_heading_bin_count_below_threshold"
            )
        if candidate_pool_metrics["heading_span_deg"] < _MIN_HEADING_SPAN_DEG:
            advisory_reasons.append("candidate_pool_heading_span_below_threshold")

    return {
        "ready_for_free_planar": not free_planar_reasons,
        "ready_for_local_search": not local_search_reasons,
        "free_planar_reasons": free_planar_reasons,
        "local_search_reasons": local_search_reasons,
        "advisory_reasons": advisory_reasons,
        "locked_window_ids": list(active_metrics["anchor_window_ids"]),
        "thresholds": thresholds,
        "active_selection": active_metrics,
        "candidate_pool": candidate_pool_metrics,
    }
