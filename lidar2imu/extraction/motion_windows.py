from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.motion_information import motion_information_components
from lidar2lidar.extrinsic_io import transform_dict_from_matrix


def relative_motion(
    transform_world_start: np.ndarray, transform_world_end: np.ndarray
) -> np.ndarray:
    return np.linalg.inv(transform_world_end) @ transform_world_start


def motion_excitation(delta_transform: np.ndarray) -> tuple[float, float]:
    rotation_deg = float(
        np.degrees(np.linalg.norm(R.from_matrix(delta_transform[:3, :3]).as_rotvec()))
    )
    translation_m = float(np.linalg.norm(delta_transform[:3, 3]))
    return rotation_deg, translation_m


def motion_translation_heading_deg(delta_transform: np.ndarray) -> float | None:
    translation = np.asarray(delta_transform[:3, 3], dtype=float).reshape(3)
    if np.linalg.norm(translation[:2]) <= 1e-9:
        return None
    return float(np.degrees(np.arctan2(translation[1], translation[0])))


def motion_signed_yaw_deg(delta_transform: np.ndarray) -> float:
    try:
        return float(
            np.degrees(R.from_matrix(delta_transform[:3, :3]).as_euler("ZYX")[0])
        )
    except ValueError:
        return 0.0


def motion_rotation_axis_abs(delta_transform: np.ndarray) -> list[float]:
    rotvec = R.from_matrix(delta_transform[:3, :3]).as_rotvec()
    norm = float(np.linalg.norm(rotvec))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    axis = np.abs(rotvec / norm)
    return [float(axis[0]), float(axis[1]), float(axis[2])]


def scan_review_object_descriptor(meta: Any, *, point_count: int | None = None) -> dict:
    return {
        "object_type": "raw_scan",
        "anchor_index": None,
        "support_frame_count": 1,
        "point_count": None if point_count is None else int(point_count),
        "builder_mode": "raw_scan",
        "refined_support_count": 0,
        "support_records": [
            {
                "meta_index": None,
                "timestamp_ns": int(meta.timestamp_ns),
                "record_path": meta.record_path,
                "pose_sync_dt_ms": None,
                "point_count": None if point_count is None else int(point_count),
                "transform_anchor_support": transform_dict_from_matrix(np.eye(4)),
            }
        ],
    }


def submap_review_object_descriptor(submap_info: dict, *, object_type: str) -> dict:
    return {
        "object_type": object_type,
        "anchor_index": int(submap_info["anchor_index"]),
        "support_frame_count": int(submap_info["support_frame_count"]),
        "point_count": int(submap_info.get("point_count", 0)),
        "builder_mode": str(submap_info.get("builder_mode", "pose_only")),
        "refined_support_count": int(submap_info.get("refined_support_count", 0)),
        "support_records": copy.deepcopy(submap_info.get("support_records", [])),
    }


def serialize_motion_review_candidate(
    candidate: dict,
    *,
    lidar_topic: str,
    pose_topic: str,
    selected_for_calibration: bool,
) -> dict:
    return {
        "window_id": int(candidate["window_id"]),
        "start_index": int(candidate["start_index"]),
        "end_index": int(candidate["end_index"]),
        "start_timestamp_ns": int(candidate["start_meta"].timestamp_ns),
        "end_timestamp_ns": int(candidate["end_meta"].timestamp_ns),
        "frame_stride": int(candidate["stride"]),
        "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
        "pose_translation_m": float(candidate["pose_translation_m"]),
        "window_score": float(candidate["score"]),
        "information_score": float(candidate["information_score"]),
        "probabilistic_information_score": (
            None
            if candidate.get("probabilistic_information_score") is None
            else float(candidate["probabilistic_information_score"])
        ),
        "probabilistic_window_score": (
            None
            if candidate.get("probabilistic_window_score") is None
            else float(candidate["probabilistic_window_score"])
        ),
        "information_uncertainty_scale": (
            None
            if candidate.get("information_uncertainty_scale") is None
            else float(candidate["information_uncertainty_scale"])
        ),
        "information_rotation_confidence": (
            None
            if candidate.get("information_rotation_confidence") is None
            else float(candidate["information_rotation_confidence"])
        ),
        "information_translation_confidence": (
            None
            if candidate.get("information_translation_confidence") is None
            else float(candidate["information_translation_confidence"])
        ),
        "observability_segment_id": (
            None
            if candidate.get("observability_segment_id") is None
            else int(candidate["observability_segment_id"])
        ),
        "observability_combined_min_eigenvalue": (
            None
            if candidate.get("observability_combined_min_eigenvalue") is None
            else float(candidate["observability_combined_min_eigenvalue"])
        ),
        "observability_combined_condition_number": (
            None
            if candidate.get("observability_combined_condition_number") is None
            else float(candidate["observability_combined_condition_number"])
        ),
        "observability_min_eigenvalue_gain": (
            None
            if candidate.get("observability_min_eigenvalue_gain") is None
            else float(candidate["observability_min_eigenvalue_gain"])
        ),
        "observability_min_eigenvalue_gain_ratio": (
            None
            if candidate.get("observability_min_eigenvalue_gain_ratio") is None
            else float(candidate["observability_min_eigenvalue_gain_ratio"])
        ),
        "observability_condition_worsening_ratio": (
            None
            if candidate.get("observability_condition_worsening_ratio") is None
            else float(candidate["observability_condition_worsening_ratio"])
        ),
        "observability_capacity_weight": (
            None
            if candidate.get("observability_capacity_weight") is None
            else float(candidate["observability_capacity_weight"])
        ),
        "imu_preintegration_delta_translation_m": (
            None
            if candidate.get("imu_preintegration_delta_translation_m") is None
            else [
                float(value)
                for value in np.asarray(
                    candidate["imu_preintegration_delta_translation_m"], dtype=float
                ).reshape(3)
            ]
        ),
        "imu_preintegration_delta_velocity_mps": (
            None
            if candidate.get("imu_preintegration_delta_velocity_mps") is None
            else [
                float(value)
                for value in np.asarray(
                    candidate["imu_preintegration_delta_velocity_mps"], dtype=float
                ).reshape(3)
            ]
        ),
        "imu_preintegration_confidence": (
            None
            if candidate.get("imu_preintegration_confidence") is None
            else float(candidate["imu_preintegration_confidence"])
        ),
        "imu_preintegration_sample_count": (
            None
            if candidate.get("imu_preintegration_sample_count") is None
            else int(candidate["imu_preintegration_sample_count"])
        ),
        "imu_preintegration_valid_step_count": (
            None
            if candidate.get("imu_preintegration_valid_step_count") is None
            else int(candidate["imu_preintegration_valid_step_count"])
        ),
        "imu_preintegration_duration_sec": (
            None
            if candidate.get("imu_preintegration_duration_sec") is None
            else float(candidate["imu_preintegration_duration_sec"])
        ),
        "imu_preintegration_mean_specific_accel_mps2": (
            None
            if candidate.get("imu_preintegration_mean_specific_accel_mps2") is None
            else float(candidate["imu_preintegration_mean_specific_accel_mps2"])
        ),
        "imu_preintegration_source": (
            None
            if candidate.get("imu_preintegration_source") is None
            else str(candidate["imu_preintegration_source"])
        ),
        "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
        "imu_translation_heading_deg": (
            None
            if candidate.get("imu_translation_heading_deg") is None
            else float(candidate["imu_translation_heading_deg"])
        ),
        "registration_fitness": float(candidate["registration_fitness"]),
        "registration_inlier_rmse": float(candidate["registration_inlier_rmse"]),
        "registered_overlap_quality_score": (
            None
            if candidate.get("registered_overlap_quality_score") is None
            else float(candidate["registered_overlap_quality_score"])
        ),
        "registered_overlap_within_0p4m_ratio": (
            None
            if candidate.get("registered_overlap_within_0p4m_ratio") is None
            else float(candidate["registered_overlap_within_0p4m_ratio"])
        ),
        "registered_overlap_nn_mean_m": (
            None
            if candidate.get("registered_overlap_nn_mean_m") is None
            else float(candidate["registered_overlap_nn_mean_m"])
        ),
        "global_selection_score": (
            None
            if candidate.get("global_selection_score") is None
            else float(candidate["global_selection_score"])
        ),
        "selected_for_calibration": bool(selected_for_calibration),
        "motion_registration_mode": str(candidate["motion_registration_mode"]),
        "lidar_topic": lidar_topic,
        "pose_topic": pose_topic,
        "record_path_start": candidate["start_meta"].record_path,
        "record_path_end": candidate["end_meta"].record_path,
        "imu_delta": transform_dict_from_matrix(candidate["imu_delta"]),
        "lidar_delta": transform_dict_from_matrix(candidate["lidar_delta"]),
        "source_registration_object": copy.deepcopy(
            candidate["source_registration_object"]
        ),
        "target_registration_object": copy.deepcopy(
            candidate["target_registration_object"]
        ),
    }


def candidate_identity(candidate: dict) -> tuple[int, int, int]:
    return (
        int(candidate["start_index"]),
        int(candidate["end_index"]),
        int(candidate["stride"]),
    )


def circular_distance_deg(angle_a: float, angle_b: float) -> float:
    return float(abs(((angle_a - angle_b) + 180.0) % 360.0 - 180.0))


def motion_turn_sign(signed_yaw_deg: float) -> str | None:
    if signed_yaw_deg > 0.5:
        return "left"
    if signed_yaw_deg < -0.5:
        return "right"
    return None


def heading_bin(heading_deg: float | None, bin_width_deg: float = 45.0) -> int | None:
    if heading_deg is None:
        return None
    return int(np.floor((float(heading_deg) + 180.0) / float(bin_width_deg))) % int(
        round(360.0 / float(bin_width_deg))
    )


def candidate_axis_abs_vector(candidate: dict) -> np.ndarray:
    axis = candidate.get("imu_rotation_axis_abs") or [0.0, 0.0, 0.0]
    return np.asarray(axis, dtype=float).reshape(3)


def candidate_overlaps_used_ranges(
    candidate: dict, used_ranges: list[tuple[int, int]]
) -> bool:
    for used_start, used_end in used_ranges:
        if not (
            candidate["end_index"] < used_start or candidate["start_index"] > used_end
        ):
            return True
    return False


def _candidate_information_score(candidate: dict) -> float:
    value = candidate.get("probabilistic_information_score")
    if value is not None:
        return float(value)
    return float(candidate["information_score"])


def _candidate_window_score(candidate: dict) -> float:
    value = candidate.get("probabilistic_window_score")
    if value is not None:
        return float(value)
    return float(candidate["score"])


def select_window_candidates(
    candidates: list[dict], max_candidates_per_window: int
) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_candidate_information_score(item),
            -_candidate_window_score(item),
            -item["pose_rotation_deg"],
            -item["pose_translation_m"],
            -item["stride"],
            item["start_index"],
            item["end_index"],
        ),
    )
    selected: list[dict] = []
    selected_ids: set[tuple[int, int, int]] = set()

    stride_best: dict[int, dict] = {}
    for candidate in ranked:
        stride_best.setdefault(int(candidate["stride"]), candidate)
    for stride in sorted(
        stride_best,
        key=lambda value: -_candidate_information_score(stride_best[value]),
    ):
        candidate = stride_best[stride]
        candidate_id = candidate_identity(candidate)
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        if len(selected) >= max_candidates_per_window:
            return selected

    heading_best: dict[int, dict] = {}
    for candidate in ranked:
        heading_deg = candidate.get("imu_translation_heading_deg")
        if heading_deg is None:
            continue
        candidate_heading_bin = int(np.floor((float(heading_deg) + 180.0) / 45.0))
        heading_best.setdefault(candidate_heading_bin, candidate)
    for candidate_heading_bin in sorted(
        heading_best,
        key=lambda value: -_candidate_information_score(heading_best[value]),
    ):
        candidate = heading_best[candidate_heading_bin]
        candidate_id = candidate_identity(candidate)
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        if len(selected) >= max_candidates_per_window:
            return selected

    for candidate in ranked:
        candidate_id = candidate_identity(candidate)
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        if len(selected) >= max_candidates_per_window:
            break
    return selected


def summarize_motion_candidate(candidate: dict) -> dict:
    return {
        "start_timestamp_ns": int(candidate["start_meta"].timestamp_ns),
        "end_timestamp_ns": int(candidate["end_meta"].timestamp_ns),
        "frame_stride": int(candidate["stride"]),
        "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
        "pose_translation_m": float(candidate["pose_translation_m"]),
        "window_score": float(candidate["score"]),
        "information_score": float(candidate["information_score"]),
        "probabilistic_information_score": (
            None
            if candidate.get("probabilistic_information_score") is None
            else float(candidate["probabilistic_information_score"])
        ),
        "probabilistic_window_score": (
            None
            if candidate.get("probabilistic_window_score") is None
            else float(candidate["probabilistic_window_score"])
        ),
        "information_uncertainty_scale": (
            None
            if candidate.get("information_uncertainty_scale") is None
            else float(candidate["information_uncertainty_scale"])
        ),
        "information_rotation_confidence": (
            None
            if candidate.get("information_rotation_confidence") is None
            else float(candidate["information_rotation_confidence"])
        ),
        "information_translation_confidence": (
            None
            if candidate.get("information_translation_confidence") is None
            else float(candidate["information_translation_confidence"])
        ),
        "observability_segment_id": (
            None
            if candidate.get("observability_segment_id") is None
            else int(candidate["observability_segment_id"])
        ),
        "observability_combined_min_eigenvalue": (
            None
            if candidate.get("observability_combined_min_eigenvalue") is None
            else float(candidate["observability_combined_min_eigenvalue"])
        ),
        "observability_combined_condition_number": (
            None
            if candidate.get("observability_combined_condition_number") is None
            else float(candidate["observability_combined_condition_number"])
        ),
        "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
        "imu_translation_heading_deg": (
            None
            if candidate.get("imu_translation_heading_deg") is None
            else float(candidate["imu_translation_heading_deg"])
        ),
        "registered_overlap_quality_score": (
            None
            if candidate.get("registered_overlap_quality_score") is None
            else float(candidate["registered_overlap_quality_score"])
        ),
        "registered_overlap_within_0p4m_ratio": (
            None
            if candidate.get("registered_overlap_within_0p4m_ratio") is None
            else float(candidate["registered_overlap_within_0p4m_ratio"])
        ),
        "registered_overlap_nn_mean_m": (
            None
            if candidate.get("registered_overlap_nn_mean_m") is None
            else float(candidate["registered_overlap_nn_mean_m"])
        ),
    }


def global_motion_selection_score(
    candidate: dict, selected_candidates: list[dict], base_stride: int
) -> float:
    info_components = motion_information_components(candidate, base_stride=base_stride)
    base_quality = max(float(info_components["probabilistic_information_score"]), 1e-6)
    multiplier = float(info_components["probabilistic_stride_multiplier"])
    uncertainty_scale = float(info_components["uncertainty_scale"])
    if uncertainty_scale < 0.8:
        multiplier -= min((0.8 - uncertainty_scale) * 0.35, 0.20)
    elif uncertainty_scale > 1.2:
        multiplier += min((uncertainty_scale - 1.2) * 0.15, 0.10)

    candidate_stride = int(candidate["stride"])
    selected_stride_counts: dict[int, int] = {}
    for item in selected_candidates:
        stride = int(item["stride"])
        selected_stride_counts[stride] = selected_stride_counts.get(stride, 0) + 1
    same_stride_count = selected_stride_counts.get(candidate_stride, 0)
    if same_stride_count == 0:
        multiplier += 0.35
    else:
        multiplier -= min(0.10 * float(same_stride_count), 0.30)

    turn_sign = motion_turn_sign(float(candidate["imu_signed_yaw_deg"]))
    selected_turn_counts = {"left": 0, "right": 0}
    for item in selected_candidates:
        item_turn_sign = motion_turn_sign(float(item["imu_signed_yaw_deg"]))
        if item_turn_sign in selected_turn_counts:
            selected_turn_counts[item_turn_sign] += 1
    if turn_sign is not None:
        opposite_turn_sign = "right" if turn_sign == "left" else "left"
        if selected_turn_counts[turn_sign] == 0:
            multiplier += 0.20
        if selected_turn_counts[turn_sign] < selected_turn_counts[opposite_turn_sign]:
            multiplier += 0.20
        elif selected_turn_counts[turn_sign] > selected_turn_counts[opposite_turn_sign]:
            multiplier -= min(
                0.06
                * float(
                    selected_turn_counts[turn_sign]
                    - selected_turn_counts[opposite_turn_sign]
                ),
                0.18,
            )

    heading_deg = candidate.get("imu_translation_heading_deg")
    selected_headings = [
        float(item["imu_translation_heading_deg"])
        for item in selected_candidates
        if item.get("imu_translation_heading_deg") is not None
    ]
    candidate_heading_bin = heading_bin(
        None if heading_deg is None else float(heading_deg)
    )
    selected_heading_bin_counts: dict[int, int] = {}
    for existing_heading in selected_headings:
        existing_heading_bin = heading_bin(existing_heading)
        if existing_heading_bin is None:
            continue
        selected_heading_bin_counts[existing_heading_bin] = (
            selected_heading_bin_counts.get(existing_heading_bin, 0) + 1
        )
    if heading_deg is not None:
        same_heading_bin_count = (
            0
            if candidate_heading_bin is None
            else selected_heading_bin_counts.get(candidate_heading_bin, 0)
        )
        if same_heading_bin_count == 0:
            multiplier += 0.35
        else:
            multiplier -= min(0.08 * float(same_heading_bin_count), 0.24)
        if not selected_headings:
            multiplier += 0.15
        else:
            min_heading_distance = min(
                circular_distance_deg(float(heading_deg), existing_heading)
                for existing_heading in selected_headings
            )
            multiplier += min_heading_distance / 240.0
            if min_heading_distance < 20.0:
                multiplier -= 0.12

    candidate_axis = candidate_axis_abs_vector(candidate)
    selected_axes = [candidate_axis_abs_vector(item) for item in selected_candidates]
    if selected_axes:
        min_axis_distance = min(
            float(np.linalg.norm(candidate_axis - existing_axis))
            for existing_axis in selected_axes
        )
        multiplier += 0.20 * min(min_axis_distance, 1.0)
        selected_axis_z_mean = float(
            np.mean([float(existing_axis[2]) for existing_axis in selected_axes])
        )
        if candidate_axis[2] < 0.95 and selected_axis_z_mean >= 0.95:
            multiplier += 0.20
        elif candidate_axis[2] >= 0.98 and selected_axis_z_mean >= 0.95:
            multiplier -= 0.08
    elif candidate_axis[2] < 0.95:
        multiplier += 0.10

    return float(base_quality * max(multiplier, 0.45))


def build_motion_windows(
    motion_candidates: list[dict],
    lidar_meta_count: int,
    max_motion_samples: int,
    base_stride: int,
    min_window_rotation_deg: float,
    min_window_translation_m: float,
    max_candidates_per_window: int = 3,
) -> list[dict]:
    if not motion_candidates:
        return []

    window_size = max(
        base_stride * 4,
        int(np.ceil(lidar_meta_count / max(max_motion_samples, 1))),
    )
    windows_by_id: dict[int, dict] = {}
    for candidate in motion_candidates:
        midpoint_index = (candidate["start_index"] + candidate["end_index"]) // 2
        window_id = midpoint_index // window_size
        candidate_with_window = {
            **candidate,
            "midpoint_index": midpoint_index,
            "window_id": window_id,
        }
        window_record = windows_by_id.setdefault(
            window_id,
            {
                "window_id": window_id,
                "window_start_index": window_id * window_size,
                "window_end_index": min(
                    lidar_meta_count - 1, ((window_id + 1) * window_size) - 1
                ),
                "candidates": [],
            },
        )
        window_record["candidates"].append(candidate_with_window)

    windows = []
    for window_id in sorted(windows_by_id):
        window_record = windows_by_id[window_id]
        candidates = sorted(
            window_record["candidates"],
            key=lambda item: (
                -_candidate_information_score(item),
                -_candidate_window_score(item),
                -item["pose_rotation_deg"],
                -item["pose_translation_m"],
                item["start_index"],
                item["end_index"],
            ),
        )
        rotation_candidates = [
            candidate
            for candidate in candidates
            if candidate["pose_rotation_deg"] >= min_window_rotation_deg
        ]
        preferred_candidates = rotation_candidates or candidates
        best_candidate = preferred_candidates[0]
        gate_reasons = []
        if (
            best_candidate["pose_rotation_deg"] < min_window_rotation_deg
            and best_candidate["pose_translation_m"] < min_window_translation_m
        ):
            gate_reasons.append("weak_window_excitation")
        windows.append(
            {
                "window_id": window_id,
                "window_start_index": window_record["window_start_index"],
                "window_end_index": window_record["window_end_index"],
                "candidate_count": len(candidates),
                "rotation_candidate_count": len(rotation_candidates),
                "has_rotation_candidate": bool(rotation_candidates),
                "best_score": float(best_candidate["score"]),
                "best_information_score": float(best_candidate["information_score"]),
                "best_probabilistic_score": float(
                    _candidate_window_score(best_candidate)
                ),
                "best_probabilistic_information_score": float(
                    _candidate_information_score(best_candidate)
                ),
                "best_pose_rotation_deg": float(best_candidate["pose_rotation_deg"]),
                "best_pose_translation_m": float(best_candidate["pose_translation_m"]),
                "stride_values": sorted(
                    {int(candidate["stride"]) for candidate in preferred_candidates}
                ),
                "valid": not gate_reasons,
                "gate_reasons": gate_reasons,
                "candidates": select_window_candidates(
                    preferred_candidates, max_candidates_per_window
                ),
            }
        )
    return sorted(windows, key=lambda item: item["window_start_index"])
