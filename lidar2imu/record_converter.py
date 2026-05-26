#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created Date: 2026-02-09
# Author: daohu527

from __future__ import annotations

# isort: off
import argparse
import bisect
import copy
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import (
    circular_span_deg,
    normalize_plane,
    normalize_vector,
)
from lidar2imu.io import load_dataset, write_outputs
from lidar2imu.models import CalibrationConfig
from lidar2imu.pipeline import run_calibration
from lidar2lidar.extrinsic_io import load_extrinsics_file
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic
from lidar2lidar.prepared_dataset import (
    ImuSample,
    PoseSample,
    collect_imu_samples,
    collect_pose_samples,
    load_prepared_rig_dataset,
)
from lidar2lidar.record_utils import (
    build_transform_graph,
    collect_pointcloud_metadata,
    discover_record_files,
    extract_tf_edges,
    get_topic_frame_ids,
    load_pointcloud_from_meta,
    lookup_transform,
)

# isort: on

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RUN_PROFILE_PRESETS = {
    "baseline": {
        "gravity_source": "pose",
        "motion_frame_stride": 5,
        "min_registration_fitness": 0.55,
        "motion_registration_mode": "scan_to_scan",
        "planar_motion_policy": "auto",
        "auto_reextract_if_needed": False,
    },
    "production": {
        "gravity_source": "pose",
        "motion_frame_stride": 5,
        "min_registration_fitness": 0.55,
        "motion_registration_mode": "submap_to_map",
        "submap_half_window": 2,
        "submap_support_stride": 5,
        "submap_min_support_frames": 3,
        "map_half_window": 6,
        "map_support_stride": 10,
        "map_min_support_frames": 5,
        "planar_motion_policy": "auto",
        "auto_reextract_if_needed": True,
    },
}

RUN_PROFILE_OPTION_NAMES = {
    "gravity_source": ("--gravity-source",),
    "motion_frame_stride": ("--motion-frame-stride",),
    "min_registration_fitness": ("--min-registration-fitness",),
    "motion_registration_mode": ("--motion-registration-mode",),
    "planar_motion_policy": ("--planar-motion-policy",),
    "submap_half_window": ("--submap-half-window",),
    "submap_support_stride": ("--submap-support-stride",),
    "submap_min_support_frames": ("--submap-min-support-frames",),
    "map_half_window": ("--map-half-window",),
    "map_support_stride": ("--map-support-stride",),
    "map_min_support_frames": ("--map-min-support-frames",),
    "auto_reextract_if_needed": ("--auto-reextract-if-needed",),
}


def _explicit_cli_options(argv: list[str]) -> set[str]:
    options = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        options.add(token.split("=", 1)[0])
    return options


def _apply_run_profile(args: argparse.Namespace, explicit_options: set[str]) -> None:
    if args.profile is None:
        return
    preset = RUN_PROFILE_PRESETS[args.profile]
    for field_name, value in preset.items():
        option_names = RUN_PROFILE_OPTION_NAMES.get(field_name, ())
        if any(option_name in explicit_options for option_name in option_names):
            continue
        setattr(args, field_name, value)


def _uniform_indices(length: int, max_items: int) -> list[int]:
    if length <= 0:
        return []
    if max_items <= 0 or length <= max_items:
        return list(range(length))
    return np.linspace(0, length - 1, num=max_items, dtype=int).tolist()


def _nearest_index(sorted_timestamps: list[int], timestamp_ns: int) -> int | None:
    if not sorted_timestamps:
        return None
    index = bisect.bisect_left(sorted_timestamps, timestamp_ns)
    candidates = []
    if index < len(sorted_timestamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: abs(sorted_timestamps[candidate] - timestamp_ns),
    )


def _nearest_sample(
    samples: list, timestamps: list[int], timestamp_ns: int, max_delta_ns: int
):
    index = _nearest_index(timestamps, timestamp_ns)
    if index is None:
        return None, None
    sample = samples[index]
    delta_ns = abs(timestamps[index] - timestamp_ns)
    if delta_ns > max_delta_ns:
        return None, delta_ns
    return sample, delta_ns


def _windowed_imu_gravity(
    imu_samples: list[ImuSample],
    imu_timestamps: list[int],
    timestamp_ns: int,
    window_ns: int,
) -> np.ndarray | None:
    if not imu_samples:
        return None
    left = bisect.bisect_left(imu_timestamps, timestamp_ns - window_ns)
    right = bisect.bisect_right(imu_timestamps, timestamp_ns + window_ns)
    if left >= right:
        sample, _ = _nearest_sample(
            imu_samples, imu_timestamps, timestamp_ns, window_ns
        )
        if sample is None:
            return None
        return sample.linear_acceleration
    values = np.asarray(
        [sample.linear_acceleration for sample in imu_samples[left:right]], dtype=float
    )
    return np.median(values, axis=0)


def _filter_ground_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    radius = np.linalg.norm(points[:, :2], axis=1)
    mask = np.isfinite(points).all(axis=1)
    mask &= radius >= 3.0
    mask &= radius <= 60.0
    filtered = points[mask]
    if filtered.shape[0] < 500:
        return filtered
    z_limit = float(np.percentile(filtered[:, 2], 35))
    mask_low = filtered[:, 2] <= z_limit + 0.8
    return filtered[mask_low]


def _extract_ground_plane(
    cloud: o3d.geometry.PointCloud,
    expected_up_lidar: np.ndarray,
    plane_dist_thresh: float,
    normal_angle_thresh_deg: float,
) -> tuple[dict | None, dict]:
    points = _filter_ground_points(np.asarray(cloud.points))
    diagnostics = {
        "input_points": int(len(cloud.points)),
        "candidate_points": int(points.shape[0]),
    }
    if points.shape[0] < 200:
        diagnostics["reason"] = "too_few_candidate_points"
        return None, diagnostics

    candidate_cloud = o3d.geometry.PointCloud()
    candidate_cloud.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = candidate_cloud.segment_plane(
        distance_threshold=plane_dist_thresh,
        ransac_n=3,
        num_iterations=500,
    )
    normal, offset = normalize_plane(
        np.asarray(plane_model[:3], dtype=float), float(plane_model[3])
    )
    if float(np.dot(normal, expected_up_lidar)) < 0.0:
        normal = -normal
        offset = -offset
    normal, offset = normalize_plane(normal, offset)
    angle_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(normal, expected_up_lidar), -1.0, 1.0)))
    )
    diagnostics.update(
        {
            "inlier_count": int(len(inliers)),
            "inlier_ratio": float(len(inliers) / max(len(points), 1)),
            "normal_angle_to_expected_up_deg": angle_deg,
        }
    )
    if angle_deg > normal_angle_thresh_deg:
        diagnostics["reason"] = "plane_not_ground_like"
        return None, diagnostics

    result = {
        "lidar_plane_normal": [float(value) for value in normal.tolist()],
        "lidar_plane_offset": float(offset),
        "inlier_ratio": float(len(inliers) / max(len(points), 1)),
        "normal_angle_to_expected_up_deg": angle_deg,
    }
    return result, diagnostics


def _relative_motion(
    transform_world_start: np.ndarray, transform_world_end: np.ndarray
) -> np.ndarray:
    return np.linalg.inv(transform_world_end) @ transform_world_start


def _rotation_from_matrix(matrix: np.ndarray) -> R:
    return R.from_matrix(np.array(matrix, dtype=np.float64, copy=True))


def _motion_excitation(delta_transform: np.ndarray) -> tuple[float, float]:
    rotation_deg = float(
        np.degrees(np.linalg.norm(_rotation_from_matrix(delta_transform[:3, :3]).as_rotvec()))
    )
    translation_m = float(np.linalg.norm(delta_transform[:3, 3]))
    return rotation_deg, translation_m


def _motion_translation_heading_deg(delta_transform: np.ndarray) -> float | None:
    translation = np.asarray(delta_transform[:3, 3], dtype=float).reshape(3)
    if np.linalg.norm(translation[:2]) <= 1e-9:
        return None
    return float(np.degrees(np.arctan2(translation[1], translation[0])))


def _motion_signed_yaw_deg(delta_transform: np.ndarray) -> float:
    try:
        return float(
            np.degrees(_rotation_from_matrix(delta_transform[:3, :3]).as_euler("ZYX")[0])
        )
    except ValueError:
        return 0.0


def _motion_rotation_axis_abs(delta_transform: np.ndarray) -> list[float]:
    rotvec = _rotation_from_matrix(delta_transform[:3, :3]).as_rotvec()
    norm = float(np.linalg.norm(rotvec))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    axis = np.abs(rotvec / norm)
    return [float(axis[0]), float(axis[1]), float(axis[2])]


def _candidate_identity(candidate: dict) -> tuple[int, int, int]:
    return (
        int(candidate["start_index"]),
        int(candidate["end_index"]),
        int(candidate["stride"]),
    )


def _circular_distance_deg(angle_a: float, angle_b: float) -> float:
    return float(abs(((angle_a - angle_b) + 180.0) % 360.0 - 180.0))


def _motion_turn_sign(signed_yaw_deg: float) -> str | None:
    if signed_yaw_deg > 0.5:
        return "left"
    if signed_yaw_deg < -0.5:
        return "right"
    return None


def _heading_bin(heading_deg: float | None, bin_width_deg: float = 45.0) -> int | None:
    if heading_deg is None:
        return None
    return int(np.floor((float(heading_deg) + 180.0) / float(bin_width_deg))) % int(
        round(360.0 / float(bin_width_deg))
    )


def _candidate_axis_abs_vector(candidate: dict) -> np.ndarray:
    axis = candidate.get("imu_rotation_axis_abs") or [0.0, 0.0, 0.0]
    return np.asarray(axis, dtype=float).reshape(3)


def _load_cached_cloud(meta, cloud_cache: dict) -> o3d.geometry.PointCloud:
    cache_key = (meta.topic, int(meta.timestamp_ns))
    if cache_key not in cloud_cache:
        cloud_cache[cache_key] = load_pointcloud_from_meta(meta)
    return cloud_cache[cache_key]


def _resolve_world_lidar_transform(
    meta_index: int,
    *,
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    extraction_transform: np.ndarray,
    alignment_cache: dict[int, dict],
) -> dict:
    if meta_index in alignment_cache:
        return alignment_cache[meta_index]
    meta = lidar_metas[meta_index]
    pose_sample, pose_dt_ns = _nearest_sample(
        pose_samples,
        pose_timestamps,
        meta.timestamp_ns,
        sync_threshold_ns,
    )
    if pose_sample is None:
        alignment_cache[meta_index] = {
            "valid": False,
            "pose_sync_dt_ms": (
                None if pose_dt_ns is None else float(pose_dt_ns / 1e6)
            ),
            "reason": "missing_pose_sync",
        }
        return alignment_cache[meta_index]
    alignment_cache[meta_index] = {
        "valid": True,
        "transform_world_lidar": pose_sample.transform_world_imu @ extraction_transform,
        "pose_sync_dt_ms": float(pose_dt_ns / 1e6),
    }
    return alignment_cache[meta_index]


def _build_local_lidar_submap(
    anchor_index: int,
    *,
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    extraction_transform: np.ndarray,
    submap_half_window: int,
    submap_support_stride: int,
    submap_min_support_frames: int,
    submap_voxel_size: float,
    alignment_cache: dict[int, dict],
    cloud_cache: dict,
    submap_cache: dict,
) -> tuple[o3d.geometry.PointCloud | None, dict]:
    cache_key = (
        int(anchor_index),
        int(submap_half_window),
        int(submap_support_stride),
        int(submap_min_support_frames),
        float(submap_voxel_size),
    )
    if cache_key in submap_cache:
        cloud, info = submap_cache[cache_key]
        return copy.deepcopy(cloud), dict(info)

    anchor_alignment = _resolve_world_lidar_transform(
        anchor_index,
        lidar_metas=lidar_metas,
        pose_samples=pose_samples,
        pose_timestamps=pose_timestamps,
        sync_threshold_ns=sync_threshold_ns,
        extraction_transform=extraction_transform,
        alignment_cache=alignment_cache,
    )
    if not anchor_alignment["valid"]:
        info = {
            "valid": False,
            "reason": "anchor_missing_pose_sync",
            "anchor_index": int(anchor_index),
            "support_frame_count": 0,
        }
        submap_cache[cache_key] = (None, info)
        return None, dict(info)

    anchor_pose = np.asarray(anchor_alignment["transform_world_lidar"], dtype=float)
    merged_cloud = o3d.geometry.PointCloud()
    support_records = []
    max_index = len(lidar_metas) - 1
    support_indices = []
    for offset in range(-int(submap_half_window), int(submap_half_window) + 1):
        support_index = anchor_index + (offset * int(submap_support_stride))
        if 0 <= support_index <= max_index:
            support_indices.append(int(support_index))
    support_indices = sorted(set(support_indices))

    for support_index in support_indices:
        support_alignment = _resolve_world_lidar_transform(
            support_index,
            lidar_metas=lidar_metas,
            pose_samples=pose_samples,
            pose_timestamps=pose_timestamps,
            sync_threshold_ns=sync_threshold_ns,
            extraction_transform=extraction_transform,
            alignment_cache=alignment_cache,
        )
        if not support_alignment["valid"]:
            continue
        support_pose = np.asarray(
            support_alignment["transform_world_lidar"], dtype=float
        )
        support_cloud = copy.deepcopy(
            _load_cached_cloud(lidar_metas[support_index], cloud_cache)
        )
        transform_anchor_support = np.linalg.inv(anchor_pose) @ support_pose
        support_cloud.transform(transform_anchor_support)
        merged_cloud += support_cloud
        support_records.append(
            {
                "meta_index": int(support_index),
                "timestamp_ns": int(lidar_metas[support_index].timestamp_ns),
                "pose_sync_dt_ms": float(support_alignment["pose_sync_dt_ms"]),
                "point_count": int(len(support_cloud.points)),
            }
        )

    if len(support_records) < int(submap_min_support_frames):
        info = {
            "valid": False,
            "reason": "insufficient_submap_support",
            "anchor_index": int(anchor_index),
            "support_frame_count": int(len(support_records)),
            "support_records": support_records,
        }
        submap_cache[cache_key] = (None, info)
        return None, dict(info)

    if submap_voxel_size > 0.0:
        merged_cloud = merged_cloud.voxel_down_sample(float(submap_voxel_size))

    info = {
        "valid": True,
        "anchor_index": int(anchor_index),
        "support_frame_count": int(len(support_records)),
        "point_count": int(len(merged_cloud.points)),
        "support_records": support_records,
    }
    submap_cache[cache_key] = (merged_cloud, info)
    return copy.deepcopy(merged_cloud), dict(info)


def _build_motion_candidates(
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    base_stride: int,
) -> list[dict]:
    if base_stride < 1:
        raise ValueError("motion_frame_stride must be >= 1.")

    candidate_records: list[dict] = []
    stride_values = []
    stride = base_stride
    max_stride = max(base_stride, min(len(lidar_metas) // 2, base_stride * 8))
    while stride <= max_stride:
        stride_values.append(stride)
        stride *= 2

    for stride in stride_values:
        for start_index in range(0, len(lidar_metas) - stride):
            end_index = start_index + stride
            start_meta = lidar_metas[start_index]
            end_meta = lidar_metas[end_index]
            start_pose, start_pose_dt_ns = _nearest_sample(
                pose_samples,
                pose_timestamps,
                start_meta.timestamp_ns,
                sync_threshold_ns,
            )
            end_pose, end_pose_dt_ns = _nearest_sample(
                pose_samples,
                pose_timestamps,
                end_meta.timestamp_ns,
                sync_threshold_ns,
            )
            if start_pose is None or end_pose is None:
                continue
            imu_delta = _relative_motion(
                start_pose.transform_world_imu, end_pose.transform_world_imu
            )
            rotation_deg, translation_m = _motion_excitation(imu_delta)
            information_score = rotation_deg * 10.0 + translation_m
            candidate_records.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "stride": stride,
                    "start_meta": start_meta,
                    "end_meta": end_meta,
                    "start_pose": start_pose,
                    "end_pose": end_pose,
                    "start_pose_dt_ns": start_pose_dt_ns,
                    "end_pose_dt_ns": end_pose_dt_ns,
                    "imu_delta": imu_delta,
                    "pose_rotation_deg": rotation_deg,
                    "pose_translation_m": translation_m,
                    "imu_translation_heading_deg": _motion_translation_heading_deg(
                        imu_delta
                    ),
                    "imu_signed_yaw_deg": _motion_signed_yaw_deg(imu_delta),
                    "imu_rotation_axis_abs": _motion_rotation_axis_abs(imu_delta),
                    "information_score": information_score,
                    "score": information_score / float(stride),
                }
            )

    candidate_records.sort(
        key=lambda item: (
            -item["pose_rotation_deg"],
            -item["pose_translation_m"],
            item["start_index"],
            item["end_index"],
        )
    )

    return candidate_records


def _candidate_overlaps_used_ranges(
    candidate: dict, used_ranges: list[tuple[int, int]]
) -> bool:
    for used_start, used_end in used_ranges:
        if not (
            candidate["end_index"] < used_start or candidate["start_index"] > used_end
        ):
            return True
    return False


def _select_window_candidates(
    candidates: list[dict], max_candidates_per_window: int
) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["information_score"],
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
        stride_best, key=lambda value: -stride_best[value]["information_score"]
    ):
        candidate = stride_best[stride]
        candidate_id = _candidate_identity(candidate)
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
        heading_bin = int(np.floor((float(heading_deg) + 180.0) / 45.0))
        heading_best.setdefault(heading_bin, candidate)
    for heading_bin in sorted(
        heading_best, key=lambda value: -heading_best[value]["information_score"]
    ):
        candidate = heading_best[heading_bin]
        candidate_id = _candidate_identity(candidate)
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        if len(selected) >= max_candidates_per_window:
            return selected

    for candidate in ranked:
        candidate_id = _candidate_identity(candidate)
        if candidate_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        if len(selected) >= max_candidates_per_window:
            break
    return selected


def _summarize_motion_candidate(candidate: dict) -> dict:
    return {
        "start_timestamp_ns": int(candidate["start_meta"].timestamp_ns),
        "end_timestamp_ns": int(candidate["end_meta"].timestamp_ns),
        "frame_stride": int(candidate["stride"]),
        "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
        "pose_translation_m": float(candidate["pose_translation_m"]),
        "window_score": float(candidate["score"]),
        "information_score": float(candidate["information_score"]),
        "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
        "imu_translation_heading_deg": (
            None
            if candidate.get("imu_translation_heading_deg") is None
            else float(candidate["imu_translation_heading_deg"])
        ),
    }


def _global_motion_selection_score(
    candidate: dict, selected_candidates: list[dict], base_stride: int
) -> float:
    base_quality = max(float(candidate["registration_fitness"]), 1e-3) * max(
        float(candidate["information_score"]), 1e-6
    )
    stride_bonus = min(
        np.log2(float(candidate["stride"]) / float(max(base_stride, 1)) + 1.0),
        2.0,
    )
    multiplier = 1.0 + (0.08 * float(stride_bonus))

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

    turn_sign = _motion_turn_sign(float(candidate["imu_signed_yaw_deg"]))
    selected_turn_counts = {"left": 0, "right": 0}
    for item in selected_candidates:
        item_turn_sign = _motion_turn_sign(float(item["imu_signed_yaw_deg"]))
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
    candidate_heading_bin = _heading_bin(
        None if heading_deg is None else float(heading_deg)
    )
    selected_heading_bin_counts: dict[int, int] = {}
    for existing_heading in selected_headings:
        heading_bin = _heading_bin(existing_heading)
        if heading_bin is None:
            continue
        selected_heading_bin_counts[heading_bin] = (
            selected_heading_bin_counts.get(heading_bin, 0) + 1
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
                _circular_distance_deg(float(heading_deg), existing_heading)
                for existing_heading in selected_headings
            )
            multiplier += min_heading_distance / 240.0
            if min_heading_distance < 20.0:
                multiplier -= 0.12

    candidate_axis = _candidate_axis_abs_vector(candidate)
    selected_axes = [_candidate_axis_abs_vector(item) for item in selected_candidates]
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


def _build_motion_windows(
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
                -item["information_score"],
                -item["score"],
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
                "best_pose_rotation_deg": float(best_candidate["pose_rotation_deg"]),
                "best_pose_translation_m": float(best_candidate["pose_translation_m"]),
                "stride_values": sorted(
                    {int(candidate["stride"]) for candidate in preferred_candidates}
                ),
                "valid": not gate_reasons,
                "gate_reasons": gate_reasons,
                "candidates": _select_window_candidates(
                    preferred_candidates, max_candidates_per_window
                ),
            }
        )
    return sorted(windows, key=lambda item: item["window_start_index"])


def convert_record_to_standardized_samples(
    record_path: str | None,
    output_dir: str,
    lidar_topic: str,
    pose_topic: str,
    imu_topic: str,
    parent_frame: str,
    child_frame: str | None,
    initial_transform_path: str | None,
    extraction_transform_path: str | None,
    identity_initial_transform: bool,
    gravity_source: str,
    ground_pose_sync_threshold_ms: float,
    motion_pose_sync_threshold_ms: float,
    imu_gravity_window_ms: float,
    max_ground_samples: int,
    max_motion_samples: int,
    motion_frame_stride: int,
    plane_dist_thresh: float,
    plane_normal_thresh_deg: float,
    registration_voxel_size: float,
    min_registration_fitness: float,
    calibration_loss: str,
    calibration_motion_rotation_deg: float,
    calibration_planar_motion_policy: str,
    motion_registration_mode: str = "scan_to_scan",
    submap_half_window: int = 2,
    submap_support_stride: int | None = None,
    submap_min_support_frames: int = 3,
    submap_voxel_size: float | None = None,
    map_half_window: int | None = None,
    map_support_stride: int | None = None,
    map_min_support_frames: int | None = None,
    map_voxel_size: float | None = None,
    run_profile: str | None = None,
    prepared_dataset_yaml: str | None = None,
) -> tuple[Path, dict]:
    prepared_dataset = (
        load_prepared_rig_dataset(prepared_dataset_yaml)
        if prepared_dataset_yaml is not None
        else None
    )
    if prepared_dataset is not None:
        if lidar_topic not in prepared_dataset.metadata_by_topic:
            raise RuntimeError(
                f"LiDAR topic {lidar_topic} is not present in prepared dataset {prepared_dataset_yaml}."
            )
        if prepared_dataset.pose_topic and pose_topic != prepared_dataset.pose_topic:
            raise RuntimeError(
                f"Requested pose topic {pose_topic} does not match prepared dataset topic "
                f"{prepared_dataset.pose_topic}."
            )
        if (
            gravity_source == "imu"
            and prepared_dataset.imu_topic
            and imu_topic != prepared_dataset.imu_topic
        ):
            raise RuntimeError(
                f"Requested IMU topic {imu_topic} does not match prepared dataset topic "
                f"{prepared_dataset.imu_topic}."
            )
        record_files = prepared_dataset.record_files
        lidar_frame = child_frame or prepared_dataset.topic_infos.get(
            lidar_topic, {}
        ).get("frame_id", "")
        if not lidar_frame and prepared_dataset.metadata_by_topic[lidar_topic]:
            lidar_frame = prepared_dataset.metadata_by_topic[lidar_topic][0].frame_id
        tf_edges = prepared_dataset.tf_edges
        metadata_by_topic = {
            lidar_topic: prepared_dataset.metadata_by_topic[lidar_topic]
        }
    else:
        if record_path is None:
            raise RuntimeError(
                "Either record_path or prepared_dataset_yaml must be provided."
            )
        record_files = discover_record_files(record_path)
        topic_frame_ids = get_topic_frame_ids(record_files, [lidar_topic])
        lidar_frame = child_frame or topic_frame_ids.get(lidar_topic, "")
        if not lidar_frame:
            raise RuntimeError(
                f"Failed to infer frame_id for lidar topic {lidar_topic}."
            )
        tf_edges = extract_tf_edges(record_files)
        metadata_by_topic = collect_pointcloud_metadata(record_files, [lidar_topic])

    if not lidar_frame:
        raise RuntimeError(f"Failed to infer frame_id for lidar topic {lidar_topic}.")
    tf_graph = build_transform_graph(tf_edges)
    record_reference_transform = lookup_transform(tf_graph, lidar_frame, parent_frame)
    reference_transform_source = "/tf_static or merged tf graph"
    initial_transform_source = reference_transform_source

    def _load_override_transform(path: str, *, kind: str) -> tuple[np.ndarray, str]:
        transform, file_parent_frame, file_child_frame, _, _ = load_extrinsics_file(
            path
        )
        if file_parent_frame and file_parent_frame != parent_frame:
            raise RuntimeError(
                f"{kind} parent frame {file_parent_frame} does not match requested "
                f"parent frame {parent_frame}."
            )
        if file_child_frame and file_child_frame != lidar_frame:
            raise RuntimeError(
                f"{kind} child frame {file_child_frame} does not match LiDAR frame "
                f"{lidar_frame}."
            )
        return np.asarray(transform, dtype=float), str(Path(path).resolve())

    if initial_transform_path is not None:
        initial_transform, initial_transform_source = _load_override_transform(
            initial_transform_path,
            kind="Initial transform",
        )
    else:
        initial_transform = record_reference_transform
        if initial_transform is None:
            if not identity_initial_transform:
                raise RuntimeError(
                    f"Could not find transform from {lidar_frame} to {parent_frame}. "
                    "Provide --initial-transform or enable --identity-initial-transform "
                    "for exploratory runs."
                )
            initial_transform = np.eye(4, dtype=float)
            initial_transform_source = "identity_fallback"
            reference_transform_source = None
    if extraction_transform_path is not None:
        extraction_transform, extraction_transform_source = _load_override_transform(
            extraction_transform_path,
            kind="Extraction transform",
        )
    elif record_reference_transform is not None:
        extraction_transform = np.asarray(record_reference_transform, dtype=float)
        extraction_transform_source = reference_transform_source
    else:
        extraction_transform = np.asarray(initial_transform, dtype=float)
        extraction_transform_source = initial_transform_source

    localization_to_imu = lookup_transform(tf_graph, parent_frame, "localization")
    if localization_to_imu is None:
        raise RuntimeError(
            f"Could not find transform from {parent_frame} to localization. "
            "This conversion currently expects the localization frame to be present."
        )

    lidar_metas = metadata_by_topic[lidar_topic]
    if not lidar_metas:
        raise RuntimeError(f"No point clouds found for topic {lidar_topic}.")

    pose_samples = (
        prepared_dataset.pose_samples
        if prepared_dataset is not None
        else collect_pose_samples(record_files, pose_topic, localization_to_imu)
    )
    if not pose_samples:
        raise RuntimeError(f"No pose messages found on {pose_topic}.")
    pose_timestamps = [sample.timestamp_ns for sample in pose_samples]

    imu_samples = (
        (
            prepared_dataset.imu_samples
            if prepared_dataset is not None
            else collect_imu_samples(record_files, imu_topic)
        )
        if gravity_source == "imu"
        else []
    )
    imu_timestamps = [sample.timestamp_ns for sample in imu_samples]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_path = output_path / "standardized_samples.yaml"
    diagnostics_path = output_path / "conversion_diagnostics.yaml"

    ground_samples = []
    ground_diagnostics = []
    ground_indices = _uniform_indices(len(lidar_metas), max_ground_samples)
    ground_sync_threshold_ns = int(ground_pose_sync_threshold_ms * 1e6)
    imu_window_ns = int(imu_gravity_window_ms * 1e6)

    for meta_index in ground_indices:
        lidar_meta = lidar_metas[meta_index]
        pose_sample, pose_dt_ns = _nearest_sample(
            pose_samples,
            pose_timestamps,
            lidar_meta.timestamp_ns,
            ground_sync_threshold_ns,
        )
        diagnostic = {
            "timestamp_ns": int(lidar_meta.timestamp_ns),
            "record_path": lidar_meta.record_path,
            "pose_sync_dt_ms": None if pose_dt_ns is None else float(pose_dt_ns / 1e6),
        }
        if pose_sample is None:
            diagnostic["reason"] = "missing_pose_sync"
            ground_diagnostics.append(diagnostic)
            continue

        if gravity_source == "imu":
            gravity_imu = _windowed_imu_gravity(
                imu_samples, imu_timestamps, lidar_meta.timestamp_ns, imu_window_ns
            )
            if gravity_imu is None:
                diagnostic["reason"] = "missing_imu_window"
                ground_diagnostics.append(diagnostic)
                continue
        else:
            gravity_imu = pose_sample.gravity_imu

        cloud = load_pointcloud_from_meta(lidar_meta)
        expected_up_imu = -normalize_vector(gravity_imu)
        expected_up_lidar = normalize_vector(
            extraction_transform[:3, :3].T @ expected_up_imu
        )
        plane_result, plane_diag = _extract_ground_plane(
            cloud,
            expected_up_lidar=expected_up_lidar,
            plane_dist_thresh=plane_dist_thresh,
            normal_angle_thresh_deg=plane_normal_thresh_deg,
        )
        diagnostic.update(plane_diag)
        if plane_result is None:
            ground_diagnostics.append(diagnostic)
            continue

        lidar_plane_normal = np.asarray(plane_result["lidar_plane_normal"], dtype=float)
        lidar_plane_offset = float(plane_result["lidar_plane_offset"])
        imu_plane_normal = extraction_transform[:3, :3] @ lidar_plane_normal
        imu_ground_height = float(
            lidar_plane_offset - imu_plane_normal @ extraction_transform[:3, 3]
        )
        weight = max(float(plane_result["inlier_ratio"]), 1e-3)

        ground_samples.append(
            {
                "timestamp_ns": int(lidar_meta.timestamp_ns),
                "lidar_plane_normal": plane_result["lidar_plane_normal"],
                "lidar_plane_offset": lidar_plane_offset,
                "imu_gravity": [
                    float(value)
                    for value in np.asarray(gravity_imu, dtype=float).tolist()
                ],
                "imu_ground_height": imu_ground_height,
                "weight": weight,
                "sync_dt_ms": float(pose_dt_ns / 1e6),
                "metadata": {
                    "record_path": lidar_meta.record_path,
                    "gravity_source": gravity_source,
                    "lidar_topic": lidar_topic,
                    "pose_topic": pose_topic,
                    "plane_inlier_ratio": float(plane_result["inlier_ratio"]),
                    "plane_normal_angle_to_expected_up_deg": float(
                        plane_result["normal_angle_to_expected_up_deg"]
                    ),
                },
            }
        )
        diagnostic["selected"] = True
        ground_diagnostics.append(diagnostic)

    motion_samples = []
    motion_diagnostics = []
    motion_window_diagnostics = []
    motion_rejected_low_fitness = 0
    motion_registered_candidates: list[dict] = []
    motion_sync_threshold_ns = int(motion_pose_sync_threshold_ms * 1e6)
    submap_support_stride = int(
        motion_frame_stride if submap_support_stride is None else submap_support_stride
    )
    if motion_registration_mode not in (
        "scan_to_scan",
        "submap_to_submap",
        "submap_to_map",
    ):
        raise ValueError(
            "motion_registration_mode must be 'scan_to_scan', "
            "'submap_to_submap', or 'submap_to_map'."
        )
    if (
        motion_registration_mode in ("submap_to_submap", "submap_to_map")
        and submap_half_window < 1
    ):
        raise ValueError(
            "submap_half_window must be >= 1 for submap_to_submap/submap_to_map."
        )
    if submap_support_stride < 1:
        raise ValueError("submap_support_stride must be >= 1.")
    if submap_min_support_frames < 1:
        raise ValueError("submap_min_support_frames must be >= 1.")
    submap_voxel_size = float(
        registration_voxel_size if submap_voxel_size is None else submap_voxel_size
    )
    map_half_window = int(
        max(submap_half_window + 1, submap_half_window * 2)
        if map_half_window is None
        else map_half_window
    )
    map_support_stride = int(
        submap_support_stride if map_support_stride is None else map_support_stride
    )
    map_min_support_frames = int(
        max(submap_min_support_frames + 2, submap_min_support_frames)
        if map_min_support_frames is None
        else map_min_support_frames
    )
    map_voxel_size = float(
        submap_voxel_size if map_voxel_size is None else map_voxel_size
    )
    if motion_registration_mode == "submap_to_map" and map_half_window < 1:
        raise ValueError("map_half_window must be >= 1 for submap_to_map.")
    if map_support_stride < 1:
        raise ValueError("map_support_stride must be >= 1.")
    if map_min_support_frames < 1:
        raise ValueError("map_min_support_frames must be >= 1.")
    motion_candidates = _build_motion_candidates(
        lidar_metas=lidar_metas,
        pose_samples=pose_samples,
        pose_timestamps=pose_timestamps,
        sync_threshold_ns=motion_sync_threshold_ns,
        base_stride=motion_frame_stride,
    )
    motion_windows = _build_motion_windows(
        motion_candidates=motion_candidates,
        lidar_meta_count=len(lidar_metas),
        max_motion_samples=max_motion_samples,
        base_stride=motion_frame_stride,
        min_window_rotation_deg=float(calibration_motion_rotation_deg),
        min_window_translation_m=0.5,
    )
    preprocessing_params = {
        "voxel_size": registration_voxel_size,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": max(plane_dist_thresh, 0.1),
        "height_range": None,
        "remove_ground": False,
        "remove_walls": False,
    }
    cloud_cache = {}
    alignment_cache: dict[int, dict] = {}
    submap_cache = {}

    for window in motion_windows:
        window_diagnostic = {
            "window_id": int(window["window_id"]),
            "window_start_index": int(window["window_start_index"]),
            "window_end_index": int(window["window_end_index"]),
            "candidate_count": int(window["candidate_count"]),
            "attempt_candidate_count": int(len(window["candidates"])),
            "best_pose_rotation_deg": float(window["best_pose_rotation_deg"]),
            "best_pose_translation_m": float(window["best_pose_translation_m"]),
            "stride_values": list(window.get("stride_values", [])),
            "valid": bool(window["valid"]),
            "gate_reasons": list(window["gate_reasons"]),
        }
        if not window["valid"]:
            motion_window_diagnostics.append(window_diagnostic)
            continue

        attempt_count = 0
        rejection_reasons = []
        registered_candidate_summaries = []
        for candidate in window["candidates"]:
            start_meta = candidate["start_meta"]
            end_meta = candidate["end_meta"]
            start_pose_dt_ns = candidate["start_pose_dt_ns"]
            end_pose_dt_ns = candidate["end_pose_dt_ns"]
            diagnostic = {
                "window_id": int(window["window_id"]),
                "start_timestamp_ns": int(start_meta.timestamp_ns),
                "end_timestamp_ns": int(end_meta.timestamp_ns),
                "start_pose_sync_dt_ms": (
                    None if start_pose_dt_ns is None else float(start_pose_dt_ns / 1e6)
                ),
                "end_pose_sync_dt_ms": (
                    None if end_pose_dt_ns is None else float(end_pose_dt_ns / 1e6)
                ),
                "frame_stride": int(candidate["stride"]),
                "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
                "pose_translation_m": float(candidate["pose_translation_m"]),
                "window_score": float(candidate["score"]),
                "information_score": float(candidate["information_score"]),
                "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
                "imu_translation_heading_deg": (
                    None
                    if candidate.get("imu_translation_heading_deg") is None
                    else float(candidate["imu_translation_heading_deg"])
                ),
                "motion_registration_mode": motion_registration_mode,
                "registered": False,
                "passed_registration_gate": False,
                "selected": False,
            }
            attempt_count += 1
            imu_delta = candidate["imu_delta"]
            lidar_initial_guess = (
                np.linalg.inv(extraction_transform) @ imu_delta @ extraction_transform
            )
            if motion_registration_mode in ("submap_to_submap", "submap_to_map"):
                source_cloud, source_submap_info = _build_local_lidar_submap(
                    int(candidate["start_index"]),
                    lidar_metas=lidar_metas,
                    pose_samples=pose_samples,
                    pose_timestamps=pose_timestamps,
                    sync_threshold_ns=motion_sync_threshold_ns,
                    extraction_transform=extraction_transform,
                    submap_half_window=int(submap_half_window),
                    submap_support_stride=int(submap_support_stride),
                    submap_min_support_frames=int(submap_min_support_frames),
                    submap_voxel_size=float(submap_voxel_size),
                    alignment_cache=alignment_cache,
                    cloud_cache=cloud_cache,
                    submap_cache=submap_cache,
                )
                if motion_registration_mode == "submap_to_map":
                    target_cloud, target_submap_info = _build_local_lidar_submap(
                        int(candidate["end_index"]),
                        lidar_metas=lidar_metas,
                        pose_samples=pose_samples,
                        pose_timestamps=pose_timestamps,
                        sync_threshold_ns=motion_sync_threshold_ns,
                        extraction_transform=extraction_transform,
                        submap_half_window=int(map_half_window),
                        submap_support_stride=int(map_support_stride),
                        submap_min_support_frames=int(map_min_support_frames),
                        submap_voxel_size=float(map_voxel_size),
                        alignment_cache=alignment_cache,
                        cloud_cache=cloud_cache,
                        submap_cache=submap_cache,
                    )
                else:
                    target_cloud, target_submap_info = _build_local_lidar_submap(
                        int(candidate["end_index"]),
                        lidar_metas=lidar_metas,
                        pose_samples=pose_samples,
                        pose_timestamps=pose_timestamps,
                        sync_threshold_ns=motion_sync_threshold_ns,
                        extraction_transform=extraction_transform,
                        submap_half_window=int(submap_half_window),
                        submap_support_stride=int(submap_support_stride),
                        submap_min_support_frames=int(submap_min_support_frames),
                        submap_voxel_size=float(submap_voxel_size),
                        alignment_cache=alignment_cache,
                        cloud_cache=cloud_cache,
                        submap_cache=submap_cache,
                    )
                diagnostic["source_submap"] = source_submap_info
                if motion_registration_mode == "submap_to_map":
                    diagnostic["target_map"] = target_submap_info
                else:
                    diagnostic["target_submap"] = target_submap_info
                if source_cloud is None or target_cloud is None:
                    diagnostic["reason"] = (
                        "local_map_build_failed"
                        if motion_registration_mode == "submap_to_map"
                        else "submap_build_failed"
                    )
                    rejection_reasons.append(diagnostic["reason"])
                    motion_diagnostics.append(diagnostic)
                    continue
            else:
                source_cloud = load_pointcloud_from_meta(start_meta)
                target_cloud = load_pointcloud_from_meta(end_meta)
            lidar_delta, _, registration_result = calibrate_lidar_extrinsic(
                source_cloud,
                target_cloud,
                is_draw_registration=False,
                preprocessing_params=preprocessing_params,
                method=2,
                initial_transform=lidar_initial_guess,
            )
            if lidar_delta is None or registration_result is None:
                diagnostic["reason"] = "registration_failed"
                rejection_reasons.append("registration_failed")
                motion_diagnostics.append(diagnostic)
                continue

            registration_fitness = float(registration_result.fitness)
            registration_inlier_rmse = float(registration_result.inlier_rmse)
            diagnostic.update(
                {
                    "registered": True,
                    "registration_fitness": registration_fitness,
                    "registration_inlier_rmse": registration_inlier_rmse,
                }
            )
            if registration_fitness < min_registration_fitness:
                diagnostic["reason"] = "low_registration_fitness"
                motion_rejected_low_fitness += 1
                rejection_reasons.append("low_registration_fitness")
                motion_diagnostics.append(diagnostic)
                continue

            diagnostic["passed_registration_gate"] = True
            weight = max(registration_fitness, 1e-3)
            imu_quat = _rotation_from_matrix(imu_delta[:3, :3]).as_quat()
            lidar_quat = _rotation_from_matrix(lidar_delta[:3, :3]).as_quat()
            registered_candidate = {
                **candidate,
                "lidar_delta": lidar_delta,
                "registration_fitness": registration_fitness,
                "registration_inlier_rmse": registration_inlier_rmse,
                "weight": weight,
                "sync_dt_ms": float(max(start_pose_dt_ns, end_pose_dt_ns) / 1e6),
                "imu_quat": imu_quat,
                "lidar_quat": lidar_quat,
            }
            motion_registered_candidates.append(registered_candidate)
            registered_candidate_summaries.append(
                {
                    **_summarize_motion_candidate(candidate),
                    "registration_fitness": registration_fitness,
                    "registration_inlier_rmse": registration_inlier_rmse,
                }
            )
            motion_diagnostics.append(diagnostic)

        window_diagnostic["attempt_count"] = int(attempt_count)
        window_diagnostic["registered_candidate_count"] = int(
            len(registered_candidate_summaries)
        )
        if registered_candidate_summaries:
            window_diagnostic["registered_candidates"] = registered_candidate_summaries
            window_diagnostic["selected"] = False
        else:
            window_diagnostic["selected"] = False
            if rejection_reasons:
                window_diagnostic["reason"] = "window_no_candidate_passing_gate"
                window_diagnostic["candidate_rejection_reasons"] = rejection_reasons
            else:
                window_diagnostic["reason"] = "window_not_selected"
        motion_window_diagnostics.append(window_diagnostic)

    motion_window_diagnostics_by_id = {
        int(item["window_id"]): item for item in motion_window_diagnostics
    }
    selected_registered_candidates = []
    used_motion_ranges: list[tuple[int, int]] = []
    for window in motion_windows:
        if len(selected_registered_candidates) >= max_motion_samples:
            break
        window_candidates = [
            candidate
            for candidate in motion_registered_candidates
            if int(candidate["window_id"]) == int(window["window_id"])
        ]
        if not window_candidates:
            continue

        best_candidate = None
        best_selection_score = None
        overlap_rejected = 0
        for candidate in window_candidates:
            if _candidate_overlaps_used_ranges(candidate, used_motion_ranges):
                overlap_rejected += 1
                continue
            selection_score = _global_motion_selection_score(
                candidate, selected_registered_candidates, motion_frame_stride
            )
            if best_candidate is None or selection_score > best_selection_score:
                best_candidate = candidate
                best_selection_score = selection_score

        window_diagnostic = motion_window_diagnostics_by_id[int(window["window_id"])]
        window_diagnostic["global_overlap_rejected"] = int(overlap_rejected)
        if best_candidate is None:
            window_diagnostic["reason"] = "window_candidates_overlap_selected_ranges"
            continue

        best_candidate["global_selection_score"] = float(best_selection_score)
        selected_registered_candidates.append(best_candidate)
        used_motion_ranges.append(
            (int(best_candidate["start_index"]), int(best_candidate["end_index"]))
        )
        window_diagnostic["selected"] = True
        window_diagnostic["selected_candidate"] = {
            **_summarize_motion_candidate(best_candidate),
            "registration_fitness": float(best_candidate["registration_fitness"]),
            "registration_inlier_rmse": float(
                best_candidate["registration_inlier_rmse"]
            ),
            "global_selection_score": float(best_selection_score),
        }

    selected_candidate_keys = {
        (
            int(candidate["start_meta"].timestamp_ns),
            int(candidate["end_meta"].timestamp_ns),
            int(candidate["stride"]),
        )
        for candidate in selected_registered_candidates
    }
    for diagnostic in motion_diagnostics:
        candidate_key = (
            int(diagnostic["start_timestamp_ns"]),
            int(diagnostic["end_timestamp_ns"]),
            int(diagnostic["frame_stride"]),
        )
        if candidate_key in selected_candidate_keys:
            diagnostic["selected"] = True

    for candidate in sorted(
        selected_registered_candidates,
        key=lambda item: int(item["start_meta"].timestamp_ns),
    ):
        imu_delta = candidate["imu_delta"]
        lidar_delta = candidate["lidar_delta"]
        motion_samples.append(
            {
                "start_timestamp_ns": int(candidate["start_meta"].timestamp_ns),
                "end_timestamp_ns": int(candidate["end_meta"].timestamp_ns),
                "imu_delta": {
                    "translation": {
                        "x": float(imu_delta[0, 3]),
                        "y": float(imu_delta[1, 3]),
                        "z": float(imu_delta[2, 3]),
                    },
                    "rotation": {
                        "x": float(candidate["imu_quat"][0]),
                        "y": float(candidate["imu_quat"][1]),
                        "z": float(candidate["imu_quat"][2]),
                        "w": float(candidate["imu_quat"][3]),
                    },
                },
                "lidar_delta": {
                    "translation": {
                        "x": float(lidar_delta[0, 3]),
                        "y": float(lidar_delta[1, 3]),
                        "z": float(lidar_delta[2, 3]),
                    },
                    "rotation": {
                        "x": float(candidate["lidar_quat"][0]),
                        "y": float(candidate["lidar_quat"][1]),
                        "z": float(candidate["lidar_quat"][2]),
                        "w": float(candidate["lidar_quat"][3]),
                    },
                },
                "weight": float(candidate["weight"]),
                "sync_dt_ms": float(candidate["sync_dt_ms"]),
                "metadata": {
                    "record_path_start": candidate["start_meta"].record_path,
                    "record_path_end": candidate["end_meta"].record_path,
                    "lidar_topic": lidar_topic,
                    "pose_topic": pose_topic,
                    "window_id": int(candidate["window_id"]),
                    "motion_registration_mode": motion_registration_mode,
                    "frame_stride": int(candidate["stride"]),
                    "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
                    "pose_translation_m": float(candidate["pose_translation_m"]),
                    "information_score": float(candidate["information_score"]),
                    "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
                    "imu_translation_heading_deg": (
                        None
                        if candidate.get("imu_translation_heading_deg") is None
                        else float(candidate["imu_translation_heading_deg"])
                    ),
                    "registration_fitness": float(candidate["registration_fitness"]),
                    "registration_inlier_rmse": float(
                        candidate["registration_inlier_rmse"]
                    ),
                    "global_selection_score": float(
                        candidate["global_selection_score"]
                    ),
                },
            }
        )

    selected_motion_headings = [
        float(sample["metadata"]["imu_translation_heading_deg"])
        for sample in motion_samples
        if sample["metadata"].get("imu_translation_heading_deg") is not None
    ]
    selected_turn_counts = {"left": 0, "right": 0}
    selected_heading_bins = set()
    selected_axis_vectors = []
    for candidate in selected_registered_candidates:
        turn_sign = _motion_turn_sign(float(candidate["imu_signed_yaw_deg"]))
        if turn_sign in selected_turn_counts:
            selected_turn_counts[turn_sign] += 1
        heading_bin = _heading_bin(candidate.get("imu_translation_heading_deg"))
        if heading_bin is not None:
            selected_heading_bins.add(int(heading_bin))
        selected_axis_vectors.append(_candidate_axis_abs_vector(candidate))
    selected_axis_mean = (
        np.mean(np.asarray(selected_axis_vectors, dtype=float), axis=0)
        if selected_axis_vectors
        else None
    )

    payload = {
        "parent_frame": parent_frame,
        "child_frame": lidar_frame,
        "initial_transform": {
            "header": {
                "stamp": {"secs": 0, "nsecs": 0},
                "seq": 0,
                "frame_id": parent_frame,
            },
            "transform": {
                "translation": {
                    "x": float(initial_transform[0, 3]),
                    "y": float(initial_transform[1, 3]),
                    "z": float(initial_transform[2, 3]),
                },
                "rotation": {
                    **{
                        key: float(value)
                        for key, value in zip(
                            ("x", "y", "z", "w"),
                            _rotation_from_matrix(initial_transform[:3, :3]).as_quat(),
                        )
                    }
                },
            },
            "child_frame_id": lidar_frame,
        },
        "config": {
            "loss": calibration_loss,
            "min_motion_rotation_deg": float(calibration_motion_rotation_deg),
            "planar_motion_policy": calibration_planar_motion_policy,
        },
        "metadata": {
            "run_profile": run_profile or "custom",
            "record_path": record_path,
            "record_files": record_files,
            "lidar_topic": lidar_topic,
            "pose_topic": pose_topic,
            "imu_topic": imu_topic,
            "gravity_source": gravity_source,
            "ground_pose_sync_threshold_ms": float(ground_pose_sync_threshold_ms),
            "motion_pose_sync_threshold_ms": float(motion_pose_sync_threshold_ms),
            "motion_frame_stride": int(motion_frame_stride),
            "motion_selection_strategy": "global_diversity",
            "motion_registration_mode": motion_registration_mode,
            "submap_half_window": int(submap_half_window),
            "submap_support_stride": int(submap_support_stride),
            "submap_min_support_frames": int(submap_min_support_frames),
            "submap_voxel_size": float(submap_voxel_size),
            "map_half_window": int(map_half_window),
            "map_support_stride": int(map_support_stride),
            "map_min_support_frames": int(map_min_support_frames),
            "map_voxel_size": float(map_voxel_size),
            "min_registration_fitness": float(min_registration_fitness),
            "initial_transform_source": initial_transform_source,
            "extraction_transform_source": extraction_transform_source,
            "reference_transform_source": reference_transform_source,
            "ground_selected": len(ground_samples),
            "motion_selected": len(motion_samples),
            "motion_candidate_count": len(motion_candidates),
            "motion_window_count": len(motion_windows),
            "motion_valid_window_count": int(
                sum(1 for window in motion_windows if window["valid"])
            ),
            "motion_registered_candidate_count": len(motion_registered_candidates),
        },
        "ground_samples": ground_samples,
        "motion_samples": motion_samples,
    }
    payload["extraction_transform"] = {
        "header": {
            "stamp": {"secs": 0, "nsecs": 0},
            "seq": 0,
            "frame_id": parent_frame,
        },
        "transform": {
            "translation": {
                "x": float(extraction_transform[0, 3]),
                "y": float(extraction_transform[1, 3]),
                "z": float(extraction_transform[2, 3]),
            },
            "rotation": {
                **{
                    key: float(value)
                    for key, value in zip(
                        ("x", "y", "z", "w"),
                        _rotation_from_matrix(extraction_transform[:3, :3]).as_quat(),
                    )
                }
            },
        },
        "child_frame_id": lidar_frame,
    }
    if record_reference_transform is not None:
        payload["reference_transform"] = {
            "header": {
                "stamp": {"secs": 0, "nsecs": 0},
                "seq": 0,
                "frame_id": parent_frame,
            },
            "transform": {
                "translation": {
                    "x": float(record_reference_transform[0, 3]),
                    "y": float(record_reference_transform[1, 3]),
                    "z": float(record_reference_transform[2, 3]),
                },
                "rotation": {
                    **{
                        key: float(value)
                        for key, value in zip(
                            ("x", "y", "z", "w"),
                            _rotation_from_matrix(
                                record_reference_transform[:3, :3]
                            ).as_quat(),
                        )
                    }
                },
            },
            "child_frame_id": lidar_frame,
        }
    diagnostics = {
        "summary": {
            "run_profile": run_profile or "custom",
            "record_files": record_files,
            "lidar_topic": lidar_topic,
            "lidar_frame": lidar_frame,
            "parent_frame": parent_frame,
            "ground_selected": len(ground_samples),
            "motion_selected": len(motion_samples),
            "ground_attempted": len(ground_indices),
            "motion_attempted": len(motion_candidates),
            "motion_rejected_low_fitness": motion_rejected_low_fitness,
            "motion_registered_candidate_count": len(motion_registered_candidates),
            "motion_window_count": len(motion_windows),
            "motion_valid_window_count": int(
                sum(1 for window in motion_windows if window["valid"])
            ),
            "min_registration_fitness": float(min_registration_fitness),
            "motion_selection_strategy": "global_diversity",
            "motion_registration_mode": motion_registration_mode,
            "map_half_window": int(map_half_window),
            "map_support_stride": int(map_support_stride),
            "map_min_support_frames": int(map_min_support_frames),
            "map_voxel_size": float(map_voxel_size),
        },
        "ground": ground_diagnostics,
        "motion_windows": motion_window_diagnostics,
        "motion": motion_diagnostics,
        "motion_selection": {
            "strategy": "global_diversity",
            "registration_mode": motion_registration_mode,
            "selected_frame_strides": sorted(
                {int(sample["metadata"]["frame_stride"]) for sample in motion_samples}
            ),
            "selected_turn_counts": {
                "left": int(selected_turn_counts["left"]),
                "right": int(selected_turn_counts["right"]),
            },
            "selected_heading_bin_count": int(len(selected_heading_bins)),
            "selected_translation_heading_span_deg": circular_span_deg(
                selected_motion_headings
            ),
            "selected_rotation_axis_abs_mean_xyz": (
                {
                    "x": float(selected_axis_mean[0]),
                    "y": float(selected_axis_mean[1]),
                    "z": float(selected_axis_mean[2]),
                }
                if selected_axis_mean is not None
                else None
            ),
            "registered_candidate_count": len(motion_registered_candidates),
        },
    }
    with open(sample_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    with open(diagnostics_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(diagnostics, file, sort_keys=False)
    return sample_path, diagnostics


def _recommendation_rank(recommendation: str | None) -> int:
    ranking = {
        "full_6dof_candidate": 5,
        "reference_conflict_review": 4,
        "holdout_review": 3,
        "z_roll_pitch_priority": 2,
        "basin_sensitivity_review": 1,
        "reextract_review": 0,
        "insufficient_data": 0,
    }
    return ranking.get(recommendation or "", -1)


def _status_rank(status: str | None) -> int:
    ranking = {"pass": 2, "warning": 1}
    return ranking.get(status or "", 0)


def _delta_rotation_deg(delta: dict | None) -> float:
    if not isinstance(delta, dict):
        return float("inf")
    return float(delta.get("rotation_deg", float("inf")))


def _delta_translation_m(delta: dict | None) -> float:
    if not isinstance(delta, dict):
        return float("inf")
    return float(delta.get("translation_norm_m", float("inf")))


def _build_reextract_pass_summary(
    pass_name: str,
    pass_dir: Path,
    sample_path: Path,
    diagnostics: dict,
    result: dict,
    manifest: dict,
) -> dict:
    metrics = result["metrics"]
    coarse_metrics = metrics.get("coarse_metrics", {})
    assessment = metrics.get("vehicle_motion_assessment", {})
    summary = metrics.get("summary", {})
    return {
        "pass_name": pass_name,
        "pass_dir": str(pass_dir),
        "sample_path": str(sample_path),
        "calibration_dir": str(pass_dir / "calibration"),
        "calibrated_extrinsics": manifest["artifacts"]["calibrated_extrinsics"],
        "recommendation": assessment.get("recommendation"),
        "solver_policy": summary.get("solver_policy"),
        "trusted_reference_consistency": assessment.get(
            "trusted_reference_consistency"
        ),
        "extraction_consistency": assessment.get("extraction_consistency"),
        "planar_basin_stability": assessment.get("planar_basin_stability"),
        "full_prior_robustness": assessment.get("full_prior_robustness"),
        "delta_to_reference": summary.get("delta_to_reference"),
        "delta_to_extraction": summary.get("delta_to_extraction"),
        "motion_rotation_residual_p95_deg": coarse_metrics.get(
            "motion_rotation_residual_p95_deg"
        ),
        "motion_translation_residual_p95_m": coarse_metrics.get(
            "motion_translation_residual_p95_m"
        ),
        "motion_registration_fitness_p05": coarse_metrics.get(
            "motion_registration_fitness_p05"
        ),
        "ground_selected": diagnostics["summary"]["ground_selected"],
        "motion_selected": diagnostics["summary"]["motion_selected"],
    }


def _rewrite_pass_summary_paths(pass_summary: dict, pass_dir: Path) -> None:
    pass_summary["pass_dir"] = str(pass_dir)
    pass_summary["sample_path"] = str(pass_dir / "standardized_samples.yaml")
    pass_summary["calibration_dir"] = str(pass_dir / "calibration")
    calibrated_extrinsics = Path(pass_summary["calibrated_extrinsics"])
    pass_summary["calibrated_extrinsics"] = str(
        pass_dir / "calibration" / "calibrated" / calibrated_extrinsics.name
    )


def _reextract_trigger_reasons(pass_summary: dict) -> list[str]:
    reasons: list[str] = []
    if pass_summary.get("recommendation") == "reextract_review":
        reasons.append("recommendation=reextract_review")
    if pass_summary.get("extraction_consistency") == "warning":
        reasons.append("extraction_consistency=warning")
    return reasons


def _pass_selection_score(pass_summary: dict) -> tuple[float, ...]:
    return (
        float(_recommendation_rank(pass_summary.get("recommendation"))),
        float(_status_rank(pass_summary.get("trusted_reference_consistency"))),
        float(_status_rank(pass_summary.get("extraction_consistency"))),
        float(_status_rank(pass_summary.get("planar_basin_stability"))),
        float(_status_rank(pass_summary.get("full_prior_robustness"))),
        -_delta_rotation_deg(pass_summary.get("delta_to_reference")),
        -_delta_translation_m(pass_summary.get("delta_to_reference")),
        -_delta_rotation_deg(pass_summary.get("delta_to_extraction")),
        -_delta_translation_m(pass_summary.get("delta_to_extraction")),
        -float(pass_summary.get("motion_rotation_residual_p95_deg", float("inf"))),
        -float(pass_summary.get("motion_translation_residual_p95_m", float("inf"))),
        float(pass_summary.get("motion_registration_fitness_p05", 0.0)),
    )


def _select_reextract_pass(pass_summaries: list[dict]) -> dict:
    return max(pass_summaries, key=_pass_selection_score)


def _copy_output_artifacts(source_dir: Path, target_dir: Path) -> None:
    for name in (
        "standardized_samples.yaml",
        "conversion_diagnostics.yaml",
        "calibration",
    ):
        source_path = source_dir / name
        target_path = target_dir / name
        if not source_path.exists():
            continue
        if target_path.is_dir():
            shutil.rmtree(target_path)
        elif target_path.exists():
            target_path.unlink()
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _refresh_calibration_manifest_paths(calibration_dir: Path) -> None:
    manifest_path = calibration_dir / "diagnostics" / "manifest.yaml"
    if not manifest_path.exists():
        return
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file) or {}
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["calibrated_tf"] = str(calibration_dir / "calibrated_tf.yaml")
    artifacts["metrics"] = str(calibration_dir / "metrics.yaml")
    parent_frame = manifest.get("parent_frame")
    child_frame = manifest.get("child_frame")
    if parent_frame and child_frame:
        filename = f"{parent_frame}_{child_frame}_extrinsics.yaml"
        artifacts["initial_guess"] = str(calibration_dir / "initial_guess" / filename)
        artifacts["calibrated_extrinsics"] = str(
            calibration_dir / "calibrated" / filename
        )
    diagnostics = artifacts.setdefault("diagnostics", {})
    diagnostics["algorithm"] = str(calibration_dir / "diagnostics" / "algorithm.yaml")
    diagnostics["evaluation"] = str(calibration_dir / "diagnostics" / "evaluation.yaml")
    diagnostics["observability"] = str(
        calibration_dir / "diagnostics" / "observability.yaml"
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)


def _run_conversion_and_calibration(
    args: argparse.Namespace,
    output_dir: Path,
    initial_transform_path: str | None,
    extraction_transform_path: str | None,
    identity_initial_transform: bool,
) -> dict:
    sample_path, diagnostics = convert_record_to_standardized_samples(
        record_path=args.record_path,
        output_dir=str(output_dir),
        lidar_topic=args.lidar_topic,
        pose_topic=args.pose_topic,
        imu_topic=args.imu_topic,
        parent_frame=args.parent_frame,
        child_frame=args.child_frame,
        initial_transform_path=initial_transform_path,
        extraction_transform_path=extraction_transform_path,
        identity_initial_transform=identity_initial_transform,
        gravity_source=args.gravity_source,
        ground_pose_sync_threshold_ms=args.ground_pose_sync_threshold_ms,
        motion_pose_sync_threshold_ms=args.motion_pose_sync_threshold_ms,
        imu_gravity_window_ms=args.imu_gravity_window_ms,
        max_ground_samples=args.max_ground_samples,
        max_motion_samples=args.max_motion_samples,
        motion_frame_stride=args.motion_frame_stride,
        plane_dist_thresh=args.plane_dist_thresh,
        plane_normal_thresh_deg=args.plane_normal_thresh_deg,
        registration_voxel_size=args.registration_voxel_size,
        min_registration_fitness=args.min_registration_fitness,
        calibration_loss=args.loss,
        calibration_motion_rotation_deg=args.min_motion_rotation_deg,
        calibration_planar_motion_policy=args.planar_motion_policy,
        motion_registration_mode=args.motion_registration_mode,
        submap_half_window=args.submap_half_window,
        submap_support_stride=args.submap_support_stride,
        submap_min_support_frames=args.submap_min_support_frames,
        submap_voxel_size=args.submap_voxel_size,
        map_half_window=args.map_half_window,
        map_support_stride=args.map_support_stride,
        map_min_support_frames=args.map_min_support_frames,
        map_voxel_size=args.map_voxel_size,
        run_profile=args.profile,
        prepared_dataset_yaml=args.prepared_dataset_yaml,
    )
    logging.info("Wrote standardized samples to %s", sample_path)
    logging.info(
        "Converted %d ground samples and %d motion samples.",
        diagnostics["summary"]["ground_selected"],
        diagnostics["summary"]["motion_selected"],
    )
    calibration_output_dir = output_dir / "calibration"
    dataset, config, raw_payload = load_dataset(str(sample_path))
    config_updates = {
        "loss": args.loss,
        "min_motion_rotation_deg": args.min_motion_rotation_deg,
        "planar_motion_policy": args.planar_motion_policy,
    }
    config = CalibrationConfig(**{**config.__dict__, **config_updates})
    result = run_calibration(
        dataset, config=config, output_dir=str(calibration_output_dir)
    )
    manifest = write_outputs(
        output_dir=calibration_output_dir,
        dataset=dataset,
        initial_transform=result["initial_transform"],
        final_transform=result["final_transform"],
        metrics_output=result["metrics"],
        algorithm_report={
            "input_file": str(sample_path.resolve()),
            "config": config.__dict__,
            "dataset_metadata": dataset.metadata,
            "raw_metadata": raw_payload.get("metadata", {}),
            "conversion_summary": diagnostics["summary"],
            "stages": result["stages"],
        },
        evaluation_report=result["evaluation"],
    )
    logging.info("Saved lidar2imu calibration outputs to %s", calibration_output_dir)
    logging.info("Manifest: %s", manifest)
    return {
        "sample_path": sample_path,
        "diagnostics": diagnostics,
        "result": result,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Apollo record data into standardized lidar2imu samples and optionally run calibration."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILE_PRESETS.keys()),
        default=None,
        help="Apply a fixed lidar2imu preset. baseline=stable scan-to-scan reference; production=current submap-to-map production candidate. Explicit CLI flags still override preset values.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--record-path",
        help="Path to a record file or split-record directory.",
    )
    input_group.add_argument(
        "--prepared-dataset-yaml",
        default=None,
        help="Path to diagnostics/prepared_rig_dataset.yaml generated by lidar2lidar-rig-dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2imu/record_conversion",
        help="Output directory.",
    )
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/lslidar_main/PointCloud2",
        help="LiDAR point cloud topic.",
    )
    parser.add_argument(
        "--pose-topic",
        default="/apollo/localization/pose",
        help="Pose topic used for IMU-side motion.",
    )
    parser.add_argument(
        "--imu-topic",
        default="/apollo/sensor/gnss/imu",
        help="IMU topic used when gravity-source=imu.",
    )
    parser.add_argument(
        "--parent-frame", default="imu", help="Parent frame for the output extrinsics."
    )
    parser.add_argument(
        "--child-frame",
        default=None,
        help="Optional explicit child frame. Defaults to the LiDAR topic frame_id.",
    )
    parser.add_argument(
        "--initial-transform",
        default=None,
        help="Optional extrinsics YAML/JSON used when the bag does not contain lidar->parent TF.",
    )
    parser.add_argument(
        "--extraction-transform",
        default=None,
        help=(
            "Optional extrinsics YAML/JSON used for record-side geometry "
            "construction. Use this for intentional re-extraction while keeping "
            "the in-record TF as the trusted reference."
        ),
    )
    parser.add_argument(
        "--identity-initial-transform",
        action="store_true",
        help="Use identity as the initial lidar->parent transform when TF is missing. Exploratory only.",
    )
    parser.add_argument(
        "--gravity-source",
        choices=["pose", "imu"],
        default="pose",
        help="Source for imu_gravity in ground samples.",
    )
    parser.add_argument(
        "--ground-pose-sync-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp gap for ground sample pose sync.",
    )
    parser.add_argument(
        "--motion-pose-sync-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp gap for motion sample pose sync.",
    )
    parser.add_argument(
        "--imu-gravity-window-ms",
        type=float,
        default=100.0,
        help="Window size when averaging IMU gravity samples.",
    )
    parser.add_argument(
        "--max-ground-samples",
        type=int,
        default=16,
        help="Maximum ground samples to export.",
    )
    parser.add_argument(
        "--max-motion-samples",
        type=int,
        default=12,
        help="Maximum motion samples to export.",
    )
    parser.add_argument(
        "--motion-frame-stride",
        type=int,
        default=5,
        help="LiDAR frame stride between motion sample endpoints.",
    )
    parser.add_argument(
        "--plane-dist-thresh",
        type=float,
        default=0.15,
        help="RANSAC plane fitting threshold in meters.",
    )
    parser.add_argument(
        "--plane-normal-thresh-deg",
        type=float,
        default=20.0,
        help="Max angle between extracted plane normal and expected up.",
    )
    parser.add_argument(
        "--registration-voxel-size",
        type=float,
        default=0.3,
        help="Voxel size used for LiDAR-to-LiDAR registration.",
    )
    parser.add_argument(
        "--motion-registration-mode",
        default="scan_to_scan",
        choices=["scan_to_scan", "submap_to_submap", "submap_to_map"],
        help="Use direct pair registration, symmetric local submaps, or local-submap to larger local-map registration for motion factors.",
    )
    parser.add_argument(
        "--submap-half-window",
        type=int,
        default=2,
        help="Number of support frames on each side of the anchor when motion-registration-mode=submap_to_submap.",
    )
    parser.add_argument(
        "--submap-support-stride",
        type=int,
        default=None,
        help="Stride between support frames inside each local submap. Defaults to motion-frame-stride.",
    )
    parser.add_argument(
        "--submap-min-support-frames",
        type=int,
        default=3,
        help="Minimum aligned LiDAR frames required to keep a local submap candidate.",
    )
    parser.add_argument(
        "--submap-voxel-size",
        type=float,
        default=None,
        help="Optional voxel size for the merged local submap. Defaults to registration-voxel-size.",
    )
    parser.add_argument(
        "--map-half-window",
        type=int,
        default=None,
        help="Number of support frames on each side of the target anchor when motion-registration-mode=submap_to_map. Defaults to a larger window than submap-half-window.",
    )
    parser.add_argument(
        "--map-support-stride",
        type=int,
        default=None,
        help="Stride between support frames inside the target local map when motion-registration-mode=submap_to_map. Defaults to submap-support-stride.",
    )
    parser.add_argument(
        "--map-min-support-frames",
        type=int,
        default=None,
        help="Minimum aligned LiDAR frames required to keep a target local map candidate. Defaults above submap-min-support-frames.",
    )
    parser.add_argument(
        "--map-voxel-size",
        type=float,
        default=None,
        help="Optional voxel size for the target local map in submap_to_map mode. Defaults to submap-voxel-size.",
    )
    parser.add_argument(
        "--min-registration-fitness",
        type=float,
        default=0.55,
        help="Reject motion pairs whose LiDAR-LiDAR registration fitness is below this threshold.",
    )
    parser.add_argument(
        "--loss",
        default="huber",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Calibration robust loss.",
    )
    parser.add_argument(
        "--min-motion-rotation-deg",
        type=float,
        default=1.0,
        help="Minimum motion rotation threshold for calibration.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run lidar2imu calibration immediately after conversion.",
    )
    parser.add_argument(
        "--auto-reextract-if-needed",
        action="store_true",
        help="When calibration reports stale extraction geometry, rerun one second extraction/calibration pass using the pass-1 calibrated transform as the extraction seed.",
    )
    parser.add_argument(
        "--planar-motion-policy",
        default="auto",
        choices=["auto", "free", "freeze_xyyaw"],
        help="How to handle weak planar observability during calibration.",
    )
    args = parser.parse_args()
    _apply_run_profile(args, _explicit_cli_options(sys.argv[1:]))
    if args.profile is not None:
        logging.info("Using lidar2imu %s profile.", args.profile)

    if args.auto_reextract_if_needed and not args.calibrate:
        parser.error("--auto-reextract-if-needed requires --calibrate")

    output_dir = Path(args.output_dir)
    if not args.calibrate:
        sample_path, diagnostics = convert_record_to_standardized_samples(
            record_path=args.record_path,
            output_dir=str(output_dir),
            lidar_topic=args.lidar_topic,
            pose_topic=args.pose_topic,
            imu_topic=args.imu_topic,
            parent_frame=args.parent_frame,
            child_frame=args.child_frame,
            initial_transform_path=args.initial_transform,
            extraction_transform_path=args.extraction_transform,
            identity_initial_transform=args.identity_initial_transform,
            gravity_source=args.gravity_source,
            ground_pose_sync_threshold_ms=args.ground_pose_sync_threshold_ms,
            motion_pose_sync_threshold_ms=args.motion_pose_sync_threshold_ms,
            imu_gravity_window_ms=args.imu_gravity_window_ms,
            max_ground_samples=args.max_ground_samples,
            max_motion_samples=args.max_motion_samples,
            motion_frame_stride=args.motion_frame_stride,
            plane_dist_thresh=args.plane_dist_thresh,
            plane_normal_thresh_deg=args.plane_normal_thresh_deg,
            registration_voxel_size=args.registration_voxel_size,
            min_registration_fitness=args.min_registration_fitness,
            calibration_loss=args.loss,
            calibration_motion_rotation_deg=args.min_motion_rotation_deg,
            calibration_planar_motion_policy=args.planar_motion_policy,
            motion_registration_mode=args.motion_registration_mode,
            submap_half_window=args.submap_half_window,
            submap_support_stride=args.submap_support_stride,
            submap_min_support_frames=args.submap_min_support_frames,
            submap_voxel_size=args.submap_voxel_size,
            map_half_window=args.map_half_window,
            map_support_stride=args.map_support_stride,
            map_min_support_frames=args.map_min_support_frames,
            map_voxel_size=args.map_voxel_size,
            run_profile=args.profile,
            prepared_dataset_yaml=args.prepared_dataset_yaml,
        )
        logging.info("Wrote standardized samples to %s", sample_path)
        logging.info(
            "Converted %d ground samples and %d motion samples.",
            diagnostics["summary"]["ground_selected"],
            diagnostics["summary"]["motion_selected"],
        )
        return

    first_pass = _run_conversion_and_calibration(
        args=args,
        output_dir=output_dir,
        initial_transform_path=args.initial_transform,
        extraction_transform_path=args.extraction_transform,
        identity_initial_transform=args.identity_initial_transform,
    )

    pass_summaries = [
        _build_reextract_pass_summary(
            pass_name="pass1",
            pass_dir=output_dir,
            sample_path=Path(first_pass["sample_path"]),
            diagnostics=first_pass["diagnostics"],
            result=first_pass["result"],
            manifest=first_pass["manifest"],
        )
    ]
    trigger_reasons = _reextract_trigger_reasons(pass_summaries[0])
    reextract_summary = {
        "enabled": bool(args.auto_reextract_if_needed),
        "triggered": False,
        "trigger_reasons": trigger_reasons,
        "chosen_pass": "pass1",
        "passes": pass_summaries,
    }
    if not args.auto_reextract_if_needed:
        return

    if trigger_reasons:
        reextract_summary["triggered"] = True
        pass1_dir = output_dir / "reextract_pass1"
        pass2_dir = output_dir / "reextract_pass2"
        if pass1_dir.exists():
            shutil.rmtree(pass1_dir)
        if pass2_dir.exists():
            shutil.rmtree(pass2_dir)
        pass1_dir.mkdir(parents=True, exist_ok=True)
        _copy_output_artifacts(output_dir, pass1_dir)
        _rewrite_pass_summary_paths(pass_summaries[0], pass1_dir)
        first_pass_extrinsics = first_pass["manifest"]["artifacts"][
            "calibrated_extrinsics"
        ]
        try:
            second_pass = _run_conversion_and_calibration(
                args=args,
                output_dir=pass2_dir,
                initial_transform_path=first_pass_extrinsics,
                extraction_transform_path=first_pass_extrinsics,
                identity_initial_transform=False,
            )
        except Exception as exc:
            reextract_summary["pass2_error"] = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "Auto re-extraction pass2 failed; keeping pass1 outputs. %s",
                reextract_summary["pass2_error"],
            )
        else:
            pass_summaries.append(
                _build_reextract_pass_summary(
                    pass_name="pass2",
                    pass_dir=pass2_dir,
                    sample_path=Path(second_pass["sample_path"]),
                    diagnostics=second_pass["diagnostics"],
                    result=second_pass["result"],
                    manifest=second_pass["manifest"],
                )
            )
            chosen_pass = _select_reextract_pass(pass_summaries)
            reextract_summary["chosen_pass"] = chosen_pass["pass_name"]
            if chosen_pass["pass_name"] == "pass2":
                _copy_output_artifacts(pass2_dir, output_dir)
                _refresh_calibration_manifest_paths(output_dir / "calibration")
    reextract_summary_path = output_dir / "reextract_summary.yaml"
    with open(reextract_summary_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(reextract_summary, file, sort_keys=False)
    logging.info("Wrote re-extraction summary to %s", reextract_summary_path)


if __name__ == "__main__":
    main()
