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
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import normalize_plane, normalize_vector
from lidar2imu.io import load_dataset, write_outputs
from lidar2imu.models import CalibrationConfig
from lidar2imu.pipeline import run_calibration
from lidar2lidar.extrinsic_io import load_extrinsics_file
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic
from lidar2lidar.record_adapter import Record, ensure_record_available
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


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    transform_world_localization: np.ndarray
    transform_world_imu: np.ndarray
    gravity_imu: np.ndarray


@dataclass(frozen=True)
class ImuSample:
    timestamp_ns: int
    linear_acceleration: np.ndarray
    angular_velocity: np.ndarray


def _pose_to_matrix(position, orientation) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_quat(
        [
            float(orientation.qx),
            float(orientation.qy),
            float(orientation.qz),
            float(orientation.qw),
        ]
    ).as_matrix()
    transform[:3, 3] = [
        float(position.x),
        float(position.y),
        float(position.z),
    ]
    return transform


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


def _collect_pose_samples(
    record_files: list[str], pose_topic: str, transform_localization_to_imu: np.ndarray
) -> list[PoseSample]:
    ensure_record_available()
    samples: list[PoseSample] = []
    gravity_world = np.array([0.0, 0.0, -9.81], dtype=float)
    for record_file in record_files:
        with Record(record_file) as record:
            for _, msg, timestamp_ns in record.read_messages(topics=[pose_topic]):
                pose = msg.pose
                transform_world_localization = _pose_to_matrix(
                    pose.position, pose.orientation
                )
                transform_world_imu = (
                    transform_world_localization @ transform_localization_to_imu
                )
                gravity_imu = transform_world_imu[:3, :3].T @ gravity_world
                samples.append(
                    PoseSample(
                        timestamp_ns=int(timestamp_ns),
                        transform_world_localization=transform_world_localization,
                        transform_world_imu=transform_world_imu,
                        gravity_imu=gravity_imu,
                    )
                )
    samples.sort(key=lambda item: item.timestamp_ns)
    return samples


def _collect_imu_samples(record_files: list[str], imu_topic: str) -> list[ImuSample]:
    ensure_record_available()
    samples: list[ImuSample] = []
    for record_file in record_files:
        with Record(record_file) as record:
            for _, msg, timestamp_ns in record.read_messages(topics=[imu_topic]):
                linear_acceleration = getattr(msg, "linear_acceleration", None)
                angular_velocity = getattr(msg, "angular_velocity", None)
                if linear_acceleration is None or angular_velocity is None:
                    imu_pose = getattr(msg, "imu", None)
                    linear_acceleration = getattr(imu_pose, "linear_acceleration", None)
                    angular_velocity = getattr(imu_pose, "angular_velocity", None)
                if linear_acceleration is None or angular_velocity is None:
                    raise RuntimeError(
                        f"Unsupported IMU message layout on topic {imu_topic}."
                    )
                samples.append(
                    ImuSample(
                        timestamp_ns=int(timestamp_ns),
                        linear_acceleration=np.array(
                            [
                                float(linear_acceleration.x),
                                float(linear_acceleration.y),
                                float(linear_acceleration.z),
                            ],
                            dtype=float,
                        ),
                        angular_velocity=np.array(
                            [
                                float(angular_velocity.x),
                                float(angular_velocity.y),
                                float(angular_velocity.z),
                            ],
                            dtype=float,
                        ),
                    )
                )
    samples.sort(key=lambda item: item.timestamp_ns)
    return samples


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


def _motion_excitation(delta_transform: np.ndarray) -> tuple[float, float]:
    rotation_deg = float(
        np.degrees(np.linalg.norm(R.from_matrix(delta_transform[:3, :3]).as_rotvec()))
    )
    translation_m = float(np.linalg.norm(delta_transform[:3, 3]))
    return rotation_deg, translation_m


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
                    "score": (rotation_deg * 10.0 + translation_m) / float(stride),
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
                "best_pose_rotation_deg": float(best_candidate["pose_rotation_deg"]),
                "best_pose_translation_m": float(
                    best_candidate["pose_translation_m"]
                ),
                "valid": not gate_reasons,
                "gate_reasons": gate_reasons,
                "candidates": preferred_candidates[:max_candidates_per_window],
            }
        )
    return sorted(
        windows,
        key=lambda item: (
            -int(item["valid"]),
            -int(item["has_rotation_candidate"]),
            -item["best_score"],
            item["window_start_index"],
        ),
    )


def convert_record_to_standardized_samples(
    record_path: str,
    output_dir: str,
    lidar_topic: str,
    pose_topic: str,
    imu_topic: str,
    parent_frame: str,
    child_frame: str | None,
    initial_transform_path: str | None,
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
) -> tuple[Path, dict]:
    record_files = discover_record_files(record_path)
    topic_frame_ids = get_topic_frame_ids(record_files, [lidar_topic])
    lidar_frame = child_frame or topic_frame_ids.get(lidar_topic, "")
    if not lidar_frame:
        raise RuntimeError(f"Failed to infer frame_id for lidar topic {lidar_topic}.")

    tf_edges = extract_tf_edges(record_files)
    tf_graph = build_transform_graph(tf_edges)
    initial_transform_source = "/tf_static or merged tf graph"
    if initial_transform_path is not None:
        initial_transform, file_parent_frame, file_child_frame, _, _ = (
            load_extrinsics_file(initial_transform_path)
        )
        if file_parent_frame and file_parent_frame != parent_frame:
            raise RuntimeError(
                f"Initial transform parent frame {file_parent_frame} does not match "
                f"requested parent frame {parent_frame}."
            )
        if file_child_frame and file_child_frame != lidar_frame:
            raise RuntimeError(
                f"Initial transform child frame {file_child_frame} does not match "
                f"LiDAR frame {lidar_frame}."
            )
        initial_transform_source = str(Path(initial_transform_path).resolve())
    else:
        initial_transform = lookup_transform(tf_graph, lidar_frame, parent_frame)
        if initial_transform is None:
            if not identity_initial_transform:
                raise RuntimeError(
                    f"Could not find transform from {lidar_frame} to {parent_frame}. "
                    "Provide --initial-transform or enable --identity-initial-transform "
                    "for exploratory runs."
                )
            initial_transform = np.eye(4, dtype=float)
            initial_transform_source = "identity_fallback"

    localization_to_imu = lookup_transform(tf_graph, parent_frame, "localization")
    if localization_to_imu is None:
        raise RuntimeError(
            f"Could not find transform from {parent_frame} to localization. "
            "This conversion currently expects the localization frame to be present."
        )

    metadata_by_topic = collect_pointcloud_metadata(record_files, [lidar_topic])
    lidar_metas = metadata_by_topic[lidar_topic]
    if not lidar_metas:
        raise RuntimeError(f"No point clouds found for topic {lidar_topic}.")

    pose_samples = _collect_pose_samples(record_files, pose_topic, localization_to_imu)
    if not pose_samples:
        raise RuntimeError(f"No pose messages found on {pose_topic}.")
    pose_timestamps = [sample.timestamp_ns for sample in pose_samples]

    imu_samples = (
        _collect_imu_samples(record_files, imu_topic) if gravity_source == "imu" else []
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
            initial_transform[:3, :3].T @ expected_up_imu
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
        imu_plane_normal = initial_transform[:3, :3] @ lidar_plane_normal
        imu_ground_height = float(
            lidar_plane_offset - imu_plane_normal @ initial_transform[:3, 3]
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
    motion_sync_threshold_ns = int(motion_pose_sync_threshold_ms * 1e6)
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
    used_motion_ranges: list[tuple[int, int]] = []

    for window in motion_windows:
        window_diagnostic = {
            "window_id": int(window["window_id"]),
            "window_start_index": int(window["window_start_index"]),
            "window_end_index": int(window["window_end_index"]),
            "candidate_count": int(window["candidate_count"]),
            "best_pose_rotation_deg": float(window["best_pose_rotation_deg"]),
            "best_pose_translation_m": float(window["best_pose_translation_m"]),
            "valid": bool(window["valid"]),
            "gate_reasons": list(window["gate_reasons"]),
        }
        if not window["valid"]:
            motion_window_diagnostics.append(window_diagnostic)
            continue

        selected_candidate_summary = None
        attempt_count = 0
        overlap_skipped = 0
        rejection_reasons = []
        for candidate in window["candidates"]:
            if len(motion_samples) >= max_motion_samples:
                break
            if _candidate_overlaps_used_ranges(candidate, used_motion_ranges):
                overlap_skipped += 1
                continue

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
            }
            attempt_count += 1
            imu_delta = candidate["imu_delta"]
            lidar_initial_guess = (
                np.linalg.inv(initial_transform) @ imu_delta @ initial_transform
            )
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
                    "selected": True,
                    "registration_fitness": registration_fitness,
                    "registration_inlier_rmse": registration_inlier_rmse,
                }
            )
            if registration_fitness < min_registration_fitness:
                diagnostic["selected"] = False
                diagnostic["reason"] = "low_registration_fitness"
                motion_rejected_low_fitness += 1
                rejection_reasons.append("low_registration_fitness")
                motion_diagnostics.append(diagnostic)
                continue

            weight = max(registration_fitness, 1e-3)
            imu_quat = R.from_matrix(imu_delta[:3, :3]).as_quat()
            lidar_quat = R.from_matrix(lidar_delta[:3, :3]).as_quat()
            motion_samples.append(
                {
                    "start_timestamp_ns": int(start_meta.timestamp_ns),
                    "end_timestamp_ns": int(end_meta.timestamp_ns),
                    "imu_delta": {
                        "translation": {
                            "x": float(imu_delta[0, 3]),
                            "y": float(imu_delta[1, 3]),
                            "z": float(imu_delta[2, 3]),
                        },
                        "rotation": {
                            "x": float(imu_quat[0]),
                            "y": float(imu_quat[1]),
                            "z": float(imu_quat[2]),
                            "w": float(imu_quat[3]),
                        },
                    },
                    "lidar_delta": {
                        "translation": {
                            "x": float(lidar_delta[0, 3]),
                            "y": float(lidar_delta[1, 3]),
                            "z": float(lidar_delta[2, 3]),
                        },
                        "rotation": {
                            "x": float(lidar_quat[0]),
                            "y": float(lidar_quat[1]),
                            "z": float(lidar_quat[2]),
                            "w": float(lidar_quat[3]),
                        },
                    },
                    "weight": weight,
                    "sync_dt_ms": float(max(start_pose_dt_ns, end_pose_dt_ns) / 1e6),
                    "metadata": {
                        "record_path_start": start_meta.record_path,
                        "record_path_end": end_meta.record_path,
                        "lidar_topic": lidar_topic,
                        "pose_topic": pose_topic,
                        "window_id": int(window["window_id"]),
                        "frame_stride": int(candidate["stride"]),
                        "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
                        "pose_translation_m": float(candidate["pose_translation_m"]),
                        "registration_fitness": registration_fitness,
                        "registration_inlier_rmse": registration_inlier_rmse,
                    },
                }
            )
            used_motion_ranges.append((candidate["start_index"], candidate["end_index"]))
            selected_candidate_summary = {
                "start_timestamp_ns": int(start_meta.timestamp_ns),
                "end_timestamp_ns": int(end_meta.timestamp_ns),
                "frame_stride": int(candidate["stride"]),
                "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
                "pose_translation_m": float(candidate["pose_translation_m"]),
                "registration_fitness": registration_fitness,
                "registration_inlier_rmse": registration_inlier_rmse,
            }
            motion_diagnostics.append(diagnostic)
            break

        window_diagnostic["attempt_count"] = int(attempt_count)
        window_diagnostic["overlap_skipped"] = int(overlap_skipped)
        if selected_candidate_summary is not None:
            window_diagnostic["selected"] = True
            window_diagnostic["selected_candidate"] = selected_candidate_summary
        else:
            window_diagnostic["selected"] = False
            if len(motion_samples) >= max_motion_samples:
                window_diagnostic["reason"] = "max_motion_samples_reached"
            elif attempt_count <= 0 and overlap_skipped > 0:
                window_diagnostic["reason"] = "window_candidates_overlap_selected_ranges"
            elif rejection_reasons:
                window_diagnostic["reason"] = "window_no_candidate_passing_gate"
                window_diagnostic["candidate_rejection_reasons"] = rejection_reasons
            else:
                window_diagnostic["reason"] = "window_not_selected"
        motion_window_diagnostics.append(window_diagnostic)
        if len(motion_samples) >= max_motion_samples:
            break

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
                            R.from_matrix(initial_transform[:3, :3]).as_quat(),
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
            "record_path": record_path,
            "record_files": record_files,
            "lidar_topic": lidar_topic,
            "pose_topic": pose_topic,
            "imu_topic": imu_topic,
            "gravity_source": gravity_source,
            "ground_pose_sync_threshold_ms": float(ground_pose_sync_threshold_ms),
            "motion_pose_sync_threshold_ms": float(motion_pose_sync_threshold_ms),
            "motion_frame_stride": int(motion_frame_stride),
            "min_registration_fitness": float(min_registration_fitness),
            "initial_transform_source": initial_transform_source,
            "ground_selected": len(ground_samples),
            "motion_selected": len(motion_samples),
            "motion_candidate_count": len(motion_candidates),
            "motion_window_count": len(motion_windows),
            "motion_valid_window_count": int(
                sum(1 for window in motion_windows if window["valid"])
            ),
        },
        "ground_samples": ground_samples,
        "motion_samples": motion_samples,
    }
    diagnostics = {
        "summary": {
            "record_files": record_files,
            "lidar_topic": lidar_topic,
            "lidar_frame": lidar_frame,
            "parent_frame": parent_frame,
            "ground_selected": len(ground_samples),
            "motion_selected": len(motion_samples),
            "ground_attempted": len(ground_indices),
            "motion_attempted": len(motion_candidates),
            "motion_rejected_low_fitness": motion_rejected_low_fitness,
            "motion_window_count": len(motion_windows),
            "motion_valid_window_count": int(
                sum(1 for window in motion_windows if window["valid"])
            ),
            "min_registration_fitness": float(min_registration_fitness),
        },
        "ground": ground_diagnostics,
        "motion_windows": motion_window_diagnostics,
        "motion": motion_diagnostics,
    }
    with open(sample_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    with open(diagnostics_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(diagnostics, file, sort_keys=False)
    return sample_path, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Apollo record data into standardized lidar2imu samples and optionally run calibration."
    )
    parser.add_argument(
        "--record-path",
        required=True,
        help="Path to a record file or split-record directory.",
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
        "--planar-motion-policy",
        default="auto",
        choices=["auto", "free", "freeze_xyyaw"],
        help="How to handle weak planar observability during calibration.",
    )
    args = parser.parse_args()

    sample_path, diagnostics = convert_record_to_standardized_samples(
        record_path=args.record_path,
        output_dir=args.output_dir,
        lidar_topic=args.lidar_topic,
        pose_topic=args.pose_topic,
        imu_topic=args.imu_topic,
        parent_frame=args.parent_frame,
        child_frame=args.child_frame,
        initial_transform_path=args.initial_transform,
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
    )
    logging.info("Wrote standardized samples to %s", sample_path)
    logging.info(
        "Converted %d ground samples and %d motion samples.",
        diagnostics["summary"]["ground_selected"],
        diagnostics["summary"]["motion_selected"],
    )

    if not args.calibrate:
        return

    calibration_output_dir = Path(args.output_dir) / "calibration"
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


if __name__ == "__main__":
    main()
