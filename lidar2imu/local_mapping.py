from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class LocalMapBuildConfig:
    registration_voxel_sizes: tuple[float, ...] = (0.40, 0.20)
    min_source_points: int = 50
    min_target_points: int = 80
    min_fitness: float = 0.35
    max_correspondence_distance_scale: float = 4.0
    max_correspondence_distance_min: float = 0.40
    max_iteration: int = 50
    local_map_voxel_size: float = 0.15
    final_map_voxel_size: float = 0.20
    max_correction_translation_m: float = 0.15
    max_correction_rotation_deg: float = 1.50


def preprocess_registration_cloud(
    cloud: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.PointCloud:
    prepared = o3d.geometry.PointCloud(cloud)
    if prepared.is_empty():
        return prepared
    prepared = prepared.voxel_down_sample(float(voxel_size))
    if prepared.is_empty():
        return prepared
    points = np.asarray(prepared.points)
    finite_mask = np.isfinite(points).all(axis=1)
    if not finite_mask.all():
        prepared = prepared.select_by_index(np.where(finite_mask)[0])
    if len(prepared.points) >= 20:
        prepared.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(float(voxel_size) * 2.5, 0.3), max_nn=30
            )
        )
    return prepared


def refine_scan_to_local_map(
    source_cloud: o3d.geometry.PointCloud,
    target_map: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    *,
    voxel_sizes: tuple[float, ...] = (0.40, 0.20),
    min_source_points: int = 50,
    min_target_points: int = 80,
    min_fitness: float = 0.35,
    max_correspondence_distance_scale: float = 4.0,
    max_correspondence_distance_min: float = 0.40,
    max_iteration: int = 50,
) -> tuple[np.ndarray, dict[str, Any]]:
    current_transform = np.asarray(initial_transform, dtype=float)
    final_result = None
    attempted_stages = 0
    for voxel_size in voxel_sizes:
        source_stage = preprocess_registration_cloud(source_cloud, voxel_size)
        target_stage = preprocess_registration_cloud(target_map, voxel_size)
        if len(source_stage.points) < int(min_source_points) or len(
            target_stage.points
        ) < int(min_target_points):
            continue
        attempted_stages += 1
        result = o3d.pipelines.registration.registration_generalized_icp(
            source_stage,
            target_stage,
            max(
                float(voxel_size) * float(max_correspondence_distance_scale),
                float(max_correspondence_distance_min),
            ),
            current_transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=int(max_iteration),
            ),
        )
        if not np.isfinite(result.transformation).all():
            break
        current_transform = np.asarray(result.transformation, dtype=float)
        final_result = result
    if final_result is None or float(final_result.fitness) < float(min_fitness):
        return np.asarray(initial_transform, dtype=float), {
            "refined": False,
            "registration_fitness": (
                None if final_result is None else float(final_result.fitness)
            ),
            "registration_inlier_rmse": (
                None if final_result is None else float(final_result.inlier_rmse)
            ),
            "attempted_stages": int(attempted_stages),
        }
    return current_transform, {
        "refined": True,
        "registration_fitness": float(final_result.fitness),
        "registration_inlier_rmse": float(final_result.inlier_rmse),
        "attempted_stages": int(attempted_stages),
    }


def _pose_prior_transform(entry: dict[str, Any]) -> np.ndarray:
    transform = entry.get("seed_transform", entry["initial_transform"])
    return np.asarray(transform, dtype=float)


def _transform_delta_metrics(
    initial_transform: np.ndarray, refined_transform: np.ndarray
) -> tuple[float, float]:
    delta = np.linalg.inv(np.asarray(initial_transform, dtype=float)) @ np.asarray(
        refined_transform, dtype=float
    )
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    rotation_deg = float(
        np.degrees(np.linalg.norm(R.from_matrix(delta[:3, :3]).as_rotvec()))
    )
    return translation_m, rotation_deg


def _build_propagated_seed(
    previous_prior: np.ndarray,
    previous_refined: np.ndarray,
    current_prior: np.ndarray,
) -> np.ndarray:
    return np.asarray(previous_refined, dtype=float) @ (
        np.linalg.inv(np.asarray(previous_prior, dtype=float))
        @ np.asarray(current_prior, dtype=float)
    )


def _anchor_record(entry: dict[str, Any]) -> dict[str, Any]:
    identity = np.eye(4, dtype=float)
    return {
        "timestamp_ns": int(entry["timestamp_ns"]),
        "record_path": str(entry["record_path"]),
        "initial_transform": identity.copy(),
        "seed_transform": identity.copy(),
        "refined_transform": identity.copy(),
        "local_refined": False,
        "registration_fitness": None,
        "registration_inlier_rmse": None,
        "registration_attempted_stages": 0,
        "seed_correction_translation_m": 0.0,
        "seed_correction_rotation_deg": 0.0,
        "accepted": True,
        "skip_reason": None,
    }


def _ordered_bidirectional_indices(
    entry_count: int, anchor_index: int
) -> list[tuple[int, str]]:
    ordered: list[tuple[int, str]] = []
    for offset in range(1, entry_count):
        backward_index = int(anchor_index) - offset
        if backward_index >= 0:
            ordered.append((backward_index, "backward"))
        forward_index = int(anchor_index) + offset
        if forward_index < int(entry_count):
            ordered.append((forward_index, "forward"))
    return ordered


def build_bidirectional_local_map(
    entries: list[dict[str, Any]],
    *,
    anchor_index: int,
    config: LocalMapBuildConfig,
) -> dict[str, Any]:
    if not entries:
        return {
            "merged_cloud": o3d.geometry.PointCloud(),
            "support_records": [],
            "rejected_records": [],
            "frame_count": 0,
            "requested_frame_count": 0,
            "rejected_frame_count": 0,
            "accepted_frame_ratio": 0.0,
            "skip_reason_counts": {},
            "refinement_mode": "bidirectional_scan_to_map_gicp",
        }

    if not (0 <= int(anchor_index) < len(entries)):
        raise IndexError("anchor_index is out of range for local map entries.")

    normalized_entries = [
        {
            **entry,
            "timestamp_ns": int(entry["timestamp_ns"]),
            "record_path": str(entry["record_path"]),
            "initial_transform": np.asarray(entry["initial_transform"], dtype=float),
            "seed_transform": np.asarray(
                entry.get("seed_transform", entry["initial_transform"]), dtype=float
            ),
            "cloud": o3d.geometry.PointCloud(entry["cloud"]),
        }
        for entry in entries
    ]
    anchor_entry = normalized_entries[int(anchor_index)]
    anchor_cloud = o3d.geometry.PointCloud(anchor_entry["cloud"])
    anchor_record = _anchor_record(anchor_entry)

    accepted_records = [anchor_record]
    rejected_records: list[dict[str, Any]] = []
    transformed_clouds: list[o3d.geometry.PointCloud] = []
    local_map = o3d.geometry.PointCloud(anchor_cloud)
    chain_states = {
        "backward": {
            "previous_prior": np.eye(4, dtype=float),
            "previous_refined": np.eye(4, dtype=float),
        },
        "forward": {
            "previous_prior": np.eye(4, dtype=float),
            "previous_refined": np.eye(4, dtype=float),
        },
    }

    for index, direction in _ordered_bidirectional_indices(
        len(normalized_entries), int(anchor_index)
    ):
        entry = normalized_entries[index]
        state = chain_states[str(direction)]
        prior_transform = _pose_prior_transform(entry)
        propagated_seed = _build_propagated_seed(
            state["previous_prior"], state["previous_refined"], prior_transform
        )
        refined_transform, refinement_info = refine_scan_to_local_map(
            entry["cloud"],
            local_map,
            propagated_seed,
            voxel_sizes=config.registration_voxel_sizes,
            min_source_points=config.min_source_points,
            min_target_points=config.min_target_points,
            min_fitness=config.min_fitness,
            max_correspondence_distance_scale=config.max_correspondence_distance_scale,
            max_correspondence_distance_min=config.max_correspondence_distance_min,
            max_iteration=config.max_iteration,
        )
        correction_translation_m, correction_rotation_deg = _transform_delta_metrics(
            propagated_seed, refined_transform
        )
        record = {
            "timestamp_ns": int(entry["timestamp_ns"]),
            "record_path": str(entry["record_path"]),
            "initial_transform": np.asarray(entry["initial_transform"], dtype=float),
            "seed_transform": np.asarray(propagated_seed, dtype=float),
            "refined_transform": np.asarray(refined_transform, dtype=float),
            "local_refined": bool(refinement_info["refined"]),
            "registration_fitness": refinement_info["registration_fitness"],
            "registration_inlier_rmse": refinement_info["registration_inlier_rmse"],
            "registration_attempted_stages": int(refinement_info["attempted_stages"]),
            "seed_correction_translation_m": float(correction_translation_m),
            "seed_correction_rotation_deg": float(correction_rotation_deg),
            "accepted": False,
            "skip_reason": None,
        }
        if not refinement_info["refined"]:
            record["skip_reason"] = "low_fitness_or_insufficient_overlap"
            rejected_records.append(record)
            continue
        if correction_translation_m > float(config.max_correction_translation_m):
            record["skip_reason"] = "excessive_translation_correction"
            rejected_records.append(record)
            continue
        if correction_rotation_deg > float(config.max_correction_rotation_deg):
            record["skip_reason"] = "excessive_rotation_correction"
            rejected_records.append(record)
            continue

        record["accepted"] = True
        transformed_cloud = o3d.geometry.PointCloud(entry["cloud"])
        transformed_cloud.transform(refined_transform)
        transformed_clouds.append(transformed_cloud)
        local_map += transformed_cloud
        local_map = local_map.voxel_down_sample(float(config.local_map_voxel_size))
        accepted_records.append(record)
        state["previous_prior"] = prior_transform
        state["previous_refined"] = np.asarray(refined_transform, dtype=float)

    accepted_records.sort(key=lambda item: int(item["timestamp_ns"]))
    rejected_records.sort(key=lambda item: int(item["timestamp_ns"]))

    merged_cloud = o3d.geometry.PointCloud(anchor_cloud)
    for cloud in transformed_clouds:
        merged_cloud += cloud
    if not merged_cloud.is_empty():
        merged_cloud = merged_cloud.voxel_down_sample(
            float(config.final_map_voxel_size)
        )

    skip_reason_counts = Counter(
        str(record["skip_reason"])
        for record in rejected_records
        if record.get("skip_reason") is not None
    )
    requested_frame_count = int(len(normalized_entries))
    frame_count = int(len(accepted_records))
    rejected_frame_count = int(len(rejected_records))
    return {
        "merged_cloud": merged_cloud,
        "support_records": accepted_records,
        "rejected_records": rejected_records,
        "frame_count": frame_count,
        "requested_frame_count": requested_frame_count,
        "rejected_frame_count": rejected_frame_count,
        "accepted_frame_ratio": (
            float(frame_count / requested_frame_count)
            if requested_frame_count > 0
            else 0.0
        ),
        "skip_reason_counts": dict(skip_reason_counts),
        "refinement_mode": "bidirectional_scan_to_map_gicp",
    }
