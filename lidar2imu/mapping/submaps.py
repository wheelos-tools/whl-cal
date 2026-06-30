from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d

from lidar2imu.extraction.timing import nearest_sample, shift_timestamp_ns
from lidar2imu.local_mapping import (LocalMapBuildConfig,
                                     build_bidirectional_local_map)
from lidar2lidar.extrinsic_io import transform_dict_from_matrix
from lidar2lidar.prepared_dataset import PoseSample
from lidar2lidar.record_utils import load_pointcloud_from_meta


@dataclass(frozen=True)
class LocalSubmapBuildConfig:
    half_window: int
    support_stride: int
    min_support_frames: int
    voxel_size: float
    builder_mode: str


def load_cached_cloud(meta: Any, cloud_cache: dict) -> o3d.geometry.PointCloud:
    cache_key = (meta.topic, int(meta.timestamp_ns))
    if cache_key not in cloud_cache:
        cloud_cache[cache_key] = load_pointcloud_from_meta(meta)
    return cloud_cache[cache_key]


def resolve_world_lidar_transform(
    meta_index: int,
    *,
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    pose_time_offset_ns: int,
    sync_threshold_ns: int,
    extraction_transform: np.ndarray,
    alignment_cache: dict[int, dict],
) -> dict:
    if meta_index in alignment_cache:
        return alignment_cache[meta_index]
    meta = lidar_metas[meta_index]
    adjusted_timestamp_ns = shift_timestamp_ns(meta.timestamp_ns, pose_time_offset_ns)
    pose_sample, pose_dt_ns = nearest_sample(
        pose_samples,
        pose_timestamps,
        adjusted_timestamp_ns,
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


def _serialize_support_record(
    entry: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "meta_index": int(entry["meta_index"]),
        "timestamp_ns": int(record["timestamp_ns"]),
        "record_path": str(record["record_path"]),
        "pose_sync_dt_ms": float(entry["pose_sync_dt_ms"]),
        "point_count": int(entry["point_count"]),
        "initial_transform_anchor_support": transform_dict_from_matrix(
            record["initial_transform"]
        ),
        "seed_transform_anchor_support": transform_dict_from_matrix(
            record["seed_transform"]
        ),
        "transform_anchor_support": transform_dict_from_matrix(
            record["refined_transform"]
        ),
        "local_refined": bool(record["local_refined"]),
        "local_registration_fitness": record["registration_fitness"],
        "local_registration_inlier_rmse": record["registration_inlier_rmse"],
        "local_registration_attempted_stages": int(
            record["registration_attempted_stages"]
        ),
        "seed_correction_translation_m": float(record["seed_correction_translation_m"]),
        "seed_correction_rotation_deg": float(record["seed_correction_rotation_deg"]),
    }


def _serialize_rejected_record(
    entry: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    return {
        **_serialize_support_record(entry, record),
        "skip_reason": str(record["skip_reason"]),
    }


def _support_indices(
    anchor_index: int, *, meta_count: int, config: LocalSubmapBuildConfig
) -> list[int]:
    max_index = int(meta_count) - 1
    indices = []
    for offset in range(-int(config.half_window), int(config.half_window) + 1):
        support_index = int(anchor_index) + (offset * int(config.support_stride))
        if 0 <= support_index <= max_index:
            indices.append(int(support_index))
    return sorted(set(indices))


def _pose_only_submap(
    entry_records: list[dict[str, Any]], *, voxel_size: float
) -> dict:
    merged_cloud = o3d.geometry.PointCloud()
    support_records = []
    for entry in entry_records:
        support_cloud = o3d.geometry.PointCloud(entry["cloud"])
        support_cloud.transform(entry["initial_transform"])
        merged_cloud += support_cloud
        support_records.append(
            {
                "meta_index": int(entry["meta_index"]),
                "timestamp_ns": int(entry["timestamp_ns"]),
                "record_path": str(entry["record_path"]),
                "pose_sync_dt_ms": float(entry["pose_sync_dt_ms"]),
                "point_count": int(entry["point_count"]),
                "initial_transform_anchor_support": transform_dict_from_matrix(
                    entry["initial_transform"]
                ),
                "seed_transform_anchor_support": transform_dict_from_matrix(
                    entry["seed_transform"]
                ),
                "transform_anchor_support": transform_dict_from_matrix(
                    entry["initial_transform"]
                ),
                "local_refined": False,
                "local_registration_fitness": None,
                "local_registration_inlier_rmse": None,
                "local_registration_attempted_stages": 0,
                "seed_correction_translation_m": 0.0,
                "seed_correction_rotation_deg": 0.0,
            }
        )
    if voxel_size > 0.0:
        merged_cloud = merged_cloud.voxel_down_sample(float(voxel_size))
    return {
        "merged_cloud": merged_cloud,
        "support_records": support_records,
        "rejected_records": [],
    }


def build_local_lidar_submap(
    anchor_index: int,
    *,
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    pose_time_offset_ns: int,
    sync_threshold_ns: int,
    extraction_transform: np.ndarray,
    config: LocalSubmapBuildConfig,
    alignment_cache: dict[int, dict],
    cloud_cache: dict,
    submap_cache: dict,
) -> tuple[o3d.geometry.PointCloud | None, dict]:
    cache_key = (
        int(anchor_index),
        int(config.half_window),
        int(config.support_stride),
        int(config.min_support_frames),
        float(config.voxel_size),
        str(config.builder_mode),
    )
    if cache_key in submap_cache:
        cloud, info = submap_cache[cache_key]
        return cloud, dict(info)

    anchor_alignment = resolve_world_lidar_transform(
        anchor_index,
        lidar_metas=lidar_metas,
        pose_samples=pose_samples,
        pose_timestamps=pose_timestamps,
        pose_time_offset_ns=pose_time_offset_ns,
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
    entry_records = []
    anchor_entry_index = None
    for support_index in _support_indices(
        anchor_index, meta_count=len(lidar_metas), config=config
    ):
        support_alignment = resolve_world_lidar_transform(
            support_index,
            lidar_metas=lidar_metas,
            pose_samples=pose_samples,
            pose_timestamps=pose_timestamps,
            pose_time_offset_ns=pose_time_offset_ns,
            sync_threshold_ns=sync_threshold_ns,
            extraction_transform=extraction_transform,
            alignment_cache=alignment_cache,
        )
        if not support_alignment["valid"]:
            continue
        support_pose = np.asarray(
            support_alignment["transform_world_lidar"], dtype=float
        )
        support_cloud = load_cached_cloud(lidar_metas[support_index], cloud_cache)
        if support_cloud.is_empty():
            continue
        initial_transform_anchor_support = np.linalg.inv(anchor_pose) @ support_pose
        entry_records.append(
            {
                "meta_index": int(support_index),
                "timestamp_ns": int(lidar_metas[support_index].timestamp_ns),
                "record_path": lidar_metas[support_index].record_path,
                "pose_sync_dt_ms": float(support_alignment["pose_sync_dt_ms"]),
                "point_count": int(len(support_cloud.points)),
                "initial_transform": np.asarray(
                    initial_transform_anchor_support, dtype=float
                ),
                "seed_transform": np.asarray(
                    initial_transform_anchor_support, dtype=float
                ),
                "cloud": support_cloud,
            }
        )
        if int(support_index) == int(anchor_index):
            anchor_entry_index = len(entry_records) - 1

    if anchor_entry_index is None:
        info = {
            "valid": False,
            "reason": "anchor_missing_support_frame",
            "anchor_index": int(anchor_index),
            "support_frame_count": 0,
        }
        submap_cache[cache_key] = (None, info)
        return None, dict(info)

    if str(config.builder_mode) == "dense_scan_to_map_gicp":
        build_result = build_bidirectional_local_map(
            entry_records,
            anchor_index=int(anchor_entry_index),
            config=LocalMapBuildConfig(
                local_map_voxel_size=max(min(float(config.voxel_size), 0.20), 0.15),
                final_map_voxel_size=max(float(config.voxel_size), 0.20),
                max_correction_translation_m=0.20,
                max_correction_rotation_deg=1.75,
            ),
        )
        merged_cloud = o3d.geometry.PointCloud(build_result["merged_cloud"])
        entry_lookup = {
            (int(entry["timestamp_ns"]), str(entry["record_path"])): entry
            for entry in entry_records
        }
        support_records = [
            _serialize_support_record(
                entry_lookup[(int(record["timestamp_ns"]), str(record["record_path"]))],
                record,
            )
            for record in build_result["support_records"]
        ]
        rejected_records = [
            _serialize_rejected_record(
                entry_lookup[(int(record["timestamp_ns"]), str(record["record_path"]))],
                record,
            )
            for record in build_result["rejected_records"]
        ]
    else:
        pose_only_result = _pose_only_submap(
            entry_records,
            voxel_size=float(config.voxel_size),
        )
        merged_cloud = pose_only_result["merged_cloud"]
        support_records = pose_only_result["support_records"]
        rejected_records = pose_only_result["rejected_records"]

    if len(support_records) < int(config.min_support_frames):
        info = {
            "valid": False,
            "reason": "insufficient_submap_support",
            "anchor_index": int(anchor_index),
            "support_frame_count": int(len(support_records)),
            "support_records": support_records,
            "rejected_frame_count": int(len(rejected_records)),
            "rejected_records": rejected_records,
        }
        submap_cache[cache_key] = (None, info)
        return None, dict(info)

    info = {
        "valid": True,
        "anchor_index": int(anchor_index),
        "support_frame_count": int(len(support_records)),
        "point_count": int(len(merged_cloud.points)),
        "requested_frame_count": int(len(entry_records)),
        "rejected_frame_count": int(len(rejected_records)),
        "accepted_frame_ratio": (
            float(len(support_records) / len(entry_records)) if entry_records else 0.0
        ),
        "builder_mode": (
            "bidirectional_scan_to_map_gicp"
            if str(config.builder_mode) == "dense_scan_to_map_gicp"
            else str(config.builder_mode)
        ),
        "refined_support_count": int(
            sum(1 for record in support_records if record.get("local_refined"))
        ),
        "support_records": support_records,
        "rejected_records": rejected_records,
        "skip_reason_counts": {
            reason: int(
                sum(
                    1
                    for record in rejected_records
                    if record.get("skip_reason") == reason
                )
            )
            for reason in sorted(
                {
                    str(record["skip_reason"])
                    for record in rejected_records
                    if record.get("skip_reason") is not None
                }
            )
        },
    }
    submap_cache[cache_key] = (merged_cloud, info)
    return merged_cloud, dict(info)
