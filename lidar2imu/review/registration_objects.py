from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp

from lidar2lidar.extrinsic_io import matrix_from_transform_dict
from lidar2lidar.record_utils import (
    PointCloudMeta,
    collect_pointcloud_metadata,
    load_pointcloud_from_meta,
)


def _registration_object_cache_key(
    descriptor: dict[str, Any], *, lidar_topic: str, frame_id: str
) -> tuple[Any, ...]:
    return (
        str(lidar_topic),
        str(frame_id),
        str(descriptor.get("object_type", "")),
        str(descriptor.get("builder_mode", "")),
        descriptor.get("anchor_index"),
        tuple(
            (
                int(record.get("timestamp_ns", 0)),
                str(record.get("record_path", "")),
            )
            for record in descriptor.get("support_records", [])
            if isinstance(record, dict)
        ),
    )


def _descriptor_anchor_timestamp_ns(descriptor: dict[str, Any]) -> int | None:
    for record in descriptor.get("support_records", []):
        if not isinstance(record, dict):
            continue
        transform_payload = record.get("transform_anchor_support")
        if not isinstance(transform_payload, dict):
            continue
        transform = matrix_from_transform_dict(transform_payload)
        if np.linalg.norm(transform[:3, 3]) < 1e-6 and np.allclose(
            transform[:3, :3], np.eye(3), atol=1e-6
        ):
            return int(record.get("timestamp_ns", 0))
    support_records = [
        record
        for record in descriptor.get("support_records", [])
        if isinstance(record, dict) and record.get("timestamp_ns") is not None
    ]
    if not support_records:
        return None
    support_records.sort(key=lambda item: int(item["timestamp_ns"]))
    return int(support_records[len(support_records) // 2]["timestamp_ns"])


def _review_dense_half_window_ns(
    descriptor: dict[str, Any], sparse_records: list[dict[str, Any]]
) -> int:
    if len(sparse_records) < 2:
        return int(1.5e9)
    timestamps = sorted(int(record["timestamp_ns"]) for record in sparse_records)
    original_span_ns = int(timestamps[-1] - timestamps[0])
    object_type = str(descriptor.get("object_type", ""))
    cap_ns = int(1.5e9) if object_type == "source_submap" else int(2.0e9)
    min_ns = int(1.0e9)
    return int(max(min(original_span_ns // 4, cap_ns), min_ns))


def _load_review_topic_metadata(
    record_paths: list[str],
    *,
    lidar_topic: str,
    metadata_cache: dict[tuple[Any, ...], list[PointCloudMeta]],
) -> list[PointCloudMeta]:
    cache_key = (tuple(sorted(record_paths)), str(lidar_topic))
    if cache_key not in metadata_cache:
        metadata = collect_pointcloud_metadata(cache_key[0], [str(lidar_topic)])
        metadata_cache[cache_key] = list(metadata.get(str(lidar_topic), []))
    return metadata_cache[cache_key]


def _interpolate_sparse_local_transform(
    sparse_records: list[dict[str, Any]], timestamp_ns: int
) -> np.ndarray:
    if not sparse_records:
        return np.eye(4, dtype=float)
    timestamps = [int(record["timestamp_ns"]) for record in sparse_records]
    index = bisect_left(timestamps, int(timestamp_ns))
    if index <= 0:
        return np.asarray(sparse_records[0]["initial_transform"], dtype=float).copy()
    if index >= len(sparse_records):
        return np.asarray(sparse_records[-1]["initial_transform"], dtype=float).copy()
    left = sparse_records[index - 1]
    right = sparse_records[index]
    left_ts = int(left["timestamp_ns"])
    right_ts = int(right["timestamp_ns"])
    if right_ts <= left_ts:
        return np.asarray(left["initial_transform"], dtype=float).copy()
    fraction = float((int(timestamp_ns) - left_ts) / (right_ts - left_ts))
    left_transform = np.asarray(left["initial_transform"], dtype=float)
    right_transform = np.asarray(right["initial_transform"], dtype=float)
    left_rot = R.from_matrix(left_transform[:3, :3])
    right_rot = R.from_matrix(right_transform[:3, :3])
    slerp = Slerp(
        [float(left_ts), float(right_ts)],
        R.concatenate([left_rot, right_rot]),
    )
    interpolated_rotation = slerp([float(timestamp_ns)])[0].as_matrix()
    interpolated_translation = (1.0 - fraction) * left_transform[:3, 3] + fraction * (
        right_transform[:3, 3]
    )
    interpolated_transform = np.eye(4, dtype=float)
    interpolated_transform[:3, :3] = interpolated_rotation
    interpolated_transform[:3, 3] = interpolated_translation
    return interpolated_transform


def _load_review_scan_cloud(
    *,
    record_path: str,
    timestamp_ns: int,
    lidar_topic: str,
    frame_id: str,
    scan_cache: dict[tuple[Any, ...], o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    cache_key = (str(lidar_topic), str(frame_id), int(timestamp_ns), str(record_path))
    if cache_key not in scan_cache:
        cloud = load_pointcloud_from_meta(
            PointCloudMeta(
                topic=str(lidar_topic),
                frame_id=str(frame_id),
                timestamp_ns=int(timestamp_ns),
                record_path=str(record_path),
            )
        )
        if not cloud.is_empty():
            cloud = cloud.voxel_down_sample(0.15)
        scan_cache[cache_key] = cloud
    return o3d.geometry.PointCloud(scan_cache[cache_key])


def _dense_review_entries(
    descriptor: dict[str, Any],
    *,
    lidar_topic: str,
    frame_id: str,
    scan_cache: dict[tuple[Any, ...], o3d.geometry.PointCloud],
    metadata_cache: dict[tuple[Any, ...], list[PointCloudMeta]],
) -> tuple[list[dict[str, Any]], int]:
    sparse_initial_records = []
    sparse_refined_records = []
    record_paths = []
    for record in descriptor.get("support_records", []):
        if not isinstance(record, dict):
            continue
        record_path = record.get("record_path")
        if not record_path:
            continue
        initial_transform_payload = record.get(
            "initial_transform_anchor_support"
        ) or record.get("transform_anchor_support")
        refined_transform_payload = record.get("transform_anchor_support")
        sparse_initial_records.append(
            {
                "timestamp_ns": int(record.get("timestamp_ns", 0)),
                "record_path": str(record_path),
                "initial_transform": (
                    matrix_from_transform_dict(initial_transform_payload)
                    if isinstance(initial_transform_payload, dict)
                    else np.eye(4, dtype=float)
                ),
            }
        )
        sparse_refined_records.append(
            {
                "timestamp_ns": int(record.get("timestamp_ns", 0)),
                "record_path": str(record_path),
                "initial_transform": (
                    matrix_from_transform_dict(refined_transform_payload)
                    if isinstance(refined_transform_payload, dict)
                    else np.eye(4, dtype=float)
                ),
            }
        )
        record_paths.append(str(record_path))
    if not sparse_initial_records:
        return [], 0
    sparse_initial_records.sort(key=lambda item: int(item["timestamp_ns"]))
    sparse_refined_records.sort(key=lambda item: int(item["timestamp_ns"]))
    anchor_timestamp_ns = _descriptor_anchor_timestamp_ns(descriptor)
    if anchor_timestamp_ns is None:
        anchor_timestamp_ns = int(
            sparse_initial_records[len(sparse_initial_records) // 2]["timestamp_ns"]
        )
    dense_half_window_ns = _review_dense_half_window_ns(
        descriptor, sparse_initial_records
    )
    metas = _load_review_topic_metadata(
        sorted(set(record_paths)),
        lidar_topic=str(lidar_topic),
        metadata_cache=metadata_cache,
    )
    dense_metas = [
        meta
        for meta in metas
        if abs(int(meta.timestamp_ns) - int(anchor_timestamp_ns))
        <= dense_half_window_ns
    ]
    if not dense_metas:
        dense_metas = [
            PointCloudMeta(
                topic=str(lidar_topic),
                frame_id=str(frame_id),
                timestamp_ns=int(record["timestamp_ns"]),
                record_path=str(record["record_path"]),
            )
            for record in sparse_initial_records
        ]
    max_dense_frames = 41
    if len(dense_metas) > max_dense_frames:
        indices = np.linspace(0, len(dense_metas) - 1, num=max_dense_frames, dtype=int)
        dense_metas = [dense_metas[index] for index in sorted(set(indices.tolist()))]
    entries = []
    for meta in dense_metas:
        cloud = _load_review_scan_cloud(
            record_path=str(meta.record_path),
            timestamp_ns=int(meta.timestamp_ns),
            lidar_topic=str(lidar_topic),
            frame_id=str(frame_id),
            scan_cache=scan_cache,
        )
        if cloud.is_empty():
            continue
        entries.append(
            {
                "timestamp_ns": int(meta.timestamp_ns),
                "record_path": str(meta.record_path),
                "initial_transform": _interpolate_sparse_local_transform(
                    sparse_initial_records, int(meta.timestamp_ns)
                ),
                "seed_transform": _interpolate_sparse_local_transform(
                    sparse_refined_records, int(meta.timestamp_ns)
                ),
                "cloud": cloud,
            }
        )
    if not entries:
        return [], 0
    entries.sort(key=lambda item: int(item["timestamp_ns"]))
    anchor_index = min(
        range(len(entries)),
        key=lambda index: abs(
            int(entries[index]["timestamp_ns"]) - int(anchor_timestamp_ns)
        ),
    )
    anchor_entry = entries[anchor_index]
    anchor_entry["initial_transform"] = np.eye(4, dtype=float)
    anchor_entry["seed_transform"] = np.eye(4, dtype=float)
    return entries, int(anchor_index)


def _copy_transform_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "initial_transform": np.asarray(
                record["initial_transform"], dtype=float
            ).copy(),
            "seed_transform": np.asarray(record["seed_transform"], dtype=float).copy(),
            "refined_transform": np.asarray(
                record["refined_transform"], dtype=float
            ).copy(),
        }
        for record in records
    ]


def _cacheable_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "cloud"}


def _descriptor_support_records(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    support_records = []
    for record in descriptor.get("support_records", []):
        if not isinstance(record, dict):
            continue
        record_path = record.get("record_path")
        if not record_path:
            continue
        initial_transform_payload = record.get(
            "initial_transform_anchor_support"
        ) or record.get("transform_anchor_support")
        refined_transform_payload = record.get("transform_anchor_support")
        initial_transform = (
            matrix_from_transform_dict(initial_transform_payload)
            if isinstance(initial_transform_payload, dict)
            else np.eye(4, dtype=float)
        )
        refined_transform = (
            matrix_from_transform_dict(refined_transform_payload)
            if isinstance(refined_transform_payload, dict)
            else np.asarray(initial_transform, dtype=float)
        )
        support_records.append(
            {
                "timestamp_ns": int(record.get("timestamp_ns", 0)),
                "record_path": str(record_path),
                "point_count": (
                    None
                    if record.get("point_count") is None
                    else int(record.get("point_count", 0))
                ),
                "initial_transform": np.asarray(initial_transform, dtype=float),
                "seed_transform": np.asarray(refined_transform, dtype=float),
                "refined_transform": np.asarray(refined_transform, dtype=float),
                "local_refined": bool(record.get("local_refined", False)),
                "registration_fitness": record.get("local_registration_fitness"),
                "registration_inlier_rmse": record.get(
                    "local_registration_inlier_rmse"
                ),
                "registration_attempted_stages": int(
                    record.get("local_registration_attempted_stages", 0)
                ),
                "seed_correction_translation_m": float(
                    record.get("seed_correction_translation_m", 0.0)
                ),
                "seed_correction_rotation_deg": float(
                    record.get("seed_correction_rotation_deg", 0.0)
                ),
            }
        )
    support_records.sort(key=lambda item: int(item["timestamp_ns"]))
    return support_records


def _dense_scene_records(
    entries: list[dict[str, Any]], anchor_index: int
) -> list[dict[str, Any]]:
    scene_records = []
    for index, entry in enumerate(entries):
        refined_transform = np.asarray(entry["seed_transform"], dtype=float)
        if int(index) == int(anchor_index):
            refined_transform = np.eye(4, dtype=float)
        scene_records.append(
            {
                "timestamp_ns": int(entry["timestamp_ns"]),
                "record_path": str(entry["record_path"]),
                "initial_transform": np.asarray(
                    entry["initial_transform"], dtype=float
                ),
                "seed_transform": refined_transform.copy(),
                "refined_transform": refined_transform.copy(),
                "local_refined": False,
                "registration_fitness": None,
                "registration_inlier_rmse": None,
                "registration_attempted_stages": 0,
                "seed_correction_translation_m": 0.0,
                "seed_correction_rotation_deg": 0.0,
                "cloud": o3d.geometry.PointCloud(entry["cloud"]),
            }
        )
    scene_records.sort(key=lambda item: int(item["timestamp_ns"]))
    return scene_records


def _support_scene_records(
    support_records: list[dict[str, Any]],
    *,
    lidar_topic: str,
    frame_id: str,
    scan_cache: dict[tuple[Any, ...], o3d.geometry.PointCloud],
) -> list[dict[str, Any]]:
    scene_records = []
    for record in support_records:
        cloud = _load_review_scan_cloud(
            record_path=str(record["record_path"]),
            timestamp_ns=int(record["timestamp_ns"]),
            lidar_topic=str(lidar_topic),
            frame_id=str(frame_id),
            scan_cache=scan_cache,
        )
        if cloud.is_empty():
            continue
        scene_records.append(
            {
                **record,
                "cloud": cloud,
            }
        )
    return scene_records


def _merge_scene_cloud(scene_records: list[dict[str, Any]]) -> o3d.geometry.PointCloud:
    merged_cloud = o3d.geometry.PointCloud()
    for record in scene_records:
        cloud = o3d.geometry.PointCloud(record["cloud"])
        cloud.transform(np.asarray(record["refined_transform"], dtype=float))
        merged_cloud += cloud
    if not merged_cloud.is_empty():
        merged_cloud = merged_cloud.voxel_down_sample(0.20)
    return merged_cloud


def build_registration_object_review(
    descriptor: dict[str, Any],
    *,
    lidar_topic: str,
    frame_id: str,
    scan_cache: dict[tuple[Any, ...], o3d.geometry.PointCloud],
    metadata_cache: dict[tuple[Any, ...], list[PointCloudMeta]],
    object_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = _registration_object_cache_key(
        descriptor, lidar_topic=lidar_topic, frame_id=frame_id
    )
    cached = object_cache.get(cache_key)
    if cached is not None:
        return {
            "merged_cloud": o3d.geometry.PointCloud(cached["merged_cloud"]),
            "support_records": _copy_transform_records(cached["support_records"]),
            "trajectory_records": _copy_transform_records(cached["trajectory_records"]),
            "refinement_mode": str(cached["refinement_mode"]),
            "frame_count": int(cached["frame_count"]),
            "requested_frame_count": int(cached["requested_frame_count"]),
            "rejected_frame_count": int(cached["rejected_frame_count"]),
            "skip_reason_counts": dict(cached["skip_reason_counts"]),
            "review_input_mode": str(cached["review_input_mode"]),
        }

    support_records = _descriptor_support_records(descriptor)
    entries, anchor_index = _dense_review_entries(
        descriptor,
        lidar_topic=str(lidar_topic),
        frame_id=str(frame_id),
        scan_cache=scan_cache,
        metadata_cache=metadata_cache,
    )
    if entries:
        scene_records = _dense_scene_records(entries, anchor_index)
        refinement_mode = "dense_interpolated_scene"
        review_input_mode = "dense_support_pose_interpolation"
    else:
        scene_records = _support_scene_records(
            support_records,
            lidar_topic=str(lidar_topic),
            frame_id=str(frame_id),
            scan_cache=scan_cache,
        )
        refinement_mode = "exact_sparse_support_scene"
        review_input_mode = "exact_support_records"
    if not scene_records:
        return {
            "merged_cloud": o3d.geometry.PointCloud(),
            "support_records": [],
            "trajectory_records": [],
            "refinement_mode": refinement_mode,
            "frame_count": 0,
            "requested_frame_count": 0,
            "rejected_frame_count": 0,
            "skip_reason_counts": {},
            "review_input_mode": review_input_mode,
        }
    cached_payload = {
        "merged_cloud": _merge_scene_cloud(scene_records),
        "support_records": [_cacheable_record(record) for record in support_records],
        "trajectory_records": [_cacheable_record(record) for record in scene_records],
        "refinement_mode": refinement_mode,
        "frame_count": int(len(scene_records)),
        "requested_frame_count": int(len(scene_records)),
        "rejected_frame_count": 0,
        "skip_reason_counts": {},
        "review_input_mode": review_input_mode,
    }
    object_cache[cache_key] = cached_payload
    return {
        "merged_cloud": o3d.geometry.PointCloud(cached_payload["merged_cloud"]),
        "support_records": _copy_transform_records(cached_payload["support_records"]),
        "trajectory_records": _copy_transform_records(
            cached_payload["trajectory_records"]
        ),
        "refinement_mode": str(cached_payload["refinement_mode"]),
        "frame_count": int(cached_payload["frame_count"]),
        "requested_frame_count": int(cached_payload["requested_frame_count"]),
        "rejected_frame_count": int(cached_payload["rejected_frame_count"]),
        "skip_reason_counts": dict(cached_payload["skip_reason_counts"]),
        "review_input_mode": str(cached_payload["review_input_mode"]),
    }
