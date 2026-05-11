#!/usr/bin/env python3

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

from lidar2lidar.record_adapter import Record, ensure_record_available
from lidar2lidar.record_utils import (PointCloudMeta, TransformEdge,
                                      build_transform_graph,
                                      collect_pointcloud_metadata,
                                      discover_record_files, extract_tf_edges,
                                      get_topic_frame_ids,
                                      load_pointcloud_from_meta,
                                      lookup_transform, topic_sensor_name)


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


@dataclass(frozen=True)
class PreparedRigDataset:
    dataset_path: str
    record_files: list[str]
    lidar_topics: list[str]
    topic_infos: dict[str, dict]
    metadata_by_topic: dict[str, list[PointCloudMeta]]
    tf_edges: list[TransformEdge]
    pose_topic: str
    imu_topic: str
    pose_samples: list[PoseSample]
    imu_samples: list[ImuSample]
    synchronized_snapshots: list[dict]
    manifest: dict


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


def collect_pose_samples(
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


def collect_imu_samples(record_files: list[str], imu_topic: str) -> list[ImuSample]:
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


def _sanitize_topic(topic: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", topic.strip("/"))


def _nearest_meta(
    metas: list[PointCloudMeta],
    timestamps: list[int],
    timestamp_ns: int,
    max_delta_ns: int,
) -> tuple[PointCloudMeta | None, int | None]:
    if not metas:
        return None, None
    index = bisect.bisect_left(timestamps, timestamp_ns)
    candidates: list[int] = []
    if index < len(metas):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None, None
    best_index = min(
        candidates, key=lambda candidate: abs(timestamps[candidate] - timestamp_ns)
    )
    delta_ns = abs(timestamps[best_index] - timestamp_ns)
    if delta_ns > max_delta_ns:
        return None, delta_ns
    return metas[best_index], delta_ns


def _serialize_transform(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]


def _serialize_tf_edge(edge: TransformEdge) -> dict:
    return {
        "parent_frame": edge.parent_frame,
        "child_frame": edge.child_frame,
        "transform": _serialize_transform(edge.transform),
        "source_topic": edge.source_topic,
        "timestamp_ns": edge.timestamp_ns,
        "is_static": bool(edge.is_static),
    }


def _deserialize_tf_edge(payload: dict) -> TransformEdge:
    return TransformEdge(
        parent_frame=str(payload["parent_frame"]),
        child_frame=str(payload["child_frame"]),
        transform=np.asarray(payload["transform"], dtype=float),
        source_topic=str(payload.get("source_topic", "")),
        timestamp_ns=(
            None
            if payload.get("timestamp_ns") is None
            else int(payload["timestamp_ns"])
        ),
        is_static=bool(payload.get("is_static", False)),
    )


def _export_cloud(
    meta: PointCloudMeta,
    output_path: Path,
    *,
    voxel_size: float | None,
) -> dict:
    cloud = load_pointcloud_from_meta(meta)
    if voxel_size is not None and voxel_size > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), cloud):
        raise RuntimeError(f"Failed to write cached point cloud to {output_path}.")
    return {
        "artifact_path": str(output_path),
        "point_count": int(len(cloud.points)),
    }


def _summarize_rate(metas: list[PointCloudMeta]) -> dict:
    if len(metas) < 2:
        return {"count": int(len(metas)), "duration_sec": 0.0, "hz": None}
    duration_sec = (metas[-1].timestamp_ns - metas[0].timestamp_ns) / 1e9
    hz = None if duration_sec <= 0.0 else float(len(metas) / duration_sec)
    return {"count": int(len(metas)), "duration_sec": float(duration_sec), "hz": hz}


def build_prepared_rig_dataset(
    *,
    record_path: str,
    output_dir: str,
    lidar_topics: list[str],
    pose_topic: str,
    imu_topic: str,
    parent_frame: str,
    reference_topic: str | None,
    sync_threshold_ms: float,
    frame_stride: int,
    max_snapshots: int | None,
    export_voxel_size: float | None,
) -> Path:
    if not lidar_topics:
        raise RuntimeError("At least one raw LiDAR topic is required.")
    record_files = discover_record_files(record_path)
    reference_topic = reference_topic or lidar_topics[0]
    if reference_topic not in lidar_topics:
        raise RuntimeError(
            f"Reference topic {reference_topic} must be one of the prepared LiDAR topics."
        )

    output_path = Path(output_dir)
    diagnostics_dir = output_path / "diagnostics"
    cache_dir = output_path / "cache"
    pointcloud_cache_dir = cache_dir / "pointclouds"
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_cache_dir.mkdir(parents=True, exist_ok=True)

    tf_edges = extract_tf_edges(record_files)
    tf_graph = build_transform_graph(tf_edges)
    localization_to_imu = lookup_transform(tf_graph, parent_frame, "localization")
    if localization_to_imu is None:
        raise RuntimeError(
            f"Could not find transform from {parent_frame} to localization."
        )

    topic_frame_ids = get_topic_frame_ids(record_files, lidar_topics)
    raw_metadata_by_topic = collect_pointcloud_metadata(record_files, lidar_topics)
    pose_samples = collect_pose_samples(record_files, pose_topic, localization_to_imu)
    imu_samples = collect_imu_samples(record_files, imu_topic)

    reference_metas = raw_metadata_by_topic[reference_topic]
    if not reference_metas:
        raise RuntimeError(
            f"No point clouds found for reference topic {reference_topic}."
        )
    sampled_reference = reference_metas[:: max(1, int(frame_stride))]
    if (
        max_snapshots is not None
        and max_snapshots > 0
        and len(sampled_reference) > max_snapshots
    ):
        indices = np.linspace(
            0, len(sampled_reference) - 1, num=max_snapshots, dtype=int
        )
        sampled_reference = [sampled_reference[index] for index in indices.tolist()]

    sync_threshold_ns = int(sync_threshold_ms * 1e6)
    topic_timestamps = {
        topic: [meta.timestamp_ns for meta in raw_metadata_by_topic[topic]]
        for topic in lidar_topics
    }
    sampled_metadata_by_topic: dict[str, list[PointCloudMeta]] = {
        topic: [] for topic in lidar_topics
    }
    cached_by_key: dict[tuple[str, int], dict] = {}
    synchronized_snapshots = []

    for snapshot_index, reference_meta in enumerate(sampled_reference):
        snapshot_topics = {}
        snapshot_metas: dict[str, PointCloudMeta] = {}
        valid_snapshot = True
        for topic in lidar_topics:
            meta, delta_ns = _nearest_meta(
                raw_metadata_by_topic[topic],
                topic_timestamps[topic],
                reference_meta.timestamp_ns,
                sync_threshold_ns,
            )
            if meta is None:
                valid_snapshot = False
                break
            key = (meta.topic, int(meta.timestamp_ns))
            cached_payload = cached_by_key.get(key)
            if cached_payload is None:
                topic_dir = pointcloud_cache_dir / _sanitize_topic(topic)
                file_path = topic_dir / f"{snapshot_index:05d}_{meta.timestamp_ns}.pcd"
                cached_payload = _export_cloud(
                    meta, file_path, voxel_size=export_voxel_size
                )
                cached_by_key[key] = cached_payload
            cached_meta = PointCloudMeta(
                topic=meta.topic,
                frame_id=meta.frame_id,
                timestamp_ns=int(meta.timestamp_ns),
                record_path=meta.record_path,
                artifact_path=str(cached_payload["artifact_path"]),
            )
            snapshot_metas[topic] = cached_meta
            snapshot_topics[topic] = {
                "timestamp_ns": int(cached_meta.timestamp_ns),
                "frame_id": cached_meta.frame_id,
                "record_path": cached_meta.record_path,
                "artifact_path": cached_meta.artifact_path,
                "sync_dt_ms": float((delta_ns or 0) / 1e6),
                "point_count": int(cached_payload["point_count"]),
            }
        if not valid_snapshot:
            continue
        for topic, meta in snapshot_metas.items():
            sampled_metadata_by_topic[topic].append(meta)
        synchronized_snapshots.append(
            {
                "snapshot_index": int(len(synchronized_snapshots)),
                "reference_topic": reference_topic,
                "reference_timestamp_ns": int(reference_meta.timestamp_ns),
                "topics": snapshot_topics,
            }
        )

    # Deduplicate per-topic entries in case one non-reference frame is reused.
    for topic, metas in sampled_metadata_by_topic.items():
        unique: dict[int, PointCloudMeta] = {}
        for meta in metas:
            unique[int(meta.timestamp_ns)] = meta
        sampled_metadata_by_topic[topic] = [
            unique[timestamp_ns] for timestamp_ns in sorted(unique)
        ]

    state_path = cache_dir / "state.npz"
    np.savez_compressed(
        state_path,
        pose_timestamps_ns=np.asarray(
            [sample.timestamp_ns for sample in pose_samples], dtype=np.int64
        ),
        pose_transform_world_localization=np.asarray(
            [sample.transform_world_localization for sample in pose_samples],
            dtype=float,
        ),
        pose_transform_world_imu=np.asarray(
            [sample.transform_world_imu for sample in pose_samples], dtype=float
        ),
        pose_gravity_imu=np.asarray(
            [sample.gravity_imu for sample in pose_samples], dtype=float
        ),
        imu_timestamps_ns=np.asarray(
            [sample.timestamp_ns for sample in imu_samples], dtype=np.int64
        ),
        imu_linear_acceleration=np.asarray(
            [sample.linear_acceleration for sample in imu_samples], dtype=float
        ),
        imu_angular_velocity=np.asarray(
            [sample.angular_velocity for sample in imu_samples], dtype=float
        ),
    )

    topic_info_summary = {}
    for topic in lidar_topics:
        raw_summary = _summarize_rate(raw_metadata_by_topic[topic])
        sampled_summary = _summarize_rate(sampled_metadata_by_topic[topic])
        topic_info_summary[topic] = {
            "frame_id": topic_frame_ids.get(topic, ""),
            "sensor_name": topic_sensor_name(topic),
            "count": int(len(sampled_metadata_by_topic[topic])),
            "sampled_count": int(len(sampled_metadata_by_topic[topic])),
            "original_count": int(len(raw_metadata_by_topic[topic])),
            "original_rate_hz": raw_summary["hz"],
            "sampled_rate_hz": sampled_summary["hz"],
        }

    dataset_manifest = {
        "record_files": record_files,
        "pose_topic": pose_topic,
        "imu_topic": imu_topic,
        "parent_frame": parent_frame,
        "reference_topic": reference_topic,
        "lidar_topics": lidar_topics,
        "cache_policy": {
            "sync_threshold_ms": float(sync_threshold_ms),
            "frame_stride": int(max(1, frame_stride)),
            "max_snapshots": (None if max_snapshots is None else int(max_snapshots)),
            "export_voxel_size": (
                None if export_voxel_size is None else float(export_voxel_size)
            ),
            "pointcloud_format": "pcd",
        },
        "summary": {
            "synchronized_snapshot_count": int(len(synchronized_snapshots)),
            "pose_sample_count": int(len(pose_samples)),
            "imu_sample_count": int(len(imu_samples)),
            "pointcloud_cache_count": int(
                sum(len(metas) for metas in sampled_metadata_by_topic.values())
            ),
        },
        "topics": topic_info_summary,
        "metadata_by_topic": {
            topic: [
                {
                    "timestamp_ns": int(meta.timestamp_ns),
                    "frame_id": meta.frame_id,
                    "record_path": meta.record_path,
                    "artifact_path": meta.artifact_path,
                }
                for meta in sampled_metadata_by_topic[topic]
            ]
            for topic in lidar_topics
        },
        "synchronized_snapshots": synchronized_snapshots,
        "tf_edges": [_serialize_tf_edge(edge) for edge in tf_edges],
        "artifacts": {
            "prepared_dataset": str(diagnostics_dir / "prepared_rig_dataset.yaml"),
            "state_npz": str(state_path),
            "pointcloud_cache_dir": str(pointcloud_cache_dir),
        },
    }

    dataset_path = diagnostics_dir / "prepared_rig_dataset.yaml"
    with open(dataset_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(dataset_manifest, file, sort_keys=False)

    logging.info(
        "Prepared rig dataset ready | snapshots=%d | topics=%d | path=%s",
        len(synchronized_snapshots),
        len(lidar_topics),
        dataset_path,
    )
    return dataset_path


def load_prepared_rig_dataset(dataset_yaml: str) -> PreparedRigDataset:
    dataset_path = Path(dataset_yaml)
    with open(dataset_path, "r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file) or {}
    artifacts = manifest.get("artifacts", {})
    state_path = Path(artifacts["state_npz"])
    state = np.load(state_path)

    metadata_by_topic = {
        topic: [
            PointCloudMeta(
                topic=topic,
                frame_id=str(item.get("frame_id", "")),
                timestamp_ns=int(item["timestamp_ns"]),
                record_path=str(item.get("record_path", "")),
                artifact_path=(
                    None
                    if item.get("artifact_path") is None
                    else str(item["artifact_path"])
                ),
            )
            for item in manifest.get("metadata_by_topic", {}).get(topic, [])
        ]
        for topic in manifest.get("lidar_topics", [])
    }
    tf_edges = [
        _deserialize_tf_edge(payload) for payload in manifest.get("tf_edges", [])
    ]
    pose_samples = [
        PoseSample(
            timestamp_ns=int(timestamp_ns),
            transform_world_localization=np.asarray(
                transform_world_localization, dtype=float
            ),
            transform_world_imu=np.asarray(transform_world_imu, dtype=float),
            gravity_imu=np.asarray(gravity_imu, dtype=float),
        )
        for timestamp_ns, transform_world_localization, transform_world_imu, gravity_imu in zip(
            state["pose_timestamps_ns"],
            state["pose_transform_world_localization"],
            state["pose_transform_world_imu"],
            state["pose_gravity_imu"],
        )
    ]
    imu_samples = [
        ImuSample(
            timestamp_ns=int(timestamp_ns),
            linear_acceleration=np.asarray(linear_acceleration, dtype=float),
            angular_velocity=np.asarray(angular_velocity, dtype=float),
        )
        for timestamp_ns, linear_acceleration, angular_velocity in zip(
            state["imu_timestamps_ns"],
            state["imu_linear_acceleration"],
            state["imu_angular_velocity"],
        )
    ]

    return PreparedRigDataset(
        dataset_path=str(dataset_path),
        record_files=[str(path) for path in manifest.get("record_files", [])],
        lidar_topics=[str(topic) for topic in manifest.get("lidar_topics", [])],
        topic_infos={
            str(topic): dict(info)
            for topic, info in (manifest.get("topics") or {}).items()
        },
        metadata_by_topic=metadata_by_topic,
        tf_edges=tf_edges,
        pose_topic=str(manifest.get("pose_topic", "")),
        imu_topic=str(manifest.get("imu_topic", "")),
        pose_samples=pose_samples,
        imu_samples=imu_samples,
        synchronized_snapshots=list(manifest.get("synchronized_snapshots", [])),
        manifest=manifest,
    )
