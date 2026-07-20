"""Load synchronized LiDAR frames and export manual alignment results."""

from __future__ import annotations

import copy
import bisect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as ScipyRotation

from lidar2lidar.auto_calib import load_cached_cloud
from lidar2lidar.extrinsic_io import (
    build_extrinsics_payload,
    extrinsics_filename,
    save_extrinsics_yaml,
)
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic
from lidar2lidar.pcl_registration import register_generalized_icp
from lidar2lidar.loop_closure import (
    optimize_loop_closure,
    select_loop_graph_edges,
)
from lidar2lidar.record_utils import (
    PointCloudMeta,
    build_transform_graph,
    collect_pointcloud_metadata,
    discover_record_files,
    find_synchronized_pairs,
    get_topic_frame_ids,
    infer_pointcloud_topics,
    list_topics,
    load_pointcloud_from_meta,
    load_transform_edges_from_dir,
    lookup_transform,
    topic_sensor_name,
    transform_delta_metrics,
    voxel_overlap_ratio,
)
from lidar2lidar.workflow import load_workflow_config, resolve_workflow_plan

DEFAULT_VANJEE_TOPICS = [
    "/apollo/sensor/vanjeelidar/left_front/PointCloud2",
    "/apollo/sensor/vanjeelidar/left_back/PointCloud2",
    "/apollo/sensor/vanjeelidar/right_front/PointCloud2",
    "/apollo/sensor/vanjeelidar/right_back/PointCloud2",
]

DEFAULT_MANUAL_ALIGN_WORKFLOW = (
    Path(__file__).resolve().parent / "conf" / "workflow_raw4_perimeter_loop.yaml"
)

FRAME_LABEL_ZH = {
    "left_front": "左前",
    "left_back": "左后",
    "right_front": "右前",
    "right_back": "右后",
}

SENSOR_COLORS = {
    "left_front": [0.0, 1.0, 0.0],
    "left_back": [1.0, 0.0, 0.0],
    "right_front": [1.0, 0.8, 0.0],
    "right_back": [0.0, 0.0, 1.0],
}

OVERLAP_VOXEL_SIZE = 0.04
AUTO_OVERLAP_VOXEL_SIZE = 0.5
SEARCH_DOWNSAMPLE_VOXEL = 0.10
LOW_OVERLAP_THRESHOLD = 0.05
MIN_OVERLAP_GAIN = 0.0004
MIN_GICP_FITNESS = 0.01
MAX_REGISTRATION_ROTATION_DEG = 45.0
MAX_REGISTRATION_TRANSLATION_M = 3.0
MAX_LOOP_EDGE_ROTATION_DEG = 45.0
MAX_LOOP_EDGE_TRANSLATION_M = 3.0
OVERLAP_CROP_MARGIN_M = 2.0
LOW_OVERLAP_USE_GICP = True


@dataclass
class SensorFrame:
    topic: str
    frame_id: str
    sensor_name: str
    timestamp_ns: int
    record_path: str
    point_count: int
    positions: list[float]
    color_rgb: list[float]
    fixed: bool = False
    sync_dt_ms: float = 0.0
    initial_transform: list[list[float]] = field(default_factory=list)
    current_transform: list[list[float]] = field(default_factory=list)


@dataclass
class ManualAlignSession:
    record_files: list[str]
    conf_dir: str
    target_topic: str
    target_frame: str
    sync_threshold_ms: float
    voxel_size: float
    frame_index: int
    sensors: list[SensorFrame]
    reference_timestamp_ns: int
    workflow_path: str = ""
    align_edges: list[dict] = field(default_factory=list)
    align_settings: dict = field(default_factory=dict)
    sync_mode: str = "strict"
    sync_warnings: list[str] = field(default_factory=list)
    available_frame_indices: list[int] = field(default_factory=list)
    strictly_synced_frame_indices: list[int] = field(default_factory=list)
    target_frame_count: int = 0
    _metadata_by_topic: dict = field(default_factory=dict, repr=False, compare=False)

    def to_api_dict(self) -> dict:
        return {
            "record_files": self.record_files,
            "conf_dir": self.conf_dir,
            "target_topic": self.target_topic,
            "target_frame": self.target_frame,
            "reference_timestamp_ns": self.reference_timestamp_ns,
            "sync_threshold_ms": self.sync_threshold_ms,
            "voxel_size": self.voxel_size,
            "frame_index": self.frame_index,
            "workflow_path": self.workflow_path,
            "align_edges": self.align_edges,
            "align_settings": self.align_settings,
            "sync_mode": self.sync_mode,
            "sync_warnings": self.sync_warnings,
            "target_frame_count": int(self.target_frame_count),
            "available_frame_indices": list(self.available_frame_indices),
            "strictly_synced_frame_indices": list(self.strictly_synced_frame_indices),
            "sensors": [
                {
                    "topic": sensor.topic,
                    "frame_id": sensor.frame_id,
                    "sensor_name": sensor.sensor_name,
                    "timestamp_ns": sensor.timestamp_ns,
                    "point_count": sensor.point_count,
                    "color_rgb": sensor.color_rgb,
                    "fixed": sensor.fixed,
                    "sync_dt_ms": sensor.sync_dt_ms,
                    "initial_transform": sensor.initial_transform,
                    "current_transform": sensor.current_transform,
                }
                for sensor in self.sensors
            ],
        }


def _matrix_to_rows(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).reshape(4, 4).tolist()


def _downsample_positions(
    meta,
    cloud_cache: dict,
    voxel_size: float,
) -> tuple[list[float], int]:
    cloud = load_cached_cloud(meta, cloud_cache)
    if voxel_size > 0 and len(cloud.points) > 0:
        cloud = cloud.voxel_down_sample(float(voxel_size))
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.size == 0:
        return [], 0
    return points.reshape(-1).tolist(), int(points.shape[0])


def find_first_synchronized_frame_index(
    metadata_by_topic: dict[str, list],
    target_topic: str,
    topics: list[str],
    sync_threshold_ns: int,
) -> int:
    index = try_find_first_synchronized_frame_index(
        metadata_by_topic,
        target_topic,
        topics,
        sync_threshold_ns,
    )
    if index is None:
        raise RuntimeError(
            "No synchronized frame found where all LiDAR topics align within "
            f"{sync_threshold_ns / 1e6:.1f} ms."
        )
    return index


def try_find_first_synchronized_frame_index(
    metadata_by_topic: dict[str, list],
    target_topic: str,
    topics: list[str],
    sync_threshold_ns: int,
) -> int | None:
    target_metas = metadata_by_topic.get(target_topic, [])
    for index, reference_meta in enumerate(target_metas):
        ok = True
        for topic in topics:
            if topic == target_topic:
                continue
            matches = find_synchronized_pairs(
                metadata_by_topic[topic],
                [reference_meta],
                sync_threshold_ns,
                max_pairs=1,
            )
            if not matches:
                ok = False
                break
        if ok:
            return index
    return None


def _frame_has_strict_sync(
    metadata_by_topic: dict[str, list],
    target_topic: str,
    topics: list[str],
    sync_threshold_ns: int,
    frame_index: int,
) -> bool:
    target_metas = metadata_by_topic.get(target_topic, [])
    if frame_index < 0 or frame_index >= len(target_metas):
        return False
    reference_meta = target_metas[frame_index]
    for topic in topics:
        if topic == target_topic:
            continue
        matches = find_synchronized_pairs(
            metadata_by_topic[topic],
            [reference_meta],
            sync_threshold_ns,
            max_pairs=1,
        )
        if not matches:
            return False
    return True


def _find_nearest_meta_to_timestamp(
    metas: list,
    reference_timestamp_ns: int,
) -> tuple[object, int]:
    if not metas:
        raise RuntimeError("No point cloud frames available.")
    timestamps = [int(meta.timestamp_ns) for meta in metas]
    index = bisect.bisect_left(timestamps, int(reference_timestamp_ns))
    candidate_indices = []
    if index < len(metas):
        candidate_indices.append(index)
    if index > 0:
        candidate_indices.append(index - 1)
    best_index = min(
        candidate_indices,
        key=lambda candidate: abs(timestamps[candidate] - int(reference_timestamp_ns)),
    )
    meta = metas[best_index]
    delta_ns = abs(int(meta.timestamp_ns) - int(reference_timestamp_ns))
    return meta, delta_ns


def _resolve_frame_index(
    metadata_by_topic: dict[str, list],
    target_topic: str,
    topics: list[str],
    sync_threshold_ns: int,
    sync_threshold_ms: float,
    requested_index: int,
) -> tuple[int, str, list[str]]:
    target_metas = metadata_by_topic.get(target_topic, [])
    if not target_metas:
        raise RuntimeError(f"Target topic has no frames: {target_topic}")

    frame_index = int(requested_index)
    if frame_index < 0 or frame_index >= len(target_metas):
        frame_index = 0

    if _frame_has_strict_sync(
        metadata_by_topic,
        target_topic,
        topics,
        sync_threshold_ns,
        frame_index,
    ):
        return frame_index, "strict", []

    return (
        frame_index,
        "nearest_fallback",
        [
            f"帧 {frame_index} 在 {sync_threshold_ms:.1f} ms 内未严格同步，"
            "其余雷达已匹配最近时间戳帧。"
        ],
    )


def probe_manual_align_frames(
    *,
    record_path: str,
    conf_dir: str | None = None,
    target_topic: str | None = None,
    topics: list[str] | None = None,
    sync_threshold_ms: float = 50.0,
    workflow_yaml: str | None = None,
) -> dict:
    record_files = discover_record_files(record_path)
    workflow_plan, selected_topics, target_topic, workflow_path = _resolve_manual_align_workflow(
        record_files=record_files,
        workflow_yaml=workflow_yaml,
        conf_dir=conf_dir,
        target_topic=target_topic,
        topics=topics,
    )
    metadata_by_topic = collect_pointcloud_metadata(record_files, selected_topics)
    target_metas = metadata_by_topic.get(target_topic, [])
    if not target_metas:
        raise RuntimeError(f"Target topic has no frames: {target_topic}")

    sync_threshold_ns = int(sync_threshold_ms * 1e6)
    available_frames: list[dict] = []
    strictly_synced_indices: list[int] = []
    for index, meta in enumerate(target_metas):
        strict_sync = _frame_has_strict_sync(
            metadata_by_topic,
            target_topic,
            selected_topics,
            sync_threshold_ns,
            index,
        )
        if strict_sync:
            strictly_synced_indices.append(index)
        available_frames.append(
            {
                "index": int(index),
                "timestamp_ns": int(meta.timestamp_ns),
                "strict_sync": bool(strict_sync),
            }
        )

    default_frame_index = (
        strictly_synced_indices[0] if strictly_synced_indices else 0
    )
    return {
        "record_files": record_files,
        "conf_dir": str(conf_dir or ""),
        "target_topic": target_topic,
        "target_frame_count": len(target_metas),
        "available_frame_indices": [frame["index"] for frame in available_frames],
        "strictly_synced_frame_indices": strictly_synced_indices,
        "available_frames": available_frames,
        "default_frame_index": int(default_frame_index),
        "sync_threshold_ms": float(sync_threshold_ms),
        "workflow_path": workflow_path,
    }


def _resolve_sensor_meta_for_reference(
    *,
    topic: str,
    target_topic: str,
    reference_meta,
    metadata_by_topic: dict[str, list],
    sync_threshold_ns: int,
    sync_threshold_ms: float,
    sync_mode: str,
) -> tuple[object, float, str | None]:
    if topic == target_topic:
        return reference_meta, 0.0, None

    topic_metas = metadata_by_topic.get(topic, [])
    if not topic_metas:
        raise RuntimeError(f"Topic has no frames: {topic}")

    matches = find_synchronized_pairs(
        topic_metas,
        [reference_meta],
        sync_threshold_ns,
        max_pairs=1,
    )
    if matches:
        meta, _, delta_ns = matches[0]
        return meta, float(delta_ns / 1e6), None

    meta, delta_ns = _find_nearest_meta_to_timestamp(
        topic_metas,
        int(reference_meta.timestamp_ns),
    )
    sync_dt_ms = float(delta_ns / 1e6)
    if sync_mode != "nearest_fallback":
        return meta, sync_dt_ms, None

    sensor_name = topic_sensor_name(topic)
    if sync_dt_ms <= sync_threshold_ms:
        return meta, sync_dt_ms, None

    return (
        meta,
        sync_dt_ms,
        f"{sensor_name}: 最近帧偏差 {sync_dt_ms:.1f} ms（阈值 {sync_threshold_ms:.1f} ms）",
    )


def _edge_label_zh(source_frame: str, target_frame: str) -> str:
    source_label = FRAME_LABEL_ZH.get(source_frame, source_frame)
    target_label = FRAME_LABEL_ZH.get(target_frame, target_frame)
    return f"{source_label}→{target_label}"


def _build_align_edges_from_plan(workflow_plan: dict) -> list[dict]:
    edges: list[dict] = []
    for relation in workflow_plan["relations"]:
        if relation.get("role") == "supporting" and not relation.get("required"):
            continue
        edges.append(
            {
                "relation_id": relation["relation_id"],
                "label": _edge_label_zh(relation["source_frame"], relation["target_frame"]),
                "source_frame": relation["source_frame"],
                "target_frame": relation["target_frame"],
                "source_topic": relation["source_topic"],
                "target_topic": relation["target_topic"],
            }
        )
    return edges


def _align_settings_from_plan(
    workflow_plan: dict,
    *,
    preview_voxel_size: float | None,
) -> dict:
    scene = workflow_plan.get("scene_sufficiency") or {}
    visualization = workflow_plan.get("visualization") or {}
    preview = float(
        preview_voxel_size
        if preview_voxel_size is not None
        else visualization.get("downsample_voxel_size", 0.10)
    )
    registration_voxel = min(0.04, max(0.02, preview * 0.4))
    return {
        "min_overlap_ratio": float(scene.get("min_overlap_ratio", 0.07)),
        "registration_voxel_size": registration_voxel,
        "preview_voxel_size": preview,
        "search_downsample_voxel": float(
            visualization.get("downsample_voxel_size", SEARCH_DOWNSAMPLE_VOXEL)
        ),
        "plane_distance_threshold": float(
            visualization.get("plane_distance_threshold", 0.05)
        ),
        "dynamic_distance_threshold_m": float(
            scene.get("dynamic_distance_threshold_m", 0.40)
        ),
        "max_windows_per_edge": 1,
        "overlap_voxel_size": AUTO_OVERLAP_VOXEL_SIZE,
        "enable_loop_closure": bool(workflow_plan.get("enable_global_optimization", True)),
    }


def _resolve_manual_align_workflow(
    *,
    record_files: list[str],
    workflow_yaml: str | None,
    conf_dir: str | None,
    target_topic: str | None,
    topics: list[str] | None,
) -> tuple[dict, list[str], str, str]:
    workflow_path = (
        str(Path(workflow_yaml).expanduser().resolve())
        if workflow_yaml
        else str(DEFAULT_MANUAL_ALIGN_WORKFLOW)
    )
    topic_counts = list_topics(record_files)
    pointcloud_topics = infer_pointcloud_topics(topic_counts)
    candidate_topics = list(
        dict.fromkeys(
            [
                *pointcloud_topics,
                *DEFAULT_VANJEE_TOPICS,
                *(topics or []),
                *topic_counts.keys(),
            ]
        )
    )
    topic_frame_ids = get_topic_frame_ids(record_files, candidate_topics)
    topic_infos = {
        topic: {
            "frame_id": topic_frame_ids.get(topic, topic_sensor_name(topic)),
            "topic": topic,
        }
        for topic in candidate_topics
        if topic in topic_counts
    }
    default_target = target_topic
    if default_target is None:
        for candidate in DEFAULT_VANJEE_TOPICS:
            if candidate in topic_infos:
                default_target = candidate
                break
        default_target = default_target or next(iter(topic_infos), DEFAULT_VANJEE_TOPICS[0])

    workflow_config = load_workflow_config(workflow_path)
    plan = resolve_workflow_plan(
        workflow_config=workflow_config,
        workflow_path=workflow_path,
        pointcloud_topics=[topic for topic in pointcloud_topics if topic in topic_infos],
        topic_infos=topic_infos,
        tf_edges=load_transform_edges_from_dir(conf_dir) or [],
        default_target_topic=default_target,
        cli_source_topics=topics,
        default_min_overlap=0.07,
        default_enable_global_optimization=True,
        default_save_visuals=False,
    )

    resolved_target = target_topic or plan["target_topic"]
    if topics:
        selected_topics = list(dict.fromkeys(topics))
        if resolved_target not in selected_topics:
            selected_topics.insert(0, resolved_target)
    else:
        selected_topics = list(plan["selected_topics"])
        if resolved_target != plan["target_topic"]:
            selected_topics = [
                resolved_target,
                *[topic for topic in selected_topics if topic != resolved_target],
            ]

    selected_topics = [topic for topic in selected_topics if topic in topic_counts]
    if not selected_topics:
        raise RuntimeError("No PointCloud2 topics selected from workflow.")
    if resolved_target not in selected_topics:
        selected_topics.insert(0, resolved_target)
    return plan, selected_topics, resolved_target, workflow_path


def _session_edge_map(session: ManualAlignSession) -> dict[str, dict]:
    return {edge["relation_id"]: edge for edge in session.align_edges}


def load_manual_align_session(
    *,
    record_path: str,
    conf_dir: str | None = None,
    target_topic: str | None = None,
    topics: list[str] | None = None,
    sync_threshold_ms: float = 50.0,
    voxel_size: float | None = None,
    frame_index: int = 0,
    workflow_yaml: str | None = None,
) -> ManualAlignSession:
    record_files = discover_record_files(record_path)
    workflow_plan, selected_topics, target_topic, workflow_path = _resolve_manual_align_workflow(
        record_files=record_files,
        workflow_yaml=workflow_yaml,
        conf_dir=conf_dir,
        target_topic=target_topic,
        topics=topics,
    )
    align_settings = _align_settings_from_plan(
        workflow_plan,
        preview_voxel_size=voxel_size,
    )
    align_edges = _build_align_edges_from_plan(workflow_plan)
    if not align_edges:
        raise RuntimeError("Workflow did not define any manual-align edges.")
    effective_voxel_size = float(align_settings["preview_voxel_size"])

    topic_frame_ids = get_topic_frame_ids(record_files, selected_topics)
    target_frame = topic_frame_ids.get(target_topic, topic_sensor_name(target_topic))
    metadata_by_topic = collect_pointcloud_metadata(record_files, selected_topics)
    if not metadata_by_topic.get(target_topic):
        raise RuntimeError(f"Target topic has no frames: {target_topic}")

    sync_threshold_ns = int(sync_threshold_ms * 1e6)
    target_metas = metadata_by_topic[target_topic]
    if not target_metas:
        raise RuntimeError(f"Target topic has no frames: {target_topic}")

    available_frame_indices = list(range(len(target_metas)))
    strictly_synced_frame_indices = [
        index
        for index in available_frame_indices
        if _frame_has_strict_sync(
            metadata_by_topic,
            target_topic,
            selected_topics,
            sync_threshold_ns,
            index,
        )
    ]

    frame_index, sync_mode, sync_warnings = _resolve_frame_index(
        metadata_by_topic,
        target_topic,
        selected_topics,
        sync_threshold_ns,
        float(sync_threshold_ms),
        int(frame_index),
    )
    reference_meta = target_metas[frame_index]

    tf_graph = build_transform_graph(load_transform_edges_from_dir(conf_dir))
    cloud_cache: dict = {}
    sensors: list[SensorFrame] = []

    for topic in selected_topics:
        frame_id = topic_frame_ids.get(topic, topic_sensor_name(topic))
        sensor_name = topic_sensor_name(topic)
        is_target = topic == target_topic

        meta, sync_dt_ms, warning = _resolve_sensor_meta_for_reference(
            topic=topic,
            target_topic=target_topic,
            reference_meta=reference_meta,
            metadata_by_topic=metadata_by_topic,
            sync_threshold_ns=sync_threshold_ns,
            sync_threshold_ms=float(sync_threshold_ms),
            sync_mode=sync_mode,
        )
        if warning:
            sync_warnings.append(warning)

        positions, point_count = _downsample_positions(meta, cloud_cache, effective_voxel_size)
        if point_count == 0:
            raise RuntimeError(f"Point cloud is empty for topic {topic}.")

        if is_target:
            transform = np.eye(4, dtype=float)
        else:
            seed = lookup_transform(tf_graph, frame_id, target_frame)
            transform = (
                np.asarray(seed, dtype=float)
                if seed is not None
                else np.eye(4, dtype=float)
            )

        color = SENSOR_COLORS.get(frame_id, [0.7, 0.7, 0.7])
        sensors.append(
            SensorFrame(
                topic=topic,
                frame_id=frame_id,
                sensor_name=sensor_name,
                timestamp_ns=int(meta.timestamp_ns),
                record_path=str(meta.record_path),
                point_count=point_count,
                positions=positions,
                color_rgb=[float(v) for v in color],
                fixed=is_target,
                sync_dt_ms=float(sync_dt_ms),
                initial_transform=_matrix_to_rows(transform),
                current_transform=_matrix_to_rows(transform),
            )
        )

    return ManualAlignSession(
        record_files=record_files,
        conf_dir=str(conf_dir or ""),
        target_topic=target_topic,
        target_frame=target_frame,
        sync_threshold_ms=float(sync_threshold_ms),
        voxel_size=effective_voxel_size,
        frame_index=frame_index,
        sensors=sensors,
        reference_timestamp_ns=int(reference_meta.timestamp_ns),
        workflow_path=workflow_path,
        align_edges=align_edges,
        align_settings=align_settings,
        sync_mode=sync_mode,
        sync_warnings=sync_warnings,
        available_frame_indices=available_frame_indices,
        strictly_synced_frame_indices=strictly_synced_frame_indices,
        target_frame_count=len(target_metas),
        _metadata_by_topic=metadata_by_topic,
    )


def _sensor_meta(sensor: SensorFrame) -> PointCloudMeta:
    return PointCloudMeta(
        topic=sensor.topic,
        frame_id=sensor.frame_id,
        timestamp_ns=int(sensor.timestamp_ns),
        record_path=sensor.record_path,
    )


def _sensor_lookup(session: ManualAlignSession) -> dict[str, SensorFrame]:
    return {sensor.frame_id: sensor for sensor in session.sensors}


def _seed_transform_to_target(
    frame_id: str,
    target_frame: str,
    sensors_by_frame: dict[str, SensorFrame],
    transforms_to_target: dict[str, np.ndarray],
    seed_transforms: dict[str, list[list[float]]] | None,
) -> np.ndarray:
    if frame_id in transforms_to_target:
        return np.asarray(transforms_to_target[frame_id], dtype=float)
    if seed_transforms and frame_id in seed_transforms:
        return np.asarray(seed_transforms[frame_id], dtype=float).reshape(4, 4)
    sensor = sensors_by_frame[frame_id]
    rows = sensor.current_transform or sensor.initial_transform
    return np.asarray(rows, dtype=float).reshape(4, 4)


def _resolve_default_topics(topic_counts: dict[str, int]) -> list[str]:
    vanjee = [topic for topic in DEFAULT_VANJEE_TOPICS if topic in topic_counts]
    if len(vanjee) >= 2:
        return vanjee
    return infer_pointcloud_topics(topic_counts)


def _downsample_cloud(cloud: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size <= 0 or len(cloud.points) == 0:
        return copy.deepcopy(cloud)
    return cloud.voxel_down_sample(float(voxel_size))


def _pairwise_overlap(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    source_to_target: np.ndarray,
    *,
    voxel_size: float = OVERLAP_VOXEL_SIZE,
) -> float:
    return float(
        voxel_overlap_ratio(
            source_cloud,
            target_cloud,
            np.asarray(source_to_target, dtype=float),
            float(voxel_size),
        )
    )


def _compose_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return matrix


def _crop_to_overlap_region(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    source_to_target: np.ndarray,
    *,
    margin: float = OVERLAP_CROP_MARGIN_M,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    source_copy = copy.deepcopy(source_cloud)
    source_copy.transform(np.asarray(source_to_target, dtype=float))
    source_points = np.asarray(source_copy.points)
    target_points = np.asarray(target_cloud.points)
    if source_points.size == 0 or target_points.size == 0:
        return source_cloud, target_cloud

    source_min = source_points.min(axis=0)
    source_max = source_points.max(axis=0)
    target_min = target_points.min(axis=0)
    target_max = target_points.max(axis=0)
    lower = np.maximum(source_min, target_min) - float(margin)
    upper = np.minimum(source_max, target_max) + float(margin)
    if np.any(lower >= upper):
        return source_cloud, target_cloud

    def _crop_native(cloud: o3d.geometry.PointCloud, transform: np.ndarray | None) -> o3d.geometry.PointCloud:
        cropped = copy.deepcopy(cloud)
        if transform is not None:
            cropped.transform(transform)
        points = np.asarray(cropped.points)
        if points.size == 0:
            return cloud
        mask = np.all((points >= lower) & (points <= upper), axis=1)
        indices = np.where(mask)[0]
        if indices.size == 0:
            return cloud
        cropped = cropped.select_by_index(indices.tolist())
        if transform is not None:
            cropped.transform(np.linalg.inv(transform))
        return cropped

    return (
        _crop_native(source_cloud, np.asarray(source_to_target, dtype=float)),
        _crop_native(target_cloud, None),
    )


def _local_rigid_search(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
) -> np.ndarray:
    """Search small translation/yaw adjustments around the seed transform."""
    initial = np.asarray(initial_transform, dtype=float)
    source_eval = _downsample_cloud(source_cloud, SEARCH_DOWNSAMPLE_VOXEL)
    target_eval = _downsample_cloud(target_cloud, SEARCH_DOWNSAMPLE_VOXEL)
    rotation_seed = initial[:3, :3]
    translation_seed = initial[:3, 3]
    best_overlap = _pairwise_overlap(
        source_eval,
        target_eval,
        initial,
        voxel_size=SEARCH_DOWNSAMPLE_VOXEL,
    )
    best_transform = initial.copy()

    for delta_x in np.linspace(-0.6, 0.6, 7):
        for delta_y in np.linspace(-0.6, 0.6, 7):
            for delta_z in np.linspace(-0.12, 0.12, 3):
                candidate = _compose_transform(
                    rotation_seed,
                    translation_seed + np.array([delta_x, delta_y, delta_z], dtype=float),
                )
                overlap = _pairwise_overlap(
                    source_eval,
                    target_eval,
                    candidate,
                    voxel_size=SEARCH_DOWNSAMPLE_VOXEL,
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_transform = candidate

    rotation_best = best_transform[:3, :3]
    translation_best = best_transform[:3, 3]
    for yaw_deg in np.linspace(-6.0, 6.0, 7):
        rotation = ScipyRotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ rotation_seed
        candidate = _compose_transform(rotation, translation_best)
        overlap = _pairwise_overlap(
            source_eval,
            target_eval,
            candidate,
            voxel_size=SEARCH_DOWNSAMPLE_VOXEL,
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_transform = candidate

    return best_transform


def _register_with_initial(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    initial_transform: np.ndarray | None,
    preprocessing_params: dict,
    method: int,
) -> tuple[np.ndarray | None, object | None]:
    if int(method) == 2:
        voxel_size = float(preprocessing_params.get("voxel_size", 0.04))
        max_correspondence_distance = max(voxel_size * 5, 0.02)
        transform, pcl_result = register_generalized_icp(
            source_cloud,
            target_cloud,
            initial_transform=initial_transform,
            preprocessing_params=preprocessing_params,
            max_correspondence_distance=max_correspondence_distance,
        )
        if transform is not None and pcl_result is not None:

            class _PclRegistrationResult:
                fitness = pcl_result.fitness
                inlier_rmse = pcl_result.inlier_rmse

            return np.asarray(transform, dtype=float), _PclRegistrationResult()

        logging.warning("PCL GICP failed, falling back to Open3D registration.")

    final_transform, _, reg_result = calibrate_lidar_extrinsic(
        source_cloud,
        target_cloud,
        is_draw_registration=False,
        preprocessing_params=preprocessing_params,
        method=int(method),
        initial_transform=initial_transform,
    )
    if final_transform is None or reg_result is None:
        return None, None
    return np.asarray(final_transform, dtype=float), reg_result


def _collect_registration_candidates(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    preprocessing_params: dict,
    method: int,
    *,
    settings: dict | None = None,
) -> list[dict]:
    seed_overlap = _pairwise_overlap(source_cloud, target_cloud, initial_transform)
    candidates: list[dict] = [
        {
            "name": "seed",
            "transform": np.asarray(initial_transform, dtype=float),
            "overlap_ratio": seed_overlap,
            "fitness": None,
            "inlier_rmse": None,
        }
    ]

    local_transform = _local_rigid_search(source_cloud, target_cloud, initial_transform)
    local_overlap = _pairwise_overlap(source_cloud, target_cloud, local_transform)
    candidates.append(
        {
            "name": "local_search",
            "transform": local_transform,
            "overlap_ratio": local_overlap,
            "fitness": None,
            "inlier_rmse": None,
        }
    )

    cropped_source, cropped_target = _crop_to_overlap_region(
        source_cloud,
        target_cloud,
        initial_transform,
    )
    gicp_initial = local_transform if seed_overlap < LOW_OVERLAP_THRESHOLD else initial_transform
    transform, reg_result = _register_with_initial(
        cropped_source,
        cropped_target,
        gicp_initial,
        preprocessing_params,
        method,
    )
    if transform is not None:
        candidates.append(
            {
                "name": (
                    "pcl_local_gicp"
                    if gicp_initial is local_transform
                    else "pcl_seed_gicp"
                ),
                "transform": transform,
                "overlap_ratio": _pairwise_overlap(source_cloud, target_cloud, transform),
                "fitness": float(reg_result.fitness),
                "inlier_rmse": float(reg_result.inlier_rmse),
            }
        )

    if seed_overlap < float((settings or {}).get("min_overlap_ratio", 0.15)):
        transform, reg_result = _register_with_initial(
            cropped_source,
            cropped_target,
            None,
            preprocessing_params,
            method,
        )
        if transform is not None:
            candidates.append(
                {
                    "name": "pcl_feature_gicp",
                    "transform": transform,
                    "overlap_ratio": _pairwise_overlap(source_cloud, target_cloud, transform),
                    "fitness": float(reg_result.fitness),
                    "inlier_rmse": float(reg_result.inlier_rmse),
                }
            )

    return candidates


def _registration_preprocessing_params(
    registration_voxel_size: float,
    *,
    plane_dist_thresh: float = 0.05,
) -> dict:
    return {
        "voxel_size": float(registration_voxel_size),
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": float(plane_dist_thresh),
        "height_range": None,
        "remove_ground": False,
        "remove_walls": False,
    }


def _compute_pairwise_initial_transform(
    *,
    source_frame: str,
    relation_target_frame: str,
    target_frame: str,
    sensors_by_frame: dict[str, SensorFrame],
    transforms_to_target: dict[str, np.ndarray],
    seed_transforms: dict[str, list[list[float]]] | None,
) -> np.ndarray:
    if relation_target_frame == target_frame:
        return _seed_transform_to_target(
            source_frame,
            target_frame,
            sensors_by_frame,
            transforms_to_target,
            seed_transforms,
        )
    target_to_ref = _seed_transform_to_target(
        relation_target_frame,
        target_frame,
        sensors_by_frame,
        transforms_to_target,
        seed_transforms,
    )
    source_to_ref = _seed_transform_to_target(
        source_frame,
        target_frame,
        sensors_by_frame,
        transforms_to_target,
        seed_transforms,
    )
    return np.linalg.inv(target_to_ref) @ source_to_ref


def _accept_registration(
    *,
    seed_overlap: float,
    best: dict,
    final_transform: np.ndarray,
    initial_transform: np.ndarray,
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    settings: dict,
) -> bool:
    delta = transform_delta_metrics(initial_transform, final_transform)
    max_rotation_deg = float(
        settings.get("max_registration_rotation_deg", MAX_REGISTRATION_ROTATION_DEG)
    )
    max_translation_m = float(
        settings.get("max_registration_translation_m", MAX_REGISTRATION_TRANSLATION_M)
    )
    if seed_overlap < LOW_OVERLAP_THRESHOLD and (
        float(delta["rotation_deg"]) > max_rotation_deg
        or float(delta["translation_norm_m"]) > max_translation_m
    ):
        return False
    if best["name"] in {"pcl_feature_gicp", "feature_gicp"} and (
        float(delta["rotation_deg"]) > max_rotation_deg * 0.6
        or float(delta["translation_norm_m"]) > max_translation_m * 0.6
    ):
        return False

    overlap_gain = float(best["overlap_ratio"]) - float(seed_overlap)
    min_gain = (
        MIN_OVERLAP_GAIN
        if seed_overlap >= LOW_OVERLAP_THRESHOLD
        else MIN_OVERLAP_GAIN * 0.25
    )
    if best["name"] in {"pcl_seed_gicp", "pcl_local_gicp", "pcl_feature_gicp", "seed_gicp", "local_gicp", "feature_gicp"}:
        fitness = float(best.get("fitness") or 0.0)
        if fitness >= MIN_GICP_FITNESS:
            return True
        coarse_overlap = _pairwise_overlap(
            source_cloud,
            target_cloud,
            final_transform,
            voxel_size=float(settings.get("overlap_voxel_size", AUTO_OVERLAP_VOXEL_SIZE)),
        )
        min_overlap = float(settings.get("min_overlap_ratio", 0.15))
        if coarse_overlap >= min_overlap * 0.5:
            return True
    return overlap_gain >= min_gain


def _pick_best_candidate(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda item: (
            float(item["overlap_ratio"]),
            float(item.get("fitness") or 0.0),
        ),
    )


def _align_cloud_pair(
    *,
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    preprocessing_params: dict,
    method: int,
    settings: dict,
) -> dict:
    candidates = _collect_registration_candidates(
        source_cloud,
        target_cloud,
        initial_transform,
        preprocessing_params,
        int(method),
        settings=settings,
    )
    seed_overlap = float(candidates[0]["overlap_ratio"])
    best = _pick_best_candidate(candidates)
    final_transform = np.asarray(best["transform"], dtype=float)
    accepted = _accept_registration(
        seed_overlap=seed_overlap,
        best=best,
        final_transform=final_transform,
        initial_transform=initial_transform,
        source_cloud=source_cloud,
        target_cloud=target_cloud,
        settings=settings,
    )
    attempted_delta = transform_delta_metrics(
        initial_transform,
        np.asarray(best["transform"], dtype=float),
    )
    max_rotation_deg = float(
        settings.get("max_registration_rotation_deg", MAX_REGISTRATION_ROTATION_DEG)
    )
    max_translation_m = float(
        settings.get("max_registration_translation_m", MAX_REGISTRATION_TRANSLATION_M)
    )
    if not accepted:
        final_transform = np.asarray(initial_transform, dtype=float)
        best = candidates[0]
    delta = transform_delta_metrics(initial_transform, final_transform)
    coarse_overlap = _pairwise_overlap(
        source_cloud,
        target_cloud,
        final_transform,
        voxel_size=float(settings.get("overlap_voxel_size", AUTO_OVERLAP_VOXEL_SIZE)),
    )
    return {
        "success": bool(accepted),
        "reason": None
        if accepted
        else (
            "excessive_registration_delta"
            if (
                float(attempted_delta["rotation_deg"]) > max_rotation_deg
                or float(attempted_delta["translation_norm_m"]) > max_translation_m
            )
            else "insufficient_overlap_improvement"
        ),
        "attempt": str(best["name"]),
        "seed_overlap_ratio": float(seed_overlap),
        "overlap_ratio": float(best["overlap_ratio"]),
        "coarse_overlap_ratio": float(coarse_overlap),
        "fitness": None if best.get("fitness") is None else float(best["fitness"]),
        "inlier_rmse": None if best.get("inlier_rmse") is None else float(best["inlier_rmse"]),
        "final_transform": final_transform,
        "delta_translation_m": float(delta["translation_norm_m"]),
        "delta_rotation_deg": float(delta["rotation_deg"]),
    }


def _edge_is_usable_for_loop_closure(edge: dict) -> bool:
    if not edge.get("success"):
        return False
    if float(edge.get("delta_rotation_deg") or 0.0) > MAX_LOOP_EDGE_ROTATION_DEG:
        return False
    if float(edge.get("delta_translation_m") or 0.0) > MAX_LOOP_EDGE_TRANSLATION_M:
        return False
    seed_overlap = float(edge.get("seed_overlap_ratio") or 0.0)
    overlap_ratio = float(edge.get("overlap_ratio") or 0.0)
    if seed_overlap < 0.01 and overlap_ratio < 0.05:
        return False
    return True


def _apply_loop_closure_refinement(
    session: ManualAlignSession,
    edge_results: list[dict],
    transforms_to_target: dict[str, np.ndarray],
) -> dict | None:
    settings = session.align_settings or {}
    graph_edges = []
    skipped_edges = []
    for edge in edge_results:
        if not _edge_is_usable_for_loop_closure(edge):
            if edge.get("success"):
                skipped_edges.append(
                    {
                        "relation_id": edge.get("relation_id"),
                        "source_frame": edge.get("source_frame"),
                        "target_frame": edge.get("target_frame"),
                        "delta_rotation_deg": float(edge.get("delta_rotation_deg") or 0.0),
                        "delta_translation_m": float(edge.get("delta_translation_m") or 0.0),
                        "seed_overlap_ratio": float(edge.get("seed_overlap_ratio") or 0.0),
                        "overlap_ratio": float(edge.get("overlap_ratio") or 0.0),
                    }
                )
            continue
        graph_edges.append(
            {
                "source_topic": edge["source_topic"],
                "target_topic": edge["target_topic"],
                "source_frame": edge["source_frame"],
                "target_frame": edge["target_frame"],
                "relation_id": edge["relation_id"],
                "overlap_ratio": float(edge.get("overlap_ratio") or 0.0),
                "best_run": {
                    "transformation": edge["pairwise_transform"],
                    "fitness": float(edge.get("fitness") or 0.0),
                    "inlier_rmse": float(edge.get("inlier_rmse") or 0.0),
                },
            }
        )
    if len(graph_edges) < 2:
        return {
            "success": False,
            "reason": "insufficient_loop_edges",
            "skipped_edges": skipped_edges,
        }

    topic_to_frame = {sensor.topic: sensor.frame_id for sensor in session.sensors}
    initial_topic_transforms = {session.target_topic: np.eye(4, dtype=float)}
    for sensor in session.sensors:
        if sensor.fixed:
            continue
        initial_topic_transforms[sensor.topic] = np.asarray(
            transforms_to_target[sensor.frame_id],
            dtype=float,
        )

    # Anchor loop closure to the edge-aligned transforms, not conf extrinsics.
    prior_topic_transforms = copy.deepcopy(initial_topic_transforms)
    graph_selection = select_loop_graph_edges(
        graph_edges,
        session.target_topic,
        required_topics=[sensor.topic for sensor in session.sensors],
    )
    loop_result = optimize_loop_closure(
        session.target_topic,
        initial_topic_transforms,
        graph_selection["graph_edges"],
        prior_topic_transforms=prior_topic_transforms,
        prior_translation_weight=float(settings.get("loop_prior_translation_weight", 0.15)),
        prior_rotation_weight=float(settings.get("loop_prior_rotation_weight", 0.5)),
    )
    if skipped_edges:
        loop_result["skipped_edges"] = skipped_edges
    if not loop_result.get("success"):
        return loop_result

    for topic, transform in loop_result["optimized_topic_transforms"].items():
        frame_id = topic_to_frame.get(topic)
        if frame_id and frame_id != session.target_frame:
            transforms_to_target[frame_id] = np.asarray(transform, dtype=float)
    return loop_result


def _initialize_transforms_to_target(
    session: ManualAlignSession,
    seed_transforms: dict[str, list[list[float]]] | None,
) -> dict[str, np.ndarray]:
    sensors_by_frame = _sensor_lookup(session)
    target_frame = session.target_frame
    transforms_to_target: dict[str, np.ndarray] = {
        target_frame: np.eye(4, dtype=float)
    }
    for sensor in session.sensors:
        if sensor.fixed:
            continue
        transforms_to_target[sensor.frame_id] = _seed_transform_to_target(
            sensor.frame_id,
            target_frame,
            sensors_by_frame,
            transforms_to_target,
            seed_transforms,
        )
    return transforms_to_target


def _session_transforms_from_target_map(
    session: ManualAlignSession,
    transforms_to_target: dict[str, np.ndarray],
    sensors_by_frame: dict[str, SensorFrame],
    seed_transforms: dict[str, list[list[float]]] | None,
) -> dict[str, list[list[float]]]:
    target_frame = session.target_frame
    transforms: dict[str, list[list[float]]] = {}
    for sensor in session.sensors:
        if sensor.fixed:
            sensor.current_transform = _matrix_to_rows(np.eye(4, dtype=float))
            transforms[sensor.frame_id] = sensor.current_transform
            continue
        matrix = transforms_to_target.get(
            sensor.frame_id,
            _seed_transform_to_target(
                sensor.frame_id,
                target_frame,
                sensors_by_frame,
                transforms_to_target,
                seed_transforms,
            ),
        )
        sensor.current_transform = _matrix_to_rows(matrix)
        transforms[sensor.frame_id] = sensor.current_transform
    return transforms


def _summarize_loop_closure_report(report: dict | None) -> dict | None:
    if not report:
        return None
    return {
        "success": bool(report.get("success")),
        "message": str(report.get("message", "")),
        "cost": float(report.get("cost", 0.0)),
        "nfev": int(report.get("nfev", 0)),
    }


def _run_edge_alignment(
    edge: dict[str, str],
    *,
    session: ManualAlignSession,
    sensors_by_frame: dict[str, SensorFrame],
    target_frame: str,
    transforms_to_target: dict[str, np.ndarray],
    seed_transforms: dict[str, list[list[float]]] | None,
    preprocessing_params: dict,
    method: int,
    settings: dict,
    max_windows: int | None = None,
) -> dict:
    del max_windows  # auto-align always uses the currently loaded synchronized frame
    source_frame = edge["source_frame"]
    relation_target_frame = edge["target_frame"]
    source_topic = edge.get("source_topic") or sensors_by_frame[source_frame].topic
    target_topic = edge.get("target_topic") or sensors_by_frame[relation_target_frame].topic
    if source_frame not in sensors_by_frame:
        return {**edge, "success": False, "reason": "source_frame_missing"}
    if relation_target_frame not in sensors_by_frame:
        return {**edge, "success": False, "reason": "target_frame_missing"}

    source_sensor = sensors_by_frame[source_frame]
    target_sensor = sensors_by_frame[relation_target_frame]
    source_cloud = load_pointcloud_from_meta(_sensor_meta(source_sensor))
    target_cloud = load_pointcloud_from_meta(_sensor_meta(target_sensor))
    if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
        return {
            **edge,
            "source_topic": source_topic,
            "target_topic": target_topic,
            "source_frame": source_frame,
            "target_frame": relation_target_frame,
            "success": False,
            "reason": "empty_point_cloud",
        }

    initial_transform = _compute_pairwise_initial_transform(
        source_frame=source_frame,
        relation_target_frame=relation_target_frame,
        target_frame=target_frame,
        sensors_by_frame=sensors_by_frame,
        transforms_to_target=transforms_to_target,
        seed_transforms=seed_transforms,
    )
    alignment = _align_cloud_pair(
        source_cloud=source_cloud,
        target_cloud=target_cloud,
        initial_transform=initial_transform,
        preprocessing_params=preprocessing_params,
        method=int(method),
        settings=settings,
    )
    final_transform = np.asarray(alignment["final_transform"], dtype=float)
    if alignment["success"]:
        if relation_target_frame == target_frame:
            transforms_to_target[source_frame] = final_transform
        else:
            target_to_ref = transforms_to_target[relation_target_frame]
            transforms_to_target[source_frame] = target_to_ref @ final_transform

    return {
        **edge,
        "source_topic": source_topic,
        "target_topic": target_topic,
        "source_frame": source_frame,
        "target_frame": relation_target_frame,
        "output_frame": target_frame,
        "success": bool(alignment["success"]),
        "reason": alignment.get("reason"),
        "method": int(method),
        "attempt": str(alignment["attempt"]),
        "seed_overlap_ratio": float(alignment["seed_overlap_ratio"]),
        "overlap_ratio": float(alignment["overlap_ratio"]),
        "coarse_overlap_ratio": float(alignment.get("coarse_overlap_ratio") or 0.0),
        "fitness": alignment.get("fitness"),
        "inlier_rmse": alignment.get("inlier_rmse"),
        "delta_translation_m": float(alignment["delta_translation_m"]),
        "delta_rotation_deg": float(alignment["delta_rotation_deg"]),
        "frame_index": int(session.frame_index),
        "reference_timestamp_ns": int(session.reference_timestamp_ns),
        "pairwise_transform": _matrix_to_rows(final_transform),
        "composed_transform": _matrix_to_rows(transforms_to_target[source_frame]),
    }


def align_edge_session(
    session: ManualAlignSession,
    relation_id: str,
    *,
    method: int = 2,
    registration_voxel_size: float | None = None,
    seed_transforms: dict[str, list[list[float]]] | None = None,
    max_windows: int | None = None,
    loop_closure: bool = False,
) -> dict:
    """Register one source LiDAR to a target using multi-window GICP."""
    edge_map = _session_edge_map(session)
    edge = edge_map.get(relation_id)
    if edge is None:
        known = ", ".join(sorted(edge_map))
        raise ValueError(f"Unknown relation_id {relation_id!r}. Expected one of: {known}")

    sensors_by_frame = _sensor_lookup(session)
    target_frame = session.target_frame
    if target_frame not in sensors_by_frame:
        raise RuntimeError(f"Reference frame {target_frame} is missing from session.")

    settings = session.align_settings or {}
    effective_registration_voxel = float(
        registration_voxel_size
        if registration_voxel_size is not None
        else settings.get("registration_voxel_size", 0.04)
    )
    preprocessing_params = _registration_preprocessing_params(
        effective_registration_voxel,
        plane_dist_thresh=float(settings.get("plane_distance_threshold", 0.05)),
    )
    transforms_to_target = _initialize_transforms_to_target(session, seed_transforms)
    edge_result = _run_edge_alignment(
        edge,
        session=session,
        sensors_by_frame=sensors_by_frame,
        target_frame=target_frame,
        transforms_to_target=transforms_to_target,
        seed_transforms=seed_transforms,
        preprocessing_params=preprocessing_params,
        method=int(method),
        settings=settings,
        max_windows=max_windows,
    )
    loop_closure_report = None
    if loop_closure and edge_result.get("success"):
        loop_closure_report = _apply_loop_closure_refinement(
            session,
            [edge_result],
            transforms_to_target,
        )
    transforms = _session_transforms_from_target_map(
        session,
        transforms_to_target,
        sensors_by_frame,
        seed_transforms,
    )
    return {
        "transforms": transforms,
        "edge_results": [edge_result],
        "loop_closure": _summarize_loop_closure_report(loop_closure_report),
        "summary": {
            "relation_id": relation_id,
            "label": edge["label"],
            "aligned_count": 1 if edge_result.get("success") else 0,
            "edge_count": 1,
            "movable_count": sum(1 for sensor in session.sensors if not sensor.fixed),
            "method": int(method),
            "registration_voxel_size": effective_registration_voxel,
            "strategy": "single_edge_current_frame",
            "workflow_path": session.workflow_path,
            "frame_index": int(session.frame_index),
            "loop_closure_applied": bool(loop_closure_report and loop_closure_report.get("success")),
        },
    }


def auto_align_session(
    session: ManualAlignSession,
    *,
    method: int = 2,
    registration_voxel_size: float | None = None,
    seed_transforms: dict[str, list[list[float]]] | None = None,
    max_windows: int | None = None,
    loop_closure: bool | None = None,
) -> dict:
    """Register LiDARs following workflow relation order with optional loop closure."""
    sensors_by_frame = _sensor_lookup(session)
    target_frame = session.target_frame
    if target_frame not in sensors_by_frame:
        raise RuntimeError(f"Reference frame {target_frame} is missing from session.")

    settings = session.align_settings or {}
    effective_registration_voxel = float(
        registration_voxel_size
        if registration_voxel_size is not None
        else settings.get("registration_voxel_size", 0.04)
    )
    use_loop_closure = (
        bool(loop_closure)
        if loop_closure is not None
        else bool(settings.get("enable_loop_closure", True))
    )
    preprocessing_params = _registration_preprocessing_params(
        effective_registration_voxel,
        plane_dist_thresh=float(settings.get("plane_distance_threshold", 0.05)),
    )
    transforms_to_target = _initialize_transforms_to_target(session, seed_transforms)
    edge_results: list[dict] = []

    for edge in session.align_edges:
        edge_result = _run_edge_alignment(
            edge,
            session=session,
            sensors_by_frame=sensors_by_frame,
            target_frame=target_frame,
            transforms_to_target=transforms_to_target,
            seed_transforms=seed_transforms,
            preprocessing_params=preprocessing_params,
            method=int(method),
            settings=settings,
            max_windows=max_windows,
        )
        edge_results.append(edge_result)

    loop_closure_report = None
    if use_loop_closure:
        loop_closure_report = _apply_loop_closure_refinement(
            session,
            edge_results,
            transforms_to_target,
        )

    transforms = _session_transforms_from_target_map(
        session,
        transforms_to_target,
        sensors_by_frame,
        seed_transforms,
    )
    succeeded = sum(1 for item in edge_results if item.get("success"))
    return {
        "transforms": transforms,
        "edge_results": edge_results,
        "loop_closure": _summarize_loop_closure_report(loop_closure_report),
        "summary": {
            "aligned_count": succeeded,
            "edge_count": len(session.align_edges),
            "movable_count": sum(1 for sensor in session.sensors if not sensor.fixed),
            "method": int(method),
            "registration_voxel_size": effective_registration_voxel,
            "strategy": "perimeter_chain_current_frame",
            "workflow_path": session.workflow_path,
            "frame_index": int(session.frame_index),
            "loop_closure_applied": bool(loop_closure_report and loop_closure_report.get("success")),
        },
    }


def export_manual_alignment(
    session: ManualAlignSession,
    transforms: dict[str, list[list[float]]],
    output_dir: str | Path,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calibrated_dir = output_path / "calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    target_frame = session.target_frame
    merged = o3d.geometry.PointCloud()
    colored = o3d.geometry.PointCloud()
    extrinsics_payload = []
    saved_files: list[str] = []

    for sensor in session.sensors:
        matrix_rows = transforms.get(sensor.frame_id, sensor.current_transform)
        matrix = np.asarray(matrix_rows, dtype=float).reshape(4, 4)
        sensor.current_transform = _matrix_to_rows(matrix)

        points = np.asarray(sensor.positions, dtype=np.float32).reshape(-1, 3)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.copy())
        if not sensor.fixed:
            cloud.transform(matrix)
        merged += cloud

        colored_cloud = copy.deepcopy(cloud)
        colored_cloud.paint_uniform_color(sensor.color_rgb)
        colored += colored_cloud

        if not sensor.fixed:
            metadata = {
                "topic": sensor.topic,
                "sensor_name": sensor.sensor_name,
                "source": "manual_align_web",
                "reference_timestamp_ns": session.reference_timestamp_ns,
                "sensor_timestamp_ns": sensor.timestamp_ns,
            }
            payload = build_extrinsics_payload(
                parent_frame=target_frame,
                child_frame=sensor.frame_id,
                matrix=matrix,
                metadata=metadata,
            )
            extrinsics_payload.append(payload)
            file_path = calibrated_dir / extrinsics_filename(
                target_frame,
                sensor.frame_id,
            )
            save_extrinsics_yaml(
                str(file_path),
                parent_frame=target_frame,
                child_frame=sensor.frame_id,
                matrix=matrix,
                metadata=metadata,
            )
            saved_files.append(str(file_path))

    merged_pcd = output_path / "merged_cloud.pcd"
    colored_ply = output_path / "merged_cloud_colored.ply"
    o3d.io.write_point_cloud(str(merged_pcd), merged)
    o3d.io.write_point_cloud(str(colored_ply), colored)

    calibrated_tf = {
        "base_topic": session.target_topic,
        "base_frame": target_frame,
        "extrinsics": extrinsics_payload,
    }
    calibrated_tf_path = output_path / "calibrated_tf.yaml"
    with open(calibrated_tf_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(calibrated_tf, file, sort_keys=False)

    state_path = output_path / "manual_align_state.yaml"
    state_payload = {
        "record_files": session.record_files,
        "conf_dir": session.conf_dir,
        "target_topic": session.target_topic,
        "target_frame": target_frame,
        "reference_timestamp_ns": session.reference_timestamp_ns,
        "transforms": {
            sensor.frame_id: _matrix_to_rows(
                np.asarray(
                    transforms.get(sensor.frame_id, sensor.current_transform),
                    dtype=float,
                ).reshape(4, 4)
            )
            for sensor in session.sensors
        },
        "artifacts": {
            "calibrated_tf": str(calibrated_tf_path),
            "merged_cloud": str(merged_pcd),
            "merged_cloud_colored": str(colored_ply),
            "calibrated_files": saved_files,
        },
    }
    with open(state_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(state_payload, file, sort_keys=False)

    return {
        "output_dir": str(output_path),
        "calibrated_tf": str(calibrated_tf_path),
        "merged_cloud": str(merged_pcd),
        "merged_cloud_colored": str(colored_ply),
        "manual_align_state": str(state_path),
        "calibrated_files": saved_files,
        "point_count": int(len(merged.points)),
    }
