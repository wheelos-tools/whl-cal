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

import bisect
import copy
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from lidar2lidar.extrinsic_io import extrinsics_filename, load_extrinsics_file, save_extrinsics_yaml
from lidar2lidar.record_adapter import Record, ensure_record_available


@dataclass(frozen=True)
class PointCloudMeta:
    topic: str
    frame_id: str
    timestamp_ns: int
    record_path: str


@dataclass(frozen=True)
class TransformEdge:
    parent_frame: str
    child_frame: str
    transform: np.ndarray
    source_topic: str
    timestamp_ns: int | None
    is_static: bool


def discover_record_files(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Record path not found: {input_path}")

    record_files = sorted(
        str(child) for child in path.iterdir()
        if child.is_file() and (".record." in child.name or child.suffix == ".record")
    )
    if record_files:
        return record_files

    return sorted(str(child) for child in path.iterdir() if child.is_file())


def list_topics(record_files: Iterable[str]) -> dict[str, int]:
    ensure_record_available()
    counts: Counter[str] = Counter()
    for record_file in record_files:
        with Record(record_file) as record:
            for channel, _, _, _ in record.read_raw_messages():
                counts[channel] += 1
    return dict(counts.most_common())


def infer_pointcloud_topics(topic_counts: dict[str, int]) -> list[str]:
    return sorted(topic for topic in topic_counts if topic.endswith("/PointCloud2"))


def topic_sensor_name(topic: str) -> str:
    parts = topic.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return topic


def get_topic_frame_ids(record_files: Iterable[str], topics: Iterable[str]) -> dict[str, str]:
    ensure_record_available()
    pending = set(topics)
    frame_ids: dict[str, str] = {}

    for record_file in record_files:
        if not pending:
            break
        with Record(record_file) as record:
            for channel, msg, _ in record.read_messages(topics=tuple(pending)):
                header = getattr(msg, "header", None)
                frame_id = getattr(header, "frame_id", "")
                if channel not in frame_ids:
                    frame_ids[channel] = frame_id
                    pending.discard(channel)
                if not pending:
                    break

    for topic in topics:
        frame_ids.setdefault(topic, "")
    return frame_ids


def collect_pointcloud_metadata(record_files: Iterable[str], topics: Iterable[str]) -> dict[str, list[PointCloudMeta]]:
    ensure_record_available()
    metadata: dict[str, list[PointCloudMeta]] = defaultdict(list)
    topic_set = tuple(topics)

    for record_file in record_files:
        with Record(record_file) as record:
            for channel, msg, timestamp_ns in record.read_messages(topics=topic_set):
                header = getattr(msg, "header", None)
                frame_id = getattr(header, "frame_id", "")
                metadata[channel].append(
                    PointCloudMeta(
                        topic=channel,
                        frame_id=frame_id,
                        timestamp_ns=int(timestamp_ns),
                        record_path=record_file,
                    )
                )

    for topic in topics:
        metadata.setdefault(topic, [])

    for metas in metadata.values():
        metas.sort(key=lambda item: item.timestamp_ns)
    return metadata


def pointcloud_message_to_open3d(msg) -> o3d.geometry.PointCloud:
    points = np.array([(point.x, point.y, point.z) for point in msg.point], dtype=np.float64)
    if points.size == 0:
        return o3d.geometry.PointCloud()

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def load_pointcloud_from_meta(meta: PointCloudMeta) -> o3d.geometry.PointCloud:
    ensure_record_available()
    with Record(meta.record_path) as record:
        for channel, msg, timestamp_ns in record.read_messages(topics=[meta.topic]):
            if channel == meta.topic and int(timestamp_ns) == meta.timestamp_ns:
                return pointcloud_message_to_open3d(msg)
    raise RuntimeError(
        f"Failed to reload point cloud from {meta.record_path} topic {meta.topic} at {meta.timestamp_ns}."
    )


def find_synchronized_pairs(source_metas: list[PointCloudMeta],
                            target_metas: list[PointCloudMeta],
                            max_delta_ns: int,
                            max_pairs: int) -> list[tuple[PointCloudMeta, PointCloudMeta, int]]:
    if not source_metas or not target_metas:
        return []

    target_times = [meta.timestamp_ns for meta in target_metas]
    candidates: list[tuple[PointCloudMeta, PointCloudMeta, int]] = []
    for source_meta in source_metas:
        index = bisect.bisect_left(target_times, source_meta.timestamp_ns)
        neighbor_indices = []
        if index < len(target_metas):
            neighbor_indices.append(index)
        if index > 0:
            neighbor_indices.append(index - 1)

        best_pair = None
        for target_index in neighbor_indices:
            target_meta = target_metas[target_index]
            delta_ns = abs(source_meta.timestamp_ns - target_meta.timestamp_ns)
            if delta_ns <= max_delta_ns:
                if best_pair is None or delta_ns < best_pair[2]:
                    best_pair = (source_meta, target_meta, delta_ns)
        if best_pair is not None:
            candidates.append(best_pair)

    candidates.sort(key=lambda item: (item[2], item[0].timestamp_ns, item[1].timestamp_ns))
    selected: list[tuple[PointCloudMeta, PointCloudMeta, int]] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for source_meta, target_meta, delta_ns in candidates:
        if source_meta.timestamp_ns in used_sources or target_meta.timestamp_ns in used_targets:
            continue
        selected.append((source_meta, target_meta, delta_ns))
        used_sources.add(source_meta.timestamp_ns)
        used_targets.add(target_meta.timestamp_ns)

    selected.sort(key=lambda item: item[0].timestamp_ns)
    if max_pairs > 0 and len(selected) > max_pairs:
        indices = np.linspace(0, len(selected) - 1, num=max_pairs, dtype=int)
        selected = [selected[index] for index in indices]
    return selected


def proto_transform_to_matrix(transform_proto) -> np.ndarray:
    translation = transform_proto.translation
    rotation = transform_proto.rotation

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_quat([
        float(rotation.qx),
        float(rotation.qy),
        float(rotation.qz),
        float(rotation.qw),
    ]).as_matrix()
    transform[:3, 3] = [
        float(translation.x),
        float(translation.y),
        float(translation.z),
    ]
    return transform


def extract_tf_edges(record_files: Iterable[str]) -> list[TransformEdge]:
    ensure_record_available()
    static_edges: dict[tuple[str, str], TransformEdge] = {}
    dynamic_edges: dict[tuple[str, str], TransformEdge] = {}

    for record_file in record_files:
        with Record(record_file) as record:
            for topic, msg, timestamp_ns in record.read_messages(topics=("/tf_static", "/tf")):
                for transform_stamped in msg.transforms:
                    parent_frame = getattr(transform_stamped.header, "frame_id", "")
                    child_frame = getattr(transform_stamped, "child_frame_id", "")
                    if not parent_frame or not child_frame:
                        continue
                    edge = TransformEdge(
                        parent_frame=parent_frame,
                        child_frame=child_frame,
                        transform=proto_transform_to_matrix(transform_stamped.transform),
                        source_topic=topic,
                        timestamp_ns=int(timestamp_ns),
                        is_static=(topic == "/tf_static"),
                    )
                    key = (parent_frame, child_frame)
                    if edge.is_static:
                        static_edges[key] = edge
                    else:
                        dynamic_edges[key] = edge

    edges = list(static_edges.values())
    edges.extend(dynamic_edges.values())
    edges.sort(key=lambda item: (item.parent_frame, item.child_frame, item.source_topic))
    return edges


def load_transform_edges_from_dir(conf_dir: str | None) -> list[TransformEdge]:
    if not conf_dir:
        return []

    directory = Path(conf_dir)
    if not directory.exists() or not directory.is_dir():
        return []

    edges = []
    for file_path in sorted(directory.glob("*_extrinsics.yaml")):
        matrix, parent_frame, child_frame, stamp_ns, _ = load_extrinsics_file(str(file_path))
        if not parent_frame or not child_frame:
            continue
        edges.append(
            TransformEdge(
                parent_frame=parent_frame,
                child_frame=child_frame,
                transform=matrix,
                source_topic="conf",
                timestamp_ns=stamp_ns,
                is_static=True,
            )
        )
    return edges


def save_transform_edges_to_dir(output_dir: str | Path,
                                edges: Iterable[TransformEdge],
                                include_dynamic: bool = False) -> list[str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for edge in sorted(edges, key=lambda item: (item.parent_frame, item.child_frame, item.source_topic)):
        if not include_dynamic and not edge.is_static:
            continue
        file_path = directory / extrinsics_filename(edge.parent_frame, edge.child_frame)
        save_extrinsics_yaml(
            str(file_path),
            parent_frame=edge.parent_frame,
            child_frame=edge.child_frame,
            matrix=edge.transform,
            stamp_ns=edge.timestamp_ns,
            metadata={
                "source_topic": edge.source_topic,
                "is_static": bool(edge.is_static),
            },
        )
        saved_paths.append(str(file_path))
    return saved_paths


def build_transform_graph(edges: Iterable[TransformEdge]) -> dict[str, dict[str, np.ndarray]]:
    graph: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for edge in edges:
        # Apollo tf stores the child pose in the parent frame, so the matrix maps
        # points from child coordinates into parent coordinates.
        graph[edge.child_frame][edge.parent_frame] = edge.transform
        graph[edge.parent_frame][edge.child_frame] = np.linalg.inv(edge.transform)
    return graph


def merge_transform_edges(primary_edges: Iterable[TransformEdge],
                          secondary_edges: Iterable[TransformEdge]) -> list[TransformEdge]:
    merged: dict[tuple[str, str], TransformEdge] = {}
    for edge in secondary_edges:
        merged[(edge.parent_frame, edge.child_frame)] = edge
    for edge in primary_edges:
        merged[(edge.parent_frame, edge.child_frame)] = edge
    return sorted(merged.values(), key=lambda item: (item.parent_frame, item.child_frame, item.source_topic))


def lookup_transform(graph: dict[str, dict[str, np.ndarray]], source_frame: str, target_frame: str) -> np.ndarray | None:
    if source_frame == target_frame:
        return np.eye(4, dtype=float)
    if source_frame not in graph or target_frame not in graph:
        return None

    visited = {source_frame}
    queue = deque([(source_frame, np.eye(4, dtype=float))])
    while queue:
        current_frame, current_transform = queue.popleft()
        for next_frame, step_transform in graph[current_frame].items():
            if next_frame in visited:
                continue
            next_transform = step_transform @ current_transform
            if next_frame == target_frame:
                return next_transform
            visited.add(next_frame)
            queue.append((next_frame, next_transform))
    return None


def render_tf_tree(edges: Iterable[TransformEdge]) -> str:
    lines = ["TF edges found in record:"]
    for edge in sorted(edges, key=lambda item: (item.parent_frame, item.child_frame, item.source_topic)):
        kind = "static" if edge.is_static else "dynamic"
        lines.append(f"- [{kind}] {edge.parent_frame} -> {edge.child_frame} ({edge.source_topic})")
    return "\n".join(lines)


def tf_tree_payload(edges: Iterable[TransformEdge]) -> dict:
    return {
        "edges": [
            {
                "parent_frame": edge.parent_frame,
                "child_frame": edge.child_frame,
                "source_topic": edge.source_topic,
                "timestamp_ns": edge.timestamp_ns,
                "is_static": bool(edge.is_static),
            }
            for edge in sorted(edges, key=lambda item: (item.parent_frame, item.child_frame, item.source_topic))
        ]
    }


def analyze_pointcloud_roots(pointcloud_frames: Iterable[str], edges: Iterable[TransformEdge]) -> dict:
    frames = sorted({frame for frame in pointcloud_frames if frame})
    indegree = {frame: 0 for frame in frames}
    adjacency = {frame: set() for frame in frames}

    for edge in edges:
        if not edge.is_static:
            continue
        if edge.parent_frame in indegree and edge.child_frame in indegree:
            indegree[edge.child_frame] += 1
            adjacency[edge.parent_frame].add(edge.child_frame)
            adjacency[edge.child_frame].add(edge.parent_frame)

    root_frames = sorted(frame for frame, degree in indegree.items() if degree == 0)
    disconnected_frames = sorted(frame for frame, neighbors in adjacency.items() if not neighbors)
    preferred_root_frame = root_frames[0] if len(root_frames) == 1 else None

    return {
        "pointcloud_frames": frames,
        "root_frames": root_frames,
        "preferred_root_frame": preferred_root_frame,
        "disconnected_frames": disconnected_frames,
        "has_unique_root": preferred_root_frame is not None,
    }


def find_missing_transform_frames(graph: dict[str, dict[str, np.ndarray]],
                                  base_frame: str,
                                  pointcloud_frames: Iterable[str]) -> list[str]:
    missing = []
    for frame in sorted({frame for frame in pointcloud_frames if frame and frame != base_frame}):
        if lookup_transform(graph, frame, base_frame) is None:
            missing.append(frame)
    return missing


def voxel_overlap_ratio(source_cloud: o3d.geometry.PointCloud,
                        target_cloud: o3d.geometry.PointCloud,
                        source_to_target: np.ndarray,
                        voxel_size: float) -> float:
    if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
        return 0.0

    source_copy = copy.deepcopy(source_cloud)
    source_copy.transform(source_to_target)
    source_copy = source_copy.voxel_down_sample(voxel_size)
    target_copy = target_cloud.voxel_down_sample(voxel_size)

    source_points = np.asarray(source_copy.points)
    target_points = np.asarray(target_copy.points)
    if source_points.size == 0 or target_points.size == 0:
        return 0.0

    source_voxels = np.unique(np.floor(source_points / voxel_size).astype(np.int64), axis=0)
    target_voxels = np.unique(np.floor(target_points / voxel_size).astype(np.int64), axis=0)
    source_set = {tuple(item) for item in source_voxels.tolist()}
    target_set = {tuple(item) for item in target_voxels.tolist()}
    if not source_set or not target_set:
        return 0.0

    return len(source_set & target_set) / min(len(source_set), len(target_set))


def rotation_angle_degrees(rotation_matrix: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def transform_delta_metrics(initial_transform: np.ndarray, refined_transform: np.ndarray) -> dict[str, float]:
    delta = refined_transform @ np.linalg.inv(initial_transform)
    return {
        "translation_norm_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": rotation_angle_degrees(delta[:3, :3]),
    }


def compute_information_metrics(source_cloud: o3d.geometry.PointCloud,
                                target_cloud: o3d.geometry.PointCloud,
                                transformation: np.ndarray,
                                max_correspondence_distance: float,
                                downsample_voxel_size: float) -> dict:
    source_eval = source_cloud.voxel_down_sample(downsample_voxel_size)
    target_eval = target_cloud.voxel_down_sample(downsample_voxel_size)
    info_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_eval,
        target_eval,
        max_correspondence_distance,
        transformation,
    )
    eigenvalues = np.linalg.eigvalsh(info_matrix)
    positive = eigenvalues[eigenvalues > 1e-9]
    if len(positive) >= 2:
        condition_number = float(positive[-1] / positive[0])
    else:
        condition_number = float("inf")

    degenerate = bool(len(positive) < 6 or condition_number > 1e6)
    return {
        "matrix": info_matrix.tolist(),
        "eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "condition_number": condition_number,
        "degenerate": degenerate,
    }
