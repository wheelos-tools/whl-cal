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

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def transform_delta_components(
    reference_transform: np.ndarray,
    candidate_transform: np.ndarray,
) -> dict:
    delta = candidate_transform @ np.linalg.inv(reference_transform)
    yaw_deg, pitch_deg, roll_deg = (
        R.from_matrix(delta[:3, :3]).as_euler("zyx", degrees=True).tolist()
    )
    return {
        "translation_norm_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(R.from_matrix(delta[:3, :3]).magnitude() * 180.0 / np.pi),
        "translation_xyz_m": {
            "x": float(delta[0, 3]),
            "y": float(delta[1, 3]),
            "z": float(delta[2, 3]),
        },
        "rotation_yaw_pitch_roll_deg": {
            "yaw": float(yaw_deg),
            "pitch": float(pitch_deg),
            "roll": float(roll_deg),
        },
    }


def summarize_metric(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "min": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def select_loop_candidate_edges(
    candidate_pairs: list[dict],
    topic_infos: dict[str, dict],
    allowed_topics: set[str],
    min_overlap: float,
) -> list[dict]:
    selected = []
    for pair in candidate_pairs:
        topic_a = pair["topic_a"]
        topic_b = pair["topic_b"]
        overlap_ratio = pair.get("overlap_ratio")
        if topic_a not in allowed_topics or topic_b not in allowed_topics:
            continue
        if overlap_ratio is None or float(overlap_ratio) < float(min_overlap):
            continue
        selected.append(
            {
                "source_topic": topic_a,
                "target_topic": topic_b,
                "registration_target_topic": topic_b,
                "source_frame": topic_infos[topic_a]["frame_id"],
                "target_frame": topic_infos[topic_b]["frame_id"],
                "overlap_ratio": float(overlap_ratio),
                "sync_dt_ms": pair["sync_dt_ms"],
                "initial_transform": pair["initial_transform"],
                "pair_role": "graph_candidate",
            }
        )
    return selected


def _edge_quality_score(edge_result: dict) -> tuple[float, float, float]:
    best_run = edge_result["best_run"]
    return (
        float(best_run["fitness"]),
        float(edge_result["overlap_ratio"]),
        -float(best_run["inlier_rmse"]),
    )


def _edge_quality_scalar(edge_result: dict) -> float:
    best_run = edge_result["best_run"]
    return (
        float(best_run["fitness"])
        * float(edge_result["overlap_ratio"])
        / (1.0 + float(best_run["inlier_rmse"]))
    )


def filter_loop_measurement_edges(
    edge_results: list[dict],
    *,
    min_quality_score: float,
    max_prior_translation_delta_m: float | None,
    max_prior_rotation_delta_deg: float | None,
) -> tuple[list[dict], list[dict]]:
    accepted_edges = []
    skipped_edges = []
    for edge_result in edge_results:
        quality_score = _edge_quality_scalar(edge_result)
        delta_to_initial = edge_result["best_run"].get("delta_to_initial", {})
        translation_delta = delta_to_initial.get("translation_norm_m")
        rotation_delta = delta_to_initial.get("rotation_deg")
        gate_reasons = []
        if quality_score < float(min_quality_score):
            gate_reasons.append("quality_score_below_threshold")
        if (
            max_prior_translation_delta_m is not None
            and translation_delta is not None
            and float(translation_delta) > float(max_prior_translation_delta_m)
        ):
            gate_reasons.append("translation_delta_to_prior_above_threshold")
        if (
            max_prior_rotation_delta_deg is not None
            and rotation_delta is not None
            and float(rotation_delta) > float(max_prior_rotation_delta_deg)
        ):
            gate_reasons.append("rotation_delta_to_prior_above_threshold")
        if gate_reasons:
            skipped_edges.append(
                {
                    "source_topic": edge_result["source_topic"],
                    "target_topic": edge_result["target_topic"],
                    "source_frame": edge_result["source_frame"],
                    "target_frame": edge_result["target_frame"],
                    "fitness": float(edge_result["best_run"]["fitness"]),
                    "inlier_rmse": float(edge_result["best_run"]["inlier_rmse"]),
                    "overlap_ratio": float(edge_result["overlap_ratio"]),
                    "quality_score": float(quality_score),
                    "delta_to_prior": delta_to_initial,
                    "reason": "production_edge_gate_failed",
                    "gate_reasons": gate_reasons,
                }
            )
            continue
        accepted_edges.append(edge_result)
    return accepted_edges, skipped_edges


def _find_parent(parents: dict[str, str], node: str) -> str:
    while parents[node] != node:
        parents[node] = parents[parents[node]]
        node = parents[node]
    return node


def _union_parent(parents: dict[str, str], node_a: str, node_b: str) -> bool:
    root_a = _find_parent(parents, node_a)
    root_b = _find_parent(parents, node_b)
    if root_a == root_b:
        return False
    parents[root_b] = root_a
    return True


def select_loop_graph_edges(
    edge_results: list[dict],
    target_topic: str,
    *,
    required_topics: Iterable[str] | None = None,
) -> dict:
    selected_topics = set(required_topics or [])
    selected_topics.add(target_topic)
    for edge_result in edge_results:
        selected_topics.add(edge_result["source_topic"])
        selected_topics.add(edge_result["target_topic"])

    parents = {topic: topic for topic in selected_topics}
    tree_edges: list[dict] = []
    loop_edges: list[dict] = []

    sorted_edges = sorted(
        edge_results,
        key=lambda item: (
            -_edge_quality_score(item)[0],
            -_edge_quality_score(item)[1],
            _edge_quality_score(item)[2],
            item["source_topic"],
            item["target_topic"],
        ),
    )

    for edge_result in sorted_edges:
        topic_a = edge_result["source_topic"]
        topic_b = edge_result["target_topic"]
        if _union_parent(parents, topic_a, topic_b):
            tree_edges.append(edge_result)
        else:
            loop_edges.append(edge_result)

    connected_components = defaultdict(list)
    for topic in selected_topics:
        connected_components[_find_parent(parents, topic)].append(topic)

    return {
        "topics": sorted(selected_topics),
        "tree_edges": tree_edges,
        "loop_edges": loop_edges,
        "graph_edges": tree_edges + loop_edges,
        "connected_components": [
            sorted(component) for component in connected_components.values()
        ],
    }


def compose_topic_transforms(
    target_topic: str,
    edge_results: list[dict],
    *,
    required_topics: Iterable[str] | None = None,
) -> dict:
    adjacency: dict[str, list[tuple[str, np.ndarray, dict]]] = defaultdict(list)
    topics = set(required_topics or [])
    topics.add(target_topic)
    for edge_result in edge_results:
        source_topic = edge_result["source_topic"]
        relation_target = edge_result["target_topic"]
        transform = np.asarray(edge_result["best_run"]["transformation"], dtype=float)
        adjacency[source_topic].append((relation_target, transform, edge_result))
        adjacency[relation_target].append(
            (source_topic, np.linalg.inv(transform), edge_result)
        )
        topics.add(source_topic)
        topics.add(relation_target)

    topic_transforms = {target_topic: np.eye(4, dtype=float)}
    relation_paths = {target_topic: []}
    stack = [target_topic]
    while stack:
        current_topic = stack.pop()
        for next_topic, step_transform, edge_result in adjacency.get(current_topic, []):
            if next_topic in topic_transforms:
                continue
            topic_transforms[next_topic] = (
                topic_transforms[current_topic] @ step_transform
            )
            relation_paths[next_topic] = relation_paths[current_topic] + [
                edge_result.get(
                    "relation_id",
                    f"{edge_result['source_topic']}__to__{edge_result['target_topic']}",
                )
            ]
            stack.append(next_topic)

    unresolved_topics = sorted(topic for topic in topics if topic not in topic_transforms)

    component_topics = set(topics)
    connected_components = []
    visited = set()
    for topic in sorted(component_topics):
        if topic in visited:
            continue
        component = []
        queue = [topic]
        visited.add(topic)
        while queue:
            current_topic = queue.pop()
            component.append(current_topic)
            for next_topic, _, _ in adjacency.get(current_topic, []):
                if next_topic in visited:
                    continue
                visited.add(next_topic)
                queue.append(next_topic)
        connected_components.append(sorted(component))

    return {
        "topic_transforms": topic_transforms,
        "unresolved_topics": unresolved_topics,
        "relation_paths": relation_paths,
        "connected_components": sorted(connected_components),
    }


def build_initial_topic_transforms(
    target_topic: str,
    target_frame: str,
    topic_infos: dict[str, dict],
    baseline_edge_results: list[dict],
    tf_graph: dict[str, dict[str, np.ndarray]],
    selected_topics: Iterable[str],
) -> dict[str, np.ndarray]:
    transforms = {target_topic: np.eye(4, dtype=float)}
    for edge_result in baseline_edge_results:
        transforms[edge_result["source_topic"]] = np.asarray(
            edge_result["best_run"]["transformation"],
            dtype=float,
        )

    for topic in selected_topics:
        if topic in transforms:
            continue
        frame_id = topic_infos[topic]["frame_id"]
        seed = None
        if frame_id:
            from lidar2lidar.record_utils import lookup_transform

            seed = lookup_transform(tf_graph, frame_id, target_frame)
        transforms[topic] = (
            np.asarray(seed, dtype=float)
            if seed is not None
            else np.eye(4, dtype=float)
        )
    return transforms


def build_prior_topic_transforms(
    target_topic: str,
    target_frame: str,
    topic_infos: dict[str, dict],
    tf_graph: dict[str, dict[str, np.ndarray]],
    selected_topics: Iterable[str],
    *,
    fallback_topic_transforms: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    transforms = {target_topic: np.eye(4, dtype=float)}
    sources = {target_topic: "fixed_target"}
    for topic in selected_topics:
        if topic == target_topic:
            continue
        frame_id = topic_infos[topic]["frame_id"]
        seed = None
        if frame_id:
            from lidar2lidar.record_utils import lookup_transform

            seed = lookup_transform(tf_graph, frame_id, target_frame)
        if seed is not None:
            transforms[topic] = np.asarray(seed, dtype=float)
            sources[topic] = "tf_graph"
            continue
        if fallback_topic_transforms is not None and topic in fallback_topic_transforms:
            transforms[topic] = np.asarray(fallback_topic_transforms[topic], dtype=float)
            sources[topic] = "baseline_or_identity_fallback"
            continue
        transforms[topic] = np.eye(4, dtype=float)
        sources[topic] = "identity_fallback"
    return transforms, sources


def _matrix_to_pose_vector(matrix: np.ndarray) -> np.ndarray:
    rotation_vector = R.from_matrix(matrix[:3, :3]).as_rotvec()
    translation = matrix[:3, 3]
    return np.concatenate([rotation_vector, translation], axis=0)


def _pose_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = R.from_rotvec(vector[:3]).as_matrix()
    matrix[:3, 3] = vector[3:6]
    return matrix


def _predict_relative_transform(
    topic_transforms: dict[str, np.ndarray],
    source_topic: str,
    target_topic: str,
) -> np.ndarray:
    source_to_root = topic_transforms[source_topic]
    target_to_root = topic_transforms[target_topic]
    return np.linalg.inv(target_to_root) @ source_to_root


def evaluate_graph_consistency(
    edge_results: list[dict],
    topic_transforms: dict[str, np.ndarray],
) -> dict:
    per_edge = []
    translation_residuals = []
    rotation_residuals = []

    for edge_result in edge_results:
        measured_transform = np.asarray(
            edge_result["best_run"]["transformation"],
            dtype=float,
        )
        predicted_transform = _predict_relative_transform(
            topic_transforms,
            edge_result["source_topic"],
            edge_result["target_topic"],
        )
        residual_transform = predicted_transform @ np.linalg.inv(measured_transform)
        translation_residual_m = float(np.linalg.norm(residual_transform[:3, 3]))
        rotation_residual_deg = float(
            R.from_matrix(residual_transform[:3, :3]).magnitude() * 180.0 / np.pi
        )
        translation_residuals.append(translation_residual_m)
        rotation_residuals.append(rotation_residual_deg)
        per_edge.append(
            {
                "source_topic": edge_result["source_topic"],
                "target_topic": edge_result["target_topic"],
                "source_frame": edge_result["source_frame"],
                "target_frame": edge_result["target_frame"],
                "fitness": float(edge_result["best_run"]["fitness"]),
                "inlier_rmse": float(edge_result["best_run"]["inlier_rmse"]),
                "overlap_ratio": float(edge_result["overlap_ratio"]),
                "translation_residual_m": translation_residual_m,
                "rotation_residual_deg": rotation_residual_deg,
                "quality_score": _edge_quality_scalar(edge_result),
            }
        )

    return {
        "per_edge": per_edge,
        "translation_residual_m": summarize_metric(translation_residuals),
        "rotation_residual_deg": summarize_metric(rotation_residuals),
    }


def optimize_loop_closure(
    target_topic: str,
    initial_topic_transforms: dict[str, np.ndarray],
    graph_edges: list[dict],
    *,
    rotation_weight: float = 5.0,
    prior_topic_transforms: dict[str, np.ndarray] | None = None,
    prior_translation_weight: float = 1.0,
    prior_rotation_weight: float = 5.0,
) -> dict:
    variable_topics = sorted(
        topic for topic in initial_topic_transforms if topic != target_topic
    )
    if len(variable_topics) < 2 or len(graph_edges) < len(variable_topics):
        return {
            "success": False,
            "reason": "insufficient_graph_constraints",
            "optimized_topic_transforms": initial_topic_transforms,
        }

    initial_vector = np.concatenate(
        [
            _matrix_to_pose_vector(initial_topic_transforms[topic])
            for topic in variable_topics
        ],
        axis=0,
    )

    def unpack(vector: np.ndarray) -> dict[str, np.ndarray]:
        topic_transforms = {target_topic: np.eye(4, dtype=float)}
        for index, topic in enumerate(variable_topics):
            start = index * 6
            topic_transforms[topic] = _pose_vector_to_matrix(vector[start : start + 6])
        return topic_transforms

    def residuals(vector: np.ndarray) -> np.ndarray:
        topic_transforms = unpack(vector)
        residual_chunks = []
        for edge_result in graph_edges:
            measurement = np.asarray(
                edge_result["best_run"]["transformation"],
                dtype=float,
            )
            predicted = _predict_relative_transform(
                topic_transforms,
                edge_result["source_topic"],
                edge_result["target_topic"],
            )
            delta = predicted @ np.linalg.inv(measurement)
            weight = np.sqrt(max(_edge_quality_scalar(edge_result), 1e-3))
            rotation_residual = R.from_matrix(delta[:3, :3]).as_rotvec() * float(
                rotation_weight
            )
            translation_residual = delta[:3, 3]
            residual_chunks.append(
                np.concatenate(
                    [
                        rotation_residual * weight,
                        translation_residual * weight,
                    ],
                    axis=0,
                )
            )
        if prior_topic_transforms is not None:
            for topic in variable_topics:
                prior_transform = prior_topic_transforms.get(topic)
                if prior_transform is None:
                    continue
                delta_to_prior = topic_transforms[topic] @ np.linalg.inv(prior_transform)
                residual_chunks.append(
                    np.concatenate(
                        [
                            R.from_matrix(delta_to_prior[:3, :3]).as_rotvec()
                            * float(prior_rotation_weight),
                            delta_to_prior[:3, 3] * float(prior_translation_weight),
                        ],
                        axis=0,
                    )
                )
        return np.concatenate(residual_chunks, axis=0)

    result = least_squares(
        residuals,
        initial_vector,
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=200,
    )
    optimized_topic_transforms = unpack(result.x)
    baseline_consistency = evaluate_graph_consistency(
        graph_edges,
        initial_topic_transforms,
    )
    optimized_consistency = evaluate_graph_consistency(
        graph_edges,
        optimized_topic_transforms,
    )

    delta_to_baseline = {}
    delta_to_prior = {}
    for topic, optimized_transform in optimized_topic_transforms.items():
        delta_to_baseline[topic] = transform_delta_components(
            initial_topic_transforms[topic],
            optimized_transform,
        )
        if prior_topic_transforms is not None and topic in prior_topic_transforms:
            delta_to_prior[topic] = transform_delta_components(
                prior_topic_transforms[topic],
                optimized_transform,
            )

    return {
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "optimized_topic_transforms": optimized_topic_transforms,
        "baseline_consistency": baseline_consistency,
        "optimized_consistency": optimized_consistency,
        "delta_to_baseline": delta_to_baseline,
        "delta_to_prior": delta_to_prior,
    }


def build_aligned_snapshot(
    *,
    target_topic: str,
    reference_target_meta,
    topic_transforms: dict[str, np.ndarray],
    metadata_by_topic: dict,
    sync_threshold_ns: int,
    cloud_cache: dict,
    load_cloud: Callable,
) -> dict:
    colors = {
        0: np.array([1.0, 0.0, 0.0], dtype=float),
        1: np.array([0.0, 1.0, 0.0], dtype=float),
        2: np.array([0.0, 0.0, 1.0], dtype=float),
        3: np.array([1.0, 0.8, 0.0], dtype=float),
        4: np.array([1.0, 0.0, 1.0], dtype=float),
        5: np.array([0.0, 1.0, 1.0], dtype=float),
    }

    aligned_clouds = []
    merged_cloud = o3d.geometry.PointCloud()
    colored_merged_cloud = o3d.geometry.PointCloud()
    sorted_topics = sorted(topic_transforms)

    for index, topic in enumerate(sorted_topics):
        if topic == target_topic:
            matches = [(reference_target_meta, reference_target_meta, 0)]
        else:
            from lidar2lidar.record_utils import find_synchronized_pairs

            matches = find_synchronized_pairs(
                metadata_by_topic[topic],
                [reference_target_meta],
                sync_threshold_ns,
                max_pairs=1,
            )
        if not matches:
            continue
        source_meta, _, delta_ns = matches[0]
        cloud = copy.deepcopy(load_cloud(source_meta, cloud_cache))
        transform = np.asarray(topic_transforms[topic], dtype=float)
        if topic != target_topic:
            cloud.transform(transform)

        colored_cloud = copy.deepcopy(cloud)
        color = colors[index % len(colors)]
        colored_cloud.paint_uniform_color(color)

        aligned_clouds.append(
            {
                "topic": topic,
                "frame_id": source_meta.frame_id,
                "timestamp_ns": int(source_meta.timestamp_ns),
                "sync_dt_ms": float(delta_ns / 1e6),
                "color_rgb": [float(value) for value in color.tolist()],
                "cloud": cloud,
                "colored_cloud": colored_cloud,
            }
        )
        merged_cloud += cloud
        colored_merged_cloud += colored_cloud

    return {
        "target_topic": target_topic,
        "target_timestamp_ns": int(reference_target_meta.timestamp_ns),
        "aligned_clouds": aligned_clouds,
        "merged_cloud": merged_cloud,
        "colored_merged_cloud": colored_merged_cloud,
    }


def save_snapshot_clouds(
    snapshot: dict,
    *,
    plain_output_path: str | None,
    colored_output_path: str | None,
) -> dict:
    output = {
        "target_topic": snapshot["target_topic"],
        "target_timestamp_ns": snapshot["target_timestamp_ns"],
        "topics": [
            {
                "topic": item["topic"],
                "frame_id": item["frame_id"],
                "timestamp_ns": item["timestamp_ns"],
                "sync_dt_ms": item["sync_dt_ms"],
                "color_rgb": item["color_rgb"],
            }
            for item in snapshot["aligned_clouds"]
        ],
        "plain_output_path": plain_output_path,
        "colored_output_path": colored_output_path,
        "plain_saved": False,
        "colored_saved": False,
    }
    if plain_output_path is not None:
        output["plain_saved"] = bool(
            o3d.io.write_point_cloud(plain_output_path, snapshot["merged_cloud"])
        )
    if colored_output_path is not None:
        output["colored_saved"] = bool(
            o3d.io.write_point_cloud(
                colored_output_path,
                snapshot["colored_merged_cloud"],
            )
        )
    return output


def compute_visual_plane_metrics(
    snapshot: dict,
    *,
    downsample_voxel_size: float = 0.1,
    plane_distance_threshold: float = 0.08,
    max_planes: int = 4,
    min_plane_points: int = 600,
    corner_angle_tolerance_deg: float = 20.0,
    corner_distance_threshold_m: float = 0.12,
    slice_bin_size_m: float = 0.25,
    min_slice_points: int = 120,
) -> dict:
    def extract_wall_normal_bins(normals: list[np.ndarray]) -> int:
        if not normals:
            return 0
        bins = set()
        for normal in normals:
            yaw_deg = float(np.degrees(np.arctan2(normal[1], normal[0])))
            yaw_deg = abs(((yaw_deg + 90.0) % 180.0) - 90.0)
            bins.add(int(round(yaw_deg / 15.0)))
        return len(bins)

    def line_from_planes(
        normal_a: np.ndarray,
        offset_a: float,
        normal_b: np.ndarray,
        offset_b: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        direction = np.cross(normal_a, normal_b)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return None
        direction /= direction_norm
        system = np.vstack([normal_a, normal_b, direction])
        rhs = np.array([-offset_a, -offset_b, 0.0], dtype=float)
        try:
            point = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            point, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        return point, direction

    def point_line_distance(points: np.ndarray, point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        vectors = points - point
        projected = np.outer(vectors @ direction, direction)
        return np.linalg.norm(vectors - projected, axis=1)

    def compute_slice_metric(points: np.ndarray, axis_a: int, axis_b: int) -> dict:
        if points.size == 0:
            return {
                "bin_size_m": float(slice_bin_size_m),
                "usable_slice_count": 0,
                "thickness_m": summarize_metric([]),
                "sharpness_score": None,
            }
        coord_a = points[:, axis_a]
        coord_b = points[:, axis_b]
        if coord_a.size == 0:
            return {
                "bin_size_m": float(slice_bin_size_m),
                "usable_slice_count": 0,
                "thickness_m": summarize_metric([]),
                "sharpness_score": None,
            }
        coord_min = float(np.min(coord_a))
        bin_indices = np.floor((coord_a - coord_min) / float(slice_bin_size_m)).astype(int)
        spans = []
        for bin_index in np.unique(bin_indices):
            mask = bin_indices == bin_index
            if int(np.sum(mask)) < int(min_slice_points):
                continue
            values = coord_b[mask]
            spans.append(
                float(np.percentile(values, 97.5) - np.percentile(values, 2.5))
            )
        thickness = summarize_metric(spans)
        sharpness_score = (
            float(1.0 / (thickness["mean"] + 1e-6))
            if thickness["mean"] is not None
            else None
        )
        return {
            "bin_size_m": float(slice_bin_size_m),
            "usable_slice_count": int(thickness["count"]),
            "thickness_m": thickness,
            "sharpness_score": sharpness_score,
        }

    merged_cloud = snapshot["merged_cloud"].voxel_down_sample(downsample_voxel_size)
    working_cloud = copy.deepcopy(merged_cloud)
    wall_planes = []
    wall_geometry = []
    extracted_planes = 0

    while (
        extracted_planes < max_planes and len(working_cloud.points) >= min_plane_points
    ):
        plane_model, inliers = working_cloud.segment_plane(
            distance_threshold=plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        if len(inliers) < min_plane_points:
            break

        normal = np.asarray(plane_model[:3], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-9)
        plane_offset = float(plane_model[3])
        is_wall = bool(abs(normal[2]) <= 0.3)
        plane_cloud = working_cloud.select_by_index(inliers)

        if is_wall:
            points = np.asarray(plane_cloud.points)
            merged_distances = points @ normal + plane_offset
            per_sensor = []
            sensor_means = []
            for aligned in snapshot["aligned_clouds"]:
                sensor_points = np.asarray(aligned["cloud"].points)
                if sensor_points.size == 0:
                    continue
                distances = sensor_points @ normal + plane_offset
                inlier_mask = np.abs(distances) <= plane_distance_threshold
                if int(np.sum(inlier_mask)) < 100:
                    continue
                inlier_distances = distances[inlier_mask]
                mean_signed_distance = float(np.mean(inlier_distances))
                sensor_means.append(mean_signed_distance)
                per_sensor.append(
                    {
                        "topic": aligned["topic"],
                        "inlier_count": int(inlier_distances.size),
                        "mean_signed_distance_m": mean_signed_distance,
                        "abs_distance_p95_m": float(
                            np.percentile(np.abs(inlier_distances), 95)
                        ),
                    }
                )
            signed_span = None
            if merged_distances.size > 0:
                signed_span = float(
                    np.percentile(merged_distances, 97.5)
                    - np.percentile(merged_distances, 2.5)
                )
            wall_planes.append(
                {
                    "normal_xyz": [float(value) for value in normal.tolist()],
                    "offset_m": plane_offset,
                    "merged_inlier_count": int(len(inliers)),
                    "merged_abs_distance_p95_m": float(
                        np.percentile(np.abs(merged_distances), 95)
                    ),
                    "merged_signed_span_p95_m": signed_span,
                    "sensor_mean_offset_spread_m": (
                        float(max(sensor_means) - min(sensor_means))
                        if len(sensor_means) >= 2
                        else None
                    ),
                    "per_sensor": per_sensor,
                }
            )
            wall_geometry.append(
                {
                    "normal": normal,
                    "offset": plane_offset,
                    "points": points,
                }
            )

        working_cloud = working_cloud.select_by_index(inliers, invert=True)
        extracted_planes += 1

    thickness_values = [
        plane["merged_signed_span_p95_m"]
        for plane in wall_planes
        if plane["merged_signed_span_p95_m"] is not None
    ]
    spread_values = [
        plane["sensor_mean_offset_spread_m"]
        for plane in wall_planes
        if plane["sensor_mean_offset_spread_m"] is not None
    ]
    wall_normals = [plane["normal"] for plane in wall_geometry]
    corner_records = []
    merged_points = np.asarray(merged_cloud.points)
    for index_a in range(len(wall_geometry)):
        for index_b in range(index_a + 1, len(wall_geometry)):
            plane_a = wall_geometry[index_a]
            plane_b = wall_geometry[index_b]
            cosine = float(np.clip(np.dot(plane_a["normal"], plane_b["normal"]), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(abs(cosine))))
            if abs(angle_deg - 90.0) > float(corner_angle_tolerance_deg):
                continue
            line = line_from_planes(
                plane_a["normal"],
                plane_a["offset"],
                plane_b["normal"],
                plane_b["offset"],
            )
            if line is None:
                continue
            line_point, line_direction = line
            distances_a = np.abs(merged_points @ plane_a["normal"] + plane_a["offset"])
            distances_b = np.abs(merged_points @ plane_b["normal"] + plane_b["offset"])
            support_mask = (
                (distances_a <= float(corner_distance_threshold_m))
                & (distances_b <= float(corner_distance_threshold_m))
            )
            if int(np.sum(support_mask)) < int(min_slice_points):
                continue
            support_points = merged_points[support_mask]
            line_distances = point_line_distance(
                support_points,
                line_point,
                line_direction,
            )
            corner_records.append(
                {
                    "plane_indices": [index_a, index_b],
                    "angle_deg": angle_deg,
                    "support_point_count": int(support_points.shape[0]),
                    "spread_radius_p95_m": float(np.percentile(line_distances, 95)),
                    "spread_radius_mean_m": float(np.mean(line_distances)),
                }
            )
    corner_spreads = [record["spread_radius_p95_m"] for record in corner_records]
    slice_metrics = {
        "xy": compute_slice_metric(merged_points, 0, 1),
        "xz": compute_slice_metric(merged_points, 0, 2),
        "yz": compute_slice_metric(merged_points, 1, 2),
    }
    return {
        "wall_plane_count": len(wall_planes),
        "plane_normal_diversity": int(extract_wall_normal_bins(wall_normals)),
        "wall_planes": wall_planes,
        "wall_signed_span_p95_m": summarize_metric(thickness_values),
        "wall_double_edge_separation_m": summarize_metric(thickness_values),
        "sensor_offset_spread_m": summarize_metric(spread_values),
        "corner_metrics": {
            "corner_pair_count": len(corner_records),
            "corners": corner_records,
            "corner_spread_radius_m": summarize_metric(corner_spreads),
        },
        "slice_metrics": slice_metrics,
    }
