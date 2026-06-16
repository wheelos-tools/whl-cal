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

import argparse
import copy
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from calibration_common.evaluation import (
    build_final_acceptance,
    write_acceptance_artifacts,
    write_paradigm_artifacts,
    write_table_csv,
)
from lidar2lidar.extrinsic_io import (
    build_extrinsics_payload,
    extrinsics_filename,
    save_extrinsics_yaml,
)
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic
from lidar2lidar.loop_closure import (
    build_aligned_snapshot,
    build_initial_topic_transforms,
    build_prior_topic_transforms,
    compose_topic_transforms,
    compute_visual_plane_metrics,
    evaluate_graph_consistency,
    filter_loop_measurement_edges,
    optimize_loop_closure,
    save_snapshot_clouds,
    select_loop_candidate_edges,
    select_loop_graph_edges,
)
from lidar2lidar.prepared_dataset import load_prepared_rig_dataset
from lidar2lidar.record_utils import (
    analyze_pointcloud_roots,
    build_transform_graph,
    collect_pointcloud_metadata,
    compute_information_metrics,
    discover_record_files,
    extract_tf_edges,
    find_missing_transform_frames,
    find_synchronized_pairs,
    get_topic_frame_ids,
    infer_pointcloud_topics,
    list_topics,
    load_pointcloud_from_meta,
    load_transform_edges_from_dir,
    lookup_transform,
    merge_transform_edges,
    save_transform_edges_to_dir,
    tf_tree_payload,
    topic_sensor_name,
    transform_delta_metrics,
    voxel_overlap_ratio,
)
from lidar2lidar.workflow import load_workflow_config, resolve_workflow_plan

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def summarize_values(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "max": None,
            "min": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def topic_priority(topic: str) -> float:
    priority = 0.0
    lowered = topic.lower()
    if "main" in lowered:
        priority += 0.2
    if "fusion" in lowered:
        priority -= 1.0
    return priority


def topic_preference_sort_key(topic: str, topic_infos: dict[str, dict]) -> tuple:
    lowered = topic.lower()
    return (
        "fusion" in lowered,
        0 if "main" in lowered else 1,
        -topic_infos[topic]["count"],
        topic,
    )


def choose_target_topic(
    topic_infos: dict[str, dict],
    candidate_pairs: list[dict],
    root_analysis: dict,
    explicit_target: str | None,
) -> str:
    if explicit_target is not None:
        if explicit_target not in topic_infos:
            raise ValueError(f"Unknown target topic: {explicit_target}")
        return explicit_target

    frame_representatives: dict[str, list[str]] = defaultdict(list)
    for topic, info in topic_infos.items():
        frame_representatives[info["frame_id"]].append(topic)

    raw_candidates = []
    for topics in frame_representatives.values():
        preferred = sorted(
            topics, key=lambda topic: topic_preference_sort_key(topic, topic_infos)
        )[0]
        raw_candidates.append(preferred)

    raw_candidates = [topic for topic in raw_candidates if topic]
    if not raw_candidates:
        raw_candidates = list(topic_infos)

    preferred_root_frame = root_analysis.get("preferred_root_frame")
    if preferred_root_frame:
        root_topics = frame_representatives.get(preferred_root_frame, [])
        if root_topics:
            return sorted(
                root_topics,
                key=lambda topic: topic_preference_sort_key(topic, topic_infos),
            )[0]

    scores = {topic: topic_priority(topic) for topic in raw_candidates}
    for pair in candidate_pairs:
        overlap = pair["overlap_ratio"]
        if overlap is None:
            continue
        if pair["topic_a"] in scores:
            scores[pair["topic_a"]] += overlap
        if pair["topic_b"] in scores:
            scores[pair["topic_b"]] += overlap

    return max(raw_candidates, key=lambda topic: scores[topic])


def select_edges_for_target(
    topic_infos: dict[str, dict],
    candidate_pairs: list[dict],
    tf_graph: dict,
    target_topic: str,
    target_frame: str,
    source_topics: list[str] | None,
    min_overlap: float,
) -> tuple[list[dict], list[dict]]:
    requested_sources = set(source_topics or [])
    selected_edges = []
    skipped_precheck = []
    same_frame_target_topics = [
        topic for topic, info in topic_infos.items() if info["frame_id"] == target_frame
    ]

    for topic, info in topic_infos.items():
        if topic == target_topic:
            continue
        if requested_sources and topic not in requested_sources:
            continue
        if info["frame_id"] == target_frame:
            skipped_precheck.append(
                {
                    "source_topic": topic,
                    "target_topic": target_topic,
                    "reason": "same_target_frame",
                }
            )
            continue

        initial_transform = lookup_transform(tf_graph, info["frame_id"], target_frame)
        if initial_transform is None:
            skipped_precheck.append(
                {
                    "source_topic": topic,
                    "target_topic": target_topic,
                    "reason": "no_tf_path",
                }
            )
            continue

        proxy_candidates = [
            pair
            for pair in candidate_pairs
            if topic in {pair["topic_a"], pair["topic_b"]}
            and ({pair["topic_a"], pair["topic_b"]} & set(same_frame_target_topics))
        ]
        proxy_candidates = [
            pair for pair in proxy_candidates if pair["overlap_ratio"] is not None
        ]
        if not proxy_candidates:
            skipped_precheck.append(
                {
                    "source_topic": topic,
                    "target_topic": target_topic,
                    "reason": "no_overlap_probe",
                }
            )
            continue

        pair_match = max(proxy_candidates, key=lambda pair: pair["overlap_ratio"])
        registration_target_topic = (
            pair_match["topic_b"]
            if pair_match["topic_a"] == topic
            else pair_match["topic_a"]
        )

        if pair_match["overlap_ratio"] < min_overlap:
            skipped_precheck.append(
                {
                    "source_topic": topic,
                    "target_topic": target_topic,
                    "registration_target_topic": registration_target_topic,
                    "reason": "low_overlap",
                    "overlap_ratio": float(pair_match["overlap_ratio"]),
                }
            )
            continue

        selected_edges.append(
            {
                "source_topic": topic,
                "target_topic": target_topic,
                "registration_target_topic": registration_target_topic,
                "source_frame": info["frame_id"],
                "target_frame": target_frame,
                "overlap_ratio": float(pair_match["overlap_ratio"]),
                "sync_dt_ms": pair_match["sync_dt_ms"],
                "initial_transform": initial_transform.tolist(),
            }
        )

    return selected_edges, skipped_precheck


def _pair_key(topic_a: str, topic_b: str) -> frozenset[str]:
    return frozenset((topic_a, topic_b))


def build_selected_edges_for_relations(
    relations: list[dict],
    candidate_pairs: list[dict],
    tf_graph: dict,
    min_overlap: float,
) -> tuple[list[dict], list[dict]]:
    pair_lookup = {
        _pair_key(pair["topic_a"], pair["topic_b"]): pair for pair in candidate_pairs
    }
    selected_edges = []
    skipped_precheck = []
    for relation in relations:
        pair = pair_lookup.get(
            _pair_key(relation["source_topic"], relation["target_topic"])
        )
        if pair is None:
            skipped_precheck.append(
                {
                    **relation,
                    "reason": "pair_not_found",
                }
            )
            continue
        initial_transform = lookup_transform(
            tf_graph,
            relation["source_frame"],
            relation["target_frame"],
        )
        if initial_transform is None:
            skipped_precheck.append(
                {
                    **relation,
                    "reason": "no_tf_path",
                }
            )
            continue
        if pair.get("overlap_ratio") is None:
            skipped_precheck.append(
                {
                    **relation,
                    "reason": pair.get("reason", "no_overlap_probe"),
                    "candidate_pair": pair,
                }
            )
            continue
        if float(pair["overlap_ratio"]) < float(min_overlap):
            skipped_precheck.append(
                {
                    **relation,
                    "reason": "low_overlap",
                    "overlap_ratio": float(pair["overlap_ratio"]),
                    "candidate_pair": pair,
                }
            )
            continue
        selected_edges.append(
            {
                **relation,
                "registration_target_topic": relation["target_topic"],
                "overlap_ratio": float(pair["overlap_ratio"]),
                "sync_dt_ms": pair.get("sync_dt_ms"),
                "initial_transform": np.asarray(
                    initial_transform, dtype=float
                ).tolist(),
                "candidate_pair": {
                    "sync_dt_ms": pair.get("sync_dt_ms"),
                    "overlap_ratio": pair.get("overlap_ratio"),
                    "reason": pair.get("reason"),
                },
            }
        )
    return selected_edges, skipped_precheck


def summarize_repeatability_runs(
    runs: list[dict],
    reference_transform: np.ndarray,
) -> dict:
    translation_values = []
    rotation_values = []
    fitness_values = []
    rmse_values = []
    per_window = []
    for run in runs:
        delta = transform_delta_metrics(
            reference_transform,
            np.asarray(run["transformation"], dtype=float),
        )
        translation_values.append(float(delta["translation_norm_m"]))
        rotation_values.append(float(delta["rotation_deg"]))
        fitness_values.append(float(run["fitness"]))
        rmse_values.append(float(run["inlier_rmse"]))
        per_window.append(
            {
                "source_timestamp_ns": int(run["source_timestamp_ns"]),
                "target_timestamp_ns": int(run["target_timestamp_ns"]),
                "sync_dt_ms": float(run["sync_dt_ms"]),
                "fitness": float(run["fitness"]),
                "inlier_rmse": float(run["inlier_rmse"]),
                "delta_to_reference": delta,
            }
        )
    return {
        "window_count": len(runs),
        "translation_to_reference_m": summarize_values(translation_values),
        "rotation_to_reference_deg": summarize_values(rotation_values),
        "fitness": summarize_values(fitness_values),
        "inlier_rmse": summarize_values(rmse_values),
        "per_window": per_window,
    }


def _single_direction_unmatched_ratio(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    *,
    distance_threshold_m: float,
) -> float | None:
    source_points = np.asarray(source_cloud.points)
    if source_points.size == 0 or len(target_cloud.points) == 0:
        return None
    tree = o3d.geometry.KDTreeFlann(target_cloud)
    unmatched = 0
    for point in source_points:
        count, _, distance2 = tree.search_knn_vector_3d(point, 1)
        if count <= 0 or distance2[0] > float(distance_threshold_m) ** 2:
            unmatched += 1
    return float(unmatched / source_points.shape[0])


def estimate_bidirectional_unmatched_ratio(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    *,
    transform: np.ndarray,
    distance_threshold_m: float,
    voxel_size: float,
) -> float | None:
    source_down = source_cloud.voxel_down_sample(voxel_size)
    target_down = target_cloud.voxel_down_sample(voxel_size)
    if len(source_down.points) == 0 or len(target_down.points) == 0:
        return None
    aligned_source = copy.deepcopy(source_down)
    aligned_source.transform(transform)
    forward = _single_direction_unmatched_ratio(
        aligned_source,
        target_down,
        distance_threshold_m=distance_threshold_m,
    )
    backward = _single_direction_unmatched_ratio(
        target_down,
        aligned_source,
        distance_threshold_m=distance_threshold_m,
    )
    ratios = [ratio for ratio in (forward, backward) if ratio is not None]
    if not ratios:
        return None
    return float(np.mean(ratios))


def build_scene_sufficiency_report(
    relations: list[dict],
    metadata_by_topic: dict,
    sync_threshold_ns: int,
    load_cloud,
    cloud_cache: dict,
    config: dict,
) -> dict:
    relation_reports = []
    suggestions = []
    valid_relation_count = 0
    for relation in relations:
        matches = find_synchronized_pairs(
            metadata_by_topic.get(relation["source_topic"], []),
            metadata_by_topic.get(relation["target_topic"], []),
            sync_threshold_ns,
            max_pairs=int(config["max_windows_per_relation"]),
        )
        windows = []
        valid_windows = 0
        for source_meta, target_meta, delta_ns in matches:
            source_cloud = load_cloud(source_meta, cloud_cache)
            target_cloud = load_cloud(target_meta, cloud_cache)
            if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
                continue
            snapshot = build_aligned_snapshot(
                target_topic=relation["target_topic"],
                reference_target_meta=target_meta,
                topic_transforms={
                    relation["target_topic"]: np.eye(4, dtype=float),
                    relation["source_topic"]: np.asarray(
                        relation["initial_transform"],
                        dtype=float,
                    ),
                },
                metadata_by_topic={
                    relation["target_topic"]: [target_meta],
                    relation["source_topic"]: [source_meta],
                },
                sync_threshold_ns=sync_threshold_ns,
                cloud_cache=cloud_cache,
                load_cloud=load_cloud,
            )
            visual_metrics = compute_visual_plane_metrics(
                snapshot,
                downsample_voxel_size=0.10,
                plane_distance_threshold=0.08,
                max_planes=4,
                min_plane_points=400,
                corner_angle_tolerance_deg=20.0,
                corner_distance_threshold_m=0.12,
                slice_bin_size_m=0.25,
                min_slice_points=100,
            )
            unmatched_ratio = estimate_bidirectional_unmatched_ratio(
                source_cloud,
                target_cloud,
                transform=np.asarray(relation["initial_transform"], dtype=float),
                distance_threshold_m=float(config["dynamic_distance_threshold_m"]),
                voxel_size=0.20,
            )
            window = {
                "source_timestamp_ns": int(source_meta.timestamp_ns),
                "target_timestamp_ns": int(target_meta.timestamp_ns),
                "sync_dt_ms": float(delta_ns / 1e6),
                "overlap_ratio": float(relation["overlap_ratio"]),
                "dynamic_unmatched_ratio": unmatched_ratio,
                "plane_normal_diversity": int(visual_metrics["plane_normal_diversity"]),
                "wall_plane_count": int(visual_metrics["wall_plane_count"]),
                "corner_pair_count": int(
                    visual_metrics["corner_metrics"]["corner_pair_count"]
                ),
                "slice_metrics": visual_metrics["slice_metrics"],
            }
            healthy_window = (
                float(window["overlap_ratio"]) >= float(config["min_overlap_ratio"])
                and int(window["wall_plane_count"])
                >= int(config["min_wall_plane_count"])
                and (
                    unmatched_ratio is None
                    or float(unmatched_ratio)
                    <= float(config["max_dynamic_unmatched_ratio"])
                )
            )
            window["healthy"] = bool(healthy_window)
            if healthy_window:
                valid_windows += 1
            windows.append(window)
        overlap_values = [window["overlap_ratio"] for window in windows]
        sync_values = [window["sync_dt_ms"] for window in windows]
        unmatched_values = [
            window["dynamic_unmatched_ratio"]
            for window in windows
            if window["dynamic_unmatched_ratio"] is not None
        ]
        wall_counts = [window["wall_plane_count"] for window in windows]
        corner_counts = [window["corner_pair_count"] for window in windows]
        relation_status = (
            "pass"
            if valid_windows >= int(config["min_valid_windows_per_relation"])
            else "warning"
        )
        if relation_status == "pass":
            valid_relation_count += 1
        relation_report = {
            "relation_id": relation["relation_id"],
            "source_topic": relation["source_topic"],
            "target_topic": relation["target_topic"],
            "role": relation["role"],
            "required": bool(relation["required"]),
            "window_count": len(windows),
            "valid_window_count": int(valid_windows),
            "status": relation_status,
            "overlap_ratio": summarize_values(overlap_values),
            "timestamp_skew_ms": summarize_values(sync_values),
            "dynamic_unmatched_ratio": summarize_values(unmatched_values),
            "wall_plane_count": summarize_values(wall_counts),
            "corner_pair_count": summarize_values(corner_counts),
            "windows": windows,
        }
        relation_reports.append(relation_report)
        if relation_status != "pass":
            suggestions.append(
                f"{relation['source_topic']} -> {relation['target_topic']} is weak; add more static windows or stronger shared structure."
            )
        elif relation_report["corner_pair_count"]["max"] is not None and float(
            relation_report["corner_pair_count"]["max"]
        ) < float(config["min_corner_pair_count"]):
            suggestions.append(
                f"{relation['source_topic']} -> {relation['target_topic']} lacks strong corners; rely more on long walls than yaw-sensitive refinement."
            )
    required_relations = [relation for relation in relations if relation["required"]]
    healthy_required_relations = [
        relation
        for relation in relation_reports
        if relation["required"] and relation["status"] == "pass"
    ]
    scene_class = "open_space_weak"
    max_corner_support = max(
        (
            relation["corner_pair_count"]["max"]
            for relation in relation_reports
            if relation["corner_pair_count"]["max"] is not None
        ),
        default=0,
    )
    max_wall_support = max(
        (
            relation["wall_plane_count"]["max"]
            for relation in relation_reports
            if relation["wall_plane_count"]["max"] is not None
        ),
        default=0,
    )
    if max_corner_support >= int(config["min_corner_pair_count"]):
        scene_class = "corner_rich"
    elif max_wall_support >= int(config["min_wall_plane_count"]):
        scene_class = "wall_dominant"
    return {
        "thresholds": config,
        "summary": {
            "relation_count": len(relations),
            "valid_relation_count": valid_relation_count,
            "required_relation_count": len(required_relations),
            "healthy_required_relation_count": len(healthy_required_relations),
            "scene_class": scene_class,
            "status": (
                "pass"
                if len(healthy_required_relations) == len(required_relations)
                else "warning"
            ),
        },
        "relations": relation_reports,
        "suggestions": suggestions,
    }


def load_cached_cloud(meta, cloud_cache: dict) -> o3d.geometry.PointCloud:
    key = (meta.topic, int(meta.timestamp_ns))
    if key not in cloud_cache:
        cloud_cache[key] = load_pointcloud_from_meta(meta)
    return cloud_cache[key]


def build_candidate_pairs(
    topic_infos: dict[str, dict],
    metadata_by_topic: dict,
    tf_graph: dict,
    sync_threshold_ns: int,
    overlap_voxel_size: float,
    cloud_cache: dict | None = None,
) -> list[dict]:
    if cloud_cache is None:
        cloud_cache = {}
    topics = list(topic_infos)
    candidate_pairs: list[dict] = []

    for index_a in range(len(topics)):
        for index_b in range(index_a + 1, len(topics)):
            topic_a = topics[index_a]
            topic_b = topics[index_b]
            frame_a = topic_infos[topic_a]["frame_id"]
            frame_b = topic_infos[topic_b]["frame_id"]
            pair_record = {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "frame_a": frame_a,
                "frame_b": frame_b,
                "same_frame": frame_a == frame_b and bool(frame_a),
                "overlap_ratio": None,
                "sync_dt_ms": None,
                "reason": None,
            }

            if not frame_a or not frame_b:
                pair_record["reason"] = "missing_frame_id"
                candidate_pairs.append(pair_record)
                continue
            if frame_a == frame_b:
                pair_record["reason"] = "duplicate_frame"
                pair_record["overlap_ratio"] = 1.0
                pair_record["sync_dt_ms"] = 0.0
                candidate_pairs.append(pair_record)
                continue

            initial_transform = lookup_transform(tf_graph, frame_a, frame_b)
            if initial_transform is None:
                pair_record["reason"] = "no_tf_path"
                candidate_pairs.append(pair_record)
                continue

            matches = find_synchronized_pairs(
                metadata_by_topic[topic_a],
                metadata_by_topic[topic_b],
                sync_threshold_ns,
                max_pairs=1,
            )
            if not matches:
                pair_record["reason"] = "no_synced_frames"
                candidate_pairs.append(pair_record)
                continue

            source_meta, target_meta, delta_ns = matches[0]
            source_cloud = load_cached_cloud(source_meta, cloud_cache)
            target_cloud = load_cached_cloud(target_meta, cloud_cache)
            overlap_ratio = voxel_overlap_ratio(
                source_cloud, target_cloud, initial_transform, overlap_voxel_size
            )

            pair_record["overlap_ratio"] = float(overlap_ratio)
            pair_record["sync_dt_ms"] = float(delta_ns / 1e6)
            pair_record["initial_transform"] = initial_transform.tolist()
            candidate_pairs.append(pair_record)

    candidate_pairs.sort(
        key=lambda item: (
            -(item["overlap_ratio"] if item["overlap_ratio"] is not None else -1.0),
            item["topic_a"],
            item["topic_b"],
        )
    )
    return candidate_pairs


def summarize_method_runs(method: int, runs: list[dict]) -> dict:
    fitness_values = [run["fitness"] for run in runs]
    rmse_values = [run["inlier_rmse"] for run in runs]
    delta_translation_values = [
        run["delta_to_initial"]["translation_norm_m"] for run in runs
    ]
    delta_rotation_values = [run["delta_to_initial"]["rotation_deg"] for run in runs]
    condition_number_values = [
        run["information_matrix"]["condition_number"] for run in runs
    ]
    avg_fitness = float(np.mean(fitness_values))
    avg_rmse = float(np.mean(rmse_values))
    avg_delta_translation = float(np.mean(delta_translation_values))
    avg_delta_rotation = float(np.mean(delta_rotation_values))
    degenerate_runs = int(
        sum(1 for run in runs if run["information_matrix"]["degenerate"])
    )
    best_run = sorted(
        runs,
        key=lambda run: (
            run["information_matrix"]["degenerate"],
            -run["fitness"],
            run["inlier_rmse"],
            run["delta_to_initial"]["rotation_deg"],
        ),
    )[0]
    return {
        "method": method,
        "runs": len(runs),
        "avg_fitness": avg_fitness,
        "avg_inlier_rmse": avg_rmse,
        "avg_delta_translation_m": avg_delta_translation,
        "avg_delta_rotation_deg": avg_delta_rotation,
        "fitness_distribution": summarize_values(fitness_values),
        "rmse_distribution": summarize_values(rmse_values),
        "delta_translation_distribution_m": summarize_values(delta_translation_values),
        "delta_rotation_distribution_deg": summarize_values(delta_rotation_values),
        "condition_number_distribution": summarize_values(condition_number_values),
        "degenerate_runs": degenerate_runs,
        "best_run": best_run,
    }


def clean_generated_yaml_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for file_path in directory.glob("*_extrinsics.yaml"):
        if file_path.is_file():
            file_path.unlink()


def prepare_output_layout(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for legacy_file in (
        output_dir / "manifest.yaml",
        output_dir / "topology.yaml",
        output_dir / "extraction.yaml",
        output_dir / "tf_tree.yaml",
        output_dir / "calibration.yaml",
        output_dir / "loop_closed_tf.yaml",
        output_dir / "merged_cloud.pcd",
    ):
        if legacy_file.exists() and legacy_file.is_file():
            legacy_file.unlink()

    for diagnostics_file in (
        diagnostics_dir / "manifest.yaml",
        diagnostics_dir / "topology.yaml",
        diagnostics_dir / "extraction.yaml",
        diagnostics_dir / "tf_tree.yaml",
        diagnostics_dir / "workflow.yaml",
        diagnostics_dir / "calibration.yaml",
        diagnostics_dir / "loop_closure.yaml",
        diagnostics_dir / "scene_sufficiency.yaml",
        diagnostics_dir / "visual_evaluation.yaml",
        diagnostics_dir / "merged_cloud.pcd",
        diagnostics_dir / "merged_cloud_baseline_colored.ply",
        diagnostics_dir / "merged_cloud_solution.pcd",
        diagnostics_dir / "merged_cloud_solution_colored.ply",
        diagnostics_dir / "merged_cloud_loop_closure.pcd",
        diagnostics_dir / "merged_cloud_loop_closure_colored.ply",
    ):
        if diagnostics_file.exists() and diagnostics_file.is_file():
            diagnostics_file.unlink()

    initial_guess_dir = output_dir / "initial_guess"
    calibrated_dir = output_dir / "calibrated"
    loop_closed_dir = output_dir / "loop_closed"
    clean_generated_yaml_dir(initial_guess_dir)
    clean_generated_yaml_dir(calibrated_dir)
    clean_generated_yaml_dir(loop_closed_dir)
    return initial_guess_dir, calibrated_dir, loop_closed_dir, diagnostics_dir


def build_topic_transforms_from_edge_results(
    target_topic: str,
    edge_results: list[dict],
) -> dict[str, np.ndarray]:
    topic_transforms = {target_topic: np.eye(4, dtype=float)}
    for edge_result in edge_results:
        topic_transforms[edge_result["source_topic"]] = np.asarray(
            edge_result["best_run"]["transformation"],
            dtype=float,
        )
    return topic_transforms


def build_tf_output_from_topic_transforms(
    target_topic: str,
    target_frame: str,
    topic_infos: dict[str, dict],
    topic_transforms: dict[str, np.ndarray],
    *,
    extra_metadata: dict[str, dict] | None = None,
) -> dict:
    extrinsics = []
    for topic in sorted(topic_transforms):
        if topic == target_topic:
            continue
        topic_info = topic_infos[topic]
        metadata = {
            "topic": topic,
            "sensor_name": topic_info["sensor_name"],
        }
        if extra_metadata and topic in extra_metadata:
            metadata.update(extra_metadata[topic])
        extrinsics.append(
            build_extrinsics_payload(
                parent_frame=target_frame,
                child_frame=topic_info["frame_id"],
                matrix=np.asarray(topic_transforms[topic], dtype=float),
                metadata=metadata,
            )
        )

    return {
        "base_topic": target_topic,
        "base_frame": target_frame,
        "extrinsics": extrinsics,
    }


def write_topic_transform_files(
    output_dir: Path,
    target_frame: str,
    topic_infos: dict[str, dict],
    topic_transforms: dict[str, np.ndarray],
    *,
    extra_metadata: dict[str, dict] | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for topic in sorted(topic_transforms):
        if topic_infos[topic]["frame_id"] == target_frame:
            continue
        file_path = output_dir / extrinsics_filename(
            target_frame,
            topic_infos[topic]["frame_id"],
        )
        metadata = {
            "source_topic": topic,
        }
        if extra_metadata and topic in extra_metadata:
            metadata.update(extra_metadata[topic])
        save_extrinsics_yaml(
            str(file_path),
            parent_frame=target_frame,
            child_frame=topic_infos[topic]["frame_id"],
            matrix=np.asarray(topic_transforms[topic], dtype=float),
            metadata=metadata,
        )
        saved_paths.append(str(file_path))
    return saved_paths


def pick_reference_target_meta(
    metadata_by_topic: dict,
    target_topic: str,
    edge_results: list[dict],
    sync_threshold_ns: int,
):
    metas = metadata_by_topic[target_topic]
    if not metas:
        return None

    if edge_results:
        reference_timestamp_ns = edge_results[0]["best_run"]["target_timestamp_ns"]
        nearest_meta = min(
            metas, key=lambda meta: abs(meta.timestamp_ns - reference_timestamp_ns)
        )
        if abs(nearest_meta.timestamp_ns - reference_timestamp_ns) <= sync_threshold_ns:
            return nearest_meta

    return metas[len(metas) // 2]


def save_merged_cloud(
    output_path: Path,
    metadata_by_topic: dict,
    target_topic: str,
    edge_results: list[dict],
    sync_threshold_ns: int,
    cloud_cache: dict | None = None,
) -> dict:
    if cloud_cache is None:
        cloud_cache = {}
    reference_target_meta = pick_reference_target_meta(
        metadata_by_topic,
        target_topic,
        edge_results,
        sync_threshold_ns,
    )
    if reference_target_meta is None:
        return {"saved": False, "reason": "no_target_frame"}

    merged_cloud = copy.deepcopy(load_cached_cloud(reference_target_meta, cloud_cache))
    used_sources = []
    merge_sync_threshold_ns = max(sync_threshold_ns, int(50e6))

    for edge_result in edge_results:
        matches = find_synchronized_pairs(
            metadata_by_topic[edge_result["source_topic"]],
            [reference_target_meta],
            merge_sync_threshold_ns,
            max_pairs=1,
        )
        if not matches:
            continue

        source_meta, _, delta_ns = matches[0]
        source_cloud = copy.deepcopy(load_cached_cloud(source_meta, cloud_cache))
        source_cloud.transform(
            np.array(edge_result["best_run"]["transformation"], dtype=float)
        )
        merged_cloud += source_cloud
        used_sources.append(
            {
                "topic": edge_result["source_topic"],
                "timestamp_ns": source_meta.timestamp_ns,
                "dt_ms": float(delta_ns / 1e6),
            }
        )

    saved = o3d.io.write_point_cloud(str(output_path), merged_cloud)
    return {
        "saved": bool(saved),
        "output_path": str(output_path),
        "target_topic": target_topic,
        "target_timestamp_ns": reference_target_meta.timestamp_ns,
        "used_sources": used_sources,
    }


def calibrate_selected_edges(
    selected_edges: list[dict],
    metadata_by_topic: dict,
    sync_threshold_ns: int,
    args,
    cloud_cache: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    if cloud_cache is None:
        cloud_cache = {}
    edge_results: list[dict] = []
    skipped_edges: list[dict] = []

    preprocessing_params = {
        "voxel_size": args.voxel_size,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": 0.05,
        "height_range": args.max_height,
        "remove_ground": args.remove_ground,
        "remove_walls": args.remove_walls,
    }

    for edge in selected_edges:
        logging.info(
            "Calibrating %s -> %s using target cloud %s with overlap %.3f",
            edge["source_topic"],
            edge["target_topic"],
            edge["registration_target_topic"],
            edge["overlap_ratio"],
        )
        matches = find_synchronized_pairs(
            metadata_by_topic[edge["source_topic"]],
            metadata_by_topic[edge["registration_target_topic"]],
            sync_threshold_ns,
            max_pairs=args.max_samples,
        )
        if not matches:
            skipped_edges.append(
                {
                    **edge,
                    "reason": "no_synced_frames_for_calibration",
                }
            )
            continue

        method_runs: dict[int, list[dict]] = defaultdict(list)
        for source_meta, target_meta, delta_ns in matches:
            source_cloud = load_cached_cloud(source_meta, cloud_cache)
            target_cloud = load_cached_cloud(target_meta, cloud_cache)
            if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
                continue

            for method in args.methods:
                final_transform, _, reg_result = calibrate_lidar_extrinsic(
                    source_cloud,
                    target_cloud,
                    is_draw_registration=False,
                    preprocessing_params=preprocessing_params,
                    method=method,
                    initial_transform=np.array(edge["initial_transform"], dtype=float),
                )
                if final_transform is None or reg_result is None:
                    continue

                info_metrics = compute_information_metrics(
                    source_cloud,
                    target_cloud,
                    final_transform,
                    max_correspondence_distance=max(args.voxel_size * 5, 0.1),
                    downsample_voxel_size=max(args.voxel_size, 0.1),
                )
                run_record = {
                    "method": method,
                    "source_timestamp_ns": source_meta.timestamp_ns,
                    "target_timestamp_ns": target_meta.timestamp_ns,
                    "sync_dt_ms": float(delta_ns / 1e6),
                    "fitness": float(reg_result.fitness),
                    "inlier_rmse": float(reg_result.inlier_rmse),
                    "transformation": final_transform.tolist(),
                    "delta_to_initial": transform_delta_metrics(
                        np.array(edge["initial_transform"], dtype=float),
                        final_transform,
                    ),
                    "information_matrix": info_metrics,
                }
                method_runs[method].append(run_record)

        if not method_runs:
            skipped_edges.append(
                {
                    **edge,
                    "reason": "calibration_failed",
                }
            )
            continue

        method_summaries = [
            summarize_method_runs(method, runs) for method, runs in method_runs.items()
        ]
        method_summaries.sort(
            key=lambda summary: (
                summary["degenerate_runs"] == summary["runs"],
                -summary["avg_fitness"],
                summary["avg_inlier_rmse"],
                summary["avg_delta_rotation_deg"],
            )
        )
        chosen_summary = method_summaries[0]
        chosen_run = chosen_summary["best_run"]
        chosen_method_runs = method_runs[chosen_summary["method"]]
        repeatability = summarize_repeatability_runs(
            chosen_method_runs,
            np.asarray(chosen_run["transformation"], dtype=float),
        )
        quality_gate_reasons = []
        if chosen_run["information_matrix"]["degenerate"]:
            quality_gate_reasons.append("degenerate_information_matrix")
        if float(chosen_run["information_matrix"]["condition_number"]) > float(
            args.max_condition_number
        ):
            quality_gate_reasons.append("condition_number_above_threshold")
        if float(chosen_run["fitness"]) < float(args.min_fitness):
            quality_gate_reasons.append("fitness_below_threshold")
        if quality_gate_reasons:
            skipped_edges.append(
                {
                    **edge,
                    "reason": "quality_gate_failed",
                    "quality_gate_reasons": quality_gate_reasons,
                    "candidate_best_run": chosen_run,
                    "candidate_method_summaries": method_summaries,
                }
            )
            continue
        edge_results.append(
            {
                **edge,
                "method_summaries": method_summaries,
                "chosen_method": chosen_summary["method"],
                "best_run": chosen_run,
                "repeatability": repeatability,
            }
        )

    edge_results.sort(
        key=lambda item: (-item["best_run"]["fitness"], item["best_run"]["inlier_rmse"])
    )
    return edge_results, skipped_edges


def build_extraction_output(
    record_files: list[str],
    conf_dir: str,
    topic_counts: dict[str, int],
    pointcloud_topics: list[str],
    topic_infos: dict[str, dict],
    record_tf_edges: list,
    conf_tf_edges: list,
    root_analysis: dict,
    candidate_pairs: list[dict],
    selected_edges: list[dict],
    skipped_precheck: list[dict],
) -> dict:
    overlap_values = [
        float(pair["overlap_ratio"])
        for pair in candidate_pairs
        if pair.get("overlap_ratio") is not None
    ]
    sync_values = [
        float(pair["sync_dt_ms"])
        for pair in candidate_pairs
        if pair.get("sync_dt_ms") is not None
    ]
    return {
        "record_files": record_files,
        "conf_dir": conf_dir,
        "summary": {
            "pointcloud_topic_count": len(pointcloud_topics),
            "record_tf_edges": len(record_tf_edges),
            "conf_tf_edges": len(conf_tf_edges),
            "candidate_pair_count": len(candidate_pairs),
            "selected_edge_count": len(selected_edges),
            "skipped_precheck_count": len(skipped_precheck),
            "candidate_overlap_ratio": summarize_values(overlap_values),
            "candidate_sync_dt_ms": summarize_values(sync_values),
        },
        "topic_counts": topic_counts,
        "pointcloud_topics": pointcloud_topics,
        "topics": topic_infos,
        "root_analysis": root_analysis,
        "candidate_pairs": candidate_pairs,
        "selected_edges": selected_edges,
        "skipped_precheck": skipped_precheck,
    }


def _quality_status(
    value: float | None, threshold: float, *, smaller_is_better: bool
) -> str:
    if value is None:
        return "unknown"
    if smaller_is_better:
        return "pass" if value <= threshold else "warning"
    return "pass" if value >= threshold else "warning"


def build_tf_output(
    base_frame: str, target_topic: str, edge_results: list[dict]
) -> dict:
    extrinsics = []
    for edge_result in edge_results:
        extrinsics.append(
            build_extrinsics_payload(
                parent_frame=base_frame,
                child_frame=edge_result["source_frame"],
                matrix=np.array(edge_result["best_run"]["transformation"], dtype=float),
                stamp_ns=edge_result["best_run"]["target_timestamp_ns"],
                metadata={
                    "topic": edge_result["source_topic"],
                    "registration_target_topic": edge_result[
                        "registration_target_topic"
                    ],
                    "chosen_method": int(edge_result["chosen_method"]),
                },
            )
        )

    return {
        "base_topic": target_topic,
        "base_frame": base_frame,
        "extrinsics": extrinsics,
    }


def build_metrics_output(
    record_files: list[str],
    target_topic: str,
    target_frame: str,
    root_analysis: dict,
    edge_results: list[dict],
    skipped_edges: list[dict],
    extraction_output: dict,
    args,
    output_dir: Path,
    merged_summary: dict | None,
) -> dict:
    per_edge = []
    condition_numbers = []
    fitness_values = []
    rmse_values = []
    overlap_values = []

    for edge_result in edge_results:
        info_metrics = edge_result["best_run"]["information_matrix"]
        condition_numbers.append(float(info_metrics["condition_number"]))
        fitness_values.append(float(edge_result["best_run"]["fitness"]))
        rmse_values.append(float(edge_result["best_run"]["inlier_rmse"]))
        overlap_values.append(float(edge_result["overlap_ratio"]))
        relation_target_frame = edge_result.get("target_frame", target_frame)
        extrinsics_file = None
        if relation_target_frame == target_frame:
            extrinsics_file = str(
                output_dir
                / "calibrated"
                / extrinsics_filename(target_frame, edge_result["source_frame"])
            )
        per_edge.append(
            {
                "extrinsics_file": extrinsics_file,
                "relation_id": edge_result.get("relation_id"),
                "role": edge_result.get("role", "primary"),
                "required": bool(edge_result.get("required", True)),
                "source_topic": edge_result["source_topic"],
                "registration_target_topic": edge_result["registration_target_topic"],
                "source_frame": edge_result["source_frame"],
                "target_topic": edge_result.get("target_topic", target_topic),
                "target_frame": relation_target_frame,
                "chosen_method": int(edge_result["chosen_method"]),
                "fitness": float(edge_result["best_run"]["fitness"]),
                "inlier_rmse": float(edge_result["best_run"]["inlier_rmse"]),
                "overlap_ratio": float(edge_result["overlap_ratio"]),
                "sync_dt_ms": float(edge_result["best_run"]["sync_dt_ms"]),
                "delta_to_initial": edge_result["best_run"]["delta_to_initial"],
                "repeatability": edge_result.get("repeatability"),
                "information_matrix": {
                    "condition_number": float(info_metrics["condition_number"]),
                    "degenerate": bool(info_metrics["degenerate"]),
                    "eigenvalues": info_metrics["eigenvalues"],
                },
                "method_summaries": edge_result["method_summaries"],
            }
        )

    summary = {
        "calibrated_edges": len(edge_results),
        "skipped_edges": len(skipped_edges),
        "average_fitness": float(np.mean(fitness_values)) if fitness_values else None,
        "average_inlier_rmse": float(np.mean(rmse_values)) if rmse_values else None,
        "min_overlap_ratio": float(min(overlap_values)) if overlap_values else None,
        "max_condition_number": (
            float(max(condition_numbers)) if condition_numbers else None
        ),
    }

    coarse_metrics = {
        **summary,
        "statuses": {
            "coverage": "pass" if edge_results else "warning",
            "overlap": _quality_status(
                summary["min_overlap_ratio"],
                float(args.min_overlap),
                smaller_is_better=False,
            ),
            "fitness": _quality_status(
                summary["average_fitness"],
                float(args.min_fitness),
                smaller_is_better=False,
            ),
            "condition_number": _quality_status(
                summary["max_condition_number"],
                float(args.max_condition_number),
                smaller_is_better=True,
            ),
            "degeneracy": (
                "warning"
                if any(
                    edge_result["best_run"]["information_matrix"]["degenerate"]
                    for edge_result in edge_results
                )
                else "pass"
            ),
        },
    }

    fine_metrics = {
        "per_edge": per_edge,
        "skipped_edges": skipped_edges,
        "target_selection": {
            "preferred_root_frame": root_analysis.get("preferred_root_frame"),
            "strategy": root_analysis.get("target_selection_strategy"),
            "missing_transform_frames_to_target": root_analysis.get(
                "missing_transform_frames_to_target", []
            ),
        },
        "extraction_summary": extraction_output["summary"],
    }

    return {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "target_selection": {
            "preferred_root_frame": root_analysis.get("preferred_root_frame"),
            "strategy": root_analysis.get("target_selection_strategy"),
            "missing_transform_frames_to_target": root_analysis.get(
                "missing_transform_frames_to_target", []
            ),
        },
        "summary": summary,
        "per_edge": per_edge,
        "skipped_edges": skipped_edges,
        "coarse_metrics": coarse_metrics,
        "fine_metrics": fine_metrics,
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "initial_guess_dir": str(output_dir / "initial_guess"),
            "calibrated_dir": str(output_dir / "calibrated"),
            "diagnostics_dir": str(output_dir / "diagnostics"),
            "merged_cloud": merged_summary,
        },
    }


def _build_lidar2lidar_final_acceptance(metrics_output: dict) -> dict:
    coarse_metrics = metrics_output["coarse_metrics"]
    statuses = coarse_metrics["statuses"]
    gates = [
        {
            "name": "coverage",
            "status": statuses.get("coverage", "unknown"),
            "severity": "required",
            "evidence": f"calibrated_edges={coarse_metrics.get('calibrated_edges')}",
            "action": "Resolve missing TF, sync, or overlap before accepting a run.",
        },
        {
            "name": "overlap",
            "status": statuses.get("overlap", "unknown"),
            "severity": "required",
            "evidence": f"min_overlap_ratio={coarse_metrics.get('min_overlap_ratio')}",
            "action": "Reject low-overlap relations or collect scenes with shared structure.",
        },
        {
            "name": "fitness",
            "status": statuses.get("fitness", "unknown"),
            "severity": "required",
            "evidence": f"average_fitness={coarse_metrics.get('average_fitness')}",
            "action": "Review registration inputs and dynamic objects before trusting the edge.",
        },
        {
            "name": "condition_number",
            "status": statuses.get("condition_number", "unknown"),
            "severity": "required",
            "evidence": f"max_condition_number={coarse_metrics.get('max_condition_number')}",
            "action": "Treat geometrically degenerate edges as diagnostic-only.",
        },
        {
            "name": "degeneracy",
            "status": statuses.get("degeneracy", "unknown"),
            "severity": "required",
            "evidence": "information_matrix.degenerate",
            "action": "Use richer geometry or stronger topology priors before promotion.",
        },
        {
            "name": "scene_sufficiency",
            "status": statuses.get("scene_sufficiency", "unknown"),
            "severity": "required",
            "evidence": "diagnostics/scene_sufficiency.yaml",
            "action": "Collect scenes with enough overlap, wall support, and corner/slice structure.",
        },
        {
            "name": "relation_connectivity",
            "status": statuses.get("relation_connectivity", "unknown"),
            "severity": "required",
            "evidence": "solution_graph.baseline_unresolved_topics",
            "action": "Do not promote rigs with unresolved required relations.",
        },
        {
            "name": "repeatability",
            "status": statuses.get("repeatability", "unknown"),
            "severity": "required",
            "evidence": (
                "edge_repeatability_translation_p95_m="
                f"{coarse_metrics.get('edge_repeatability_translation_p95_m')}, "
                "edge_repeatability_rotation_p95_deg="
                f"{coarse_metrics.get('edge_repeatability_rotation_p95_deg')}"
            ),
            "action": "Require multi-window consistency or keep the edge as diagnostic-only.",
        },
        {
            "name": "visual_geometry",
            "status": statuses.get("visual_geometry", "unknown"),
            "severity": "required",
            "evidence": (
                f"wall_thickness_p95_m={coarse_metrics.get('wall_thickness_p95_m')}, "
                "corner_spread_radius_p95_m="
                f"{coarse_metrics.get('corner_spread_radius_p95_m')}, "
                f"min_slice_sharpness_score={coarse_metrics.get('min_slice_sharpness_score')}"
            ),
            "action": "Inspect colored clouds and require wall/corner/slice agreement.",
        },
        {
            "name": "loop_closure",
            "status": statuses.get("loop_closure", "unknown"),
            "severity": "advisory",
            "evidence": "diagnostics/loop_closure.yaml",
            "action": "Use graph consistency as a rig-level check when loop edges exist.",
        },
    ]
    return build_final_acceptance(
        module="lidar2lidar",
        gates=gates,
        pass_recommendation="release_rig_extrinsics",
        review_recommendation="review_metrics_and_visuals",
        fail_recommendation="reject_and_recollect_or_fix_topology",
    )


def write_calibrated_edge_files(
    output_dir: Path, base_frame: str, edge_results: list[dict]
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for edge_result in edge_results:
        child_frame = edge_result["source_frame"]
        file_path = output_dir / extrinsics_filename(base_frame, child_frame)
        save_extrinsics_yaml(
            str(file_path),
            parent_frame=base_frame,
            child_frame=child_frame,
            matrix=np.array(edge_result["best_run"]["transformation"], dtype=float),
            stamp_ns=edge_result["best_run"]["target_timestamp_ns"],
            metadata={
                "source_topic": edge_result["source_topic"],
                "registration_target_topic": edge_result["registration_target_topic"],
                "chosen_method": int(edge_result["chosen_method"]),
            },
        )
        saved_paths.append(str(file_path))
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatic multi-LiDAR calibration pipeline for Apollo record files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--record-path",
        help="Path to a .record file or a directory containing split record files.",
    )
    input_group.add_argument(
        "--prepared-dataset-yaml",
        default=None,
        help="Path to diagnostics/prepared_rig_dataset.yaml generated by lidar2lidar-rig-dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2lidar/auto_calib",
        help="Directory for reports and output files.",
    )
    parser.add_argument(
        "--conf-dir",
        default="lidar2lidar/conf",
        help="Directory that stores fallback extrinsics YAML files.",
    )
    parser.add_argument(
        "--prefer-conf-tf",
        action="store_true",
        help="Use conf-dir extrinsics as the preferred initial TF when the same edge also exists in the record.",
    )
    parser.add_argument(
        "--workflow-yaml",
        default=None,
        help="Optional workflow YAML that defines relation planning, scene sufficiency thresholds, repeatability windows, and visualization settings.",
    )
    parser.add_argument(
        "--bootstrap-conf",
        action="store_true",
        help="Export record-derived static TF edges into the conf directory.",
    )
    parser.add_argument(
        "--target-topic",
        default=None,
        help="Target point cloud topic. If omitted, the pipeline selects one automatically.",
    )
    parser.add_argument(
        "--source-topics",
        nargs="*",
        default=None,
        help="Optional explicit source point cloud topics.",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=10.0,
        help="Maximum timestamp difference for frame synchronization.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.30,
        help="Minimum voxel overlap ratio required to calibrate a topic pair.",
    )
    parser.add_argument(
        "--overlap-voxel-size",
        type=float,
        default=0.5,
        help="Voxel size used during overlap pre-check.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2,
        help="Maximum synchronized frame pairs to evaluate per selected edge.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Calibration methods to compare (1: point-to-plane, 2: GICP, 3: point-to-point).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="Voxel size for registration preprocessing.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Optional max height filter for preprocessing.",
    )
    parser.add_argument(
        "--remove-ground",
        action="store_true",
        help="Remove the dominant ground plane during preprocessing.",
    )
    parser.add_argument(
        "--remove-walls",
        action="store_true",
        help="Remove vertical planes during preprocessing.",
    )
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.0,
        help="Minimum fitness required to keep a calibrated edge.",
    )
    parser.add_argument(
        "--max-condition-number",
        type=float,
        default=1e6,
        help="Maximum information-matrix condition number allowed for accepted edges.",
    )
    parser.add_argument(
        "--save-merged-pcd",
        action="store_true",
        help="Save a merged point cloud for one synchronized reference timestamp.",
    )
    parser.add_argument(
        "--loop-closure",
        action="store_true",
        help="Add a graph-based loop-closure/global-consistency refinement on top of the pairwise baseline.",
    )
    parser.add_argument(
        "--loop-min-quality-score",
        type=float,
        default=0.005,
        help="Minimum loop-edge quality score required to keep a graph measurement in production mode.",
    )
    parser.add_argument(
        "--loop-max-prior-translation-delta-m",
        type=float,
        default=3.0,
        help="Reject loop edges whose calibrated transform drifts too far from the trusted TF prior in translation.",
    )
    parser.add_argument(
        "--loop-max-prior-rotation-delta-deg",
        type=float,
        default=20.0,
        help="Reject loop edges whose calibrated transform drifts too far from the trusted TF prior in rotation.",
    )
    parser.add_argument(
        "--loop-prior-translation-weight",
        type=float,
        default=1.5,
        help="Translation regularization weight that keeps loop-closure poses near the trusted rig prior.",
    )
    parser.add_argument(
        "--loop-prior-rotation-weight",
        type=float,
        default=6.0,
        help="Rotation regularization weight that keeps loop-closure poses near the trusted rig prior.",
    )
    args = parser.parse_args()
    workflow_config = load_workflow_config(args.workflow_yaml)

    prepared_dataset = (
        load_prepared_rig_dataset(args.prepared_dataset_yaml)
        if args.prepared_dataset_yaml is not None
        else None
    )
    record_files = (
        prepared_dataset.record_files
        if prepared_dataset is not None
        else discover_record_files(args.record_path)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_guess_dir, calibrated_dir, loop_closed_dir, diagnostics_dir = (
        prepare_output_layout(output_dir)
    )

    logging.info("Using record files: %s", record_files)
    if prepared_dataset is not None:
        pointcloud_topics = list(prepared_dataset.lidar_topics)
        if not pointcloud_topics:
            raise RuntimeError(
                f"No LiDAR topics were found in prepared dataset {args.prepared_dataset_yaml}."
            )
        topic_infos = {
            topic: {
                "count": int(
                    prepared_dataset.topic_infos.get(topic, {}).get(
                        "count", len(prepared_dataset.metadata_by_topic.get(topic, []))
                    )
                ),
                "frame_id": prepared_dataset.topic_infos.get(topic, {}).get(
                    "frame_id", ""
                ),
                "sensor_name": prepared_dataset.topic_infos.get(topic, {}).get(
                    "sensor_name", topic_sensor_name(topic)
                ),
            }
            for topic in pointcloud_topics
        }
        topic_counts = {
            topic: int(info["count"]) for topic, info in topic_infos.items()
        }
    else:
        topic_counts = list_topics(record_files)
        pointcloud_topics = infer_pointcloud_topics(topic_counts)
        if not pointcloud_topics:
            raise RuntimeError("No PointCloud2 topics were found in the input record.")

        topic_frame_ids = get_topic_frame_ids(record_files, pointcloud_topics)
        topic_infos = {
            topic: {
                "count": int(topic_counts.get(topic, 0)),
                "frame_id": topic_frame_ids.get(topic, ""),
                "sensor_name": topic_sensor_name(topic),
            }
            for topic in pointcloud_topics
        }

    pointcloud_frames = sorted(
        {info["frame_id"] for info in topic_infos.values() if info["frame_id"]}
    )

    record_tf_edges = (
        prepared_dataset.tf_edges
        if prepared_dataset is not None
        else extract_tf_edges(record_files)
    )
    conf_tf_edges = load_transform_edges_from_dir(args.conf_dir)
    tf_edges = (
        merge_transform_edges(conf_tf_edges, record_tf_edges)
        if args.prefer_conf_tf
        else merge_transform_edges(record_tf_edges, conf_tf_edges)
    )
    tf_graph = {}
    if tf_edges:
        tf_graph = build_transform_graph(tf_edges)

    pointcloud_related_static_edges = [
        edge
        for edge in record_tf_edges
        if edge.is_static
        and (
            edge.parent_frame in pointcloud_frames
            or edge.child_frame in pointcloud_frames
        )
    ]
    if args.bootstrap_conf and pointcloud_related_static_edges:
        Path(args.conf_dir).mkdir(parents=True, exist_ok=True)
        save_transform_edges_to_dir(args.conf_dir, pointcloud_related_static_edges)

    metadata_by_topic = (
        {
            topic: list(prepared_dataset.metadata_by_topic.get(topic, []))
            for topic in pointcloud_topics
        }
        if prepared_dataset is not None
        else collect_pointcloud_metadata(record_files, pointcloud_topics)
    )
    cloud_cache = {}
    sync_threshold_ns = int(args.sync_threshold_ms * 1e6)
    root_analysis = analyze_pointcloud_roots(pointcloud_frames, tf_edges)
    candidate_pairs = build_candidate_pairs(
        topic_infos,
        metadata_by_topic,
        tf_graph,
        sync_threshold_ns,
        args.overlap_voxel_size,
        cloud_cache=cloud_cache,
    )

    default_target_topic = choose_target_topic(
        topic_infos, candidate_pairs, root_analysis, args.target_topic
    )
    workflow_plan = resolve_workflow_plan(
        workflow_config=workflow_config,
        workflow_path=args.workflow_yaml,
        pointcloud_topics=pointcloud_topics,
        topic_infos=topic_infos,
        tf_edges=tf_edges,
        default_target_topic=default_target_topic,
        cli_source_topics=args.source_topics,
        default_min_overlap=float(args.min_overlap),
        default_enable_global_optimization=bool(args.loop_closure),
        default_save_visuals=bool(args.save_merged_pcd),
    )
    target_topic = workflow_plan["target_topic"]
    target_frame = topic_infos[target_topic]["frame_id"]
    root_analysis["topics_on_preferred_root_frame"] = [
        topic
        for topic, info in topic_infos.items()
        if info["frame_id"] == root_analysis.get("preferred_root_frame")
    ]
    root_analysis["target_selection_strategy"] = (
        "workflow_explicit"
        if workflow_plan["source"] != "cli_defaults"
        and workflow_plan["target_topic"] != default_target_topic
        else (
            "tf_static_root"
            if root_analysis.get("preferred_root_frame") == target_frame
            else "overlap_fallback"
        )
    )
    root_analysis["selected_target_topic"] = target_topic
    root_analysis["selected_target_frame"] = target_frame
    root_analysis["missing_transform_frames_to_target"] = find_missing_transform_frames(
        tf_graph,
        target_frame,
        pointcloud_frames,
    )
    logging.info("Selected target topic: %s (%s)", target_topic, target_frame)

    selected_edges, skipped_precheck = build_selected_edges_for_relations(
        workflow_plan["relations"],
        candidate_pairs,
        tf_graph,
        float(args.min_overlap),
    )
    args.max_samples = int(
        max(args.max_samples, workflow_plan["repeatability"]["max_windows_per_edge"])
    )

    edge_results, skipped_edges = calibrate_selected_edges(
        selected_edges,
        metadata_by_topic,
        sync_threshold_ns,
        args,
        cloud_cache=cloud_cache,
    )
    skipped_edges = skipped_precheck + skipped_edges

    extraction_report = build_extraction_output(
        record_files=record_files,
        conf_dir=args.conf_dir,
        topic_counts=topic_counts,
        pointcloud_topics=pointcloud_topics,
        topic_infos=topic_infos,
        record_tf_edges=record_tf_edges,
        conf_tf_edges=conf_tf_edges,
        root_analysis=root_analysis,
        candidate_pairs=candidate_pairs,
        selected_edges=selected_edges,
        skipped_precheck=skipped_precheck,
    )
    extraction_report["workflow"] = workflow_plan

    initial_guess_files = save_transform_edges_to_dir(
        initial_guess_dir, pointcloud_related_static_edges
    )
    scene_sufficiency_report = build_scene_sufficiency_report(
        selected_edges,
        metadata_by_topic,
        sync_threshold_ns,
        load_cached_cloud,
        cloud_cache,
        workflow_plan["scene_sufficiency"],
    )
    scene_sufficiency_report["skipped_relations"] = skipped_precheck

    topology_report = {
        "record_files": record_files,
        "conf_dir": args.conf_dir,
        "record_tf_edges": len(record_tf_edges),
        "conf_tf_edges": len(conf_tf_edges),
        "target_topic": target_topic,
        "target_frame": target_frame,
        "root_analysis": root_analysis,
        "workflow": workflow_plan,
        "topics": topic_infos,
        "candidate_pairs": candidate_pairs,
        "selected_edges": selected_edges,
        "skipped_edges": skipped_edges,
    }
    with open(diagnostics_dir / "topology.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(topology_report, file, sort_keys=False)

    with open(diagnostics_dir / "extraction.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(extraction_report, file, sort_keys=False)

    with open(diagnostics_dir / "workflow.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(workflow_plan, file, sort_keys=False)

    with open(
        diagnostics_dir / "scene_sufficiency.yaml", "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(scene_sufficiency_report, file, sort_keys=False)

    with open(diagnostics_dir / "tf_tree.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                **tf_tree_payload(tf_edges),
                "root_analysis": root_analysis,
            },
            file,
            sort_keys=False,
        )

    calibration_report = {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "workflow": workflow_plan,
        "edge_results": edge_results,
        "skipped_edges": skipped_edges,
        "scene_sufficiency": scene_sufficiency_report,
    }
    with open(diagnostics_dir / "calibration.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(calibration_report, file, sort_keys=False)

    baseline_solution_edges = [
        edge_result
        for edge_result in edge_results
        if bool(edge_result.get("use_for_solution", True))
    ]
    baseline_solution = compose_topic_transforms(
        target_topic,
        baseline_solution_edges,
        required_topics=workflow_plan["selected_topics"],
    )
    baseline_topic_transforms = baseline_solution["topic_transforms"]
    calibrated_extra_metadata = {
        topic: {
            "solution": "baseline_relation_graph",
            "relation_path": baseline_solution["relation_paths"].get(topic, []),
        }
        for topic in baseline_topic_transforms
        if topic != target_topic
    }
    calibrated_files = write_topic_transform_files(
        calibrated_dir,
        target_frame,
        topic_infos,
        baseline_topic_transforms,
        extra_metadata=calibrated_extra_metadata,
    )
    tf_output = build_tf_output_from_topic_transforms(
        target_topic,
        target_frame,
        topic_infos,
        baseline_topic_transforms,
        extra_metadata=calibrated_extra_metadata,
    )
    with open(output_dir / "calibrated_tf.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(tf_output, file, sort_keys=False)

    merged_summary = None
    loop_closure_report = None
    visual_evaluation_report = None
    loop_closed_tf_output = None
    loop_closed_files: list[str] = []
    loop_closed_merged_summary = None
    loop_result: dict = {}
    reference_target_meta = None
    visual_config = workflow_plan["visualization"]
    should_render_visuals = bool(
        visual_config["enabled"]
        or visual_config["save_merged_clouds"]
        or args.save_merged_pcd
    )
    if should_render_visuals and baseline_topic_transforms:
        reference_target_meta = pick_reference_target_meta(
            metadata_by_topic,
            target_topic,
            baseline_solution_edges or edge_results,
            sync_threshold_ns,
        )

    final_topic_transforms = baseline_topic_transforms
    use_legacy_loop_candidates = (
        workflow_plan["source"] == "cli_defaults"
        and workflow_plan["planner_mode"] == "target_star"
        and bool(args.loop_closure)
    )
    if workflow_plan["enable_global_optimization"] or use_legacy_loop_candidates:
        allowed_topics = set(workflow_plan["selected_topics"])
        graph_candidate_edges = select_loop_candidate_edges(
            candidate_pairs if use_legacy_loop_candidates else [],
            topic_infos,
            allowed_topics,
            float(args.min_overlap),
        )
        graph_edge_results = []
        graph_skipped_edges = []
        graph_filtered_out_edges = []
        if use_legacy_loop_candidates:
            graph_edge_results, graph_skipped_edges = calibrate_selected_edges(
                graph_candidate_edges,
                metadata_by_topic,
                sync_threshold_ns,
                args,
                cloud_cache=cloud_cache,
            )
        else:
            graph_candidate_edges = [
                edge_result
                for edge_result in edge_results
                if bool(edge_result.get("use_for_optimization", False))
            ]
            graph_edge_results = list(graph_candidate_edges)
        filtered_graph_edge_results, graph_filtered_out_edges = (
            filter_loop_measurement_edges(
                graph_edge_results,
                min_quality_score=float(args.loop_min_quality_score),
                max_prior_translation_delta_m=float(
                    args.loop_max_prior_translation_delta_m
                ),
                max_prior_rotation_delta_deg=float(
                    args.loop_max_prior_rotation_delta_deg
                ),
            )
        )
        graph_selection = select_loop_graph_edges(
            filtered_graph_edge_results,
            target_topic,
            required_topics=workflow_plan["selected_topics"],
        )
        prior_topic_transforms, prior_sources = build_prior_topic_transforms(
            target_topic,
            target_frame,
            topic_infos,
            tf_graph,
            workflow_plan["selected_topics"],
            fallback_topic_transforms=baseline_topic_transforms,
        )
        initial_topic_transforms = {
            topic: np.asarray(
                baseline_topic_transforms.get(topic, prior_topic_transforms[topic]),
                dtype=float,
            )
            for topic in workflow_plan["selected_topics"]
            if topic in prior_topic_transforms
        }
        loop_result = optimize_loop_closure(
            target_topic,
            initial_topic_transforms,
            graph_selection["graph_edges"],
            prior_topic_transforms=prior_topic_transforms,
            prior_translation_weight=float(args.loop_prior_translation_weight),
            prior_rotation_weight=float(args.loop_prior_rotation_weight),
        )
        loop_closure_report = {
            "enabled": True,
            "workflow_mode": workflow_plan["planner_mode"],
            "candidate_edge_count": len(graph_candidate_edges),
            "accepted_measurement_edge_count": len(graph_edge_results),
            "retained_measurement_edge_count": len(filtered_graph_edge_results),
            "skipped_edge_count": len(graph_skipped_edges),
            "filtered_out_edge_count": len(graph_filtered_out_edges),
            "graph_structure": {
                "topic_count": len(graph_selection["topics"]),
                "topics": graph_selection["topics"],
                "tree_edge_count": len(graph_selection["tree_edges"]),
                "loop_edge_count": len(graph_selection["loop_edges"]),
                "connected_components": graph_selection["connected_components"],
                "has_effective_loop_evidence": bool(
                    graph_selection["loop_edges"]
                    and len(graph_selection["connected_components"]) == 1
                ),
                "tree_edges": [
                    {
                        "source_topic": edge["source_topic"],
                        "target_topic": edge["target_topic"],
                        "fitness": float(edge["best_run"]["fitness"]),
                        "inlier_rmse": float(edge["best_run"]["inlier_rmse"]),
                        "overlap_ratio": float(edge["overlap_ratio"]),
                    }
                    for edge in graph_selection["tree_edges"]
                ],
                "loop_edges": [
                    {
                        "source_topic": edge["source_topic"],
                        "target_topic": edge["target_topic"],
                        "fitness": float(edge["best_run"]["fitness"]),
                        "inlier_rmse": float(edge["best_run"]["inlier_rmse"]),
                        "overlap_ratio": float(edge["overlap_ratio"]),
                    }
                    for edge in graph_selection["loop_edges"]
                ],
            },
            "baseline_components": baseline_solution["connected_components"],
            "baseline_unresolved_topics": baseline_solution["unresolved_topics"],
            "prior_sources": prior_sources,
            "prior_regularization": {
                "translation_weight": float(args.loop_prior_translation_weight),
                "rotation_weight": float(args.loop_prior_rotation_weight),
            },
            "edge_gating": {
                "min_quality_score": float(args.loop_min_quality_score),
                "max_prior_translation_delta_m": float(
                    args.loop_max_prior_translation_delta_m
                ),
                "max_prior_rotation_delta_deg": float(
                    args.loop_max_prior_rotation_delta_deg
                ),
            },
            "accepted_measurement_edges": [
                {
                    "source_topic": edge["source_topic"],
                    "target_topic": edge["target_topic"],
                    "source_frame": edge["source_frame"],
                    "target_frame": edge["target_frame"],
                    "fitness": float(edge["best_run"]["fitness"]),
                    "inlier_rmse": float(edge["best_run"]["inlier_rmse"]),
                    "overlap_ratio": float(edge["overlap_ratio"]),
                    "chosen_method": int(edge["chosen_method"]),
                }
                for edge in graph_edge_results
            ],
            "retained_measurement_edges": [
                {
                    "source_topic": edge["source_topic"],
                    "target_topic": edge["target_topic"],
                    "source_frame": edge["source_frame"],
                    "target_frame": edge["target_frame"],
                    "fitness": float(edge["best_run"]["fitness"]),
                    "inlier_rmse": float(edge["best_run"]["inlier_rmse"]),
                    "overlap_ratio": float(edge["overlap_ratio"]),
                    "quality_score": float(
                        edge["best_run"]["fitness"]
                        * edge["overlap_ratio"]
                        / (1.0 + edge["best_run"]["inlier_rmse"])
                    ),
                    "delta_to_prior": edge["best_run"]["delta_to_initial"],
                    "chosen_method": int(edge["chosen_method"]),
                }
                for edge in filtered_graph_edge_results
            ],
            "skipped_edges": graph_skipped_edges,
            "filtered_out_edges": graph_filtered_out_edges,
            "baseline_consistency": (
                loop_result.get("baseline_consistency")
                if loop_result.get("success")
                else evaluate_graph_consistency(
                    graph_selection["graph_edges"],
                    initial_topic_transforms,
                )
            ),
            "optimized_consistency": (
                loop_result.get("optimized_consistency")
                if loop_result.get("success")
                else None
            ),
            "delta_to_baseline": (
                {
                    topic: delta
                    for topic, delta in loop_result.get(
                        "delta_to_baseline",
                        {},
                    ).items()
                    if topic != target_topic
                }
                if loop_result.get("success")
                else {}
            ),
            "delta_to_prior": (
                {
                    topic: delta
                    for topic, delta in loop_result.get("delta_to_prior", {}).items()
                    if topic != target_topic
                }
                if loop_result.get("success")
                else {}
            ),
            "optimization": {
                "success": bool(loop_result.get("success")),
                "reason": loop_result.get("reason"),
                "message": loop_result.get("message"),
                "cost": loop_result.get("cost"),
                "nfev": loop_result.get("nfev"),
            },
        }
        if loop_result.get("success"):
            loop_closed_topic_transforms = loop_result["optimized_topic_transforms"]
            final_topic_transforms = loop_closed_topic_transforms
            extra_metadata = {
                topic: {
                    "solution": "loop_closure",
                    "delta_to_pairwise_baseline": loop_result["delta_to_baseline"][
                        topic
                    ],
                }
                for topic in loop_closed_topic_transforms
                if topic != target_topic
            }
            loop_closed_tf_output = build_tf_output_from_topic_transforms(
                target_topic,
                target_frame,
                topic_infos,
                loop_closed_topic_transforms,
                extra_metadata=extra_metadata,
            )
            with open(
                output_dir / "loop_closed_tf.yaml", "w", encoding="utf-8"
            ) as file:
                yaml.safe_dump(loop_closed_tf_output, file, sort_keys=False)
            loop_closed_files = write_topic_transform_files(
                loop_closed_dir,
                target_frame,
                topic_infos,
                loop_closed_topic_transforms,
                extra_metadata=extra_metadata,
            )

        loop_closure_report["visual_evaluation_available"] = False

        with open(diagnostics_dir / "loop_closure.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(loop_closure_report, file, sort_keys=False)

    if (
        should_render_visuals
        and reference_target_meta is not None
        and final_topic_transforms
    ):
        baseline_snapshot = build_aligned_snapshot(
            target_topic=target_topic,
            reference_target_meta=reference_target_meta,
            topic_transforms=baseline_topic_transforms,
            metadata_by_topic=metadata_by_topic,
            sync_threshold_ns=max(sync_threshold_ns, int(50e6)),
            cloud_cache=cloud_cache,
            load_cloud=load_cached_cloud,
        )
        baseline_metrics = compute_visual_plane_metrics(
            baseline_snapshot,
            downsample_voxel_size=float(visual_config["downsample_voxel_size"]),
            plane_distance_threshold=float(visual_config["plane_distance_threshold"]),
            max_planes=int(visual_config["max_planes"]),
            min_plane_points=int(visual_config["min_plane_points"]),
            corner_angle_tolerance_deg=float(
                visual_config["corner_angle_tolerance_deg"]
            ),
            corner_distance_threshold_m=float(
                visual_config["corner_distance_threshold_m"]
            ),
            slice_bin_size_m=float(visual_config["slice_bin_size_m"]),
            min_slice_points=int(visual_config["min_slice_points"]),
        )
        if loop_closure_report is not None and loop_result.get("success"):
            loop_snapshot = build_aligned_snapshot(
                target_topic=target_topic,
                reference_target_meta=reference_target_meta,
                topic_transforms=final_topic_transforms,
                metadata_by_topic=metadata_by_topic,
                sync_threshold_ns=max(sync_threshold_ns, int(50e6)),
                cloud_cache=cloud_cache,
                load_cloud=load_cached_cloud,
            )
            baseline_colored_summary = save_snapshot_clouds(
                baseline_snapshot,
                plain_output_path=(
                    str(diagnostics_dir / "merged_cloud.pcd")
                    if args.save_merged_pcd
                    else None
                ),
                colored_output_path=str(
                    diagnostics_dir / "merged_cloud_baseline_colored.ply"
                ),
            )
            loop_closed_merged_summary = save_snapshot_clouds(
                loop_snapshot,
                plain_output_path=str(
                    diagnostics_dir / "merged_cloud_loop_closure.pcd"
                ),
                colored_output_path=str(
                    diagnostics_dir / "merged_cloud_loop_closure_colored.ply"
                ),
            )
            visual_evaluation_report = {
                "baseline_pairwise": {
                    "artifacts": baseline_colored_summary,
                    "wall_metrics": baseline_metrics,
                },
                "loop_closure": {
                    "artifacts": loop_closed_merged_summary,
                    "wall_metrics": compute_visual_plane_metrics(
                        loop_snapshot,
                        downsample_voxel_size=float(
                            visual_config["downsample_voxel_size"]
                        ),
                        plane_distance_threshold=float(
                            visual_config["plane_distance_threshold"]
                        ),
                        max_planes=int(visual_config["max_planes"]),
                        min_plane_points=int(visual_config["min_plane_points"]),
                        corner_angle_tolerance_deg=float(
                            visual_config["corner_angle_tolerance_deg"]
                        ),
                        corner_distance_threshold_m=float(
                            visual_config["corner_distance_threshold_m"]
                        ),
                        slice_bin_size_m=float(visual_config["slice_bin_size_m"]),
                        min_slice_points=int(visual_config["min_slice_points"]),
                    ),
                },
            }
            merged_summary = baseline_colored_summary
            loop_closure_report["visual_evaluation_available"] = True
        else:
            merged_summary = save_snapshot_clouds(
                baseline_snapshot,
                plain_output_path=(
                    str(diagnostics_dir / "merged_cloud.pcd")
                    if args.save_merged_pcd or visual_config["save_merged_clouds"]
                    else None
                ),
                colored_output_path=(
                    str(diagnostics_dir / "merged_cloud_solution_colored.ply")
                    if visual_config["enabled"]
                    else None
                ),
            )
            visual_evaluation_report = {
                "final_solution": {
                    "artifacts": merged_summary,
                    "wall_metrics": baseline_metrics,
                }
            }

    if visual_evaluation_report is not None:
        with open(
            diagnostics_dir / "visual_evaluation.yaml",
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(visual_evaluation_report, file, sort_keys=False)

    metrics_output = build_metrics_output(
        record_files,
        target_topic,
        target_frame,
        root_analysis,
        edge_results,
        skipped_edges,
        extraction_report,
        args,
        output_dir,
        merged_summary,
    )
    repeatability_translation_p95 = [
        edge_result["repeatability"]["translation_to_reference_m"]["p95"]
        for edge_result in edge_results
        if edge_result.get("repeatability")
        and edge_result["repeatability"]["translation_to_reference_m"]["p95"]
        is not None
    ]
    repeatability_rotation_p95 = [
        edge_result["repeatability"]["rotation_to_reference_deg"]["p95"]
        for edge_result in edge_results
        if edge_result.get("repeatability")
        and edge_result["repeatability"]["rotation_to_reference_deg"]["p95"] is not None
    ]
    metrics_output["workflow"] = workflow_plan
    metrics_output["scene_sufficiency"] = scene_sufficiency_report
    metrics_output["coarse_metrics"]["required_relation_count"] = int(
        workflow_plan["summary"]["required_relation_count"]
    )
    metrics_output["coarse_metrics"]["healthy_required_relation_count"] = int(
        scene_sufficiency_report["summary"]["healthy_required_relation_count"]
    )
    metrics_output["coarse_metrics"]["edge_repeatability_translation_p95_m"] = (
        float(max(repeatability_translation_p95))
        if repeatability_translation_p95
        else None
    )
    metrics_output["coarse_metrics"]["edge_repeatability_rotation_p95_deg"] = (
        float(max(repeatability_rotation_p95)) if repeatability_rotation_p95 else None
    )
    metrics_output["coarse_metrics"]["statuses"]["scene_sufficiency"] = (
        scene_sufficiency_report["summary"]["status"]
    )
    metrics_output["coarse_metrics"]["statuses"]["relation_connectivity"] = (
        "pass" if not baseline_solution["unresolved_topics"] else "warning"
    )
    repeatability_pass = (
        metrics_output["coarse_metrics"]["edge_repeatability_translation_p95_m"]
        is not None
        and metrics_output["coarse_metrics"]["edge_repeatability_rotation_p95_deg"]
        is not None
        and metrics_output["coarse_metrics"]["edge_repeatability_translation_p95_m"]
        <= float(workflow_plan["repeatability"]["pass_translation_p95_m"])
        and metrics_output["coarse_metrics"]["edge_repeatability_rotation_p95_deg"]
        <= float(workflow_plan["repeatability"]["pass_rotation_p95_deg"])
    )
    metrics_output["coarse_metrics"]["statuses"]["repeatability"] = (
        "pass" if repeatability_pass else "warning"
    )
    metrics_output["fine_metrics"]["scene_sufficiency"] = scene_sufficiency_report
    metrics_output["fine_metrics"]["workflow"] = workflow_plan
    metrics_output["fine_metrics"]["solution_graph"] = {
        "baseline_components": baseline_solution["connected_components"],
        "baseline_unresolved_topics": baseline_solution["unresolved_topics"],
        "relation_paths": baseline_solution["relation_paths"],
    }
    if loop_closure_report is not None:
        optimized_consistency = loop_closure_report.get("optimized_consistency")
        baseline_consistency = loop_closure_report["baseline_consistency"]
        loop_status = "warning"
        has_effective_loop = (
            int(loop_closure_report["graph_structure"]["loop_edge_count"]) > 0
            and len(loop_closure_report["graph_structure"]["connected_components"]) == 1
        )
        if optimized_consistency is not None and has_effective_loop:
            baseline_p95 = baseline_consistency["translation_residual_m"]["p95"]
            optimized_p95 = optimized_consistency["translation_residual_m"]["p95"]
            baseline_rot_p95 = baseline_consistency["rotation_residual_deg"]["p95"]
            optimized_rot_p95 = optimized_consistency["rotation_residual_deg"]["p95"]
            improved_translation = (
                baseline_p95 is not None
                and optimized_p95 is not None
                and optimized_p95 < baseline_p95
            )
            improved_rotation = (
                baseline_rot_p95 is not None
                and optimized_rot_p95 is not None
                and optimized_rot_p95 < baseline_rot_p95
            )
            loop_status = (
                "pass" if improved_translation or improved_rotation else "warning"
            )
        metrics_output["coarse_metrics"]["statuses"]["loop_closure"] = loop_status
        metrics_output["comparison"] = {
            "baseline_solution": "pairwise_star",
            "loop_closure_solution": (
                "rig_graph_loop_closure"
                if loop_closure_report["optimization"]["success"]
                else "not_available"
            ),
            "loop_closure": {
                "graph_structure": loop_closure_report["graph_structure"],
                "optimization": loop_closure_report["optimization"],
                "baseline_consistency": baseline_consistency,
                "optimized_consistency": optimized_consistency,
                "delta_to_pairwise_baseline": loop_closure_report["delta_to_baseline"],
                "visual_evaluation_available": loop_closure_report.get(
                    "visual_evaluation_available",
                    False,
                ),
            },
        }
        metrics_output["fine_metrics"]["loop_closure"] = loop_closure_report
        if visual_evaluation_report is not None:
            metrics_output["fine_metrics"][
                "visual_evaluation"
            ] = visual_evaluation_report
        metrics_output["artifacts"]["loop_closed_tf"] = (
            str(output_dir / "loop_closed_tf.yaml")
            if loop_closed_tf_output is not None
            else None
        )
        metrics_output["artifacts"]["loop_closed_dir"] = str(loop_closed_dir)
        metrics_output["artifacts"][
            "loop_closed_merged_cloud"
        ] = loop_closed_merged_summary
    if visual_evaluation_report is not None:
        visual_target = visual_evaluation_report.get(
            "loop_closure"
        ) or visual_evaluation_report.get("final_solution")
        if visual_target is not None:
            wall_p95 = visual_target["wall_metrics"]["wall_signed_span_p95_m"]["p95"]
            corner_p95 = visual_target["wall_metrics"]["corner_metrics"][
                "corner_spread_radius_m"
            ]["p95"]
            slice_scores = [
                metric["sharpness_score"]
                for metric in visual_target["wall_metrics"]["slice_metrics"].values()
                if metric["sharpness_score"] is not None
            ]
            min_slice_score = min(slice_scores) if slice_scores else None
            visual_status = "pass"
            if wall_p95 is not None and wall_p95 > float(
                workflow_plan["visualization"]["max_wall_double_edge_m"]
            ):
                visual_status = "warning"
            if corner_p95 is not None and corner_p95 > float(
                workflow_plan["visualization"]["max_corner_spread_p95_m"]
            ):
                visual_status = "warning"
            if min_slice_score is not None and min_slice_score < float(
                workflow_plan["visualization"]["min_slice_sharpness_score"]
            ):
                visual_status = "warning"
            metrics_output["coarse_metrics"]["statuses"][
                "visual_geometry"
            ] = visual_status
            metrics_output["coarse_metrics"]["wall_thickness_p95_m"] = wall_p95
            metrics_output["coarse_metrics"]["corner_spread_radius_p95_m"] = corner_p95
            metrics_output["coarse_metrics"][
                "min_slice_sharpness_score"
            ] = min_slice_score
    final_acceptance = _build_lidar2lidar_final_acceptance(metrics_output)
    metrics_output["final_acceptance"] = final_acceptance
    metrics_output["summary"]["final_acceptance_status"] = final_acceptance["status"]
    metrics_output["summary"]["release_ready"] = final_acceptance["release_ready"]
    acceptance_artifacts = write_acceptance_artifacts(diagnostics_dir, final_acceptance)
    table_artifacts = {
        "edge_metrics_csv": write_table_csv(
            diagnostics_dir / "edge_metrics.csv",
            metrics_output.get("per_edge", []),
        ),
        "skipped_edges_csv": write_table_csv(
            diagnostics_dir / "skipped_edges.csv",
            metrics_output.get("skipped_edges", []),
        ),
    }
    standardized_data = {
        "schema_version": 1,
        "module": "lidar2lidar",
        "representation": "record_or_prepared_dataset_plus_relation_graph",
        "record_files": record_files,
        "input": {
            "record_path": args.record_path,
            "prepared_dataset_yaml": args.prepared_dataset_yaml,
            "conf_dir": args.conf_dir,
            "workflow_yaml": args.workflow_yaml,
        },
        "target": {
            "topic": target_topic,
            "frame": target_frame,
        },
        "workflow": {
            "planner_mode": workflow_plan["planner_mode"],
            "required_relation_count": workflow_plan["summary"][
                "required_relation_count"
            ],
            "relation_count": workflow_plan["summary"]["relation_count"],
        },
        "normalized_entities": {
            "calibrated_edges": len(edge_results),
            "skipped_edges": len(skipped_edges),
        },
    }
    data_quality = {
        "schema_version": 1,
        "module": "lidar2lidar",
        "status": final_acceptance["status"],
        "release_ready": final_acceptance["release_ready"],
        "quality_gates": final_acceptance["gates"],
        "coarse_statuses": metrics_output["coarse_metrics"]["statuses"],
        "scene_sufficiency": scene_sufficiency_report["summary"],
        "recommendation": final_acceptance["recommendation"],
    }
    visual_layers = [
        str(diagnostics_dir / "visual_evaluation.yaml"),
        table_artifacts["edge_metrics_csv"],
        table_artifacts["skipped_edges_csv"],
    ]
    if merged_summary is not None:
        visual_layers.append(str(merged_summary))
    if loop_closed_merged_summary is not None:
        visual_layers.append(str(loop_closed_merged_summary))
    visualization_index = {
        "schema_version": 1,
        "module": "lidar2lidar",
        "layers": {
            "conclusion": [
                acceptance_artifacts["acceptance_report"],
                acceptance_artifacts["status_summary_csv"],
            ],
            "detail_metrics": [
                str(output_dir / "metrics.yaml"),
                str(diagnostics_dir / "scene_sufficiency.yaml"),
                str(diagnostics_dir / "calibration.yaml"),
                str(diagnostics_dir / "loop_closure.yaml"),
                table_artifacts["edge_metrics_csv"],
                table_artifacts["skipped_edges_csv"],
            ],
            "visual_review": visual_layers,
        },
        "manual_review": [
            "Open colored PLY/PCD overlays in CloudCompare or Open3D when present.",
            "Check wall double edges, corner spread, slice sharpness, and per-edge repeatability.",
            "Treat missing visual geometry as review-only, not release-ready.",
        ],
    }
    paradigm_artifacts = write_paradigm_artifacts(
        diagnostics_dir,
        standardized_data=standardized_data,
        data_quality=data_quality,
        visualization_index=visualization_index,
    )
    metrics_output["artifacts"].update(acceptance_artifacts)
    metrics_output["artifacts"].update(table_artifacts)
    metrics_output["artifacts"].update(paradigm_artifacts)
    with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)

    manifest = {
        "record_files": record_files,
        "conf_dir": args.conf_dir,
        "workflow_yaml": args.workflow_yaml,
        "workflow_mode": workflow_plan["planner_mode"],
        "bootstrap_conf": bool(args.bootstrap_conf),
        "selected_target_topic": target_topic,
        "selected_target_frame": target_frame,
        "root_analysis": root_analysis,
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "metrics": str(output_dir / "metrics.yaml"),
            "initial_guess_dir": str(initial_guess_dir),
            "calibrated_dir": str(calibrated_dir),
            "diagnostics_dir": str(diagnostics_dir),
            "initial_guess_files": initial_guess_files,
            "calibrated_files": calibrated_files,
            "loop_closed_tf": (
                str(output_dir / "loop_closed_tf.yaml")
                if loop_closed_tf_output is not None
                else None
            ),
            "loop_closed_dir": str(loop_closed_dir),
            "loop_closed_files": loop_closed_files,
            "diagnostics": {
                "extraction": str(diagnostics_dir / "extraction.yaml"),
                "topology": str(diagnostics_dir / "topology.yaml"),
                "tf_tree": str(diagnostics_dir / "tf_tree.yaml"),
                "workflow": str(diagnostics_dir / "workflow.yaml"),
                "calibration": str(diagnostics_dir / "calibration.yaml"),
                "scene_sufficiency": str(diagnostics_dir / "scene_sufficiency.yaml"),
                "manifest": str(diagnostics_dir / "manifest.yaml"),
                "loop_closure": (
                    str(diagnostics_dir / "loop_closure.yaml")
                    if loop_closure_report is not None
                    else None
                ),
                "visual_evaluation": (
                    str(diagnostics_dir / "visual_evaluation.yaml")
                    if visual_evaluation_report is not None
                    else None
                ),
                "acceptance_report": acceptance_artifacts["acceptance_report"],
                "status_summary_csv": acceptance_artifacts["status_summary_csv"],
                "standardized_data": paradigm_artifacts["standardized_data"],
                "data_quality": paradigm_artifacts["data_quality"],
                "visualization_index": paradigm_artifacts["visualization_index"],
                "edge_metrics_csv": table_artifacts["edge_metrics_csv"],
                "skipped_edges_csv": table_artifacts["skipped_edges_csv"],
                "merged_cloud": merged_summary,
                "loop_closed_merged_cloud": loop_closed_merged_summary,
            },
        },
    }
    with open(diagnostics_dir / "manifest.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    logging.info("Selected target topic: %s", target_topic)
    for edge_result in edge_results:
        logging.info(
            "Accepted %s -> %s | overlap=%.3f | method=%d | fitness=%.4f | rmse=%.4f",
            edge_result["source_topic"],
            edge_result["target_topic"],
            edge_result["overlap_ratio"],
            edge_result["chosen_method"],
            edge_result["best_run"]["fitness"],
            edge_result["best_run"]["inlier_rmse"],
        )
    if root_analysis["missing_transform_frames_to_target"]:
        logging.warning(
            "Frames missing transform path to target: %s",
            root_analysis["missing_transform_frames_to_target"],
        )
    logging.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
