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

from lidar2lidar.extrinsic_io import build_extrinsics_payload, extrinsics_filename, save_extrinsics_yaml
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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


def choose_target_topic(topic_infos: dict[str, dict],
                        candidate_pairs: list[dict],
                        root_analysis: dict,
                        explicit_target: str | None) -> str:
    if explicit_target is not None:
        if explicit_target not in topic_infos:
            raise ValueError(f"Unknown target topic: {explicit_target}")
        return explicit_target

    frame_representatives: dict[str, list[str]] = defaultdict(list)
    for topic, info in topic_infos.items():
        frame_representatives[info["frame_id"]].append(topic)

    raw_candidates = []
    for topics in frame_representatives.values():
        preferred = sorted(topics, key=lambda topic: topic_preference_sort_key(topic, topic_infos))[0]
        raw_candidates.append(preferred)

    raw_candidates = [topic for topic in raw_candidates if topic]
    if not raw_candidates:
        raw_candidates = list(topic_infos)

    preferred_root_frame = root_analysis.get("preferred_root_frame")
    if preferred_root_frame:
        root_topics = frame_representatives.get(preferred_root_frame, [])
        if root_topics:
            return sorted(root_topics, key=lambda topic: topic_preference_sort_key(topic, topic_infos))[0]

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


def select_edges_for_target(topic_infos: dict[str, dict],
                            candidate_pairs: list[dict],
                            tf_graph: dict,
                            target_topic: str,
                            target_frame: str,
                            source_topics: list[str] | None,
                            min_overlap: float) -> tuple[list[dict], list[dict]]:
    requested_sources = set(source_topics or [])
    selected_edges = []
    skipped_precheck = []
    same_frame_target_topics = [
        topic for topic, info in topic_infos.items()
        if info["frame_id"] == target_frame
    ]

    for topic, info in topic_infos.items():
        if topic == target_topic:
            continue
        if requested_sources and topic not in requested_sources:
            continue
        if info["frame_id"] == target_frame:
            skipped_precheck.append({
                "source_topic": topic,
                "target_topic": target_topic,
                "reason": "same_target_frame",
            })
            continue

        initial_transform = lookup_transform(tf_graph, info["frame_id"], target_frame)
        if initial_transform is None:
            skipped_precheck.append({
                "source_topic": topic,
                "target_topic": target_topic,
                "reason": "no_tf_path",
            })
            continue

        proxy_candidates = [
            pair for pair in candidate_pairs
            if topic in {pair["topic_a"], pair["topic_b"]}
            and ({pair["topic_a"], pair["topic_b"]} & set(same_frame_target_topics))
        ]
        proxy_candidates = [
            pair for pair in proxy_candidates
            if pair["overlap_ratio"] is not None
        ]
        if not proxy_candidates:
            skipped_precheck.append({
                "source_topic": topic,
                "target_topic": target_topic,
                "reason": "no_overlap_probe",
            })
            continue

        pair_match = max(proxy_candidates, key=lambda pair: pair["overlap_ratio"])
        registration_target_topic = (
            pair_match["topic_b"] if pair_match["topic_a"] == topic else pair_match["topic_a"]
        )

        if pair_match["overlap_ratio"] < min_overlap:
            skipped_precheck.append({
                "source_topic": topic,
                "target_topic": target_topic,
                "registration_target_topic": registration_target_topic,
                "reason": "low_overlap",
                "overlap_ratio": float(pair_match["overlap_ratio"]),
            })
            continue

        selected_edges.append({
            "source_topic": topic,
            "target_topic": target_topic,
            "registration_target_topic": registration_target_topic,
            "source_frame": info["frame_id"],
            "target_frame": target_frame,
            "overlap_ratio": float(pair_match["overlap_ratio"]),
            "sync_dt_ms": pair_match["sync_dt_ms"],
            "initial_transform": initial_transform.tolist(),
        })

    return selected_edges, skipped_precheck


def load_cached_cloud(meta, cloud_cache: dict) -> o3d.geometry.PointCloud:
    key = (meta.topic, int(meta.timestamp_ns))
    if key not in cloud_cache:
        cloud_cache[key] = load_pointcloud_from_meta(meta)
    return cloud_cache[key]


def build_candidate_pairs(topic_infos: dict[str, dict],
                          metadata_by_topic: dict,
                          tf_graph: dict,
                          sync_threshold_ns: int,
                          overlap_voxel_size: float,
                          cloud_cache: dict | None = None) -> list[dict]:
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
            overlap_ratio = voxel_overlap_ratio(source_cloud, target_cloud, initial_transform, overlap_voxel_size)

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
    delta_translation_values = [run["delta_to_initial"]["translation_norm_m"] for run in runs]
    delta_rotation_values = [run["delta_to_initial"]["rotation_deg"] for run in runs]
    condition_number_values = [run["information_matrix"]["condition_number"] for run in runs]
    avg_fitness = float(np.mean(fitness_values))
    avg_rmse = float(np.mean(rmse_values))
    avg_delta_translation = float(np.mean(delta_translation_values))
    avg_delta_rotation = float(np.mean(delta_rotation_values))
    degenerate_runs = int(sum(1 for run in runs if run["information_matrix"]["degenerate"]))
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


def prepare_output_layout(output_dir: Path) -> tuple[Path, Path, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for legacy_file in (
        output_dir / "manifest.yaml",
        output_dir / "topology.yaml",
        output_dir / "extraction.yaml",
        output_dir / "tf_tree.yaml",
        output_dir / "calibration.yaml",
        output_dir / "merged_cloud.pcd",
    ):
        if legacy_file.exists() and legacy_file.is_file():
            legacy_file.unlink()

    for diagnostics_file in (
        diagnostics_dir / "manifest.yaml",
        diagnostics_dir / "topology.yaml",
        diagnostics_dir / "extraction.yaml",
        diagnostics_dir / "tf_tree.yaml",
        diagnostics_dir / "calibration.yaml",
        diagnostics_dir / "merged_cloud.pcd",
    ):
        if diagnostics_file.exists() and diagnostics_file.is_file():
            diagnostics_file.unlink()

    initial_guess_dir = output_dir / "initial_guess"
    calibrated_dir = output_dir / "calibrated"
    clean_generated_yaml_dir(initial_guess_dir)
    clean_generated_yaml_dir(calibrated_dir)
    return initial_guess_dir, calibrated_dir, diagnostics_dir


def pick_reference_target_meta(metadata_by_topic: dict,
                               target_topic: str,
                               edge_results: list[dict],
                               sync_threshold_ns: int):
    metas = metadata_by_topic[target_topic]
    if not metas:
        return None

    if edge_results:
        reference_timestamp_ns = edge_results[0]["best_run"]["target_timestamp_ns"]
        nearest_meta = min(metas, key=lambda meta: abs(meta.timestamp_ns - reference_timestamp_ns))
        if abs(nearest_meta.timestamp_ns - reference_timestamp_ns) <= sync_threshold_ns:
            return nearest_meta

    return metas[len(metas) // 2]


def save_merged_cloud(output_path: Path,
                      metadata_by_topic: dict,
                      target_topic: str,
                      edge_results: list[dict],
                      sync_threshold_ns: int,
                      cloud_cache: dict | None = None) -> dict:
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
        source_cloud.transform(np.array(edge_result["best_run"]["transformation"], dtype=float))
        merged_cloud += source_cloud
        used_sources.append({
            "topic": edge_result["source_topic"],
            "timestamp_ns": source_meta.timestamp_ns,
            "dt_ms": float(delta_ns / 1e6),
        })

    saved = o3d.io.write_point_cloud(str(output_path), merged_cloud)
    return {
        "saved": bool(saved),
        "output_path": str(output_path),
        "target_topic": target_topic,
        "target_timestamp_ns": reference_target_meta.timestamp_ns,
        "used_sources": used_sources,
    }


def calibrate_selected_edges(selected_edges: list[dict],
                             metadata_by_topic: dict,
                             sync_threshold_ns: int,
                             args,
                             cloud_cache: dict | None = None) -> tuple[list[dict], list[dict]]:
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
            skipped_edges.append({
                **edge,
                "reason": "no_synced_frames_for_calibration",
            })
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
            skipped_edges.append({
                **edge,
                "reason": "calibration_failed",
            })
            continue

        method_summaries = [summarize_method_runs(method, runs) for method, runs in method_runs.items()]
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
        quality_gate_reasons = []
        if chosen_run["information_matrix"]["degenerate"]:
            quality_gate_reasons.append("degenerate_information_matrix")
        if float(chosen_run["information_matrix"]["condition_number"]) > float(args.max_condition_number):
            quality_gate_reasons.append("condition_number_above_threshold")
        if float(chosen_run["fitness"]) < float(args.min_fitness):
            quality_gate_reasons.append("fitness_below_threshold")
        if quality_gate_reasons:
            skipped_edges.append({
                **edge,
                "reason": "quality_gate_failed",
                "quality_gate_reasons": quality_gate_reasons,
                "candidate_best_run": chosen_run,
                "candidate_method_summaries": method_summaries,
            })
            continue
        edge_results.append({
            **edge,
            "method_summaries": method_summaries,
            "chosen_method": chosen_summary["method"],
            "best_run": chosen_run,
        })

    edge_results.sort(key=lambda item: (-item["best_run"]["fitness"], item["best_run"]["inlier_rmse"]))
    return edge_results, skipped_edges


def build_extraction_output(record_files: list[str],
                            conf_dir: str,
                            topic_counts: dict[str, int],
                            pointcloud_topics: list[str],
                            topic_infos: dict[str, dict],
                            record_tf_edges: list,
                            conf_tf_edges: list,
                            root_analysis: dict,
                            candidate_pairs: list[dict],
                            selected_edges: list[dict],
                            skipped_precheck: list[dict]) -> dict:
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


def _quality_status(value: float | None, threshold: float, *, smaller_is_better: bool) -> str:
    if value is None:
        return "unknown"
    if smaller_is_better:
        return "pass" if value <= threshold else "warning"
    return "pass" if value >= threshold else "warning"


def build_tf_output(base_frame: str, target_topic: str, edge_results: list[dict]) -> dict:
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
                    "registration_target_topic": edge_result["registration_target_topic"],
                    "chosen_method": int(edge_result["chosen_method"]),
                },
            )
        )

    return {
        "base_topic": target_topic,
        "base_frame": base_frame,
        "extrinsics": extrinsics,
    }


def build_metrics_output(record_files: list[str],
                         target_topic: str,
                         target_frame: str,
                         root_analysis: dict,
                         edge_results: list[dict],
                         skipped_edges: list[dict],
                         extraction_output: dict,
                         args,
                         output_dir: Path,
                         merged_summary: dict | None) -> dict:
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
        per_edge.append({
            "extrinsics_file": str(output_dir / "calibrated" / extrinsics_filename(target_frame, edge_result["source_frame"])),
            "source_topic": edge_result["source_topic"],
            "registration_target_topic": edge_result["registration_target_topic"],
            "source_frame": edge_result["source_frame"],
            "target_topic": target_topic,
            "target_frame": target_frame,
            "chosen_method": int(edge_result["chosen_method"]),
            "fitness": float(edge_result["best_run"]["fitness"]),
            "inlier_rmse": float(edge_result["best_run"]["inlier_rmse"]),
            "overlap_ratio": float(edge_result["overlap_ratio"]),
            "sync_dt_ms": float(edge_result["best_run"]["sync_dt_ms"]),
            "delta_to_initial": edge_result["best_run"]["delta_to_initial"],
            "information_matrix": {
                "condition_number": float(info_metrics["condition_number"]),
                "degenerate": bool(info_metrics["degenerate"]),
                "eigenvalues": info_metrics["eigenvalues"],
            },
            "method_summaries": edge_result["method_summaries"],
        })

    summary = {
        "calibrated_edges": len(edge_results),
        "skipped_edges": len(skipped_edges),
        "average_fitness": float(np.mean(fitness_values)) if fitness_values else None,
        "average_inlier_rmse": float(np.mean(rmse_values)) if rmse_values else None,
        "min_overlap_ratio": float(min(overlap_values)) if overlap_values else None,
        "max_condition_number": float(max(condition_numbers)) if condition_numbers else None,
    }

    coarse_metrics = {
        **summary,
        "statuses": {
            "coverage": "pass" if edge_results else "warning",
            "overlap": _quality_status(summary["min_overlap_ratio"], float(args.min_overlap), smaller_is_better=False),
            "fitness": _quality_status(summary["average_fitness"], float(args.min_fitness), smaller_is_better=False),
            "condition_number": _quality_status(summary["max_condition_number"], float(args.max_condition_number), smaller_is_better=True),
            "degeneracy": "warning" if any(
                edge_result["best_run"]["information_matrix"]["degenerate"] for edge_result in edge_results
            ) else "pass",
        },
    }

    fine_metrics = {
        "per_edge": per_edge,
        "skipped_edges": skipped_edges,
        "target_selection": {
            "preferred_root_frame": root_analysis.get("preferred_root_frame"),
            "strategy": root_analysis.get("target_selection_strategy"),
            "missing_transform_frames_to_target": root_analysis.get("missing_transform_frames_to_target", []),
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
            "missing_transform_frames_to_target": root_analysis.get("missing_transform_frames_to_target", []),
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


def write_calibrated_edge_files(output_dir: Path, base_frame: str, edge_results: list[dict]) -> list[str]:
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
    parser = argparse.ArgumentParser(description="Automatic multi-LiDAR calibration pipeline for Apollo record files.")
    parser.add_argument("--record-path", required=True, help="Path to a .record file or a directory containing split record files.")
    parser.add_argument("--output-dir", default="outputs/lidar2lidar/auto_calib", help="Directory for reports and output files.")
    parser.add_argument("--conf-dir", default="lidar2lidar/conf", help="Directory that stores fallback extrinsics YAML files.")
    parser.add_argument("--bootstrap-conf", action="store_true", help="Export record-derived static TF edges into the conf directory.")
    parser.add_argument("--target-topic", default=None, help="Target point cloud topic. If omitted, the pipeline selects one automatically.")
    parser.add_argument("--source-topics", nargs="*", default=None, help="Optional explicit source point cloud topics.")
    parser.add_argument("--sync-threshold-ms", type=float, default=10.0, help="Maximum timestamp difference for frame synchronization.")
    parser.add_argument("--min-overlap", type=float, default=0.30, help="Minimum voxel overlap ratio required to calibrate a topic pair.")
    parser.add_argument("--overlap-voxel-size", type=float, default=0.5, help="Voxel size used during overlap pre-check.")
    parser.add_argument("--max-samples", type=int, default=2, help="Maximum synchronized frame pairs to evaluate per selected edge.")
    parser.add_argument("--methods", nargs="+", type=int, default=[2, 3], help="Calibration methods to compare (1: point-to-plane, 2: GICP, 3: point-to-point).")
    parser.add_argument("--voxel-size", type=float, default=0.04, help="Voxel size for registration preprocessing.")
    parser.add_argument("--max-height", type=float, default=None, help="Optional max height filter for preprocessing.")
    parser.add_argument("--remove-ground", action="store_true", help="Remove the dominant ground plane during preprocessing.")
    parser.add_argument("--remove-walls", action="store_true", help="Remove vertical planes during preprocessing.")
    parser.add_argument("--min-fitness", type=float, default=0.0, help="Minimum fitness required to keep a calibrated edge.")
    parser.add_argument("--max-condition-number", type=float, default=1e6, help="Maximum information-matrix condition number allowed for accepted edges.")
    parser.add_argument("--save-merged-pcd", action="store_true", help="Save a merged point cloud for one synchronized reference timestamp.")
    args = parser.parse_args()

    record_files = discover_record_files(args.record_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_guess_dir, calibrated_dir, diagnostics_dir = prepare_output_layout(output_dir)

    logging.info("Using record files: %s", record_files)
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

    pointcloud_frames = sorted({info["frame_id"] for info in topic_infos.values() if info["frame_id"]})

    record_tf_edges = extract_tf_edges(record_files)
    conf_tf_edges = load_transform_edges_from_dir(args.conf_dir)
    tf_edges = merge_transform_edges(record_tf_edges, conf_tf_edges)
    tf_graph = {}
    if tf_edges:
        tf_graph = build_transform_graph(tf_edges)

    pointcloud_related_static_edges = [
        edge for edge in record_tf_edges
        if edge.is_static and (edge.parent_frame in pointcloud_frames or edge.child_frame in pointcloud_frames)
    ]
    if args.bootstrap_conf and pointcloud_related_static_edges:
        Path(args.conf_dir).mkdir(parents=True, exist_ok=True)
        save_transform_edges_to_dir(args.conf_dir, pointcloud_related_static_edges)

    metadata_by_topic = collect_pointcloud_metadata(record_files, pointcloud_topics)
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

    target_topic = choose_target_topic(topic_infos, candidate_pairs, root_analysis, args.target_topic)
    target_frame = topic_infos[target_topic]["frame_id"]
    root_analysis["topics_on_preferred_root_frame"] = [
        topic for topic, info in topic_infos.items()
        if info["frame_id"] == root_analysis.get("preferred_root_frame")
    ]
    root_analysis["target_selection_strategy"] = (
        "tf_static_root" if root_analysis.get("preferred_root_frame") == target_frame else "overlap_fallback"
    )
    root_analysis["selected_target_topic"] = target_topic
    root_analysis["selected_target_frame"] = target_frame
    root_analysis["missing_transform_frames_to_target"] = find_missing_transform_frames(
        tf_graph,
        target_frame,
        pointcloud_frames,
    )
    logging.info("Selected target topic: %s (%s)", target_topic, target_frame)

    selected_edges, skipped_precheck = select_edges_for_target(
        topic_infos=topic_infos,
        candidate_pairs=candidate_pairs,
        tf_graph=tf_graph,
        target_topic=target_topic,
        target_frame=target_frame,
        source_topics=args.source_topics,
        min_overlap=args.min_overlap,
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

    initial_guess_files = save_transform_edges_to_dir(initial_guess_dir, pointcloud_related_static_edges)
    calibrated_files = write_calibrated_edge_files(calibrated_dir, target_frame, edge_results)

    topology_report = {
        "record_files": record_files,
        "conf_dir": args.conf_dir,
        "record_tf_edges": len(record_tf_edges),
        "conf_tf_edges": len(conf_tf_edges),
        "target_topic": target_topic,
        "target_frame": target_frame,
        "root_analysis": root_analysis,
        "topics": topic_infos,
        "candidate_pairs": candidate_pairs,
        "selected_edges": selected_edges,
        "skipped_edges": skipped_edges,
    }
    with open(diagnostics_dir / "topology.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(topology_report, file, sort_keys=False)

    with open(diagnostics_dir / "extraction.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(extraction_report, file, sort_keys=False)

    with open(diagnostics_dir / "tf_tree.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump({
            **tf_tree_payload(tf_edges),
            "root_analysis": root_analysis,
        }, file, sort_keys=False)

    calibration_report = {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "edge_results": edge_results,
        "skipped_edges": skipped_edges,
    }
    with open(diagnostics_dir / "calibration.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(calibration_report, file, sort_keys=False)

    tf_output = build_tf_output(target_frame, target_topic, edge_results)
    with open(output_dir / "calibrated_tf.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(tf_output, file, sort_keys=False)

    merged_summary = None
    if args.save_merged_pcd and edge_results:
        merged_summary = save_merged_cloud(
            diagnostics_dir / "merged_cloud.pcd",
            metadata_by_topic,
            target_topic,
            edge_results,
            sync_threshold_ns,
            cloud_cache=cloud_cache,
        )

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
    with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)

    manifest = {
        "record_files": record_files,
        "conf_dir": args.conf_dir,
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
            "diagnostics": {
                "extraction": str(diagnostics_dir / "extraction.yaml"),
                "topology": str(diagnostics_dir / "topology.yaml"),
                "tf_tree": str(diagnostics_dir / "tf_tree.yaml"),
                "calibration": str(diagnostics_dir / "calibration.yaml"),
                "manifest": str(diagnostics_dir / "manifest.yaml"),
                "merged_cloud": merged_summary,
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
        logging.warning("Frames missing transform path to target: %s", root_analysis["missing_transform_frames_to_target"])
    logging.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
