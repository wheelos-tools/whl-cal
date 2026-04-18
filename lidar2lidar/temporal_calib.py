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
import bisect
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.auto_calib import (  # noqa: E402
    build_candidate_pairs,
    build_extraction_output,
    calibrate_selected_edges,
    choose_target_topic,
    prepare_output_layout,
    select_edges_for_target,
    summarize_values,
)
from lidar2lidar.extrinsic_io import (  # noqa: E402
    build_extrinsics_payload,
    extrinsics_filename,
    save_extrinsics_yaml,
    stamp_ns_to_dict,
)
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic, preprocess_point_cloud  # noqa: E402
from lidar2lidar.record_utils import (  # noqa: E402
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
    rotation_angle_degrees,
    save_transform_edges_to_dir,
    tf_tree_payload,
    topic_sensor_name,
    transform_delta_metrics,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_temporal_windows(anchor_pairs: list[dict], strides: list[int], max_windows: int) -> list[dict]:
    if len(anchor_pairs) < 2:
        return []

    unique_strides = sorted({int(stride) for stride in strides if int(stride) > 0})
    if not unique_strides:
        unique_strides = [10, 20, 30]

    windows = {}
    per_stride_limit = max(1, int(np.ceil(max_windows / max(len(unique_strides), 1))))
    for stride in unique_strides:
        if stride >= len(anchor_pairs):
            continue
        available = len(anchor_pairs) - stride
        sample_count = min(available, per_stride_limit)
        indices = np.linspace(0, available - 1, num=sample_count, dtype=int)
        for start_index in sorted({int(index) for index in indices.tolist()}):
            end_index = start_index + stride
            start_anchor = anchor_pairs[start_index]
            end_anchor = anchor_pairs[end_index]
            key = (start_index, end_index)
            windows[key] = {
                "start_index": start_index,
                "end_index": end_index,
                "stride": int(stride),
                "source_start_timestamp_ns": int(start_anchor["source_timestamp_ns"]),
                "source_end_timestamp_ns": int(end_anchor["source_timestamp_ns"]),
                "target_start_timestamp_ns": int(start_anchor["target_timestamp_ns"]),
                "target_end_timestamp_ns": int(end_anchor["target_timestamp_ns"]),
                "window_dt_ms": float(
                    max(
                        end_anchor["source_timestamp_ns"] - start_anchor["source_timestamp_ns"],
                        end_anchor["target_timestamp_ns"] - start_anchor["target_timestamp_ns"],
                    ) / 1e6
                ),
            }

    ordered_windows = sorted(
        windows.values(),
        key=lambda item: (-item["stride"], item["start_index"]),
    )
    return ordered_windows[:max_windows]


def find_nearest_meta(metas: list, timestamp_ns: int, max_delta_ns: int):
    if not metas:
        return None, None, None
    times = [meta.timestamp_ns for meta in metas]
    index = bisect.bisect_left(times, timestamp_ns)
    best_meta = None
    best_delta_ns = None
    best_index = None
    for candidate_index in (index - 1, index):
        if candidate_index < 0 or candidate_index >= len(metas):
            continue
        candidate_meta = metas[candidate_index]
        delta_ns = abs(candidate_meta.timestamp_ns - timestamp_ns)
        if delta_ns <= max_delta_ns and (best_delta_ns is None or delta_ns < best_delta_ns):
            best_meta = candidate_meta
            best_delta_ns = delta_ns
            best_index = candidate_index
    return best_meta, best_delta_ns, best_index


def build_temporal_dataset(edge: dict,
                           metadata_by_topic: dict,
                           sync_threshold_ns: int,
                           strides: list[int],
                           max_windows: int) -> dict:
    source_metas = metadata_by_topic[edge["source_topic"]]
    target_metas = metadata_by_topic[edge["target_topic"]]
    anchor_pairs = []
    for source_meta in source_metas:
        target_meta, delta_ns, target_index = find_nearest_meta(target_metas, source_meta.timestamp_ns, sync_threshold_ns)
        if target_meta is None:
            continue
        anchor_pairs.append({
            "source_timestamp_ns": int(source_meta.timestamp_ns),
            "target_timestamp_ns": int(target_meta.timestamp_ns),
            "target_index": int(target_index),
            "sync_dt_ms": float(delta_ns / 1e6),
        })

    unique_strides = sorted({int(stride) for stride in strides if int(stride) > 0})
    if not unique_strides:
        unique_strides = [5, 10, 20]
    windows = {}
    per_stride_limit = max(1, int(np.ceil(max_windows / max(len(unique_strides), 1))))
    for stride in unique_strides:
        if stride >= len(source_metas):
            continue
        available = len(source_metas) - stride
        sample_count = min(available, per_stride_limit)
        indices = np.linspace(0, available - 1, num=sample_count, dtype=int)
        for start_index in sorted({int(index) for index in indices.tolist()}):
            end_index = start_index + stride
            source_start = source_metas[start_index]
            source_end = source_metas[end_index]
            target_start, start_delta_ns, target_start_index = find_nearest_meta(
                target_metas,
                source_start.timestamp_ns,
                sync_threshold_ns,
            )
            target_end, end_delta_ns, target_end_index = find_nearest_meta(
                target_metas,
                source_end.timestamp_ns,
                sync_threshold_ns,
            )
            if target_start is None or target_end is None:
                continue
            windows[(start_index, end_index)] = {
                "start_index": start_index,
                "end_index": end_index,
                "stride": int(stride),
                "source_start_timestamp_ns": int(source_start.timestamp_ns),
                "source_end_timestamp_ns": int(source_end.timestamp_ns),
                "target_start_timestamp_ns": int(target_start.timestamp_ns),
                "target_end_timestamp_ns": int(target_end.timestamp_ns),
                "target_start_index": int(target_start_index),
                "target_end_index": int(target_end_index),
                "window_dt_ms": float((source_end.timestamp_ns - source_start.timestamp_ns) / 1e6),
                "start_sync_dt_ms": float(start_delta_ns / 1e6),
                "end_sync_dt_ms": float(end_delta_ns / 1e6),
            }
    windows = sorted(
        windows.values(),
        key=lambda item: (-item["stride"], item["start_index"]),
    )[:max_windows]
    return {
        "source_topic": edge["source_topic"],
        "target_topic": edge["target_topic"],
        "source_frame": edge["source_frame"],
        "target_frame": edge["target_frame"],
        "registration_target_topic": edge["registration_target_topic"],
        "anchor_pairs": anchor_pairs,
        "candidate_windows": windows,
        "summary": {
            "anchor_pair_count": len(anchor_pairs),
            "candidate_window_count": len(windows),
            "anchor_sync_dt_ms": summarize_values([pair["sync_dt_ms"] for pair in anchor_pairs]),
            "window_dt_ms": summarize_values([window["window_dt_ms"] for window in windows]),
            "start_sync_dt_ms": summarize_values([window["start_sync_dt_ms"] for window in windows]),
            "end_sync_dt_ms": summarize_values([window["end_sync_dt_ms"] for window in windows]),
        },
    }


def load_cached_cloud(meta, cloud_cache: dict) -> o3d.geometry.PointCloud:
    key = (meta.topic, int(meta.timestamp_ns))
    if key not in cloud_cache:
        cloud_cache[key] = load_pointcloud_from_meta(meta)
    return cloud_cache[key]


def _freeze_preprocessing_params(preprocessing_params: dict) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((str(key), preprocessing_params[key]) for key in preprocessing_params))


def load_preprocessed_cloud(meta,
                            *,
                            preprocessing_params: dict,
                            cloud_cache: dict,
                            preprocessed_cloud_cache: dict) -> o3d.geometry.PointCloud:
    cache_key = (
        meta.topic,
        int(meta.timestamp_ns),
        _freeze_preprocessing_params(preprocessing_params),
    )
    if cache_key not in preprocessed_cloud_cache:
        preprocessed_cloud_cache[cache_key] = preprocess_point_cloud(
            load_cached_cloud(meta, cloud_cache),
            **preprocessing_params,
        )
    return preprocessed_cloud_cache[cache_key]


def register_motion_clouds(source_cloud,
                           target_cloud,
                           *,
                           start_timestamp_ns: int,
                           end_timestamp_ns: int,
                           motion_method: int,
                           preprocessing_params: dict,
                           motion_voxel_size: float,
                           stagnation_dt_ms: float,
                           stagnation_translation_m: float,
                           stagnation_rotation_deg: float,
                           attempts: list[dict] | None = None,
                           allow_early_break: bool = True) -> dict | None:
    if len(source_cloud.points) == 0 or len(target_cloud.points) == 0:
        return None

    if attempts is None:
        attempts = [
            {"name": "identity_bootstrap", "initial_transform": np.eye(4, dtype=float)},
            {"name": "feature_bootstrap", "initial_transform": None},
        ]

    eval_distance = max(motion_voxel_size * 5.0, 0.2)
    eval_voxel_size = max(motion_voxel_size, 0.1)
    source_eval = source_cloud.voxel_down_sample(eval_voxel_size)
    target_eval = target_cloud.voxel_down_sample(eval_voxel_size)

    def build_motion_record(candidate_name: str,
                            transform: np.ndarray,
                            *,
                            fitness: float,
                            inlier_rmse: float) -> dict:
        info_metrics = compute_information_metrics(
            source_cloud,
            target_cloud,
            transform,
            max_correspondence_distance=eval_distance,
            downsample_voxel_size=eval_voxel_size,
        )
        record = {
            "attempt": candidate_name,
            "start_timestamp_ns": int(start_timestamp_ns),
            "end_timestamp_ns": int(end_timestamp_ns),
            "window_dt_ms": float((end_timestamp_ns - start_timestamp_ns) / 1e6),
            "fitness": float(fitness),
            "inlier_rmse": float(inlier_rmse),
            "transformation": transform.tolist(),
            "motion_magnitude": {
                "translation_m": float(np.linalg.norm(transform[:3, 3])),
                "rotation_deg": float(rotation_angle_degrees(transform[:3, :3])),
            },
            "delta_to_identity": transform_delta_metrics(np.eye(4, dtype=float), transform),
            "information_matrix": info_metrics,
        }
        record["stagnant_solution"] = bool(
            record["window_dt_ms"] >= stagnation_dt_ms
            and record["motion_magnitude"]["translation_m"] <= stagnation_translation_m
            and record["motion_magnitude"]["rotation_deg"] <= stagnation_rotation_deg
        )
        return record

    best_record = None
    for attempt in attempts:
        final_transform, initial_transform, reg_result = calibrate_lidar_extrinsic(
            source_cloud,
            target_cloud,
            is_draw_registration=False,
            preprocessing_params=preprocessing_params,
            method=motion_method,
            initial_transform=attempt["initial_transform"],
        )

        candidate_records = []
        if initial_transform is not None:
            coarse_eval = o3d.pipelines.registration.evaluate_registration(
                source_eval,
                target_eval,
                eval_distance,
                np.asarray(initial_transform, dtype=float),
            )
            candidate_records.append(build_motion_record(
                f"{attempt['name']}_coarse",
                np.asarray(initial_transform, dtype=float),
                fitness=float(coarse_eval.fitness),
                inlier_rmse=float(coarse_eval.inlier_rmse),
            ))
        if final_transform is not None and reg_result is not None:
            candidate_records.append(build_motion_record(
                attempt["name"],
                np.asarray(final_transform, dtype=float),
                fitness=float(reg_result.fitness),
                inlier_rmse=float(reg_result.inlier_rmse),
            ))

        attempt_best = None
        for record in candidate_records:
            if attempt_best is None or (
                record["stagnant_solution"],
                record["information_matrix"]["degenerate"],
                -record["fitness"],
                record["inlier_rmse"],
                -record["motion_magnitude"]["translation_m"],
                -record["motion_magnitude"]["rotation_deg"],
            ) < (
                attempt_best["stagnant_solution"],
                attempt_best["information_matrix"]["degenerate"],
                -attempt_best["fitness"],
                attempt_best["inlier_rmse"],
                -attempt_best["motion_magnitude"]["translation_m"],
                -attempt_best["motion_magnitude"]["rotation_deg"],
            ):
                attempt_best = record

        if attempt_best is None:
            continue

        if best_record is None or (
            attempt_best["stagnant_solution"],
            attempt_best["information_matrix"]["degenerate"],
            -attempt_best["fitness"],
            attempt_best["inlier_rmse"],
            -attempt_best["motion_magnitude"]["translation_m"],
            -attempt_best["motion_magnitude"]["rotation_deg"],
        ) < (
            best_record["stagnant_solution"],
            best_record["information_matrix"]["degenerate"],
            -best_record["fitness"],
            best_record["inlier_rmse"],
            -best_record["motion_magnitude"]["translation_m"],
            -best_record["motion_magnitude"]["rotation_deg"],
        ):
            best_record = attempt_best
        if (
            allow_early_break
            and attempt["name"] == "identity_bootstrap"
            and not attempt_best["stagnant_solution"]
            and not attempt_best["information_matrix"]["degenerate"]
            and attempt_best["fitness"] >= 0.85
        ):
            break

    return best_record


def estimate_sensor_motion(start_meta,
                           end_meta,
                           *,
                           motion_method: int,
                           preprocessing_params: dict,
                           motion_voxel_size: float,
                           stagnation_dt_ms: float,
                           stagnation_translation_m: float,
                           stagnation_rotation_deg: float,
                           cloud_cache: dict,
                           motion_cache: dict) -> dict | None:
    cache_key = (start_meta.topic, int(start_meta.timestamp_ns), int(end_meta.timestamp_ns), motion_method)
    if cache_key in motion_cache:
        return motion_cache[cache_key]

    start_cloud = load_cached_cloud(start_meta, cloud_cache)
    end_cloud = load_cached_cloud(end_meta, cloud_cache)
    best_record = register_motion_clouds(
        start_cloud,
        end_cloud,
        start_timestamp_ns=int(start_meta.timestamp_ns),
        end_timestamp_ns=int(end_meta.timestamp_ns),
        motion_method=motion_method,
        preprocessing_params=preprocessing_params,
        motion_voxel_size=motion_voxel_size,
        stagnation_dt_ms=stagnation_dt_ms,
        stagnation_translation_m=stagnation_translation_m,
        stagnation_rotation_deg=stagnation_rotation_deg,
    )
    motion_cache[cache_key] = best_record
    return best_record


def motion_pair_score(source_motion: dict, target_motion: dict) -> float:
    rotation_score = min(
        source_motion["motion_magnitude"]["rotation_deg"],
        target_motion["motion_magnitude"]["rotation_deg"],
    )
    translation_score = min(
        source_motion["motion_magnitude"]["translation_m"],
        target_motion["motion_magnitude"]["translation_m"],
    )
    quality_score = min(source_motion["fitness"], target_motion["fitness"])
    return float(quality_score * (rotation_score + translation_score * 10.0))


def compose_relative_transform(metas: list,
                               source_index: int,
                               target_index: int,
                               *,
                               local_step: int,
                               motion_method: int,
                               preprocessing_params: dict,
                               motion_voxel_size: float,
                               stagnation_dt_ms: float,
                               stagnation_translation_m: float,
                               stagnation_rotation_deg: float,
                               cloud_cache: dict,
                               motion_cache: dict) -> tuple[np.ndarray, list[dict]] | None:
    if source_index < 0 or target_index < 0 or source_index >= len(metas) or target_index >= len(metas):
        return None
    if source_index == target_index:
        return np.eye(4, dtype=float), []

    effective_step = max(1, int(local_step))
    current_index = source_index
    composed_transform = np.eye(4, dtype=float)
    path_records = []
    while current_index != target_index:
        if current_index < target_index:
            next_index = min(current_index + effective_step, target_index)
            step_record = estimate_sensor_motion(
                metas[current_index],
                metas[next_index],
                motion_method=motion_method,
                preprocessing_params=preprocessing_params,
                motion_voxel_size=motion_voxel_size,
                stagnation_dt_ms=stagnation_dt_ms,
                stagnation_translation_m=stagnation_translation_m,
                stagnation_rotation_deg=stagnation_rotation_deg,
                cloud_cache=cloud_cache,
                motion_cache=motion_cache,
            )
            if step_record is None:
                return None
            step_transform = np.array(step_record["transformation"], dtype=float)
        else:
            next_index = max(current_index - effective_step, target_index)
            step_record = estimate_sensor_motion(
                metas[next_index],
                metas[current_index],
                motion_method=motion_method,
                preprocessing_params=preprocessing_params,
                motion_voxel_size=motion_voxel_size,
                stagnation_dt_ms=stagnation_dt_ms,
                stagnation_translation_m=stagnation_translation_m,
                stagnation_rotation_deg=stagnation_rotation_deg,
                cloud_cache=cloud_cache,
                motion_cache=motion_cache,
            )
            if step_record is None:
                return None
            step_transform = np.linalg.inv(np.array(step_record["transformation"], dtype=float))

        path_records.append({
            "from_index": int(current_index),
            "to_index": int(next_index),
            "from_timestamp_ns": int(metas[current_index].timestamp_ns),
            "to_timestamp_ns": int(metas[next_index].timestamp_ns),
            "fitness": float(step_record["fitness"]),
            "inlier_rmse": float(step_record["inlier_rmse"]),
            "stagnant_solution": bool(step_record["stagnant_solution"]),
            "information_matrix": step_record["information_matrix"],
            "delta_to_identity": transform_delta_metrics(np.eye(4, dtype=float), step_transform),
        })
        composed_transform = step_transform @ composed_transform
        current_index = next_index

    return composed_transform, path_records


def select_submap_support_indices(reference_index: int,
                                  other_index: int,
                                  *,
                                  local_step: int,
                                  max_support_nodes: int) -> list[int]:
    indices = [int(reference_index)]
    support_limit = max(0, int(max_support_nodes))
    if support_limit == 0 or reference_index == other_index:
        return indices

    effective_step = max(1, int(local_step))
    midpoint = 0.5 * (reference_index + other_index)
    if reference_index < other_index:
        candidate_index = reference_index + effective_step
        while len(indices) - 1 < support_limit and candidate_index < midpoint:
            indices.append(int(candidate_index))
            candidate_index += effective_step
    else:
        candidate_index = reference_index - effective_step
        while len(indices) - 1 < support_limit and candidate_index > midpoint:
            indices.append(int(candidate_index))
            candidate_index -= effective_step
    return indices


def build_local_submap(metas: list,
                       reference_index: int,
                       support_indices: list[int],
                       *,
                       local_step: int,
                       submap_voxel_size: float,
                       motion_method: int,
                       preprocessing_params: dict,
                       motion_voxel_size: float,
                       stagnation_dt_ms: float,
                       stagnation_translation_m: float,
                       stagnation_rotation_deg: float,
                       cloud_cache: dict,
                       motion_cache: dict) -> tuple[o3d.geometry.PointCloud, dict] | None:
    reference_cloud = copy.deepcopy(load_cached_cloud(metas[reference_index], cloud_cache))
    if len(reference_cloud.points) == 0:
        return None

    merged_cloud = reference_cloud
    support_records = []
    for support_index in support_indices:
        if support_index == reference_index:
            continue
        transform_result = compose_relative_transform(
            metas,
            support_index,
            reference_index,
            local_step=local_step,
            motion_method=motion_method,
            preprocessing_params=preprocessing_params,
            motion_voxel_size=motion_voxel_size,
            stagnation_dt_ms=stagnation_dt_ms,
            stagnation_translation_m=stagnation_translation_m,
            stagnation_rotation_deg=stagnation_rotation_deg,
            cloud_cache=cloud_cache,
            motion_cache=motion_cache,
        )
        if transform_result is None:
            continue
        support_transform, path_records = transform_result
        support_cloud = copy.deepcopy(load_cached_cloud(metas[support_index], cloud_cache))
        if len(support_cloud.points) == 0:
            continue
        support_cloud.transform(support_transform)
        merged_cloud += support_cloud
        support_records.append({
            "scan_index": int(support_index),
            "timestamp_ns": int(metas[support_index].timestamp_ns),
            "node_hops": len(path_records),
            "path_fitness": summarize_values([record["fitness"] for record in path_records]),
            "path_inlier_rmse": summarize_values([record["inlier_rmse"] for record in path_records]),
            "path_condition_number": summarize_values([
                float(record["information_matrix"]["condition_number"])
                for record in path_records
            ]),
            "delta_to_reference": transform_delta_metrics(np.eye(4, dtype=float), support_transform),
            "path_records": path_records,
        })

    if submap_voxel_size and submap_voxel_size > 0:
        merged_cloud = merged_cloud.voxel_down_sample(float(submap_voxel_size))

    return merged_cloud, {
        "reference_index": int(reference_index),
        "reference_timestamp_ns": int(metas[reference_index].timestamp_ns),
        "scan_indices": [int(index) for index in support_indices],
        "scan_count": int(1 + len(support_records)),
        "point_count": int(len(merged_cloud.points)),
        "support_records": support_records,
    }


def build_window_motion_from_local_chain(metas: list,
                                         start_index: int,
                                         end_index: int,
                                         *,
                                         local_step: int,
                                         motion_method: int,
                                         preprocessing_params: dict,
                                         motion_voxel_size: float,
                                         stagnation_dt_ms: float,
                                         stagnation_translation_m: float,
                                         stagnation_rotation_deg: float,
                                         cloud_cache: dict,
                                         motion_cache: dict) -> dict | None:
    if start_index >= end_index or start_index < 0 or end_index >= len(metas):
        return None

    step_records = []
    composed_transform = np.eye(4, dtype=float)
    effective_step = max(1, int(local_step))
    for index in range(start_index, end_index, effective_step):
        segment_end_index = min(index + effective_step, end_index)
        step_record = estimate_sensor_motion(
            metas[index],
            metas[segment_end_index],
            motion_method=motion_method,
            preprocessing_params=preprocessing_params,
            motion_voxel_size=motion_voxel_size,
            stagnation_dt_ms=stagnation_dt_ms,
            stagnation_translation_m=stagnation_translation_m,
            stagnation_rotation_deg=stagnation_rotation_deg,
            cloud_cache=cloud_cache,
            motion_cache=motion_cache,
        )
        if step_record is None:
            return None
        step_records.append(step_record)
        composed_transform = np.array(step_record["transformation"], dtype=float) @ composed_transform

    condition_numbers = [
        float(step_record["information_matrix"]["condition_number"])
        for step_record in step_records
    ]
    fitness_values = [float(step_record["fitness"]) for step_record in step_records]
    rmse_values = [float(step_record["inlier_rmse"]) for step_record in step_records]
    window_dt_ms = float((metas[end_index].timestamp_ns - metas[start_index].timestamp_ns) / 1e6)
    motion_record = {
        "attempt": "local_odometry_chain",
        "motion_frontend": "chain",
        "start_timestamp_ns": int(metas[start_index].timestamp_ns),
        "end_timestamp_ns": int(metas[end_index].timestamp_ns),
        "window_dt_ms": window_dt_ms,
        "fitness": float(np.mean(fitness_values)),
        "inlier_rmse": float(np.mean(rmse_values)),
        "transformation": composed_transform.tolist(),
        "motion_magnitude": {
            "translation_m": float(np.linalg.norm(composed_transform[:3, 3])),
            "rotation_deg": float(rotation_angle_degrees(composed_transform[:3, :3])),
        },
        "delta_to_identity": transform_delta_metrics(np.eye(4, dtype=float), composed_transform),
        "information_matrix": {
            "matrix": None,
            "eigenvalues": None,
            "condition_number": float(max(condition_numbers)) if condition_numbers else float("inf"),
            "degenerate": bool(any(step_record["information_matrix"]["degenerate"] for step_record in step_records)),
        },
        "local_odometry": {
            "step_count": len(step_records),
            "local_step": effective_step,
            "step_fitness": summarize_values(fitness_values),
            "step_inlier_rmse": summarize_values(rmse_values),
            "step_condition_number": summarize_values(condition_numbers),
            "step_records": step_records,
        },
    }
    motion_record["stagnant_solution"] = bool(
        motion_record["window_dt_ms"] >= stagnation_dt_ms
        and motion_record["motion_magnitude"]["translation_m"] <= stagnation_translation_m
        and motion_record["motion_magnitude"]["rotation_deg"] <= stagnation_rotation_deg
    )
    return motion_record


def build_window_motion_from_submaps(metas: list,
                                     start_index: int,
                                     end_index: int,
                                     *,
                                     local_step: int,
                                     submap_support_nodes: int,
                                     submap_voxel_size: float,
                                     motion_method: int,
                                     preprocessing_params: dict,
                                     motion_voxel_size: float,
                                     stagnation_dt_ms: float,
                                     stagnation_translation_m: float,
                                     stagnation_rotation_deg: float,
                                     cloud_cache: dict,
                                     motion_cache: dict) -> dict | None:
    if start_index >= end_index or start_index < 0 or end_index >= len(metas):
        return None

    cache_key = (
        "submap",
        metas[start_index].topic,
        int(metas[start_index].timestamp_ns),
        int(metas[end_index].timestamp_ns),
        int(motion_method),
        int(local_step),
        int(submap_support_nodes),
        float(submap_voxel_size),
    )
    if cache_key in motion_cache:
        return motion_cache[cache_key]

    start_support_indices = select_submap_support_indices(
        start_index,
        end_index,
        local_step=local_step,
        max_support_nodes=submap_support_nodes,
    )
    end_support_indices = select_submap_support_indices(
        end_index,
        start_index,
        local_step=local_step,
        max_support_nodes=submap_support_nodes,
    )
    start_submap_result = build_local_submap(
        metas,
        start_index,
        start_support_indices,
        local_step=local_step,
        submap_voxel_size=submap_voxel_size,
        motion_method=motion_method,
        preprocessing_params=preprocessing_params,
        motion_voxel_size=motion_voxel_size,
        stagnation_dt_ms=stagnation_dt_ms,
        stagnation_translation_m=stagnation_translation_m,
        stagnation_rotation_deg=stagnation_rotation_deg,
        cloud_cache=cloud_cache,
        motion_cache=motion_cache,
    )
    end_submap_result = build_local_submap(
        metas,
        end_index,
        end_support_indices,
        local_step=local_step,
        submap_voxel_size=submap_voxel_size,
        motion_method=motion_method,
        preprocessing_params=preprocessing_params,
        motion_voxel_size=motion_voxel_size,
        stagnation_dt_ms=stagnation_dt_ms,
        stagnation_translation_m=stagnation_translation_m,
        stagnation_rotation_deg=stagnation_rotation_deg,
        cloud_cache=cloud_cache,
        motion_cache=motion_cache,
    )
    if start_submap_result is None or end_submap_result is None:
        motion_cache[cache_key] = None
        return None

    start_submap, start_submap_info = start_submap_result
    end_submap, end_submap_info = end_submap_result
    motion_record = register_motion_clouds(
        start_submap,
        end_submap,
        start_timestamp_ns=int(metas[start_index].timestamp_ns),
        end_timestamp_ns=int(metas[end_index].timestamp_ns),
        motion_method=motion_method,
        preprocessing_params=preprocessing_params,
        motion_voxel_size=motion_voxel_size,
        stagnation_dt_ms=stagnation_dt_ms,
        stagnation_translation_m=stagnation_translation_m,
        stagnation_rotation_deg=stagnation_rotation_deg,
        attempts=[
            {"name": "feature_bootstrap", "initial_transform": None},
            {"name": "identity_bootstrap", "initial_transform": np.eye(4, dtype=float)},
        ],
        allow_early_break=False,
    )
    if motion_record is None:
        motion_cache[cache_key] = None
        return None

    motion_record["attempt"] = f"submap_{motion_record['attempt']}"
    motion_record["motion_frontend"] = "submap"
    motion_record["submap_odometry"] = {
        "local_step": int(max(1, local_step)),
        "submap_support_nodes": int(max(0, submap_support_nodes)),
        "submap_voxel_size": float(submap_voxel_size),
        "start_submap": start_submap_info,
        "end_submap": end_submap_info,
    }
    motion_cache[cache_key] = motion_record
    return motion_record


def estimate_window_motion(metas: list,
                           start_index: int,
                           end_index: int,
                           *,
                           motion_frontend: str,
                           local_step: int,
                           submap_support_nodes: int,
                           submap_voxel_size: float,
                           motion_method: int,
                           preprocessing_params: dict,
                           motion_voxel_size: float,
                           stagnation_dt_ms: float,
                           stagnation_translation_m: float,
                           stagnation_rotation_deg: float,
                           cloud_cache: dict,
                           motion_cache: dict) -> dict | None:
    if motion_frontend == "submap":
        return build_window_motion_from_submaps(
            metas,
            start_index,
            end_index,
            local_step=local_step,
            submap_support_nodes=submap_support_nodes,
            submap_voxel_size=submap_voxel_size,
            motion_method=motion_method,
            preprocessing_params=preprocessing_params,
            motion_voxel_size=motion_voxel_size,
            stagnation_dt_ms=stagnation_dt_ms,
            stagnation_translation_m=stagnation_translation_m,
            stagnation_rotation_deg=stagnation_rotation_deg,
            cloud_cache=cloud_cache,
            motion_cache=motion_cache,
        )
    return build_window_motion_from_local_chain(
        metas,
        start_index,
        end_index,
        local_step=local_step,
        motion_method=motion_method,
        preprocessing_params=preprocessing_params,
        motion_voxel_size=motion_voxel_size,
        stagnation_dt_ms=stagnation_dt_ms,
        stagnation_translation_m=stagnation_translation_m,
        stagnation_rotation_deg=stagnation_rotation_deg,
        cloud_cache=cloud_cache,
        motion_cache=motion_cache,
    )


def collect_motion_pairs(edge: dict,
                         temporal_dataset: dict,
                         metadata_by_topic: dict,
                         args,
                         cloud_cache: dict,
                         motion_cache: dict) -> tuple[list[dict], list[dict]]:
    source_metas = metadata_by_topic[edge["source_topic"]]
    target_metas = metadata_by_topic[edge["target_topic"]]
    accepted = []
    skipped = []
    preprocessing_params = {
        "voxel_size": args.motion_voxel_size,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": 0.05,
        "height_range": args.max_height,
        "remove_ground": args.remove_ground,
        "remove_walls": args.remove_walls,
    }

    for window in temporal_dataset["candidate_windows"]:
        if (
            window["start_index"] < 0
            or window["end_index"] >= len(source_metas)
            or window["target_start_index"] < 0
            or window["target_end_index"] >= len(target_metas)
        ):
            skipped.append({
                **window,
                "reason": "missing_window_metadata",
            })
            continue

        source_motion = estimate_window_motion(
            source_metas,
            window["start_index"],
            window["end_index"],
            motion_frontend=args.motion_frontend,
            local_step=args.local_odometry_step,
            submap_support_nodes=args.submap_support_nodes,
            submap_voxel_size=args.submap_voxel_size,
            motion_method=args.motion_method,
            preprocessing_params=preprocessing_params,
            motion_voxel_size=args.motion_voxel_size,
            stagnation_dt_ms=args.motion_stagnation_dt_ms,
            stagnation_translation_m=args.motion_stagnation_translation_m,
            stagnation_rotation_deg=args.motion_stagnation_rotation_deg,
            cloud_cache=cloud_cache,
            motion_cache=motion_cache,
        )
        target_motion = estimate_window_motion(
            target_metas,
            window["target_start_index"],
            window["target_end_index"],
            motion_frontend=args.motion_frontend,
            local_step=args.local_odometry_step,
            submap_support_nodes=args.submap_support_nodes,
            submap_voxel_size=args.submap_voxel_size,
            motion_method=args.motion_method,
            preprocessing_params=preprocessing_params,
            motion_voxel_size=args.motion_voxel_size,
            stagnation_dt_ms=args.motion_stagnation_dt_ms,
            stagnation_translation_m=args.motion_stagnation_translation_m,
            stagnation_rotation_deg=args.motion_stagnation_rotation_deg,
            cloud_cache=cloud_cache,
            motion_cache=motion_cache,
        )
        if source_motion is None or target_motion is None:
            skipped.append({
                **window,
                "reason": "motion_estimation_failed",
            })
            continue

        reasons = []
        if source_motion["fitness"] < args.motion_min_fitness:
            reasons.append("source_motion_fitness_below_threshold")
        if target_motion["fitness"] < args.motion_min_fitness:
            reasons.append("target_motion_fitness_below_threshold")
        if source_motion["information_matrix"]["degenerate"]:
            reasons.append("source_motion_degenerate")
        if target_motion["information_matrix"]["degenerate"]:
            reasons.append("target_motion_degenerate")
        if source_motion.get("stagnant_solution"):
            reasons.append("source_motion_stagnant")
        if target_motion.get("stagnant_solution"):
            reasons.append("target_motion_stagnant")

        max_rotation = max(
            source_motion["motion_magnitude"]["rotation_deg"],
            target_motion["motion_magnitude"]["rotation_deg"],
        )
        max_translation = max(
            source_motion["motion_magnitude"]["translation_m"],
            target_motion["motion_magnitude"]["translation_m"],
        )
        if max_rotation < args.motion_min_rotation_deg and max_translation < args.motion_min_translation_m:
            reasons.append("low_motion_excitation")

        motion_pair = {
            **window,
            "source_motion": source_motion,
            "target_motion": target_motion,
            "weight": float(min(source_motion["fitness"], target_motion["fitness"])),
            "score": motion_pair_score(source_motion, target_motion),
        }
        if reasons:
            skipped.append({
                **motion_pair,
                "reason": "motion_pair_rejected",
                "reasons": reasons,
            })
            continue
        accepted.append(motion_pair)

    accepted.sort(key=lambda item: (-item["score"], -item["stride"], item["start_index"]))
    return accepted[:args.max_motion_pairs], skipped


def solve_handeye_rotation(motion_pairs: list[dict]) -> tuple[np.ndarray | None, dict]:
    source_rotvecs = []
    target_rotvecs = []
    weights = []
    for pair in motion_pairs:
        source_transform = np.array(pair["source_motion"]["transformation"], dtype=float)
        target_transform = np.array(pair["target_motion"]["transformation"], dtype=float)
        source_rotvec = R.from_matrix(source_transform[:3, :3]).as_rotvec()
        target_rotvec = R.from_matrix(target_transform[:3, :3]).as_rotvec()
        if np.linalg.norm(source_rotvec) < 1e-6 or np.linalg.norm(target_rotvec) < 1e-6:
            continue
        source_rotvecs.append(source_rotvec)
        target_rotvecs.append(target_rotvec)
        weights.append(float(pair["weight"]))

    if len(source_rotvecs) < 2:
        return None, {
            "usable_motion_pairs": len(source_rotvecs),
            "reason": "insufficient_rotational_excitation",
        }

    source_stack = np.vstack(source_rotvecs)
    target_stack = np.vstack(target_rotvecs)
    cross_covariance = np.zeros((3, 3), dtype=float)
    for source_rotvec, target_rotvec, weight in zip(source_stack, target_stack, weights):
        cross_covariance += weight * np.outer(target_rotvec, source_rotvec)

    u, singular_values, vt = np.linalg.svd(cross_covariance)
    rotation_matrix = u @ vt
    if np.linalg.det(rotation_matrix) < 0:
        u[:, -1] *= -1.0
        rotation_matrix = u @ vt

    axes = []
    for rotvec in source_stack:
        norm = np.linalg.norm(rotvec)
        if norm > 1e-9:
            axes.append(rotvec / norm)
    if axes:
        axes_singular_values = np.linalg.svd(np.vstack(axes), compute_uv=False)
        axis_rank = int(sum(value > 0.2 * axes_singular_values[0] for value in axes_singular_values))
    else:
        axes_singular_values = np.zeros(3)
        axis_rank = 0

    return rotation_matrix, {
        "usable_motion_pairs": len(source_rotvecs),
        "singular_values": [float(value) for value in singular_values.tolist()],
        "rotation_axis_singular_values": [float(value) for value in axes_singular_values.tolist()],
        "rotation_axis_rank": axis_rank,
    }


def solve_handeye_translation(rotation_matrix: np.ndarray, motion_pairs: list[dict]) -> tuple[np.ndarray | None, dict]:
    left_blocks = []
    right_blocks = []
    for pair in motion_pairs:
        source_transform = np.array(pair["source_motion"]["transformation"], dtype=float)
        target_transform = np.array(pair["target_motion"]["transformation"], dtype=float)
        weight = float(np.sqrt(max(pair["weight"], 1e-6)))
        left_blocks.append(weight * (target_transform[:3, :3] - np.eye(3, dtype=float)))
        right_blocks.append(weight * (rotation_matrix @ source_transform[:3, 3] - target_transform[:3, 3]))

    if not left_blocks:
        return None, {
            "rank": 0,
            "condition_number": float("inf"),
            "reason": "no_motion_pairs",
        }

    left_matrix = np.vstack(left_blocks)
    right_vector = np.concatenate(right_blocks)
    solution, _, rank, singular_values = np.linalg.lstsq(left_matrix, right_vector, rcond=None)
    positive = singular_values[singular_values > 1e-9]
    condition_number = float(positive[-1] / positive[0]) if len(positive) >= 2 else float("inf")
    return solution, {
        "rank": int(rank),
        "condition_number": condition_number,
        "singular_values": [float(value) for value in singular_values.tolist()],
    }


def pack_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def handeye_residual_components(transform: np.ndarray, motion_pairs: list[dict]) -> list[dict]:
    inverse_transform = np.linalg.inv(transform)
    residuals = []
    for pair in motion_pairs:
        source_transform = np.array(pair["source_motion"]["transformation"], dtype=float)
        target_transform = np.array(pair["target_motion"]["transformation"], dtype=float)
        delta = target_transform @ transform @ np.linalg.inv(source_transform) @ inverse_transform
        residuals.append({
            "start_index": int(pair["start_index"]),
            "end_index": int(pair["end_index"]),
            "weight": float(pair["weight"]),
            "rotation_deg": float(rotation_angle_degrees(delta[:3, :3])),
            "translation_m": float(np.linalg.norm(delta[:3, 3])),
        })
    return residuals


def refine_handeye_solution(initial_transform: np.ndarray, motion_pairs: list[dict]) -> tuple[np.ndarray, dict]:
    initial_rotation = R.from_matrix(initial_transform[:3, :3]).as_rotvec()
    initial_translation = initial_transform[:3, 3]

    def residual_vector(parameters: np.ndarray) -> np.ndarray:
        transform = pack_transform(R.from_rotvec(parameters[:3]).as_matrix(), parameters[3:])
        inverse_transform = np.linalg.inv(transform)
        residuals = []
        for pair in motion_pairs:
            source_transform = np.array(pair["source_motion"]["transformation"], dtype=float)
            target_transform = np.array(pair["target_motion"]["transformation"], dtype=float)
            delta = target_transform @ transform @ np.linalg.inv(source_transform) @ inverse_transform
            weight = float(np.sqrt(max(pair["weight"], 1e-6)))
            residuals.extend((weight * R.from_matrix(delta[:3, :3]).as_rotvec()).tolist())
            residuals.extend((weight * delta[:3, 3]).tolist())
        return np.asarray(residuals, dtype=float)

    optimized = least_squares(
        residual_vector,
        np.concatenate([initial_rotation, initial_translation]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=200,
    )
    refined_transform = pack_transform(
        R.from_rotvec(optimized.x[:3]).as_matrix(),
        optimized.x[3:],
    )
    residual_components = handeye_residual_components(refined_transform, motion_pairs)
    return refined_transform, {
        "success": bool(optimized.success),
        "cost": float(optimized.cost),
        "nfev": int(optimized.nfev),
        "status": int(optimized.status),
        "message": str(optimized.message),
        "rotation_residual_deg": summarize_values([item["rotation_deg"] for item in residual_components]),
        "translation_residual_m": summarize_values([item["translation_m"] for item in residual_components]),
        "per_motion_pair": residual_components,
    }


def evaluate_transform_on_pairs(transform: np.ndarray,
                                source_topic: str,
                                target_topic: str,
                                metadata_by_topic: dict,
                                sync_threshold_ns: int,
                                sample_count: int,
                                preprocessing_params: dict,
                                evaluation_distance: float,
                                cloud_cache: dict,
                                preprocessed_cloud_cache: dict) -> dict:
    matches = find_synchronized_pairs(
        metadata_by_topic[source_topic],
        metadata_by_topic[target_topic],
        sync_threshold_ns,
        max_pairs=sample_count,
    )
    if not matches:
        return {
            "samples": 0,
            "fitness": summarize_values([]),
            "inlier_rmse": summarize_values([]),
        }

    fitness_values = []
    rmse_values = []
    per_sample = []
    for source_meta, target_meta, delta_ns in matches:
        source_cloud = load_preprocessed_cloud(
            source_meta,
            preprocessing_params=preprocessing_params,
            cloud_cache=cloud_cache,
            preprocessed_cloud_cache=preprocessed_cloud_cache,
        )
        target_cloud = load_preprocessed_cloud(
            target_meta,
            preprocessing_params=preprocessing_params,
            cloud_cache=cloud_cache,
            preprocessed_cloud_cache=preprocessed_cloud_cache,
        )
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_cloud,
            target_cloud,
            evaluation_distance,
            transform,
        )
        fitness_values.append(float(evaluation.fitness))
        rmse_values.append(float(evaluation.inlier_rmse))
        per_sample.append({
            "source_timestamp_ns": int(source_meta.timestamp_ns),
            "target_timestamp_ns": int(target_meta.timestamp_ns),
            "sync_dt_ms": float(delta_ns / 1e6),
            "fitness": float(evaluation.fitness),
            "inlier_rmse": float(evaluation.inlier_rmse),
        })

    return {
        "samples": len(matches),
        "fitness": summarize_values(fitness_values),
        "inlier_rmse": summarize_values(rmse_values),
        "per_sample": per_sample,
    }


def summarize_pairwise_baseline(edge_results: list[dict], skipped_edges: list[dict]) -> dict:
    fitness_values = [float(item["best_run"]["fitness"]) for item in edge_results]
    rmse_values = [float(item["best_run"]["inlier_rmse"]) for item in edge_results]
    return {
        "summary": {
            "accepted_edges": len(edge_results),
            "skipped_edges": len(skipped_edges),
            "fitness": summarize_values(fitness_values),
            "inlier_rmse": summarize_values(rmse_values),
        },
        "per_edge": [
            {
                "source_topic": item["source_topic"],
                "target_topic": item["target_topic"],
                "registration_target_topic": item["registration_target_topic"],
                "fitness": float(item["best_run"]["fitness"]),
                "inlier_rmse": float(item["best_run"]["inlier_rmse"]),
                "transformation": item["best_run"]["transformation"],
                "chosen_method": int(item["chosen_method"]),
            }
            for item in edge_results
        ],
        "skipped_edges": skipped_edges,
    }


def build_temporal_tf_output(base_frame: str, target_topic: str, edge_results: list[dict]) -> dict:
    return {
        "base_topic": target_topic,
        "base_frame": base_frame,
        "extrinsics": [
            build_extrinsics_payload(
                parent_frame=base_frame,
                child_frame=edge_result["source_frame"],
                matrix=np.array(edge_result["refined_transform"], dtype=float),
                stamp_ns=edge_result["latest_timestamp_ns"],
                metadata={
                    "mode": "temporal_handeye",
                    "source_topic": edge_result["source_topic"],
                    "registration_target_topic": edge_result["registration_target_topic"],
                    "motion_pair_count": int(edge_result["motion_pair_count"]),
                },
            )
            for edge_result in edge_results
        ],
    }


def write_temporal_edge_files(output_dir: Path, base_frame: str, edge_results: list[dict]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for edge_result in edge_results:
        file_path = output_dir / extrinsics_filename(base_frame, edge_result["source_frame"])
        save_extrinsics_yaml(
            str(file_path),
            parent_frame=base_frame,
            child_frame=edge_result["source_frame"],
            matrix=np.array(edge_result["refined_transform"], dtype=float),
            stamp_ns=edge_result["latest_timestamp_ns"],
            metadata={
                "mode": "temporal_handeye",
                "source_topic": edge_result["source_topic"],
                "registration_target_topic": edge_result["registration_target_topic"],
                "motion_pair_count": int(edge_result["motion_pair_count"]),
            },
        )
        saved_paths.append(str(file_path))
    return saved_paths


def calibrate_temporal_edges(selected_edges: list[dict],
                             metadata_by_topic: dict,
                             sync_threshold_ns: int,
                             args,
                             pairwise_baseline_by_topic: dict[str, dict]) -> tuple[list[dict], list[dict], list[dict]]:
    temporal_results = []
    skipped_edges = []
    temporal_datasets = []
    cloud_cache = {}
    motion_cache = {}
    preprocessed_cloud_cache = {}
    evaluation_preprocessing = {
        "voxel_size": args.voxel_size,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": 0.05,
        "height_range": args.max_height,
        "remove_ground": args.remove_ground,
        "remove_walls": args.remove_walls,
    }
    evaluation_distance = max(args.voxel_size * 5.0, 0.2)

    for edge in selected_edges:
        logging.info(
            "Temporal hand-eye calibration for %s -> %s using %s",
            edge["source_topic"],
            edge["target_topic"],
            edge["registration_target_topic"],
        )
        dataset = build_temporal_dataset(
            edge,
            metadata_by_topic,
            sync_threshold_ns,
            args.temporal_window_strides,
            args.max_temporal_windows,
        )
        temporal_datasets.append(dataset)
        motion_pairs, skipped_motion_pairs = collect_motion_pairs(
            edge,
            dataset,
            metadata_by_topic,
            args,
            cloud_cache,
            motion_cache,
        )
        if len(motion_pairs) < args.min_motion_pairs:
            skipped_edges.append({
                **edge,
                "reason": "insufficient_motion_pairs",
                "usable_motion_pairs": len(motion_pairs),
                "dataset_summary": dataset["summary"],
                "skipped_motion_pairs": skipped_motion_pairs,
            })
            continue

        rotation_matrix, rotation_info = solve_handeye_rotation(motion_pairs)
        if rotation_matrix is None:
            skipped_edges.append({
                **edge,
                "reason": "rotation_solve_failed",
                "usable_motion_pairs": len(motion_pairs),
                "rotation_info": rotation_info,
                "skipped_motion_pairs": skipped_motion_pairs,
            })
            continue

        translation, translation_info = solve_handeye_translation(rotation_matrix, motion_pairs)
        if translation is None:
            skipped_edges.append({
                **edge,
                "reason": "translation_solve_failed",
                "rotation_info": rotation_info,
                "translation_info": translation_info,
                "skipped_motion_pairs": skipped_motion_pairs,
            })
            continue

        initial_transform = pack_transform(rotation_matrix, translation)
        refined_transform, refinement_info = refine_handeye_solution(initial_transform, motion_pairs)
        transform_delta = transform_delta_metrics(
            np.array(edge["initial_transform"], dtype=float),
            refined_transform,
        )
        pairwise_edge = pairwise_baseline_by_topic.get(edge["source_topic"])
        comparison = {
            "delta_to_initial": transform_delta,
        }
        if pairwise_edge is not None:
            comparison["delta_to_pairwise"] = transform_delta_metrics(
                np.array(pairwise_edge["best_run"]["transformation"], dtype=float),
                refined_transform,
            )

        holdout_temporal = evaluate_transform_on_pairs(
            refined_transform,
            edge["source_topic"],
            edge["registration_target_topic"],
            metadata_by_topic,
            sync_threshold_ns,
            args.comparison_samples,
            evaluation_preprocessing,
            evaluation_distance,
            cloud_cache,
            preprocessed_cloud_cache,
        )
        holdout_initial = evaluate_transform_on_pairs(
            np.array(edge["initial_transform"], dtype=float),
            edge["source_topic"],
            edge["registration_target_topic"],
            metadata_by_topic,
            sync_threshold_ns,
            args.comparison_samples,
            evaluation_preprocessing,
            evaluation_distance,
            cloud_cache,
            preprocessed_cloud_cache,
        )
        holdout_pairwise = None
        if pairwise_edge is not None:
            holdout_pairwise = evaluate_transform_on_pairs(
                np.array(pairwise_edge["best_run"]["transformation"], dtype=float),
                edge["source_topic"],
                edge["registration_target_topic"],
                metadata_by_topic,
                sync_threshold_ns,
                args.comparison_samples,
                evaluation_preprocessing,
                evaluation_distance,
                cloud_cache,
                preprocessed_cloud_cache,
            )

        quality_reasons = []
        if rotation_info.get("rotation_axis_rank", 0) < args.min_rotation_axis_rank:
            quality_reasons.append("insufficient_rotation_axis_diversity")
        if translation_info["rank"] < 3:
            quality_reasons.append("translation_rank_deficient")
        if translation_info["condition_number"] > args.max_translation_condition_number:
            quality_reasons.append("translation_condition_number_above_threshold")
        rotation_p95 = refinement_info["rotation_residual_deg"]["p95"]
        translation_p95 = refinement_info["translation_residual_m"]["p95"]
        if rotation_p95 is not None and rotation_p95 > args.max_handeye_rotation_residual_deg:
            quality_reasons.append("rotation_residual_above_threshold")
        if translation_p95 is not None and translation_p95 > args.max_handeye_translation_residual_m:
            quality_reasons.append("translation_residual_above_threshold")

        result = {
            **edge,
            "latest_timestamp_ns": max(
                pair["source_motion"]["end_timestamp_ns"]
                for pair in motion_pairs
            ),
            "initial_transform": edge["initial_transform"],
            "initial_transform_stamp": stamp_ns_to_dict(None),
            "temporal_dataset_summary": dataset["summary"],
            "motion_pair_count": len(motion_pairs),
            "motion_pairs": motion_pairs,
            "skipped_motion_pairs": skipped_motion_pairs,
            "rotation_info": rotation_info,
            "translation_info": translation_info,
            "initial_solution": initial_transform.tolist(),
            "refined_transform": refined_transform.tolist(),
            "refinement": refinement_info,
            "comparison": comparison,
            "holdout_evaluation": {
                "initial": holdout_initial,
                "pairwise": holdout_pairwise,
                "temporal": holdout_temporal,
            },
            "quality_gate_reasons": quality_reasons,
        }
        if quality_reasons:
            skipped_edges.append({
                **result,
                "reason": "quality_gate_failed",
            })
            continue
        temporal_results.append(result)

    temporal_results.sort(
        key=lambda item: (
            -(item["holdout_evaluation"]["temporal"]["fitness"]["mean"] or -1.0),
            item["refinement"]["translation_residual_m"]["p95"] or float("inf"),
        )
    )
    return temporal_results, skipped_edges, temporal_datasets


def build_temporal_metrics_output(record_files: list[str],
                                  target_topic: str,
                                  target_frame: str,
                                  root_analysis: dict,
                                  temporal_results: list[dict],
                                  skipped_edges: list[dict],
                                  extraction_output: dict,
                                  temporal_datasets: list[dict],
                                  pairwise_baseline: dict,
                                  output_dir: Path) -> dict:
    temporal_holdout_fitness = [
        float(item["holdout_evaluation"]["temporal"]["fitness"]["mean"])
        for item in temporal_results
        if item["holdout_evaluation"]["temporal"]["fitness"]["mean"] is not None
    ]
    temporal_holdout_rmse = [
        float(item["holdout_evaluation"]["temporal"]["inlier_rmse"]["mean"])
        for item in temporal_results
        if item["holdout_evaluation"]["temporal"]["inlier_rmse"]["mean"] is not None
    ]
    motion_pair_counts = [int(item["motion_pair_count"]) for item in temporal_results]
    rotation_residual_p95 = [
        float(item["refinement"]["rotation_residual_deg"]["p95"])
        for item in temporal_results
        if item["refinement"]["rotation_residual_deg"]["p95"] is not None
    ]
    translation_residual_p95 = [
        float(item["refinement"]["translation_residual_m"]["p95"])
        for item in temporal_results
        if item["refinement"]["translation_residual_m"]["p95"] is not None
    ]

    per_edge = []
    for result in temporal_results:
        per_edge.append({
            "extrinsics_file": str(output_dir / "calibrated" / extrinsics_filename(target_frame, result["source_frame"])),
            "source_topic": result["source_topic"],
            "target_topic": result["target_topic"],
            "registration_target_topic": result["registration_target_topic"],
            "motion_pair_count": int(result["motion_pair_count"]),
            "rotation_axis_rank": int(result["rotation_info"]["rotation_axis_rank"]),
            "translation_rank": int(result["translation_info"]["rank"]),
            "translation_condition_number": float(result["translation_info"]["condition_number"]),
            "rotation_residual_deg": result["refinement"]["rotation_residual_deg"],
            "translation_residual_m": result["refinement"]["translation_residual_m"],
            "delta_to_initial": result["comparison"]["delta_to_initial"],
            "delta_to_pairwise": result["comparison"].get("delta_to_pairwise"),
            "holdout_evaluation": result["holdout_evaluation"],
        })

    summary = {
        "accepted_edges": len(temporal_results),
        "skipped_edges": len(skipped_edges),
        "motion_pair_count": summarize_values(motion_pair_counts),
        "temporal_holdout_fitness": summarize_values(temporal_holdout_fitness),
        "temporal_holdout_inlier_rmse": summarize_values(temporal_holdout_rmse),
        "rotation_residual_p95_deg": summarize_values(rotation_residual_p95),
        "translation_residual_p95_m": summarize_values(translation_residual_p95),
    }
    coarse_metrics = {
        **summary,
        "statuses": {
            "coverage": "pass" if temporal_results else "warning",
            "motion_pairs": "pass" if motion_pair_counts and min(motion_pair_counts) >= 3 else "warning",
            "rotation_residual": "pass" if rotation_residual_p95 and max(rotation_residual_p95) <= 2.0 else "warning",
            "translation_residual": "pass" if translation_residual_p95 and max(translation_residual_p95) <= 1.0 else "warning",
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
        "temporal_dataset_summary": [
            {
                "source_topic": dataset["source_topic"],
                "target_topic": dataset["target_topic"],
                "summary": dataset["summary"],
            }
            for dataset in temporal_datasets
        ],
    }
    return {
        "record_files": record_files,
        "pipeline_mode": "temporal_handeye_compare",
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
        "pairwise_baseline": pairwise_baseline,
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "initial_guess_dir": str(output_dir / "initial_guess"),
            "calibrated_dir": str(output_dir / "calibrated"),
            "diagnostics_dir": str(output_dir / "diagnostics"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal multi-frame LiDAR-to-LiDAR hand-eye calibration with pairwise ICP comparison."
    )
    parser.add_argument("--record-path", required=True, help="Path to a .record file or record directory.")
    parser.add_argument("--output-dir", default="outputs/lidar2lidar/temporal_calib", help="Directory for reports and output files.")
    parser.add_argument("--conf-dir", default="lidar2lidar/conf", help="Directory that stores fallback extrinsics YAML files.")
    parser.add_argument("--bootstrap-conf", action="store_true", help="Export record-derived static TF edges into the conf directory.")
    parser.add_argument("--target-topic", default=None, help="Target point cloud topic. If omitted, the pipeline selects one automatically.")
    parser.add_argument("--source-topics", nargs="*", default=None, help="Optional explicit source point cloud topics.")
    parser.add_argument("--sync-threshold-ms", type=float, default=20.0, help="Maximum timestamp difference for frame synchronization.")
    parser.add_argument("--min-overlap", type=float, default=0.30, help="Minimum voxel overlap ratio required to calibrate a topic pair.")
    parser.add_argument("--overlap-voxel-size", type=float, default=0.5, help="Voxel size used during overlap pre-check.")
    parser.add_argument("--max-samples", type=int, default=1, help="Maximum synchronized frame pairs to evaluate per pairwise ICP edge.")
    parser.add_argument("--methods", nargs="+", type=int, default=[2], help="Pairwise ICP methods to compare (1: point-to-plane, 2: GICP, 3: point-to-point).")
    parser.add_argument("--voxel-size", type=float, default=0.04, help="Voxel size for cross-sensor evaluation preprocessing.")
    parser.add_argument("--motion-frontend", choices=["chain", "submap"], default="submap", help="Same-sensor motion frontend used before AX=XB solving.")
    parser.add_argument("--motion-voxel-size", type=float, default=0.08, help="Voxel size for same-sensor motion estimation.")
    parser.add_argument("--motion-method", type=int, default=2, help="Registration method used for same-sensor motion estimation.")
    parser.add_argument("--max-height", type=float, default=None, help="Optional max height filter for preprocessing.")
    parser.add_argument("--remove-ground", action="store_true", help="Remove the dominant ground plane during preprocessing.")
    parser.add_argument("--remove-walls", action="store_true", help="Remove vertical planes during preprocessing.")
    parser.add_argument("--min-fitness", type=float, default=0.0, help="Minimum fitness required to keep a pairwise ICP edge.")
    parser.add_argument("--max-condition-number", type=float, default=1e6, help="Maximum information-matrix condition number allowed for pairwise ICP edges.")
    parser.add_argument("--temporal-window-strides", nargs="+", type=int, default=[10, 20, 30, 40], help="Anchor-pair strides used to build temporal windows.")
    parser.add_argument("--max-temporal-windows", type=int, default=12, help="Maximum temporal windows to evaluate per edge.")
    parser.add_argument("--max-motion-pairs", type=int, default=8, help="Maximum accepted motion pairs kept per edge.")
    parser.add_argument("--min-motion-pairs", type=int, default=3, help="Minimum accepted motion pairs required for hand-eye solving.")
    parser.add_argument("--motion-min-fitness", type=float, default=0.10, help="Minimum same-sensor motion fitness required to keep a motion pair.")
    parser.add_argument("--motion-min-rotation-deg", type=float, default=0.3, help="Minimum rotational excitation for a temporal motion pair.")
    parser.add_argument("--motion-min-translation-m", type=float, default=0.05, help="Minimum translational excitation for a temporal motion pair.")
    parser.add_argument("--local-odometry-step", type=int, default=4, help="Number of frames between chained local-odometry nodes.")
    parser.add_argument("--submap-support-nodes", type=int, default=1, help="Maximum extra local-odometry nodes fused into each temporal submap.")
    parser.add_argument("--submap-voxel-size", type=float, default=0.08, help="Optional downsample voxel size applied after building each temporal submap.")
    parser.add_argument("--motion-stagnation-dt-ms", type=float, default=250.0, help="Minimum window duration that activates stagnant-motion rejection.")
    parser.add_argument("--motion-stagnation-translation-m", type=float, default=0.02, help="Translation magnitude below this is treated as stagnant for long windows.")
    parser.add_argument("--motion-stagnation-rotation-deg", type=float, default=0.05, help="Rotation magnitude below this is treated as stagnant for long windows.")
    parser.add_argument("--min-rotation-axis-rank", type=int, default=1, help="Minimum approximate rank of rotation-axis diversity required for acceptance.")
    parser.add_argument("--max-translation-condition-number", type=float, default=1e5, help="Maximum translation linear-system condition number allowed for temporal acceptance.")
    parser.add_argument("--max-handeye-rotation-residual-deg", type=float, default=5.0, help="Maximum p95 AX=XB rotation residual allowed for temporal acceptance.")
    parser.add_argument("--max-handeye-translation-residual-m", type=float, default=2.0, help="Maximum p95 AX=XB translation residual allowed for temporal acceptance.")
    parser.add_argument("--comparison-samples", type=int, default=3, help="Number of synchronized cross-sensor samples used for fixed-transform comparison.")
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
    tf_graph = build_transform_graph(tf_edges) if tf_edges else {}

    pointcloud_related_static_edges = [
        edge for edge in record_tf_edges
        if edge.is_static and (edge.parent_frame in pointcloud_frames or edge.child_frame in pointcloud_frames)
    ]
    if args.bootstrap_conf and pointcloud_related_static_edges:
        Path(args.conf_dir).mkdir(parents=True, exist_ok=True)
        save_transform_edges_to_dir(args.conf_dir, pointcloud_related_static_edges)

    metadata_by_topic = collect_pointcloud_metadata(record_files, pointcloud_topics)
    sync_threshold_ns = int(args.sync_threshold_ms * 1e6)
    root_analysis = analyze_pointcloud_roots(pointcloud_frames, tf_edges)
    candidate_pairs = build_candidate_pairs(
        topic_infos,
        metadata_by_topic,
        tf_graph,
        sync_threshold_ns,
        args.overlap_voxel_size,
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

    selected_edges, skipped_precheck = select_edges_for_target(
        topic_infos=topic_infos,
        candidate_pairs=candidate_pairs,
        tf_graph=tf_graph,
        target_topic=target_topic,
        target_frame=target_frame,
        source_topics=args.source_topics,
        min_overlap=args.min_overlap,
    )

    pairwise_results, pairwise_skipped_runtime = calibrate_selected_edges(
        selected_edges,
        metadata_by_topic,
        sync_threshold_ns,
        args,
    )
    pairwise_skipped = skipped_precheck + pairwise_skipped_runtime
    pairwise_baseline = summarize_pairwise_baseline(pairwise_results, pairwise_skipped)
    pairwise_baseline_by_topic = {
        item["source_topic"]: item
        for item in pairwise_results
    }

    temporal_results, temporal_skipped_runtime, temporal_datasets = calibrate_temporal_edges(
        selected_edges,
        metadata_by_topic,
        sync_threshold_ns,
        args,
        pairwise_baseline_by_topic,
    )
    temporal_skipped = skipped_precheck + temporal_skipped_runtime

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
    calibrated_files = write_temporal_edge_files(calibrated_dir, target_frame, temporal_results)

    with open(diagnostics_dir / "extraction.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(extraction_report, file, sort_keys=False)
    with open(diagnostics_dir / "tf_tree.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump({
            **tf_tree_payload(tf_edges),
            "root_analysis": root_analysis,
        }, file, sort_keys=False)
    with open(diagnostics_dir / "pairwise_baseline.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(pairwise_baseline, file, sort_keys=False)
    with open(diagnostics_dir / "temporal_dataset.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump({"edges": temporal_datasets}, file, sort_keys=False)
    with open(diagnostics_dir / "temporal_calibration.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump({
            "record_files": record_files,
            "target_topic": target_topic,
            "target_frame": target_frame,
            "edge_results": temporal_results,
            "skipped_edges": temporal_skipped,
        }, file, sort_keys=False)

    tf_output = build_temporal_tf_output(target_frame, target_topic, temporal_results)
    with open(output_dir / "calibrated_tf.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(tf_output, file, sort_keys=False)

    metrics_output = build_temporal_metrics_output(
        record_files=record_files,
        target_topic=target_topic,
        target_frame=target_frame,
        root_analysis=root_analysis,
        temporal_results=temporal_results,
        skipped_edges=temporal_skipped,
        extraction_output=extraction_report,
        temporal_datasets=temporal_datasets,
        pairwise_baseline=pairwise_baseline,
        output_dir=output_dir,
    )
    with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)

    manifest = {
        "record_files": record_files,
        "conf_dir": args.conf_dir,
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
                "tf_tree": str(diagnostics_dir / "tf_tree.yaml"),
                "pairwise_baseline": str(diagnostics_dir / "pairwise_baseline.yaml"),
                "temporal_dataset": str(diagnostics_dir / "temporal_dataset.yaml"),
                "temporal_calibration": str(diagnostics_dir / "temporal_calibration.yaml"),
                "manifest": str(diagnostics_dir / "manifest.yaml"),
            },
        },
    }
    with open(diagnostics_dir / "manifest.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    logging.info("Selected target topic: %s", target_topic)
    for edge_result in temporal_results:
        logging.info(
            "Accepted temporal %s -> %s | motion_pairs=%d | rot_res_p95=%.3f deg | trans_res_p95=%.3f m",
            edge_result["source_topic"],
            edge_result["target_topic"],
            edge_result["motion_pair_count"],
            edge_result["refinement"]["rotation_residual_deg"]["p95"],
            edge_result["refinement"]["translation_residual_m"]["p95"],
        )
    if root_analysis["missing_transform_frames_to_target"]:
        logging.warning("Frames missing transform path to target: %s", root_analysis["missing_transform_frames_to_target"])
    logging.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
