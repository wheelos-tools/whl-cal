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
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.extrinsic_io import (
    build_extrinsics_payload,
    extrinsics_filename,
    matrix_from_transform_dict,
    save_extrinsics_yaml,
)
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic, preprocess_point_cloud
from lidar2lidar.record_utils import (
    build_transform_graph,
    collect_pointcloud_metadata,
    compute_information_metrics,
    discover_record_files,
    extract_tf_edges,
    get_topic_frame_ids,
    infer_pointcloud_topics,
    list_topics,
    load_pointcloud_from_meta,
    load_transform_edges_from_dir,
    lookup_transform,
    merge_transform_edges,
    topic_sensor_name,
    transform_delta_metrics,
    voxel_overlap_ratio,
)


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


def _quality_status(
    value: float | None, threshold: float, *, smaller_is_better: bool
) -> str:
    if value is None:
        return "unknown"
    if smaller_is_better:
        return "pass" if value <= threshold else "warning"
    return "pass" if value >= threshold else "warning"


def uniform_sample(items: list, max_items: int) -> list:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[index] for index in indices]


def load_cached_cloud(meta, cloud_cache: dict) -> o3d.geometry.PointCloud:
    key = (meta.topic, int(meta.timestamp_ns))
    if key not in cloud_cache:
        cloud_cache[key] = load_pointcloud_from_meta(meta)
    return cloud_cache[key]


def freeze_preprocessing_params(
    preprocessing_params: dict,
) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted((str(key), preprocessing_params[key]) for key in preprocessing_params)
    )


def load_preprocessed_source_cloud(
    meta,
    *,
    preprocessing_params: dict,
    cloud_cache: dict,
    preprocessed_source_cache: dict,
) -> o3d.geometry.PointCloud:
    cache_key = (
        meta.topic,
        int(meta.timestamp_ns),
        freeze_preprocessing_params(preprocessing_params),
    )
    if cache_key not in preprocessed_source_cache:
        preprocessed_source_cache[cache_key] = preprocess_point_cloud(
            copy.deepcopy(load_cached_cloud(meta, cloud_cache)),
            **preprocessing_params,
        )
    return preprocessed_source_cache[cache_key]


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    return np.linalg.inv(transform)


def matrix_to_pose_components(transform: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(transform, dtype=float)
    yaw_deg, pitch_deg, roll_deg = R.from_matrix(matrix[:3, :3]).as_euler(
        "zyx", degrees=True
    )
    return {
        "x": float(matrix[0, 3]),
        "y": float(matrix[1, 3]),
        "z": float(matrix[2, 3]),
        "yaw": float(yaw_deg),
        "pitch": float(pitch_deg),
        "roll": float(roll_deg),
    }


def pose_components_to_matrix(components: dict[str, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = R.from_euler(
        "zyx",
        [
            float(components["yaw"]),
            float(components["pitch"]),
            float(components["roll"]),
        ],
        degrees=True,
    ).as_matrix()
    matrix[:3, 3] = [
        float(components["x"]),
        float(components["y"]),
        float(components["z"]),
    ]
    return matrix


def normalize_locked_components(components: list[str] | None) -> list[str]:
    ordered = []
    for name in ("x", "y", "z", "yaw", "pitch", "roll"):
        if components and name in components and name not in ordered:
            ordered.append(name)
    return ordered


def constrain_transform_components(
    transform: np.ndarray,
    *,
    reference_transform: np.ndarray | None,
    locked_components: list[str] | None,
) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    locked = normalize_locked_components(locked_components)
    if reference_transform is None or not locked:
        return matrix
    components = matrix_to_pose_components(matrix)
    reference_components = matrix_to_pose_components(reference_transform)
    for name in locked:
        components[name] = reference_components[name]
    return pose_components_to_matrix(components)


def resolve_constraint_reference(
    *,
    initial_transform: np.ndarray,
    baseline_transform: np.ndarray | None,
    constraint_reference: str,
) -> tuple[np.ndarray | None, str]:
    if constraint_reference == "initial":
        return np.asarray(initial_transform, dtype=float), "initial"
    if constraint_reference == "scan2scan_baseline":
        return (
            (
                np.asarray(baseline_transform, dtype=float)
                if baseline_transform is not None
                else None
            ),
            "scan2scan_baseline",
        )
    return None, "none"


def wrap_angle_degrees(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def component_delta_metrics(
    reference_transform: np.ndarray, candidate_transform: np.ndarray
) -> dict[str, float]:
    reference = matrix_to_pose_components(reference_transform)
    candidate = matrix_to_pose_components(candidate_transform)
    delta_x = float(candidate["x"] - reference["x"])
    delta_y = float(candidate["y"] - reference["y"])
    delta_z = float(candidate["z"] - reference["z"])
    delta_yaw = wrap_angle_degrees(candidate["yaw"] - reference["yaw"])
    delta_pitch = float(candidate["pitch"] - reference["pitch"])
    delta_roll = float(candidate["roll"] - reference["roll"])
    return {
        "x_m": delta_x,
        "y_m": delta_y,
        "z_m": delta_z,
        "planar_translation_norm_m": float(np.linalg.norm([delta_x, delta_y])),
        "yaw_deg": delta_yaw,
        "pitch_deg": delta_pitch,
        "roll_deg": delta_roll,
        "pitch_roll_norm_deg": float(np.linalg.norm([delta_pitch, delta_roll])),
    }


def build_holdout_delta_metrics(
    reference_evaluation: dict | None, candidate_evaluation: dict | None
) -> dict | None:
    if not reference_evaluation or not candidate_evaluation:
        return None
    if (
        int(reference_evaluation.get("count", 0)) == 0
        or int(candidate_evaluation.get("count", 0)) == 0
    ):
        return None

    def mean_or_none(evaluation: dict, field: str) -> float | None:
        value = evaluation.get(field, {}).get("mean")
        return None if value is None else float(value)

    def max_or_none(evaluation: dict, field: str) -> float | None:
        value = evaluation.get(field, {}).get("max")
        return None if value is None else float(value)

    baseline_condition = max_or_none(reference_evaluation, "condition_number")
    candidate_condition = max_or_none(candidate_evaluation, "condition_number")
    return {
        "fitness_mean_delta": float(
            mean_or_none(candidate_evaluation, "fitness")
            - mean_or_none(reference_evaluation, "fitness")
        ),
        "inlier_rmse_mean_delta": float(
            mean_or_none(candidate_evaluation, "inlier_rmse")
            - mean_or_none(reference_evaluation, "inlier_rmse")
        ),
        "overlap_ratio_mean_delta": float(
            mean_or_none(candidate_evaluation, "overlap_ratio")
            - mean_or_none(reference_evaluation, "overlap_ratio")
        ),
        "condition_number_max_delta": (
            None
            if baseline_condition is None or candidate_condition is None
            else float(candidate_condition - baseline_condition)
        ),
        "condition_number_ratio": (
            None
            if baseline_condition in (None, 0.0) or candidate_condition is None
            else float(candidate_condition / baseline_condition)
        ),
    }


def build_vehicle_rig_assessment(
    *, delta_components: dict | None, holdout_delta: dict | None
) -> dict:
    thresholds = {
        "planar_translation_norm_m": 0.05,
        "yaw_deg": 1.0,
        "z_m": 0.05,
        "pitch_roll_norm_deg": 1.0,
        "fitness_mean_delta": 0.01,
        "overlap_ratio_mean_delta": 0.02,
        "inlier_rmse_mean_delta": -0.001,
        "condition_number_ratio": 1.5,
    }
    if delta_components is None or holdout_delta is None:
        return {
            "thresholds": thresholds,
            "statuses": {
                "planar_drift": "unknown",
                "vertical_drift": "unknown",
                "holdout_gain": "unknown",
                "conditioning": "unknown",
            },
            "recommendation": "insufficient_data",
        }

    planar_ok = float(delta_components["planar_translation_norm_m"]) <= float(
        thresholds["planar_translation_norm_m"]
    ) and abs(float(delta_components["yaw_deg"])) <= float(thresholds["yaw_deg"])
    vertical_ok = abs(float(delta_components["z_m"])) <= float(
        thresholds["z_m"]
    ) and float(delta_components["pitch_roll_norm_deg"]) <= float(
        thresholds["pitch_roll_norm_deg"]
    )
    holdout_gain = (
        float(holdout_delta["fitness_mean_delta"])
        >= float(thresholds["fitness_mean_delta"])
        and float(holdout_delta["overlap_ratio_mean_delta"])
        >= float(thresholds["overlap_ratio_mean_delta"])
        and float(holdout_delta["inlier_rmse_mean_delta"])
        <= float(thresholds["inlier_rmse_mean_delta"])
    )
    conditioning_ok = holdout_delta.get("condition_number_ratio") is None or float(
        holdout_delta["condition_number_ratio"]
    ) <= float(thresholds["condition_number_ratio"])

    if holdout_gain and planar_ok and vertical_ok and conditioning_ok:
        recommendation = "accept_candidate"
    elif holdout_gain and planar_ok and not vertical_ok:
        recommendation = "planar_only_or_diagnostic"
    else:
        recommendation = "keep_baseline"

    return {
        "thresholds": thresholds,
        "statuses": {
            "planar_drift": "pass" if planar_ok else "warning",
            "vertical_drift": "pass" if vertical_ok else "warning",
            "holdout_gain": "pass" if holdout_gain else "warning",
            "conditioning": "pass" if conditioning_ok else "warning",
        },
        "recommendation": recommendation,
    }


def nearest_index(sorted_values: list[int], value: int) -> int | None:
    if not sorted_values:
        return None
    index = bisect.bisect_left(sorted_values, value)
    candidates = []
    if index < len(sorted_values):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(sorted_values[candidate] - value))


def find_nearest_meta(
    metas: list, timestamps: list[int], timestamp_ns: int, max_delta_ns: int
):
    index = nearest_index(timestamps, timestamp_ns)
    if index is None:
        return None, None
    delta_ns = abs(int(timestamps[index]) - int(timestamp_ns))
    if delta_ns > max_delta_ns:
        return None, delta_ns
    return metas[index], delta_ns


def prepare_output_layout(output_dir: Path) -> tuple[Path, Path, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for legacy_file in (
        output_dir / "calibrated_tf.yaml",
        output_dir / "metrics.yaml",
    ):
        if legacy_file.exists() and legacy_file.is_file():
            legacy_file.unlink()

    for diagnostics_file in (
        diagnostics_dir / "scan2map_dataset.yaml",
        diagnostics_dir / "scan2map_optimization.yaml",
        diagnostics_dir / "evaluation.yaml",
        diagnostics_dir / "manifest.yaml",
    ):
        if diagnostics_file.exists() and diagnostics_file.is_file():
            diagnostics_file.unlink()

    initial_guess_dir = output_dir / "initial_guess"
    calibrated_dir = output_dir / "calibrated"
    initial_guess_dir.mkdir(parents=True, exist_ok=True)
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    for directory in (initial_guess_dir, calibrated_dir):
        for file_path in directory.glob("*_extrinsics.yaml"):
            if file_path.is_file():
                file_path.unlink()
    return initial_guess_dir, calibrated_dir, diagnostics_dir


def load_dataset_report(dataset_path: str) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = yaml.safe_load(file)
    for frame in dataset.get("aligned_frames", []):
        frame["transform_map_lidar_matrix"] = matrix_from_transform_dict(
            frame["map_pose"]
        )
    return dataset


def load_scan2scan_baseline_transforms(path: str | None) -> dict[str, np.ndarray]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    transforms = {}
    for extrinsic in payload.get("extrinsics", []):
        child_frame = extrinsic.get("child_frame_id", "")
        if child_frame:
            transforms[child_frame] = matrix_from_transform_dict(extrinsic["transform"])
    return transforms


def choose_source_topics(
    topic_infos: dict[str, dict],
    *,
    target_topic: str,
    target_frame: str,
    explicit_sources: list[str] | None,
    tf_graph: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict], list[dict]]:
    requested = set(explicit_sources or [])
    selected = []
    skipped = []
    for topic, info in sorted(topic_infos.items()):
        if topic == target_topic:
            continue
        if requested and topic not in requested:
            continue
        if info["frame_id"] == target_frame:
            skipped.append(
                {
                    "source_topic": topic,
                    "target_topic": target_topic,
                    "reason": "same_target_frame",
                }
            )
            continue
        initial_transform = lookup_transform(tf_graph, info["frame_id"], target_frame)
        if initial_transform is None:
            skipped.append(
                {
                    "source_topic": topic,
                    "target_topic": target_topic,
                    "reason": "no_tf_path",
                }
            )
            continue
        selected.append(
            {
                "source_topic": topic,
                "source_frame": info["frame_id"],
                "target_topic": target_topic,
                "target_frame": target_frame,
                "initial_transform": initial_transform.tolist(),
            }
        )
    return selected, skipped


def build_submap_lookup(
    dataset: dict,
) -> tuple[dict[str, dict], dict[str, dict], list[dict], list[dict]]:
    aligned_frames = dataset.get("aligned_frames", [])
    keyframes = [frame for frame in aligned_frames if frame.get("keyframe_selected")]
    keyframe_by_id = {
        frame["keyframe_id"]: frame for frame in keyframes if frame.get("keyframe_id")
    }
    submaps = dataset.get("submaps", [])
    submap_by_id = {submap["submap_id"]: submap for submap in submaps}
    holdout_frames = [
        frame for frame in aligned_frames if frame.get("dataset_role") == "holdout"
    ]
    return keyframe_by_id, submap_by_id, keyframes, holdout_frames


def build_submap_cloud(
    submap: dict,
    *,
    keyframe_by_id: dict[str, dict],
    target_meta_by_timestamp: dict[int, object],
    cloud_cache: dict,
    submap_cache: dict,
    submap_voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, dict]:
    cache_key = (submap["submap_id"], float(submap_voxel_size))
    if cache_key in submap_cache:
        return submap_cache[cache_key]

    anchor_frame = keyframe_by_id[submap["anchor_keyframe_id"]]
    anchor_pose = anchor_frame["transform_map_lidar_matrix"]
    merged_cloud = o3d.geometry.PointCloud()
    support_records = []

    for keyframe_id in submap["support_keyframe_ids"]:
        support_frame = keyframe_by_id[keyframe_id]
        support_meta = target_meta_by_timestamp.get(int(support_frame["timestamp_ns"]))
        if support_meta is None:
            continue
        support_cloud = copy.deepcopy(load_cached_cloud(support_meta, cloud_cache))
        support_pose = support_frame["transform_map_lidar_matrix"]
        transform_anchor_support = inverse_transform(anchor_pose) @ support_pose
        support_cloud.transform(transform_anchor_support)
        merged_cloud += support_cloud
        support_records.append(
            {
                "keyframe_id": keyframe_id,
                "timestamp_ns": int(support_frame["timestamp_ns"]),
                "point_count": int(len(support_cloud.points)),
                "distance_to_anchor_m": float(
                    np.linalg.norm(
                        support_frame["transform_map_lidar_matrix"][:3, 3]
                        - anchor_pose[:3, 3]
                    )
                ),
            }
        )

    if submap_voxel_size > 0:
        merged_cloud = merged_cloud.voxel_down_sample(float(submap_voxel_size))

    info = {
        "submap_id": submap["submap_id"],
        "anchor_keyframe_id": submap["anchor_keyframe_id"],
        "anchor_timestamp_ns": int(submap["anchor_timestamp_ns"]),
        "support_keyframe_count": int(len(support_records)),
        "point_count": int(len(merged_cloud.points)),
        "support_records": support_records,
    }
    submap_cache[cache_key] = (merged_cloud, info)
    return merged_cloud, info


def load_preprocessed_submap(
    submap: dict,
    *,
    keyframe_by_id: dict[str, dict],
    target_meta_by_timestamp: dict[int, object],
    cloud_cache: dict,
    submap_cache: dict,
    preprocessing_params: dict,
    preprocessed_submap_cache: dict,
    submap_voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, dict]:
    cache_key = (
        submap["submap_id"],
        float(submap_voxel_size),
        freeze_preprocessing_params(preprocessing_params),
    )
    if cache_key not in preprocessed_submap_cache:
        submap_cloud, submap_info = build_submap_cloud(
            submap,
            keyframe_by_id=keyframe_by_id,
            target_meta_by_timestamp=target_meta_by_timestamp,
            cloud_cache=cloud_cache,
            submap_cache=submap_cache,
            submap_voxel_size=submap_voxel_size,
        )
        preprocessed_submap_cache[cache_key] = (
            preprocess_point_cloud(copy.deepcopy(submap_cloud), **preprocessing_params),
            submap_info,
        )
    return preprocessed_submap_cache[cache_key]


def average_transforms(
    transforms: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    translations = np.asarray(
        [transform[:3, 3] for transform in transforms], dtype=float
    )
    matrix[:3, 3] = np.average(translations, axis=0, weights=weights)
    rotations = R.from_matrix(
        np.asarray([transform[:3, :3] for transform in transforms], dtype=float)
    )
    matrix[:3, :3] = rotations.mean(
        weights=np.asarray(weights, dtype=float)
    ).as_matrix()
    return matrix


def summarize_delta_to_reference(
    reference_transform: np.ndarray, transforms: list[np.ndarray]
) -> dict:
    translation_deltas = []
    rotation_deltas = []
    for transform in transforms:
        delta = transform_delta_metrics(reference_transform, transform)
        translation_deltas.append(float(delta["translation_norm_m"]))
        rotation_deltas.append(float(delta["rotation_deg"]))
    return {
        "translation_m": summarize_values(translation_deltas),
        "rotation_deg": summarize_values(rotation_deltas),
    }


def aggregate_method_runs(
    run_records: list[dict],
    *,
    initial_transform: np.ndarray,
    min_consensus_runs: int,
    consensus_translation_m: float,
    consensus_rotation_deg: float,
    constraint_reference_transform: np.ndarray | None,
    constraint_reference_name: str,
    locked_components: list[str] | None,
) -> tuple[dict | None, list[dict], list[dict]]:
    if not run_records:
        return None, [], []

    locked_components = normalize_locked_components(locked_components)
    projected_runs = []
    for item in run_records:
        raw_transform = np.asarray(item["transformation"], dtype=float)
        projected_transform = constrain_transform_components(
            raw_transform,
            reference_transform=constraint_reference_transform,
            locked_components=locked_components,
        )
        projected_item = dict(item)
        projected_item["transformation"] = projected_transform.tolist()
        if locked_components and constraint_reference_transform is not None:
            projected_item["raw_transformation"] = raw_transform.tolist()
            projected_item["projection_delta_to_raw"] = transform_delta_metrics(
                raw_transform,
                projected_transform,
            )
            projected_item["constraint"] = {
                "reference": constraint_reference_name,
                "locked_components": list(locked_components),
            }
        projected_runs.append(projected_item)

    filtered_runs = sorted(
        projected_runs,
        key=lambda item: (
            -float(item["fitness"]),
            float(item["inlier_rmse"]),
            float(item["information_matrix"]["condition_number"]),
        ),
    )
    consensus_runs = filtered_runs
    candidate_transform = None
    for _ in range(2):
        weights = [
            max(float(item["fitness"]), 1e-3) / max(float(item["inlier_rmse"]), 1e-3)
            for item in consensus_runs
        ]
        candidate_transform = average_transforms(
            [
                np.asarray(item["transformation"], dtype=float)
                for item in consensus_runs
            ],
            weights,
        )
        next_consensus = []
        for item in consensus_runs:
            delta = transform_delta_metrics(
                candidate_transform, np.asarray(item["transformation"], dtype=float)
            )
            if float(delta["translation_norm_m"]) <= float(
                consensus_translation_m
            ) and float(delta["rotation_deg"]) <= float(consensus_rotation_deg):
                next_consensus.append(item)
        if len(next_consensus) < min_consensus_runs or len(next_consensus) == len(
            consensus_runs
        ):
            break
        consensus_runs = next_consensus

    if candidate_transform is None or len(consensus_runs) < min_consensus_runs:
        return None, filtered_runs, consensus_runs

    delta_to_initial = transform_delta_metrics(initial_transform, candidate_transform)
    condition_numbers = [
        float(item["information_matrix"]["condition_number"]) for item in consensus_runs
    ]
    optimization_fitness = [float(item["fitness"]) for item in consensus_runs]
    optimization_rmse = [float(item["inlier_rmse"]) for item in consensus_runs]
    overlap_values = [float(item["overlap_ratio"]) for item in consensus_runs]
    aggregated = {
        "candidate_transform": candidate_transform.tolist(),
        "delta_to_initial": delta_to_initial,
        "consensus_runs": len(consensus_runs),
        "optimization_fitness": summarize_values(optimization_fitness),
        "optimization_inlier_rmse": summarize_values(optimization_rmse),
        "optimization_overlap_ratio": summarize_values(overlap_values),
        "optimization_condition_number": summarize_values(condition_numbers),
        "consensus_delta_to_candidate": summarize_delta_to_reference(
            candidate_transform,
            [
                np.asarray(item["transformation"], dtype=float)
                for item in consensus_runs
            ],
        ),
        "constraint": {
            "reference": constraint_reference_name,
            "locked_components": list(locked_components),
        },
    }
    return aggregated, filtered_runs, consensus_runs


def collect_optimization_pairs(
    edge: dict,
    *,
    optimization_submaps: list[dict],
    source_metas: list,
    source_timestamps: list[int],
    sync_threshold_ns: int,
    keyframe_by_id: dict[str, dict],
    max_pairs: int,
) -> tuple[list[dict], list[dict]]:
    pairs = []
    skipped = []
    for submap in optimization_submaps:
        anchor_frame = keyframe_by_id[submap["anchor_keyframe_id"]]
        source_meta, delta_ns = find_nearest_meta(
            source_metas,
            source_timestamps,
            int(anchor_frame["timestamp_ns"]),
            sync_threshold_ns,
        )
        if source_meta is None:
            skipped.append(
                {
                    "source_topic": edge["source_topic"],
                    "submap_id": submap["submap_id"],
                    "anchor_timestamp_ns": int(anchor_frame["timestamp_ns"]),
                    "reason": "no_synced_source_frame",
                    "sync_dt_ms": (
                        float(delta_ns / 1e6) if delta_ns is not None else None
                    ),
                }
            )
            continue
        pairs.append(
            {
                "submap_id": submap["submap_id"],
                "anchor_keyframe_id": submap["anchor_keyframe_id"],
                "anchor_timestamp_ns": int(anchor_frame["timestamp_ns"]),
                "source_meta": source_meta,
                "sync_dt_ms": float(delta_ns / 1e6),
            }
        )
    return select_pairs_for_iteration(pairs, max_pairs, "anchor_timestamp_ns"), skipped


def choose_holdout_submap(
    frame: dict, optimization_submaps: list[dict], keyframe_by_id: dict[str, dict]
) -> dict | None:
    if not optimization_submaps:
        return None
    frame_position = frame["transform_map_lidar_matrix"][:3, 3]
    return min(
        optimization_submaps,
        key=lambda submap: (
            float(
                np.linalg.norm(
                    keyframe_by_id[submap["anchor_keyframe_id"]][
                        "transform_map_lidar_matrix"
                    ][:3, 3]
                    - frame_position
                )
            ),
            abs(int(submap["anchor_timestamp_ns"]) - int(frame["timestamp_ns"])),
        ),
    )


def select_pairs_for_iteration(
    pairs: list[dict], max_items: int, timestamp_key: str
) -> list[dict]:
    if max_items <= 0 or len(pairs) <= max_items:
        return list(pairs)
    shortlist = sorted(
        pairs,
        key=lambda item: (
            float(item["sync_dt_ms"]),
            int(item[timestamp_key]),
        ),
    )[: max(max_items * 3, max_items)]
    shortlist.sort(key=lambda item: int(item[timestamp_key]))
    return uniform_sample(shortlist, max_items)


def collect_holdout_pairs(
    edge: dict,
    *,
    holdout_frames: list[dict],
    optimization_submaps: list[dict],
    source_metas: list,
    source_timestamps: list[int],
    sync_threshold_ns: int,
    keyframe_by_id: dict[str, dict],
    max_pairs: int,
) -> tuple[list[dict], list[dict]]:
    pairs = []
    skipped = []
    for frame in holdout_frames:
        source_meta, delta_ns = find_nearest_meta(
            source_metas,
            source_timestamps,
            int(frame["timestamp_ns"]),
            sync_threshold_ns,
        )
        if source_meta is None:
            skipped.append(
                {
                    "source_topic": edge["source_topic"],
                    "target_timestamp_ns": int(frame["timestamp_ns"]),
                    "reason": "no_synced_source_frame",
                    "sync_dt_ms": (
                        float(delta_ns / 1e6) if delta_ns is not None else None
                    ),
                }
            )
            continue
        submap = choose_holdout_submap(frame, optimization_submaps, keyframe_by_id)
        if submap is None:
            skipped.append(
                {
                    "source_topic": edge["source_topic"],
                    "target_timestamp_ns": int(frame["timestamp_ns"]),
                    "reason": "no_holdout_submap",
                }
            )
            continue
        anchor_frame = keyframe_by_id[submap["anchor_keyframe_id"]]
        transform_anchor_target = (
            inverse_transform(anchor_frame["transform_map_lidar_matrix"])
            @ frame["transform_map_lidar_matrix"]
        )
        pairs.append(
            {
                "holdout_timestamp_ns": int(frame["timestamp_ns"]),
                "source_meta": source_meta,
                "submap_id": submap["submap_id"],
                "anchor_keyframe_id": submap["anchor_keyframe_id"],
                "sync_dt_ms": float(delta_ns / 1e6),
                "transform_anchor_target": transform_anchor_target.tolist(),
                "distance_to_anchor_m": float(
                    np.linalg.norm(
                        anchor_frame["transform_map_lidar_matrix"][:3, 3]
                        - frame["transform_map_lidar_matrix"][:3, 3]
                    )
                ),
            }
        )
    return select_pairs_for_iteration(pairs, max_pairs, "holdout_timestamp_ns"), skipped


def evaluate_fixed_transform(
    fixed_transform: np.ndarray,
    pairs: list[dict],
    *,
    keyframe_by_id: dict[str, dict],
    submap_by_id: dict[str, dict],
    target_meta_by_timestamp: dict[int, object],
    cloud_cache: dict,
    submap_cache: dict,
    source_preprocess_cache: dict,
    submap_preprocess_cache: dict,
    evaluation_preprocessing: dict,
    evaluation_distance: float,
    evaluation_overlap_voxel_size: float,
    source_cloud_raw_cache: dict,
    target_submap_voxel_size: float,
) -> dict:
    fitness_values = []
    rmse_values = []
    overlap_values = []
    condition_numbers = []
    samples = []
    for pair in pairs:
        source_meta = pair["source_meta"]
        submap = submap_by_id[pair["submap_id"]]
        submap_cloud_raw, submap_info = build_submap_cloud(
            submap,
            keyframe_by_id=keyframe_by_id,
            target_meta_by_timestamp=target_meta_by_timestamp,
            cloud_cache=cloud_cache,
            submap_cache=submap_cache,
            submap_voxel_size=target_submap_voxel_size,
        )
        source_cloud_raw = load_cached_cloud(source_meta, source_cloud_raw_cache)
        source_cloud = load_preprocessed_source_cloud(
            source_meta,
            preprocessing_params=evaluation_preprocessing,
            cloud_cache=cloud_cache,
            preprocessed_source_cache=source_preprocess_cache,
        )
        target_cloud, _ = load_preprocessed_submap(
            submap,
            keyframe_by_id=keyframe_by_id,
            target_meta_by_timestamp=target_meta_by_timestamp,
            cloud_cache=cloud_cache,
            submap_cache=submap_cache,
            preprocessing_params=evaluation_preprocessing,
            preprocessed_submap_cache=submap_preprocess_cache,
            submap_voxel_size=target_submap_voxel_size,
        )
        evaluation_transform = (
            np.asarray(pair["transform_anchor_target"], dtype=float) @ fixed_transform
        )
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_cloud,
            target_cloud,
            evaluation_distance,
            evaluation_transform,
        )
        overlap_ratio = voxel_overlap_ratio(
            source_cloud_raw,
            submap_cloud_raw,
            evaluation_transform,
            evaluation_overlap_voxel_size,
        )
        info_metrics = compute_information_metrics(
            source_cloud_raw,
            submap_cloud_raw,
            evaluation_transform,
            max_correspondence_distance=evaluation_distance,
            downsample_voxel_size=max(
                evaluation_overlap_voxel_size, evaluation_preprocessing["voxel_size"]
            ),
        )
        fitness_values.append(float(evaluation.fitness))
        rmse_values.append(float(evaluation.inlier_rmse))
        overlap_values.append(float(overlap_ratio))
        condition_numbers.append(float(info_metrics["condition_number"]))
        samples.append(
            {
                "holdout_timestamp_ns": int(pair["holdout_timestamp_ns"]),
                "source_timestamp_ns": int(source_meta.timestamp_ns),
                "submap_id": pair["submap_id"],
                "anchor_keyframe_id": pair["anchor_keyframe_id"],
                "sync_dt_ms": float(pair["sync_dt_ms"]),
                "distance_to_anchor_m": float(pair["distance_to_anchor_m"]),
                "fitness": float(evaluation.fitness),
                "inlier_rmse": float(evaluation.inlier_rmse),
                "overlap_ratio": float(overlap_ratio),
                "condition_number": float(info_metrics["condition_number"]),
                "submap_point_count": int(submap_info["point_count"]),
            }
        )
    return {
        "count": int(len(samples)),
        "fitness": summarize_values(fitness_values),
        "inlier_rmse": summarize_values(rmse_values),
        "overlap_ratio": summarize_values(overlap_values),
        "condition_number": summarize_values(condition_numbers),
        "samples": samples,
    }


def build_method_summary(
    edge: dict,
    *,
    method: int,
    run_records: list[dict],
    initial_transform: np.ndarray,
    baseline_transform: np.ndarray | None,
    holdout_pairs: list[dict],
    keyframe_by_id: dict[str, dict],
    submap_by_id: dict[str, dict],
    target_meta_by_timestamp: dict[int, object],
    cloud_cache: dict,
    submap_cache: dict,
    source_preprocess_cache: dict,
    submap_preprocess_cache: dict,
    evaluation_preprocessing: dict,
    evaluation_distance: float,
    evaluation_overlap_voxel_size: float,
    source_cloud_raw_cache: dict,
    submap_voxel_size: float,
    args,
) -> dict:
    constraint_reference_transform, constraint_reference_name = (
        resolve_constraint_reference(
            initial_transform=initial_transform,
            baseline_transform=baseline_transform,
            constraint_reference=str(args.constraint_reference),
        )
    )
    locked_components = normalize_locked_components(args.lock_components)
    aggregated, filtered_runs, consensus_runs = aggregate_method_runs(
        run_records,
        initial_transform=initial_transform,
        min_consensus_runs=int(args.min_consensus_runs),
        consensus_translation_m=float(args.consensus_translation_m),
        consensus_rotation_deg=float(args.consensus_rotation_deg),
        constraint_reference_transform=constraint_reference_transform,
        constraint_reference_name=constraint_reference_name,
        locked_components=locked_components,
    )
    summary = {
        "method": int(method),
        "attempted_pairs": int(len(run_records)),
        "accepted_pairs_before_consensus": int(len(filtered_runs)),
        "consensus_pairs": int(len(consensus_runs)),
        "optimization_runs": filtered_runs,
        "consensus_runs_detail": consensus_runs,
        "candidate": None,
        "holdout_evaluation": {
            "initial": evaluate_fixed_transform(
                initial_transform,
                holdout_pairs,
                keyframe_by_id=keyframe_by_id,
                submap_by_id=submap_by_id,
                target_meta_by_timestamp=target_meta_by_timestamp,
                cloud_cache=cloud_cache,
                submap_cache=submap_cache,
                source_preprocess_cache=source_preprocess_cache,
                submap_preprocess_cache=submap_preprocess_cache,
                evaluation_preprocessing=evaluation_preprocessing,
                evaluation_distance=evaluation_distance,
                evaluation_overlap_voxel_size=evaluation_overlap_voxel_size,
                source_cloud_raw_cache=source_cloud_raw_cache,
                target_submap_voxel_size=submap_voxel_size,
            ),
            "scan2scan_baseline": (
                evaluate_fixed_transform(
                    baseline_transform,
                    holdout_pairs,
                    keyframe_by_id=keyframe_by_id,
                    submap_by_id=submap_by_id,
                    target_meta_by_timestamp=target_meta_by_timestamp,
                    cloud_cache=cloud_cache,
                    submap_cache=submap_cache,
                    source_preprocess_cache=source_preprocess_cache,
                    submap_preprocess_cache=submap_preprocess_cache,
                    evaluation_preprocessing=evaluation_preprocessing,
                    evaluation_distance=evaluation_distance,
                    evaluation_overlap_voxel_size=evaluation_overlap_voxel_size,
                    source_cloud_raw_cache=source_cloud_raw_cache,
                    target_submap_voxel_size=submap_voxel_size,
                )
                if baseline_transform is not None
                else None
            ),
            "scan2map_candidate": None,
        },
        "quality_gate_reasons": [],
        "constraint": {
            "reference": constraint_reference_name,
            "locked_components": list(locked_components),
        },
    }
    if aggregated is None:
        summary["quality_gate_reasons"].append("insufficient_consensus_runs")
        return summary

    candidate_transform = np.asarray(aggregated["candidate_transform"], dtype=float)
    holdout_candidate = evaluate_fixed_transform(
        candidate_transform,
        holdout_pairs,
        keyframe_by_id=keyframe_by_id,
        submap_by_id=submap_by_id,
        target_meta_by_timestamp=target_meta_by_timestamp,
        cloud_cache=cloud_cache,
        submap_cache=submap_cache,
        source_preprocess_cache=source_preprocess_cache,
        submap_preprocess_cache=submap_preprocess_cache,
        evaluation_preprocessing=evaluation_preprocessing,
        evaluation_distance=evaluation_distance,
        evaluation_overlap_voxel_size=evaluation_overlap_voxel_size,
        source_cloud_raw_cache=source_cloud_raw_cache,
        target_submap_voxel_size=submap_voxel_size,
    )
    summary["candidate"] = aggregated
    summary["holdout_evaluation"]["scan2map_candidate"] = holdout_candidate
    summary["candidate"]["pose_components"] = matrix_to_pose_components(
        candidate_transform
    )
    summary["candidate"]["delta_to_initial_components"] = component_delta_metrics(
        initial_transform,
        candidate_transform,
    )
    summary["candidate"]["holdout_delta_to_initial"] = build_holdout_delta_metrics(
        summary["holdout_evaluation"]["initial"],
        holdout_candidate,
    )
    if baseline_transform is not None:
        summary["candidate"]["delta_to_scan2scan_baseline"] = transform_delta_metrics(
            baseline_transform,
            candidate_transform,
        )
        summary["candidate"]["delta_to_scan2scan_baseline_components"] = (
            component_delta_metrics(
                baseline_transform,
                candidate_transform,
            )
        )
        summary["candidate"]["holdout_delta_to_scan2scan_baseline"] = (
            build_holdout_delta_metrics(
                summary["holdout_evaluation"]["scan2scan_baseline"],
                holdout_candidate,
            )
        )
        summary["candidate"]["vehicle_rig_assessment"] = build_vehicle_rig_assessment(
            delta_components=summary["candidate"][
                "delta_to_scan2scan_baseline_components"
            ],
            holdout_delta=summary["candidate"]["holdout_delta_to_scan2scan_baseline"],
        )
    if holdout_candidate["count"] == 0:
        summary["quality_gate_reasons"].append("no_holdout_pairs")
    if holdout_candidate["fitness"]["mean"] is None or float(
        holdout_candidate["fitness"]["mean"]
    ) < float(args.min_holdout_fitness):
        summary["quality_gate_reasons"].append("holdout_fitness_below_threshold")
    if holdout_candidate["condition_number"]["max"] is not None and float(
        holdout_candidate["condition_number"]["max"]
    ) > float(args.max_condition_number):
        summary["quality_gate_reasons"].append("condition_number_above_threshold")
    return summary


def build_edge_result(edge: dict, method_summaries: list[dict]) -> dict | None:
    valid = [
        summary
        for summary in method_summaries
        if not summary["quality_gate_reasons"] and summary["candidate"] is not None
    ]
    if not valid:
        return None
    valid.sort(
        key=lambda summary: (
            -(
                summary["holdout_evaluation"]["scan2map_candidate"]["fitness"]["mean"]
                or -1.0
            ),
            summary["holdout_evaluation"]["scan2map_candidate"]["inlier_rmse"]["mean"]
            or float("inf"),
            -int(summary["candidate"]["consensus_runs"]),
        ),
    )
    chosen = valid[0]
    return {
        **edge,
        "chosen_method": int(chosen["method"]),
        "candidate_transform": chosen["candidate"]["candidate_transform"],
        "candidate": chosen["candidate"],
        "holdout_evaluation": chosen["holdout_evaluation"],
        "method_summaries": method_summaries,
    }


def write_calibrated_edge_files(
    output_dir: Path, base_frame: str, edge_results: list[dict]
) -> list[str]:
    saved_paths = []
    for edge_result in edge_results:
        child_frame = edge_result["source_frame"]
        file_path = output_dir / extrinsics_filename(base_frame, child_frame)
        save_extrinsics_yaml(
            str(file_path),
            parent_frame=base_frame,
            child_frame=child_frame,
            matrix=np.asarray(edge_result["candidate_transform"], dtype=float),
            metadata={
                "source_topic": edge_result["source_topic"],
                "chosen_method": int(edge_result["chosen_method"]),
                "pipeline": "scan2map_candidate",
                "constraint_reference": edge_result["candidate"]
                .get("constraint", {})
                .get("reference", "none"),
                "locked_components": edge_result["candidate"]
                .get("constraint", {})
                .get("locked_components", []),
            },
        )
        saved_paths.append(str(file_path))
    return saved_paths


def build_tf_output(base_topic: str, base_frame: str, edge_results: list[dict]) -> dict:
    return {
        "base_topic": base_topic,
        "base_frame": base_frame,
        "extrinsics": [
            build_extrinsics_payload(
                parent_frame=base_frame,
                child_frame=edge_result["source_frame"],
                matrix=np.asarray(edge_result["candidate_transform"], dtype=float),
                metadata={
                    "source_topic": edge_result["source_topic"],
                    "chosen_method": int(edge_result["chosen_method"]),
                    "pipeline": "scan2map_candidate",
                    "constraint_reference": edge_result["candidate"]
                    .get("constraint", {})
                    .get("reference", "none"),
                    "locked_components": edge_result["candidate"]
                    .get("constraint", {})
                    .get("locked_components", []),
                },
            )
            for edge_result in edge_results
        ],
    }


def build_metrics_output(
    record_files: list[str],
    *,
    target_topic: str,
    target_frame: str,
    edge_results: list[dict],
    skipped_edges: list[dict],
    dataset_report: dict,
    output_dir: Path,
    runtime_sec: float,
    args,
) -> dict:
    holdout_fitness = []
    holdout_rmse = []
    holdout_overlap = []
    max_condition_numbers = []
    accepted_keyframes = []
    accepted_submaps = []
    per_edge = []

    for edge_result in edge_results:
        holdout = edge_result["holdout_evaluation"]["scan2map_candidate"]
        candidate = edge_result["candidate"]
        holdout_fitness.append(float(holdout["fitness"]["mean"]))
        holdout_rmse.append(float(holdout["inlier_rmse"]["mean"]))
        holdout_overlap.append(float(holdout["overlap_ratio"]["mean"]))
        max_condition_numbers.append(float(holdout["condition_number"]["max"]))
        accepted_keyframes.append(int(candidate["consensus_runs"]))
        accepted_submaps.append(int(candidate["consensus_runs"]))
        per_edge.append(
            {
                "extrinsics_file": str(
                    output_dir
                    / "calibrated"
                    / extrinsics_filename(target_frame, edge_result["source_frame"])
                ),
                "source_topic": edge_result["source_topic"],
                "source_frame": edge_result["source_frame"],
                "target_topic": target_topic,
                "target_frame": target_frame,
                "chosen_method": int(edge_result["chosen_method"]),
                "accepted_keyframes": int(candidate["consensus_runs"]),
                "accepted_submaps": int(candidate["consensus_runs"]),
                "delta_to_initial": candidate["delta_to_initial"],
                "delta_to_scan2scan_baseline": candidate.get(
                    "delta_to_scan2scan_baseline"
                ),
                "delta_to_initial_components": candidate.get(
                    "delta_to_initial_components"
                ),
                "delta_to_scan2scan_baseline_components": candidate.get(
                    "delta_to_scan2scan_baseline_components"
                ),
                "holdout_delta_to_initial": candidate.get("holdout_delta_to_initial"),
                "holdout_delta_to_scan2scan_baseline": candidate.get(
                    "holdout_delta_to_scan2scan_baseline"
                ),
                "vehicle_rig_assessment": candidate.get("vehicle_rig_assessment"),
                "constraint": candidate.get("constraint"),
                "optimization_fitness": candidate["optimization_fitness"],
                "optimization_inlier_rmse": candidate["optimization_inlier_rmse"],
                "optimization_overlap_ratio": candidate["optimization_overlap_ratio"],
                "optimization_condition_number": candidate[
                    "optimization_condition_number"
                ],
                "holdout_evaluation": edge_result["holdout_evaluation"],
                "method_summaries": edge_result["method_summaries"],
            }
        )

    summary = {
        "accepted_edges": int(len(edge_results)),
        "skipped_edges": int(len(skipped_edges)),
        "accepted_keyframes": summarize_values(
            [float(value) for value in accepted_keyframes]
        ),
        "accepted_submaps": summarize_values(
            [float(value) for value in accepted_submaps]
        ),
        "average_holdout_fitness": (
            float(np.mean(holdout_fitness)) if holdout_fitness else None
        ),
        "average_holdout_inlier_rmse": (
            float(np.mean(holdout_rmse)) if holdout_rmse else None
        ),
        "average_holdout_overlap_ratio": (
            float(np.mean(holdout_overlap)) if holdout_overlap else None
        ),
        "max_condition_number": (
            float(max(max_condition_numbers)) if max_condition_numbers else None
        ),
        "runtime_sec": float(runtime_sec),
    }
    coarse_metrics = {
        **summary,
        "statuses": {
            "coverage": "pass" if edge_results else "warning",
            "holdout_fitness": _quality_status(
                summary["average_holdout_fitness"],
                float(args.min_holdout_fitness),
                smaller_is_better=False,
            ),
            "condition_number": _quality_status(
                summary["max_condition_number"],
                float(args.max_condition_number),
                smaller_is_better=True,
            ),
            "consensus": (
                "pass"
                if accepted_keyframes
                and min(accepted_keyframes) >= int(args.min_consensus_runs)
                else "warning"
            ),
        },
    }
    fine_metrics = {
        "per_edge": per_edge,
        "skipped_edges": skipped_edges,
        "dataset_summary": dataset_report.get("summary", {}),
        "path_coverage": dataset_report.get("path_coverage", {}),
    }
    return {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "constraint": {
            "reference": str(args.constraint_reference),
            "locked_components": list(
                normalize_locked_components(args.lock_components)
            ),
        },
        "summary": summary,
        "per_edge": per_edge,
        "skipped_edges": skipped_edges,
        "coarse_metrics": coarse_metrics,
        "fine_metrics": fine_metrics,
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "metrics": str(output_dir / "metrics.yaml"),
            "diagnostics_dir": str(output_dir / "diagnostics"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="scan2map candidate calibration using optimization / holdout slices."
    )
    parser.add_argument(
        "--record-path",
        required=True,
        help="Path to a .record file or a directory containing split record files.",
    )
    parser.add_argument(
        "--dataset-yaml",
        required=True,
        help="Path to diagnostics/scan2map_dataset.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2lidar/scan2map_candidate",
        help="Directory for reports and output files.",
    )
    parser.add_argument(
        "--conf-dir",
        default="lidar2lidar/conf",
        help="Directory that stores fallback extrinsics YAML files.",
    )
    parser.add_argument(
        "--target-topic",
        default=None,
        help="Override target topic. Defaults to the topic recorded in the dataset artifact.",
    )
    parser.add_argument(
        "--source-topics",
        nargs="*",
        default=None,
        help="Optional explicit source point cloud topics.",
    )
    parser.add_argument(
        "--scan2scan-baseline-tf",
        default=None,
        help="Optional calibrated_tf.yaml from lidar2lidar-auto for transform delta comparison.",
    )
    parser.add_argument(
        "--constraint-reference",
        choices=["none", "initial", "scan2scan_baseline"],
        default="none",
        help="Reference transform used when locking transform components.",
    )
    parser.add_argument(
        "--lock-components",
        nargs="*",
        choices=["x", "y", "z", "yaw", "pitch", "roll"],
        default=None,
        help="Transform components to keep fixed to the chosen constraint reference.",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum source/target timestamp difference for synchronized scan selection.",
    )
    parser.add_argument(
        "--max-optimization-keyframes",
        type=int,
        default=15,
        help="Maximum optimization submaps sampled per source edge. Set to 0 to use all.",
    )
    parser.add_argument(
        "--max-holdout-frames",
        type=int,
        default=8,
        help="Maximum holdout frames sampled per source edge. Set to 0 to use all.",
    )
    parser.add_argument(
        "--submap-voxel-size",
        type=float,
        default=0.08,
        help="Voxel size applied after merging each target submap.",
    )
    parser.add_argument(
        "--overlap-voxel-size",
        type=float,
        default=0.5,
        help="Voxel size used for overlap reporting.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=int,
        default=[2],
        help="Registration methods to compare (1: point-to-plane, 2: GICP, 3: point-to-point).",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
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
        "--min-keyframe-fitness",
        type=float,
        default=0.90,
        help="Minimum optimization registration fitness kept before consensus.",
    )
    parser.add_argument(
        "--max-keyframe-rmse",
        type=float,
        default=0.05,
        help="Maximum optimization registration RMSE kept before consensus.",
    )
    parser.add_argument(
        "--min-consensus-runs",
        type=int,
        default=4,
        help="Minimum optimization runs required to keep a scan2map edge.",
    )
    parser.add_argument(
        "--consensus-translation-m",
        type=float,
        default=0.25,
        help="Maximum translation spread allowed inside the scan2map transform consensus set.",
    )
    parser.add_argument(
        "--consensus-rotation-deg",
        type=float,
        default=1.0,
        help="Maximum rotation spread allowed inside the scan2map transform consensus set.",
    )
    parser.add_argument(
        "--min-holdout-fitness",
        type=float,
        default=0.90,
        help="Minimum mean holdout fitness required to keep a scan2map edge.",
    )
    parser.add_argument(
        "--max-condition-number",
        type=float,
        default=5000.0,
        help="Maximum holdout information-matrix condition number allowed for accepted edges.",
    )
    args = parser.parse_args()
    args.lock_components = normalize_locked_components(args.lock_components)
    if args.lock_components and str(args.constraint_reference) == "none":
        parser.error("--lock-components requires --constraint-reference.")
    if (
        str(args.constraint_reference) == "scan2scan_baseline"
        and not args.scan2scan_baseline_tf
    ):
        parser.error(
            "--constraint-reference scan2scan_baseline requires --scan2scan-baseline-tf."
        )

    start_time = time.perf_counter()
    record_files = discover_record_files(args.record_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_guess_dir, calibrated_dir, diagnostics_dir = prepare_output_layout(
        output_dir
    )
    dataset_report = load_dataset_report(args.dataset_yaml)
    shutil.copyfile(args.dataset_yaml, diagnostics_dir / "scan2map_dataset.yaml")

    target_topic = args.target_topic or dataset_report["lidar_topic"]
    target_frame = dataset_report["lidar_frame"]
    keyframe_by_id, submap_by_id, keyframes, holdout_frames = build_submap_lookup(
        dataset_report
    )
    optimization_submaps = list(dataset_report.get("submaps", []))
    holdout_frames = list(holdout_frames)

    topic_counts = list_topics(record_files)
    pointcloud_topics = infer_pointcloud_topics(topic_counts)
    topic_frame_ids = get_topic_frame_ids(record_files, pointcloud_topics)
    topic_infos = {
        topic: {
            "count": int(topic_counts.get(topic, 0)),
            "frame_id": topic_frame_ids.get(topic, ""),
            "sensor_name": topic_sensor_name(topic),
        }
        for topic in pointcloud_topics
    }

    record_tf_edges = extract_tf_edges(record_files)
    conf_tf_edges = load_transform_edges_from_dir(args.conf_dir)
    tf_edges = merge_transform_edges(record_tf_edges, conf_tf_edges)
    tf_graph = build_transform_graph(tf_edges) if tf_edges else {}
    baseline_transforms = load_scan2scan_baseline_transforms(args.scan2scan_baseline_tf)

    selected_edges, skipped_precheck = choose_source_topics(
        topic_infos,
        target_topic=target_topic,
        target_frame=target_frame,
        explicit_sources=args.source_topics,
        tf_graph=tf_graph,
    )

    source_topics = [target_topic] + [edge["source_topic"] for edge in selected_edges]
    metadata_by_topic = collect_pointcloud_metadata(record_files, source_topics)
    target_meta_by_timestamp = {
        int(meta.timestamp_ns): meta for meta in metadata_by_topic[target_topic]
    }

    cloud_cache = {}
    submap_cache = {}
    source_preprocess_cache = {}
    submap_preprocess_cache = {}
    source_cloud_raw_cache = {}
    preprocessing_params = {
        "voxel_size": float(args.voxel_size),
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": 0.05,
        "height_range": args.max_height,
        "remove_ground": bool(args.remove_ground),
        "remove_walls": bool(args.remove_walls),
    }
    evaluation_preprocessing = dict(preprocessing_params)
    evaluation_distance = max(float(args.voxel_size) * 5.0, 0.1)
    sync_threshold_ns = int(args.sync_threshold_ms * 1e6)

    edge_results = []
    skipped_edges = list(skipped_precheck)
    for edge in selected_edges:
        source_metas = metadata_by_topic.get(edge["source_topic"], [])
        source_timestamps = [int(meta.timestamp_ns) for meta in source_metas]
        optimization_pairs, skipped_optimization_pairs = collect_optimization_pairs(
            edge,
            optimization_submaps=optimization_submaps,
            source_metas=source_metas,
            source_timestamps=source_timestamps,
            sync_threshold_ns=sync_threshold_ns,
            keyframe_by_id=keyframe_by_id,
            max_pairs=int(args.max_optimization_keyframes),
        )
        holdout_pairs, skipped_holdout_pairs = collect_holdout_pairs(
            edge,
            holdout_frames=holdout_frames,
            optimization_submaps=optimization_submaps,
            source_metas=source_metas,
            source_timestamps=source_timestamps,
            sync_threshold_ns=sync_threshold_ns,
            keyframe_by_id=keyframe_by_id,
            max_pairs=int(args.max_holdout_frames),
        )
        if not optimization_pairs:
            skipped_edges.append(
                {
                    **edge,
                    "reason": "no_optimization_pairs",
                    "skipped_optimization_pairs": skipped_optimization_pairs,
                }
            )
            continue

        method_summaries = []
        initial_transform = np.asarray(edge["initial_transform"], dtype=float)
        baseline_transform = baseline_transforms.get(edge["source_frame"])
        for method in args.methods:
            run_records = []
            method_skips = []
            for pair in optimization_pairs:
                source_meta = pair["source_meta"]
                submap = submap_by_id[pair["submap_id"]]
                source_cloud = load_cached_cloud(source_meta, cloud_cache)
                target_submap, submap_info = build_submap_cloud(
                    submap,
                    keyframe_by_id=keyframe_by_id,
                    target_meta_by_timestamp=target_meta_by_timestamp,
                    cloud_cache=cloud_cache,
                    submap_cache=submap_cache,
                    submap_voxel_size=float(args.submap_voxel_size),
                )
                if len(source_cloud.points) == 0 or len(target_submap.points) == 0:
                    method_skips.append(
                        {
                            "submap_id": pair["submap_id"],
                            "source_timestamp_ns": int(source_meta.timestamp_ns),
                            "reason": "empty_cloud",
                        }
                    )
                    continue
                final_transform, _, reg_result = calibrate_lidar_extrinsic(
                    source_cloud,
                    target_submap,
                    is_draw_registration=False,
                    preprocessing_params=preprocessing_params,
                    method=int(method),
                    initial_transform=initial_transform,
                )
                if final_transform is None or reg_result is None:
                    method_skips.append(
                        {
                            "submap_id": pair["submap_id"],
                            "source_timestamp_ns": int(source_meta.timestamp_ns),
                            "reason": "registration_failed",
                        }
                    )
                    continue
                overlap_ratio = voxel_overlap_ratio(
                    source_cloud,
                    target_submap,
                    final_transform,
                    float(args.overlap_voxel_size),
                )
                info_metrics = compute_information_metrics(
                    source_cloud,
                    target_submap,
                    final_transform,
                    max_correspondence_distance=evaluation_distance,
                    downsample_voxel_size=max(
                        float(args.voxel_size), float(args.submap_voxel_size)
                    ),
                )
                run_record = {
                    "method": int(method),
                    "submap_id": pair["submap_id"],
                    "anchor_keyframe_id": pair["anchor_keyframe_id"],
                    "anchor_timestamp_ns": int(pair["anchor_timestamp_ns"]),
                    "source_timestamp_ns": int(source_meta.timestamp_ns),
                    "sync_dt_ms": float(pair["sync_dt_ms"]),
                    "fitness": float(reg_result.fitness),
                    "inlier_rmse": float(reg_result.inlier_rmse),
                    "overlap_ratio": float(overlap_ratio),
                    "transformation": final_transform.tolist(),
                    "delta_to_initial": transform_delta_metrics(
                        initial_transform, final_transform
                    ),
                    "information_matrix": info_metrics,
                    "submap_info": submap_info,
                }
                if float(run_record["fitness"]) >= float(
                    args.min_keyframe_fitness
                ) and float(run_record["inlier_rmse"]) <= float(args.max_keyframe_rmse):
                    run_records.append(run_record)
                else:
                    method_skips.append(
                        {
                            **run_record,
                            "reason": "keyframe_quality_gate_failed",
                        }
                    )

            method_summary = build_method_summary(
                edge,
                method=int(method),
                run_records=run_records,
                initial_transform=initial_transform,
                baseline_transform=baseline_transform,
                holdout_pairs=holdout_pairs,
                keyframe_by_id=keyframe_by_id,
                submap_by_id=submap_by_id,
                target_meta_by_timestamp=target_meta_by_timestamp,
                cloud_cache=cloud_cache,
                submap_cache=submap_cache,
                source_preprocess_cache=source_preprocess_cache,
                submap_preprocess_cache=submap_preprocess_cache,
                evaluation_preprocessing=evaluation_preprocessing,
                evaluation_distance=evaluation_distance,
                evaluation_overlap_voxel_size=float(args.overlap_voxel_size),
                source_cloud_raw_cache=source_cloud_raw_cache,
                submap_voxel_size=float(args.submap_voxel_size),
                args=args,
            )
            method_summary["skipped_optimization_pairs"] = method_skips
            method_summary["skipped_holdout_pairs"] = skipped_holdout_pairs
            method_summary["optimization_pair_summary"] = {
                "attempted": int(len(optimization_pairs)),
                "kept_after_keyframe_gate": int(len(run_records)),
                "skipped": int(len(method_skips)),
            }
            method_summary["holdout_pair_summary"] = {
                "attempted": int(len(holdout_frames)),
                "kept": int(len(holdout_pairs)),
                "skipped": int(len(skipped_holdout_pairs)),
            }
            method_summaries.append(method_summary)

        edge_result = build_edge_result(edge, method_summaries)
        if edge_result is None:
            skipped_edges.append(
                {
                    **edge,
                    "reason": "scan2map_quality_gate_failed",
                    "method_summaries": method_summaries,
                    "skipped_optimization_pairs": skipped_optimization_pairs,
                    "skipped_holdout_pairs": skipped_holdout_pairs,
                }
            )
            continue
        edge_results.append(edge_result)

    edge_results.sort(
        key=lambda item: (
            -(
                item["holdout_evaluation"]["scan2map_candidate"]["fitness"]["mean"]
                or -1.0
            ),
            item["holdout_evaluation"]["scan2map_candidate"]["inlier_rmse"]["mean"]
            or float("inf"),
        ),
    )

    initial_guess_paths = []
    for edge in selected_edges:
        file_path = initial_guess_dir / extrinsics_filename(
            target_frame, edge["source_frame"]
        )
        save_extrinsics_yaml(
            str(file_path),
            parent_frame=target_frame,
            child_frame=edge["source_frame"],
            matrix=np.asarray(edge["initial_transform"], dtype=float),
            metadata={
                "source_topic": edge["source_topic"],
                "pipeline": "scan2map_candidate_initial_guess",
            },
        )
        initial_guess_paths.append(str(file_path))

    calibrated_paths = write_calibrated_edge_files(
        calibrated_dir, target_frame, edge_results
    )
    tf_output = build_tf_output(target_topic, target_frame, edge_results)
    with open(output_dir / "calibrated_tf.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(tf_output, file, sort_keys=False)

    runtime_sec = time.perf_counter() - start_time
    metrics_output = build_metrics_output(
        record_files,
        target_topic=target_topic,
        target_frame=target_frame,
        edge_results=edge_results,
        skipped_edges=skipped_edges,
        dataset_report=dataset_report,
        output_dir=output_dir,
        runtime_sec=runtime_sec,
        args=args,
    )
    with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)

    optimization_report = {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "optimization_submap_ids": [
            submap["submap_id"] for submap in optimization_submaps
        ],
        "holdout_timestamp_ns": [
            int(frame["timestamp_ns"]) for frame in holdout_frames
        ],
        "edge_results": edge_results,
        "skipped_edges": skipped_edges,
    }
    with open(
        diagnostics_dir / "scan2map_optimization.yaml", "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(optimization_report, file, sort_keys=False)

    evaluation_report = {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "summary": metrics_output["summary"],
        "per_edge": metrics_output["per_edge"],
    }
    with open(diagnostics_dir / "evaluation.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_report, file, sort_keys=False)

    manifest = {
        "record_files": record_files,
        "target_topic": target_topic,
        "target_frame": target_frame,
        "dataset_yaml": str(args.dataset_yaml),
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "metrics": str(output_dir / "metrics.yaml"),
            "initial_guess_dir": str(initial_guess_dir),
            "calibrated_dir": str(calibrated_dir),
            "diagnostics": {
                "scan2map_dataset": str(diagnostics_dir / "scan2map_dataset.yaml"),
                "scan2map_optimization": str(
                    diagnostics_dir / "scan2map_optimization.yaml"
                ),
                "evaluation": str(diagnostics_dir / "evaluation.yaml"),
                "manifest": str(diagnostics_dir / "manifest.yaml"),
            },
            "initial_guess_files": initial_guess_paths,
            "calibrated_files": calibrated_paths,
        },
    }
    with open(diagnostics_dir / "manifest.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    logging.info(
        "scan2map candidate finished | target=%s | accepted_edges=%d | runtime=%.2fs",
        target_topic,
        len(edge_results),
        runtime_sec,
    )


if __name__ == "__main__":
    main()
