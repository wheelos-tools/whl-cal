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
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2lidar.extrinsic_io import transform_dict_from_matrix
from lidar2lidar.record_adapter import Record, ensure_record_available
from lidar2lidar.record_utils import (
    build_transform_graph,
    collect_pointcloud_metadata,
    discover_record_files,
    extract_tf_edges,
    get_topic_frame_ids,
    infer_pointcloud_topics,
    list_topics,
    load_transform_edges_from_dir,
    lookup_transform,
    merge_transform_edges,
    save_transform_edges_to_dir,
    topic_sensor_name,
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


def _pose_to_matrix(position, orientation) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_quat([
        float(orientation.qx),
        float(orientation.qy),
        float(orientation.qz),
        float(orientation.qw),
    ]).as_matrix()
    transform[:3, 3] = [
        float(position.x),
        float(position.y),
        float(position.z),
    ]
    return transform


def rotation_angle_degrees(rotation_matrix: np.ndarray) -> float:
    trace = float(np.trace(rotation_matrix))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def relative_transform(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.inv(reference) @ target


def yaw_degrees(transform: np.ndarray) -> float:
    return float(R.from_matrix(transform[:3, :3]).as_euler("zyx", degrees=True)[0])


def topic_preference_sort_key(topic: str, topic_infos: dict[str, dict]) -> tuple:
    lowered = topic.lower()
    return (
        "fusion" in lowered,
        0 if "main" in lowered else 1,
        -topic_infos[topic]["count"],
        topic,
    )


def choose_lidar_topic(topic_infos: dict[str, dict], explicit_topic: str | None) -> str:
    if explicit_topic is not None:
        if explicit_topic not in topic_infos:
            raise ValueError(f"Unknown LiDAR topic: {explicit_topic}")
        return explicit_topic

    candidates = sorted(topic_infos, key=lambda topic: topic_preference_sort_key(topic, topic_infos))
    if not candidates:
        raise RuntimeError("No PointCloud2 topics were found in the input record.")
    return candidates[0]


def prepare_output_layout(output_dir: Path) -> Path:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for file_path in (
        diagnostics_dir / "scan2map_dataset.yaml",
        diagnostics_dir / "manifest.yaml",
    ):
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
    return diagnostics_dir


def collect_pose_samples(record_files: list[str], pose_topic: str) -> list[dict]:
    ensure_record_available()
    samples = []
    for record_file in record_files:
        with Record(record_file) as record:
            for _, msg, timestamp_ns in record.read_messages(topics=[pose_topic]):
                pose = getattr(msg, "pose", None)
                if pose is None:
                    continue
                transform_world_localization = _pose_to_matrix(pose.position, pose.orientation)
                samples.append({
                    "timestamp_ns": int(timestamp_ns),
                    "transform_world_localization": transform_world_localization,
                })
    samples.sort(key=lambda item: item["timestamp_ns"])
    return samples


def nearest_index(sorted_timestamps: list[int], timestamp_ns: int) -> int | None:
    if not sorted_timestamps:
        return None
    index = bisect.bisect_left(sorted_timestamps, timestamp_ns)
    candidates = []
    if index < len(sorted_timestamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(sorted_timestamps[candidate] - timestamp_ns))


def nearest_pose_sample(pose_samples: list[dict],
                        pose_timestamps: list[int],
                        timestamp_ns: int,
                        max_delta_ns: int) -> tuple[dict | None, int | None]:
    index = nearest_index(pose_timestamps, timestamp_ns)
    if index is None:
        return None, None
    delta_ns = abs(int(pose_timestamps[index]) - int(timestamp_ns))
    if delta_ns > max_delta_ns:
        return None, delta_ns
    return pose_samples[index], delta_ns


def build_path_coverage(aligned_frames: list[dict]) -> tuple[dict, list[float], list[float], list[float]]:
    if not aligned_frames:
        return {
            "aligned_frame_count": 0,
            "total_path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "total_rotation_change_deg": 0.0,
            "left_turn_count": 0,
            "right_turn_count": 0,
            "straight_segment_count": 0,
        }, [], [], []

    translation_steps = []
    rotation_steps = []
    yaw_steps = []
    left_turn_count = 0
    right_turn_count = 0
    straight_segment_count = 0
    for index in range(1, len(aligned_frames)):
        previous = aligned_frames[index - 1]["transform_map_lidar"]
        current = aligned_frames[index]["transform_map_lidar"]
        delta = relative_transform(previous, current)
        translation_m = float(np.linalg.norm(delta[:3, 3]))
        rotation_deg = rotation_angle_degrees(delta[:3, :3])
        yaw_delta_deg = yaw_degrees(current) - yaw_degrees(previous)
        while yaw_delta_deg > 180.0:
            yaw_delta_deg -= 360.0
        while yaw_delta_deg < -180.0:
            yaw_delta_deg += 360.0
        translation_steps.append(translation_m)
        rotation_steps.append(rotation_deg)
        yaw_steps.append(float(yaw_delta_deg))
        if yaw_delta_deg > 1.0:
            left_turn_count += 1
        elif yaw_delta_deg < -1.0:
            right_turn_count += 1
        else:
            straight_segment_count += 1

    first_position = aligned_frames[0]["transform_map_lidar"][:3, 3]
    last_position = aligned_frames[-1]["transform_map_lidar"][:3, 3]
    coverage = {
        "aligned_frame_count": int(len(aligned_frames)),
        "total_path_length_m": float(sum(translation_steps)),
        "net_displacement_m": float(np.linalg.norm(last_position - first_position)),
        "total_rotation_change_deg": float(sum(abs(value) for value in rotation_steps)),
        "left_turn_count": int(left_turn_count),
        "right_turn_count": int(right_turn_count),
        "straight_segment_count": int(straight_segment_count),
    }
    return coverage, translation_steps, rotation_steps, yaw_steps


def serialize_frame(frame: dict) -> dict:
    transform = frame["transform_map_lidar"]
    payload = {
        "aligned_index": int(frame["aligned_index"]),
        "timestamp_ns": int(frame["timestamp_ns"]),
        "record_path": frame["record_path"],
        "frame_id": frame["frame_id"],
        "pose_timestamp_ns": int(frame["pose_timestamp_ns"]),
        "pose_sync_dt_ms": float(frame["pose_sync_dt_ms"]),
        "dataset_role": frame["dataset_role"],
        "keyframe_selected": bool(frame["keyframe_selected"]),
        "keyframe_id": frame["keyframe_id"],
        "map_pose": transform_dict_from_matrix(transform),
        "map_position": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
        "yaw_deg": yaw_degrees(transform),
    }
    if frame.get("delta_from_previous_aligned") is not None:
        payload["delta_from_previous_aligned"] = frame["delta_from_previous_aligned"]
    if frame.get("delta_from_previous_keyframe") is not None:
        payload["delta_from_previous_keyframe"] = frame["delta_from_previous_keyframe"]
    if frame.get("keyframe_triggers"):
        payload["keyframe_triggers"] = frame["keyframe_triggers"]
    if frame.get("keyframe_skip_reasons"):
        payload["keyframe_skip_reasons"] = frame["keyframe_skip_reasons"]
    return payload


def build_submaps(keyframes: list[dict],
                  *,
                  max_submap_keyframes: int,
                  min_submap_keyframes: int,
                  submap_radius_m: float) -> tuple[list[dict], list[dict]]:
    submaps = []
    skipped_submaps = []
    if not keyframes:
        return submaps, skipped_submaps

    for anchor in keyframes:
        anchor_position = anchor["transform_map_lidar"][:3, 3]
        support_candidates = []
        for candidate in keyframes:
            candidate_position = candidate["transform_map_lidar"][:3, 3]
            distance_m = float(np.linalg.norm(candidate_position - anchor_position))
            if distance_m <= submap_radius_m:
                support_candidates.append((distance_m, abs(candidate["aligned_index"] - anchor["aligned_index"]), candidate))
        support_candidates.sort(key=lambda item: (item[0], item[1], item[2]["timestamp_ns"]))
        support_keyframes = [item[2] for item in support_candidates[:max_submap_keyframes]]
        support_keyframes.sort(key=lambda item: item["timestamp_ns"])

        if len(support_keyframes) < min_submap_keyframes:
            skipped_submaps.append({
                "anchor_keyframe_id": anchor["keyframe_id"],
                "anchor_timestamp_ns": int(anchor["timestamp_ns"]),
                "reason": "insufficient_support_keyframes",
                "support_keyframe_count": int(len(support_keyframes)),
            })
            continue

        support_distances = [
            float(np.linalg.norm(candidate["transform_map_lidar"][:3, 3] - anchor_position))
            for candidate in support_keyframes
        ]
        time_span_s = float((support_keyframes[-1]["timestamp_ns"] - support_keyframes[0]["timestamp_ns"]) / 1e9)
        path_span_m = 0.0
        for index in range(1, len(support_keyframes)):
            previous_position = support_keyframes[index - 1]["transform_map_lidar"][:3, 3]
            current_position = support_keyframes[index]["transform_map_lidar"][:3, 3]
            path_span_m += float(np.linalg.norm(current_position - previous_position))

        submaps.append({
            "submap_id": f"submap_{anchor['keyframe_id']}",
            "anchor_keyframe_id": anchor["keyframe_id"],
            "anchor_timestamp_ns": int(anchor["timestamp_ns"]),
            "support_keyframe_ids": [candidate["keyframe_id"] for candidate in support_keyframes],
            "support_keyframe_count": int(len(support_keyframes)),
            "max_support_radius_m": float(max(support_distances)) if support_distances else 0.0,
            "median_support_radius_m": float(np.median(support_distances)) if support_distances else 0.0,
            "time_span_s": time_span_s,
            "path_span_m": float(path_span_m),
        })

    return submaps, skipped_submaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a scan2map dataset artifact from Apollo record files.")
    parser.add_argument("--record-path", required=True, help="Path to a .record file or a directory containing split record files.")
    parser.add_argument("--output-dir", default="outputs/lidar2lidar/scan2map_dataset", help="Directory for reports and output files.")
    parser.add_argument("--conf-dir", default="lidar2lidar/conf", help="Directory that stores fallback extrinsics YAML files.")
    parser.add_argument("--bootstrap-conf", action="store_true", help="Export record-derived static TF edges into the conf directory.")
    parser.add_argument("--lidar-topic", default=None, help="Point cloud topic used for scan2map extraction. Defaults to the preferred main/raw topic.")
    parser.add_argument("--pose-topic", default="/apollo/localization/pose", help="Pose topic used for map-frame alignment.")
    parser.add_argument("--pose-frame", default="localization", help="Pose frame expected by the static TF graph.")
    parser.add_argument("--map-frame", choices=["world", "localization"], default="world", help="Map frame used for dataset poses.")
    parser.add_argument("--pose-sync-threshold-ms", type=float, default=50.0, help="Maximum timestamp difference allowed between a scan and its aligned pose sample.")
    parser.add_argument("--holdout-stride", type=int, default=3, help="Every Nth aligned scan is assigned to holdout. Set to 0 to disable holdout.")
    parser.add_argument("--keyframe-translation-m", type=float, default=0.5, help="Minimum translation from the previous optimization keyframe before selecting a new keyframe.")
    parser.add_argument("--keyframe-rotation-deg", type=float, default=5.0, help="Minimum rotation from the previous optimization keyframe before selecting a new keyframe.")
    parser.add_argument("--min-keyframe-dt-sec", type=float, default=0.2, help="Minimum time spacing preferred between optimization keyframes.")
    parser.add_argument("--max-keyframe-interval-sec", type=float, default=2.0, help="Force a new optimization keyframe if this interval is exceeded.")
    parser.add_argument("--max-interval-min-translation-m", type=float, default=0.2, help="Minimum translation still required when the max keyframe interval is used as a fallback trigger.")
    parser.add_argument("--max-interval-min-rotation-deg", type=float, default=1.0, help="Minimum rotation still required when the max keyframe interval is used as a fallback trigger.")
    parser.add_argument("--max-submap-keyframes", type=int, default=20, help="Maximum optimization keyframes kept in each local submap definition.")
    parser.add_argument("--min-submap-keyframes", type=int, default=5, help="Minimum optimization keyframes required for an accepted submap definition.")
    parser.add_argument("--submap-radius-m", type=float, default=15.0, help="Maximum map-frame radius used when gathering support keyframes for a submap.")
    args = parser.parse_args()

    record_files = discover_record_files(args.record_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = prepare_output_layout(output_dir)

    logging.info("Using record files: %s", record_files)
    topic_counts = list_topics(record_files)
    if args.pose_topic not in topic_counts:
        raise RuntimeError(f"Pose topic {args.pose_topic} was not found in the input record.")

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
    lidar_topic = choose_lidar_topic(topic_infos, args.lidar_topic)
    lidar_frame = topic_infos[lidar_topic]["frame_id"]
    if not lidar_frame:
        raise RuntimeError(f"LiDAR topic {lidar_topic} does not expose a frame_id.")

    record_tf_edges = extract_tf_edges(record_files)
    conf_tf_edges = load_transform_edges_from_dir(args.conf_dir)
    tf_edges = merge_transform_edges(record_tf_edges, conf_tf_edges)
    tf_graph = build_transform_graph(tf_edges) if tf_edges else {}

    pointcloud_related_static_edges = [
        edge for edge in record_tf_edges
        if edge.is_static and (edge.parent_frame == lidar_frame or edge.child_frame == lidar_frame)
    ]
    if args.bootstrap_conf and pointcloud_related_static_edges:
        Path(args.conf_dir).mkdir(parents=True, exist_ok=True)
        save_transform_edges_to_dir(args.conf_dir, pointcloud_related_static_edges)

    transform_localization_lidar = lookup_transform(tf_graph, args.pose_frame, lidar_frame)
    if transform_localization_lidar is None:
        raise RuntimeError(f"Missing TF path from {args.pose_frame} to {lidar_frame}.")

    metadata_by_topic = collect_pointcloud_metadata(record_files, [lidar_topic])
    lidar_metas = metadata_by_topic[lidar_topic]
    if not lidar_metas:
        raise RuntimeError(f"No point clouds found on topic {lidar_topic}.")

    pose_samples = collect_pose_samples(record_files, args.pose_topic)
    if not pose_samples:
        raise RuntimeError(f"No pose samples found on topic {args.pose_topic}.")
    pose_timestamps = [sample["timestamp_ns"] for sample in pose_samples]

    sync_threshold_ns = int(args.pose_sync_threshold_ms * 1e6)
    aligned_frames = []
    skipped_frames = []
    for meta in lidar_metas:
        pose_sample, delta_ns = nearest_pose_sample(
            pose_samples,
            pose_timestamps,
            int(meta.timestamp_ns),
            sync_threshold_ns,
        )
        if pose_sample is None:
            skipped_frames.append({
                "timestamp_ns": int(meta.timestamp_ns),
                "record_path": meta.record_path,
                "frame_id": meta.frame_id,
                "reason": "no_pose_within_sync_threshold",
                "pose_sync_dt_ms": float(delta_ns / 1e6) if delta_ns is not None else None,
            })
            continue

        transform_map_localization = (
            pose_sample["transform_world_localization"]
            if args.map_frame == "world"
            else np.eye(4, dtype=float)
        )
        transform_map_lidar = transform_map_localization @ transform_localization_lidar
        aligned_frames.append({
            "aligned_index": int(len(aligned_frames)),
            "timestamp_ns": int(meta.timestamp_ns),
            "record_path": meta.record_path,
            "frame_id": meta.frame_id,
            "pose_timestamp_ns": int(pose_sample["timestamp_ns"]),
            "pose_sync_dt_ms": float(delta_ns / 1e6),
            "transform_map_lidar": transform_map_lidar,
            "dataset_role": None,
            "keyframe_selected": False,
            "keyframe_id": None,
            "keyframe_triggers": [],
            "keyframe_skip_reasons": [],
            "delta_from_previous_aligned": None,
            "delta_from_previous_keyframe": None,
        })

    if not aligned_frames:
        raise RuntimeError("No LiDAR frames could be aligned to pose samples.")

    coverage, aligned_translation_steps, aligned_rotation_steps, yaw_steps = build_path_coverage(aligned_frames)
    for index in range(1, len(aligned_frames)):
        previous_transform = aligned_frames[index - 1]["transform_map_lidar"]
        current_transform = aligned_frames[index]["transform_map_lidar"]
        delta = relative_transform(previous_transform, current_transform)
        aligned_frames[index]["delta_from_previous_aligned"] = {
            "translation_m": float(np.linalg.norm(delta[:3, 3])),
            "rotation_deg": rotation_angle_degrees(delta[:3, :3]),
            "dt_sec": float((aligned_frames[index]["timestamp_ns"] - aligned_frames[index - 1]["timestamp_ns"]) / 1e9),
        }

    keyframes = []
    keyframe_translation_steps = []
    keyframe_rotation_steps = []
    last_keyframe = None
    for frame in aligned_frames:
        if args.holdout_stride > 0 and frame["aligned_index"] % int(args.holdout_stride) == int(args.holdout_stride) - 1:
            frame["dataset_role"] = "holdout"
            frame["keyframe_skip_reasons"] = ["holdout_slice"]
            continue

        frame["dataset_role"] = "optimization"
        if last_keyframe is None:
            triggers = ["first_optimization_frame"]
            delta_translation_m = None
            delta_rotation_deg = None
            delta_dt_sec = None
        else:
            delta = relative_transform(last_keyframe["transform_map_lidar"], frame["transform_map_lidar"])
            delta_translation_m = float(np.linalg.norm(delta[:3, 3]))
            delta_rotation_deg = rotation_angle_degrees(delta[:3, :3])
            delta_dt_sec = float((frame["timestamp_ns"] - last_keyframe["timestamp_ns"]) / 1e9)
            triggers = []
            if delta_translation_m >= float(args.keyframe_translation_m):
                triggers.append("translation_threshold")
            if delta_rotation_deg >= float(args.keyframe_rotation_deg):
                triggers.append("rotation_threshold")
            if (
                delta_dt_sec >= float(args.max_keyframe_interval_sec)
                and (
                    delta_translation_m >= float(args.max_interval_min_translation_m)
                    or delta_rotation_deg >= float(args.max_interval_min_rotation_deg)
                )
            ):
                triggers.append("max_interval_threshold")
            if not triggers and delta_dt_sec < float(args.min_keyframe_dt_sec):
                frame["keyframe_skip_reasons"] = ["min_keyframe_dt"]
                continue
            if not triggers and delta_dt_sec >= float(args.max_keyframe_interval_sec):
                frame["keyframe_skip_reasons"] = ["low_motion_interval"]
                continue
            if not triggers:
                frame["keyframe_skip_reasons"] = ["redundant_motion"]
                continue

        frame["keyframe_selected"] = True
        frame["keyframe_triggers"] = triggers
        frame["keyframe_id"] = f"kf_{len(keyframes):04d}"
        if last_keyframe is not None:
            frame["delta_from_previous_keyframe"] = {
                "translation_m": float(delta_translation_m),
                "rotation_deg": float(delta_rotation_deg),
                "dt_sec": float(delta_dt_sec),
            }
            keyframe_translation_steps.append(float(delta_translation_m))
            keyframe_rotation_steps.append(float(delta_rotation_deg))
        keyframes.append(frame)
        last_keyframe = frame

    submaps, skipped_submaps = build_submaps(
        keyframes,
        max_submap_keyframes=int(max(1, args.max_submap_keyframes)),
        min_submap_keyframes=int(max(1, args.min_submap_keyframes)),
        submap_radius_m=float(args.submap_radius_m),
    )

    keyframe_windows = []
    for index in range(1, len(keyframes)):
        previous = keyframes[index - 1]
        current = keyframes[index]
        delta = current["delta_from_previous_keyframe"]
        keyframe_windows.append({
            "start_keyframe_id": previous["keyframe_id"],
            "end_keyframe_id": current["keyframe_id"],
            "start_timestamp_ns": int(previous["timestamp_ns"]),
            "end_timestamp_ns": int(current["timestamp_ns"]),
            "translation_m": float(delta["translation_m"]),
            "rotation_deg": float(delta["rotation_deg"]),
            "dt_sec": float(delta["dt_sec"]),
        })

    dataset_report = {
        "record_files": record_files,
        "conf_dir": args.conf_dir,
        "pose_topic": args.pose_topic,
        "pose_frame": args.pose_frame,
        "map_frame": args.map_frame,
        "lidar_topic": lidar_topic,
        "lidar_frame": lidar_frame,
        "selection_policy": {
            "holdout_stride": int(args.holdout_stride),
            "selection_pool": "optimization",
            "keyframe_translation_m": float(args.keyframe_translation_m),
            "keyframe_rotation_deg": float(args.keyframe_rotation_deg),
            "min_keyframe_dt_sec": float(args.min_keyframe_dt_sec),
            "max_keyframe_interval_sec": float(args.max_keyframe_interval_sec),
            "max_interval_min_translation_m": float(args.max_interval_min_translation_m),
            "max_interval_min_rotation_deg": float(args.max_interval_min_rotation_deg),
            "max_submap_keyframes": int(args.max_submap_keyframes),
            "min_submap_keyframes": int(args.min_submap_keyframes),
            "submap_radius_m": float(args.submap_radius_m),
            "pose_sync_threshold_ms": float(args.pose_sync_threshold_ms),
        },
        "initial_transform_source": {
            "type": "merged_tf_graph",
            "map_frame": args.map_frame,
            "pose_frame": args.pose_frame,
            "lidar_frame": lidar_frame,
            "transform_pose_to_lidar": transform_dict_from_matrix(transform_localization_lidar),
        },
        "summary": {
            "raw_pointcloud_count": int(len(lidar_metas)),
            "pose_sample_count": int(len(pose_samples)),
            "aligned_frame_count": int(len(aligned_frames)),
            "skipped_frame_count": int(len(skipped_frames)),
            "optimization_frame_count": int(sum(1 for frame in aligned_frames if frame["dataset_role"] == "optimization")),
            "holdout_frame_count": int(sum(1 for frame in aligned_frames if frame["dataset_role"] == "holdout")),
            "keyframe_count": int(len(keyframes)),
            "submap_count": int(len(submaps)),
            "skipped_submap_count": int(len(skipped_submaps)),
            "pose_sync_dt_ms": summarize_values([float(frame["pose_sync_dt_ms"]) for frame in aligned_frames]),
            "aligned_step_translation_m": summarize_values(aligned_translation_steps),
            "aligned_step_rotation_deg": summarize_values(aligned_rotation_steps),
            "keyframe_step_translation_m": summarize_values(keyframe_translation_steps),
            "keyframe_step_rotation_deg": summarize_values(keyframe_rotation_steps),
            "submap_support_keyframe_count": summarize_values([
                float(submap["support_keyframe_count"]) for submap in submaps
            ]),
            "submap_support_radius_m": summarize_values([
                float(submap["max_support_radius_m"]) for submap in submaps
            ]),
        },
        "path_coverage": {
            **coverage,
            "aligned_step_yaw_deg": summarize_values(yaw_steps),
        },
        "topic_counts": topic_counts,
        "topics": topic_infos,
        "aligned_frames": [serialize_frame(frame) for frame in aligned_frames],
        "skipped_frames": skipped_frames,
        "keyframe_windows": keyframe_windows,
        "submaps": submaps,
        "skipped_submaps": skipped_submaps,
    }

    dataset_path = diagnostics_dir / "scan2map_dataset.yaml"
    with open(dataset_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(dataset_report, file, sort_keys=False)

    manifest = {
        "record_files": record_files,
        "selected_lidar_topic": lidar_topic,
        "selected_lidar_frame": lidar_frame,
        "pose_topic": args.pose_topic,
        "pose_frame": args.pose_frame,
        "map_frame": args.map_frame,
        "artifacts": {
            "scan2map_dataset": str(dataset_path),
            "manifest": str(diagnostics_dir / "manifest.yaml"),
        },
    }
    with open(diagnostics_dir / "manifest.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)

    logging.info(
        "scan2map dataset ready | lidar=%s | aligned=%d | keyframes=%d | submaps=%d",
        lidar_topic,
        len(aligned_frames),
        len(keyframes),
        len(submaps),
    )
    logging.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
