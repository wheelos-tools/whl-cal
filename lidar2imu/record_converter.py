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

# isort: off
import argparse
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import (
    circular_span_deg,
    normalize_vector,
)
from lidar2imu.motion_information import (
    motion_information_components as _motion_information_components,
)
from lidar2imu.extraction.ground import extract_ground_plane as _extract_ground_plane
from lidar2imu.extraction.motion_candidates import (
    build_motion_candidates as _build_motion_candidates,
)
from lidar2imu.extraction.observability_screening import (
    registered_candidate_key as _registered_candidate_key,
)
from lidar2imu.extraction.observability_screening import (
    screen_motion_candidates_by_gril_observability as _screen_observability_candidates,
)
from lidar2imu.extraction.motion_windows import (
    build_motion_windows as _build_motion_windows,
    candidate_axis_abs_vector as _candidate_axis_abs_vector,
    candidate_overlaps_used_ranges as _candidate_overlaps_used_ranges,
    global_motion_selection_score as _global_motion_selection_score,
    heading_bin as _heading_bin,
    motion_turn_sign as _motion_turn_sign,
    scan_review_object_descriptor as _scan_review_object_descriptor,
    serialize_motion_review_candidate as _serialize_motion_review_candidate,
    submap_review_object_descriptor as _submap_review_object_descriptor,
    summarize_motion_candidate as _summarize_motion_candidate,
)
from lidar2imu.extraction.timing import (
    estimate_pose_time_offset_ns as _estimate_pose_time_offset_ns,
    nearest_sample as _nearest_sample,
    shift_timestamp_ns as _shift_timestamp_ns,
    uniform_indices as _uniform_indices,
    windowed_imu_gravity as _windowed_imu_gravity,
)
from lidar2imu.mapping.submaps import (
    LocalSubmapBuildConfig,
    build_local_lidar_submap as _build_local_lidar_submap,
    load_cached_cloud as _load_cached_cloud,
)
from lidar2imu.mapping.overlap import (
    overlap_quality_score as _overlap_quality_score,
    passes_overlap_gate as _passes_overlap_gate,
    point_cloud_overlap_metrics as _point_cloud_overlap_metrics,
)
from lidar2imu.io import load_dataset, write_outputs
from lidar2imu.models import CalibrationConfig
from lidar2imu.pipeline import run_calibration
from lidar2lidar.extrinsic_io import load_extrinsics_file
from lidar2lidar.lidar2lidar import calibrate_lidar_extrinsic
from lidar2lidar.prepared_dataset import (
    collect_record_bundle,
    load_prepared_rig_dataset,
)
from lidar2lidar.record_utils import (
    build_transform_graph,
    lookup_transform,
    prefetch_pointcloud_cache,
)

# isort: on

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RUN_PROFILE_PRESETS = {
    "baseline": {
        "gravity_source": "pose",
        "motion_frame_stride": 5,
        "motion_max_candidates_per_window": 3,
        "motion_observability_screening": "off",
        "min_registration_fitness": 0.55,
        "motion_registration_mode": "scan_to_scan",
        "solver_family": "baseline",
        "planar_motion_policy": "auto",
        "auto_reextract_if_needed": False,
    },
    "production": {
        "gravity_source": "pose",
        "motion_frame_stride": 5,
        "motion_max_candidates_per_window": 2,
        "max_motion_samples": 24,
        "motion_observability_screening": "gril_fisher",
        "motion_observability_turn_condition_relax_scale": 2.4,
        "motion_observability_straight_condition_strict_scale": 0.75,
        "motion_observability_turn_segment_min_turn_ratio": 0.30,
        "motion_observability_turn_segment_yaw_p95_deg": 8.0,
        "motion_observability_straight_segment_max_turn_ratio": 0.12,
        "motion_observability_straight_segment_yaw_p95_deg": 3.0,
        "motion_observability_auto_window_rescue_count": 4,
        "motion_observability_auto_window_rescue_min_relative_score": 0.55,
        "motion_observability_auto_window_rescue_condition_scale": 1.8,
        "min_registration_fitness": 0.50,
        "motion_registration_mode": "submap_to_map",
        "submap_builder_mode": "dense_scan_to_map_gicp",
        "submap_half_window": 2,
        "submap_support_stride": 5,
        "submap_min_support_frames": 3,
        "map_half_window": 6,
        "map_support_stride": 10,
        "map_min_support_frames": 5,
        "solver_family": "baseline",
        "planar_motion_policy": "auto",
        "auto_reextract_if_needed": True,
    },
}

RUN_PROFILE_OPTION_NAMES = {
    "gravity_source": ("--gravity-source",),
    "motion_frame_stride": ("--motion-frame-stride",),
    "motion_max_candidates_per_window": ("--motion-max-candidates-per-window",),
    "motion_observability_screening": ("--motion-observability-screening",),
    "motion_observability_window_sec": ("--motion-observability-window-sec",),
    "motion_observability_min_window_sec": ("--motion-observability-min-window-sec",),
    "motion_observability_min_samples": ("--motion-observability-min-samples",),
    "motion_observability_top_segments": ("--motion-observability-top-segments",),
    "motion_observability_min_rotation_lambda": (
        "--motion-observability-min-rotation-lambda",
    ),
    "motion_observability_min_planar_lambda": (
        "--motion-observability-min-planar-lambda",
    ),
    "motion_observability_max_condition_number": (
        "--motion-observability-max-condition-number",
    ),
    "motion_observability_turn_condition_relax_scale": (
        "--motion-observability-turn-condition-relax-scale",
    ),
    "motion_observability_straight_condition_strict_scale": (
        "--motion-observability-straight-condition-strict-scale",
    ),
    "motion_observability_turn_segment_min_turn_ratio": (
        "--motion-observability-turn-segment-min-turn-ratio",
    ),
    "motion_observability_turn_segment_yaw_p95_deg": (
        "--motion-observability-turn-segment-yaw-p95-deg",
    ),
    "motion_observability_straight_segment_max_turn_ratio": (
        "--motion-observability-straight-segment-max-turn-ratio",
    ),
    "motion_observability_straight_segment_yaw_p95_deg": (
        "--motion-observability-straight-segment-yaw-p95-deg",
    ),
    "motion_observability_rotation_sigma_rad": (
        "--motion-observability-rotation-sigma-rad",
    ),
    "motion_observability_translation_sigma_m": (
        "--motion-observability-translation-sigma-m",
    ),
    "motion_observability_max_merged_segments": (
        "--motion-observability-max-merged-segments",
    ),
    "motion_observability_auto_window_rescue_count": (
        "--motion-observability-auto-window-rescue-count",
    ),
    "motion_observability_auto_window_rescue_min_relative_score": (
        "--motion-observability-auto-window-rescue-min-relative-score",
    ),
    "motion_observability_auto_window_rescue_condition_scale": (
        "--motion-observability-auto-window-rescue-condition-scale",
    ),
    "max_motion_samples": ("--max-motion-samples",),
    "imu_preintegration_translation_weight": (
        "--imu-preintegration-translation-weight",
    ),
    "imu_preintegration_translation_scale_m": (
        "--imu-preintegration-translation-scale-m",
    ),
    "min_registration_fitness": ("--min-registration-fitness",),
    "motion_registration_mode": ("--motion-registration-mode",),
    "solver_family": ("--solver-family",),
    "submap_builder_mode": ("--submap-builder-mode",),
    "planar_motion_policy": ("--planar-motion-policy",),
    "submap_half_window": ("--submap-half-window",),
    "submap_support_stride": ("--submap-support-stride",),
    "submap_min_support_frames": ("--submap-min-support-frames",),
    "map_half_window": ("--map-half-window",),
    "map_support_stride": ("--map-support-stride",),
    "map_min_support_frames": ("--map-min-support-frames",),
    "auto_reextract_if_needed": ("--auto-reextract-if-needed",),
}

SUBMAP_REGISTERED_OVERLAP_MIN_WITHIN_0P4M_RATIO = 0.80
SUBMAP_REGISTERED_OVERLAP_MAX_NN_MEAN_M = 0.45


def _explicit_cli_options(argv: list[str]) -> set[str]:
    options = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        options.add(token.split("=", 1)[0])
    return options


def _apply_run_profile(args: argparse.Namespace, explicit_options: set[str]) -> None:
    if args.profile is None:
        return
    preset = RUN_PROFILE_PRESETS[args.profile]
    for field_name, value in preset.items():
        option_names = RUN_PROFILE_OPTION_NAMES.get(field_name, ())
        if any(option_name in explicit_options for option_name in option_names):
            continue
        setattr(args, field_name, value)


def _observability_segment_gain_by_id(
    observability_summary: dict[str, Any],
) -> dict[int, dict[str, float]]:
    information_capacity = observability_summary.get("information_capacity") or {}
    selection_trace = information_capacity.get("selection_trace") or []
    segment_gain_by_id: dict[int, dict[str, float]] = {}
    for item in selection_trace:
        if not isinstance(item, dict):
            continue
        try:
            segment_id = int(item["segment_id"])
        except (TypeError, ValueError, KeyError):
            continue
        segment_gain_by_id[segment_id] = {
            "min_eigenvalue_gain": float(item.get("min_eigenvalue_gain", 0.0)),
            "min_eigenvalue_gain_ratio": float(
                item.get("min_eigenvalue_gain_ratio", 0.0)
            ),
            "condition_worsening_ratio": float(
                item.get("condition_worsening_ratio", 1.0)
            ),
        }
    return segment_gain_by_id


def _observability_candidate_metadata(
    observability_summary: dict[str, Any],
) -> dict[tuple[int, int, int], dict[str, float | int]]:
    selected_segments = observability_summary.get("selected_segments") or []
    segment_gain_by_id = _observability_segment_gain_by_id(observability_summary)
    candidate_metadata: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for segment in selected_segments:
        if not isinstance(segment, dict):
            continue
        try:
            segment_id = int(segment["segment_id"])
        except (TypeError, ValueError, KeyError):
            continue
        combined_min_eigenvalue = float(segment.get("combined_min_eigenvalue", 0.0))
        combined_condition_number = float(
            segment.get("combined_condition_number", float("inf"))
        )
        segment_gain = segment_gain_by_id.get(segment_id, {})
        min_eigenvalue_gain = float(segment_gain.get("min_eigenvalue_gain", 0.0))
        min_eigenvalue_gain_ratio = float(
            segment_gain.get("min_eigenvalue_gain_ratio", 0.0)
        )
        condition_worsening_ratio = float(
            segment_gain.get("condition_worsening_ratio", 1.0)
        )
        capacity_weight = float(
            np.log1p(max(combined_min_eigenvalue, 0.0) * 10.0)
            / max(np.log1p(max(combined_condition_number, 1.0)), 1e-9)
        )
        capacity_weight *= float(1.0 + min(min_eigenvalue_gain_ratio, 2.0) * 0.5)
        for key_payload in segment.get("candidate_keys", []):
            if not isinstance(key_payload, dict):
                continue
            key = (
                int(key_payload["start_timestamp_ns"]),
                int(key_payload["end_timestamp_ns"]),
                int(key_payload["frame_stride"]),
            )
            metadata = {
                "observability_segment_id": int(segment_id),
                "observability_combined_min_eigenvalue": float(combined_min_eigenvalue),
                "observability_combined_condition_number": float(
                    combined_condition_number
                ),
                "observability_min_eigenvalue_gain": float(min_eigenvalue_gain),
                "observability_min_eigenvalue_gain_ratio": float(
                    min_eigenvalue_gain_ratio
                ),
                "observability_condition_worsening_ratio": float(
                    condition_worsening_ratio
                ),
                "observability_capacity_weight": float(max(capacity_weight, 1e-3)),
            }
            previous = candidate_metadata.get(key)
            if (
                previous is None
                or float(previous["observability_combined_min_eigenvalue"])
                < combined_min_eigenvalue
            ):
                candidate_metadata[key] = metadata
    return candidate_metadata


def _observability_candidate_decision_by_key(
    observability_summary: dict[str, Any],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    decisions = observability_summary.get("candidate_decisions") or []
    decision_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        try:
            key = (
                int(decision["start_timestamp_ns"]),
                int(decision["end_timestamp_ns"]),
                int(decision["frame_stride"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        decision_by_key[key] = decision
    return decision_by_key


def _parse_motion_candidate_key_specs(
    raw_specs: list[str] | None,
) -> set[tuple[int, int, int]]:
    key_specs: set[tuple[int, int, int]] = set()
    for raw_item in raw_specs or []:
        for token in str(raw_item).split(","):
            item = token.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 3:
                raise ValueError(
                    "Invalid --motion-observability-force-include-candidate-key "
                    f"value {item!r}; expected start_ns:end_ns:frame_stride."
                )
            try:
                start_timestamp_ns = int(parts[0])
                end_timestamp_ns = int(parts[1])
                frame_stride = int(parts[2])
            except ValueError as exc:
                raise ValueError(
                    "Invalid --motion-observability-force-include-candidate-key "
                    f"value {item!r}; expected integer start/end/stride."
                ) from exc
            key_specs.add((start_timestamp_ns, end_timestamp_ns, frame_stride))
    return key_specs


def _candidate_key_payload_from_tuple(
    candidate_key: tuple[int, int, int],
) -> dict[str, int]:
    start_timestamp_ns, end_timestamp_ns, frame_stride = candidate_key
    return {
        "start_timestamp_ns": int(start_timestamp_ns),
        "end_timestamp_ns": int(end_timestamp_ns),
        "frame_stride": int(frame_stride),
    }


def _screening_rejection_brief(
    decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(decision, dict):
        return None
    return {
        "classification": str(decision.get("rejection_classification", "")),
        "best_segment_id": (
            None
            if decision.get("best_segment_id") is None
            else int(decision["best_segment_id"])
        ),
        "best_segment_passes_rules": (
            None
            if decision.get("best_segment_passes_rules") is None
            else bool(decision["best_segment_passes_rules"])
        ),
        "best_segment_rule_failures": [
            str(reason)
            for reason in decision.get("best_segment_rule_failures", [])
            if isinstance(reason, str)
        ],
        "rule_failure_counts": {
            str(reason): int(count)
            for reason, count in (decision.get("rule_failure_counts") or {}).items()
        },
        "best_segment_rotation_min_eigenvalue": (
            None
            if decision.get("best_segment_rotation_min_eigenvalue") is None
            else float(decision["best_segment_rotation_min_eigenvalue"])
        ),
        "best_segment_planar_min_eigenvalue": (
            None
            if decision.get("best_segment_planar_min_eigenvalue") is None
            else float(decision["best_segment_planar_min_eigenvalue"])
        ),
        "best_segment_combined_condition_number": (
            None
            if decision.get("best_segment_combined_condition_number") is None
            else float(decision["best_segment_combined_condition_number"])
        ),
    }


def _support_indices_for_prefetch(
    anchor_index: int,
    *,
    meta_count: int,
    half_window: int,
    support_stride: int,
) -> set[int]:
    max_index = int(meta_count) - 1
    support_indices: set[int] = set()
    for offset in range(-int(half_window), int(half_window) + 1):
        support_index = int(anchor_index) + (offset * int(support_stride))
        if 0 <= support_index <= max_index:
            support_indices.add(int(support_index))
    return support_indices


def _estimate_imu_preintegration_delta_translation(
    imu_samples: list[Any],
    imu_timestamps: list[int],
    *,
    start_timestamp_ns: int,
    end_timestamp_ns: int,
    start_gravity_imu: np.ndarray,
    end_gravity_imu: np.ndarray,
) -> dict[str, Any] | None:
    if end_timestamp_ns <= start_timestamp_ns:
        return None
    if len(imu_samples) < 2 or len(imu_timestamps) < 2:
        return None
    start_index = int(
        np.searchsorted(imu_timestamps, int(start_timestamp_ns), side="left")
    )
    end_index = (
        int(np.searchsorted(imu_timestamps, int(end_timestamp_ns), side="right")) - 1
    )
    if start_index < 0:
        start_index = 0
    if end_index >= len(imu_samples):
        end_index = len(imu_samples) - 1
    if end_index <= start_index:
        return None
    selected_samples = imu_samples[start_index : end_index + 1]
    if len(selected_samples) < 2:
        return None

    duration_sec = max(float(end_timestamp_ns - start_timestamp_ns) / 1e9, 1e-6)
    gravity_start = np.asarray(start_gravity_imu, dtype=float).reshape(3)
    gravity_end = np.asarray(end_gravity_imu, dtype=float).reshape(3)
    rotation_start_to_current = np.eye(3, dtype=float)
    delta_velocity_start = np.zeros(3, dtype=float)
    delta_translation_start = np.zeros(3, dtype=float)
    valid_step_count = 0
    specific_accel_norms: list[float] = []

    previous_sample = selected_samples[0]
    for current_sample in selected_samples[1:]:
        dt_sec = (
            float(int(current_sample.timestamp_ns) - int(previous_sample.timestamp_ns))
            / 1e9
        )
        if dt_sec <= 1e-5 or dt_sec > 0.10:
            previous_sample = current_sample
            continue
        omega = 0.5 * (
            np.asarray(previous_sample.angular_velocity, dtype=float).reshape(3)
            + np.asarray(current_sample.angular_velocity, dtype=float).reshape(3)
        )
        rotation_start_to_current = (
            rotation_start_to_current @ R.from_rotvec(omega * dt_sec).as_matrix()
        )
        alpha = float(
            (int(current_sample.timestamp_ns) - int(start_timestamp_ns))
            / max(int(end_timestamp_ns) - int(start_timestamp_ns), 1)
        )
        gravity_current = (1.0 - alpha) * gravity_start + alpha * gravity_end
        specific_accel_current = (
            np.asarray(current_sample.linear_acceleration, dtype=float).reshape(3)
            - gravity_current
        )
        specific_accel_start = rotation_start_to_current.T @ specific_accel_current
        previous_delta_velocity = delta_velocity_start.copy()
        delta_velocity_start = previous_delta_velocity + specific_accel_start * dt_sec
        delta_translation_start = (
            delta_translation_start
            + previous_delta_velocity * dt_sec
            + 0.5 * specific_accel_start * (dt_sec * dt_sec)
        )
        specific_accel_norms.append(float(np.linalg.norm(specific_accel_start)))
        valid_step_count += 1
        previous_sample = current_sample

    if valid_step_count < 2:
        return None
    mean_specific_accel = (
        float(np.mean(specific_accel_norms)) if specific_accel_norms else 0.0
    )
    confidence = min(max(mean_specific_accel / 0.6, 0.1), 2.5)
    confidence *= min(max(duration_sec / 0.5, 0.3), 1.5)
    confidence *= min(float(valid_step_count) / 20.0, 1.5)
    return {
        "delta_translation_m": [
            float(value) for value in delta_translation_start.reshape(3).tolist()
        ],
        "delta_velocity_mps": [
            float(value) for value in delta_velocity_start.reshape(3).tolist()
        ],
        "sample_count": int(len(selected_samples)),
        "valid_step_count": int(valid_step_count),
        "duration_sec": float(duration_sec),
        "mean_specific_accel_mps2": float(mean_specific_accel),
        "confidence": float(max(confidence, 1e-3)),
    }


def convert_record_to_standardized_samples(
    record_path: str | None,
    output_dir: str,
    lidar_topic: str,
    pose_topic: str,
    imu_topic: str,
    parent_frame: str,
    child_frame: str | None,
    initial_transform_path: str | None,
    extraction_transform_path: str | None,
    identity_initial_transform: bool,
    gravity_source: str,
    ground_pose_sync_threshold_ms: float,
    motion_pose_sync_threshold_ms: float,
    imu_gravity_window_ms: float,
    max_ground_samples: int,
    max_motion_samples: int,
    motion_frame_stride: int,
    motion_max_candidates_per_window: int,
    plane_dist_thresh: float,
    plane_normal_thresh_deg: float,
    registration_voxel_size: float,
    min_registration_fitness: float,
    calibration_loss: str,
    calibration_motion_rotation_deg: float,
    calibration_solver_family: str,
    calibration_planar_motion_policy: str,
    calibration_imu_preintegration_translation_weight: float = 0.0,
    calibration_imu_preintegration_translation_scale_m: float = 0.08,
    motion_registration_mode: str = "scan_to_scan",
    submap_builder_mode: str = "pose_only",
    submap_half_window: int = 2,
    submap_support_stride: int | None = None,
    submap_min_support_frames: int = 3,
    submap_voxel_size: float | None = None,
    map_half_window: int | None = None,
    map_support_stride: int | None = None,
    map_min_support_frames: int | None = None,
    map_voxel_size: float | None = None,
    motion_observability_screening: str = "off",
    motion_observability_window_sec: float = 10.0,
    motion_observability_min_window_sec: float = 6.0,
    motion_observability_min_samples: int = 3,
    motion_observability_top_segments: int = 8,
    motion_observability_min_rotation_lambda: float = 1e-4,
    motion_observability_min_planar_lambda: float = 1e-3,
    motion_observability_max_condition_number: float = 2e3,
    motion_observability_turn_condition_relax_scale: float = 2.0,
    motion_observability_straight_condition_strict_scale: float = 0.8,
    motion_observability_turn_segment_min_turn_ratio: float = 0.35,
    motion_observability_turn_segment_yaw_p95_deg: float = 8.0,
    motion_observability_straight_segment_max_turn_ratio: float = 0.15,
    motion_observability_straight_segment_yaw_p95_deg: float = 3.0,
    motion_observability_rotation_sigma_rad: float = 0.02,
    motion_observability_translation_sigma_m: float = 0.05,
    motion_observability_max_merged_segments: int = 2,
    motion_observability_auto_window_rescue_count: int = 0,
    motion_observability_auto_window_rescue_min_relative_score: float = 0.65,
    motion_observability_auto_window_rescue_condition_scale: float = 1.5,
    motion_observability_force_include_candidate_keys: list[str] | None = None,
    motion_observability_force_include_outside_count: int = 0,
    run_profile: str | None = None,
    prepared_dataset_yaml: str | None = None,
    pose_time_offset_ms: float | None = None,
    estimate_pose_time_offset: bool = False,
    pose_time_offset_estimator: str = "nearest_median",
) -> tuple[Path, dict]:
    prepared_dataset = (
        load_prepared_rig_dataset(prepared_dataset_yaml)
        if prepared_dataset_yaml is not None
        else None
    )
    if prepared_dataset is not None:
        if lidar_topic not in prepared_dataset.metadata_by_topic:
            raise RuntimeError(
                "LiDAR topic "
                f"{lidar_topic} is not present in prepared dataset "
                f"{prepared_dataset_yaml}."
            )
        if prepared_dataset.pose_topic and pose_topic != prepared_dataset.pose_topic:
            raise RuntimeError(
                "Requested pose topic "
                f"{pose_topic} does not match prepared dataset topic "
                f"{prepared_dataset.pose_topic}."
            )
        if (
            gravity_source == "imu"
            and prepared_dataset.imu_topic
            and imu_topic != prepared_dataset.imu_topic
        ):
            raise RuntimeError(
                "Requested IMU topic "
                f"{imu_topic} does not match prepared dataset topic "
                f"{prepared_dataset.imu_topic}."
            )
        record_files = prepared_dataset.record_files
        lidar_frame = child_frame or prepared_dataset.topic_infos.get(
            lidar_topic, {}
        ).get("frame_id", "")
        if not lidar_frame and prepared_dataset.metadata_by_topic[lidar_topic]:
            lidar_frame = prepared_dataset.metadata_by_topic[lidar_topic][0].frame_id
        tf_edges = prepared_dataset.tf_edges
        metadata_by_topic = {
            lidar_topic: prepared_dataset.metadata_by_topic[lidar_topic]
        }
        localization_to_imu_source = str(
            prepared_dataset.manifest.get("summary", {}).get(
                "localization_to_imu_source", "prepared_dataset"
            )
        )
    else:
        if record_path is None:
            raise RuntimeError(
                "Either record_path or prepared_dataset_yaml must be provided."
            )
        record_bundle = collect_record_bundle(
            record_path=record_path,
            lidar_topics=[lidar_topic],
            pose_topic=pose_topic,
            imu_topic=imu_topic if gravity_source == "imu" else None,
            parent_frame=parent_frame,
        )
        record_files = record_bundle.record_files
        tf_edges = record_bundle.tf_edges
        metadata_by_topic = record_bundle.metadata_by_topic
        lidar_frame = child_frame or record_bundle.topic_frame_ids.get(lidar_topic, "")
        if not lidar_frame and metadata_by_topic[lidar_topic]:
            lidar_frame = metadata_by_topic[lidar_topic][0].frame_id
        if not lidar_frame:
            raise RuntimeError(
                f"Failed to infer frame_id for lidar topic {lidar_topic}."
            )
        localization_to_imu_source = record_bundle.localization_to_imu_source

    if not lidar_frame:
        raise RuntimeError(f"Failed to infer frame_id for lidar topic {lidar_topic}.")
    tf_graph = build_transform_graph(tf_edges)
    record_reference_transform = lookup_transform(tf_graph, lidar_frame, parent_frame)
    reference_transform_source = "/tf_static or merged tf graph"
    initial_transform_source = reference_transform_source

    def _load_override_transform(path: str, *, kind: str) -> tuple[np.ndarray, str]:
        transform, file_parent_frame, file_child_frame, _, _ = load_extrinsics_file(
            path
        )
        if file_parent_frame and file_parent_frame != parent_frame:
            raise RuntimeError(
                f"{kind} parent frame {file_parent_frame} does not match requested "
                f"parent frame {parent_frame}."
            )
        if file_child_frame and file_child_frame != lidar_frame:
            raise RuntimeError(
                f"{kind} child frame {file_child_frame} does not match LiDAR frame "
                f"{lidar_frame}."
            )
        return np.asarray(transform, dtype=float), str(Path(path).resolve())

    if initial_transform_path is not None:
        initial_transform, initial_transform_source = _load_override_transform(
            initial_transform_path,
            kind="Initial transform",
        )
    else:
        initial_transform = record_reference_transform
        if initial_transform is None:
            if not identity_initial_transform:
                raise RuntimeError(
                    f"Could not find transform from {lidar_frame} to {parent_frame}. "
                    "Provide --initial-transform or enable "
                    "--identity-initial-transform for exploratory runs."
                )
            initial_transform = np.eye(4, dtype=float)
            initial_transform_source = "identity_fallback"
            reference_transform_source = None
    if extraction_transform_path is not None:
        extraction_transform, extraction_transform_source = _load_override_transform(
            extraction_transform_path,
            kind="Extraction transform",
        )
    elif record_reference_transform is not None:
        extraction_transform = np.asarray(record_reference_transform, dtype=float)
        extraction_transform_source = reference_transform_source
    else:
        extraction_transform = np.asarray(initial_transform, dtype=float)
        extraction_transform_source = initial_transform_source

    lidar_metas = metadata_by_topic[lidar_topic]
    if not lidar_metas:
        raise RuntimeError(f"No point clouds found for topic {lidar_topic}.")

    pose_samples = (
        prepared_dataset.pose_samples
        if prepared_dataset is not None
        else record_bundle.pose_samples
    )
    if not pose_samples:
        raise RuntimeError(f"No pose messages found on {pose_topic}.")
    pose_timestamps = [sample.timestamp_ns for sample in pose_samples]

    imu_samples = (
        prepared_dataset.imu_samples
        if prepared_dataset is not None
        else record_bundle.imu_samples
    )
    imu_timestamps = [sample.timestamp_ns for sample in imu_samples]
    pose_time_offset_sync_threshold_ns = int(
        max(float(ground_pose_sync_threshold_ms), float(motion_pose_sync_threshold_ms))
        * 1e6
    )
    pose_time_offset_ns = 0
    pose_time_offset_source = "default_zero"
    pose_time_offset_diagnostics = {
        "enabled": False,
        "reason": "disabled",
        "sample_count": 0,
        "matched_count": 0,
        "estimated_offset_ns": 0,
        "estimated_offset_ms": 0.0,
    }
    if pose_time_offset_ms is not None:
        pose_time_offset_ns = int(round(float(pose_time_offset_ms) * 1e6))
        pose_time_offset_source = "explicit"
        pose_time_offset_diagnostics = {
            "enabled": True,
            "reason": None,
            "source": "explicit",
            "sample_count": 0,
            "matched_count": 0,
            "estimated_offset_ns": int(pose_time_offset_ns),
            "estimated_offset_ms": float(pose_time_offset_ns / 1e6),
        }
    elif estimate_pose_time_offset:
        pose_time_offset_ns, pose_time_offset_diagnostics = (
            _estimate_pose_time_offset_ns(
                lidar_metas,
                pose_samples,
                pose_timestamps,
                pose_time_offset_sync_threshold_ns,
                estimator=pose_time_offset_estimator,
            )
        )
        pose_time_offset_source = (
            "estimated"
            if pose_time_offset_diagnostics.get("reason") is None
            else "estimated_fallback_zero"
        )
        pose_time_offset_diagnostics["source"] = pose_time_offset_source

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_path = output_path / "standardized_samples.yaml"
    diagnostics_path = output_path / "conversion_diagnostics.yaml"
    cloud_cache: dict = {}

    ground_samples = []
    ground_diagnostics = []
    ground_indices = _uniform_indices(len(lidar_metas), max_ground_samples)
    ground_sync_threshold_ns = int(ground_pose_sync_threshold_ms * 1e6)
    imu_window_ns = int(imu_gravity_window_ms * 1e6)

    for meta_index in ground_indices:
        lidar_meta = lidar_metas[meta_index]
        adjusted_timestamp_ns = _shift_timestamp_ns(
            lidar_meta.timestamp_ns, pose_time_offset_ns
        )
        pose_sample, pose_dt_ns = _nearest_sample(
            pose_samples,
            pose_timestamps,
            adjusted_timestamp_ns,
            ground_sync_threshold_ns,
        )
        diagnostic = {
            "timestamp_ns": int(lidar_meta.timestamp_ns),
            "record_path": lidar_meta.record_path,
            "pose_sync_dt_ms": None if pose_dt_ns is None else float(pose_dt_ns / 1e6),
        }
        if pose_sample is None:
            diagnostic["reason"] = "missing_pose_sync"
            ground_diagnostics.append(diagnostic)
            continue

        if gravity_source == "imu":
            gravity_imu = _windowed_imu_gravity(
                imu_samples, imu_timestamps, adjusted_timestamp_ns, imu_window_ns
            )
            if gravity_imu is None:
                diagnostic["reason"] = "missing_imu_window"
                ground_diagnostics.append(diagnostic)
                continue
        else:
            gravity_imu = pose_sample.gravity_imu

        cloud = _load_cached_cloud(lidar_meta, cloud_cache)
        expected_up_imu = -normalize_vector(gravity_imu)
        expected_up_lidar = normalize_vector(
            extraction_transform[:3, :3].T @ expected_up_imu
        )
        plane_result, plane_diag = _extract_ground_plane(
            cloud,
            expected_up_lidar=expected_up_lidar,
            plane_dist_thresh=plane_dist_thresh,
            normal_angle_thresh_deg=plane_normal_thresh_deg,
        )
        diagnostic.update(plane_diag)
        if plane_result is None:
            ground_diagnostics.append(diagnostic)
            continue

        lidar_plane_normal = np.asarray(plane_result["lidar_plane_normal"], dtype=float)
        lidar_plane_offset = float(plane_result["lidar_plane_offset"])
        imu_plane_normal = extraction_transform[:3, :3] @ lidar_plane_normal
        imu_ground_height = float(
            lidar_plane_offset - imu_plane_normal @ extraction_transform[:3, 3]
        )
        weight = max(float(plane_result["inlier_ratio"]), 1e-3)

        ground_samples.append(
            {
                "timestamp_ns": int(lidar_meta.timestamp_ns),
                "lidar_plane_normal": plane_result["lidar_plane_normal"],
                "lidar_plane_offset": lidar_plane_offset,
                "imu_gravity": [
                    float(value)
                    for value in np.asarray(gravity_imu, dtype=float).tolist()
                ],
                "imu_ground_height": imu_ground_height,
                "weight": weight,
                "sync_dt_ms": float(pose_dt_ns / 1e6),
                "metadata": {
                    "record_path": lidar_meta.record_path,
                    "gravity_source": gravity_source,
                    "lidar_topic": lidar_topic,
                    "pose_topic": pose_topic,
                    "plane_inlier_ratio": float(plane_result["inlier_ratio"]),
                    "plane_normal_angle_to_expected_up_deg": float(
                        plane_result["normal_angle_to_expected_up_deg"]
                    ),
                },
            }
        )
        diagnostic["selected"] = True
        ground_diagnostics.append(diagnostic)

    motion_samples = []
    motion_diagnostics = []
    motion_window_diagnostics = []
    motion_rejected_low_fitness = 0
    motion_rejected_low_overlap = 0
    motion_registered_candidates: list[dict] = []
    motion_sync_threshold_ns = int(motion_pose_sync_threshold_ms * 1e6)
    submap_support_stride = int(
        motion_frame_stride if submap_support_stride is None else submap_support_stride
    )
    if motion_registration_mode not in (
        "scan_to_scan",
        "submap_to_submap",
        "submap_to_map",
    ):
        raise ValueError(
            "motion_registration_mode must be 'scan_to_scan', "
            "'submap_to_submap', or 'submap_to_map'."
        )
    if motion_observability_screening not in ("off", "gril_fisher"):
        raise ValueError(
            "motion_observability_screening must be 'off' or 'gril_fisher'."
        )
    if motion_observability_window_sec <= 0.0:
        raise ValueError("motion_observability_window_sec must be > 0.")
    if motion_observability_min_window_sec <= 0.0:
        raise ValueError("motion_observability_min_window_sec must be > 0.")
    if motion_observability_min_samples < 1:
        raise ValueError("motion_observability_min_samples must be >= 1.")
    if motion_observability_top_segments < 1:
        raise ValueError("motion_observability_top_segments must be >= 1.")
    if motion_observability_min_rotation_lambda < 0.0:
        raise ValueError("motion_observability_min_rotation_lambda must be >= 0.")
    if motion_observability_min_planar_lambda < 0.0:
        raise ValueError("motion_observability_min_planar_lambda must be >= 0.")
    if motion_observability_max_condition_number <= 0.0:
        raise ValueError("motion_observability_max_condition_number must be > 0.")
    if motion_observability_turn_condition_relax_scale < 1.0:
        raise ValueError(
            "motion_observability_turn_condition_relax_scale must be >= 1."
        )
    if not (0.0 < motion_observability_straight_condition_strict_scale <= 1.0):
        raise ValueError(
            "motion_observability_straight_condition_strict_scale must be in (0, 1]."
        )
    if not (0.0 <= motion_observability_turn_segment_min_turn_ratio <= 1.0):
        raise ValueError(
            "motion_observability_turn_segment_min_turn_ratio must be in [0, 1]."
        )
    if motion_observability_turn_segment_yaw_p95_deg < 0.0:
        raise ValueError("motion_observability_turn_segment_yaw_p95_deg must be >= 0.")
    if not (0.0 <= motion_observability_straight_segment_max_turn_ratio <= 1.0):
        raise ValueError(
            "motion_observability_straight_segment_max_turn_ratio must be in [0, 1]."
        )
    if motion_observability_straight_segment_yaw_p95_deg < 0.0:
        raise ValueError(
            "motion_observability_straight_segment_yaw_p95_deg must be >= 0."
        )
    if motion_observability_rotation_sigma_rad <= 0.0:
        raise ValueError("motion_observability_rotation_sigma_rad must be > 0.")
    if motion_observability_translation_sigma_m <= 0.0:
        raise ValueError("motion_observability_translation_sigma_m must be > 0.")
    if motion_observability_max_merged_segments < 1:
        raise ValueError("motion_observability_max_merged_segments must be >= 1.")
    if motion_observability_auto_window_rescue_count < 0:
        raise ValueError("motion_observability_auto_window_rescue_count must be >= 0.")
    if motion_observability_auto_window_rescue_min_relative_score < 0.0:
        raise ValueError(
            "motion_observability_auto_window_rescue_min_relative_score must be >= 0."
        )
    if motion_observability_auto_window_rescue_condition_scale < 1.0:
        raise ValueError(
            "motion_observability_auto_window_rescue_condition_scale must be >= 1."
        )
    if motion_max_candidates_per_window < 1:
        raise ValueError("motion_max_candidates_per_window must be >= 1.")
    if motion_observability_force_include_outside_count < 0:
        raise ValueError(
            "motion_observability_force_include_outside_count must be >= 0."
        )
    if calibration_imu_preintegration_translation_weight < 0.0:
        raise ValueError(
            "calibration_imu_preintegration_translation_weight must be >= 0."
        )
    if calibration_imu_preintegration_translation_scale_m <= 0.0:
        raise ValueError(
            "calibration_imu_preintegration_translation_scale_m must be > 0."
        )
    if submap_builder_mode not in ("pose_only", "dense_scan_to_map_gicp"):
        raise ValueError(
            "submap_builder_mode must be 'pose_only' or " "'dense_scan_to_map_gicp'."
        )
    if (
        motion_registration_mode in ("submap_to_submap", "submap_to_map")
        and submap_half_window < 1
    ):
        raise ValueError(
            "submap_half_window must be >= 1 for submap_to_submap/submap_to_map."
        )
    if submap_support_stride < 1:
        raise ValueError("submap_support_stride must be >= 1.")
    if submap_min_support_frames < 1:
        raise ValueError("submap_min_support_frames must be >= 1.")
    submap_voxel_size = float(
        registration_voxel_size if submap_voxel_size is None else submap_voxel_size
    )
    map_half_window = int(
        max(submap_half_window + 1, submap_half_window * 2)
        if map_half_window is None
        else map_half_window
    )
    map_support_stride = int(
        submap_support_stride if map_support_stride is None else map_support_stride
    )
    map_min_support_frames = int(
        max(submap_min_support_frames + 2, submap_min_support_frames)
        if map_min_support_frames is None
        else map_min_support_frames
    )
    map_voxel_size = float(
        submap_voxel_size if map_voxel_size is None else map_voxel_size
    )
    submap_build_config = LocalSubmapBuildConfig(
        half_window=int(submap_half_window),
        support_stride=int(submap_support_stride),
        min_support_frames=int(submap_min_support_frames),
        voxel_size=float(submap_voxel_size),
        builder_mode=str(submap_builder_mode),
    )
    map_build_config = LocalSubmapBuildConfig(
        half_window=int(map_half_window),
        support_stride=int(map_support_stride),
        min_support_frames=int(map_min_support_frames),
        voxel_size=float(map_voxel_size),
        builder_mode=str(submap_builder_mode),
    )
    if motion_registration_mode == "submap_to_map" and map_half_window < 1:
        raise ValueError("map_half_window must be >= 1 for submap_to_map.")
    if map_support_stride < 1:
        raise ValueError("map_support_stride must be >= 1.")
    if map_min_support_frames < 1:
        raise ValueError("map_min_support_frames must be >= 1.")
    motion_candidates = _build_motion_candidates(
        lidar_metas=lidar_metas,
        pose_samples=pose_samples,
        pose_timestamps=pose_timestamps,
        pose_time_offset_ns=pose_time_offset_ns,
        sync_threshold_ns=motion_sync_threshold_ns,
        base_stride=motion_frame_stride,
    )
    motion_windows = _build_motion_windows(
        motion_candidates=motion_candidates,
        lidar_meta_count=len(lidar_metas),
        max_motion_samples=max_motion_samples,
        base_stride=motion_frame_stride,
        min_window_rotation_deg=float(calibration_motion_rotation_deg),
        min_window_translation_m=0.5,
        max_candidates_per_window=motion_max_candidates_per_window,
    )
    prefetch_meta_indices: set[int] = {int(index) for index in ground_indices}
    for window in motion_windows:
        if not bool(window["valid"]):
            continue
        for candidate in window["candidates"]:
            start_index = int(candidate["start_index"])
            end_index = int(candidate["end_index"])
            prefetch_meta_indices.add(start_index)
            prefetch_meta_indices.add(end_index)
            if motion_registration_mode not in ("submap_to_submap", "submap_to_map"):
                continue
            prefetch_meta_indices.update(
                _support_indices_for_prefetch(
                    start_index,
                    meta_count=len(lidar_metas),
                    half_window=int(submap_build_config.half_window),
                    support_stride=int(submap_build_config.support_stride),
                )
            )
            target_build_config = (
                map_build_config
                if motion_registration_mode == "submap_to_map"
                else submap_build_config
            )
            prefetch_meta_indices.update(
                _support_indices_for_prefetch(
                    end_index,
                    meta_count=len(lidar_metas),
                    half_window=int(target_build_config.half_window),
                    support_stride=int(target_build_config.support_stride),
                )
            )
    prefetch_target_cloud_count = int(len(prefetch_meta_indices))
    prefetch_loaded_cloud_count = 0
    if prefetch_meta_indices:
        prefetch_metas = [lidar_metas[index] for index in sorted(prefetch_meta_indices)]
        prefetched_cloud_cache = prefetch_pointcloud_cache(prefetch_metas)
        cloud_cache.update(prefetched_cloud_cache)
        prefetch_loaded_cloud_count = int(len(prefetched_cloud_cache))

    preprocessing_params = {
        "voxel_size": registration_voxel_size,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": max(plane_dist_thresh, 0.1),
        "height_range": None,
        "remove_ground": False,
        "remove_walls": False,
    }
    alignment_cache: dict[int, dict] = {}
    submap_cache = {}

    for window in motion_windows:
        window_diagnostic = {
            "window_id": int(window["window_id"]),
            "window_start_index": int(window["window_start_index"]),
            "window_end_index": int(window["window_end_index"]),
            "candidate_count": int(window["candidate_count"]),
            "attempt_candidate_count": int(len(window["candidates"])),
            "best_pose_rotation_deg": float(window["best_pose_rotation_deg"]),
            "best_pose_translation_m": float(window["best_pose_translation_m"]),
            "stride_values": list(window.get("stride_values", [])),
            "valid": bool(window["valid"]),
            "gate_reasons": list(window["gate_reasons"]),
        }
        if not window["valid"]:
            motion_window_diagnostics.append(window_diagnostic)
            continue

        attempt_count = 0
        rejection_reasons = []
        registered_candidate_summaries = []
        for candidate in window["candidates"]:
            start_meta = candidate["start_meta"]
            end_meta = candidate["end_meta"]
            start_pose_dt_ns = candidate["start_pose_dt_ns"]
            end_pose_dt_ns = candidate["end_pose_dt_ns"]
            diagnostic = {
                "window_id": int(window["window_id"]),
                "start_timestamp_ns": int(start_meta.timestamp_ns),
                "end_timestamp_ns": int(end_meta.timestamp_ns),
                "start_pose_sync_dt_ms": (
                    None if start_pose_dt_ns is None else float(start_pose_dt_ns / 1e6)
                ),
                "end_pose_sync_dt_ms": (
                    None if end_pose_dt_ns is None else float(end_pose_dt_ns / 1e6)
                ),
                "frame_stride": int(candidate["stride"]),
                "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
                "pose_translation_m": float(candidate["pose_translation_m"]),
                "window_score": float(candidate["score"]),
                "information_score": float(candidate["information_score"]),
                "probabilistic_information_score": (
                    None
                    if candidate.get("probabilistic_information_score") is None
                    else float(candidate["probabilistic_information_score"])
                ),
                "probabilistic_window_score": (
                    None
                    if candidate.get("probabilistic_window_score") is None
                    else float(candidate["probabilistic_window_score"])
                ),
                "information_uncertainty_scale": (
                    None
                    if candidate.get("information_uncertainty_scale") is None
                    else float(candidate["information_uncertainty_scale"])
                ),
                "information_rotation_confidence": (
                    None
                    if candidate.get("information_rotation_confidence") is None
                    else float(candidate["information_rotation_confidence"])
                ),
                "information_translation_confidence": (
                    None
                    if candidate.get("information_translation_confidence") is None
                    else float(candidate["information_translation_confidence"])
                ),
                "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
                "imu_translation_heading_deg": (
                    None
                    if candidate.get("imu_translation_heading_deg") is None
                    else float(candidate["imu_translation_heading_deg"])
                ),
                "motion_registration_mode": motion_registration_mode,
                "registered": False,
                "passed_registration_gate": False,
                "selected": False,
            }
            attempt_count += 1
            imu_delta = candidate["imu_delta"]
            lidar_initial_guess = (
                np.linalg.inv(extraction_transform) @ imu_delta @ extraction_transform
            )
            if motion_registration_mode in ("submap_to_submap", "submap_to_map"):
                source_cloud, source_submap_info = _build_local_lidar_submap(
                    int(candidate["start_index"]),
                    lidar_metas=lidar_metas,
                    pose_samples=pose_samples,
                    pose_timestamps=pose_timestamps,
                    pose_time_offset_ns=pose_time_offset_ns,
                    sync_threshold_ns=motion_sync_threshold_ns,
                    extraction_transform=extraction_transform,
                    config=submap_build_config,
                    alignment_cache=alignment_cache,
                    cloud_cache=cloud_cache,
                    submap_cache=submap_cache,
                )
                if motion_registration_mode == "submap_to_map":
                    target_cloud, target_submap_info = _build_local_lidar_submap(
                        int(candidate["end_index"]),
                        lidar_metas=lidar_metas,
                        pose_samples=pose_samples,
                        pose_timestamps=pose_timestamps,
                        pose_time_offset_ns=pose_time_offset_ns,
                        sync_threshold_ns=motion_sync_threshold_ns,
                        extraction_transform=extraction_transform,
                        config=map_build_config,
                        alignment_cache=alignment_cache,
                        cloud_cache=cloud_cache,
                        submap_cache=submap_cache,
                    )
                else:
                    target_cloud, target_submap_info = _build_local_lidar_submap(
                        int(candidate["end_index"]),
                        lidar_metas=lidar_metas,
                        pose_samples=pose_samples,
                        pose_timestamps=pose_timestamps,
                        pose_time_offset_ns=pose_time_offset_ns,
                        sync_threshold_ns=motion_sync_threshold_ns,
                        extraction_transform=extraction_transform,
                        config=submap_build_config,
                        alignment_cache=alignment_cache,
                        cloud_cache=cloud_cache,
                        submap_cache=submap_cache,
                    )
                diagnostic["source_submap"] = source_submap_info
                if motion_registration_mode == "submap_to_map":
                    diagnostic["target_map"] = target_submap_info
                else:
                    diagnostic["target_submap"] = target_submap_info
                if source_cloud is None or target_cloud is None:
                    diagnostic["reason"] = (
                        "local_map_build_failed"
                        if motion_registration_mode == "submap_to_map"
                        else "submap_build_failed"
                    )
                    rejection_reasons.append(diagnostic["reason"])
                    motion_diagnostics.append(diagnostic)
                    continue
                source_registration_object = _submap_review_object_descriptor(
                    source_submap_info,
                    object_type="source_submap",
                )
                target_registration_object = _submap_review_object_descriptor(
                    target_submap_info,
                    object_type=(
                        "target_map"
                        if motion_registration_mode == "submap_to_map"
                        else "target_submap"
                    ),
                )
            else:
                source_cloud = _load_cached_cloud(start_meta, cloud_cache)
                target_cloud = _load_cached_cloud(end_meta, cloud_cache)
                source_registration_object = _scan_review_object_descriptor(
                    start_meta,
                    point_count=int(len(source_cloud.points)),
                )
                target_registration_object = _scan_review_object_descriptor(
                    end_meta,
                    point_count=int(len(target_cloud.points)),
                )
            lidar_delta, _, registration_result = calibrate_lidar_extrinsic(
                source_cloud,
                target_cloud,
                is_draw_registration=False,
                preprocessing_params=preprocessing_params,
                method=2,
                initial_transform=lidar_initial_guess,
            )
            if lidar_delta is None or registration_result is None:
                diagnostic["reason"] = "registration_failed"
                rejection_reasons.append("registration_failed")
                motion_diagnostics.append(diagnostic)
                continue

            registration_fitness = float(registration_result.fitness)
            registration_inlier_rmse = float(registration_result.inlier_rmse)
            diagnostic.update(
                {
                    "registered": True,
                    "registration_fitness": registration_fitness,
                    "registration_inlier_rmse": registration_inlier_rmse,
                }
            )
            if registration_fitness < min_registration_fitness:
                diagnostic["reason"] = "low_registration_fitness"
                motion_rejected_low_fitness += 1
                rejection_reasons.append("low_registration_fitness")
                motion_diagnostics.append(diagnostic)
                continue

            registered_overlap = None
            registered_overlap_quality_score = None
            if motion_registration_mode in ("submap_to_submap", "submap_to_map"):
                registered_source_cloud = o3d.geometry.PointCloud(source_cloud)
                registered_source_cloud.transform(lidar_delta)
                registered_overlap = _point_cloud_overlap_metrics(
                    registered_source_cloud, target_cloud
                )
                registered_overlap_quality_score = _overlap_quality_score(
                    registered_overlap
                )
                diagnostic["registered_overlap"] = dict(registered_overlap)
                diagnostic["registered_overlap_quality_score"] = float(
                    registered_overlap_quality_score
                )
                if not _passes_overlap_gate(
                    registered_overlap,
                    min_within_0p4m_ratio=(
                        SUBMAP_REGISTERED_OVERLAP_MIN_WITHIN_0P4M_RATIO
                    ),
                    max_nn_mean_m=SUBMAP_REGISTERED_OVERLAP_MAX_NN_MEAN_M,
                ):
                    diagnostic["reason"] = "low_registered_overlap"
                    motion_rejected_low_overlap += 1
                    rejection_reasons.append("low_registered_overlap")
                    motion_diagnostics.append(diagnostic)
                    continue

            diagnostic["passed_registration_gate"] = True
            weight = max(registration_fitness, 1e-3)
            imu_quat = R.from_matrix(imu_delta[:3, :3]).as_quat()
            lidar_quat = R.from_matrix(lidar_delta[:3, :3]).as_quat()
            registered_candidate = {
                **candidate,
                "lidar_delta": lidar_delta,
                "registration_fitness": registration_fitness,
                "registration_inlier_rmse": registration_inlier_rmse,
                "weight": weight,
                "sync_dt_ms": float(max(start_pose_dt_ns, end_pose_dt_ns) / 1e6),
                "imu_quat": imu_quat,
                "lidar_quat": lidar_quat,
                "motion_registration_mode": motion_registration_mode,
                "source_registration_object": source_registration_object,
                "target_registration_object": target_registration_object,
                "registered_overlap_quality_score": registered_overlap_quality_score,
                "registered_overlap_within_0p4m_ratio": (
                    None
                    if registered_overlap is None
                    else float(registered_overlap["within_0p4m_ratio"])
                ),
                "registered_overlap_nn_mean_m": (
                    None
                    if registered_overlap is None
                    else float(registered_overlap["nn_mean_m"])
                ),
            }
            imu_preintegration = _estimate_imu_preintegration_delta_translation(
                imu_samples,
                imu_timestamps,
                start_timestamp_ns=int(start_meta.timestamp_ns),
                end_timestamp_ns=int(end_meta.timestamp_ns),
                start_gravity_imu=np.asarray(
                    candidate["start_pose"].gravity_imu, dtype=float
                ),
                end_gravity_imu=np.asarray(
                    candidate["end_pose"].gravity_imu, dtype=float
                ),
            )
            if imu_preintegration is None:
                duration_sec = max(
                    float(int(end_meta.timestamp_ns) - int(start_meta.timestamp_ns))
                    / 1e9,
                    1e-6,
                )
                fallback_delta = np.asarray(imu_delta[:3, 3], dtype=float).reshape(3)
                imu_preintegration = {
                    "delta_translation_m": [
                        float(value) for value in fallback_delta.tolist()
                    ],
                    "delta_velocity_mps": [
                        float(value)
                        for value in (fallback_delta / duration_sec).tolist()
                    ],
                    "sample_count": 0,
                    "valid_step_count": 0,
                    "duration_sec": float(duration_sec),
                    "mean_specific_accel_mps2": 0.0,
                    "confidence": 0.2,
                    "source": "imu_delta_fallback",
                }
            preintegration_source = str(
                imu_preintegration.get("source") or "imu_preintegration"
            )
            registered_candidate.update(
                {
                    "imu_preintegration_delta_translation_m": np.asarray(
                        imu_preintegration["delta_translation_m"], dtype=float
                    ),
                    "imu_preintegration_delta_velocity_mps": np.asarray(
                        imu_preintegration["delta_velocity_mps"], dtype=float
                    ),
                    "imu_preintegration_confidence": float(
                        imu_preintegration["confidence"]
                    ),
                    "imu_preintegration_sample_count": int(
                        imu_preintegration["sample_count"]
                    ),
                    "imu_preintegration_valid_step_count": int(
                        imu_preintegration["valid_step_count"]
                    ),
                    "imu_preintegration_duration_sec": float(
                        imu_preintegration["duration_sec"]
                    ),
                    "imu_preintegration_mean_specific_accel_mps2": float(
                        imu_preintegration["mean_specific_accel_mps2"]
                    ),
                    "imu_preintegration_source": preintegration_source,
                }
            )
            diagnostic["imu_preintegration_confidence"] = float(
                imu_preintegration["confidence"]
            )
            diagnostic["imu_preintegration_sample_count"] = int(
                imu_preintegration["sample_count"]
            )
            diagnostic["imu_preintegration_source"] = preintegration_source
            info_components = _motion_information_components(
                registered_candidate,
                base_stride=motion_frame_stride,
            )
            registered_candidate.update(
                {
                    "probabilistic_information_score": float(
                        info_components["probabilistic_information_score"]
                    ),
                    "probabilistic_window_score": float(
                        info_components["probabilistic_window_score"]
                    ),
                    "information_uncertainty_scale": float(
                        info_components["uncertainty_scale"]
                    ),
                    "information_rotation_confidence": float(
                        info_components["rotation_confidence"]
                    ),
                    "information_translation_confidence": float(
                        info_components["translation_confidence"]
                    ),
                }
            )
            diagnostic["probabilistic_information_score"] = float(
                info_components["probabilistic_information_score"]
            )
            diagnostic["probabilistic_window_score"] = float(
                info_components["probabilistic_window_score"]
            )
            motion_registered_candidates.append(registered_candidate)
            registered_candidate_summaries.append(
                {
                    **_summarize_motion_candidate(registered_candidate),
                    "registration_fitness": registration_fitness,
                    "registration_inlier_rmse": registration_inlier_rmse,
                }
            )
            motion_diagnostics.append(diagnostic)

        window_diagnostic["attempt_count"] = int(attempt_count)
        window_diagnostic["registered_candidate_count"] = int(
            len(registered_candidate_summaries)
        )
        if registered_candidate_summaries:
            window_diagnostic["registered_candidates"] = registered_candidate_summaries
            window_diagnostic["selected"] = False
        else:
            window_diagnostic["selected"] = False
            if rejection_reasons:
                window_diagnostic["reason"] = "window_no_candidate_passing_gate"
                window_diagnostic["candidate_rejection_reasons"] = rejection_reasons
            else:
                window_diagnostic["reason"] = "window_not_selected"
        motion_window_diagnostics.append(window_diagnostic)

    motion_window_diagnostics_by_id = {
        int(item["window_id"]): item for item in motion_window_diagnostics
    }
    observability_screening_summary = _screen_observability_candidates(
        motion_registered_candidates,
        mode=motion_observability_screening,
        target_window_sec=float(motion_observability_window_sec),
        min_window_sec=float(motion_observability_min_window_sec),
        min_samples=int(motion_observability_min_samples),
        top_segments=int(motion_observability_top_segments),
        min_rotation_lambda=float(motion_observability_min_rotation_lambda),
        min_planar_lambda=float(motion_observability_min_planar_lambda),
        max_combined_condition_number=float(motion_observability_max_condition_number),
        turn_condition_relax_scale=float(
            motion_observability_turn_condition_relax_scale
        ),
        straight_condition_strict_scale=float(
            motion_observability_straight_condition_strict_scale
        ),
        turn_segment_min_turn_ratio=float(
            motion_observability_turn_segment_min_turn_ratio
        ),
        turn_segment_yaw_p95_deg=float(motion_observability_turn_segment_yaw_p95_deg),
        straight_segment_max_turn_ratio=float(
            motion_observability_straight_segment_max_turn_ratio
        ),
        straight_segment_yaw_p95_deg=float(
            motion_observability_straight_segment_yaw_p95_deg
        ),
        rotation_sigma_rad=float(motion_observability_rotation_sigma_rad),
        translation_sigma_m=float(motion_observability_translation_sigma_m),
        max_merged_segments=int(motion_observability_max_merged_segments),
        max_selected_candidates=int(max(max_motion_samples * 3, max_motion_samples)),
        auto_window_rescue_count=int(motion_observability_auto_window_rescue_count),
        auto_window_rescue_min_relative_score=float(
            motion_observability_auto_window_rescue_min_relative_score
        ),
        auto_window_rescue_condition_scale=float(
            motion_observability_auto_window_rescue_condition_scale
        ),
    )
    observability_candidate_metadata = _observability_candidate_metadata(
        observability_screening_summary
    )
    observability_candidate_decisions_by_key = _observability_candidate_decision_by_key(
        observability_screening_summary
    )
    for candidate in motion_registered_candidates:
        candidate_key = _registered_candidate_key(candidate)
        if candidate_key not in observability_candidate_metadata:
            continue
        candidate.update(
            {
                **candidate,
                **observability_candidate_metadata[candidate_key],
            }
        )
    selection_candidate_pool = list(motion_registered_candidates)
    registered_candidates_by_key = {
        _registered_candidate_key(candidate): candidate
        for candidate in motion_registered_candidates
    }
    observability_selected_keys = {
        (
            int(item["start_timestamp_ns"]),
            int(item["end_timestamp_ns"]),
            int(item["frame_stride"]),
        )
        for item in observability_screening_summary.get("selected_candidate_keys", [])
    }
    original_observability_selected_keys = set(observability_selected_keys)
    observability_selected_window_ids = {
        int(window_id)
        for window_id in observability_screening_summary.get("selected_window_ids", [])
    }
    forced_include_requested_keys = _parse_motion_candidate_key_specs(
        motion_observability_force_include_candidate_keys
    )
    forced_include_missing_keys: list[tuple[int, int, int]] = []
    forced_include_explicit_applied_keys: list[tuple[int, int, int]] = []
    forced_include_auto_applied_keys: list[tuple[int, int, int]] = []
    if observability_screening_summary.get("applied", False):
        forced_include_missing_keys = sorted(
            forced_include_requested_keys - set(registered_candidates_by_key)
        )
        forced_include_explicit_applied_keys = sorted(
            forced_include_requested_keys & set(registered_candidates_by_key)
        )
        for key in forced_include_explicit_applied_keys:
            observability_selected_keys.add(key)
        if motion_observability_force_include_outside_count > 0:
            outside_candidates = [
                candidate
                for candidate in motion_registered_candidates
                if _registered_candidate_key(candidate)
                not in observability_selected_keys
            ]
            outside_candidates.sort(
                key=lambda candidate: (
                    float(candidate.get("registered_overlap_quality_score") or 0.0),
                    float(candidate.get("registration_fitness") or 0.0),
                    float(candidate.get("probabilistic_information_score") or 0.0),
                    float(candidate.get("information_score") or 0.0),
                ),
                reverse=True,
            )
            for candidate in outside_candidates:
                if (
                    len(forced_include_auto_applied_keys)
                    >= motion_observability_force_include_outside_count
                ):
                    break
                key = _registered_candidate_key(candidate)
                if key in observability_selected_keys:
                    continue
                observability_selected_keys.add(key)
                forced_include_auto_applied_keys.append(key)
    elif (
        forced_include_requested_keys
        or motion_observability_force_include_outside_count > 0
    ):
        logging.warning(
            "Ignoring observability force-include options because screening is not "
            "applied in this run."
        )
    forced_include_applied_keys = sorted(
        set(forced_include_explicit_applied_keys)
        | set(forced_include_auto_applied_keys)
    )
    forced_include_applied_key_set = set(forced_include_applied_keys)
    if forced_include_applied_keys:
        observability_selected_window_ids.update(
            {
                int(registered_candidates_by_key[key]["window_id"])
                for key in forced_include_applied_keys
                if key in registered_candidates_by_key
                and registered_candidates_by_key[key].get("window_id") is not None
            }
        )
    for key in forced_include_applied_keys:
        if key in observability_candidate_decisions_by_key:
            observability_candidate_decisions_by_key[key][
                "force_included_for_ablation"
            ] = True
    observability_screening_summary["force_include"] = {
        "requested_candidate_keys": [
            _candidate_key_payload_from_tuple(key)
            for key in sorted(forced_include_requested_keys)
        ],
        "requested_outside_count": int(
            motion_observability_force_include_outside_count
        ),
        "missing_candidate_keys": [
            _candidate_key_payload_from_tuple(key)
            for key in forced_include_missing_keys
        ],
        "applied_explicit_candidate_keys": [
            _candidate_key_payload_from_tuple(key)
            for key in forced_include_explicit_applied_keys
        ],
        "applied_auto_candidate_keys": [
            _candidate_key_payload_from_tuple(key)
            for key in forced_include_auto_applied_keys
        ],
        "applied_candidate_keys": [
            _candidate_key_payload_from_tuple(key)
            for key in forced_include_applied_keys
        ],
        "applied_count": int(len(forced_include_applied_keys)),
        "applied_window_ids": sorted(
            int(window_id) for window_id in observability_selected_window_ids
        ),
    }
    if observability_screening_summary.get("applied", False):
        rejection_class_counter = Counter()
        failure_reason_counter = Counter()
        for key, decision in observability_candidate_decisions_by_key.items():
            if key in original_observability_selected_keys:
                continue
            rejection_class = str(decision.get("rejection_classification") or "unknown")
            rejection_class_counter[rejection_class] += 1
            for reason, count in (decision.get("rule_failure_counts") or {}).items():
                if not isinstance(reason, str):
                    continue
                try:
                    failure_reason_counter[reason] += int(count)
                except (TypeError, ValueError):
                    continue
        logging.info(
            "Observability screening selected %d/%d registered candidates "
            "(forced include %d).",
            len(observability_selected_keys),
            len(motion_registered_candidates),
            len(forced_include_applied_keys),
        )
        if rejection_class_counter:
            logging.info(
                "Observability rejection classes: %s",
                dict(rejection_class_counter),
            )
        if failure_reason_counter:
            logging.info(
                "Observability rule-failure counts: %s",
                dict(failure_reason_counter),
            )
    if observability_screening_summary.get("applied", False):
        selection_candidate_pool = [
            candidate
            for candidate in motion_registered_candidates
            if _registered_candidate_key(candidate) in observability_selected_keys
        ]
    if observability_screening_summary.get("applied", False):
        for diagnostic in motion_diagnostics:
            if not diagnostic.get("passed_registration_gate"):
                continue
            candidate_key = (
                int(diagnostic["start_timestamp_ns"]),
                int(diagnostic["end_timestamp_ns"]),
                int(diagnostic["frame_stride"]),
            )
            in_selected_segment = candidate_key in original_observability_selected_keys
            force_included = candidate_key in forced_include_applied_key_set
            diagnostic["passed_observability_screening"] = bool(in_selected_segment)
            diagnostic["outside_observability_screening_original"] = bool(
                not in_selected_segment
            )
            diagnostic["force_included_observability_outside"] = bool(force_included)
            rejection_brief = _screening_rejection_brief(
                observability_candidate_decisions_by_key.get(candidate_key)
            )
            if rejection_brief is not None:
                diagnostic["observability_screening_rejection"] = rejection_brief
            if (
                not in_selected_segment
                and not force_included
                and diagnostic.get("reason") is None
            ):
                diagnostic["reason"] = "outside_observability_screening_segment"
        window_decisions: dict[
            int, list[tuple[tuple[int, int, int], dict[str, Any]]]
        ] = {}
        for candidate_key, decision in observability_candidate_decisions_by_key.items():
            window_id_raw = decision.get("window_id")
            if window_id_raw is None:
                continue
            try:
                window_id = int(window_id_raw)
            except (TypeError, ValueError):
                continue
            window_decisions.setdefault(window_id, []).append((candidate_key, decision))
        for window_diagnostic in motion_window_diagnostics:
            window_id = int(window_diagnostic["window_id"])
            in_selected_segment = window_id in observability_selected_window_ids
            window_diagnostic["in_observability_segment"] = bool(in_selected_segment)
            decisions_in_window = window_decisions.get(window_id, [])
            if decisions_in_window:
                rejection_class_counts = Counter()
                rule_failure_counts = Counter()
                forced_include_count = 0
                for candidate_key, decision in decisions_in_window:
                    if candidate_key in forced_include_applied_key_set:
                        forced_include_count += 1
                    if candidate_key in original_observability_selected_keys:
                        continue
                    rejection_class = str(
                        decision.get("rejection_classification") or "unknown"
                    )
                    rejection_class_counts[rejection_class] += 1
                    for reason, count in (
                        decision.get("rule_failure_counts") or {}
                    ).items():
                        if not isinstance(reason, str):
                            continue
                        try:
                            rule_failure_counts[reason] += int(count)
                        except (TypeError, ValueError):
                            continue
                window_diagnostic["observability_screening_summary"] = {
                    "registered_candidate_count": int(len(decisions_in_window)),
                    "outside_original_segment_count": int(
                        sum(
                            1
                            for candidate_key, _ in decisions_in_window
                            if candidate_key not in original_observability_selected_keys
                        )
                    ),
                    "force_included_candidate_count": int(forced_include_count),
                    "rejection_class_counts": dict(rejection_class_counts),
                    "rule_failure_counts": dict(rule_failure_counts),
                }
            if in_selected_segment:
                continue
            if (
                window_diagnostic.get("reason") is None
                and int(
                    window_diagnostic.get("observability_screening_summary", {}).get(
                        "force_included_candidate_count", 0
                    )
                )
                == 0
                and int(window_diagnostic.get("registered_candidate_count", 0)) > 0
            ):
                window_diagnostic["reason"] = "outside_observability_screening_segment"

    motion_selection_strategy = "global_diversity"
    if observability_screening_summary.get("applied", False):
        motion_selection_strategy = "gril_fisher_segments_then_global_diversity"
        if forced_include_applied_keys:
            motion_selection_strategy += "_with_forced_outside_candidates"
    elif motion_observability_screening != "off":
        motion_selection_strategy = "global_diversity_with_gril_fisher_diagnostics"

    selected_registered_candidates = []
    used_motion_ranges: list[tuple[int, int]] = []
    for window in motion_windows:
        if len(selected_registered_candidates) >= max_motion_samples:
            break
        window_candidates = [
            candidate
            for candidate in selection_candidate_pool
            if int(candidate["window_id"]) == int(window["window_id"])
        ]
        if not window_candidates:
            continue

        best_candidate = None
        best_selection_score = None
        overlap_rejected = 0
        for candidate in window_candidates:
            if _candidate_overlaps_used_ranges(candidate, used_motion_ranges):
                overlap_rejected += 1
                continue
            selection_score = _global_motion_selection_score(
                candidate, selected_registered_candidates, motion_frame_stride
            )
            if best_candidate is None or selection_score > best_selection_score:
                best_candidate = candidate
                best_selection_score = selection_score

        window_diagnostic = motion_window_diagnostics_by_id[int(window["window_id"])]
        window_diagnostic["global_overlap_rejected"] = int(overlap_rejected)
        if best_candidate is None:
            window_diagnostic["reason"] = "window_candidates_overlap_selected_ranges"
            continue

        best_candidate["global_selection_score"] = float(best_selection_score)
        selected_registered_candidates.append(best_candidate)
        used_motion_ranges.append(
            (int(best_candidate["start_index"]), int(best_candidate["end_index"]))
        )
        window_diagnostic["selected"] = True
        if "reason" in window_diagnostic:
            window_diagnostic.pop("reason", None)
        window_diagnostic.pop("candidate_rejection_reasons", None)
        window_diagnostic["selected_candidate"] = {
            **_summarize_motion_candidate(best_candidate),
            "registration_fitness": float(best_candidate["registration_fitness"]),
            "registration_inlier_rmse": float(
                best_candidate["registration_inlier_rmse"]
            ),
            "global_selection_score": float(best_selection_score),
        }

    selected_candidate_keys = {
        (
            int(candidate["start_meta"].timestamp_ns),
            int(candidate["end_meta"].timestamp_ns),
            int(candidate["stride"]),
        )
        for candidate in selected_registered_candidates
    }
    selected_candidates_by_window_id = {
        int(candidate["window_id"]): candidate
        for candidate in selected_registered_candidates
    }
    for diagnostic in motion_diagnostics:
        candidate_key = (
            int(diagnostic["start_timestamp_ns"]),
            int(diagnostic["end_timestamp_ns"]),
            int(diagnostic["frame_stride"]),
        )
        if candidate_key in selected_candidate_keys:
            diagnostic["selected"] = True
        if candidate_key in forced_include_applied_key_set:
            diagnostic["selected_via_forced_observability_include"] = True

    review_motion_candidates = []
    for window in motion_windows:
        window_candidates = [
            candidate
            for candidate in motion_registered_candidates
            if int(candidate["window_id"]) == int(window["window_id"])
        ]
        if not window_candidates:
            continue
        review_candidate = selected_candidates_by_window_id.get(
            int(window["window_id"])
        )
        if review_candidate is None:
            review_candidate = max(
                window_candidates,
                key=lambda item: (
                    float(item.get("registered_overlap_quality_score") or 0.0),
                    float(item["registration_fitness"]),
                    float(item["information_score"]),
                    float(item["score"]),
                ),
            )
        review_candidate_key = (
            int(review_candidate["start_meta"].timestamp_ns),
            int(review_candidate["end_meta"].timestamp_ns),
            int(review_candidate["stride"]),
        )
        review_payload = _serialize_motion_review_candidate(
            review_candidate,
            lidar_topic=lidar_topic,
            pose_topic=pose_topic,
            selected_for_calibration=review_candidate_key in selected_candidate_keys,
        )
        motion_window_diagnostics_by_id[int(window["window_id"])][
            "review_candidate"
        ] = {
            "start_timestamp_ns": int(review_payload["start_timestamp_ns"]),
            "end_timestamp_ns": int(review_payload["end_timestamp_ns"]),
            "frame_stride": int(review_payload["frame_stride"]),
            "registration_fitness": float(review_payload["registration_fitness"]),
            "registration_inlier_rmse": float(
                review_payload["registration_inlier_rmse"]
            ),
            "registered_overlap_quality_score": (
                None
                if review_payload.get("registered_overlap_quality_score") is None
                else float(review_payload["registered_overlap_quality_score"])
            ),
            "registered_overlap_within_0p4m_ratio": (
                None
                if review_payload.get("registered_overlap_within_0p4m_ratio") is None
                else float(review_payload["registered_overlap_within_0p4m_ratio"])
            ),
            "registered_overlap_nn_mean_m": (
                None
                if review_payload.get("registered_overlap_nn_mean_m") is None
                else float(review_payload["registered_overlap_nn_mean_m"])
            ),
            "selected_for_calibration": bool(
                review_payload["selected_for_calibration"]
            ),
        }
        review_motion_candidates.append(review_payload)
    review_motion_candidates.sort(
        key=lambda item: (
            int(item["start_timestamp_ns"]),
            int(item["end_timestamp_ns"]),
        )
    )

    for candidate in sorted(
        selected_registered_candidates,
        key=lambda item: int(item["start_meta"].timestamp_ns),
    ):
        imu_delta = candidate["imu_delta"]
        lidar_delta = candidate["lidar_delta"]
        motion_samples.append(
            {
                "start_timestamp_ns": int(candidate["start_meta"].timestamp_ns),
                "end_timestamp_ns": int(candidate["end_meta"].timestamp_ns),
                "imu_delta": {
                    "translation": {
                        "x": float(imu_delta[0, 3]),
                        "y": float(imu_delta[1, 3]),
                        "z": float(imu_delta[2, 3]),
                    },
                    "rotation": {
                        "x": float(candidate["imu_quat"][0]),
                        "y": float(candidate["imu_quat"][1]),
                        "z": float(candidate["imu_quat"][2]),
                        "w": float(candidate["imu_quat"][3]),
                    },
                },
                "lidar_delta": {
                    "translation": {
                        "x": float(lidar_delta[0, 3]),
                        "y": float(lidar_delta[1, 3]),
                        "z": float(lidar_delta[2, 3]),
                    },
                    "rotation": {
                        "x": float(candidate["lidar_quat"][0]),
                        "y": float(candidate["lidar_quat"][1]),
                        "z": float(candidate["lidar_quat"][2]),
                        "w": float(candidate["lidar_quat"][3]),
                    },
                },
                "weight": float(candidate["weight"]),
                "sync_dt_ms": float(candidate["sync_dt_ms"]),
                "metadata": {
                    "record_path_start": candidate["start_meta"].record_path,
                    "record_path_end": candidate["end_meta"].record_path,
                    "lidar_topic": lidar_topic,
                    "pose_topic": pose_topic,
                    "window_id": int(candidate["window_id"]),
                    "motion_registration_mode": motion_registration_mode,
                    "frame_stride": int(candidate["stride"]),
                    "pose_rotation_deg": float(candidate["pose_rotation_deg"]),
                    "pose_translation_m": float(candidate["pose_translation_m"]),
                    "information_score": float(candidate["information_score"]),
                    "imu_signed_yaw_deg": float(candidate["imu_signed_yaw_deg"]),
                    "imu_translation_heading_deg": (
                        None
                        if candidate.get("imu_translation_heading_deg") is None
                        else float(candidate["imu_translation_heading_deg"])
                    ),
                    "registration_fitness": float(candidate["registration_fitness"]),
                    "registration_inlier_rmse": float(
                        candidate["registration_inlier_rmse"]
                    ),
                    "registered_overlap_quality_score": (
                        None
                        if candidate.get("registered_overlap_quality_score") is None
                        else float(candidate["registered_overlap_quality_score"])
                    ),
                    "registered_overlap_within_0p4m_ratio": (
                        None
                        if candidate.get("registered_overlap_within_0p4m_ratio") is None
                        else float(candidate["registered_overlap_within_0p4m_ratio"])
                    ),
                    "registered_overlap_nn_mean_m": (
                        None
                        if candidate.get("registered_overlap_nn_mean_m") is None
                        else float(candidate["registered_overlap_nn_mean_m"])
                    ),
                    "observability_segment_id": (
                        None
                        if candidate.get("observability_segment_id") is None
                        else int(candidate["observability_segment_id"])
                    ),
                    "observability_combined_min_eigenvalue": (
                        None
                        if candidate.get("observability_combined_min_eigenvalue")
                        is None
                        else float(candidate["observability_combined_min_eigenvalue"])
                    ),
                    "observability_combined_condition_number": (
                        None
                        if candidate.get("observability_combined_condition_number")
                        is None
                        else float(candidate["observability_combined_condition_number"])
                    ),
                    "observability_min_eigenvalue_gain": (
                        None
                        if candidate.get("observability_min_eigenvalue_gain") is None
                        else float(candidate["observability_min_eigenvalue_gain"])
                    ),
                    "observability_min_eigenvalue_gain_ratio": (
                        None
                        if candidate.get("observability_min_eigenvalue_gain_ratio")
                        is None
                        else float(candidate["observability_min_eigenvalue_gain_ratio"])
                    ),
                    "observability_condition_worsening_ratio": (
                        None
                        if candidate.get("observability_condition_worsening_ratio")
                        is None
                        else float(candidate["observability_condition_worsening_ratio"])
                    ),
                    "observability_capacity_weight": (
                        None
                        if candidate.get("observability_capacity_weight") is None
                        else float(candidate["observability_capacity_weight"])
                    ),
                    "imu_preintegration_delta_translation_m": (
                        None
                        if candidate.get("imu_preintegration_delta_translation_m")
                        is None
                        else [
                            float(value)
                            for value in np.asarray(
                                candidate["imu_preintegration_delta_translation_m"],
                                dtype=float,
                            ).reshape(3)
                        ]
                    ),
                    "imu_preintegration_delta_velocity_mps": (
                        None
                        if candidate.get("imu_preintegration_delta_velocity_mps")
                        is None
                        else [
                            float(value)
                            for value in np.asarray(
                                candidate["imu_preintegration_delta_velocity_mps"],
                                dtype=float,
                            ).reshape(3)
                        ]
                    ),
                    "imu_preintegration_confidence": (
                        None
                        if candidate.get("imu_preintegration_confidence") is None
                        else float(candidate["imu_preintegration_confidence"])
                    ),
                    "imu_preintegration_sample_count": (
                        None
                        if candidate.get("imu_preintegration_sample_count") is None
                        else int(candidate["imu_preintegration_sample_count"])
                    ),
                    "imu_preintegration_valid_step_count": (
                        None
                        if candidate.get("imu_preintegration_valid_step_count") is None
                        else int(candidate["imu_preintegration_valid_step_count"])
                    ),
                    "imu_preintegration_duration_sec": (
                        None
                        if candidate.get("imu_preintegration_duration_sec") is None
                        else float(candidate["imu_preintegration_duration_sec"])
                    ),
                    "imu_preintegration_mean_specific_accel_mps2": (
                        None
                        if candidate.get("imu_preintegration_mean_specific_accel_mps2")
                        is None
                        else float(
                            candidate["imu_preintegration_mean_specific_accel_mps2"]
                        )
                    ),
                    "imu_preintegration_source": (
                        None
                        if candidate.get("imu_preintegration_source") is None
                        else str(candidate["imu_preintegration_source"])
                    ),
                    "global_selection_score": float(
                        candidate["global_selection_score"]
                    ),
                },
            }
        )

    selected_motion_headings = [
        float(sample["metadata"]["imu_translation_heading_deg"])
        for sample in motion_samples
        if sample["metadata"].get("imu_translation_heading_deg") is not None
    ]
    selected_turn_counts = {"left": 0, "right": 0}
    selected_heading_bins = set()
    selected_axis_vectors = []
    for candidate in selected_registered_candidates:
        turn_sign = _motion_turn_sign(float(candidate["imu_signed_yaw_deg"]))
        if turn_sign in selected_turn_counts:
            selected_turn_counts[turn_sign] += 1
        heading_bin = _heading_bin(candidate.get("imu_translation_heading_deg"))
        if heading_bin is not None:
            selected_heading_bins.add(int(heading_bin))
        selected_axis_vectors.append(_candidate_axis_abs_vector(candidate))
    selected_axis_mean = (
        np.mean(np.asarray(selected_axis_vectors, dtype=float), axis=0)
        if selected_axis_vectors
        else None
    )

    payload = {
        "parent_frame": parent_frame,
        "child_frame": lidar_frame,
        "initial_transform": {
            "header": {
                "stamp": {"secs": 0, "nsecs": 0},
                "seq": 0,
                "frame_id": parent_frame,
            },
            "transform": {
                "translation": {
                    "x": float(initial_transform[0, 3]),
                    "y": float(initial_transform[1, 3]),
                    "z": float(initial_transform[2, 3]),
                },
                "rotation": {
                    **{
                        key: float(value)
                        for key, value in zip(
                            ("x", "y", "z", "w"),
                            R.from_matrix(initial_transform[:3, :3]).as_quat(),
                        )
                    }
                },
            },
            "child_frame_id": lidar_frame,
        },
        "config": {
            "loss": calibration_loss,
            "min_motion_rotation_deg": float(calibration_motion_rotation_deg),
            "solver_family": calibration_solver_family,
            "planar_motion_policy": calibration_planar_motion_policy,
            "imu_preintegration_translation_weight": float(
                calibration_imu_preintegration_translation_weight
            ),
            "imu_preintegration_translation_scale_m": float(
                calibration_imu_preintegration_translation_scale_m
            ),
        },
        "metadata": {
            "run_profile": run_profile or "custom",
            "solver_family": calibration_solver_family,
            "record_path": record_path,
            "record_files": record_files,
            "lidar_topic": lidar_topic,
            "pose_topic": pose_topic,
            "imu_topic": imu_topic,
            "gravity_source": gravity_source,
            "ground_pose_sync_threshold_ms": float(ground_pose_sync_threshold_ms),
            "motion_pose_sync_threshold_ms": float(motion_pose_sync_threshold_ms),
            "pose_time_offset_ms": float(pose_time_offset_ns / 1e6),
            "pose_time_offset_source": pose_time_offset_source,
            "pose_time_offset_estimator": pose_time_offset_estimator,
            "pose_time_offset_diagnostics": pose_time_offset_diagnostics,
            "motion_frame_stride": int(motion_frame_stride),
            "motion_selection_strategy": motion_selection_strategy,
            "motion_observability_screening": str(motion_observability_screening),
            "motion_observability_target_window_sec": float(
                motion_observability_window_sec
            ),
            "motion_observability_min_window_sec": float(
                motion_observability_min_window_sec
            ),
            "motion_observability_min_samples": int(motion_observability_min_samples),
            "motion_observability_min_rotation_lambda": float(
                motion_observability_min_rotation_lambda
            ),
            "motion_observability_min_planar_lambda": float(
                motion_observability_min_planar_lambda
            ),
            "motion_observability_max_condition_number": float(
                motion_observability_max_condition_number
            ),
            "motion_observability_turn_condition_relax_scale": float(
                motion_observability_turn_condition_relax_scale
            ),
            "motion_observability_straight_condition_strict_scale": float(
                motion_observability_straight_condition_strict_scale
            ),
            "motion_observability_turn_segment_min_turn_ratio": float(
                motion_observability_turn_segment_min_turn_ratio
            ),
            "motion_observability_turn_segment_yaw_p95_deg": float(
                motion_observability_turn_segment_yaw_p95_deg
            ),
            "motion_observability_straight_segment_max_turn_ratio": float(
                motion_observability_straight_segment_max_turn_ratio
            ),
            "motion_observability_straight_segment_yaw_p95_deg": float(
                motion_observability_straight_segment_yaw_p95_deg
            ),
            "motion_observability_rotation_sigma_rad": float(
                motion_observability_rotation_sigma_rad
            ),
            "motion_observability_translation_sigma_m": float(
                motion_observability_translation_sigma_m
            ),
            "motion_observability_max_merged_segments": int(
                motion_observability_max_merged_segments
            ),
            "motion_observability_auto_window_rescue_count": int(
                motion_observability_auto_window_rescue_count
            ),
            "motion_observability_auto_window_rescue_min_relative_score": float(
                motion_observability_auto_window_rescue_min_relative_score
            ),
            "motion_observability_auto_window_rescue_condition_scale": float(
                motion_observability_auto_window_rescue_condition_scale
            ),
            "motion_observability_force_include_candidate_keys": [
                _candidate_key_payload_from_tuple(key)
                for key in sorted(forced_include_requested_keys)
            ],
            "motion_observability_force_include_outside_count": int(
                motion_observability_force_include_outside_count
            ),
            "motion_observability_force_included_count": int(
                len(forced_include_applied_keys)
            ),
            "motion_observability_applied": bool(
                observability_screening_summary.get("applied", False)
            ),
            "calibration_imu_preintegration_translation_weight": float(
                calibration_imu_preintegration_translation_weight
            ),
            "calibration_imu_preintegration_translation_scale_m": float(
                calibration_imu_preintegration_translation_scale_m
            ),
            "motion_registration_mode": motion_registration_mode,
            "submap_builder_mode": submap_builder_mode,
            "submap_half_window": int(submap_half_window),
            "submap_support_stride": int(submap_support_stride),
            "submap_min_support_frames": int(submap_min_support_frames),
            "submap_voxel_size": float(submap_voxel_size),
            "timestamp_policy": {
                "pointcloud": (
                    "measurement_time -> header.timestamp_sec -> record_timestamp"
                ),
                "pose": (
                    "measurement_time -> header.timestamp_sec -> record_timestamp"
                ),
                "imu": ("measurement_time -> header.timestamp_sec -> record_timestamp"),
                "gnss_best_pose_heading": (
                    "measurement_time(GPS) -> unix conversion -> "
                    "header.timestamp_sec"
                ),
            },
            "map_half_window": int(map_half_window),
            "map_support_stride": int(map_support_stride),
            "map_min_support_frames": int(map_min_support_frames),
            "map_voxel_size": float(map_voxel_size),
            "min_registration_fitness": float(min_registration_fitness),
            "min_registered_overlap_within_0p4m_ratio": (
                SUBMAP_REGISTERED_OVERLAP_MIN_WITHIN_0P4M_RATIO
                if motion_registration_mode in ("submap_to_submap", "submap_to_map")
                else None
            ),
            "max_registered_overlap_nn_mean_m": (
                SUBMAP_REGISTERED_OVERLAP_MAX_NN_MEAN_M
                if motion_registration_mode in ("submap_to_submap", "submap_to_map")
                else None
            ),
            "initial_transform_source": initial_transform_source,
            "extraction_transform_source": extraction_transform_source,
            "reference_transform_source": reference_transform_source,
            "localization_to_imu_source": localization_to_imu_source,
            "ground_selected": len(ground_samples),
            "motion_selected": len(motion_samples),
            "motion_candidate_count": len(motion_candidates),
            "motion_window_count": len(motion_windows),
            "motion_valid_window_count": int(
                sum(1 for window in motion_windows if window["valid"])
            ),
            "motion_max_candidates_per_window": int(motion_max_candidates_per_window),
            "cloud_prefetch_target_count": int(prefetch_target_cloud_count),
            "cloud_prefetch_loaded_count": int(prefetch_loaded_cloud_count),
            "motion_registered_candidate_count": len(motion_registered_candidates),
            "motion_registration_attempt_count": int(
                sum(
                    int(window.get("attempt_count", 0) or 0)
                    for window in motion_window_diagnostics
                )
            ),
            "motion_rejected_low_overlap": int(motion_rejected_low_overlap),
            "review_motion_candidate_count": len(review_motion_candidates),
            "observability_segment_count": int(
                observability_screening_summary.get("segment_count", 0)
            ),
        },
        "ground_samples": ground_samples,
        "motion_samples": motion_samples,
        "review_motion_candidates": review_motion_candidates,
    }
    payload["extraction_transform"] = {
        "header": {
            "stamp": {"secs": 0, "nsecs": 0},
            "seq": 0,
            "frame_id": parent_frame,
        },
        "transform": {
            "translation": {
                "x": float(extraction_transform[0, 3]),
                "y": float(extraction_transform[1, 3]),
                "z": float(extraction_transform[2, 3]),
            },
            "rotation": {
                **{
                    key: float(value)
                    for key, value in zip(
                        ("x", "y", "z", "w"),
                        R.from_matrix(extraction_transform[:3, :3]).as_quat(),
                    )
                }
            },
        },
        "child_frame_id": lidar_frame,
    }
    if record_reference_transform is not None:
        payload["reference_transform"] = {
            "header": {
                "stamp": {"secs": 0, "nsecs": 0},
                "seq": 0,
                "frame_id": parent_frame,
            },
            "transform": {
                "translation": {
                    "x": float(record_reference_transform[0, 3]),
                    "y": float(record_reference_transform[1, 3]),
                    "z": float(record_reference_transform[2, 3]),
                },
                "rotation": {
                    **{
                        key: float(value)
                        for key, value in zip(
                            ("x", "y", "z", "w"),
                            R.from_matrix(record_reference_transform[:3, :3]).as_quat(),
                        )
                    }
                },
            },
            "child_frame_id": lidar_frame,
        }
    diagnostics = {
        "summary": {
            "run_profile": run_profile or "custom",
            "record_files": record_files,
            "lidar_topic": lidar_topic,
            "lidar_frame": lidar_frame,
            "parent_frame": parent_frame,
            "localization_to_imu_source": localization_to_imu_source,
            "pose_time_offset_ms": float(pose_time_offset_ns / 1e6),
            "pose_time_offset_source": pose_time_offset_source,
            "pose_time_offset_estimator": pose_time_offset_estimator,
            "ground_selected": len(ground_samples),
            "motion_selected": len(motion_samples),
            "ground_attempted": len(ground_indices),
            "motion_attempted": len(motion_candidates),
            "motion_rejected_low_fitness": motion_rejected_low_fitness,
            "motion_rejected_low_overlap": int(motion_rejected_low_overlap),
            "motion_registered_candidate_count": len(motion_registered_candidates),
            "motion_registration_attempt_count": int(
                sum(
                    int(window.get("attempt_count", 0) or 0)
                    for window in motion_window_diagnostics
                )
            ),
            "motion_window_count": len(motion_windows),
            "motion_valid_window_count": int(
                sum(1 for window in motion_windows if window["valid"])
            ),
            "motion_max_candidates_per_window": int(motion_max_candidates_per_window),
            "cloud_prefetch_target_count": int(prefetch_target_cloud_count),
            "cloud_prefetch_loaded_count": int(prefetch_loaded_cloud_count),
            "review_motion_candidate_count": len(review_motion_candidates),
            "submap_builder_mode": submap_builder_mode,
            "min_registration_fitness": float(min_registration_fitness),
            "min_registered_overlap_within_0p4m_ratio": (
                SUBMAP_REGISTERED_OVERLAP_MIN_WITHIN_0P4M_RATIO
                if motion_registration_mode in ("submap_to_submap", "submap_to_map")
                else None
            ),
            "max_registered_overlap_nn_mean_m": (
                SUBMAP_REGISTERED_OVERLAP_MAX_NN_MEAN_M
                if motion_registration_mode in ("submap_to_submap", "submap_to_map")
                else None
            ),
            "motion_selection_strategy": motion_selection_strategy,
            "motion_registration_mode": motion_registration_mode,
            "map_half_window": int(map_half_window),
            "map_support_stride": int(map_support_stride),
            "map_min_support_frames": int(map_min_support_frames),
            "map_voxel_size": float(map_voxel_size),
            "timestamp_basis": "canonical_sensor_time",
            "motion_observability_screening": str(motion_observability_screening),
            "motion_observability_target_window_sec": float(
                motion_observability_window_sec
            ),
            "motion_observability_min_window_sec": float(
                motion_observability_min_window_sec
            ),
            "motion_observability_min_samples": int(motion_observability_min_samples),
            "motion_observability_min_rotation_lambda": float(
                motion_observability_min_rotation_lambda
            ),
            "motion_observability_min_planar_lambda": float(
                motion_observability_min_planar_lambda
            ),
            "motion_observability_max_condition_number": float(
                motion_observability_max_condition_number
            ),
            "motion_observability_turn_condition_relax_scale": float(
                motion_observability_turn_condition_relax_scale
            ),
            "motion_observability_straight_condition_strict_scale": float(
                motion_observability_straight_condition_strict_scale
            ),
            "motion_observability_turn_segment_min_turn_ratio": float(
                motion_observability_turn_segment_min_turn_ratio
            ),
            "motion_observability_turn_segment_yaw_p95_deg": float(
                motion_observability_turn_segment_yaw_p95_deg
            ),
            "motion_observability_straight_segment_max_turn_ratio": float(
                motion_observability_straight_segment_max_turn_ratio
            ),
            "motion_observability_straight_segment_yaw_p95_deg": float(
                motion_observability_straight_segment_yaw_p95_deg
            ),
            "motion_observability_rotation_sigma_rad": float(
                motion_observability_rotation_sigma_rad
            ),
            "motion_observability_translation_sigma_m": float(
                motion_observability_translation_sigma_m
            ),
            "motion_observability_max_merged_segments": int(
                motion_observability_max_merged_segments
            ),
            "motion_observability_auto_window_rescue_count": int(
                motion_observability_auto_window_rescue_count
            ),
            "motion_observability_auto_window_rescue_min_relative_score": float(
                motion_observability_auto_window_rescue_min_relative_score
            ),
            "motion_observability_auto_window_rescue_condition_scale": float(
                motion_observability_auto_window_rescue_condition_scale
            ),
            "motion_observability_force_include_candidate_keys": [
                _candidate_key_payload_from_tuple(key)
                for key in sorted(forced_include_requested_keys)
            ],
            "motion_observability_force_include_outside_count": int(
                motion_observability_force_include_outside_count
            ),
            "motion_observability_force_included_count": int(
                len(forced_include_applied_keys)
            ),
            "motion_observability_applied": bool(
                observability_screening_summary.get("applied", False)
            ),
            "calibration_imu_preintegration_translation_weight": float(
                calibration_imu_preintegration_translation_weight
            ),
            "calibration_imu_preintegration_translation_scale_m": float(
                calibration_imu_preintegration_translation_scale_m
            ),
        },
        "ground": ground_diagnostics,
        "motion_windows": motion_window_diagnostics,
        "motion": motion_diagnostics,
        "motion_selection": {
            "strategy": motion_selection_strategy,
            "registration_mode": motion_registration_mode,
            "selected_frame_strides": sorted(
                {int(sample["metadata"]["frame_stride"]) for sample in motion_samples}
            ),
            "selected_turn_counts": {
                "left": int(selected_turn_counts["left"]),
                "right": int(selected_turn_counts["right"]),
            },
            "selected_heading_bin_count": int(len(selected_heading_bins)),
            "selected_translation_heading_span_deg": circular_span_deg(
                selected_motion_headings
            ),
            "selected_rotation_axis_abs_mean_xyz": (
                {
                    "x": float(selected_axis_mean[0]),
                    "y": float(selected_axis_mean[1]),
                    "z": float(selected_axis_mean[2]),
                }
                if selected_axis_mean is not None
                else None
            ),
            "registered_candidate_count": len(motion_registered_candidates),
            "review_candidate_count": len(review_motion_candidates),
            "observability_screening": observability_screening_summary,
        },
    }
    with open(sample_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    with open(diagnostics_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(diagnostics, file, sort_keys=False)
    return sample_path, diagnostics


def _recommendation_rank(recommendation: str | None) -> int:
    ranking = {
        "full_6dof_candidate": 5,
        "reference_conflict_review": 4,
        "holdout_review": 3,
        "z_roll_pitch_priority": 2,
        "basin_sensitivity_review": 1,
        "reextract_review": 0,
        "insufficient_data": 0,
    }
    return ranking.get(recommendation or "", -1)


def _status_rank(status: str | None) -> int:
    ranking = {"pass": 2, "warning": 1}
    return ranking.get(status or "", 0)


def _delta_rotation_deg(delta: dict | None) -> float:
    if not isinstance(delta, dict):
        return float("inf")
    return float(delta.get("rotation_deg", float("inf")))


def _delta_translation_m(delta: dict | None) -> float:
    if not isinstance(delta, dict):
        return float("inf")
    return float(delta.get("translation_norm_m", float("inf")))


def _build_reextract_pass_summary(
    pass_name: str,
    pass_dir: Path,
    sample_path: Path,
    diagnostics: dict,
    result: dict,
    manifest: dict,
) -> dict:
    metrics = result["metrics"]
    coarse_metrics = metrics.get("coarse_metrics", {})
    assessment = metrics.get("vehicle_motion_assessment", {})
    summary = metrics.get("summary", {})
    return {
        "pass_name": pass_name,
        "pass_dir": str(pass_dir),
        "sample_path": str(sample_path),
        "calibration_dir": str(pass_dir / "calibration"),
        "calibrated_extrinsics": manifest["artifacts"]["calibrated_extrinsics"],
        "recommendation": assessment.get("recommendation"),
        "solver_policy": summary.get("solver_policy"),
        "trusted_reference_consistency": assessment.get(
            "trusted_reference_consistency"
        ),
        "extraction_consistency": assessment.get("extraction_consistency"),
        "planar_basin_stability": assessment.get("planar_basin_stability"),
        "full_prior_robustness": assessment.get("full_prior_robustness"),
        "delta_to_reference": summary.get("delta_to_reference"),
        "delta_to_extraction": summary.get("delta_to_extraction"),
        "motion_rotation_residual_p95_deg": coarse_metrics.get(
            "motion_rotation_residual_p95_deg"
        ),
        "motion_translation_residual_p95_m": coarse_metrics.get(
            "motion_translation_residual_p95_m"
        ),
        "motion_registration_fitness_p05": coarse_metrics.get(
            "motion_registration_fitness_p05"
        ),
        "ground_selected": diagnostics["summary"]["ground_selected"],
        "motion_selected": diagnostics["summary"]["motion_selected"],
    }


def _rewrite_pass_summary_paths(pass_summary: dict, pass_dir: Path) -> None:
    pass_summary["pass_dir"] = str(pass_dir)
    pass_summary["sample_path"] = str(pass_dir / "standardized_samples.yaml")
    pass_summary["calibration_dir"] = str(pass_dir / "calibration")
    calibrated_extrinsics = Path(pass_summary["calibrated_extrinsics"])
    pass_summary["calibrated_extrinsics"] = str(
        pass_dir / "calibration" / "calibrated" / calibrated_extrinsics.name
    )


def _reextract_trigger_reasons(pass_summary: dict) -> list[str]:
    reasons: list[str] = []
    if pass_summary.get("recommendation") == "reextract_review":
        reasons.append("recommendation=reextract_review")
    if pass_summary.get("extraction_consistency") == "warning":
        reasons.append("extraction_consistency=warning")
    return reasons


def _pass_selection_score(pass_summary: dict) -> tuple[float, ...]:
    return (
        float(_recommendation_rank(pass_summary.get("recommendation"))),
        float(_status_rank(pass_summary.get("trusted_reference_consistency"))),
        float(_status_rank(pass_summary.get("extraction_consistency"))),
        float(_status_rank(pass_summary.get("planar_basin_stability"))),
        float(_status_rank(pass_summary.get("full_prior_robustness"))),
        -_delta_rotation_deg(pass_summary.get("delta_to_reference")),
        -_delta_translation_m(pass_summary.get("delta_to_reference")),
        -_delta_rotation_deg(pass_summary.get("delta_to_extraction")),
        -_delta_translation_m(pass_summary.get("delta_to_extraction")),
        -float(pass_summary.get("motion_rotation_residual_p95_deg", float("inf"))),
        -float(pass_summary.get("motion_translation_residual_p95_m", float("inf"))),
        float(pass_summary.get("motion_registration_fitness_p05", 0.0)),
    )


def _select_reextract_pass(pass_summaries: list[dict]) -> dict:
    return max(pass_summaries, key=_pass_selection_score)


def _copy_output_artifacts(source_dir: Path, target_dir: Path) -> None:
    for name in (
        "standardized_samples.yaml",
        "conversion_diagnostics.yaml",
        "calibration",
    ):
        source_path = source_dir / name
        target_path = target_dir / name
        if not source_path.exists():
            continue
        if target_path.is_dir():
            shutil.rmtree(target_path)
        elif target_path.exists():
            target_path.unlink()
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _refresh_calibration_manifest_paths(calibration_dir: Path) -> None:
    manifest_path = calibration_dir / "diagnostics" / "manifest.yaml"
    if not manifest_path.exists():
        return
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file) or {}
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["calibrated_tf"] = str(calibration_dir / "calibrated_tf.yaml")
    artifacts["metrics"] = str(calibration_dir / "metrics.yaml")
    parent_frame = manifest.get("parent_frame")
    child_frame = manifest.get("child_frame")
    if parent_frame and child_frame:
        filename = f"{parent_frame}_{child_frame}_extrinsics.yaml"
        artifacts["initial_guess"] = str(calibration_dir / "initial_guess" / filename)
        artifacts["calibrated_extrinsics"] = str(
            calibration_dir / "calibrated" / filename
        )
    diagnostics = artifacts.setdefault("diagnostics", {})
    diagnostics["algorithm"] = str(calibration_dir / "diagnostics" / "algorithm.yaml")
    diagnostics["evaluation"] = str(calibration_dir / "diagnostics" / "evaluation.yaml")
    diagnostics["observability"] = str(
        calibration_dir / "diagnostics" / "observability.yaml"
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)


def _run_conversion_and_calibration(
    args: argparse.Namespace,
    output_dir: Path,
    initial_transform_path: str | None,
    extraction_transform_path: str | None,
    identity_initial_transform: bool,
) -> dict:
    sample_path, diagnostics = convert_record_to_standardized_samples(
        record_path=args.record_path,
        output_dir=str(output_dir),
        lidar_topic=args.lidar_topic,
        pose_topic=args.pose_topic,
        imu_topic=args.imu_topic,
        parent_frame=args.parent_frame,
        child_frame=args.child_frame,
        initial_transform_path=initial_transform_path,
        extraction_transform_path=extraction_transform_path,
        identity_initial_transform=identity_initial_transform,
        gravity_source=args.gravity_source,
        ground_pose_sync_threshold_ms=args.ground_pose_sync_threshold_ms,
        motion_pose_sync_threshold_ms=args.motion_pose_sync_threshold_ms,
        pose_time_offset_ms=args.pose_time_offset_ms,
        estimate_pose_time_offset=args.estimate_pose_time_offset,
        pose_time_offset_estimator=args.pose_time_offset_estimator,
        imu_gravity_window_ms=args.imu_gravity_window_ms,
        max_ground_samples=args.max_ground_samples,
        max_motion_samples=args.max_motion_samples,
        motion_frame_stride=args.motion_frame_stride,
        motion_max_candidates_per_window=args.motion_max_candidates_per_window,
        plane_dist_thresh=args.plane_dist_thresh,
        plane_normal_thresh_deg=args.plane_normal_thresh_deg,
        registration_voxel_size=args.registration_voxel_size,
        min_registration_fitness=args.min_registration_fitness,
        calibration_loss=args.loss,
        calibration_motion_rotation_deg=args.min_motion_rotation_deg,
        calibration_solver_family=args.solver_family,
        calibration_planar_motion_policy=args.planar_motion_policy,
        calibration_imu_preintegration_translation_weight=(
            args.imu_preintegration_translation_weight
        ),
        calibration_imu_preintegration_translation_scale_m=(
            args.imu_preintegration_translation_scale_m
        ),
        motion_registration_mode=args.motion_registration_mode,
        submap_builder_mode=args.submap_builder_mode,
        submap_half_window=args.submap_half_window,
        submap_support_stride=args.submap_support_stride,
        submap_min_support_frames=args.submap_min_support_frames,
        submap_voxel_size=args.submap_voxel_size,
        map_half_window=args.map_half_window,
        map_support_stride=args.map_support_stride,
        map_min_support_frames=args.map_min_support_frames,
        map_voxel_size=args.map_voxel_size,
        motion_observability_screening=args.motion_observability_screening,
        motion_observability_window_sec=args.motion_observability_window_sec,
        motion_observability_min_window_sec=(args.motion_observability_min_window_sec),
        motion_observability_min_samples=args.motion_observability_min_samples,
        motion_observability_top_segments=args.motion_observability_top_segments,
        motion_observability_min_rotation_lambda=(
            args.motion_observability_min_rotation_lambda
        ),
        motion_observability_min_planar_lambda=(
            args.motion_observability_min_planar_lambda
        ),
        motion_observability_max_condition_number=(
            args.motion_observability_max_condition_number
        ),
        motion_observability_turn_condition_relax_scale=(
            args.motion_observability_turn_condition_relax_scale
        ),
        motion_observability_straight_condition_strict_scale=(
            args.motion_observability_straight_condition_strict_scale
        ),
        motion_observability_turn_segment_min_turn_ratio=(
            args.motion_observability_turn_segment_min_turn_ratio
        ),
        motion_observability_turn_segment_yaw_p95_deg=(
            args.motion_observability_turn_segment_yaw_p95_deg
        ),
        motion_observability_straight_segment_max_turn_ratio=(
            args.motion_observability_straight_segment_max_turn_ratio
        ),
        motion_observability_straight_segment_yaw_p95_deg=(
            args.motion_observability_straight_segment_yaw_p95_deg
        ),
        motion_observability_rotation_sigma_rad=(
            args.motion_observability_rotation_sigma_rad
        ),
        motion_observability_translation_sigma_m=(
            args.motion_observability_translation_sigma_m
        ),
        motion_observability_max_merged_segments=(
            args.motion_observability_max_merged_segments
        ),
        motion_observability_auto_window_rescue_count=(
            args.motion_observability_auto_window_rescue_count
        ),
        motion_observability_auto_window_rescue_min_relative_score=(
            args.motion_observability_auto_window_rescue_min_relative_score
        ),
        motion_observability_auto_window_rescue_condition_scale=(
            args.motion_observability_auto_window_rescue_condition_scale
        ),
        motion_observability_force_include_candidate_keys=(
            args.motion_observability_force_include_candidate_key
        ),
        motion_observability_force_include_outside_count=(
            args.motion_observability_force_include_outside_count
        ),
        run_profile=args.profile,
        prepared_dataset_yaml=args.prepared_dataset_yaml,
    )
    logging.info("Wrote standardized samples to %s", sample_path)
    logging.info(
        "Converted %d ground samples and %d motion samples.",
        diagnostics["summary"]["ground_selected"],
        diagnostics["summary"]["motion_selected"],
    )
    calibration_output_dir = output_dir / "calibration"
    dataset, config, raw_payload = load_dataset(str(sample_path))
    config_updates = {
        "loss": args.loss,
        "min_motion_rotation_deg": args.min_motion_rotation_deg,
        "solver_family": args.solver_family,
        "planar_motion_policy": args.planar_motion_policy,
    }
    config = CalibrationConfig(**{**config.__dict__, **config_updates})
    result = run_calibration(
        dataset, config=config, output_dir=str(calibration_output_dir)
    )
    manifest = write_outputs(
        output_dir=calibration_output_dir,
        dataset=dataset,
        initial_transform=result["initial_transform"],
        final_transform=result["final_transform"],
        metrics_output=result["metrics"],
        algorithm_report={
            "input_file": str(sample_path.resolve()),
            "config": config.__dict__,
            "dataset_metadata": dataset.metadata,
            "raw_metadata": raw_payload.get("metadata", {}),
            "conversion_summary": diagnostics["summary"],
            "stages": result["stages"],
        },
        evaluation_report=result["evaluation"],
        raw_payload=raw_payload,
    )
    logging.info("Saved lidar2imu calibration outputs to %s", calibration_output_dir)
    logging.info("Manifest: %s", manifest)
    return {
        "sample_path": sample_path,
        "diagnostics": diagnostics,
        "result": result,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Apollo record data into standardized lidar2imu samples "
            "and optionally run calibration."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILE_PRESETS.keys()),
        default=None,
        help=(
            "Apply a fixed lidar2imu preset. baseline=stable scan-to-scan "
            "reference; production=current submap-to-map production candidate. "
            "Explicit CLI flags still override preset values."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--record-path",
        help="Path to a record file or split-record directory.",
    )
    input_group.add_argument(
        "--prepared-dataset-yaml",
        default=None,
        help=(
            "Path to diagnostics/prepared_rig_dataset.yaml generated by "
            "lidar2lidar-rig-dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2imu/record_conversion",
        help="Output directory.",
    )
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/lslidar_main/PointCloud2",
        help="LiDAR point cloud topic.",
    )
    parser.add_argument(
        "--pose-topic",
        default="/apollo/localization/pose",
        help="Pose topic used for IMU-side motion.",
    )
    parser.add_argument(
        "--imu-topic",
        default="/apollo/sensor/gnss/imu",
        help="IMU topic used when gravity-source=imu.",
    )
    parser.add_argument(
        "--parent-frame", default="imu", help="Parent frame for the output extrinsics."
    )
    parser.add_argument(
        "--child-frame",
        default=None,
        help="Optional explicit child frame. Defaults to the LiDAR topic frame_id.",
    )
    parser.add_argument(
        "--initial-transform",
        default=None,
        help=(
            "Optional extrinsics YAML/JSON used when the bag does not contain "
            "lidar->parent TF."
        ),
    )
    parser.add_argument(
        "--extraction-transform",
        default=None,
        help=(
            "Optional extrinsics YAML/JSON used for record-side geometry "
            "construction. Use this for intentional re-extraction while keeping "
            "the in-record TF as the trusted reference."
        ),
    )
    parser.add_argument(
        "--identity-initial-transform",
        action="store_true",
        help=(
            "Use identity as the initial lidar->parent transform when TF is "
            "missing. Exploratory only."
        ),
    )
    parser.add_argument(
        "--gravity-source",
        choices=["pose", "imu"],
        default="pose",
        help="Source for imu_gravity in ground samples.",
    )
    parser.add_argument(
        "--ground-pose-sync-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp gap for ground sample pose sync.",
    )
    parser.add_argument(
        "--motion-pose-sync-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp gap for motion sample pose sync.",
    )
    parser.add_argument(
        "--pose-time-offset-ms",
        type=float,
        default=None,
        help=(
            "Apply a fixed LiDAR-to-pose/IMU time offset before matching. "
            "Positive values shift pose/IMU matching forward relative to "
            "LiDAR timestamps."
        ),
    )
    parser.add_argument(
        "--estimate-pose-time-offset",
        action="store_true",
        help=(
            "Estimate a single global LiDAR-to-pose/IMU offset from "
            "representative samples before extraction."
        ),
    )
    parser.add_argument(
        "--pose-time-offset-estimator",
        choices=["nearest_median", "xcorr_angular_speed"],
        default="nearest_median",
        help="Estimator used by --estimate-pose-time-offset.",
    )
    parser.add_argument(
        "--imu-gravity-window-ms",
        type=float,
        default=100.0,
        help="Window size when averaging IMU gravity samples.",
    )
    parser.add_argument(
        "--max-ground-samples",
        type=int,
        default=16,
        help="Maximum ground samples to export.",
    )
    parser.add_argument(
        "--max-motion-samples",
        type=int,
        default=12,
        help="Maximum motion samples to export.",
    )
    parser.add_argument(
        "--motion-observability-screening",
        choices=["off", "gril_fisher"],
        default="off",
        help=(
            "Extraction-time motion screening mode. gril_fisher evaluates "
            "sliding-window Fisher observability and greedily merges the top "
            "passing segments before final global diversity selection."
        ),
    )
    parser.add_argument(
        "--motion-observability-window-sec",
        type=float,
        default=10.0,
        help="Target duration of the extraction-time observability segment.",
    )
    parser.add_argument(
        "--motion-observability-min-window-sec",
        type=float,
        default=6.0,
        help=(
            "Minimum segment duration required for observability screening "
            "eligibility."
        ),
    )
    parser.add_argument(
        "--motion-observability-min-samples",
        type=int,
        default=3,
        help="Minimum motion candidates in one screened observability segment.",
    )
    parser.add_argument(
        "--motion-observability-top-segments",
        type=int,
        default=8,
        help="Number of ranked observability segments to keep in diagnostics.",
    )
    parser.add_argument(
        "--motion-observability-min-rotation-lambda",
        type=float,
        default=1e-4,
        help=(
            "Minimum rotational Fisher eigenvalue for a segment to pass the "
            "observability hard gate."
        ),
    )
    parser.add_argument(
        "--motion-observability-min-planar-lambda",
        type=float,
        default=1e-3,
        help=(
            "Minimum planar-XY Fisher eigenvalue for a segment to pass the "
            "observability hard gate."
        ),
    )
    parser.add_argument(
        "--motion-observability-max-condition-number",
        type=float,
        default=2e3,
        help=(
            "Maximum combined Fisher condition number allowed for one "
            "observability segment."
        ),
    )
    parser.add_argument(
        "--motion-observability-turn-condition-relax-scale",
        type=float,
        default=2.0,
        help=(
            "Condition-number threshold scale for turn-dominant segments. "
            "Values >1 relax the combined-condition gate on turns."
        ),
    )
    parser.add_argument(
        "--motion-observability-straight-condition-strict-scale",
        type=float,
        default=0.8,
        help=(
            "Condition-number threshold scale for straight-dominant segments. "
            "Values <1 tighten the combined-condition gate on near-straight motion."
        ),
    )
    parser.add_argument(
        "--motion-observability-turn-segment-min-turn-ratio",
        type=float,
        default=0.35,
        help=(
            "Minimum active-turn sample ratio to classify a segment as "
            "turn-dominant for condition-gate relaxation."
        ),
    )
    parser.add_argument(
        "--motion-observability-turn-segment-yaw-p95-deg",
        type=float,
        default=8.0,
        help=(
            "Alternative yaw-excitation threshold (p95 abs yaw, deg) for "
            "turn-dominant segment classification."
        ),
    )
    parser.add_argument(
        "--motion-observability-straight-segment-max-turn-ratio",
        type=float,
        default=0.15,
        help=(
            "Maximum active-turn sample ratio to classify a segment as "
            "straight-dominant for stricter condition gating."
        ),
    )
    parser.add_argument(
        "--motion-observability-straight-segment-yaw-p95-deg",
        type=float,
        default=3.0,
        help=(
            "Maximum yaw-excitation threshold (p95 abs yaw, deg) for "
            "straight-dominant segment classification."
        ),
    )
    parser.add_argument(
        "--motion-observability-rotation-sigma-rad",
        type=float,
        default=0.02,
        help=(
            "Base rotation noise scale (rad) used when building "
            "J^T Sigma^-1 J in extraction screening."
        ),
    )
    parser.add_argument(
        "--motion-observability-translation-sigma-m",
        type=float,
        default=0.05,
        help=(
            "Base translation noise scale (m) used when building "
            "J^T Sigma^-1 J in extraction screening."
        ),
    )
    parser.add_argument(
        "--motion-observability-max-merged-segments",
        type=int,
        default=2,
        help=(
            "Maximum number of top-ranked passing observability segments to "
            "greedily merge before global diversity selection."
        ),
    )
    parser.add_argument(
        "--motion-observability-auto-window-rescue-count",
        type=int,
        default=0,
        help=(
            "Automatically rescue up to N outside-screening windows using "
            "window-mean observability and condition checks."
        ),
    )
    parser.add_argument(
        "--motion-observability-auto-window-rescue-min-relative-score",
        type=float,
        default=0.65,
        help=(
            "Minimum window-mean observability score ratio (relative to current "
            "selected pool mean) for automatic rescue."
        ),
    )
    parser.add_argument(
        "--motion-observability-auto-window-rescue-condition-scale",
        type=float,
        default=1.5,
        help=(
            "Condition-number allowance scale for automatic window rescue "
            "(relative to max-condition-number)."
        ),
    )
    parser.add_argument(
        "--motion-observability-force-include-candidate-key",
        action="append",
        default=[],
        help=(
            "Force-include one motion candidate outside observability-selected "
            "segments for ablation. Format: start_ns:end_ns:frame_stride. "
            "Repeat this option to include multiple keys."
        ),
    )
    parser.add_argument(
        "--motion-observability-force-include-outside-count",
        type=int,
        default=0,
        help=(
            "Force-include the top-N outside-observability candidates "
            "(ranked by overlap/fitness/information score) for ablation."
        ),
    )
    parser.add_argument(
        "--motion-frame-stride",
        type=int,
        default=5,
        help="LiDAR frame stride between motion sample endpoints.",
    )
    parser.add_argument(
        "--motion-max-candidates-per-window",
        type=int,
        default=3,
        help=(
            "Maximum registration candidates retained per motion window before "
            "observability and global diversity selection."
        ),
    )
    parser.add_argument(
        "--plane-dist-thresh",
        type=float,
        default=0.15,
        help="RANSAC plane fitting threshold in meters.",
    )
    parser.add_argument(
        "--plane-normal-thresh-deg",
        type=float,
        default=20.0,
        help="Max angle between extracted plane normal and expected up.",
    )
    parser.add_argument(
        "--registration-voxel-size",
        type=float,
        default=0.3,
        help="Voxel size used for LiDAR-to-LiDAR registration.",
    )
    parser.add_argument(
        "--motion-registration-mode",
        default="scan_to_scan",
        choices=["scan_to_scan", "submap_to_submap", "submap_to_map"],
        help=(
            "Use direct pair registration, symmetric local submaps, or "
            "local-submap to larger local-map registration for motion factors."
        ),
    )
    parser.add_argument(
        "--submap-builder-mode",
        default="pose_only",
        choices=["pose_only", "dense_scan_to_map_gicp"],
        help=(
            "How to build each local submap before pair registration. "
            "pose_only keeps sparse pose-warp accumulation; "
            "dense_scan_to_map_gicp refines each support frame against a "
            "growing local map."
        ),
    )
    parser.add_argument(
        "--submap-half-window",
        type=int,
        default=2,
        help=(
            "Number of support frames on each side of the anchor when "
            "motion-registration-mode=submap_to_submap."
        ),
    )
    parser.add_argument(
        "--submap-support-stride",
        type=int,
        default=None,
        help=(
            "Stride between support frames inside each local submap. Defaults "
            "to motion-frame-stride."
        ),
    )
    parser.add_argument(
        "--submap-min-support-frames",
        type=int,
        default=3,
        help="Minimum aligned LiDAR frames required to keep a local submap candidate.",
    )
    parser.add_argument(
        "--submap-voxel-size",
        type=float,
        default=None,
        help=(
            "Optional voxel size for the merged local submap. Defaults to "
            "registration-voxel-size."
        ),
    )
    parser.add_argument(
        "--map-half-window",
        type=int,
        default=None,
        help=(
            "Number of support frames on each side of the target anchor when "
            "motion-registration-mode=submap_to_map. Defaults to a larger "
            "window than submap-half-window."
        ),
    )
    parser.add_argument(
        "--map-support-stride",
        type=int,
        default=None,
        help=(
            "Stride between support frames inside the target local map when "
            "motion-registration-mode=submap_to_map. Defaults to "
            "submap-support-stride."
        ),
    )
    parser.add_argument(
        "--map-min-support-frames",
        type=int,
        default=None,
        help=(
            "Minimum aligned LiDAR frames required to keep a target local map "
            "candidate. Defaults above submap-min-support-frames."
        ),
    )
    parser.add_argument(
        "--map-voxel-size",
        type=float,
        default=None,
        help=(
            "Optional voxel size for the target local map in submap_to_map "
            "mode. Defaults to submap-voxel-size."
        ),
    )
    parser.add_argument(
        "--min-registration-fitness",
        type=float,
        default=0.55,
        help=(
            "Reject motion pairs whose LiDAR-LiDAR registration fitness is "
            "below this threshold."
        ),
    )
    parser.add_argument(
        "--loss",
        default="huber",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Calibration robust loss.",
    )
    parser.add_argument(
        "--min-motion-rotation-deg",
        type=float,
        default=1.0,
        help="Minimum motion rotation threshold for calibration.",
    )
    parser.add_argument(
        "--imu-preintegration-translation-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for IMU preintegration translation residuals in "
            "planar/joint stages. 0 disables the residual."
        ),
    )
    parser.add_argument(
        "--imu-preintegration-translation-scale-m",
        type=float,
        default=0.08,
        help=(
            "Scale (meters) for IMU preintegration translation residuals when "
            "the preintegration block is enabled."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run lidar2imu calibration immediately after conversion.",
    )
    parser.add_argument(
        "--solver-family",
        default="baseline",
        choices=["baseline", "gril_staged", "gril_prob", "gril_prob_nhc"],
        help=(
            "Calibration solver family. baseline keeps the current path; "
            "gril_staged enables screened staged solving; gril_prob adds "
            "information-weighted motion residuals; gril_prob_nhc adds "
            "weak-motion NHC prior gating."
        ),
    )
    parser.add_argument(
        "--auto-reextract-if-needed",
        action="store_true",
        help=(
            "When calibration reports stale extraction geometry, rerun one "
            "second extraction/calibration pass using the pass-1 calibrated "
            "transform as the extraction seed."
        ),
    )
    parser.add_argument(
        "--planar-motion-policy",
        default="auto",
        choices=["auto", "free", "freeze_xyyaw"],
        help="How to handle weak planar observability during calibration.",
    )
    args = parser.parse_args()
    _apply_run_profile(args, _explicit_cli_options(sys.argv[1:]))
    if args.profile is not None:
        logging.info("Using lidar2imu %s profile.", args.profile)
    if args.estimate_pose_time_offset and args.pose_time_offset_ms is not None:
        parser.error(
            "--estimate-pose-time-offset cannot be combined with --pose-time-offset-ms"
        )

    if args.auto_reextract_if_needed and not args.calibrate:
        parser.error("--auto-reextract-if-needed requires --calibrate")

    output_dir = Path(args.output_dir)
    if not args.calibrate:
        sample_path, diagnostics = convert_record_to_standardized_samples(
            record_path=args.record_path,
            output_dir=str(output_dir),
            lidar_topic=args.lidar_topic,
            pose_topic=args.pose_topic,
            imu_topic=args.imu_topic,
            parent_frame=args.parent_frame,
            child_frame=args.child_frame,
            initial_transform_path=args.initial_transform,
            extraction_transform_path=args.extraction_transform,
            identity_initial_transform=args.identity_initial_transform,
            gravity_source=args.gravity_source,
            ground_pose_sync_threshold_ms=args.ground_pose_sync_threshold_ms,
            motion_pose_sync_threshold_ms=args.motion_pose_sync_threshold_ms,
            pose_time_offset_ms=args.pose_time_offset_ms,
            estimate_pose_time_offset=args.estimate_pose_time_offset,
            pose_time_offset_estimator=args.pose_time_offset_estimator,
            imu_gravity_window_ms=args.imu_gravity_window_ms,
            max_ground_samples=args.max_ground_samples,
            max_motion_samples=args.max_motion_samples,
            motion_frame_stride=args.motion_frame_stride,
            motion_max_candidates_per_window=args.motion_max_candidates_per_window,
            plane_dist_thresh=args.plane_dist_thresh,
            plane_normal_thresh_deg=args.plane_normal_thresh_deg,
            registration_voxel_size=args.registration_voxel_size,
            min_registration_fitness=args.min_registration_fitness,
            calibration_loss=args.loss,
            calibration_motion_rotation_deg=args.min_motion_rotation_deg,
            calibration_solver_family=args.solver_family,
            calibration_planar_motion_policy=args.planar_motion_policy,
            calibration_imu_preintegration_translation_weight=(
                args.imu_preintegration_translation_weight
            ),
            calibration_imu_preintegration_translation_scale_m=(
                args.imu_preintegration_translation_scale_m
            ),
            motion_registration_mode=args.motion_registration_mode,
            submap_builder_mode=args.submap_builder_mode,
            submap_half_window=args.submap_half_window,
            submap_support_stride=args.submap_support_stride,
            submap_min_support_frames=args.submap_min_support_frames,
            submap_voxel_size=args.submap_voxel_size,
            map_half_window=args.map_half_window,
            map_support_stride=args.map_support_stride,
            map_min_support_frames=args.map_min_support_frames,
            map_voxel_size=args.map_voxel_size,
            motion_observability_screening=args.motion_observability_screening,
            motion_observability_window_sec=args.motion_observability_window_sec,
            motion_observability_min_window_sec=(
                args.motion_observability_min_window_sec
            ),
            motion_observability_min_samples=args.motion_observability_min_samples,
            motion_observability_top_segments=args.motion_observability_top_segments,
            motion_observability_min_rotation_lambda=(
                args.motion_observability_min_rotation_lambda
            ),
            motion_observability_min_planar_lambda=(
                args.motion_observability_min_planar_lambda
            ),
            motion_observability_max_condition_number=(
                args.motion_observability_max_condition_number
            ),
            motion_observability_turn_condition_relax_scale=(
                args.motion_observability_turn_condition_relax_scale
            ),
            motion_observability_straight_condition_strict_scale=(
                args.motion_observability_straight_condition_strict_scale
            ),
            motion_observability_turn_segment_min_turn_ratio=(
                args.motion_observability_turn_segment_min_turn_ratio
            ),
            motion_observability_turn_segment_yaw_p95_deg=(
                args.motion_observability_turn_segment_yaw_p95_deg
            ),
            motion_observability_straight_segment_max_turn_ratio=(
                args.motion_observability_straight_segment_max_turn_ratio
            ),
            motion_observability_straight_segment_yaw_p95_deg=(
                args.motion_observability_straight_segment_yaw_p95_deg
            ),
            motion_observability_rotation_sigma_rad=(
                args.motion_observability_rotation_sigma_rad
            ),
            motion_observability_translation_sigma_m=(
                args.motion_observability_translation_sigma_m
            ),
            motion_observability_max_merged_segments=(
                args.motion_observability_max_merged_segments
            ),
            motion_observability_auto_window_rescue_count=(
                args.motion_observability_auto_window_rescue_count
            ),
            motion_observability_auto_window_rescue_min_relative_score=(
                args.motion_observability_auto_window_rescue_min_relative_score
            ),
            motion_observability_auto_window_rescue_condition_scale=(
                args.motion_observability_auto_window_rescue_condition_scale
            ),
            motion_observability_force_include_candidate_keys=(
                args.motion_observability_force_include_candidate_key
            ),
            motion_observability_force_include_outside_count=(
                args.motion_observability_force_include_outside_count
            ),
            run_profile=args.profile,
            prepared_dataset_yaml=args.prepared_dataset_yaml,
        )
        logging.info("Wrote standardized samples to %s", sample_path)
        logging.info(
            "Converted %d ground samples and %d motion samples.",
            diagnostics["summary"]["ground_selected"],
            diagnostics["summary"]["motion_selected"],
        )
        return

    first_pass = _run_conversion_and_calibration(
        args=args,
        output_dir=output_dir,
        initial_transform_path=args.initial_transform,
        extraction_transform_path=args.extraction_transform,
        identity_initial_transform=args.identity_initial_transform,
    )

    pass_summaries = [
        _build_reextract_pass_summary(
            pass_name="pass1",
            pass_dir=output_dir,
            sample_path=Path(first_pass["sample_path"]),
            diagnostics=first_pass["diagnostics"],
            result=first_pass["result"],
            manifest=first_pass["manifest"],
        )
    ]
    trigger_reasons = _reextract_trigger_reasons(pass_summaries[0])
    reextract_summary = {
        "enabled": bool(args.auto_reextract_if_needed),
        "triggered": False,
        "trigger_reasons": trigger_reasons,
        "chosen_pass": "pass1",
        "passes": pass_summaries,
    }
    if not args.auto_reextract_if_needed:
        return

    if trigger_reasons:
        reextract_summary["triggered"] = True
        pass1_dir = output_dir / "reextract_pass1"
        pass2_dir = output_dir / "reextract_pass2"
        if pass1_dir.exists():
            shutil.rmtree(pass1_dir)
        if pass2_dir.exists():
            shutil.rmtree(pass2_dir)
        pass1_dir.mkdir(parents=True, exist_ok=True)
        _copy_output_artifacts(output_dir, pass1_dir)
        _rewrite_pass_summary_paths(pass_summaries[0], pass1_dir)
        first_pass_extrinsics = first_pass["manifest"]["artifacts"][
            "calibrated_extrinsics"
        ]
        try:
            second_pass = _run_conversion_and_calibration(
                args=args,
                output_dir=pass2_dir,
                initial_transform_path=first_pass_extrinsics,
                extraction_transform_path=first_pass_extrinsics,
                identity_initial_transform=False,
            )
        except Exception as exc:
            reextract_summary["pass2_error"] = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "Auto re-extraction pass2 failed; keeping pass1 outputs. %s",
                reextract_summary["pass2_error"],
            )
        else:
            pass_summaries.append(
                _build_reextract_pass_summary(
                    pass_name="pass2",
                    pass_dir=pass2_dir,
                    sample_path=Path(second_pass["sample_path"]),
                    diagnostics=second_pass["diagnostics"],
                    result=second_pass["result"],
                    manifest=second_pass["manifest"],
                )
            )
            chosen_pass = _select_reextract_pass(pass_summaries)
            reextract_summary["chosen_pass"] = chosen_pass["pass_name"]
            if chosen_pass["pass_name"] == "pass2":
                _copy_output_artifacts(pass2_dir, output_dir)
                _refresh_calibration_manifest_paths(output_dir / "calibration")
    reextract_summary_path = output_dir / "reextract_summary.yaml"
    with open(reextract_summary_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(reextract_summary, file, sort_keys=False)
    logging.info("Wrote re-extraction summary to %s", reextract_summary_path)


if __name__ == "__main__":
    main()
