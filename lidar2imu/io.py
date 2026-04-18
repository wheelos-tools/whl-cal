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

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lidar2imu.algorithms import normalize_plane, normalize_vector
from lidar2imu.models import CalibrationConfig, CalibrationDataset, GroundSample, MotionSample
from lidar2lidar.extrinsic_io import (
    build_extrinsics_payload,
    extrinsics_filename,
    parse_transform_payload,
    save_extrinsics_yaml,
)


def _load_payload(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        if Path(path).suffix.lower() == ".json":
            return json.load(file)
        return yaml.safe_load(file)


def _vector_from_payload(payload: Any, key: str) -> np.ndarray:
    if isinstance(payload, dict) and key in payload:
        payload = payload[key]
    if isinstance(payload, dict):
        if {"x", "y", "z"} <= payload.keys():
            return np.array([payload["x"], payload["y"], payload["z"]], dtype=float)
    return np.asarray(payload, dtype=float).reshape(3)


def _sample_timestamp(payload: dict, key: str, fallback: int = 0) -> int:
    value = payload.get(key, fallback)
    return int(value if value is not None else fallback)


def load_dataset(path: str,
                 parent_frame_override: str | None = None,
                 child_frame_override: str | None = None) -> tuple[CalibrationDataset, CalibrationConfig, dict]:
    payload = _load_payload(path)
    parent_frame = parent_frame_override or str(payload.get("parent_frame", "imu"))
    child_frame = child_frame_override or str(payload.get("child_frame", "lidar"))

    initial_transform = np.eye(4, dtype=float)
    if "initial_transform" in payload:
        initial_transform, _, _, _ = parse_transform_payload(payload["initial_transform"])

    ground_samples = []
    for raw_sample in payload.get("ground_samples", []):
        normal_payload = raw_sample.get("lidar_plane_normal", raw_sample.get("plane_normal"))
        if normal_payload is None and isinstance(raw_sample.get("lidar_plane"), dict):
            normal_payload = raw_sample["lidar_plane"].get("normal")
        offset_value = raw_sample.get("lidar_plane_offset", raw_sample.get("plane_offset"))
        if offset_value is None and isinstance(raw_sample.get("lidar_plane"), dict):
            offset_value = raw_sample["lidar_plane"].get("offset")
        if normal_payload is None or offset_value is None:
            raise ValueError("Each ground sample must contain lidar_plane_normal and lidar_plane_offset.")
        normal, offset = normalize_plane(_vector_from_payload(normal_payload, "normal"), float(offset_value))
        gravity = normalize_vector(_vector_from_payload(raw_sample, "imu_gravity"))
        metadata = dict(raw_sample.get("metadata", {}))
        for reserved_key in (
            "timestamp_ns",
            "lidar_plane_normal",
            "plane_normal",
            "lidar_plane_offset",
            "plane_offset",
            "lidar_plane",
            "imu_gravity",
            "imu_ground_height",
            "weight",
            "sync_dt_ms",
        ):
            metadata.pop(reserved_key, None)
        ground_samples.append(
            GroundSample(
                timestamp_ns=_sample_timestamp(raw_sample, "timestamp_ns"),
                lidar_plane_normal=normal,
                lidar_plane_offset=offset,
                imu_gravity=gravity,
                imu_ground_height=(
                    float(raw_sample["imu_ground_height"])
                    if raw_sample.get("imu_ground_height") is not None
                    else None
                ),
                weight=float(raw_sample.get("weight", 1.0)),
                sync_dt_ms=(
                    float(raw_sample["sync_dt_ms"])
                    if raw_sample.get("sync_dt_ms") is not None
                    else None
                ),
                metadata=metadata,
            )
        )

    motion_samples = []
    for raw_sample in payload.get("motion_samples", []):
        if "imu_delta" not in raw_sample or "lidar_delta" not in raw_sample:
            raise ValueError("Each motion sample must contain imu_delta and lidar_delta.")
        imu_delta, _, _, _ = parse_transform_payload(raw_sample["imu_delta"])
        lidar_delta, _, _, _ = parse_transform_payload(raw_sample["lidar_delta"])
        metadata = dict(raw_sample.get("metadata", {}))
        for reserved_key in (
            "start_timestamp_ns",
            "end_timestamp_ns",
            "imu_delta",
            "lidar_delta",
            "weight",
            "sync_dt_ms",
        ):
            metadata.pop(reserved_key, None)
        motion_samples.append(
            MotionSample(
                start_timestamp_ns=_sample_timestamp(raw_sample, "start_timestamp_ns"),
                end_timestamp_ns=_sample_timestamp(raw_sample, "end_timestamp_ns"),
                imu_delta_rotation=np.asarray(imu_delta[:3, :3], dtype=float),
                imu_delta_translation=np.asarray(imu_delta[:3, 3], dtype=float),
                lidar_delta_rotation=np.asarray(lidar_delta[:3, :3], dtype=float),
                lidar_delta_translation=np.asarray(lidar_delta[:3, 3], dtype=float),
                weight=float(raw_sample.get("weight", 1.0)),
                sync_dt_ms=(
                    float(raw_sample["sync_dt_ms"])
                    if raw_sample.get("sync_dt_ms") is not None
                    else None
                ),
                metadata=metadata,
            )
        )

    raw_config = payload.get("config", {})
    config = CalibrationConfig(
        **{key: value for key, value in raw_config.items() if key in CalibrationConfig.__dataclass_fields__}
    )
    dataset = CalibrationDataset(
        parent_frame=parent_frame,
        child_frame=child_frame,
        ground_samples=ground_samples,
        motion_samples=motion_samples,
        initial_transform=initial_transform,
        metadata=dict(payload.get("metadata", {})),
    )
    return dataset, config, payload


def prepare_output_layout(output_dir: Path) -> tuple[Path, Path, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for file_path in (
        output_dir / "calibrated_tf.yaml",
        output_dir / "metrics.yaml",
        diagnostics_dir / "manifest.yaml",
        diagnostics_dir / "algorithm.yaml",
        diagnostics_dir / "evaluation.yaml",
        diagnostics_dir / "observability.yaml",
    ):
        if file_path.exists() and file_path.is_file():
            file_path.unlink()

    initial_guess_dir = output_dir / "initial_guess"
    calibrated_dir = output_dir / "calibrated"
    initial_guess_dir.mkdir(parents=True, exist_ok=True)
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    return initial_guess_dir, calibrated_dir, diagnostics_dir


def write_outputs(output_dir: Path,
                  dataset: CalibrationDataset,
                  initial_transform: np.ndarray,
                  final_transform: np.ndarray,
                  metrics_output: dict,
                  algorithm_report: dict,
                  evaluation_report: dict) -> dict:
    initial_guess_dir, calibrated_dir, diagnostics_dir = prepare_output_layout(output_dir)
    calibrated_file = calibrated_dir / extrinsics_filename(dataset.parent_frame, dataset.child_frame)
    initial_guess_file = initial_guess_dir / extrinsics_filename(dataset.parent_frame, dataset.child_frame)

    save_extrinsics_yaml(
        str(initial_guess_file),
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        matrix=initial_transform,
        metadata={"source": "input_initial_transform"},
    )
    save_extrinsics_yaml(
        str(calibrated_file),
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        matrix=final_transform,
        metrics=metrics_output["coarse_metrics"],
        metadata={"source": "lidar2imu-calibrate"},
    )

    tf_payload = {
        "base_frame": dataset.parent_frame,
        "extrinsics": [
            build_extrinsics_payload(
                parent_frame=dataset.parent_frame,
                child_frame=dataset.child_frame,
                matrix=final_transform,
                metrics=metrics_output["coarse_metrics"],
                metadata={"source": "lidar2imu-calibrate"},
            )
        ],
    }
    with open(output_dir / "calibrated_tf.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(tf_payload, file, sort_keys=False)
    with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)
    with open(diagnostics_dir / "algorithm.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(algorithm_report, file, sort_keys=False)
    with open(diagnostics_dir / "evaluation.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_report, file, sort_keys=False)
    with open(diagnostics_dir / "observability.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_report.get("observability", {}), file, sort_keys=False)

    manifest = {
        "parent_frame": dataset.parent_frame,
        "child_frame": dataset.child_frame,
        "sample_counts": {
            "ground_samples": len(dataset.ground_samples),
            "motion_samples": len(dataset.motion_samples),
        },
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "metrics": str(output_dir / "metrics.yaml"),
            "initial_guess": str(initial_guess_file),
            "calibrated_extrinsics": str(calibrated_file),
            "diagnostics": {
                "algorithm": str(diagnostics_dir / "algorithm.yaml"),
                "evaluation": str(diagnostics_dir / "evaluation.yaml"),
                "observability": str(diagnostics_dir / "observability.yaml"),
            },
        },
    }
    with open(diagnostics_dir / "manifest.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    return manifest
