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
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration_common.evaluation import write_acceptance_artifacts
from calibration_common.evaluation import write_paradigm_artifacts
from calibration_common.evaluation import write_table_csv
from lidar2imu.algorithms import normalize_plane, normalize_vector
from lidar2imu.models import CalibrationConfig
from lidar2imu.models import CalibrationDataset
from lidar2imu.models import GroundSample
from lidar2imu.models import MotionSample
from lidar2imu.visualization import build_visualization_artifacts
from lidar2lidar.extrinsic_io import build_extrinsics_payload
from lidar2lidar.extrinsic_io import extrinsics_filename
from lidar2lidar.extrinsic_io import parse_transform_payload
from lidar2lidar.extrinsic_io import save_extrinsics_yaml

# isort: on


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


def _motion_sample_from_payload(raw_sample: dict) -> MotionSample:
    if "imu_delta" not in raw_sample or "lidar_delta" not in raw_sample:
        raise ValueError("Each motion sample must contain imu_delta and lidar_delta.")
    imu_delta, _, _, _ = parse_transform_payload(raw_sample["imu_delta"])
    lidar_delta, _, _, _ = parse_transform_payload(raw_sample["lidar_delta"])
    sample_metadata = dict(raw_sample.get("metadata", {}))
    for reserved_key in (
        "start_timestamp_ns",
        "end_timestamp_ns",
        "imu_delta",
        "lidar_delta",
        "weight",
        "sync_dt_ms",
    ):
        sample_metadata.pop(reserved_key, None)
    return MotionSample(
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
        metadata=sample_metadata,
    )


def _motion_sample_from_review_candidate(candidate: dict[str, Any]) -> MotionSample:
    return _motion_sample_from_payload(
        {
            "start_timestamp_ns": candidate["start_timestamp_ns"],
            "end_timestamp_ns": candidate["end_timestamp_ns"],
            "imu_delta": candidate["imu_delta"],
            "lidar_delta": candidate["lidar_delta"],
            "weight": 1.0,
            "sync_dt_ms": candidate.get("sync_dt_ms"),
            "metadata": {
                "record_path_start": candidate.get("record_path_start"),
                "record_path_end": candidate.get("record_path_end"),
                "lidar_topic": candidate.get("lidar_topic"),
                "pose_topic": candidate.get("pose_topic"),
                "window_id": (
                    None
                    if candidate.get("window_id") is None
                    else int(candidate["window_id"])
                ),
                "motion_registration_mode": candidate.get("motion_registration_mode"),
                "frame_stride": (
                    None
                    if candidate.get("frame_stride") is None
                    else int(candidate["frame_stride"])
                ),
                "pose_rotation_deg": float(candidate.get("pose_rotation_deg", 0.0)),
                "pose_translation_m": float(candidate.get("pose_translation_m", 0.0)),
                "information_score": float(candidate.get("information_score", 0.0)),
                "probabilistic_information_score": candidate.get(
                    "probabilistic_information_score"
                ),
                "probabilistic_window_score": candidate.get(
                    "probabilistic_window_score"
                ),
                "information_uncertainty_scale": candidate.get(
                    "information_uncertainty_scale"
                ),
                "information_rotation_confidence": candidate.get(
                    "information_rotation_confidence"
                ),
                "information_translation_confidence": candidate.get(
                    "information_translation_confidence"
                ),
                "observability_segment_id": candidate.get("observability_segment_id"),
                "observability_combined_min_eigenvalue": candidate.get(
                    "observability_combined_min_eigenvalue"
                ),
                "observability_combined_condition_number": candidate.get(
                    "observability_combined_condition_number"
                ),
                "observability_min_eigenvalue_gain": candidate.get(
                    "observability_min_eigenvalue_gain"
                ),
                "observability_min_eigenvalue_gain_ratio": candidate.get(
                    "observability_min_eigenvalue_gain_ratio"
                ),
                "observability_condition_worsening_ratio": candidate.get(
                    "observability_condition_worsening_ratio"
                ),
                "observability_capacity_weight": candidate.get(
                    "observability_capacity_weight"
                ),
                "imu_signed_yaw_deg": float(candidate.get("imu_signed_yaw_deg", 0.0)),
                "imu_translation_heading_deg": candidate.get(
                    "imu_translation_heading_deg"
                ),
                "imu_preintegration_delta_translation_m": candidate.get(
                    "imu_preintegration_delta_translation_m"
                ),
                "imu_preintegration_delta_velocity_mps": candidate.get(
                    "imu_preintegration_delta_velocity_mps"
                ),
                "imu_preintegration_confidence": candidate.get(
                    "imu_preintegration_confidence"
                ),
                "imu_preintegration_sample_count": candidate.get(
                    "imu_preintegration_sample_count"
                ),
                "imu_preintegration_valid_step_count": candidate.get(
                    "imu_preintegration_valid_step_count"
                ),
                "imu_preintegration_duration_sec": candidate.get(
                    "imu_preintegration_duration_sec"
                ),
                "imu_preintegration_mean_specific_accel_mps2": candidate.get(
                    "imu_preintegration_mean_specific_accel_mps2"
                ),
                "imu_preintegration_source": candidate.get("imu_preintegration_source"),
                "registration_fitness": float(
                    candidate.get("registration_fitness", 0.0)
                ),
                "registration_inlier_rmse": float(
                    candidate.get("registration_inlier_rmse", 0.0)
                ),
                "registered_overlap_quality_score": candidate.get(
                    "registered_overlap_quality_score"
                ),
                "registered_overlap_within_0p4m_ratio": candidate.get(
                    "registered_overlap_within_0p4m_ratio"
                ),
                "registered_overlap_nn_mean_m": candidate.get(
                    "registered_overlap_nn_mean_m"
                ),
                "global_selection_score": float(
                    candidate.get("global_selection_score") or 0.0
                ),
                "selected_for_calibration": bool(
                    candidate.get("selected_for_calibration", False)
                ),
            },
        }
    )


def load_dataset(
    path: str,
    parent_frame_override: str | None = None,
    child_frame_override: str | None = None,
) -> tuple[CalibrationDataset, CalibrationConfig, dict]:
    payload = _load_payload(path)
    parent_frame = parent_frame_override or str(payload.get("parent_frame", "imu"))
    child_frame = child_frame_override or str(payload.get("child_frame", "lidar"))

    initial_transform = np.eye(4, dtype=float)
    if "initial_transform" in payload:
        initial_transform, _, _, _ = parse_transform_payload(
            payload["initial_transform"]
        )
    extraction_transform = None
    if "extraction_transform" in payload:
        extraction_transform, _, _, _ = parse_transform_payload(
            payload["extraction_transform"]
        )
    reference_transform = None
    if "reference_transform" in payload:
        reference_transform, _, _, _ = parse_transform_payload(
            payload["reference_transform"]
        )
    dataset_metadata = dict(payload.get("metadata", {}))
    extraction_transform_source = dataset_metadata.get("extraction_transform_source")
    if extraction_transform is None:
        if (
            extraction_transform_source is not None
            and extraction_transform_source
            == dataset_metadata.get("reference_transform_source")
            and reference_transform is not None
        ):
            extraction_transform = reference_transform
        elif (
            extraction_transform_source is not None
            and extraction_transform_source
            == dataset_metadata.get("initial_transform_source")
        ):
            extraction_transform = initial_transform
        else:
            extraction_transform = initial_transform

    ground_samples = []
    for raw_sample in payload.get("ground_samples", []):
        normal_payload = raw_sample.get(
            "lidar_plane_normal", raw_sample.get("plane_normal")
        )
        if normal_payload is None and isinstance(raw_sample.get("lidar_plane"), dict):
            normal_payload = raw_sample["lidar_plane"].get("normal")
        offset_value = raw_sample.get(
            "lidar_plane_offset", raw_sample.get("plane_offset")
        )
        if offset_value is None and isinstance(raw_sample.get("lidar_plane"), dict):
            offset_value = raw_sample["lidar_plane"].get("offset")
        if normal_payload is None or offset_value is None:
            raise ValueError(
                "Each ground sample must contain lidar_plane_normal and "
                "lidar_plane_offset."
            )
        normal, offset = normalize_plane(
            _vector_from_payload(normal_payload, "normal"), float(offset_value)
        )
        gravity = normalize_vector(_vector_from_payload(raw_sample, "imu_gravity"))
        sample_metadata = dict(raw_sample.get("metadata", {}))
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
            sample_metadata.pop(reserved_key, None)
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
                metadata=sample_metadata,
            )
        )

    motion_samples = []
    for raw_sample in payload.get("motion_samples", []):
        motion_samples.append(_motion_sample_from_payload(raw_sample))

    motion_candidate_pool = []
    for candidate in payload.get("review_motion_candidates", []):
        if not isinstance(candidate, dict):
            continue
        motion_candidate_pool.append(_motion_sample_from_review_candidate(candidate))

    raw_config = payload.get("config", {})
    config = CalibrationConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in CalibrationConfig.__dataclass_fields__
        }
    )
    dataset = CalibrationDataset(
        parent_frame=parent_frame,
        child_frame=child_frame,
        ground_samples=ground_samples,
        motion_samples=motion_samples,
        initial_transform=initial_transform,
        extraction_transform=extraction_transform,
        reference_transform=reference_transform,
        motion_candidate_pool=motion_candidate_pool,
        metadata=dataset_metadata,
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


def write_outputs(
    output_dir: Path,
    dataset: CalibrationDataset,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    metrics_output: dict,
    algorithm_report: dict,
    evaluation_report: dict,
    raw_payload: dict | None = None,
) -> dict:
    initial_guess_dir, calibrated_dir, diagnostics_dir = prepare_output_layout(
        output_dir
    )
    calibrated_file = calibrated_dir / extrinsics_filename(
        dataset.parent_frame, dataset.child_frame
    )
    initial_guess_file = initial_guess_dir / extrinsics_filename(
        dataset.parent_frame, dataset.child_frame
    )

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
    with open(diagnostics_dir / "algorithm.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(algorithm_report, file, sort_keys=False)
    with open(diagnostics_dir / "evaluation.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_report, file, sort_keys=False)
    with open(diagnostics_dir / "observability.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(
            evaluation_report.get("observability", {}), file, sort_keys=False
        )
    acceptance_artifacts = write_acceptance_artifacts(
        diagnostics_dir, metrics_output["final_acceptance"]
    )
    table_artifacts = {
        "ground_residuals_csv": write_table_csv(
            diagnostics_dir / "ground_residuals.csv",
            evaluation_report.get("ground_per_sample", []),
        ),
        "motion_residuals_csv": write_table_csv(
            diagnostics_dir / "motion_residuals.csv",
            evaluation_report.get("motion_per_sample", []),
        ),
        "holdout_motion_residuals_csv": write_table_csv(
            diagnostics_dir / "holdout_motion_residuals.csv",
            evaluation_report.get("holdout_motion_per_sample", []),
        ),
        "cloud_thickness_window_frames_csv": write_table_csv(
            diagnostics_dir / "cloud_thickness_window_frames.csv",
            evaluation_report.get("cloud_thickness_window_frames", []),
        ),
    }
    solver_window_ids = None
    stages = algorithm_report.get("stages", {})
    if isinstance(stages, dict):
        planar_stage = stages.get("planar_stage", {})
        if isinstance(planar_stage, dict):
            raw_window_ids = planar_stage.get("used_window_ids")
            if isinstance(raw_window_ids, list):
                solver_window_ids = [int(window_id) for window_id in raw_window_ids]
    visualization_artifacts = build_visualization_artifacts(
        diagnostics_dir,
        dataset=dataset,
        final_transform=final_transform,
        summary=metrics_output.get("summary", {}),
        final_acceptance=metrics_output["final_acceptance"],
        motion_assessment=metrics_output.get("vehicle_motion_assessment", {}),
        ground_rows=evaluation_report.get("ground_per_sample", []),
        motion_rows=evaluation_report.get("motion_per_sample", []),
        holdout_motion_rows=evaluation_report.get("holdout_motion_per_sample", []),
        observability=evaluation_report.get("observability", {}),
        raw_payload=raw_payload or {},
        artifact_links={
            "acceptance_report": acceptance_artifacts["acceptance_report"],
            "metrics": str(output_dir / "metrics.yaml"),
            "data_quality": str(diagnostics_dir / "data_quality.yaml"),
            "observability": str(diagnostics_dir / "observability.yaml"),
            "ground_residuals_csv": table_artifacts["ground_residuals_csv"],
            "motion_residuals_csv": table_artifacts["motion_residuals_csv"],
            "holdout_motion_residuals_csv": table_artifacts[
                "holdout_motion_residuals_csv"
            ],
            "cloud_thickness_window_frames_csv": table_artifacts[
                "cloud_thickness_window_frames_csv"
            ],
        },
        solver_window_ids=solver_window_ids,
    )
    standardized_data = {
        "schema_version": 1,
        "module": "lidar2imu",
        "representation": "standardized_samples",
        "frames": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
        },
        "sample_counts": {
            "ground_samples": len(dataset.ground_samples),
            "motion_samples": len(dataset.motion_samples),
        },
        "metadata": dataset.metadata,
        "transforms": {
            "has_initial_transform": dataset.initial_transform is not None,
            "has_extraction_transform": dataset.extraction_transform is not None,
            "has_reference_transform": dataset.reference_transform is not None,
        },
    }
    data_quality = {
        "schema_version": 1,
        "module": "lidar2imu",
        "status": metrics_output["final_acceptance"]["status"],
        "release_ready": metrics_output["final_acceptance"]["release_ready"],
        "quality_gates": metrics_output["final_acceptance"]["gates"],
        "coarse_statuses": metrics_output["coarse_metrics"]["statuses"],
        "recommendation": metrics_output["final_acceptance"]["recommendation"],
    }
    visualization_index = {
        "schema_version": 1,
        "module": "lidar2imu",
        "layers": {
            "conclusion": [
                acceptance_artifacts["acceptance_report"],
                acceptance_artifacts["status_summary_csv"],
            ],
            "detail_metrics": [
                str(output_dir / "metrics.yaml"),
                str(diagnostics_dir / "evaluation.yaml"),
                str(diagnostics_dir / "observability.yaml"),
                table_artifacts["ground_residuals_csv"],
                table_artifacts["motion_residuals_csv"],
                table_artifacts["holdout_motion_residuals_csv"],
                table_artifacts["cloud_thickness_window_frames_csv"],
            ],
            "visual_artifacts": [
                visualization_artifacts["review_report"],
                visualization_artifacts["ground_residuals_plot"],
                visualization_artifacts["ground_height_residuals_plot"],
                visualization_artifacts["motion_rotation_residuals_plot"],
                visualization_artifacts["motion_residuals_plot"],
                visualization_artifacts["motion_registration_fitness_plot"],
                *(
                    [visualization_artifacts["trajectory_overlay_plot"]]
                    if "trajectory_overlay_plot" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["trajectory_position_gap_plot"]]
                    if "trajectory_position_gap_plot" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["holdout_motion_residuals_plot"]]
                    if "holdout_motion_residuals_plot" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["imu_trajectory_cloud"]]
                    if "imu_trajectory_cloud" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["lidar_trajectory_cloud"]]
                    if "lidar_trajectory_cloud" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["trajectory_overlay_cloud"]]
                    if "trajectory_overlay_cloud" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["registration_review_yaml"]]
                    if "registration_review_yaml" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["registration_review_csv"]]
                    if "registration_review_csv" in visualization_artifacts
                    else []
                ),
                *(
                    [visualization_artifacts["yaw_cost_scan"]]
                    if "yaw_cost_scan" in visualization_artifacts
                    else []
                ),
            ],
            "visual_review": [
                (
                    "Open diagnostics/review_report.html in a browser for the "
                    "quickest visual triage."
                ),
                "Inspect ground_residuals_plot.svg for ground-plane angle stability.",
                (
                    "Inspect motion_residuals_plot.svg for translation residuals "
                    "and motion quality."
                ),
                (
                    "Inspect trajectory_overlay.svg and "
                    "trajectory_position_gap_plot.svg to compare IMU and LiDAR "
                    "BEV review trajectories across the sequence."
                ),
                (
                    "Open trajectory_overlay_cloud.ply in CloudCompare/Open3D to "
                    "inspect actual registered-object overlap: gray target "
                    "geometry, blue IMU-predicted source geometry, and red "
                    "LiDAR-registered source geometry."
                ),
                (
                    "Inspect registration_review.yaml or registration_review.csv "
                    "for per-window overlap ratios and nearest-neighbor tails."
                ),
                (
                    "Inspect cloud_thickness_window_frames.csv and "
                    "metrics.yaml fine_metrics.cloud_thickness for the holdout "
                    "5s straight accel/brake stitched-cloud thickness gate."
                ),
                (
                    "Inspect yaw_cost_scan.svg for a sharp, well-supported yaw "
                    "optimum when present."
                ),
            ],
        },
    }
    paradigm_artifacts = write_paradigm_artifacts(
        diagnostics_dir,
        standardized_data=standardized_data,
        data_quality=data_quality,
        visualization_index=visualization_index,
    )
    metrics_output["fine_metrics"]["artifacts"].update(acceptance_artifacts)
    metrics_output["fine_metrics"]["artifacts"].update(table_artifacts)
    metrics_output["fine_metrics"]["artifacts"].update(visualization_artifacts)
    metrics_output["fine_metrics"]["artifacts"].update(paradigm_artifacts)
    with open(output_dir / "metrics.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)

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
                "acceptance_report": acceptance_artifacts["acceptance_report"],
                "status_summary_csv": acceptance_artifacts["status_summary_csv"],
                "standardized_data": paradigm_artifacts["standardized_data"],
                "data_quality": paradigm_artifacts["data_quality"],
                "visualization_index": paradigm_artifacts["visualization_index"],
                "ground_residuals_csv": table_artifacts["ground_residuals_csv"],
                "motion_residuals_csv": table_artifacts["motion_residuals_csv"],
                "holdout_motion_residuals_csv": table_artifacts[
                    "holdout_motion_residuals_csv"
                ],
                "cloud_thickness_window_frames_csv": table_artifacts[
                    "cloud_thickness_window_frames_csv"
                ],
                "ground_residuals_plot": visualization_artifacts[
                    "ground_residuals_plot"
                ],
                "ground_height_residuals_plot": visualization_artifacts[
                    "ground_height_residuals_plot"
                ],
                "motion_rotation_residuals_plot": visualization_artifacts[
                    "motion_rotation_residuals_plot"
                ],
                "motion_residuals_plot": visualization_artifacts[
                    "motion_residuals_plot"
                ],
                "motion_registration_fitness_plot": visualization_artifacts[
                    "motion_registration_fitness_plot"
                ],
                "trajectory_overlay_plot": visualization_artifacts.get(
                    "trajectory_overlay_plot"
                ),
                "trajectory_position_gap_plot": visualization_artifacts.get(
                    "trajectory_position_gap_plot"
                ),
                "imu_trajectory_cloud": visualization_artifacts.get(
                    "imu_trajectory_cloud"
                ),
                "lidar_trajectory_cloud": visualization_artifacts.get(
                    "lidar_trajectory_cloud"
                ),
                "trajectory_overlay_cloud": visualization_artifacts.get(
                    "trajectory_overlay_cloud"
                ),
                "registration_review_yaml": visualization_artifacts.get(
                    "registration_review_yaml"
                ),
                "registration_review_csv": visualization_artifacts.get(
                    "registration_review_csv"
                ),
                "holdout_motion_residuals_plot": visualization_artifacts.get(
                    "holdout_motion_residuals_plot"
                ),
                "yaw_cost_scan": visualization_artifacts.get("yaw_cost_scan"),
                "review_report": visualization_artifacts["review_report"],
            },
        },
    }
    with open(diagnostics_dir / "manifest.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    return manifest
