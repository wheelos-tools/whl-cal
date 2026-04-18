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

"""Sweep lidar2imu record-conversion settings and summarize results."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lidar2imu.io import load_dataset, write_outputs  # noqa: E402
from lidar2imu.models import CalibrationConfig  # noqa: E402
from lidar2imu.pipeline import run_calibration  # noqa: E402
from lidar2imu.record_converter import (  # noqa: E402
    convert_record_to_standardized_samples,
)


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _trial_name(
    gravity_source: str,
    motion_frame_stride: int,
    min_registration_fitness: float,
    min_motion_rotation_deg: float,
    planar_motion_policy: str,
) -> str:
    return (
        f"gravity_{gravity_source}"
        f"__stride_{motion_frame_stride}"
        f"__fitness_{_slugify_float(min_registration_fitness)}"
        f"__minrot_{_slugify_float(min_motion_rotation_deg)}"
        f"__policy_{planar_motion_policy}"
    )


def _warning_count(statuses: dict | None) -> int:
    if not isinstance(statuses, dict):
        return 99
    return sum(1 for value in statuses.values() if value != "pass")


def _recommendation_rank(recommendation: str | None) -> int:
    order = {
        "full_6dof_candidate": 0,
        "z_roll_pitch_priority": 1,
        "recollect_data": 2,
        None: 3,
    }
    return order.get(recommendation, 4)


def _status_rank(status: str) -> int:
    order = {
        "ok": 0,
        "calibration_failed": 1,
        "conversion_failed": 2,
    }
    return order.get(status, 3)


def _flatten_result(result: dict) -> dict:
    conversion = result.get("conversion_summary", {})
    coarse = result.get("coarse_metrics", {})
    assessment = result.get("vehicle_motion_assessment", {})
    summary = result.get("final_summary", {})
    translation = summary.get("final_translation_m", {})
    euler = summary.get("final_euler_deg", {})
    delta_to_initial = summary.get("delta_to_initial", {})
    return {
        "trial_name": result.get("trial_name"),
        "status": result.get("status"),
        "gravity_source": result.get("gravity_source"),
        "motion_frame_stride": result.get("motion_frame_stride"),
        "min_registration_fitness": result.get("min_registration_fitness"),
        "min_motion_rotation_deg": result.get("min_motion_rotation_deg"),
        "planar_motion_policy": result.get("planar_motion_policy"),
        "ground_selected": conversion.get("ground_selected"),
        "motion_selected": conversion.get("motion_selected"),
        "motion_rejected_low_fitness": conversion.get("motion_rejected_low_fitness"),
        "recommendation": assessment.get("recommendation"),
        "ground_normal_angle_p95_deg": coarse.get("ground_normal_angle_p95_deg"),
        "ground_height_residual_p95_m": coarse.get("ground_height_residual_p95_m"),
        "motion_rotation_residual_p95_deg": coarse.get(
            "motion_rotation_residual_p95_deg"
        ),
        "motion_translation_residual_p95_m": coarse.get(
            "motion_translation_residual_p95_m"
        ),
        "motion_registration_fitness_p05": coarse.get(
            "motion_registration_fitness_p05"
        ),
        "motion_registration_inlier_rmse_p95": coarse.get(
            "motion_registration_inlier_rmse_p95"
        ),
        "left_turn_count": coarse.get("left_turn_count"),
        "right_turn_count": coarse.get("right_turn_count"),
        "turn_balance_ratio": coarse.get("turn_balance_ratio"),
        "joint_condition_number": coarse.get("joint_condition_number"),
        "final_x": translation.get("x"),
        "final_y": translation.get("y"),
        "final_z": translation.get("z"),
        "final_yaw_deg": euler.get("yaw"),
        "final_roll_deg": euler.get("roll"),
        "final_pitch_deg": euler.get("pitch"),
        "delta_translation_norm_m": delta_to_initial.get("translation_norm_m"),
        "delta_rotation_deg": delta_to_initial.get("rotation_deg"),
        "applied_solver_planar_motion_policy": result.get(
            "vehicle_motion_assessment", {}
        ).get("applied_solver_planar_motion_policy"),
        "error": result.get("error"),
    }


def _result_sort_key(result: dict) -> tuple:
    coarse = result.get("coarse_metrics", {})
    statuses = coarse.get("statuses", {})
    conversion = result.get("conversion_summary", {})
    delta_to_initial = result.get("final_summary", {}).get("delta_to_initial", {})
    return (
        _status_rank(result.get("status", "")),
        _recommendation_rank(
            result.get("vehicle_motion_assessment", {}).get("recommendation")
        ),
        float(delta_to_initial.get("translation_norm_m") or 1e9),
        float(delta_to_initial.get("rotation_deg") or 1e9),
        _warning_count(statuses),
        -int(conversion.get("ground_selected", 0) or 0),
        -int(conversion.get("motion_selected", 0) or 0),
        float(coarse.get("motion_translation_residual_p95_m") or 1e9),
        float(coarse.get("motion_rotation_residual_p95_deg") or 1e9),
        -float(coarse.get("motion_registration_fitness_p05") or -1e9),
    )


def _run_trial(
    args: argparse.Namespace,
    gravity_source: str,
    motion_frame_stride: int,
    min_registration_fitness: float,
    min_motion_rotation_deg: float,
    planar_motion_policy: str,
) -> dict:
    trial_name = _trial_name(
        gravity_source=gravity_source,
        motion_frame_stride=motion_frame_stride,
        min_registration_fitness=min_registration_fitness,
        min_motion_rotation_deg=min_motion_rotation_deg,
        planar_motion_policy=planar_motion_policy,
    )
    trial_output_dir = Path(args.output_dir) / trial_name
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "trial_name": trial_name,
        "gravity_source": gravity_source,
        "motion_frame_stride": motion_frame_stride,
        "min_registration_fitness": min_registration_fitness,
        "min_motion_rotation_deg": min_motion_rotation_deg,
        "planar_motion_policy": planar_motion_policy,
        "status": "conversion_failed",
    }

    try:
        sample_path, diagnostics = convert_record_to_standardized_samples(
            record_path=args.record_path,
            output_dir=str(trial_output_dir),
            lidar_topic=args.lidar_topic,
            pose_topic=args.pose_topic,
            imu_topic=args.imu_topic,
            parent_frame=args.parent_frame,
            child_frame=args.child_frame,
            initial_transform_path=args.initial_transform,
            identity_initial_transform=args.identity_initial_transform,
            gravity_source=gravity_source,
            ground_pose_sync_threshold_ms=args.ground_pose_sync_threshold_ms,
            motion_pose_sync_threshold_ms=args.motion_pose_sync_threshold_ms,
            imu_gravity_window_ms=args.imu_gravity_window_ms,
            max_ground_samples=args.max_ground_samples,
            max_motion_samples=args.max_motion_samples,
            motion_frame_stride=motion_frame_stride,
            plane_dist_thresh=args.plane_dist_thresh,
            plane_normal_thresh_deg=args.plane_normal_thresh_deg,
            registration_voxel_size=args.registration_voxel_size,
            min_registration_fitness=min_registration_fitness,
            calibration_loss=args.loss,
            calibration_motion_rotation_deg=min_motion_rotation_deg,
            calibration_planar_motion_policy=planar_motion_policy,
        )
        result["conversion_summary"] = diagnostics.get("summary", {})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if not args.calibrate:
        result["status"] = "ok"
        return result

    try:
        dataset, config, raw_payload = load_dataset(str(sample_path))
        config = CalibrationConfig(
            **{
                **config.__dict__,
                "loss": args.loss,
                "min_motion_rotation_deg": min_motion_rotation_deg,
                "planar_motion_policy": planar_motion_policy,
            }
        )
        calibration_output_dir = trial_output_dir / "calibration"
        calibration_output_dir.mkdir(parents=True, exist_ok=True)
        calibration_result = run_calibration(
            dataset, config=config, output_dir=str(calibration_output_dir)
        )
        manifest = write_outputs(
            output_dir=calibration_output_dir,
            dataset=dataset,
            initial_transform=calibration_result["initial_transform"],
            final_transform=calibration_result["final_transform"],
            metrics_output=calibration_result["metrics"],
            algorithm_report={
                "input_file": str(sample_path.resolve()),
                "config": config.__dict__,
                "dataset_metadata": dataset.metadata,
                "raw_metadata": raw_payload.get("metadata", {}),
                "conversion_summary": diagnostics["summary"],
                "stages": calibration_result["stages"],
            },
            evaluation_report=calibration_result["evaluation"],
        )
        result["status"] = "ok"
        result["coarse_metrics"] = calibration_result["metrics"].get(
            "coarse_metrics", {}
        )
        result["vehicle_motion_assessment"] = calibration_result["metrics"].get(
            "vehicle_motion_assessment", {}
        )
        result["final_summary"] = calibration_result["metrics"].get("summary", {})
        result["manifest"] = manifest
        return result
    except Exception as exc:
        result["status"] = "calibration_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep lidar2imu record-conversion/calibration parameters."
    )
    parser.add_argument(
        "--record-path", required=True, help="Record file or directory."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will contain one subdirectory per tuning trial.",
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
        "--parent-frame", default="imu", help="Parent frame for output extrinsics."
    )
    parser.add_argument(
        "--child-frame",
        default=None,
        help="Optional explicit child frame. Defaults to LiDAR topic frame_id.",
    )
    parser.add_argument(
        "--initial-transform",
        default=None,
        help="Optional extrinsics YAML/JSON used as the initial lidar->parent transform.",
    )
    parser.add_argument(
        "--identity-initial-transform",
        action="store_true",
        help="Use identity as the initial lidar->parent transform.",
    )
    parser.add_argument(
        "--gravity-sources",
        default="pose",
        help="Comma-separated gravity sources to sweep, e.g. pose,imu.",
    )
    parser.add_argument(
        "--motion-frame-strides",
        default="1,2,5",
        help="Comma-separated motion frame strides to sweep.",
    )
    parser.add_argument(
        "--min-registration-fitness-values",
        default="0.35,0.45,0.55",
        help="Comma-separated registration fitness thresholds to sweep.",
    )
    parser.add_argument(
        "--min-motion-rotation-values",
        default="1.0",
        help="Comma-separated minimum motion rotation thresholds to sweep.",
    )
    parser.add_argument(
        "--planar-motion-policies",
        default="auto",
        help="Comma-separated planar motion policies to sweep, e.g. auto,free.",
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
        "--imu-gravity-window-ms",
        type=float,
        default=100.0,
        help="Window size when averaging IMU gravity samples.",
    )
    parser.add_argument(
        "--max-ground-samples", type=int, default=12, help="Maximum ground samples."
    )
    parser.add_argument(
        "--max-motion-samples", type=int, default=8, help="Maximum motion samples."
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
        "--loss",
        default="huber",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Robust loss for calibration.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run full lidar2imu calibration for each trial.",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    gravity_sources = _parse_csv_strings(args.gravity_sources)
    motion_frame_strides = _parse_csv_ints(args.motion_frame_strides)
    min_registration_fitness_values = _parse_csv_floats(
        args.min_registration_fitness_values
    )
    min_motion_rotation_values = _parse_csv_floats(args.min_motion_rotation_values)
    planar_motion_policies = _parse_csv_strings(args.planar_motion_policies)

    results = []
    for (
        gravity_source,
        motion_frame_stride,
        min_registration_fitness,
        min_motion_rotation_deg,
        planar_motion_policy,
    ) in itertools.product(
        gravity_sources,
        motion_frame_strides,
        min_registration_fitness_values,
        min_motion_rotation_values,
        planar_motion_policies,
    ):
        trial_result = _run_trial(
            args=args,
            gravity_source=gravity_source,
            motion_frame_stride=motion_frame_stride,
            min_registration_fitness=min_registration_fitness,
            min_motion_rotation_deg=min_motion_rotation_deg,
            planar_motion_policy=planar_motion_policy,
        )
        results.append(trial_result)
        print(
            f"[{trial_result['status']}] {trial_result['trial_name']}",
            flush=True,
        )

    ranked_results = sorted(results, key=_result_sort_key)
    summary_payload = {
        "record_path": args.record_path,
        "output_dir": str(Path(args.output_dir).resolve()),
        "initial_transform": args.initial_transform,
        "identity_initial_transform": bool(args.identity_initial_transform),
        "sweep": {
            "gravity_sources": gravity_sources,
            "motion_frame_strides": motion_frame_strides,
            "min_registration_fitness_values": min_registration_fitness_values,
            "min_motion_rotation_values": min_motion_rotation_values,
            "planar_motion_policies": planar_motion_policies,
        },
        "results": ranked_results,
        "top_trial": ranked_results[0] if ranked_results else None,
    }
    summary_path = Path(args.output_dir) / "tuning_summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(summary_payload, sort_keys=False), encoding="utf-8"
    )

    csv_rows = [_flatten_result(item) for item in ranked_results]
    csv_path = Path(args.output_dir) / "tuning_summary.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"Saved tuning summary to {summary_path}")
    if csv_rows:
        print(f"Saved tuning CSV to {csv_path}")
        print("Top trial:")
        print(yaml.safe_dump(ranked_results[0], sort_keys=False))


if __name__ == "__main__":
    main()
