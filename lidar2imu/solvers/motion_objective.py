from __future__ import annotations

# isort: off
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import solve_translation_from_motion
from lidar2imu.algorithms import solve_yaw_from_motion
from lidar2imu.algorithms import transform_from_components
from lidar2imu.algorithms import yaw_roll_pitch_from_matrix
from lidar2imu.models import CalibrationConfig, CalibrationDataset, MotionSample
from lidar2imu.solvers.joint import run_joint_polish
from lidar2imu.solvers.nhc_prior import resolve_nhc_prior
from lidar2imu.solvers.motion_screening import resolve_planar_motion_policy

# isort: on


def window_id(sample: MotionSample) -> int | None:
    raw_window_id = sample.metadata.get("window_id")
    return None if raw_window_id is None else int(raw_window_id)


def motion_sample_window_ids(motion_samples: list[MotionSample]) -> list[int]:
    return sorted(
        {
            int(raw_window_id)
            for raw_window_id in (window_id(sample) for sample in motion_samples)
            if raw_window_id is not None
        }
    )


def motion_sample_frame_strides(motion_samples: list[MotionSample]) -> list[int]:
    return sorted(
        {
            int(sample.metadata["frame_stride"])
            for sample in motion_samples
            if sample.metadata.get("frame_stride") is not None
        }
    )


def candidate_dataset(
    dataset: CalibrationDataset, motion_samples: list[MotionSample]
) -> CalibrationDataset:
    return CalibrationDataset(
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        ground_samples=list(dataset.ground_samples),
        motion_samples=list(motion_samples),
        initial_transform=np.asarray(dataset.initial_transform, dtype=float),
        extraction_transform=(
            None
            if dataset.extraction_transform is None
            else np.asarray(dataset.extraction_transform, dtype=float)
        ),
        reference_transform=(
            None
            if dataset.reference_transform is None
            else np.asarray(dataset.reference_transform, dtype=float)
        ),
        motion_candidate_pool=list(dataset.motion_candidate_pool),
        metadata=dict(dataset.metadata),
    )


def run_planar_motion_stage(
    motion_samples: list[MotionSample],
    config: CalibrationConfig,
    ground_stage: dict[str, Any],
    initial_transform: np.ndarray,
    *,
    stage_family: str,
    screening_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    initial_transform = np.asarray(initial_transform, dtype=float)
    initial_yaw, _, _ = yaw_roll_pitch_from_matrix(initial_transform)
    initial_translation = np.asarray(initial_transform[:3, 3], dtype=float)
    fixed_roll = float(ground_stage["orientation_stage"]["parameters"]["roll_rad"])
    fixed_pitch = float(ground_stage["orientation_stage"]["parameters"]["pitch_rad"])

    if len(motion_samples) < int(config.min_motion_samples):
        if config.planar_motion_policy == "free":
            raise ValueError(
                "Free planar solve requested, but screened motion samples are "
                "insufficient."
            )
        motion_rotation_stage = {
            "success": False,
            "skipped": True,
            "reason": "insufficient_motion_samples",
            "used_samples": int(len(motion_samples)),
            "parameters": {
                "yaw_rad": float(initial_yaw),
                "yaw_deg": float(np.degrees(initial_yaw)),
            },
            "observability": {
                "degenerate": True,
                "reasons": ["insufficient_motion_samples"],
                "cost_scan": {},
            },
        }
        yaw = float(initial_yaw)
    else:
        yaw, motion_rotation_stage = solve_yaw_from_motion(
            motion_samples,
            fixed_roll=fixed_roll,
            fixed_pitch=fixed_pitch,
            initial_yaw=initial_yaw,
            config=config,
        )
    if screening_summary is not None:
        motion_rotation_stage["screening"] = screening_summary

    motion_rotation = transform_from_components(
        yaw=yaw,
        roll=fixed_roll,
        pitch=fixed_pitch,
        translation=np.zeros(3, dtype=float),
    )[:3, :3]
    solver_policy = resolve_planar_motion_policy(
        motion_samples,
        config,
        motion_rotation_stage,
        screening_summary=screening_summary,
    )
    translation_locked_axes = (
        (True, True, False)
        if solver_policy["applied"] == "freeze_xyyaw"
        else (False, False, False)
    )
    nhc_prior = resolve_nhc_prior(
        config,
        solver_policy=solver_policy,
        screening_summary=screening_summary,
    )
    motion_translation_initial = np.asarray(
        ground_stage["translation"], dtype=float
    ).copy()
    if solver_policy["applied"] == "freeze_xyyaw":
        motion_translation_initial[:2] = initial_translation[:2]
    motion_translation, motion_translation_stage = solve_translation_from_motion(
        motion_samples,
        rotation=motion_rotation,
        initial_translation=motion_translation_initial,
        locked_axes=translation_locked_axes,
        nhc_prior=nhc_prior,
        config=config,
    )
    motion_translation = np.asarray(motion_translation, dtype=float).reshape(3)
    if screening_summary is not None:
        motion_translation_stage["screening"] = screening_summary

    return {
        "yaw": float(
            initial_yaw if solver_policy["applied"] == "freeze_xyyaw" else yaw
        ),
        "rotation": motion_rotation,
        "translation": motion_translation,
        "motion_samples": list(motion_samples),
        "rotation_stage": motion_rotation_stage,
        "translation_stage": motion_translation_stage,
        "solver_policy": solver_policy,
        "nhc_prior": nhc_prior,
        "summary": {
            "stage_family": stage_family,
            "input_sample_count": int(len(motion_samples)),
            "used_window_ids": motion_sample_window_ids(motion_samples),
            "used_frame_strides": motion_sample_frame_strides(motion_samples),
            "screening": screening_summary,
            "policy": solver_policy,
            "nhc_prior": nhc_prior,
            "parameters": {
                "yaw_rad": float(yaw),
                "yaw_deg": float(np.degrees(yaw)),
                "translation": [float(value) for value in motion_translation.tolist()],
            },
        },
    }


def evaluate_motion_subset_objective(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    ground_stage: dict[str, Any],
    initial_transform: np.ndarray,
    motion_samples: list[MotionSample],
) -> tuple[float, dict[str, float]]:
    candidate_planar = run_planar_motion_stage(
        motion_samples,
        config,
        ground_stage,
        initial_transform,
        stage_family="gril_planar_local_search",
        screening_summary=None,
    )
    candidate_joint = run_joint_polish(
        candidate_dataset(dataset, motion_samples),
        config,
        initial_transform,
        ground_stage,
        candidate_planar,
    )
    final_transform = np.asarray(candidate_joint["final_transform"], dtype=float)
    rotation = final_transform[:3, :3]
    translation = final_transform[:3, 3]
    rotation_residuals_deg = []
    translation_residuals_m = []
    for sample in motion_samples:
        rotation_error = (
            sample.imu_delta_rotation
            @ rotation
            @ sample.lidar_delta_rotation.T
            @ rotation.T
        )
        rotation_residuals_deg.append(
            float(np.degrees(np.linalg.norm(R.from_matrix(rotation_error).as_rotvec())))
        )
        translation_error = (
            sample.imu_delta_rotation - np.eye(3, dtype=float)
        ) @ translation - (
            rotation @ sample.lidar_delta_translation - sample.imu_delta_translation
        )
        translation_residuals_m.append(float(np.linalg.norm(translation_error)))
    rotation_p95_deg = float(np.percentile(rotation_residuals_deg, 95))
    translation_p95_m = float(np.percentile(translation_residuals_m, 95))
    objective = float(rotation_p95_deg + (5.0 * translation_p95_m))
    return objective, {
        "rotation_p95_deg": rotation_p95_deg,
        "translation_p95_m": translation_p95_m,
    }
