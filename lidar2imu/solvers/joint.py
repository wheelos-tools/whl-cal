from __future__ import annotations

# isort: off
import numpy as np

from lidar2imu.algorithms import refine_joint_solution
from lidar2imu.algorithms import yaw_roll_pitch_from_matrix
from lidar2imu.models import CalibrationConfig, CalibrationDataset

# isort: on


def run_joint_polish(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    initial_transform: np.ndarray,
    ground_stage: dict,
    planar_stage: dict,
) -> dict:
    initial_yaw, _, _ = yaw_roll_pitch_from_matrix(
        np.asarray(initial_transform, dtype=float)
    )
    solver_policy = planar_stage["solver_policy"]
    motion_samples = planar_stage.get("motion_samples", dataset.motion_samples)
    nhc_prior = planar_stage.get("nhc_prior")
    joint_yaw = (
        initial_yaw
        if solver_policy["applied"] == "freeze_xyyaw"
        else planar_stage["yaw"]
    )
    joint_initial = np.array(
        [
            joint_yaw,
            ground_stage["orientation_stage"]["parameters"]["roll_rad"],
            ground_stage["orientation_stage"]["parameters"]["pitch_rad"],
            planar_stage["translation"][0],
            planar_stage["translation"][1],
            planar_stage["translation"][2],
        ],
        dtype=float,
    )
    final_transform, joint_stage = refine_joint_solution(
        dataset.ground_samples,
        motion_samples,
        initial_params=joint_initial,
        config=config,
        locked_mask=(
            solver_policy["applied"] == "freeze_xyyaw",
            False,
            False,
            solver_policy["applied"] == "freeze_xyyaw",
            solver_policy["applied"] == "freeze_xyyaw",
            False,
        ),
        nhc_prior=nhc_prior,
    )
    return {
        "final_transform": np.asarray(final_transform, dtype=float),
        "stage": joint_stage,
        "summary": {
            "stage_family": "light_joint_polish",
            "applied_policy": solver_policy["applied"],
            "motion_sample_count": int(len(motion_samples)),
            "locked_components": list(joint_stage["parameters"]["locked_components"]),
            "imu_preintegration_enabled": bool(
                (joint_stage.get("imu_preintegration") or {}).get("enabled", False)
            ),
            "nhc_prior_enabled": bool(nhc_prior and nhc_prior.get("enabled", False)),
        },
    }
