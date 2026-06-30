from __future__ import annotations

# isort: off
import numpy as np

from lidar2imu.models import CalibrationConfig, CalibrationDataset
from lidar2imu.solvers import motion_information_weighting as _miw
from lidar2imu.solvers.ground import run_ground_stage
from lidar2imu.solvers.joint import run_joint_polish
from lidar2imu.solvers.planar import run_baseline_planar_stage, run_gril_planar_stage

# isort: on


def run_calibration_once(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    initial_transform: np.ndarray,
) -> dict:
    initial_transform = np.asarray(initial_transform, dtype=float)
    solver_family = str(config.solver_family or "baseline")
    solver_dataset = dataset
    solver_config = config
    information_weighting_summary = None

    if solver_family in {"gril_prob", "gril_prob_nhc"}:
        solver_dataset, information_weighting_summary = (
            _miw.build_information_weighted_dataset(
                dataset,
                config,
            )
        )
    if solver_family == "gril_prob_nhc" and config.nhc_prior_mode == "off":
        solver_config = CalibrationConfig(
            **{**config.__dict__, "nhc_prior_mode": "auto"}
        )

    ground_stage = run_ground_stage(solver_dataset, solver_config, initial_transform)
    if solver_family == "baseline":
        planar_stage = run_baseline_planar_stage(
            solver_dataset, solver_config, ground_stage, initial_transform
        )
    elif solver_family == "gril_staged":
        planar_stage = run_gril_planar_stage(
            solver_dataset, solver_config, ground_stage, initial_transform
        )
    elif solver_family == "gril_prob":
        planar_stage = run_gril_planar_stage(
            solver_dataset,
            solver_config,
            ground_stage,
            initial_transform,
            stage_family_name="gril_prob_planar",
            seed_stage_family_name="gril_prob_planar_seed",
        )
    elif solver_family == "gril_prob_nhc":
        planar_stage = run_gril_planar_stage(
            solver_dataset,
            solver_config,
            ground_stage,
            initial_transform,
            stage_family_name="gril_prob_nhc_planar",
            seed_stage_family_name="gril_prob_nhc_planar_seed",
        )
    else:
        raise ValueError(f"Unsupported solver_family: {solver_family}")
    if information_weighting_summary is not None:
        planar_stage["summary"]["information_weighting"] = information_weighting_summary

    joint_polish = run_joint_polish(
        solver_dataset,
        solver_config,
        initial_transform,
        ground_stage,
        planar_stage,
    )
    stages = {
        "ground_orientation": ground_stage["orientation_stage"],
        "ground_translation": ground_stage["translation_stage"],
        "motion_rotation": planar_stage["rotation_stage"],
        "motion_translation": planar_stage["translation_stage"],
        "solver_policy": planar_stage["solver_policy"],
        "joint": joint_polish["stage"],
        "solver_family": {
            "name": solver_family,
            "nhc_prior_mode": str(solver_config.nhc_prior_mode),
            "information_weighting": information_weighting_summary,
        },
        "ground_stage": ground_stage["summary"],
        "planar_stage": planar_stage["summary"],
        "joint_polish": joint_polish["summary"],
    }
    return {
        "initial_transform": initial_transform,
        "final_transform": joint_polish["final_transform"],
        "stages": stages,
    }
