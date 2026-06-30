from __future__ import annotations

import numpy as np

from lidar2imu.algorithms import (
    solve_ground_translation,
    solve_roll_pitch_from_ground,
    yaw_roll_pitch_from_matrix,
)
from lidar2imu.models import CalibrationConfig, CalibrationDataset


def run_ground_stage(
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    initial_transform: np.ndarray,
) -> dict:
    initial_transform = np.asarray(initial_transform, dtype=float)
    _, initial_roll, initial_pitch = yaw_roll_pitch_from_matrix(initial_transform)
    initial_translation = np.asarray(initial_transform[:3, 3], dtype=float)
    height_prior_count = int(
        sum(
            1
            for sample in dataset.ground_samples
            if sample.imu_ground_height is not None
        )
    )

    ground_rotation, orientation_stage = solve_roll_pitch_from_ground(
        dataset.ground_samples,
        initial_roll=initial_roll,
        initial_pitch=initial_pitch,
        config=config,
    )
    ground_translation, translation_stage = solve_ground_translation(
        dataset.ground_samples,
        rotation=ground_rotation,
        initial_translation=initial_translation,
    )
    ground_translation = np.asarray(ground_translation, dtype=float).reshape(3)

    return {
        "rotation": np.asarray(ground_rotation, dtype=float),
        "translation": ground_translation,
        "orientation_stage": orientation_stage,
        "translation_stage": translation_stage,
        "summary": {
            "stage_family": "gril_ground",
            "sample_count": int(len(dataset.ground_samples)),
            "height_prior_count": height_prior_count,
            "gates": {
                "orientation_ready": True,
                "translation_ready": not bool(translation_stage.get("skipped", False)),
                "skip_reasons": (
                    [str(translation_stage.get("reason"))]
                    if translation_stage.get("reason")
                    else []
                ),
            },
            "parameters": {
                "roll_rad": float(orientation_stage["parameters"]["roll_rad"]),
                "pitch_rad": float(orientation_stage["parameters"]["pitch_rad"]),
                "roll_deg": float(orientation_stage["parameters"]["roll_deg"]),
                "pitch_deg": float(orientation_stage["parameters"]["pitch_deg"]),
                "translation": [float(value) for value in ground_translation.tolist()],
                "z_m": float(ground_translation[2]),
            },
        },
    }
