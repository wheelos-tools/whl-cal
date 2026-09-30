#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=Path,
    default=Path(os.environ.get("GRIL_ALIGNED_STATES", "aligned_accel_states.txt")),
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(os.environ.get("GRIL_VALIDATION_OUTPUT", ".")),
)
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)
GRAVITY = np.array([0.0, 0.0, -9.81])


def skew_batch(v):
    result = np.zeros((len(v), 3, 3))
    result[:, 0, 1] = -v[:, 2]
    result[:, 0, 2] = v[:, 1]
    result[:, 1, 0] = v[:, 2]
    result[:, 1, 2] = -v[:, 0]
    result[:, 2, 0] = -v[:, 1]
    result[:, 2, 1] = v[:, 0]
    return result


def shifted_acceleration(offset_s):
    return np.column_stack(
        [
            np.interp(timestamps + offset_s, timestamps, imu_acc[:, axis])
            for axis in range(3)
        ]
    )


def design_target(indices, yaw_deg, shifted_imu):
    r_li = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    r_il = r_li.T
    target = (
        (r_il @ shifted_imu[indices].T).T
        + np.einsum(
            "nji,njk,k->ni",
            r_world_lidar[indices],
            r_world_ground[indices],
            GRAVITY,
        )
        - np.einsum("nji,nj->ni", r_world_lidar[indices], lidar_acc[indices])
    )
    design = np.concatenate(
        [
            jacobian[indices, :, :2],
            np.broadcast_to(np.eye(3), (len(indices), 3, 3)),
        ],
        axis=2,
    )
    return design.reshape(-1, 5), target.reshape(-1), r_li


def fit_evaluate(train, holdout, yaw_deg, shifted_imu):
    design_train, target_train, r_li = design_target(train, yaw_deg, shifted_imu)
    parameters, *_ = np.linalg.lstsq(design_train, target_train, rcond=None)
    design_holdout, target_holdout, _ = design_target(holdout, yaw_deg, shifted_imu)
    residual = design_holdout @ parameters - target_holdout
    t_il = np.array([parameters[0], parameters[1], 0.0])
    t_li = -r_li @ t_il
    return float(np.sqrt(np.mean(residual**2))), t_li[:2]


data = np.loadtxt(INPUT)
timestamps = data[:, 0]
r_world_lidar = data[:, 1:10].reshape(-1, 3, 3)
imu_acc = data[:, 10:13]
lidar_acc = data[:, 13:16]
omega = data[:, 16:19]
alpha = data[:, 19:22]
r_world_ground = data[:, 22:31].reshape(-1, 3, 3)
jacobian = skew_batch(omega) @ skew_batch(omega) + skew_batch(alpha)

valid = np.flatnonzero(
    (timestamps >= timestamps[0] + 0.51) & (timestamps <= timestamps[-1] - 0.51)
)
folds = np.array_split(valid, 4)
yaws = np.arange(0.0, 180.1, 5.0)
offsets = np.arange(-0.50, 0.2001, 0.01)
shifted_by_offset = {float(offset): shifted_acceleration(offset) for offset in offsets}

grid = []
per_fold_best = []
for yaw in yaws:
    for offset in offsets:
        fold_results = []
        shifted = shifted_by_offset[float(offset)]
        for fold_number, holdout in enumerate(folds):
            train = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_number]
            )
            rmse, translation = fit_evaluate(train, holdout, float(yaw), shifted)
            fold_results.append(
                {
                    "fold": fold_number,
                    "rmse": rmse,
                    "translation": translation,
                }
            )
        grid.append(
            {
                "yaw_deg": float(yaw),
                "offset_s": float(offset),
                "mean_rmse": float(np.mean([item["rmse"] for item in fold_results])),
                "fold_results": fold_results,
            }
        )

for fold_number in range(4):
    best = min(grid, key=lambda item: item["fold_results"][fold_number]["rmse"])
    fold_result = best["fold_results"][fold_number]
    per_fold_best.append(
        {
            "fold": fold_number,
            "yaw_deg": best["yaw_deg"],
            "additional_offset_s": best["offset_s"],
            "holdout_rmse_m_s2": fold_result["rmse"],
            "translation_xy_lidar_to_imu_m": fold_result["translation"].tolist(),
        }
    )

best = min(grid, key=lambda item: item["mean_rmse"])
configured = min(
    grid,
    key=lambda item: abs(item["yaw_deg"] - 90.0) + 1000.0 * abs(item["offset_s"]),
)
near_best_0_1_percent = [
    item for item in grid if item["mean_rmse"] <= best["mean_rmse"] * 1.001
]
near_best_1_percent = [
    item for item in grid if item["mean_rmse"] <= best["mean_rmse"] * 1.01
]
best_translations = np.array([item["translation"] for item in best["fold_results"]])
pairwise = np.linalg.norm(
    best_translations[:, None, :] - best_translations[None, :, :], axis=2
)
report = {
    "verdict": "no_unique_generalizing_solution",
    "reference_transform_used_for_selection": False,
    "selection_metric": "mean RMSE over four contiguous held-out quarters",
    "grid": {
        "yaw_range_deg": [float(yaws[0]), float(yaws[-1])],
        "yaw_step_deg": 5.0,
        "additional_time_offset_range_s": [
            float(offsets[0]),
            float(offsets[-1]),
        ],
        "time_step_s": 0.01,
    },
    "global_cross_validation_best": {
        "yaw_deg": best["yaw_deg"],
        "additional_offset_s": best["offset_s"],
        "mean_holdout_rmse_m_s2": best["mean_rmse"],
        "fold_translation_xy_lidar_to_imu_m": best_translations.tolist(),
        "maximum_fold_translation_difference_m": float(np.max(pairwise)),
        "rmse_improvement_over_yaw_90_current_time_percent": float(
            100.0
            * (configured["mean_rmse"] - best["mean_rmse"])
            / configured["mean_rmse"]
        ),
    },
    "yaw_90_current_time_mean_holdout_rmse_m_s2": configured["mean_rmse"],
    "near_optimal_surface": {
        "within_0_1_percent": {
            "grid_point_count": len(near_best_0_1_percent),
            "yaw_range_deg": [
                min(item["yaw_deg"] for item in near_best_0_1_percent),
                max(item["yaw_deg"] for item in near_best_0_1_percent),
            ],
            "additional_offset_range_s": [
                min(item["offset_s"] for item in near_best_0_1_percent),
                max(item["offset_s"] for item in near_best_0_1_percent),
            ],
        },
        "within_1_percent": {
            "grid_point_count": len(near_best_1_percent),
            "yaw_range_deg": [
                min(item["yaw_deg"] for item in near_best_1_percent),
                max(item["yaw_deg"] for item in near_best_1_percent),
            ],
            "additional_offset_range_s": [
                min(item["offset_s"] for item in near_best_1_percent),
                max(item["offset_s"] for item in near_best_1_percent),
            ],
        },
    },
    "per_holdout_best": per_fold_best,
    "per_holdout_best_yaw_range_deg": float(
        np.ptp([item["yaw_deg"] for item in per_fold_best])
    ),
    "per_holdout_best_time_range_s": float(
        np.ptp([item["additional_offset_s"] for item in per_fold_best])
    ),
    "reason": (
        "If yaw, timing, and lever arm were supported by one rigid-body model, "
        "independent contiguous holdouts would prefer compatible parameters. "
        "The grid deliberately does not use the configured/default transform."
    ),
}
with (OUTPUT / "gril_yaw_time_holdout_grid.yaml").open("w") as stream:
    yaml.safe_dump(report, stream, sort_keys=False)
