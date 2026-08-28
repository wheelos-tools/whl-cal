#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares
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
parser.add_argument(
    "--rotation-xyz-deg",
    default=os.environ.get("GRIL_ROTATION_XYZ_DEG", "0.447966,0.021440,86.084444"),
)
parser.add_argument(
    "--time-offset-s",
    type=float,
    default=float(os.environ.get("GRIL_TIME_OFFSET_S", "-0.01660758")),
)
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)
GRAVITY = np.array([0.0, 0.0, -9.81])
R_LI = Rotation.from_euler(
    "xyz",
    [float(value) for value in args.rotation_xyz_deg.split(",")],
    degrees=True,
).as_matrix()
R_IL = R_LI.T
TIME_OFFSET_S = args.time_offset_s


def skew_batch(v):
    result = np.zeros((len(v), 3, 3))
    result[:, 0, 1] = -v[:, 2]
    result[:, 0, 2] = v[:, 1]
    result[:, 1, 0] = v[:, 2]
    result[:, 1, 2] = -v[:, 0]
    result[:, 2, 0] = -v[:, 1]
    result[:, 2, 1] = v[:, 0]
    return result


def design_and_target(indices):
    local_target = (
        (R_IL @ imu_acc[indices].T).T
        + np.einsum(
            "nji,njk,k->ni",
            r_world_lidar[indices],
            r_world_ground[indices],
            GRAVITY,
        )
        - np.einsum("nji,nj->ni", r_world_lidar[indices], lidar_acc[indices])
    )
    design = np.concatenate(
        [jacobian[indices, :, :2], np.broadcast_to(np.eye(3), (len(indices), 3, 3))],
        axis=2,
    )
    return design.reshape(-1, 5), local_target.reshape(-1)


def solve(indices, loss="linear"):
    design, target = design_and_target(indices)
    if loss == "linear":
        parameters, *_ = np.linalg.lstsq(design, target, rcond=None)
    else:
        result = least_squares(
            lambda value: design @ value - target,
            np.zeros(5),
            loss=loss,
            f_scale=0.5,
            max_nfev=2000,
        )
        parameters = result.x
    return parameters


def evaluate(parameters, indices):
    design, target = design_and_target(indices)
    residual = design @ parameters - target
    return float(np.sqrt(np.mean(residual**2)))


def observability(indices):
    j_xy = jacobian[indices, :, :2]
    marginalized = j_xy - np.mean(j_xy, axis=0, keepdims=True)
    singular_values = np.linalg.svd(marginalized.reshape(-1, 2), compute_uv=False)
    return singular_values


def lidar_to_imu_translation(parameters):
    t_il = np.array([parameters[0], parameters[1], 0.0])
    return (-R_LI @ t_il)[:2]


data = np.loadtxt(INPUT)
timestamps = data[:, 0]
r_world_lidar = data[:, 1:10].reshape(-1, 3, 3)
imu_acc = data[:, 10:13]
imu_acc = np.column_stack(
    [
        np.interp(timestamps + TIME_OFFSET_S, timestamps, imu_acc[:, axis])
        for axis in range(3)
    ]
)
lidar_acc = data[:, 13:16]
omega = data[:, 16:19]
alpha = data[:, 19:22]
r_world_ground = data[:, 22:31].reshape(-1, 3, 3)
jacobian = skew_batch(omega) @ skew_batch(omega) + skew_batch(alpha)
all_indices = np.arange(len(data))

null_parameters = np.zeros(5)
_, full_target = design_and_target(all_indices)
null_parameters[2:5] = full_target.reshape(-1, 3).mean(axis=0)
full_fit = solve(all_indices)
full_robust_fit = solve(all_indices, loss="cauchy")
full_null_rmse = evaluate(null_parameters, all_indices)
full_fit_rmse = evaluate(full_fit, all_indices)

report = {
    "verdict": "not_self_consistent",
    "reference_transform_used_for_verdict": False,
    "assumption": (
        "A rigid LiDAR-IMU lever arm is constant. With fixed rotation and timing, "
        "independent sufficiently excited time blocks must estimate the same "
        "horizontal lever arm and predict held-out acceleration."
    ),
    "sample_count": int(len(data)),
    "duration_s": float(timestamps[-1] - timestamps[0]),
    "full_fit": {
        "translation_xy_lidar_to_imu_m": lidar_to_imu_translation(full_fit).tolist(),
        "robust_translation_xy_lidar_to_imu_m": lidar_to_imu_translation(
            full_robust_fit
        ).tolist(),
        "bias_lidar_m_s2": full_fit[2:5].tolist(),
        "null_model_rmse_m_s2": full_null_rmse,
        "lever_arm_model_rmse_m_s2": full_fit_rmse,
        "rmse_improvement_percent": float(
            100.0 * (full_null_rmse - full_fit_rmse) / full_null_rmse
        ),
        "bias_marginalized_singular_values": observability(all_indices).tolist(),
        "condition_number": float(
            observability(all_indices)[0] / observability(all_indices)[-1]
        ),
    },
    "contiguous_cross_validation": [],
    "excitation_quantiles": [],
    "sliding_windows": [],
}

folds = np.array_split(all_indices, 4)
for fold_number, holdout in enumerate(folds):
    train = np.concatenate(
        [fold for index, fold in enumerate(folds) if index != fold_number]
    )
    fit = solve(train)
    null = np.zeros(5)
    _, train_target = design_and_target(train)
    null[2:5] = train_target.reshape(-1, 3).mean(axis=0)
    report["contiguous_cross_validation"].append(
        {
            "holdout_quarter": int(fold_number),
            "translation_xy_lidar_to_imu_m": lidar_to_imu_translation(fit).tolist(),
            "holdout_null_rmse_m_s2": evaluate(null, holdout),
            "holdout_lever_arm_rmse_m_s2": evaluate(fit, holdout),
            "holdout_improvement_percent": float(
                100.0
                * (evaluate(null, holdout) - evaluate(fit, holdout))
                / evaluate(null, holdout)
            ),
            "train_observability_singular_values": observability(train).tolist(),
        }
    )

excitation = np.linalg.norm(jacobian[:, :, :2], axis=(1, 2))
for quantile in (0.0, 0.25, 0.5, 0.75):
    threshold = float(np.quantile(excitation, quantile))
    indices = np.flatnonzero(excitation >= threshold)
    fit = solve(indices)
    report["excitation_quantiles"].append(
        {
            "retained_top_fraction": float(1.0 - quantile),
            "sample_count": int(len(indices)),
            "minimum_excitation": threshold,
            "translation_xy_lidar_to_imu_m": lidar_to_imu_translation(fit).tolist(),
            "observability_singular_values": observability(indices).tolist(),
            "condition_number": float(
                observability(indices)[0] / observability(indices)[-1]
            ),
        }
    )

window_duration_s = 20.0
window_step_s = 10.0
start_time = timestamps[0]
while start_time + window_duration_s <= timestamps[-1]:
    indices = np.flatnonzero(
        (timestamps >= start_time) & (timestamps < start_time + window_duration_s)
    )
    singular_values = observability(indices)
    fit = solve(indices)
    report["sliding_windows"].append(
        {
            "start_s": float(start_time - timestamps[0]),
            "sample_count": int(len(indices)),
            "translation_xy_lidar_to_imu_m": lidar_to_imu_translation(fit).tolist(),
            "observability_singular_values": singular_values.tolist(),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "fit_rmse_m_s2": evaluate(fit, indices),
        }
    )
    start_time += window_step_s

window_translations = np.array(
    [item["translation_xy_lidar_to_imu_m"] for item in report["sliding_windows"]]
)
pairwise_distances = np.linalg.norm(
    window_translations[:, None, :] - window_translations[None, :, :], axis=2
)
report["sliding_window_summary"] = {
    "translation_xy_range_m": np.ptp(window_translations, axis=0).tolist(),
    "maximum_pairwise_difference_m": float(np.max(pairwise_distances)),
    "median_translation_xy_lidar_to_imu_m": np.median(
        window_translations, axis=0
    ).tolist(),
}
report["acceptance_rule"] = {
    "reference_free": True,
    "maximum_window_pairwise_difference_m": 0.10,
    "all_holdout_improvements_must_be_positive": True,
    "full_fit_rmse_improvement_min_percent": 5.0,
}
rejection_reasons = []
if report["full_fit"]["rmse_improvement_percent"] < 5.0:
    rejection_reasons.append(
        "The lever-arm term explains too little acceleration variance."
    )
if report["sliding_window_summary"]["maximum_pairwise_difference_m"] > 0.10:
    rejection_reasons.append(
        "Independent time windows estimate incompatible rigid lever arms."
    )
if any(
    item["holdout_improvement_percent"] <= 0.0
    for item in report["contiguous_cross_validation"]
):
    rejection_reasons.append(
        "Some contiguous holdouts are predicted worse after fitting the lever arm."
    )
report["rejection_reasons"] = rejection_reasons
report["verdict"] = "accepted" if not rejection_reasons else "not_self_consistent"

with (OUTPUT / "gril_first_principles_validation.yaml").open("w") as stream:
    yaml.safe_dump(report, stream, sort_keys=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(
    window_translations[:, 0],
    window_translations[:, 1],
    c=np.arange(len(window_translations)),
    cmap="viridis",
)
for index, point in enumerate(window_translations):
    axes[0].annotate(str(index), point)
axes[0].set(
    xlabel="Tx LiDAR->IMU (m)",
    ylabel="Ty LiDAR->IMU (m)",
    title="20 s GRIL lever-arm estimates",
)
axes[0].axis("equal")

holdout_improvements = [
    item["holdout_improvement_percent"]
    for item in report["contiguous_cross_validation"]
]
axes[1].bar(np.arange(4), holdout_improvements)
axes[1].axhline(0.0, color="black", linewidth=1)
axes[1].set(
    xlabel="Held-out contiguous quarter",
    ylabel="RMSE improvement (%)",
    title="Cross-segment prediction vs no lever arm",
)
fig.tight_layout()
fig.savefig(OUTPUT / "gril_first_principles_validation.png", dpi=180)
fig.savefig(OUTPUT / "gril_first_principles_validation.svg")
