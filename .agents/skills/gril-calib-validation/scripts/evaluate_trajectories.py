#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from profile_gril_batch_timing import read_states
from scipy.spatial.transform import Rotation

from lidar2lidar.record_adapter import Record
from lidar2lidar.record_utils import message_timestamp_ns

parser = argparse.ArgumentParser()
parser.add_argument("--record-file", action="append", required=True)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--trajectory", type=Path)
source.add_argument("--batch-trace", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--pose-topic", default="/apollo/sensor/gnss/odometry")
args = parser.parse_args()

RECORDS = args.record_file
POSE_TOPIC = args.pose_topic
OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)

if args.batch_trace:
    lines = args.batch_trace.read_text().splitlines()
    if not lines or lines[0].strip() != "GRIL_BATCH_TRACE 1":
        raise ValueError("Batch trajectory requires GRIL_BATCH_TRACE 1")
    lidar_states = read_states(lines, "lidar")
    lidar_time = lidar_states[:, 0]
    lidar_xy = lidar_states[:, 10:12]
    lidar_yaw = np.unwrap(
        Rotation.from_matrix(lidar_states[:, 1:10].reshape(-1, 3, 3)).as_euler("xyz")[
            :, 2
        ]
    )
else:
    lidar = np.loadtxt(args.trajectory, comments="#")
    lidar_time = lidar[:, 0]
    lidar_xy = lidar[:, 1:3]

pose_samples = []
for record_path in RECORDS:
    with Record(record_path) as record:
        for topic, message, record_timestamp_ns in record.read_messages(
            topics=[POSE_TOPIC]
        ):
            timestamp_ns = message_timestamp_ns(
                topic, message, int(record_timestamp_ns)
            )
            pose = getattr(message, "pose", None)
            if pose is None:
                pose = getattr(message, "localization", None)
            if pose is None:
                raise RuntimeError(f"{POSE_TOPIC} has neither pose nor localization")
            if args.batch_trace and not hasattr(pose, "orientation"):
                raise RuntimeError(
                    f"{POSE_TOPIC} has no orientation for yaw comparison"
                )
            orientation = getattr(pose, "orientation", None)
            pose_samples.append(
                (
                    timestamp_ns * 1e-9,
                    float(pose.position.x),
                    float(pose.position.y),
                    *(
                        (
                            float(orientation.qx),
                            float(orientation.qy),
                            float(orientation.qz),
                            float(orientation.qw),
                        )
                        if args.batch_trace
                        else ()
                    ),
                )
            )

pose_samples = np.asarray(pose_samples)
if np.any(np.diff(pose_samples[:, 0]) <= 0):
    raise ValueError("GNSS/INS odometry timestamps must be strictly increasing")
if lidar_time[0] < pose_samples[0, 0] or lidar_time[-1] > pose_samples[-1, 0]:
    raise ValueError("GNSS/INS odometry does not cover the LiDAR trajectory")
ins_xy = np.column_stack(
    [
        np.interp(lidar_time, pose_samples[:, 0], pose_samples[:, axis])
        for axis in (1, 2)
    ]
)

lidar_center = lidar_xy.mean(axis=0)
ins_center = ins_xy.mean(axis=0)
covariance = (lidar_xy - lidar_center).T @ (ins_xy - ins_center)
u, _, vt = np.linalg.svd(covariance)
rotation = vt.T @ u.T
if np.linalg.det(rotation) < 0:
    vt[-1] *= -1
    rotation = vt.T @ u.T
translation = ins_center - rotation @ lidar_center
aligned_lidar_xy = (rotation @ lidar_xy.T).T + translation

ate = np.linalg.norm(aligned_lidar_xy - ins_xy, axis=1)
step_indices = np.searchsorted(lidar_time, lidar_time + 1.0)
valid = step_indices < len(lidar_time)
starts = np.flatnonzero(valid)
ends = step_indices[valid]
rpe = np.linalg.norm(
    (aligned_lidar_xy[ends] - aligned_lidar_xy[starts])
    - (ins_xy[ends] - ins_xy[starts]),
    axis=1,
)
lidar_steps = np.linalg.norm(np.diff(aligned_lidar_xy, axis=0), axis=1)
ins_steps = np.linalg.norm(np.diff(ins_xy, axis=0), axis=1)

metrics = {
    "alignment": "SE(2) rigid, no scale",
    "pose_count": int(len(lidar_time)),
    "duration_s": float(lidar_time[-1] - lidar_time[0]),
    "ate_xy_rmse_m": float(np.sqrt(np.mean(ate**2))),
    "ate_xy_p95_m": float(np.percentile(ate, 95)),
    "ate_xy_max_m": float(np.max(ate)),
    "rpe_xy_1s_rmse_m": float(np.sqrt(np.mean(rpe**2))),
    "rpe_xy_1s_p95_m": float(np.percentile(rpe, 95)),
    "lidar_path_length_m": float(np.sum(lidar_steps)),
    "ins_path_length_m": float(np.sum(ins_steps)),
    "lidar_max_step_m": float(np.max(lidar_steps)),
    "ins_max_step_m": float(np.max(ins_steps)),
}
if args.batch_trace:
    metrics["position_comparison_caveat"] = (
        "INS and LiDAR positions refer to different sensor origins; no lever-arm "
        "compensation was applied. Path length, ATE, and RPE are exploratory "
        "frontend diagnostics, not extrinsic accuracy metrics."
    )
    ins_yaw = np.interp(
        lidar_time,
        pose_samples[:, 0],
        np.unwrap(Rotation.from_quat(pose_samples[:, 3:7]).as_euler("xyz")[:, 2]),
    )
    yaw_difference = np.degrees((lidar_yaw - lidar_yaw[0]) - (ins_yaw - ins_yaw[0]))
    windows = []
    edges = np.linspace(lidar_time[0], lidar_time[-1], 5)
    for index in range(4):
        indices = np.flatnonzero(
            (lidar_time >= edges[index])
            & (
                lidar_time < edges[index + 1]
                if index < 3
                else lidar_time <= edges[index + 1]
            )
        )
        if len(indices) < 2:
            raise ValueError(f"Window {index} has fewer than two LiDAR states")
        first, last = indices[0], indices[-1]
        windows.append(
            {
                "window": index,
                "lidar_yaw_change_deg": float(
                    np.degrees(lidar_yaw[last] - lidar_yaw[first])
                ),
                "ins_yaw_change_deg": float(np.degrees(ins_yaw[last] - ins_yaw[first])),
                "yaw_drift_deg": float(yaw_difference[last] - yaw_difference[first]),
                "lidar_path_length_m": float(np.sum(lidar_steps[first:last])),
                "ins_path_length_m": float(np.sum(ins_steps[first:last])),
            }
        )
    metrics["yaw_comparison"] = {
        "reference_role": (
            "INS pose is coupled to input IMU; frontend motion check only, "
            "not extrinsic ground truth"
        ),
        "final_drift_deg": float(yaw_difference[-1]),
        "p95_absolute_drift_deg": float(np.percentile(abs(yaw_difference), 95)),
        "windows": windows,
    }
    fig_yaw, ax_yaw = plt.subplots(figsize=(9, 4))
    ax_yaw.plot(
        lidar_time - lidar_time[0], yaw_difference, label="LiDAR - INS yaw change"
    )
    ax_yaw.set_xlabel("Time from first LiDAR state [s]")
    ax_yaw.set_ylabel("Relative yaw drift [deg]")
    ax_yaw.grid(True, alpha=0.3)
    ax_yaw.legend()
    fig_yaw.tight_layout()
    fig_yaw.savefig(OUTPUT / "frontend_yaw_drift.png", dpi=180)
    plt.close(fig_yaw)
with (OUTPUT / "frontend_trajectory.yaml").open("w") as stream:
    yaml.safe_dump(metrics, stream, sort_keys=False)

fig, axis = plt.subplots(figsize=(8, 7))
axis.plot(ins_xy[:, 0] - ins_xy[0, 0], ins_xy[:, 1] - ins_xy[0, 1], label="GNSS/INS")
axis.plot(
    aligned_lidar_xy[:, 0] - ins_xy[0, 0],
    aligned_lidar_xy[:, 1] - ins_xy[0, 1],
    label="GRIL LiDAR odometry",
)
axis.set_aspect("equal")
axis.set_xlabel("x [m]")
axis.set_ylabel("y [m]")
axis.grid(True, alpha=0.3)
axis.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "frontend_trajectory.png", dpi=180)
plt.close(fig)
