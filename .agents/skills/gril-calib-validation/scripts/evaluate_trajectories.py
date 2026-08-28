#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from lidar2lidar.record_adapter import Record
from lidar2lidar.record_utils import message_timestamp_ns

parser = argparse.ArgumentParser()
parser.add_argument("--record-file", action="append", required=True)
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--pose-topic", default="/apollo/sensor/gnss/odometry")
args = parser.parse_args()

RECORDS = args.record_file
POSE_TOPIC = args.pose_topic
TRAJECTORY = args.trajectory
OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)


lidar = np.loadtxt(TRAJECTORY, comments="#")
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
            pose_samples.append(
                (
                    timestamp_ns * 1e-9,
                    float(pose.position.x),
                    float(pose.position.y),
                )
            )

pose_samples = np.asarray(pose_samples)
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
