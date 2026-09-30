#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from lidar2lidar.record_adapter import Record
from lidar2lidar.record_utils import message_timestamp_ns


def parse_result(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    text = path.read_text()

    def values(label: str, count: int) -> np.ndarray:
        match = re.search(rf"{label}[^=]*=\s*([^\n]+)", text)
        if match is None:
            raise ValueError(f"Missing {label!r} in {path}")
        parsed = np.fromstring(match.group(1), sep=" ")
        if len(parsed) != count:
            raise ValueError(f"Expected {count} values for {label!r}")
        return parsed

    rotation = values("Rotation LiDAR to IMU", 3)
    translation = values("Translation LiDAR to IMU", 3)
    time_lag = float(values("Time Lag IMU to LiDAR", 1)[0])
    return rotation, translation, time_lag


def write_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype="<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(points.tobytes())


parser = argparse.ArgumentParser(
    description=(
        "Build a submap using independent IMU/GNSS odometry, the final GRIL "
        "extrinsic, and per-point timestamps."
    )
)
parser.add_argument("--record-file", action="append", required=True)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--lidar-topic", default="/apollo/sensor/vanjeelidar/up/PointCloud2"
)
parser.add_argument("--pose-topic", default="/apollo/sensor/gnss/odometry")
parser.add_argument("--scan-stride", type=int, default=5)
parser.add_argument("--point-stride", type=int, default=8)
parser.add_argument("--max-range-m", type=float, default=60.0)
parser.add_argument("--voxel-size-m", type=float, default=0.10)
parser.add_argument("--thickness-neighbors", type=int, default=20)
parser.add_argument("--thickness-samples", type=int, default=20000)
args = parser.parse_args()

rotation_deg, translation_li, time_lag_s = parse_result(args.result)
rotation_li = Rotation.from_euler("xyz", rotation_deg, degrees=True)

pose_samples = []
for record_file in args.record_file:
    with Record(record_file) as record:
        for topic, message, record_timestamp_ns in record.read_messages(
            topics=[args.pose_topic]
        ):
            pose = getattr(message, "pose", None)
            if pose is None:
                pose = getattr(message, "localization", None)
            if pose is None:
                raise RuntimeError(
                    f"{args.pose_topic} has neither pose nor localization"
                )
            timestamp_ns = message_timestamp_ns(
                topic, message, int(record_timestamp_ns)
            )
            pose_samples.append(
                (
                    timestamp_ns * 1e-9,
                    [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                    ],
                    [
                        float(pose.orientation.qx),
                        float(pose.orientation.qy),
                        float(pose.orientation.qz),
                        float(pose.orientation.qw),
                    ],
                )
            )

pose_time = np.array([sample[0] for sample in pose_samples])
pose_position = np.array([sample[1] for sample in pose_samples])
pose_rotation = Rotation.from_quat([sample[2] for sample in pose_samples])
pose_slerp = Slerp(pose_time, pose_rotation)

world_points = []
scan_count = 0
used_scan_count = 0
for record_file in args.record_file:
    with Record(record_file) as record:
        for topic, payload, type_name, _ in record.read_raw_messages(
            [args.lidar_topic]
        ):
            if scan_count % args.scan_stride:
                scan_count += 1
                continue
            scan_count += 1
            message = record.decode_message(topic, payload, type_name)
            raw = message.point[:: args.point_stride]
            xyz = np.array([(point.x, point.y, point.z) for point in raw], dtype=float)
            timestamp = (
                np.array([point.timestamp for point in raw], dtype=np.float64) * 1e-9
                + time_lag_s
            )
            valid = (
                np.isfinite(xyz).all(axis=1)
                & (np.linalg.norm(xyz, axis=1) <= args.max_range_m)
                & (timestamp >= pose_time[0])
                & (timestamp <= pose_time[-1])
            )
            xyz = xyz[valid]
            timestamp = timestamp[valid]
            if not len(xyz):
                continue
            imu_points = rotation_li.apply(xyz) + translation_li
            world_rotation = pose_slerp(timestamp)
            world_position = np.column_stack(
                [
                    np.interp(timestamp, pose_time, pose_position[:, axis])
                    for axis in range(3)
                ]
            )
            world_points.append(world_rotation.apply(imu_points) + world_position)
            used_scan_count += 1

points = np.concatenate(world_points)
origin = points.min(axis=0)
voxel_index = np.floor((points - origin) / args.voxel_size_m).astype(np.int64)
_, inverse, counts = np.unique(
    voxel_index, axis=0, return_inverse=True, return_counts=True
)
downsampled = np.column_stack(
    [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
)

rng = np.random.default_rng(0)
sample_index = rng.choice(
    len(downsampled),
    min(args.thickness_samples, len(downsampled)),
    replace=False,
)
tree = cKDTree(downsampled)
_, neighbors = tree.query(downsampled[sample_index], k=args.thickness_neighbors)
thickness = []
for indices in neighbors:
    local = downsampled[indices]
    covariance = np.cov(local - local.mean(axis=0), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[1] > 0.0 and eigenvalues[0] / eigenvalues[1] < 0.25:
        thickness.append(2.0 * np.sqrt(max(eigenvalues[0], 0.0)))
thickness = np.asarray(thickness)

args.output_dir.mkdir(parents=True, exist_ok=True)
write_ply(args.output_dir / "imu_extrinsic_submap.ply", downsampled)
report = {
    "pose_source": args.pose_topic,
    "motion_compensation": "per-point SE(3) interpolation",
    "rotation_lidar_to_imu_deg": rotation_deg.tolist(),
    "translation_lidar_to_imu_m": translation_li.tolist(),
    "time_lag_imu_to_lidar_s": time_lag_s,
    "input_scan_count": scan_count,
    "used_scan_count": used_scan_count,
    "raw_sampled_point_count": int(len(points)),
    "voxel_point_count": int(len(downsampled)),
    "voxel_size_m": args.voxel_size_m,
    "local_planar_thickness_m": {
        "sample_count": int(len(thickness)),
        "p50": float(np.percentile(thickness, 50)),
        "p95": float(np.percentile(thickness, 95)),
        "p99": float(np.percentile(thickness, 99)),
    },
    "interpretation": (
        "Lower thickness is better only when comparing candidates with identical "
        "records, sampling, pose source, crop, and voxel settings."
    ),
}
with (args.output_dir / "submap_metrics.yaml").open("w") as stream:
    yaml.safe_dump(report, stream, sort_keys=False)

fig, axis = plt.subplots(figsize=(8, 8))
plot_index = rng.choice(len(downsampled), min(100000, len(downsampled)), replace=False)
axis.scatter(
    downsampled[plot_index, 0],
    downsampled[plot_index, 1],
    s=0.1,
)
axis.set_aspect("equal")
axis.set_xlabel("world x [m]")
axis.set_ylabel("world y [m]")
axis.set_title("IMU/GNSS odometry + GRIL extrinsic submap")
fig.tight_layout()
fig.savefig(args.output_dir / "imu_extrinsic_submap_bev.png", dpi=180)
