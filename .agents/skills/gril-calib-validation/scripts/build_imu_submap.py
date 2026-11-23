#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from profile_gril_batch_timing import read_states  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

from lidar2lidar.extrinsic_io import load_extrinsics_file  # noqa: E402
from lidar2lidar.record_adapter import Record  # noqa: E402
from lidar2lidar.record_utils import message_timestamp_ns  # noqa: E402


def parse_result(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        mat, _, _, _, data = load_extrinsics_file(str(path))
        rot_deg = Rotation.from_matrix(mat[:3, :3]).as_euler("xyz", degrees=True)
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        tau = float(meta.get("time_offset_s", 0.0))
        return rot_deg, np.asarray(mat[:3, 3], dtype=np.float64), tau

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


def _load_pose_to_imu_transform(args: argparse.Namespace) -> np.ndarray:
    """Return T_{pose_body <- imu} mapping IMU-origin points into pose body frame."""
    T = np.eye(4, dtype=np.float64)
    if args.imu_in_pose_yaml is not None:
        mat, _, _, _, _ = load_extrinsics_file(str(args.imu_in_pose_yaml))
        return np.asarray(mat, dtype=np.float64)
    if args.imu_in_pose_xyz is not None:
        T[:3, 3] = np.asarray(args.imu_in_pose_xyz, dtype=np.float64)
    return T


def _estimate_pose_to_imu_from_batch_trace(
    batch_trace: Path,
    pose_time: np.ndarray,
    pose_position: np.ndarray,
    pose_slerp: Slerp,
    translation_li: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Estimate t_{pose_body <- imu} = t_{pose <- lidar} - t_{imu <- lidar} via
    SE(2) + rigid-body lever-arm regression against LiDAR odometry states.
    """
    lines = batch_trace.read_text().splitlines()
    if not lines or lines[0].strip() != "GRIL_BATCH_TRACE 1":
        return None, {"available": False, "reason": "invalid_batch_trace_format"}
    lidar_states = read_states(lines, "lidar")
    t_lid = lidar_states[:, 0]
    p_lid = lidar_states[:, 10:12]
    in_cov = (t_lid >= pose_time[0]) & (t_lid <= pose_time[-1])
    if int(np.sum(in_cov)) < 30:
        return None, {"available": False, "reason": "insufficient_pose_coverage"}
    t_q = t_lid[in_cov]
    p_q = p_lid[in_cov]
    ins_xy = np.column_stack(
        [np.interp(t_q, pose_time, pose_position[:, ax]) for ax in (0, 1)]
    )
    R_wo = pose_slerp(t_q).as_matrix()
    ins_yaw = np.unwrap(Rotation.from_matrix(R_wo).as_euler("xyz")[:, 2])
    if float(np.degrees(np.max(ins_yaw) - np.min(ins_yaw))) < 15.0:
        return None, {"available": False, "reason": "insufficient_yaw_excitation"}

    n = len(t_q)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = ins_xy.reshape(-1)
    A[0::2, 0] = p_q[:, 0]
    A[0::2, 1] = -p_q[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 0] = p_q[:, 1]
    A[1::2, 1] = p_q[:, 0]
    A[1::2, 3] = 1.0
    A[0::2, 4] = -R_wo[:, 0, 0]
    A[0::2, 5] = -R_wo[:, 0, 1]
    A[1::2, 4] = -R_wo[:, 1, 0]
    A[1::2, 5] = -R_wo[:, 1, 1]
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    theta = float(np.arctan2(sol[1], sol[0]))
    R_opt = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    b4 = b - (R_opt @ p_q.T).T.reshape(-1)
    sol4, _, _, _ = np.linalg.lstsq(A[:, 2:], b4, rcond=None)
    t_OL_xy = sol4[2:4]
    t_OI_xy = t_OL_xy - translation_li[:2]
    T_OI = np.eye(4, dtype=np.float64)
    T_OI[:2, 3] = t_OI_xy
    return T_OI, {
        "available": True,
        "estimated_pose_to_lidar_xy_m": [float(v) for v in t_OL_xy],
        "implied_imu_in_pose_xy_m": [float(v) for v in t_OI_xy],
        "implied_imu_in_pose_norm_m": float(np.linalg.norm(t_OI_xy)),
    }


def _audit_pose_turning_lever_arm(
    pose_time: np.ndarray,
    pose_position: np.ndarray,
    pose_rotation: Rotation,
) -> dict[str, object]:
    """Check whether --pose-topic turns off ICR (e.g. ant2outposoffset)."""
    if len(pose_time) < 30:
        return {"available": False}
    dt = float(np.median(np.diff(pose_time)))
    if dt <= 0.0:
        return {"available": False}
    R_wb = pose_rotation.as_matrix()
    v_world = np.gradient(pose_position, dt, axis=0)
    v_body = np.einsum("nji,nj->ni", R_wb, v_world)
    yaws = np.unwrap(pose_rotation.as_euler("xyz")[:, 2])
    wz = np.gradient(yaws, dt)
    turn = np.abs(wz) > 0.10
    if int(np.sum(turn)) < 20:
        return {"available": False, "reason": "insufficient_turn_samples"}
    lat_over_wz = float(np.median(v_body[turn, 0] / wz[turn]))
    return {
        "available": True,
        "turn_samples": int(np.sum(turn)),
        "median_pose_lateral_vel_over_wz_m": lat_over_wz,
        "ant2outposoffset_suspected": bool(abs(lat_over_wz) >= 0.35),
    }


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


def main() -> None:
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
    parser.add_argument(
        "--imu-in-pose-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "IMU origin [m] expressed in the --pose-topic body frame "
            "(needed when GNSS/INS uses ant2outposoffset to rear axle)."
        ),
    )
    parser.add_argument(
        "--imu-in-pose-yaml",
        type=Path,
        help="Optional 4x4 extrinsic YAML specifying T_{pose_body <- imu}.",
    )
    parser.add_argument(
        "--batch-trace",
        type=Path,
        help=(
            "Optional GRIL_batch_trace_v1.txt used to audit or auto-compensate "
            "pose-origin lever-arm offset (ant2outposoffset)."
        ),
    )
    parser.add_argument(
        "--auto-compensate-pose-offset",
        action="store_true",
        help=(
            "If --imu-in-pose-* is omitted and --batch-trace is supplied, "
            "auto-estimate and apply t_{pose_body <- imu} before submap deskew."
        ),
    )
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

    # Sort and deduplicate pose timestamps across multi-shard records
    pose_samples.sort(key=lambda item: item[0])
    dedup = []
    last_t = -1e18
    for sample in pose_samples:
        if sample[0] > last_t + 1e-9:
            dedup.append(sample)
            last_t = sample[0]
    pose_samples = dedup

    pose_time = np.array([sample[0] for sample in pose_samples])
    pose_position = np.array([sample[1] for sample in pose_samples])
    pose_rotation = Rotation.from_quat([sample[2] for sample in pose_samples])
    pose_slerp = Slerp(pose_time, pose_rotation)

    T_pose_imu = _load_pose_to_imu_transform(args)
    pose_offset_source = (
        "explicit_imu_in_pose"
        if (args.imu_in_pose_xyz is not None or args.imu_in_pose_yaml is not None)
        else "identity_assumed"
    )
    pose_audit: dict[str, object] = _audit_pose_turning_lever_arm(
        pose_time, pose_position, pose_rotation
    )
    if args.batch_trace is not None:
        T_auto, batch_audit = _estimate_pose_to_imu_from_batch_trace(
            args.batch_trace,
            pose_time,
            pose_position,
            pose_slerp,
            translation_li,
        )
        pose_audit["batch_trace_regression"] = batch_audit
        if (
            args.auto_compensate_pose_offset
            and pose_offset_source == "identity_assumed"
            and T_auto is not None
        ):
            T_pose_imu = T_auto
            pose_offset_source = "auto_estimated_from_batch_trace"

    R_pose_imu = T_pose_imu[:3, :3]
    t_pose_imu = T_pose_imu[:3, 3]

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
                xyz = np.array(
                    [(point.x, point.y, point.z) for point in raw], dtype=float
                )
                timestamp = (
                    np.array([point.timestamp for point in raw], dtype=np.float64)
                    * 1e-9
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
                pose_body_points = imu_points @ R_pose_imu.T + t_pose_imu
                world_rotation = pose_slerp(timestamp)
                world_position = np.column_stack(
                    [
                        np.interp(timestamp, pose_time, pose_position[:, axis])
                        for axis in range(3)
                    ]
                )
                world_points.append(
                    world_rotation.apply(pose_body_points) + world_position
                )
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
        "pose_to_imu_offset_source": pose_offset_source,
        "pose_to_imu_translation_m": t_pose_imu.tolist(),
        "pose_origin_audit": pose_audit,
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
            "records, sampling, pose source, pose-to-imu offset, crop, and voxel."
        ),
    }
    with (args.output_dir / "submap_metrics.yaml").open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)

    fig, axis = plt.subplots(figsize=(8, 8))
    plot_index = rng.choice(
        len(downsampled), min(100000, len(downsampled)), replace=False
    )
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
    plt.close(fig)


if __name__ == "__main__":
    main()
