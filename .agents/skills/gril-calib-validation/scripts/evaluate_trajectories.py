#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from profile_gril_batch_timing import read_states  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from lidar2lidar.extrinsic_io import load_extrinsics_file  # noqa: E402
from lidar2lidar.record_adapter import Record  # noqa: E402
from lidar2lidar.record_utils import message_timestamp_ns  # noqa: E402


def _load_lidar_to_imu_xy(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".yaml", ".yml"}:
        mat, _, _, _, _ = load_extrinsics_file(str(path))
        return np.asarray(mat[:2, 3], dtype=np.float64)
    text = path.read_text()
    match = re.search(r"Translation LiDAR to IMU[^=]*=\s*([^\n]+)", text)
    if match is None:
        raise ValueError(f"Missing 'Translation LiDAR to IMU' in {path}")
    vals = [float(v) for v in match.group(1).split()]
    return np.asarray(vals[:2], dtype=np.float64)


def _load_imu_in_pose_xy(args: argparse.Namespace) -> np.ndarray | None:
    if args.imu_in_pose_xyz is not None:
        return np.asarray(args.imu_in_pose_xyz[:2], dtype=np.float64)
    if args.imu_in_pose_yaml is not None:
        mat, _, _, _, _ = load_extrinsics_file(str(args.imu_in_pose_yaml))
        return np.asarray(mat[:2, 3], dtype=np.float64)
    return None


def _compute_error_stats(
    aligned_xy: np.ndarray,
    ref_xy: np.ndarray,
    lidar_time: np.ndarray,
) -> dict[str, float]:
    ate = np.linalg.norm(aligned_xy - ref_xy, axis=1)
    step_indices = np.searchsorted(lidar_time, lidar_time + 1.0)
    valid = step_indices < len(lidar_time)
    starts = np.flatnonzero(valid)
    ends = step_indices[valid]
    rpe = np.linalg.norm(
        (aligned_xy[ends] - aligned_xy[starts]) - (ref_xy[ends] - ref_xy[starts]),
        axis=1,
    )
    lidar_steps = np.linalg.norm(np.diff(aligned_xy, axis=0), axis=1)
    ins_steps = np.linalg.norm(np.diff(ref_xy, axis=0), axis=1)
    return {
        "ate_xy_rmse_m": float(np.sqrt(np.mean(ate**2))),
        "ate_xy_p95_m": float(np.percentile(ate, 95)),
        "ate_xy_max_m": float(np.max(ate)),
        "rpe_xy_1s_rmse_m": float(np.sqrt(np.mean(rpe**2))) if len(rpe) else 0.0,
        "rpe_xy_1s_p95_m": float(np.percentile(rpe, 95)) if len(rpe) else 0.0,
        "lidar_path_length_m": float(np.sum(lidar_steps)),
        "ins_path_length_m": float(np.sum(ins_steps)),
        "lidar_max_step_m": float(np.max(lidar_steps)) if len(lidar_steps) else 0.0,
        "ins_max_step_m": float(np.max(ins_steps)) if len(ins_steps) else 0.0,
    }


def _solve_se2_with_lever_arm(
    lidar_xy: np.ndarray,
    ins_xy: np.ndarray,
    R_wo_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve R(theta)*p_L + t - R_wo*t_OL = p_O with strict R in SO(2)."""
    n = len(lidar_xy)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = ins_xy.reshape(-1)
    px, py = lidar_xy[:, 0], lidar_xy[:, 1]
    A[0::2, 0] = px
    A[0::2, 1] = -py
    A[0::2, 2] = 1.0
    A[1::2, 0] = py
    A[1::2, 1] = px
    A[1::2, 3] = 1.0
    A[0::2, 4] = -R_wo_2d[:, 0, 0]
    A[0::2, 5] = -R_wo_2d[:, 0, 1]
    A[1::2, 4] = -R_wo_2d[:, 1, 0]
    A[1::2, 5] = -R_wo_2d[:, 1, 1]
    sol_lin, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    theta0 = float(np.arctan2(sol_lin[1], sol_lin[0]))

    def _res(x: np.ndarray) -> np.ndarray:
        th, tx, ty, ox, oy = x
        c, s = np.cos(th), np.sin(th)
        rx = c * px - s * py + tx - (R_wo_2d[:, 0, 0] * ox + R_wo_2d[:, 0, 1] * oy)
        ry = s * px + c * py + ty - (R_wo_2d[:, 1, 0] * ox + R_wo_2d[:, 1, 1] * oy)
        return np.concatenate([rx - ins_xy[:, 0], ry - ins_xy[:, 1]])

    x0 = np.array([theta0, sol_lin[2], sol_lin[3], sol_lin[4], sol_lin[5]])
    opt = least_squares(_res, x0, method="lm", max_nfev=60)
    th, tx, ty, ox, oy = opt.x
    R_opt = np.array(
        [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]],
        dtype=np.float64,
    )
    return (
        R_opt,
        np.array([tx, ty], dtype=np.float64),
        np.array([ox, oy], dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-file", action="append", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path)
    source.add_argument("--batch-trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pose-topic", default="/apollo/sensor/gnss/odometry")
    parser.add_argument(
        "--result",
        type=Path,
        help="Optional GRIL result txt or standardized extrinsic YAML (T_{I<-L}).",
    )
    parser.add_argument(
        "--imu-in-pose-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Known IMU origin position [m] in the --pose-topic body frame.",
    )
    parser.add_argument(
        "--imu-in-pose-yaml",
        type=Path,
        help="Optional YAML file giving T_{pose_body<-imu}.",
    )
    parser.add_argument(
        "--lever-arm-threshold-m",
        type=float,
        default=0.35,
        help="Threshold on |t_{pose<-imu}| to flag ant2outposoffset mismatch.",
    )
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
            Rotation.from_matrix(lidar_states[:, 1:10].reshape(-1, 3, 3)).as_euler(
                "xyz"
            )[:, 2]
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
                    raise RuntimeError(
                        f"{POSE_TOPIC} has neither pose nor localization"
                    )
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
                            if orientation is not None
                            else ()
                        ),
                    )
                )

    pose_samples = np.asarray(pose_samples, dtype=np.float64)
    order = np.argsort(pose_samples[:, 0])
    pose_samples = pose_samples[order]
    _, unique_idx = np.unique(pose_samples[:, 0], return_index=True)
    pose_samples = pose_samples[np.sort(unique_idx)]
    if np.any(np.diff(pose_samples[:, 0]) <= 0):
        raise ValueError("GNSS/INS odometry timestamps must be strictly increasing")

    # Clamp LiDAR trajectory to pose coverage if records only cover a subset
    in_cov = (lidar_time >= pose_samples[0, 0]) & (lidar_time <= pose_samples[-1, 0])
    if int(np.sum(in_cov)) < 10:
        raise ValueError("GNSS/INS odometry does not cover the LiDAR trajectory")
    lidar_time = lidar_time[in_cov]
    lidar_xy = lidar_xy[in_cov]
    if args.batch_trace:
        lidar_yaw = lidar_yaw[in_cov]

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

    raw_stats = _compute_error_stats(aligned_lidar_xy, ins_xy, lidar_time)
    metrics = {
        "alignment": "SE(2) rigid, no scale",
        "pose_count": int(len(lidar_time)),
        "duration_s": float(lidar_time[-1] - lidar_time[0]),
        **raw_stats,
    }

    aligned_comp_xy = None
    has_orientation = pose_samples.shape[1] >= 7
    if has_orientation:
        ins_q = np.column_stack(
            [
                np.interp(lidar_time, pose_samples[:, 0], pose_samples[:, ax])
                for ax in (3, 4, 5, 6)
            ]
        )
        ins_q /= np.maximum(np.linalg.norm(ins_q, axis=1, keepdims=True), 1e-12)
        R_wo = Rotation.from_quat(ins_q).as_matrix()
        ins_yaw = np.unwrap(Rotation.from_matrix(R_wo).as_euler("xyz")[:, 2])
        total_yaw_span_deg = float(np.degrees(np.max(ins_yaw) - np.min(ins_yaw)))

        t_IL_xy = (
            _load_lidar_to_imu_xy(args.result) if args.result is not None else None
        )
        t_OI_xy = _load_imu_in_pose_xy(args)

        if t_IL_xy is not None and t_OI_xy is not None:
            t_OL_explicit = t_OI_xy + t_IL_xy
            ins_at_L = ins_xy + np.einsum("nij,j->ni", R_wo[:, :2, :2], t_OL_explicit)
            lc = lidar_xy.mean(axis=0)
            ic = ins_at_L.mean(axis=0)
            cov_e = (lidar_xy - lc).T @ (ins_at_L - ic)
            ue, _, vte = np.linalg.svd(cov_e)
            Re = vte.T @ ue.T
            if np.linalg.det(Re) < 0:
                vte[-1] *= -1
                Re = vte.T @ ue.T
            te = ic - Re @ lc
            aligned_explicit = (
                (Re @ lidar_xy.T).T
                + te
                - np.einsum("nij,j->ni", R_wo[:, :2, :2], t_OL_explicit)
            )
            comp_stats = _compute_error_stats(aligned_explicit, ins_xy, lidar_time)
            aligned_comp_xy = aligned_explicit
            metrics["lever_arm_compensated"] = {
                "mode": "explicit_imu_in_pose_plus_result",
                "applied_pose_to_lidar_xy_m": [float(v) for v in t_OL_explicit],
                **comp_stats,
            }
        elif total_yaw_span_deg >= 15.0:
            R_opt, t_opt, t_OL_est = _solve_se2_with_lever_arm(
                lidar_xy, ins_xy, R_wo[:, :2, :2]
            )
            aligned_est = (
                (R_opt @ lidar_xy.T).T
                + t_opt
                - np.einsum("nij,j->ni", R_wo[:, :2, :2], t_OL_est)
            )
            comp_stats = _compute_error_stats(aligned_est, ins_xy, lidar_time)
            aligned_comp_xy = aligned_est
            comp_block: dict[str, object] = {
                "mode": "joint_se2_plus_lever_arm_regression",
                "estimated_pose_to_lidar_xy_m": [float(v) for v in t_OL_est],
                **comp_stats,
            }
            if t_IL_xy is not None:
                implied_t_OI = t_OL_est - t_IL_xy
                offset_norm = float(np.linalg.norm(implied_t_OI))
                comp_block["provided_imu_to_lidar_xy_m"] = [float(v) for v in t_IL_xy]
                comp_block["implied_imu_in_pose_xy_m"] = [
                    float(v) for v in implied_t_OI
                ]
                comp_block["implied_imu_in_pose_norm_m"] = offset_norm
                comp_block["ant2outposoffset_detected"] = bool(
                    offset_norm >= args.lever_arm_threshold_m
                )
            metrics["lever_arm_compensated"] = comp_block

    if args.batch_trace:
        metrics["position_comparison_caveat"] = (
            "INS and LiDAR positions refer to different sensor origins (and INS "
            "odometry may include ant2outposoffset to the rear axle). Use "
            "lever_arm_compensated metrics when comparing trajectory fidelity."
        )
        yaw_difference = np.degrees((lidar_yaw - lidar_yaw[0]) - (ins_yaw - ins_yaw[0]))
        windows = []
        edges = np.linspace(lidar_time[0], lidar_time[-1], 5)
        lidar_steps = np.linalg.norm(np.diff(aligned_lidar_xy, axis=0), axis=1)
        ins_steps = np.linalg.norm(np.diff(ins_xy, axis=0), axis=1)
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
                    "ins_yaw_change_deg": float(
                        np.degrees(ins_yaw[last] - ins_yaw[first])
                    ),
                    "yaw_drift_deg": float(
                        yaw_difference[last] - yaw_difference[first]
                    ),
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
    axis.plot(
        ins_xy[:, 0] - ins_xy[0, 0],
        ins_xy[:, 1] - ins_xy[0, 1],
        label="GNSS/INS (pose origin)",
    )
    axis.plot(
        aligned_lidar_xy[:, 0] - ins_xy[0, 0],
        aligned_lidar_xy[:, 1] - ins_xy[0, 1],
        alpha=0.55,
        label=f"Raw LiDAR SE(2) (ATE={raw_stats['ate_xy_rmse_m']:.2f}m)",
    )
    if aligned_comp_xy is not None:
        c_rmse = metrics["lever_arm_compensated"]["ate_xy_rmse_m"]
        axis.plot(
            aligned_comp_xy[:, 0] - ins_xy[0, 0],
            aligned_comp_xy[:, 1] - ins_xy[0, 1],
            "--",
            label=f"LiDAR lever-arm compensated (ATE={c_rmse:.2f}m)",
        )
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "frontend_trajectory.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
