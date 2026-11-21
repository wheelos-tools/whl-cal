"""Post-run calibration quality gates for ROS-free GRIL LiDAR-IMU runs.

Strictly read-only on completed run artifacts (`GRIL_Calib_result.txt`,
`GRIL_full_frontend_trace_v1.txt`, `GRIL_batch_trace_v1.txt`, `gril_native.conf`).
Quality thresholds never alter frozen GRIL algorithm execution.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.geometry import euler_xyz_degrees_to_matrix


@dataclass(frozen=True)
class QualityGateThresholds:
    min_lidar_states: int = 900
    min_accum_duration_s: float = 60.0
    max_odom_step_m: float = 1.5
    max_odom_tilt_p95_deg: float = 10.0
    max_degenerate_skip_ratio: float = 0.05
    min_path_length_m: float = 50.0
    min_total_yaw_deg: float = 60.0
    min_p95_omega_rad_s: float = 0.08
    min_rms_alpha_rad_s2: float = 0.05
    min_acc_xy_std_m_s2: float = 0.15
    min_rot_hessian_min_eig: float = 5.0
    min_trans_xy_min_sv: float = 1.5
    min_trans_3d_min_sv: float = 0.5
    min_time_alpha_energy: float = 2.0
    max_abs_time_offset_s: float = 0.20
    min_xcorr_peak_pearson: float = 0.50
    max_repeat_rot_deg: float = 0.10
    max_repeat_trans_m: float = 0.01
    max_repeat_time_s: float = 0.001
    max_seed_rot_deg: float = 1.00
    max_seed_trans_xy_m: float = 0.15
    max_seed_time_s: float = 0.005
    max_window_rot_deg: float = 2.50
    max_window_trans_xy_m: float = 0.35
    max_window_time_s: float = 0.015
    max_holdout_gyro_rmse_rad_s: float = 0.08
    max_holdout_acc_rmse_m_s2: float = 0.80


def parse_detailed_gril_result(path: Path) -> dict[str, Any]:
    text = Path(path).read_text()

    def _vals(label: str) -> list[float]:
        match = re.search(rf"{label}[^=]*=\s*([^\n]+)", text)
        if match is None:
            raise ValueError(f"Missing {label!r} in {path}")
        return [float(v) for v in match.group(1).split()]

    euler_deg = _vals("Rotation LiDAR to IMU")
    trans_m = _vals("Translation LiDAR to IMU")
    time_lag_s = _vals("Time Lag IMU to LiDAR")[0]
    gyro_bias = _vals("Bias of Gyroscope")
    acc_bias = _vals("Bias of Accelerometer")

    lines = text.splitlines()
    mat_idx = None
    for idx, line in enumerate(lines):
        if "Homogeneous Transformation Matrix" in line:
            mat_idx = idx + 1
            break
    if mat_idx is not None and mat_idx + 4 <= len(lines):
        hom = np.array(
            [[float(x) for x in lines[mat_idx + r].split()] for r in range(4)],
            dtype=np.float64,
        )
        R_LI = hom[:3, :3]
        T_LI = hom[:3, 3]
    else:
        R_LI = euler_xyz_degrees_to_matrix(euler_deg)
        T_LI = np.asarray(trans_m, dtype=np.float64)

    T_IL = -(R_LI.T @ T_LI)
    yaw_deg = float(math.degrees(math.atan2(R_LI[1, 0], R_LI[0, 0])))
    pitch_deg = float(math.degrees(math.asin(max(-1.0, min(1.0, float(-R_LI[2, 0]))))))
    roll_deg = float(math.degrees(math.atan2(R_LI[2, 1], R_LI[2, 2])))
    return {
        "rotation_xyz_deg": euler_deg,
        "zyx_rpy_deg": [roll_deg, pitch_deg, yaw_deg],
        "translation_LI_m": [float(x) for x in T_LI],
        "translation_IL_m": [float(x) for x in T_IL],
        "rotation_LI_matrix": R_LI.tolist(),
        "time_offset_s": float(time_lag_s),
        "gyro_bias_rad_s": gyro_bias,
        "acc_bias_m_s2": acc_bias,
    }


def _parse_native_conf(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    out: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        key, val = parts
        if val in {"true", "false"}:
            out[key] = val == "true"
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def parse_batch_trace_arrays(path: Path) -> dict[str, Any]:
    imu_t: list[float] = []
    imu_w: list[list[float]] = []
    imu_a: list[list[float]] = []
    lid_t: list[float] = []
    lid_R: list[np.ndarray] = []
    lid_p: list[list[float]] = []
    lid_w: list[list[float]] = []
    lid_v: list[list[float]] = []
    g_lid_q: list[list[float]] = []
    g_imu_q: list[list[float]] = []
    g_norm: list[list[float]] = []
    g_dist: list[float] = []
    meta: dict[str, float] = {}

    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("orig_odom_freq "):
                meta["orig_odom_freq"] = float(line.split()[1])
            elif line.startswith("cut_frame_num "):
                meta["cut_frame_num"] = float(line.split()[1])
            elif line.startswith("timediff_imu_wrt_lidar "):
                meta["timediff_imu_wrt_lidar"] = float(line.split()[1])
            elif line.startswith("move_start_time "):
                meta["move_start_time"] = float(line.split()[1])
            elif line.startswith("imu "):
                vals = [float(x) for x in line.split()[1:]]
                imu_t.append(vals[0])
                imu_w.append(vals[13:16])
                imu_a.append(vals[22:25])
            elif line.startswith("lidar "):
                vals = [float(x) for x in line.split()[1:]]
                lid_t.append(vals[0])
                lid_R.append(np.array(vals[1:10], dtype=np.float64).reshape(3, 3))
                lid_p.append(vals[10:13])
                lid_w.append(vals[13:16])
                lid_v.append(vals[16:19])
            elif line.startswith("ground "):
                vals = [float(x) for x in line.split()[1:]]
                g_lid_q.append(vals[0:4])
                g_imu_q.append(vals[4:8])
                g_norm.append(vals[8:11])
                g_dist.append(vals[11])

    return {
        "meta": meta,
        "imu_t": np.asarray(imu_t, dtype=np.float64),
        "imu_w": np.asarray(imu_w, dtype=np.float64),
        "imu_a": np.asarray(imu_a, dtype=np.float64),
        "lid_t": np.asarray(lid_t, dtype=np.float64),
        "lid_R": np.asarray(lid_R, dtype=np.float64),
        "lid_p": np.asarray(lid_p, dtype=np.float64),
        "lid_w": np.asarray(lid_w, dtype=np.float64),
        "lid_v": np.asarray(lid_v, dtype=np.float64),
        "ground_lidar_wxyz": np.asarray(g_lid_q, dtype=np.float64),
        "ground_imu_wxyz": np.asarray(g_imu_q, dtype=np.float64),
        "ground_normal": np.asarray(g_norm, dtype=np.float64),
        "ground_dist": np.asarray(g_dist, dtype=np.float64),
    }


def _parse_frontend_health(trace_path: Path | None) -> dict[str, Any]:
    if trace_path is None or not trace_path.is_file():
        return {"available": False}
    updates = 0
    deg_skips = 0
    feat_counts: list[int] = []
    positions: list[list[float]] = []
    tilts_deg: list[float] = []
    with trace_path.open() as handle:
        for line in handle:
            if line.startswith("odometry "):
                parts = line.split()
                if parts[2] == "1":
                    updates += 1
                    feat_counts.append(int(parts[5]))
            elif line.startswith("frame_skipped degenerate_features"):
                deg_skips += 1
            elif line.startswith("updated "):
                vals = [float(x) for x in line.split()[1:13]]
                R = np.array(vals[:9], dtype=np.float64).reshape(3, 3)
                positions.append(vals[9:12])
                tilts_deg.append(
                    math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
                )
    pos = np.asarray(positions, dtype=np.float64) if positions else np.zeros((1, 3))
    steps = (
        np.linalg.norm(np.diff(pos, axis=0), axis=1)
        if len(pos) > 1
        else np.zeros(1, dtype=np.float64)
    )
    tilts = np.asarray(tilts_deg, dtype=np.float64) if tilts_deg else np.zeros(1)
    total_frames = max(1, updates + deg_skips)
    return {
        "available": True,
        "odometry_updates": int(updates),
        "degenerate_skips": int(deg_skips),
        "degenerate_skip_ratio": float(deg_skips / total_frames),
        "min_effective_features": int(min(feat_counts)) if feat_counts else 0,
        "p05_effective_features": (
            float(np.percentile(feat_counts, 5)) if feat_counts else 0.0
        ),
        "max_step_m": float(steps.max()),
        "p95_tilt_deg": float(np.percentile(tilts, 95)),
        "max_tilt_deg": float(tilts.max()),
        "z_span_m": float(pos[:, 2].max() - pos[:, 2].min()),
    }


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _interp_vec3(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_dst), 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(t_dst, t_src, v_src[:, axis])
    return out


def _zero_phase_smooth(signal: np.ndarray, window: int = 7) -> np.ndarray:
    if len(signal) < window * 2:
        return signal.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(signal)
    for axis in range(signal.shape[1]):
        fwd = np.convolve(signal[:, axis], kernel, mode="same")
        out[:, axis] = np.convolve(fwd[::-1], kernel, mode="same")[::-1]
    return out


def _project_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    R_orth = u @ vt
    if np.linalg.det(R_orth) < 0.0:
        u[:, -1] *= -1.0
        R_orth = u @ vt
    return R_orth


def _rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    if np.array_equal(R_a, R_b):
        return 0.0
    delta = _project_so3(R_a).T @ _project_so3(R_b)
    skew_part = 0.5 * (delta - delta.T)
    sin_theta = float(
        math.sqrt(skew_part[2, 1] ** 2 + skew_part[0, 2] ** 2 + skew_part[1, 0] ** 2)
    )
    cos_theta = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.atan2(sin_theta, cos_theta)))


def evaluate_holdout_trace(
    candidate_result: dict[str, Any],
    holdout_batch_trace_path: Path,
) -> dict[str, Any]:
    arrays = parse_batch_trace_arrays(holdout_batch_trace_path)
    lid_t = arrays["lid_t"]
    if len(lid_t) < 60:
        raise ValueError(f"Holdout trace has too few LiDAR states: {len(lid_t)}")
    R_LI = np.asarray(candidate_result["rotation_LI_matrix"], dtype=np.float64)
    R_IL = R_LI.T
    T_IL = np.asarray(candidate_result["translation_IL_m"], dtype=np.float64)
    tau = float(candidate_result["time_offset_s"])
    ba_L = R_IL @ np.asarray(candidate_result["acc_bias_m_s2"], dtype=np.float64)
    bg_I = np.asarray(candidate_result["gyro_bias_rad_s"], dtype=np.float64)

    # Trim boundary edge states so central differences and tau shift stay interior
    valid = slice(20, len(lid_t) - 20)
    t_eval = lid_t[valid]
    dt_med = float(np.median(np.diff(lid_t)))
    imu_w_sync = _zero_phase_smooth(
        _interp_vec3(arrays["imu_t"], arrays["imu_w"] - bg_I, lid_t + tau)
    )
    imu_a_sync = _zero_phase_smooth(
        _interp_vec3(arrays["imu_t"], arrays["imu_a"], lid_t + tau)
    )
    imu_alpha = np.gradient(imu_w_sync, dt_med, axis=0)
    lid_v_smooth = _zero_phase_smooth(arrays["lid_v"])
    lid_a = np.gradient(lid_v_smooth, dt_med, axis=0)
    lid_w_smooth = _zero_phase_smooth(arrays["lid_w"])

    # Uncalibrated identity baseline vs candidate on identical holdout kinematics
    gyro_res_id = lid_w_smooth[valid] - imu_w_sync[valid]
    gyro_res_cal = lid_w_smooth[valid] - (imu_w_sync[valid] @ R_IL.T)

    std_grav = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    acc_res_cal_list = []
    acc_res_zero_trans_list = []
    for idx in range(20, len(lid_t) - 20):
        R_ll0 = arrays["lid_R"][idx]
        R_gl = _quat_wxyz_to_rot(arrays["ground_lidar_wxyz"][idx])
        omega_L = R_IL @ imu_w_sync[idx]
        alpha_L = R_IL @ imu_alpha[idx]
        lever = np.cross(omega_L, np.cross(omega_L, T_IL)) + np.cross(alpha_L, T_IL)
        base_term = (
            R_ll0 @ (R_IL @ imu_a_sync[idx])
            - R_ll0 @ ba_L
            + R_gl @ std_grav
            - lid_a[idx]
        )
        acc_res_zero_trans_list.append(base_term)
        acc_res_cal_list.append(base_term - R_ll0 @ lever)

    acc_res_cal = np.asarray(acc_res_cal_list, dtype=np.float64)
    acc_res_zero_t = np.asarray(acc_res_zero_trans_list, dtype=np.float64)

    gyro_rmse = float(np.sqrt(np.mean(np.sum(gyro_res_cal**2, axis=1))))
    gyro_rmse_identity = float(np.sqrt(np.mean(np.sum(gyro_res_id**2, axis=1))))
    acc_rmse = float(np.sqrt(np.mean(np.sum(acc_res_cal**2, axis=1))))
    acc_rmse_zero_trans = float(np.sqrt(np.mean(np.sum(acc_res_zero_t**2, axis=1))))

    return {
        "holdout_trace": str(Path(holdout_batch_trace_path).resolve()),
        "evaluated_states": int(len(t_eval)),
        "duration_s": float(t_eval[-1] - t_eval[0]),
        "gyro_rmse_rad_s": gyro_rmse,
        "gyro_rmse_identity_rad_s": gyro_rmse_identity,
        "acc_rmse_m_s2": acc_rmse,
        "acc_rmse_zero_translation_m_s2": acc_rmse_zero_trans,
        "lever_arm_improved_holdout_acc": bool(acc_rmse <= acc_rmse_zero_trans + 1e-4),
    }


def evaluate_run_quality_gates(
    run_dir: Path,
    *,
    repeat_run_dirs: list[Path] | None = None,
    seed_run_dirs: list[Path] | None = None,
    window_run_dirs: list[Path] | None = None,
    holdout_batch_trace: Path | None = None,
    thresholds: QualityGateThresholds = QualityGateThresholds(),
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    result_path = run_dir / "GRIL_Calib_result.txt"
    batch_trace_path = run_dir / "GRIL_batch_trace_v1.txt"
    frontend_trace_path = run_dir / "GRIL_full_frontend_trace_v1.txt"
    native_conf_path = run_dir / "gril_native.conf"

    primary = parse_detailed_gril_result(result_path)
    native_conf = _parse_native_conf(native_conf_path)
    arrays = parse_batch_trace_arrays(batch_trace_path)
    frontend = _parse_frontend_health(frontend_trace_path)

    lid_t = arrays["lid_t"]
    lid_p = arrays["lid_p"]
    lid_R = arrays["lid_R"]
    imu_t = arrays["imu_t"]
    imu_w = arrays["imu_w"]
    imu_a = arrays["imu_a"]

    n_states = int(len(lid_t))
    duration_s = float(lid_t[-1] - lid_t[0]) if n_states > 1 else 0.0
    dt_med = float(np.median(np.diff(lid_t))) if n_states > 1 else 1.0 / 30.0
    step_xy = (
        np.linalg.norm(np.diff(lid_p[:, :2], axis=0), axis=1)
        if n_states > 1
        else np.zeros(1)
    )
    path_length_m = float(np.sum(step_xy))

    # 1. Data Sufficiency Pillar
    max_step_m = (
        frontend["max_step_m"]
        if frontend.get("available")
        else float(np.linalg.norm(np.diff(lid_p, axis=0), axis=1).max())
    )
    p95_tilt_deg = (
        frontend["p95_tilt_deg"]
        if frontend.get("available")
        else float(
            np.percentile(
                [
                    math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
                    for R in lid_R
                ],
                95,
            )
        )
    )
    deg_skip_ratio = float(frontend.get("degenerate_skip_ratio", 0.0))
    data_checks = {
        "min_lidar_states": n_states >= thresholds.min_lidar_states,
        "min_accum_duration": duration_s >= thresholds.min_accum_duration_s,
        "trajectory_continuity": max_step_m <= thresholds.max_odom_step_m,
        "attitude_tilt_stability": p95_tilt_deg <= thresholds.max_odom_tilt_p95_deg,
        "feature_continuity": deg_skip_ratio <= thresholds.max_degenerate_skip_ratio,
    }

    # 2. Motion Excitation Pillar
    imu_w_sync = _zero_phase_smooth(_interp_vec3(imu_t, imu_w, lid_t))
    imu_a_sync = _zero_phase_smooth(_interp_vec3(imu_t, imu_a, lid_t))
    imu_alpha = np.gradient(imu_w_sync, dt_med, axis=0)
    omega_norm = np.linalg.norm(imu_w_sync, axis=1)
    total_abs_yaw_deg = float(math.degrees(np.sum(np.abs(imu_w_sync[:, 2])) * dt_med))
    left_turn_deg = float(
        math.degrees(np.sum(np.maximum(0.0, imu_w_sync[:, 2])) * dt_med)
    )
    right_turn_deg = float(
        math.degrees(np.sum(np.maximum(0.0, -imu_w_sync[:, 2])) * dt_med)
    )
    roll_pitch_exc_deg = float(
        math.degrees(np.sum(np.linalg.norm(imu_w_sync[:, :2], axis=1)) * dt_med)
    )
    p95_omega = float(np.percentile(omega_norm, 95))
    rms_alpha = float(np.sqrt(np.mean(np.sum(imu_alpha**2, axis=1))))
    acc_xy_std = float(np.linalg.norm(np.std(imu_a_sync[:, :2], axis=0)))

    excitation_checks = {
        "planar_path_length": path_length_m >= thresholds.min_path_length_m,
        "yaw_rotation_excitation": total_abs_yaw_deg >= thresholds.min_total_yaw_deg,
        "angular_velocity_p95": p95_omega >= thresholds.min_p95_omega_rad_s,
        "angular_acceleration_rms": rms_alpha >= thresholds.min_rms_alpha_rad_s2,
        "planar_linear_acc_variation": acc_xy_std >= thresholds.min_acc_xy_std_m_s2,
    }

    # 3. Parameter Observability Pillar (Rotation, Translation XY/Z, Time Offset)
    R_LI = np.asarray(primary["rotation_LI_matrix"], dtype=np.float64)
    R_IL = R_LI.T

    # 3a) Rotation SO(3) Gauss-Newton information matrix:
    # Combine gyro alignment sensitivity -[R_IL w_I]_x, ground-normal alignment,
    # and dynamic planar acceleration sensitivity -[R_IL (a_I - g_I)]_x.
    H_rot = np.zeros((3, 3), dtype=np.float64)
    a_mean = np.mean(imu_a_sync, axis=0)
    for k in range(20, max(21, n_states - 20)):
        J_gyro = _skew(R_IL @ imu_w_sync[k])
        J_acc = _skew(R_IL @ (imu_a_sync[k] - a_mean))
        H_rot += J_gyro.T @ J_gyro + 0.25 * (J_acc.T @ J_acc)
    if len(arrays["ground_normal"]) > 0:
        n_mean = np.mean(arrays["ground_normal"], axis=0)
        J_gnd = _skew(n_mean)
        H_rot += float(len(arrays["ground_normal"])) * 0.05 * (J_gnd.T @ J_gnd)
    rot_eigs = np.linalg.eigvalsh(H_rot)
    rot_min_eig = float(rot_eigs[0])
    rot_cond = float(rot_eigs[-1] / max(1e-9, rot_eigs[0]))

    # 3b) Translation lever-arm kinematic regressor A_k = [w_L]_x^2 + [alpha_L]_x
    A_blocks = []
    for k in range(20, max(21, n_states - 20)):
        w_L = R_IL @ imu_w_sync[k]
        al_L = R_IL @ imu_alpha[k]
        W = _skew(w_L)
        A_blocks.append(W @ W + _skew(al_L))
    A_stack = np.vstack(A_blocks) if A_blocks else np.zeros((3, 3), dtype=np.float64)
    sv_3d = np.linalg.svd(A_stack, compute_uv=False)
    sv_xy = np.linalg.svd(A_stack[:, :2], compute_uv=False)

    set_boundary = bool(native_conf.get("set_boundary", False))
    bound_th = float(native_conf.get("bound_th", 0.1))
    seed_T_IL = np.array(
        [
            float(native_conf.get("trans_IL_x", 0.0)),
            float(native_conf.get("trans_IL_y", 0.0)),
            float(native_conf.get("trans_IL_z", 0.0)),
        ],
        dtype=np.float64,
    )
    est_T_IL = np.asarray(primary["translation_IL_m"], dtype=np.float64)
    delta_from_seed_abs = np.abs(est_T_IL - seed_T_IL)
    boundary_hit_axes = (
        [
            axis
            for axis, delta in zip(("x", "y", "z"), delta_from_seed_abs)
            if delta >= 0.97 * bound_th
        ]
        if set_boundary and bound_th > 0.0
        else []
    )

    trans_xy_observable = (
        float(sv_xy[-1]) >= thresholds.min_trans_xy_min_sv
        and "x" not in boundary_hit_axes
        and "y" not in boundary_hit_axes
    )
    trans_z_kinematically_observable = (
        float(sv_3d[-1]) >= thresholds.min_trans_3d_min_sv
        and "z" not in boundary_hit_axes
    )

    # 3c) Time-offset observability (angular acceleration energy + bounded Pearson peak)
    alpha_energy = float(np.sum(imu_alpha[20 : max(21, n_states - 20)] ** 2) * dt_med)
    lid_w_norm = np.linalg.norm(_zero_phase_smooth(arrays["lid_w"]), axis=1)
    di = omega_norm - np.mean(omega_norm)
    dl = lid_w_norm - np.mean(lid_w_norm)
    denom = float(np.sqrt(np.sum(di**2) * np.sum(dl**2)))
    max_lag_samples = min(max(1, n_states // 4), int(round(0.5 / max(1e-3, dt_med))))
    best_pearson = 0.0
    best_lag_s = 0.0
    if denom > 1e-12 and n_states > 40:
        best_pearson = -2.0
        for lag in range(-max_lag_samples, max_lag_samples + 1):
            if lag < 0:
                a_seg, b_seg = di[-lag:], dl[: n_states + lag]
            elif lag > 0:
                a_seg, b_seg = di[: n_states - lag], dl[lag:]
            else:
                a_seg, b_seg = di, dl
            d_sub = float(np.sqrt(np.sum(a_seg**2) * np.sum(b_seg**2)))
            p_val = float(np.sum(a_seg * b_seg) / d_sub) if d_sub > 1e-12 else 0.0
            if p_val > best_pearson:
                best_pearson = p_val
                best_lag_s = -lag * dt_med

    time_observable = (
        alpha_energy >= thresholds.min_time_alpha_energy
        and best_pearson >= thresholds.min_xcorr_peak_pearson
        and abs(primary["time_offset_s"]) <= thresholds.max_abs_time_offset_s
    )

    observability = {
        "rotation": {
            "status": (
                "observable"
                if rot_min_eig >= thresholds.min_rot_hessian_min_eig
                else "unobservable"
            ),
            "hessian_eigenvalues": [float(x) for x in rot_eigs],
            "condition_number": rot_cond,
        },
        "translation": {
            "xy_status": (
                "observable" if trans_xy_observable else "unobservable_or_clamped"
            ),
            "z_status": (
                "observable_3d"
                if trans_z_kinematically_observable
                else "prior_coupled_planar_motion"
            ),
            "singular_values_3d": [float(x) for x in sv_3d],
            "singular_values_xy": [float(x) for x in sv_xy],
            "set_boundary": set_boundary,
            "bound_th_m": bound_th,
            "boundary_hit_axes": boundary_hit_axes,
        },
        "time_offset": {
            "status": (
                "observable" if time_observable else "unobservable_or_out_of_bounds"
            ),
            "angular_acc_energy_rad2_s3": alpha_energy,
            "bounded_xcorr_peak_pearson": best_pearson,
            "bounded_xcorr_lag_s": best_lag_s,
            "estimated_time_offset_s": primary["time_offset_s"],
        },
    }

    # 4. Repeatability Gate
    repeatability: dict[str, Any] = {"provided": False, "passed": False}
    if repeat_run_dirs:
        all_runs = [primary] + [
            parse_detailed_gril_result(Path(d) / "GRIL_Calib_result.txt")
            for d in repeat_run_dirs
        ]
        max_rot_spread = 0.0
        max_trans_spread = 0.0
        max_time_spread = 0.0
        for i in range(len(all_runs)):
            for j in range(i + 1, len(all_runs)):
                R_i = np.asarray(all_runs[i]["rotation_LI_matrix"])
                R_j = np.asarray(all_runs[j]["rotation_LI_matrix"])
                T_i = np.asarray(all_runs[i]["translation_LI_m"])
                T_j = np.asarray(all_runs[j]["translation_LI_m"])
                max_rot_spread = max(max_rot_spread, _rotation_error_deg(R_i, R_j))
                max_trans_spread = max(
                    max_trans_spread, float(np.linalg.norm(T_i - T_j))
                )
                max_time_spread = max(
                    max_time_spread,
                    abs(all_runs[i]["time_offset_s"] - all_runs[j]["time_offset_s"]),
                )
        rep_passed = (
            max_rot_spread <= thresholds.max_repeat_rot_deg
            and max_trans_spread <= thresholds.max_repeat_trans_m
            and max_time_spread <= thresholds.max_repeat_time_s
        )
        repeatability = {
            "provided": True,
            "run_count": len(all_runs),
            "max_rotation_spread_deg": max_rot_spread,
            "max_translation_spread_m": max_trans_spread,
            "max_time_offset_spread_s": max_time_spread,
            "passed": rep_passed,
        }

    # 4b. Seed Sensitivity Gate (optional multi-seed basin check)
    seed_sensitivity: dict[str, Any] = {"provided": False, "passed": False}
    if seed_run_dirs:
        seed_runs = [primary] + [
            parse_detailed_gril_result(Path(d) / "GRIL_Calib_result.txt")
            for d in seed_run_dirs
        ]
        max_s_rot = 0.0
        max_s_txy = 0.0
        max_s_t3d = 0.0
        max_s_time = 0.0
        for i in range(len(seed_runs)):
            for j in range(i + 1, len(seed_runs)):
                R_i = np.asarray(seed_runs[i]["rotation_LI_matrix"])
                R_j = np.asarray(seed_runs[j]["rotation_LI_matrix"])
                T_i = np.asarray(seed_runs[i]["translation_LI_m"])
                T_j = np.asarray(seed_runs[j]["translation_LI_m"])
                max_s_rot = max(max_s_rot, _rotation_error_deg(R_i, R_j))
                max_s_txy = max(max_s_txy, float(np.linalg.norm(T_i[:2] - T_j[:2])))
                max_s_t3d = max(max_s_t3d, float(np.linalg.norm(T_i - T_j)))
                max_s_time = max(
                    max_s_time,
                    abs(seed_runs[i]["time_offset_s"] - seed_runs[j]["time_offset_s"]),
                )
        seed_passed = (
            max_s_rot <= thresholds.max_seed_rot_deg
            and max_s_txy <= thresholds.max_seed_trans_xy_m
            and max_s_time <= thresholds.max_seed_time_s
        )
        seed_sensitivity = {
            "provided": True,
            "seed_count": len(seed_runs),
            "max_rotation_spread_deg": max_s_rot,
            "max_translation_xy_spread_m": max_s_txy,
            "max_translation_3d_spread_m": max_s_t3d,
            "max_time_offset_spread_s": max_s_time,
            "passed": seed_passed,
        }

    # 5. Multi-Window Consistency Gate
    multi_window: dict[str, Any] = {"provided": False, "passed": False}
    if window_run_dirs:
        win_runs = [primary] + [
            parse_detailed_gril_result(Path(d) / "GRIL_Calib_result.txt")
            for d in window_run_dirs
        ]
        max_w_rot = 0.0
        max_w_txy = 0.0
        max_w_t3d = 0.0
        max_w_time = 0.0
        for i in range(len(win_runs)):
            for j in range(i + 1, len(win_runs)):
                R_i = np.asarray(win_runs[i]["rotation_LI_matrix"])
                R_j = np.asarray(win_runs[j]["rotation_LI_matrix"])
                T_i = np.asarray(win_runs[i]["translation_LI_m"])
                T_j = np.asarray(win_runs[j]["translation_LI_m"])
                max_w_rot = max(max_w_rot, _rotation_error_deg(R_i, R_j))
                max_w_txy = max(max_w_txy, float(np.linalg.norm(T_i[:2] - T_j[:2])))
                max_w_t3d = max(max_w_t3d, float(np.linalg.norm(T_i - T_j)))
                max_w_time = max(
                    max_w_time,
                    abs(win_runs[i]["time_offset_s"] - win_runs[j]["time_offset_s"]),
                )
        win_passed = (
            max_w_rot <= thresholds.max_window_rot_deg
            and max_w_txy <= thresholds.max_window_trans_xy_m
            and max_w_time <= thresholds.max_window_time_s
        )
        multi_window = {
            "provided": True,
            "window_count": len(win_runs),
            "max_rotation_spread_deg": max_w_rot,
            "max_translation_xy_spread_m": max_w_txy,
            "max_translation_3d_spread_m": max_w_t3d,
            "max_time_offset_spread_s": max_w_time,
            "passed": win_passed,
        }

    # 6. Independent Holdout Gate
    holdout: dict[str, Any] = {"provided": False, "passed": False}
    if holdout_batch_trace is not None:
        h_metrics = evaluate_holdout_trace(primary, Path(holdout_batch_trace))
        h_passed = (
            h_metrics["gyro_rmse_rad_s"] <= thresholds.max_holdout_gyro_rmse_rad_s
            and h_metrics["acc_rmse_m_s2"] <= thresholds.max_holdout_acc_rmse_m_s2
            and h_metrics["gyro_rmse_rad_s"]
            <= h_metrics["gyro_rmse_identity_rad_s"] + 1e-4
            and h_metrics["lever_arm_improved_holdout_acc"]
        )
        holdout = {"provided": True, **h_metrics, "passed": h_passed}

    # Explicit state classification
    hard_failures: list[str] = []
    unobservable_reasons: list[str] = []

    for name, ok in data_checks.items():
        if not ok:
            hard_failures.append(f"data_sufficiency.{name}")
    for name, ok in excitation_checks.items():
        if not ok:
            unobservable_reasons.append(f"motion_excitation.{name}")
    if observability["rotation"]["status"] != "observable":
        unobservable_reasons.append("parameter_observability.rotation")
    if observability["time_offset"]["status"] != "observable":
        if abs(primary["time_offset_s"]) > thresholds.max_abs_time_offset_s:
            hard_failures.append("parameter_observability.time_offset_out_of_bounds")
        else:
            unobservable_reasons.append("parameter_observability.time_offset")
    if observability["translation"]["xy_status"] != "observable":
        unobservable_reasons.append("parameter_observability.translation_xy")
    if boundary_hit_axes:
        axes_tag = "".join(boundary_hit_axes)
        unobservable_reasons.append(
            f"parameter_observability.translation_boundary_hit_{axes_tag}"
        )

    if repeatability["provided"] and not repeatability["passed"]:
        hard_failures.append("repeatability.spread_exceeded")
    if not repeatability["provided"]:
        unobservable_reasons.append("repeatability.not_provided")

    if seed_sensitivity["provided"] and not seed_sensitivity["passed"]:
        hard_failures.append("seed_sensitivity.basin_divergence")

    if multi_window["provided"] and not multi_window["passed"]:
        hard_failures.append("multi_window.inconsistent_extrinsics")
    if not multi_window["provided"]:
        unobservable_reasons.append("multi_window.not_provided")

    if holdout["provided"] and not holdout["passed"]:
        hard_failures.append("holdout.residual_failed")
    if not holdout["provided"]:
        unobservable_reasons.append("holdout.not_provided")

    if hard_failures:
        state = "failed"
        verdict = "rejected"
        release_ready = False
    elif unobservable_reasons:
        state = "degraded_unobservable"
        verdict = "review_only"
        release_ready = False
    else:
        state = "converged_and_verified"
        verdict = "accepted"
        release_ready = True

    notes: list[str] = []
    if observability["translation"]["z_status"] == "prior_coupled_planar_motion":
        notes.append(
            "Flat-road planar vehicle motion leaves lever-arm Z coupled to "
            "Patchwork++ ground height and imu_sensor_height prior; treat T_LI.z "
            "as prior-anchored unless 3D pitch/roll excitation is captured."
        )

    return {
        "schema_version": 1,
        "module": "lidar2imu_gril_quality_gates",
        "run_dir": str(run_dir.resolve()),
        "state": state,
        "verdict": verdict,
        "release_ready": release_ready,
        "hard_failures": hard_failures,
        "unobservable_or_missing_gates": unobservable_reasons,
        "notes": notes,
        "result": primary,
        "pillars": {
            "data_sufficiency": {
                "passed": all(data_checks.values()),
                "checks": data_checks,
                "metrics": {
                    "lidar_states": n_states,
                    "accum_duration_s": duration_s,
                    "max_odom_step_m": max_step_m,
                    "p95_odom_tilt_deg": p95_tilt_deg,
                    "degenerate_skip_ratio": deg_skip_ratio,
                    "frontend": frontend,
                },
            },
            "motion_excitation": {
                "passed": all(excitation_checks.values()),
                "checks": excitation_checks,
                "metrics": {
                    "planar_path_length_m": path_length_m,
                    "total_abs_yaw_deg": total_abs_yaw_deg,
                    "left_turn_yaw_deg": left_turn_deg,
                    "right_turn_yaw_deg": right_turn_deg,
                    "roll_pitch_excitation_deg": roll_pitch_exc_deg,
                    "p95_omega_rad_s": p95_omega,
                    "rms_alpha_rad_s2": rms_alpha,
                    "acc_xy_std_m_s2": acc_xy_std,
                },
            },
            "parameter_observability": observability,
        },
        "repeatability": repeatability,
        "seed_sensitivity": seed_sensitivity,
        "multi_window": multi_window,
        "holdout": holdout,
    }


def write_quality_gate_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(report, sort_keys=False))
    return output_path
