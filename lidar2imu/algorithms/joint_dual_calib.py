#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created Date: 2026-11-20
# Author: daohu527

"""Joint spatiotemporal calibration combining 2x LiDAR-to-IMU (L2I) and
upgraded LiDAR-to-LiDAR (L2L) constraints across three staged levels:

- v1: Anisotropic SE(3)^2 factor-graph fusion of L2I info, multi-window GICP L2L,
      multi-heading gravity-ground alignment, and explicit z_cm nullspace prior.
- v2: Upgrades L2L edges with 4-channel geometric primitives, range/incidence-
      calibrated intensity, ego-occlusion-aware 3D shared FoV, and local info.
- v3: Turn-window 100 Hz pure-IMU preintegrated cross-time multi-LiDAR plane BA
      refining common-mode lever-arm and continuous-time offsets (tau_A, tau_B).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from lidar2imu.evaluation.dual_sensor_diagnostics import (
    FastCanonicalScanAccess,
    _filter_range,
    _fit_ground_plane_ransac,
)
from lidar2imu.evaluation.quality_gates import (
    _interp_vec3,
    _project_so3,
    _rotation_error_deg,
    _skew,
    _zero_phase_smooth,
    parse_batch_trace_arrays,
    parse_detailed_gril_result,
)
from lidar2lidar.extrinsic_io import extrinsics_filename, save_extrinsics_yaml
from lidar2lidar.lie import information_left_to_right, se3_log6, so3_right_jacobian
from lidar2lidar.semantic_registration import (
    LidarFrameData,
    PrimitiveExtractionConfig,
    SemanticRegistrationConfig,
    VisibilityConfig,
    extract_geometric_primitives,
    register_semantic_l2l,
)


@dataclass(frozen=True)
class JointCalibrationConfig:
    max_static_windows: int = 5
    static_stack_half_scans: int = 3
    min_static_shared_cells: int = 10
    static_screening_cell_m: float = 1.5
    min_static_temporal_consistency: float = 0.62
    enable_low_speed_submap_fallback: bool = True
    max_sync_dt_ms: float = 60.0
    max_turn_windows: int = 4
    turn_duration_s: float = 2.6
    min_turn_separation_s: float = 12.0
    min_turn_yaw_deg: float = 18.0
    turn_point_stride: int = 2
    z_cm_prior_m: float | None = None
    z_cm_prior_sigma_m: float = 0.04
    l2i_xy_prior_sigma_m: float = 0.45
    l2i_rot_prior_sigma_deg: float = 0.75
    l2i_yaw_prior_sigma_deg: float = 1.05
    gravity_sigma_deg: float = 0.65
    ego_bbox_min_imu: tuple[float, float, float] | None = None
    ego_bbox_max_imu: tuple[float, float, float] | None = None
    v3_outer_iters: int = 5
    v3_voxel_schedule_m: tuple[float, ...] = (0.85, 0.75, 0.65, 0.60, 0.55)
    v3_plane_sigma_m: float = 0.06
    v3_gyro_sigma_rad_s: float = 0.025
    v3_tau_prior_sigma_s: float = 0.005
    v3_dtau_prior_sigma_s: float = 0.0008
    v3_z_lock_sigma_m: float = 0.002
    gate_min_side_turn_deg: float = 90.0
    gate_min_turn_symmetry_ratio: float = 0.25
    gate_min_p95_planar_acc_m_s2: float = 0.50
    gate_min_max_abs_wz_rad_s: float = 0.32
    gate_min_p99_rp_omega_rad_s: float = 0.30


def evaluate_common_mode_excitation_gate(
    batch: dict[str, Any],
    l2l_edges: list[dict[str, Any]],
    cfg: JointCalibrationConfig = JointCalibrationConfig(),
) -> dict[str, Any]:
    """Post-run common-mode excitation & L2L health gate.

    Distinguishes well-excited bidirectional runs (e.g. S3, Full_0_680s) from
    one-sided / low-acceleration segments (e.g. S1 zero-left-turn, S4 weak-right-turn)
    where common-mode Yaw and horizontal lever-arm can drift without altering L2L.
    """
    lid_t = batch["lid_t"]
    imu_t = batch["imu_t"]
    imu_w = batch["imu_w"]
    imu_a = batch["imu_a"]
    if len(lid_t) < 20 or len(imu_t) < 20:
        return {
            "verdict": "needs_more_excitation",
            "z_cm_observable": False,
            "reasons": ["insufficient_samples"],
            "metrics": {},
        }

    # Restrict strictly to the active LiDAR span so full-file IMU tails never leak
    t_start = float(lid_t[0]) - 2.0
    t_end = float(lid_t[-1]) + 2.0
    m = (imu_t >= t_start) & (imu_t <= t_end)
    if int(np.sum(m)) < 20:
        m = np.ones(len(imu_t), dtype=bool)
    ts = imu_t[m]
    ws = imu_w[m]
    acc = imu_a[m]

    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    wz = ws[:, 2]
    left_deg = float(np.degrees(np.sum(wz[wz > 0.08]) * dt))
    right_deg = float(np.degrees(np.sum(-wz[wz < -0.08]) * dt))
    min_side = min(left_deg, right_deg)
    max_side = max(left_deg, right_deg)
    sym_ratio = float(min_side / max(max_side, 1e-6))

    # 0.3s smoothed dynamic planar acceleration (demeaned per segment)
    ax_dyn = acc[:, 0] - float(np.median(acc[:, 0]))
    ay_dyn = acc[:, 1] - float(np.median(acc[:, 1]))
    win_len = min(31, max(3, (len(ts) // 5) | 1))
    a_sm = _zero_phase_smooth(np.column_stack([ax_dyn, ay_dyn]), window=win_len)
    p95_ax = float(np.percentile(np.abs(a_sm[:, 0]), 95))
    p95_ay = float(np.percentile(np.abs(a_sm[:, 1]), 95))
    p95_planar_acc = max(p95_ax, p95_ay)
    max_abs_wz = float(np.max(np.abs(wz)))
    p99_rp = float(np.percentile(np.hypot(ws[:, 0], ws[:, 1]), 99))

    healthy_l2l = sum(
        1
        for e in l2l_edges
        if not e.get("downweighted_by_weak_support", False)
        and not e.get("consensus_outlier", False)
        and not e.get("downweighted_by_gravity_gate", False)
    )

    reasons: list[str] = []
    if min_side < cfg.gate_min_side_turn_deg:
        reasons.append(
            f"insufficient_bidirectional_turns("
            f"left={left_deg:.1f}deg,right={right_deg:.1f}deg)"
        )
    if sym_ratio < cfg.gate_min_turn_symmetry_ratio:
        reasons.append(
            f"asymmetric_turn_ratio("
            f"{sym_ratio:.3f}<{cfg.gate_min_turn_symmetry_ratio:.2f})"
        )
    if p95_planar_acc < cfg.gate_min_p95_planar_acc_m_s2:
        reasons.append(
            f"low_planar_acceleration("
            f"p95={p95_planar_acc:.3f}<{cfg.gate_min_p95_planar_acc_m_s2:.2f})"
        )
    if max_abs_wz < cfg.gate_min_max_abs_wz_rad_s:
        reasons.append(
            f"low_peak_yaw_rate("
            f"max_wz={max_abs_wz:.3f}<{cfg.gate_min_max_abs_wz_rad_s:.2f})"
        )
    if healthy_l2l < 1:
        reasons.append("no_healthy_l2l_edge")

    z_cm_obs = bool(p99_rp >= cfg.gate_min_p99_rp_omega_rad_s)
    verdict = "accepted" if not reasons else "needs_more_excitation"
    return {
        "verdict": verdict,
        "z_cm_observable": z_cm_obs,
        "z_cm_status": "observable" if z_cm_obs else "prior_anchored_planar_nullspace",
        "reasons": reasons,
        "metrics": {
            "left_turn_deg": round(left_deg, 2),
            "right_turn_deg": round(right_deg, 2),
            "turn_symmetry_ratio": round(sym_ratio, 4),
            "p95_planar_acc_m_s2": round(p95_planar_acc, 4),
            "max_abs_wz_rad_s": round(max_abs_wz, 4),
            "p99_rp_omega_rad_s": round(p99_rp, 4),
            "healthy_l2l_edges": int(healthy_l2l),
        },
    }


def _se3_from_rt(R_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _project_so3(R_mat)
    mat[:3, 3] = np.asarray(t_vec, dtype=np.float64)
    return mat


def _se3_log6(T: np.ndarray) -> np.ndarray:
    """Return SE(3) log coordinates [rotation, translation] in the right tangent."""
    return se3_log6(T)


def _tangent_basis_s2(unit_vec: np.ndarray) -> np.ndarray:
    u = np.asarray(unit_vec, dtype=np.float64)
    u = u / max(1e-12, float(np.linalg.norm(u)))
    ref = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(u[0]) < 0.8
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    b1 = np.cross(u, ref)
    b1 /= max(1e-12, float(np.linalg.norm(b1)))
    b2 = np.cross(u, b1)
    b2 /= max(1e-12, float(np.linalg.norm(b2)))
    return np.column_stack([b1, b2])  # (3, 2)


def _safe_cholesky_whitener(H: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    H_sym = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    eigvals_clamped = np.maximum(eigvals, min_eig)
    return np.diag(np.sqrt(eigvals_clamped)) @ eigvecs.T


def _stack_static_frame_with_consistency(
    access: FastCanonicalScanAccess,
    center_idx: int,
    half_scans: int,
) -> tuple[LidarFrameData, float]:
    xyz_list: list[np.ndarray] = []
    int_list: list[np.ndarray] = []
    ring_list: list[np.ndarray] = []
    scan_indices: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []
    early_pts: list[np.ndarray] = []
    late_pts: list[np.ndarray] = []
    has_int = True
    has_ring = True

    for idx in range(
        max(0, center_idx - half_scans),
        min(len(access.scan_t_s), center_idx + half_scans + 1),
    ):
        xyz, _, inten, rng = access.read_scan_with_attributes(idx)
        if len(xyz) == 0:
            continue
        r_xy = np.linalg.norm(xyz[:, :2], axis=1)
        keep = (r_xy >= 2.0) & (r_xy <= 48.0) & np.isfinite(xyz).all(axis=1)
        if not np.any(keep):
            continue
        pts_k = xyz[keep]
        xyz_list.append(pts_k)
        scan_indices.append(np.full(len(pts_k), idx, dtype=np.int64))
        point_indices.append(np.flatnonzero(keep))
        if idx < center_idx:
            early_pts.append(pts_k)
        elif idx > center_idx:
            late_pts.append(pts_k)
        if inten is not None and len(inten) == len(xyz):
            int_list.append(inten[keep])
        else:
            has_int = False
        if rng is not None and len(rng) == len(xyz):
            ring_list.append(rng[keep])
        else:
            has_ring = False

    if not xyz_list:
        return LidarFrameData(points=np.zeros((0, 3), dtype=np.float64)), 0.0

    consistency = 1.0
    if early_pts and late_pts:
        pe = np.vstack(early_pts)
        pl = np.vstack(late_pts)
        ve = set(map(tuple, np.floor(pe / 0.75).astype(int)))
        vl = set(map(tuple, np.floor(pl / 0.75).astype(int)))
        if ve and vl:
            consistency = float(len(ve & vl) / max(1, min(len(ve), len(vl))))

    return (
        LidarFrameData(
            points=np.vstack(xyz_list),
            intensity=np.concatenate(int_list) if has_int and int_list else None,
            ring=np.concatenate(ring_list) if has_ring and ring_list else None,
            scan_indices=np.concatenate(scan_indices),
            point_indices=np.concatenate(point_indices),
        ),
        consistency,
    )


def _stack_motion_compensated_frame_with_consistency(
    access: FastCanonicalScanAccess,
    batch: dict[str, Any],
    center_idx: int,
    tau_s: float,
    half_scans: int = 2,
) -> tuple[LidarFrameData, float]:
    """Stack low-speed driving scans into center scan frame using batch odometry."""
    lid_t = batch["lid_t"]
    if len(lid_t) < 2:
        return LidarFrameData(points=np.zeros((0, 3), dtype=np.float64)), 0.0

    def _pose_at(t_phys: float) -> tuple[np.ndarray, np.ndarray]:
        k = int(np.clip(np.searchsorted(lid_t + tau_s, t_phys), 0, len(lid_t) - 1))
        if 0 < k < len(lid_t):
            t0 = float(lid_t[k - 1] + tau_s)
            t1 = float(lid_t[k] + tau_s)
            a = float(np.clip((t_phys - t0) / max(1e-6, t1 - t0), 0.0, 1.0))
            R0 = batch["lid_R"][k - 1]
            R1 = batch["lid_R"][k]
            dR = Rot.from_matrix(_project_so3(R0.T @ R1)).as_rotvec()
            Rm = R0 @ Rot.from_rotvec(a * dR).as_matrix()
            pm = (1.0 - a) * batch["lid_p"][k - 1] + a * batch["lid_p"][k]
            return Rm, pm
        return batch["lid_R"][k], batch["lid_p"][k]

    t_center = float(access.scan_t_s[center_idx] + tau_s)
    R_c, p_c = _pose_at(t_center)

    xyz_list: list[np.ndarray] = []
    int_list: list[np.ndarray] = []
    ring_list: list[np.ndarray] = []
    scan_indices: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []
    early_pts: list[np.ndarray] = []
    late_pts: list[np.ndarray] = []
    has_int = True
    has_ring = True

    for idx in range(
        max(0, center_idx - half_scans),
        min(len(access.scan_t_s), center_idx + half_scans + 1),
    ):
        xyz, _, inten, rng = access.read_scan_with_attributes(idx)
        if len(xyz) == 0:
            continue
        r_xy = np.linalg.norm(xyz[:, :2], axis=1)
        keep = (r_xy >= 2.0) & (r_xy <= 48.0) & np.isfinite(xyz).all(axis=1)
        if not np.any(keep):
            continue
        R_i, p_i = _pose_at(float(access.scan_t_s[idx] + tau_s))
        dR_ci = R_c.T @ R_i
        dp_ci = R_c.T @ (p_i - p_c)
        pts_k = xyz[keep] @ dR_ci.T + dp_ci
        xyz_list.append(pts_k)
        scan_indices.append(np.full(len(pts_k), idx, dtype=np.int64))
        point_indices.append(np.flatnonzero(keep))
        if idx < center_idx:
            early_pts.append(pts_k)
        elif idx > center_idx:
            late_pts.append(pts_k)
        if inten is not None and len(inten) == len(xyz):
            int_list.append(inten[keep])
        else:
            has_int = False
        if rng is not None and len(rng) == len(xyz):
            ring_list.append(rng[keep])
        else:
            has_ring = False

    if not xyz_list:
        return LidarFrameData(points=np.zeros((0, 3), dtype=np.float64)), 0.0

    consistency = 1.0
    if early_pts and late_pts:
        pe = np.vstack(early_pts)
        pl = np.vstack(late_pts)
        ve = set(map(tuple, np.floor(pe / 0.75).astype(int)))
        vl = set(map(tuple, np.floor(pl / 0.75).astype(int)))
        if ve and vl:
            consistency = float(len(ve & vl) / max(1, min(len(ve), len(vl))))

    return (
        LidarFrameData(
            points=np.vstack(xyz_list),
            intensity=np.concatenate(int_list) if has_int and int_list else None,
            ring=np.concatenate(ring_list) if has_ring and ring_list else None,
            scan_indices=np.concatenate(scan_indices),
            point_indices=np.concatenate(point_indices),
        ),
        consistency,
    )


def _stack_static_frame(
    access: FastCanonicalScanAccess,
    center_idx: int,
    half_scans: int,
) -> LidarFrameData:
    frame, _ = _stack_static_frame_with_consistency(access, center_idx, half_scans)
    return frame


def _build_preintegrated_imu_table(
    imu_t: np.ndarray,
    imu_w: np.ndarray,
    imu_a: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    gyro_bias: np.ndarray,
    acc_bias: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = (imu_t >= t_start - 0.35) & (imu_t <= t_end + 0.35)
    ts = imu_t[mask]
    ws = imu_w[mask] - gyro_bias
    fs = imu_a[mask] - acc_bias
    if len(ts) < 6:
        raise RuntimeError("Insufficient IMU samples inside turn window")

    idx0 = int(np.argmin(np.abs(ts - t_start)))
    t0 = float(ts[idx0])
    n = len(ts)
    rots = np.zeros((n, 3, 3), dtype=np.float64)
    rots[0] = np.eye(3)
    for k in range(n - 1):
        dt = float(ts[k + 1] - ts[k])
        w_mid = 0.5 * (ws[k] + ws[k + 1])
        rots[k + 1] = rots[k] @ Rot.from_rotvec(w_mid * dt).as_matrix()

    R_anchor = rots[idx0].copy()
    rots = np.einsum("ij,njk->nik", R_anchor.T, rots)

    acc_world = np.einsum("nij,nj->ni", rots, fs)
    vel_spec = np.zeros((n, 3), dtype=np.float64)
    for k in range(n - 1):
        dt = float(ts[k + 1] - ts[k])
        vel_spec[k + 1] = vel_spec[k] + 0.5 * (acc_world[k] + acc_world[k + 1]) * dt
    vel_spec -= vel_spec[idx0]

    pos_spec = np.zeros((n, 3), dtype=np.float64)
    for k in range(n - 1):
        dt = float(ts[k + 1] - ts[k])
        pos_spec[k + 1] = pos_spec[k] + 0.5 * (vel_spec[k] + vel_spec[k + 1]) * dt
    pos_spec -= pos_spec[idx0]

    return {
        "t": ts,
        "t0": np.array([t0], dtype=np.float64),
        "rots": rots,
        "vel_spec": vel_spec,
        "pos_spec": pos_spec,
        "omega": ws,
    }


def _query_imu_preint(
    table: dict[str, np.ndarray],
    t_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuous-time SE(3) preintegration lookup with geodesic SO(3) interp."""
    ts = table["t"]
    t0 = float(table["t0"][0])
    t_clamped = np.clip(t_query, ts[0] + 1e-6, ts[-1] - 1e-6)
    idx = np.clip(np.searchsorted(ts, t_clamped, side="right") - 1, 0, len(ts) - 2)
    dt_seg = np.maximum(ts[idx + 1] - ts[idx], 1e-9)
    alpha = (t_clamped - ts[idx]) / dt_seg

    w_mid = 0.5 * (table["omega"][idx] + table["omega"][idx + 1])
    dR = Rot.from_rotvec(w_mid * (alpha * dt_seg)[:, None]).as_matrix()
    dR_q = np.einsum("nij,njk->nik", table["rots"][idx], dR)

    v0 = table["vel_spec"][idx]
    v1 = table["vel_spec"][idx + 1]
    dv_q = (1.0 - alpha)[:, None] * v0 + alpha[:, None] * v1

    p0 = table["pos_spec"][idx]
    p1 = table["pos_spec"][idx + 1]
    dp_q = (
        (1.0 - alpha)[:, None] * p0
        + alpha[:, None] * p1
        + 0.5 * (alpha * (alpha - 1.0) * dt_seg)[:, None] * (v1 - v0)
    )
    w_q = (1.0 - alpha)[:, None] * table["omega"][idx] + alpha[:, None] * table[
        "omega"
    ][idx + 1]
    dt_from_t0 = t_query - t0
    return dR_q, dp_q, dv_q, w_q, dt_from_t0


def screen_joint_calibration_windows(
    access_a: FastCanonicalScanAccess,
    access_b: FastCanonicalScanAccess,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    cfg: JointCalibrationConfig = JointCalibrationConfig(),
    *,
    include_turn_windows: bool = True,
    feature_rejections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage 1: Screen static L2L, multi-heading gravity, and turn-BA windows."""
    R_LI_a = _project_so3(np.asarray(result_a["rotation_LI_matrix"], dtype=np.float64))
    T_LI_a = np.asarray(result_a["translation_LI_m"], dtype=np.float64)
    T_I_A = _se3_from_rt(R_LI_a, T_LI_a)
    tau_a = float(result_a["time_offset_s"])

    R_LI_b = _project_so3(np.asarray(result_b["rotation_LI_matrix"], dtype=np.float64))
    T_LI_b = np.asarray(result_b["translation_LI_m"], dtype=np.float64)
    T_I_B = _se3_from_rt(R_LI_b, T_LI_b)
    tau_b = float(result_b["time_offset_s"])

    T_AB_init = np.linalg.inv(T_I_A) @ T_I_B

    # 1. Static L2L candidate windows
    lid_t = batch_a["lid_t"]
    t0_ref = float(access_a.scan_t_s[0])
    cand_times: list[float] = []
    if len(lid_t) > 0 and (lid_t[0] - t0_ref) > 4.0:
        cand_times.append(t0_ref + 0.45 * (float(lid_t[0]) - t0_ref))
    if len(lid_t) > 15:
        speed = np.linalg.norm(batch_a["lid_v"], axis=1)
        omega = np.linalg.norm(batch_a["lid_w"], axis=1)
        quiet = np.flatnonzero((speed <= 0.12) & (omega <= 0.045))
        order = quiet[np.argsort(speed[quiet] + 0.5 * omega[quiet])]
        for idx in order:
            t_val = float(lid_t[int(idx)])
            if all(abs(t_val - prev) > 15.0 for prev in cand_times):
                cand_times.append(t_val)
            if len(cand_times) >= cfg.max_static_windows * 2:
                break
    if not cand_times:
        quiet_i = int(np.argmin(np.linalg.norm(access_a.imu_w, axis=1)))
        cand_times.append(float(access_a.imu_t_s[quiet_i]))

    static_windows: list[dict[str, Any]] = []
    static_screening_log: list[dict[str, Any]] = []

    def retain_rejection(
        reason: str,
        frame_a: LidarFrameData | None = None,
        frame_b: LidarFrameData | None = None,
    ) -> None:
        if feature_rejections is not None:
            feature_rejections.append(
                {
                    "metadata": {
                        "window_id": f"rejected_{len(feature_rejections) + 1:03d}",
                        "rel_time_s": float(t_phys - t0_ref),
                        "scan_index_a": int(ia),
                        "scan_index_b": int(ib),
                        "sync_dt_ms": sync_dt_ms,
                        "accepted": False,
                        "skip_reason": reason,
                    },
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                }
            )

    for t_cand in cand_times:
        ia = int(np.argmin(np.abs((access_a.scan_t_s + tau_a) - t_cand)))
        t_phys = float(access_a.scan_t_s[ia] + tau_a)
        ib = int(np.argmin(np.abs((access_b.scan_t_s + tau_b) - t_phys)))
        sync_dt_ms = float(((access_b.scan_t_s[ib] + tau_b) - t_phys) * 1e3)
        if abs(sync_dt_ms) > cfg.max_sync_dt_ms:
            static_screening_log.append(
                {
                    "rel_time_s": float(t_phys - t0_ref),
                    "accepted": False,
                    "skip_reason": "sync_dt_exceeded",
                    "sync_dt_ms": sync_dt_ms,
                }
            )
            retain_rejection("sync_dt_exceeded")
            continue

        frame_a, cons_a = _stack_static_frame_with_consistency(
            access_a, ia, cfg.static_stack_half_scans
        )
        frame_b, cons_b = _stack_static_frame_with_consistency(
            access_b, ib, cfg.static_stack_half_scans
        )
        temporal_consistency = min(cons_a, cons_b)
        if len(frame_a.points) < 120 or len(frame_b.points) < 120:
            static_screening_log.append(
                {
                    "rel_time_s": float(t_phys - t0_ref),
                    "accepted": False,
                    "skip_reason": "insufficient_points",
                }
            )
            retain_rejection("insufficient_points", frame_a, frame_b)
            continue
        if temporal_consistency < cfg.min_static_temporal_consistency:
            static_screening_log.append(
                {
                    "rel_time_s": float(t_phys - t0_ref),
                    "accepted": False,
                    "skip_reason": "dynamic_traffic_contamination",
                    "temporal_static_consistency": float(temporal_consistency),
                }
            )
            retain_rejection("dynamic_traffic_contamination", frame_a, frame_b)
            continue

        pa_imu = frame_a.points @ R_LI_a.T + T_LI_a
        pb_imu = frame_b.points @ R_LI_b.T + T_LI_b
        z_hint = float(np.percentile(np.concatenate([pa_imu[:, 2], pb_imu[:, 2]]), 12))
        cell_sz = float(cfg.static_screening_cell_m)
        va = set(
            map(
                tuple,
                np.floor(pa_imu[pa_imu[:, 2] > z_hint + 0.45, :2] / cell_sz).astype(
                    int
                ),
            )
        )
        vb = set(
            map(
                tuple,
                np.floor(pb_imu[pb_imu[:, 2] > z_hint + 0.45, :2] / cell_sz).astype(
                    int
                ),
            )
        )
        shared_cells = len(va & vb)
        if shared_cells < cfg.min_static_shared_cells:
            static_screening_log.append(
                {
                    "rel_time_s": float(t_phys - t0_ref),
                    "accepted": False,
                    "skip_reason": "insufficient_shared_vertical_cells",
                    "shared_cells": int(shared_cells),
                }
            )
            retain_rejection("insufficient_shared_vertical_cells", frame_a, frame_b)
            continue

        static_windows.append(
            {
                "window_id": f"static_{len(static_windows) + 1}",
                "rel_time_s": float(t_phys - t0_ref),
                "scan_index_a": int(ia),
                "scan_index_b": int(ib),
                "sync_dt_ms": sync_dt_ms,
                "shared_vertical_cells_0p5m": int(shared_cells),
                "temporal_static_consistency": float(temporal_consistency),
                "frame_a": frame_a,
                "frame_b": frame_b,
            }
        )
        static_screening_log.append(
            {
                "window_id": static_windows[-1]["window_id"],
                "rel_time_s": float(t_phys - t0_ref),
                "accepted": True,
                "sync_dt_ms": sync_dt_ms,
                "shared_cells": int(shared_cells),
                "temporal_static_consistency": float(temporal_consistency),
            }
        )
        if len(static_windows) >= cfg.max_static_windows:
            break

    # 1b. Low-speed motion-compensated micro-submap fallback if < 2 stationary stops
    if (
        cfg.enable_low_speed_submap_fallback
        and len(static_windows) < 2
        and len(lid_t) > 30
    ):
        speed = np.linalg.norm(batch_a["lid_v"], axis=1)
        omega = np.linalg.norm(batch_a["lid_w"], axis=1)
        smooth_drive = np.flatnonzero(
            (speed > 0.15) & (speed <= 5.5) & (omega <= 0.035)
        )
        # Prefer slower smooth sections spaced apart from existing windows
        existing_t = [t0_ref + w["rel_time_s"] for w in static_windows]
        for idx in smooth_drive[
            np.argsort(speed[smooth_drive] + 10.0 * omega[smooth_drive])
        ]:
            if len(static_windows) >= min(3, cfg.max_static_windows):
                break
            t_cand = float(lid_t[int(idx)])
            if any(abs(t_cand - prev) < 20.0 for prev in existing_t):
                continue
            ia = int(np.argmin(np.abs((access_a.scan_t_s + tau_a) - t_cand)))
            t_phys = float(access_a.scan_t_s[ia] + tau_a)
            ib = int(np.argmin(np.abs((access_b.scan_t_s + tau_b) - t_phys)))
            sync_dt_ms = float(((access_b.scan_t_s[ib] + tau_b) - t_phys) * 1e3)
            if abs(sync_dt_ms) > cfg.max_sync_dt_ms:
                retain_rejection("low_speed_sync_dt_exceeded")
                continue
            frame_a, cons_a = _stack_motion_compensated_frame_with_consistency(
                access_a, batch_a, ia, tau_a, half_scans=2
            )
            frame_b, cons_b = _stack_motion_compensated_frame_with_consistency(
                access_b, batch_b, ib, tau_b, half_scans=2
            )
            temporal_consistency = min(cons_a, cons_b)
            if len(frame_a.points) < 120 or len(frame_b.points) < 120:
                retain_rejection("low_speed_insufficient_points", frame_a, frame_b)
                continue
            if temporal_consistency < 0.58:
                static_screening_log.append(
                    {
                        "rel_time_s": float(t_phys - t0_ref),
                        "accepted": False,
                        "skip_reason": "low_speed_dynamic_contamination",
                        "temporal_static_consistency": float(temporal_consistency),
                    }
                )
                retain_rejection("low_speed_dynamic_contamination", frame_a, frame_b)
                continue
            pa_imu = frame_a.points @ R_LI_a.T + T_LI_a
            pb_imu = frame_b.points @ R_LI_b.T + T_LI_b
            z_hint = float(
                np.percentile(np.concatenate([pa_imu[:, 2], pb_imu[:, 2]]), 12)
            )
            cell_sz = float(cfg.static_screening_cell_m)
            va = set(
                map(
                    tuple,
                    np.floor(pa_imu[pa_imu[:, 2] > z_hint + 0.45, :2] / cell_sz).astype(
                        int
                    ),
                )
            )
            vb = set(
                map(
                    tuple,
                    np.floor(pb_imu[pb_imu[:, 2] > z_hint + 0.45, :2] / cell_sz).astype(
                        int
                    ),
                )
            )
            shared_cells = len(va & vb)
            if shared_cells < max(cfg.min_static_shared_cells, 16):
                retain_rejection(
                    "low_speed_insufficient_shared_vertical_cells", frame_a, frame_b
                )
                continue
            existing_t.append(t_cand)
            static_windows.append(
                {
                    "window_id": f"low_speed_{len(static_windows) + 1}",
                    "rel_time_s": float(t_phys - t0_ref),
                    "scan_index_a": int(ia),
                    "scan_index_b": int(ib),
                    "sync_dt_ms": sync_dt_ms,
                    "shared_vertical_cells_0p5m": int(shared_cells),
                    "temporal_static_consistency": float(temporal_consistency),
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                }
            )
            static_screening_log.append(
                {
                    "window_id": static_windows[-1]["window_id"],
                    "rel_time_s": float(t_phys - t0_ref),
                    "accepted": True,
                    "fallback_low_speed_submap": True,
                    "sync_dt_ms": sync_dt_ms,
                    "shared_cells": int(shared_cells),
                    "temporal_static_consistency": float(temporal_consistency),
                }
            )

    # 2. Multi-heading gravity-ground windows
    ba = 0.5 * (
        np.asarray(result_a["acc_bias_m_s2"], dtype=np.float64)
        + np.asarray(result_b["acc_bias_m_s2"], dtype=np.float64)
    )
    bg = 0.5 * (
        np.asarray(result_a["gyro_bias_rad_s"], dtype=np.float64)
        + np.asarray(result_b["gyro_bias_rad_s"], dtype=np.float64)
    )
    grav_cands: list[tuple[float, float, np.ndarray]] = []
    if len(lid_t) > 0 and (lid_t[0] - t0_ref) > 4.0:
        grav_cands.append(
            (
                t0_ref + 0.5 * float(lid_t[0] - t0_ref),
                0.0,
                np.zeros(3, dtype=np.float64),
            )
        )
    if len(lid_t) > 30:
        dt_med = float(np.median(np.diff(lid_t)))
        v_smooth = _zero_phase_smooth(batch_a["lid_v"], window=9)
        a_world = np.gradient(v_smooth, dt_med, axis=0)
        omega_norm = np.linalg.norm(batch_a["lid_w"], axis=1)
        acc_norm = np.linalg.norm(a_world, axis=1)
        yaws = np.array(
            [
                math.degrees(math.atan2(float(Rm[1, 0]), float(Rm[0, 0])))
                for Rm in batch_a["lid_R"]
            ]
        )
        for bin_deg in range(-180, 180, 30):
            in_bin = np.flatnonzero(
                (yaws >= bin_deg)
                & (yaws < bin_deg + 30)
                & (omega_norm < 0.04)
                & (acc_norm < 0.25)
            )
            if len(in_bin) == 0:
                continue
            best_i = int(in_bin[np.argmin(omega_norm[in_bin] + 0.5 * acc_norm[in_bin])])
            a_local = batch_a["lid_R"][best_i].T @ a_world[best_i]
            grav_cands.append(
                (float(lid_t[best_i]), float(yaws[best_i]), R_LI_a @ a_local)
            )
    if not grav_cands:
        grav_cands.append((t0_ref, 0.0, np.zeros(3, dtype=np.float64)))

    gravity_windows: list[dict[str, Any]] = []
    for t_win, yaw_deg, a_kin_imu in grav_cands:
        ia = int(np.argmin(np.abs((access_a.scan_t_s + tau_a) - t_win)))
        ib = int(np.argmin(np.abs((access_b.scan_t_s + tau_b) - t_win)))
        imu_mask = np.abs(access_a.imu_t_s - t_win) <= 0.45
        if int(imu_mask.sum()) < 5:
            continue
        f_mean = access_a.imu_a[imu_mask].mean(axis=0) - ba
        g_up = f_mean - a_kin_imu
        g_norm = float(np.linalg.norm(g_up))
        if g_norm < 7.0 or g_norm > 12.0:
            continue
        u_g = g_up / g_norm
        if u_g[2] < 0.0:
            u_g = -u_g

        raw_a, _ = access_a.read_scan(ia)
        raw_b, _ = access_b.read_scan(ib)
        pts_a_i, _ = _filter_range(raw_a @ R_LI_a.T + T_LI_a)
        pts_b_i, _ = _filter_range(raw_b @ R_LI_b.T + T_LI_b)
        if len(pts_a_i) < 40 or len(pts_b_i) < 40:
            continue
        z_hint = float(
            np.percentile(np.concatenate([pts_a_i[:, 2], pts_b_i[:, 2]]), 12)
        )
        n_a_i, _, _ = _fit_ground_plane_ransac(pts_a_i, z_hint=z_hint)
        n_b_i, _, _ = _fit_ground_plane_ransac(pts_b_i, z_hint=z_hint)
        # Express normals in each sensor's native frame
        n_a_local = R_LI_a.T @ n_a_i
        n_b_local = R_LI_b.T @ n_b_i
        gravity_windows.append(
            {
                "rel_time_s": float(t_win - t0_ref),
                "heading_yaw_deg": float(yaw_deg),
                "u_g_imu": u_g,
                "n_a_lidar": n_a_local / max(1e-9, float(np.linalg.norm(n_a_local))),
                "n_b_lidar": n_b_local / max(1e-9, float(np.linalg.norm(n_b_local))),
            }
        )

    # 3. High-excitation turn windows for v3 IMU-preintegrated BA
    turn_windows: list[dict[str, Any]] = []
    turn_screening_log: list[dict[str, Any]] = []
    if include_turn_windows and len(lid_t) > 20:
        dt_med = float(np.median(np.diff(lid_t)))
        win_steps = max(
            4, min(len(lid_t) - 1, int(round(cfg.turn_duration_s / dt_med)))
        )
        scored_turns: list[tuple[float, int, float]] = []
        stride = max(1, win_steps // 6)
        for idx in range(0, max(1, len(lid_t) - win_steps), stride):
            dR = batch_a["lid_R"][idx].T @ batch_a["lid_R"][idx + win_steps]
            ang = float(np.degrees(Rot.from_matrix(_project_so3(dR)).magnitude()))
            spd = float(
                np.mean(np.linalg.norm(batch_a["lid_v"][idx : idx + win_steps], axis=1))
            )
            score = ang * (1.0 if spd >= 1.0 else 0.4)
            scored_turns.append((score, idx, ang))
        scored_turns.sort(key=lambda item: item[0], reverse=True)

        chosen_t_centers: list[float] = []
        for score, idx, ang in scored_turns:
            if len(turn_windows) >= cfg.max_turn_windows:
                break
            i1 = min(len(lid_t) - 1, idx + win_steps)
            t_start = float(lid_t[idx] + tau_a)
            t_end = float(lid_t[i1] + tau_a)
            t_mid = 0.5 * (t_start + t_end)
            if any(
                abs(t_mid - prev) < cfg.min_turn_separation_s
                for prev in chosen_t_centers
            ):
                continue
            if ang < cfg.min_turn_yaw_deg:
                turn_screening_log.append(
                    {
                        "rel_start_s": float(t_start - t0_ref),
                        "rel_end_s": float(t_end - t0_ref),
                        "yaw_deg": float(ang),
                        "accepted": False,
                        "skip_reason": "insufficient_yaw_rotation",
                    }
                )
                continue

            table = _build_preintegrated_imu_table(
                access_a.imu_t_s,
                access_a.imu_w,
                access_a.imu_a,
                t_start=t_start,
                t_end=t_end,
                gyro_bias=bg,
                acc_bias=ba,
            )
            # Initial velocity & effective gravity in frame I(t_start)
            dR_01_L = _project_so3(batch_a["lid_R"][idx].T @ batch_a["lid_R"][i1])
            R_I0_I1 = R_LI_a @ dR_01_L @ R_LI_a.T
            v_L0_local = batch_a["lid_R"][idx].T @ batch_a["lid_v"][idx]
            v_L1_local = batch_a["lid_R"][i1].T @ batch_a["lid_v"][i1]
            v_I0 = R_LI_a @ (
                v_L0_local + np.cross(batch_a["lid_w"][idx], -(R_LI_a.T @ T_LI_a))
            )
            v_I1_in_I1 = R_LI_a @ (
                v_L1_local + np.cross(batch_a["lid_w"][i1], -(R_LI_a.T @ T_LI_a))
            )
            v_I1_in_I0 = R_I0_I1 @ v_I1_in_I1
            span = max(1e-3, float(t_end - t_start))
            _, _, dv_spec_end, _, _ = _query_imu_preint(table, np.array([t_end]))
            g_init = (v_I1_in_I0 - v_I0 - dv_spec_end[0]) / span
            g_norm = float(np.linalg.norm(g_init))
            if g_norm > 1.0:
                g_init = g_init * (9.81 / g_norm)
            else:
                g_init = np.array([0.0, 0.0, -9.81], dtype=np.float64)

            # Extract structural points per scan in turn window (keeping raw times)
            def _collect_turn_points(
                access: FastCanonicalScanAccess,
                R_LI_est: np.ndarray,
                T_LI_est: np.ndarray,
                tau_est: float,
            ) -> tuple[np.ndarray, np.ndarray]:
                scan_phys = access.scan_t_s + tau_est
                sel = np.flatnonzero((scan_phys >= t_start) & (scan_phys <= t_end))
                p_chunks: list[np.ndarray] = []
                t_chunks: list[np.ndarray] = []
                for s_idx in sel:
                    xyz_l, pt_t = access.read_scan(int(s_idx))
                    if len(xyz_l) == 0:
                        continue
                    xyz_l = xyz_l[:: cfg.turn_point_stride]
                    pt_t = pt_t[:: cfg.turn_point_stride]
                    r_xy = np.linalg.norm(xyz_l[:, :2], axis=1)
                    # Keep structural elevation band (exclude near-chassis & far sparse)
                    z_approx = (xyz_l @ R_LI_est.T + T_LI_est)[:, 2]
                    keep = (
                        (r_xy >= 2.5)
                        & (r_xy <= 38.0)
                        & (z_approx >= -1.2)
                        & (z_approx <= 4.5)
                    )
                    if not np.any(keep):
                        continue
                    xyz_k = xyz_l[keep]
                    t_raw_k = float(access.scan_t_s[int(s_idx)]) + pt_t[keep]
                    p_chunks.append(xyz_k)
                    t_chunks.append(t_raw_k)
                if not p_chunks:
                    return np.zeros((0, 3), dtype=np.float64), np.zeros(0)
                pts_all = np.vstack(p_chunks)
                ts_all = np.concatenate(t_chunks)
                if len(pts_all) > 7500:
                    sub = np.linspace(0, len(pts_all) - 1, num=7500, dtype=int)
                    pts_all = pts_all[sub]
                    ts_all = ts_all[sub]
                return pts_all, ts_all

            pts_a_turn, t_raw_a = _collect_turn_points(access_a, R_LI_a, T_LI_a, tau_a)
            pts_b_turn, t_raw_b = _collect_turn_points(access_b, R_LI_b, T_LI_b, tau_b)
            if len(pts_a_turn) < 300 or len(pts_b_turn) < 300:
                turn_screening_log.append(
                    {
                        "rel_start_s": float(t_start - t0_ref),
                        "rel_end_s": float(t_end - t0_ref),
                        "yaw_deg": float(ang),
                        "accepted": False,
                        "skip_reason": "insufficient_turn_points",
                    }
                )
                continue

            chosen_t_centers.append(t_mid)
            turn_windows.append(
                {
                    "window_id": f"turn_{len(turn_windows) + 1}",
                    "rel_start_s": float(t_start - t0_ref),
                    "rel_end_s": float(t_end - t0_ref),
                    "yaw_deg": float(ang),
                    "table": table,
                    "v_u0_init": v_I0,
                    "g_u0_init": g_init,
                    "pts_a": pts_a_turn,
                    "t_raw_a": t_raw_a,
                    "pts_b": pts_b_turn,
                    "t_raw_b": t_raw_b,
                }
            )
            turn_screening_log.append(
                {
                    "window_id": turn_windows[-1]["window_id"],
                    "rel_start_s": float(t_start - t0_ref),
                    "rel_end_s": float(t_end - t0_ref),
                    "yaw_deg": float(ang),
                    "points_a": int(len(pts_a_turn)),
                    "points_b": int(len(pts_b_turn)),
                    "accepted": True,
                }
            )

    return {
        "T_I_A_init": T_I_A,
        "T_I_B_init": T_I_B,
        "T_AB_init": T_AB_init,
        "tau_a_init": tau_a,
        "tau_b_init": tau_b,
        "bg_mean": bg,
        "ba_mean": ba,
        "static_windows": static_windows,
        "static_screening_log": static_screening_log,
        "gravity_windows": gravity_windows,
        "turn_windows": turn_windows,
        "turn_screening_log": turn_screening_log,
    }


def _compute_l2i_anisotropic_info(
    result: dict[str, Any],
    batch: dict[str, Any],
    cfg: JointCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build anisotropic 3x3 rotation and 3x3 IMU-frame translation info."""
    R_LI = _project_so3(np.asarray(result["rotation_LI_matrix"], dtype=np.float64))
    R_IL = R_LI.T
    lid_t = batch["lid_t"]
    dt_med = float(np.median(np.diff(lid_t))) if len(lid_t) > 1 else 1.0 / 30.0
    imu_w_sync = _zero_phase_smooth(_interp_vec3(batch["imu_t"], batch["imu_w"], lid_t))
    imu_alpha = np.gradient(imu_w_sync, dt_med, axis=0)

    # Anisotropic rotation info in IMU frame:
    # - Roll/Pitch (X, Y) are strongly grounded by gravity + 3D gyro.
    # - Yaw (Z) has weaker observability under planar driving and is scaled by
    #   each sensor's linear acceleration consistency against IMU specific force.
    sigma_rp = math.radians(cfg.l2i_rot_prior_sigma_deg)
    sigma_yaw = math.radians(cfg.l2i_yaw_prior_sigma_deg)
    yaw_quality_scale = 1.0
    if len(lid_t) > 30:
        tau_s = float(result.get("time_offset_s", 0.0))
        ba = np.asarray(result.get("acc_bias_m_s2", [0.0, 0.0, 0.0]), dtype=np.float64)
        T_IL = -(R_IL @ np.asarray(result["translation_LI_m"], dtype=np.float64))
        v_sm = _zero_phase_smooth(batch["lid_v"], window=11)
        a_w = np.gradient(v_sm, dt_med, axis=0)
        a_l = np.einsum("nji,nj->ni", batch["lid_R"], a_w)
        w_l = imu_w_sync @ R_LI
        al_l = imu_alpha @ R_LI
        lever_l = np.cross(al_l, T_IL) + np.cross(w_l, np.cross(w_l, T_IL))
        a_imu_from_l = (a_l + _zero_phase_smooth(lever_l, window=11)) @ R_LI.T
        f_imu = _zero_phase_smooth(
            _interp_vec3(batch["imu_t"], batch["imu_a"] - ba, lid_t + tau_s),
            window=11,
        )
        # Compare horizontal dynamic acceleration variations (zero-meaned)
        da_xy = a_imu_from_l[:, :2] - np.mean(a_imu_from_l[:, :2], axis=0)
        df_xy = f_imu[:, :2] - np.mean(f_imu[:, :2], axis=0)
        acc_rmse = float(np.sqrt(np.mean(np.sum((da_xy - df_xy) ** 2, axis=1))))
        # Reference accuracy ~0.35 m/s^2; noisier sensor gets smaller yaw precision
        yaw_quality_scale = float(np.clip((0.35 / max(0.15, acc_rmse)) ** 2, 0.65, 1.5))
    H_rot = np.diag(
        [
            1.0 / (sigma_rp**2),
            1.0 / (sigma_rp**2),
            yaw_quality_scale / (sigma_yaw**2),
        ]
    )

    # Translation info in IMU frame: evaluate kinematic singular values along X, Y, Z
    A_blocks = []
    for k in range(10, max(11, len(lid_t) - 10)):
        w_I = imu_w_sync[k]
        al_I = imu_alpha[k]
        W = _skew(w_I)
        A_blocks.append(W @ W + _skew(al_I))
    A_stack = np.vstack(A_blocks) if A_blocks else np.zeros((3, 3), dtype=np.float64)
    # Transform regressor A_I^T A_I eigenvalues to physical weight
    H_kin = A_stack.T @ A_stack
    eigvals, eigvecs = np.linalg.eigh(H_kin)
    max_ev = max(float(eigvals[-1]), 1e-6)
    rel_info = np.clip(eigvals / max_ev, 0.0, 1.0)
    # Planar xy gets l2i_xy_prior_sigma_m; unobservable z gets near-zero weight.
    base_xy_info = 1.0 / (cfg.l2i_xy_prior_sigma_m**2)
    # Keep tiny numerical ridge (1e-4) only; true z-anchor comes from z_cm_prior
    scaled_eigs = np.where(rel_info > 0.05, base_xy_info * rel_info, 1e-4)
    H_trans_imu = eigvecs @ np.diag(scaled_eigs) @ eigvecs.T
    # Preserve R_IL check sanity (already in IMU frame because w_I, al_I are IMU frame)
    _ = R_IL
    return H_rot, H_trans_imu


def _evaluate_l2l_edges_gicp(
    screened: dict[str, Any],
) -> list[dict[str, Any]]:
    """v1 L2L edges using Open3D GICP + physical covariance scaling."""
    import open3d as o3d

    T_AB_init = screened["T_AB_init"]
    edges: list[dict[str, Any]] = []
    for win in screened["static_windows"]:
        pts_a = win["frame_a"].points
        pts_b = win["frame_b"].points
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(pts_a)
        pcd_tgt = pcd_tgt.voxel_down_sample(0.15)
        pcd_tgt.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30)
        )

        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(pts_b)
        pcd_src = pcd_src.voxel_down_sample(0.15)
        pcd_src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30)
        )

        T_curr = T_AB_init.copy()
        reg = None
        for max_corr in (1.1, 0.45, 0.22):
            reg = o3d.pipelines.registration.registration_generalized_icp(
                pcd_src,
                pcd_tgt,
                max_corr,
                T_curr,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
            T_curr = np.asarray(reg.transformation, dtype=np.float64)
        assert reg is not None

        raw_info = np.asarray(
            o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcd_src, pcd_tgt, 0.25, T_curr
            ),
            dtype=np.float64,
        )
        # Open3D's point-to-point proxy is in the left-target tangent; the joint
        # edge residual Log(inv(T_edge) @ T_pred) uses the right-source tangent.
        raw_info = information_left_to_right(raw_info, T_curr)
        sigma_eff = max(0.04, float(reg.inlier_rmse))
        # Normalize Open3D point-to-point correspondence Hessian onto realistic physical
        # multi-window Huber scale.
        info_6d = raw_info / (sigma_eff**2 * max(1.0, len(pcd_src.points) / 40.0))
        eigvals = np.linalg.eigvalsh(0.5 * (info_6d + info_6d.T))
        cond = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-9 else float("inf")
        edges.append(
            {
                "window_id": win["window_id"],
                "rel_time_s": win["rel_time_s"],
                "method": "gicp_v1",
                "success": True,
                "transformation": T_curr,
                "info_matrix_6d": info_6d,
                "information_tangent": "right_source",
                "information_model": "scaled_point_to_point_proxy",
                "fitness": float(reg.fitness),
                "inlier_rmse_m": float(reg.inlier_rmse),
                "condition_number_6d": cond,
                "eigenvalues_6d": [float(v) for v in eigvals],
            }
        )
    return edges


def _evaluate_l2l_edges_semantic(
    screened: dict[str, Any],
    T_AB_seed: np.ndarray,
    cfg: JointCalibrationConfig,
    *,
    feature_traces: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """v2 L2L edges using 4-channel primitives, calibrated intensity, and 3D FoV."""
    ego_min = (
        np.asarray(cfg.ego_bbox_min_imu, dtype=np.float64)
        if cfg.ego_bbox_min_imu is not None
        else None
    )
    ego_max = (
        np.asarray(cfg.ego_bbox_max_imu, dtype=np.float64)
        if cfg.ego_bbox_max_imu is not None
        else None
    )
    T_I_A = screened["T_I_A_init"]
    ego_min_A = None
    ego_max_A = None
    if ego_min is not None and ego_max is not None:
        corners_i = np.array(
            [
                [x, y, z]
                for x in (ego_min[0], ego_max[0])
                for y in (ego_min[1], ego_max[1])
                for z in (ego_min[2], ego_max[2])
            ],
            dtype=np.float64,
        )
        T_A_I = np.linalg.inv(T_I_A)
        corners_a = corners_i @ T_A_I[:3, :3].T + T_A_I[:3, 3]
        ego_min_A = tuple(float(v) for v in corners_a.min(axis=0))
        ego_max_A = tuple(float(v) for v in corners_a.max(axis=0))

    vis_cfg = VisibilityConfig(
        ego_bbox_min_lidar=ego_min_A,
        ego_bbox_max_lidar=ego_max_A,
    )
    edges: list[dict[str, Any]] = []
    for win in screened["static_windows"]:
        trace = {} if feature_traces is not None else None
        res = register_semantic_l2l(
            win["frame_b"],
            win["frame_a"],
            T_AB_seed,
            prim_cfg=PrimitiveExtractionConfig(),
            vis_cfg=vis_cfg,
            reg_cfg=SemanticRegistrationConfig(),
            trace=trace,
        )
        if feature_traces is not None:
            feature_traces.append(
                {
                    "trace": trace,
                    "result": res,
                    "metadata": {
                        k: v for k, v in win.items() if k not in {"frame_a", "frame_b"}
                    },
                }
            )
        if not res.get("success"):
            if failures is not None:
                failures.append(
                    {
                        "window_id": win["window_id"],
                        "rel_time_s": win["rel_time_s"],
                        "method": "semantic_v2",
                        "success": False,
                        "reason": res["reason"],
                    }
                )
            continue
        # Check roll/pitch consistency against gravity-grounded L2I prior (T_AB_init):
        # if an L2L static window tilts roll/pitch > 1.2 deg relative to gravity,
        # downweight it so outlier dynamic-traffic stops cannot corrupt valid stops.
        T_rel_err = np.linalg.inv(screened["T_AB_init"]) @ res["transformation"]
        tilt_err_deg = math.degrees(
            math.hypot(float(T_rel_err[2, 1]), float(T_rel_err[2, 0]))
        )
        matches = res["channel_matches"]
        weak_support = (
            matches["facade"] + matches["pole"] + matches["intensity"]
        ) < 35 or (matches["facade"] < 20 and matches["pole"] < 12)
        info_scale = 0.02 if (tilt_err_deg > 1.2 or weak_support) else 1.0
        quality_score = (
            (1.0 if info_scale >= 1.0 else 0.05)
            * (2.0 * matches["facade"] + matches["pole"] + 0.5 * matches["intensity"])
            / max(0.02, float(res["sigma_eff_m"]))
        )
        edges.append(
            {
                "window_id": win["window_id"],
                "rel_time_s": win["rel_time_s"],
                "method": "semantic_v2",
                "success": True,
                "transformation": res["transformation"],
                "info_matrix_6d": res["info_matrix_6d"] * info_scale,
                "information_tangent": res["information_tangent"],
                "information_model": "weighted_residual_gauss_newton",
                "gravity_tilt_delta_deg": float(tilt_err_deg),
                "downweighted_by_gravity_gate": bool(tilt_err_deg > 1.2),
                "downweighted_by_weak_support": bool(weak_support),
                "consensus_outlier": False,
                "quality_score": float(quality_score),
                "sigma_eff_m": res["sigma_eff_m"],
                "localizability": res["localizability"],
                "directed_overlap": res["directed_overlap"],
                "channel_matches": res["channel_matches"],
                "channel_unique_targets": res.get("channel_unique_targets"),
                "channel_rmse_m": res["channel_rmse_m"],
                "condition_number_6d": res["localizability"]["condition_number_6d"],
                "registration_eigenvalues_6d": res["localizability"]["eigenvalues_6d"],
                "eigenvalues_6d": [
                    info_scale * value
                    for value in res["localizability"]["eigenvalues_6d"]
                ],
            }
        )

    # Multi-window IRWLS consensus: MAD-adaptive Cauchy gating for outlier stops
    if len(edges) >= 2:
        anchor = max(edges, key=lambda item: item["quality_score"])
        T_ref = anchor["transformation"]
        if len(edges) >= 3:
            # Find edge minimizing weighted distance to all other edges
            best_med_cost = float("inf")
            for cand in edges:
                T_c = cand["transformation"]
                c_sum = 0.0
                for other in edges:
                    T_o = other["transformation"]
                    d = np.linalg.inv(T_c) @ T_o
                    dt = float(np.linalg.norm(d[:3, 3]))
                    dr = _rotation_error_deg(np.eye(3), d[:3, :3])
                    c_sum += other["quality_score"] * (dt + 0.05 * dr)
                if c_sum < best_med_cost:
                    best_med_cost = c_sum
                    T_ref = T_c

        dts = []
        drs = []
        for edge in edges:
            d = np.linalg.inv(T_ref) @ edge["transformation"]
            dts.append(float(np.linalg.norm(d[:3, 3])))
            drs.append(_rotation_error_deg(np.eye(3), d[:3, :3]))
        mad_dt = 1.4826 * float(np.median(dts))
        mad_dr = 1.4826 * float(np.median(drs))
        dt_gate = max(0.08, 2.5 * mad_dt)
        dr_gate = max(0.35, 2.5 * mad_dr)

        for edge, dt, dr in zip(edges, dts, drs):
            if dt > dt_gate or dr > dr_gate:
                cauchy_w = 1.0 / (1.0 + (dt / 0.05) ** 2 + (dr / 0.30) ** 2)
                if dt > 2.0 * dt_gate or dr > 2.0 * dr_gate:
                    cauchy_w = min(cauchy_w, 0.02)
                edge["info_matrix_6d"] = edge["info_matrix_6d"] * cauchy_w
                edge["eigenvalues_6d"] = [
                    value * cauchy_w for value in edge["eigenvalues_6d"]
                ]
                edge["consensus_outlier"] = True
                edge["consensus_weight"] = float(cauchy_w)
    return edges


def _solve_anisotropic_factor_graph(
    *,
    screened: dict[str, Any],
    l2l_edges: list[dict[str, Any]],
    H_rot_a: np.ndarray,
    H_trans_a: np.ndarray,
    H_rot_b: np.ndarray,
    H_trans_b: np.ndarray,
    cfg: JointCalibrationConfig,
    T_I_A_init: np.ndarray | None = None,
    T_I_B_init: np.ndarray | None = None,
) -> dict[str, Any]:
    T_IA0 = screened["T_I_A_init"] if T_I_A_init is None else T_I_A_init
    T_IB0 = screened["T_I_B_init"] if T_I_B_init is None else T_I_B_init
    T_IA_l2i = screened["T_I_A_init"]
    T_IB_l2i = screened["T_I_B_init"]

    z_cm_prior = (
        float(cfg.z_cm_prior_m)
        if cfg.z_cm_prior_m is not None
        else 0.5 * float(T_IA_l2i[2, 3] + T_IB_l2i[2, 3])
    )
    W_rot_a = _safe_cholesky_whitener(H_rot_a)
    W_trans_a = _safe_cholesky_whitener(H_trans_a)
    W_rot_b = _safe_cholesky_whitener(H_rot_b)
    W_trans_b = _safe_cholesky_whitener(H_trans_b)
    l2l_whiteners = [_safe_cholesky_whitener(e["info_matrix_6d"]) for e in l2l_edges]
    w_grav = 1.0 / math.radians(cfg.gravity_sigma_deg)
    w_z_prior = 1.0 / max(1e-3, cfg.z_cm_prior_sigma_m)

    def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R_A = Rot.from_rotvec(x[0:3]).as_matrix() @ T_IA0[:3, :3]
        t_A = T_IA0[:3, 3] + x[3:6]
        R_B = Rot.from_rotvec(x[6:9]).as_matrix() @ T_IB0[:3, :3]
        t_B = T_IB0[:3, 3] + x[9:12]
        return _se3_from_rt(R_A, t_A), _se3_from_rt(R_B, t_B)

    def _residuals(x: np.ndarray) -> np.ndarray:
        T_IA, T_IB = _unpack(x)
        chunks: list[np.ndarray] = []

        # 1. Anisotropic L2I priors
        chunks.append(
            W_rot_a @ Rot.from_matrix(T_IA[:3, :3] @ T_IA_l2i[:3, :3].T).as_rotvec()
        )
        chunks.append(W_trans_a @ (T_IA[:3, 3] - T_IA_l2i[:3, 3]))
        chunks.append(
            W_rot_b @ Rot.from_matrix(T_IB[:3, :3] @ T_IB_l2i[:3, :3].T).as_rotvec()
        )
        chunks.append(W_trans_b @ (T_IB[:3, 3] - T_IB_l2i[:3, 3]))

        # 2. Multi-window L2L SE(3) factors
        T_AB_pred = np.linalg.inv(T_IA) @ T_IB
        for edge, W_edge in zip(l2l_edges, l2l_whiteners):
            T_err = np.linalg.inv(edge["transformation"]) @ T_AB_pred
            chunks.append(W_edge @ _se3_log6(T_err))

        # 3. Multi-heading gravity-ground alignment factors
        for gw in screened["gravity_windows"]:
            u_g = gw["u_g_imu"]
            P_tan = np.eye(3, dtype=np.float64) - np.outer(u_g, u_g)
            chunks.append(w_grav * (P_tan @ (T_IA[:3, :3] @ gw["n_a_lidar"])))
            chunks.append(w_grav * (P_tan @ (T_IB[:3, :3] @ gw["n_b_lidar"])))

        # 4. Nullspace anchor on common-mode vertical height z_cm
        z_cm = 0.5 * (T_IA[2, 3] + T_IB[2, 3])
        chunks.append(np.array([w_z_prior * (z_cm - z_cm_prior)], dtype=np.float64))
        return np.concatenate(chunks)

    sol = least_squares(
        _residuals,
        np.zeros(12, dtype=np.float64),
        method="trf",
        loss="huber",
        f_scale=2.5,
        max_nfev=120,
    )
    T_IA_opt, T_IB_opt = _unpack(sol.x)

    # Convert additive solver parameters to left SE(3) tangents in the shared IMU
    # frame. A common rigid rotation also rotates both lever arms; otherwise L2L
    # spuriously appears to observe the rig's common-mode rotation.
    J = sol.jac.copy()
    for offset, transform in ((0, T_IA_opt), (6, T_IB_opt)):
        left_jacobian = so3_right_jacobian(-sol.x[offset : offset + 3])
        J[:, offset : offset + 3] = np.linalg.solve(
            left_jacobian.T, J[:, offset : offset + 3].T
        ).T - J[:, offset + 3 : offset + 6] @ _skew(transform[:3, 3])
    H_12 = J.T @ J
    s_half = math.sqrt(0.5)
    # Orthogonal 12x12 transform: [cm(6), diff(6)] = Q @ [xi_A(6), xi_B(6)]
    I6 = np.eye(6, dtype=np.float64)
    Q_cd = np.block([[s_half * I6, s_half * I6], [-s_half * I6, s_half * I6]])
    H_cd = Q_cd @ H_12 @ Q_cd.T
    H_cm = H_cd[:6, :6]
    H_diff = H_cd[6:12, 6:12]
    eig_cm = np.linalg.eigvalsh(0.5 * (H_cm + H_cm.T))
    eig_diff = np.linalg.eigvalsh(0.5 * (H_diff + H_diff.T))

    return {
        "T_I_A": T_IA_opt,
        "T_I_B": T_IB_opt,
        "T_A_B": np.linalg.inv(T_IA_opt) @ T_IB_opt,
        "tau_a_s": screened["tau_a_init"],
        "tau_b_s": screened["tau_b_init"],
        "z_cm_prior_m": z_cm_prior,
        "cost": float(sol.cost),
        "solver": {
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": str(sol.message),
            "nfev": int(sol.nfev),
        },
        "observability": {
            "tangent": "left_imu_se3",
            "common_mode_6d_eigenvalues": [float(v) for v in eig_cm],
            "differential_mode_6d_eigenvalues": [float(v) for v in eig_diff],
            "common_mode_z_prior_anchored": True,
        },
    }


def _evaluate_l2l_window_influence(
    *,
    screened: dict[str, Any],
    l2l_edges: list[dict[str, Any]],
    solution: dict[str, Any],
    H_rot_a: np.ndarray,
    H_trans_a: np.ndarray,
    H_rot_b: np.ndarray,
    H_trans_b: np.ndarray,
    cfg: JointCalibrationConfig,
    T_I_A_init: np.ndarray,
    T_I_B_init: np.ndarray,
) -> dict[str, Any]:
    """Post-run deletion sensitivity with the original factors and priors fixed."""
    window_ids = list(dict.fromkeys(edge["window_id"] for edge in l2l_edges))
    undownweighted = [
        edge["window_id"]
        for edge in l2l_edges
        if not any(
            edge.get(flag, False)
            for flag in (
                "downweighted_by_gravity_gate",
                "downweighted_by_weak_support",
                "consensus_outlier",
            )
        )
    ]
    report: dict[str, Any] = {
        "status": "review_only",
        "scope": "fixed_factor_l2l_window_deletion",
        "window_count": len(window_ids),
        "nominal_solver": solution["solver"],
        "undownweighted_window_ids": list(dict.fromkeys(undownweighted)),
        "reasons": [],
        "windows": [],
        "max_change_from_nominal": None,
        "notes": [
            "Delete all L2L factors from one window; keep L2I, gravity, height "
            "prior, remaining factors/weights and optimizer initialization fixed.",
            "Does not rerun screening, registration or consensus; not a full "
            "pipeline jackknife, calibrated covariance or absolute accuracy.",
            "L2L cannot validate the shared IMU datum or prior-anchored height.",
            "Distinct target indices are not independent physical objects.",
            "Diagnostic only; does not change transforms, weights or verdict.",
        ],
    }
    if len(set(undownweighted)) < 2:
        report["reasons"].append("fewer_than_two_undownweighted_windows")
    if not solution["solver"]["success"]:
        report["status"] = "nominal_solver_failed"
        report["reasons"].append("cannot_compare_to_unconverged_nominal")
        return report
    if len(window_ids) < 2:
        report["status"] = "insufficient_windows"
        report["reasons"].append("fewer_than_two_l2l_windows")
        return report
    for window_id in window_ids:
        remaining = [edge for edge in l2l_edges if edge["window_id"] != window_id]
        omitted_solution = _solve_anisotropic_factor_graph(
            screened=screened,
            l2l_edges=remaining,
            H_rot_a=H_rot_a,
            H_trans_a=H_trans_a,
            H_rot_b=H_rot_b,
            H_trans_b=H_trans_b,
            cfg=cfg,
            T_I_A_init=T_I_A_init,
            T_I_B_init=T_I_B_init,
        )
        changes = {}
        if omitted_solution["solver"]["success"]:
            for key in ("T_I_A", "T_I_B", "T_A_B"):
                delta = np.linalg.inv(solution[key]) @ omitted_solution[key]
                changes[key] = {
                    "translation_m": float(np.linalg.norm(delta[:3, 3])),
                    "rotation_deg": _rotation_error_deg(np.eye(3), delta[:3, :3]),
                }
        else:
            report["status"] = "solver_failed"
            report["reasons"].append(f"deletion_solver_failed:{window_id}")
        report["windows"].append(
            {
                "omitted_window_id": window_id,
                "remaining_window_count": len(window_ids) - 1,
                "solver": omitted_solution["solver"],
                "change_from_nominal": changes or None,
            }
        )
    if report["status"] != "solver_failed":
        report["max_change_from_nominal"] = {
            key: {
                metric: max(
                    row["change_from_nominal"][key][metric] for row in report["windows"]
                )
                for metric in ("translation_m", "rotation_deg")
            }
            for key in ("T_I_A", "T_I_B", "T_A_B")
        }
    return report


def _solve_turn_imu_ba_v3(
    *,
    screened: dict[str, Any],
    v2_solution: dict[str, Any],
    l2l_edges: list[dict[str, Any]],
    H_rot_a: np.ndarray,
    H_trans_a: np.ndarray,
    H_rot_b: np.ndarray,
    H_trans_b: np.ndarray,
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    cfg: JointCalibrationConfig,
) -> dict[str, Any]:
    """v3: Tightly-coupled 100 Hz IMU-preintegrated dual-LiDAR plane BA over turn
    windows + velocity-level kinematic consistency + static L2L + gravity + gyro.
    """
    turn_windows = screened["turn_windows"]
    if not turn_windows:
        out = dict(v2_solution)
        out["v3_skipped_reason"] = "no_turn_windows_available"
        return out

    T_IA_curr = v2_solution["T_I_A"].copy()
    T_IB_curr = v2_solution["T_I_B"].copy()
    T_IA_l2i = screened["T_I_A_init"]
    T_IB_l2i = screened["T_I_B_init"]
    tau_a_curr = float(v2_solution["tau_a_s"])
    tau_b_curr = float(v2_solution["tau_b_s"])
    z_cm_prior = float(v2_solution["z_cm_prior_m"])

    W_rot_a = _safe_cholesky_whitener(H_rot_a)
    # In v3, active turn BA + velocity kinematics supersede EKF acceleration
    # differentiation; relax planar L2I translation prior to a loose fallback.
    l2i_relax = 0.25 if len(turn_windows) >= 2 else 0.5
    W_trans_a = _safe_cholesky_whitener(H_trans_a * l2i_relax)
    W_rot_b = _safe_cholesky_whitener(H_rot_b)
    W_trans_b = _safe_cholesky_whitener(H_trans_b * l2i_relax)

    v_u0_list = [tw["v_u0_init"].copy() for tw in turn_windows]
    g_u0_list = [tw["g_u0_init"].copy() for tw in turn_windows]

    # Precompute per-turn local LiDAR velocities at 5 sample timestamps for
    # velocity-level IMU preintegration consistency (no velocity differentiation!).
    turn_vel_samples: list[dict[str, Any]] = []
    for tw in turn_windows:
        t_start = float(tw["table"]["t0"][0])
        t_end = t_start + float(tw["rel_end_s"] - tw["rel_start_s"])
        t_samples = np.linspace(t_start, t_end, 5, dtype=np.float64)

        # Local LiDAR velocities interpolated from batch_a / batch_b
        def _local_v(batch: dict[str, Any], tau_est: float) -> np.ndarray:
            v_loc = np.einsum("nji,nj->ni", batch["lid_R"], batch["lid_v"])
            return _interp_vec3(batch["lid_t"] + tau_est, v_loc, t_samples)

        turn_vel_samples.append(
            {
                "t": t_samples,
                "va_local": _local_v(batch_a, screened["tau_a_init"]),
                "vb_local": _local_v(batch_b, screened["tau_b_init"]),
            }
        )

    # Precompute full-drive gyro alignment samples (30 Hz interpolated)
    ta_grid = batch_a["lid_t"]
    tb_grid = batch_b["lid_t"]
    w_la_sm = _zero_phase_smooth(batch_a["lid_w"], window=5)
    w_lb_sm = _zero_phase_smooth(batch_b["lid_w"], window=5)
    # Finite-diff angular acc of IMU for analytical tau Jacobian in gyro term
    dt_imu = max(1e-4, float(np.median(np.diff(batch_a["imu_t"]))))
    w_imu_raw = batch_a["imu_w"] - screened["bg_mean"]
    w_imu_sm = _zero_phase_smooth(w_imu_raw, window=5)
    dw_imu_dt = np.gradient(w_imu_sm, dt_imu, axis=0)

    # Subsample gyro evaluation times across active segments for speed (<= 450 pts)
    step_ga = max(1, len(ta_grid) // 400)
    step_gb = max(1, len(tb_grid) // 400)
    ga_idx = np.arange(10, max(11, len(ta_grid) - 10), step_ga)
    gb_idx = np.arange(10, max(11, len(tb_grid) - 10), step_gb)

    z_cm_v2 = 0.5 * float(T_IA_curr[2, 3] + T_IB_curr[2, 3])

    l2l_whiteners = [_safe_cholesky_whitener(e["info_matrix_6d"]) for e in l2l_edges]
    w_grav = 1.0 / math.radians(cfg.gravity_sigma_deg)
    w_z_lock = 1.0 / max(1e-4, cfg.v3_z_lock_sigma_m)
    w_gyro = 1.0 / cfg.v3_gyro_sigma_rad_s
    w_tau_prior = 1.0 / max(1e-4, cfg.v3_tau_prior_sigma_s)
    w_dtau_prior = 1.0 / max(1e-4, cfg.v3_dtau_prior_sigma_s)
    tau_a_l2i = float(screened["tau_a_init"])
    tau_b_l2i = float(screened["tau_b_init"])
    dtau_l2i = tau_a_l2i - tau_b_l2i

    def _full_gyro_rmse(
        R_IA: np.ndarray, tau_a: float, R_IB: np.ndarray, tau_b: float
    ) -> float:
        errs = []
        for idx_arr, t_l_arr, w_l_arr, R_Ik, tau_k in (
            (ga_idx, ta_grid, w_la_sm, R_IA, tau_a),
            (gb_idx, tb_grid, w_lb_sm, R_IB, tau_b),
        ):
            w_i_q = _interp_vec3(batch_a["imu_t"], w_imu_sm, t_l_arr[idx_arr] + tau_k)
            w_l_rot = w_l_arr[idx_arr] @ R_Ik.T
            errs.append(np.sum((w_l_rot - w_i_q) ** 2, axis=1))
        return float(np.sqrt(np.mean(np.concatenate(errs))))

    v2_gyro_rmse = _full_gyro_rmse(
        T_IA_curr[:3, :3], tau_a_curr, T_IB_curr[:3, :3], tau_b_curr
    )

    def _extract_turn_plane_correspondences(
        T_IA: np.ndarray,
        T_IB: np.ndarray,
        tau_a: float,
        tau_b: float,
        voxel_m: float,
        active_indices: list[int] | None = None,
        v_override: list[np.ndarray] | None = None,
        g_override: list[np.ndarray] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        indices = (
            list(range(len(turn_windows)))
            if active_indices is None
            else list(active_indices)
        )
        v_src = v_u0_list if v_override is None else v_override
        g_src = g_u0_list if g_override is None else g_override
        per_turn_data: list[dict[str, Any]] = []
        all_sigmas_cm: list[float] = []
        for pos_i, u_idx in enumerate(indices):
            tw = turn_windows[u_idx]
            v_u0 = v_src[pos_i] if v_override is not None else v_src[u_idx]
            g_u0 = g_src[pos_i] if g_override is not None else g_src[u_idx]
            table = tw["table"]

            def _proj(
                pts_l: np.ndarray, t_raw: np.ndarray, T_Ik: np.ndarray, tau_k: float
            ):
                dR, dp, _, _, dt0 = _query_imu_preint(table, t_raw + tau_k)
                p_i = pts_l @ T_Ik[:3, :3].T + T_Ik[:3, 3]
                p_w = (
                    dp
                    + np.outer(dt0, v_u0)
                    + 0.5 * np.outer(dt0**2, g_u0)
                    + np.einsum("nij,nj->ni", dR, p_i)
                )
                return p_w

            pw_a = _proj(tw["pts_a"], tw["t_raw_a"], T_IA, tau_a)
            pw_b = _proj(tw["pts_b"], tw["t_raw_b"], T_IB, tau_b)

            pw_all = np.vstack([pw_a, pw_b])
            sensor_id = np.concatenate(
                [np.zeros(len(pw_a), dtype=np.int8), np.ones(len(pw_b), dtype=np.int8)]
            )
            local_idx = np.concatenate(
                [
                    np.arange(len(pw_a), dtype=np.int32),
                    np.arange(len(pw_b), dtype=np.int32),
                ]
            )
            t_all = np.concatenate([tw["t_raw_a"], tw["t_raw_b"]])

            vox = np.floor(pw_all / voxel_m).astype(np.int64)
            order = np.lexsort((vox[:, 2], vox[:, 1], vox[:, 0]))
            s_vox = vox[order]
            s_pw = pw_all[order]
            s_sid = sensor_id[order]
            s_lidx = local_idx[order]
            s_t = t_all[order]
            diffs = np.any(np.diff(s_vox, axis=0) != 0, axis=1)
            bounds = np.concatenate([[0], np.where(diffs)[0] + 1, [len(s_pw)]])

            sel_a_idx: list[np.ndarray] = []
            sel_a_q: list[np.ndarray] = []
            sel_a_n: list[np.ndarray] = []
            sel_b_idx: list[np.ndarray] = []
            sel_b_q: list[np.ndarray] = []
            sel_b_n: list[np.ndarray] = []

            for s, e in zip(bounds[:-1], bounds[1:]):
                cnt = int(e - s)
                if cnt < 14:
                    continue
                if float(np.max(s_t[s:e]) - np.min(s_t[s:e])) < 0.35:
                    continue
                block = s_pw[s:e]
                mu = block.mean(axis=0)
                cov = ((block - mu).T @ (block - mu)) / float(cnt)
                eigvals, eigvecs = np.linalg.eigh(cov)
                if eigvals[2] < 1e-4:
                    continue
                planarity = (eigvals[1] - eigvals[0]) / eigvals[2]
                sigma_n = math.sqrt(max(0.0, float(eigvals[0])))
                if planarity < 0.42 or sigma_n > 0.14:
                    continue
                normal = eigvecs[:, 0]
                if abs(float(normal[2])) < 0.35:
                    all_sigmas_cm.append(sigma_n * 100.0)
                stride_v = max(1, cnt // 18)
                sub_range = slice(s, e, stride_v)
                sid_sub = s_sid[sub_range]
                lidx_sub = s_lidx[sub_range]
                ma = sid_sub == 0
                mb = sid_sub == 1
                if np.any(ma):
                    idx_a_v = lidx_sub[ma]
                    sel_a_idx.append(idx_a_v)
                    sel_a_q.append(np.tile(mu, (len(idx_a_v), 1)))
                    sel_a_n.append(np.tile(normal, (len(idx_a_v), 1)))
                if np.any(mb):
                    idx_b_v = lidx_sub[mb]
                    sel_b_idx.append(idx_b_v)
                    sel_b_q.append(np.tile(mu, (len(idx_b_v), 1)))
                    sel_b_n.append(np.tile(normal, (len(idx_b_v), 1)))

            per_turn_data.append(
                {
                    "idx_a": (
                        np.concatenate(sel_a_idx)
                        if sel_a_idx
                        else np.zeros(0, dtype=np.int32)
                    ),
                    "q_a": np.vstack(sel_a_q) if sel_a_q else np.zeros((0, 3)),
                    "n_a": np.vstack(sel_a_n) if sel_a_n else np.zeros((0, 3)),
                    "idx_b": (
                        np.concatenate(sel_b_idx)
                        if sel_b_idx
                        else np.zeros(0, dtype=np.int32)
                    ),
                    "q_b": np.vstack(sel_b_q) if sel_b_q else np.zeros((0, 3)),
                    "n_b": np.vstack(sel_b_n) if sel_b_n else np.zeros((0, 3)),
                }
            )
        med_sigma = float(np.median(all_sigmas_cm)) if all_sigmas_cm else 999.0
        return per_turn_data, med_sigma

    def _fit_single_turn_kinematics_frozen(
        u_idx: int,
        T_IA: np.ndarray,
        T_IB: np.ndarray,
        tau_a: float,
        tau_b: float,
    ) -> float:
        """Fit only (v_u0, g_u0) on a held-out turn with extrinsics frozen."""
        tw = turn_windows[u_idx]
        table = tw["table"]
        tv = turn_vel_samples[u_idx]
        dR_v, _, dv_v, w_v, dt0_v = _query_imu_preint(table, tv["t"])
        v_lid_a = np.einsum(
            "nij,nj->ni",
            dR_v,
            tv["va_local"] @ T_IA[:3, :3].T - np.cross(w_v, T_IA[:3, 3]),
        )
        v_lid_b = np.einsum(
            "nij,nj->ni",
            dR_v,
            tv["vb_local"] @ T_IB[:3, :3].T - np.cross(w_v, T_IB[:3, 3]),
        )
        v_curr = tw["v_u0_init"].copy()
        g_curr = tw["g_u0_init"].copy()
        B_g = _tangent_basis_s2(g_curr)

        def _vel_res(x5: np.ndarray) -> np.ndarray:
            v_try = v_curr + x5[0:3]
            g_try = Rot.from_rotvec(B_g @ x5[3:5]).as_matrix() @ g_curr
            v_pred = v_try[None, :] + dv_v + np.outer(dt0_v, g_try)
            return np.concatenate(
                [
                    (v_pred - v_lid_a).reshape(-1) / 0.10,
                    (v_pred - v_lid_b).reshape(-1) / 0.10,
                    x5[3:5] / math.radians(3.5),
                ]
            )

        sol_k = least_squares(_vel_res, np.zeros(5), method="trf", max_nfev=25)
        v_fit = v_curr + sol_k.x[0:3]
        g_fit = Rot.from_rotvec(B_g @ sol_k.x[3:5]).as_matrix() @ g_curr
        _, sig_cm = _extract_turn_plane_correspondences(
            T_IA,
            T_IB,
            tau_a,
            tau_b,
            0.70,
            active_indices=[u_idx],
            v_override=[v_fit],
            g_override=[g_fit],
        )
        return sig_cm

    def _solve_v3_on_subset(
        active_indices: list[int],
        max_outer_iters: int,
    ) -> dict[str, Any]:
        T_IA_s = v2_solution["T_I_A"].copy()
        T_IB_s = v2_solution["T_I_B"].copy()
        tau_a_s = float(v2_solution["tau_a_s"])
        tau_b_s = float(v2_solution["tau_b_s"])
        v_sub = [turn_windows[i]["v_u0_init"].copy() for i in active_indices]
        g_sub = [turn_windows[i]["g_u0_init"].copy() for i in active_indices]
        n_sub = len(active_indices)
        n_p = 14 + 5 * n_sub
        J_last = np.zeros((1, n_p), dtype=np.float64)

        for outer_idx in range(max_outer_iters):
            voxel_m = cfg.v3_voxel_schedule_m[
                min(outer_idx, len(cfg.v3_voxel_schedule_m) - 1)
            ]
            turn_corrs, _ = _extract_turn_plane_correspondences(
                T_IA_s,
                T_IB_s,
                tau_a_s,
                tau_b_s,
                voxel_m,
                active_indices=active_indices,
                v_override=v_sub,
                g_override=g_sub,
            )
            g_bases = [_tangent_basis_s2(g_u) for g_u in g_sub]

            def _eval_v3(x: np.ndarray, compute_jac: bool = True):
                R_IA = Rot.from_rotvec(x[0:3]).as_matrix() @ T_IA_s[:3, :3]
                t_IA = T_IA_s[:3, 3] + x[3:6]
                R_IB = Rot.from_rotvec(x[6:9]).as_matrix() @ T_IB_s[:3, :3]
                t_IB = T_IB_s[:3, 3] + x[9:12]
                tau_a = tau_a_s + float(x[12])
                tau_b = tau_b_s + float(x[13])

                res_chunks: list[np.ndarray] = []
                jac_chunks: list[np.ndarray] = []

                for sub_pos, (u_idx, tc, B_g) in enumerate(
                    zip(active_indices, turn_corrs, g_bases)
                ):
                    tw = turn_windows[u_idx]
                    p_off = 14 + 5 * sub_pos
                    v_u0 = v_sub[sub_pos] + x[p_off : p_off + 3]
                    dtheta_g = x[p_off + 3 : p_off + 5]
                    g_u0 = Rot.from_rotvec(B_g @ dtheta_g).as_matrix() @ g_sub[sub_pos]
                    table = tw["table"]

                    total_u_pts = len(tc["idx_a"]) + len(tc["idx_b"])
                    if total_u_pts > 0:
                        w_ba = 1.0 / (
                            cfg.v3_plane_sigma_m
                            * math.sqrt(max(1.0, total_u_pts / 80.0))
                        )
                        for (
                            s_idx_arr,
                            q_arr,
                            n_arr,
                            pts_src,
                            t_raw_src,
                            R_Ik,
                            t_Ik,
                            tau_k,
                            rot_col,
                            tr_col,
                            tau_col,
                        ) in (
                            (
                                tc["idx_a"],
                                tc["q_a"],
                                tc["n_a"],
                                tw["pts_a"],
                                tw["t_raw_a"],
                                R_IA,
                                t_IA,
                                tau_a,
                                0,
                                3,
                                12,
                            ),
                            (
                                tc["idx_b"],
                                tc["q_b"],
                                tc["n_b"],
                                tw["pts_b"],
                                tw["t_raw_b"],
                                R_IB,
                                t_IB,
                                tau_b,
                                6,
                                9,
                                13,
                            ),
                        ):
                            if len(s_idx_arr) == 0:
                                continue
                            pts_sel = pts_src[s_idx_arr]
                            t_sel = t_raw_src[s_idx_arr]
                            dR, dp, dv_spec, w_q, dt0 = _query_imu_preint(
                                table, t_sel + tau_k
                            )
                            p_rot = pts_sel @ R_Ik.T
                            p_i = p_rot + t_Ik
                            p_w = (
                                dp
                                + np.outer(dt0, v_u0)
                                + 0.5 * np.outer(dt0**2, g_u0)
                                + np.einsum("nij,nj->ni", dR, p_i)
                            )
                            res_chunks.append(
                                w_ba * np.sum((p_w - q_arr) * n_arr, axis=1)
                            )
                            if compute_jac:
                                m = len(pts_sel)
                                J_u = np.zeros((m, n_p), dtype=np.float64)
                                n_in_i = np.einsum("nji,nj->ni", dR, n_arr)
                                J_u[:, rot_col : rot_col + 3] = w_ba * np.cross(
                                    p_rot, n_in_i
                                )
                                J_u[:, tr_col : tr_col + 3] = w_ba * n_in_i
                                dp_dtau = (
                                    dv_spec
                                    + v_u0[None, :]
                                    + np.outer(dt0, g_u0)
                                    + np.einsum("nij,nj->ni", dR, np.cross(w_q, p_i))
                                )
                                J_u[:, tau_col] = w_ba * np.sum(n_arr * dp_dtau, axis=1)
                                J_u[:, p_off : p_off + 3] = w_ba * (
                                    dt0[:, None] * n_arr
                                )
                                dg_block = (
                                    -0.5
                                    * (dt0**2)[:, None]
                                    * (n_arr @ (_skew(g_u0) @ B_g))
                                )
                                J_u[:, p_off + 3 : p_off + 5] = w_ba * dg_block
                                jac_chunks.append(J_u)

                    tv = turn_vel_samples[u_idx]
                    dR_v, _, dv_v, w_v, dt0_v = _query_imu_preint(table, tv["t"])
                    v_imu_pred = v_u0[None, :] + dv_v + np.outer(dt0_v, g_u0)
                    w_vel = 1.0 / 0.10
                    for v_loc_arr, R_Ik, t_Ik, rot_c, tr_c in (
                        (tv["va_local"], R_IA, t_IA, 0, 3),
                        (tv["vb_local"], R_IB, t_IB, 6, 9),
                    ):
                        v_lid_in_i = v_loc_arr @ R_Ik.T - np.cross(w_v, t_Ik)
                        v_lid_pred = np.einsum("nij,nj->ni", dR_v, v_lid_in_i)
                        res_chunks.append(w_vel * (v_imu_pred - v_lid_pred).reshape(-1))
                        if compute_jac:
                            m_v = len(tv["t"])
                            J_vel = np.zeros((m_v * 3, n_p), dtype=np.float64)
                            v_rot_part = v_loc_arr @ R_Ik.T
                            for k_v in range(m_v):
                                r0, r1 = 3 * k_v, 3 * k_v + 3
                                J_vel[r0:r1, p_off : p_off + 3] = w_vel * np.eye(3)
                                J_vel[r0:r1, p_off + 3 : p_off + 5] = (
                                    -w_vel * dt0_v[k_v] * (_skew(g_u0) @ B_g)
                                )
                                J_vel[r0:r1, rot_c : rot_c + 3] = w_vel * (
                                    dR_v[k_v] @ _skew(v_rot_part[k_v])
                                )
                                J_vel[r0:r1, tr_c : tr_c + 3] = w_vel * (
                                    dR_v[k_v] @ _skew(w_v[k_v])
                                )
                            jac_chunks.append(J_vel)

                    w_g_prior = 1.0 / math.radians(3.5)
                    res_chunks.append(w_g_prior * dtheta_g)
                    if compute_jac:
                        Jg = np.zeros((2, n_p), dtype=np.float64)
                        Jg[:, p_off + 3 : p_off + 5] = w_g_prior * np.eye(2)
                        jac_chunks.append(Jg)

                # Anisotropic L2I priors
                r_ra = W_rot_a @ Rot.from_matrix(R_IA @ T_IA_l2i[:3, :3].T).as_rotvec()
                r_ta = W_trans_a @ (t_IA - T_IA_l2i[:3, 3])
                r_rb = W_rot_b @ Rot.from_matrix(R_IB @ T_IB_l2i[:3, :3].T).as_rotvec()
                r_tb = W_trans_b @ (t_IB - T_IB_l2i[:3, 3])
                res_chunks.extend([r_ra, r_ta, r_rb, r_tb])
                if compute_jac:
                    J_l2i = np.zeros((12, n_p), dtype=np.float64)
                    J_l2i[0:3, 0:3] = W_rot_a
                    J_l2i[3:6, 3:6] = W_trans_a
                    J_l2i[6:9, 6:9] = W_rot_b
                    J_l2i[9:12, 9:12] = W_trans_b
                    jac_chunks.append(J_l2i)

                # Static multi-window L2L SE(3) factors
                T_IA_mat = _se3_from_rt(R_IA, t_IA)
                T_IB_mat = _se3_from_rt(R_IB, t_IB)
                T_AB_pred = np.linalg.inv(T_IA_mat) @ T_IB_mat
                for edge, W_edge in zip(l2l_edges, l2l_whiteners):
                    r_edge = W_edge @ _se3_log6(
                        np.linalg.inv(edge["transformation"]) @ T_AB_pred
                    )
                    res_chunks.append(r_edge)
                    if compute_jac:
                        J_edge = np.zeros((6, n_p), dtype=np.float64)
                        eps = 1e-6
                        for col in range(12):
                            dx_c = np.zeros(12, dtype=np.float64)
                            dx_c[col] = eps
                            Rp_a = (
                                Rot.from_rotvec(x[0:3] + dx_c[0:3]).as_matrix()
                                @ T_IA_s[:3, :3]
                            )
                            tp_a = T_IA_s[:3, 3] + x[3:6] + dx_c[3:6]
                            Rp_b = (
                                Rot.from_rotvec(x[6:9] + dx_c[6:9]).as_matrix()
                                @ T_IB_s[:3, :3]
                            )
                            tp_b = T_IB_s[:3, 3] + x[9:12] + dx_c[9:12]
                            T_AB_p = np.linalg.inv(
                                _se3_from_rt(Rp_a, tp_a)
                            ) @ _se3_from_rt(Rp_b, tp_b)
                            r_p = W_edge @ _se3_log6(
                                np.linalg.inv(edge["transformation"]) @ T_AB_p
                            )
                            J_edge[:, col] = (r_p - r_edge) / eps
                        jac_chunks.append(J_edge)

                # Gravity-ground alignment
                for gw in screened["gravity_windows"]:
                    u_g = gw["u_g_imu"]
                    P_tan = np.eye(3, dtype=np.float64) - np.outer(u_g, u_g)
                    na_i = R_IA @ gw["n_a_lidar"]
                    nb_i = R_IB @ gw["n_b_lidar"]
                    res_chunks.append(w_grav * (P_tan @ na_i))
                    res_chunks.append(w_grav * (P_tan @ nb_i))
                    if compute_jac:
                        Jga = np.zeros((3, n_p), dtype=np.float64)
                        Jga[:, 0:3] = -w_grav * (P_tan @ _skew(na_i))
                        jac_chunks.append(Jga)
                        Jgb = np.zeros((3, n_p), dtype=np.float64)
                        Jgb[:, 6:9] = -w_grav * (P_tan @ _skew(nb_i))
                        jac_chunks.append(Jgb)

                # Full-drive 3D gyro alignment
                for idx_arr, t_l_arr, w_l_arr, R_Ik, tau_k, rot_col, tau_col in (
                    (ga_idx, ta_grid, w_la_sm, R_IA, tau_a, 0, 12),
                    (gb_idx, tb_grid, w_lb_sm, R_IB, tau_b, 6, 13),
                ):
                    t_q = t_l_arr[idx_arr] + tau_k
                    w_i_q = _interp_vec3(batch_a["imu_t"], w_imu_sm, t_q)
                    dw_i_q = _interp_vec3(batch_a["imu_t"], dw_imu_dt, t_q)
                    w_l_rot = w_l_arr[idx_arr] @ R_Ik.T
                    w_g_eff = w_gyro / math.sqrt(max(1.0, len(idx_arr) / 60.0))
                    res_chunks.append(w_g_eff * (w_l_rot - w_i_q).reshape(-1))
                    if compute_jac:
                        m_g = len(idx_arr)
                        J_gy = np.zeros((m_g * 3, n_p), dtype=np.float64)
                        for k_row in range(m_g):
                            J_gy[3 * k_row : 3 * k_row + 3, rot_col : rot_col + 3] = (
                                -w_g_eff * _skew(w_l_rot[k_row])
                            )
                        J_gy[:, tau_col] = (-w_g_eff * dw_i_q).reshape(-1)
                        jac_chunks.append(J_gy)

                # Locked z_cm + L2I temporal priors + shared-clock dtau lock
                z_cm = 0.5 * (t_IA[2] + t_IB[2])
                res_chunks.append(
                    np.array(
                        [
                            w_z_lock * (z_cm - z_cm_v2),
                            w_tau_prior * (tau_a - tau_a_l2i),
                            w_tau_prior * (tau_b - tau_b_l2i),
                            w_dtau_prior * ((tau_a - tau_b) - dtau_l2i),
                        ],
                        dtype=np.float64,
                    )
                )
                if compute_jac:
                    Jz = np.zeros((4, n_p), dtype=np.float64)
                    Jz[0, 5] = 0.5 * w_z_lock
                    Jz[0, 11] = 0.5 * w_z_lock
                    Jz[1, 12] = w_tau_prior
                    Jz[2, 13] = w_tau_prior
                    Jz[3, 12] = w_dtau_prior
                    Jz[3, 13] = -w_dtau_prior
                    jac_chunks.append(Jz)

                r_vec = np.concatenate(res_chunks)
                if not compute_jac:
                    return r_vec
                return r_vec, np.vstack(jac_chunks)

            sol = least_squares(
                lambda x: _eval_v3(x, compute_jac=False),
                np.zeros(n_p, dtype=np.float64),
                jac=lambda x: _eval_v3(x, compute_jac=True)[1],
                method="trf",
                loss="huber",
                f_scale=2.5,
                max_nfev=35,
            )
            dx = sol.x
            T_IA_s = _se3_from_rt(
                Rot.from_rotvec(dx[0:3]).as_matrix() @ T_IA_s[:3, :3],
                T_IA_s[:3, 3] + dx[3:6],
            )
            T_IB_s = _se3_from_rt(
                Rot.from_rotvec(dx[6:9]).as_matrix() @ T_IB_s[:3, :3],
                T_IB_s[:3, 3] + dx[9:12],
            )
            tau_a_s = float(np.clip(tau_a_s + dx[12], -0.25, 0.25))
            tau_b_s = float(np.clip(tau_b_s + dx[13], -0.25, 0.25))
            for sub_pos in range(n_sub):
                p_off = 14 + 5 * sub_pos
                v_sub[sub_pos] += dx[p_off : p_off + 3]
                g_sub[sub_pos] = (
                    Rot.from_rotvec(
                        g_bases[sub_pos] @ dx[p_off + 3 : p_off + 5]
                    ).as_matrix()
                    @ g_sub[sub_pos]
                )
            _, J_last = _eval_v3(np.zeros(n_p), compute_jac=True)

        return {
            "T_I_A": T_IA_s,
            "T_I_B": T_IB_s,
            "tau_a_s": tau_a_s,
            "tau_b_s": tau_b_s,
            "J_final": J_last,
            "n_sub": n_sub,
        }

    all_idx = list(range(len(turn_windows)))
    full_res = _solve_v3_on_subset(all_idx, cfg.v3_outer_iters)
    T_IA_curr = full_res["T_I_A"]
    T_IB_curr = full_res["T_I_B"]
    tau_a_curr = full_res["tau_a_s"]
    tau_b_curr = full_res["tau_b_s"]
    J_final = full_res["J_final"]
    n_turns = len(turn_windows)

    _, v2_turn_sigma_cm = _extract_turn_plane_correspondences(
        v2_solution["T_I_A"],
        v2_solution["T_I_B"],
        float(v2_solution["tau_a_s"]),
        float(v2_solution["tau_b_s"]),
        0.70,
    )
    _, v3_turn_sigma_cm = _extract_turn_plane_correspondences(
        T_IA_curr, T_IB_curr, tau_a_curr, tau_b_curr, 0.70
    )
    v3_gyro_rmse = _full_gyro_rmse(
        T_IA_curr[:3, :3], tau_a_curr, T_IB_curr[:3, :3], tau_b_curr
    )

    # Leave-One-Turn-Out (LOTO) out-of-sample cross-validation guard
    loto_v2_sigmas: list[float] = []
    loto_v3_sigmas: list[float] = []
    if n_turns >= 2:
        for u_hold in range(n_turns):
            train_idx = [i for i in range(n_turns) if i != u_hold]
            fold_res = _solve_v3_on_subset(train_idx, min(3, cfg.v3_outer_iters))
            sig_v2_u = _fit_single_turn_kinematics_frozen(
                u_hold,
                v2_solution["T_I_A"],
                v2_solution["T_I_B"],
                float(v2_solution["tau_a_s"]),
                float(v2_solution["tau_b_s"]),
            )
            sig_v3_u = _fit_single_turn_kinematics_frozen(
                u_hold,
                fold_res["T_I_A"],
                fold_res["T_I_B"],
                fold_res["tau_a_s"],
                fold_res["tau_b_s"],
            )
            loto_v2_sigmas.append(sig_v2_u)
            loto_v3_sigmas.append(sig_v3_u)

    loto_v2_mean = (
        float(np.mean(loto_v2_sigmas)) if loto_v2_sigmas else float(v2_turn_sigma_cm)
    )
    loto_v3_mean = (
        float(np.mean(loto_v3_sigmas)) if loto_v3_sigmas else float(v3_turn_sigma_cm)
    )

    reverted = False
    if loto_v3_mean > loto_v2_mean + 0.02 or v3_gyro_rmse > v2_gyro_rmse + 0.0002:
        reverted = True
        T_IA_curr = v2_solution["T_I_A"].copy()
        T_IB_curr = v2_solution["T_I_B"].copy()
        tau_a_curr = float(v2_solution["tau_a_s"])
        tau_b_curr = float(v2_solution["tau_b_s"])

    # Compute final 14x14 observability on the calibration subset
    H_full = J_final.T @ J_final
    H_cc = H_full[:14, :14]
    H_ca = H_full[:14, 14:]
    H_aa = H_full[14:, 14:] + 1e-6 * np.eye(5 * n_turns)
    H_calib = H_cc - H_ca @ np.linalg.solve(H_aa, H_ca.T)

    s_half = math.sqrt(0.5)
    I6 = np.eye(6, dtype=np.float64)
    Q_cd = np.block([[s_half * I6, s_half * I6], [-s_half * I6, s_half * I6]])
    H_cd = Q_cd @ H_calib[:12, :12] @ Q_cd.T
    eig_cm = np.linalg.eigvalsh(0.5 * (H_cd[:6, :6] + H_cd[:6, :6].T))
    eig_diff = np.linalg.eigvalsh(0.5 * (H_cd[6:12, 6:12] + H_cd[6:12, 6:12].T))
    eig_tau = np.linalg.eigvalsh(
        0.5 * (H_calib[12:14, 12:14] + H_calib[12:14, 12:14].T)
    )

    return {
        "T_I_A": T_IA_curr,
        "T_I_B": T_IB_curr,
        "T_A_B": np.linalg.inv(T_IA_curr) @ T_IB_curr,
        "tau_a_s": tau_a_curr,
        "tau_b_s": tau_b_curr,
        "z_cm_prior_m": z_cm_prior,
        "v2_turn_wall_sigma_cm": float(v2_turn_sigma_cm),
        "v3_turn_wall_sigma_cm": float(v3_turn_sigma_cm),
        "loto_v2_turn_wall_sigma_cm": loto_v2_mean,
        "loto_v3_turn_wall_sigma_cm": loto_v3_mean,
        "v2_gyro_rmse_rad_s": float(v2_gyro_rmse),
        "v3_gyro_rmse_rad_s": float(v3_gyro_rmse),
        "reverted_to_v2": reverted,
        "observability": {
            "common_mode_6d_eigenvalues": [float(v) for v in eig_cm],
            "differential_mode_6d_eigenvalues": [float(v) for v in eig_diff],
            "temporal_2d_eigenvalues": [float(v) for v in eig_tau],
            "common_mode_z_prior_anchored": True,
        },
    }


def _solution_summary(
    name: str,
    sol: dict[str, Any],
    screened: dict[str, Any],
) -> dict[str, Any]:
    T_IA = sol["T_I_A"]
    T_IB = sol["T_I_B"]
    T_AB = sol["T_A_B"]
    rpy_a = Rot.from_matrix(T_IA[:3, :3]).as_euler("xyz", degrees=True).tolist()
    rpy_b = Rot.from_matrix(T_IB[:3, :3]).as_euler("xyz", degrees=True).tolist()
    dT_a_rot = _rotation_error_deg(screened["T_I_A_init"][:3, :3], T_IA[:3, :3])
    dT_b_rot = _rotation_error_deg(screened["T_I_B_init"][:3, :3], T_IB[:3, :3])
    dT_a_t = float(np.linalg.norm(T_IA[:3, 3] - screened["T_I_A_init"][:3, 3]))
    dT_b_t = float(np.linalg.norm(T_IB[:3, 3] - screened["T_I_B_init"][:3, 3]))
    return {
        "stage": name,
        "T_I_A": T_IA.tolist(),
        "T_I_B": T_IB.tolist(),
        "T_A_B": T_AB.tolist(),
        "sensor_a_xyz_deg": [float(v) for v in rpy_a],
        "sensor_a_t_LI_m": [float(v) for v in T_IA[:3, 3]],
        "sensor_a_tau_s": float(sol["tau_a_s"]),
        "sensor_b_xyz_deg": [float(v) for v in rpy_b],
        "sensor_b_t_LI_m": [float(v) for v in T_IB[:3, 3]],
        "sensor_b_tau_s": float(sol["tau_b_s"]),
        "delta_from_l2i_baseline": {
            "sensor_a_rot_deg": dT_a_rot,
            "sensor_a_trans_m": dT_a_t,
            "sensor_b_rot_deg": dT_b_rot,
            "sensor_b_trans_m": dT_b_t,
        },
        "observability": sol.get("observability", {}),
        "solver": sol.get("solver", {}),
    }


def _export_l2l_feature_debug(
    feature_dir: Path,
    feature_traces: list[dict[str, Any]],
    feature_rejections: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    screened: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    from lidar2lidar.feature_visualization import (
        export_feature_index,
        export_feature_window,
    )

    lookup = {e["window_id"]: e for e in edges if e["method"] == "semantic_v2"}
    reports = []
    for record in feature_traces:
        window = record["metadata"]
        reports.append(
            export_feature_window(
                feature_dir / window["window_id"],
                record["trace"],
                record["result"],
                {**window, **provenance},
                edge=lookup.get(window["window_id"]),
            )
        )
    for rejected in feature_rejections:
        metadata = {**rejected["metadata"], **provenance, "screening_rejected": True}
        if rejected["frame_a"] is None or rejected["frame_b"] is None:
            reports.append(
                {
                    **metadata,
                    "status": "screening-rejected",
                    "solver_success": False,
                    "failure_reason": metadata["skip_reason"],
                    "has_gallery": False,
                }
            )
            continue
        trace = {
            "source": {},
            "target": {},
            "stages": [],
            "T_initial": screened["T_AB_init"],
            "configs": {"primitive_extraction": asdict(PrimitiveExtractionConfig())},
        }
        for side, frame_key in (("source", "frame_b"), ("target", "frame_a")):
            extract_geometric_primitives(rejected[frame_key], trace=trace[side])
        reports.append(
            export_feature_window(
                feature_dir / metadata["window_id"],
                trace,
                {"success": False, "reason": metadata["skip_reason"]},
                metadata,
            )
        )
    screening = screened["static_screening_log"]
    if feature_rejections:
        screening = [r for r in screening if r.get("accepted")] + [
            r["metadata"] for r in feature_rejections
        ]
    export_feature_index(feature_dir, reports, screening, metadata=provenance)


def run_joint_dual_calibration(
    *,
    dataset_a: Path,
    run_dir_a: Path,
    dataset_b: Path,
    run_dir_b: Path,
    output_dir: Path,
    label_a: str = "left_front",
    label_b: str = "right_back",
    mode: str = "v2",
    config: JointCalibrationConfig = JointCalibrationConfig(),
    save_l2l_feature_debug: bool = False,
) -> dict[str, Any]:
    """Run staged Extraction -> Joint Solve (v1/v2/v3) -> Standardized Export."""
    if mode not in {"v1", "v2", "v3", "all"}:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if save_l2l_feature_debug and mode == "v1":
        raise ValueError("L2L feature debugging requires semantic mode v2, v3 or all")
    if save_l2l_feature_debug:
        from importlib.util import find_spec

        if find_spec("matplotlib") is None:
            raise RuntimeError(
                'L2L feature debugging requires pip install -e ".[diagnostics]"'
            )

    output_dir = Path(output_dir)
    diag_dir = output_dir / "diagnostics"
    ext_dir = output_dir / "extrinsics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    access_a = FastCanonicalScanAccess(Path(dataset_a))
    access_b = FastCanonicalScanAccess(Path(dataset_b))
    result_a = parse_detailed_gril_result(Path(run_dir_a) / "GRIL_Calib_result.txt")
    result_b = parse_detailed_gril_result(Path(run_dir_b) / "GRIL_Calib_result.txt")
    batch_a = parse_batch_trace_arrays(Path(run_dir_a) / "GRIL_batch_trace_v1.txt")
    batch_b = parse_batch_trace_arrays(Path(run_dir_b) / "GRIL_batch_trace_v1.txt")
    feature_rejections: list[dict[str, Any]] = []
    provenance = {
        "source_label": label_b,
        "target_label": label_a,
        "source_dataset": str(Path(dataset_b).resolve()),
        "target_dataset": str(Path(dataset_a).resolve()),
        "run_dir_a": str(Path(run_dir_a).resolve()),
        "run_dir_b": str(Path(run_dir_b).resolve()),
    }

    screened = screen_joint_calibration_windows(
        access_a,
        access_b,
        result_a,
        result_b,
        batch_a,
        batch_b,
        config,
        include_turn_windows=(mode in {"v3", "all"}),
        feature_rejections=feature_rejections if save_l2l_feature_debug else None,
    )
    if not screened["static_windows"]:
        if save_l2l_feature_debug:
            _export_l2l_feature_debug(
                diag_dir / "l2l_features",
                [],
                feature_rejections,
                [],
                screened,
                provenance,
            )
        raise RuntimeError("No valid static shared-FoV windows found for L2L coupling")

    H_rot_a, H_trans_a = _compute_l2i_anisotropic_info(result_a, batch_a, config)
    H_rot_b, H_trans_b = _compute_l2i_anisotropic_info(result_b, batch_b, config)

    stages: dict[str, Any] = {}
    selected_stage = "v1"

    # v1: Anisotropic factor graph + multi-window GICP L2L
    edges_v1 = _evaluate_l2l_edges_gicp(screened)
    sol_v1 = _solve_anisotropic_factor_graph(
        screened=screened,
        l2l_edges=edges_v1,
        H_rot_a=H_rot_a,
        H_trans_a=H_trans_a,
        H_rot_b=H_rot_b,
        H_trans_b=H_trans_b,
        cfg=config,
    )
    stages["v1"] = _solution_summary("v1_anisotropic_gicp_graph", sol_v1, screened)
    final_sol = sol_v1

    edges_v2: list[dict[str, Any]] = []
    feature_traces: list[dict[str, Any]] = []
    semantic_failures: list[dict[str, Any]] = []
    if mode in {"v2", "v3", "all"}:
        edges_v2 = _evaluate_l2l_edges_semantic(
            screened,
            sol_v1["T_A_B"],
            config,
            feature_traces=feature_traces if save_l2l_feature_debug else None,
            failures=semantic_failures,
        )
        if edges_v2:
            sol_v2 = _solve_anisotropic_factor_graph(
                screened=screened,
                l2l_edges=edges_v2,
                H_rot_a=H_rot_a,
                H_trans_a=H_trans_a,
                H_rot_b=H_rot_b,
                H_trans_b=H_trans_b,
                cfg=config,
                T_I_A_init=sol_v1["T_I_A"],
                T_I_B_init=sol_v1["T_I_B"],
            )
            stages["v2"] = _solution_summary(
                "v2_semantic_reflectivity_3dfov_graph", sol_v2, screened
            )
            final_sol = sol_v2
            selected_stage = "v2"
        else:
            logging.getLogger(__name__).warning(
                "No semantic L2L edges succeeded; falling back to v1 GICP edges. "
                "Failure details are recorded in diagnostics/l2l_edges.yaml."
            )

    if mode in {"v3", "all"}:
        sol_v3 = _solve_turn_imu_ba_v3(
            screened=screened,
            v2_solution=final_sol,
            l2l_edges=edges_v2 if edges_v2 else edges_v1,
            H_rot_a=H_rot_a,
            H_trans_a=H_trans_a,
            H_rot_b=H_rot_b,
            H_trans_b=H_trans_b,
            batch_a=batch_a,
            batch_b=batch_b,
            cfg=config,
        )
        stages["v3"] = _solution_summary("v3_turn_imu_preint_ba", sol_v3, screened)
        stages["v3"]["v2_turn_wall_sigma_cm"] = sol_v3.get("v2_turn_wall_sigma_cm")
        stages["v3"]["v3_turn_wall_sigma_cm"] = sol_v3.get("v3_turn_wall_sigma_cm")
        stages["v3"]["loto_v2_turn_wall_sigma_cm"] = sol_v3.get(
            "loto_v2_turn_wall_sigma_cm"
        )
        stages["v3"]["loto_v3_turn_wall_sigma_cm"] = sol_v3.get(
            "loto_v3_turn_wall_sigma_cm"
        )
        stages["v3"]["v2_gyro_rmse_rad_s"] = sol_v3.get("v2_gyro_rmse_rad_s")
        stages["v3"]["v3_gyro_rmse_rad_s"] = sol_v3.get("v3_gyro_rmse_rad_s")
        stages["v3"]["reverted_to_v2"] = sol_v3.get("reverted_to_v2", False)
        if not sol_v3.get("reverted_to_v2", False):
            final_sol = sol_v3
            selected_stage = "v3"

    # Export standardized extrinsics for selected final stage + per-stage YAMLs
    frame_imu = access_a.imu_frame
    frame_a = (
        access_a.lidar_frame
        if access_a.lidar_frame != access_b.lidar_frame
        else label_a
    )
    frame_b = (
        access_b.lidar_frame
        if access_a.lidar_frame != access_b.lidar_frame
        else label_b
    )

    def _export_stage_extrinsics(
        target_dir: Path, sol: dict[str, Any], tag: str
    ) -> dict[str, str]:
        target_dir.mkdir(parents=True, exist_ok=True)
        pa = target_dir / extrinsics_filename(frame_imu, frame_a)
        pb = target_dir / extrinsics_filename(frame_imu, frame_b)
        pab = target_dir / extrinsics_filename(frame_a, frame_b)
        save_extrinsics_yaml(
            str(pa),
            parent_frame=frame_imu,
            child_frame=frame_a,
            matrix=sol["T_I_A"],
            metadata={"time_offset_s": float(sol["tau_a_s"]), "stage": tag},
        )
        save_extrinsics_yaml(
            str(pb),
            parent_frame=frame_imu,
            child_frame=frame_b,
            matrix=sol["T_I_B"],
            metadata={"time_offset_s": float(sol["tau_b_s"]), "stage": tag},
        )
        save_extrinsics_yaml(
            str(pab),
            parent_frame=frame_a,
            child_frame=frame_b,
            matrix=sol["T_A_B"],
            metadata={
                "relative_time_offset_s": float(sol["tau_a_s"] - sol["tau_b_s"]),
                "stage": tag,
            },
        )
        return {
            "imu_a": str(pa.resolve()),
            "imu_b": str(pb.resolve()),
            "a_b": str(pab.resolve()),
        }

    exported_final = _export_stage_extrinsics(ext_dir, final_sol, selected_stage)
    stage_exports: dict[str, dict[str, str]] = {}
    for st_name, st_sol_dict in stages.items():
        st_sol_mat = {
            "T_I_A": np.asarray(st_sol_dict["T_I_A"], dtype=np.float64),
            "T_I_B": np.asarray(st_sol_dict["T_I_B"], dtype=np.float64),
            "T_A_B": np.asarray(st_sol_dict["T_A_B"], dtype=np.float64),
            "tau_a_s": float(st_sol_dict["sensor_a_tau_s"]),
            "tau_b_s": float(st_sol_dict["sensor_b_tau_s"]),
        }
        stage_exports[st_name] = _export_stage_extrinsics(
            output_dir / "stages" / st_name, st_sol_mat, st_name
        )

    # Write Stage 1 & 2 diagnostics
    screening_report = {
        "static_windows_accepted": len(screened["static_windows"]),
        "static_candidates": screened["static_screening_log"],
        "gravity_windows_accepted": len(screened["gravity_windows"]),
        "turn_windows_accepted": len(screened["turn_windows"]),
        "turn_candidates": screened["turn_screening_log"],
    }
    (diag_dir / "window_screening.yaml").write_text(
        yaml.safe_dump(screening_report, sort_keys=False)
    )

    def _clean_edge(e: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in e.items()
            if k != "info_matrix_6d"
        }

    l2l_report = {
        "v1_gicp_edges": [_clean_edge(e) for e in edges_v1],
        "v2_semantic_edges": (
            [_clean_edge(e) for e in edges_v2] if mode in {"v2", "v3", "all"} else []
        ),
        "v2_semantic_failures": semantic_failures,
        "semantic_fallback_to_gicp": mode != "v1" and not edges_v2,
    }
    (diag_dir / "l2l_edges.yaml").write_text(
        yaml.safe_dump(l2l_report, sort_keys=False)
    )

    active_edges = edges_v2 if edges_v2 else edges_v1
    excitation_gate = evaluate_common_mode_excitation_gate(
        batch_a, active_edges, config
    )
    if selected_stage == "v2":
        window_influence = _evaluate_l2l_window_influence(
            screened=screened,
            l2l_edges=edges_v2,
            solution=sol_v2,
            H_rot_a=H_rot_a,
            H_trans_a=H_trans_a,
            H_rot_b=H_rot_b,
            H_trans_b=H_trans_b,
            cfg=config,
            T_I_A_init=sol_v1["T_I_A"],
            T_I_B_init=sol_v1["T_I_B"],
        )
    else:
        window_influence = {
            "status": "not_evaluated",
            "reason": "requires_selected_v2_graph",
        }
    (diag_dir / "l2l_window_influence.yaml").write_text(
        yaml.safe_dump(window_influence, sort_keys=False)
    )

    report = {
        "schema_version": 1,
        "module": "lidar2imu_joint_dual_calib",
        "verdict": excitation_gate["verdict"],
        "verdict_scope": "common_mode_excitation_only",
        "requested_mode": mode,
        "selected_stage": selected_stage,
        "semantic_fallback_to_gicp": mode != "v1" and not edges_v2,
        "frames": {"imu": frame_imu, "sensor_a": frame_a, "sensor_b": frame_b},
        "extrinsic_files": exported_final,
        "stage_extrinsic_files": stage_exports,
        "common_mode_excitation_gate": excitation_gate,
        "l2l_window_influence": window_influence,
        "screening_summary": {
            "static_windows": len(screened["static_windows"]),
            "gravity_windows": len(screened["gravity_windows"]),
            "turn_windows": len(screened["turn_windows"]),
        },
        "stages": stages,
        "final": stages[selected_stage],
    }
    (output_dir / "joint_calibration_result.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False)
    )
    if save_l2l_feature_debug:
        _export_l2l_feature_debug(
            diag_dir / "l2l_features",
            feature_traces,
            feature_rejections,
            edges_v2,
            screened,
            {
                **provenance,
                "validation_artifacts": {
                    "Fixed-factor L2L window influence": "../l2l_window_influence.yaml"
                },
            },
        )
    return report
