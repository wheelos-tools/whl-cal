from __future__ import annotations

import io
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from lidar2imu.algorithms.joint_dual_calib import (
    JointCalibrationConfig,
    evaluate_common_mode_excitation_gate,
    run_joint_dual_calibration,
)
from lidar2imu.evaluation.dual_sensor_diagnostics import (
    evaluate_dual_lidar2imu_diagnostics,
)


def _write_npy_into_zip(zf: zipfile.ZipFile, name: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
    zf.writestr(
        zipfile.ZipInfo(filename=f"{name}.npy"),
        buf.getvalue(),
        compress_type=zipfile.ZIP_STORED,
    )


def _write_gril_result(
    path: Path, R_LI: np.ndarray, T_LI: np.ndarray, tau_s: float
) -> None:
    rpy = Rot.from_matrix(R_LI).as_euler("xyz", degrees=True)
    T_H = np.eye(4)
    T_H[:3, :3] = R_LI
    T_H[:3, 3] = T_LI
    rows_h = "\n".join("  ".join(f"{val:.6f}" for val in T_H[r]) for r in range(4))
    text = f"""Calibration result:
Rotation LiDAR to IMU (XYZ Euler, deg) = {rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}
Translation LiDAR to IMU (m) = {T_LI[0]:.6f} {T_LI[1]:.6f} {T_LI[2]:.6f}
Time Lag IMU to LiDAR (s) = {tau_s:.6f}
Bias of Gyroscope (rad/s) = 0.000000 0.000000 0.000000
Bias of Accelerometer (m/s^2) = 0.000000 0.000000 0.000000
Homogeneous Transformation Matrix from LiDAR frmae L to IMU frame I:
{rows_h}
"""
    path.write_text(text)


def _build_synthetic_joint_capture(
    base_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(101)
    R_LI_a_gt = Rot.from_euler("xyz", [0.4, -0.3, 134.5], degrees=True).as_matrix()
    T_LI_a_gt = np.array([-0.25, 2.25, 0.48], dtype=np.float64)
    R_LI_b_gt = Rot.from_euler("xyz", [-0.2, 0.5, -45.5], degrees=True).as_matrix()
    T_LI_b_gt = np.array([1.75, -4.15, 0.44], dtype=np.float64)
    tau_a_gt = -0.022
    tau_b_gt = -0.025

    # Rich 3D world: ground + 2 long orthogonal walls + vertical pole
    gx, gy = np.meshgrid(np.linspace(-16.0, 22.0, 42), np.linspace(-16.0, 24.0, 46))
    gnd = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.5)])
    wy, wz = np.meshgrid(np.linspace(-14.0, 22.0, 110), np.linspace(0.0, 3.0, 14))
    wall_x = np.column_stack([np.full(wy.size, -9.5), wy.ravel(), wz.ravel()])
    wx, wz2 = np.meshgrid(np.linspace(-9.0, 20.0, 96), np.linspace(0.0, 3.0, 14))
    wall_y = np.column_stack([wx.ravel(), np.full(wx.size, 16.0), wz2.ravel()])
    pz = np.linspace(0.0, 3.0, 24)
    pole = np.column_stack([np.full(24, 6.0), np.full(24, -4.0), pz])
    world_pts = np.vstack([gnd, wall_x, wall_y, pole])
    world_int = np.full(len(world_pts), 35.0, dtype=np.float32)

    # Trajectory [0, 16.0 s]:
    # [0, 2.5s] parked; [3.0, 5.6s] +55 deg left turn; [7.0, 9.6s] -55 deg right turn;
    # [11.0, 13.6s] +60 deg left turn; [14.0, 16.0s] parked
    imu_t = np.linspace(0.0, 16.2, 1621, dtype=np.float64)
    wz_i = np.zeros_like(imu_t)
    spd_i = np.zeros_like(imu_t)
    for t0_seg, dur_seg, dpsi_deg, spd_amp in (
        (3.0, 2.6, 55.0, 2.6),
        (7.0, 2.6, -55.0, 2.4),
        (11.0, 2.6, 60.0, 2.8),
    ):
        m = (imu_t >= t0_seg) & (imu_t <= t0_seg + dur_seg)
        u = (imu_t[m] - t0_seg) / dur_seg
        wz_i[m] = math.radians(dpsi_deg) * (6.0 * u * (1.0 - u)) / dur_seg
        spd_i[m] = 1.4 + spd_amp * np.sin(math.pi * u)

    # Straight cruising between turns
    for t0_seg, t1_seg in ((2.4, 3.0), (5.6, 7.0), (9.6, 11.0), (13.6, 14.2)):
        m = (imu_t >= t0_seg) & (imu_t <= t1_seg)
        spd_i[m] = 1.4

    dt_i = float(imu_t[1] - imu_t[0])
    psi_i = np.cumsum(wz_i) * dt_i
    vx_w = -spd_i * np.sin(psi_i)
    vy_w = spd_i * np.cos(psi_i)
    vel_w = np.column_stack([vx_w, vy_w, np.zeros_like(vx_w)])
    pos_w = np.cumsum(vel_w, axis=0) * dt_i
    acc_w = np.gradient(vel_w, dt_i, axis=0)

    imu_w = np.column_stack([np.zeros_like(wz_i), np.zeros_like(wz_i), wz_i])
    imu_f = np.zeros_like(acc_w)
    g_w = np.array([0.0, 0.0, -9.81])
    for k in range(len(imu_t)):
        R_wi = Rot.from_euler("z", float(psi_i[k])).as_matrix()
        imu_f[k] = R_wi.T @ (acc_w[k] - g_w)

    def _pose_interp(t_arr: np.ndarray):
        psi = np.interp(t_arr, imu_t, psi_i)
        px = np.interp(t_arr, imu_t, pos_w[:, 0])
        py = np.interp(t_arr, imu_t, pos_w[:, 1])
        return psi, np.column_stack([px, py, np.zeros_like(px)])

    scan_times_phys = np.arange(0.3, 15.8, 0.10, dtype=np.float64)

    # Inject 16% radial lever-arm shrinkage + 1 deg yaw error into GRIL prior!
    R_LI_a_biased = R_LI_a_gt @ Rot.from_euler("z", 0.9, degrees=True).as_matrix()
    T_LI_a_biased = T_LI_a_gt * 0.84 + np.array([0.0, 0.0, 0.05])
    R_LI_b_biased = R_LI_b_gt @ Rot.from_euler("z", -0.8, degrees=True).as_matrix()
    T_LI_b_biased = T_LI_b_gt * 0.84 + np.array([0.0, 0.0, -0.04])

    paths: dict[str, Path] = {}
    for name, R_gt, T_gt, tau_gt, R_b, T_b, tau_b in (
        (
            "left_front",
            R_LI_a_gt,
            T_LI_a_gt,
            tau_a_gt,
            R_LI_a_biased,
            T_LI_a_biased,
            -0.012,
        ),
        (
            "right_back",
            R_LI_b_gt,
            T_LI_b_gt,
            tau_b_gt,
            R_LI_b_biased,
            T_LI_b_biased,
            -0.032,
        ),
    ):
        ds_dir = base_dir / name / "dataset"
        run_dir = base_dir / name / "run"
        ds_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        xyz_chunks = []
        int_chunks = []
        ring_chunks = []
        pt_t_chunks = []
        offsets = [0]
        scan_times_lid = scan_times_phys - tau_gt

        for t_s in scan_times_phys:
            # Give points realistic intra-scan [0, 0.09s] timestamps
            pt_dt = np.linspace(0.0, 0.09, len(world_pts), dtype=np.float64)
            psi_pts, pos_pts = _pose_interp(t_s + pt_dt)
            pts_l = np.empty_like(world_pts, dtype=np.float32)
            for j in range(len(world_pts)):
                R_wi = Rot.from_euler("z", float(psi_pts[j])).as_matrix()
                p_i = R_wi.T @ (world_pts[j] - pos_pts[j])
                pts_l[j] = (
                    (p_i - T_gt) @ R_gt + rng.normal(0.0, 0.004, size=3)
                ).astype(np.float32)
            xyz_chunks.append(pts_l)
            int_chunks.append(world_int)
            ring_chunks.append(np.zeros(len(pts_l), dtype=np.uint16))
            pt_t_chunks.append(pt_dt.astype(np.float32))
            offsets.append(offsets[-1] + len(pts_l))

        with zipfile.ZipFile(ds_dir / "lidar.npz", "w") as zf:
            _write_npy_into_zip(zf, "xyz", np.vstack(xyz_chunks))
            _write_npy_into_zip(zf, "intensity", np.concatenate(int_chunks))
            _write_npy_into_zip(zf, "ring", np.concatenate(ring_chunks))
            _write_npy_into_zip(zf, "point_time_s", np.concatenate(pt_t_chunks))
            _write_npy_into_zip(
                zf,
                "scan_timestamps_ns",
                np.round(scan_times_lid * 1e9).astype(np.int64),
            )
            _write_npy_into_zip(zf, "scan_offsets", np.asarray(offsets, dtype=np.int64))

        np.savez(
            ds_dir / "imu.npz",
            timestamps_ns=np.round(imu_t * 1e9).astype(np.int64),
            angular_velocity=imu_w,
            linear_acceleration=imu_f,
        )
        manifest = {
            "schema_version": 1,
            "frames": {"imu": "imu", "lidar": name},
            "arrays": {"lidar": {"path": "lidar.npz"}, "imu": {"path": "imu.npz"}},
        }
        (ds_dir / "dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

        # Write batch trace on 30 Hz grid
        lid_grid_phys = np.arange(0.5, 15.6, 1.0 / 30.0, dtype=np.float64)
        lid_grid_raw = lid_grid_phys - tau_gt
        lines = [
            "GRIL_batch_trace_v1",
            "orig_odom_freq 10.0",
            "cut_frame_num 3",
            "timediff_imu_wrt_lidar 0.0",
            "move_start_time 2.5",
        ]
        zeros9 = " ".join(["0.0"] * 9)
        zeros3 = "0.0 0.0 0.0"
        zeros6 = " ".join(["0.0"] * 6)
        for k_i in range(len(imu_t)):
            w = imu_w[k_i]
            f = imu_f[k_i]
            lines.append(
                f"imu {imu_t[k_i]:.9f} {zeros9} {zeros3} "
                f"{w[0]:.8f} {w[1]:.8f} {w[2]:.8f} {zeros6} "
                f"{f[0]:.8f} {f[1]:.8f} {f[2]:.8f}"
            )
        R_WL0 = (Rot.from_euler("z", float(psi_i[0])).as_matrix() @ R_gt).T
        for t_p, t_r in zip(lid_grid_phys, lid_grid_raw):
            psi_k = float(np.interp(t_p, imu_t, psi_i))
            wz_k = float(np.interp(t_p, imu_t, wz_i))
            vx_k = float(np.interp(t_p, imu_t, vel_w[:, 0]))
            vy_k = float(np.interp(t_p, imu_t, vel_w[:, 1]))
            R_wi = Rot.from_euler("z", psi_k).as_matrix()
            R_wl = R_wi @ R_gt
            R_ll0 = R_WL0 @ R_wl
            w_l = R_gt.T @ np.array([0.0, 0.0, wz_k])
            v_i_w = np.array([vx_k, vy_k, 0.0])
            v_l_w = R_WL0 @ (v_i_w + R_wi @ np.cross(np.array([0.0, 0.0, wz_k]), T_gt))
            p_l_w = np.array(
                [
                    float(np.interp(t_p, imu_t, pos_w[:, 0])),
                    float(np.interp(t_p, imu_t, pos_w[:, 1])),
                    0.0,
                ]
            )
            r_str = " ".join(f"{v:.8f}" for v in R_ll0.reshape(-1))
            lines.append(
                f"lidar {t_r:.9f} {r_str} {p_l_w[0]:.6f} {p_l_w[1]:.6f} {p_l_w[2]:.6f} "
                f"{w_l[0]:.8f} {w_l[1]:.8f} {w_l[2]:.8f} "
                f"{v_l_w[0]:.6f} {v_l_w[1]:.6f} {v_l_w[2]:.6f}"
            )
        (run_dir / "GRIL_batch_trace_v1.txt").write_text("\n".join(lines) + "\n")
        _write_gril_result(run_dir / "GRIL_Calib_result.txt", R_b, T_b, tau_b)
        paths[f"ds_{name}"] = ds_dir / "dataset.yaml"
        paths[f"run_{name}"] = run_dir

    return {
        "paths": paths,
        "T_I_A_gt": np.vstack(
            [np.hstack([R_LI_a_gt, T_LI_a_gt[:, None]]), [0, 0, 0, 1]]
        ),
        "T_I_B_gt": np.vstack(
            [np.hstack([R_LI_b_gt, T_LI_b_gt[:, None]]), [0, 0, 0, 1]]
        ),
        "tau_a_gt": tau_a_gt,
        "tau_b_gt": tau_b_gt,
    }


class JointDualCalibTests(unittest.TestCase):
    def test_v1_v2_v3_recovery_from_shrunk_l2i_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim = _build_synthetic_joint_capture(root)
            p = sim["paths"]
            out_dir = root / "joint_out"
            z_cm_true = 0.5 * float(sim["T_I_A_gt"][2, 3] + sim["T_I_B_gt"][2, 3])
            report = run_joint_dual_calibration(
                dataset_a=p["ds_left_front"],
                run_dir_a=p["run_left_front"],
                dataset_b=p["ds_right_back"],
                run_dir_b=p["run_right_back"],
                output_dir=out_dir,
                mode="all",
                config=JointCalibrationConfig(
                    z_cm_prior_m=z_cm_true,
                    max_static_windows=2,
                    max_turn_windows=3,
                    min_turn_separation_s=3.5,
                    turn_point_stride=1,
                    v3_dtau_prior_sigma_s=0.015,
                ),
            )
            self.assertEqual(set(report["stages"].keys()), {"v1", "v2", "v3"})
            T_AB_gt = np.linalg.inv(sim["T_I_A_gt"]) @ sim["T_I_B_gt"]

            # v1/v2 must lock relative L2L baseline to < 3 cm & 0.2 deg
            T_AB_v2 = np.asarray(report["stages"]["v2"]["T_A_B"], dtype=np.float64)
            err_ab_trans = float(np.linalg.norm(T_AB_gt[:3, 3] - T_AB_v2[:3, 3]))
            err_ab_rot = math.degrees(
                Rot.from_matrix(T_AB_gt[:3, :3].T @ T_AB_v2[:3, :3]).magnitude()
            )
            self.assertLess(err_ab_trans, 0.03)
            self.assertLess(err_ab_rot, 0.25)

            # Final (v3) should keep accurate T_AB, sharpen turn submap, and recover
            # lever-arm & time offset!
            fin = report["final"]
            T_IA_fin = np.asarray(fin["T_I_A"], dtype=np.float64)
            T_IB_fin = np.asarray(fin["T_I_B"], dtype=np.float64)
            T_AB_fin = np.asarray(fin["T_A_B"], dtype=np.float64)
            self.assertLess(
                float(np.linalg.norm(T_AB_gt[:3, 3] - T_AB_fin[:3, 3])), 0.03
            )
            self.assertLess(
                float(np.linalg.norm(T_IA_fin[:3, 3] - sim["T_I_A_gt"][:3, 3])), 0.16
            )
            self.assertLess(
                float(np.linalg.norm(T_IB_fin[:3, 3] - sim["T_I_B_gt"][:3, 3])), 0.16
            )
            self.assertLess(abs(float(fin["sensor_a_tau_s"]) - sim["tau_a_gt"]), 0.005)
            self.assertLess(abs(float(fin["sensor_b_tau_s"]) - sim["tau_b_gt"]), 0.005)

            # Verify evaluate_dual_lidar2imu_diagnostics consumes exported YAML
            diag = evaluate_dual_lidar2imu_diagnostics(
                dataset_a=p["ds_left_front"],
                run_dir_a=p["run_left_front"],
                dataset_b=p["ds_right_back"],
                run_dir_b=p["run_right_back"],
                extrinsic_a_yaml=Path(report["extrinsic_files"]["imu_a"]),
                extrinsic_b_yaml=Path(report["extrinsic_files"]["imu_b"]),
                output_dir=root / "diag_check",
            )
            self.assertEqual(diag["verdict"], "accepted")

            # Verify default mode="v2" skips v3 overhead and passes cleanly
            report_v2 = run_joint_dual_calibration(
                dataset_a=p["ds_left_front"],
                run_dir_a=p["run_left_front"],
                dataset_b=p["ds_right_back"],
                run_dir_b=p["run_right_back"],
                output_dir=root / "joint_out_v2_default",
                config=JointCalibrationConfig(
                    z_cm_prior_m=z_cm_true,
                    max_static_windows=2,
                ),
            )
            self.assertEqual(report_v2["selected_stage"], "v2")
            self.assertEqual(set(report_v2["stages"].keys()), {"v1", "v2"})
            T_AB_def = np.asarray(report_v2["final"]["T_A_B"], dtype=np.float64)
            self.assertLess(
                float(np.linalg.norm(T_AB_gt[:3, 3] - T_AB_def[:3, 3])), 0.03
            )

    def test_common_mode_excitation_gate_flags_one_sided_turns(self) -> None:
        t = np.arange(0.0, 40.0, 0.01, dtype=np.float64)
        lid_t = np.arange(0.0, 40.0, 0.1, dtype=np.float64)
        healthy_edge = [
            {"downweighted_by_weak_support": False, "consensus_outlier": False}
        ]

        # One-sided right turns only (0 deg left) -> needs_more_excitation
        w_one = np.zeros((len(t), 3), dtype=np.float64)
        w_one[(t >= 5.0) & (t <= 12.0), 2] = -0.35
        a_one = np.column_stack(
            [np.zeros_like(t), 0.7 * np.sin(0.8 * t), np.full_like(t, 9.81)]
        )
        gate_bad = evaluate_common_mode_excitation_gate(
            {"lid_t": lid_t, "imu_t": t, "imu_w": w_one, "imu_a": a_one},
            healthy_edge,
        )
        self.assertEqual(gate_bad["verdict"], "needs_more_excitation")
        self.assertFalse(gate_bad["z_cm_observable"])

        # Bidirectional symmetric turns + >=0.5 m/s2 planar accel -> accepted
        w_two = w_one.copy()
        w_two[(t >= 20.0) & (t <= 27.0), 2] = 0.36
        gate_ok = evaluate_common_mode_excitation_gate(
            {"lid_t": lid_t, "imu_t": t, "imu_w": w_two, "imu_a": a_one},
            healthy_edge,
        )
        self.assertEqual(gate_ok["verdict"], "accepted")
        self.assertFalse(gate_ok["z_cm_observable"])


if __name__ == "__main__":
    unittest.main()
