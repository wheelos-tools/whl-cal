from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from lidar2imu.evaluation.dual_sensor_diagnostics import (
    FastCanonicalScanAccess,
    evaluate_dual_lidar2imu_diagnostics,
    write_colored_dual_pcd,
)


def _write_npy_into_zip(
    zf: zipfile.ZipFile, name: str, arr: np.ndarray, compress_type: int
) -> None:
    import io

    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
    zf.writestr(
        zipfile.ZipInfo(filename=f"{name}.npy"),
        buf.getvalue(),
        compress_type=compress_type,
    )


def _write_gril_result_file(
    path: Path,
    R_LI: np.ndarray,
    T_LI: np.ndarray,
    tau_s: float = -0.020,
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


def _build_synthetic_dual_capture(
    base_dir: Path,
    *,
    zip_compress: int = zipfile.ZIP_STORED,
    perturb_b_lever_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(7)
    R_LI_a = Rot.from_euler("xyz", [0.0, 0.0, 135.0], degrees=True).as_matrix()
    T_LI_a = np.array([-0.4, 2.0, 0.5], dtype=np.float64)
    R_LI_b = Rot.from_euler("xyz", [0.0, 0.0, -45.0], degrees=True).as_matrix()
    T_LI_b = np.array([1.2, -3.2, 0.5], dtype=np.float64)
    tau = -0.020

    # World points: flat ground z = -0.5 + two orthogonal walls x = -8.0 & y = +9.0
    gx, gy = np.meshgrid(np.linspace(-12.0, 12.0, 26), np.linspace(-10.0, 10.0, 22))
    ground_pts = np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, -0.5, dtype=np.float64)]
    )
    wy, wz = np.meshgrid(np.linspace(-9.0, 9.0, 32), np.linspace(0.1, 2.4, 10))
    wall_x = np.column_stack([np.full(wy.size, -8.0), wy.ravel(), wz.ravel()])
    wx, wz2 = np.meshgrid(np.linspace(-7.0, 9.0, 32), np.linspace(0.1, 2.4, 10))
    wall_y = np.column_stack([wx.ravel(), np.full(wx.size, 9.0), wz2.ravel()])
    world_pts = np.vstack([ground_pts, wall_x, wall_y])

    # Trajectory over [0, 8.0 s]: [0, 3.0 s] parked, [3.5, 6.5 s] smooth 55 deg yaw turn
    imu_t = np.linspace(0.0, 8.2, 821, dtype=np.float64)
    psi_max = math.radians(55.0)

    def _pose_at(t_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.clip((t_arr - 3.5) / 3.0, 0.0, 1.0)
        psi = psi_max * (3.0 * u**2 - 2.0 * u**3)
        wz = np.where(
            (t_arr >= 3.5) & (t_arr <= 6.5),
            psi_max * (6.0 * u * (1.0 - u)) / 3.0,
            0.0,
        )
        # Mild circular arc velocity in world frame
        spd = np.where((t_arr >= 3.5) & (t_arr <= 6.5), 2.0 * np.sin(math.pi * u), 0.0)
        vx_w = -spd * np.sin(psi)
        vy_w = spd * np.cos(psi)
        px = np.cumsum(vx_w) * (t_arr[1] - t_arr[0]) if len(t_arr) > 1 else np.zeros(1)
        py = np.cumsum(vy_w) * (t_arr[1] - t_arr[0]) if len(t_arr) > 1 else np.zeros(1)
        pos = np.column_stack([px, py, np.zeros_like(px)])
        vel_w = np.column_stack([vx_w, vy_w, np.zeros_like(px)])
        return psi, wz, pos, vel_w

    psi_i, wz_i, pos_i, vel_i = _pose_at(imu_t)
    dt_i = float(imu_t[1] - imu_t[0])
    acc_w = np.gradient(vel_i, dt_i, axis=0)
    imu_w = np.column_stack([np.zeros_like(wz_i), np.zeros_like(wz_i), wz_i])
    imu_f = np.zeros_like(acc_w)
    g_w = np.array([0.0, 0.0, -9.81])
    for k in range(len(imu_t)):
        R_wi = Rot.from_euler("z", float(psi_i[k])).as_matrix()
        imu_f[k] = R_wi.T @ (acc_w[k] - g_w)

    scan_times_phys = np.arange(0.2, 7.8, 0.10, dtype=np.float64)
    scan_times_lid = scan_times_phys - tau

    def _write_sensor_bundle(
        name: str,
        R_LI: np.ndarray,
        T_LI: np.ndarray,
        reported_T_LI: np.ndarray,
    ) -> tuple[Path, Path]:
        ds_dir = base_dir / name / "dataset"
        run_dir = base_dir / name / "run"
        ds_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        xyz_chunks = []
        pt_t_chunks = []
        offsets = [0]
        for t_s in scan_times_phys:
            idx = int(np.argmin(np.abs(imu_t - t_s)))
            R_wi = Rot.from_euler("z", float(psi_i[idx])).as_matrix()
            p_wi = pos_i[idx]
            pts_i = (world_pts - p_wi) @ R_wi
            pts_l = (
                (pts_i - T_LI) @ R_LI + rng.normal(0.0, 0.005, size=pts_i.shape)
            ).astype(np.float32)
            pt_dt = np.zeros(len(pts_l), dtype=np.float32)
            xyz_chunks.append(pts_l)
            pt_t_chunks.append(pt_dt)
            offsets.append(offsets[-1] + len(pts_l))

        xyz_all = np.vstack(xyz_chunks)
        pt_t_all = np.concatenate(pt_t_chunks)
        scan_ns = np.round(scan_times_lid * 1e9).astype(np.int64)
        scan_offsets = np.asarray(offsets, dtype=np.int64)

        with zipfile.ZipFile(ds_dir / "lidar.npz", "w") as zf:
            _write_npy_into_zip(zf, "xyz", xyz_all, zip_compress)
            _write_npy_into_zip(zf, "point_time_s", pt_t_all, zip_compress)
            _write_npy_into_zip(zf, "scan_timestamps_ns", scan_ns, zip_compress)
            _write_npy_into_zip(zf, "scan_offsets", scan_offsets, zip_compress)

        np.savez(
            ds_dir / "imu.npz",
            timestamps_ns=np.round(imu_t * 1e9).astype(np.int64),
            angular_velocity=imu_w,
            linear_acceleration=imu_f,
        )
        manifest = {
            "schema_version": 1,
            "frames": {"imu": "imu", "lidar": name},
            "arrays": {
                "lidar": {"path": "lidar.npz"},
                "imu": {"path": "imu.npz"},
            },
        }
        (ds_dir / "dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

        # Write batch trace on 30 Hz grid over [2.5, 7.5 s]
        lid_grid_phys = np.arange(2.5, 7.5, 1.0 / 30.0, dtype=np.float64)
        lid_grid_raw = lid_grid_phys - tau
        T_IL = -(R_LI.T @ T_LI)
        lines = [
            "GRIL_batch_trace_v1",
            "orig_odom_freq 10.0",
            "cut_frame_num 3",
            "timediff_imu_wrt_lidar 0.0",
            "move_start_time 3.5",
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
        R_WL0 = (Rot.from_euler("z", float(psi_i[0])).as_matrix() @ R_LI).T
        for k_l, t_p in enumerate(lid_grid_phys):
            psi_k = float(np.interp(t_p, imu_t, psi_i))
            wz_k = float(np.interp(t_p, imu_t, wz_i))
            vx_k = float(np.interp(t_p, imu_t, vel_i[:, 0]))
            vy_k = float(np.interp(t_p, imu_t, vel_i[:, 1]))
            R_wi = Rot.from_euler("z", psi_k).as_matrix()
            R_wl = R_wi @ R_LI
            R_rel = R_WL0 @ R_wl
            r_flat = " ".join(f"{v:.8f}" for v in R_rel.reshape(-1))
            w_i_vec = np.array([0.0, 0.0, wz_k])
            w_l_vec = R_LI.T @ w_i_vec
            v_i_local = R_wi.T @ np.array([vx_k, vy_k, 0.0])
            v_l_local = R_LI.T @ v_i_local - np.cross(w_l_vec, T_IL)
            v_in_l0 = R_rel @ v_l_local
            lines.append(
                f"lidar {lid_grid_raw[k_l]:.9f} {r_flat} 0.0 0.0 0.0 "
                f"{w_l_vec[0]:.8f} {w_l_vec[1]:.8f} {w_l_vec[2]:.8f} "
                f"{v_in_l0[0]:.8f} {v_in_l0[1]:.8f} {v_in_l0[2]:.8f}"
            )
        (run_dir / "GRIL_batch_trace_v1.txt").write_text("\n".join(lines) + "\n")
        _write_gril_result_file(
            run_dir / "GRIL_Calib_result.txt", R_LI, reported_T_LI, tau_s=tau
        )
        return ds_dir, run_dir

    ds_a, run_a = _write_sensor_bundle("left_front", R_LI_a, T_LI_a, T_LI_a)
    ds_b, run_b = _write_sensor_bundle(
        "right_back",
        R_LI_b,
        T_LI_b,
        T_LI_b + np.asarray(perturb_b_lever_m, dtype=np.float64),
    )
    return ds_a, run_a, ds_b, run_b


class TestDualSensorDiagnostics(unittest.TestCase):
    def test_fast_scan_access_stored_and_deflated_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_stored, _, _, _ = _build_synthetic_dual_capture(
                root / "stored", zip_compress=zipfile.ZIP_STORED
            )
            ds_deflated, _, _, _ = _build_synthetic_dual_capture(
                root / "deflated", zip_compress=zipfile.ZIP_DEFLATED
            )
            acc_stored = FastCanonicalScanAccess(ds_stored)
            acc_deflated = FastCanonicalScanAccess(ds_deflated)
            self.assertIsNotNone(acc_stored._xyz_offset)
            self.assertIsNone(acc_deflated._xyz_offset)
            xyz_s, t_s = acc_stored.read_scan(5)
            xyz_d, t_d = acc_deflated.read_scan(5)
            np.testing.assert_allclose(xyz_s, xyz_d, atol=1e-6)
            np.testing.assert_allclose(t_s, t_d, atol=1e-7)

    def test_write_colored_dual_pcd_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pcd_path = Path(tmp) / "dual.pcd"
            pts_a = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 1.0]])
            pts_b = np.array([[-1.0, -2.0, 0.5]])
            write_colored_dual_pcd(pts_a, pts_b, pcd_path)
            raw = pcd_path.read_bytes()
            self.assertIn(b"FIELDS x y z rgb\n", raw)
            self.assertIn(b"POINTS 3\n", raw)
            try:
                import open3d as o3d

                pcd = o3d.io.read_point_cloud(str(pcd_path))
                self.assertEqual(len(pcd.points), 3)
                self.assertTrue(pcd.has_colors())
            except ImportError:
                pass

    def test_dual_diagnostics_accepts_consistent_and_flags_lever_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_a, run_a, ds_b, run_b = _build_synthetic_dual_capture(root / "clean")
            out_clean = root / "out_clean"
            report_clean = evaluate_dual_lidar2imu_diagnostics(
                dataset_a=ds_a,
                run_dir_a=run_a,
                dataset_b=ds_b,
                run_dir_b=run_b,
                output_dir=out_clean,
            )
            self.assertEqual(report_clean["verdict"], "accepted")
            self.assertTrue(report_clean["checks"]["static_overlay_consistent"])
            self.assertTrue(report_clean["checks"]["turn_submap_sharp"])
            self.assertGreater(
                report_clean["pillars"]["turn_imu_submap"][
                    "common_mode_blur_inflation_ratio"
                ],
                1.15,
            )
            self.assertTrue((out_clean / "dual_diagnostics.yaml").is_file())
            self.assertEqual(len(list((out_clean / "extrinsics").glob("*.yaml"))), 3)
            self.assertGreaterEqual(
                len(list((out_clean / "pointclouds").glob("*.pcd"))), 3
            )

            # Inject a 0.60 m lever-arm bias on Sensor B -> must trigger review_only!
            ds_a2, run_a2, ds_b2, run_b2 = _build_synthetic_dual_capture(
                root / "biased", perturb_b_lever_m=(0.45, -0.45, 0.0)
            )
            out_bias = root / "out_bias"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lidar2imu.cli",
                    "evaluate-dual-diagnostics",
                    "--dataset-a",
                    str(ds_a2),
                    "--run-dir-a",
                    str(run_a2),
                    "--dataset-b",
                    str(ds_b2),
                    "--run-dir-b",
                    str(run_b2),
                    "--output-dir",
                    str(out_bias),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("dual_diagnostics.yaml", proc.stdout)
            report_bias = yaml.safe_load(
                (out_bias / "dual_diagnostics.yaml").read_text()
            )
            self.assertEqual(report_bias["verdict"], "review_only")
            self.assertFalse(report_bias["checks"]["static_overlay_consistent"])
            self.assertGreater(
                report_bias["pillars"]["static_dual_overlay"]["max_abs_wall_gap_cm"],
                30.0,
            )

    def test_evaluate_trajectories_recovers_ant2outposoffset_lever_arm(self) -> None:
        import importlib.util

        script_dir = (
            Path(__file__).resolve().parents[1]
            / ".agents/skills/gril-calib-validation/scripts"
        )
        sys.path.insert(0, str(script_dir))
        try:
            spec = importlib.util.spec_from_file_location(
                "evaluate_trajectories", script_dir / "evaluate_trajectories.py"
            )
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)

        # S-turn trajectory: rear-axle outpos [-0.69, +2.81] + t_IL [-0.37, +2.52]
        t_ol_true = np.array([-1.06, 5.33], dtype=np.float64)
        theta_true = 0.42
        R_world = np.array(
            [
                [np.cos(theta_true), -np.sin(theta_true)],
                [np.sin(theta_true), np.cos(theta_true)],
            ]
        )
        t_s = np.linspace(0.0, 20.0, 201)
        wz = np.where(t_s < 10.0, 0.25, -0.25)
        yaws = np.cumsum(wz) * float(t_s[1] - t_s[0])
        R_wo_2d = np.array(
            [[[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]] for y in yaws]
        )
        vel_world = np.column_stack([-3.0 * np.sin(yaws), 3.0 * np.cos(yaws)])
        p_outpos = np.cumsum(vel_world, axis=0) * float(t_s[1] - t_s[0])
        p_lidar_world = p_outpos + np.einsum("nij,j->ni", R_wo_2d, t_ol_true)
        p_lidar_local = (R_world.T @ (p_lidar_world - np.array([3.5, -2.0])).T).T

        R_est, t_est, t_ol_est = mod._solve_se2_with_lever_arm(
            p_lidar_local, p_outpos, R_wo_2d
        )
        aligned = (
            (R_est @ p_lidar_local.T).T
            + t_est
            - np.einsum("nij,j->ni", R_wo_2d, t_ol_est)
        )
        np.testing.assert_allclose(t_ol_est, t_ol_true, atol=1e-3)
        self.assertLess(float(np.max(np.linalg.norm(aligned - p_outpos, axis=1))), 1e-3)


if __name__ == "__main__":
    unittest.main()
