"""Dual-sensor LiDAR-to-IMU visual and quantitative cross-diagnostics.

Implements five complementary post-run checks for a two-LiDAR + single-IMU rig
without modifying any calibration solver:
1. Structured static frame colored dual-PCD export + 3 orthogonal 2D slices.
2. High-yaw-rate turn pure-IMU deskew + cross-time submap sharpness & common-mode
   gauge-freedom sensitivity check.
3. 3-edge cycle consistency (`2 x lidar2imu` induced L2L vs direct GICP `L2L`).
4. Multi-heading / opposite-heading ground normal vs IMU gravity alignment.
5. Three-curves-in-one rigid-body angular velocity & lever-arm acceleration check.
"""

from __future__ import annotations

import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.lib import format as npformat
from scipy.spatial.transform import Rotation as Rot

from lidar2imu.dataset_io import MANIFEST_NAME
from lidar2imu.evaluation.quality_gates import (
    _interp_vec3,
    _project_so3,
    _rotation_error_deg,
    _zero_phase_smooth,
    parse_batch_trace_arrays,
    parse_detailed_gril_result,
)
from lidar2lidar.extrinsic_io import (
    extrinsics_filename,
    load_extrinsics_file,
    save_extrinsics_yaml,
)

COLOR_SENSOR_A = np.array([0.92, 0.26, 0.21], dtype=np.float64)  # Coral red
COLOR_SENSOR_B = np.array([0.12, 0.62, 0.88], dtype=np.float64)  # Cyan blue


@dataclass(frozen=True)
class DualDiagnosticThresholds:
    max_static_ground_z_gap_cm: float = 5.0
    max_static_wall_gap_cm: float = 8.0
    max_turn_wall_thickness_cm: float = 8.0
    max_turn_wall_gap_cm: float = 10.0
    min_common_bias_blur_ratio: float = 1.15
    max_l2l_loop_rot_deg: float = 1.5
    max_l2l_loop_trans_m: float = 0.20
    max_inter_lidar_ground_tilt_deg: float = 1.0
    max_imu_gravity_tilt_bias_deg: float = 1.5
    max_gyro_cross_rmse_rad_s: float = 0.03


class FastCanonicalScanAccess:
    """Random-access reader for canonical `lidar.npz` + `imu.npz` bundles.

    Uses direct byte-offset seeking when `lidar.npz` is `ZIP_STORED` so reading
    individual scans from multi-GB NAS archives takes milliseconds instead of
    loading the entire point cloud array into memory.
    """

    def __init__(self, dataset_path: Path) -> None:
        manifest_path = (
            Path(dataset_path)
            if Path(dataset_path).is_file()
            else Path(dataset_path) / MANIFEST_NAME
        )
        self.root = manifest_path.parent
        self.manifest = yaml.safe_load(manifest_path.read_text())
        self.lidar_frame = str(self.manifest["frames"]["lidar"])
        self.imu_frame = str(self.manifest["frames"]["imu"])
        self.lidar_npz = self.root / self.manifest["arrays"]["lidar"]["path"]
        self.imu_npz = self.root / self.manifest["arrays"]["imu"]["path"]

        with np.load(self.imu_npz, allow_pickle=False) as imu_data:
            self.imu_timestamps_ns = np.asarray(
                imu_data["timestamps_ns"], dtype=np.int64
            )
            self.imu_w = np.asarray(imu_data["angular_velocity"], dtype=np.float64)
            self.imu_a = np.asarray(imu_data["linear_acceleration"], dtype=np.float64)
            imu_disp = (
                np.asarray(imu_data["dispatch_timestamps_ns"], dtype=np.int64)
                if "dispatch_timestamps_ns" in imu_data
                else None
            )

        self._xyz_offset: int | None = None
        self._time_offset: int | None = None
        self._intensity_offset: int | None = None
        self._intensity_dtype: np.dtype = np.dtype(np.float32)
        self._ring_offset: int | None = None
        self._ring_dtype: np.dtype = np.dtype(np.uint16)
        self._cached_xyz: np.ndarray | None = None
        self._cached_time: np.ndarray | None = None
        self._cached_intensity: np.ndarray | None = None
        self._cached_ring: np.ndarray | None = None

        with zipfile.ZipFile(self.lidar_npz, "r") as zf:
            with zf.open("scan_timestamps_ns.npy") as f_ts:
                self.scan_timestamps_ns = np.asarray(np.load(f_ts), dtype=np.int64)
            with zf.open("scan_offsets.npy") as f_off:
                self.scan_offsets = np.asarray(np.load(f_off), dtype=np.int64)
            lidar_disp = None
            lidar_meas = None
            names = set(zf.namelist())
            if "dispatch_timestamps_ns.npy" in names:
                with zf.open("dispatch_timestamps_ns.npy") as f_d:
                    lidar_disp = np.asarray(np.load(f_d), dtype=np.int64)
            if "measurement_timestamps_ns.npy" in names:
                with zf.open("measurement_timestamps_ns.npy") as f_m:
                    lidar_meas = np.asarray(np.load(f_m), dtype=np.int64)

            xyz_info = zf.getinfo("xyz.npy")
            time_info = zf.getinfo("point_time_s.npy")
            if (
                xyz_info.compress_type == zipfile.ZIP_STORED
                and time_info.compress_type == zipfile.ZIP_STORED
            ):
                self._xyz_offset, _ = self._stored_member_data_offset(zf, "xyz.npy")
                self._time_offset, _ = self._stored_member_data_offset(
                    zf, "point_time_s.npy"
                )
            if (
                "intensity.npy" in names
                and zf.getinfo("intensity.npy").compress_type == zipfile.ZIP_STORED
            ):
                self._intensity_offset, self._intensity_dtype = (
                    self._stored_member_data_offset(zf, "intensity.npy")
                )
            if (
                "ring.npy" in names
                and zf.getinfo("ring.npy").compress_type == zipfile.ZIP_STORED
            ):
                self._ring_offset, self._ring_dtype = self._stored_member_data_offset(
                    zf, "ring.npy"
                )

        self.hard_offset_s = self._compute_hard_offset_s(
            lidar_meas=lidar_meas,
            lidar_disp=lidar_disp,
            imu_meas=self.imu_timestamps_ns,
            imu_disp=imu_disp,
        )
        self.scan_t_s = self.scan_timestamps_ns.astype(np.float64) * 1e-9
        self.imu_t_s = (
            self.imu_timestamps_ns.astype(np.float64) * 1e-9 - self.hard_offset_s
        )

    @staticmethod
    def _stored_member_data_offset(
        zf: zipfile.ZipFile, member_name: str
    ) -> tuple[int, np.dtype]:
        info = zf.getinfo(member_name)
        with zf.open(member_name) as member:
            version = npformat.read_magic(member)
            _, _, dtype = npformat._read_array_header(member, version)
            npy_hdr_len = int(member.tell())
        npz_path = Path(zf.filename)  # type: ignore[arg-type]
        with npz_path.open("rb") as raw:
            raw.seek(info.header_offset)
            local_hdr = raw.read(30)
            unpacked = struct.unpack("<4sHHHHHIIIHH", local_hdr)
            if unpacked[0] != b"PK\x03\x04":
                raise ValueError(f"Invalid ZIP local header in {npz_path}")
            fname_len, extra_len = unpacked[9], unpacked[10]
        return int(info.header_offset + 30 + fname_len + extra_len + npy_hdr_len), dtype

    @staticmethod
    def _compute_hard_offset_s(
        *,
        lidar_meas: np.ndarray | None,
        lidar_disp: np.ndarray | None,
        imu_meas: np.ndarray,
        imu_disp: np.ndarray | None,
    ) -> float:
        if (
            lidar_meas is None
            or lidar_disp is None
            or imu_disp is None
            or len(lidar_meas) == 0
            or len(imu_meas) == 0
        ):
            return 0.0
        valid_l = (lidar_meas > 0) & (lidar_disp > 0)
        valid_i = (imu_meas > 0) & (imu_disp > 0)
        if not np.any(valid_l) or not np.any(valid_i):
            return 0.0
        dl_s = float(
            np.median((lidar_disp[valid_l] - lidar_meas[valid_l]).astype(np.float64))
            * 1e-9
        )
        di_s = float(
            np.median((imu_disp[valid_i] - imu_meas[valid_i]).astype(np.float64)) * 1e-9
        )
        raw_h = round(dl_s - di_s)
        return float(raw_h) if abs(raw_h) >= 1 else 0.0

    def _ensure_fallback_loaded(self) -> None:
        if self._cached_xyz is None or self._cached_time is None:
            with np.load(self.lidar_npz, allow_pickle=False) as values:
                self._cached_xyz = np.asarray(values["xyz"], dtype=np.float32)
                self._cached_time = np.asarray(values["point_time_s"], dtype=np.float32)
                self._cached_intensity = (
                    np.asarray(values["intensity"], dtype=np.float32)
                    if "intensity" in values
                    else None
                )
                self._cached_ring = (
                    np.asarray(values["ring"], dtype=np.int32)
                    if "ring" in values
                    else None
                )

    def read_scan(self, scan_idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.scan_offsets[scan_idx])
        end = int(self.scan_offsets[scan_idx + 1])
        count = end - start
        if count <= 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        if self._xyz_offset is not None and self._time_offset is not None:
            with self.lidar_npz.open("rb") as raw:
                raw.seek(self._xyz_offset + start * 12)
                xyz = (
                    np.frombuffer(raw.read(count * 12), dtype=np.float32)
                    .reshape(-1, 3)
                    .astype(np.float64)
                )
                raw.seek(self._time_offset + start * 4)
                pt_t = np.frombuffer(raw.read(count * 4), dtype=np.float32).astype(
                    np.float64
                )
            return xyz, pt_t
        self._ensure_fallback_loaded()
        assert self._cached_xyz is not None and self._cached_time is not None
        return (
            self._cached_xyz[start:end].astype(np.float64),
            self._cached_time[start:end].astype(np.float64),
        )

    def read_scan_with_attributes(
        self, scan_idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        xyz, pt_t = self.read_scan(scan_idx)
        start = int(self.scan_offsets[scan_idx])
        end = int(self.scan_offsets[scan_idx + 1])
        count = end - start
        if count <= 0:
            return xyz, pt_t, None, None
        intensity: np.ndarray | None = None
        ring: np.ndarray | None = None
        if self._intensity_offset is not None:
            item_sz = int(self._intensity_dtype.itemsize)
            with self.lidar_npz.open("rb") as raw:
                raw.seek(self._intensity_offset + start * item_sz)
                intensity = np.frombuffer(
                    raw.read(count * item_sz), dtype=self._intensity_dtype
                ).astype(np.float64)
        else:
            self._ensure_fallback_loaded()
            if self._cached_intensity is not None:
                intensity = self._cached_intensity[start:end].astype(np.float64)

        if self._ring_offset is not None:
            item_sz = int(self._ring_dtype.itemsize)
            with self.lidar_npz.open("rb") as raw:
                raw.seek(self._ring_offset + start * item_sz)
                ring = np.frombuffer(
                    raw.read(count * item_sz), dtype=self._ring_dtype
                ).astype(np.int32)
        else:
            self._ensure_fallback_loaded()
            if self._cached_ring is not None:
                ring = self._cached_ring[start:end].astype(np.int32)
        return xyz, pt_t, intensity, ring


def _extrinsic_matrix_LI(result: dict[str, Any]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _project_so3(np.asarray(result["rotation_LI_matrix"], dtype=float))
    mat[:3, 3] = np.asarray(result["translation_LI_m"], dtype=float)
    return mat


def _filter_range(
    xyz_imu: np.ndarray,
    pt_t: np.ndarray | None = None,
    *,
    min_xy_m: float = 2.4,
    max_xy_m: float = 42.0,
    min_z_m: float = -2.5,
    max_z_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    r_xy = np.linalg.norm(xyz_imu[:, :2], axis=1)
    mask = (
        (r_xy >= min_xy_m)
        & (r_xy <= max_xy_m)
        & (xyz_imu[:, 2] >= min_z_m)
        & (xyz_imu[:, 2] <= max_z_m)
    )
    return xyz_imu[mask], (pt_t[mask] if pt_t is not None else None)


def write_colored_dual_pcd(
    points_a: np.ndarray,
    points_b: np.ndarray,
    output_path: Path,
    *,
    color_a: np.ndarray = COLOR_SENSOR_A,
    color_b: np.ndarray = COLOR_SENSOR_B,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.vstack([points_a, points_b]).astype(np.float64)
    cols = np.vstack(
        [
            np.tile(color_a, (len(points_a), 1)),
            np.tile(color_b, (len(points_b), 1)),
        ]
    ).astype(np.float64)
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = o3d.utility.Vector3dVector(cols)
        if not o3d.io.write_point_cloud(str(output_path), cloud, write_ascii=False):
            raise RuntimeError(f"Failed to write PCD via Open3D: {output_path}")
    except ImportError:
        rgb_u8 = np.clip(np.round(cols * 255.0), 0, 255).astype(np.uint32)
        packed_u32 = (rgb_u8[:, 0] << 16) | (rgb_u8[:, 1] << 8) | rgb_u8[:, 2]
        packed_f32 = packed_u32.view(np.float32)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {len(pts)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(pts)}\n"
            "DATA binary\n"
        ).encode("ascii")
        rec = np.empty(
            len(pts),
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")],
        )
        rec["x"] = pts[:, 0].astype(np.float32)
        rec["y"] = pts[:, 1].astype(np.float32)
        rec["z"] = pts[:, 2].astype(np.float32)
        rec["rgb"] = packed_f32
        with output_path.open("wb") as handle:
            handle.write(header)
            handle.write(rec.tobytes())
    return output_path


def _fit_ground_plane_ransac(
    pts_imu: np.ndarray,
    *,
    z_hint: float,
    inlier_thresh_m: float = 0.06,
    rng_seed: int = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    r_xy = np.linalg.norm(pts_imu[:, :2], axis=1)
    cand_mask = (
        (r_xy >= 3.2)
        & (r_xy <= 18.0)
        & (pts_imu[:, 2] >= z_hint - 0.35)
        & (pts_imu[:, 2] <= z_hint + 0.35)
    )
    cand = pts_imu[cand_mask]
    if len(cand) < 30:
        cand = pts_imu
    if len(cand) < 6:
        return np.array([0.0, 0.0, 1.0]), float(z_hint), cand

    rng = np.random.default_rng(rng_seed)
    best_inliers: np.ndarray | None = None
    n_pts = len(cand)
    for _ in range(80):
        idx = rng.choice(n_pts, size=3, replace=False)
        v1 = cand[idx[1]] - cand[idx[0]]
        v2 = cand[idx[2]] - cand[idx[0]]
        normal = np.cross(v1, v2)
        norm_val = float(np.linalg.norm(normal))
        if norm_val < 1e-8:
            continue
        normal /= norm_val
        if normal[2] < 0.0:
            normal = -normal
        if normal[2] < 0.90:
            continue
        dist = np.abs((cand - cand[idx[0]]) @ normal)
        inliers = dist < inlier_thresh_m
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers

    inlier_pts = (
        cand[best_inliers]
        if best_inliers is not None and int(best_inliers.sum()) >= 10
        else cand
    )
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0.0:
        normal = -normal
    offset_d = -float(normal @ centroid)
    return normal, offset_d, inlier_pts


def _extract_orthogonal_wall_slices(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    z_ground: float,
) -> list[dict[str, Any]]:
    vert_a = pts_a[(pts_a[:, 2] > z_ground + 0.45) & (pts_a[:, 2] < z_ground + 3.2)]
    vert_b = pts_b[(pts_b[:, 2] > z_ground + 0.45) & (pts_b[:, 2] < z_ground + 3.2)]
    if len(vert_a) < 20 or len(vert_b) < 20:
        return []

    cell_m = 1.5
    keys_a = np.floor(vert_a[:, :2] / cell_m).astype(int)
    unique_cells = sorted(set(map(tuple, keys_a)))

    candidates: list[dict[str, Any]] = []
    for cx, cy in unique_cells:
        center_xy = np.array([(cx + 0.5) * cell_m, (cy + 0.5) * cell_m])
        win_a = vert_a[
            (np.abs(vert_a[:, 0] - center_xy[0]) <= 1.4)
            & (np.abs(vert_a[:, 1] - center_xy[1]) <= 1.4)
        ]
        if len(win_a) < 16:
            continue
        mean_xy = win_a[:, :2].mean(axis=0)
        cov_a = np.cov((win_a[:, :2] - mean_xy).T)
        eigvals_a, eigvecs_a = np.linalg.eigh(cov_a)
        if eigvals_a[1] < 2e-4 or (eigvals_a[0] / eigvals_a[1]) > 0.08:
            continue
        n_xy = eigvecs_a[:, 0]
        if float(n_xy @ mean_xy) < 0.0:
            n_xy = -n_xy
        t_xy = np.array([-n_xy[1], n_xy[0]])

        na_all = (win_a[:, :2] - mean_xy) @ n_xy
        ua_all = (win_a[:, :2] - mean_xy) @ t_xy
        in_band_a = (np.abs(na_all) <= 0.22) & (np.abs(ua_all) <= 1.35)
        if int(in_band_a.sum()) < 14:
            continue

        # Search Sensor B along the wall normal up to ±1.8 m so even meter-scale
        # lever-arm shrinkage on longitudinal/lateral walls is captured honestly.
        rel_b = vert_b[:, :2] - mean_xy
        nb_proj = rel_b @ n_xy
        ub_proj = rel_b @ t_xy
        cand_b_mask = (np.abs(ub_proj) <= 1.35) & (np.abs(nb_proj) <= 1.80)
        if int(cand_b_mask.sum()) < 14:
            continue
        nb_cand = nb_proj[cand_b_mask]
        ub_cand = ub_proj[cand_b_mask]
        med_nb = float(np.median(nb_cand))
        tight_b = np.abs(nb_cand - med_nb) <= 0.22
        if int(tight_b.sum()) < 12:
            continue
        pts_b_wall = vert_b[cand_b_mask][tight_b, :2]
        cov_b = np.cov((pts_b_wall - pts_b_wall.mean(axis=0)).T)
        eigvals_b, eigvecs_b = np.linalg.eigh(cov_b)
        if eigvals_b[1] < 2e-4 or (eigvals_b[0] / eigvals_b[1]) > 0.10:
            continue
        n_b_xy = eigvecs_b[:, 0]
        if abs(float(n_xy @ n_b_xy)) < 0.92:
            continue

        na_in = na_all[in_band_a]
        ua_in = ua_all[in_band_a]
        nb_in = nb_cand[tight_b]
        ub_in = ub_cand[tight_b]
        # Re-center plot coordinates around midpoint of the two walls
        mid_n = 0.5 * (float(np.median(na_in)) + float(np.median(nb_in)))
        na_centered = na_in - mid_n
        nb_centered = nb_in - mid_n
        gap_cm = float((np.median(na_in) - np.median(nb_in)) * 100.0)
        single_std_cm = float(0.5 * (np.std(na_in) + np.std(nb_in)) * 100.0)
        if min(len(na_in), len(nb_in)) < 20 or single_std_cm > 6.0:
            continue
        combined_std_cm = float(
            np.std(np.concatenate([na_centered, nb_centered])) * 100.0
        )
        score = float(min(len(na_in), len(nb_in))) / (
            1.0 + eigvals_a[0] / eigvals_a[1] + eigvals_b[0] / eigvals_b[1]
        )
        candidates.append(
            {
                "score": score,
                "center_xy_m": [float(mean_xy[0]), float(mean_xy[1])],
                "normal_xy": [float(n_xy[0]), float(n_xy[1])],
                "normal_azimuth_deg": float(math.degrees(math.atan2(n_xy[1], n_xy[0]))),
                "signed_normal_gap_cm": gap_cm,
                "abs_normal_gap_cm": abs(gap_cm),
                "single_wall_std_cm": single_std_cm,
                "combined_plane_std_cm": combined_std_cm,
                "count_a": int(len(na_in)),
                "count_b": int(len(nb_in)),
                "u_a_m": ua_in,
                "n_a_cm": na_centered * 100.0,
                "u_b_m": ub_in,
                "n_b_cm": nb_centered * 100.0,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        return []
    first = candidates[0]
    chosen = [first]
    n1 = np.asarray(first["normal_xy"], dtype=np.float64)
    c1 = np.asarray(first["center_xy_m"], dtype=np.float64)
    for item in candidates[1:]:
        n2 = np.asarray(item["normal_xy"], dtype=np.float64)
        c2 = np.asarray(item["center_xy_m"], dtype=np.float64)
        if np.linalg.norm(c2 - c1) < 2.5:
            continue
        if abs(float(n1 @ n2)) < 0.65:
            chosen.append(item)
            break
    return chosen


def _find_stationary_candidates(
    access_a: FastCanonicalScanAccess,
    batch_a: dict[str, Any],
) -> list[float]:
    candidates: list[float] = []
    t0_scan = float(access_a.scan_t_s[0])
    lid_t = batch_a["lid_t"]
    if len(lid_t) > 0 and (lid_t[0] - t0_scan) > 4.0:
        pre_mid = t0_scan + 0.45 * (float(lid_t[0]) - t0_scan)
        candidates.append(pre_mid)

    if len(lid_t) > 10:
        speed = np.linalg.norm(batch_a["lid_v"], axis=1)
        omega = np.linalg.norm(batch_a["lid_w"], axis=1)
        valid_idx = np.where((speed <= 0.12) & (omega <= 0.045))[0]
        order = valid_idx[np.argsort(speed[valid_idx] + 0.5 * omega[valid_idx])]
        for idx in order:
            t_val = float(lid_t[int(idx)])
            if all(abs(t_val - prev) > 12.0 for prev in candidates):
                candidates.append(t_val)
            if len(candidates) >= 8:
                break

    if not candidates:
        imu_w_norm = np.linalg.norm(access_a.imu_w, axis=1)
        quiet_idx = int(np.argmin(imu_w_norm))
        candidates.append(float(access_a.imu_t_s[quiet_idx]))
    return candidates


def evaluate_static_dual_overlay(
    access_a: FastCanonicalScanAccess,
    access_b: FastCanonicalScanAccess,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    batch_a: dict[str, Any],
    output_dir: Path,
    *,
    static_time_s: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    T_I_A = _extrinsic_matrix_LI(result_a)
    T_I_B = _extrinsic_matrix_LI(result_b)
    tau_a = float(result_a["time_offset_s"])
    tau_b = float(result_b["time_offset_s"])

    if static_time_s is not None:
        cand_times = [float(access_a.scan_t_s[0]) + float(static_time_s)]
    else:
        cand_times = _find_stationary_candidates(access_a, batch_a)

    best: dict[str, Any] | None = None
    for t_target in cand_times:
        ia = int(np.argmin(np.abs((access_a.scan_t_s + tau_a) - t_target)))
        t_phys = float(access_a.scan_t_s[ia] + tau_a)
        ib = int(np.argmin(np.abs((access_b.scan_t_s + tau_b) - t_phys)))
        sync_dt_ms = float(((access_b.scan_t_s[ib] + tau_b) - t_phys) * 1e3)
        if abs(sync_dt_ms) > 60.0:
            continue
        raw_a, _ = access_a.read_scan(ia)
        raw_b, _ = access_b.read_scan(ib)
        if len(raw_a) < 50 or len(raw_b) < 50:
            continue
        pts_a, _ = _filter_range(raw_a @ T_I_A[:3, :3].T + T_I_A[:3, 3])
        pts_b, _ = _filter_range(raw_b @ T_I_B[:3, :3].T + T_I_B[:3, 3])
        if len(pts_a) < 30 or len(pts_b) < 30:
            continue
        z_hint = float(np.percentile(np.concatenate([pts_a[:, 2], pts_b[:, 2]]), 12))
        va = set(
            map(
                tuple,
                np.floor(pts_a[pts_a[:, 2] > z_hint + 0.45, :2] / 0.5).astype(int),
            )
        )
        vb = set(
            map(
                tuple,
                np.floor(pts_b[pts_b[:, 2] > z_hint + 0.45, :2] / 0.5).astype(int),
            )
        )
        shared_cells = len(va & vb)
        cand_slices = _extract_orthogonal_wall_slices(pts_a, pts_b, z_ground=z_hint)
        cand_score = float(shared_cells) * (1.45 if len(cand_slices) >= 2 else 1.0)
        if best is None or cand_score > best["cand_score"]:
            best = {
                "ia": ia,
                "ib": ib,
                "t_phys": t_phys,
                "rel_time_s": float(t_phys - access_a.scan_t_s[0]),
                "sync_dt_ms": sync_dt_ms,
                "pts_a": pts_a,
                "pts_b": pts_b,
                "z_hint": z_hint,
                "shared_cells": shared_cells,
                "cand_score": cand_score,
                "wall_slices": cand_slices,
            }

    if best is None:
        raise RuntimeError("No synchronized static scan pair found between datasets")

    pts_a = best["pts_a"]
    pts_b = best["pts_b"]
    z_hint = best["z_hint"]

    pcd_dir = Path(output_dir) / "pointclouds"
    pcd_path = write_colored_dual_pcd(
        pts_a, pts_b, pcd_dir / "static_dual_calibrated_imu.pcd"
    )

    n_a, d_a, g_in_a = _fit_ground_plane_ransac(pts_a, z_hint=z_hint)
    n_b, d_b, g_in_b = _fit_ground_plane_ransac(pts_b, z_hint=z_hint)
    ground_angle_deg = float(
        math.degrees(math.acos(np.clip(float(n_a @ n_b), -1.0, 1.0)))
    )
    # Evaluate ground Z gap on shared lateral patches (`|x| in [3, 14] m, |y| < 6 m`)
    g_strip_a = g_in_a[(np.abs(g_in_a[:, 0]) >= 3.0) & (np.abs(g_in_a[:, 1]) <= 6.0)]
    g_strip_b = g_in_b[(np.abs(g_in_b[:, 0]) >= 3.0) & (np.abs(g_in_b[:, 1]) <= 6.0)]
    if len(g_strip_a) < 10 or len(g_strip_b) < 10:
        g_strip_a, g_strip_b = g_in_a, g_in_b
    n_mid = 0.5 * (n_a + n_b)
    n_mid /= max(1e-9, float(np.linalg.norm(n_mid)))
    ref_xy = 0.5 * (g_strip_a.mean(axis=0) + g_strip_b.mean(axis=0))
    z_on_a = (-d_a - n_a[0] * ref_xy[0] - n_a[1] * ref_xy[1]) / max(1e-6, n_a[2])
    z_on_b = (-d_b - n_b[0] * ref_xy[0] - n_b[1] * ref_xy[1]) / max(1e-6, n_b[2])
    ground_z_gap_cm = float((z_on_a - z_on_b) * 100.0)

    wall_slices = best["wall_slices"]
    wall_summary = [
        {
            "slice_id": f"wall_{idx + 1}",
            "center_xy_m": item["center_xy_m"],
            "normal_azimuth_deg": item["normal_azimuth_deg"],
            "signed_normal_gap_cm": item["signed_normal_gap_cm"],
            "abs_normal_gap_cm": item["abs_normal_gap_cm"],
            "single_wall_std_cm": item["single_wall_std_cm"],
            "combined_plane_std_cm": item["combined_plane_std_cm"],
            "count_a": item["count_a"],
            "count_b": item["count_b"],
        }
        for idx, item in enumerate(wall_slices)
    ]
    max_wall_gap_cm = (
        max(item["abs_normal_gap_cm"] for item in wall_summary) if wall_summary else 0.0
    )

    metrics = {
        "selected_rel_time_s": best["rel_time_s"],
        "scan_index_a": int(best["ia"]),
        "scan_index_b": int(best["ib"]),
        "sync_Residual_dt_ms": best["sync_dt_ms"],
        "points_a": int(len(pts_a)),
        "points_b": int(len(pts_b)),
        "shared_vertical_cells_0p5m": int(best["shared_cells"]),
        "ground_plane": {
            "normal_a_imu": [float(v) for v in n_a],
            "normal_b_imu": [float(v) for v in n_b],
            "inter_lidar_ground_angle_deg": ground_angle_deg,
            "signed_ground_z_gap_cm": ground_z_gap_cm,
            "abs_ground_z_gap_cm": abs(ground_z_gap_cm),
        },
        "orthogonal_wall_slices": wall_summary,
        "max_abs_wall_gap_cm": float(max_wall_gap_cm),
        "colored_pcd_path": str(pcd_path.resolve()),
        "blind_spot_note": (
            "Static dual-PCD verifies T_A_B = inv(T_I_A) @ T_I_B only; common-mode "
            "IMU bias T_err is gauge-invariant here and checked in Pillar 2/4/5."
        ),
    }
    plot_data = {
        "pts_a": pts_a,
        "pts_b": pts_b,
        "g_strip_a": g_strip_a,
        "g_strip_b": g_strip_b,
        "wall_slices": wall_slices,
        "z_hint": z_hint,
    }
    return metrics, plot_data


def _integrate_short_imu_trajectory(
    imu_t: np.ndarray,
    imu_w: np.ndarray,
    imu_a: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    gyro_bias: np.ndarray,
    acc_bias: np.ndarray,
    v_imu_start: np.ndarray,
    v_imu_end: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = (imu_t >= t_start - 0.15) & (imu_t <= t_end + 0.15)
    ts = imu_t[mask]
    ws = imu_w[mask] - gyro_bias
    fs = imu_a[mask] - acc_bias
    if len(ts) < 4:
        raise RuntimeError("Insufficient IMU samples inside turn submap window")

    n = len(ts)
    rots = np.zeros((n, 3, 3), dtype=np.float64)
    rots[0] = np.eye(3)
    for k in range(n - 1):
        dt = float(ts[k + 1] - ts[k])
        w_mid = 0.5 * (ws[k] + ws[k + 1])
        rots[k + 1] = rots[k] @ Rot.from_rotvec(w_mid * dt).as_matrix()

    idx0 = int(np.argmin(np.abs(ts - t_start)))
    idx1 = int(np.argmin(np.abs(ts - t_end)))
    if idx1 <= idx0:
        idx0, idx1 = 0, n - 1
    R_anchor = rots[idx0].copy()
    for k in range(n):
        rots[k] = R_anchor.T @ rots[k]

    acc_world = np.einsum("nij,nj->ni", rots, fs)
    vel_spec = np.zeros((n, 3), dtype=np.float64)
    for k in range(n - 1):
        dt = float(ts[k + 1] - ts[k])
        vel_spec[k + 1] = vel_spec[k] + 0.5 * (acc_world[k] + acc_world[k + 1]) * dt
    vel_spec -= vel_spec[idx0]

    span = max(1e-3, float(ts[idx1] - ts[idx0]))
    d_spec = vel_spec[idx1] - vel_spec[idx0]
    g_eff = (v_imu_end - v_imu_start - d_spec) / span

    vel = v_imu_start[None, :] + vel_spec + np.outer(ts - ts[idx0], g_eff)
    pos = np.zeros((n, 3), dtype=np.float64)
    for k in range(n - 1):
        dt = float(ts[k + 1] - ts[k])
        pos[k + 1] = pos[k] + 0.5 * (vel[k] + vel[k + 1]) * dt
    pos -= pos[idx0]
    rotvecs = Rot.from_matrix(rots).as_rotvec()
    return {
        "t": ts,
        "pos": pos,
        "rotvec": rotvecs,
        "rots": rots,
        "g_eff_norm": np.array([float(np.linalg.norm(g_eff))]),
    }


def _accumulate_imu_submap(
    access: FastCanonicalScanAccess,
    traj: dict[str, np.ndarray],
    *,
    t_start: float,
    t_end: float,
    R_LI: np.ndarray,
    T_LI: np.ndarray,
    tau_s: float,
    point_stride: int = 3,
) -> np.ndarray:
    scan_phys = access.scan_t_s + tau_s
    indices = np.where((scan_phys >= t_start) & (scan_phys <= t_end))[0]
    chunks: list[np.ndarray] = []
    traj_t = traj["t"]
    traj_pos = traj["pos"]
    traj_rv = traj["rotvec"]

    for idx in indices:
        xyz_l, pt_t = access.read_scan(int(idx))
        if len(xyz_l) == 0:
            continue
        xyz_l = xyz_l[::point_stride]
        pt_t = pt_t[::point_stride]
        xyz_i = xyz_l @ R_LI.T + T_LI
        xyz_i, pt_t_f = _filter_range(
            xyz_i, pt_t, min_xy_m=2.5, max_xy_m=38.0, min_z_m=-2.0, max_z_m=5.0
        )
        if len(xyz_i) == 0 or pt_t_f is None:
            continue
        t_pts = float(access.scan_t_s[int(idx)]) + pt_t_f + tau_s
        # Bin intra-scan points into 10 ms slices (10 slices per 100 ms scan) for exact
        # 100 Hz IMU deskewing without millions of individual matrix exponentials.
        bins = np.clip(np.floor((t_pts - t_pts.min()) / 0.010).astype(int), 0, 15)
        out_scan = np.empty_like(xyz_i)
        for b in np.unique(bins):
            bm = bins == b
            t_rep = float(np.mean(t_pts[bm]))
            rv = np.array([np.interp(t_rep, traj_t, traj_rv[:, ax]) for ax in range(3)])
            p_rep = np.array(
                [np.interp(t_rep, traj_t, traj_pos[:, ax]) for ax in range(3)]
            )
            R_rep = Rot.from_rotvec(rv).as_matrix()
            out_scan[bm] = xyz_i[bm] @ R_rep.T + p_rep
        chunks.append(out_scan)

    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(chunks)


def _single_sensor_wall_voxels(pts: np.ndarray, voxel_m: float) -> list[dict[str, Any]]:
    if len(pts) < 24:
        return []
    vox_keys = np.floor(pts / voxel_m).astype(np.int64)
    order = np.lexsort((vox_keys[:, 2], vox_keys[:, 1], vox_keys[:, 0]))
    sorted_keys = vox_keys[order]
    sorted_pts = pts[order]
    diffs = np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)
    starts = np.concatenate([[0], np.where(diffs)[0] + 1, [len(sorted_pts)]])
    patches: list[dict[str, Any]] = []
    for s, e in zip(starts[:-1], starts[1:]):
        cnt = e - s
        if cnt < 20:
            continue
        patch = sorted_pts[s:e]
        mean = patch.mean(axis=0)
        cov = (patch - mean).T @ (patch - mean) / float(cnt)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[2] < 1e-4:
            continue
        planarity = (eigvals[1] - eigvals[0]) / eigvals[2]
        normal = eigvecs[:, 0]
        if planarity < 0.35 or abs(float(normal[2])) > 0.35:
            continue
        patches.append(
            {
                "count": int(cnt),
                "mean": mean,
                "normal": normal,
                "thickness_cm": float(math.sqrt(max(0.0, float(eigvals[0]))) * 100.0),
            }
        )
    return patches


def _score_submap_wall_sharpness(
    sub_a: np.ndarray,
    sub_b: np.ndarray,
    *,
    z_ground: float,
    voxel_m: float = 0.75,
) -> dict[str, Any]:
    va = sub_a[(sub_a[:, 2] > z_ground + 0.45) & (sub_a[:, 2] < z_ground + 3.5)]
    vb = sub_b[(sub_b[:, 2] > z_ground + 0.45) & (sub_b[:, 2] < z_ground + 3.5)]
    if len(va) < 30 or len(vb) < 30:
        return {
            "planar_voxel_count": 0,
            "shared_vertical_cells_0p5m": 0,
            "median_wall_thickness_cm": 0.0,
            "p90_wall_thickness_cm": 0.0,
            "median_ab_wall_gap_cm": 0.0,
            "best_patch": None,
        }

    cells_a = set(map(tuple, np.floor(va[:, :2] / 0.5).astype(int)))
    cells_b = set(map(tuple, np.floor(vb[:, :2] / 0.5).astype(int)))
    shared_cells_0p5 = len(cells_a & cells_b)

    patches_a = _single_sensor_wall_voxels(va, voxel_m=voxel_m)
    patches_b = _single_sensor_wall_voxels(vb, voxel_m=voxel_m)
    thicknesses_cm = [p["thickness_cm"] for p in patches_a] + [
        p["thickness_cm"] for p in patches_b
    ]

    # Macro cross-sensor wall-to-wall normal distance up to ±2.2 m along wall normal
    macro_gaps_cm: list[float] = []
    best_patch: dict[str, Any] | None = None
    for pa in patches_a:
        mean_a = pa["mean"]
        n_a = pa["normal"]
        n_xy = n_a[:2] / max(1e-6, float(np.linalg.norm(n_a[:2])))
        t_xy = np.array([-n_xy[1], n_xy[0]])
        rel = vb - mean_a
        u_dist = rel[:, :2] @ t_xy
        z_dist = rel[:, 2]
        n_dist = rel @ n_a
        mask = (
            (np.abs(u_dist) <= 0.85)
            & (np.abs(z_dist) <= 0.85)
            & (np.abs(n_dist) <= 2.20)
        )
        if int(mask.sum()) < 16:
            continue
        nd_sub = n_dist[mask]
        med_nd = float(np.median(nd_sub))
        in_wall = np.abs(nd_sub - med_nd) <= 0.22
        if int(in_wall.sum()) < 14:
            continue
        gap_cm = abs(float(np.median(nd_sub[in_wall]))) * 100.0
        macro_gaps_cm.append(gap_cm)
        min_ab = min(pa["count"], int(in_wall.sum()))
        if best_patch is None or min_ab > best_patch["min_ab"]:
            best_patch = {
                "min_ab": min_ab,
                "center": mean_a + 0.5 * med_nd * n_a,
                "normal": n_a,
            }

    if not thicknesses_cm:
        return {
            "planar_voxel_count": 0,
            "shared_vertical_cells_0p5m": int(shared_cells_0p5),
            "median_wall_thickness_cm": 0.0,
            "p90_wall_thickness_cm": 0.0,
            "median_ab_wall_gap_cm": 0.0,
            "best_patch": None,
        }

    return {
        "planar_voxel_count": int(len(thicknesses_cm)),
        "shared_vertical_cells_0p5m": int(shared_cells_0p5),
        "median_wall_thickness_cm": float(np.median(thicknesses_cm)),
        "p90_wall_thickness_cm": float(np.percentile(thicknesses_cm, 90)),
        "median_ab_wall_gap_cm": (
            float(np.median(macro_gaps_cm)) if macro_gaps_cm else 0.0
        ),
        "best_patch": best_patch,
    }


def evaluate_turn_imu_submap(
    access_a: FastCanonicalScanAccess,
    access_b: FastCanonicalScanAccess,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    batch_a: dict[str, Any],
    output_dir: Path,
    *,
    turn_time_s: float | None = None,
    turn_duration_s: float = 2.6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    R_LI_a = _project_so3(np.asarray(result_a["rotation_LI_matrix"], dtype=float))
    T_LI_a = np.asarray(result_a["translation_LI_m"], dtype=float)
    T_IL_a = np.asarray(result_a["translation_IL_m"], dtype=float)
    tau_a = float(result_a["time_offset_s"])
    bg = 0.5 * (
        np.asarray(result_a["gyro_bias_rad_s"], dtype=float)
        + np.asarray(result_b["gyro_bias_rad_s"], dtype=float)
    )
    ba = 0.5 * (
        np.asarray(result_a["acc_bias_m_s2"], dtype=float)
        + np.asarray(result_b["acc_bias_m_s2"], dtype=float)
    )

    R_LI_b = _project_so3(np.asarray(result_b["rotation_LI_matrix"], dtype=float))
    T_LI_b = np.asarray(result_b["translation_LI_m"], dtype=float)
    tau_b = float(result_b["time_offset_s"])

    lid_t = batch_a["lid_t"]
    dt_med = float(np.median(np.diff(lid_t))) if len(lid_t) > 1 else 1.0 / 30.0
    win_steps = max(4, min(len(lid_t) - 1, int(round(turn_duration_s / dt_med))))

    if turn_time_s is not None:
        t_center = float(access_a.scan_t_s[0]) + float(turn_time_s)
        i0 = int(
            np.clip(
                np.argmin(np.abs(lid_t - (t_center - 0.5 * turn_duration_s))),
                0,
                max(0, len(lid_t) - win_steps - 1),
            )
        )
    else:
        best_ang = -1.0
        i0 = 0
        stride = max(1, win_steps // 6)
        for idx in range(0, max(1, len(lid_t) - win_steps), stride):
            dR = batch_a["lid_R"][idx].T @ batch_a["lid_R"][idx + win_steps]
            ang = float(np.degrees(Rot.from_matrix(_project_so3(dR)).magnitude()))
            spd = float(
                np.mean(np.linalg.norm(batch_a["lid_v"][idx : idx + win_steps], axis=1))
            )
            score = ang * (1.0 if spd >= 1.2 else 0.4)
            if score > best_ang:
                best_ang = score
                i0 = idx

    i1 = min(len(lid_t) - 1, i0 + win_steps)
    t_start = float(lid_t[i0] + tau_a)
    t_end = float(lid_t[i1] + tau_a)

    # Initial and final IMU-origin velocity in frame I(t0)
    dR_01_L = _project_so3(batch_a["lid_R"][i0].T @ batch_a["lid_R"][i1])
    R_I0_I1 = R_LI_a @ dR_01_L @ R_LI_a.T
    v_L0_local = batch_a["lid_R"][i0].T @ batch_a["lid_v"][i0]
    v_L1_local = batch_a["lid_R"][i1].T @ batch_a["lid_v"][i1]
    v_I0 = R_LI_a @ (v_L0_local + np.cross(batch_a["lid_w"][i0], T_IL_a))
    v_I1_in_I1 = R_LI_a @ (v_L1_local + np.cross(batch_a["lid_w"][i1], T_IL_a))
    v_I1_in_I0 = R_I0_I1 @ v_I1_in_I1

    traj = _integrate_short_imu_trajectory(
        access_a.imu_t_s,
        access_a.imu_w,
        access_a.imu_a,
        t_start=t_start,
        t_end=t_end,
        gyro_bias=bg,
        acc_bias=ba,
        v_imu_start=v_I0,
        v_imu_end=v_I1_in_I0,
    )
    total_imu_rot_deg = float(
        np.degrees(np.linalg.norm(traj["rotvec"][-1] - traj["rotvec"][0]))
    )

    # 1. Calibrated submap
    cal_a = _accumulate_imu_submap(
        access_a,
        traj,
        t_start=t_start,
        t_end=t_end,
        R_LI=R_LI_a,
        T_LI=T_LI_a,
        tau_s=tau_a,
    )
    cal_b = _accumulate_imu_submap(
        access_b,
        traj,
        t_start=t_start,
        t_end=t_end,
        R_LI=R_LI_b,
        T_LI=T_LI_b,
        tau_s=tau_b,
    )
    z_ground = (
        float(np.percentile(np.vstack([cal_a, cal_b])[:, 2], 10))
        if len(cal_a) and len(cal_b)
        else -0.4
    )
    sharp_cal = _score_submap_wall_sharpness(cal_a, cal_b, z_ground=z_ground)

    # 2. Zero lever-arm & zero time-lag ablation
    zero_a = _accumulate_imu_submap(
        access_a,
        traj,
        t_start=t_start,
        t_end=t_end,
        R_LI=R_LI_a,
        T_LI=np.zeros(3),
        tau_s=0.0,
    )
    zero_b = _accumulate_imu_submap(
        access_b,
        traj,
        t_start=t_start,
        t_end=t_end,
        R_LI=R_LI_b,
        T_LI=np.zeros(3),
        tau_s=0.0,
    )
    sharp_zero = _score_submap_wall_sharpness(zero_a, zero_b, z_ground=z_ground)

    # 3. Injected COMMON-MODE lever + attitude/timing bias (invisible in static PCD)
    dR_common = Rot.from_euler("xyz", [2.5, -2.5, 0.0], degrees=True).as_matrix()
    dt_common = np.array([0.45, 0.45, 0.0], dtype=np.float64)
    dtau_common = 0.045
    cm_a = _accumulate_imu_submap(
        access_a,
        traj,
        t_start=t_start,
        t_end=t_end,
        R_LI=dR_common @ R_LI_a,
        T_LI=dR_common @ T_LI_a + dt_common,
        tau_s=tau_a + dtau_common,
    )
    cm_b = _accumulate_imu_submap(
        access_b,
        traj,
        t_start=t_start,
        t_end=t_end,
        R_LI=dR_common @ R_LI_b,
        T_LI=dR_common @ T_LI_b + dt_common,
        tau_s=tau_b + dtau_common,
    )
    sharp_cm = _score_submap_wall_sharpness(cm_a, cm_b, z_ground=z_ground)

    pcd_dir = Path(output_dir) / "pointclouds"
    cal_pcd = write_colored_dual_pcd(
        cal_a, cal_b, pcd_dir / "turn_submap_calibrated_imu.pcd"
    )
    cm_pcd = write_colored_dual_pcd(
        cm_a, cm_b, pcd_dir / "turn_submap_common_mode_bias_imu.pcd"
    )

    base_thick = max(0.1, sharp_cal["median_wall_thickness_cm"])
    cm_blur_ratio = (
        sharp_cm["median_wall_thickness_cm"] / base_thick
        if sharp_cm["planar_voxel_count"] > 0
        else 1.0
    )
    zero_blur_ratio = (
        sharp_zero["median_wall_thickness_cm"] / base_thick
        if sharp_zero["planar_voxel_count"] > 0
        else 1.0
    )

    def _clean(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k != "best_patch"}

    metrics = {
        "window_rel_start_s": float(t_start - access_a.scan_t_s[0]),
        "window_rel_end_s": float(t_end - access_a.scan_t_s[0]),
        "duration_s": float(t_end - t_start),
        "integrated_imu_rotation_deg": total_imu_rot_deg,
        "effective_gravity_norm_m_s2": float(traj["g_eff_norm"][0]),
        "calibrated": _clean(sharp_cal),
        "ablation_zero_lever_zero_lag": _clean(sharp_zero),
        "ablation_common_mode_imu_bias": _clean(sharp_cm),
        "common_mode_blur_inflation_ratio": float(cm_blur_ratio),
        "zero_lever_blur_inflation_ratio": float(zero_blur_ratio),
        "calibrated_submap_pcd_path": str(cal_pcd.resolve()),
        "common_mode_control_pcd_path": str(cm_pcd.resolve()),
    }
    plot_data = {
        "cal_a": cal_a,
        "cal_b": cal_b,
        "zero_a": zero_a,
        "zero_b": zero_b,
        "cm_a": cm_a,
        "cm_b": cm_b,
        "traj": traj,
        "z_ground": z_ground,
        "best_patch": sharp_cal.get("best_patch"),
    }
    return metrics, plot_data


def evaluate_l2l_cycle_consistency(
    access_a: FastCanonicalScanAccess,
    access_b: FastCanonicalScanAccess,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    *,
    center_scan_a: int,
    center_scan_b: int,
    output_dir: Path | None = None,
    accum_scans: int = 8,
) -> dict[str, Any]:
    T_I_A = _extrinsic_matrix_LI(result_a)
    T_I_B = _extrinsic_matrix_LI(result_b)
    # Induced transform mapping points in B -> A: p_A = inv(T_I_A) @ T_I_B @ p_B
    T_A_B_induced = np.linalg.inv(T_I_A) @ T_I_B

    half = max(1, accum_scans // 2)
    pts_a_list = []
    int_a_list = []
    has_int_a = True
    for idx in range(
        max(0, center_scan_a - half),
        min(len(access_a.scan_t_s), center_scan_a + half + 1),
    ):
        xyz, _, inten, _ = access_a.read_scan_with_attributes(idx)
        r = np.linalg.norm(xyz[:, :2], axis=1)
        m = (r >= 2.4) & (r <= 42.0)
        pts_a_list.append(xyz[m])
        if inten is not None and len(inten) == len(xyz):
            int_a_list.append(inten[m])
        else:
            has_int_a = False
    pts_b_list = []
    int_b_list = []
    has_int_b = True
    for idx in range(
        max(0, center_scan_b - half),
        min(len(access_b.scan_t_s), center_scan_b + half + 1),
    ):
        xyz, _, inten, _ = access_b.read_scan_with_attributes(idx)
        r = np.linalg.norm(xyz[:, :2], axis=1)
        m = (r >= 2.4) & (r <= 42.0)
        pts_b_list.append(xyz[m])
        if inten is not None and len(inten) == len(xyz):
            int_b_list.append(inten[m])
        else:
            has_int_b = False

    raw_a = np.vstack(pts_a_list) if pts_a_list else np.zeros((0, 3))
    raw_b = np.vstack(pts_b_list) if pts_b_list else np.zeros((0, 3))
    raw_int_a = np.concatenate(int_a_list) if has_int_a and int_a_list else None
    raw_int_b = np.concatenate(int_b_list) if has_int_b and int_b_list else None

    try:
        import open3d as o3d
    except ImportError:
        return {
            "available": False,
            "reason": "open3d not installed",
        }

    if len(raw_a) < 100 or len(raw_b) < 100:
        return {
            "available": False,
            "reason": "insufficient static points for L2L GICP",
        }

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(raw_a)
    pcd_target = pcd_target.voxel_down_sample(0.15)
    pcd_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30)
    )

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(raw_b)
    pcd_source = pcd_source.voxel_down_sample(0.15)
    pcd_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30)
    )

    T_curr = T_A_B_induced.copy()
    reg = None
    for max_corr in (1.0, 0.45, 0.20):
        reg = o3d.pipelines.registration.registration_generalized_icp(
            pcd_source,
            pcd_target,
            max_corr,
            T_curr,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        T_curr = np.asarray(reg.transformation, dtype=np.float64)

    assert reg is not None
    info_mat = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd_source, pcd_target, 0.25, T_curr
    )
    eigvals = np.linalg.eigvalsh(np.asarray(info_mat, dtype=np.float64))
    cond_num = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1e-6 else float("inf")

    T_loop = np.linalg.inv(T_A_B_induced) @ T_curr
    rot_res_deg = _rotation_error_deg(np.eye(3), T_loop[:3, :3])
    trans_res_m = float(np.linalg.norm(T_loop[:3, 3]))
    rpy_res_deg = Rot.from_matrix(_project_so3(T_loop[:3, :3])).as_euler(
        "xyz", degrees=True
    )
    T_I_B_gicp = T_I_A @ T_curr
    imu_xyz_residual_m = (T_I_B_gicp[:3, 3] - T_I_B[:3, 3]).tolist()

    loop_pcd_str = ""
    if output_dir is not None:
        raw_a0, _ = access_a.read_scan(center_scan_a)
        raw_b0, _ = access_b.read_scan(center_scan_b)
        pa_i, _ = _filter_range(raw_a0 @ T_I_A[:3, :3].T + T_I_A[:3, 3])
        pb_i_aligned, _ = _filter_range(
            raw_b0 @ T_I_B_gicp[:3, :3].T + T_I_B_gicp[:3, 3]
        )
        loop_pcd = write_colored_dual_pcd(
            pa_i,
            pb_i_aligned,
            Path(output_dir) / "pointclouds" / "static_dual_l2l_loop_aligned_imu.pcd",
        )
        loop_pcd_str = str(loop_pcd.resolve())

    from lidar2lidar.semantic_registration import LidarFrameData, register_semantic_l2l

    sem_summary: dict[str, Any] = {"available": False}
    sem_res = register_semantic_l2l(
        LidarFrameData(points=raw_b, intensity=raw_int_b),
        LidarFrameData(points=raw_a, intensity=raw_int_a),
        T_A_B_induced,
    )
    if sem_res.get("success"):
        T_sem = np.asarray(sem_res["transformation"], dtype=np.float64)
        T_loop_sem = np.linalg.inv(T_A_B_induced) @ T_sem
        sem_summary = {
            "available": True,
            "corridor_degenerate": bool(
                sem_res["localizability"]["corridor_degenerate"]
            ),
            "translation_condition_number": float(
                sem_res["localizability"]["translation_condition_number"]
            ),
            "loop_rotation_residual_deg": float(
                _rotation_error_deg(np.eye(3), T_loop_sem[:3, :3])
            ),
            "loop_translation_residual_m": float(np.linalg.norm(T_loop_sem[:3, 3])),
            "channel_matches": sem_res["channel_matches"],
            "channel_rmse_m": sem_res["channel_rmse_m"],
            "direct_semantic_T_A_B": T_sem.tolist(),
        }

    return {
        "available": True,
        "stacked_static_scans": int(accum_scans),
        "downsampled_points_a": int(len(pcd_target.points)),
        "downsampled_points_b": int(len(pcd_source.points)),
        "gicp_fitness_overlap": float(reg.fitness),
        "gicp_inlier_rmse_m": float(reg.inlier_rmse),
        "info_matrix_eigenvalues": [float(v) for v in eigvals],
        "info_matrix_condition_number": cond_num,
        "loop_rotation_residual_deg": float(rot_res_deg),
        "loop_translation_residual_m": float(trans_res_m),
        "loop_rpy_residual_deg": [float(v) for v in rpy_res_deg],
        "loop_xyz_residual_m": [float(v) for v in T_loop[:3, 3]],
        "loop_imu_frame_xyz_residual_m": [float(v) for v in imu_xyz_residual_m],
        "induced_T_A_B": T_A_B_induced.tolist(),
        "direct_gicp_T_A_B": T_curr.tolist(),
        "l2l_loop_aligned_pcd_path": loop_pcd_str,
        "semantic_l2l": sem_summary,
    }


def evaluate_gravity_ground_alignment(
    access_a: FastCanonicalScanAccess,
    access_b: FastCanonicalScanAccess,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    batch_a: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    R_LI_a = _project_so3(np.asarray(result_a["rotation_LI_matrix"], dtype=float))
    T_LI_a = np.asarray(result_a["translation_LI_m"], dtype=float)
    tau_a = float(result_a["time_offset_s"])
    R_LI_b = _project_so3(np.asarray(result_b["rotation_LI_matrix"], dtype=float))
    T_LI_b = np.asarray(result_b["translation_LI_m"], dtype=float)
    tau_b = float(result_b["time_offset_s"])
    ba = 0.5 * (
        np.asarray(result_a["acc_bias_m_s2"], dtype=float)
        + np.asarray(result_b["acc_bias_m_s2"], dtype=float)
    )

    lid_t = batch_a["lid_t"]
    samples: list[dict[str, Any]] = []

    # Candidate windows: pre-departure rest + low-angular-rate quasi-uniform windows
    # across distinct vehicle yaw bins to capture opposite headings (road-slope cancel).
    cand_times: list[tuple[float, float, np.ndarray]] = []
    if len(lid_t) > 0 and (lid_t[0] - access_a.scan_t_s[0]) > 4.0:
        t_rest = float(access_a.scan_t_s[0]) + 0.5 * float(
            lid_t[0] - access_a.scan_t_s[0]
        )
        cand_times.append((t_rest, 0.0, np.zeros(3)))

    if len(lid_t) > 30:
        dt_med = float(np.median(np.diff(lid_t)))
        v_smooth = _zero_phase_smooth(batch_a["lid_v"], window=9)
        a_world = np.gradient(v_smooth, dt_med, axis=0)
        omega_norm = np.linalg.norm(batch_a["lid_w"], axis=1)
        acc_norm = np.linalg.norm(a_world, axis=1)
        yaws = np.array(
            [
                math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
                for R in batch_a["lid_R"]
            ]
        )
        for bin_deg in range(-180, 180, 30):
            in_bin = np.where(
                (yaws >= bin_deg)
                & (yaws < bin_deg + 30)
                & (omega_norm < 0.04)
                & (acc_norm < 0.25)
            )[0]
            if len(in_bin) == 0:
                continue
            best_i = int(in_bin[np.argmin(omega_norm[in_bin] + 0.5 * acc_norm[in_bin])])
            a_local = batch_a["lid_R"][best_i].T @ a_world[best_i]
            cand_times.append(
                (float(lid_t[best_i]), float(yaws[best_i]), R_LI_a @ a_local)
            )

    if not cand_times:
        cand_times.append((float(access_a.scan_t_s[0]), 0.0, np.zeros(3)))

    for t_win, yaw_deg, a_kin_imu in cand_times:
        ia = int(np.argmin(np.abs((access_a.scan_t_s + tau_a) - t_win)))
        ib = int(np.argmin(np.abs((access_b.scan_t_s + tau_b) - t_win)))
        imu_mask = np.abs(access_a.imu_t_s - t_win) <= 0.45
        if int(imu_mask.sum()) < 5:
            continue
        f_mean = access_a.imu_a[imu_mask].mean(axis=0) - ba
        # Specific force f_I = a_kin_I - g_I => upward gravity u_up = norm(f_I - a_kin)
        g_up = f_mean - a_kin_imu
        g_norm = float(np.linalg.norm(g_up))
        if g_norm < 7.0 or g_norm > 12.0:
            continue
        u_g = g_up / g_norm
        if u_g[2] < 0.0:
            u_g = -u_g

        raw_a, _ = access_a.read_scan(ia)
        raw_b, _ = access_b.read_scan(ib)
        pts_a, _ = _filter_range(raw_a @ R_LI_a.T + T_LI_a)
        pts_b, _ = _filter_range(raw_b @ R_LI_b.T + T_LI_b)
        if len(pts_a) < 40 or len(pts_b) < 40:
            continue
        z_hint = float(np.percentile(np.concatenate([pts_a[:, 2], pts_b[:, 2]]), 12))
        n_a, _, _ = _fit_ground_plane_ransac(pts_a, z_hint=z_hint)
        n_b, _, _ = _fit_ground_plane_ransac(pts_b, z_hint=z_hint)

        ab_deg = float(math.degrees(math.acos(np.clip(float(n_a @ n_b), -1.0, 1.0))))
        ag_deg = float(math.degrees(math.acos(np.clip(float(n_a @ u_g), -1.0, 1.0))))
        bg_deg = float(math.degrees(math.acos(np.clip(float(n_b @ u_g), -1.0, 1.0))))
        n_mid = 0.5 * (n_a + n_b)
        n_mid /= max(1e-9, float(np.linalg.norm(n_mid)))
        diff_vec_deg = np.degrees(n_mid[:2] - u_g[:2])
        samples.append(
            {
                "rel_time_s": float(t_win - access_a.scan_t_s[0]),
                "heading_yaw_deg": float(yaw_deg),
                "inter_lidar_normal_deg": ab_deg,
                "a_to_imu_gravity_deg": ag_deg,
                "b_to_imu_gravity_deg": bg_deg,
                "tilt_xy_diff_deg": [float(diff_vec_deg[0]), float(diff_vec_deg[1])],
            }
        )

    if not samples:
        return (
            {
                "evaluated_windows": 0,
                "inter_lidar_normal_median_deg": 0.0,
                "a_to_imu_gravity_median_deg": 0.0,
                "b_to_imu_gravity_median_deg": 0.0,
                "opposite_heading_pairs": 0,
                "slope_cancelled_tilt_bias_deg": 0.0,
            },
            [],
        )

    # Opposite-heading pairs (`150 <= |dyaw| <= 210 deg`): physical road slope flips
    # sign in body RFU while any true sensor-to-IMU mounting bias stays constant!
    opp_biases: list[np.ndarray] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            dyaw = abs(
                (samples[i]["heading_yaw_deg"] - samples[j]["heading_yaw_deg"] + 180.0)
                % 360.0
                - 180.0
            )
            if dyaw >= 150.0:
                v_i = np.asarray(samples[i]["tilt_xy_diff_deg"], dtype=float)
                v_j = np.asarray(samples[j]["tilt_xy_diff_deg"], dtype=float)
                opp_biases.append(0.5 * (v_i + v_j))

    if opp_biases:
        mean_bias = np.median(np.vstack(opp_biases), axis=0)
    else:
        mean_bias = np.median(
            np.array([s["tilt_xy_diff_deg"] for s in samples], dtype=float), axis=0
        )

    summary = {
        "evaluated_windows": int(len(samples)),
        "inter_lidar_normal_median_deg": float(
            np.median([s["inter_lidar_normal_deg"] for s in samples])
        ),
        "a_to_imu_gravity_median_deg": float(
            np.median([s["a_to_imu_gravity_deg"] for s in samples])
        ),
        "b_to_imu_gravity_median_deg": float(
            np.median([s["b_to_imu_gravity_deg"] for s in samples])
        ),
        "opposite_heading_pairs": int(len(opp_biases)),
        "slope_cancelled_xy_tilt_bias_deg": [
            float(mean_bias[0]),
            float(mean_bias[1]),
        ],
        "slope_cancelled_tilt_bias_norm_deg": float(np.linalg.norm(mean_bias)),
    }
    return summary, samples


def evaluate_three_curve_dynamics(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    batch_a: dict[str, Any],
    batch_b: dict[str, Any],
    *,
    focus_center_s: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    R_LI_a = _project_so3(np.asarray(result_a["rotation_LI_matrix"], dtype=float))
    T_IL_a = np.asarray(result_a["translation_IL_m"], dtype=float)
    tau_a = float(result_a["time_offset_s"])
    R_LI_b = _project_so3(np.asarray(result_b["rotation_LI_matrix"], dtype=float))
    T_IL_b = np.asarray(result_b["translation_IL_m"], dtype=float)
    tau_b = float(result_b["time_offset_s"])
    bg = 0.5 * (
        np.asarray(result_a["gyro_bias_rad_s"], dtype=float)
        + np.asarray(result_b["gyro_bias_rad_s"], dtype=float)
    )

    ta = batch_a["lid_t"] + tau_a
    tb = batch_b["lid_t"] + tau_b
    t_min = max(float(ta[0]), float(tb[0])) + 0.5
    t_max = min(float(ta[-1]), float(tb[-1])) - 0.5
    dt_med = float(np.median(np.diff(ta))) if len(ta) > 1 else 1.0 / 30.0
    t_grid = np.arange(t_min, t_max, dt_med)
    if len(t_grid) < 30:
        t_grid = ta.copy()

    w_imu = _zero_phase_smooth(
        _interp_vec3(batch_a["imu_t"], batch_a["imu_w"] - bg, t_grid), window=7
    )
    w_la = _zero_phase_smooth(_interp_vec3(ta, batch_a["lid_w"], t_grid), window=7)
    w_lb = _zero_phase_smooth(_interp_vec3(tb, batch_b["lid_w"], t_grid), window=7)
    w_a_in_i = w_la @ R_LI_a.T
    w_b_in_i = w_lb @ R_LI_b.T

    # Body-frame kinematic linear acceleration at each sensor vs projected to IMU origin
    def _body_acc(batch: dict[str, Any], t_src: np.ndarray) -> np.ndarray:
        v_sm = _zero_phase_smooth(batch["lid_v"], window=11)
        dt_b = float(np.median(np.diff(t_src))) if len(t_src) > 1 else dt_med
        a_w = np.gradient(v_sm, dt_b, axis=0)
        a_l = np.einsum("nji,nj->ni", batch["lid_R"], a_w)
        return _zero_phase_smooth(_interp_vec3(t_src, a_l, t_grid), window=9)

    a_la = _body_acc(batch_a, ta)
    a_lb = _body_acc(batch_b, tb)
    alpha_i = np.gradient(w_imu, dt_med, axis=0)
    omega_la = w_imu @ R_LI_a  # R_IL @ w_imu
    alpha_la = alpha_i @ R_LI_a
    omega_lb = w_imu @ R_LI_b
    alpha_lb = alpha_i @ R_LI_b

    lever_a_in_l = _zero_phase_smooth(
        _zero_phase_smooth(
            np.cross(alpha_la, T_IL_a) + np.cross(omega_la, np.cross(omega_la, T_IL_a)),
            window=11,
        ),
        window=9,
    )
    lever_b_in_l = _zero_phase_smooth(
        _zero_phase_smooth(
            np.cross(alpha_lb, T_IL_b) + np.cross(omega_lb, np.cross(omega_lb, T_IL_b)),
            window=11,
        ),
        window=9,
    )

    a_a_uncomp_i = a_la @ R_LI_a.T
    a_b_uncomp_i = a_lb @ R_LI_b.T
    a_a_comp_i = (a_la + lever_a_in_l) @ R_LI_a.T
    a_b_comp_i = (a_lb + lever_b_in_l) @ R_LI_b.T

    # Non-differentiated velocity check (immune to EKF derivative damping):
    # v_I = R_LI * v_L - w_I x t_LI = R_LI * (v_L + w_L x T_IL)
    va_loc = np.einsum("nji,nj->ni", batch_a["lid_R"], batch_a["lid_v"])
    vb_loc = np.einsum("nji,nj->ni", batch_b["lid_R"], batch_b["lid_v"])
    va_i_raw = _interp_vec3(ta, va_loc, t_grid) @ R_LI_a.T
    vb_i_raw = _interp_vec3(tb, vb_loc, t_grid) @ R_LI_b.T
    va_i_comp = va_i_raw + np.cross(w_imu, R_LI_a @ T_IL_a)
    vb_i_comp = vb_i_raw + np.cross(w_imu, R_LI_b @ T_IL_b)

    # Evaluate lever-arm benefit during active angular acceleration / turn
    active = np.abs(alpha_i[:, 2]) >= float(np.percentile(np.abs(alpha_i[:, 2]), 80))
    if int(active.sum()) < 10:
        active = np.ones(len(t_grid), dtype=bool)
    active_vel = np.abs(w_imu[:, 2]) >= float(np.percentile(np.abs(w_imu[:, 2]), 85))
    if int(active_vel.sum()) < 10:
        active_vel = active

    uncomp_ab_vel_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (va_i_raw[active_vel, :2] - vb_i_raw[active_vel, :2]) ** 2, axis=1
                )
            )
        )
    )
    comp_ab_vel_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (va_i_comp[active_vel, :2] - vb_i_comp[active_vel, :2]) ** 2, axis=1
                )
            )
        )
    )
    vel_reduction_pct = (
        float((uncomp_ab_vel_rms - comp_ab_vel_rms) / uncomp_ab_vel_rms * 100.0)
        if uncomp_ab_vel_rms > 1e-6
        else 0.0
    )

    uncomp_ab_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (a_a_uncomp_i[active, :2] - a_b_uncomp_i[active, :2]) ** 2, axis=1
                )
            )
        )
    )
    comp_ab_rms = float(
        np.sqrt(
            np.mean(
                np.sum((a_a_comp_i[active, :2] - a_b_comp_i[active, :2]) ** 2, axis=1)
            )
        )
    )
    reduction_pct = (
        float((uncomp_ab_rms - comp_ab_rms) / uncomp_ab_rms * 100.0)
        if uncomp_ab_rms > 1e-6
        else 0.0
    )

    gyro_rmse_a = float(np.sqrt(np.mean(np.sum((w_a_in_i - w_imu) ** 2, axis=1))))
    gyro_rmse_b = float(np.sqrt(np.mean(np.sum((w_b_in_i - w_imu) ** 2, axis=1))))
    gyro_rmse_ab = float(np.sqrt(np.mean(np.sum((w_a_in_i - w_b_in_i) ** 2, axis=1))))
    corr_a = (
        float(np.corrcoef(w_imu[:, 2], w_a_in_i[:, 2])[0, 1])
        if np.std(w_imu[:, 2]) > 1e-6
        else 1.0
    )
    corr_b = (
        float(np.corrcoef(w_imu[:, 2], w_b_in_i[:, 2])[0, 1])
        if np.std(w_imu[:, 2]) > 1e-6
        else 1.0
    )

    if focus_center_s is None:
        peak_idx = int(np.argmax(np.abs(alpha_i[:, 2]) + 0.5 * np.abs(w_imu[:, 2])))
        focus_center_s = float(t_grid[peak_idx])
    focus_mask = (t_grid >= focus_center_s - 6.0) & (t_grid <= focus_center_s + 6.0)
    if int(focus_mask.sum()) < 10:
        focus_mask = np.ones(len(t_grid), dtype=bool)

    metrics = {
        "evaluated_samples": int(len(t_grid)),
        "gyro_a_to_imu_rmse_rad_s": gyro_rmse_a,
        "gyro_b_to_imu_rmse_rad_s": gyro_rmse_b,
        "gyro_a_to_b_rmse_rad_s": gyro_rmse_ab,
        "wz_pearson_a": corr_a,
        "wz_pearson_b": corr_b,
        "active_turn_ab_planar_acc_gap_uncompensated_m_s2": uncomp_ab_rms,
        "active_turn_ab_planar_acc_gap_compensated_m_s2": comp_ab_rms,
        "lever_arm_ab_acc_gap_reduction_pct": reduction_pct,
        "active_turn_ab_planar_vel_gap_uncompensated_m_s": uncomp_ab_vel_rms,
        "active_turn_ab_planar_vel_gap_compensated_m_s": comp_ab_vel_rms,
        "lever_arm_ab_vel_gap_reduction_pct": vel_reduction_pct,
    }
    plot_data = {
        "t_rel": t_grid[focus_mask] - float(t_grid[0]),
        "w_imu": w_imu[focus_mask],
        "w_a": w_a_in_i[focus_mask],
        "w_b": w_b_in_i[focus_mask],
        "a_a_uncomp": a_a_uncomp_i[focus_mask],
        "a_b_uncomp": a_b_uncomp_i[focus_mask],
        "a_a_comp": a_a_comp_i[focus_mask],
        "a_b_comp": a_b_comp_i[focus_mask],
    }
    return metrics, plot_data


def _render_diagnostic_figures(
    *,
    label_a: str,
    label_b: str,
    report: dict[str, Any],
    static_plot: dict[str, Any],
    turn_plot: dict[str, Any],
    grav_samples: list[dict[str, Any]],
    dyn_plot: dict[str, Any],
    diag_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    diag_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    col_a = "#eb4236"
    col_b = "#1f9ee0"

    # Figure 1: Static Dual-Color BEV + Ground Z-Slice + Orthogonal Wall Cross-Sections
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    pa = static_plot["pts_a"]
    pb = static_plot["pts_b"]
    ax = axes1[0, 0]
    ax.scatter(pa[::2, 0], pa[::2, 1], s=0.35, c=col_a, alpha=0.55, label=label_a)
    ax.scatter(pb[::2, 0], pb[::2, 1], s=0.35, c=col_b, alpha=0.55, label=label_b)
    for idx, ws in enumerate(static_plot["wall_slices"]):
        cx, cy = ws["center_xy_m"]
        ax.plot(cx, cy, "ko", ms=6)
        ax.annotate(
            f"W{idx + 1} (Δn={ws['signed_normal_gap_cm']:+.1f}cm)",
            xy=(cx, cy),
            xytext=(cx + 1.5, cy + 1.5),
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85},
            arrowprops={"arrowstyle": "->", "lw": 1.0},
        )
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("IMU X Right (m)")
    ax.set_ylabel("IMU Y Forward (m)")
    st_m = report["pillars"]["static_dual_overlay"]
    t_rel_s = st_m["selected_rel_time_s"]
    ax.set_title(
        f"Static Structured Dual-PCD in IMU Frame (t={t_rel_s:.1f}s)\n"
        f"Shared 0.5m Vertical Cells = {st_m['shared_vertical_cells_0p5m']}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(markerscale=10, loc="upper right")

    axg = axes1[0, 1]
    ga = static_plot["g_strip_a"]
    gb = static_plot["g_strip_b"]
    axg.scatter(ga[:, 0], ga[:, 2] * 100.0, s=2.0, c=col_a, alpha=0.45, label=label_a)
    axg.scatter(gb[:, 0], gb[:, 2] * 100.0, s=2.0, c=col_b, alpha=0.45, label=label_b)
    gp = st_m["ground_plane"]
    axg.set_xlabel("Lateral X in IMU Frame (m)")
    axg.set_ylabel("Ground Height Z in IMU Frame (cm)")
    axg.grid(True, alpha=0.3)
    axg.set_title(
        "Orthogonal Ground Z-Slice Profile\n"
        f"Δz = {gp['signed_ground_z_gap_cm']:+.2f} cm | "
        f"Normal Angle = {gp['inter_lidar_ground_angle_deg']:.3f}°",
        fontsize=10.5,
        fontweight="bold",
    )
    axg.legend(markerscale=4, loc="best")

    for w_idx in range(2):
        axw = axes1[1, w_idx]
        if w_idx < len(static_plot["wall_slices"]):
            ws = static_plot["wall_slices"][w_idx]
            axw.scatter(
                ws["u_a_m"], ws["n_a_cm"], s=8, c=col_a, alpha=0.65, label=label_a
            )
            axw.scatter(
                ws["u_b_m"], ws["n_b_cm"], s=8, c=col_b, alpha=0.65, label=label_b
            )
            axw.axhline(float(np.median(ws["n_a_cm"])), color=col_a, ls="--", lw=1.2)
            axw.axhline(float(np.median(ws["n_b_cm"])), color=col_b, ls="--", lw=1.2)
            max_n_plot = max(
                22.0,
                1.25
                * float(
                    max(
                        np.max(np.abs(ws["n_a_cm"])),
                        np.max(np.abs(ws["n_b_cm"])),
                    )
                ),
            )
            axw.set_ylim(-max_n_plot, max_n_plot)
            axw.set_xlabel("Wall Tangent Coordinate u (m)")
            axw.set_ylabel("Signed Wall-Normal Residual n (cm)")
            az_deg = ws["normal_azimuth_deg"]
            gap_val = ws["signed_normal_gap_cm"]
            s_std = ws["single_wall_std_cm"]
            axw.set_title(
                f"Vertical Wall Slice W{w_idx + 1} (Azimuth={az_deg:.0f}°)\n"
                f"Signed Gap Δn = {gap_val:+.2f} cm | Wall σ = {s_std:.2f} cm",
                fontsize=10.5,
                fontweight="bold",
            )
            axw.grid(True, alpha=0.3)
            axw.legend(loc="upper right")
        else:
            axw.text(0.5, 0.5, "No second orthogonal wall in static FoV", ha="center")
            axw.axis("off")

    fig1.tight_layout()
    f1_path = diag_dir / "fig1_static_dual_pcd_and_slices.png"
    fig1.savefig(f1_path)
    plt.close(fig1)
    written.append(str(f1_path.resolve()))

    # Figure 2: Turn-Window Pure-IMU Submap & Common-Mode Blind-Spot Proof
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12), dpi=160)
    tm = report["pillars"]["turn_imu_submap"]
    cal_a, cal_b = turn_plot["cal_a"], turn_plot["cal_b"]
    cm_a, cm_b = turn_plot["cm_a"], turn_plot["cm_b"]
    zg = turn_plot["z_ground"]

    ax = axes2[0, 0]
    va = cal_a[cal_a[:, 2] > zg + 0.35]
    vb = cal_b[cal_b[:, 2] > zg + 0.35]
    ax.scatter(va[::3, 0], va[::3, 1], s=0.3, c=col_a, alpha=0.45, label=label_a)
    ax.scatter(vb[::3, 0], vb[::3, 1], s=0.3, c=col_b, alpha=0.45, label=label_b)
    ax.plot(
        turn_plot["traj"]["pos"][:, 0],
        turn_plot["traj"]["pos"][:, 1],
        "k-",
        lw=2.0,
        label="100Hz IMU Strapdown",
    )
    bp = turn_plot.get("best_patch")
    if bp is not None:
        cx, cy = float(bp["center"][0]), float(bp["center"][1])
        ax.plot(cx, cy, "yo", ms=8, mec="k")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-32, 32)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Local I0 X (m)")
    ax.set_ylabel("Local I0 Y (m)")
    t0_w = tm["window_rel_start_s"]
    t1_w = tm["window_rel_end_s"]
    dpsi = tm["integrated_imu_rotation_deg"]
    c_stat = st_m["shared_vertical_cells_0p5m"]
    c_turn = tm["calibrated"]["shared_vertical_cells_0p5m"]
    ax.set_title(
        f"Turn Pure-IMU Cross-Time Submap ([{t0_w:.1f}, {t1_w:.1f}]s, Δψ={dpsi:.1f}°)\n"
        f"Shared Vertical Cells: {c_stat} (static) -> {c_turn} (cross-time)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.legend(markerscale=8, loc="upper right")

    # Zoomed region on best shared wall: Calibrated vs Common-Mode IMU Bias
    if bp is not None:
        z_cx, z_cy = float(bp["center"][0]), float(bp["center"][1])
    else:
        z_cx, z_cy = 0.0, 10.0
    half_w = 5.5
    cal_sig = tm["calibrated"]["median_wall_thickness_cm"]
    cal_gap = tm["calibrated"]["median_ab_wall_gap_cm"]
    cm_sig = tm["ablation_common_mode_imu_bias"]["median_wall_thickness_cm"]
    cm_rat = tm["common_mode_blur_inflation_ratio"]
    for sub_ax, pts1, pts2, title_txt in (
        (
            axes2[0, 1],
            cal_a,
            cal_b,
            "Zoomed Wall — Calibrated Extrinsics\n"
            f"Single-Sensor Wall σ = {cal_sig:.2f} cm | "
            f"Macro A-B Gap = {cal_gap:.1f} cm",
        ),
        (
            axes2[1, 0],
            cm_a,
            cm_b,
            "Zoomed Wall — Common-Mode IMU Bias (+2.5°/+45cm/+45ms)\n"
            f"(Invisible in Static PCD! Wall σ = {cm_sig:.2f} cm, {cm_rat:.2f}x blur)",
        ),
    ):
        m1 = (
            (pts1[:, 2] > zg + 0.4)
            & (np.abs(pts1[:, 0] - z_cx) <= half_w)
            & (np.abs(pts1[:, 1] - z_cy) <= half_w)
        )
        m2 = (
            (pts2[:, 2] > zg + 0.4)
            & (np.abs(pts2[:, 0] - z_cx) <= half_w)
            & (np.abs(pts2[:, 1] - z_cy) <= half_w)
        )
        sub_ax.scatter(
            pts1[m1, 0], pts1[m1, 1], s=2.2, c=col_a, alpha=0.6, label=label_a
        )
        sub_ax.scatter(
            pts2[m2, 0], pts2[m2, 1], s=2.2, c=col_b, alpha=0.6, label=label_b
        )
        sub_ax.set_aspect("equal")
        sub_ax.grid(True, alpha=0.3)
        sub_ax.set_xlabel("X (m)")
        sub_ax.set_ylabel("Y (m)")
        sub_ax.set_title(title_txt, fontsize=10.0, fontweight="bold")
        sub_ax.legend(markerscale=4, loc="upper right")

    axb = axes2[1, 1]
    labels_bar = [
        "Calibrated\n(R_LI, T_LI, τ)",
        "Zero Lever/Lag\n(T_LI=0, τ=0)",
        "Common-Mode Bias\n(Gauge Blind Spot)",
    ]
    med_vals = [
        tm["calibrated"]["median_wall_thickness_cm"],
        tm["ablation_zero_lever_zero_lag"]["median_wall_thickness_cm"],
        tm["ablation_common_mode_imu_bias"]["median_wall_thickness_cm"],
    ]
    p90_vals = [
        tm["calibrated"]["p90_wall_thickness_cm"],
        tm["ablation_zero_lever_zero_lag"]["p90_wall_thickness_cm"],
        tm["ablation_common_mode_imu_bias"]["p90_wall_thickness_cm"],
    ]
    x_pos = np.arange(3)
    axb.bar(
        x_pos - 0.17, med_vals, width=0.32, color="#2b8cbe", label="Median Wall σ (cm)"
    )
    axb.bar(
        x_pos + 0.17, p90_vals, width=0.32, color="#e34a33", label="P90 Wall σ (cm)"
    )
    for idx_b, (mv, pv) in enumerate(zip(med_vals, p90_vals)):
        axb.text(idx_b - 0.17, mv + 0.25, f"{mv:.1f}", ha="center", fontsize=8.5)
        axb.text(idx_b + 0.17, pv + 0.25, f"{pv:.1f}", ha="center", fontsize=8.5)
    axb.set_xticks(x_pos)
    axb.set_xticklabels(labels_bar, fontsize=9.0)
    axb.set_ylabel("Voxel Plane Point-to-Plane Thickness (cm)")
    axb.set_title(
        "Turn-Submap Wall Sharpness Breaks Gauge Blind Spot\n"
        "(Lower = Crisper 3D Walls)",
        fontsize=10.5,
        fontweight="bold",
    )
    axb.grid(True, axis="y", alpha=0.3)
    axb.legend(loc="upper left")
    fig2.tight_layout()
    f2_path = diag_dir / "fig2_turn_imu_submap_sharpness.png"
    fig2.savefig(f2_path)
    plt.close(fig2)
    written.append(str(f2_path.resolve()))

    # Fig 3: 3-Curve Dynamics + Lever-Arm Cancellation + Gravity + Scorecard
    fig3, axes3 = plt.subplots(2, 2, figsize=(15.5, 11.5), dpi=160)
    dm = report["pillars"]["three_curve_dynamics"]
    tr = dyn_plot["t_rel"]
    ax = axes3[0, 0]
    ax.plot(tr, dyn_plot["w_imu"][:, 2], "k-", lw=2.0, label="IMU ω_z (measured)")
    ax.plot(
        tr,
        dyn_plot["w_a"][:, 2],
        color=col_a,
        ls="--",
        lw=1.5,
        label=f"{label_a} -> IMU",
    )
    ax.plot(
        tr,
        dyn_plot["w_b"][:, 2],
        color=col_b,
        ls=":",
        lw=1.7,
        label=f"{label_b} -> IMU",
    )
    ax.set_xlabel("Window Time (s)")
    ax.set_ylabel("Yaw Angular Velocity ω_z (rad/s)")
    ax.set_title(
        f"Three-Curves-in-One Yaw Rate Alignment\n"
        f"RMSE(A,IMU)={dm['gyro_a_to_imu_rmse_rad_s']:.4f} | "
        f"RMSE(B,IMU)={dm['gyro_b_to_imu_rmse_rad_s']:.4f} rad/s",
        fontsize=10.3,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    axa = axes3[0, 1]
    axa.plot(
        tr,
        dyn_plot["a_a_uncomp"][:, 0] - dyn_plot["a_b_uncomp"][:, 0],
        color="#999999",
        ls="--",
        lw=1.5,
        label="Uncompensated Δa_x (A - B, opposite corners diverge)",
    )
    axa.plot(
        tr,
        dyn_plot["a_a_comp"][:, 0] - dyn_plot["a_b_comp"][:, 0],
        color="#1b7837",
        lw=1.9,
        label="Lever-Arm Compensated Δa_x at IMU Origin",
    )
    axa.axhline(0.0, color="k", lw=0.8)
    axa.set_xlabel("Window Time (s)")
    axa.set_ylabel("Cross-Corner Lateral Acc Discrepancy Δa_x (m/s²)")
    uncomp_g = dm["active_turn_ab_planar_acc_gap_uncompensated_m_s2"]
    comp_g = dm["active_turn_ab_planar_acc_gap_compensated_m_s2"]
    red_pct = dm["lever_arm_ab_acc_gap_reduction_pct"]
    axa.set_title(
        "Opposite-Corner Tangential Lever-Arm Cancellation\n"
        f"Active-Turn A-B Gap: {uncomp_g:.3f} -> {comp_g:.3f} m/s² (-{red_pct:.1f}%)",
        fontsize=10.2,
        fontweight="bold",
    )
    axa.grid(True, alpha=0.3)
    axa.legend(loc="best", fontsize=8.5)

    axv = axes3[1, 0]
    gm = report["pillars"]["gravity_ground_alignment"]
    if grav_samples:
        hdg = [s["heading_yaw_deg"] for s in grav_samples]
        ab_n = [s["inter_lidar_normal_deg"] for s in grav_samples]
        ag_n = [s["a_to_imu_gravity_deg"] for s in grav_samples]
        bg_n = [s["b_to_imu_gravity_deg"] for s in grav_samples]
        axv.scatter(
            hdg, ab_n, c="#2ca25f", s=42, label="Inter-LiDAR Ground ∠ (slope-free)"
        )
        axv.scatter(
            hdg, ag_n, c=col_a, marker="^", s=38, label=f"{label_a} Ground vs IMU g"
        )
        axv.scatter(
            hdg, bg_n, c=col_b, marker="s", s=38, label=f"{label_b} Ground vs IMU g"
        )
    axv.set_xlabel("Vehicle Heading Yaw (deg)")
    axv.set_ylabel("Normal Alignment Angle (deg)")
    n_win = gm["evaluated_windows"]
    axv.set_title(
        f"Multi-Heading Ground Normal vs IMU Gravity ({n_win} windows)\n"
        f"Inter-LiDAR Med={gm['inter_lidar_normal_median_deg']:.2f}° | "
        f"Slope-Cancelled IMU Bias={gm['slope_cancelled_tilt_bias_norm_deg']:.2f}°",
        fontsize=10.0,
        fontweight="bold",
    )
    axv.grid(True, alpha=0.3)
    axv.legend(loc="best", fontsize=8.5)

    axt = axes3[1, 1]
    axt.axis("off")
    l2l = report["pillars"]["l2l_cycle_consistency"]
    sem_l2l = l2l.get("semantic_l2l", {})
    if (
        l2l.get("available")
        and sem_l2l.get("available")
        and not sem_l2l.get("corridor_degenerate", True)
    ):
        l2l_str = (
            f"ΔR={sem_l2l['loop_rotation_residual_deg']:.2f}°, "
            f"Δt={sem_l2l['loop_translation_residual_m']*100:.1f} cm "
            f"(sem cond={sem_l2l['translation_condition_number']:.1f})"
        )
    elif l2l.get("available"):
        l2l_str = (
            f"ΔR={l2l['loop_rotation_residual_deg']:.2f}°, "
            f"Δt={l2l['loop_translation_residual_m']*100:.1f} cm "
            f"(gicp cond={l2l['info_matrix_condition_number']:.1f})"
        )
    else:
        l2l_str = "skipped"
    dz_g = gp["signed_ground_z_gap_cm"]
    mw_g = st_m["max_abs_wall_gap_cm"]
    max_w_rmse = max(dm["gyro_a_to_imu_rmse_rad_s"], dm["gyro_b_to_imu_rmse_rad_s"])
    table_rows = [
        [
            "1. Static Dual-PCD & Slices",
            f"Δz={dz_g:+.2f} cm, max|Δn_wall|={mw_g:.1f} cm",
            "PASS" if report["checks"]["static_overlay_consistent"] else "REVIEW",
        ],
        [
            "2. Turn Pure-IMU Submap",
            f"σ_single={cal_sig:.2f} cm ({cm_rat:.2f}x CM), A-B={cal_gap:.1f} cm",
            "PASS" if report["checks"]["turn_submap_sharp"] else "REVIEW",
        ],
        [
            "3. 2xL2I vs Direct L2L Loop",
            l2l_str,
            "PASS" if report["checks"]["l2l_cycle_consistent"] else "REVIEW",
        ],
        [
            "4. Gravity vs Ground Normals",
            f"A-B={gm['inter_lidar_normal_median_deg']:.2f}°, "
            f"IMU bias={gm['slope_cancelled_tilt_bias_norm_deg']:.2f}°",
            "PASS" if report["checks"]["gravity_alignment_consistent"] else "REVIEW",
        ],
        [
            "5. 3-Curve Dynamics & Lever",
            f"ω RMSE={max_w_rmse:.4f} rad/s, Δa -{red_pct:.1f}%",
            "PASS" if report["checks"]["three_curve_dynamics_consistent"] else "REVIEW",
        ],
        [
            "DUAL-L2I VERDICT",
            f"ΔYaw={report['dual_geometry']['yaw_separation_deg']:.2f}°, "
            f"Δτ={report['dual_geometry']['time_offset_diff_ms']:.2f} ms",
            report["verdict"].upper(),
        ],
    ]
    tbl = axt.table(
        cellText=table_rows,
        colLabels=["Diagnostic Pillar", "Key Measured Evidence", "Status"],
        colWidths=[0.33, 0.49, 0.16],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.7)
    tbl.scale(1.0, 1.85)
    for (r_i, c_i), cell in tbl.get_celld().items():
        if r_i == 0:
            cell.set_facecolor("#223344")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r_i == len(table_rows):
            cell.set_facecolor("#e5f5e0")
            cell.get_text().set_weight("bold")
    axt.set_title(
        "5-Pillar Dual LiDAR-to-IMU Cross-Diagnostic Scorecard",
        fontsize=10.5,
        fontweight="bold",
    )
    fig3.tight_layout()
    f3_path = diag_dir / "fig3_dynamics_gravity_loop_closure.png"
    fig3.savefig(f3_path)
    plt.close(fig3)
    written.append(str(f3_path.resolve()))
    return written


def _apply_extrinsic_override(
    result: dict[str, Any],
    extrinsic_yaml: Path | None,
) -> dict[str, Any]:
    if extrinsic_yaml is None:
        return result
    mat, _, _, _, payload = load_extrinsics_file(str(extrinsic_yaml))
    R_LI = _project_so3(np.asarray(mat[:3, :3], dtype=np.float64))
    T_LI = np.asarray(mat[:3, 3], dtype=np.float64)
    T_IL = -(R_LI.T @ T_LI)
    euler_xyz = Rot.from_matrix(R_LI).as_euler("xyz", degrees=True).tolist()
    yaw_deg = float(math.degrees(math.atan2(R_LI[1, 0], R_LI[0, 0])))
    pitch_deg = float(math.degrees(math.asin(max(-1.0, min(1.0, float(-R_LI[2, 0]))))))
    roll_deg = float(math.degrees(math.atan2(R_LI[2, 1], R_LI[2, 2])))
    updated = dict(result)
    updated["rotation_LI_matrix"] = R_LI.tolist()
    updated["rotation_xyz_deg"] = [float(v) for v in euler_xyz]
    updated["zyx_rpy_deg"] = [roll_deg, pitch_deg, yaw_deg]
    updated["translation_LI_m"] = [float(v) for v in T_LI]
    updated["translation_IL_m"] = [float(v) for v in T_IL]
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if isinstance(meta, dict) and "time_offset_s" in meta:
        updated["time_offset_s"] = float(meta["time_offset_s"])
    updated["override_extrinsic_yaml"] = str(Path(extrinsic_yaml).resolve())
    return updated


def evaluate_dual_lidar2imu_diagnostics(
    *,
    dataset_a: Path,
    run_dir_a: Path,
    dataset_b: Path,
    run_dir_b: Path,
    output_dir: Path,
    label_a: str = "left_front",
    label_b: str = "right_back",
    static_time_s: float | None = None,
    turn_time_s: float | None = None,
    extrinsic_a_yaml: Path | None = None,
    extrinsic_b_yaml: Path | None = None,
    thresholds: DualDiagnosticThresholds = DualDiagnosticThresholds(),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir_a = Path(run_dir_a)
    run_dir_b = Path(run_dir_b)

    access_a = FastCanonicalScanAccess(Path(dataset_a))
    access_b = FastCanonicalScanAccess(Path(dataset_b))
    result_a = _apply_extrinsic_override(
        parse_detailed_gril_result(run_dir_a / "GRIL_Calib_result.txt"),
        extrinsic_a_yaml,
    )
    result_b = _apply_extrinsic_override(
        parse_detailed_gril_result(run_dir_b / "GRIL_Calib_result.txt"),
        extrinsic_b_yaml,
    )
    batch_a = parse_batch_trace_arrays(run_dir_a / "GRIL_batch_trace_v1.txt")
    batch_b = parse_batch_trace_arrays(run_dir_b / "GRIL_batch_trace_v1.txt")

    T_I_A = _extrinsic_matrix_LI(result_a)
    T_I_B = _extrinsic_matrix_LI(result_b)
    T_A_B = np.linalg.inv(T_I_A) @ T_I_B

    ext_dir = output_dir / "extrinsics"
    ext_dir.mkdir(parents=True, exist_ok=True)
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
    ext_a_path = ext_dir / extrinsics_filename(frame_imu, frame_a)
    ext_b_path = ext_dir / extrinsics_filename(frame_imu, frame_b)
    ext_ab_path = ext_dir / extrinsics_filename(frame_a, frame_b)
    save_extrinsics_yaml(
        str(ext_a_path),
        parent_frame=frame_imu,
        child_frame=frame_a,
        matrix=T_I_A,
        metadata={"time_offset_s": result_a["time_offset_s"], "source": str(run_dir_a)},
    )
    save_extrinsics_yaml(
        str(ext_b_path),
        parent_frame=frame_imu,
        child_frame=frame_b,
        matrix=T_I_B,
        metadata={"time_offset_s": result_b["time_offset_s"], "source": str(run_dir_b)},
    )
    save_extrinsics_yaml(
        str(ext_ab_path),
        parent_frame=frame_a,
        child_frame=frame_b,
        matrix=T_A_B,
        metadata={
            "derivation": "inv(T_imu_a) @ T_imu_b",
            "relative_time_offset_s": float(
                result_a["time_offset_s"] - result_b["time_offset_s"]
            ),
        },
    )

    static_metrics, static_plot = evaluate_static_dual_overlay(
        access_a,
        access_b,
        result_a,
        result_b,
        batch_a,
        output_dir,
        static_time_s=static_time_s,
    )
    turn_metrics, turn_plot = evaluate_turn_imu_submap(
        access_a,
        access_b,
        result_a,
        result_b,
        batch_a,
        output_dir,
        turn_time_s=turn_time_s,
    )
    l2l_metrics = evaluate_l2l_cycle_consistency(
        access_a,
        access_b,
        result_a,
        result_b,
        center_scan_a=static_metrics["scan_index_a"],
        center_scan_b=static_metrics["scan_index_b"],
        output_dir=output_dir,
    )
    grav_metrics, grav_samples = evaluate_gravity_ground_alignment(
        access_a, access_b, result_a, result_b, batch_a
    )
    dyn_metrics, dyn_plot = evaluate_three_curve_dynamics(
        result_a,
        result_b,
        batch_a,
        batch_b,
        focus_center_s=(
            float(access_a.scan_t_s[0])
            + 0.5
            * (turn_metrics["window_rel_start_s"] + turn_metrics["window_rel_end_s"])
        ),
    )

    yaw_a = float(result_a["zyx_rpy_deg"][2])
    yaw_b = float(result_b["zyx_rpy_deg"][2])
    dyaw = float((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)
    if dyaw < 0.0:
        dyaw += 360.0
    dtau_ms = abs(float(result_a["time_offset_s"] - result_b["time_offset_s"])) * 1e3
    dz_cm = abs(float(T_I_A[2, 3] - T_I_B[2, 3])) * 100.0

    checks = {
        "static_overlay_consistent": bool(
            static_metrics["ground_plane"]["abs_ground_z_gap_cm"]
            <= thresholds.max_static_ground_z_gap_cm
            and static_metrics["max_abs_wall_gap_cm"]
            <= thresholds.max_static_wall_gap_cm
        ),
        "turn_submap_sharp": bool(
            turn_metrics["calibrated"]["median_wall_thickness_cm"]
            <= thresholds.max_turn_wall_thickness_cm
            and turn_metrics["calibrated"]["median_ab_wall_gap_cm"]
            <= thresholds.max_turn_wall_gap_cm
            and (
                turn_metrics["integrated_imu_rotation_deg"] < 15.0
                or turn_metrics["common_mode_blur_inflation_ratio"]
                >= thresholds.min_common_bias_blur_ratio
            )
        ),
        "turn_single_sensor_imu_deskew_sharp": bool(
            turn_metrics["calibrated"]["median_wall_thickness_cm"]
            <= thresholds.max_turn_wall_thickness_cm
            and (
                turn_metrics["integrated_imu_rotation_deg"] < 15.0
                or turn_metrics["common_mode_blur_inflation_ratio"]
                >= thresholds.min_common_bias_blur_ratio
            )
        ),
        "l2l_cycle_consistent": bool(
            not l2l_metrics.get("available")
            or (
                (
                    l2l_metrics["semantic_l2l"]["loop_rotation_residual_deg"]
                    <= thresholds.max_l2l_loop_rot_deg
                    and l2l_metrics["semantic_l2l"]["loop_translation_residual_m"]
                    <= thresholds.max_l2l_loop_trans_m
                )
                if (
                    l2l_metrics.get("semantic_l2l", {}).get("available")
                    and not l2l_metrics["semantic_l2l"].get("corridor_degenerate", True)
                )
                else (
                    l2l_metrics["loop_rotation_residual_deg"]
                    <= thresholds.max_l2l_loop_rot_deg
                    and l2l_metrics["loop_translation_residual_m"]
                    <= thresholds.max_l2l_loop_trans_m
                )
            )
        ),
        "gravity_alignment_consistent": bool(
            grav_metrics["inter_lidar_normal_median_deg"]
            <= thresholds.max_inter_lidar_ground_tilt_deg
            and grav_metrics["slope_cancelled_tilt_bias_norm_deg"]
            <= thresholds.max_imu_gravity_tilt_bias_deg
        ),
        "three_curve_dynamics_consistent": bool(
            max(
                dyn_metrics["gyro_a_to_imu_rmse_rad_s"],
                dyn_metrics["gyro_b_to_imu_rmse_rad_s"],
            )
            <= thresholds.max_gyro_cross_rmse_rad_s
            and dyn_metrics["lever_arm_ab_acc_gap_reduction_pct"] >= 0.0
        ),
    }

    verdict = "accepted" if all(checks.values()) else "review_only"
    report: dict[str, Any] = {
        "schema_version": 1,
        "module": "lidar2imu_dual_sensor_diagnostics",
        "verdict": verdict,
        "sensors": {
            "sensor_a": {
                "label": label_a,
                "frame_id": frame_a,
                "run_dir": str(run_dir_a.resolve()),
                "extrinsic_yaml": str(ext_a_path.resolve()),
                "result": result_a,
            },
            "sensor_b": {
                "label": label_b,
                "frame_id": frame_b,
                "run_dir": str(run_dir_b.resolve()),
                "extrinsic_yaml": str(ext_b_path.resolve()),
                "result": result_b,
            },
        },
        "dual_geometry": {
            "imu_frame": frame_imu,
            "induced_extrinsic_yaml": str(ext_ab_path.resolve()),
            "yaw_separation_deg": dyaw,
            "height_diff_cm": dz_cm,
            "baseline_distance_m": float(np.linalg.norm(T_I_A[:3, 3] - T_I_B[:3, 3])),
            "time_offset_diff_ms": dtau_ms,
        },
        "checks": checks,
        "pillars": {
            "static_dual_overlay": static_metrics,
            "turn_imu_submap": turn_metrics,
            "l2l_cycle_consistency": l2l_metrics,
            "gravity_ground_alignment": grav_metrics,
            "three_curve_dynamics": dyn_metrics,
        },
    }

    figures = _render_diagnostic_figures(
        label_a=label_a,
        label_b=label_b,
        report=report,
        static_plot=static_plot,
        turn_plot=turn_plot,
        grav_samples=grav_samples,
        dyn_plot=dyn_plot,
        diag_dir=output_dir / "diagnostics",
    )
    report["artifacts"] = {
        "pcd_files": [
            static_metrics["colored_pcd_path"],
            turn_metrics["calibrated_submap_pcd_path"],
            turn_metrics["common_mode_control_pcd_path"],
        ],
        "diagnostic_figures": figures,
        "extrinsic_files": [
            str(ext_a_path.resolve()),
            str(ext_b_path.resolve()),
            str(ext_ab_path.resolve()),
        ],
    }

    report_path = output_dir / "dual_diagnostics.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    return report
