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

"""Upgraded L2L registration with 4-channel primitives, calibrated reflectivity,
self-occlusion-aware 3D shared-FoV, and X-ICP localizability diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from lidar2lidar.fov import AngularSupport, estimate_angular_support


@dataclass(frozen=True)
class ReflectivityConfig:
    range_power: float = 1.5
    incidence_power: float = 0.5
    min_incidence_cos: float = 0.2588  # cos(75 deg)
    retro_z_threshold: float = 1.8
    max_retro_points: int = 260


@dataclass(frozen=True)
class PrimitiveExtractionConfig:
    min_range_m: float = 2.0
    max_range_m: float = 48.0
    voxel_size_m: float = 0.50
    min_voxel_points: int = 5
    max_sphericity: float = 0.28
    min_planarity: float = 0.42
    min_linearity: float = 0.55
    ground_normal_min_z: float = 0.90
    ground_inlier_dist_m: float = 0.08
    max_ground_points: int = 1600
    max_facade_nz: float = 0.35
    min_pole_vz: float = 0.82
    max_facade_points: int = 2400
    max_pole_points: int = 600


@dataclass(frozen=True)
class VisibilityConfig:
    ego_bbox_min_lidar: tuple[float, float, float] | None = None
    ego_bbox_max_lidar: tuple[float, float, float] | None = None
    min_elevation_margin_deg: float = 4.0
    min_azimuth_margin_deg: float = 4.0
    normal_covisibility_cos: float = 0.819  # cos(35 deg)
    reciprocal_check: bool = True


@dataclass(frozen=True)
class SemanticRegistrationConfig:
    correspondence_schedule_m: tuple[float, ...] = (1.20, 0.55, 0.25, 0.16)
    uncertainty_schedule: tuple[tuple[float, float], ...] = (
        (8.0, 1.20),
        (4.0, 0.50),
        (2.0, 0.22),
        (1.2, 0.12),
    )
    facade_weight: float = 1.0
    ground_weight: float = 0.45
    pole_weight: float = 1.35
    intensity_weight: float = 0.55
    huber_scale_m: float = 0.08
    noise_floor_m: float = 0.025
    corridor_cond_threshold: float = 150.0
    min_trans_info_threshold: float = 25.0
    enable_coarse_fallback: bool = True
    coarse_yaw_range_deg: float = 30.0
    coarse_yaw_step_deg: float = 3.0


@dataclass
class LidarFrameData:
    points: np.ndarray
    intensity: np.ndarray | None = None
    ring: np.ndarray | None = None
    normals: np.ndarray | None = None


@dataclass
class GeometricPrimitives:
    ground_points: np.ndarray
    ground_normals: np.ndarray
    facade_points: np.ndarray
    facade_normals: np.ndarray
    pole_points: np.ndarray
    pole_directions: np.ndarray
    intensity_points: np.ndarray
    intensity_z: np.ndarray
    intensity_normals: np.ndarray
    all_filtered_points: np.ndarray
    stats: dict[str, int] = field(default_factory=dict)


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _tangent_basis(unit_dir: np.ndarray) -> np.ndarray:
    u = np.asarray(unit_dir, dtype=np.float64)
    u = u / max(1e-12, float(np.linalg.norm(u)))
    ref = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(u[0]) < 0.8
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    t1 = np.cross(u, ref)
    t1 /= max(1e-12, float(np.linalg.norm(t1)))
    t2 = np.cross(u, t1)
    t2 /= max(1e-12, float(np.linalg.norm(t2)))
    return np.vstack([t1, t2])  # (2, 3)


def normalize_reflectivity(
    points: np.ndarray,
    intensity: np.ndarray | None,
    normals: np.ndarray | None = None,
    config: ReflectivityConfig = ReflectivityConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Range/incidence-compensated and sensor-internal robust-normalized intensity."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if intensity is None or len(intensity) != len(pts) or len(pts) == 0:
        zeros = np.zeros(len(pts), dtype=np.float64)
        return zeros, zeros

    raw_i = np.asarray(intensity, dtype=np.float64).reshape(-1)
    valid = np.isfinite(raw_i) & (raw_i > 0.0)
    if int(np.sum(valid)) < 16 or float(np.std(raw_i[valid])) < 1e-9:
        zeros = np.zeros(len(pts), dtype=np.float64)
        return zeros, zeros

    ranges = np.maximum(np.linalg.norm(pts, axis=1), 1.0)
    if normals is not None and len(normals) == len(pts):
        view_dir = pts / ranges[:, None]
        cos_inc = np.maximum(
            np.abs(np.sum(view_dir * normals, axis=1)),
            config.min_incidence_cos,
        )
    else:
        cos_inc = np.ones(len(pts), dtype=np.float64)

    compensated = (
        raw_i
        * np.power(ranges, config.range_power)
        / np.power(cos_inc, config.incidence_power)
    )
    log_comp = np.log1p(np.maximum(compensated[valid], 0.0))
    med = float(np.median(log_comp))
    q25, q75 = np.percentile(log_comp, [25.0, 75.0])
    scale = float(q75 - q25)
    if scale < 1e-6:
        scale = float(np.std(log_comp))
    if scale < 1e-6:
        zeros = np.zeros(len(pts), dtype=np.float64)
        return zeros, zeros

    z_scores = np.zeros(len(pts), dtype=np.float64)
    z_scores[valid] = np.clip(
        (np.log1p(np.maximum(compensated[valid], 0.0)) - med) / (0.7413 * scale),
        -5.0,
        8.0,
    )
    return z_scores, compensated


def _uniform_cap(arr_idx: np.ndarray, max_count: int) -> np.ndarray:
    if len(arr_idx) <= max_count:
        return arr_idx
    sel = np.linspace(0, len(arr_idx) - 1, num=max_count, dtype=int)
    return arr_idx[sel]


def _fit_dominant_ground(
    pts: np.ndarray,
    *,
    normal_min_z: float = 0.90,
    dist_thresh_m: float = 0.08,
    rng_seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) < 20:
        return np.zeros(len(pts), dtype=bool), np.array([0.0, 0.0, 1.0])

    z_q35 = float(np.percentile(pts[:, 2], 35))
    cand_idx = np.flatnonzero(pts[:, 2] <= z_q35 + 0.35)
    if len(cand_idx) < 20:
        cand_idx = np.arange(len(pts))
    cand = pts[cand_idx]

    rng = np.random.default_rng(rng_seed)
    best_inliers: np.ndarray | None = None
    best_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_cand = len(cand)
    for _ in range(70):
        idx = rng.choice(n_cand, size=3, replace=False)
        n = np.cross(cand[idx[1]] - cand[idx[0]], cand[idx[2]] - cand[idx[0]])
        norm_v = float(np.linalg.norm(n))
        if norm_v < 1e-8:
            continue
        n /= norm_v
        if n[2] < 0.0:
            n = -n
        if n[2] < normal_min_z:
            continue
        inliers = np.abs((cand - cand[idx[0]]) @ n) <= dist_thresh_m
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
            best_n = n

    mask = np.zeros(len(pts), dtype=bool)
    if best_inliers is None or int(best_inliers.sum()) < 15:
        return mask, best_n

    g_pts = cand[best_inliers]
    centroid = g_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(g_pts - centroid, full_matrices=False)
    refined_n = vt[-1]
    if refined_n[2] < 0.0:
        refined_n = -refined_n
    if refined_n[2] < normal_min_z:
        return mask, best_n
    mask = np.abs((pts - centroid) @ refined_n) <= dist_thresh_m
    return mask, refined_n


def extract_geometric_primitives(
    frame: LidarFrameData,
    *,
    prim_cfg: PrimitiveExtractionConfig = PrimitiveExtractionConfig(),
    refl_cfg: ReflectivityConfig = ReflectivityConfig(),
) -> GeometricPrimitives:
    """Decompose raw scan/submap into ground, structural facade, pole/edge, and
    calibrated reflectivity channels while rejecting high-sphericity clutter.
    """
    pts = np.asarray(frame.points, dtype=np.float64).reshape(-1, 3)
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    intensity = (
        np.asarray(frame.intensity, dtype=np.float64).reshape(-1)[finite_mask]
        if frame.intensity is not None and len(frame.intensity) == len(finite_mask)
        else None
    )

    r_xy = np.linalg.norm(pts[:, :2], axis=1)
    range_mask = (r_xy >= prim_cfg.min_range_m) & (r_xy <= prim_cfg.max_range_m)
    pts = pts[range_mask]
    if intensity is not None:
        intensity = intensity[range_mask]

    if len(pts) == 0:
        empty3 = np.zeros((0, 3), dtype=np.float64)
        empty1 = np.zeros((0,), dtype=np.float64)
        return GeometricPrimitives(
            ground_points=empty3,
            ground_normals=empty3,
            facade_points=empty3,
            facade_normals=empty3,
            pole_points=empty3,
            pole_directions=empty3,
            intensity_points=empty3,
            intensity_z=empty1,
            intensity_normals=empty3,
            all_filtered_points=empty3,
            stats={
                "input_points": 0,
                "ground": 0,
                "facade": 0,
                "pole": 0,
                "intensity": 0,
                "scatter_rejected_voxels": 0,
            },
        )

    ground_mask, ground_normal = _fit_dominant_ground(
        pts,
        normal_min_z=prim_cfg.ground_normal_min_z,
        dist_thresh_m=prim_cfg.ground_inlier_dist_m,
    )
    g_idx = _uniform_cap(np.flatnonzero(ground_mask), prim_cfg.max_ground_points)
    ground_pts = pts[g_idx]
    ground_normals = np.tile(ground_normal, (len(ground_pts), 1))

    # Assign per-point normals from ground or local voxel PCA for intensity incidence
    point_normals = np.tile(ground_normal, (len(pts), 1))

    non_ground_idx = np.flatnonzero(~ground_mask)
    non_ground_pts = pts[non_ground_idx]

    facade_pts_list: list[np.ndarray] = []
    facade_normals_list: list[np.ndarray] = []
    pole_pts_list: list[np.ndarray] = []
    pole_dirs_list: list[np.ndarray] = []
    scatter_rejected = 0

    if len(non_ground_pts) >= prim_cfg.min_voxel_points:
        vox_keys = np.floor(non_ground_pts / prim_cfg.voxel_size_m).astype(np.int64)
        order = np.lexsort((vox_keys[:, 2], vox_keys[:, 1], vox_keys[:, 0]))
        s_keys = vox_keys[order]
        s_pts = non_ground_pts[order]
        s_orig_idx = non_ground_idx[order]
        diffs = np.any(np.diff(s_keys, axis=0) != 0, axis=1)
        bounds = np.concatenate([[0], np.where(diffs)[0] + 1, [len(s_pts)]])

        for s, e in zip(bounds[:-1], bounds[1:]):
            cnt = int(e - s)
            if cnt < prim_cfg.min_voxel_points:
                continue
            block = s_pts[s:e]
            mu = block.mean(axis=0)
            centered = block - mu
            cov = (centered.T @ centered) / float(cnt)
            eigvals, eigvecs = np.linalg.eigh(cov)
            l3, l2, l1 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
            if l1 < 1e-6:
                continue
            sphericity = l3 / l1
            if sphericity > prim_cfg.max_sphericity:
                scatter_rejected += 1
                continue

            planarity = (l2 - l3) / l1
            linearity = (l1 - l2) / l1
            normal = eigvecs[:, 0].copy()
            # Orient surface normal toward sensor origin (0, 0, 0) for same-side gate
            if float(normal @ mu) > 0.0:
                normal = -normal
            point_normals[s_orig_idx[s:e]] = normal

            axis_dir = eigvecs[:, 2].copy()
            if axis_dir[2] < 0.0:
                axis_dir = -axis_dir

            if (
                planarity >= prim_cfg.min_planarity
                and abs(float(normal[2])) <= prim_cfg.max_facade_nz
            ):
                facade_pts_list.append(mu)
                facade_normals_list.append(normal)
            elif (
                linearity >= prim_cfg.min_linearity
                and abs(float(axis_dir[2])) >= prim_cfg.min_pole_vz
            ):
                pole_pts_list.append(mu)
                pole_dirs_list.append(axis_dir)

    facade_pts = (
        np.asarray(facade_pts_list, dtype=np.float64)
        if facade_pts_list
        else np.zeros((0, 3), dtype=np.float64)
    )
    facade_normals = (
        np.asarray(facade_normals_list, dtype=np.float64)
        if facade_normals_list
        else np.zeros((0, 3), dtype=np.float64)
    )
    if len(facade_pts) > prim_cfg.max_facade_points:
        keep_f = _uniform_cap(np.arange(len(facade_pts)), prim_cfg.max_facade_points)
        facade_pts = facade_pts[keep_f]
        facade_normals = facade_normals[keep_f]

    pole_pts = (
        np.asarray(pole_pts_list, dtype=np.float64)
        if pole_pts_list
        else np.zeros((0, 3), dtype=np.float64)
    )
    pole_dirs = (
        np.asarray(pole_dirs_list, dtype=np.float64)
        if pole_dirs_list
        else np.zeros((0, 3), dtype=np.float64)
    )
    if len(pole_pts) > prim_cfg.max_pole_points:
        keep_p = _uniform_cap(np.arange(len(pole_pts)), prim_cfg.max_pole_points)
        pole_pts = pole_pts[keep_p]
        pole_dirs = pole_dirs[keep_p]

    z_intensity, _ = normalize_reflectivity(pts, intensity, point_normals, refl_cfg)
    retro_idx = np.flatnonzero(z_intensity >= refl_cfg.retro_z_threshold)
    if len(retro_idx) > refl_cfg.max_retro_points:
        top_order = np.argsort(z_intensity[retro_idx])[::-1]
        retro_idx = retro_idx[top_order[: refl_cfg.max_retro_points]]
    int_pts = pts[retro_idx]
    int_z = z_intensity[retro_idx]
    int_normals = point_normals[retro_idx]

    return GeometricPrimitives(
        ground_points=ground_pts,
        ground_normals=ground_normals,
        facade_points=facade_pts,
        facade_normals=facade_normals,
        pole_points=pole_pts,
        pole_directions=pole_dirs,
        intensity_points=int_pts,
        intensity_z=int_z,
        intensity_normals=int_normals,
        all_filtered_points=pts,
        stats={
            "input_points": int(len(pts)),
            "ground": int(len(ground_pts)),
            "facade": int(len(facade_pts)),
            "pole": int(len(pole_pts)),
            "intensity": int(len(int_pts)),
            "scatter_rejected_voxels": int(scatter_rejected),
        },
    )


def ray_intersects_aabb_mask(
    points: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    *,
    t_min_frac: float = 0.04,
    t_max_frac: float = 0.96,
) -> np.ndarray:
    """Return True for rays from origin (0,0,0) to `points` blocked by an AABB."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    bmin = np.asarray(box_min, dtype=np.float64)
    bmax = np.asarray(box_max, dtype=np.float64)
    t_enter = np.full(len(pts), t_min_frac, dtype=np.float64)
    t_exit = np.full(len(pts), t_max_frac, dtype=np.float64)

    for ax in range(3):
        d = pts[:, ax]
        parallel = np.abs(d) < 1e-12
        outside_parallel = parallel & ((0.0 < bmin[ax]) | (0.0 > bmax[ax]))
        t_enter[outside_parallel] = 2.0
        t_exit[outside_parallel] = -1.0

        nz = ~parallel
        inv_d = np.where(nz, 1.0 / np.where(nz, d, 1.0), 0.0)
        t0 = bmin[ax] * inv_d
        t1 = bmax[ax] * inv_d
        t_low = np.minimum(t0, t1)
        t_high = np.maximum(t0, t1)
        t_enter = np.where(nz, np.maximum(t_enter, t_low), t_enter)
        t_exit = np.where(nz, np.minimum(t_exit, t_high), t_exit)

    return t_exit >= t_enter


@dataclass(frozen=True)
class FrustumSupport3D:
    azimuth: AngularSupport
    elev_min_deg: float
    elev_max_deg: float
    range_min_m: float
    range_max_m: float


def estimate_3d_frustum_support(points: np.ndarray) -> FrustumSupport3D:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    az = estimate_angular_support([pts], coverage_ratio=0.995, margin_deg=1.5)
    r = np.linalg.norm(pts, axis=1)
    elev = np.degrees(np.arcsin(np.clip(pts[:, 2] / np.maximum(r, 1e-6), -1.0, 1.0)))
    return FrustumSupport3D(
        azimuth=az,
        elev_min_deg=float(np.percentile(elev, 0.5)),
        elev_max_deg=float(np.percentile(elev, 99.5)),
        range_min_m=float(max(0.5, np.percentile(r, 0.5))),
        range_max_m=float(np.percentile(r, 99.8)),
    )


def compute_3d_shared_visibility_mask(
    pts_in_other_frame: np.ndarray,
    other_frustum: FrustumSupport3D,
    *,
    sigma_rot_deg: float = 4.0,
    sigma_trans_m: float = 0.5,
    ego_bbox_min_other: np.ndarray | None = None,
    ego_bbox_max_other: np.ndarray | None = None,
    min_margin_deg: float = 3.0,
) -> np.ndarray:
    """Uncertainty-buffered 3D azimuth + elevation + ego-occlusion visibility mask."""
    pts = np.asarray(pts_in_other_frame, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    r = np.linalg.norm(pts, axis=1)
    r_safe = np.maximum(r, 2.0)
    ang_margin_deg = (
        min_margin_deg + 3.0 * sigma_rot_deg + np.degrees(3.0 * sigma_trans_m / r_safe)
    )
    az = np.mod(np.degrees(np.arctan2(pts[:, 1], pts[:, 0])), 360.0)
    el = np.degrees(np.arcsin(np.clip(pts[:, 2] / np.maximum(r, 1e-6), -1.0, 1.0)))

    if other_frustum.azimuth.span_deg >= 360.0:
        in_az = np.ones(len(pts), dtype=bool)
    else:
        center = (
            other_frustum.azimuth.start_deg + 0.5 * other_frustum.azimuth.span_deg
        ) % 360.0
        half_span = 0.5 * other_frustum.azimuth.span_deg + ang_margin_deg
        diff = np.abs((az - center + 180.0) % 360.0 - 180.0)
        in_az = diff <= half_span

    in_el = (el >= other_frustum.elev_min_deg - ang_margin_deg) & (
        el <= other_frustum.elev_max_deg + ang_margin_deg
    )
    in_r = (r >= max(0.5, other_frustum.range_min_m - 3.0 * sigma_trans_m)) & (
        r <= other_frustum.range_max_m + 3.0 * sigma_trans_m
    )
    mask = in_az & in_el & in_r

    if ego_bbox_min_other is not None and ego_bbox_max_other is not None:
        occluded = ray_intersects_aabb_mask(pts, ego_bbox_min_other, ego_bbox_max_other)
        mask &= ~occluded
    return mask


def _analyze_localizability(
    info_matrix: np.ndarray,
    reg_cfg: SemanticRegistrationConfig,
) -> dict[str, Any]:
    eigvals_6, eigvecs_6 = np.linalg.eigh(0.5 * (info_matrix + info_matrix.T))
    eigvals_6 = np.maximum(eigvals_6, 0.0)
    cond_6 = (
        float(eigvals_6[-1] / eigvals_6[0]) if eigvals_6[0] > 1e-9 else float("inf")
    )

    # Rotation block (0:3) and translation block (3:6) in right-perturbation tangent
    H_rr = info_matrix[:3, :3]
    H_tt = info_matrix[3:6, 3:6]
    eig_r, vec_r = np.linalg.eigh(0.5 * (H_rr + H_rr.T))
    eig_t, vec_t = np.linalg.eigh(0.5 * (H_tt + H_tt.T))
    eig_r = np.maximum(eig_r, 0.0)
    eig_t = np.maximum(eig_t, 0.0)
    trans_cond = float(eig_t[-1] / eig_t[0]) if eig_t[0] > 1e-9 else float("inf")
    rot_cond = float(eig_r[-1] / eig_r[0]) if eig_r[0] > 1e-9 else float("inf")

    corridor_degenerate = bool(
        trans_cond > reg_cfg.corridor_cond_threshold
        or float(eig_t[0]) < reg_cfg.min_trans_info_threshold
    )
    return {
        "eigenvalues_6d": [float(v) for v in eigvals_6],
        "condition_number_6d": cond_6,
        "rotation_eigenvalues": [float(v) for v in eig_r],
        "rotation_condition_number": rot_cond,
        "translation_eigenvalues": [float(v) for v in eig_t],
        "translation_condition_number": trans_cond,
        "weakest_translation_direction_target": [float(v) for v in vec_t[:, 0]],
        "weakest_rotation_direction_target": [float(v) for v in vec_r[:, 0]],
        "corridor_degenerate": corridor_degenerate,
    }


def _coarse_align_4dof_bev(
    prim_src: GeometricPrimitives,
    prim_tgt: GeometricPrimitives,
    T_init: np.ndarray,
    *,
    yaw_range_deg: float = 30.0,
    yaw_step_deg: float = 3.0,
    cell_m: float = 0.75,
    half_extent_m: float = 48.0,
) -> np.ndarray | None:
    """Fast 4-DoF coarse fallback: ground leveling + normal-binned BEV FFT search."""
    if (len(prim_src.facade_points) + len(prim_src.pole_points)) < 8 or (
        len(prim_tgt.facade_points) + len(prim_tgt.pole_points)
    ) < 8:
        return None

    R0 = T_init[:3, :3].copy()
    t0 = T_init[:3, 3].copy()

    # 1. Level ground normal & vertical z offset if ground primitives are available
    if len(prim_src.ground_normals) > 0 and len(prim_tgt.ground_normals) > 0:
        n_s = R0 @ prim_src.ground_normals[0]
        n_t = prim_tgt.ground_normals[0]
        n_s /= max(1e-9, float(np.linalg.norm(n_s)))
        n_t /= max(1e-9, float(np.linalg.norm(n_t)))
        v = np.cross(n_s, n_t)
        s = float(np.linalg.norm(v))
        c = float(np.clip(n_s @ n_t, -1.0, 1.0))
        if s > 1e-6:
            K = _skew(v / s)
            R_level = np.eye(3) + s * K + (1.0 - c) * (K @ K)
            R0 = R_level @ R0
        if len(prim_src.ground_points) >= 5 and len(prim_tgt.ground_points) >= 5:
            z_src = float(np.median((prim_src.ground_points @ R0.T + t0)[:, 2]))
            z_tgt = float(np.median(prim_tgt.ground_points[:, 2]))
            t0[2] += z_tgt - z_src

    # 2. Signed-normal facade + pole 5-channel 2D BEV FFT search around source pivot
    grid_n = int(round((2.0 * half_extent_m) / cell_m))
    fft_n = 1 << (grid_n * 2 - 1).bit_length()

    def _rasterize_5ch(
        fac_xy: np.ndarray, fac_n_xy: np.ndarray, pole_xy: np.ndarray
    ) -> np.ndarray:
        grid = np.zeros((5, fft_n, fft_n), dtype=np.float64)
        if len(fac_xy) > 0:
            ix = np.floor((fac_xy[:, 0] + half_extent_m) / cell_m).astype(int)
            iy = np.floor((fac_xy[:, 1] + half_extent_m) / cell_m).astype(int)
            ok = (ix >= 0) & (ix < grid_n) & (iy >= 0) & (iy < grid_n)
            if np.any(ok):
                ix_k, iy_k, n_k = ix[ok], iy[ok], fac_n_xy[ok]
                # Bin facade normal into cardinal directions (0:+X, 1:-X, 2:+Y, 3:-Y)
                dom_x = np.abs(n_k[:, 0]) >= np.abs(n_k[:, 1])
                ch = np.where(
                    dom_x,
                    np.where(n_k[:, 0] >= 0.0, 0, 1),
                    np.where(n_k[:, 1] >= 0.0, 2, 3),
                )
                np.add.at(grid, (ch, ix_k, iy_k), 1.0)
                np.clip(grid[:4], 0.0, 2.0, out=grid[:4])
        if len(pole_xy) > 0:
            px = np.floor((pole_xy[:, 0] + half_extent_m) / cell_m).astype(int)
            py = np.floor((pole_xy[:, 1] + half_extent_m) / cell_m).astype(int)
            okp = (px >= 0) & (px < grid_n) & (py >= 0) & (py < grid_n)
            if np.any(okp):
                np.add.at(grid, (4, px[okp], py[okp]), 3.5)
                np.clip(grid[4], 0.0, 7.0, out=grid[4])
        return grid

    grid_tgt = _rasterize_5ch(
        prim_tgt.facade_points[:, :2],
        prim_tgt.facade_normals[:, :2],
        prim_tgt.pole_points[:, :2],
    )
    fft_tgt = np.fft.rfft2(grid_tgt, axes=(-2, -1))

    fac_in_tgt0 = prim_src.facade_points @ R0.T + t0
    fac_n_in_tgt0 = prim_src.facade_normals @ R0.T
    pole_in_tgt0 = prim_src.pole_points @ R0.T + t0

    all_src_xy = np.vstack([fac_in_tgt0[:, :2], pole_in_tgt0[:, :2]])
    pivot_xy = np.mean(all_src_xy, axis=0)

    max_shift_cells = int(round(8.0 / cell_m))
    idx_1d = np.arange(fft_n)
    signed_idx = np.where(idx_1d <= fft_n // 2, idx_1d, idx_1d - fft_n)
    valid_shift = np.abs(signed_idx) <= max_shift_cells
    shift_mask_2d = np.outer(valid_shift, valid_shift)

    best_score = -1.0
    best_dyaw = 0.0
    best_dxy = np.zeros(2, dtype=np.float64)
    yaw_grid = np.arange(-yaw_range_deg, yaw_range_deg + 1e-6, yaw_step_deg)
    for dyaw in yaw_grid:
        c_y = math.cos(math.radians(float(dyaw)))
        s_y = math.sin(math.radians(float(dyaw)))
        rot2 = np.array([[c_y, -s_y], [s_y, c_y]], dtype=np.float64)
        fac_xy = (fac_in_tgt0[:, :2] - pivot_xy) @ rot2.T + pivot_xy
        fac_n_xy = fac_n_in_tgt0[:, :2] @ rot2.T
        pole_xy = (pole_in_tgt0[:, :2] - pivot_xy) @ rot2.T + pivot_xy
        fft_src = np.fft.rfft2(_rasterize_5ch(fac_xy, fac_n_xy, pole_xy), axes=(-2, -1))
        corr_5ch = np.fft.irfft2(
            fft_tgt * np.conj(fft_src), s=(fft_n, fft_n), axes=(-2, -1)
        )
        corr = np.sum(corr_5ch, axis=0)
        corr = np.where(shift_mask_2d, corr, -1.0)
        flat_idx = int(np.argmax(corr))
        px, py = divmod(flat_idx, fft_n)
        score = float(corr[px, py])
        if score > best_score:
            best_score = score
            best_dyaw = float(dyaw)
            sx = px if px <= fft_n // 2 else px - fft_n
            sy = py if py <= fft_n // 2 else py - fft_n
            best_dxy = np.array([sx * cell_m, sy * cell_m], dtype=np.float64)

    R_z = R.from_euler("z", best_dyaw, degrees=True).as_matrix()
    pivot_3 = np.array([pivot_xy[0], pivot_xy[1], 0.0], dtype=np.float64)
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = R_z @ R0
    T_out[:3, 3] = (
        R_z @ (t0 - pivot_3) + pivot_3 + np.array([best_dxy[0], best_dxy[1], 0.0])
    )
    return T_out


def register_semantic_l2l(
    source_frame: LidarFrameData,
    target_frame: LidarFrameData,
    initial_T_target_source: np.ndarray,
    *,
    prim_cfg: PrimitiveExtractionConfig = PrimitiveExtractionConfig(),
    refl_cfg: ReflectivityConfig = ReflectivityConfig(),
    vis_cfg: VisibilityConfig = VisibilityConfig(),
    reg_cfg: SemanticRegistrationConfig = SemanticRegistrationConfig(),
) -> dict[str, Any]:
    """Register source (B) into target (A) using 4-channel primitives, calibrated
    intensity features, 3D visibility/self-occlusion gating, and X-ICP analysis.
    """
    prim_src = extract_geometric_primitives(
        source_frame, prim_cfg=prim_cfg, refl_cfg=refl_cfg
    )
    prim_tgt = extract_geometric_primitives(
        target_frame, prim_cfg=prim_cfg, refl_cfg=refl_cfg
    )
    if (len(prim_src.facade_points) + len(prim_src.ground_points)) < 15 or (
        len(prim_tgt.facade_points) + len(prim_tgt.ground_points)
    ) < 15:
        return {
            "success": False,
            "reason": "insufficient_structural_primitives",
            "source_stats": prim_src.stats,
            "target_stats": prim_tgt.stats,
        }

    frustum_tgt = estimate_3d_frustum_support(prim_tgt.all_filtered_points)
    frustum_src = estimate_3d_frustum_support(prim_src.all_filtered_points)

    ego_min_tgt = (
        np.asarray(vis_cfg.ego_bbox_min_lidar, dtype=np.float64)
        if vis_cfg.ego_bbox_min_lidar is not None
        else None
    )
    ego_max_tgt = (
        np.asarray(vis_cfg.ego_bbox_max_lidar, dtype=np.float64)
        if vis_cfg.ego_bbox_max_lidar is not None
        else None
    )

    tree_gnd_tgt = (
        cKDTree(prim_tgt.ground_points) if len(prim_tgt.ground_points) >= 5 else None
    )
    tree_fac_tgt = (
        cKDTree(prim_tgt.facade_points) if len(prim_tgt.facade_points) >= 5 else None
    )
    tree_pole_tgt = (
        cKDTree(prim_tgt.pole_points) if len(prim_tgt.pole_points) >= 3 else None
    )
    tree_int_tgt = (
        cKDTree(prim_tgt.intensity_points)
        if len(prim_tgt.intensity_points) >= 8
        else None
    )

    # Also build source trees for optional reciprocal correspondence validation
    tree_fac_src = (
        cKDTree(prim_src.facade_points) if len(prim_src.facade_points) >= 5 else None
    )

    T_curr = np.asarray(initial_T_target_source, dtype=np.float64).copy()

    # Fast 4-DoF BEV FFT coarse fallback only if seed has weak stage-0 support
    if (
        reg_cfg.enable_coarse_fallback
        and (len(prim_src.facade_points) + len(prim_src.pole_points) >= 12)
        and (len(prim_tgt.facade_points) + len(prim_tgt.pole_points) >= 12)
    ):
        src_feat = np.vstack(
            [p for p in (prim_src.facade_points, prim_src.pole_points) if len(p) > 0]
        )
        tgt_feat = np.vstack(
            [p for p in (prim_tgt.facade_points, prim_tgt.pole_points) if len(p) > 0]
        )
        tree_feat_tgt = cKDTree(tgt_feat)
        init_pred = src_feat @ T_curr[:3, :3].T + T_curr[:3, 3]
        init_dists, _ = tree_feat_tgt.query(init_pred, k=1)
        init_support = float(np.mean(init_dists <= 0.85))
        if init_support < 0.28:
            T_coarse = _coarse_align_4dof_bev(
                prim_src,
                prim_tgt,
                T_curr,
                yaw_range_deg=reg_cfg.coarse_yaw_range_deg,
                yaw_step_deg=reg_cfg.coarse_yaw_step_deg,
            )
            if T_coarse is not None:
                c_pred = src_feat @ T_coarse[:3, :3].T + T_coarse[:3, 3]
                c_dists, _ = tree_feat_tgt.query(c_pred, k=1)
                if float(np.mean(c_dists <= 0.85)) > init_support + 0.08:
                    T_curr = T_coarse

    last_diag: dict[str, Any] = {}

    for stage_idx, (max_corr, (sig_rot_deg, sig_t_m)) in enumerate(
        zip(reg_cfg.correspondence_schedule_m, reg_cfg.uncertainty_schedule)
    ):
        R_curr = T_curr[:3, :3]
        t_curr = T_curr[:3, 3]

        # 1) Ground correspondences (no self-occlusion clip needed at medium distance,
        #    use 3D frustum visibility)
        g_src_pts = prim_src.ground_points
        g_tgt_pts = np.zeros((0, 3), dtype=np.float64)
        g_tgt_n = np.zeros((0, 3), dtype=np.float64)
        g_src_sel = np.zeros((0, 3), dtype=np.float64)
        if tree_gnd_tgt is not None and len(g_src_pts) > 0:
            g_pred = g_src_pts @ R_curr.T + t_curr
            vis_g = compute_3d_shared_visibility_mask(
                g_pred,
                frustum_tgt,
                sigma_rot_deg=sig_rot_deg,
                sigma_trans_m=sig_t_m,
                ego_bbox_min_other=ego_min_tgt,
                ego_bbox_max_other=ego_max_tgt,
            )
            g_idx_vis = np.flatnonzero(vis_g)
            if len(g_idx_vis) > 0:
                dists, nn_idx = tree_gnd_tgt.query(g_pred[g_idx_vis], k=1)
                n_dot = np.sum(
                    (prim_src.ground_normals[g_idx_vis] @ R_curr.T)
                    * prim_tgt.ground_normals[nn_idx],
                    axis=1,
                )
                gnd_search_m = max(max_corr * 2.5, 0.85)
                plane_dist = np.abs(
                    np.sum(
                        (g_pred[g_idx_vis] - prim_tgt.ground_points[nn_idx])
                        * prim_tgt.ground_normals[nn_idx],
                        axis=1,
                    )
                )
                valid = (
                    (dists <= gnd_search_m)
                    & (plane_dist <= max_corr)
                    & (n_dot >= vis_cfg.normal_covisibility_cos)
                )
                if np.any(valid):
                    g_src_sel = g_src_pts[g_idx_vis[valid]]
                    g_tgt_pts = prim_tgt.ground_points[nn_idx[valid]]
                    g_tgt_n = prim_tgt.ground_normals[nn_idx[valid]]

        # 2) Facade correspondences with same-side normal & reciprocal gate
        f_src_sel = np.zeros((0, 3), dtype=np.float64)
        f_tgt_pts = np.zeros((0, 3), dtype=np.float64)
        f_tgt_n = np.zeros((0, 3), dtype=np.float64)
        if tree_fac_tgt is not None and len(prim_src.facade_points) > 0:
            f_pred = prim_src.facade_points @ R_curr.T + t_curr
            vis_f = compute_3d_shared_visibility_mask(
                f_pred,
                frustum_tgt,
                sigma_rot_deg=sig_rot_deg,
                sigma_trans_m=sig_t_m,
                ego_bbox_min_other=ego_min_tgt,
                ego_bbox_max_other=ego_max_tgt,
            )
            f_idx_vis = np.flatnonzero(vis_f)
            if len(f_idx_vis) > 0:
                dists, nn_idx = tree_fac_tgt.query(f_pred[f_idx_vis], k=1)
                n_src_rot = prim_src.facade_normals[f_idx_vis] @ R_curr.T
                n_dot = np.sum(n_src_rot * prim_tgt.facade_normals[nn_idx], axis=1)
                valid = (dists <= max_corr) & (n_dot >= vis_cfg.normal_covisibility_cos)
                if (
                    vis_cfg.reciprocal_check
                    and tree_fac_src is not None
                    and np.any(valid)
                ):
                    T_inv = np.linalg.inv(T_curr)
                    tgt_matched_in_src = (
                        prim_tgt.facade_points[nn_idx[valid]] @ T_inv[:3, :3].T
                        + T_inv[:3, 3]
                    )
                    _, back_idx = tree_fac_src.query(tgt_matched_in_src, k=1)
                    back_dist = np.linalg.norm(
                        prim_src.facade_points[f_idx_vis[valid]]
                        - prim_src.facade_points[back_idx],
                        axis=1,
                    )
                    valid_idx = np.flatnonzero(valid)[back_dist <= max(max_corr, 0.75)]
                else:
                    valid_idx = np.flatnonzero(valid)

                if len(valid_idx) > 0:
                    f_src_sel = prim_src.facade_points[f_idx_vis[valid_idx]]
                    f_tgt_pts = prim_tgt.facade_points[nn_idx[valid_idx]]
                    f_tgt_n = prim_tgt.facade_normals[nn_idx[valid_idx]]

        # 3) Vertical pole/corner point-to-line correspondences
        p_src_sel = np.zeros((0, 3), dtype=np.float64)
        p_tgt_pts = np.zeros((0, 3), dtype=np.float64)
        p_tgt_b = np.zeros((0, 2, 3), dtype=np.float64)
        if tree_pole_tgt is not None and len(prim_src.pole_points) > 0:
            p_pred = prim_src.pole_points @ R_curr.T + t_curr
            dists, nn_idx = tree_pole_tgt.query(p_pred, k=1)
            dir_dot = np.abs(
                np.sum(
                    (prim_src.pole_directions @ R_curr.T)
                    * prim_tgt.pole_directions[nn_idx],
                    axis=1,
                )
            )
            valid = (dists <= max(max_corr * 1.35, 0.45)) & (dir_dot >= 0.88)
            if np.any(valid):
                p_src_sel = prim_src.pole_points[valid]
                p_tgt_pts = prim_tgt.pole_points[nn_idx[valid]]
                p_tgt_b = np.array(
                    [_tangent_basis(d) for d in prim_tgt.pole_directions[nn_idx[valid]]]
                )

        # 4) Calibrated high-reflectivity feature correspondences
        m_src_sel = np.zeros((0, 3), dtype=np.float64)
        m_tgt_pts = np.zeros((0, 3), dtype=np.float64)
        if tree_int_tgt is not None and len(prim_src.intensity_points) >= 8:
            m_pred = prim_src.intensity_points @ R_curr.T + t_curr
            dists, nn_idx = tree_int_tgt.query(m_pred, k=1)
            z_diff = np.abs(prim_src.intensity_z - prim_tgt.intensity_z[nn_idx])
            valid = (dists <= max(max_corr * 1.25, 0.35)) & (z_diff <= 2.5)
            if np.any(valid):
                m_src_sel = prim_src.intensity_points[valid]
                m_tgt_pts = prim_tgt.intensity_points[nn_idx[valid]]

        total_matches = (
            len(g_src_sel) + len(f_src_sel) + len(p_src_sel) + len(m_src_sel)
        )
        if total_matches < 12:
            if stage_idx == 0:
                return {
                    "success": False,
                    "reason": "insufficient_covisible_correspondences",
                    "source_stats": prim_src.stats,
                    "target_stats": prim_tgt.stats,
                }
            break

        # Inner Gauss-Newton / TRF solve using right-perturbation T_curr @ exp(xi)
        def _eval_res_and_jac(xi: np.ndarray, compute_jac: bool = True):
            dR = R.from_rotvec(xi[:3]).as_matrix()
            dt = xi[3:6]
            R_try = R_curr @ dR
            t_try = R_curr @ dt + t_curr

            res_chunks: list[np.ndarray] = []
            jac_chunks: list[np.ndarray] = []

            for src_p, tgt_q, tgt_n, w in (
                (f_src_sel, f_tgt_pts, f_tgt_n, reg_cfg.facade_weight),
                (g_src_sel, g_tgt_pts, g_tgt_n, reg_cfg.ground_weight),
            ):
                if len(src_p) == 0:
                    continue
                p_rot = src_p @ R_try.T
                diff = p_rot + t_try - tgt_q
                res_chunks.append(w * np.sum(diff * tgt_n, axis=1))
                if compute_jac:
                    n_in_curr = tgt_n @ R_try
                    # n^T R_try (dphi x p) = (p x (R_try^T n))^T dphi
                    j_rot = np.cross(src_p, n_in_curr)
                    j_t = n_in_curr
                    jac_chunks.append(w * np.hstack([j_rot, j_t]))

            if len(p_src_sel) > 0:
                p_rot = p_src_sel @ R_try.T
                diff = p_rot + t_try - p_tgt_pts
                w_p = reg_cfg.pole_weight
                # 2 orthogonal directions per pole point
                r_pole = w_p * np.einsum("kij,kj->ki", p_tgt_b, diff).reshape(-1)
                res_chunks.append(r_pole)
                if compute_jac:
                    b_in_curr = np.einsum("kij,jl->kil", p_tgt_b, R_try)  # (K, 2, 3)
                    j1_rot = np.cross(p_src_sel, b_in_curr[:, 0, :])
                    j2_rot = np.cross(p_src_sel, b_in_curr[:, 1, :])
                    j_pole = np.empty((len(p_src_sel) * 2, 6), dtype=np.float64)
                    j_pole[0::2, :3] = j1_rot
                    j_pole[0::2, 3:6] = b_in_curr[:, 0, :]
                    j_pole[1::2, :3] = j2_rot
                    j_pole[1::2, 3:6] = b_in_curr[:, 1, :]
                    jac_chunks.append(w_p * j_pole)

            if len(m_src_sel) > 0:
                p_rot = m_src_sel @ R_try.T
                diff = p_rot + t_try - m_tgt_pts
                w_m = reg_cfg.intensity_weight
                # Full 3D point-to-point on retro landmarks locks planar XY markings
                res_chunks.append(w_m * diff.reshape(-1))
                if compute_jac:
                    n_m = len(m_src_sel)
                    j_m = np.empty((n_m * 3, 6), dtype=np.float64)
                    for ax in range(3):
                        e_ax = np.tile(R_try[ax, :], (n_m, 1))
                        j_m[ax::3, :3] = np.cross(m_src_sel, e_ax)
                        j_m[ax::3, 3:6] = e_ax
                    jac_chunks.append(w_m * j_m)

            res_vec = np.concatenate(res_chunks)
            if not compute_jac:
                return res_vec
            jac_mat = np.vstack(jac_chunks)
            return res_vec, jac_mat

        sol = least_squares(
            lambda x: _eval_res_and_jac(x, compute_jac=False),
            np.zeros(6, dtype=np.float64),
            jac=lambda x: _eval_res_and_jac(x, compute_jac=True)[1],
            method="trf",
            loss="huber",
            f_scale=reg_cfg.huber_scale_m,
            max_nfev=35,
        )
        xi_opt = sol.x
        T_step = np.eye(4, dtype=np.float64)
        T_step[:3, :3] = R.from_rotvec(xi_opt[:3]).as_matrix()
        T_step[:3, 3] = xi_opt[3:6]
        T_curr = T_curr @ T_step

        res_final, jac_final = _eval_res_and_jac(np.zeros(6), compute_jac=True)
        mad_m = float(1.4826 * np.median(np.abs(res_final))) if len(res_final) else 0.05
        sigma_eff = max(reg_cfg.noise_floor_m, mad_m)
        # Huber weights at solution
        abs_r = np.abs(res_final)
        h_w = np.where(
            abs_r <= reg_cfg.huber_scale_m,
            1.0,
            reg_cfg.huber_scale_m / np.maximum(abs_r, 1e-9),
        )
        J_w = jac_final * np.sqrt(h_w)[:, None]
        H_info = (J_w.T @ J_w) / (sigma_eff**2)

        def _rms(arr: np.ndarray) -> float:
            return float(np.sqrt(np.mean(arr**2))) if len(arr) else 0.0

        f_res = (
            np.sum(
                ((f_src_sel @ T_curr[:3, :3].T + T_curr[:3, 3]) - f_tgt_pts) * f_tgt_n,
                axis=1,
            )
            if len(f_src_sel)
            else np.zeros(0)
        )
        g_res = (
            np.sum(
                ((g_src_sel @ T_curr[:3, :3].T + T_curr[:3, 3]) - g_tgt_pts) * g_tgt_n,
                axis=1,
            )
            if len(g_src_sel)
            else np.zeros(0)
        )
        p_res = (
            np.einsum(
                "kij,kj->ki",
                p_tgt_b,
                (p_src_sel @ T_curr[:3, :3].T + T_curr[:3, 3]) - p_tgt_pts,
            ).reshape(-1)
            if len(p_src_sel)
            else np.zeros(0)
        )

        loc = _analyze_localizability(H_info, reg_cfg)
        last_diag = {
            "success": True,
            "transformation": T_curr.copy(),
            "info_matrix_6d": H_info,
            "sigma_eff_m": float(sigma_eff),
            "mad_residual_m": float(mad_m),
            "localizability": loc,
            "channel_matches": {
                "facade": int(len(f_src_sel)),
                "ground": int(len(g_src_sel)),
                "pole": int(len(p_src_sel)),
                "intensity": int(len(m_src_sel)),
            },
            "channel_rmse_m": {
                "facade": _rms(f_res),
                "ground": _rms(g_res),
                "pole": _rms(p_res),
                "overall_weighted": _rms(res_final),
            },
            "source_stats": prim_src.stats,
            "target_stats": prim_tgt.stats,
        }

    if last_diag.get("success"):
        # Compute directional shared overlap once after convergence on 2k subsamples
        sig_rot_deg, sig_t_m = reg_cfg.uncertainty_schedule[-1]
        src_sub = prim_src.all_filtered_points[
            :: max(1, len(prim_src.all_filtered_points) // 2000)
        ]
        tgt_sub = prim_tgt.all_filtered_points[
            :: max(1, len(prim_tgt.all_filtered_points) // 2000)
        ]
        src_in_tgt = src_sub @ T_curr[:3, :3].T + T_curr[:3, 3]
        T_inv_final = np.linalg.inv(T_curr)
        tgt_in_src = tgt_sub @ T_inv_final[:3, :3].T + T_inv_final[:3, 3]
        vis_b_to_a = compute_3d_shared_visibility_mask(
            src_in_tgt,
            frustum_tgt,
            sigma_rot_deg=sig_rot_deg,
            sigma_trans_m=sig_t_m,
            ego_bbox_min_other=ego_min_tgt,
            ego_bbox_max_other=ego_max_tgt,
        )
        vis_a_to_b = compute_3d_shared_visibility_mask(
            tgt_in_src,
            frustum_src,
            sigma_rot_deg=sig_rot_deg,
            sigma_trans_m=sig_t_m,
        )
        tree_tgt_sub = cKDTree(tgt_sub)
        tree_src_sub = cKDTree(src_sub)
        ov_b_to_a = (
            float(np.mean(tree_tgt_sub.query(src_in_tgt[vis_b_to_a])[0] <= 0.35))
            if np.any(vis_b_to_a)
            else 0.0
        )
        ov_a_to_b = (
            float(np.mean(tree_src_sub.query(tgt_in_src[vis_a_to_b])[0] <= 0.35))
            if np.any(vis_a_to_b)
            else 0.0
        )
        last_diag["directed_overlap"] = {
            "source_to_target_vis_ratio": float(np.mean(vis_b_to_a)),
            "target_to_source_vis_ratio": float(np.mean(vis_a_to_b)),
            "source_to_target_inlier_overlap": ov_b_to_a,
            "target_to_source_inlier_overlap": ov_a_to_b,
        }

    return last_diag
