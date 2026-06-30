from __future__ import annotations

import numpy as np
import open3d as o3d

from lidar2imu.algorithms import normalize_plane


def filter_ground_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    radius = np.linalg.norm(points[:, :2], axis=1)
    mask = np.isfinite(points).all(axis=1)
    mask &= radius >= 3.0
    mask &= radius <= 60.0
    filtered = points[mask]
    if filtered.shape[0] < 500:
        return filtered
    z_limit = float(np.percentile(filtered[:, 2], 35))
    mask_low = filtered[:, 2] <= z_limit + 0.8
    return filtered[mask_low]


def extract_ground_plane(
    cloud: o3d.geometry.PointCloud,
    expected_up_lidar: np.ndarray,
    plane_dist_thresh: float,
    normal_angle_thresh_deg: float,
) -> tuple[dict | None, dict]:
    points = filter_ground_points(np.asarray(cloud.points))
    diagnostics = {
        "input_points": int(len(cloud.points)),
        "candidate_points": int(points.shape[0]),
    }
    if points.shape[0] < 200:
        diagnostics["reason"] = "too_few_candidate_points"
        return None, diagnostics

    candidate_cloud = o3d.geometry.PointCloud()
    candidate_cloud.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = candidate_cloud.segment_plane(
        distance_threshold=plane_dist_thresh,
        ransac_n=3,
        num_iterations=500,
    )
    normal, offset = normalize_plane(
        np.asarray(plane_model[:3], dtype=float), float(plane_model[3])
    )
    if float(np.dot(normal, expected_up_lidar)) < 0.0:
        normal = -normal
        offset = -offset
    normal, offset = normalize_plane(normal, offset)
    angle_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(normal, expected_up_lidar), -1.0, 1.0)))
    )
    diagnostics.update(
        {
            "inlier_count": int(len(inliers)),
            "inlier_ratio": float(len(inliers) / max(len(points), 1)),
            "normal_angle_to_expected_up_deg": angle_deg,
        }
    )
    if angle_deg > normal_angle_thresh_deg:
        diagnostics["reason"] = "plane_not_ground_like"
        return None, diagnostics

    result = {
        "lidar_plane_normal": [float(value) for value in normal.tolist()],
        "lidar_plane_offset": float(offset),
        "inlier_ratio": float(len(inliers) / max(len(points), 1)),
        "normal_angle_to_expected_up_deg": angle_deg,
    }
    return result, diagnostics
