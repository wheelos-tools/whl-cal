#!/usr/bin/env python3
"""Simple smoke runner for lidar2camera using a synthetic dataset.

This script generates a small synthetic set of board poses, projects them
through a known LiDAR->Camera transform, then runs the optimizer to recover
that transform. It prints delta metrics so users see the calibration surface.

If SciPy is missing, the script exits cleanly with a message.
"""

from __future__ import annotations

import sys
import os
import argparse
import numpy as np

# Ensure repo root is importable when this script is executed from tools/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    print("[SKIP] scipy is not available — install scipy to run the smoke test.")
    sys.exit(0)

from lidar2camera.reference_pipeline import _build_board_template, _optimize_dataset
from lidar2camera.models import (
    ReferenceCalibrationDataset,
    ReferencePoseObservation,
    ReferenceCalibrationConfig,
)
from lidar2camera.metrics import transform_delta_metrics


def project_points(camera_matrix: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    X = points_3d
    u = (camera_matrix[0, 0] * X[:, 0] / X[:, 2]) + camera_matrix[0, 2]
    v = (camera_matrix[1, 1] * X[:, 1] / X[:, 2]) + camera_matrix[1, 2]
    return np.vstack([u, v]).T


def make_true_transform() -> np.ndarray:
    rotvec = np.array([0.01, -0.02, 0.03])
    t = np.array([0.1, 0.0, 1.2])
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def run_smoke(pattern_size=(4, 3), square_size=0.05, num_poses=5):
    # camera intrinsics
    fx = fy = 500.0
    w, h = 640, 480
    camera_matrix = np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    camera_distortion = np.zeros(5)

    board_template = _build_board_template(pattern_size, square_size)
    true_T = make_true_transform()

    observations = []
    for i in range(num_poses):
        theta = (i - num_poses / 2) * 0.08
        rotvec = np.array([0.0, 0.0, theta])
        t = np.array([0.02 * (i - num_poses / 2), 0.01 * i, 2.0])
        Rb = R.from_rotvec(rotvec).as_matrix()
        obj_pts = (Rb @ board_template.T).T + t
        cam_pts = (true_T[:3, :3] @ obj_pts.T).T + true_T[:3, 3]
        image_pts = project_points(camera_matrix, cam_pts)
        obs = ReferencePoseObservation(
            pose_id=f"pose_{i}",
            image_path="",
            pcd_path="",
            image_size_wh=(w, h),
            image_points=np.asarray(image_pts, dtype=float),
            object_points=np.asarray(obj_pts, dtype=float),
            metadata={},
        )
        observations.append(obs)

    dataset = ReferenceCalibrationDataset(
        parent_frame="camera",
        child_frame="lidar",
        camera_matrix=camera_matrix,
        camera_distortion=camera_distortion,
        observations=observations,
        initial_transform=true_T,
        metadata={},
    )

    config = ReferenceCalibrationConfig(min_pose_count=3)
    initial_transform = true_T

    print("[INFO] Running optimizer on synthetic dataset...")
    final_transform, optimization = _optimize_dataset(
        dataset, config, initial_transform
    )

    delta = transform_delta_metrics(true_T, final_transform)
    print("\n[RESULT] True -> Recovered delta:")
    print(f"  translation_norm_m: {delta['translation_norm_m']:.6f}")
    print(f"  rotation_deg: {delta['rotation_deg']:.6f}")
    print(f"  distinct_solution_indicator: {optimization.get('success', False)}")

    # Simple pass/fail heuristic
    if delta["translation_norm_m"] < 0.05 and delta["rotation_deg"] < 5.0:
        print(
            "[PASS] Smoke test passed — recovered transform is close to ground truth."
        )
        return 0
    else:
        print("[FAIL] Smoke test failed — result deviates from ground truth.")
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a lightweight lidar2camera smoke test on synthetic data."
    )
    parser.add_argument(
        "--poses",
        type=int,
        default=5,
        help="Number of synthetic board poses to generate",
    )
    args = parser.parse_args()
    rc = run_smoke(num_poses=args.poses)
    sys.exit(rc)
