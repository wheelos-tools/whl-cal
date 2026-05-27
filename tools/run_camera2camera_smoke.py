#!/usr/bin/env python3
"""Synthetic smoke runner for camera2camera."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def _project_points(
    camera_matrix: np.ndarray, transform: np.ndarray, object_points: np.ndarray
) -> np.ndarray:
    points = (transform[:3, :3] @ object_points.T).T + transform[:3, 3]
    u = camera_matrix[0, 0] * points[:, 0] / points[:, 2] + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * points[:, 1] / points[:, 2] + camera_matrix[1, 2]
    return np.vstack([u, v]).T


def _make_transform(rotvec: list[float], translation: list[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def run_smoke(pattern_size=(6, 5), square_size=0.04, num_pairs=8) -> int:
    from camera2camera.metrics import (build_metrics_output,
                                       transform_delta_metrics)
    from camera2camera.models import (StereoCalibrationConfig,
                                      StereoCalibrationDataset,
                                      StereoCalibrationObservation)
    from camera2camera.reference_pipeline import (_build_board_template,
                                                  _evaluate_dataset,
                                                  _optimize_dataset)

    width = 1280
    height = 720
    parent_camera_matrix = np.array(
        [[920.0, 0.0, width / 2.0], [0.0, 915.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    child_camera_matrix = np.array(
        [
            [905.0, 0.0, width / 2.0 + 12.0],
            [0.0, 900.0, height / 2.0 - 6.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    board_template = _build_board_template(pattern_size, square_size)
    true_transform = _make_transform([0.01, -0.015, 0.02], [0.22, 0.01, -0.02])
    initial_transform = _make_transform([0.015, -0.02, 0.028], [0.19, 0.02, -0.01])

    observations = []
    for index in range(num_pairs):
        depth = 1.4 + 0.08 * index
        tx = -0.18 + 0.05 * index
        ty = -0.08 + 0.025 * index
        board_transform = _make_transform(
            [0.08 * np.sin(index * 0.7), -0.06 * np.cos(index * 0.5), 0.03 * index],
            [tx, ty, depth],
        )
        parent_points = _project_points(
            parent_camera_matrix, board_transform, board_template
        )
        child_points = _project_points(
            child_camera_matrix,
            true_transform @ board_transform,
            board_template,
        )
        parent_points += np.random.default_rng(index).normal(
            scale=0.15, size=parent_points.shape
        )
        child_points += np.random.default_rng(index + 100).normal(
            scale=0.15, size=child_points.shape
        )
        observations.append(
            StereoCalibrationObservation(
                pose_id=f"pair_{index}",
                parent_image_path="",
                child_image_path="",
                parent_image_size_wh=(width, height),
                child_image_size_wh=(width, height),
                parent_image_points=np.asarray(parent_points, dtype=float),
                child_image_points=np.asarray(child_points, dtype=float),
                object_points=np.asarray(board_template, dtype=float),
                metadata={
                    "initial_parent_board_transform_matrix": board_transform.tolist(),
                    "selected_permutation": "identity",
                },
            )
        )

    dataset = StereoCalibrationDataset(
        parent_frame="camera_parent",
        child_frame="camera_child",
        parent_camera_matrix=parent_camera_matrix,
        parent_camera_distortion=np.zeros(5, dtype=float),
        child_camera_matrix=child_camera_matrix,
        child_camera_distortion=np.zeros(5, dtype=float),
        observations=observations,
        initial_transform=initial_transform,
        metadata={},
    )
    config = StereoCalibrationConfig(
        min_pair_count=6,
        optimization_max_refinement_rounds=1,
        metrics_min_leave_one_out_pair_count=6,
    )
    optimized_dataset, final_transform, board_transforms, optimization_report = (
        _optimize_dataset(
            dataset,
            config,
            initial_transform,
        )
    )
    evaluation = _evaluate_dataset(
        optimized_dataset,
        config,
        initial_transform=initial_transform,
        final_transform=final_transform,
        board_transforms=board_transforms,
    )
    metrics_output = build_metrics_output(
        dataset=optimized_dataset,
        config=config,
        initial_transform=initial_transform,
        final_transform=final_transform,
        extraction_report={
            "pairing_summary": {"paired_count": num_pairs},
            "final_inlier_pair_ratio": len(optimized_dataset.observations)
            / float(num_pairs),
            "ordering_resolution": {"changed_pair_count": 0},
        },
        optimization_report=optimization_report,
        evaluation=evaluation,
    )
    delta = transform_delta_metrics(true_transform, final_transform)
    print("[RESULT] camera2camera smoke")
    print(f"  translation_norm_m: {delta['translation_norm_m']:.6f}")
    print(f"  rotation_deg: {delta['rotation_deg']:.6f}")
    print(f"  final_rms_px: {metrics_output['summary']['final_rms_px']:.6f}")
    print(f"  release_ready: {metrics_output['summary']['release_ready']}")
    if delta["translation_norm_m"] < 0.02 and delta["rotation_deg"] < 0.3:
        print("[PASS] Smoke test passed.")
        return 0
    print("[FAIL] Smoke test failed.")
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a lightweight camera2camera smoke test on synthetic data."
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=8,
        help="Number of synthetic stereo pairs to generate.",
    )
    args = parser.parse_args()
    raise SystemExit(run_smoke(num_pairs=args.pairs))
