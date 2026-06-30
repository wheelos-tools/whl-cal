from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lidar2imu.models import CalibrationDataset
from lidar2lidar.extrinsic_io import matrix_from_transform_dict


@dataclass(frozen=True)
class TrajectoryNode:
    timestamp_ns: int
    record_path: str | None
    imu_pose: np.ndarray
    lidar_pose: np.ndarray


def _matrix_from_components(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def trajectory_nodes(
    dataset: CalibrationDataset, final_transform: np.ndarray
) -> list[TrajectoryNode]:
    ordered_samples = sorted(
        dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if not ordered_samples:
        return []

    final_transform = np.asarray(final_transform, dtype=float)
    final_transform_inv = np.linalg.inv(final_transform)
    imu_pose = np.eye(4, dtype=float)
    lidar_pose = np.eye(4, dtype=float)
    nodes = [
        TrajectoryNode(
            timestamp_ns=int(ordered_samples[0].start_timestamp_ns),
            record_path=ordered_samples[0].metadata.get("record_path_start"),
            imu_pose=imu_pose.copy(),
            lidar_pose=lidar_pose.copy(),
        )
    ]
    for sample in ordered_samples:
        imu_delta = _matrix_from_components(
            sample.imu_delta_rotation, sample.imu_delta_translation
        )
        lidar_delta = _matrix_from_components(
            sample.lidar_delta_rotation, sample.lidar_delta_translation
        )
        lidar_delta_in_imu = final_transform @ lidar_delta @ final_transform_inv
        imu_pose = imu_pose @ np.linalg.inv(imu_delta)
        lidar_pose = lidar_pose @ np.linalg.inv(lidar_delta_in_imu)
        nodes.append(
            TrajectoryNode(
                timestamp_ns=int(sample.end_timestamp_ns),
                record_path=sample.metadata.get("record_path_end"),
                imu_pose=imu_pose.copy(),
                lidar_pose=lidar_pose.copy(),
            )
        )
    return nodes


def trajectory_cloud_nodes(
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    *,
    segment_spacing_m: float = 12.0,
) -> list[TrajectoryNode]:
    ordered_samples = sorted(
        dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if not ordered_samples:
        return []

    final_transform = np.asarray(final_transform, dtype=float)
    final_transform_inv = np.linalg.inv(final_transform)
    nodes: list[TrajectoryNode] = []
    segment_anchor_x = 0.0

    for sample in ordered_samples:
        anchor = np.eye(4, dtype=float)
        anchor[0, 3] = float(segment_anchor_x)

        nodes.append(
            TrajectoryNode(
                timestamp_ns=int(sample.start_timestamp_ns),
                record_path=sample.metadata.get("record_path_start"),
                imu_pose=anchor.copy(),
                lidar_pose=anchor.copy(),
            )
        )

        imu_delta = _matrix_from_components(
            sample.imu_delta_rotation, sample.imu_delta_translation
        )
        lidar_delta = _matrix_from_components(
            sample.lidar_delta_rotation, sample.lidar_delta_translation
        )
        lidar_delta_in_imu = final_transform @ lidar_delta @ final_transform_inv

        nodes.append(
            TrajectoryNode(
                timestamp_ns=int(sample.end_timestamp_ns),
                record_path=sample.metadata.get("record_path_end"),
                imu_pose=anchor @ np.linalg.inv(imu_delta),
                lidar_pose=anchor @ np.linalg.inv(lidar_delta_in_imu),
            )
        )

        segment_anchor_x += float(segment_spacing_m)

    return nodes


def trajectory_xy_points(
    nodes: list[TrajectoryNode], *, source: str
) -> list[tuple[float, float]]:
    points = []
    for node in nodes:
        pose = node.imu_pose if source == "imu" else node.lidar_pose
        points.append((float(pose[0, 3]), float(pose[1, 3])))
    return points


def trajectory_gap_points(nodes: list[TrajectoryNode]) -> list[tuple[float, float]]:
    points = []
    for index, node in enumerate(nodes):
        gap = float(np.linalg.norm(node.imu_pose[:3, 3] - node.lidar_pose[:3, 3]))
        points.append((float(index), gap))
    return points


def review_motion_candidates(
    raw_payload: dict[str, Any],
    *,
    selected_only: bool = False,
    selected_window_ids: set[int] | list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    candidates = raw_payload.get("review_motion_candidates") or []
    if not isinstance(candidates, list):
        return []
    review_candidates = [item for item in candidates if isinstance(item, dict)]
    if selected_window_ids is not None:
        allowed_window_ids = {int(window_id) for window_id in selected_window_ids}
        review_candidates = [
            item
            for item in review_candidates
            if int(item.get("window_id", -1)) in allowed_window_ids
        ]
    elif selected_only:
        selected_candidates = [
            item
            for item in review_candidates
            if bool(item.get("selected_for_calibration", False))
        ]
        if selected_candidates:
            review_candidates = selected_candidates
    review_candidates.sort(
        key=lambda item: (
            int(item.get("start_timestamp_ns", 0)),
            int(item.get("end_timestamp_ns", 0)),
        )
    )
    return review_candidates


def review_trajectory_nodes(
    raw_payload: dict[str, Any],
    final_transform: np.ndarray,
    *,
    selected_only: bool = True,
    selected_window_ids: set[int] | list[int] | tuple[int, ...] | None = None,
) -> list[TrajectoryNode]:
    review_candidates = review_motion_candidates(
        raw_payload,
        selected_only=selected_only,
        selected_window_ids=selected_window_ids,
    )
    if not review_candidates:
        return []

    final_transform = np.asarray(final_transform, dtype=float)
    final_transform_inv = np.linalg.inv(final_transform)
    imu_pose = np.eye(4, dtype=float)
    lidar_pose = np.eye(4, dtype=float)
    nodes = [
        TrajectoryNode(
            timestamp_ns=int(review_candidates[0].get("start_timestamp_ns", 0)),
            record_path=review_candidates[0].get("record_path_start"),
            imu_pose=imu_pose.copy(),
            lidar_pose=lidar_pose.copy(),
        )
    ]
    for candidate in review_candidates:
        imu_delta = matrix_from_transform_dict(candidate["imu_delta"])
        lidar_delta = matrix_from_transform_dict(candidate["lidar_delta"])
        lidar_delta_in_imu = final_transform @ lidar_delta @ final_transform_inv
        imu_pose = imu_pose @ np.linalg.inv(imu_delta)
        lidar_pose = lidar_pose @ np.linalg.inv(lidar_delta_in_imu)
        nodes.append(
            TrajectoryNode(
                timestamp_ns=int(candidate.get("end_timestamp_ns", 0)),
                record_path=candidate.get("record_path_end"),
                imu_pose=imu_pose.copy(),
                lidar_pose=lidar_pose.copy(),
            )
        )
    return nodes
