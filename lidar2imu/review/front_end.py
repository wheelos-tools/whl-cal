from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

from lidar2imu.mapping.overlap import point_cloud_overlap_metrics
from lidar2imu.models import CalibrationDataset
from lidar2imu.review.io_utils import write_csv_rows, write_point_cloud
from lidar2imu.review.registration_objects import build_registration_object_review
from lidar2imu.review.trajectory import (
    TrajectoryNode,
    review_motion_candidates,
)
from lidar2lidar.extrinsic_io import matrix_from_transform_dict
from lidar2lidar.record_utils import PointCloudMeta, load_pointcloud_from_meta


def local_pose_in_parent(
    local_pose: np.ndarray, final_transform: np.ndarray
) -> np.ndarray:
    final_transform = np.asarray(final_transform, dtype=float)
    return final_transform @ local_pose @ np.linalg.inv(final_transform)


def refinement_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    movable_records = records[1:] if len(records) > 1 else []
    refined_records = [
        record
        for record in movable_records
        if record.get("refined", record.get("local_refined"))
    ]
    fitness_values = [
        float(record["registration_fitness"])
        for record in refined_records
        if record.get("registration_fitness") is not None
    ]
    rmse_values = [
        float(record["registration_inlier_rmse"])
        for record in refined_records
        if record.get("registration_inlier_rmse") is not None
    ]
    return {
        "support_frame_count": int(len(records)),
        "movable_support_count": int(len(movable_records)),
        "refined_support_count": int(len(refined_records)),
        "refined_support_ratio": (
            float(len(refined_records) / len(movable_records))
            if movable_records
            else 0.0
        ),
        "refinement_fitness_mean": (
            None if not fitness_values else float(np.mean(fitness_values))
        ),
        "refinement_inlier_rmse_mean": (
            None if not rmse_values else float(np.mean(rmse_values))
        ),
    }


def trajectory_records(review: dict[str, Any]) -> list[dict[str, Any]]:
    records = review.get("trajectory_records")
    if isinstance(records, list) and records:
        return records
    return list(review.get("support_records", []))


def build_registration_review_artifacts(
    diagnostics_dir: Path,
    *,
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    raw_payload: dict[str, Any],
    selected_window_ids: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], list[TrajectoryNode]]:
    review_candidates = review_motion_candidates(
        raw_payload,
        selected_only=True,
        selected_window_ids=selected_window_ids,
    )
    candidate_scope = (
        "solver_inlier_windows"
        if selected_window_ids is not None
        else "selected_for_calibration"
    )
    if not review_candidates and selected_window_ids is not None:
        review_candidates = review_motion_candidates(raw_payload, selected_only=True)
        candidate_scope = "selected_for_calibration"
    if not review_candidates:
        review_candidates = review_motion_candidates(raw_payload)
        candidate_scope = "all_review_candidates"
    if not review_candidates:
        return {}, [], []

    final_transform = np.asarray(final_transform, dtype=float)
    final_transform_inv = np.linalg.inv(final_transform)
    scan_cache: dict[tuple[Any, ...], o3d.geometry.PointCloud] = {}
    metadata_cache: dict[tuple[Any, ...], list[PointCloudMeta]] = {}
    object_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    imu_cloud = o3d.geometry.PointCloud()
    lidar_cloud = o3d.geometry.PointCloud()
    overlay_cloud = o3d.geometry.PointCloud()
    review_rows: list[dict[str, Any]] = []
    dense_nodes_by_timestamp: dict[int, TrajectoryNode] = {}
    imu_anchor_pose = np.eye(4, dtype=float)
    lidar_anchor_pose = np.eye(4, dtype=float)
    first_candidate_written = False

    for segment_index, candidate in enumerate(review_candidates):
        lidar_topic = candidate.get("lidar_topic") or dataset.metadata.get(
            "lidar_topic"
        )
        if not lidar_topic:
            continue
        source_descriptor = candidate.get("source_registration_object") or {}
        target_descriptor = candidate.get("target_registration_object") or {}
        source_review = build_registration_object_review(
            source_descriptor,
            lidar_topic=str(lidar_topic),
            frame_id=dataset.child_frame,
            scan_cache=scan_cache,
            metadata_cache=metadata_cache,
            object_cache=object_cache,
        )
        target_review = build_registration_object_review(
            target_descriptor,
            lidar_topic=str(lidar_topic),
            frame_id=dataset.child_frame,
            scan_cache=scan_cache,
            metadata_cache=metadata_cache,
            object_cache=object_cache,
        )
        source_cloud = o3d.geometry.PointCloud(source_review["merged_cloud"])
        target_cloud = o3d.geometry.PointCloud(target_review["merged_cloud"])
        if source_cloud.is_empty() or target_cloud.is_empty():
            continue

        lidar_delta = matrix_from_transform_dict(candidate["lidar_delta"])
        imu_delta = matrix_from_transform_dict(candidate["imu_delta"])
        predicted_lidar_delta = final_transform_inv @ imu_delta @ final_transform
        lidar_delta_in_imu = final_transform @ lidar_delta @ final_transform_inv
        end_imu_anchor_pose = imu_anchor_pose @ np.linalg.inv(imu_delta)
        end_lidar_anchor_pose = lidar_anchor_pose @ np.linalg.inv(lidar_delta_in_imu)

        source_trajectory_records = trajectory_records(source_review)
        target_trajectory_records = trajectory_records(target_review)
        if not first_candidate_written:
            for record in sorted(
                source_trajectory_records,
                key=lambda item: int(item["timestamp_ns"]),
            ):
                dense_nodes_by_timestamp[int(record["timestamp_ns"])] = TrajectoryNode(
                    timestamp_ns=int(record["timestamp_ns"]),
                    record_path=str(record["record_path"]),
                    imu_pose=imu_anchor_pose
                    @ local_pose_in_parent(
                        record["initial_transform"], final_transform
                    ),
                    lidar_pose=lidar_anchor_pose
                    @ local_pose_in_parent(
                        record["refined_transform"], final_transform
                    ),
                )
            first_candidate_written = True
        for record in sorted(
            target_trajectory_records, key=lambda item: int(item["timestamp_ns"])
        ):
            dense_nodes_by_timestamp[int(record["timestamp_ns"])] = TrajectoryNode(
                timestamp_ns=int(record["timestamp_ns"]),
                record_path=str(record["record_path"]),
                imu_pose=end_imu_anchor_pose
                @ local_pose_in_parent(record["initial_transform"], final_transform),
                lidar_pose=end_lidar_anchor_pose
                @ local_pose_in_parent(record["refined_transform"], final_transform),
            )

        world_target_pose = end_imu_anchor_pose @ final_transform

        target_segment = o3d.geometry.PointCloud(target_cloud)
        target_segment.transform(world_target_pose)
        target_segment.paint_uniform_color((0.45, 0.45, 0.45))

        lidar_source_segment = o3d.geometry.PointCloud(source_cloud)
        lidar_source_segment.transform(world_target_pose @ lidar_delta)
        lidar_source_segment.paint_uniform_color((0.862, 0.239, 0.176))

        imu_source_segment = o3d.geometry.PointCloud(source_cloud)
        imu_source_segment.transform(world_target_pose @ predicted_lidar_delta)
        imu_source_segment.paint_uniform_color((0.145, 0.388, 0.922))

        imu_cloud += target_segment + imu_source_segment
        lidar_cloud += o3d.geometry.PointCloud(target_segment) + lidar_source_segment
        overlay_cloud += (
            o3d.geometry.PointCloud(target_segment)
            + o3d.geometry.PointCloud(lidar_source_segment)
            + o3d.geometry.PointCloud(imu_source_segment)
        )

        lidar_overlap = point_cloud_overlap_metrics(
            o3d.geometry.PointCloud(lidar_source_segment),
            o3d.geometry.PointCloud(target_segment),
        )
        imu_overlap = point_cloud_overlap_metrics(
            o3d.geometry.PointCloud(imu_source_segment),
            o3d.geometry.PointCloud(target_segment),
        )
        source_refinement = refinement_summary(source_review["support_records"])
        target_refinement = refinement_summary(target_review["support_records"])
        review_rows.append(
            {
                "segment_index": int(segment_index),
                "window_id": int(candidate.get("window_id", -1)),
                "selected_for_calibration": bool(
                    candidate.get("selected_for_calibration", False)
                ),
                "motion_registration_mode": str(
                    candidate.get("motion_registration_mode", "")
                ),
                "source_object_type": str(source_descriptor.get("object_type", "")),
                "target_object_type": str(target_descriptor.get("object_type", "")),
                "source_support_frame_count": int(
                    source_descriptor.get("support_frame_count", 0)
                ),
                "target_support_frame_count": int(
                    target_descriptor.get("support_frame_count", 0)
                ),
                "source_registration_object_builder": str(
                    source_descriptor.get("builder_mode", "raw_scan")
                ),
                "target_registration_object_builder": str(
                    target_descriptor.get("builder_mode", "raw_scan")
                ),
                "source_review_frame_count": int(source_review.get("frame_count", 0)),
                "target_review_frame_count": int(target_review.get("frame_count", 0)),
                "source_review_requested_frame_count": int(
                    source_review.get("requested_frame_count", 0)
                ),
                "target_review_requested_frame_count": int(
                    target_review.get("requested_frame_count", 0)
                ),
                "source_review_rejected_frame_count": int(
                    source_review.get("rejected_frame_count", 0)
                ),
                "target_review_rejected_frame_count": int(
                    target_review.get("rejected_frame_count", 0)
                ),
                "source_review_refinement_mode": str(
                    source_review.get("refinement_mode", "")
                ),
                "target_review_refinement_mode": str(
                    target_review.get("refinement_mode", "")
                ),
                "source_review_input_mode": str(
                    source_review.get("review_input_mode", "")
                ),
                "target_review_input_mode": str(
                    target_review.get("review_input_mode", "")
                ),
                "source_review_skip_reason_counts": dict(
                    source_review.get("skip_reason_counts", {})
                ),
                "target_review_skip_reason_counts": dict(
                    target_review.get("skip_reason_counts", {})
                ),
                "source_local_refined_support_count": int(
                    source_refinement["refined_support_count"]
                ),
                "target_local_refined_support_count": int(
                    target_refinement["refined_support_count"]
                ),
                "source_local_refined_support_ratio": float(
                    source_refinement["refined_support_ratio"]
                ),
                "target_local_refined_support_ratio": float(
                    target_refinement["refined_support_ratio"]
                ),
                "source_local_refinement_fitness_mean": (
                    None
                    if source_refinement["refinement_fitness_mean"] is None
                    else float(source_refinement["refinement_fitness_mean"])
                ),
                "target_local_refinement_fitness_mean": (
                    None
                    if target_refinement["refinement_fitness_mean"] is None
                    else float(target_refinement["refinement_fitness_mean"])
                ),
                "frame_stride": int(candidate.get("frame_stride", 0)),
                "registration_fitness": float(
                    candidate.get("registration_fitness", 0.0)
                ),
                "registration_inlier_rmse": float(
                    candidate.get("registration_inlier_rmse", 0.0)
                ),
                "pose_rotation_deg": float(candidate.get("pose_rotation_deg", 0.0)),
                "pose_translation_m": float(candidate.get("pose_translation_m", 0.0)),
                "imu_predicted_nn_mean_m": float(imu_overlap["nn_mean_m"]),
                "imu_predicted_nn_median_m": float(imu_overlap["nn_median_m"]),
                "imu_predicted_nn_p95_m": float(imu_overlap["nn_p95_m"]),
                "imu_predicted_within_0p1m_ratio": float(
                    imu_overlap["within_0p1m_ratio"]
                ),
                "imu_predicted_within_0p2m_ratio": float(
                    imu_overlap["within_0p2m_ratio"]
                ),
                "imu_predicted_within_0p4m_ratio": float(
                    imu_overlap["within_0p4m_ratio"]
                ),
                "lidar_registered_nn_mean_m": float(lidar_overlap["nn_mean_m"]),
                "lidar_registered_nn_median_m": float(lidar_overlap["nn_median_m"]),
                "lidar_registered_nn_p95_m": float(lidar_overlap["nn_p95_m"]),
                "lidar_registered_within_0p1m_ratio": float(
                    lidar_overlap["within_0p1m_ratio"]
                ),
                "lidar_registered_within_0p2m_ratio": float(
                    lidar_overlap["within_0p2m_ratio"]
                ),
                "lidar_registered_within_0p4m_ratio": float(
                    lidar_overlap["within_0p4m_ratio"]
                ),
            }
        )
        imu_anchor_pose = end_imu_anchor_pose
        lidar_anchor_pose = end_lidar_anchor_pose

    if not review_rows:
        return {}, [], []

    artifacts: dict[str, str] = {}
    if not imu_cloud.is_empty():
        artifacts["imu_trajectory_cloud"] = write_point_cloud(
            diagnostics_dir / "imu_trajectory_cloud.ply",
            imu_cloud.voxel_down_sample(0.20),
        )
    if not lidar_cloud.is_empty():
        artifacts["lidar_trajectory_cloud"] = write_point_cloud(
            diagnostics_dir / "lidar_trajectory_cloud.ply",
            lidar_cloud.voxel_down_sample(0.20),
        )
    if not overlay_cloud.is_empty():
        artifacts["trajectory_overlay_cloud"] = write_point_cloud(
            diagnostics_dir / "trajectory_overlay_cloud.ply",
            overlay_cloud.voxel_down_sample(0.20),
        )

    review_summary = {
        "schema_version": 1,
        "module": "lidar2imu",
        "review_surface": "selected_solver_world_scene_overlap",
        "segment_count": len(review_rows),
        "world_coordinate_frame": "first_selected_window_source_imu",
        "submap_builder": "dense_scan_to_map_gicp",
        "trajectory_builder": (
            "solver_inlier_world_chain"
            if candidate_scope == "solver_inlier_windows"
            else "selected_candidate_world_chain"
        ),
        "candidate_scope": candidate_scope,
        "notes": {
            "imu_trajectory_cloud": (
                "Gray target-map geometry plus blue source geometry warped by the "
                "calibrated IMU-predicted LiDAR delta and placed in the accumulated "
                "selected-window world frame."
            ),
            "lidar_trajectory_cloud": (
                "Gray target-map geometry plus red source geometry warped by the "
                "LiDAR registration delta and placed in the same accumulated world "
                "frame; scene scans are replayed from the stored support-pose chain "
                "without a second local-registration pass."
            ),
            "trajectory_overlay_cloud": (
                "Combined gray target, blue IMU-predicted source, and red "
                "LiDAR-registered source in one accumulated world frame for direct "
                "ghosting review on the exact selected solver windows when available."
            ),
        },
        "candidates": review_rows,
    }
    review_yaml_path = diagnostics_dir / "registration_review.yaml"
    review_yaml_path.write_text(
        yaml.safe_dump(review_summary, sort_keys=False), encoding="utf-8"
    )
    artifacts["registration_review_yaml"] = str(review_yaml_path)
    review_csv_path = diagnostics_dir / "registration_review.csv"
    csv_artifact = write_csv_rows(review_csv_path, review_rows)
    if csv_artifact:
        artifacts["registration_review_csv"] = csv_artifact
    dense_nodes = [
        dense_nodes_by_timestamp[timestamp_ns]
        for timestamp_ns in sorted(dense_nodes_by_timestamp)
    ]
    return artifacts, review_rows, dense_nodes


def load_node_clouds(
    dataset: CalibrationDataset, nodes: list[TrajectoryNode]
) -> list[tuple[TrajectoryNode, o3d.geometry.PointCloud]]:
    if not nodes:
        return []
    motion_samples = sorted(
        dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if not motion_samples:
        return []
    lidar_topic = motion_samples[0].metadata.get("lidar_topic")
    if not lidar_topic:
        return []

    node_clouds = []
    for node in nodes:
        if not node.record_path:
            return []
        if not Path(node.record_path).exists():
            return []
        cloud = load_pointcloud_from_meta(
            PointCloudMeta(
                topic=str(lidar_topic),
                frame_id=dataset.child_frame,
                timestamp_ns=int(node.timestamp_ns),
                record_path=str(node.record_path),
            )
        )
        if cloud.is_empty():
            continue
        cloud = cloud.voxel_down_sample(0.20)
        node_clouds.append((node, cloud))
    return node_clouds


def build_stitched_cloud(
    node_clouds: list[tuple[TrajectoryNode, o3d.geometry.PointCloud]],
    *,
    pose_source: str,
    final_transform: np.ndarray,
    color: tuple[float, float, float],
) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for node, cloud in node_clouds:
        transformed = o3d.geometry.PointCloud(cloud)
        base_pose = node.imu_pose if pose_source == "imu" else node.lidar_pose
        transformed.transform(base_pose @ final_transform)
        transformed.paint_uniform_color(list(color))
        merged += transformed
    if merged.is_empty():
        return merged
    return merged.voxel_down_sample(0.20)
