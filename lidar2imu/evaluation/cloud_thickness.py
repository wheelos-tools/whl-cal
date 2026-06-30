from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from lidar2imu.algorithms import summarize_values
from lidar2imu.models import CalibrationConfig, CalibrationDataset, MotionSample
from lidar2lidar.record_utils import PointCloudMeta, load_pointcloud_from_meta


@dataclass(frozen=True)
class _WindowSelection:
    start_index: int
    end_index: int
    duration_sec: float
    yaw_abs_p95_deg: float
    speed_range_mps: float
    accel_abs_p90_mps2: float
    score: float


def _rotation_yaw_deg(rotation_matrix: np.ndarray) -> float:
    try:
        return float(
            np.degrees(
                R.from_matrix(np.asarray(rotation_matrix, dtype=float)).as_euler(
                    "ZYX", degrees=False
                )[0]
            )
        )
    except ValueError:
        return 0.0


def _sample_yaw_deg(sample: MotionSample) -> float:
    metadata_yaw = sample.metadata.get("imu_signed_yaw_deg")
    if metadata_yaw is not None:
        return float(metadata_yaw)
    return _rotation_yaw_deg(sample.imu_delta_rotation)


def _sample_duration_sec(sample: MotionSample) -> float:
    return max(
        float(sample.end_timestamp_ns - sample.start_timestamp_ns) * 1e-9,
        1e-3,
    )


def _sample_speed_mps(sample: MotionSample) -> float:
    return float(
        np.linalg.norm(sample.imu_delta_translation) / _sample_duration_sec(sample)
    )


def _window_features(
    samples: list[MotionSample],
    start_index: int,
    end_index: int,
) -> _WindowSelection:
    window = samples[start_index : end_index + 1]
    duration_sec = (
        float(window[-1].end_timestamp_ns - window[0].start_timestamp_ns) * 1e-9
    )
    yaw_abs = np.asarray(
        [abs(_sample_yaw_deg(sample)) for sample in window], dtype=float
    )
    speeds = np.asarray([_sample_speed_mps(sample) for sample in window], dtype=float)
    yaw_abs_p95 = float(np.percentile(yaw_abs, 95)) if yaw_abs.size else 0.0
    speed_range = float(np.max(speeds) - np.min(speeds)) if speeds.size else 0.0

    if speeds.size >= 3:
        dt = np.asarray(
            [_sample_duration_sec(sample) for sample in window], dtype=float
        )
        accel = np.diff(speeds) / np.maximum(dt[1:], 1e-3)
        accel_abs_p90 = float(np.percentile(np.abs(accel), 90))
        has_accel = bool(np.any(accel > 0.2))
        has_brake = bool(np.any(accel < -0.2))
    else:
        accel_abs_p90 = 0.0
        has_accel = False
        has_brake = False

    straightness_score = max(0.0, 1.0 - (yaw_abs_p95 / 6.0))
    speed_score = min(speed_range / 2.0, 1.0)
    accel_score = min(accel_abs_p90 / 2.0, 1.0)
    accel_brake_bonus = 0.5 if (has_accel and has_brake) else 0.0
    score = float(
        2.0 * straightness_score + speed_score + accel_score + accel_brake_bonus
    )
    return _WindowSelection(
        start_index=start_index,
        end_index=end_index,
        duration_sec=float(duration_sec),
        yaw_abs_p95_deg=yaw_abs_p95,
        speed_range_mps=speed_range,
        accel_abs_p90_mps2=accel_abs_p90,
        score=score,
    )


def _select_holdout_window(
    samples: list[MotionSample],
    *,
    target_duration_sec: float,
) -> _WindowSelection | None:
    if len(samples) < 2:
        return None
    min_duration_sec = max(2.0, 0.8 * float(target_duration_sec))
    max_duration_sec = max(float(target_duration_sec), 1.8 * float(target_duration_sec))
    candidates: list[_WindowSelection] = []
    for start_index in range(len(samples)):
        for end_index in range(start_index + 1, len(samples)):
            duration_sec = (
                float(
                    samples[end_index].end_timestamp_ns
                    - samples[start_index].start_timestamp_ns
                )
                * 1e-9
            )
            if duration_sec < min_duration_sec:
                continue
            if duration_sec > max_duration_sec:
                break
            candidates.append(_window_features(samples, start_index, end_index))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.score)


def _matrix_from_motion_sample(sample: MotionSample) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(sample.imu_delta_rotation, dtype=float)
    transform[:3, 3] = np.asarray(sample.imu_delta_translation, dtype=float)
    return transform


def _subsample_indices(count: int, max_count: int) -> list[int]:
    if count <= max_count:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, num=max_count, dtype=int).tolist()))


def _crop_cloud_for_plane_metrics(
    cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points, dtype=float)
    if points.size == 0:
        return o3d.geometry.PointCloud()
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if points.size == 0:
        return o3d.geometry.PointCloud()
    horizontal_norm = np.linalg.norm(points[:, :2], axis=1)
    mask = (horizontal_norm <= 40.0) & (np.abs(points[:, 2]) <= 4.0)
    points = points[mask]
    if points.size == 0:
        return o3d.geometry.PointCloud()
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(points)
    return cropped


def _plane_thickness_from_points(
    points: np.ndarray,
    normal: np.ndarray,
    intercept: float,
) -> dict:
    distances = np.abs(points @ normal + float(intercept))
    if distances.size == 0:
        return {
            "sample_count": 0,
            "thickness_p95_m": None,
            "thickness_mean_m": None,
            "thickness_std_m": None,
        }
    return {
        "sample_count": int(distances.size),
        "thickness_p95_m": float(np.percentile(distances, 95)),
        "thickness_mean_m": float(np.mean(distances)),
        "thickness_std_m": float(np.std(distances)),
    }


def _extract_plane_thicknesses(
    stitched_cloud: o3d.geometry.PointCloud,
) -> tuple[dict | None, dict | None]:
    evaluation_cloud = stitched_cloud.voxel_down_sample(0.08)
    if evaluation_cloud.is_empty():
        return None, None
    all_points = np.asarray(evaluation_cloud.points, dtype=float)
    if all_points.shape[0] < 1200:
        return None, None

    working_cloud = o3d.geometry.PointCloud(evaluation_cloud)
    ground_plane = None
    wall_plane = None
    min_plane_points = 600
    for _ in range(6):
        if len(working_cloud.points) < min_plane_points:
            break
        model, inliers = working_cloud.segment_plane(
            distance_threshold=0.06,
            ransac_n=3,
            num_iterations=800,
        )
        if len(inliers) < min_plane_points:
            break
        normal = np.asarray(model[:3], dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-9:
            break
        normal = normal / normal_norm
        intercept = float(model[3]) / normal_norm
        z_alignment = abs(float(normal[2]))
        near_distances = np.abs(all_points @ normal + intercept)
        support_points = all_points[near_distances <= 0.30]
        plane_payload = {
            "normal_xyz": {
                "x": float(normal[0]),
                "y": float(normal[1]),
                "z": float(normal[2]),
            },
            "z_alignment": z_alignment,
            "inlier_count": int(len(inliers)),
            "support": _plane_thickness_from_points(support_points, normal, intercept),
        }
        if z_alignment >= 0.85:
            if ground_plane is None or int(plane_payload["inlier_count"]) > int(
                ground_plane["inlier_count"]
            ):
                ground_plane = plane_payload
        elif z_alignment <= 0.35:
            if wall_plane is None or int(plane_payload["inlier_count"]) > int(
                wall_plane["inlier_count"]
            ):
                wall_plane = plane_payload
        working_cloud = working_cloud.select_by_index(inliers, invert=True)
        if ground_plane is not None and wall_plane is not None:
            break
    return ground_plane, wall_plane


def evaluate_cloud_thickness(
    *,
    calibration_dataset: CalibrationDataset,
    holdout_dataset: CalibrationDataset | None,
    final_transform: np.ndarray,
    config: CalibrationConfig,
    enable_expensive_metrics: bool,
) -> tuple[dict, list[dict]]:
    thresholds = {
        "production_thickness_m": float(config.metrics_cloud_thickness_production_m),
        "warning_thickness_m": float(config.metrics_cloud_thickness_warning_m),
    }
    if not enable_expensive_metrics:
        return {
            "enabled": False,
            "status": "unknown",
            "primary_cause": "expensive_cloud_metrics_disabled",
            "thresholds": thresholds,
        }, []

    source_dataset = holdout_dataset or calibration_dataset
    source_split = "holdout" if holdout_dataset is not None else "calibration_fallback"
    ordered_samples = sorted(
        source_dataset.motion_samples,
        key=lambda sample: (sample.start_timestamp_ns, sample.end_timestamp_ns),
    )
    if len(ordered_samples) < 2:
        return {
            "enabled": True,
            "status": "unknown",
            "primary_cause": "insufficient_motion_samples",
            "source_split": source_split,
            "thresholds": thresholds,
        }, []

    selected_window = _select_holdout_window(
        ordered_samples,
        target_duration_sec=float(config.metrics_cloud_thickness_target_duration_sec),
    )
    if selected_window is None:
        return {
            "enabled": True,
            "status": "unknown",
            "primary_cause": "no_window_meets_duration",
            "source_split": source_split,
            "thresholds": thresholds,
        }, []

    selected_samples = ordered_samples[
        selected_window.start_index : selected_window.end_index + 1
    ]
    lidar_topic = None
    frame_rows: list[dict] = []
    frame_entries: list[dict] = []
    imu_pose = np.eye(4, dtype=float)
    first_sample = selected_samples[0]
    first_record_path = first_sample.metadata.get("record_path_start")
    if first_record_path:
        frame_entries.append(
            {
                "timestamp_ns": int(first_sample.start_timestamp_ns),
                "record_path": str(first_record_path),
                "imu_pose": imu_pose.copy(),
            }
        )

    for sample in selected_samples:
        lidar_topic = lidar_topic or sample.metadata.get("lidar_topic")
        imu_delta = _matrix_from_motion_sample(sample)
        imu_pose = imu_pose @ np.linalg.inv(imu_delta)
        record_path_end = sample.metadata.get("record_path_end")
        if record_path_end:
            frame_entries.append(
                {
                    "timestamp_ns": int(sample.end_timestamp_ns),
                    "record_path": str(record_path_end),
                    "imu_pose": imu_pose.copy(),
                }
            )

    lidar_topic = lidar_topic or calibration_dataset.metadata.get("lidar_topic")
    if not lidar_topic or not frame_entries:
        return {
            "enabled": True,
            "status": "unknown",
            "primary_cause": "missing_lidar_topic_or_record_path",
            "source_split": source_split,
            "thresholds": thresholds,
            "window": {
                "duration_sec": float(selected_window.duration_sec),
                "sample_count": int(len(selected_samples)),
                "yaw_abs_p95_deg": float(selected_window.yaw_abs_p95_deg),
                "speed_range_mps": float(selected_window.speed_range_mps),
                "accel_abs_p90_mps2": float(selected_window.accel_abs_p90_mps2),
            },
        }, frame_rows

    frame_entries = [
        frame_entries[index]
        for index in _subsample_indices(
            len(frame_entries),
            max(int(config.metrics_cloud_thickness_max_frames), 4),
        )
    ]

    stitched_cloud = o3d.geometry.PointCloud()
    loaded_frame_count = 0
    load_failures = 0
    for entry in frame_entries:
        meta = PointCloudMeta(
            topic=str(lidar_topic),
            frame_id=str(calibration_dataset.child_frame),
            timestamp_ns=int(entry["timestamp_ns"]),
            record_path=str(entry["record_path"]),
        )
        try:
            cloud = load_pointcloud_from_meta(meta)
        except RuntimeError as error:
            load_failures += 1
            frame_rows.append(
                {
                    "timestamp_ns": int(entry["timestamp_ns"]),
                    "record_path": str(entry["record_path"]),
                    "status": "load_failed",
                    "reason": str(error),
                }
            )
            continue
        if cloud.is_empty():
            frame_rows.append(
                {
                    "timestamp_ns": int(entry["timestamp_ns"]),
                    "record_path": str(entry["record_path"]),
                    "status": "empty_cloud",
                }
            )
            continue
        cloud = cloud.voxel_down_sample(
            float(config.metrics_cloud_thickness_voxel_size_m)
        )
        cloud = _crop_cloud_for_plane_metrics(cloud)
        if cloud.is_empty():
            frame_rows.append(
                {
                    "timestamp_ns": int(entry["timestamp_ns"]),
                    "record_path": str(entry["record_path"]),
                    "status": "filtered_empty_cloud",
                }
            )
            continue
        world_pose = np.asarray(entry["imu_pose"], dtype=float) @ np.asarray(
            final_transform, dtype=float
        )
        cloud.transform(world_pose)
        stitched_cloud += cloud
        loaded_frame_count += 1
        frame_rows.append(
            {
                "timestamp_ns": int(entry["timestamp_ns"]),
                "record_path": str(entry["record_path"]),
                "status": "loaded",
                "point_count": int(len(cloud.points)),
            }
        )

    if stitched_cloud.is_empty() or loaded_frame_count < 3:
        return {
            "enabled": True,
            "status": "unknown",
            "primary_cause": "insufficient_loaded_cloud_frames",
            "source_split": source_split,
            "thresholds": thresholds,
            "window": {
                "duration_sec": float(selected_window.duration_sec),
                "sample_count": int(len(selected_samples)),
                "yaw_abs_p95_deg": float(selected_window.yaw_abs_p95_deg),
                "speed_range_mps": float(selected_window.speed_range_mps),
                "accel_abs_p90_mps2": float(selected_window.accel_abs_p90_mps2),
            },
            "loaded_frame_count": int(loaded_frame_count),
            "load_failure_count": int(load_failures),
        }, frame_rows

    ground_plane, wall_plane = _extract_plane_thicknesses(stitched_cloud)
    thickness_candidates = []
    for plane in (ground_plane, wall_plane):
        if plane is None:
            continue
        value = plane.get("support", {}).get("thickness_p95_m")
        if value is not None:
            thickness_candidates.append(float(value))
    representative_thickness = (
        None if not thickness_candidates else float(max(thickness_candidates))
    )

    if representative_thickness is None:
        status = "unknown"
        primary_cause = "no_ground_or_wall_plane"
    elif representative_thickness < float(config.metrics_cloud_thickness_production_m):
        status = "pass"
        primary_cause = "cloud_thickness_production_grade"
    elif representative_thickness > float(config.metrics_cloud_thickness_warning_m):
        status = "warning"
        primary_cause = "cloud_thickness_exceeds_warning_threshold"
    else:
        status = "warning"
        primary_cause = "cloud_thickness_borderline"

    window_speeds = [_sample_speed_mps(sample) for sample in selected_samples]
    return {
        "enabled": True,
        "status": status,
        "primary_cause": primary_cause,
        "source_split": source_split,
        "thresholds": thresholds,
        "window": {
            "duration_sec": float(selected_window.duration_sec),
            "sample_count": int(len(selected_samples)),
            "start_timestamp_ns": int(selected_samples[0].start_timestamp_ns),
            "end_timestamp_ns": int(selected_samples[-1].end_timestamp_ns),
            "yaw_abs_p95_deg": float(selected_window.yaw_abs_p95_deg),
            "speed_range_mps": float(selected_window.speed_range_mps),
            "accel_abs_p90_mps2": float(selected_window.accel_abs_p90_mps2),
            "speed_mps": summarize_values(window_speeds),
        },
        "cloud_inputs": {
            "lidar_topic": str(lidar_topic),
            "requested_frame_count": int(len(frame_entries)),
            "loaded_frame_count": int(loaded_frame_count),
            "load_failure_count": int(load_failures),
            "stitched_point_count": int(len(stitched_cloud.points)),
        },
        "representative_thickness_p95_m": representative_thickness,
        "ground_plane": ground_plane,
        "wall_plane": wall_plane,
    }, frame_rows
