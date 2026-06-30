from __future__ import annotations

# isort: off
from lidar2imu.extraction.motion_windows import motion_excitation
from lidar2imu.extraction.motion_windows import motion_rotation_axis_abs
from lidar2imu.extraction.motion_windows import motion_signed_yaw_deg
from lidar2imu.extraction.motion_windows import motion_translation_heading_deg
from lidar2imu.extraction.motion_windows import relative_motion
from lidar2imu.extraction.timing import nearest_sample, shift_timestamp_ns
from lidar2imu.motion_information import motion_information_components
from lidar2lidar.prepared_dataset import PoseSample

# isort: on


def build_motion_candidates(
    lidar_metas: list,
    *,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    pose_time_offset_ns: int,
    sync_threshold_ns: int,
    base_stride: int,
) -> list[dict]:
    if base_stride < 1:
        raise ValueError("motion_frame_stride must be >= 1.")

    candidate_records: list[dict] = []
    stride_values = []
    stride = int(base_stride)
    max_stride = max(int(base_stride), min(len(lidar_metas) // 2, int(base_stride) * 8))
    while stride <= max_stride:
        stride_values.append(int(stride))
        stride *= 2

    for stride in stride_values:
        for start_index in range(0, len(lidar_metas) - stride):
            end_index = start_index + stride
            start_meta = lidar_metas[start_index]
            end_meta = lidar_metas[end_index]
            start_timestamp_ns = shift_timestamp_ns(
                start_meta.timestamp_ns, pose_time_offset_ns
            )
            end_timestamp_ns = shift_timestamp_ns(
                end_meta.timestamp_ns, pose_time_offset_ns
            )
            start_pose, start_pose_dt_ns = nearest_sample(
                pose_samples,
                pose_timestamps,
                start_timestamp_ns,
                sync_threshold_ns,
            )
            end_pose, end_pose_dt_ns = nearest_sample(
                pose_samples,
                pose_timestamps,
                end_timestamp_ns,
                sync_threshold_ns,
            )
            if start_pose is None or end_pose is None:
                continue
            imu_delta = relative_motion(
                start_pose.transform_world_imu, end_pose.transform_world_imu
            )
            rotation_deg, translation_m = motion_excitation(imu_delta)
            info_components = motion_information_components(
                {
                    "pose_rotation_deg": rotation_deg,
                    "pose_translation_m": translation_m,
                    "stride": stride,
                    "weight": 1.0,
                },
                base_stride=base_stride,
            )
            information_score = float(info_components["base_information_score"])
            candidate_records.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "stride": stride,
                    "start_meta": start_meta,
                    "end_meta": end_meta,
                    "start_pose": start_pose,
                    "end_pose": end_pose,
                    "start_pose_dt_ns": start_pose_dt_ns,
                    "end_pose_dt_ns": end_pose_dt_ns,
                    "imu_delta": imu_delta,
                    "pose_rotation_deg": rotation_deg,
                    "pose_translation_m": translation_m,
                    "imu_translation_heading_deg": motion_translation_heading_deg(
                        imu_delta
                    ),
                    "imu_signed_yaw_deg": motion_signed_yaw_deg(imu_delta),
                    "imu_rotation_axis_abs": motion_rotation_axis_abs(imu_delta),
                    "information_score": information_score,
                    "probabilistic_information_score": float(
                        info_components["probabilistic_information_score"]
                    ),
                    "probabilistic_window_score": float(
                        info_components["probabilistic_window_score"]
                    ),
                    "information_uncertainty_scale": float(
                        info_components["uncertainty_scale"]
                    ),
                    "information_rotation_confidence": float(
                        info_components["rotation_confidence"]
                    ),
                    "information_translation_confidence": float(
                        info_components["translation_confidence"]
                    ),
                    "score": float(info_components["probabilistic_window_score"]),
                }
            )

    candidate_records.sort(
        key=lambda item: (
            -item["pose_rotation_deg"],
            -item["pose_translation_m"],
            item["start_index"],
            item["end_index"],
        )
    )
    return candidate_records
