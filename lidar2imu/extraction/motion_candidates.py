from __future__ import annotations

import numpy as np

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
    timing_diagnostics: dict | None = None,
) -> list[dict]:
    if base_stride < 1:
        raise ValueError("motion_frame_stride must be >= 1.")

    candidate_records: list[dict] = []
    positive_frame_deltas_ns = np.asarray(
        [
            int(current.timestamp_ns) - int(previous.timestamp_ns)
            for previous, current in zip(lidar_metas, lidar_metas[1:])
            if int(current.timestamp_ns) > int(previous.timestamp_ns)
        ],
        dtype=np.int64,
    )
    median_frame_delta_ns = (
        int(np.median(positive_frame_deltas_ns)) if positive_frame_deltas_ns.size else 0
    )
    rejected_frame_gap_count = 0
    frame_gap_examples = []
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
            pair_duration_ns = int(end_meta.timestamp_ns) - int(start_meta.timestamp_ns)
            expected_duration_ns = median_frame_delta_ns * int(stride)
            max_duration_ns = max(
                int(round(expected_duration_ns * 2.5)),
                expected_duration_ns + 250_000_000,
            )
            if pair_duration_ns <= 0 or (
                expected_duration_ns > 0 and pair_duration_ns > max_duration_ns
            ):
                rejected_frame_gap_count += 1
                if len(frame_gap_examples) < 20:
                    frame_gap_examples.append(
                        {
                            "start_index": int(start_index),
                            "end_index": int(end_index),
                            "stride": int(stride),
                            "pair_duration_ms": float(pair_duration_ns / 1e6),
                            "expected_duration_ms": float(expected_duration_ns / 1e6),
                            "max_duration_ms": float(max_duration_ns / 1e6),
                        }
                    )
                continue
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
                    "pair_duration_ms": float(pair_duration_ns / 1e6),
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
                    "pair_duration_ms": float(pair_duration_ns / 1e6),
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
    if timing_diagnostics is not None:
        timing_diagnostics.update(
            {
                "median_frame_delta_ms": float(median_frame_delta_ns / 1e6),
                "rejected_frame_gap_count": int(rejected_frame_gap_count),
                "frame_gap_examples": frame_gap_examples,
            }
        )
    return candidate_records
