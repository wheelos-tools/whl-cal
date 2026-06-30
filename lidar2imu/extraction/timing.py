from __future__ import annotations

import bisect

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2lidar.prepared_dataset import ImuSample, PoseSample


def uniform_indices(length: int, max_items: int) -> list[int]:
    if length <= 0:
        return []
    if max_items <= 0 or length <= max_items:
        return list(range(length))
    return np.linspace(0, length - 1, num=max_items, dtype=int).tolist()


def nearest_index(sorted_timestamps: list[int], timestamp_ns: int) -> int | None:
    if not sorted_timestamps:
        return None
    index = bisect.bisect_left(sorted_timestamps, timestamp_ns)
    candidates = []
    if index < len(sorted_timestamps):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: abs(sorted_timestamps[candidate] - timestamp_ns),
    )


def nearest_sample(
    samples: list, timestamps: list[int], timestamp_ns: int, max_delta_ns: int
):
    index = nearest_index(timestamps, timestamp_ns)
    if index is None:
        return None, None
    sample = samples[index]
    delta_ns = abs(timestamps[index] - timestamp_ns)
    if delta_ns > max_delta_ns:
        return None, delta_ns
    return sample, delta_ns


def nearest_sample_signed(
    samples: list, timestamps: list[int], timestamp_ns: int, max_delta_ns: int
):
    index = nearest_index(timestamps, timestamp_ns)
    if index is None:
        return None, None
    sample = samples[index]
    delta_ns = int(timestamps[index] - timestamp_ns)
    if abs(delta_ns) > max_delta_ns:
        return None, delta_ns
    return sample, delta_ns


def rotation_delta_angle_rad(
    rotation_prev: np.ndarray, rotation_curr: np.ndarray
) -> float:
    delta = np.asarray(rotation_prev, dtype=float).T @ np.asarray(
        rotation_curr, dtype=float
    )
    return float(np.linalg.norm(R.from_matrix(delta).as_rotvec()))


def pose_angular_speed_series(
    pose_samples: list[PoseSample],
) -> tuple[np.ndarray, np.ndarray]:
    timestamps_ns: list[int] = []
    angular_speeds_rad_s: list[float] = []
    for prev_sample, curr_sample in zip(pose_samples[:-1], pose_samples[1:]):
        dt_ns = int(curr_sample.timestamp_ns - prev_sample.timestamp_ns)
        if dt_ns <= 0:
            continue
        dt_sec = float(dt_ns * 1e-9)
        delta_angle_rad = rotation_delta_angle_rad(
            prev_sample.transform_world_imu[:3, :3],
            curr_sample.transform_world_imu[:3, :3],
        )
        timestamps_ns.append(int(curr_sample.timestamp_ns))
        angular_speeds_rad_s.append(float(delta_angle_rad / dt_sec))
    if not timestamps_ns:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float)
    return np.asarray(timestamps_ns, dtype=np.int64), np.asarray(
        angular_speeds_rad_s, dtype=float
    )


def lidar_proxy_angular_speed_series(
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    *,
    sample_count: int = 24,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not lidar_metas or not pose_samples:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
            {
                "requested_lidar_samples": 0,
                "matched_lidar_samples": 0,
                "reason": "missing_lidar_or_pose_samples",
            },
        )

    candidate_indices = uniform_indices(len(lidar_metas), sample_count)
    matched_lidar_timestamps_ns: list[int] = []
    matched_pose_rotations: list[np.ndarray] = []
    for meta_index in candidate_indices:
        timestamp_ns = int(lidar_metas[meta_index].timestamp_ns)
        pose_sample, delta_ns = nearest_sample_signed(
            pose_samples, pose_timestamps, timestamp_ns, sync_threshold_ns
        )
        if pose_sample is None or delta_ns is None:
            continue
        matched_lidar_timestamps_ns.append(int(timestamp_ns))
        matched_pose_rotations.append(
            np.asarray(pose_sample.transform_world_imu[:3, :3], dtype=float)
        )

    if len(matched_lidar_timestamps_ns) < 3:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
            {
                "requested_lidar_samples": int(len(candidate_indices)),
                "matched_lidar_samples": int(len(matched_lidar_timestamps_ns)),
                "reason": "insufficient_matches",
            },
        )

    angular_speed_timestamps_ns: list[int] = []
    angular_speeds_rad_s: list[float] = []
    for index in range(1, len(matched_lidar_timestamps_ns)):
        prev_ts = int(matched_lidar_timestamps_ns[index - 1])
        curr_ts = int(matched_lidar_timestamps_ns[index])
        dt_ns = curr_ts - prev_ts
        if dt_ns <= 0:
            continue
        delta_angle_rad = rotation_delta_angle_rad(
            matched_pose_rotations[index - 1], matched_pose_rotations[index]
        )
        angular_speed_timestamps_ns.append(curr_ts)
        angular_speeds_rad_s.append(float(delta_angle_rad / float(dt_ns * 1e-9)))

    if len(angular_speed_timestamps_ns) < 3:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
            {
                "requested_lidar_samples": int(len(candidate_indices)),
                "matched_lidar_samples": int(len(matched_lidar_timestamps_ns)),
                "reason": "insufficient_angular_speed_pairs",
            },
        )

    return (
        np.asarray(angular_speed_timestamps_ns, dtype=np.int64),
        np.asarray(angular_speeds_rad_s, dtype=float),
        {
            "requested_lidar_samples": int(len(candidate_indices)),
            "matched_lidar_samples": int(len(matched_lidar_timestamps_ns)),
            "reason": None,
        },
    )


def estimate_pose_time_offset_nearest_median_ns(
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    *,
    sample_count: int = 24,
) -> tuple[int, dict]:
    if not lidar_metas or not pose_samples:
        return 0, {
            "enabled": True,
            "estimator": "nearest_median",
            "sample_count": 0,
            "matched_count": 0,
            "reason": "missing_lidar_or_pose_samples",
        }

    candidate_indices = uniform_indices(len(lidar_metas), sample_count)
    signed_deltas_ns = []
    for meta_index in candidate_indices:
        timestamp_ns = int(lidar_metas[meta_index].timestamp_ns)
        _, delta_ns = nearest_sample_signed(
            pose_samples, pose_timestamps, timestamp_ns, sync_threshold_ns
        )
        if delta_ns is None:
            continue
        signed_deltas_ns.append(int(delta_ns))

    if len(signed_deltas_ns) < 3:
        return 0, {
            "enabled": True,
            "estimator": "nearest_median",
            "sample_count": int(len(candidate_indices)),
            "matched_count": int(len(signed_deltas_ns)),
            "reason": "insufficient_matches",
        }

    series = np.asarray(signed_deltas_ns, dtype=float)
    offset_ns = int(np.round(np.median(series)))
    return offset_ns, {
        "enabled": True,
        "estimator": "nearest_median",
        "sample_count": int(len(candidate_indices)),
        "matched_count": int(series.size),
        "estimated_offset_ns": int(offset_ns),
        "estimated_offset_ms": float(offset_ns / 1e6),
        "median_sync_dt_ms": float(np.median(series) / 1e6),
        "mean_sync_dt_ms": float(np.mean(series) / 1e6),
        "std_sync_dt_ms": float(np.std(series) / 1e6),
        "min_sync_dt_ms": float(np.min(series) / 1e6),
        "max_sync_dt_ms": float(np.max(series) / 1e6),
        "abs_p95_sync_dt_ms": float(np.percentile(np.abs(series), 95) / 1e6),
        "reason": None,
    }


def estimate_pose_time_offset_xcorr_ns(
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    *,
    sample_count: int = 96,
) -> tuple[int, dict]:
    if not lidar_metas or len(pose_samples) < 3:
        return 0, {
            "enabled": True,
            "estimator": "xcorr_angular_speed",
            "sample_count": 0,
            "matched_count": 0,
            "reason": "missing_lidar_or_pose_samples",
        }

    pose_speed_ts_ns, pose_speed_values = pose_angular_speed_series(pose_samples)
    (
        lidar_proxy_ts_ns,
        lidar_proxy_values,
        proxy_info,
    ) = lidar_proxy_angular_speed_series(
        lidar_metas,
        pose_samples,
        pose_timestamps,
        sync_threshold_ns,
        sample_count=sample_count,
    )
    if pose_speed_ts_ns.size < 4 or lidar_proxy_ts_ns.size < 4:
        return 0, {
            "enabled": True,
            "estimator": "xcorr_angular_speed",
            "sample_count": int(proxy_info.get("requested_lidar_samples", 0)),
            "matched_count": int(proxy_info.get("matched_lidar_samples", 0)),
            "pose_speed_count": int(pose_speed_ts_ns.size),
            "lidar_proxy_speed_count": int(lidar_proxy_ts_ns.size),
            "reason": "insufficient_angular_speed_series",
        }

    lidar_diff_ns = np.diff(lidar_proxy_ts_ns.astype(np.int64))
    lidar_diff_ns = lidar_diff_ns[lidar_diff_ns > 0]
    if lidar_diff_ns.size == 0:
        return 0, {
            "enabled": True,
            "estimator": "xcorr_angular_speed",
            "sample_count": int(proxy_info.get("requested_lidar_samples", 0)),
            "matched_count": int(proxy_info.get("matched_lidar_samples", 0)),
            "pose_speed_count": int(pose_speed_ts_ns.size),
            "lidar_proxy_speed_count": int(lidar_proxy_ts_ns.size),
            "reason": "invalid_lidar_proxy_timing",
        }

    lag_step_ns = max(int(np.round(np.median(lidar_diff_ns))), 1)
    max_lag_ns = max(int(sync_threshold_ns), lag_step_ns)
    lag_values_ns = np.arange(-max_lag_ns, max_lag_ns + lag_step_ns, lag_step_ns)

    pose_speed_ts_float = pose_speed_ts_ns.astype(float)
    pose_speed_values_float = pose_speed_values.astype(float)
    lidar_proxy_ts_float = lidar_proxy_ts_ns.astype(float)
    lidar_proxy_values_float = lidar_proxy_values.astype(float)

    best_corr = -np.inf
    best_lag_ns = 0
    best_pair_count = 0
    evaluated_lags = 0
    for lag_ns in lag_values_ns:
        query_ts = lidar_proxy_ts_float + float(lag_ns)
        valid_mask = (query_ts >= pose_speed_ts_float[0]) & (
            query_ts <= pose_speed_ts_float[-1]
        )
        if int(np.count_nonzero(valid_mask)) < 4:
            continue
        lidar_series = lidar_proxy_values_float[valid_mask]
        pose_series = np.interp(
            query_ts[valid_mask], pose_speed_ts_float, pose_speed_values_float
        )
        lidar_centered = lidar_series - float(np.mean(lidar_series))
        pose_centered = pose_series - float(np.mean(pose_series))
        denom = float(np.linalg.norm(lidar_centered) * np.linalg.norm(pose_centered))
        if denom <= 1e-12:
            continue
        corr = float(np.dot(lidar_centered, pose_centered) / denom)
        evaluated_lags += 1
        if corr > best_corr:
            best_corr = corr
            best_lag_ns = int(lag_ns)
            best_pair_count = int(np.count_nonzero(valid_mask))

    if not np.isfinite(best_corr):
        return 0, {
            "enabled": True,
            "estimator": "xcorr_angular_speed",
            "sample_count": int(proxy_info.get("requested_lidar_samples", 0)),
            "matched_count": int(proxy_info.get("matched_lidar_samples", 0)),
            "pose_speed_count": int(pose_speed_ts_ns.size),
            "lidar_proxy_speed_count": int(lidar_proxy_ts_ns.size),
            "reason": "insufficient_correlation_support",
        }

    return int(best_lag_ns), {
        "enabled": True,
        "estimator": "xcorr_angular_speed",
        "sample_count": int(proxy_info.get("requested_lidar_samples", 0)),
        "matched_count": int(proxy_info.get("matched_lidar_samples", 0)),
        "pose_speed_count": int(pose_speed_ts_ns.size),
        "lidar_proxy_speed_count": int(lidar_proxy_ts_ns.size),
        "estimated_offset_ns": int(best_lag_ns),
        "estimated_offset_ms": float(best_lag_ns / 1e6),
        "best_correlation": float(best_corr),
        "best_pair_count": int(best_pair_count),
        "lag_step_ms": float(lag_step_ns / 1e6),
        "lag_search_range_ms": {
            "min": float(-max_lag_ns / 1e6),
            "max": float(max_lag_ns / 1e6),
        },
        "evaluated_lag_count": int(evaluated_lags),
        "reason": None,
    }


def estimate_pose_time_offset_ns(
    lidar_metas: list,
    pose_samples: list[PoseSample],
    pose_timestamps: list[int],
    sync_threshold_ns: int,
    *,
    estimator: str = "nearest_median",
) -> tuple[int, dict]:
    if estimator == "nearest_median":
        return estimate_pose_time_offset_nearest_median_ns(
            lidar_metas, pose_samples, pose_timestamps, sync_threshold_ns
        )
    if estimator == "xcorr_angular_speed":
        return estimate_pose_time_offset_xcorr_ns(
            lidar_metas, pose_samples, pose_timestamps, sync_threshold_ns
        )
    raise ValueError(f"Unsupported pose time offset estimator: {estimator}")


def shift_timestamp_ns(timestamp_ns: int, offset_ns: int) -> int:
    return int(timestamp_ns) + int(offset_ns)


def windowed_imu_gravity(
    imu_samples: list[ImuSample],
    imu_timestamps: list[int],
    timestamp_ns: int,
    window_ns: int,
) -> np.ndarray | None:
    if not imu_samples:
        return None
    left = bisect.bisect_left(imu_timestamps, timestamp_ns - window_ns)
    right = bisect.bisect_right(imu_timestamps, timestamp_ns + window_ns)
    if left >= right:
        sample, _ = nearest_sample(imu_samples, imu_timestamps, timestamp_ns, window_ns)
        if sample is None:
            return None
        return sample.linear_acceleration
    values = np.asarray(
        [sample.linear_acceleration for sample in imu_samples[left:right]], dtype=float
    )
    return np.median(values, axis=0)
