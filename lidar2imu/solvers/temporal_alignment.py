from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lidar2imu.models import MotionSample
from lidar2imu.solvers.motion_objective import (motion_sample_window_ids,
                                                window_id)


def _signed_yaw_deg(rotation: np.ndarray) -> float:
    try:
        return float(np.degrees(R.from_matrix(rotation).as_euler("ZYX")[0]))
    except ValueError:
        return 0.0


def _median_step_ms(timestamps_ns: list[int]) -> float | None:
    if len(timestamps_ns) < 2:
        return None
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    deltas_ms = np.diff(timestamps).astype(float) / 1e6
    if deltas_ms.size == 0:
        return None
    return float(np.median(deltas_ms))


def _summarize(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _normalized_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if values_a.size != values_b.size or values_a.size < 2:
        return None
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _lag_scan(
    imu_signed_yaw_deg: list[float],
    lidar_signed_yaw_deg: list[float],
    timestamps_ns: list[int],
) -> dict[str, Any]:
    length = min(len(imu_signed_yaw_deg), len(lidar_signed_yaw_deg), len(timestamps_ns))
    if length < 3:
        return {
            "enabled": False,
            "reason": "insufficient_samples",
            "sample_count": int(length),
            "rows": [],
        }
    imu = np.asarray(imu_signed_yaw_deg[:length], dtype=float)
    lidar = np.asarray(lidar_signed_yaw_deg[:length], dtype=float)
    max_lag = min(3, length - 2)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            imu_segment = imu[: length + lag]
            lidar_segment = lidar[-lag:length]
        elif lag > 0:
            imu_segment = imu[lag:length]
            lidar_segment = lidar[: length - lag]
        else:
            imu_segment = imu
            lidar_segment = lidar
        correlation = _normalized_correlation(imu_segment, lidar_segment)
        row = {
            "lag_samples": int(lag),
            "paired_count": int(imu_segment.size),
            "correlation": correlation,
        }
        rows.append(row)
        if correlation is None:
            continue
        if best_row is None or float(correlation) > float(best_row["correlation"]):
            best_row = row
    median_step_ms = _median_step_ms(timestamps_ns[:length])
    return {
        "enabled": True,
        "sample_count": int(length),
        "median_step_ms": median_step_ms,
        "rows": rows,
        "best": (
            None
            if best_row is None
            else {
                **best_row,
                "lag_ms": (
                    None
                    if median_step_ms is None
                    else float(best_row["lag_samples"] * median_step_ms)
                ),
            }
        ),
    }


def _series_rows(motion_samples: list[MotionSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in sorted(
        motion_samples,
        key=lambda item: (item.start_timestamp_ns, item.end_timestamp_ns),
    ):
        rows.append(
            {
                "window_id": window_id(sample),
                "timestamp_ns": int(
                    (sample.start_timestamp_ns + sample.end_timestamp_ns) // 2
                ),
                "imu_signed_yaw_deg": _signed_yaw_deg(sample.imu_delta_rotation),
                "lidar_signed_yaw_deg": _signed_yaw_deg(sample.lidar_delta_rotation),
                "sync_dt_ms": (
                    None if sample.sync_dt_ms is None else float(sample.sync_dt_ms)
                ),
            }
        )
    return rows


def _summarize_temporal_samples(motion_samples: list[MotionSample]) -> dict[str, Any]:
    rows = _series_rows(motion_samples)
    timestamps = [int(row["timestamp_ns"]) for row in rows]
    imu_series = [float(row["imu_signed_yaw_deg"]) for row in rows]
    lidar_series = [float(row["lidar_signed_yaw_deg"]) for row in rows]
    yaw_gap_series = [
        float(row["imu_signed_yaw_deg"] - row["lidar_signed_yaw_deg"]) for row in rows
    ]
    sync_values = [
        float(row["sync_dt_ms"]) for row in rows if row.get("sync_dt_ms") is not None
    ]
    return {
        "sample_count": int(len(rows)),
        "window_ids": motion_sample_window_ids(motion_samples),
        "sync_dt_ms": _summarize(sync_values),
        "imu_signed_yaw_deg": _summarize(imu_series),
        "lidar_signed_yaw_deg": _summarize(lidar_series),
        "yaw_gap_deg": _summarize(yaw_gap_series),
        "lag_scan": _lag_scan(imu_series, lidar_series, timestamps),
        "rows": rows,
    }


def assess_temporal_alignment(
    active_motion_samples: list[MotionSample], candidate_pool: list[MotionSample]
) -> dict[str, Any]:
    pool_samples = candidate_pool if candidate_pool else active_motion_samples
    return {
        "active_selection": _summarize_temporal_samples(active_motion_samples),
        "candidate_pool": _summarize_temporal_samples(pool_samples),
    }
