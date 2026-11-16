"""One fixed sensor-clock initialization at the offline replay boundary."""

from __future__ import annotations

import numpy as np

from lidar2imu.models import CanonicalDataset


def _median_ns(values: np.ndarray) -> int:
    ordered = np.sort(values)
    middle = len(ordered) // 2
    return (int(ordered[middle]) + int(ordered[(len(ordered) - 1) // 2])) // 2


def measurement_clocks_disjoint(dataset: CanonicalDataset) -> bool:
    """Legacy range-separation guard, not evidence that clocks are shared."""
    lidar, imu = dataset.lidar, dataset.imu
    left = max(int(lidar.scan_timestamps_ns[0]), int(imu.timestamps_ns[0]))
    right = min(int(lidar.scan_timestamps_ns[-1]), int(imu.timestamps_ns[-1]))
    return left - right > 1_000_000_000


def replay_hard_offset_ns(dataset: CanonicalDataset) -> int:
    """Return H in t_IMU_aligned = t_IMU - H, never including configured C.

    Complete publication evidence takes precedence over measurement-range
    overlap. Only offsets exceeding GRIL's one-second hard-offset threshold
    are applied. Publication latency asymmetry remains a coarse timing error
    for GRIL to estimate, not jitter to inject into individual samples.
    """
    lidar, imu = dataset.lidar.normalized(), dataset.imu.normalized()
    measurement = lidar.measurement_timestamps_ns
    lidar_dispatch = lidar.dispatch_timestamps_ns
    imu_dispatch = imu.dispatch_timestamps_ns
    if (
        measurement is None
        or lidar_dispatch is None
        or imu_dispatch is None
        or np.any(measurement == 0)
        or np.any(lidar_dispatch == 0)
        or np.any(imu_dispatch == 0)
    ):
        if not measurement_clocks_disjoint(dataset):
            return 0
        raise ValueError(
            "Disjoint sensor clocks require complete independent dispatch times "
            "and LiDAR scan-end measurement_time; re-extract the original record. "
            "Legacy canonical/bag timestamps alone cannot establish an epoch offset."
        )
    candidate = _median_ns(imu.timestamps_ns - imu_dispatch) - _median_ns(
        measurement - lidar_dispatch
    )
    if abs(candidate) <= 1_000_000_000:
        return 0
    if max(int(lidar_dispatch.min()), int(imu_dispatch.min())) > min(
        int(lidar_dispatch.max()), int(imu_dispatch.max())
    ):
        raise ValueError("Sensor dispatch time ranges do not overlap")
    return candidate
